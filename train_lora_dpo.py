import argparse
import copy
import json
import logging
import math
import os
import random
from pathlib import Path
from typing import Dict, List, Optional
import toml
from datetime import datetime

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers.optimization import get_scheduler
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm.auto import tqdm
from accelerate.utils import set_seed
from transformers import AutoTokenizer, AutoModel, AutoConfig
from peft import LoraConfig, get_peft_model, PeftModel, get_peft_model_state_dict, set_peft_model_state_dict
from safetensors.torch import load_file, save_file

from models.model import NewbieModel
from models.components import NewbieConfig
from dataset import DPODataset
from train_newbie_lora import load_model_and_tokenizer, setup_scheduler, setup_optimizer, save_checkpoint, save_lora_model
from transport import create_transport
try:
    import bitsandbytes as bnb
except ImportError:
    logging.warning("bitsandbytes not available, 8-bit optimizer disabled")

logger = get_logger(__name__)

class EMAModel:
    """
    Exponential Moving Average of models weights
    """
    def __init__(self, parameters: List[torch.nn.Parameter], decay=0.999):
        self.decay = decay
        self.shadow = {id(p): p.clone().detach() for p in parameters if p.requires_grad}
        self.collected_params = []

    def step(self, parameters: List[torch.nn.Parameter]):
        for p in parameters:
            if p.requires_grad:
                p_id = id(p)
                if p_id in self.shadow:
                    new_average = (1.0 - self.decay) * p.detach() + self.decay * self.shadow[p_id]
                    self.shadow[p_id] = new_average.clone()

    def copy_to(self, parameters: List[torch.nn.Parameter]):
        """
        Copy current parameters to collected_params to save them, 
        and replace current parameters with shadow weights.
        """
        self.collected_params = [p.clone().detach() for p in parameters if p.requires_grad]
        for p in parameters:
            if p.requires_grad and id(p) in self.shadow:
                p.data.copy_(self.shadow[id(p)].data)

    def restore(self, parameters: List[torch.nn.Parameter]):
        """
        Restore the parameters saved in copy_to.
        """
        for p, saved in zip([p for p in parameters if p.requires_grad], self.collected_params):
            p.data.copy_(saved.data)
        self.collected_params = []
    
    def state_dict(self):
        return self.shadow
    
    def load_state_dict(self, state_dict):
        self.shadow = state_dict

def collate_fn(batch):
    pixel_values_chosen = torch.stack([item["pixel_values_chosen"] for item in batch])
    pixel_values_rejected = torch.stack([item["pixel_values_rejected"] for item in batch])
    captions = [item["caption"] for item in batch]
    
    return {
        "pixel_values_chosen": pixel_values_chosen,
        "pixel_values_rejected": pixel_values_rejected,
        "captions": captions
    }

def calculate_sdpo_loss(model_chosen_mse, model_rejected_mse, ref_chosen_mse, ref_rejected_mse, beta=0.1, margin=0.5):
    """
    实现 Safeguarded DPO Loss
    在扩散模型中，log-prob 的差值可以用负 MSE 的差值来近似
    """
    # 隐式奖励计算 (Reward = -MSE)
    model_pref_delta = (model_rejected_mse - model_chosen_mse)
    ref_pref_delta = (ref_rejected_mse - ref_chosen_mse)
    
    # DPO 核心项
    logits = model_pref_delta - ref_pref_delta
    
    # SDPO 加入安全边界 margin
    loss = -F.logsigmoid(beta * (logits - margin)).mean()
    return loss

def compute_loss(model, ref_model, vae, text_encoder, tokenizer, clip_model, clip_tokenizer, transport, batch, device, gemma3_prompt="", beta=1000.0, mu=0.0, dmpo_alpha=0.0):
    pixel_values_chosen = batch["pixel_values_chosen"].to(device)
    pixel_values_rejected = batch["pixel_values_rejected"].to(device)
    captions = batch["captions"]
    batch_size = pixel_values_chosen.shape[0]
    scaling_factor = getattr(vae.config, 'scaling_factor', 0.3611)
    shift_factor = getattr(vae.config, 'shift_factor', 0.1159)

    with torch.no_grad():
        latents_chosen = vae.encode(pixel_values_chosen).latent_dist.sample()
        latents_chosen = (latents_chosen - shift_factor) * scaling_factor
        latents_rejected = vae.encode(pixel_values_rejected).latent_dist.sample()
        latents_rejected = (latents_rejected - shift_factor) * scaling_factor
        
        # Gemma 编码
        gemma_texts = [(gemma3_prompt + cap) if (gemma3_prompt and cap) else cap for cap in captions]
        gemma_inputs = tokenizer(
            gemma_texts, padding=True, pad_to_multiple_of=8,
            truncation=True, max_length=1280, return_tensors="pt"
        ).to(device)
        gemma_outputs = text_encoder(**gemma_inputs, output_hidden_states=True)
        cap_feats = gemma_outputs.hidden_states[-2]
        cap_mask = gemma_inputs.attention_mask
        
        # CLIP 编码
        clip_inputs = clip_tokenizer(
            captions, padding=True, truncation=True,
            max_length=2048, return_tensors="pt"
        ).to(device)
        clip_text_pooled = clip_model.get_text_features(**clip_inputs)

    model_kwargs = dict(cap_feats=cap_feats, cap_mask=cap_mask, clip_text_pooled=clip_text_pooled)

    ############ 损失计算 ############
    loss = transport.training_dpo_losses(model, ref_model, latents_chosen, latents_rejected, beta, mu, dmpo_alpha, model_kwargs)["loss"].mean()
    
    return loss

def main():
    """主训练函数"""
    parser = argparse.ArgumentParser(description="Newbie LoRA DPO Trainer")
    parser.add_argument("--config_file", type=str, required=True, help="Path to .toml config file")
    args = parser.parse_args()

    # 加载DPO训练参数
    with open(args.config_file, 'r', encoding='utf-8') as f:
        config = toml.load(f)

    # 若配置缺少 Optimization 段，补默认值避免 KeyError
    opt_cfg = config.setdefault('Optimization', {})
    opt_cfg.setdefault('optimizer_type', 'AdamW')
    opt_cfg.setdefault('use_flash_attention_2', False)

    gradient_accumulation_steps = config['Model'].get('gradient_accumulation_steps', 1)

    output_dir = config['Model']['output_dir']    
    os.makedirs(output_dir, exist_ok=True)

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
    accelerator = Accelerator(
        mixed_precision=config['Model'].get('mixed_precision', 'no'),
        gradient_accumulation_steps=gradient_accumulation_steps,
        log_with="wandb",
        project_dir=output_dir,
        kwargs_handlers=[ddp_kwargs],
        step_scheduler_with_optimizer=False
    )
    if accelerator.is_main_process:
        tracker_config = {
            **config['Model'], 
            **config['Optimization']
        }
        accelerator.init_trackers(
            project_name="Newbie_LoRA_DPO",
            config=tracker_config,
            init_kwargs={"wandb": {"name": config['Model'].get('output_name', 'run')}}
        )

    set_seed(42)

    if not config['Optimization'].get('use_flash_attention_2', False):
        torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False)

    if accelerator.is_main_process:
        os.makedirs(config['Model']['output_dir'], exist_ok=True)
    ####################################################################################################
    # 加载模型
    model, vae, text_encoder, tokenizer, clip_model, clip_tokenizer = load_model_and_tokenizer(config)

    # 应用 LoRA
    sft_lora_path = config['Model'].get('sft_lora_path', "")
    try:
        print(f"Loading LoRA from: {sft_lora_path}")
        model = PeftModel.from_pretrained(model, sft_lora_path, is_trainable=True)
        model.to(accelerator.device)
    except:
        print(f'{sft_lora_path} is not a valid PEFT LoRA directory path! Falling back to random init.')

    # 创建参考模型 (Reference Model) - 冻结的 SFT 状态
    ref_model = copy.deepcopy(model)
    ref_model.eval()
    ref_model.requires_grad_(False)

    # 创建 Rectified Flow transport
    #seq_len = (resolution // 16) ** 2
    transport = create_transport(
        path_type="Linear",
        prediction="velocity",
        snr_type="lognorm",
        do_shift=True,
        #seq_len=seq_len
    )
    logger.info(f"Rectified Flow transport created.")

    # 数据加载
    batch_size = config['Model']['train_batch_size']
    train_dataset = DPODataset(
        preference_json_path=config['Model']['preference_json'],
        caption_dropout_rate=config['Model'].get('caption_dropout_rate', 0.1), 
        real_ratio=config['Model'].get('real_ratio', 0.2), 
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=config['Model'].get('dataloader_num_workers', 4)
    )

     # 优化器与调度器
    optimizer = setup_optimizer(model, config)
    scheduler, num_training_steps = setup_scheduler(optimizer, config, train_dataloader)

    # 梯度检查点
    if config['Model'].get('gradient_checkpointing', True):
        model.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing enabled")

    # EMA 设置
    ema_model = None
    if config['Model'].get('use_ema', True):
        ema_model = EMAModel(model.parameters(), decay=config['Model'].get('ema_decay', 0.999))

    # accelerator准备模型
    model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)
    if text_encoder is not None:
        text_encoder = text_encoder.to(accelerator.device)
        text_encoder.eval()
        text_encoder.requires_grad_(False)
    if clip_model is not None:
        clip_model = clip_model.to(accelerator.device)
        clip_model.eval()
        clip_model.requires_grad_(False)
    if vae is not None:
        vae = vae.to(accelerator.device)
        vae.eval()
        vae.requires_grad_(False)
    ref_model.to(accelerator.device)

    # 训练检查点
    start_step = load_checkpoint(accelerator, model, optimizer, scheduler, config)
    
    logger.info("Training started")
    global_step = start_step
    session_start_step = start_step

    if accelerator.is_main_process:
        wandb_key = config['Model'].get('wandb_key')
        if wandb_key:
            try:
                import wandb
                wandb.login(key=wandb_key)
                logger.info("Successfully logged into WandB using config token.")
            except ImportError:
                logger.warning("wandb not installed, skipping login.")
            except Exception as e:
                logger.warning(f"Failed to login to WandB: {e}")

    start_time = datetime.now()
    max_grad_norm = config['Optimization'].get('gradient_clip_norm', 1.0)
    save_epochs_interval = config['Model'].get('save_epochs_interval', 0)

    #steps_per_epoch = len(train_dataloader)
    #start_epoch = start_step // steps_per_epoch
    #steps_to_skip_in_first_epoch = start_step % steps_per_epoch
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)
    start_epoch = start_step // num_update_steps_per_epoch
    resume_step_in_epoch = start_step % num_update_steps_per_epoch
    steps_to_skip_in_first_epoch = resume_step_in_epoch * gradient_accumulation_steps
    
    if start_step > 0:
        logger.info(f"Resuming from epoch {start_epoch+1}, will skip {steps_to_skip_in_first_epoch} steps in first epoch")

    for epoch in range(start_epoch, config['Model']['num_epochs']):
        epoch_losses = []
        model.train()
        progress_bar = tqdm(
            train_dataloader,
            desc=f"Epoch {epoch+1}/{config['Model']['num_epochs']}",
            disable=not accelerator.is_main_process
        )
        
        for batch_idx, batch in enumerate(progress_bar):
            if epoch == start_epoch and batch_idx < steps_to_skip_in_first_epoch:
                continue

            with accelerator.accumulate(model):
                loss = compute_loss(
                    model, 
                    ref_model,
                    vae, 
                    text_encoder, 
                    tokenizer, 
                    clip_model, 
                    clip_tokenizer, 
                    transport, 
                    batch, 
                    accelerator.device, 
                    config['Model'].get('gemma3_prompt', ''),
                    beta=config['Model']['beta']
                    mu=config['Model']['mu']
                    dmpo_alpha=config['Model']['dmpo_alpha']
                )
                epoch_losses.append(loss.item())
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    if max_grad_norm > 0:
                        accelerator.clip_grad_norm_(model.parameters(), max_grad_norm)

                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                    if ema_model is not None:
                        ema_model.step(model.parameters())

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    accelerator.log({"loss": loss.item(), "learning_rate": scheduler.get_last_lr()[0]}, step=global_step)
                    elapsed = datetime.now() - start_time
                    steps_in_session = global_step - session_start_step
                    steps_per_sec = steps_in_session / elapsed.total_seconds() if elapsed.total_seconds() > 0 else 0
                    progress_bar.set_postfix({
                        'loss': f'{loss.item():.4f}',
                        'lr': f'{scheduler.get_last_lr()[0]:.2e}',
                        'speed': f'{steps_per_sec:.2f} steps/s'
                    })
                if global_step % 100 == 0:
                    elapsed = datetime.now() - start_time
                    steps_in_session = global_step - session_start_step
                    steps_per_sec = steps_in_session / elapsed.total_seconds() if elapsed.total_seconds() > 0 else 0
                    logger.info(f"Epoch {epoch+1}/{config['Model']['num_epochs']}, Step {global_step}/{num_training_steps}, Loss {loss.item():.4f}, LR {scheduler.get_last_lr()[0]:.7f}, Speed {steps_per_sec:.2f} steps/s")

                if global_step % 1000 == 0:
                    save_checkpoint(accelerator, model, optimizer, scheduler, global_step, config)
                    if ema_model:
                        torch.save(ema_model.state_dict(), os.path.join(config['Model']['output_dir'], "ema_weights.pt"))

        avg_epoch_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0.0
        logger.info(f"Epoch {epoch+1}/{config['Model']['num_epochs']} completed - Average Loss: {avg_epoch_loss:.4f}")

        if save_epochs_interval == 0 or (epoch + 1) % save_epochs_interval == 0:
            save_checkpoint(accelerator, model, optimizer, scheduler, global_step, config)
            if ema_model:
                torch.save(ema_model.state_dict(), os.path.join(config['Model']['output_dir'], "ema_weights.pt"))
            logger.info(f"Checkpoint saved at epoch {epoch+1}")

    logger.info("Training complete, saving final model")
    save_lora_model(accelerator, model, config)

    if accelerator.is_main_process:
        accelerator.end_training()

    logger.info("Training finished")


if __name__ == "__main__":
    main()
