#!/usr/bin/env python3
"""Anima LoRA 训练器 - 基于 Rectified Flow 的 LoRA DPO"""

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

import sys
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "AnimaLoraToolkit"))
from anima_train import (
    load_anima_model,
    load_vae,
    load_text_encoders,
    ensure_models_namespace,
    _build_qwen_text_from_prompt,
    encode_qwen,
    _parse_weighted_tag,
    tokenize_t5_weighted
)

from dataset import DPODataset, DPOBucketBatchSampler
from train_anima_lora import (
    load_model_and_tokenizer,
    setup_scheduler,
    setup_optimizer,
    save_checkpoint,
    save_lora_model,
    load_checkpoint,
    EMAModel,
    enable_anima_gradient_checkpointing
)
from transport import create_transport
try:
    import bitsandbytes as bnb
except ImportError:
    logging.warning("bitsandbytes not available, 8-bit optimizer disabled")

logger = get_logger(__name__)

def collate_fn(batch):
    if batch[0].get("cached", False):
        latents_chosen = torch.stack([item["latents_chosen"] for item in batch])
        latents_rejected = torch.stack([item["latents_rejected"] for item in batch])
        captions = [item["caption"] for item in batch]
        return {
            "latents_chosen": latents_chosen,
            "latents_rejected": latents_rejected,
            "captions": captions,
            "cached": True
        }
    else:
        pixel_values_chosen = torch.stack([item["pixel_values_chosen"] for item in batch])
        pixel_values_rejected = torch.stack([item["pixel_values_rejected"] for item in batch])
        captions = [item["caption"] for item in batch]

        return {
            "pixel_values_chosen": pixel_values_chosen,
            "pixel_values_rejected": pixel_values_rejected,
            "captions": captions,
            "cached": False
        }

def compute_loss(model, ref_model, vae, qwen_model, qwen_tokenizer, t5_tokenizer, transport, batch, device, beta=1000.0, mu=0.0, dmpo_alpha=0.0):
    if batch.get("cached", False):
        latents_chosen = batch["latents_chosen"].to(device).unsqueeze(2)
        latents_rejected = batch["latents_rejected"].to(device).unsqueeze(2)
    else:
        if vae is None:
            raise RuntimeError("VAE required for non-cached data")
        captions = batch["captions"]
        pixel_values_chosen = batch["pixel_values_chosen"].to(device)
        pixel_values_rejected = batch["pixel_values_rejected"].to(device)
        with torch.no_grad():
            pixel_values_chosen = pixel_values_chosen.unsqueeze(2).to(dtype=next(vae.model.parameters()).dtype)
            pixel_values_rejected = pixel_values_rejected.unsqueeze(2).to(dtype=next(vae.model.parameters()).dtype)
            latents_chosen = vae.model.encode(pixel_values_chosen, vae.scale)
            latents_rejected = vae.model.encode(pixel_values_rejected, vae.scale)

    bs = latents_chosen.shape[0]
    
    with torch.no_grad():
        # 处理 Qwen 文本 (提取纯净标签)
        qwen_texts = [_build_qwen_text_from_prompt(cap) for cap in captions]
        qwen_embeds, qwen_attn = encode_qwen(qwen_model, qwen_tokenizer, qwen_texts, device)
        # 处理 T5 文本 (保留权重语法)
        t5_ids, t5_attn, t5_w = tokenize_t5_weighted(t5_tokenizer, captions, max_length=1024)
        t5_ids = t5_ids.to(device)
        # LLMAdapter 桥接
        unwrapped_model = model.module if hasattr(model, "module") else model
        cross = unwrapped_model.preprocess_text_embeds(qwen_embeds, t5_ids)
        if cross.shape[1] < 1024:
            cross = torch.nn.functional.pad(cross, (0, 0, 0, 1024 - cross.shape[1]))

    pad_mask = torch.zeros(bs, 1, latents_chosen.shape[-2], latents_chosen.shape[-1], device=device, dtype=latents.dtype)
    model_kwargs = dict(crossattn_emb=cross, padding_mask=pad_mask)

    ############ 损失计算 ############
    loss = transport.training_dpo_losses(model, ref_model, latents_chosen, latents_rejected, beta, mu, dmpo_alpha, model_kwargs)["loss"].mean()

    return loss

class ReferenceModelWrapper(torch.nn.Module):
    """
    一个包装器，用于复用 Base Model。
    在执行 forward 时临时切换到 'reference' adapter，
    执行完毕后恢复到 'default' adapter。
    """
    def __init__(self, model, accelerator):
        super().__init__()
        self.model = model
        self.accelerator = accelerator

    def forward(self, *args, **kwargs):
        # 获取底层的 PeftModel
        unwrapped = self.accelerator.unwrap_model(self.model)
        
        # 1. 切换到参考 Adapter
        # 保存之前的状态（主要是 training/eval 模式）
        was_training = unwrapped.training
        previous_adapter = unwrapped.active_adapter
        
        # 切换 adapter 并强制设为 eval 模式 (关闭 Dropout 等)
        unwrapped.set_adapter("reference")
        unwrapped.eval()
        
        try:
            # 2. 执行前向传播
            return self.model(*args, **kwargs)
        finally:
            # 3. 恢复
            if was_training:
                unwrapped.train()
            unwrapped.set_adapter(previous_adapter)

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
            project_name="Anima_LoRA_DPO",
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
    model, vae, qwen_model, qwen_tokenizer, t5_tokenizer, _ = load_model_and_tokenizer(config)

    # 应用 LoRA
    sft_lora_path = config['Model'].get('sft_lora_path', "")
    try:
        print(f"Loading Trainable LoRA (Policy) from: {sft_lora_path}")
        # 1. 加载用于训练的 Adapter (默认名称为 'default')
        model = PeftModel.from_pretrained(model, sft_lora_path, is_trainable=True)
        # 2. 再次加载相同的 LoRA 作为参考 Adapter (命名为 'reference'), 这样我们就在显存中只存了一份 Base Model，但有两份轻量级的 LoRA
        print(f"Loading Frozen LoRA (Reference) from: {sft_lora_path}")
        model.load_adapter(sft_lora_path, adapter_name="reference")
        model.set_adapter("default")
        model.print_trainable_parameters()
        model.to(accelerator.device)
    except Exception as e:
        print(f'{sft_lora_path} is not a valid PEFT LoRA directory path!')
        raise

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
    use_cache = config['Model'].get('use_cache', False)
    train_dataset = DPODataset(
        preference_json_path=config['Model']['preference_json'],
        caption_dropout_rate=config['Model'].get('caption_dropout_rate', 0.1), 
        real_ratio=config['Model'].get('real_ratio', 0.2), 
        use_cache=use_cache, 
        vae=vae,
        device=accelerator.device,
    )
    if use_cache:
        print("Cache generation complete. Unloading VAE from memory.")
        del vae
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        vae = None
    bucket_sampler = DPOBucketBatchSampler(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        seed=42,
        num_replicas=accelerator.num_processes,
        rank=accelerator.process_index
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_sampler=bucket_sampler,
        collate_fn=collate_fn,
        num_workers=config['Model'].get('dataloader_num_workers', 4)
    )

    # 优化器与调度器
    optimizer = setup_optimizer(model, config)
    scheduler, num_training_steps = setup_scheduler(optimizer, config, train_dataloader)

    # 梯度检查点
    if config['Model'].get('gradient_checkpointing', True):
        unwrapped_model = model
        if hasattr(unwrapped_model, "base_model"): 
            unwrapped_model = unwrapped_model.base_model
        if hasattr(unwrapped_model, "model"):
            unwrapped_model = unwrapped_model.model
        enable_anima_gradient_checkpointing(unwrapped_model)
        logger.info("Gradient checkpointing enabled")

    # EMA 设置
    ema_model = None
    if config['Model'].get('use_ema', True):
        ema_model = EMAModel(model, decay=config['Model'].get('ema_decay', 0.999))

    # accelerator准备模型
    model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)
    ref_model = ReferenceModelWrapper(model, accelerator)
    if qwen_model is not None:
        qwen_model = qwen_model.to(accelerator.device)
        qwen_model.eval()
        qwen_model.requires_grad_(False)
    if vae is not None:
        vae.model = vae.model.to(accelerator.device)
        vae.scale = [s.to(device=accelerator.device, dtype=next(vae.model.parameters()).dtype) for s in vae.scale]
        vae.model.eval()
        vae.model.requires_grad_(False)

    # 训练检查点
    start_step = load_checkpoint(accelerator, model, optimizer, scheduler, config, ema_model)
    logger.info("Training started")
    global_step = start_step
    session_start_step = start_step
    
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
        accumulated_loss = 0.0
        micro_step_count = 0
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
                    qwen_model,
                    qwen_tokenizer,
                    t5_tokenizer,
                    transport, 
                    batch, 
                    accelerator.device, 
                    beta=config['Model']['beta'],
                    mu=config['Model']['mu'],
                    dmpo_alpha=config['Model']['dmpo_alpha'],
                )
                epoch_losses.append(loss.item())
                accumulated_loss += loss.item()
                micro_step_count += 1
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    if max_grad_norm > 0:
                        accelerator.clip_grad_norm_(model.parameters(), max_grad_norm)

                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                    if ema_model is not None:
                        ema_model.step(accelerator.unwrap_model(model))

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                avg_step_loss = accumulated_loss / max(micro_step_count, 1)
                if accelerator.is_main_process:
                    accelerator.log({"loss": avg_step_loss, "learning_rate": scheduler.get_last_lr()[0]}, step=global_step)
                    
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
                    logger.info(f"Epoch {epoch+1}/{config['Model']['num_epochs']}, Step {global_step}/{num_training_steps}, Loss {avg_step_loss:.4f}, LR {scheduler.get_last_lr()[0]:.7f}, Speed {steps_per_sec:.2f} steps/s")

                if accelerator.is_main_process and global_step % 1000 == 0:
                    save_checkpoint(accelerator, model, optimizer, scheduler, global_step, config, ema_model)

                accumulated_loss = 0.0
                micro_step_count = 0

        avg_epoch_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0.0
        logger.info(f"Epoch {epoch+1}/{config['Model']['num_epochs']} completed - Average Loss: {avg_epoch_loss:.4f}")

        if accelerator.is_main_process and save_epochs_interval > 0 and (epoch + 1) % save_epochs_interval == 0:
            save_checkpoint(accelerator, model, optimizer, scheduler, global_step, config, ema_model)
            logger.info(f"Checkpoint saved at epoch {epoch+1}")

    logger.info("Training complete, saving final model")
    save_lora_model(accelerator, model, config, step=None, ema_model=ema_model)

    if accelerator.is_main_process:
        accelerator.end_training()

    logger.info("Training finished")


if __name__ == "__main__":
    main()
