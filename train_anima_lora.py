#!/usr/bin/env python3
"""Anima LoRA 训练器 - 基于 Rectified Flow 的 LoRA 微调"""

import os
import sys
import argparse
import toml
import json
import logging
import random
import math
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import set_seed
from diffusers import AutoencoderKL
from diffusers.optimization import get_scheduler
from transformers import AutoTokenizer, AutoModel, AutoConfig
from peft import LoraConfig, get_peft_model, PeftModel, get_peft_model_state_dict, set_peft_model_state_dict
from safetensors.torch import load_file, save_file
from tqdm import tqdm
import re
import random
from dataset import ImageCaptionDataset, BucketBatchSampler, collate_fn

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "AnimaLoraToolkit"))
from transport import create_transport
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

try:
    import bitsandbytes as bnb
except ImportError:
    logging.warning("bitsandbytes not available, 8-bit optimizer disabled")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("newbie_lora_trainer")

class EMAModel:
    """
    EMA 模型
    """
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        for name, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[name] = p.clone().detach()

    def step(self, model):
        # 遍历模型的当前参数
        for name, p in model.named_parameters():
            if p.requires_grad:
                if name in self.shadow:
                    new_average = (1.0 - self.decay) * p.detach() + self.decay * self.shadow[name]
                    self.shadow[name] = new_average.clone()

    def copy_to(self, model):
        """将 EMA 权重应用到模型"""
        for name, p in model.named_parameters():
            if p.requires_grad and name in self.shadow:
                self.backup[name] = p.data.clone()
                p.data.copy_(self.shadow[name].data)

    def restore(self, model):
        """恢复备份的原始权重"""
        for name, p in model.named_parameters():
            if name in self.backup:
                p.data.copy_(self.backup[name])
        self.backup = {}
    
    def state_dict(self):
        return self.shadow
    
    def load_state_dict(self, state_dict):
        for name, p in state_dict.items():
            if name in self.shadow:
                self.shadow[name].copy_(p)
            else:
                self.shadow[name] = p.clone().detach()

    def to(self, device):
        for name in self.shadow:
            self.shadow[name] = self.shadow[name].to(device)
        return self
    
def load_encoders_only(config):
    mixed_precision = config['Model'].get('mixed_precision', 'no')
    model_dtype = torch.bfloat16 if mixed_precision == 'bf16' else (torch.float16 if mixed_precision == 'fp16' else torch.float32)
    device = "cpu"
    repo_root = Path(config['Model'].get('anima_repo_root', '.'))
    
    logger.info("Loading Anima encoders (VAE, Qwen, T5)...")
    
    vae = load_vae(config['Model']['vae_path'], device, model_dtype, repo_root)
    vae.scale = [s.to(device=device, dtype=model_dtype) for s in vae.scale]
    
    qwen_model, qwen_tokenizer, t5_tokenizer = load_text_encoders(
        config['Model']['qwen_model_path'], 
        config['Model'].get('t5_tokenizer_path', ''), 
        device, 
        model_dtype
    )
    
    logger.info("Anima Encoders loaded successfully")
    return vae, qwen_model, qwen_tokenizer, t5_tokenizer, None

def load_transformer_only(config):
    mixed_precision = config['Model'].get('mixed_precision', 'no')
    model_dtype = torch.bfloat16 if mixed_precision == 'bf16' else (torch.float16 if mixed_precision == 'fp16' else torch.float32)
    device = "cpu"
    repo_root = Path(config['Model'].get('anima_repo_root', '.'))
    transformer_path = config['Model']['transformer_path']
    
    logger.info(f"Loading Anima transformer from: {transformer_path}")
    
    model = load_anima_model(transformer_path, device, model_dtype, repo_root)
    model.train()
    
    logger.info("Anima Transformer loaded.")
    return model

def load_model_and_tokenizer(config):
    mixed_precision = config['Model'].get('mixed_precision', 'no')
    model_dtype = torch.bfloat16 if mixed_precision == 'bf16' else (torch.float16 if mixed_precision == 'fp16' else torch.float32)
    device = "cpu"
    repo_root = Path(config['Model'].get('anima_repo_root', '.'))
    
    logger.info("Loading full Anima model pipeline...")
    
    model = load_transformer_only(config)
    vae = load_vae(config['Model']['vae_path'], device, model_dtype, repo_root)
    vae.scale = [s.to(device=device, dtype=model_dtype) for s in vae.scale]
    qwen_model, qwen_tokenizer, t5_tokenizer = load_text_encoders(
        config['Model']['qwen_model_path'], 
        config['Model'].get('t5_tokenizer_path', ''), 
        device, 
        model_dtype
    )
    
    return model, vae, qwen_model, qwen_tokenizer, t5_tokenizer, None

def setup_lora(model, config):
    """Apply adapter (PEFT) to model"""
    # 获取配置参数
    lora_rank = config['Model'].get('lora_rank', 32)
    lora_alpha = config['Model'].get('lora_alpha', lora_rank)
    lora_dropout = config['Model'].get('lora_dropout', 0.05)
    use_dora=config['Model'].get('use_dora', False)
    use_rslora=config['Model'].get('use_rslora', False)
    train_norm=config['Model'].get('train_norm', False)
    resume_lora_path = config['Model'].get('resume_from_lora', None)
    rank_pattern = config['Model'].get('lora_rank_pattern', {})
    alpha_pattern = config['Model'].get('lora_alpha_pattern', {})
    
    if resume_lora_path:
        peft_model = PeftModel.from_pretrained(model, resume_lora_path, is_trainable=True)
        logger.info(f"load LoRA weights from {resume_lora_path}")
    else: 
        # 获取目标模块
        default_target_modules = [
            "q_proj", 
            "k_proj", 
            "v_proj", 
            "output_proj",
            "mlp.layer1",
            "mlp.layer2",
        ]
        
        target_modules = config['Model'].get('lora_target_modules') or default_target_modules
    
        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type=None,
            use_dora=use_dora,
            use_rslora=use_rslora,
            rank_pattern=rank_pattern,
            alpha_pattern=alpha_pattern,
            exclude_modules="llm_adapter.*",
        )
        
        peft_model = get_peft_model(model, lora_config, low_cpu_mem_usage=False)
    
        peft_model._adapter_type = "lora"
        peft_model._adapter_rank = lora_rank
        peft_model._adapter_alpha = lora_alpha

        logger.info(f"  Target modules: {target_modules}")
        logger.info(f"  LoRA rank={lora_rank}, alpha={lora_alpha}")

    trainable_params = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in peft_model.parameters())
    logger.info(
        f"LoRA applied: {trainable_params/1e6:.2f}M/{total_params/1e6:.2f}M trainable "
        f"({trainable_params/total_params*100:.2f}%)"
    )

    return peft_model


def setup_optimizer(model, config):
    """Configure optimizer (supports LoRA/LyCORIS)"""
    optimizer_type = config['Optimization']['optimizer_type']
    learning_rate = config['Model']['learning_rate']
    adapter_type = getattr(model, "_adapter_type", "lora")
    weight_decay = config['Optimization'].get('weight_decay', 0.01)

    trainable_params = [p for p in model.parameters() if p.requires_grad]

    adam_kwargs = {"lr": learning_rate, "betas": (0.9, 0.999), "eps": 1e-8, "weight_decay": weight_decay}

    if optimizer_type == "AdamW8bit":
        try:
            optimizer = bnb.optim.AdamW8bit(trainable_params, **adam_kwargs)
            logger.info("Using 8-bit AdamW optimizer")
        except Exception as e:
            logger.warning(f"8-bit AdamW failed, using standard AdamW: {e}")
            optimizer = optim.AdamW(trainable_params, **adam_kwargs)
    else:
        optimizer = optim.AdamW(trainable_params, **adam_kwargs)
        logger.info("Using standard AdamW optimizer")

    return optimizer


def setup_scheduler(optimizer, config, train_dataloader):
    """设置学习率调度器"""
    num_epochs = config['Model']['num_epochs']
    gradient_accumulation_steps = config['Model'].get('gradient_accumulation_steps', 1)
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)
    num_training_steps = num_epochs * num_update_steps_per_epoch
    scheduler_type = config['Model']['lr_scheduler']
    lr_warmup_steps = config['Model'].get('lr_warmup_steps', 100)

    scheduler = get_scheduler(
        name=scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=lr_warmup_steps,
        num_training_steps=num_training_steps
    )

    logger.info(f"LR scheduler: {scheduler_type}, {num_training_steps} steps, {lr_warmup_steps} warmup")

    return scheduler, num_training_steps


def generate_noise(batch_size, num_channels, height, width, device):
    """生成随机噪声"""
    return torch.randn((batch_size, num_channels, height, width), device=device)


def print_memory_usage(stage_name, profiler_enabled=False):
    """打印显存使用情况"""
    if not profiler_enabled:
        return

    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        max_allocated = torch.cuda.max_memory_allocated() / 1024**3

        logger.info(f"=" * 80)
        logger.info(f"GPU Memory at [{stage_name}]")
        logger.info(f"  Allocated:     {allocated:.2f} GB")
        logger.info(f"  Reserved:      {reserved:.2f} GB")
        logger.info(f"  Max Allocated: {max_allocated:.2f} GB")
        logger.info(f"=" * 80)


def get_tensors_summary(profiler_enabled=False):
    """获取所有 CUDA 张量的摘要"""
    if not profiler_enabled or not torch.cuda.is_available():
        return

    import gc
    tensors = []
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) and obj.is_cuda:
                tensors.append((type(obj).__name__, obj.size(), obj.dtype, obj.element_size() * obj.nelement() / 1024**2))
        except:
            pass

    # 按大小排序
    tensors.sort(key=lambda x: x[3], reverse=True)

    logger.info(f"\n{'='*80}")
    logger.info(f"Top 20 Largest Tensors in GPU Memory:")
    logger.info(f"{'Type':<20} {'Shape':<30} {'Dtype':<15} {'Size (MB)':<10}")
    logger.info(f"{'-'*80}")

    for i, (ttype, shape, dtype, size_mb) in enumerate(tensors[:20]):
        logger.info(f"{ttype:<20} {str(shape):<30} {str(dtype):<15} {size_mb:>10.2f}")

    total_size = sum(t[3] for t in tensors)
    logger.info(f"{'-'*80}")
    logger.info(f"Total tensor size: {total_size:.2f} MB ({len(tensors)} tensors)")
    logger.info(f"{'='*80}\n")

def enable_anima_gradient_checkpointing(anima_model):
    """为 Anima 架构动态注入支持梯度检查点的前向传播方法"""
    import types
    from torch.utils.checkpoint import checkpoint

    def checkpointed_forward(
        self,
        x_B_C_T_H_W: torch.Tensor,
        timesteps_B_T: torch.Tensor,
        crossattn_emb: torch.Tensor,
        fps=None,
        padding_mask=None,
    ):
        # 1. 准备序列与位置编码
        x_B_T_H_W_D, rope_emb_L_1_1_D, extra_pos_emb = self.prepare_embedded_sequence(
            x_B_C_T_H_W,
            fps=fps,
            padding_mask=padding_mask,
        )

        if timesteps_B_T.ndim == 1:
            timesteps_B_T = timesteps_B_T.unsqueeze(1)
        
        t_embedding_B_T_D, adaln_lora_B_T_3D = self.t_embedder(timesteps_B_T)
        t_embedding_B_T_D = self.t_embedding_norm(t_embedding_B_T_D)

        block_kwargs = {
            "rope_emb_L_1_1_D": rope_emb_L_1_1_D,
            "adaln_lora_B_T_3D": adaln_lora_B_T_3D,
            "extra_per_block_pos_emb": extra_pos_emb,
        }

        # 2. 逐层应用 torch.utils.checkpoint
        for block in self.blocks:
            def custom_forward(x, blk=block):
                return blk(x, t_embedding_B_T_D, crossattn_emb, **block_kwargs)
            
            # 使用 use_reentrant=False 是 PyTorch 新版本的推荐做法，能更好兼容 LoRA
            x_B_T_H_W_D = checkpoint(custom_forward, x_B_T_H_W_D, use_reentrant=False)

        # 3. 输出层处理
        x_B_T_H_W_O = self.final_layer(x_B_T_H_W_D, t_embedding_B_T_D, adaln_lora_B_T_3D=adaln_lora_B_T_3D)
        x_B_C_Tt_Hp_Wp = self.unpatchify(x_B_T_H_W_O)
        
        return x_B_C_Tt_Hp_Wp

    # 将原有 forward 方法替换为支持 checkpoint 的版本
    anima_model.forward = types.MethodType(checkpointed_forward, anima_model)
    anima_model.gradient_checkpointing = True

def compute_loss(model, vae, qwen_model, qwen_tokenizer, t5_tokenizer, transport, batch, device, gemma3_prompt=""):
    """计算 Rectified Flow 训练损失"""
    if batch.get("cached", False):
        latents = batch["latents"].to(device).unsqueeze(2)
        captions = batch["captions"]
    else:
        if vae is None:
            raise RuntimeError("VAE required for non-cached data")
        pixel_values = batch["pixel_values"].to(device)
        captions = batch["captions"]
        with torch.no_grad():
            vae_dtype = next(vae.model.parameters()).dtype
            pixel_values_5d = pixel_values.unsqueeze(2).to(dtype=vae_dtype)
            vae_scale = [s.to(device=pixel_values_5d.device) for s in vae.scale]
            latents = vae.model.encode(pixel_values_5d, vae_scale)

    bs = latents.shape[0]
    
    with torch.no_grad():
        # 处理 Qwen 文本 (提取纯净标签)
        qwen_texts = [_build_qwen_text_from_prompt(cap) for cap in captions]
        # 添加全局 prompt 前缀（如果配置了 gemma3_prompt）
        qwen_texts = [(gemma3_prompt + cap) if (gemma3_prompt and cap) else cap for cap in qwen_texts]
        qwen_embeds, qwen_attn = encode_qwen(qwen_model, qwen_tokenizer, qwen_texts, device)
        # 处理 T5 文本 (保留权重语法)
        t5_texts = [(gemma3_prompt + cap) if (gemma3_prompt and cap) else cap for cap in captions]
        t5_ids, t5_attn, t5_w = tokenize_t5_weighted(t5_tokenizer, t5_texts, max_length=512)
        t5_ids = t5_ids.to(device)
        # LLMAdapter 桥接
        unwrapped_model = model.module if hasattr(model, "module") else model
        cross = unwrapped_model.preprocess_text_embeds(qwen_embeds, t5_ids)
        if cross.shape[1] < 1024:
            cross = torch.nn.functional.pad(cross, (0, 0, 0, 1024 - cross.shape[1]))
        
    pad_mask = torch.zeros(bs, 1, latents.shape[-2], latents.shape[-1], device=device, dtype=latents.dtype)
    model_kwargs = dict(crossattn_emb=cross, padding_mask=pad_mask)

    ############ 损失计算 ############
    # 原始分辨率损失
    loss = transport.training_losses(model, latents, model_kwargs)["loss"].mean()
    return loss


def save_checkpoint(accelerator, model, optimizer, scheduler, step, config, ema_model=None):
    """Save training checkpoint"""
    checkpoint_dir = os.path.join(config['Model']['output_dir'], "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{step}.pt")
    unwrapped = accelerator.unwrap_model(model)
    adapter_type = getattr(unwrapped, "_adapter_type", "lora")

    adapter_state = {
        k: v.detach().cpu()
        for k, v in get_peft_model_state_dict(unwrapped).items()
    }
    checkpoint = {
        "step": step,
        "adapter_type": adapter_type,      
        "adapter_state_dict": adapter_state,        
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
    }
    # 保存 EMA 状态
    if ema_model is not None:
        checkpoint["ema_state_dict"] = ema_model.state_dict()
    accelerator.save(checkpoint, checkpoint_path)
    logger.info(f"Checkpoint saved: {checkpoint_path}")
    save_lora_model(accelerator, model, config, step, ema_model)


def save_lora_model(accelerator, model, config, step=None, ema_model=None):
    """Save adapter weights (PEFT format & ComfyUI compatible format)"""
    output_dir = config['Model']['output_dir']
    output_name = config['Model']['output_name']
    os.makedirs(output_dir, exist_ok=True)

    def _save_implementation(model_to_save, suffix=""):
        save_dir = os.path.join(output_dir, f"{output_name}_step_{step}{suffix}" if step else f"{output_name}{suffix}")
        os.makedirs(save_dir, exist_ok=True)

        unwrapped = accelerator.unwrap_model(model_to_save)
        lora_state_dict = get_peft_model_state_dict(unwrapped)
            
        if accelerator.is_main_process:
            # --- A. 保存标准 PEFT 格式 ---
            unwrapped.save_pretrained(
                save_dir,
                is_main_process=accelerator.is_main_process,
                safe_serialization=True,
            )
            logger.info(f"PEFT LoRA model{suffix} saved to: {save_dir}")

            # --- B. 保存 ComfyUI 兼容格式 ---
            comfy_state_dict = {}
            for key, value in lora_state_dict.items():
                if key.endswith(".lora_magnitude_vector"):
                    key = key.replace(".lora_magnitude_vector", ".dora_scale")
                    if len(value.shape) == 1:
                        value = value.unsqueeze(1)
                if key.startswith("base_model.model."):
                    key = "diffusion_model." + key[len("base_model.model."):]
                if "lora_A.weight" in key:
                    key = key.replace("lora_A.weight", "lora_down.weight")
                if "lora_B.weight" in key:
                    key = key.replace("lora_B.weight", "lora_up.weight")
                comfy_state_dict[key] = value
            
            comfy_filename = f"{output_name}_step_{step}{suffix}.safetensors" if step else f"{output_name}{suffix}.safetensors"
            comfy_path = os.path.join(output_dir, comfy_filename)
            save_file(comfy_state_dict, comfy_path)
            logger.info(f"ComfyUI compatible LoRA{suffix} saved to: {comfy_path}")

    # 1. 保存普通模型
    _save_implementation(model, suffix="")

    # 2. 保存 EMA 模型 (如果有)
    if ema_model is not None:
        logger.info("Saving EMA weights...")
        unwrapped_model = accelerator.unwrap_model(model)
        ema_model.copy_to(unwrapped_model)
        try:
            _save_implementation(model, suffix="_ema")
        finally:
            ema_model.restore(unwrapped_model)

def load_checkpoint(accelerator, model, optimizer, scheduler, config, ema_model=None):
    checkpoint_dir = os.path.join(config['Model']['output_dir'], "checkpoints")
    if not os.path.exists(checkpoint_dir):
        logger.info("No checkpoint found, starting from scratch")
        return 0
    checkpoints = [
        f for f in os.listdir(checkpoint_dir)
        if f.startswith("checkpoint_") and f.endswith(".pt")
    ]
    if not checkpoints:
        logger.info("No checkpoint files found, starting from scratch")
        return 0
    checkpoints.sort(key=lambda x: int(x.split('_')[1].split('.')[0]), reverse=True)
    checkpoint_path = os.path.join(checkpoint_dir, checkpoints[0])
    logger.info(f"Loading checkpoint: {checkpoint_path}")
    unwrapped = accelerator.unwrap_model(model)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    adapter_type = checkpoint.get("adapter_type", getattr(unwrapped, "_adapter_type", "lora"))
    adapter_state = checkpoint.get("adapter_state_dict") or checkpoint.get("lora_state_dict")
    if adapter_state is None:
        raise RuntimeError("No adapter_state_dict found in checkpoint")
    else:
        set_peft_model_state_dict(unwrapped, adapter_state)
        logger.info("Loaded PEFT LoRA state dict from checkpoint")
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    # 恢复 EMA 状态
    if ema_model is not None:
        if "ema_state_dict" in checkpoint:
            ema_model.load_state_dict(checkpoint["ema_state_dict"])
            ema_model.to(accelerator.device)
            logger.info("Loaded EMA state from checkpoint")
        else:
            logger.warning("EMA enabled but no EMA state found in checkpoint!")
    logger.info(f"Resumed from step {checkpoint['step']}")
    return checkpoint["step"]


def main():
    """主训练函数"""
    parser = argparse.ArgumentParser(description="Newbie LoRA Trainer")
    parser.add_argument("--config_file", type=str, required=True, help="Path to .toml config file")
    parser.add_argument("--profiler", action="store_true", help="Enable GPU memory profiling")
    args = parser.parse_args()

    with open(args.config_file, 'r', encoding='utf-8') as f:
        config = toml.load(f)
    
    import models
    from pathlib import Path
    repo_root = str(Path(config['Model'].get('anima_repo_root', '.')).resolve())
    if repo_root not in [str(Path(p).resolve()) for p in models.__path__]:
        models.__path__.append(repo_root)
    
    # 若配置缺少 Optimization 段，补默认值避免 KeyError
    opt_cfg = config.setdefault('Optimization', {})
    opt_cfg.setdefault('optimizer_type', 'AdamW')
    opt_cfg.setdefault('use_flash_attention_2', False)

    gradient_accumulation_steps = config['Model'].get('gradient_accumulation_steps', 1)

    output_dir = config['Model']['output_dir']    
    os.makedirs(output_dir, exist_ok=True)

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
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
            project_name="Anima_LoRA",
            config=tracker_config,
            init_kwargs={"wandb": {"name": config['Model'].get('output_name', 'run')}} # 设置本次运行的名称
        )

    set_seed(42)

    if not config['Optimization'].get('use_flash_attention_2', False):
        torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False)

    use_cache = config['Model'].get('use_cache', True)
    mixed_precision = config['Model'].get('mixed_precision', 'no')
    cache_dtype = torch.bfloat16 if mixed_precision == 'bf16' else (torch.float16 if mixed_precision == 'fp16' else torch.float32)
    gemma3_prompt = config['Model'].get('gemma3_prompt', '')
    resolution = config['Model']['resolution']
    min_bucket_reso = config['Model'].get('min_bucket_reso', 256)
    max_bucket_reso = config['Model'].get('max_bucket_reso', 2048)
    bucket_reso_step = config['Model'].get('bucket_reso_step', 64)
    shuffle_caption = config['Model'].get('shuffle_caption', False)
    keep_tokens_separator = config['Model'].get('keep_tokens_separator', "|||")
    enable_wildcard = config['Model'].get('enable_wildcard', False)
    caption_dropout_rate = config['Model'].get('caption_dropout_rate', 0.0)
    caption_tag_dropout_rate = config['Model'].get('caption_tag_dropout_rate', 0.0)
    drop_artist_rate = config['Model'].get('drop_artist_rate', 0.0)

    if use_cache:
        logger.info("Checking if VAE cache files exist...")
        train_data_dir = config['Model']['train_data_dir']
        image_paths = []
        for root, _, files in os.walk(train_data_dir):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                    image_paths.append(os.path.join(root, file))

        cache_complete = True
        for image_path in image_paths:
            vae_cache = f"{image_path}.safetensors"
            #text_cache = f"{os.path.splitext(image_path)[0]}.txt.safetensors"
            if not os.path.exists(vae_cache): # or not os.path.exists(text_cache):
                cache_complete = False
                break

        logger.info("Loading encoders...")
        vae, qwen_model, qwen_tokenizer, t5_tokenizer, _ = load_encoders_only(config)

        if not cache_complete and accelerator.is_main_process:
            logger.info("Cache incomplete, generating VAE latents...")
            dataset = ImageCaptionDataset(
                train_data_dir=train_data_dir,
                resolution=resolution,
                enable_bucket=config['Model'].get('enable_bucket', True),
                use_cache=True,
                vae=vae,
                text_encoder=None,
                tokenizer=None,
                clip_model=None,
                clip_tokenizer=None,
                device=accelerator.device,
                dtype=cache_dtype,
                gemma3_prompt=gemma3_prompt,
                min_bucket_reso=min_bucket_reso,
                max_bucket_reso=max_bucket_reso,
                bucket_reso_step=bucket_reso_step,
            )
            del dataset
            import gc
            gc.collect()
            torch.cuda.empty_cache()
        accelerator.wait_for_everyone()
        
        logger.info("Unloading VAE to save memory (using cached latents)...")
        del vae
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        vae = None

        logger.info("Loading NextDiT model for training...")
        model = load_transformer_only(config)
        print_memory_usage("After loading transformer", args.profiler)

        dataset = ImageCaptionDataset(
            train_data_dir=train_data_dir,
            resolution=resolution,
            enable_bucket=config['Model'].get('enable_bucket', True),
            use_cache=True,
            vae=None,
            text_encoder=None,
            tokenizer=None,
            clip_model=None,
            clip_tokenizer=None,
            device=accelerator.device,
            dtype=cache_dtype,
            gemma3_prompt=gemma3_prompt,
            min_bucket_reso=min_bucket_reso,
            max_bucket_reso=max_bucket_reso,
            bucket_reso_step=bucket_reso_step,
            shuffle_caption=shuffle_caption,
            keep_tokens_separator=keep_tokens_separator,
            enable_wildcard=enable_wildcard,
            caption_dropout_rate=caption_dropout_rate,
            caption_tag_dropout_rate=caption_tag_dropout_rate,
            drop_artist_rate=drop_artist_rate,
        )
    else:
        model, vae, qwen_model, qwen_tokenizer, t5_tokenizer, _ = load_model_and_tokenizer(config)

        dataset = ImageCaptionDataset(
            train_data_dir=config['Model']['train_data_dir'],
            resolution=resolution,
            enable_bucket=config['Model'].get('enable_bucket', True),
            use_cache=False,
            vae=None,
            text_encoder=None,
            tokenizer=None,
            clip_model=None,
            clip_tokenizer=None,
            device=accelerator.device,
            dtype=cache_dtype,
            gemma3_prompt=gemma3_prompt,
            min_bucket_reso=min_bucket_reso,
            max_bucket_reso=max_bucket_reso,
            bucket_reso_step=bucket_reso_step,
            shuffle_caption=shuffle_caption,
            keep_tokens_separator=keep_tokens_separator,
            enable_wildcard=enable_wildcard,
            caption_dropout_rate=caption_dropout_rate,
            caption_tag_dropout_rate=caption_tag_dropout_rate,
            drop_artist_rate=drop_artist_rate,
        )

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

    num_workers = config['Model'].get('dataloader_num_workers', 4)
    batch_size = config['Model']['train_batch_size']

    if config['Model'].get('enable_bucket', True):
        batch_sampler = BucketBatchSampler(
            dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            seed=42,
            num_replicas=accelerator.num_processes,
            rank=accelerator.process_index
        )
        train_dataloader = DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True if num_workers > 0 else False,
            prefetch_factor=2 if num_workers > 0 else None
        )
    else:
        train_dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True if num_workers > 0 else False,
            prefetch_factor=2 if num_workers > 0 else None
        )
        
    print_memory_usage("Before LoRA", args.profiler)
    model = setup_lora(model, config)
    model.to(accelerator.device)
    print_memory_usage("After LoRA", args.profiler)
    
    if config['Model'].get('gradient_checkpointing', True):
        unwrapped_model = model
        if hasattr(unwrapped_model, "base_model"): 
            unwrapped_model = unwrapped_model.base_model
        if hasattr(unwrapped_model, "model"):
            unwrapped_model = unwrapped_model.model
        enable_anima_gradient_checkpointing(unwrapped_model)
        logger.info("Gradient checkpointing enabled")
    
    optimizer = setup_optimizer(model, config)
    scheduler, num_training_steps = setup_scheduler(optimizer, config, train_dataloader)

    print_memory_usage("Before accelerator.prepare", args.profiler)
    if config['Model'].get('enable_bucket', True):
        model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)
    else:
        model, optimizer, scheduler, train_dataloader = accelerator.prepare(model, optimizer, scheduler, train_dataloader)
    print_memory_usage("After accelerator.prepare", args.profiler)

    # 初始化 EMA
    ema_model = None
    if config['Model'].get('use_ema', False):
        ema_decay = config['Model'].get('ema_decay', 0.999)
        if accelerator.is_main_process:
            logger.info(f"Initializing EMA with decay: {ema_decay}")
            ema_model = EMAModel(accelerator.unwrap_model(model), decay=ema_decay)
    
    """
    # Do NOT prepare encoders - they should stay frozen and not be wrapped
    if not use_cache:
        vae = vae.to(accelerator.device)
        text_encoder = text_encoder.to(accelerator.device)
        clip_model = clip_model.to(accelerator.device)
        vae.eval()
        text_encoder.eval()
        clip_model.eval()
        vae.requires_grad_(False)
        text_encoder.requires_grad_(False)
        clip_model.requires_grad_(False)
        print_memory_usage("After loading encoders (no cache)", args.profiler)
    """
    if qwen_model is not None:
        qwen_model = qwen_model.to(accelerator.device)
        qwen_model.eval()
        qwen_model.requires_grad_(False)
    if vae is not None:
        vae.model = vae.model.to(accelerator.device)
        vae.model.eval()
        vae.model.requires_grad_(False)

    start_step = load_checkpoint(accelerator, model, optimizer, scheduler, config, ema_model)

    if args.profiler:
        get_tensors_summary(args.profiler)

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
        
    has_profiled_micro_step = False
    for epoch in range(start_epoch, config['Model']['num_epochs']):
        if config['Model'].get('enable_bucket', True) and hasattr(train_dataloader, 'batch_sampler'):
            if hasattr(train_dataloader.batch_sampler, 'set_epoch'):
                train_dataloader.batch_sampler.set_epoch(epoch)

        epoch_losses = []
        accumulated_loss = 0.0
        micro_step_count = 0
        progress_bar = tqdm(
            train_dataloader,
            desc=f"Epoch {epoch+1}/{config['Model']['num_epochs']}",
            disable=not accelerator.is_main_process
        )
        
        for batch_idx, batch in enumerate(progress_bar):
            if epoch == start_epoch and batch_idx < steps_to_skip_in_first_epoch:
                continue

            # 判断是否是用于 Profiling 的第一个全局步（通常是第0步或断点续训的起始步）
            is_profiling_step = (global_step == start_step) and args.profiler

            with accelerator.accumulate(model):
                # ================= Profiler: Before Forward =================
                # 仅在当前累积周期的第一个微步打印
                if is_profiling_step and not has_profiled_micro_step:
                    print_memory_usage("Before first forward pass", args.profiler)

                # 1. 计算 Loss
                loss = compute_loss(
                    model, 
                    vae, 
                    qwen_model,
                    qwen_tokenizer,
                    t5_tokenizer,
                    transport, 
                    batch, 
                    accelerator.device, 
                    gemma3_prompt
                )
                
                # 记录 Loss
                epoch_losses.append(loss.item())
                accumulated_loss += loss.item()
                micro_step_count += 1

                # ================= Profiler: After Forward =================
                if is_profiling_step and not has_profiled_micro_step:
                    print_memory_usage("After first forward pass", args.profiler)

                # 2. 反向传播
                accelerator.backward(loss)

                # ================= Profiler: After Backward =================
                if is_profiling_step and not has_profiled_micro_step:
                    print_memory_usage("After first backward pass", args.profiler)
                    # 标记已完成微步的 Profiling，避免在同一个累积周期内的后续微步重复打印
                    has_profiled_micro_step = True 

                # 3. 梯度裁剪、优化器步进 (仅在同步步执行)
                if accelerator.sync_gradients:
                    if max_grad_norm > 0:
                        accelerator.clip_grad_norm_(model.parameters(), max_grad_norm)

                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                    if ema_model is not None:
                        ema_model.step(accelerator.unwrap_model(model))

                    # ================= Profiler: After Optimizer & Exit =================
                    # 仅在第一个完整的 Update Step 结束后执行
                    if is_profiling_step:
                        print_memory_usage("After first optimizer step", args.profiler)
                        get_tensors_summary(args.profiler)
                        logger.info("Profiling complete for first step. You can now Ctrl+C to stop.")
                        sys.exit(0)

            # 4. 更新 Global Step 和 日志 (仅在同步步执行)
            if accelerator.sync_gradients:
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

                if accelerator.is_main_process and global_step % 100 == 0:
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
