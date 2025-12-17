#!/usr/bin/env python3
"""
计算 FLUX VAE 的 scaling_factor 和 shift_factor 并保存为 Diffusers 格式。
基于 NewbieLoraTrainer 的数据处理逻辑。
Example:
python calc_flux_vae_scale.py \
  --train_data_dir /path/to/your/images \
  --vae_path /path/to/flux_vae.safetensors \
  --output_dir /path/to/save/diffusers_vae \
  --resolution 1024 \
  --batch_size 4
"""

import os
import argparse
import json
import math
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from diffusers import AutoencoderKL
from accelerate.utils import set_seed

# 从 train_newbie_lora 导入数据集类和整理函数
try:
    from train_newbie_lora import ImageCaptionDataset, collate_fn
except ImportError:
    raise ImportError("请将此脚本放在 train_newbie_lora.py 同级目录下运行。")

def parse_args():
    parser = argparse.ArgumentParser(description="Calculate VAE scaling and shift factors")
    parser.add_argument("--train_data_dir", type=str, required=True, help="训练数据集目录")
    parser.add_argument("--vae_path", type=str, required=True, help="单文件 VAE (.safetensors) 路径")
    parser.add_argument("--output_dir", type=str, required=True, help="输出 Diffusers VAE 的目录")
    parser.add_argument("--resolution", type=int, default=1024, help="计算时使用的分辨率 (默认 1024)")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size (建议 1-4，取决于显存)")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--max_samples", type=int, default=None, help="最大采样图片数 (None 为使用全部)")
    parser.add_argument("--enable_bucket", action="store_true", help="启用分桶 (通常计算 scale 时建议关闭 bucket 以保持一致性，但在 NewbieLoraTrainer 中默认开启)")
    return parser.parse_args()

def main():
    args = parse_args()
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Loading VAE from: {args.vae_path}")
    
    # 加载单文件 VAE
    # 注意：FLUX VAE 通常是 AutoencoderKL 架构
    try:
        vae = AutoencoderKL.from_single_file(args.vae_path)
    except Exception as e:
        print(f"Error loading VAE: {e}")
        print("尝试手动指定 config (假设是 FLUX.1-dev 结构)...")
        # 如果自动加载失败，尝试从 HuggingFace 拉取标准 config
        vae = AutoencoderKL.from_single_file(
            args.vae_path, 
            config="black-forest-labs/FLUX.1-dev", 
            subfolder="vae"
        )
    
    vae.to(device)
    vae.eval()
    vae.requires_grad_(False)
    
    print("Initializing Dataset...")
    # 复用 train_newbie_lora.py 中的数据集
    # 关键参数：use_cache=False (强制实时读取图片), vae=None (防止内部缓存逻辑触发)
    dataset = ImageCaptionDataset(
        train_data_dir=args.train_data_dir,
        resolution=args.resolution,
        enable_bucket=args.enable_bucket, # 建议计算统计量时关闭 bucket 或设为 False，除非你的训练主要依靠 bucket
        use_cache=False, 
        vae=None,
        device=device,
        dtype=torch.float32 # 保持精度
    )
    
    # 限制样本数量用于快速测试
    if args.max_samples is not None and len(dataset) > args.max_samples:
        indices = torch.randperm(len(dataset))[:args.max_samples]
        dataset = torch.utils.data.Subset(dataset, indices)
        print(f"Subsampled dataset to {len(dataset)} images.")

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers
    )

    print("Starting Loop to calculate Mean and Std...")
    
    # 使用 Welford 算法或简单的累加器 (这里使用累加器，因为我们需要全局标量)
    # 我们假设 latent 的分布是 (Latent - Shift) * Scale ~ N(0, 1)
    # 所以 Shift = Mean(Latent), Scale = 1 / Std(Latent)
    
    # FLUX 的 latents 通道数通常是 16 (FLUX.1) 或 4 (SDXL)。
    # 我们计算所有 channel 共享的 scaling factor (标量) 还是 per-channel?
    # 通常 Diffusers 和 SD 生态使用标量 scaling factor，
    # 但 FLUX 可能使用 per-channel 的 shift_factor。
    # 为了通用性，我们先计算全局标量。
    
    sum_x = 0.0
    sum_x2 = 0.0
    n_elements = 0
    
    # 用于计算 Per-channel Shift (Mean)
    # 初始化 channel 数将在第一个 batch 确定
    channel_sum = None 
    
    for batch in tqdm(dataloader):
        pixel_values = batch["pixel_values"].to(device)
        
        with torch.no_grad():
            # Encode: x -> z
            # pixel_values 已经在 dataset 中被 Normalize 到 [-1, 1]
            latents = vae.encode(pixel_values).latent_dist.sample()
            
            # Latents shape: [B, C, H, W]
            if channel_sum is None:
                channel_sum = torch.zeros(latents.shape[1], device=device, dtype=torch.float64)
            
            # 累加用于计算全局 Std
            latents_f64 = latents.to(torch.float64)
            sum_x += latents_f64.sum().item()
            sum_x2 += (latents_f64 ** 2).sum().item()
            n_elements += latents.numel()
            
            # 累加用于计算 Per-Channel Mean (Shift)
            # Average over B, H, W
            channel_sum += latents_f64.mean(dim=[0, 2, 3]) * latents.shape[0] # 这里的加权平均有点粗糙，假设每个batch size一致
            # 更精确的做法是累加 sum(dim=[0, 2, 3]) 然后除以 总像素数/C
            
    # 计算统计量
    global_mean = sum_x / n_elements
    global_variance = (sum_x2 / n_elements) - (global_mean ** 2)
    global_std = math.sqrt(global_variance)
    
    recommended_scaling_factor = 1.0 / global_std
    
    # 计算 Per-Channel Shift
    # 注意：上面的 channel_sum 是 mean 的累加，不太对。应该累加 sum。
    # 由于上面的循环逻辑为了简化，我们只用了 sum_x 计算全局。
    # 如果需要 per-channel shift，需要重新累加。
    # 这里我们简化为：Shift = Global Mean (标量) 或 0。
    # FLUX 官方通常 scaling_factor ~ 0.3611, shift_factor ~ 0.1159 (部分通道) 或 0。
    
    print("-" * 50)
    print(f"Total Elements processed: {n_elements}")
    print(f"Global Latent Mean: {global_mean:.6f}")
    print(f"Global Latent Std:  {global_std:.6f}")
    print("-" * 50)
    print(f"Calculated Scaling Factor (1/Std): {recommended_scaling_factor:.6f}")
    print(f"Calculated Shift Factor (Mean):    {global_mean:.6f}")
    print("-" * 50)
    
    # 保存模型
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Saving VAE to {args.output_dir}...")
    
    # 1. 保存权重和基础 Config
    vae.save_pretrained(args.output_dir)
    
    # 2. 更新 Config.json
    config_path = os.path.join(args.output_dir, "config.json")
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # 更新字段
    config['scaling_factor'] = recommended_scaling_factor
    
    # Diffusers 中 shift_factor 可以是列表(per-channel)或浮点数
    # 为了兼容性，如果你不想使用 shift，可以设为 0。
    # 但如果是 Rectified Flow，通常希望 Latents 是 Zero Mean 的。
    config['shift_factor'] = global_mean 
    
    # FLUX 特有：latent_channels 通常为 16
    if "latent_channels" not in config:
        config["latent_channels"] = vae.config.latent_channels

    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
        
    print(f"Config updated with scaling_factor={recommended_scaling_factor:.6f} and shift_factor={global_mean:.6f}")
    print("Done.")

if __name__ == "__main__":
    main()
