#!/usr/bin/env python3
"""
计算 FLUX VAE 的 scaling_factor 和 shift_factor 并保存为 Diffusers 格式。
基于 NewbieLoraTrainer 的数据处理逻辑。

功能：
1. 遍历数据集，通过 VAE 编码图像。
2. 统计 Latent 的全局均值 (Shift) 和标准差 (Std)。
3. 计算 Scaling Factor = 1 / Std。
4. 更新 VAE 的 config.json 并保存。

Example:
python calc_flux_vae_scale.py \
  --train_data_dir /root/autodl-tmp/datasets \
  --vae_path "https://huggingface.co/Anzhc/MS-LC-EQ-D-VR_VAE/blob/main/Pad Flux EQ v2 B1.safetensors" \
  --output_dir /root/autodl-tmp/diffusers_vae \
  --resolution 1024 \
  --batch_size 4 \
  --enable_bucket \
  --vae_reflect_padding
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

# 尝试从 train_newbie_lora 导入数据集类，如果失败则尝试相对路径
try:
    # [Fix] 增加导入 BucketBatchSampler
    from dataset import ImageCaptionDataset, collate_fn, BucketBatchSampler
except ImportError:
    raise ImportError("请确保 dataset.py 在同一目录下。")

def parse_args():
    parser = argparse.ArgumentParser(description="Calculate VAE scaling and shift factors for Flux (Global Only)")
    parser.add_argument("--train_data_dir", type=str, required=True, help="训练数据集目录")
    parser.add_argument("--vae_path", type=str, required=True, help="单文件 VAE (.safetensors) 路径或 Diffusers 模型目录")
    parser.add_argument("--output_dir", type=str, required=True, help="输出 Diffusers VAE 的目录")
    parser.add_argument("--resolution", type=int, default=1024, help="计算时使用的分辨率 (默认 1024)")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--max_samples", type=int, default=100, help="最大采样图片数 (None 为使用全部)")
    parser.add_argument("--enable_bucket", action="store_true", help="启用分桶 (计算统计量时建议关闭 bucket 以保持一致性，但也支持开启)")
    parser.add_argument("--vae_reflect_padding", action="store_true", help="VAE是否使用reflect_padding")
    return parser.parse_args()

def main():
    args = parse_args()
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Loading VAE from: {args.vae_path}")   

    try:
        if os.path.isdir(args.vae_path):
            vae = AutoencoderKL.from_pretrained(args.vae_path)
        else:
            vae = AutoencoderKL.from_single_file(
                args.vae_path, 
                config='black-forest-labs/FLUX.1-dev', 
                subfolder="vae"
            )
    except Exception as e:
        print(f"Error loading VAE: {e}")
        return

    # 设置 Padding 模式
    if args.vae_reflect_padding:
        print("Enabling 'reflect' padding mode for VAE layers...")
        for module in vae.modules():
            if isinstance(module, torch.nn.Conv2d):
                pad_h, pad_w = module.padding if isinstance(module.padding, tuple) else (module.padding, module.padding)
                if pad_h > 0 or pad_w > 0:
                    module.padding_mode = "reflect"

    vae.to(device)
    vae.eval()
    vae.requires_grad_(False)
    
    print("Initializing Dataset...")
    dataset = ImageCaptionDataset(
        train_data_dir=args.train_data_dir,
        resolution=args.resolution,
        enable_bucket=args.enable_bucket,
        use_cache=False, 
        vae=None,
        device=device,
        dtype=torch.float32 
    )
    
    if args.max_samples is not None and len(dataset) > args.max_samples:
        indices = torch.randperm(len(dataset))[:args.max_samples]
        dataset = torch.utils.data.Subset(dataset, indices)
        print(f"Subsampled dataset to {len(dataset)} images.")

    # [Fix] 这里的逻辑修改了：如果开启 bucket，必须使用 BucketBatchSampler
    if args.enable_bucket:
        print("Using BucketBatchSampler because bucketing is enabled.")
        # BucketBatchSampler 会自动处理 shuffle，这里为了统计可以设 shuffle=False (或者 True 也不影响均值计算)
        # 注意：BucketBatchSampler 初始化需要 dataset 本身（如果 dataset 被 Subset 包裹了，需要小心处理）
        # 但 dataset.py 里的 BucketBatchSampler 是直接访问 dataset.image_to_bucket 等属性的。
        # 如果使用了 Subset，Subset 对象没有 image_to_bucket 属性。
        
        # 针对 Subset 的特殊处理：如果 dataset 是 Subset，我们实际上无法简单使用原来的 BucketBatchSampler，
        # 因为原 BucketBatchSampler 依赖整个数据集的 buckets 索引。
        # 简单起见，如果用了 Subset 且开启了 Bucket，我们强制 batch_size=1 或者警告。
        # 但为了稳健，如果 dataset 是 Subset，我们需要batch_size = 1。
        
        if isinstance(dataset, torch.utils.data.Subset):
            print("Warning: Using Subset with enable_bucket and batch_size > 1 is tricky. "
                  "Ideally we should rebuild buckets for the subset, but here we fallback to batch_size=1 "
                  "to avoid errors, or you can implement a Subset-aware sampler.")
            # 简单回退策略：强制 batch_size = 1
            if args.batch_size > 1:
                print("Forcing batch_size=1 due to Subset usage with Buckets.")
                args.batch_size = 1
                dataloader = DataLoader(
                    dataset,
                    batch_size=1,
                    shuffle=False,
                    collate_fn=collate_fn,
                    num_workers=args.num_workers
                )
            else:
                dataloader = DataLoader(
                    dataset,
                    batch_size=1,
                    shuffle=False,
                    collate_fn=collate_fn,
                    num_workers=args.num_workers
                )
        else:
            # 正常情况：使用 BucketBatchSampler
            batch_sampler = BucketBatchSampler(
                dataset,
                batch_size=args.batch_size,
                shuffle=False, # 统计任务不需要打乱
                seed=42
            )
            dataloader = DataLoader(
                dataset,
                batch_sampler=batch_sampler, # 互斥：使用了 batch_sampler 就不能传 batch_size 和 shuffle
                collate_fn=collate_fn,
                num_workers=args.num_workers
            )
    else:
        # 未开启 Bucket，所有图像都是固定分辨率，直接使用默认 DataLoader
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=args.num_workers
        )

    print("Starting Loop to calculate Global Mean and Std...")
    
    # 初始化累加器 (使用 float64 避免精度溢出)
    total_elements = 0
    global_sum = 0.0
    global_sq_sum = 0.0
    
    for batch in tqdm(dataloader):
        pixel_values = batch["pixel_values"].to(device, dtype=torch.float32)
        
        with torch.no_grad():
            # Encode: x -> z
            latents = vae.encode(pixel_values).latent_dist.sample() 
            
            # Latents shape: [B, C, H, W]
            
            # 转换为 float64 进行累加
            latents_f64 = latents.to(dtype=torch.float64)
            
            # 全局统计 (所有通道合并计算)
            global_sum += latents_f64.sum().item()
            global_sq_sum += (latents_f64 ** 2).sum().item()
            total_elements += latents.numel()
            
    # --- 计算统计量 ---
    
    if total_elements == 0:
        print("Error: No data processed.")
        return

    global_mean = global_sum / total_elements
    # Var = E[X^2] - (E[X])^2
    global_variance = (global_sq_sum / total_elements) - (global_mean ** 2)
    # 防止方差为负（浮点误差）
    global_variance = max(0.0, global_variance)
    global_std = math.sqrt(global_variance)
    
    # 计算 Scale
    # Scale = 1 / Std
    rec_scale = 1.0 / global_std if global_std > 1e-9 else 1.0
    
    print("\n" + "=" * 50)
    print("CALCULATION RESULTS (Global Only)")
    print("=" * 50)
    print(f"Total Elements Processed: {total_elements}")
    print("-" * 30)
    print(f"Latent Mean (Shift): {global_mean:.6f}")
    print(f"Latent Std:          {global_std:.6f}")
    print(f"Rec. Scale (1/Std):  {rec_scale:.6f}")
    print("-" * 30)
    print(f"(Ref: Flux.1 VAE default Shift ~ 0.1159, Scale ~ 0.3611)")

    final_shift = float(global_mean)
    final_scale = float(rec_scale)

    # --- 保存模型 ---
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"\nSaving VAE to {args.output_dir}...")
    
    # 1. 保存权重
    vae.save_pretrained(args.output_dir)
    
    # 2. 更新 config.json
    config_path = os.path.join(args.output_dir, "config.json")
    
    # 读取刚刚 save_pretrained 生成的 config
    with open(config_path, 'r') as f:
        saved_config = json.load(f)
    
    # 覆盖 scaling_factor 和 shift_factor
    saved_config['scaling_factor'] = final_scale
    saved_config['shift_factor'] = final_shift
    
    with open(config_path, 'w') as f:
        json.dump(saved_config, f, indent=2)
        
    print(f"Config updated successfully:")
    print(f"  scaling_factor: {final_scale}")
    print(f"  shift_factor:   {final_shift}")
    print("Done.")

if __name__ == "__main__":
    main()
