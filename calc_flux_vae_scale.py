#!/usr/bin/env python3
"""
计算 FLUX VAE 的 scaling_factor 和 shift_factor 并保存为 Diffusers 格式。
基于 NewbieLoraTrainer 的数据处理逻辑。

功能：
1. 遍历数据集，通过 VAE 编码图像。
2. 统计 Latent 的全局均值 (Shift) 和标准差 (Std)。
3. 计算 Scaling Factor = 1 / Std。
4. 更新 VAE 的 config.json 并保存。

修改说明：已移除单通道 (Per-Channel) 统计计算，仅计算全局统计量。

Example:
python calc_flux_vae_scale.py \
  --train_data_dir /path/to/your/images \
  --vae_path /path/to/flux_vae.safetensors \
  --output_dir /path/to/save/diffusers_vae \
  --resolution 1024 \
  --batch_size 4 \
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
    from dataset import ImageCaptionDataset, collate_fn
except ImportError:
    raise ImportError("请确保 dataset.py 在同一目录下。")

def parse_args():
    parser = argparse.ArgumentParser(description="Calculate VAE scaling and shift factors for Flux (Global Only)")
    parser.add_argument("--train_data_dir", type=str, required=True, help="训练数据集目录")
    parser.add_argument("--vae_path", type=str, required=True, help="单文件 VAE (.safetensors) 路径或 Diffusers 模型目录")
    parser.add_argument("--output_dir", type=str, required=True, help="输出 Diffusers VAE 的目录")
    parser.add_argument("--resolution", type=int, default=1024, help="计算时使用的分辨率 (默认 1024)")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size (建议 1-4)")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--max_samples", type=int, default=None, help="最大采样图片数 (None 为使用全部)")
    parser.add_argument("--enable_bucket", action="store_true", help="启用分桶 (计算统计量时建议关闭 bucket 以保持一致性，但也支持开启)")
    parser.add_argument("--vae_reflect_padding", action="store_true", help="VAE是否使用reflect_padding")
    return parser.parse_args()

def main():
    args = parse_args()
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Loading VAE from: {args.vae_path}")
    
    # 基础 Config，用于单文件加载
    config = {
        "_class_name": "AutoencoderKL",
        "_diffusers_version": "0.30.0",
        "act_fn": "silu",
        "block_out_channels": [128, 256, 512, 512],
        "down_block_types": [
            "DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D"
        ],
        "force_upcast": True,
        "in_channels": 3,
        "latent_channels": 16,
        "layers_per_block": 2,
        "mid_block_add_attention": True,
        "norm_num_groups": 32,
        "out_channels": 3,
        "sample_size": 1024,
        "scaling_factor": 1.0, # 初始占位
        "shift_factor": 0.0,   # 初始占位
        "up_block_types": [
            "UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D"
        ],
        "use_post_quant_conv": False,
        "use_quant_conv": False
    }

    try:
        if os.path.isdir(args.vae_path):
            vae = AutoencoderKL.from_pretrained(args.vae_path)
        else:
            vae = AutoencoderKL.from_single_file(
                args.vae_path, 
                config=config, 
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
