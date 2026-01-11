"""
python offline_dataset.py \
  --config_file lora.toml \
  --lora_path /root/autodl-tmp/output/NewBieLoRA \
  --output_dir /root/autodl-tmp/gen_dataset \
  --num_samples 2 \
  --steps 28
"""

import os
import json
import toml
import torch
import argparse
import math
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import numpy as np

# 导入仓库组件
from dataset import ImageCaptionDataset, collate_fn
from train_newbie_lora import load_model_and_tokenizer, setup_lora
from peft import set_peft_model_state_dict
from safetensors.torch import load_file
from transport import create_transport
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler

def get_lin_function(x1: float = 256, y1: float = 0.5, x2: float = 4096, y2: float = 1.15):
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return lambda x: m * x + b

def parse_args():
    parser = argparse.ArgumentParser(description="生成 DPO 离线偏好数据集样本池")
    parser.add_argument("--config_file", type=str, required=True, help="训练时使用的 .toml 配置文件 (lora.toml)")
    parser.add_argument("--lora_path", type=str, required=True, help="SFT 训练产出的 LoRA 权重路径 (PEFT 文件夹目录)")
    parser.add_argument("--output_dir", type=str, default="./sdpo_data_v1", help="样本图片存放目录")
    parser.add_argument("--num_samples", type=int, default=3, help="每个 Prompt 生成的样本数量 (N)")
    parser.add_argument("--steps", type=int, default=28, help="生成步数")
    parser.add_argument("--cfg_scale", type=float, default=6, help="Classifier-Free Guidance 强度")
    parser.add_argument("--device", type=str, default="cuda", help="使用设备")
    return parser.parse_args()

@torch.inference_mode()
def main():
    args = parse_args()
    
    # 1. 加载配置
    with open(args.config_file, 'r', encoding='utf-8') as f:
        config = toml.load(f)
    
    # 关闭数据增强部分以保证数据对应关系
    config['Model']['shuffle_caption'] = False
    config['Model']['caption_dropout_rate'] = 0.0
    config['Model']['caption_tag_dropout_rate'] = 0.0
    config['Model']['drop_artist_rate'] = 0.0
    
    os.makedirs(args.output_dir, exist_ok=True)
    gen_images_dir = os.path.join(args.output_dir, "generated")
    os.makedirs(gen_images_dir, exist_ok=True)

    # 2. 加载模型与 LoRA
    # 注意：生成需要完整的 VAE 和 Text Encoder
    model, vae, text_encoder, tokenizer, clip_model, clip_tokenizer = load_model_and_tokenizer(config)
    
    # 应用 LoRA
    if os.path.isdir(args.lora_path):
        # 如果是 PEFT 目录，直接从 Base Model 加载
        from peft import PeftModel
        print(f"Loading LoRA from: {args.lora_path}")
        model = PeftModel.from_pretrained(model, args.lora_path)
    else:
        # 只有在路径无效（或想测试未训练的随机初始化 LoRA）时，才手动 setup_lora
        print(f'{args.lora_path} is not a valid PEFT LoRA directory path! Falling back to random init.')
        model = setup_lora(model, config)
    
    model.to(args.device, dtype=torch.bfloat16).eval()
    vae.to(args.device, dtype=torch.bfloat16).eval()
    text_encoder.to(args.device, dtype=torch.bfloat16).eval()
    clip_model.to(args.device, dtype=torch.bfloat16).eval()

    print("Compiling model...")
    model = torch.compile(model, mode="default", fullgraph=False, dynamic=True) # "reduce-overhead"

    # 3. 初始化数据集 (获取动态分箱结果)
    # 不启用 cache 以便直接读取原始路径信息
    dataset = ImageCaptionDataset(
        train_data_dir=config['Model']['train_data_dir'],
        resolution=config['Model']['resolution'],
        enable_bucket=config['Model'].get('enable_bucket', True),
        use_cache=False,
        min_bucket_reso=config['Model'].get('min_bucket_reso', 256),
        max_bucket_reso=config['Model'].get('max_bucket_reso', 2048),
        bucket_reso_step=config['Model'].get('bucket_reso_step', 64),
        enable_wildcard=False, # 此处先关闭，我们在循环中手动处理 split
    )

    # 4. 预计算无条件特征 (用于 CFG)
    print("预计算负向 (Unconditional) 特征...")
    # Gemma 无条件
    uncond_input = tokenizer(
        [""], padding="max_length", pad_to_multiple_of=8,
        truncation=True, max_length=1280, return_tensors="pt"
    ).to(args.device)
    with torch.no_grad():
        uncond_outputs = text_encoder(**uncond_input, output_hidden_states=True)
        uncond_cap_feats = uncond_outputs.hidden_states[-2].to(dtype=torch.bfloat16)
        uncond_cap_mask = uncond_input.attention_mask
    # CLIP 无条件
    uncond_clip_input = clip_tokenizer(
        [""], padding=True, truncation=True,
        max_length=2048, return_tensors="pt"
    ).to(args.device)
    with torch.no_grad():
        uncond_clip_text_pooled = clip_model.get_text_features(**uncond_clip_input).to(dtype=torch.bfloat16)

    # 5. 开始生成循环
    results = []
    gemma3_prompt = config['Model'].get('gemma3_prompt', "")
    scaling_factor = getattr(vae.config, 'scaling_factor', 0.3611)
    shift_factor = getattr(vae.config, 'shift_factor', 0.1159)
 
    print(f"开始为 {len(dataset)} 个原始样本生成偏好对池...")

    batch_size = 2 # args.num_samples
    
    for i in tqdm(range(len(dataset))):
        img_path = dataset.image_paths[i]
        raw_caption = dataset.captions[i]
        target_width, target_height = dataset.image_to_bucket[i]
        
        # 处理 Wildcard: 获取所有可能的 caption 变体
        captions_to_process = [raw_caption]
        if "<split>" in raw_caption:
            captions_to_process = [c.strip() for c in raw_caption.split("<split>") if c.strip()]
        captions_to_process = [c.replace(" ||| ", ", ") for c in captions_to_process]

        for cap_idx, caption in enumerate(captions_to_process):
            gen_paths = []
            # 准备编码特征
            with torch.no_grad():
                # 编码正向特征
                gemma_text = gemma3_prompt + caption
                pos_input = tokenizer([gemma_text], padding="longest", pad_to_multiple_of=8,
                    truncation=True, max_length=1280, return_tensors="pt"
                ).to(args.device)
                current_seq_len = pos_input.input_ids.shape[1]
                pos_outputs = text_encoder(**pos_input, output_hidden_states=True)
                pos_cap_feats = pos_outputs.hidden_states[-2].to(dtype=torch.bfloat16)
                pos_cap_mask = pos_input.attention_mask
            
                pos_clip_input = clip_tokenizer(
                    [caption], padding=True, truncation=True,
                    max_length=2048, return_tensors="pt"
                ).to(args.device)
                pos_clip_text_pooled = clip_model.get_text_features(**pos_clip_input).to(dtype=torch.bfloat16)

                # 1. 重复 uncond 特征 N 次
                batch_uncond_feats = uncond_cap_feats.repeat(batch_size, 1, 1)[:, :current_seq_len, :]
                batch_uncond_mask = uncond_cap_mask.repeat(batch_size, 1)[:, :current_seq_len]
                batch_uncond_pooled = uncond_clip_text_pooled.repeat(batch_size, 1)

                # 2. 重复 cond 特征 N 次
                batch_pos_feats = pos_cap_feats.repeat(batch_size, 1, 1)
                batch_pos_mask = pos_cap_mask.repeat(batch_size, 1)
                batch_pos_pooled = pos_clip_text_pooled.repeat(batch_size, 1)

                # 3. 拼接 (Concat) -> 形状变成 [2*N, Seq, Dim]
                cat_cap_feats = torch.cat([batch_uncond_feats, batch_pos_feats])
                cat_cap_mask = torch.cat([batch_uncond_mask, batch_pos_mask])
                cat_clip_pooled = torch.cat([batch_uncond_pooled, batch_pos_pooled])

                ######################################################################
                # 1. 初始化 Scheduler
                scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, use_dynamic_shifting=True)
                mu = get_lin_function()((target_height // 16) * (target_width // 16))
                scheduler.set_timesteps(args.steps, mu=mu)
                    
                # 2. 初始化噪声
                latents = torch.randn(batch_size, 16, target_height // 8, target_width // 8, device=args.device, dtype=torch.bfloat16)
                    
                # 3. 采样循环
                for t in scheduler.timesteps:
                    t_norm = t / 1000.0
                    t_model = 1.0 - t_norm
                    t_tensor = torch.full((batch_size * 2,), t_model, device=args.device, dtype=torch.bfloat16)

                    # Forward
                    v_pred = model(
                        torch.cat([latents, latents]), 
                        t_tensor, 
                        cap_feats=cat_cap_feats, 
                        cap_mask=cat_cap_mask, 
                        clip_text_pooled=cat_clip_pooled
                    )
                        
                    # 拆分预测速度 v 并进行 CFG 混合
                    v_uncond, v_cond = v_pred.chunk(2)
                    v_final = v_uncond + args.cfg_scale * (v_cond - v_uncond)
                        
                    # 使用 Scheduler 步进更新噪声，注意取反以适配积分方向
                    latents = scheduler.step(-v_final, t, latents).prev_sample
                
                # 4. VAE 解码
                latents = (latents / scaling_factor) + shift_factor
                image = vae.decode(latents.to(vae.dtype)).sample
                image = (image / 2 + 0.5).clamp(0, 1)
                image = image.cpu().permute(0, 2, 3, 1).float().numpy()
                
                # 保存图片
                gen_paths = []
                safe_name = Path(img_path).stem
                for n in range(batch_size):
                    img_pil = Image.fromarray((image[n] * 255).astype("uint8"))
                    gen_filename = f"{safe_name}_cap{cap_idx}_sample{n}.jpg"
                    gen_path = os.path.join(gen_images_dir, gen_filename)
                    img_pil.save(gen_path, quality=100)
                    gen_paths.append(os.path.abspath(gen_path))

                # 记录数据结构
                results.append({
                    "caption": caption,
                    "real_image_path": os.path.abspath(img_path),
                    "generated_image_paths": gen_paths,
                    "resolution": [target_width, target_height],
                    "meta": {
                        "lora_source": args.lora_path,
                        "steps": args.steps,
                        "cfg_scale": args.cfg_scale,
                        "sampler": "euler"
                    }
                })

    # 6. 输出 JSON 数据库
    output_json = os.path.join(args.output_dir, "dataset.json")
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"🎉 离线数据库已建立: {output_json}")
    print(f"总计样本数: {len(results)}")

if __name__ == "__main__":
    main()
