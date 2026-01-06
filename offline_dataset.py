"""
python offline_dataset.py \
  --config_file lora.toml \
  --lora_path /root/autodl-tmp/output/NewBieLoRA_step_1000 \
  --output_dir ./sdpo_data_v1 \
  --num_samples 4 \
  --steps 28
"""

import os
import json
import toml
import torch
import argparse
from pathlib import Path
from tqdm import tqdm
from PIL import Image

# 导入仓库组件
from dataset import ImageCaptionDataset, collate_fn
from train_newbie_lora import load_model_and_tokenizer, setup_lora
from peft import set_peft_model_state_dict
from safetensors.torch import load_file
from transport import create_transport

def parse_args():
    parser = argparse.ArgumentParser(description="生成 SDPO 离线偏好数据集样本池")
    parser.add_argument("--config_file", type=str, required=True, help="训练时使用的 .toml 配置文件 (lora.toml)")
    parser.add_argument("--lora_path", type=str, required=True, help="SFT 训练产出的 LoRA 权重路径 (PEFT 文件夹目录)")
    parser.add_argument("--output_dir", type=str, default="./sdpo_data_v1", help="样本图片存放目录")
    parser.add_argument("--num_samples", type=int, default=4, help="每个 Prompt 生成的样本数量 (N)")
    parser.add_argument("--steps", type=int, default=28, help="生成步数")
    parser.add_argument("--cfg_scale", type=float, default=5, help="Classifier-Free Guidance 强度")
    parser.add_argument("--device", type=str, default="cuda", help="使用设备")
    return parser.parse_args()

@torch.no_grad()
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
    model = setup_lora(model, config)
    if os.path.isdir(args.lora_path):
        # 如果是 PEFT 目录
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.lora_path)
    else:
        print(f'{args.lora_path} is not a valid PEFT LoRA directory path！')
    
    model.to(args.device).eval()
    vae.to(args.device).eval()
    text_encoder.to(args.device).eval()
    clip_model.to(args.device).eval()

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
    uncond_input = tokenizer([""], padding=True, return_tensors="pt").to(args.device)
    uncond_outputs = text_encoder(**uncond_input, output_hidden_states=True)
    uncond_cap_feats = uncond_outputs.hidden_states[-2]
    uncond_cap_mask = uncond_input.attention_mask
    # CLIP 无条件
    uncond_clip_input = clip_tokenizer([""], padding=True, return_tensors="pt").to(args.device)
    uncond_clip_text_pooled = clip_model.get_text_features(**uncond_clip_input)

    # 5. 开始生成循环
    results = []
    gemma3_prompt = config['Model'].get('gemma3_prompt', "")
    scaling_factor = getattr(vae.config, 'scaling_factor', 0.3611)
    shift_factor = getattr(vae.config, 'shift_factor', 0.1159)

    print(f"开始为 {len(dataset)} 个原始样本生成偏好对池...")
    
    for i in tqdm(range(len(dataset))):
        img_path = dataset.image_paths[i]
        raw_caption = dataset.captions[i]
        # 获取该图片在分箱策略下的目标分辨率
        target_width, target_height = dataset.image_to_bucket[i]
        
        # 处理 Wildcard: 获取所有可能的 caption 变体
        captions_to_process = [raw_caption]
        if "<split>" in raw_caption:
            captions_to_process = [c.strip() for c in raw_caption.split("<split>") if c.strip()]

        for cap_idx, caption in enumerate(captions_to_process):
            # 准备编码特征
            with torch.no_grad():
                # 编码正向特征
                gemma_text = gemma3_prompt + caption
                pos_input = tokenizer([gemma_text], padding=True, return_tensors="pt").to(args.device)
                pos_outputs = text_encoder(**pos_input, output_hidden_states=True)
                pos_cap_feats = pos_outputs.hidden_states[-2]
                pos_cap_mask = pos_input.attention_mask
            
                pos_clip_input = clip_tokenizer([caption], padding=True, return_tensors="pt").to(args.device)
                pos_clip_text_pooled = clip_model.get_text_features(**pos_clip_input)

            # 生成 N 个样本
            gen_paths = []
            for n in range(args.num_samples):
                # 初始化噪声 [B=1, C=16, H//8, W//8]
                latents = torch.randn(1, 16, target_height // 8, target_width // 8).to(args.device)
                
                # Rectified Flow Euler 采样步
                dt = 1.0 / args.steps
                for step in range(args.steps):
                    t_val = step / args.steps
                    t = torch.ones(1).to(args.device) * t_val
                    
                    # 拼接 Batch 加速前向 (无条件 + 有条件)
                    batched_latents = torch.cat([latents, latents], dim=0)
                    batched_t = torch.cat([t, t], dim=0)
                    batched_cap_feats = torch.cat([uncond_cap_feats, pos_cap_feats], dim=0)
                    batched_cap_mask = torch.cat([uncond_cap_mask, pos_cap_mask], dim=0)
                    batched_clip_pooled = torch.cat([uncond_clip_text_pooled, pos_clip_text_pooled], dim=0)
                    
                    # 前向推理
                    v_out = model(
                        batched_latents, 
                        batched_t, 
                        cap_feats=batched_cap_feats, 
                        cap_mask=batched_cap_mask, 
                        clip_text_pooled=batched_clip_pooled
                    )
                    
                    # 拆分预测速度 v
                    v_uncond, v_cond = v_out.chunk(2)
                    
                    # CFG 混合
                    v_final = v_uncond + args.cfg_scale * (v_cond - v_uncond)
                    
                    # Euler 更新
                    latents = latents + v_final * dt
                
                # VAE 解码
                latents = (latents / scaling_factor) + shift_factor
                image = vae.decode(latents.to(vae.dtype)).sample
                image = (image / 2 + 0.5).clamp(0, 1)
                image = image.cpu().permute(0, 2, 3, 1).float().numpy()[0]
                image = Image.fromarray((image * 255).astype("uint8"))
                
                # 保存图片
                safe_name = Path(img_path).stem
                gen_filename = f"{safe_name}_cap{cap_idx}_sample{n}.jpg"
                gen_path = os.path.join(gen_images_dir, gen_filename)
                image.save(gen_path, quality=100)
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
