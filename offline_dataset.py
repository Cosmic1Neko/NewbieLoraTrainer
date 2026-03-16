"""
python offline_dataset.py \
  --config_file lora.toml \
  --lora_path /root/autodl-tmp/output/AnimaLoRA_step_1000 \
  --output_dir /root/autodl-tmp/gen_dataset \
  --num_samples 2 \
  --steps 25 \
  --max_data_samples 5000 \
  --seed 114514
"""

import os
import json
import toml
import torch
import argparse
import math
import random
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import numpy as np

# 导入仓库组件
from dataset import ImageCaptionDataset, collate_fn
from train_anima_lora import (
    load_model_and_tokenizer, 
    setup_lora, 
    _build_qwen_text_from_prompt, 
    encode_qwen, 
    tokenize_t5_weighted
)
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler

def parse_args():
    parser = argparse.ArgumentParser(description="生成 DPO 离线偏好数据集样本池")
    parser.add_argument("--config_file", type=str, required=True, help="训练时使用的 .toml 配置文件 (lora.toml)")
    parser.add_argument("--lora_path", type=str, required=True, help="SFT 训练产出的 LoRA 权重路径 (PEFT 文件夹目录)")
    parser.add_argument("--output_dir", type=str, default="./sdpo_data_v1", help="样本图片存放目录")
    parser.add_argument("--num_samples", type=int, default=3, help="每个 Prompt 生成的样本数量 (N)")
    parser.add_argument("--steps", type=int, default=28, help="生成步数")
    parser.add_argument("--cfg_scale", type=float, default=5, help="Classifier-Free Guidance 强度")
    parser.add_argument("--device", type=str, default="cuda", help="使用设备")
    parser.add_argument("--max_data_samples", type=int, default=-1, help="从数据集中随机抽取的样本数量，-1 为使用全部数据")
    parser.add_argument("--seed", type=int, default=42, help="随机抽样种子")
    return parser.parse_args()

@torch.inference_mode()
def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    gen_images_dir = os.path.join(args.output_dir, "generated")
    os.makedirs(gen_images_dir, exist_ok=True)
    output_json = os.path.join(args.output_dir, "dataset.json")

    results = []
    processed_keys = set()

    # 尝试加载现有进度
    if os.path.exists(output_json):
        try:
            print(f"检测到现有数据文件: {output_json}，正在加载以恢复进度...")
            with open(output_json, 'r', encoding='utf-8') as f:
                results = json.load(f)
            
            # 构建已处理的 (绝对路径, caption) 集合，用于快速查找
            for item in results:
                # 确保使用绝对路径作为 Key，与后续逻辑一致
                key = (item['real_image_path'], item['caption'])
                processed_keys.add(key)
            
            print(f"已加载 {len(results)} 条历史记录，将跳过这些样本。")
        except Exception as e:
            print(f"加载现有数据出错 ({e})，将重新开始生成。")
            results = []
  
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
    
    # 1. 加载配置
    with open(args.config_file, 'r', encoding='utf-8') as f:
        config = toml.load(f)

    import models
    from pathlib import Path
    repo_root = str(Path(config['Model'].get('anima_repo_root', '.')).resolve())
    if repo_root not in [str(Path(p).resolve()) for p in models.__path__]:
        models.__path__.append(repo_root)
    
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
    model, vae, qwen_model, qwen_tokenizer, t5_tokenizer, _ = load_model_and_tokenizer(config)
    
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
    vae.model.to(args.device, dtype=torch.bfloat16).eval()
    vae.scale = [s.to(device=args.device, dtype=torch.bfloat16) for s in vae.scale]
    qwen_model.to(args.device, dtype=torch.bfloat16).eval()

    #print("Compiling model...")
    #model = torch.compile(model, mode="default", fullgraph=False, dynamic=True) # "reduce-overhead"

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

    # 确定要处理的样本索引列表
    all_indices = list(range(len(dataset)))
    if args.max_data_samples > 0 and args.max_data_samples < len(dataset):
        print(f"正在从总数 {len(dataset)} 中随机抽取 {args.max_data_samples} 个样本 (Seed: {args.seed})...")
        target_indices = random.sample(all_indices, args.max_data_samples)
    else:
        print(f"使用全部 {len(dataset)} 个样本进行生成...")
        target_indices = all_indices

    # 4. 预计算无条件特征 (用于 CFG)
    print("预计算负向 (Unconditional) 特征...")
    uncond_prompt = ""
    u_qwen_text = _build_qwen_text_from_prompt(uncond_prompt)
    u_qwen_embeds, _ = encode_qwen(qwen_model, qwen_tokenizer, [u_qwen_text], args.device)
    u_t5_ids, _, _ = tokenize_t5_weighted(t5_tokenizer, [uncond_prompt], max_length=1024)
    unwrapped_model = model.module if hasattr(model, "module") else model
    if hasattr(unwrapped_model, "base_model"): unwrapped_model = unwrapped_model.base_model.model
    uncond_cross = unwrapped_model.preprocess_text_embeds(u_qwen_embeds, u_t5_ids.to(args.device))
    if uncond_cross.shape[1] < 1024:
        uncond_cross = torch.nn.functional.pad(uncond_cross, (0, 0, 0, 1024 - uncond_cross.shape[1]))

    # 5. 开始生成循环
    results = []
    gemma3_prompt = config['Model'].get('gemma3_prompt', "")
 
    print(f"开始为 {len(target_indices)} 个原始样本生成偏好对池...")

    batch_size = args.num_samples # args.num_samples
    
    for i in tqdm(target_indices, desc="Generating"):
        img_path = dataset.image_paths[i]
        raw_caption = dataset.captions[i]
        target_width, target_height = dataset.image_to_bucket[i]
        current_abs_path = os.path.abspath(img_path)
        
        # 处理 Wildcard: 获取所有可能的 caption 变体
        captions_to_process = [raw_caption]
        if "<split>" in raw_caption:
            captions_to_process = [c.strip() for c in raw_caption.split("<split>") if c.strip()]
        captions_to_process = [c.replace(" ||| ", ", ") for c in captions_to_process]

        for cap_idx, caption in enumerate(captions_to_process):
          if (current_abs_path, caption) in processed_keys:
                continue
          
          gen_paths = []
          # 准备编码特征
          with torch.no_grad():
            # 编码正向特征
            full_prompt = gemma3_prompt + caption
            p_qwen_text = _build_qwen_text_from_prompt(full_prompt)
            p_qwen_embeds, _ = encode_qwen(qwen_model, qwen_tokenizer, [p_qwen_text], args.device)
            p_t5_ids, _, _ = tokenize_t5_weighted(t5_tokenizer, [full_prompt], max_length=1024)
            pos_cross = unwrapped_model.preprocess_text_embeds(p_qwen_embeds, p_t5_ids.to(args.device))
            if pos_cross.shape[1] < 1024:
                pos_cross = torch.nn.functional.pad(pos_cross, (0, 0, 0, 1024 - pos_cross.shape[1]))

            # 准备批次特征 (Concat CFG)
            batch_cross = torch.cat([uncond_cross.repeat(args.num_samples, 1, 1), pos_cross.repeat(args.num_samples, 1, 1)])
            ######################################################################
            # 1. 初始化 Scheduler
            scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=6.0)
            scheduler.set_timesteps(args.steps)
                    
            # 2. 初始化噪声
            latents = torch.randn(batch_size, 16, 1, target_height // 8, target_width // 8, device=args.device, dtype=torch.bfloat16)
                    
            # 3. 采样循环
            for t in scheduler.timesteps:
                t_val = t / 1000.0
                t_tensor = torch.full((args.num_samples * 2,), t_val, device=args.device, dtype=torch.bfloat16)

                # Forward
                v_pred = model(
                    torch.cat([latents, latents]), 
                    t_tensor, 
                    crossattn_emb=batch_cross,
                    padding_mask=torch.zeros(args.num_samples * 2, 1, latents.shape[-2], latents.shape[-1], 
                                             device=args.device, dtype=torch.bfloat16)
                )
                        
                # 拆分预测速度 v 并进行 CFG 混合
                v_uncond, v_cond = v_pred.chunk(2)
                v_final = v_uncond + args.cfg_scale * (v_cond - v_uncond)
                latents = scheduler.step(v_final, t, latents).prev_sample
                
            # 4. VAE 解码
            with torch.no_grad():
                decoded = vae.model.decode(latents, vae.scale) 
                decoded = decoded.squeeze(2) 
                decoded = (decoded.clamp(-1, 1) + 1) / 2
                image = decoded.cpu().permute(0, 2, 3, 1).float().numpy()
                
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

            # 输出 JSON 数据库
            with open(output_json, 'w', encoding='utf-8') as f:
              json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"🎉 离线数据库已建立: {output_json}")
    print(f"总计样本数: {len(results)}")

if __name__ == "__main__":
    main()
