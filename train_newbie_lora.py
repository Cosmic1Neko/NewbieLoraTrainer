#!/usr/bin/env python3
"""Newbie LoRA 训练器 - 基于 Rectified Flow 的 LoRA 微调"""

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
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import set_seed
from diffusers import AutoencoderKL
from diffusers.optimization import get_scheduler
from transformers import AutoTokenizer, AutoModel, AutoConfig
from peft import LoraConfig, get_peft_model, PeftModel, get_peft_model_state_dict, set_peft_model_state_dict
from peft.tuners.lora import initialize_lora_eva_weights
from safetensors.torch import load_file, save_file
from tqdm import tqdm
import re
import random

sys.path.insert(0, str(Path(__file__).parent))
import models
from transport import create_transport

try:
    import bitsandbytes as bnb
except ImportError:
    logging.warning("bitsandbytes not available, 8-bit optimizer disabled")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("newbie_lora_trainer")

def apply_average_pool(latent, factor=4):
    """
    Apply average pooling to downsample the latent.

    Args:
        latent (torch.Tensor): Latent tensor with shape (1, C, H, W).
        factor (int): Downsampling factor.

    Returns:
        torch.Tensor: Downsampled latent tensor.
    """
    return torch.nn.functional.avg_pool2d(latent, kernel_size=factor, stride=factor)

def get_eva_batch_generator(dataloader, device, vae, text_encoder, tokenizer, clip_model, clip_tokenizer, gemma3_prompt, dtype, num_steps=20):
    """
    生成用于 EVA 初始化的数据 Batch。
    模拟 compute_loss 中的预处理，并添加噪声以匹配 Rectified Flow 的输入分布。
    """
    iter_loader = iter(dataloader)
    
    for _ in range(num_steps):
        try:
            batch = next(iter_loader)
        except StopIteration:
            iter_loader = iter(dataloader)
            batch = next(iter_loader)
            
        # 1. 准备 Latents 和 Caption Embeddings (复制 compute_loss 的部分逻辑)
        if batch.get("cached", False):
            latents = batch["latents"].to(device, dtype=dtype)
            captions = batch["captions"]
        else:
            pixel_values = batch["pixel_values"].to(device, dtype=dtype)
            captions = batch["captions"]
            with torch.no_grad():
                # VAE Encode
                latents = vae.encode(pixel_values).latent_dist.sample()
                scaling_factor = getattr(vae.config, 'scaling_factor', 0.13025)
                latents = latents * scaling_factor

        batch_size = latents.shape[0]

        with torch.no_grad():
            # Gemma Text Encode
            gemma_texts = [gemma3_prompt + cap if gemma3_prompt else cap for cap in captions]
            gemma_inputs = tokenizer(
                gemma_texts, padding=True, pad_to_multiple_of=8,
                truncation=True, max_length=1280, return_tensors="pt"
            ).to(device)
            gemma_outputs = text_encoder(**gemma_inputs, output_hidden_states=True)
            cap_feats = gemma_outputs.hidden_states[-2].to(dtype=dtype)
            cap_mask = gemma_inputs.attention_mask.to(device)
            
            # CLIP Text Encode
            clip_inputs = clip_tokenizer(
                captions, padding=True, truncation=True,
                max_length=2048, return_tensors="pt"
            ).to(device)
            clip_text_pooled = clip_model.get_text_features(**clip_inputs).to(dtype=dtype)

        # 2. 模拟加噪 (Rectified Flow / Flux 训练时的输入是含噪的)
        # 使用简单的线性插值模拟: x_t = (1-t)*x + t*noise
        t = torch.rand((batch_size,), device=device, dtype=dtype).view(-1, 1, 1, 1)
        noise = torch.randn_like(latents)
        x_noisy = (1 - t) * latents + t * noise

        # 3. 返回匹配 model.forward 参数的字典
        # NextDiT forward: x, t, cap_feats, cap_mask, clip_text_pooled(in kwargs)
        yield {
            "x": x_noisy,
            "t": t,
            "cap_feats": cap_feats,
            "cap_mask": cap_mask,
            "clip_text_pooled": clip_text_pooled
        }

class ImageCaptionDataset(Dataset):
    """图像-文本对数据集，支持 kohya_ss 风格目录重复"""

    def __init__(
        self,
        train_data_dir: str,
        resolution: int,
        enable_bucket: bool = True,
        use_cache: bool = True,
        vae=None,
        text_encoder=None,
        tokenizer=None,
        clip_model=None,
        clip_tokenizer=None,
        device=None,
        dtype=torch.bfloat16,
        gemma3_prompt: str = "",
        min_bucket_reso: int = 256,
        max_bucket_reso: int = 2048,
        bucket_reso_step: int = 32,
        shuffle_caption: bool = False,
        keep_tokens_separator: str = "|||",
        enable_wildcard: bool = False,
        caption_dropout_rate: float = 0.0,
        caption_tag_dropout_rate: float = 0.0,
        drop_artist_rate: float = 0.0,
    ):
        self.train_data_dir = train_data_dir
        self.resolution = resolution
        self.enable_bucket = enable_bucket
        self.use_cache = use_cache
        self.image_paths = []
        self.captions = []
        self.repeats = []
        self.image_resolutions = []
        self.buckets = {}
        self.bucket_resolutions = []
        self.image_to_bucket = {}
        self.image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'}

        self.vae = vae
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.clip_model = clip_model
        self.clip_tokenizer = clip_tokenizer
        self.device = device
        self.dtype = dtype
        self.gemma3_prompt = gemma3_prompt
        self.min_bucket_reso = min_bucket_reso
        self.max_bucket_reso = max_bucket_reso
        self.bucket_reso_step = bucket_reso_step
        self.shuffle_caption = shuffle_caption
        self.keep_tokens_separator = keep_tokens_separator
        self.enable_wildcard = enable_wildcard
        self.caption_dropout_rate = caption_dropout_rate
        self.caption_tag_dropout_rate = caption_tag_dropout_rate
        self.drop_artist_rate = drop_artist_rate
        self.caption_separator = ","

        self._load_data()
        if self.enable_bucket:
            self._generate_buckets()
            self._assign_buckets()
        else:
            for idx in range(len(self.image_paths)):
                self.image_to_bucket[idx] = (self.resolution, self.resolution)

        if self.use_cache and vae is not None:
            self._prepare_cache()

    def _load_data(self):
        from PIL import Image
        logger.info(f"Loading data from: {self.train_data_dir}")

        for root, _, files in os.walk(self.train_data_dir):
            dir_name = os.path.basename(root)
            repeats = int(dir_name.split('_')[0]) if '_' in dir_name and dir_name[0].isdigit() else 1

            for file in files:
                _, ext = os.path.splitext(file)
                if ext.lower() in self.image_extensions:
                    image_path = os.path.join(root, file)
                    caption_path = os.path.splitext(image_path)[0] + '.txt'

                    caption = ''
                    if os.path.exists(caption_path):
                        try:
                            with open(caption_path, 'r', encoding='utf-8') as f:
                                caption = f.read().strip()
                        except UnicodeDecodeError:
                            with open(caption_path, 'r', encoding='latin-1') as f:
                                caption = f.read().strip()

                    try:
                        with Image.open(image_path) as img:
                            width, height = img.size
                            self.image_resolutions.append((width, height))
                    except Exception as e:
                        logger.warning(f"Could not read image size for {image_path}: {e}")
                        self.image_resolutions.append((self.resolution, self.resolution))

                    self.image_paths.append(image_path)
                    self.captions.append(caption)
                    self.repeats.append(repeats)

        logger.info(f"Loaded {len(self.image_paths)} unique images")

    def _prepare_cache(self):
        from PIL import Image
        from tqdm import tqdm
        try:
            import cv2
            import numpy as np
            HAS_CV2 = True
        except ImportError:
            HAS_CV2 = False
            logger.warning("OpenCV not available, WebP support may be limited")

        logger.info("Checking cache files...")
        missing_indices = []

        for idx, image_path in enumerate(self.image_paths):
            vae_cache = f"{image_path}.safetensors"
            #text_cache = f"{os.path.splitext(image_path)[0]}.txt.safetensors"

            if not os.path.exists(vae_cache): #or not os.path.exists(text_cache):
                missing_indices.append(idx)

        if missing_indices:
            logger.info(f"Generating {len(missing_indices)} cache files...")
            self.vae.eval().to(self.device)
            #self.text_encoder.eval().to(self.device)
            #self.clip_model.eval().to(self.device)

            with torch.no_grad():
                for idx in tqdm(missing_indices, desc="Caching"):
                    image_path = self.image_paths[idx]
                    #caption = self.captions[idx]
                    vae_cache = f"{image_path}.safetensors"
                    #text_cache = f"{os.path.splitext(image_path)[0]}.txt.safetensors"

                    try:
                        try:
                            image = Image.open(image_path).convert("RGB")
                        except Exception as e:
                            if HAS_CV2:
                                logger.warning(f"PIL failed for {image_path}, using OpenCV")
                                img_array = cv2.imread(image_path, cv2.IMREAD_COLOR)
                                if img_array is None:
                                    raise ValueError(f"Cannot load: {image_path}")
                                img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
                                image = Image.fromarray(img_array)
                            else:
                                raise e

                        bucket_reso = self.image_to_bucket.get(idx, (self.resolution, self.resolution))
                        target_width, target_height = bucket_reso

                        bucket_ratio = target_width / target_height
                        orig_ratio = image.size[0] / image.size[1]

                        if orig_ratio > bucket_ratio:
                            resize_height = target_height
                            resize_width = int(resize_height * orig_ratio)
                        else:
                            resize_width = target_width
                            resize_height = int(resize_width / orig_ratio)

                        transform = transforms.Compose([
                            transforms.Resize((resize_height, resize_width), interpolation=InterpolationMode.LANCZOS),
                            transforms.CenterCrop((target_height, target_width)),
                            transforms.ToTensor(),
                            transforms.Normalize([0.5], [0.5])
                        ])

                        pixel_values = transform(image).unsqueeze(0).to(self.device)

                        latents = self.vae.encode(pixel_values).latent_dist.mode()
                        scaling_factor = getattr(self.vae.config, 'scaling_factor', 0.13025)
                        latents = (latents * scaling_factor).squeeze(0).cpu()

                        save_file({
                            "latents": latents,
                            "width": torch.tensor(target_width),
                            "height": torch.tensor(target_height)
                        }, vae_cache)

                        """
                        with torch.autocast(device_type='cuda', dtype=self.dtype):
                            gemma_text = self.gemma3_prompt + caption if self.gemma3_prompt else caption
                            gemma_inputs = self.tokenizer(
                                [gemma_text], padding=True, pad_to_multiple_of=8,
                                truncation=True, max_length=1280, return_tensors="pt"
                            ).to(self.device)
                            gemma_outputs = self.text_encoder(**gemma_inputs, output_hidden_states=True)
                            cap_feats = gemma_outputs.hidden_states[-2].squeeze(0).to(dtype=self.dtype).cpu()
                            cap_mask = gemma_inputs.attention_mask.squeeze(0).cpu()

                            clip_inputs = self.clip_tokenizer(
                                [caption], padding=True, truncation=True,
                                max_length=2048, return_tensors="pt"
                            ).to(self.device)
                            clip_text_pooled = self.clip_model.get_text_features(**clip_inputs).squeeze(0).to(dtype=self.dtype).cpu()

                        save_file({
                            "cap_feats": cap_feats,
                            "cap_mask": cap_mask,
                            "clip_text_pooled": clip_text_pooled
                        }, text_cache)
                        """
                        
                    except Exception as e:
                        logger.error(f"Cache error for {image_path}: {e}")

            logger.info("Cache generation complete")
        else:
            logger.info("All cache files found")

    def _generate_buckets(self):
        max_reso = self.resolution
        max_tokens = (max_reso / 16) * (max_reso / 16)
        max_area = max_reso * max_reso

        assert self.bucket_reso_step % 8 == 0, "bucket_reso_step must be divisible by 8"
        assert self.min_bucket_reso % 8 == 0, "min_bucket_reso must be divisible by 8"
        assert self.max_bucket_reso % 8 == 0, "max_bucket_reso must be divisible by 8"

        aspect_ratios = [(1, 1), (3, 4), (4, 3), (9, 16), (16, 9)]
        buckets = set()

        def quantize(value: int) -> int:
            value = max(self.min_bucket_reso, min(self.max_bucket_reso, value))
            value = max(self.bucket_reso_step, (value // self.bucket_reso_step) * self.bucket_reso_step)
            return value

        for ar_w, ar_h in aspect_ratios:
            scale = math.sqrt(max_area / (ar_w * ar_h))
            width = int(scale * ar_w)
            height = int(scale * ar_h)

            width = quantize(width)
            height = quantize(height)

            # Reduce the dominant side if rounding pushed the area over the limit.
            while width * height > max_area and (width > self.min_bucket_reso or height > self.min_bucket_reso):
                if width >= height and width > self.min_bucket_reso:
                    width = max(self.min_bucket_reso, width - self.bucket_reso_step)
                elif height > self.min_bucket_reso:
                    height = max(self.min_bucket_reso, height - self.bucket_reso_step)
                else:
                    break

            buckets.add((width, height))

        self.bucket_resolutions = sorted(list(buckets), key=lambda x: x[0] / x[1])
        self.buckets = {}
        for reso in self.bucket_resolutions:
            self.buckets[reso] = []

        logger.info(
            f"Generated {len(self.bucket_resolutions)} fixed buckets with max {max_tokens:.0f} tokens "
            f"and area limit {max_area}"
        )
        logger.info(f"Bucket resolutions: {self.bucket_resolutions}")

    def _assign_buckets(self):
        ar_errors = []

        for idx, (img_w, img_h) in enumerate(self.image_resolutions):
            aspect_ratio = img_w / img_h

            closest_bucket = min(
                self.bucket_resolutions,
                key=lambda x: abs((x[0] / x[1]) - aspect_ratio)
            )

            bucket_ar = closest_bucket[0] / closest_bucket[1]
            ar_error = abs(bucket_ar - aspect_ratio)
            ar_errors.append(ar_error)

            self.buckets[closest_bucket].append(idx)
            self.image_to_bucket[idx] = closest_bucket

        bucket_info = {k: len(v) for k, v in self.buckets.items() if len(v) > 0}
        logger.info(f"Bucket assignment: {bucket_info}")

        if ar_errors:
            import numpy as np
            ar_errors_array = np.array(ar_errors)
            logger.info(f"Mean AR error: {np.mean(ar_errors_array):.4f}")
    
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        caption = self.captions[idx]
        bucket_reso = self.image_to_bucket.get(idx, (self.resolution, self.resolution))

        if self.use_cache:
            vae_cache = f"{image_path}.safetensors"
            #text_cache = f"{os.path.splitext(image_path)[0]}.txt.safetensors"

            vae_data = load_file(vae_cache)
            #text_data = load_file(text_cache)

            caption = self.process_caption(caption)
            
            return {
                "latents": vae_data['latents'],
                #"cap_feats": text_data['cap_feats'],
                #"cap_mask": text_data['cap_mask'],
                #"clip_text_pooled": text_data['clip_text_pooled'],
                "caption": caption,
                "cached": True,
            }
        else:
            caption = self.process_caption(caption)
            
            from PIL import Image
            try:
                import cv2
                import numpy as np
                HAS_CV2 = True
            except ImportError:
                HAS_CV2 = False

            try:
                image = Image.open(image_path).convert("RGB")
            except Exception as e:
                if HAS_CV2:
                    try:
                        logger.warning(f"PIL failed for {image_path}, trying OpenCV")
                        img_array = cv2.imread(image_path, cv2.IMREAD_COLOR)
                        if img_array is not None:
                            img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
                            image = Image.fromarray(img_array)
                        else:
                            raise ValueError(f"OpenCV also failed to load {image_path}")
                    except Exception as cv_err:
                        logger.error(f"Both PIL and OpenCV failed for {image_path}: {e}, {cv_err}")
                        image = Image.new("RGB", (self.resolution, self.resolution), color="black")
                else:
                    logger.error(f"Failed to load {image_path}: {e}")
                    image = Image.new("RGB", (self.resolution, self.resolution), color="black")

            target_width, target_height = bucket_reso

            bucket_ratio = target_width / target_height
            orig_ratio = image.size[0] / image.size[1]

            if orig_ratio > bucket_ratio:
                resize_height = target_height
                resize_width = int(resize_height * orig_ratio)
            else:
                resize_width = target_width
                resize_height = int(resize_width / orig_ratio)

            transform = transforms.Compose([
                transforms.Resize((resize_height, resize_width), interpolation=InterpolationMode.LANCZOS),
                transforms.CenterCrop((target_height, target_width)),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])
            ])

            return {"pixel_values": transform(image), "caption": caption, "cached": False}

    def process_caption(self, caption):
        """
        Input example:
            1. @artist, #character1\n1girl ||| tag1, tag2<split>@artist, #character1\nA girl stands....
            2. @artist\n1girl ||| tag1, tag2<split>@artist\nA girl stands....
            3. 1girl ||| tag1, tag2<split>A girl stands....
        """
        # 1. Wildcard 随机选取多重caption (基于"<split>"符分隔)
        if self.enable_wildcard:
            if "<split>" in caption:
                caption = random.choice(caption.split("<split>"))
            
        # 2. 处理 Artist Dropout
        if self.drop_artist_rate > 0 and random.random() < self.drop_artist_rate:
            # 检查是否包含换行符(是否有作者/角色信息)
            if "\n" in caption:
                parts = caption.split("\n", 1)
                first_line = parts[0]
                rest_of_caption = parts[1]
                # 使用正则表达式移除 @artist 标签
                new_first_line = re.sub(r'@\S+', '', first_line).strip()
                # 移除标签后，可能会在开头留下一个多余的逗号和空格，我们将其清理掉
                if new_first_line.startswith(','):
                    new_first_line = new_first_line[1:].strip()
                # 如果第一行还有内容，就把它和剩余部分重新组合起来
                if new_first_line:
                    caption = new_first_line + "\n" + rest_of_caption
                else:
                    # 如果第一行只剩下 artist 标签，那么丢弃后第一行就空了
                    caption = rest_of_caption
        
        # 3. Shuffle & Tag Dropout
        if self.shuffle_caption or self.caption_tag_dropout_rate > 0:
            fixed_tokens = []
            flex_tokens = []
            
            # 处理 keep_tokens_separator
            if self.keep_tokens_separator and self.keep_tokens_separator in caption:
                parts = caption.split(self.keep_tokens_separator, 1)
                fixed_part = parts[0]
                flex_part = parts[1]
                fixed_tokens = [t.strip() for t in fixed_part.split(self.caption_separator) if t.strip()]
                flex_tokens = [t.strip() for t in flex_part.split(self.caption_separator) if t.strip()]

                # Tag Shuffle
                if self.shuffle_caption:
                    random.shuffle(flex_tokens)
                # Tag Dropout
                if self.caption_tag_dropout_rate > 0:
                    flex_tokens = [t for t in flex_tokens if random.random() >= self.caption_tag_dropout_rate]
                # 重组
                caption = ", ".join(fixed_tokens + flex_tokens)
            else:
                # 对于NLP caption (不包含keep_tokens_separator)，不进行shuffle和tag dropout
                caption = caption
        
        # 4. Caption Dropout (整句丢弃, 用于CFG)
        if self.caption_dropout_rate > 0 and random.random() < self.caption_dropout_rate:
            caption = ""
        
        return caption


class BucketBatchSampler:
    def __init__(self, dataset, batch_size, shuffle=True, seed=42):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        self._first_epoch_sorted = False

        self.bucket_to_indices = {}
        for idx in range(len(dataset)):
            bucket_reso = dataset.image_to_bucket.get(idx, (dataset.resolution, dataset.resolution))
            if bucket_reso not in self.bucket_to_indices:
                self.bucket_to_indices[bucket_reso] = []

            for _ in range(dataset.repeats[idx]):
                self.bucket_to_indices[bucket_reso].append(idx)

        total_batches = sum(math.ceil(len(indices) / batch_size) for indices in self.bucket_to_indices.values())
        logger.info(f"BucketBatchSampler: {len(self.bucket_to_indices)} buckets, {total_batches} batches")

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)

        bucket_batches = []
        for bucket_reso, indices in self.bucket_to_indices.items():
            indices_copy = indices.copy()
            if self.shuffle:
                rng.shuffle(indices_copy)

            area = bucket_reso[0] * bucket_reso[1]
            batch_count = math.ceil(len(indices_copy) / self.batch_size)
            for batch_index in range(batch_count):
                start_idx = batch_index * self.batch_size
                batch = indices_copy[start_idx:start_idx + self.batch_size]
                bucket_batches.append({"area": area, "batch": batch})

        if self.shuffle:
            if not self._first_epoch_sorted:
                bucket_batches.sort(key=lambda x: x["area"], reverse=True)
                self._first_epoch_sorted = True
            else:
                rng.shuffle(bucket_batches)

        for entry in bucket_batches:
            yield entry["batch"]

    def __len__(self):
        return sum(math.ceil(len(indices) / self.batch_size) for indices in self.bucket_to_indices.values())

    def set_epoch(self, epoch):
        self.epoch = epoch


def collate_fn(batch):
    if batch[0].get("cached", False):
        """
        max_cap_len = max(example["cap_feats"].shape[0] for example in batch)

        cap_feats_list = []
        cap_mask_list = []
        for example in batch:
            cap_feat = example["cap_feats"]
            cap_mask = example["cap_mask"]
            current_len = cap_feat.shape[0]

            if current_len < max_cap_len:
                pad_len = max_cap_len - current_len
                cap_feat = torch.cat([cap_feat, torch.zeros(pad_len, cap_feat.shape[1], dtype=cap_feat.dtype)], dim=0)
                cap_mask = torch.cat([cap_mask, torch.zeros(pad_len, dtype=cap_mask.dtype)], dim=0)

            cap_feats_list.append(cap_feat)
            cap_mask_list.append(cap_mask)
        """
        return {
            "latents": torch.stack([example["latents"] for example in batch]),
            "captions": [example["caption"] for example in batch],
            #"cap_feats": torch.stack(cap_feats_list),
            #"cap_mask": torch.stack(cap_mask_list),
            #"clip_text_pooled": torch.stack([example["clip_text_pooled"] for example in batch]),
            "cached": True,
        }
    else:
        return {
            "pixel_values": torch.stack([example["pixel_values"] for example in batch]),
            "captions": [example["caption"] for example in batch],
            "cached": False,
        }


def load_encoders_only(config):
    base_model_path = config['Model']['base_model_path']
    trust_remote_code = config['Model'].get('trust_remote_code', True)
    model_index_path = os.path.join(base_model_path, "model_index.json")
    is_diffusers_format = os.path.exists(model_index_path)

    mixed_precision = config['Model'].get('mixed_precision', 'no')
    if mixed_precision == 'bf16':
        model_dtype = torch.bfloat16
    elif mixed_precision == 'fp16':
        model_dtype = torch.float16
    else:
        model_dtype = torch.float32

    logger.info("Loading encoders only (VAE, text_encoder, CLIP) for cache generation...")

    if is_diffusers_format:
        text_encoder_path = os.path.join(base_model_path, "text_encoder")
        text_encoder = AutoModel.from_pretrained(text_encoder_path, torch_dtype=model_dtype, trust_remote_code=trust_remote_code)
        tokenizer = AutoTokenizer.from_pretrained(text_encoder_path, trust_remote_code=trust_remote_code)
        tokenizer.padding_side = "right"

        clip_model_path = os.path.join(base_model_path, "clip_model")
        clip_model = AutoModel.from_pretrained(clip_model_path, torch_dtype=model_dtype, trust_remote_code=True)
        clip_tokenizer = AutoTokenizer.from_pretrained(clip_model_path, trust_remote_code=True)

        vae_path = os.path.join(base_model_path, "vae")
        vae = AutoencoderKL.from_pretrained(
            vae_path if os.path.exists(vae_path) else "black-forest-labs/FLUX.1-dev",
            torch_dtype=torch.float32,
            trust_remote_code=trust_remote_code
        )
    else:
        gemma_path = config['Model'].get('gemma_model_path', 'google/gemma-3-4b-it')
        clip_path = config['Model'].get('clip_model_path', 'jinaai/jina-clip-v2')

        text_encoder = AutoModel.from_pretrained(gemma_path, torch_dtype=model_dtype, trust_remote_code=trust_remote_code)
        tokenizer = AutoTokenizer.from_pretrained(gemma_path, trust_remote_code=trust_remote_code)
        tokenizer.padding_side = "right"

        clip_model = AutoModel.from_pretrained(clip_path, torch_dtype=model_dtype, trust_remote_code=True)
        clip_tokenizer = AutoTokenizer.from_pretrained(clip_path, trust_remote_code=True)

        vae_path = config['Model'].get('vae_path', 'stabilityai/sdxl-vae')
        vae = AutoencoderKL.from_pretrained(vae_path, torch_dtype=torch.float32, trust_remote_code=trust_remote_code)

    if config['Model'].get('vae_reflect_padding', False):
        logger.info("Enabling 'reflect' padding for VAE")
        for module in vae.modules():
            if isinstance(module, torch.nn.Conv2d):
                pad_h, pad_w = module.padding if isinstance(module.padding, tuple) else (module.padding, module.padding)
                if pad_h > 0 or pad_w > 0:
                    module.padding_mode = "reflect"
    vae.eval()
    vae.requires_grad_(False)
    text_encoder.eval()
    text_encoder.requires_grad_(False)
    clip_model.eval()
    clip_model.requires_grad_(False)

    logger.info("Encoders loaded successfully")
    return vae, text_encoder, tokenizer, clip_model, clip_tokenizer


def load_transformer_only(config):
    base_model_path = config['Model']['base_model_path']
    trust_remote_code = config['Model'].get('trust_remote_code', True)
    model_index_path = os.path.join(base_model_path, "model_index.json")
    is_diffusers_format = os.path.exists(model_index_path)

    mixed_precision = config['Model'].get('mixed_precision', 'no')
    if mixed_precision == 'bf16':
        model_dtype = torch.bfloat16
    elif mixed_precision == 'fp16':
        model_dtype = torch.float16
    else:
        model_dtype = torch.float32

    logger.info(f"Loading transformer only from: {base_model_path}")

    if is_diffusers_format:
        transformer_path = os.path.join(base_model_path, "transformer")
        config_path = os.path.join(transformer_path, "config.json")

        with open(config_path, 'r') as f:
            model_config = json.load(f)

        text_encoder_path = os.path.join(base_model_path, "text_encoder")
        text_encoder_config = AutoConfig.from_pretrained(text_encoder_path, trust_remote_code=trust_remote_code)
        cap_feat_dim = text_encoder_config.text_config.hidden_size

        model_name = model_config.get('_class_name', 'NextDiT_3B_GQA_patch2_Adaln_Refiner_WHIT_CLIP')

        model = models.__dict__[model_name](
            in_channels=model_config.get('in_channels', 16),
            qk_norm=True,
            cap_feat_dim=cap_feat_dim,
            clip_text_dim=model_config.get('clip_text_dim', 1024),
            clip_img_dim=model_config.get('clip_img_dim', 1024),
        )

        weight_path = os.path.join(transformer_path, "diffusion_pytorch_model.safetensors")
        if os.path.exists(weight_path):
            state_dict = load_file(weight_path)
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            if missing_keys:
                logger.warning(f"Missing keys: {len(missing_keys)}")
            if unexpected_keys:
                logger.warning(f"Unexpected keys: {len(unexpected_keys)}")

        if model_dtype != torch.float32:
            model = model.to(dtype=model_dtype)
            logger.info(f"Model converted to {model_dtype}")

    else:
        gemma_path = config['Model'].get('gemma_model_path', 'google/gemma-3-4b-it')

        text_encoder_config = AutoConfig.from_pretrained(gemma_path, trust_remote_code=trust_remote_code)
        cap_feat_dim = text_encoder_config.text_config.hidden_size

        transformer_path = config['Model'].get('transformer_path', None)

        model = models.NextDiT_3B_GQA_patch2_Adaln_Refiner_WHIT_CLIP(
            in_channels=16, qk_norm=True, cap_feat_dim=cap_feat_dim,
            clip_text_dim=1024, clip_img_dim=1024
        )

        if transformer_path and os.path.exists(transformer_path):
            state_dict = load_file(transformer_path) if transformer_path.endswith('.safetensors') else torch.load(transformer_path, map_location='cpu')
            model.load_state_dict(state_dict, strict=False)

        if model_dtype != torch.float32:
            model = model.to(dtype=model_dtype)
            logger.info(f"Model converted to {model_dtype}")

    model.train()
    logger.info(f"Transformer loaded: {model.parameter_count():,} params, cap_feat_dim={cap_feat_dim}")

    return model


def load_model_and_tokenizer(config):
    base_model_path = config['Model']['base_model_path']
    trust_remote_code = config['Model'].get('trust_remote_code', True)
    model_index_path = os.path.join(base_model_path, "model_index.json")
    is_diffusers_format = os.path.exists(model_index_path)

    mixed_precision = config['Model'].get('mixed_precision', 'no')
    if mixed_precision == 'bf16':
        model_dtype = torch.bfloat16
    elif mixed_precision == 'fp16':
        model_dtype = torch.float16
    else:
        model_dtype = torch.float32

    logger.info(f"Loading model from: {base_model_path}")

    if is_diffusers_format:
        logger.info("Diffusers format detected, auto-loading components")

        text_encoder_path = os.path.join(base_model_path, "text_encoder")
        text_encoder = AutoModel.from_pretrained(text_encoder_path, torch_dtype=model_dtype, trust_remote_code=trust_remote_code)
        tokenizer = AutoTokenizer.from_pretrained(text_encoder_path, trust_remote_code=trust_remote_code)
        tokenizer.padding_side = "right"

        clip_model_path = os.path.join(base_model_path, "clip_model")
        clip_model = AutoModel.from_pretrained(clip_model_path, torch_dtype=model_dtype, trust_remote_code=True)
        clip_tokenizer = AutoTokenizer.from_pretrained(clip_model_path, trust_remote_code=True)

        transformer_path = os.path.join(base_model_path, "transformer")
        config_path = os.path.join(transformer_path, "config.json")

        with open(config_path, 'r') as f:
            model_config = json.load(f)

        cap_feat_dim = text_encoder.config.text_config.hidden_size
        model_name = model_config.get('_class_name', 'NextDiT_3B_GQA_patch2_Adaln_Refiner_WHIT_CLIP')

        model = models.__dict__[model_name](
            in_channels=model_config.get('in_channels', 16),
            qk_norm=True,
            cap_feat_dim=cap_feat_dim,
            clip_text_dim=model_config.get('clip_text_dim', 1024),
            clip_img_dim=model_config.get('clip_img_dim', 1024),
        )

        weight_path = os.path.join(transformer_path, "diffusion_pytorch_model.safetensors")
        if os.path.exists(weight_path):
            state_dict = load_file(weight_path)
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            if missing_keys:
                logger.warning(f"Missing keys: {len(missing_keys)}")
            if unexpected_keys:
                logger.warning(f"Unexpected keys: {len(unexpected_keys)}")

        # Convert model to target dtype
        if model_dtype != torch.float32:
            model = model.to(dtype=model_dtype)
            logger.info(f"Model converted to {model_dtype}")

        vae_path = os.path.join(base_model_path, "vae")
        vae = AutoencoderKL.from_pretrained(
            vae_path if os.path.exists(vae_path) else "stabilityai/sdxl-vae",
            torch_dtype=torch.float32,
            trust_remote_code=trust_remote_code
        )

    else:
        logger.info("Loading from separate paths")

        gemma_path = config['Model'].get('gemma_model_path', 'google/gemma-3-4b-it')
        clip_path = config['Model'].get('clip_model_path', 'jinaai/jina-clip-v2')
        transformer_path = config['Model'].get('transformer_path', None)

        text_encoder = AutoModel.from_pretrained(gemma_path, torch_dtype=model_dtype, trust_remote_code=trust_remote_code)
        tokenizer = AutoTokenizer.from_pretrained(gemma_path, trust_remote_code=trust_remote_code)
        tokenizer.padding_side = "right"

        clip_model = AutoModel.from_pretrained(clip_path, torch_dtype=model_dtype, trust_remote_code=True)
        clip_tokenizer = AutoTokenizer.from_pretrained(clip_path, trust_remote_code=True)

        cap_feat_dim = text_encoder.config.text_config.hidden_size
        model = models.NextDiT_3B_GQA_patch2_Adaln_Refiner_WHIT_CLIP(
            in_channels=16, qk_norm=True, cap_feat_dim=cap_feat_dim,
            clip_text_dim=1024, clip_img_dim=1024
        )

        if transformer_path and os.path.exists(transformer_path):
            state_dict = load_file(transformer_path) if transformer_path.endswith('.safetensors') else torch.load(transformer_path, map_location='cpu')
            model.load_state_dict(state_dict, strict=False)

        # Convert model to target dtype
        if model_dtype != torch.float32:
            model = model.to(dtype=model_dtype)
            logger.info(f"Model converted to {model_dtype}")

        vae_path = config['Model'].get('vae_path', 'stabilityai/sdxl-vae')
        vae = AutoencoderKL.from_pretrained(vae_path, torch_dtype=torch.float32, trust_remote_code=trust_remote_code)

    model.train()
    vae.eval()
    vae.requires_grad_(False)
    text_encoder.eval()
    text_encoder.requires_grad_(False)
    clip_model.eval()
    clip_model.requires_grad_(False)

    logger.info(f"Model loaded: {model.parameter_count():,} params, cap_feat_dim={cap_feat_dim}")

    return model, vae, text_encoder, tokenizer, clip_model, clip_tokenizer


def setup_lora(model, config):
    """Apply adapter (PEFT) to model"""
    # 获取配置参数
    lora_rank = config['Model'].get('lora_rank', 32)
    lora_alpha = config['Model'].get('lora_alpha', lora_rank)
    lora_dropout = config['Model'].get('lora_dropout', 0.05)
    use_dora=config['Model'].get('use_dora', False)
    use_rslora=config['Model'].get('use_rslora', False)
    init_lora_weights=config['Model'].get('init_lora_weights', True)
    train_norm=config['Model'].get('train_norm', False)
    
    # 获取目标模块
    default_target_modules = [
        "attention.qkv",
        "attention.out",
        "feed_forward.w2",
        "time_text_embed.1",
        "clip_text_pooled_proj.1",
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
        init_lora_weights=init_lora_weights
    )
    
    peft_model = get_peft_model(model, lora_config, low_cpu_mem_usage=True)

    peft_model._adapter_type = "lora"
    peft_model._adapter_rank = lora_rank
    peft_model._adapter_alpha = lora_alpha

    trainable_params = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in peft_model.parameters())
    logger.info(
        f"LoRA applied: {trainable_params/1e6:.2f}M/{total_params/1e6:.2f}M trainable "
        f"({trainable_params/total_params*100:.2f}%)"
    )
    logger.info(f"  Target modules: {target_modules}")
    logger.info(f"  LoRA rank={lora_rank}, alpha={lora_alpha}")

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


def compute_loss(model, vae, text_encoder, tokenizer, clip_model, clip_tokenizer, transport, batch, device, gemma3_prompt="", use_multires_loss=True, multires_factor=4):
    """计算 Rectified Flow 训练损失"""
    if batch.get("cached", False):
        latents = batch["latents"].to(device)
        #cap_feats = batch["cap_feats"].to(device)
        #cap_mask = batch["cap_mask"].to(device)
        #clip_text_pooled = batch["clip_text_pooled"].to(device)
        captions = batch["captions"]
        batch_size = latents.shape[0]
    else:
        if vae is None:
            raise RuntimeError("VAE required for non-cached data")

        pixel_values = batch["pixel_values"].to(device)
        captions = batch["captions"]
        batch_size = pixel_values.shape[0]

        with torch.no_grad():
            latents = vae.encode(pixel_values).latent_dist.sample()
            scaling_factor = getattr(vae.config, 'scaling_factor', 0.13025)
            latents = latents * scaling_factor

    with torch.no_grad():
        # Gemma 编码
        gemma_texts = [gemma3_prompt + cap if gemma3_prompt else cap for cap in captions]
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
    # 原始分辨率损失
    loss = transport.training_losses(model, latents, model_kwargs)["loss"].mean()
    # 低分辨率损失
    if use_multires_loss:
        # 下采样 Latents
        latents_low = apply_average_pool(latents, factor=multires_factor)
        loss_low = transport.training_losses(model, latents_low, model_kwargs)["loss"].mean()
        # 求和
        loss = loss + loss_low
    return loss


def save_checkpoint(accelerator, model, optimizer, scheduler, step, config):
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
    accelerator.save(checkpoint, checkpoint_path)
    logger.info(f"Checkpoint saved: {checkpoint_path}")
    save_lora_model(accelerator, model, config, step)


def save_lora_model(accelerator, model, config, step=None):
    """Save adapter weights，两种都用“目录结构”保存"""
    output_dir = config['Model']['output_dir']
    output_name = config['Model']['output_name']
    os.makedirs(output_dir, exist_ok=True)

    save_dir = os.path.join(output_dir, f"{output_name}_step_{step}" if step else output_name)
    os.makedirs(save_dir, exist_ok=True)

    unwrapped = accelerator.unwrap_model(model)
    
    save_kwargs = {}
    pissa_init_dir = config['Model'].get('pissa_init_dir')
    # 如果配置中有 pissa_init_dir 且文件存在，则启用转换参数
    if pissa_init_dir and os.path.exists(pissa_init_dir):
        save_kwargs["path_initial_model_for_weight_conversion"] = pissa_init_dir
        if accelerator.is_main_process:
            logger.info("Converting PiSSA to standard LoRA (Delta W) for ComfyUI compatibility...")
            
    unwrapped.save_pretrained(
        save_dir,
        is_main_process=accelerator.is_main_process,
        state_dict=accelerator.get_state_dict(model), 
        safe_serialization=True
        **save_kwargs
    )

    if accelerator.is_main_process:
        logger.info(f"PEFT LoRA model saved to: {save_dir}")

def load_checkpoint(accelerator, model, optimizer, scheduler, config):
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
        kwargs_handlers=[ddp_kwargs]
    )

    set_seed(42)

    if not config['Optimization'].get('use_flash_attention_2', False):
        torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False)

    use_cache = config['Model'].get('use_cache', True)
    mixed_precision = config['Model'].get('mixed_precision', 'no')
    cache_dtype = torch.bfloat16 if mixed_precision == 'bf16' else (torch.float16 if mixed_precision == 'fp16' else torch.float32)
    gemma3_prompt = config['Model'].get('gemma3_prompt', '')
    resolution = config['Model']['resolution']
    shuffle_caption = config['Model'].get('shuffle_caption', False)
    keep_tokens_separator = config['Model'].get('keep_tokens_separator', "|||")
    enable_wildcard = config['Model'].get('enable_wildcard', False)
    caption_dropout_rate = config['Model'].get('caption_dropout_rate', 0.0)
    caption_tag_dropout_rate = config['Model'].get('caption_tag_dropout_rate', 0.0)
    drop_artist_rate = config['Model'].get('drop_artist_rate', 0.0)
    use_multires_loss = config['Model'].get('use_multires_loss', True)
    multires_factor = config['Model'].get('multires_factor', 4)

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
        vae, text_encoder, tokenizer, clip_model, clip_tokenizer = load_encoders_only(config)

        if not cache_complete:
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
            )
            del dataset
            import gc
            gc.collect()
            torch.cuda.empty_cache()

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
            shuffle_caption=shuffle_caption,
            keep_tokens_separator=keep_tokens_separator,
            enable_wildcard=enable_wildcard,
            caption_dropout_rate=caption_dropout_rate,
            caption_tag_dropout_rate=caption_tag_dropout_rate,
            drop_artist_rate=drop_artist_rate,
        )
    else:
        model, vae, text_encoder, tokenizer, clip_model, clip_tokenizer = load_model_and_tokenizer(config)

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

    if config['Model'].get('gradient_checkpointing', True):
        model.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing enabled")

    # 检查是否使用了 PiSSA 初始化
    init_lora_weights = config['Model'].get('init_lora_weights', True)
    # 确保 init_lora_weights 是字符串并且以 "pissa" 开头 (例如 "pissa" 或 "pissa_niter_4")
    if isinstance(init_lora_weights, str) and init_lora_weights.startswith("pissa"):
        # 定义初始权重保存路径
        pissa_init_dir = os.path.join(config['Model']['output_dir'], "pissa_init")
        # 将路径存入 config 以便后续 save_lora_model 使用
        config['Model']['pissa_init_dir'] = pissa_init_dir
        
        # 仅在主进程保存，避免多进程冲突
        if accelerator.is_main_process:
            if not os.path.exists(pissa_init_dir):
                logger.info(f"Detected PiSSA initialization. Saving initial state to {pissa_init_dir} for later conversion...")
                # 保存未训练的初始 LoRA 权重 (A_0, B_0)
                # 注意：这里需要 unwrap 或者是直接用 model (如果是 PeftModel)
                model.save_pretrained(pissa_init_dir)
            else:
                logger.info(f"PiSSA initial state already exists at {pissa_init_dir}")
        
        # 等待主进程保存完毕
        accelerator.wait_for_everyone()

    num_workers = config['Model'].get('dataloader_num_workers', 4)
    batch_size = config['Model']['train_batch_size']

    if config['Model'].get('enable_bucket', True):
        batch_sampler = BucketBatchSampler(dataset, batch_size=batch_size, shuffle=True, seed=42)
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
    print_memory_usage("After LoRA", args.profiler)

    # 执行 EVA 初始化
    init_method = config['Model'].get('init_lora_weights', True)
    if init_method == "eva" and accelerator.is_main_process:
        logger.info("Initializing LoRA weights with EVA (Explained Variance Adaptation)...")
        
        # 准备模型的数据类型
        target_dtype = torch.bfloat16 if mixed_precision == 'bf16' else (torch.float16 if mixed_precision == 'fp16' else torch.float32)
        
        # 创建数据生成器
        data_gen = get_eva_batch_generator(
            dataloader=train_dataloader,
            device=accelerator.device,
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            clip_model=clip_model,
            clip_tokenizer=clip_tokenizer,
            gemma3_prompt=gemma3_prompt,
            dtype=target_dtype,
            num_steps=config['Model'].get('eva_num_steps', 0)
        )
        
        # 执行初始化
        initialize_lora_eva_weights(model, data_gen)
        
        logger.info("EVA initialization complete.")
        
        # 等待所有进程同步（如果使用 DDP）
        accelerator.wait_for_everyone()

    optimizer = setup_optimizer(model, config)
    scheduler, num_training_steps = setup_scheduler(optimizer, config, train_dataloader)

    print_memory_usage("Before accelerator.prepare", args.profiler)
    model, optimizer, train_dataloader, scheduler = accelerator.prepare(model, optimizer, train_dataloader, scheduler)
    print_memory_usage("After accelerator.prepare", args.profiler)
    
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

    start_step = load_checkpoint(accelerator, model, optimizer, scheduler, config)

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

    steps_per_epoch = len(train_dataloader)
    start_epoch = start_step // steps_per_epoch
    steps_to_skip_in_first_epoch = start_step % steps_per_epoch

    if start_step > 0:
        logger.info(f"Resuming from epoch {start_epoch+1}, will skip {steps_to_skip_in_first_epoch} steps in first epoch")
        
    has_profiled_micro_step = False
    for epoch in range(start_epoch, config['Model']['num_epochs']):
        if config['Model'].get('enable_bucket', True) and hasattr(train_dataloader, 'batch_sampler'):
            if hasattr(train_dataloader.batch_sampler, 'set_epoch'):
                train_dataloader.batch_sampler.set_epoch(epoch)

        epoch_losses = []
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

            # 使用 accumulate 上下文
            with accelerator.accumulate(model):
                # ================= Profiler: Before Forward =================
                # 仅在当前累积周期的第一个微步打印
                if is_profiling_step and not has_profiled_micro_step:
                    print_memory_usage("Before first forward pass", args.profiler)

                # 1. 计算 Loss
                loss = compute_loss(
                    model, 
                    vae, 
                    text_encoder, 
                    tokenizer, 
                    clip_model, 
                    clip_tokenizer, 
                    transport, 
                    batch, 
                    accelerator.device, 
                    gemma3_prompt,
                    use_multires_loss,
                    multires_factor
                )
                
                # 记录 Loss (Accelerate 会自动处理累积步的平均，这里直接 append 即可)
                epoch_losses.append(loss.item())

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

        avg_epoch_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0.0
        logger.info(f"Epoch {epoch+1}/{config['Model']['num_epochs']} completed - Average Loss: {avg_epoch_loss:.4f}")

        if save_epochs_interval == 0 or (epoch + 1) % save_epochs_interval == 0:
            save_checkpoint(accelerator, model, optimizer, scheduler, global_step, config)
            logger.info(f"Checkpoint saved at epoch {epoch+1}")

    logger.info("Training complete, saving final model")
    save_lora_model(accelerator, model, config)

    if accelerator.is_main_process:
        accelerator.end_training()

    logger.info("Training finished")


if __name__ == "__main__":
    main()









































