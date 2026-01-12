import os
import random
import math
import logging
import re
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from safetensors.torch import load_file
from PIL import Image
import numpy as np
from tqdm import tqdm
from safetensors.torch import load_file, save_file

# 设置 logger
logger = logging.getLogger(__name__)

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
            files.sort()
            
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
                        scaling_factor = getattr(self.vae.config, 'scaling_factor', 0.3611)
                        shift_factor = getattr(self.vae.config, 'shift_factor', 0.1159)
                        latents = (latents - shift_factor) * scaling_factor
                        latents = latents.squeeze(0).cpu()

                        save_file({
                            "latents": latents,
                            "width": torch.tensor(target_width),
                            "height": torch.tensor(target_height)
                        }, vae_cache)
                        
                    except Exception as e:
                        logger.error(f"Cache error for {image_path}: {e}")

            logger.info("Cache generation complete")
        else:
            logger.info("All cache files found")

    def _generate_buckets(self):
        """
        使用 sd-scripts 风格的动态分箱策略
        不再局限于预设的 5 个长宽比，而是根据 max_area 和 bucket_reso_step 动态生成所有可能的 bucket
        """
        max_reso = self.resolution
        max_area = max_reso * max_reso

        # 确保步长和最小尺寸合法
        assert self.bucket_reso_step % 8 == 0, "bucket_reso_step must be divisible by 8"
        assert self.min_bucket_reso % 8 == 0, "min_bucket_reso must be divisible by 8"
        assert self.max_bucket_reso % 8 == 0, "max_bucket_reso must be divisible by 8"

        buckets = set()

        # 1. 添加正方形 Bucket
        # 计算理论上的最大正方形边长
        sq_side = int(math.sqrt(max_area) // self.bucket_reso_step) * self.bucket_reso_step
        # 限制在允许的范围内
        sq_side = max(self.min_bucket_reso, min(self.max_bucket_reso, sq_side))
        buckets.add((sq_side, sq_side))

        # 2. 动态生成各种长宽比的 Bucket
        # 逻辑：遍历所有可能的宽度，计算对应的最大高度
        width = self.min_bucket_reso
        while width <= self.max_bucket_reso:
            # 根据面积恒定公式 area = w * h => h = area / w
            height = int((max_area // width) // self.bucket_reso_step) * self.bucket_reso_step
            
            # 限制高度不超过最大分辨率
            height = min(self.max_bucket_reso, height)

            # 如果高度也符合最小限制，则添加该 bucket (以及它的转置)
            if height >= self.min_bucket_reso:
                buckets.add((width, height))
                buckets.add((height, width))

            # 增加宽度步长
            width += self.bucket_reso_step

        # 排序并初始化
        self.bucket_resolutions = sorted(list(buckets), key=lambda x: x[0] / x[1])
        self.buckets = {}
        for reso in self.bucket_resolutions:
            self.buckets[reso] = []

        logger.info(
            f"Generated {len(self.bucket_resolutions)} dynamic buckets "
            f"(max_area: {max_area}, step: {self.bucket_reso_step}, min: {self.min_bucket_reso}, max: {self.max_bucket_reso})"
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
    def __init__(self, dataset, batch_size, shuffle=True, seed=42, num_replicas=1, rank=0):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self._first_epoch_sorted = False

        self.bucket_to_indices = {}
        for idx in range(len(dataset)):
            bucket_reso = dataset.image_to_bucket.get(idx, (dataset.resolution, dataset.resolution))
            if bucket_reso not in self.bucket_to_indices:
                self.bucket_to_indices[bucket_reso] = []

            for _ in range(dataset.repeats[idx]):
                self.bucket_to_indices[bucket_reso].append(idx)

        # 计算全局总 batch 数（仅用于日志，实际长度在 __len__ 中计算）
        total_batches = sum(math.ceil(len(indices) / batch_size) for indices in self.bucket_to_indices.values())
        logger.info(f"BucketBatchSampler: {len(self.bucket_to_indices)} buckets, {total_batches} batches (Global)")

    def __iter__(self):
        # 确保每个 epoch 的随机种子不同，且各卡同步
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
                # 第一个 epoch 按面积排序，减少显存碎片
                bucket_batches.sort(key=lambda x: x["area"], reverse=True)
                self._first_epoch_sorted = True
            else:
                rng.shuffle(bucket_batches)

        # ==========================================
        # 处理 DDP 数据分配，防止死锁
        # ==========================================
        total_len = len(bucket_batches)
        # 计算需要丢弃的 batch 数量，确保能被 GPU 数整除
        remainder = total_len % self.num_replicas
        if remainder > 0:
            # 丢弃末尾多余的 batch，保证所有 GPU 步数一致
            bucket_batches = bucket_batches[: -remainder]
        
        # 分布式切片：Rank 0 拿 0, 2, 4...; Rank 1 拿 1, 3, 5...
        local_batches = bucket_batches[self.rank::self.num_replicas]

        for entry in local_batches:
            yield entry["batch"]

    def __len__(self):
        # 计算全局总 batch 数
        total_batches = sum(math.ceil(len(indices) / self.batch_size) for indices in self.bucket_to_indices.values())
        # 减去会被丢弃的部分
        remainder = total_batches % self.num_replicas
        adjusted_total = total_batches - remainder
        # 返回单卡上的 batch 数
        return adjusted_total // self.num_replicas

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

class DPODataset(Dataset):
    def __init__(
        self,
        preference_json_path,
        resolution=512,
        caption_dropout_rate=0.1,
        real_ratio=0.2,
    ):
        """
        Args:
            preference_json_path: 人工精炼后的偏好对 JSON 路径，包含真实/生成图片，以及偏好选择
            resolution: 图片分辨率
            caption_dropout_rate:由于DPO需要保持CFG能力，需要保留caption dropout
            real_ratio: 混合真实样本的比例
        """
        self.resolution = resolution
        self.caption_dropout_rate = caption_dropout_rate
        self.real_ratio = real_ratio

        # 加载偏好对数据
        # 预期格式示例: 
        # [
        #   {
        #     "real": "path/to/real.jpg",
        #     "generated": ["path/to/gen1.jpg", "path/to/gen2.jpg"], 
        #     "caption": "a cute cat"
        #     "dpo_pair": {
        #        "chosen": 'path1',
        #        "rejected": 'path2',
        #        "margin": 1.0
        #     }，
        #     "annotation_status": 'confirm'
        #   },
        #   ...
        # ]
        # 加载单一的 JSON 数据集
        with open(preference_json_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        # 过滤掉被标记为 rejected 的数据
        self.all_data = [
            item for item in raw_data
            if item.get("annotation_status") != "rejected"
        ]
        # 筛选出有效的偏好对数据
        # 必须包含 dpo_pair 且其中有 chosen 和 rejected 路径
        self.preference_data = [
            item for item in self.all_data 
            if item.get("dpo_pair") and item.get("dpo_pair", {}).get("chosen") and item.get("dpo_pair", {}).get("rejected")
        ]
        # 筛选出可用于正则化的数据
        self.real_reg_data = [
            item for item in self.all_data
            if item.get("real") and item.get("generated")
        ]

        self.transform = transforms.Compose([
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.LANCZOS),
            transforms.CenterCrop(resolution), 
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        
        print(f"Dataset loaded from {preference_json_path}")
        print(f"  - Total Loaded: {len(raw_data)}")
        print(f"  - After Filtering 'rejected': {len(self.all_data)}")
        print(f"  - Valid Preference Pairs (DPO): {len(self.preference_data)}")
        print(f"  - Valid Real Regularization Pairs: {len(self.real_reg_data)}")
        
        if len(self.preference_data) == 0:
            print("Warning: No valid preference pairs found (checked 'dpo_pair' field). DPO training might fail.")

    def __len__(self):
        # 长度定义为偏好数据集的长度，混合逻辑在 __getitem__ 动态处理
        return len(self.preference_data)

    def load_image(self, path):
        try:
            image = Image.open(path).convert("RGB")
            return self.transform(image)
        except Exception as e:
            print(f"Error loading image {path}: {e}")
            # 返回全黑图作为 fallback，防止训练崩溃
            return torch.zeros((3, self.resolution, self.resolution))

    def _get_real_pair(self):
        """从真实vs生成数据集中采样，处理生成图片列表"""
        if not self.real_reg_data:
            return None, None, ""
        
        item = random.choice(self.real_reg_data)
        
        # 1. 获取 Chosen (Real)
        chosen_path = item.get("real")
        
        # 2. 获取 Rejected (Generated)
        # offline_dataset.py 可能会生成 num_samples > 1，所以 generated 可能是列表
        gen_candidates = item.get("generated")
        
        rejected_path = None
        if isinstance(gen_candidates, list):
            if len(gen_candidates) > 0:
                rejected_path = random.choice(gen_candidates)
        elif isinstance(gen_candidates, str):
            rejected_path = gen_candidates
            
        caption = item.get("caption", "")
        return chosen_path, rejected_path, caption

    def _get_preference_pair(self, idx):
        """从偏好数据集中获取 (DPO Target)"""
        item = self.preference_data[idx]
        
        # 直接读取 dpo_pair 结构
        dpo_pair = item.get("dpo_pair", {})
        chosen_path = dpo_pair.get("chosen")
        rejected_path = dpo_pair.get("rejected")
        caption = item.get("caption", "")
        
        return chosen_path, rejected_path, caption

    def __getitem__(self, idx):
        # 策略：按比例混合真实数据与偏好数据
        # 如果 real_ratio > 0，则有概率采样 Real vs Gen 数据作为正则项
        use_real_regularization = (
            self.real_ratio > 0 
            and len(self.real_reg_data) > 0 
            and random.random() < self.real_ratio
        )

        if use_real_regularization:
            chosen_path, rejected_path, caption = self._get_real_pair()
        else:
            chosen_path, rejected_path, caption = self._get_preference_pair(idx)

        # 加载图片
        chosen_img = self.load_image(chosen_path)
        rejected_img = self.load_image(rejected_path)

        # Caption Dropout (维持 CFG 能力)
        if random.random() < self.caption_dropout_prob:
            caption = ""

        return {
            "pixel_values_chosen": chosen_img,
            "pixel_values_rejected": rejected_img,
            "caption": caption
        }
