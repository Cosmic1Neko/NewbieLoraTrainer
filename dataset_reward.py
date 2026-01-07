import os
import json
import math
import cv2
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
import hpsv2
from mmpose.apis import MMPoseInferencer

class ImageScorer:
    """
    综合图像评分器。
    集成基于 MMPose 的解剖得分 (Anatomy Score)
    以及基于 HPSv2 的人类偏好得分 (Human Preference Score)。
    """
    def __init__(self, 
                 pose_model: str = 'rtmw-x_8xb320-270e_cocktail14-384x288',
                 device: str = 'cuda:0'):
        # 初始化 MMPose 推理器
        print(f"正在初始化 MMPose 推理器 (设备: {device})...")
        self.inferencer = MMPoseInferencer(
            pose2d=pose_model,
            device=device
        )
        
        # 预定义 COCO-WholeBody 关键点索引区间
        self.groups = {
            'body': list(range(0, 17)),      # 身体核心 17 点
            'l_feet': list(range(17, 20)),     # 左足部 3 点
            'r_feet': list(range(20, 22)),     # 右足部 3 点
            'face': list(range(23, 91)),      # 脸部 68 点
            'l_hand': list(range(91, 112)),   # 左手 21 点
            'r_hand': list(range(112, 133))   # 右手 21 点
        }
        
        # 默认权重权重分配
        self.default_group_weights = {
            'body': 0.3,
            'l_hand': 0.175,
            'r_hand': 0.175,
            'l_feet': 0.175,
            'r_feet': 0.175
        }
        
        self.target_pixels = 1024 * 1024

    def _preprocess_image(self, image_path: str) -> np.ndarray:
        """图片预处理，保持长宽比缩放至目标像素量"""
        img_pil = Image.open(image_path).convert('RGB')
        orig_w, orig_h = img_pil.size
        scale_factor = math.sqrt(self.target_pixels / (orig_w * orig_h))
        resize_w, resize_h = int(round(orig_w * scale_factor)), int(round(orig_h * scale_factor))
        processed_img = img_pil.resize((resize_w, resize_h), resample=Image.LANCZOS)
        return np.array(processed_img)

    def calculate_group_score(self, image_path: str) -> dict:
        """核心评分逻辑：针对画面中出现的每个主要部位独立打分"""
        try:
            img_rgb = self._preprocess_image(image_path)
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        except Exception as e:
            print(f"无法读取图片 {image_path}: {e}")
            return {k: 0.0 for k in self.groups.keys()}

        # 执行推理
        result = self.inferencer(img_bgr, return_vis=False, show=False, bbox_thr=0.25, nms_thr=0.25)
        result_data = next(result)

        group_names = self.groups.keys()
        if not result_data or 'predictions' not in result_data or len(result_data['predictions']) == 0:
            return {k: 0.0 for k in group_names}
        
        predictions = result_data['predictions'][0]
        # 用于存储每个人每个部位的 (分数, 权重)
        # 结构: { 'body': [(score1, weight1), (score2, weight2)], ... }
        weighted_data = {k: [] for k in group_names}

        for person in predictions:  
            # 获取 BBox
            bbox = person.get('bbox', [0, 0, 0, 0])
            area = (bbox[0][2] - bbox[0][0]) * (bbox[0][3] - bbox[0][1])
            weight = max(area, 1e-5)

            kp_scores = np.array(person['keypoint_scores']).squeeze()
            for name, indices in self.groups.items():
                group_scores = kp_scores[indices]
                avg_scores = float(np.mean(group_scores))
                # 低于阈值视为无效部位
                if (name in ['l_feet', 'r_feet']) and (avg_scores >= 3.5): # 每只脚3个点
                    weighted_data[name].append((avg_scores, weight))
                elif (name in ['l_hand', 'r_hand']) and (avg_scores >= 4): # 每只手21个点
                    weighted_data[name].append((avg_scores, weight))
                elif (name in ['body']) and (avg_scores >= 3.5): # 身体17个点:
                    weighted_data[name].append((avg_scores, weight))
                elif (name in ['face']) and (avg_scores >= 3.5): # 脸68个点:
                    weighted_data[name].append((avg_scores, weight))
                else:
                    weighted_data[name].append((float(0.0), weight))
        
        scores = {}
        for name, data_list in weighted_data.items():
            if not data_list:
                scores[name] = 0.0
                continue
            total_weighted_score = sum(s * w for s, w in data_list)
            total_weight = sum(w for s, w in data_list)
            scores[name] = total_weighted_score / total_weight
                
        return scores
    
    def calculate_anatomy_score(self, image_path: str, group_weights: dict = None) -> dict:
        """计算最终加权的解剖得分"""
        if group_weights is None:
            group_weights = self.default_group_weights

        scores = self.calculate_group_score(image_path)
        weighted_sum = 0.0
        active_weight_total = 0.0
        
        for part, weight in group_weights.items():
            part_score = scores.get(part, 0.0) 
            weighted_sum += part_score * weight
            active_weight_total += weight
            
        if active_weight_total == 0:
            scores['total'] = 0.0
        else:
            scores['total'] = weighted_sum / active_weight_total
        return scores
    
    def calculate_hps_score(self, image_path: str, prompt: str) -> float:
        """计算 HPSv2 得分"""
        try:
            # hpsv2.score 返回 list
            result = hpsv2.score(image_path, prompt, hps_version="v2.1")
            return float(result[0])
        except Exception as e:
            print(f"HPSv2 计算失败 ({image_path}): {e}")
            return 0.0

    def evaluate(self, image_path: str, prompt: str, w_anatomy=1.0, w_hps=20.0) -> dict:
        """综合评估接口"""
        anatomy_results = self.calculate_anatomy_score(image_path)
        hps_score = self.calculate_hps_score(image_path, prompt)
        
        # 归一化解剖分 (通常在0-1之间) 与 HPS 分 (通常在0.2-0.3左右) 进行加权
        total_score = (w_anatomy * anatomy_results['total']) + (w_hps * hps_score)
        
        return {
            "total_reward": total_score,
            "anatomy_score": anatomy_results['total'],
            "hps_score": hps_score,
            "anatomy_details": {k:v for k,v in anatomy_results.items() if k != 'total'}
        }

def parse_args():
    parser = argparse.ArgumentParser(description="为 SDPO 离线数据集进行奖励评分")
    parser.add_argument("--input_json", type=str, required=True, help="offline_dataset.py 输出的 json 路径")
    parser.add_argument("--output_json", type=str, default=None, help="带有得分的输出路径")
    parser.add_argument("--w_anatomy", type=float, default=1.0, help="解剖得分权重")
    parser.add_argument("--w_hps", type=float, default=20.0, help="偏好得分权重")
    parser.add_argument("--device", type=str, default="cuda:0", help="推理设备")
    return parser.parse_args()

def main():
    args = parse_args()
    
    if not os.path.exists(args.input_json):
        print(f"错误: 找不到输入文件 {args.input_json}")
        return

    # 加载数据
    with open(args.input_json, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    # 初始化评分器
    scorer = ImageScorer(device=args.device)

    print(f"开始评分，共计 {len(dataset)} 个 Prompt 组...")
    real_reward_cache = {}

    # 开始遍历评分
    for item in tqdm(dataset, desc="Scoring Dataset"):
        prompt = item['caption']
        
        # 1. 评分真实样本
        real_path = item['real_image_path']
        cache_key = (real_path, prompt)
        if cache_key in real_reward_cache:
            item['real_reward'] = real_reward_cache[cache_key]
        else:
            if os.path.exists(real_path):
                reward_info = scorer.evaluate(
                    real_path, prompt, w_anatomy=args.w_anatomy, w_hps=args.w_hps
                )
                item['real_reward'] = reward_info
                # 存入缓存
                real_reward_cache[cache_key] = reward_info
            else:
                item['real_reward'] = {"total_reward": 0.0, "anatomy_score": 0.0, "hps_score": 0.0}

        # 2. 评分生成样本
        gen_rewards = []
        for gen_path in item['generated_image_paths']:
            if os.path.exists(gen_path):
                reward_info = scorer.evaluate(
                    gen_path, prompt, w_anatomy=args.w_anatomy, w_hps=args.w_hps
                )
                gen_rewards.append(reward_info)
            else:
                gen_rewards.append({"total_reward": 0.0, "anatomy_score": 0.0, "hps_score": 0.0})
        
        item['generated_rewards'] = gen_rewards

    # 保存结果
    output_path = args.output_json if args.output_json else args.input_json.replace(".json", "_with_rewards.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    
    print(f"🎉 评分完成！结果已保存至: {output_path}")

if __name__ == "__main__":
    main()
