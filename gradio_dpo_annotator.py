"""
python gradio_dpo_annotator.py --input_json /root/autodl-tmp/gen_dataset/dataset_scored.json
"""

import gradio as gr
import json
import os
import argparse
import math
import numpy as np
from PIL import Image

def parse_args():
    parser = argparse.ArgumentParser(description="DPO 数据集人工标注工具")
    parser.add_argument("--input_json", type=str, required=True, help="dataset_reward.py 输出的 scored.json 路径")
    parser.add_argument("--output_json", type=str, default=None, help="保存标注结果的路径 (默认覆盖输入文件)")
    parser.add_argument("--image_root", type=str, default="", help="如果json中使用的是相对路径，在此指定图片根目录")
    parser.add_argument("--port", type=int, default=7860, help="Gradio 服务端口")
    parser.add_argument("--share", action="store_true", help="是否创建公开分享链接")
    parser.add_argument("--target_pixels", type=int, default=1024*1024, help="图片显示时的目标像素总量(用于缩放加速)")
    return parser.parse_args()

class DataManager:
    def __init__(self, input_path, output_path, image_root, target_pixels=1024*1024):
        self.input_path = input_path
        self.output_path = output_path if output_path else input_path
        self.image_root = image_root
        self.target_pixels = target_pixels
        self.data = []
        self.load_data()
    
    def load_data(self):
        if os.path.exists(self.input_path):
            with open(self.input_path, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
            print(f"已加载 {len(self.data)} 条数据")
        else:
            print(f"错误: 找不到文件 {self.input_path}")
            self.data = []

    def save_data(self):
        try:
            with open(self.output_path, 'w', encoding='utf-8') as f:
                json.dump(self.data, f, ensure_ascii=False, indent=2)
            return f"成功保存到 {self.output_path} (共 {len(self.data)} 条)"
        except Exception as e:
            return f"保存失败: {str(e)}"

    def get_image_path(self, path):
        if self.image_root and not os.path.isabs(path):
            full_path = os.path.join(self.image_root, path)
        else:
            full_path = path
        return full_path

    def _preprocess_image(self, image_path: str):
        """图片预处理，保持长宽比缩放至目标像素量"""
        if not image_path or not os.path.exists(image_path):
            return None
            
        try:
            img_pil = Image.open(image_path).convert('RGB')
            orig_w, orig_h = img_pil.size
            # 计算缩放比例
            scale_factor = math.sqrt(self.target_pixels / (orig_w * orig_h))
            
            # 只有当图片大于目标大小时才缩小
            if scale_factor < 1.0:
                resize_w, resize_h = int(round(orig_w * scale_factor)), int(round(orig_h * scale_factor))
                img_pil = img_pil.resize((resize_w, resize_h), resample=Image.LANCZOS)
            
            return np.array(img_pil)
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return None

    def get_scores_text(self, item, img_path):
        if 'gen_rewards' not in item:
            return "无分数数据"
        
        target_info = None
        norm_img_path = os.path.normpath(img_path)
        
        for reward in item['gen_rewards']:
            norm_reward_path = os.path.normpath(reward['path'])
            if norm_img_path == norm_reward_path or norm_img_path.endswith(norm_reward_path) or norm_reward_path.endswith(norm_img_path):
                target_info = reward
                break
        
        if target_info:
            scores = target_info.get('scores', {})
            total = scores.get('total_reward', 0)
            hps = scores.get('hps_score', 0)
            anatomy = scores.get('anatomy_score', 0)
            return f"Total: {total:.4f}\nHPSv2: {hps:.4f}\nAnatomy: {anatomy:.4f}"
        return "未找到分数"

    def get_item_data(self, index):
        if index < 0 or index >= len(self.data):
            return None
        
        item = self.data[index]
        
        if 'dpo_pair' not in item or not item['dpo_pair']:
            return {
                "index": index,
                "caption": item.get('caption', ""),
                "real_img": None,
                "chosen_img": None,
                "rejected_img": None,
                "chosen_score": "无 DPO 配对",
                "rejected_score": "无 DPO 配对",
                "status": item.get('annotation_status', 'unannotated')
            }

        # 获取路径
        real_path = self.get_image_path(item.get('real_image_path', ''))
        chosen_path = self.get_image_path(item['dpo_pair']['chosen'])
        rejected_path = self.get_image_path(item['dpo_pair']['rejected'])

        # --- 变更点：这里不再返回 path，而是返回处理后的 image array ---
        # 即使图片很大，也会被缩放到约 1024x1024，传输极快
        real_img_data = self._preprocess_image(real_path)
        chosen_img_data = self._preprocess_image(chosen_path)
        rejected_img_data = self._preprocess_image(rejected_path)

        chosen_score = self.get_scores_text(item, item['dpo_pair']['chosen'])
        rejected_score = self.get_scores_text(item, item['dpo_pair']['rejected'])

        return {
            "index": index,
            "caption": item.get('caption', ""),
            "real_img": real_img_data,
            "chosen_img": chosen_img_data,
            "rejected_img": rejected_img_data,
            "chosen_score": chosen_score,
            "rejected_score": rejected_score,
            "status": item.get('annotation_status', 'unannotated')
        }

    def mark_status(self, index, status, swap=False):
        if index < 0 or index >= len(self.data):
            return
        
        self.data[index]['annotation_status'] = status
        
        if swap and 'dpo_pair' in self.data[index]:
            pair = self.data[index]['dpo_pair']
            old_chosen = pair['chosen']
            old_rejected = pair['rejected']
            
            pair['chosen'] = old_rejected
            pair['rejected'] = old_chosen
            
            if 'margin' in pair:
                pair['margin'] = -pair['margin']
                
            self.data[index]['dpo_pair'] = pair

    def find_next_unannotated(self, start_index):
        for i in range(start_index + 1, len(self.data)):
            if self.data[i].get('annotation_status', 'unannotated') == 'unannotated':
                return i
        for i in range(0, start_index + 1):
            if self.data[i].get('annotation_status', 'unannotated') == 'unannotated':
                return i
        return start_index

def main():
    args = parse_args()
    # 传入 target_pixels 参数
    dm = DataManager(args.input_json, args.output_json, args.image_root, args.target_pixels)

    # 虽然现在直接读取图片数据，Gradio 仍可能需要权限检查，保留此逻辑无害
    allowed_dirs = set()
    allowed_dirs.add(os.path.abspath(os.getcwd()))
    if args.image_root:
        allowed_dirs.add(os.path.abspath(args.image_root))
    
    # ... (省略中间的 allowed_dirs 扫描逻辑，与之前一致，此处为了简洁省略，实际使用请保留原逻辑) ...
    # 为了保证代码完整可运行，这里还是写上扫描逻辑
    for item in dm.data:
        paths_to_check = []
        if 'real_image_path' in item:
            paths_to_check.append(item['real_image_path'])
        if 'dpo_pair' in item and item['dpo_pair']:
            paths_to_check.append(item['dpo_pair'].get('chosen'))
            paths_to_check.append(item['dpo_pair'].get('rejected'))
        for p in paths_to_check:
            if p:
                full_path = dm.get_image_path(p)
                abs_dir = os.path.dirname(os.path.abspath(full_path))
                allowed_dirs.add(abs_dir)
    allowed_paths_list = list(allowed_dirs)


    with gr.Blocks(title="DPO Annotator", theme=gr.themes.Soft()) as demo:
        current_index = gr.State(0)
        
        with gr.Row():
            gr.Markdown("## 🧬 NewbieLoraTrainer - DPO Pair Annotator")
            save_btn = gr.Button("💾 保存进度 (Save JSON)", variant="secondary", size="sm")
            save_msg = gr.Textbox(label="", show_label=False, container=False, interactive=False)
 
        with gr.Row():
            progress_md = gr.Markdown("Loading...")
        
        with gr.Row():
            prompt_box = gr.Textbox(label="Prompt / Caption", interactive=False, lines=3)
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 🖼️ Real Image (GT)")
                # --- 变更点：去掉 type="filepath"，让其接受 numpy array ---
                real_img = gr.Image(label="Real Reference", interactive=False)
            
            with gr.Column():
                gr.Markdown("### ✅ Chosen (Prefer)")
                chosen_img = gr.Image(label="Chosen", interactive=False)
                chosen_score_box = gr.Textbox(label="Chosen Scores", lines=4)
            
            with gr.Column():
                gr.Markdown("### ❌ Rejected (Dislike)")
                rejected_img = gr.Image(label="Rejected", interactive=False)
                rejected_score_box = gr.Textbox(label="Rejected Scores", lines=4)
        
        with gr.Row():
            status_box = gr.Textbox(label="Current Status", interactive=False)
        
        gr.Markdown("### 🕹️ Actions")
        with gr.Row():
            btn_swap = gr.Button("🔄 Swap", variant="stop", scale=1) 
            btn_confirm = gr.Button("✅ Confirm", variant="primary", scale=1) 
            btn_discard = gr.Button("🗑️ Discard", variant="secondary", scale=1) 

        gr.Markdown("### 🧭 Navigation")
        with gr.Row():
            btn_prev = gr.Button("⬅️ Prev", size="sm")
            btn_jump = gr.Button("⏩ Jump to Unannotated", size="sm")
            btn_next = gr.Button("Next ➡️", size="sm")
            
        
        def refresh_ui(index):
            item = dm.get_item_data(index)
            if not item:
                return [
                    index, 
                    f"Index: {index} (Out of bounds)", 
                    "", None, None, None, "", "", "Error"
                ]
            
            progress_str = f"**Sample {index + 1} / {len(dm.data)}**"
            status_str = f"Status: {item['status'].upper()}"
            
            return [
                index,
                progress_str,
                item['caption'],
                item['real_img'],     # 这里传入的是 numpy array
                item['chosen_img'],   # 这里传入的是 numpy array
                item['rejected_img'], # 这里传入的是 numpy array
                item['chosen_score'],
                item['rejected_score'],
                status_str
            ]

        # ... 事件绑定逻辑保持不变 ...
        def on_confirm(idx):
            dm.mark_status(idx, "confirmed")
            next_idx = min(idx + 1, len(dm.data) - 1)
            return refresh_ui(next_idx)

        def on_swap(idx):
            dm.mark_status(idx, "swapped", swap=True)
            next_idx = min(idx + 1, len(dm.data) - 1)
            return refresh_ui(next_idx)

        def on_discard(idx):
            dm.mark_status(idx, "discarded")
            next_idx = min(idx + 1, len(dm.data) - 1)
            return refresh_ui(next_idx)

        def on_prev(idx):
            new_idx = max(0, idx - 1)
            return refresh_ui(new_idx)

        def on_next(idx):
            new_idx = min(len(dm.data) - 1, idx + 1)
            return refresh_ui(new_idx)

        def on_jump(idx):
            new_idx = dm.find_next_unannotated(idx)
            return refresh_ui(new_idx)

        def on_save():
            return dm.save_data()

        ui_outputs = [
            current_index, progress_md, prompt_box, 
            real_img, chosen_img, rejected_img, 
            chosen_score_box, rejected_score_box, status_box
        ]

        demo.load(refresh_ui, inputs=[current_index], outputs=ui_outputs)
        btn_prev.click(on_prev, inputs=[current_index], outputs=ui_outputs)
        btn_next.click(on_next, inputs=[current_index], outputs=ui_outputs)
        btn_jump.click(on_jump, inputs=[current_index], outputs=ui_outputs)
        btn_confirm.click(on_confirm, inputs=[current_index], outputs=ui_outputs)
        btn_swap.click(on_swap, inputs=[current_index], outputs=ui_outputs)
        btn_discard.click(on_discard, inputs=[current_index], outputs=ui_outputs)
        save_btn.click(on_save, outputs=[save_msg])

    demo.launch(
        server_name="0.0.0.0", 
        server_port=args.port, 
        share=args.share,
        allowed_paths=allowed_paths_list 
    )

if __name__ == "__main__":
    main()
