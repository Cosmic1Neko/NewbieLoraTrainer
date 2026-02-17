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
    parser.add_argument("--target_pixels", type=int, default=1280*1280, help="图片显示时的目标像素总量(用于缩放加速)")
    return parser.parse_args()

class DataManager:
    def __init__(self, input_path, output_path, image_root, target_pixels=1280*1280):
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
        if not image_path or not os.path.exists(image_path):
            return None
        try:
            img_pil = Image.open(image_path).convert('RGB')
            orig_w, orig_h = img_pil.size
            scale_factor = math.sqrt(self.target_pixels / (orig_w * orig_h))
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
        real_path = self.get_image_path(item.get('real_image_path', ''))
        chosen_path = self.get_image_path(item['dpo_pair']['chosen'])
        rejected_path = self.get_image_path(item['dpo_pair']['rejected'])

        return {
            "index": index,
            "caption": item.get('caption', ""),
            "real_img": self._preprocess_image(real_path),
            "chosen_img": self._preprocess_image(chosen_path),
            "rejected_img": self._preprocess_image(rejected_path),
            "chosen_score": self.get_scores_text(item, item['dpo_pair']['chosen']),
            "rejected_score": self.get_scores_text(item, item['dpo_pair']['rejected']),
            "status": item.get('annotation_status', 'unannotated')
        }

    def mark_status(self, index, status, swap=False):
        if index < 0 or index >= len(self.data): return
        self.data[index]['annotation_status'] = status
        if swap and 'dpo_pair' in self.data[index]:
            pair = self.data[index]['dpo_pair']
            pair['chosen'], pair['rejected'] = pair['rejected'], pair['chosen']
            if 'margin' in pair: pair['margin'] = -pair['margin']

    def find_next_unannotated(self, start_index):
        for i in range(start_index + 1, len(self.data)):
            if self.data[i].get('annotation_status', 'unannotated') == 'unannotated': return i
        for i in range(0, start_index + 1):
            if self.data[i].get('annotation_status', 'unannotated') == 'unannotated': return i
        return start_index

# --- 自定义 CSS  ---
custom_css = """
#image-viewer-modal {
    display: none;
    position: fixed;
    z-index: 10000;
    left: 0;
    top: 0;
    width: 100%;
    height: 100%;
    background-color: rgba(0,0,0,0.9);
    flex-direction: column;
    justify-content: center;
    align-items: center;
}
#image-viewer-modal img {
    max-width: 95%;
    max-height: 95%;
    object-fit: contain;
    border: 2px solid #555;
}
.zoom-img {
    cursor: zoom-in;
}
.zoom-img:hover {
    filter: brightness(1.1);
}
"""

# --- 自定义 JavaScript ---
custom_js = """
function initZoomFeature() {
    // 检查是否已经初始化
    if (document.getElementById('image-viewer-modal')) return;

    // 创建 Modal 结构
    const modal = document.createElement('div');
    modal.id = 'image-viewer-modal';
    modal.innerHTML = '<img id="modal-img" src=""><p style="color:white; margin-top:10px; font-family:sans-serif;">点击任意位置退出预览</p>';
    modal.onclick = () => { modal.style.display = 'none'; };
    document.body.appendChild(modal);

    const modalImg = document.getElementById('modal-img');

    // 监听全局点击，捕获点击的图片
    document.addEventListener('click', (e) => {
        // 判断是否点击了 Gradio 图片容器中的 img 标签，且祖先节点有 .zoom-img 类
        if (e.target.tagName === 'IMG' && e.target.closest('.zoom-img')) {
            modalImg.src = e.target.src;
            modal.style.display = 'flex';
        }
    });
    console.log("Zoom feature initialized.");
}

// 在 Gradio 页面准备好后执行
document.addEventListener("DOMContentLoaded", initZoomFeature);
// 同时也通过定时器尝试，防止某些异步加载情况
setTimeout(initZoomFeature, 2000);
"""

def main():
    args = parse_args()
    dm = DataManager(args.input_json, args.output_json, args.image_root, args.target_pixels)

    # 扫描权限目录
    allowed_dirs = {os.path.abspath(os.getcwd())}
    if args.image_root: allowed_dirs.add(os.path.abspath(args.image_root))
    for item in dm.data:
        for p in [item.get('real_image_path'), item.get('dpo_pair', {}).get('chosen'), item.get('dpo_pair', {}).get('rejected')]:
            if p: allowed_dirs.add(os.path.dirname(os.path.abspath(dm.get_image_path(p))))

    with gr.Blocks(title="DPO Annotator") as demo:
        current_index = gr.State(0)
        
        with gr.Row():
            gr.Markdown("## 🧬 DPO Pair Annotator")
            save_btn = gr.Button("💾 保存 (Save)", variant="secondary", size="sm")
            save_msg = gr.Textbox(label="", show_label=False, container=False, interactive=False)

        with gr.Row():
            progress_md = gr.Markdown("Loading...")
        
        with gr.Row():
            prompt_box = gr.Textbox(label="Prompt / Caption", interactive=False, lines=2)
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 🖼️ Real Image")
                real_img = gr.Image(label="Reference", interactive=False, elem_classes="zoom-img")
            
            with gr.Column():
                gr.Markdown("### ✅ Chosen")
                chosen_img = gr.Image(label="Chosen", interactive=False, elem_classes="zoom-img")
                chosen_score_box = gr.Textbox(label="Scores", lines=3)
            
            with gr.Column():
                gr.Markdown("### ❌ Rejected")
                rejected_img = gr.Image(label="Rejected", interactive=False, elem_classes="zoom-img")
                rejected_score_box = gr.Textbox(label="Scores", lines=3)
        
        with gr.Row():
            status_box = gr.Textbox(label="Current Status", interactive=False)
        
        gr.Markdown("### 🕹️ Actions")
        with gr.Row():
            btn_swap = gr.Button("🔄 Swap", variant="stop") 
            btn_confirm = gr.Button("✅ Confirm", variant="primary") 
            btn_discard = gr.Button("🗑️ Discard", variant="secondary") 

        gr.Markdown("### 🧭 Navigation")
        with gr.Row():
            btn_prev = gr.Button("⬅️ Prev")
            btn_jump = gr.Button("⏩ Jump to Unannotated")
            btn_next = gr.Button("Next ➡️")
            
        def refresh_ui(index):
            item = dm.get_item_data(index)
            if not item: return [index, "Out of bounds", "", None, None, None, "", "", "Error"]
            
            progress_str = f"**Sample {index + 1} / {len(dm.data)}**"
            status_str = f"Status: {item['status'].upper()}"
            
            return [
                index, progress_str, item['caption'],
                item['real_img'], item['chosen_img'], item['rejected_img'],
                item['chosen_score'], item['rejected_score'], status_str
            ]

        def on_confirm(idx):
            dm.mark_status(idx, "confirmed")
            return refresh_ui(min(idx + 1, len(dm.data) - 1))

        def on_swap(idx):
            dm.mark_status(idx, "swapped", swap=True)
            return refresh_ui(min(idx + 1, len(dm.data) - 1))

        def on_discard(idx):
            dm.mark_status(idx, "discarded")
            return refresh_ui(min(idx + 1, len(dm.data) - 1))

        ui_outputs = [
            current_index, progress_md, prompt_box, 
            real_img, chosen_img, rejected_img, 
            chosen_score_box, rejected_score_box, status_box
        ]

        # 核心逻辑
        demo.load(refresh_ui, inputs=[current_index], outputs=ui_outputs)
        
        btn_prev.click(lambda i: refresh_ui(max(0, i - 1)), inputs=[current_index], outputs=ui_outputs)
        btn_next.click(lambda i: refresh_ui(min(len(dm.data)-1, i + 1)), inputs=[current_index], outputs=ui_outputs)
        btn_jump.click(lambda i: refresh_ui(dm.find_next_unannotated(i)), inputs=[current_index], outputs=ui_outputs)
        btn_confirm.click(on_confirm, inputs=[current_index], outputs=ui_outputs)
        btn_swap.click(on_swap, inputs=[current_index], outputs=ui_outputs)
        btn_discard.click(on_discard, inputs=[current_index], outputs=ui_outputs)
        save_btn.click(dm.save_data, outputs=[save_msg])

    demo.launch(
        server_name="0.0.0.0", 
        server_port=args.port, 
        share=args.share, 
        allowed_paths=list(allowed_dirs),
        theme=gr.themes.Soft(),
        css=custom_css,
        head=f"<script>{custom_js}</script>"
    )

if __name__ == "__main__":
    main()
