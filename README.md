# Newbie Trainer

`newbie trainer` is a training toolkit designed specifically for the **Newbie AI** ecosystem.  
It supports parameter-efficient fine-tuning of Newbie base models and currently provides two training modes: **LoRA** and **LoKr**. The goal is to balance output quality with lower VRAM and compute requirements, so you can quickly get started on both local machines and servers.

---

## Project Overview

The goal of this trainer is to provide Newbie AI users with a solution that is:

- **Easy to use**: Complete training workflows via configuration files and simple command-line interfaces.
- **Highly adapted**: Customized and optimized for Newbie model structures and characteristics.
- **Extensible**: Friendly to secondary development and easy to integrate into your own pipelines (e.g., ComfyUI workflows, batch generation scripts, etc.).

If you are already using Newbie inference models, this trainer will help you quickly fine-tune styles, characters, and artistic directions to build your own personalized models.

---

##  Fork Features

1. 使用[EQ-VAE](https://huggingface.co/Anzhc/MS-LC-EQ-D-VR_VAE)替换原来的Flux VAE，且修改VAE nn.conv2d的padding_mode为"reflect" (避免图像边缘问题) 。
2. 实现原始Lumina Image 2.0的多分辨率函数（原分辨率loss -> 原分辨率loss + 4倍下采样loss），可能有助于图像全局结构的维持但增加了训练成本。
3. 修复`do_shift`的潜在问题，保证时间步t能根据图像分辨率正确shift，而不是根据固定的`resolution`进行shift。
4. 删除LYCORIS LoKr微调功能。取而代之的是，全面使用PEFT库进行LoRA微调，并添加开启`DoRA`的参数。
5. 更改了use_cache的行为，默认只缓存图像latent而不缓存text embedding。
6. 增加了dropout_caption_rate(无条件生成训练，有利于CFG)和shuffle_caption等有用的caption处理功能。
7. 增加了gradient_accumulation功能，使得更大批次的训练成为可能。
8. 使用wandb替换tensorboard进行损失等日志记录。
9. 修改图像分箱策略，实现更精细的分箱，减少裁剪损失（依照sd-scripts）。
10. 修复了对latent的错误缩放策略。`latents = (latents - shift_factor) * scaling_factor`

---

## Environment & Setup

The following steps assume a Python environment. It is recommended to use **Python 3.10+** and an NVIDIA GPU with CUDA support on Linux or Windows.

### 1. Clone the Repository with Git

```bash
git clone https://github.com/NewBieAI-Lab/NewbieLoraTrainer.git
cd NewbieLoraTrainer
```

### 2. Use venv to Manage a Virtual Environment (Recommended)

Using `venv` isolates this project’s dependencies from your system Python environment and avoids conflicts between different projects.

```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate
```

### 3. Install PyTorch

Please visit the official PyTorch website and choose the correct installation command according to your **CUDA version** and operating system.  
For example (only an example, please adjust to your actual setup):

```bash
pip install torch torchvision
```

> Note: If you want GPU acceleration, make sure to install a PyTorch build with CUDA support.

### 4. Install Flash-Attention and Triton

To further improve training speed and VRAM efficiency, it is recommended to install **Flash-Attention** and **Triton**:

```bash
pip install flash-attn
pip install triton
```

Please refer to the tutorials linked below or the official documentation of each project if you run into compilation or CUDA-related issues during installation.

### 5. Install Project Dependencies

After activating the virtual environment and installing PyTorch, install the remaining dependencies required by this project with:

```bash
pip install -r requirements.txt
```

Once these steps are completed, your basic environment is ready and you can start LoRA / LoKr training following the tutorials.

---

## Training Tutorials

If this is your first time using the trainer, it is strongly recommended to read the following tutorial documents first.  
They explain data preparation, configuration files, command-line examples, and more.

- **Chinese Tutorial** (recommended for Chinese-speaking users):  
  https://www.notion.so/Newbie-AI-lora-2b84f7496d81803db524f5fc4a9c94b9?source=copy_link

- **English Tutorial** (for international / English-speaking users):  
  https://www.notion.so/Newbie-AI-lora-training-tutorial-English-2c2e4ae984ab8177b312e318827657e6?source=copy_link

The tutorials typically cover:

- Detailed environment and dependency explanations  
- How to prepare and tag your training dataset  
- Example configuration files and parameter descriptions  
- Common error patterns and troubleshooting tips  

---

## Using the Trained Results

### 1. Merging LoRA with `merge_lora.py`

After completing LoRA training, you can use the provided `merge_lora.py` script to merge the trained LoRA with a base model.  
This produces a standalone merged model that can be used directly in environments without native LoRA support.  
(LoKr merging is **not** supported yet.)

Example command (for illustration only):

```bash
python merge_lora.py   --base_model /path/to/base/model   --lora_path /path/to/trained_lora.safetensors   --output_path /path/to/merged_model
```

Please adjust paths and arguments in the script or command line according to your actual setup.

### 2. Loading LoRA Directly in ComfyUI

If you are using **ComfyUI**, you can load a trained LoRA directly through the **Newbie AI LoRA Loader node**, without merging the model beforehand.

Typical workflow:

1. Place the trained `.safetensors` LoRA file into ComfyUI’s `loras` directory (or your custom directory).
2. Add the Newbie AI LoRA Loader node in your ComfyUI workflow.
3. Select the corresponding LoRA file in the node.
4. Connect it to your Newbie base model inference pipeline and start image generation.

---

## Acknowledgements (Thanks)

The overall design and implementation of this trainer is heavily inspired by excellent open-source projects in the community, especially:

- [kohya-ss / sd-scripts](https://github.com/kohya-ss/sd-scripts)

The `newbie trainer` borrows ideas from this project in terms of training flow, parameter design, and parts of the code structure.  
We would like to express our sincere thanks to kohya-ss and all contributors to sd-scripts.

---

## License

This project is released under the **Apache License 2.0**. Under the terms of this license, you are allowed to:

- Freely use, modify, and distribute this project’s code.
- Integrate it into your personal or commercial projects.

For full details, please refer to the `LICENSE` file in this repository or the official Apache 2.0 license documentation.

If you have extended or modified this project, we kindly encourage you to credit the original source in your documentation and consider contributing improvements back to the community to help grow the Newbie ecosystem.
