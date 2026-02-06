# **NewbieLoraTrainer**

A lightweight and efficient LoRA (Low-Rank Adaptation) trainer designed for modern diffusion models. This repository focuses on both standard Supervised Fine-Tuning (SFT) and advanced RLHF-style alignment using Direct Preference Optimization (DPO).

## **🚀 Key Features**

Compared to standard or "original" LoRA trainers, this repository introduces several advanced capabilities:

* **VAE Enhancement**: Replaced the original Flux.1 VAE with EQ-VAE and modified the `padding_mode` of `nn.conv2d` to "reflect" to mitigate edge artifacts in generated images.
* **Multi-Resolution Loss**: Implemented the multi-resolution objective from Lumina Image 2.0 (Original Loss + 4x Downsampled Loss). This helps maintain global image structure during training, though it increases computational overhead.
* **Dynamic Time-Step Shifting**: Fixed potential issues with `do_shift`, ensuring that the timestep $t$ is correctly shifted based on the actual image resolution rather than a fixed global resolution.
* **LoRA-Focused Training**: Removed LYCORIS LoKr support in favor of a full migration to the PEFT library for LoRA fine-tuning. Added parameters to enable DoRA (Weight-Decomposed Low-Rank Adaptation).
* **Optimized Caching Strategy**: Modified the `use_cache` behavior to default to caching image latents only, excluding text embeddings to maintain flexibility.
* **Advanced Caption Processing**: Added useful caption features including `dropout_caption_rate` (for unconditional generation training to improve Classifier-Free Guidance) and `shuffle_caption`.
* **Gradient Accumulation**: Integrated gradient accumulation functionality to enable training with effectively larger batch sizes on consumer hardware.
* **Improved Logging**: Replaced TensorBoard with Weights & Biases (wandb) for more comprehensive experiment tracking and loss visualization.
* **Refined Resolution Bucketing**: Implemented a more granular bucketing strategy based on `sd-scripts` to minimize cropping loss and support diverse aspect ratios.
* **Latent Transformation Fix**: Corrected the latent scaling logic from `latents = latents * scaling_factor` to `latents = (latents - shift_factor) * scaling_factor`.
* **Direct Preference Optimization (DPO)**: Support for training LoRA using preference data (Win/Loss pairs), allowing models to align better with human aesthetics or specific requirements.  
* **Integrated DPO Annotator**: A built-in Gradio-based UI for quickly labeling preference pairs from generated images.  
* **Exponential Moving Average (EMA)**: Support for maintaining EMA weights `use_ema` during training. EMA helps in producing more stable models, reducing artifacts, and improving the overall aesthetic quality of generated images.

## **🛠️ Installation**

Example:
Python 3.10.8, CUDA 12.8
```
export HF_ENDPOINT=https://hf-mirror.com/
git clone https://github.com/cosmic1neko/NewbieLoraTrainer.git
cd NewbieLoraTrainer  
source venv/bin/activate

# Download base model
hf download NewBie-AI/NewBie-image-Exp0.1 --local-dir ./

# SFT
pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 --index-url https://download.pytorch.org/whl/cu128 

pip install https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.5.4/flash_attn-2.8.3+cu128torch2.9-cp310-cp310-linux_x86_64.whl
pip install -r requirements.txt

# DPO
pip install openmim
mim install "mmpose>=1.1.0"
pip install "numpy<2.0"
git clone https://github.com/tgxs002/HPSv2.git
cd HPSv2
pip install -e .
```

## **📖 Training Guide**

### **1\. Supervised Fine-Tuning (SFT)**

SFT is the standard way to train a LoRA on a set of images and captions.

**Preparation:**

* Organize your dataset: A folder containing images and corresponding .txt caption files.  
* Configure your training parameters in `lora.toml`.

**Execution:**

```
# multi_GPU
accelerate launch --multi_gpu --num_processes=2 --mixed_precision=bf16 train_newbie_lora.py --config_file lora.toml 2>&1 | tee /root/logs.log
# single_GPU
accelerate launch --mixed_precision=bf16 train_newbie_lora.py --config_file lora.toml 2>&1 | tee /root/logs.log
```

### **2\. Direct Preference Optimization (DPO)**

DPO is used to "fine-tune" a model's behavior based on preferences (e.g., choosing which image looks better).

#### **Step A: Offline dataset**

We need to use the SFT model as an image generator to generate an offline dataset, which will serve as the basis for constructing preference pairs later.

```
python offline_dataset.py \
  --config_file lora.toml \
  --lora_path /path_to_sft \
  --output_dir ./offline_dataset \
  --num_samples 2 \
  --steps 28 \
  --max_data_samples 2000 \
  --seed 114514
```

#### **Step B: Auto Annotation**

To train DPO, you need pairs of images where one is "preferred" (chosen) and the other is "rejected". Use `HPSv2.1` and `mmpose` to build preference pairing automatically:

```
python dataset_reward.py \
  --input_json ./offline_dataset/dataset.json \
  --w_anatomy 1 \
  --w_hps 20
```

#### **Step C: Manual Annotation**

Further manual annotation preferences. Use the provided Gradio tool:

```
python gradio_dpo_annotator.py --input_json ./offline_dataset/dataset_scored.json
```

* This UI allows you to compare two images side-by-side.  
* It saves a JSON file mapping prompts to "chosen" and "rejected" image paths.

#### **Step D: DPO Training**

Once you have your preference dataset, update `dpo_config.toml` to point to your annotation file.

**Execution:**

```
accelerate launch --multi_gpu --num_processes=2 --mixed_precision=bf16 train_lora_dpo.py --config_file dpo_config.toml 2>&1 | tee /root/logs.log
```

*DPO training calculates the implicit reward of the "chosen" image vs. the "rejected" image and pushes the LoRA weights to favor the preferred style.*

## **🔧 Utilities**

### **FLUX VAE Scale Calculation**

For users working with FLUX models, you can calculate the appropriate VAE scale factor:

```
python calc_flux_vae_scale.py --image_dir ./my_dataset
```

## **📂 Project Structure**

* train\_newbie\_lora.py: Entry point for standard SFT.  
* train\_lora\_dpo.py: Entry point for DPO.  
* transport/: Core logic for Flow Matching and DPM solvers.  
* models/: Model definition and LoRA injection logic.  
* dataset.py / dataset\_reward.py: Data loading pipelines.

## **Acknowledgments**

This project utilizes logic from transport for Flow Matching and is inspired by various modern diffusion training techniques.
