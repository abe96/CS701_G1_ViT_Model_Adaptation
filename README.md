# PEFT-ViT: Fine-Grained Visual Classification on unknown dataset

This repository presents an exploration of Parameter-Efficient Fine-Tuning (PEFT) techniques applied to fine-grained visual classification using a Vision Transformer (ViT) model on an unknown dataset.

The project implements and compares LoRA, a side Adapter, and hybrid approaches to optimize ViT performance, achieving a highest single-model accuracy of **74.44%** on the validation set.

Through rigorous ablation studies, the key finding was that an **Side Adapter-only** model combined with **unfreezing the last 4 backbone blocks** and a **gently regularized 3-phase training schedule** yielded the best results.

### Dataset and Constraints
Training was conducted under constraints including GPU time, VRAM limits (4 hours max on a 24 GB RTX 3090), and available data. The dataset consisted of:
- 450,000 training images (approximately 45 images per class)
- 50,000 validation images (around 5 images per class)
- An unknown test set with approximately 100,000 images

### Final Results
- **Best Single Model:** **74.44%** accuracy (`adapter` + 4 blocks unfrozen, 20 epochs, 0.2 augmentation)
- **Best LoRA Model:** **73.65%** accuracy (`lora` + 4 blocks unfrozen, 17 epochs)

For further details, please refer to the associated scripts and documentation.
