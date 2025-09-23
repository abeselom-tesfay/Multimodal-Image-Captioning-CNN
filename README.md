# Multimodal-Image-Captioning-CNN

## Overview
This project implements a **multimodal image captioning system** using **PyTorch**, combining computer vision and natural language processing. The system generates textual descriptions (captions) for images by extracting visual features with a **CNN encoder** and generating captions with an **LSTM decoder**.  

---

## Dataset
- **Flickr8k Dataset:** 8,000 images, each with 5 captions (≈40,000 captions total).  
- **Size:** ~1 GB  
- **Structure:**



---

## Model Architecture
1. **CNN Encoder:**  
   - Pretrained CNN (ResNet50 or EfficientNet) extracts image embeddings.  

2. **LSTM Decoder:**  
   - Generates captions word by word.  
   - Uses attention mechanism for better image-text alignment.  

3. **Advanced Features:**  
   - Beam search for better caption generation.  
   - Teacher forcing during training.  
   - BLEU score evaluation for captions.  

---

## Training
- **Framework:** PyTorch  
- **Loss:** CrossEntropyLoss  
- **Optimizer:** Adam  
- **Batch size:** 32  
- **Device:** GPU recommended (Colab or similar)  
- **Epochs:** 20+  

---

## Evaluation
- BLEU scores (1-4)  
- Sample predictions displayed alongside ground truth captions.  

---

## Future Work
- Replace LSTM with Transformer decoder for improved performance.  
- Use Vision Transformer (ViT) encoder for state-of-the-art image representation.  
- Scale to larger datasets (Flickr30k, MS-COCO) for better generalization.

---

## Requirements
```bash
torch
torchvision
torchtext
numpy
pandas
matplotlib
Pillow
tqdm
