# Multimodal Image Captioning-CNN

## Overview
This project implements a **multimodal image captioning system** using **PyTorch**, combining computer vision and natural language processing. The system generates textual descriptions (captions) for images by extracting visual features with a CNN encoder and generating captions with an LSTM or Transformer decoder.

This project demonstrates the application of **deep learning for vision-language tasks**, making it suitable for research-oriented MSc applications in **Machine Learning, Computer Vision, and Data Science**.

---

## Dataset
- **Flickr8k Dataset**: 8,000 images, each with 5 captions (≈40,000 captions total).  
- **Size:** ~1 GB.  
- **Source:** [Flickr8k on Kaggle](https://www.kaggle.com/datasets/adityajn105/flickr8k)

---

## Model Architecture
1. **Encoder (CNN):**  
   - Pretrained CNN (ResNet50, InceptionV3, or EfficientNet) extracts feature embeddings from images.  
   - Image embeddings are saved to speed up training.

2. **Decoder (Text Generator):**  
   - LSTM-based decoder generates captions word by word.  
   - Optional Transformer decoder can replace LSTM for research-level experimentation.  

3. **Training:**  
   - Loss function: CrossEntropyLoss  
   - Optimizer: Adam  
   - Batch size: 32  
   - Training: Colab GPU (20 epochs recommended)  

---

## Evaluation
- **Metrics:**  
  - BLEU score (1-4)  
  - METEOR  
  - CIDEr (optional)

- **Visualization:** Sample images with predicted captions shown alongside ground truth.

---

## Demo / Deployment
- **Streamlit app:** Users can upload an image and receive a generated caption.  
- **Notebook demo:** Run inference on sample images from the dataset.

---

## Future Work
- Extend decoder to **Transformer-based model** for improved performance.  
- Use **Flickr30k or MS-COCO** for larger-scale experiments.  
- Apply **Vision Transformer (ViT) encoder** for state-of-the-art image representations.

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
streamlit
tqdm
