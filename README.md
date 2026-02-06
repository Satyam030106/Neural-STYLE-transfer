# Neural Style Transfer Web Application

This repository contains a **Neural Style Transfer (NST) web application** built using **PyTorch** and **Streamlit**. The project implements **fast neural style transfer** using a Transformer-based architecture and allows users to apply artistic styles to images in real time through a simple web interface.

Two distinct artistic styles have been trained and deployed as separate models:
- **Starry Night** (inspired by Vincent van Gogh)
- **Pointillism** (inspired by Georges Seurat)

- **DEMO**- https://neural-style-transfer-v3iu7vhw9ua3owpp8uqycz.streamlit.app/

---

## Project Overview

Traditional neural style transfer methods are computationally expensive and slow during inference. This project adopts a **feed-forward TransformerNet architecture**, enabling fast and efficient style transfer after training.

The application allows users to:
1. Select a pre-trained artistic style  
2. Upload a content image  
3. Generate a stylized output image  
4. Download the processed result  

All style transformations are performed using **custom-trained models**, not prebuilt filters.

---

## Key Features

- Fast neural style transfer using a feed-forward Transformer network  
- Two independently trained artistic style models  
- Interactive web interface built with Streamlit  
- GPU acceleration support (if available)  
- Downloadable stylized output images  
- Clean and modular PyTorch implementation  

---

## Model Architecture

The core model is based on a **TransformerNet** architecture consisting of:
- Convolutional blocks with reflection padding  
- Instance normalization for style consistency  
- Multiple residual blocks for feature preservation  
- Upsampling layers for image reconstruction  

Each artistic style is trained as a **separate model**, enabling modular style selection during inference.

---

## Training Details

- **Framework:** PyTorch  
- **Feature Extractor:** Pre-trained VGG16 (used for loss computation only)  
- **Loss Functions:**
  - Content Loss  
  - Style Loss (Gram Matrix)  
  - Total Variation Loss  
- **Training Strategy:** Fast Neural Style Transfer (feed-forward network)  
- **Environment:** Google Colab (GPU)  

The final trained weights are saved and used directly for inference in the web app.

---

## Tech Stack

- Python  
- PyTorch  
- Torchvision  
- Streamlit  
- NumPy  
- Pillow  



