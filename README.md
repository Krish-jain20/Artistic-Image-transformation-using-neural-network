# Artistic-Image-transformation-using-neural-network
This project demonstrates the implementation of Neural Style Transfer, that builds a new image by merging the content of one image with the artistic style of another. Using a pretrained convolutional neural network, the algorithm extracts multi-level content and style features and optimizes a generated image to minimize style and content loss. 
# Neural Style Transfer using Deep Learning

This project implements **Neural Style Transfer (NST)** using a pretrained Convolutional Neural Network. The goal is to combine the **content** of one image with the **artistic style** of another image to generate a new, visually unique output.  
The implementation is inspired by the original work of *Leon Gatys et al.* and adapted from the open-source repository by Cédric Caruzzo.

---

## ✨ Features

- Extracts **content** and **style** representations using a pretrained CNN (VGG-based).
- Computes:
  - **Content Loss**
  - **Style Loss** (using Gram Matrices)
  - **Total Loss** (weighted combination)
- Performs **gradient-based optimization** to update a generated image.
- Supports **three modes**:
  - 🎨 *Style Transfer* (main output)
  - 🧩 *Content Reconstruction*
  - 🌈 *Style Visualization*
- Easily customizable with different content and style images.

---
---

## 🚀 How It Works

1. Load the **content** and **style** images.  
2. Pass them through a pretrained CNN to extract:
   - High-level content features  
   - Low-level style features (Gram matrices)
3. Define **Content Loss**, **Style Loss**, and **Total Loss**.
4. Initialize a **generated image** (content image or random noise).
5. Use **gradient descent** to iteratively update the image.
6. Save and display the final stylized output.

---
