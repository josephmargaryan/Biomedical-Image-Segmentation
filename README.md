## Biomedical Image Segmentation Repository

Welcome to the Biomedical Image Segmentation Repository! This repository contains all the necessary modules, classes, and functions to instantiate a project in image segmentation quickly. Whether you're a researcher, developer, or practitioner in the field of medical imaging, this repository provides essential tools for your segmentation tasks.

### Features

1. **UNet from Scratch:** Implement the classic UNet architecture from scratch, a widely used convolutional neural network for biomedical image segmentation.
<img width="637" alt="Skærmbillede 2024-06-03 kl  16 26 56" src="https://github.com/josephmargaryan/Biomedical-Image-Segmentation/assets/126695370/c18a8485-3ba2-45d1-8b10-1dfb626ebe6b">

2. **Vision Transformer (ViT) for Segmentation:** Build a Vision Transformer model specifically tailored for segmentation tasks, providing a novel approach to image segmentation using transformer architectures.
<img width="973" alt="TransformerSeg" src="https://github.com/josephmargaryan/Biomedical-Image-Segmentation/assets/126695370/8feeeff9-eea1-4132-80eb-edfe51e09bf5">


3. **Attention Map extraction:** A function that takes in a file path and returns thee averaged attention maps over each layer from the ViT
<img width="862" alt="AttentionMaps" src="https://github.com/josephmargaryan/Biomedical-Image-Segmentation/assets/126695370/5a2201e1-159a-4478-bbba-ad8180f19a91">

4. **ViT for Classification:** The Vision Transformer architecture is utilized for image classification tasks, offering a versatile solution for various image analysis applications.

5. **Attention UNet from Scratch:** Develop an Attention UNet model from scratch, combining the power of self-attention mechanisms with the UNet architecture for enhanced segmentation performance.

6. **Evaluation Class:** Quickly assess the performance of trained models using binary mean Dice and binary mean Intersection over Union metrics, ensuring rigorous evaluation of segmentation results.

7. **Training Loop Function:** Employ a robust training loop function incorporating early stopping and learning rate scheduling techniques, optimizing model training and convergence.

8. **Inference Function:** Utilize a handy inference function to process image files and generate segmentation predictions using trained models, facilitating rapid inference and deployment.

9. **Visualization Module:** Visualize images, their ground truths, and model predictions side by side using a data loader, enabling qualitative analysis of segmentation results.
