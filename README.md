# Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Networks (ESPCN)

## Overview

This project implements an image super-resolution pipeline using a machine learning model based on **Super-Resolution Efficient Sub-Pixel Convolutional Neural Networks (ESPCN)**. The algorithm enhances the resolution of low-quality images by reconstructing high-quality versions with finer details and improved clarity. This is achieved by leveraging deep learning techniques and training on high-resolution image datasets.

## Features
- Upscales low-resolution images using a trained deep learning model.
- Compares results with traditional bicubic interpolation (OpenCV).
- Calculates metrics like Peak Signal-to-Noise Ratio (PSNR) for performance evaluation.
- Visualization of results with zoomed-in areas for detailed inspection.

## Algorithm Description

The **Super-Resolution Efficient Sub-Pixel Convolutional Neural Networks (ESPCN)** is a widely-used deep learning-based method for image super-resolution. Here’s a breakdown of the algorithm:

1. **Input Preprocessing**:
   - A low-resolution image is passed as input.
   - The input is normalized and prepared for the network.

2. **Convolutional Layers**:
   - The model consists of multiple convolutional layers.
   - These layers extract and learn hierarchical features such as edges, textures, and patterns from the low-resolution input.

3. **Non-Linearity**:
   - Non-linear activation functions (e.g., ReLU) are applied after convolution to capture complex mappings between low and high-resolution representations.

4. **Upscaling**:
   - The network predicts the high-resolution details and combines them to reconstruct the output image.

5. **Training**:
   - The model is trained on a dataset of paired low- and high-resolution images.
   - The **Mean Squared Error (MSE)** loss function is used to minimize the difference between predicted and ground truth high-resolution images.

6. **Comparison with Traditional Methods**:
   - For evaluation, the model's outputs are compared against bicubic interpolation (a traditional image resizing method) in terms of PSNR.



### Compare with OpenCV Resizing
The script automatically computes and compares PSNR between:
- Low-resolution images upscaled using bicubic interpolation (OpenCV).
- High-resolution images predicted by the trained SRCNN model.


## Dataset

The model was trained on the small [BSDS500 Dataset](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/resources.html), which contains high-quality images for super-resolution tasks.

### Preprocessing
- The training images were downsampled to generate low-resolution inputs.
- Corresponding high-resolution images served as the ground truth for training.

## Results
The EXPCN model consistently outperforms bicubic interpolation in terms of visual quality and PSNR.

## References

- [Paper: Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network](https://arxiv.org/pdf/1609.05158)
- [BSDS500 Dataset](http://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/BSR_bsds500.tgz)
- [TensorFlow Documentation](https://www.tensorflow.org/)
- [OpenCV Documentation](https://docs.opencv.org/)
