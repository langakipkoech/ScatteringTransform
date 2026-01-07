# Wavelet Scattering Transform

## Description
This repository contains an implementation of the Scattering Transform, a powerful tool for signal processing and machine learning. The Scattering Transform is designed to extract stable and informative features from signals, making it particularly useful for tasks such as classification, regression, and clustering.

In this implementation, we implement the Scattering Transform for 2D signals (images) using python. The implementation is domain specific and the goal is to enhance is depth estimation in challenging maritime environments. 

## Why Scattering Transform?
The Scattering Transform offers several advantages over traditional feature extraction methods:
- **Stability**: It is stable to small deformations and translations of the input signal.
- **Invariance**: It provides invariance to certain transformations, making it robust to variations in the input data.
- **Rich Representation**: It captures multi-scale and multi-frequency information, providing a rich representation of the input signal.
- **Theoretical Foundation**: It is grounded in solid mathematical principles, ensuring reliable performance.
- **Non-learned**: Unlike deep learning methods, the Scattering Transform does not require training, making it suitable for scenarios with limited data such as the maritime domain. 

## Similarities to CNNs
The Scattering Transform shares similarities with Convolutional Neural Networks (CNNs) in that both use convolutional operations to extract features from input data. However, there are key differences:
- **Fixed Filters**: The Scattering Transform uses predefined wavelet filters, while CNNs learn filters during training.
- **No Training Required**: The Scattering Transform does not require a training phase, making it computationally efficient for feature extraction.
- **Theoretical Guarantees**: The Scattering Transform has theoretical guarantees regarding stability and invariance, which are not always present in CNNs. 
- **Hierarchical Structure**: Both methods employ a hierarchical structure to capture features at multiple scales, but the Scattering Transform does so through a mathematically defined process.


## Scattering Transform Architecture
The Scattering Transform architecture consists of multiple layers of wavelet convolutions followed by non-linear modulus operations and averaging. The key components include:
1. **Wavelet Filters**: A set of predefined wavelet filters that capture different frequency components of the input signal.
2. **Convolutional Layers**: Each layer applies the wavelet filters to the input signal, followed by a non-linear modulus operation.
3. **Averaging**: After the non-linear operation, an averaging step is applied to obtain the scattering coefficients.
4. **Multiple Orders**: The architecture can be extended to multiple orders, capturing higher-order interactions in the signal.
5. **Output**: The final output is a set of scattering coefficients that represent the input signal in a stable and informative manner.

The diagram below illustrates the architecture of a Scattering Transform

![Scattering Transform Architecture](scattering_transform_architecture.png)