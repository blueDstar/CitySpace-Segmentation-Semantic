Edge Detection Methods
📌 Overview

The Edge Detection Methods module demonstrates and compares several classical edge detection algorithms commonly used in computer vision and image processing.

Edge detection is a fundamental step in many vision tasks such as:

Feature extraction

Image segmentation

Object detection

Shape and boundary analysis

This folder provides individual Python implementations of multiple edge detection techniques, allowing users to study their behavior, strengths, and weaknesses on real images.

🧠 Edge Detection Algorithms Included
🔹 1. Canny Edge Detection

File: edge_detection_canny.py

A multi-stage edge detector

Uses Gaussian smoothing, gradient calculation, non-maximum suppression, and hysteresis thresholding

Produces thin and well-connected edges

Highly robust to noise

📌 Best suited for:
High-quality edge extraction in real-world images

🔹 2. Sobel Edge Detection

File: edge_detection_sobel.py

First-order gradient-based method

Uses Sobel kernels to compute horizontal and vertical gradients

Highlights strong intensity changes

📌 Best suited for:
Simple and fast edge detection

🔹 3. Prewitt Edge Detection

File: edge_detection_prewitt.py

Similar to Sobel but with simpler convolution masks

Less sensitive to noise compared to Sobel

📌 Best suited for:
Educational purposes and low-noise images

🔹 4. Laplacian Edge Detection

File: edge_detection_laplacian.py

Second-order derivative method

Detects edges based on zero-crossings

Very sensitive to noise

📌 Best suited for:
Highlighting fine details after smoothing

🔹 5. Roberts Cross Edge Detection

File: edge_detection_robert.py

One of the earliest edge detection methods

Uses 2×2 diagonal kernels

Extremely fast but sensitive to noise

📌 Best suited for:
Simple demonstrations and low-resolution images

🔹 6. Sobel + Canny Combination

File: edge_sobelxcanny.py

Combines Sobel gradient detection with Canny thresholding

Improves edge continuity and clarity