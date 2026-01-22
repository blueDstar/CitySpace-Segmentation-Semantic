Giới thiệu

Đây là repository nghiên cứu và triển khai phân đoạn ngữ nghĩa ảnh giao thông đô thị (Semantic Segmentation) trên bộ dữ liệu Cityscapes, sử dụng các phương pháp Thị giác máy tính và Machine Learning truyền thống, không sử dụng Deep Learning.

Mục tiêu của project là:

Hiểu rõ pipeline phân đoạn ảnh truyền thống

Nghiên cứu và so sánh các đặc trưng thủ công (hand-crafted features)

So sánh hiệu quả giữa Naive Bayes và Random Forest

Trực quan hóa và đánh giá kết quả bằng các chỉ số học máy tiêu chuẩn

Project phù hợp cho:

Học phần Computer Vision

Machine Learning

Image Processing

Nghiên cứu nền tảng trước khi tiếp cận Deep Learning

🧠 Các phương pháp sử dụng
🔹 Trích xuất đặc trưng (Feature Extraction)

Histogram of Oriented Gradients (HOG)

Dense SIFT

Multiscale Basic Features

Edge Detection:

Sobel

Canny

Roberts

Local Binary Pattern (LBP)

Gradient Magnitude

Đặc trưng màu RGB & HSV

Vị trí pixel (x, y)

Thresholding (ngưỡng ảnh)

🔹 Mô hình học máy

Gaussian Naive Bayes

Random Forest Classifier

Tối ưu siêu tham số:

GridSearchCV

RandomizedSearchCV

Kết quả đánh giá mô hình
🔹 Confusion Matrix

🔹 Biểu đồ các chỉ số đánh giá

🔹 Kết quả phân đoạn trên tập validation
--------------------------------------------------------------------------------------
📌 Introduction

This repository implements semantic segmentation on the Cityscapes dataset using traditional computer vision and machine learning techniques, without deep learning.

The project aims to:

Understand the classical semantic segmentation pipeline

Analyze the impact of hand-crafted features

Compare Naive Bayes and Random Forest

Evaluate segmentation results using standard ML metrics

This project is suitable for:

Computer Vision courses

Machine Learning fundamentals

Image Processing studies

🧠 Methods
🔹 Feature Extraction

HOG (Histogram of Oriented Gradients)

Dense SIFT

Multiscale Basic Features

Edge Detection (Sobel, Canny, Roberts)

LBP

Gradient magnitude

RGB & HSV color features

Pixel spatial coordinates (x, y)

Thresholding-based segmentation

🔹 Models

Gaussian Naive Bayes

Random Forest Classifier

Hyperparameter tuning with GridSearchCV & RandomizedSearchCV

📊 Results & Visualizations

All evaluation results, confusion matrices, learning curves, feature importance plots, edge detection outputs, and sample predictions are provided directly in this repository and visualized above.