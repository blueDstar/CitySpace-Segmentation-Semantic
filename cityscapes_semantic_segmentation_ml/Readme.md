Cityscapes Semantic Segmentation using Machine Learning
📌 Giới thiệu

Cityscapes Semantic Segmentation using Machine Learning là project nghiên cứu và triển khai phân đoạn ngữ nghĩa ảnh giao thông đô thị (semantic segmentation) dựa trên các thuật toán Machine Learning truyền thống, không sử dụng Deep Learning.

Project sử dụng bộ dữ liệu Cityscapes và áp dụng các phương pháp:

Trích xuất đặc trưng thủ công (hand-crafted features)

Phân loại pixel-level

So sánh hiệu quả giữa Naive Bayes và Random Forest

Mục tiêu chính là:

Hiểu rõ pipeline phân đoạn ảnh truyền thống

Phân tích ảnh hưởng của từng loại đặc trưng

Đánh giá mô hình bằng các chỉ số học máy tiêu chuẩn

🧠 Các phương pháp sử dụng
🔹 Feature Extraction

HOG (Histogram of Oriented Gradients)

Dense SIFT

Canny Edge Detection

Local Binary Pattern (LBP)

Gradient magnitude (Sobel)

RGB & HSV color features

Pixel position (x, y)

🔹 Machine Learning Models

Gaussian Naive Bayes

Random Forest Classifier

GridSearchCV / RandomizedSearchCV để tối ưu siêu tham số

-------------------------------------------------------------------------------------
Cityscapes Semantic Segmentation using Machine Learning
📌 Introduction

Cityscapes Semantic Segmentation using Machine Learning is a research project that focuses on implementing semantic segmentation for urban street scene images using traditional Machine Learning algorithms, without relying on Deep Learning techniques.

The project is built on the Cityscapes dataset and applies the following approaches:

Hand-crafted feature extraction

Pixel-level classification

Performance comparison between Naive Bayes and Random Forest classifiers

Main objectives

To understand the complete pipeline of traditional image semantic segmentation

To analyze the impact of different feature extraction methods

To evaluate model performance using standard Machine Learning metrics

🧠 Methods Used
🔹 Feature Extraction

The project employs multiple hand-crafted features, including:

HOG (Histogram of Oriented Gradients)

Dense SIFT

Canny Edge Detection

Local Binary Pattern (LBP)

Gradient magnitude (Sobel operator)

RGB and HSV color features

Pixel spatial position (x, y)

These features are extracted at the pixel level to provide both local appearance and contextual information for classification.

🔹 Machine Learning Models

The following Machine Learning models are used and compared:

Gaussian Naive Bayes

Random Forest Classifier

Hyperparameter optimization is performed using:

GridSearchCV

RandomizedSearchCV

to improve model performance and generalization.