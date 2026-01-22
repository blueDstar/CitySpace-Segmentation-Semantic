# Traditional Image Segmentation Methods

## 📌 Giới thiệu (Vietnamese)

Thư mục **traditional_image_segmentation** tổng hợp các phương pháp phân đoạn ảnh
(Image Segmentation) **truyền thống**, tập trung vào:

- Phân đoạn dựa trên ngưỡng (Thresholding)
- Phân đoạn theo màu sắc (HSV)
- Phân đoạn dựa trên đặc trưng thủ công (HOG, Multiscale Features)
- Phân đoạn pixel-level với Machine Learning
- Phân đoạn người thời gian thực bằng Mediapipe
- So sánh các phương pháp học máy không dùng Deep Learning

Project này phục vụ mục tiêu:
- Hiểu rõ pipeline phân đoạn ảnh truyền thống
- Phân tích ưu / nhược điểm của từng phương pháp
- Làm nền tảng trước khi học Semantic Segmentation bằng Deep Learning

---

## 🧠 Các phương pháp chính

### 🔹 Threshold-based Segmentation
- Áp dụng ngưỡng cường độ ảnh
- Phù hợp ảnh xám, bài toán đơn giản

### 🔹 Color-based Segmentation (HSV)
- Phân đoạn dựa trên không gian màu HSV
- Hiệu quả cho các vùng màu đặc trưng

### 🔹 Random Forest Segmentation
- Phân đoạn pixel-level
- Đặc trưng sử dụng:
  - RGB
  - Vị trí pixel (x, y)
  - Multiscale Basic Features
- Áp dụng cho ảnh đô thị (Cityscapes)

### 🔹 Multiscale Feature Segmentation
- Trích xuất đặc trưng đa tỉ lệ
- Kết hợp Random Forest để phân đoạn ảnh

### 🔹 Human Segmentation with Mediapipe
- Phân đoạn người từ ảnh hoặc webcam
- Xử lý real-time
- Không cần huấn luyện mô hình

---

## 📂 Danh sách file

| File | Mô tả |
|----|-----|
| `thresholding_image_segmentation.py` | Phân đoạn ảnh bằng threshold |
| `color_based_segmentation_hsv.py` | Phân đoạn theo màu HSV |
| `rgb_position_random_forest_segmentation.py` | RF segmentation với RGB + vị trí |
| `random_forest_pixel_segmentation.py` | Pixel-wise Random Forest |
| `random_forest_multiscale_segmentation.py` | RF + Multiscale Features |
| `multiscale_feature_rf_segmentation_inference.py` | Dự đoán segmentation bằng RF |
| `plot_trainable_segmentation.py` | Trực quan vùng huấn luyện |
| `image_human_segmentation_mediapipe.py` | Phân đoạn người từ ảnh |
| `real_time_human_segmentation_mediapipe.py` | Phân đoạn người realtime |
| `cityscapes_segmentation_gui.py` | Giao diện phân đoạn Cityscapes |

---

## 🎯 Mục tiêu học tập

- Hiểu rõ **segmentation không dùng deep learning**
- Làm quen với **pixel-level classification**
- Chuẩn bị nền tảng cho:
  - CNN Segmentation
  - U-Net
  - DeepLab
  - Transformer-based Segmentation

---

## 🌍 English Version

## 📌 Introduction

The **traditional_image_segmentation** folder contains implementations of
**classical image segmentation techniques**, focusing on:

- Threshold-based segmentation
- Color-based segmentation (HSV)
- Hand-crafted feature extraction
- Pixel-wise Machine Learning segmentation
- Human segmentation using Mediapipe
- Comparison between traditional ML methods

This project aims to:
- Understand traditional segmentation pipelines
- Analyze feature-based segmentation
- Build a foundation before Deep Learning segmentation

---

## 🧠 Implemented Methods

- Thresholding
- HSV color segmentation
- Random Forest pixel-wise segmentation
- Multiscale feature-based segmentation
- Real-time human segmentation with Mediapipe

---

## 🎓 Educational Purpose

This repository is designed for:
- Computer Vision students
- Machine Learning beginners
- Understanding segmentation without Deep Learning

---

## 📌 Technologies Used
- OpenCV
- scikit-image
- scikit-learn
- Mediapipe
- NumPy, Matplotlib

---

## 🧑‍💻 Author
**Nguyễn Văn Đạt**  
HUTECH University – Robotics & Artificial Intelligence

