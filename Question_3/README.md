# MLP Classifier trên tập dữ liệu MNIST với Numpy sử dụng Triplet Loss

<p align="center">
  <img src="https://github.com/nvsthinh/Interview_QAI/blob/main/data/Q3.png" />
</p>

## 1. Workflow
- **Training**: Huấn luyện mô hình MLP để học cách trích xuất đặc trưng (Với Triplet Loss: những sample có cùng class sẽ được gom lại với nhau, và cách xa những sample khác class)
- **Inference**: 
    - Sau khi có mô hình được huấn luyện, từ Input (1, 784) qua mô hình MLP (Triplet Loss) sẽ cho ra được đặc trưng Output (1, 64)
    - Ứng với mỗi class, lấy ra 1 sample, rồi đưa qua mô hình MLP (Triplet Loss) để trích xuất đặc trưng. Sau đó, gộp lại thành một list $\to$  Anchor List
    - Dùng hàm Similarity để so sánh giữa Output và Anchor List, lấy ra Index có Similarity Score cao nhất làm predict class

## 2. MLP Classifier
- Mô hình phân loại MLP với Triplet Loss. Có 2 layers với hidden size là 128, output size là 64. Được huấn luyện 10 epochs để thử với batch size là 32. 

## 3. 
## 4. Files
- `model.py`: Triển khai MLP scratch sử dụng Numpy.
- `config.py`: Chứa một số biến learning rate, batch size, ...
- `main.py`: Chứa một pipeline đơn giản sử dụng MLP Classifier trên tập dữ liệu MNIST.
- `utils.py`: Chứa một số hàm cần thiết triplet_loss_one_sample, load_data.