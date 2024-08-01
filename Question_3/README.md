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

## 3. Ưu & nhược điểm (Question 3 - b)
### 3.1. Ưu điểm
- **Ability to Capture Complex Patterns**: **MLP** có thể học và mô hình hóa các mối quan hệ phi tuyến tính phức tạp trong dữ liệu nhờ cấu trúc nhiều lớp của nó. Trong khi đó, **Random Forest** có thể gặp khó khăn với các mẫu phức tạp vì nó dựa vào tập hợp các cây quyết định, mà mỗi cây chỉ nắm bắt các mẫu đơn giản hơn.
- **Feature Learning**: **MLP** tự động học các feature từ raw, giảm thiểu nhu cầu về feature engineering. Trong khi đó, **Random Forest** cần phải có feature engineering và hoạt động không tốt nếu không có feature engineering phù hợp.
- **Scalability**: **MLP** có thể mở rộng với nhiều lớp và nơ-ron hơn để xử lý các tập dữ liệu lớn và phức tạp hơn, có thể cải thiện hiệu suất. Trong khi đó, **Random Forest** thêm nhiều cây có thể làm tăng yêu cầu về tính toán và bộ nhớ, nhưng không cải thiện khả năng nắm bắt các mẫu phức tạp.

### 3.2. Nhược điểm
- **Data Preprocessing**: **MLP** yêu cầu tiền xử lý dữ liệu để đảm bảo đào tạo ổn định và hiệu quả. Trong khi đó, có thể xử lý raw data hoặc chưa được chuẩn hóa hiệu quả hơn.
- **Risk of Overfitting**: **MLP** Dễ bị overfitting, đặc biệt trên các tập dữ liệu nhỏ hơn hoặc khi không sử dụng các kỹ thuật regularization (như dropout hoặc weight decay). Trong khi đó, **Random Forest** Ít bị overfitting hơn nhờ vào việc trung bình hóa nhiều cây quyết định.
## 4. Files
- `model.py`: Triển khai MLP scratch sử dụng Numpy.
- `config.py`: Chứa một số biến learning rate, batch size, ...
- `main.py`: Chứa một pipeline đơn giản sử dụng MLP Classifier trên tập dữ liệu MNIST.
- `utils.py`: Chứa một số hàm cần thiết triplet_loss_one_sample, load_data.
