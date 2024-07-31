# Random Forest Classifier trên tập dữ liệu MNIST với Numpy
## 1. Random Forest
Một mô hình thuộc họ Tree và có một số cải thiện nhất định. Một ví dụ minh họa, Decision Tree đóng vai trò như một chuyên gia mỹ phẩm thì Random Forest giống như một người làm việc trong lĩnh vực đó. Random Forest sẽ cho chúng ta được nhiều góc nhìn hơn, tổng quát hơn → Từ đó đưa ra quyết định có khả năng chính xác cao hơn.

## 2. Cấu trúc của Random Forest
<p align="center">
  <img src="https://github.com/nvsthinh/Interview_QAI/blob/main/data/Q1.png" width="400"/>
</p>

Radom Forest bao gồm:
- $N$ Decision Tree được xây dựng bởi Bootrapped Dataset theo phương pháp Predifined Conditions
- Từ $N$ tree sẽ cho ra $N$ kết quả dự đoán
- $N$ kết quả dự đoán qua một phương pháp Ensemble Bagging sẽ cho ra kết quả cuối cùng

## 3. Cách thức hoạt động
- **Bước 1**: Từ dữ liệu ban đầu, sẽ tạo ra một **Bootrapped Dataset** dựa trên đó.
    - **Bootrapped Dataset** là bộ dữ liệu được lựa chọn ngẫu nhiên trên bộ dữ liệu gốc. Cách thức hoạt động : gắn cho mỗi samples một xác suất bằng nhau, từ đó lựa chọn ngẫu nhiên dữ liệu cho đến khi dữ liệu sinh ra có kích thước bằng dữ liệu ban đầu (không quan trọng trùng hay không).
- **Bước 2**: Tạo Decision Tree bằng phương pháp **Predifned Conditions** trên bộ dữ liệu **Boostrapped**. Phương pháp được thực hiện:
    - **Bước 2-1**: Chọn ngẫu nhiên 2 features trong bộ dữ liệu
    - **Bước 2-2**: Xây dựng Decision Node dựa trên việc so sánh 2 features đó, features nào được chọn thì loại khỏi bộ dữ liệu
    - **Bước 2-3**: Lặp lại **Bước 2-1 → 2-2** để tìm ra Decision Tree tối ưu
- **Bước 3**: Lặp lại **Bước 1 → 2** để tìm ra $N$  Decision Tree
- **Bước 4**: Khi đưa dữ liệu cần dự đoán vào thì $N$ Decision Tree sẽ trả về $N$ kết quả. Từ đó, bằng phương pháp **Ensemble** đơn giản là **Majority Voting** để đưa ra kết quả cuối cùng


*Giải thích chi tiết về mô hình Random Forest: [Bài 5 - Random Forest [Thịnh Diablog]](https://flowery-fairy-f0d.notion.site/B-i-5-Random-Forest-d39ed94c6c1240c0b87f1708e5358f12?pvs=4)*

# 4. Files
- `main.py`: Chứa một pipeline đơn giản sử dụng Random Forest Classifier trên tập dữ liệu MNIST.
- `model.py`: Triển khai Random Forest scratch sử dụng Numpy.
- `utils.py`: Load MNIST dataset
- `notebook/Random_Forest_with_MNIST_Dataset.ipynb`: Minh họa một ví dụ triển khai Random Forest Classifier trên tập dữ liệu MNIST cho Question 1.