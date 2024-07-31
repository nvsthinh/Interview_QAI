# API Phân Loại Hình Ảnh MNIST

Kho lưu trữ này cung cấp một API dựa trên Flask để phân loại các chữ số viết tay MNIST bằng cách sử dụng mô hình Multi-Layer Perceptron (MLP) được triển khai bằng Numpy.

## Tính Năng

- Tải lên một hình ảnh của chữ số viết tay.
- Dự đoán chữ số sử dụng mô hình MLP.
- Trả về nhãn dự đoán.

## Yêu Cầu

Đảm bảo rằng bạn đã cài đặt các gói sau:

- Flask
- Numpy
- Pillow

Bạn có thể cài đặt các gói cần thiết bằng pip:

```bash
pip install -m requirements.txt
```
## Cấu Trúc Dự Án

- `app.py`: Tệp ứng dụng Flask chính chứa các route API và logic.
- `checkpoint/model.pkl`: Mô hình MLP đã được huấn luyện và lưu bằng pickle.
- `README.md`: README file

## How to Run
Khởi chạy ứng dụng Flask bằng cách chạy:
```bash
python app.py
```
Ứng dụng sẽ có sẵn tại `http://localhost:5000`