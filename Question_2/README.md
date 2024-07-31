# Triplet Loss
## 1. Định nghĩa
Triplet Loss là một hàm mất mát thường được sử dụng trong các vấn đề Image Regconition và Matching Problems, đặc biệt là trong các mô hình Deep LEarning. Mục tiêu của Triplet Loss là đảm bảo rằng các mẫu thuộc cùng một lớp gần nhau hơn trong không gian đặc trưng so với các mẫu thuộc các lớp khác nhau.
## 2. Công thức Triplet Loss với One Samples (Question 2 - a)
Triplet Loss được biểu diễn:

$$\mathcal{L} = \frac{1}{N} \sum_{i=1}^{N} \max\left(0, \|f(x_i^a) - f(x_i^p)\|^2 - \|f(x_i^a) - f(x_i^n)\|^2 + \alpha\right)$$

Trong đó
- $x_i^a$ là anchor.
- $x_i^p$ là positive sample (cùng class với anchor).
- $x_i^n$ là negative sample (khác class với anchor).
- $α$ là margin.
- $N$ là số lượng triplets được tính.

## 3. Công thức Triplet Loss với Multiple Samples (Question 2 - b)
$$\mathcal{L} = \frac{1}{A} \sum_{i=1}^{N} \max\left(0, \frac{1}{P} \sum_{p\in P}\|f(x_i^a) - f(x_i^p)\|^2 - \frac{1}{N}\sum_{n\in N} \|f(x_i^a) - f(x_i^n)\|^2 + \alpha\right)$$
Where
- $x_i^a$ là anchor.
- $x_i^p$ là positive sample.
- $x_i^n$ là negative sample.
- $α$ là margin.
- $P$ là số lượng của positive samples.
- $N$ là số lượng của negative samples.
- $A$ là số lượng triplets được tính.
## 4. Giải thích công thức
### 4.1. Hàm Embedding $f$
- $f(x)$: Một hàm (thường là mạng nơ-ron) ánh xạ một đầu vào $x$ đến một không gian embedding nơi có thể thực hiện các so sánh.
### 4.2. Distance Metric
- $\|f(x_i^a) - f(x_i^p)\|^2$: Khoảng cách bình phương giữa anchor và positive sample trong không gian embedding. (L2 Distance)

- $\|f(x_i^a) - f(x_i^n)\|^2$: Khoảng cách bình phương giữa anchor và negative sample trong không gian embedding. (L2 Distance)
### 4.3. Margin $\alpha $
- Margin được áp dụng giữa các cặp positive và negative. Margin giúp đảm bảo negative samples cách xa the anchor hơn positive examples ít nhất $\alpha$.
### 4.4. Loss Calculation
- Nếu khoảng cách của cặp mẫu dương không đủ nhỏ hơn khoảng cách của cặp mẫu âm ít nhất là $a$, thì biểu thức $\|f(x_i^a) - f(x_i^p)\|^2 - \|f(x_i^a) - f(x_i^n)\|^2 + \alpha$ sẽ dương, đóng góp vào mất mát.
- Hàm $\max$ đảm bảo rằng chỉ có các giá trị dương mới góp phần vào mất mát. Nếu sự khác biệt là âm (tức là bộ ba đã thỏa mãn điều kiện), mất mát cho bộ ba đó là bằng không.
## 5. Ưu, nhược điểm
### 5.1. Ưu điểm
- Discriminative Feature Learning
- Robust to Class Imbalance
- Good for High-Dimensional Data

### 5.2. Nhược điểm
- Margin Sensitivity
- Selection of Triplets
- Computational Cost
## 6. Ứng dụng
- **Face Recognition**: ảm bảo rằng các khuôn mặt của cùng một người gần nhau hơn trong không gian embedding so với các khuôn mặt của những người khác nhau.
- **Image Retrieval**: Giúp học các nhúng sao cho các hình ảnh tương tự gần nhau hơn trong không gian nhúng.

## 7. Files
- `triplet_loss_one_sample.py`:  Chứa triển khai Triplet Loss với 1 mẫu neo và 1 mẫu giả
- `triplet_loss_multi_samples.py`: Chứa triển khai Triplet Loss với 2 mẫu neo và 5 mẫu giả
- `notebook/Triplet_Loss.ipynb`: Chứa ví dụ triển khai với một mẫu và nhiều mẫu
