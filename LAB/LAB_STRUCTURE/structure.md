## Cấu trúc được thực hiện gọn nhẹ cho thiết bị biên 

hướng dẫn : nhóm mình sẽ thực hiện thử nghiệm các kiến trúc dưới đây và thực hiện đánh giá trên từng kiến trúc rồi tổng hợp thành kiến trúc phù hợp cho bài toán của nhóm.


#### Các vấn đề còn vướng mắc hiện tại trên các thiết bị biên.
* Hạn chế về Tài nguyên Tính toán
* Độ trễ và Yêu cầu Thời gian Thực
* Tiêu thụ Năng lượng
* Độ Chính xác trong Điều kiện Thực tế
* Quyền riêng tư và Bảo mật
* Thiếu Dữ liệu Huấn luyện Đặc thù

**-> nói chung nhóm sẽ xây dựng các cấu trúc sao cho giải quyết các vấn đề này một cách ổn định nhất**
### Các phương pháp được đề xuất cho nhóm để giải quyết các vấn đề
1 Tối ưu hóa Mô hình Nhẹ (Model Compression)

Paper : [Light-FER: Hệ thống nhận dạng cảm xúc khuôn mặt nhẹ trên thiết bị Edge](https://www.mdpi.com/1424-8220/22/23/9524)

Để giải quyết vấn đề về tài nguyên tính toán nhóm mình sẽ sử dụng cá kĩ thuật như trong paper
gồm :
+ Sử dụng Xception(depthwise separable convolution) làm nền tảng, 
+ Sử dụng kiến trúc nhẹ
+ áp dụng cắt tỉa 
+ lượng tử hóa(int8 - fp16) để giảm kích thước mô hình.

2 Độ trễ và Yêu cầu Thời gian Thực

Paper : [Fast Video Facial Expression Recognition by Deeply Tensor-Compressed LSTM](https://arxiv.org/html/2501.06663v1#bib.bib44)

Để giải quyết vấn đề độ trễ và yêu cầu realtime thì nhóm mình sẽ thử nghiệm áp dụng phương pháp nén tensor LSTM


3 Tiêu thụ Năng lượng

Paper : [LITE-FER: A Lightweight Facial Expression Recognition Framework for Children in Resource-Limited Devices" (Bicer et al., 2024)](https://brosdocs.net/fg2024/W023.pdf)

Để giải quyết vấn đề tiêu thụ năng lượng thấp nhóm sẽ phải sử dụng các kiến trúc 
nhẹ để giảm tải số lượng phép tính mà mô hình phải tính toán 



**-> Vấn đề trước mắt nhóm sẽ giải quyết các Vấn đề trên**

### Kiến trúc sẽ phải thử nghiệm 

1. MobileNet (bao gồm MobileNetV1, V2, V3)
2. EfficientNet (B0-B7)
3. MobileFaceNet
4. EfficientFormer
5. ShuffleNet (V1, V2)
6. TinyML Models (TinyCNN, MicroNet)
7. SqueezeNet
8. Xception

### Tổng quan và So sánh
| **Kiến trúc**  | **Số tham số (triệu)** | **Tốc độ (FPS trên biên)** | **Độ chính xác** | **Ưu điểm chính**             |
|----------------|--|--|------------------|--------------------------------|
| MobileNet      |  |  |          | Nhẹ, nhanh, phổ biến          |
| EfficientNet   |  |  |              | Hiệu quả năng lượng          |
| MobileFaceNet  |  |  |      | Cực nhẹ, tối ưu cho di động   |
| EfficientFormer |  |  |           | Kết hợp CNN-Transformer       |
| ShuffleNetV2   |  |  |           | Nhanh, tiết kiệm tài nguyên   |
| TinyCNN        |  |  |            | Siêu nhẹ cho vi điều khiển    |
| SqueezeNet     |  |  |           | Nhỏ gọn, tiết kiệm bộ nhớ     |
| Xception (nén) |  |  |          | Cân bằng hiệu suất            |





