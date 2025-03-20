## Bài toán FER 

**mục tiêu của nhóm là :**
0. Điều quan trọng nhất khi tiếp cận với một bài toán mới là dữ liệu, và bài toán nhận diện cảm xúc là một trong những bài toán có thiên hướng bias cao do một khuôn mặt nhưng người A nhìn ra đang cười còn người B có thể nhìn ra bình thường nên việc số lượng class của dataset cũng là một vấn đề nên tìm hiểu. 
Tìm hiểu xem các bài báo 3 năm đổ lại tiếp cận theo cách gì
Và bao nhiêu data để train và đánh giá là đủ để thuyết phục người đọc báo cáo của các em?

1. Data preprocessing: liệu 1 ảnh khuôn mặt bình thường khi cho vào train có tốt bằng ảnh sau khi detect rồi cho vào training không?

2. Model: CNN hay Transformer hay là kết hợp cả hai? Có nên thêm vào các task khác ví dụ như thêm cả dự đoán tuổi để mô hình có nhiều dữ liệu hơn để học ko?

3. Training: Có nên sử dụng AMP, EMA không? Chia train test như nào nếu là dataset tự tạo? Có thể so sánh với các kết quả trong paper không? 

4. Đánh giá: Metrics các paper gần đây nhất sử dụng là gì? Có git nào có code sẵn phần này không? 
