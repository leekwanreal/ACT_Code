# BẢN LẬP LUẬN KỸ THUẬT: LỰA CHỌN MÔ HÌNH ACT (ACTION CHUNKING WITH TRANSFORMERS) CHO BỘ ĐIỀU KHIỂN OPENARM 16-DOF

> **Tài liệu Kỹ thuật & Quyết định Kiến trúc (Architectural Decision Record - ADR)**  
> **Dự án:** VinRobotics - OpenArm Autonomous Manipulation Pipeline  
> **Hệ thống phần cứng:** 2 tay robot OpenArm (16 động cơ Damiao CAN-FD, 16-DOF) + 02 Camera Intel RealSense RGB-D (Chest + Front)  
> **Tác vụ cốt lõi:** Bimanual Pick-and-Place into Target Zone  
> **Phương pháp tiếp cận dữ liệu:** Tự thu thập dữ liệu trình diễn mẫu thông qua Teleoperation (Leader-Follower)  

---

## 1. TỔNG QUAN & BỐI CẢNH DỰ ÁN

Mục tiêu của dự án là xây dựng một pipeline hoàn chỉnh điều khiển hai cánh tay robot phối hợp khéo léo (*bimanual manipulation*, ví dụ: gắp vật phẩm, gấp khăn/vải tự động). Để hoàn thành toàn bộ khối lượng công việc trong vòng **6 tuần** — bao gồm cân chỉnh phần cứng, thu thập dữ liệu bằng tay (teleoperation), tiền xử lý, huấn luyện mô hình và triển khai điều khiển thời gian thực qua giao tiếp CAN bus — việc lựa chọn thuật toán/mô hình học máy mang tính sống còn đối với khả năng thành công của dự án.

Các họ thuật toán phổ biến nhất hiện nay cho robot manipulation bao gồm:
1. **Học tăng cường (Reinforcement Learning - RL)**: Sim-to-Real RL (PPO, SAC, Isaac Gym).
2. **Diffusion Policy (DP)**: Dựa trên mô hình sinh Denoising Diffusion Probabilistic Models.
3. **Behavioral Cloning truyền thống (Vanilla BC)**: MLP / LSTM / RNN.
4. **Action Chunking with Transformers (ACT)**: Mô hình sinh CVAE kết hợp Transformer Decoder và Temporal Ensembling (Stanford ALOHA).

Tài liệu này trình bày các căn cứ khoa học và thực tiễn kỹ thuật chứng minh **ACT là lựa chọn tối ưu và khả thi nhất** cho dự án.

---

## 2. PHÂN TÍCH SO SÁNH CÁC HƯỚNG TIẾP CẬN

### 2.1. Tại sao KHÔNG chọn Học tăng cường (Reinforcement Learning - RL)?

Mặc dù RL rất mạnh trong môi trường mô phỏng game hoặc điều khiển chân robot (quadruped/humanoid locomotion), việc áp dụng RL cho hệ hai tay thao tác khéo léo 16-DOF trong 6 tuần gần như **bất khả thi**:

1. **Hiệu suất mẫu cực thấp (Sample Inefficiency) & Nguy cơ phá hủy phần cứng:**
   * RL cần hàng triệu bước thử-sai (*trial-and-error*). Nếu train trực tiếp trên robot vật lý:
     * Tần số va chạm mất kiểm soát ở giai đoạn đầu (*exploration phase*) sẽ gây gãy bánh răng, quá nhiệt cuộn dây động cơ CAN-FD, đứt cáp hoặc va chạm nguy hiểm giữa hai tay.
     * Để đạt 1 triệu bước tương tác ở 50Hz cần liên tục hơn 5.5 giờ chạy robot không ngừng nghỉ (chưa tính thời gian reset môi trường bằng tay mỗi khi rơi đồ).
2. **Khoảng cách Sim-to-Real khổng lồ (Sim-to-Real Gap) & Vật thể biến dạng:**
   * Nếu train trong mô phỏng (Isaac Gym / MuJoCo):
     * Cần mô hình hóa chuẩn xác ma sát khớp, độ rơ nhông hộp số, độ trễ và jitter của bus CAN.
     * Với các tác vụ tiếp xúc phức tạp (đặc biệt là vật thể mềm như khăn/vải - deformable objects), việc mô phỏng động lực học vải chính xác là một đề tài nghiên cứu chuyên sâu, đòi hỏi tài nguyên tính toán lớn (GPU PhysX/FEM) và rất khó transfer về thế giới thực nếu không có Domain Randomization cực kỳ tinh vi.
   * Xây dựng và calibrate môi trường mô phỏng này sẽ tiêu tốn toàn bộ 6 tuần mà chưa chắc đã transfer thành công.
3. **Nút thắt thiết kế phần thưởng (Reward Engineering):**
   * Việc định nghĩa một hàm reward liên tục (*dense reward*) phản ánh sự khéo léo phối hợp 16 khớp là cực kỳ khó khăn. RL rất dễ bị "hack reward" (robot co giật tại chỗ hoặc tìm đường tắt dị thường) thay vì thực hiện động tác mượt mà như con người.

---

### 2.2. Tại sao chọn ACT thay vì Diffusion Policy (DP)?

Diffusion Policy (Chi et al., Columbia University - RSS 2023) là một đối thủ xuất sắc trong Imitation Learning. Tuy nhiên, khi xét trên các ràng buộc kỹ thuật của OpenArm, ACT chiếm ưu thế vượt trội:

| Tiêu chí | ACT (Stanford ALOHA) | Diffusion Policy (DP) | Tác động tới dự án OpenArm (6 tuần) |
| :--- | :--- | :--- | :--- |
| **Độ trễ suy luận (Inference Latency)** | **Cực thấp (~5 – 15 ms)**.<br>Chỉ cần **01 lần forward pass** qua Transformer Decoder. | **Đáng kể (~30 – 80 ms)**.<br>Phải lặp qua $T$ bước khử nhiễu (kể cả DDIM 8–16 steps). | **ACT thắng thế:** Bus CAN yêu cầu chu kỳ gửi lệnh đều đặn **50Hz (20ms/chu kỳ)**. ACT đáp ứng trực tiếp trên GPU máy tính xách tay/PC phổ thông mà không cần tối ưu C++/TensorRT phức tạp. |
| **Thời gian huấn luyện (Training Time)** | **Rất nhanh (~2 – 4 giờ)** cho 50–100 demo trên 1 GPU RTX 3090/4090. | **Kéo dài (8 – 16 giờ)** do phải học trường vector gradient trên không gian khuếch tán đa bước. | **ACT thắng thế:** Đội ngũ cần chu kỳ thử nghiệm nhanh: thu data ban ngày $\rightarrow$ train ban đêm $\rightarrow$ kiểm thử robot sáng hôm sau. ACT giúp tăng tốc độ lặp (iteration velocity). |
| **Độ nhạy siêu tham số (Hyperparameter Tuning)** | Ổn định, dễ hội tụ. Hàm loss chuẩn mực ($L_1 + \beta D_{KL}$). | Rất nhạy với schedule thêm nhiễu (beta schedule), số step khử nhiễu, clipping gradient. | **ACT thắng thế:** Giảm thiểu rủi ro debug mô hình trong giai đoạn nước rút. |
| **Tính trơn tru của cơ cấu chấp hành** | Tích hợp sẵn cơ chế **Temporal Ensembling** (trọng số giảm dần theo hàm mũ $e^{-m \cdot i}$). | Cần bộ lọc bên ngoài hoặc dựa vào Receding Horizon Control với tần số cao. | **ACT thắng thế:** Đảm bảo 16 động cơ không bị giật cục hay sốc mô-men xoắn, bảo vệ tuổi thọ cơ cấu cơ khí. |

---

### 2.3. Tại sao KHÔNG dùng Behavioral Cloning truyền thống (MLP / LSTM-based BC)?

Nhiều dự án ban đầu chọn MLP/LSTM vì nghĩ triển khai đơn giản, nhưng trên thực tế robot hai tay sẽ thất bại bởi hai nguyên nhân cố hữu:

1. **Vấn đề trôi sai số tích lũy (Covariate Shift / Compounding Error):**
   * Trong BC truyền thống, mô hình chỉ dự đoán hành động đơn lẻ tại bước kế tiếp: $s_t \rightarrow a_t$.
   * Trong quá trình chạy thực tế, robot bị lệch dù chỉ 1-2 mm so với quỹ đạo mẫu ban đầu. Sự sai lệch nhỏ này đưa robot vào trạng thái mà nó chưa từng thấy trong tập train (*out-of-distribution*), dẫn đến hành động sai lớn hơn $\rightarrow$ robot trôi dạt hoàn toàn và đứng yên hoặc va chạm.
   * **Giải pháp của ACT:** Cơ chế **Action Chunking** dự đoán cùng lúc một chuỗi hành động tương lai $k=50$ bước (tương đương 1 giây tại 50Hz). Điều này giúp quỹ đạo dự đoán có tính mạch lạc dài hạn, loại bỏ tình trạng phản xạ tức thời giật cục.
2. **Phân phối đa mốt của con người (Multimodal Distribution):**
   * Dữ liệu do người vận hành thu thập luôn có tính đa mốt: ví dụ tại bước xuất phát, người thu có thể chọn di chuyển tay trái trước hoặc tay phải trước; hoặc có những đoạn dừng ngắn kiểm tra góc nhìn.
   * BC cổ điển tối ưu theo hàm mất mát MSE sẽ lấy giá trị kỳ vọng (trung bình cộng của hai cách đi) $\rightarrow$ kết quả là robot đi vào vị trí trung bình giữa hai tay, hoàn toàn sai lệch logic vật lý.
   * **Giải pháp của ACT:** Tích hợp **Conditional VAE (CVAE)** với biến ẩn $z \sim \mathcal{N}(0, I)$ để tham số hóa phân phối đa mốt này một cách chính xác.

---

## 3. TÍNH TƯƠNG THÍCH ĐẶC BIỆT GIỮA ACT VÀ OPENARM

1. **Thừa kế trực tiếp từ kiến trúc Stanford ALOHA:**
   * Mô hình ACT được đề xuất bởi nhóm tác giả Stanford (Tony Z. Zhao et al., RSS 2023) chuyên biệt cho hệ robot hai tay ALOHA (14–16 DOF).
   * Hệ thống OpenArm của chúng ta có topology gần như tương đồng hoàn toàn: **2 tay $\times$ 8 động cơ = 16 bậc tự do**, phối hợp với 1 camera ngực RGB-D để triệt tiêu điểm mù thao tác.
2. **Hiệu quả mẫu vượt trội trong môi trường thiếu dữ liệu (Low-data Regime):**
   * Bài báo gốc ALOHA chứng minh: chỉ với **khoảng 50 lần thao tác mẫu (50 demonstrations)**, ACT đã học thành thạo các tác vụ đòi hỏi sự khéo léo cao (như mở nắp chai, cắm dây cáp, gắp phân loại).
   * Điều này cực kỳ phù hợp với điều kiện của nhóm là phải **tự thu thập dữ liệu bằng tay** thông qua cơ chế Master-Slave/Teleoperation trong thời gian ngắn.

---

## 4. KẾ HOẠCH TIẾN ĐỘ 6 TUẦN & QUẢN TRỊ RỦI RO (PROJECT RISK MANAGEMENT)

### 4.1. Phân bổ công việc 6 tuần theo lộ trình ACT

```
Tuần 1: Cân chỉnh phần cứng CAN-FD, Calibrate camera RGB-D & Thiết lập teleop Leader-Follower
Tuần 2: Thu thập 50 - 70 episodes chất lượng cao (HDF5 format, lọc nhiễu vận tốc và độ sâu)
Tuần 3: Tiền xử lý dữ liệu, kiểm tra tính toàn vẹn của dataset và chuẩn hóa góc khớp (Normalize Qpos/Action)
Tuần 4: Huấn luyện ACT (ResNet18 4-kênh + CVAE + Transformer Decoder), điều chỉnh beta-KL & L1 loss
Tuần 5: Tích hợp Policy vào vòng lặp CAN bus 50Hz, thực nghiệm Temporal Ensembling trên robot thật
Tuần 6: Tinh chỉnh các trường hợp biên (edge cases), đo lường tỷ lệ thành công (Success Rate) & viết báo cáo
```

### 4.2. Ma trận rủi ro khi chọn các giải pháp khác nhau

| Mô hình | Rủi ro lớn nhất | Tác động đến hạn chót 6 tuần | Khả năng hoàn thành |
| :--- | :--- | :--- | :---: |
| **RL (Isaac Gym)** | Mất 4 tuần xây dựng sim physics mà transfer ra đời thực thất bại vì sim-to-real gap. | Dự án không có sản phẩm chạy trên phần cứng thật. | **Rất thấp (< 20%)** |
| **Diffusion Policy** | Độ trễ suy luận >30ms gây nghẽn CAN bus 50Hz; mất nhiều ngày debug schedule nhiễu. | Chậm tiến độ deploy, phải dành 2 tuần tối ưu TensorRT. | **Trung bình (~ 50%)** |
| **Vanilla BC** | Robot giật cục, trôi sai số tích lũy sau 2-3 giây chạy thực tế. | Sản phẩm không hoạt động ổn định, tỷ lệ thành công thấp. | **Thấp (~ 30%)** |
| **ACT Pipeline** | Lỗi dữ liệu teleoperation không chuẩn $\rightarrow$ giải quyết nhanh bằng việc thu lại 20-30 demo. | Đảm bảo đúng tiến độ, thời gian train ngắn, suy luận thời gian thực mượt mà. | **Rất cao (> 90%)** |

---

## 5. TỔNG KẾT (EXECUTIVE SUMMARY)

Lựa chọn **ACT (Action Chunking with Transformers)** là quyết định kỹ thuật chuẩn xác và có cơ sở khoa học vững chắc nhất cho bài toán OpenArm 16-DOF bởi vì:

1. **Đáp ứng thời gian thực (Real-time Budget):** Forward pass $<15$ ms, hoàn toàn tương thích với chu kỳ bus CAN 50Hz (20 ms).
2. **Tiết kiệm dữ liệu (Data-efficient):** Chỉ cần $50 - 70$ demonstrations tự thu thập để đạt tỷ lệ thành công cao.
3. **Chuyển động mượt mà (Smooth Motion):** Cơ chế Action Chunking ($k=50$) kết hợp Temporal Ensembling ($e^{-m \cdot i}$) giải quyết triệt để bài toán trôi sai số tích lũy và bảo vệ cơ khí phần cứng.
4. **Tính khả thi tuyệt đối trong 6 tuần:** Tối ưu hóa thời gian phát triển, hạn chế tối đa rủi ro không thể kiểm soát từ việc mô phỏng RL hoặc tối ưu hoá độ trễ của Diffusion.

---

### Tài liệu tham khảo chính:
1. **Tony Z. Zhao, Vikash Kumar, Sergey Levine, Chelsea Finn.** *"Learning Fine Manipulation with Low-Cost Hardware."* Robotics: Science and Systems (RSS), 2023. (ALOHA / ACT Paper).
2. **Cheng Chi, Siyuan Feng, Yilun Du, Zhenjia Xu, Eric Cousineau, Benjamin Burchfiel, Shuran Song.** *"Diffusion Policy: Visuomotor Policy Learning via Action Diffusion."* Robotics: Science and Systems (RSS), 2023.
3. **Dean A. Pomerleau.** *"ALVINN: An Autonomous Land Vehicle in a Neural Network."* NeurIPS, 1988. (Nghiên cứu nền tảng về hạn chế của Covariate Shift trong Behavioral Cloning).
