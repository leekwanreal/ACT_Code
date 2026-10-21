# Action Chunking with Transformers (ACT) - Bimanual OpenArm 16-DOF & RGB-D

Pipeline huấn luyện, đánh giá và triển khai thuật toán **ACT (Action Chunking with Transformers)** dành riêng cho hệ thống robot hai cánh tay **Bimanual OpenArm (16 bậc tự do)** kết hợp **01 Camera ngực Intel RealSense RGB-D**.

Dự án được xây dựng dựa trên công trình nghiên cứu gốc của nhóm tác giả Stanford University (*Tony Z. Zhao, Vikash Kumar, Sergey Levine, Chelsea Finn - RSS 2023 / ALOHA*), được tùy biến tối ưu hóa cho phần cứng động cơ Damiao CAN-FD và quy trình gấp khăn/áo hai tay tự động.

---

## 📑 MỤC LỤC

1. [Cấu Trúc Thư Mục](#-cấu-trúc-thư-mục)
2. [Kiến Trúc Mô Hình & Điểm Cốt Lõi](#-kiến-trúc-mô-hình--điểm-cốt-lõi)
3. [Bảng Phân Bổ 16 Khớp Động Cơ (16-DOF)](#-bảng-phân-bổ-16-khớp-động-cơ-16-dof)
4. [Chuẩn Bị Dữ Liệu Huấn Luyện (Dataset)](#-chuẩn-bị-dữ-liệu-huấn-luyện-dataset)
5. [Hướng Dẫn Huấn Luyện (Training)](#-hướng-dẫn-huấn-luyện-training)
6. [Hướng Dẫn Đánh Giá Offline (Evaluation)](#-hướng-dẫn-đánh-giá-offline-evaluation)
7. [Triển Khai Điều Khiển Trên Robot Thật (Inference)](#-triển-khai-điều-khiển-trên-robot-thật-inference)
8. [Cơ Chế An Toàn Phần Cứng (Hardware Safety)](#-cơ-chế-an-toàn-phần-cứng-hardware-safety)
9. [Cài Đặt Môi Trường & Thư Viện](#-cài-đặt-môi-trường--thư-viện)

---

## 📂 CẤU TRÚC THƯ MỤC

```text
act_pipeline/
├── __init__.py                 # Đăng ký module act_pipeline
├── config.py                   # Cấu hình siêu tham số (ModelConfig, TrainConfig, EvalConfig)
├── train.py                    # Script huấn luyện chính (AMP FP16, Cosine LR, Checkpoint, Logging)
├── eval.py                     # Script đánh giá định lượng (L1/MSE, Jerk/Smoothness, Per-joint Error)
├── README.md                   # Tài liệu hướng dẫn toàn diện (file này)
│
├── models/                     # Kiến trúc mạng nơ-ron học sâu
│   ├── __init__.py
│   ├── backbone.py             # YOLOv8-Nano / ConvNeXt-Nano 4 kênh (RGB+Depth), C2f & SPPF, selective freezing
│   ├── cvae.py                 # CVAE Transformer Encoder (chuẩn hóa đa mốt, latent z in R^32)
│   ├── act_model.py            # Transformer Decoder tổng hợp (Vision Tokens + Qpos + Latent z)
│   └── policy.py               # ACTPolicy đóng gói quy trình train (CVAE sample) & test (z=0)
│
├── data/                       # Xử lý và nạp dữ liệu
├── __init__.py
├── dataset.py              # BimanualEpisodicDataset đọc file HDF5 theo episode
├── preprocess.py           # Tiền xử lý RGB-D (Resize, Normalization độ sâu 0.2m - 1.2m)
└── normalization.py        # Tính toán & lưu trữ mean/std chuẩn hóa (stats.json)
│
└── utils/                      # Các tiện ích bổ trợ
    ├── __init__.py
    ├── checkpoint.py           # Quản lý lưu/khôi phục checkpoint (best, latest, deploy weights)
    ├── logger.py               # Ghi nhật ký đa kênh (Console, File train.log, metrics.jsonl, TensorBoard, WandB)
    └── temporal_ensemble.py    # Bộ lọc làm mượt chuyển động lũy thừa exp(-m * i)
```

---

## 🧠 KIẾN TRÚC MÔ HÌNH & ĐIỂM CỐT LÕI

```text
[Ảnh RGB (3)] + [Ảnh Depth (1)]
            │
            ▼
 ┌────────────────────────────────────────┐
 │ YOLOv8-Nano RGB-D Backbone (Pure-Torch)│  (Stem 4-ch Trainable, C2f & SPPF Frozen)
 └───────────────────┬────────────────────┘
                     │  Feature Map Stride 16 -> 405 Visual Tokens [B, 405, 512]
                     ▼
┌────────────────────────────────────────────────────────────────────────┐
│                   TRANSFORMER POLICY (ACT DECODER)                     │
│                                                                        │
│  Inputs:                                                               │
│  - 405 Visual Tokens (2D Sinusoidal Positional Encoded)                │
│  - 1 Proprioception Token (Góc 16 khớp hiện tại qpos)                  │
│  - 1 Latent Style Token (z in R^32 sinh bởi CVAE Encoder)              │
│                                                                        │
│  Structure: 7 Layers Transformer Decoder (d_model=512, nheads=8)       │
│  Output Queries: 50 Learnable Query Tokens                             │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
                Action Chunk [B, k=50, action_dim=16]
          (Chuỗi 50 bước hành động tương lai ~ 1.0 giây tại 50Hz)
```

### 1. Vision Backbone 4 Kênh (YOLOv8 RGB-D & ConvNeXt-Nano)
- Nhận đầu vào là tensor 2 camera RGB-D `[B, 2, 4, 480, 640]` (gồm camera trước ngực `chest` và camera đối diện `front`, mỗi camera gồm 3 kênh RGB + 1 kênh Depth chuẩn hóa trong dải `[0.2m, 1.2m]`).
- **Khối C2f & SPPF**: Trích xuất đặc trưng đa tỉ lệ sắc nét, vượt trội trong việc phát hiện vật thể và nhận diện khoảng cách hình học 3D phục vụ bài toán gắp nhặt (pick-and-place).
- **Cơ chế đóng băng có chọn lọc (Selective Freezing)**: Đóng băng các tầng sâu để bảo toàn năng lực phát hiện vật thể, chỉ mở khóa huấn luyện Stem Adapter 4 kênh với learning rate nhỏ (`lr_backbone = 1e-5`) và 1x1 Conv Projection sang 512 chiều.

### 2. CVAE Transformer Encoder (Mô hình hóa dữ liệu đa mốt)
- Trong lúc dạy học bằng dắt tay (teleoperation), con người có thể thực hiện cùng một tác vụ theo nhiều cách khác nhau (ví dụ: gắp mép khăn bên trái trước hoặc bên phải trước).
- CVAE nén toàn bộ chuỗi hành động 50 bước tương lai vào một vector ngẫu nhiên $z \sim \mathcal{N}(\mu, \sigma^2)$ ($z \in \mathbb{R}^{32}$).
- **Khi huấn luyện**: Dùng Reparameterization Trick kết hợp hàm mất mát KL-Divergence ($\beta = 10.0$).
- **Khi đánh giá / chạy thực tế**: Cố định vector $z = \mathbf{0}$ để robot thực thi quỹ đạo tối ưu nhất một cách tất định.

### 3. Action Chunking ($k = 50$)
- Thay vì dự đoán từng bước rời rạc ($1$ bước hành động cho mỗi ảnh) khiến sai số tích lũy nhanh chóng (compounding error) sau vài giây, ACT dự đoán cùng lúc một **Action Chunk 50 bước liên tiếp** ($k=50$, tương đương $1.0\text{s}$ ở tần số $50\text{Hz}$).
- Triệt tiêu hoàn toàn độ trễ điều khiển và giúp robot thực hiện các chuyển động phối hợp hai tay mượt mà.

### 4. Temporal Ensembling (Bộ lọc quỹ đạo lũy thừa)
- Tại mỗi chu kỳ $t$, ACT dự đoán 50 bước hành động tiếp theo. Sau nhiều bước, các chuỗi dự đoán này sẽ gối đầu lên nhau.
- Thay vì chỉ lấy giá trị của chunk mới nhất, Temporal Ensembling lấy trung bình có trọng số của tất cả các dự đoán trong quá khứ gối lên thời điểm hiện tại:
  $$w_i = \exp(-m \cdot i)$$
  *(với $m = 0.01$, $i$ là độ cũ của dự đoán tính từ lúc chunk được sinh ra)*.
- **Tác dụng**: Triệt tiêu rung giật (jerk), loại bỏ các bước nhảy góc đột ngột, giúp động cơ chuyển động cực êm và bảo vệ hộp số cycloid/harmonic của Damiao.

---

## 🦾 BẢNG PHÂN BỔ 16 KHỚP ĐỘNG CƠ (16-DOF)

| Thứ tự Index | Tên khớp chuẩn | Mã Motor | Bus CAN | CAN Send ID | CAN Recv ID | Giới hạn góc (Radian) | Chức năng chi tiết |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **0** | `Left_J1_Base` | DM8009 | `can1` | `0x01` | `0x11` | -1.396 .. +3.491 | Xoay đế vai trái (Yaw) |
| **1** | `Left_J2_Shoulder` | DM8009 | `can1` | `0x02` | `0x12` | -3.316 .. +0.175 | Nâng/hạ bắp tay trái (Pitch) |
| **2** | `Left_J3_Elbow_Pitch` | DM4340 | `can1` | `0x03` | `0x13` | -1.571 .. +1.571 | Xoay vặn bắp tay trái (Roll) |
| **3** | `Left_J4_Elbow_Roll` | DM4340 | `can1` | `0x04` | `0x14` | 0.000 .. +2.444 | Gập/duỗi khuỷu tay trái (Pitch) |
| **4** | `Left_J5_Wrist_Pitch` | DM4310 | `can1` | `0x05` | `0x15` | -1.571 .. +1.571 | Xoay cẳng tay trái (Roll) |
| **5** | `Left_J6_Wrist_Roll` | DM4310 | `can1` | `0x06` | `0x16` | -0.785 .. +0.785 | Gập/ngửa cổ tay trái (Pitch) |
| **6** | `Left_J7_Wrist_Yaw` | DM4310 | `can1` | `0x07` | `0x17` | -1.571 .. +1.571 | Lắc cổ tay trái trái/phải (Yaw) |
| **7** | `Left_J8_Gripper` | DM4310 | `can1` | `0x08` | `0x18` | 0.000 .. +1.150 | Kẹp trái (0.0=Mở, 41.5mm=Đóng) |
| **8** | `Right_J1_Base` | DM8009 | `can0` | `0x01` | `0x11` | -1.396 .. +3.491 | Xoay đế vai phải (Yaw) |
| **9** | `Right_J2_Shoulder` | DM8009 | `can0` | `0x02` | `0x12` | -0.175 .. +3.316 | Nâng/hạ bắp tay phải (Pitch) |
| **10** | `Right_J3_Elbow_Pitch`| DM4340 | `can0` | `0x03` | `0x13` | -1.571 .. +1.571 | Xoay vặn bắp tay phải (Roll) |
| **11** | `Right_J4_Elbow_Roll` | DM4340 | `can0` | `0x04` | `0x14` | 0.000 .. +2.444 | Gập/duỗi khuỷu tay phải (Pitch) |
| **12** | `Right_J5_Wrist_Pitch`| DM4310 | `can0` | `0x05` | `0x15` | -1.571 .. +1.571 | Xoay cẳng tay phải (Roll) |
| **13** | `Right_J6_Wrist_Roll` | DM4310 | `can0` | `0x06` | `0x16` | -0.785 .. +0.785 | Gập/ngửa cổ tay phải (Pitch) |
| **14** | `Right_J7_Wrist_Yaw`  | DM4310 | `can0` | `0x07` | `0x17` | -1.571 .. +1.571 | Lắc cổ tay phải trái/phải (Yaw) |
| **15** | `Right_J8_Gripper` | DM4310 | `can0` | `0x08` | `0x18` | 0.000 .. +1.150 | Kẹp phải (0.0=Mở, 41.5mm=Đóng) |

---

## 📊 CHUẨN BỊ DỮ LIỆU HUẤN LUYỆN (DATASET)

Tập dữ liệu được tổ chức dưới định dạng file HDF5 (`.hdf5` hoặc `.h5`) cho từng episode thu thập được:

```text
data_set/pick_and_place/
├── episode_0.hdf5
├── episode_1.hdf5
├── ...
└── episode_49.hdf5
```

Mỗi file HDF5 chứa các trường dữ liệu được nén theo chuẩn:
- `/observations/qpos`: `[T, 16]` kiểu `float32` (góc hiện tại 16 khớp theo đơn vị Radian).
- `/action`: `[T, 16]` kiểu `float32` (lệnh góc mục tiêu cho 16 khớp).
- `/observations/images/chest_rgb`: `[T, 480, 640, 3]` kiểu `uint8` (ảnh màu RGB từ RealSense).
- `/observations/images/chest_depth`: `[T, 480, 640]` kiểu `uint16` (ảnh độ sâu tính bằng milimét).

---

## 🚀 HƯỚNG DẪN HUẤN LUYỆN (TRAINING)

### 1. Huấn luyện cơ bản
Chạy lệnh huấn luyện với thiết lập mặc định (500 epochs, batch size 16, Cosine Annealing LR):

```bash
python3 -m act_pipeline.train \
    --dataset_dir data_set/pick_and_place \
    --output_dir checkpoints/act_openarm \
    --epochs 500 \
    --batch_size 16 \
    --lr 1e-4 \
    --device cuda
```

### 2. Huấn luyện nâng cao (Khuyên dùng trên GPU RTX / A100)
Tận dụng Mixed Precision FP16, theo dõi trên Weights & Biases:

```bash
CUDA_VISIBLE_DEVICES=0 python3 -m act_pipeline.train \
    --dataset_dir data_set/pick_and_place \
    --output_dir checkpoints/act_openarm \
    --backbone_type yolov8_nano \
    --decoder_layers 7 \
    --yolo_stride 16 \
    --epochs 100 \
    --batch_size 16 \
    --lr 1e-4 \
    --lr_backbone 1e-5 \
    --kl_weight 10.0 \
    --warmup_epochs 15 \
    --val_split 0.15 \
    --eval_every 10 \
    --save_every 10 \
    --use_tensorboard \
    --use_wandb \
    --wandb_api_key "YOUR_WANDB_API_KEY" \
    --wandb_project act-openarm \
    --device cuda
```

### 3. Tiếp tục huấn luyện từ checkpoint cũ (Resume Training)
```bash
# Tự động tìm checkpoint mới nhất trong thư mục output_dir:
python3 -m act_pipeline.train --resume auto --output_dir checkpoints/act_openarm

# Hoặc chỉ định rõ file checkpoint:
python3 -m act_pipeline.train --resume checkpoints/act_openarm/checkpoint_epoch_250.pth
```

### 4. Giám sát quá trình huấn luyện
Mở TensorBoard trên máy tính để theo dõi đồ thị hàm mất mát theo thời gian thực:
```bash
tensorboard --logdir checkpoints/act_openarm/runs --port 6006
```
Truy cập: `http://localhost:6006` để xem đồ thị:
- `loss/total_train` & `loss/total_val`: Tổng loss hàm mục tiêu.
- `loss/recon_l1`: Sai số phục dựng quỹ đạo hành động (L1 Loss).
- `loss/kl_divergence`: Khoảng cách phân phối ẩn CVAE.
- `lr/policy`: Tốc độ học suy giảm mượt mà theo Cosine.

---

## 📈 HƯỚNG DẪN ĐÁNH GIÁ OFFLINE (EVALUATION)

Sau khi huấn luyện xong, chạy script `eval.py` để kiểm thử định lượng mô hình trên tập validation:

```bash
python3 -m act_pipeline.eval \
    --checkpoint_path checkpoints/act_openarm/best_checkpoint.pth \
    --dataset_dir data_set/pick_and_place \
    --output_dir evaluation_results \
    --device cuda
```

### Đầu ra của quá trình đánh giá:
1. **Bảng sai số từng khớp (Per-Joint MAE)**:
   - Sai số Radian và Góc độ ($^\circ$) trên từng động cơ trong số 16 khớp.
   - Sai số đạt chuẩn tốt nếu $\text{MAE} < 3^\circ$ đối với các khớp cánh tay và $\text{MAE} < 1.5\text{mm}$ đối với tay kẹp Gripper.
2. **So sánh Temporal Ensembling**:
   - So sánh độ rung giật (Jerk Metric) giữa việc xuất raw action chunk và việc hòa trộn lũy thừa $\exp(-m \cdot i)$.
3. **Báo cáo JSON**: Toàn bộ kết quả được tự động xuất ra file `evaluation_results/eval_report.json`.

---

## 🤖 TRIỂN KHAI ĐIỀU KHIỂN TRÊN ROBOT THẬT (INFERENCE)

Script `tools/infer_robot.py` chịu trách nhiệm nhận luồng ảnh trực tiếp từ camera RealSense, nạp trạng thái góc khớp thời gian thực qua SocketCAN, thực hiện suy luận mô hình ACT ở tần số **50 Hz (20ms/chu kỳ)** và gửi xung điều khiển xuống 16 động cơ.

### 1. Lệnh chạy thực tế trên phần cứng
```bash
python3 tools/infer_robot.py \
    --checkpoint checkpoints/act_openarm/best_checkpoint.pth \
    --can-left can1 \
    --can-right can0 \
    --ensemble-m 0.01 \
    --max-timesteps 1500
```

### 2. Chế độ chạy thử không cắm robot (Dry-Run Mode)
Dùng để kiểm tra tốc độ suy luận của mô hình và luồng đọc camera RealSense mà không gửi lệnh CAN xuống động cơ:
```bash
python3 tools/infer_robot.py \
    --checkpoint checkpoints/act_openarm/best_checkpoint.pth \
    --dry-run
```

---

## 🛡️ CƠ CHẾ AN TOÀN PHẦN CỨNG (HARDWARE SAFETY)

Trong quá trình điều khiển robot ngoài đời thực, pipeline áp dụng mô hình an toàn 5 tầng bảo vệ nghiêm ngặt:

1. **S-Curve Cosine Warm-up (Khởi động êm)**:
   - Khi vừa kích hoạt mô hình, góc của robot có thể đang lệch nhẹ so với góc khởi đầu của mô hình.
   - Script tự động sinh đường cong Cosine $2.5\text{s}$ để lái khớp từ vị trí thực tế sang vị trí bắt đầu một cách êm ái, **triệt tiêu hoàn toàn hiện tượng giật nảy khớp khi vừa bấm chạy**.

2. **Giới hạn vận tốc & gia tốc động học (Kinematic Clamping)**:
   - Đặt ngưỡng vận tốc tối đa và gia tốc cực đại phù hợp cho từng loại motor:
     - Khớp vai (DM8009): $v_{\max} = 1.2\text{ rad/s}$, $a_{\max} = 3.5\text{ rad/s}^2$.
     - Khớp khuỷu (DM4340): $v_{\max} = 2.0\text{ rad/s}$, $a_{\max} = 6.0\text{ rad/s}^2$.
     - Khớp cổ tay (DM4310): $v_{\max} = 3.0\text{ rad/s}$, $a_{\max} = 10.0\text{ rad/s}^2$.

3. **Điều khiển kẹp Gripper an toàn (POS_FORCE Limit)**:
   - Giới hạn lực kẹp tối đa `torque_pu = 0.15` (15% mô-men danh định). Ngăn chặn tình trạng kẹp quá chặt làm cháy cuộn dây motor hoặc vỡ vật thể.

4. **Tự động nhận diện điểm dừng (Stopping Criteria)**:
   - Khi robot hoàn thành thao tác gấp khăn và trở về vùng lân cận Home, hệ thống nhận diện độ biến thiên $|\Delta q| < 0.008\text{ rad}$ duy trì trong $1.0\text{s}$ để tự động kết thúc tác vụ và chuyển sang trạng thái chờ.

5. **Lui về Home an toàn khi ngắt khẩn cấp (Safe Home on Ctrl+C)**:
   - Nếu người vận hành bấm `Ctrl + C`, robot **KHÔNG ngắt điện đột ngột** (tránh làm rơi tự do hai cánh tay xuống bàn gây móp vỡ cơ khí).
   - Robot sẽ tự động chuyển sang đường cong S-Curve trong $2.0\text{s}$ để từ từ đưa hai cánh tay gập gọn về vị trí an toàn rồi mới ngắt mô-men.

---

## 📦 CÀI ĐẶT MÔI TRƯỜNG & THƯ VIỆN

Để chạy pipeline mượt mà trên môi trường Linux Ubuntu / WSL 2 với GPU NVIDIA:

```bash
# 1. Cài đặt PyTorch hỗ trợ CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 2. Cài đặt các thư viện bổ trợ cho Computer Vision & Dữ liệu
pip install h5py numpy opencv-python pyrealsense2 tensorboard websockets

# 3. (Tùy chọn) Theo dõi huấn luyện trực quan qua Cloud
pip install wandb
```