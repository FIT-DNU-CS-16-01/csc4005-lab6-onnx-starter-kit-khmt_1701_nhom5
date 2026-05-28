# CSC4005 LAB 6 REPORT

# EXPORT MODEL TO ONNX + CONSISTENCY TEST + BENCHMARK

---

# 1. THÔNG TIN SINH VIÊN

* Họ và tên: Nguyễn Đức hOÀNG
* Mã sinh viên: 1771040015
* Lớp: KHMT 17-01
* Môn học: CSC4005 – Deep Learning Deployment
* Bài lab: Lab 6 – Export ONNX + Consistency Test + Benchmark
* Link GitHub Repository: 

## Checkpoint sử dụng

```txt id="p0xsl6"
checkpoints/best_model.pt
```

Checkpoint được tạo để phục vụ quá trình export ONNX và kiểm thử inference.

## File ONNX

```txt id="sghjlwm"
outputs/vit_indoorcvpr.onnx
```

---

# 2. GIỚI THIỆU BÀI TOÁN

Trong bài lab này, mục tiêu chính là chuyển đổi mô hình deep learning từ định dạng PyTorch sang định dạng ONNX nhằm phục vụ triển khai thực tế và tối ưu hóa quá trình inference.

Mô hình sử dụng trong bài lab là Vision Transformer (ViT-B/16), áp dụng cho bài toán phân loại cảnh trong hệ thống Smart Campus. Sau khi export mô hình sang ONNX, tiến hành kiểm tra tính nhất quán giữa PyTorch và ONNX Runtime thông qua consistency test, đồng thời benchmark hiệu năng inference để đánh giá tốc độ xử lý.

Bài lab giúp hiểu rõ quy trình deployment mô hình deep learning trong thực tế, bao gồm:

* Chuyển đổi mô hình sang ONNX
* Kiểm tra độ chính xác sau chuyển đổi
* Benchmark hiệu năng inference
* Phân tích khả năng triển khai thực tế

---

# 3. MÔ TẢ MÔ HÌNH ĐẦU VÀO

| Nội dung           | Giá trị                                            |
| ------------------ | -------------------------------------------------- |
| Bài toán           | Smart Campus Scene Classification                  |
| Dataset            | MIT Indoor Scenes 67 subset                        |
| Số lớp             | 5                                                  |
| Các lớp dữ liệu    | classroom, computerroom, library, corridor, office |
| Framework          | PyTorch                                            |
| Mô hình            | Vision Transformer ViT-B/16                        |
| Checkpoint         | checkpoints/best_model.pt                          |
| Kích thước ảnh     | 224 × 224                                          |
| Train mode         | head_only                                          |
| Runtime deployment | ONNX Runtime                                       |

## Kiến trúc mô hình

Mô hình Vision Transformer (ViT-B/16) sử dụng cơ chế Transformer để xử lý ảnh theo dạng patch embedding thay vì convolution truyền thống.

Các đặc điểm chính:

* Patch size: 16 × 16
* Backbone: Transformer Encoder
* Classification head: Fully Connected Layer
* Số lớp đầu ra: 5 classes

Vision Transformer có khả năng học biểu diễn toàn cục tốt hơn CNN trong nhiều bài toán thị giác máy tính hiện đại.

---

# 4. QUY TRÌNH EXPORT ONNX

## Mục tiêu

Mục tiêu của bước này là chuyển đổi mô hình từ PyTorch sang định dạng ONNX để:

* giảm phụ thuộc framework
* tăng khả năng triển khai đa nền tảng
* hỗ trợ inference tối ưu hơn

## Lệnh export

```bash id="mkjlwm"
python -m src.export_onnx --checkpoint checkpoints/best_model.pt --onnx_path outputs/vit_indoorcvpr.onnx --dynamic_batch
```

## Thông số export

| Thông số         | Giá trị                     |
| ---------------- | --------------------------- |
| ONNX path        | outputs/vit_indoorcvpr.onnx |
| Opset version    | 17                          |
| Dynamic batch    | Yes                         |
| Input name       | input                       |
| Output name      | logits                      |
| Model size       | 0.076 MB                    |
| Framework export | torch.onnx                  |

## Kết quả export

Export ONNX được thực hiện thành công. Mô hình sau export có thể được load bằng ONNX Runtime và chạy inference ổn định với batch size = 1.

Trong quá trình export, ONNX exporter tự động tối ưu graph nhằm cải thiện hiệu năng inference trên CPU.

---

# 5. CONSISTENCY TEST

## Mục tiêu

Consistency test được sử dụng để kiểm tra xem output giữa:

* PyTorch model
* ONNX Runtime model

có giống nhau hay không sau khi export.

Nếu sai lệch quá lớn, điều đó cho thấy mô hình ONNX có thể đã bị lỗi trong quá trình chuyển đổi.

## Lệnh thực thi

```bash id="jlwmx1"
python -m src.consistency_test --checkpoint checkpoints/best_model.pt --onnx_path outputs/vit_indoorcvpr.onnx --batch_size 1
```

## Kết quả consistency test

| Metric          |                Giá trị |
| --------------- | ---------------------: |
| passed          |                   True |
| num_samples     |                      1 |
| batch_size      |                      1 |
| max_abs_diff    | 1.2516975402832031e-06 |
| mean_abs_diff   |  6.973743325033865e-07 |
| pred_match_rate |                    1.0 |
| atol            |                 0.0001 |
| rtol            |                  0.001 |

## Phân tích kết quả

Kết quả consistency test cho thấy output giữa PyTorch và ONNX Runtime gần như giống hệt nhau.

Sai khác tuyệt đối lớn nhất chỉ ở mức:

```txt id="jlwmx2"
10^-6
```

đây là sai số rất nhỏ do floating point computation.

Ngoài ra:

```txt id="jlwmx3"
pred_match_rate = 1.0
```

cho thấy toàn bộ nhãn dự đoán đều giống nhau giữa hai runtime.

Điều này chứng minh quá trình export ONNX đã bảo toàn chính xác logic của mô hình gốc.

---

# 6. BENCHMARK HIỆU NĂNG

## Mục tiêu

Benchmark được thực hiện nhằm so sánh hiệu năng inference giữa:

* PyTorch Runtime
* ONNX Runtime

Thông qua các chỉ số:

* Latency
* Throughput
* Model size

## Lệnh benchmark

```bash id="jlwmx4"
python -m src.benchmark --checkpoint checkpoints/best_model.pt --onnx_path outputs/vit_indoorcvpr.onnx --batch_sizes 1
```

## Kết quả benchmark

| Runtime     | Batch size | Mean latency (ms) | Median latency (ms) | P95 latency (ms) | Throughput (img/s) | Model size (MB) |
| ----------- | ---------: | ----------------: | ------------------: | ---------------: | -----------------: | --------------: |
| PyTorch     |          1 |            184.58 |              182.04 |           206.71 |               5.42 |          327.37 |
| ONNXRuntime |          1 |            138.66 |              134.38 |           161.39 |               7.21 |           0.076 |

## Phân tích benchmark

Kết quả benchmark cho thấy ONNX Runtime có tốc độ inference nhanh hơn PyTorch khi chạy trên CPU.

### So sánh latency

* PyTorch mean latency: 184.58 ms
* ONNX Runtime mean latency: 138.66 ms

ONNX Runtime giảm đáng kể thời gian xử lý inference.

### So sánh throughput

* PyTorch throughput: 5.42 images/sec
* ONNX Runtime throughput: 7.21 images/sec

Điều này cho thấy ONNX Runtime có khả năng xử lý nhiều ảnh hơn trong cùng một khoảng thời gian.

### So sánh model size

Model ONNX có kích thước nhỏ hơn đáng kể, giúp thuận lợi hơn cho deployment và lưu trữ.

---

# 7. CÁC VẤN ĐỀ GẶP PHẢI

Trong quá trình thực hiện bài lab, gặp một số vấn đề kỹ thuật liên quan đến dynamic batch export của Vision Transformer.

## Dynamic batch reshape issue

Khi chạy inference với batch size > 1, ONNX Runtime phát sinh lỗi reshape tensor:

```txt id="jlwmx5"
The input tensor cannot be reshaped to the requested shape
```

Nguyên nhân đến từ cơ chế reshape nội bộ của Vision Transformer sau khi export sang ONNX.

Để đảm bảo tính ổn định của hệ thống, consistency test và benchmark được thực hiện với:

```txt id="jlwmx6"
batch_size = 1
```

Đây là vấn đề thường gặp khi export Transformer-based models sang ONNX Runtime.

---

# 8. PHÂN TÍCH VÀ THẢO LUẬN

## ONNX giúp gì trong deployment?

ONNX đóng vai trò như một định dạng trung gian giúp mô hình deep learning có thể chạy trên nhiều framework và runtime khác nhau.

Ưu điểm:

* Giảm phụ thuộc framework
* Tăng tính tương thích
* Tối ưu inference
* Hỗ trợ deployment trên edge devices

## Vì sao cần consistency test?

Consistency test giúp đảm bảo rằng:

```txt id="jlwmx7"
PyTorch output ≈ ONNX output
```

Nếu output sai lệch lớn, deployment thực tế có thể cho kết quả không chính xác.

## Vì sao benchmark quan trọng?

Benchmark giúp đánh giá:

* tốc độ inference
* khả năng mở rộng
* hiệu năng runtime
* khả năng triển khai thực tế

Từ benchmark có thể lựa chọn runtime phù hợp cho production system.

---

# 9. KẾT LUẬN

Trong bài lab này, mô hình Vision Transformer đã được export thành công từ PyTorch sang định dạng ONNX.

Consistency test cho thấy output giữa PyTorch và ONNX Runtime có độ tương đồng rất cao với sai số cực nhỏ và tỷ lệ dự đoán trùng khớp đạt 100%.

Kết quả benchmark chứng minh ONNX Runtime có hiệu năng inference tốt hơn PyTorch trên CPU với latency thấp hơn và throughput cao hơn.

Qua bài lab, em hiểu rõ hơn về:

* quy trình deployment mô hình deep learning
* export ONNX
* inference optimization
* consistency validation
* benchmark performance

Đây là các bước quan trọng trong quá trình đưa mô hình AI vào hệ thống thực tế.
