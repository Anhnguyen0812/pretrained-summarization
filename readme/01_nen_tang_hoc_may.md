# 01. Nền tảng học máy

## 1. Học máy là gì?

Trong lập trình truyền thống, con người viết trực tiếp quy tắc:

```text
dữ liệu + quy tắc do người viết → kết quả
```

Trong học máy, con người cung cấp dữ liệu và ví dụ kết quả mong muốn. Thuật toán tìm ra một tập tham số để biến đầu vào thành đầu ra:

```text
dữ liệu + kết quả đúng → thuật toán học → mô hình có tham số
mô hình + dữ liệu mới → dự đoán
```

Với dự án này:

- Đầu vào `X`: một bài báo tiếng Việt dài.
- Đầu ra đúng `Y`: bản tóm tắt tham chiếu do tập dữ liệu cung cấp.
- Mô hình: Transformer scratch, ViT5, BARTpho hoặc Qwen.
- Mục tiêu học: với bài báo mới, sinh ra chuỗi token gần với bản tóm tắt đúng và trung thành với bài báo.

## 2. Mô hình, tham số và siêu tham số

### Tham số

Tham số là các giá trị được mô hình tự học trong quá trình train, thường là trọng số và bias trong các ma trận.

Ví dụ một lớp tuyến tính:

```math
y = Wx + b
```

`W` và `b` là tham số. Ban đầu chúng thường được khởi tạo ngẫu nhiên hoặc nạp từ checkpoint tiền huấn luyện. Sau mỗi bước train, optimizer cập nhật chúng.

Trong dự án:

- Transformer scratch có khoảng 11,47 triệu tham số, khởi tạo gần như ngẫu nhiên.
- ViT5-base có khoảng 227,72 triệu tham số đã học từ dữ liệu tiếng Việt lớn.
- ViT5 full fine-tuning cập nhật toàn bộ 227,72 triệu tham số.
- ViT5 LoRA chỉ cập nhật khoảng 1,77 triệu tham số adapter.

### Siêu tham số

Siêu tham số do con người chọn trước khi train:

- Learning rate.
- Batch size.
- Số epoch.
- Số lớp, số attention head, `d_model`.
- Dropout.
- Độ dài nguồn và đích tối đa.
- LoRA rank.
- Beam size khi giải mã.

Mô hình không tự học trực tiếp các giá trị này. Ta chọn chúng dựa trên validation.

## 3. Supervised learning trong dự án

Đây là bài toán học có giám sát vì mỗi bài viết train đi kèm một bản tóm tắt đúng:

```text
article → summary
```

Một mẫu có dạng:

```json
{
  "article": "Văn bản nguồn dài...",
  "summary": "Bản tóm tắt tham chiếu..."
}
```

Trong mỗi bước train:

1. Mô hình nhận `article`.
2. Mô hình dự đoán xác suất token tiếp theo của `summary`.
3. So sánh dự đoán với token đúng.
4. Tính loss.
5. Lan truyền gradient ngược.
6. Optimizer cập nhật tham số để lần sau dự đoán đúng hơn.

## 4. Train, validation và test

### Train

Tập train dùng để tính gradient và cập nhật trọng số. Mô hình được nhìn thấy các mẫu này nhiều lần.

Trong dự án: 10.775 mẫu train.

### Validation

Tập validation không dùng để cập nhật trọng số. Nó dùng để:

- Kiểm tra khả năng tổng quát hóa.
- Chọn checkpoint tốt nhất.
- Chọn learning rate, độ dài ngữ cảnh, LoRA rank và cấu hình giải mã.
- Phát hiện overfitting.

Trong dự án: 1.349 mẫu validation.

### Test

Tập test chỉ dùng để đánh giá cuối cùng sau khi đã chốt mô hình và cấu hình. Nếu liên tục thử cấu hình trên test, test sẽ vô tình trở thành validation và kết quả không còn khách quan.

Trong dự án: 1.344 mẫu test; không dùng để phân tích dữ liệu hoặc chọn mô hình.

## 5. Loss là gì?

Loss là một số thể hiện mức sai của mô hình trên batch hiện tại. Với sinh văn bản, loss phổ biến là cross-entropy.

Giả sử token đúng tiếp theo là `"Hà"`. Mô hình dự đoán:

| Token | Xác suất |
| --- | --- |
| Hà | 0.70 |
| Việt | 0.15 |
| thành | 0.10 |
| khác | 0.05 |

Loss tại bước đó:

```math
L = -\log P(\text{"Hà"})
```

Nếu xác suất token đúng cao, loss thấp. Nếu xác suất token đúng thấp, loss cao.

Với toàn bộ summary:

```math
\mathcal{L} = -\sum_{t=1}^{n} \log P(y_t \mid y_{<t}, X)
```

Điểm quan trọng: loss đo khả năng dự đoán token trong điều kiện train, còn ROUGE đo độ giống giữa summary được sinh hoàn chỉnh và reference. Loss giảm không đảm bảo ROUGE luôn tăng.

## 6. Gradient và gradient descent

Gradient cho biết nếu thay đổi một tham số một lượng rất nhỏ thì loss thay đổi theo hướng nào.

Quy tắc cập nhật đơn giản:

```math
\theta_{\text{new}} = \theta_{\text{old}} - \eta \nabla_\theta \mathcal{L}
```

Trong đó:

- `θ`: toàn bộ tham số mô hình.
- `∇θL`: gradient của loss theo tham số.
- `η`: learning rate.

Learning rate quá lớn có thể khiến mô hình nhảy qua điểm tốt và train không ổn định. Quá nhỏ khiến train rất chậm hoặc mắc kẹt.

## 7. Batch, step và epoch

- **Một sample**: một cặp article-summary.
- **Batch**: một nhóm sample xử lý cùng lúc.
- **Training step**: một lần forward, backward và cập nhật optimizer.
- **Epoch**: mô hình đi qua toàn bộ tập train một lần.

Ví dụ:

```text
10.775 mẫu / batch hiệu dụng 16 ≈ 674 step mỗi epoch
3 epoch ≈ 2.022 step
```

### Gradient accumulation

GPU có thể không chứa được batch lớn. Ta chia batch hiệu dụng thành nhiều micro-batch:

```text
per_device_batch_size = 2
2 GPU
gradient_accumulation_steps = 4
effective batch size = 2 × 2 × 4 = 16
```

Gradient được cộng dồn qua nhiều micro-batch rồi optimizer mới cập nhật một lần.

## 8. Underfitting, overfitting và generalization

### Underfitting

Mô hình chưa học đủ:

- Train loss còn cao.
- Validation loss cũng cao.
- Prediction chung chung hoặc sai nhiều.

Nguyên nhân có thể là mô hình quá nhỏ, train quá ít, learning rate không phù hợp hoặc dữ liệu khó.

### Overfitting

Mô hình nhớ train nhưng kém trên dữ liệu mới:

- Train loss tiếp tục giảm.
- Validation loss tăng hoặc validation ROUGE dừng tăng.
- Prediction có vẻ khớp phong cách train nhưng thiếu tổng quát.

Biện pháp dùng trong dự án:

- Dropout.
- Weight decay.
- Label smoothing.
- Early stopping.
- Chọn checkpoint theo validation ROUGE-L.
- LoRA để cập nhật ít tham số hơn.

### Generalization

Generalization là khả năng làm tốt trên dữ liệu chưa từng thấy. Mục tiêu thực sự không phải giảm train loss thấp nhất, mà là sinh tóm tắt tốt cho bài viết mới.

## 9. Optimizer AdamW

AdamW là optimizer được dùng trong dự án. Nó:

- Điều chỉnh kích thước cập nhật riêng cho từng tham số.
- Dùng trung bình động của gradient và bình phương gradient.
- Tách weight decay khỏi cập nhật gradient.

Weight decay nhẹ giúp hạn chế trọng số tăng quá lớn và giảm overfitting.

## 10. Learning-rate schedule và warmup

Đầu quá trình train, trọng số chưa ổn định. Learning rate lớn ngay lập tức có thể phá hỏng mô hình, đặc biệt với Transformer.

Warmup tăng learning rate từ nhỏ lên giá trị mục tiêu trong vài phần trăm bước đầu. Sau đó scheduler giảm learning rate dần.

Trong dự án:

- Scratch dùng warmup rồi giảm theo nghịch đảo căn bậc hai.
- Pretrained dùng warmup rồi cosine decay.

## 11. Checklist hiểu chương

Sau chương này, người đọc nên trả lời được:

1. Tham số khác siêu tham số thế nào?
2. Tại sao validation không được dùng để cập nhật trọng số?
3. Một training step gồm những bước gì?
4. Loss giảm có đồng nghĩa summary chắc chắn tốt hơn không?
5. Vì sao test chỉ nên dùng ở cuối?