# 02. Nền tảng học sâu

## 1. Từ học máy đến mạng nơ-ron

Học sâu sử dụng mạng nơ-ron có nhiều lớp để tự học biểu diễn từ dữ liệu. Thay vì con người định nghĩa thủ công đặc trưng như “số lần xuất hiện từ khóa”, mô hình học cách biến token thành vector và kết hợp các vector đó thành biểu diễn ngữ nghĩa.

Một mạng đơn giản:

```math
h = f(W_1x + b_1)
```

```math
y = W_2h + b_2
```

Trong đó:

- `x`: đầu vào.
- `h`: biểu diễn ẩn.
- `f`: hàm kích hoạt phi tuyến.
- `y`: đầu ra.

Nếu chỉ xếp các phép tuyến tính mà không có hàm kích hoạt, nhiều lớp vẫn tương đương một phép tuyến tính duy nhất. Hàm kích hoạt giúp mô hình học quan hệ phức tạp.

## 2. Tensor và shape

Tensor là mảng nhiều chiều. Hiểu shape là kỹ năng quan trọng nhất khi đọc code deep learning.

Ví dụ batch token:

```text
shape = [batch_size, sequence_length]
```

Sau embedding:

```text
shape = [batch_size, sequence_length, d_model]
```

Trong scratch Transformer:

```text
batch_size = 4
source_length = 512
d_model = 256
source embedding shape = [4, 512, 256]
```

Khi chia thành tám attention head:

```text
head_dim = d_model / num_heads = 256 / 8 = 32
Q/K/V shape = [4, 8, 512, 32]
```

## 3. Embedding

Máy tính không hiểu trực tiếp từ `"Việt Nam"`. Tokenizer biến văn bản thành ID:

```text
"Việt Nam phát triển" → [431, 928, 1752, ...]
```

Embedding là một bảng tra cứu:

```math
E \in \mathbb{R}^{V \times d}
```

- `V`: kích thước từ vựng.
- `d`: số chiều embedding.

Mỗi token ID chọn một hàng trong bảng `E`, tạo thành vector có thể học.

Embedding học được rằng các token xuất hiện trong ngữ cảnh tương tự nên có biểu diễn gần nhau. Tuy nhiên, embedding token đơn thuần không biết thứ tự token; Transformer cần positional encoding.

## 4. Forward pass

Forward pass là quá trình đưa batch qua mô hình để tạo logits và loss:

```text
input IDs
→ embeddings
→ nhiều lớp Transformer
→ logits trên toàn bộ vocabulary
→ cross-entropy loss với labels
```

Nếu batch có shape `[B, T]` và vocabulary có `V` token, logits thường có shape:

```text
[B, T, V]
```

Mỗi vị trí có một phân phối xác suất trên toàn bộ từ vựng.

## 5. Backpropagation

Backpropagation dùng quy tắc đạo hàm dây chuyền để tính gradient từ loss ngược qua mọi lớp:

```text
loss
← output projection
← decoder
← encoder/cross-attention
← embeddings
```

Sau `loss.backward()`, mỗi tham số có gradient. Sau `optimizer.step()`, tham số được cập nhật. Sau đó `optimizer.zero_grad()` xóa gradient cũ trước vòng tiếp theo, trừ khi đang gradient accumulation.

## 6. Hàm kích hoạt

### ReLU

```math
\mathrm{ReLU}(x)=\max(0,x)
```

Đơn giản và nhanh nhưng có vùng gradient bằng 0.

### GELU

GELU điều chỉnh đầu vào mềm hơn ReLU và thường được dùng trong Transformer. Scratch Transformer của dự án dùng GELU trong feed-forward network.

## 7. Residual connection

Residual connection cộng đầu vào của một khối vào đầu ra:

```math
y = x + F(x)
```

Nó giúp:

- Gradient đi qua mạng sâu dễ hơn.
- Lớp học phần điều chỉnh thay vì phải học lại toàn bộ biểu diễn.
- Huấn luyện ổn định hơn.

Transformer sử dụng residual quanh attention và feed-forward.

## 8. Layer normalization

LayerNorm chuẩn hóa các đặc trưng trong mỗi token:

```math
\mathrm{LayerNorm}(x)
```

Hai cách đặt phổ biến:

- **Post-LN**: `LayerNorm(x + F(x))`.
- **Pre-LN**: `x + F(LayerNorm(x))`.

Scratch Transformer của dự án dùng Pre-LN vì thường ổn định hơn khi train từ đầu.

## 9. Dropout

Dropout ngẫu nhiên đặt một phần activation về 0 khi train. Điều này buộc mô hình không phụ thuộc quá mức vào một đường biểu diễn cụ thể.

```text
train mode: dropout hoạt động
eval mode: dropout tắt
```

Nếu quên `model.eval()` khi đánh giá, kết quả có thể không ổn định.

## 10. Padding và mask

Các câu trong batch có độ dài khác nhau. Ta thêm token padding để chúng có cùng chiều dài:

```text
[BOS, tôi, đi, học, EOS, PAD, PAD]
[BOS, hôm, nay, trời, đẹp, EOS, PAD]
```

Padding mask bảo mô hình bỏ qua vị trí `PAD`.

Causal mask bảo decoder không nhìn token tương lai:

```text
token tại vị trí t chỉ được nhìn các vị trí ≤ t
```

Nếu không có causal mask khi train decoder, mô hình sẽ nhìn thấy đáp án tương lai và bài toán bị rò rỉ nhãn.

## 11. Precision: FP32, FP16 và BF16

- FP32: chính xác cao, tốn bộ nhớ.
- FP16: dùng ít bộ nhớ, nhanh trên GPU nhưng dễ underflow/overflow hơn.
- BF16: dải giá trị rộng hơn FP16, cần phần cứng hỗ trợ.

Các thí nghiệm pretrained dùng FP16 trên Tesla T4 để giảm bộ nhớ và tăng tốc.

## 12. Gradient clipping

Gradient đôi khi tăng quá lớn, gây cập nhật bất ổn. Gradient clipping giới hạn norm của gradient trước khi optimizer cập nhật.

Scratch Transformer sử dụng gradient clipping vì chuỗi dài và mô hình train từ ngẫu nhiên dễ gặp gradient spike.

## 13. Gradient checkpointing

Khi forward, mô hình thường lưu activation để backward. Gradient checkpointing chỉ lưu một số activation, rồi tính lại phần còn thiếu trong backward:

- Ưu: giảm bộ nhớ GPU.
- Nhược: tăng thời gian tính toán.

Pretrained pipeline bật gradient checkpointing để ViT5/Qwen vừa bộ nhớ T4.

## 14. Teacher forcing

Trong train sequence-to-sequence, decoder nhận token đúng trước đó:

```text
decoder input:  [BOS, y1, y2, y3]
labels:         [y1,  y2, y3, EOS]
```

Đây là teacher forcing. Nó giúp train hiệu quả vì có thể dự đoán nhiều vị trí song song.

Khi inference, không có đáp án đúng. Decoder phải dùng token chính nó vừa sinh:

```text
BOS → dự đoán y1 → dùng y1 dự đoán y2 → ...
```

Sự khác biệt này là một lý do prediction thực tế có thể mắc lỗi dây chuyền dù train loss thấp.

## 15. Checklist hiểu chương

Người đọc nên giải thích được:

1. Token ID khác embedding thế nào?
2. Shape `[B, T, V]` của logits có ý nghĩa gì?
3. Tại sao cần residual, LayerNorm và dropout?
4. Padding mask khác causal mask thế nào?
5. Forward, backward và optimizer step diễn ra theo thứ tự nào?

