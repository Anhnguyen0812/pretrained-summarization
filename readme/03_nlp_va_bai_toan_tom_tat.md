# 03. NLP và bài toán tóm tắt văn bản

## 1. Máy tính biểu diễn ngôn ngữ thế nào?

Văn bản là chuỗi ký tự, nhưng mô hình làm việc với số. Quy trình:

```text
văn bản
→ chuẩn hóa
→ tokenizer tách thành token/subword
→ token ID
→ embedding
→ mô hình
→ logits token
→ giải mã thành văn bản
```

Ví dụ minh họa:

```text
"Việt Nam phát triển"
→ ["Việt", " Nam", " phát", " triển"]
→ [431, 928, 1752, 816]
```

Token thực tế phụ thuộc tokenizer của từng mô hình.

## 2. Vì sao dùng subword?

Nếu mỗi từ là một token:

- Vocabulary rất lớn.
- Từ mới không có trong vocabulary trở thành unknown.
- Tiếng Việt có nhiều tên riêng, số liệu và biến thể.

Subword chia từ/cụm từ thành các mảnh phổ biến. Nhờ đó:

- Vocabulary nhỏ hơn.
- Có thể biểu diễn từ chưa gặp.
- Học được các thành phần dùng chung.

Đổi lại, một từ có thể thành nhiều token. Vì vậy “128 token” không tương đương “128 từ”.

## 3. SentencePiece BPE

Scratch Transformer dùng tokenizer SentencePiece BPE với vocabulary 16.000.

Ý tưởng BPE:

1. Bắt đầu từ các đơn vị nhỏ.
2. Đếm cặp đơn vị thường xuất hiện cạnh nhau.
3. Gộp cặp phổ biến.
4. Lặp đến khi đạt kích thước vocabulary.

SentencePiece học trực tiếp từ văn bản, không cần tách từ thủ công trước.

ViT5 cũng sử dụng tokenizer theo họ T5/SentencePiece, nhưng vocabulary đã được học trong giai đoạn tiền huấn luyện.

## 4. Tóm tắt trích xuất và tóm tắt trừu tượng

### Tóm tắt trích xuất

Chọn nguyên câu hoặc đoạn từ bài viết:

```text
bài viết → chấm điểm câu → lấy các câu quan trọng
```

Ưu điểm:

- Ít ảo giác hơn.
- Dễ triển khai.

Nhược điểm:

- Có thể dài và rời rạc.
- Không tái diễn đạt hoặc kết hợp thông tin tốt.

### Tóm tắt hướng trừu tượng

Mô hình sinh câu mới:

```text
bài viết → hiểu nội dung → sinh bản tóm tắt mới
```

Ưu điểm:

- Mạch lạc, súc tích.
- Có thể kết hợp nhiều phần của bài.

Nhược điểm:

- Có thể ảo giác tên, số, ngày tháng.
- Khó đánh giá tính đúng đắn chỉ bằng độ trùng từ.

Dự án này giải bài toán tóm tắt hướng trừu tượng.

## 5. Phát biểu bài toán dưới dạng xác suất

Cho bài viết:

```math
X=(x_1,x_2,\ldots,x_m)
```

và bản tóm tắt:

```math
Y=(y_1,y_2,\ldots,y_n)
```

Mô hình học:

```math
P(Y|X)=\prod_{t=1}^{n}P(y_t|y_{<t},X)
```

Nghĩa là xác suất cả summary được tách thành tích xác suất từng token, trong điều kiện biết bài viết và các token summary trước đó.

## 6. Ba yêu cầu của một bản tóm tắt tốt

### Độ bao phủ

Giữ được các ý quan trọng: ai, làm gì, ở đâu, khi nào, kết quả gì.

### Tính súc tích

Loại bỏ chi tiết phụ, tránh lặp và không biến summary thành phiên bản gần dài bằng bài viết.

### Tính trung thành

Không thêm thông tin nguồn không hỗ trợ. Đây là điểm khó nhất của tóm tắt hướng trừu tượng.

## 7. Encoder-decoder và causal LM nhìn bài toán khác nhau

### Encoder-decoder

```text
article → encoder → memory
previous summary tokens → decoder + cross-attention(memory) → next token
```

Nguồn và đích có hai luồng riêng. Đây là cấu trúc tự nhiên cho dịch máy và tóm tắt.

### Causal language model

```text
[instruction + article + summary] trong một chuỗi
```

Mô hình dự đoán token tiếp theo trên cùng một luồng. Khi train, loss ở phần prompt/article được mask để chỉ học phần summary.

## 8. Vấn đề độ dài ngữ cảnh

Bài viết dài hơn giới hạn token phải bị cắt:

```text
article tokens[:max_source_length]
```

Nếu thông tin quan trọng nằm sau phần bị cắt, mô hình không thể sử dụng nó. Phân tích dự án cho thấy nửa đầu bài viết chỉ bao phủ khoảng 73,51% unigram của summary validation, nên phần sau vẫn quan trọng.

Encoder-decoder có ngân sách nguồn và đích tách riêng. Causal LM phải chia sẻ ngữ cảnh cho prompt, article và summary, nên dễ mất nguồn hơn.

## 9. Sinh văn bản: greedy và beam search

### Greedy decoding

Mỗi bước chọn token xác suất cao nhất:

```text
y_t = argmax P(y_t | ...)
```

Nhanh nhưng có thể chọn một token tốt cục bộ và dẫn tới chuỗi tổng thể kém.

### Beam search

Giữ lại `k` chuỗi ứng viên tốt nhất ở mỗi bước. Beam search tìm kiếm rộng hơn nhưng chậm và tốn bộ nhớ hơn.

Trong dự án:

- Scratch dùng beam size 4 ở cấu hình cuối.
- ViT5 dùng beam size 4.
- Một số causal run dùng greedy hoặc beam 2 để tiết kiệm thời gian.

## 10. Length penalty, repetition penalty và no-repeat n-gram

- **Length penalty** điều chỉnh xu hướng ưu tiên chuỗi ngắn hoặc dài.
- **Repetition penalty** giảm xác suất token đã lặp quá nhiều.
- **No-repeat n-gram** cấm sinh lại một n-gram đã xuất hiện.

Các kỹ thuật này chỉ điều chỉnh giải mã, không làm mô hình “hiểu” tốt hơn. Cấu hình quá mạnh có thể cấm một cụm từ hợp lệ.

## 11. Hallucination

Hallucination là khi mô hình sinh thông tin không được nguồn hỗ trợ.

Ví dụ trong dự án:

```text
Nguồn: Miho Nakayama được phát hiện ngày 6/12
Qwen3 prediction: ngày 12/11
```

Prediction vẫn trôi chảy và đúng chủ đề nhưng sai số liệu. Vì vậy cần đọc mẫu dự đoán, không chỉ nhìn ROUGE.

## 12. Vì sao tiếng Việt có thách thức riêng?

- Dấu và chuẩn Unicode cần xử lý đúng.
- Từ có thể gồm nhiều âm tiết cách nhau bằng dấu cách.
- Tên riêng, địa danh và từ mượn xuất hiện nhiều.
- Tokenizer của mỗi mô hình có cách chia khác nhau.
- Dữ liệu báo chí chứa nhiều ngày tháng, số liệu và thực thể.

Hàm `clean_text` trong pipeline pretrained chuẩn hóa Unicode NFC, loại khoảng trắng đặc biệt và gom nhiều khoảng trắng thành một.

