# 12. Kết quả và cách diễn giải

## 1. Kết quả chính

### Full validation

| Mô hình | ROUGE-1 | ROUGE-2 | ROUGE-L |
|---|---:|---:|---:|
| Transformer scratch | 0.5768 | 0.2087 | 0.3053 |
| ViT5-base full fine-tuning | **0.7417** | **0.4709** | **0.4924** |
| ViT5-base LoRA rank 16 | 0.7297 | 0.4484 | 0.4733 |

### Full test

| Mô hình | ROUGE-1 | ROUGE-2 | ROUGE-L |
|---|---:|---:|---:|
| Transformer scratch | 0.5747 | 0.2038 | 0.3045 |
| ViT5-base full fine-tuning | **0.7422** | **0.4675** | **0.4889** |
| ViT5-base LoRA rank 16 | 0.7262 | 0.4408 | 0.4663 |
| BARTpho-syllable full fine-tuning | 0.7347 | 0.4617 | 0.4807 |
| ViT5 VietNews warm start | 0.7161 | 0.4426 | 0.4728 |

## 2. Kết luận trực tiếp

ViT5-base full fine-tuning là mô hình mạnh nhất đã hoàn thành.

Trên test, so với scratch:

- ROUGE-1 tăng `0.1675`.
- ROUGE-2 tăng `0.2637`.
- ROUGE-L tăng `0.1844`.

Mức tăng ROUGE-2 rất lớn cho thấy ViT5 bảo toàn các cụm từ và cấu trúc nội dung tốt hơn rõ rệt.

## 3. Vì sao validation và test gần nhau?

| Mô hình | Validation RL | Test RL | Chênh lệch |
|---|---:|---:|---:|
| Scratch | 0.3053 | 0.3045 | -0.0008 |
| ViT5 | 0.4924 | 0.4889 | -0.0035 |

Sự gần nhau cho thấy:

- Validation đại diện khá tốt cho test.
- Quy trình chọn mô hình bằng validation ổn định.
- Không có dấu hiệu dịch chuyển phân phối lớn.

Không cần và không nên phân tích mô tả nội dung test để đưa ra kết luận này.

## 4. Scratch có thất bại không?

Không. Scratch đạt mục tiêu khác:

- Chứng minh encoder-decoder Transformer tự cài đặt có thể học bài toán.
- Cho phép hiểu và kiểm soát attention, mask, residual, Pre-LN và decoding.
- Dùng mô hình nhỏ 11,47 triệu tham số.

Việc scratch thấp hơn pretrained là kết quả được kỳ vọng, không phải bằng chứng cài đặt vô dụng.

## 5. Full fine-tuning và LoRA

ViT5 LoRA:

- Chỉ train 0,777% tham số.
- Validation RL thấp hơn full fine-tuning `0.0191`.
- Test RL thấp hơn `0.0226`.

Diễn giải:

- LoRA đạt sự đánh đổi hiệu quả tốt.
- Full fine-tuning vẫn có lợi khi GPU và thời gian cho phép.
- Không nên nói LoRA “tốt hơn” chỉ vì train ít tham số; cần nêu rõ mục tiêu là chi phí hay chất lượng.

## 6. BARTpho gần ViT5

BARTpho test RL `0.4807`, chỉ thấp hơn ViT5 `0.0082`.

Điều này cho thấy:

- Pretraining tiếng Việt encoder-decoder nói chung rất hiệu quả.
- ViT5 không phải lựa chọn khả thi duy nhất.
- Khác biệt nhỏ có thể cần thêm phân tích định tính, thời gian train và độ ổn định.

## 7. Vì sao VietNews warm start không thắng?

Checkpoint đã train tóm tắt tin tức không tự động phù hợp nhất vì:

- Phong cách reference mới có thể khác.
- Domain hoặc phân phối độ dài khác.
- Fine-tuning trước đó có thể tạo bias.
- Base ViT5 linh hoạt hơn khi thích nghi toàn bộ trên dataset mới.

Đây là bài học quan trọng: checkpoint gần tác vụ chỉ là giả thuyết tốt, không phải bảo đảm.

## 8. Qwen ablation nói lên điều gì?

Kết quả tập con cho thấy context 1024 cải thiện RL rõ nhất so với context 768 trong thiết lập rank 8.

Điều này phù hợp với phân tích dữ liệu:

- 25% đầu article chỉ bao phủ 51,31% unigram summary.
- Nội dung phía sau quan trọng.
- Causal LM mất thêm context cho prompt.

Tăng rank và dữ liệu giúp R1/R2 nhưng không tiếp tục tăng RL, cho thấy vấn đề không chỉ nằm ở số tham số LoRA.

## 9. Không nên diễn giải quá mức

Các kết luận **không** được hỗ trợ:

- “Mọi causal LM đều kém seq2seq.”
- “LoRA luôn kém full fine-tuning.”
- “ViT5 luôn tốt nhất cho mọi dataset tiếng Việt.”
- “ROUGE cao nghĩa là không ảo giác.”
- “Test gần validation nghĩa là có thể tune trên test.”

Kết luận đúng phải gắn với dataset, budget, model và cấu hình cụ thể.

## 10. Cách chọn mô hình cho mục tiêu khác nhau

### Tối đa chất lượng

Chọn ViT5-base full fine-tuning.

### Giảm chi phí train/checkpoint

Chọn ViT5 LoRA, chấp nhận giảm chất lượng nhỏ.

### Học kiến trúc và kiểm soát toàn bộ

Chọn Transformer scratch.

### Cần instruction-following hoặc mở rộng nhiều tác vụ

Khảo sát Qwen LoRA, nhưng cần tăng context và kiểm tra ảo giác.

### Muốn mô hình seq2seq tiếng Việt thay thế

BARTpho là lựa chọn cạnh tranh.

## 11. Một bảng quyết định thực tế

| Câu hỏi | Nếu “có” | Hướng phù hợp |
|---|---|---|
| Chất lượng ROUGE là ưu tiên số một? | Có | ViT5 full |
| GPU hạn chế? | Có | ViT5 LoRA |
| Cần hiểu Transformer từ gốc? | Có | Scratch |
| Cần prompt/instruction linh hoạt? | Có | Causal LM LoRA |
| Cần giảm ảo giác số liệu? | Có | Seq2seq + kiểm tra hậu xử lý |

## 12. Cách viết kết luận khoa học tốt

Một kết luận tốt gồm:

1. Quan sát định lượng.
2. Giải thích có cơ sở.
3. Giới hạn của thí nghiệm.
4. Hướng kiểm chứng tiếp theo.

Ví dụ:

> Tăng context Qwen từ 768 lên 1024 token làm ROUGE-L tập con validation tăng từ 0.3333 lên 0.3792. Kết quả phù hợp với phân tích độ phủ cho thấy nội dung sau nửa đầu bài viết vẫn đóng góp đáng kể vào summary. Tuy nhiên, thí nghiệm dùng tập con và budget train hạn chế, nên chưa đủ để kết luận về mọi causal LM.

## 13. Hướng cải thiện tiếp theo có giá trị

- Đo tỷ lệ truncation theo **tokenizer thực tế**, không chỉ theo từ.
- Đánh giá entity/number/date faithfulness.
- Chấm thủ công một tập prediction cố định.
- Tăng context causal LM và giữ budget train công bằng hơn.
- Thử decode tuning trên validation.
- Bổ sung repo scratch và test đơn vị cho mask/attention.
- So sánh tốc độ, VRAM và kích thước checkpoint.

