# 08. Mô hình tiền huấn luyện sequence-to-sequence

## 1. Tiền huấn luyện là gì?

Một mô hình tiền huấn luyện đã học trên lượng văn bản lớn trước khi nhìn thấy dữ liệu của đồ án.

Quá trình tổng quát:

```text
corpus rất lớn
→ pretraining
→ checkpoint biết ngôn ngữ
→ fine-tuning trên article-summary
→ mô hình tóm tắt chuyên biệt
```

Fine-tuning không bắt đầu từ số 0. Mô hình đã có:

- Embedding tiếng Việt.
- Kiến thức cú pháp và ngữ nghĩa.
- Khả năng sinh câu trôi chảy.
- Biểu diễn nhiều cấu trúc ngôn ngữ.

Fine-tuning chủ yếu dạy mô hình cách áp dụng kiến thức đó vào bài toán tóm tắt và phân phối dữ liệu cụ thể.

## 2. Vì sao ViT5 là mô hình chính?

ViT5-base:

- Là Transformer encoder-decoder text-to-text.
- Được tiền huấn luyện cho tiếng Việt.
- Phù hợp trực tiếp với sinh chuỗi có điều kiện.
- Có khoảng 227,72 triệu tham số, dưới giới hạn 3 tỷ.
- Dùng kiến trúc T5: mọi bài toán được biểu diễn thành text-to-text.

Input trong dự án:

```text
summarize: <article>
```

Output:

```text
<summary>
```

## 3. ViT5 khác scratch ở đâu?

| Khía cạnh | Scratch | ViT5 |
|---|---|---|
| Khởi tạo | Ngẫu nhiên | Checkpoint tiền huấn luyện |
| Kiến thức tiếng Việt ban đầu | Không | Có |
| Tham số | 11,47M | 227,72M |
| Tokenizer | Tự train BPE 16K | Tokenizer của checkpoint |
| Source max | 512 | 768 |
| Target max | 128 | 160 |
| Mục tiêu fine-tune | Học từ đầu mọi thứ | Thích nghi kiến thức sẵn có |

## 4. Encoder-decoder pretrained vẫn hoạt động thế nào?

Kiến trúc cốt lõi vẫn là:

```text
source tokens
→ encoder
→ encoder hidden states
→ decoder cross-attention
→ target token probabilities
```

Điểm khác biệt lớn nhất không phải chỉ là “mô hình to hơn”, mà là trọng số encoder/decoder/embedding đã được học trước.

## 5. Full fine-tuning

Full fine-tuning cập nhật mọi tham số:

```text
227,72M tham số ViT5
→ tất cả requires_grad=True
→ tất cả nhận gradient
→ tất cả được AdamW cập nhật
```

Ưu điểm:

- Năng lực thích nghi cao nhất.
- Kết quả tốt nhất trong dự án.

Nhược điểm:

- Tốn VRAM và thời gian.
- Checkpoint lớn.
- Dễ làm hỏng kiến thức tiền huấn luyện nếu learning rate quá cao.
- Có thể overfit khi dữ liệu nhỏ.

## 6. Cấu hình ViT5 chính

File gốc: [`vit5_base.yaml`](../pretrained-summarization/configs/vit5_base.yaml)

Các giá trị quan trọng:

```yaml
model:
  name_or_path: VietAI/vit5-base

data:
  source_prefix: "summarize: "
  max_source_length: 768
  max_target_length: 160

training:
  learning_rate: 0.00003
  weight_decay: 0.01
  warmup_ratio: 0.06
  lr_scheduler_type: cosine
  label_smoothing_factor: 0.1
  dropout: 0.1
  gradient_checkpointing: true

generation:
  num_beams: 4
  length_penalty: 1.0
  no_repeat_ngram_size: 3
  repetition_penalty: 1.05
```

Lưu ý: config gốc có thể đặt số epoch lớn hơn, còn notebook chạy thực tế override xuống ba epoch. Khi xác định một run đã train bằng gì, `resolved_config.json` trong output run là nguồn đáng tin cậy nhất.

## 7. Data pipeline sequence-to-sequence

Trong [`data.py`](../pretrained-summarization/src/vn_summarization/data.py):

```python
inputs = [prefix + clean_text(text) for text in examples["article"]]
targets = [clean_text(text) for text in examples["summary"]]
```

Source được token hóa:

```python
model_inputs = tokenizer(
    inputs,
    max_length=max_source_length,
    truncation=True,
)
```

Target được token hóa riêng:

```python
labels = tokenizer(
    text_target=targets,
    max_length=max_target_length,
    truncation=True,
)
```

Kết quả mỗi sample:

```text
input_ids
attention_mask
labels
```

## 8. Data collator làm gì?

[`DataCollatorForSeq2Seq`](../pretrained-summarization/src/vn_summarization/train.py) tạo batch động:

- Pad source theo source dài nhất trong batch.
- Pad target theo target dài nhất trong batch.
- Đổi label padding thành `-100`.
- Có thể pad đến bội số 8 để FP16 hiệu quả hơn.
- Chuẩn bị decoder input phù hợp với model.

Dynamic padding tiết kiệm tính toán hơn pad mọi sample lên max length toàn cục.

## 9. Load model và tokenizer

Trong [`modeling.py`](../pretrained-summarization/src/vn_summarization/modeling.py):

```text
AutoConfig.from_pretrained()
→ load tokenizer
→ chỉnh dropout trong config
→ AutoModelForSeq2SeqLM.from_pretrained()
→ bật gradient checkpointing nếu cần
→ áp generation config
→ kiểm tra giới hạn tham số
```

### Vì sao có logic tokenizer riêng cho ViT5?

ViT5 dùng T5 SentencePiece tokenizer. Một số phiên bản Transformers/tokenizers có thể lỗi khi tự chuyển tokenizer cũ sang fast tokenizer. Code thử nạp trực tiếp `T5Tokenizer` từ `spiece.model`, rồi mới fallback sang `AutoTokenizer`.

Đây là xử lý tương thích môi trường, không phải thay đổi thuật toán tóm tắt.

## 10. Seq2SeqTrainer

[`train.py`](../pretrained-summarization/src/vn_summarization/train.py) sử dụng `Seq2SeqTrainer`.

Trainer quản lý:

- Vòng lặp train.
- Gradient accumulation.
- FP16/BF16.
- Logging.
- Evaluation theo step.
- Sinh prediction khi đánh giá.
- Lưu checkpoint.
- Load checkpoint tốt nhất.
- Early stopping callback.

`predict_with_generate=True` rất quan trọng: evaluation sinh summary thực tế để tính ROUGE, thay vì chỉ tính teacher-forced loss.

## 11. Chọn checkpoint bằng ROUGE-L

Config:

```yaml
load_best_model_at_end: true
metric_for_best_model: rougeL
greater_is_better: true
```

Điều này có nghĩa checkpoint cuối cùng không nhất thiết là checkpoint ở epoch cuối. Trainer chọn checkpoint có validation ROUGE-L cao nhất.

Đây là lựa chọn hợp lý vì mục tiêu cuối là chất lượng summary sinh, không phải train loss thấp nhất.

## 12. Early stopping

Early stopping dừng train nếu metric validation không cải thiện trong một số lần đánh giá.

Lợi ích:

- Giảm overfitting.
- Tiết kiệm thời gian.
- Tránh tiếp tục cập nhật làm hỏng checkpoint tốt.

## 13. BARTpho và ViT5 VietNews warm start

### BARTpho

BARTpho là encoder-decoder tiếng Việt được tiền huấn luyện theo mục tiêu denoising kiểu BART. Nó là đối chứng kiến trúc và pretraining khác ViT5.

Kết quả test ROUGE-L: `0.4807`, gần ViT5 nhưng thấp hơn `0.0082`.

### ViT5 VietNews warm start

Checkpoint này đã được thích nghi cho tóm tắt tin tức tiếng Việt trước đó.

Ưu:

- Có thể hội tụ nhanh.
- Đã biết phong cách tóm tắt.

Rủi ro:

- Domain hoặc cách viết summary cũ có thể không khớp dataset mới.
- Cần công bố rõ đây là warm-start checkpoint.

Kết quả test ROUGE-L `0.4728`, thấp hơn ViT5-base full fine-tuning. Task-specific warm start không tự động tốt nhất.

## 14. Vì sao pretrained tốt hơn?

### Biểu diễn từ vựng tốt

Tên, cụm từ và cấu trúc tiếng Việt không phải học hoàn toàn từ 10.775 mẫu.

### Biểu diễn ngữ nghĩa tốt

Mô hình đã học quan hệ giữa các câu và cách diễn đạt.

### Khả năng sinh ngôn ngữ tốt

Decoder đã biết tạo chuỗi trôi chảy trước fine-tuning.

### Context và target dài hơn

ViT5 đọc tối đa 768 source token và sinh tối đa 160 target token, giảm truncation so với scratch.

## 15. Rủi ro khi fine-tune pretrained

- Learning rate cao làm hỏng trọng số tốt đã có.
- Dữ liệu nhỏ gây overfitting.
- Tokenizer mismatch.
- Generation config không phù hợp làm summary quá dài/ngắn.
- Checkpoint lớn gây OOM.
- Chỉ nhìn loss mà không đọc prediction.

## 16. Một batch ViT5 đi qua hệ thống

```text
article strings
→ clean_text
→ thêm "summarize: "
→ tokenizer source
→ input_ids + attention_mask

summary strings
→ clean_text
→ tokenizer target
→ labels

DataCollatorForSeq2Seq
→ dynamic padding, label PAD=-100

ViT5 forward
→ encoder states
→ decoder logits
→ cross-entropy loss

backward
→ update toàn bộ tham số
```

