# 15. Chiến lược fine-tuning, metric, các mô hình và kết quả thực nghiệm

Chương này tổng hợp toàn bộ phần fine-tuning của dự án thành một tài liệu độc lập. Mục tiêu là trả lời chi tiết các câu hỏi:

- Ta đang fine-tune mô hình nào?
- Trọng số nào thực sự được cập nhật?
- Dữ liệu được đưa vào từng họ mô hình ra sao?
- Loss nào được tối ưu trong quá trình train?
- Metric nào dùng để chọn checkpoint và metric nào dùng để báo cáo?
- Mỗi mô hình đạt kết quả bao nhiêu?
- Tại sao kết quả giữa các mô hình lại khác nhau?
- Khi nào nên chọn full fine-tuning, LoRA, warm start hoặc causal LM?

## 1. Tổng quan bài toán fine-tuning

Dataset chứa các cặp:

```text
article → summary
```

Mục tiêu của fine-tuning là điều chỉnh một mô hình đã tiền huấn luyện để tăng xác suất sinh bản `summary` phù hợp khi nhận `article`.

Hai cách biểu diễn bài toán được dùng:

### Encoder-decoder sequence-to-sequence

Áp dụng cho ViT5, BARTpho và mT5:

```text
article
→ encoder
→ encoder hidden states
→ decoder + cross-attention
→ summary
```

Xác suất summary:

```math
P(Y|X)=\prod_{t=1}^{n}P(y_t|y_{<t},X)
```

### Causal language model

Áp dụng cho Qwen3 và DeepSeek-R1-Distill-Qwen:

```text
[instruction + article + summary]
→ một chuỗi tự hồi quy duy nhất
```

Trong train, phần instruction và article được mask khỏi loss; mô hình chỉ bị chấm điểm trên các token summary.

## 2. Fine-tuning thực sự thay đổi điều gì?

Một mô hình tiền huấn luyện đã có các ma trận trọng số học từ corpus lớn. Fine-tuning không dạy lại hoàn toàn ngôn ngữ từ đầu; nó điều chỉnh các trọng số để mô hình phù hợp hơn với:

- Dạng đầu vào là bài viết tiếng Việt dài.
- Dạng đầu ra là một đoạn tóm tắt khoảng 100 từ.
- Phong cách summary của dataset.
- Mức độ bám sát từ ngữ nguồn.
- Độ dài và cách tổ chức thông tin mong muốn.

Trong một training step:

```text
batch article-summary
→ tokenization
→ forward pass
→ logits
→ cross-entropy loss
→ backward
→ gradient
→ optimizer cập nhật tham số được phép train
```

Khác biệt giữa các phương pháp nằm chủ yếu ở:

- Kiến trúc model.
- Cách đóng gói source và target.
- Số lượng tham số được phép cập nhật.
- Độ dài ngữ cảnh.
- Ngân sách train.
- Metric chọn checkpoint.

## 3. Ba phương pháp fine-tuning chính

## 3.1. Full fine-tuning

Full fine-tuning cập nhật toàn bộ trọng số mô hình:

```text
base model parameters
→ requires_grad = True
→ nhận gradient
→ được AdamW cập nhật
```

Trong dự án, full fine-tuning được dùng cho:

- `VietAI/vit5-base`
- `vinai/bartpho-syllable`
- `VietAI/vit5-base-vietnews-summarization`

### Ưu điểm

- Khả năng thích nghi với dataset cao nhất.
- Mọi lớp encoder, decoder, attention, embedding và output head đều có thể thay đổi.
- Thường đạt chất lượng tốt nhất nếu có đủ GPU và cấu hình learning rate hợp lý.

### Nhược điểm

- Tốn VRAM vì phải lưu gradient và trạng thái optimizer cho toàn bộ model.
- Checkpoint lớn.
- Train chậm hơn LoRA.
- Learning rate quá cao có thể phá hỏng kiến thức tiền huấn luyện.
- Dataset nhỏ có thể gây overfitting.

## 3.2. LoRA fine-tuning

LoRA đóng băng trọng số gốc `W` và học cập nhật hạng thấp:

```math
W'=W+\frac{\alpha}{r}BA
```

Trong đó:

- `r` là LoRA rank.
- `A` và `B` là các ma trận nhỏ được train.
- `W` gốc không bị cập nhật.

Trong dự án, LoRA được dùng cho:

- ViT5-base.
- Qwen3-1.7B.
- DeepSeek-R1-Distill-Qwen-1.5B.

### ViT5 LoRA

LoRA được gắn vào các phép chiếu query và value:

```text
target modules = q, v
rank = 16
alpha = 32
dropout = 0.05
```

Số tham số trainable:

```text
1.769.472 / 227.720.448 ≈ 0,777%
```

### Causal-LM LoRA

LoRA được gắn rộng hơn:

```text
q_proj, k_proj, v_proj, o_proj,
gate_proj, up_proj, down_proj
```

Điều này cho phép thích nghi cả attention và feed-forward của causal LM mà không full fine-tune mô hình 1,5-1,7 tỷ tham số.

### Ưu điểm

- Giảm mạnh số tham số cần train.
- Giảm VRAM cho gradient và optimizer state.
- Adapter checkpoint nhỏ.
- Cho phép thử nghiệm model lớn trên GPU T4.
- Có thể giảm overfitting khi dữ liệu hạn chế.

### Nhược điểm

- Năng lực thích nghi thấp hơn full fine-tuning.
- Rank và target module cần chọn phù hợp.
- LoRA không tự giải quyết truncation, hallucination hoặc vấn đề kiến trúc.
- Khi inference vẫn cần nạp base model.

## 3.3. Warm-start fine-tuning

Warm start bắt đầu từ checkpoint đã được fine-tune cho một tác vụ gần giống, thay vì checkpoint pretraining tổng quát.

Trong dự án:

```text
VietAI/vit5-base-vietnews-summarization
```

Checkpoint này đã biết tóm tắt tin tức tiếng Việt trước khi được fine-tune trên dataset của đồ án.

### Giả thuyết ban đầu

Một checkpoint đã biết tóm tắt có thể:

- Hội tụ nhanh hơn.
- Sinh đúng dạng summary sớm hơn.
- Cần ít bước thích nghi hơn.

### Rủi ro

- Phong cách summary cũ có thể khác dataset mới.
- Domain cũ có thể tạo bias.
- Mô hình có thể bị “khóa” vào cách tóm tắt đã học trước.
- Warm start không bảo đảm tốt hơn checkpoint base.

Kết quả dự án xác nhận rủi ro này: VietNews warm start thấp hơn ViT5-base full fine-tuning.

## 4. Mục tiêu tối ưu trong quá trình train

## 4.1. Cross-entropy loss

Loss chính của mọi nhánh là token-level cross-entropy:

```math
\mathcal{L}
=
-\sum_{t=1}^{n}\log P(y_t|y_{<t},X)
```

Mô hình bị phạt khi gán xác suất thấp cho token summary đúng.

### Loss thấp có ý nghĩa gì?

Loss thấp cho thấy mô hình dự đoán tốt token reference trong điều kiện teacher forcing.

### Loss thấp không bảo đảm điều gì?

Loss thấp không bảo đảm:

- Summary sinh hoàn chỉnh có ROUGE cao.
- Không có hallucination.
- Độ dài phù hợp.
- Beam search tốt.
- Mô hình trung thành với số và tên riêng.

Do đó loss cần đi cùng generated metrics và phân tích định tính.

## 4.2. Teacher forcing trong seq2seq

Trong train:

```text
decoder input: [BOS, y1, y2, ..., y(n-1)]
labels:        [y1,  y2, y3, ..., EOS]
```

Decoder được nhận token đúng trước đó. Điều này giúp train ổn định và song song hóa, nhưng khác inference, khi mô hình phải dùng token tự sinh.

## 4.3. Label masking trong causal LM

Với causal LM:

```text
input:  [prompt + article][summary]
labels: [-100 ... -100  ][summary]
```

`-100` khiến cross-entropy bỏ qua phần prompt và article.

Mục đích:

- Không bắt model học chép lại prompt/article.
- Dồn tín hiệu gradient vào khả năng sinh summary.
- Làm loss phản ánh đúng tác vụ hơn.

Code liên quan:

- [`causal_data.py`](../src/vn_summarization/causal_data.py)
- [`train_causal_lm.py`](../src/vn_summarization/train_causal_lm.py)

## 4.4. Label smoothing trong seq2seq

Các nhánh sequence-to-sequence sử dụng label smoothing, thường là `0.1`.

Thay vì yêu cầu:

```text
token đúng = xác suất mục tiêu 1.0
token khác = 0.0
```

label smoothing phân phối một lượng xác suất nhỏ sang token khác.

Lợi ích:

- Giảm quá tự tin.
- Giảm overfitting.
- Có thể cải thiện generalization.

VietNews warm start dùng label smoothing thấp hơn `0.05` vì checkpoint đã được thích nghi cho tóm tắt trước đó.

## 5. Metric dùng trong dự án

Một điểm quan trọng là phải phân biệt:

1. **Training objective**: giá trị được đạo hàm để cập nhật trọng số.
2. **Checkpoint-selection metric**: giá trị dùng để chọn checkpoint tốt nhất.
3. **Reporting metric**: giá trị dùng để so sánh mô hình trong báo cáo.
4. **Diagnostic metric**: giá trị giúp phát hiện lỗi.

## 5.1. Training objective: loss

Loss được tính mỗi batch và dùng cho backward.

```text
loss → backward → gradient → optimizer.step()
```

Không thể trực tiếp backward qua ROUGE vì ROUGE dựa trên quá trình sinh rời rạc và không khả vi theo cách train thông thường.

## 5.2. ROUGE-1

ROUGE-1 đo độ trùng unigram giữa prediction và reference.

Nó phản ánh:

- Độ bao phủ từ khóa.
- Mức giữ nội dung cơ bản.

Nhưng nó không đo tốt:

- Thứ tự.
- Quan hệ giữa từ.
- Tính đúng của số hoặc thực thể.

## 5.3. ROUGE-2

ROUGE-2 đo độ trùng bigram.

Nó nghiêm ngặt hơn ROUGE-1 và phản ánh:

- Khả năng giữ cụm từ.
- Cấu trúc cục bộ.
- Mức độ gần với cách diễn đạt reference.

ViT5 tăng rất mạnh ROUGE-2 so với scratch, cho thấy tiền huấn luyện giúp bảo toàn cụm từ và cấu trúc summary.

## 5.4. ROUGE-L

ROUGE-L dựa trên dãy con chung dài nhất. Nó xét thứ tự token nhưng vẫn cho phép có khoảng cách.

Trong dự án, ROUGE-L là metric chính để:

- Chọn checkpoint seq2seq.
- So sánh chất lượng tổng thể.
- Xác định mô hình cuối.

Config sequence-to-sequence:

```yaml
load_best_model_at_end: true
metric_for_best_model: rougeL
greater_is_better: true
```

## 5.5. F1 và thang điểm

Các bảng báo cáo sử dụng ROUGE F1 trên thang `0-1`.

Code pretrained ban đầu xuất metric trên thang `0-100`:

```text
74.22 → báo cáo thành 0.7422
```

Khi so sánh run, cần kiểm tra thang điểm để tránh kết luận sai gấp 100 lần.

## 5.6. Generated length

`gen_len` đo độ dài đầu ra trung bình.

Metric này giúp phát hiện:

- Prediction luôn chạm `max_length`.
- Prediction quá ngắn.
- Model không học EOS.
- Decode config tạo output dài dòng.

ViT5 sinh khoảng 125 token trên validation, khá gần giới hạn 160 token, nên độ dài vẫn cần được theo dõi.

## 5.7. Repetition

Scratch theo dõi tỷ lệ 3-gram lặp. Các model pretrained sử dụng:

```text
no_repeat_ngram_size = 3
repetition_penalty ≈ 1.05-1.08
```

Metric lặp giúp phát hiện vòng lặp sinh, nhưng không phát hiện việc lặp cùng ý bằng cách diễn đạt khác.

## 5.8. Phân tích định tính và factuality

ROUGE không đủ để đánh giá:

- Tên riêng.
- Số liệu.
- Ngày tháng.
- Quan hệ sự kiện.
- Hallucination.

Dataset có `96,93%` số trong reference validation xuất hiện trong article. Vì vậy, sai số hoặc ngày là lỗi quan trọng.

Ví dụ Qwen:

```text
Nguồn: ngày 6/12
Prediction: ngày 12/11
```

Prediction vẫn đúng chủ đề nhưng sai factuality.

## 6. Metric chọn checkpoint theo từng họ mô hình

| Họ mô hình | Training objective | Metric chọn checkpoint | Metric báo cáo cuối |
|---|---|---|---|
| ViT5/BARTpho seq2seq | Cross-entropy + label smoothing | Validation ROUGE-L | ROUGE-1/2/L + phân tích định tính |
| ViT5 LoRA | Cross-entropy + label smoothing | Validation ROUGE-L | ROUGE-1/2/L + trainable params |
| Qwen/DeepSeek causal LM | Masked summary-token cross-entropy | `eval_loss` trong config chính | ROUGE-1/2/L trên tập con + lỗi factuality |

### Vì sao causal LM chọn `eval_loss` trong config?

Sinh summary toàn validation bằng model 1,5-1,7B chậm và tốn GPU. Chọn `eval_loss` giúp quá trình train/selection nhẹ hơn.

Tuy nhiên:

- `eval_loss` không phản ánh hoàn toàn chất lượng generation.
- Run causal cuối vẫn phải được đánh giá bằng ROUGE và prediction mẫu.
- Nếu có đủ tài nguyên, nên chọn checkpoint causal bằng generated validation ROUGE-L hoặc kết hợp loss với ROUGE.

## 7. Regularization và tối ưu hóa

## 7.1. AdamW

Mọi nhánh chính dùng AdamW.

AdamW:

- Dùng moment của gradient.
- Tự điều chỉnh mức cập nhật theo tham số.
- Tách weight decay khỏi gradient update.

## 7.2. Learning rate

Learning rate full fine-tuning thấp hơn LoRA:

| Phương pháp | Learning rate điển hình |
|---|---:|
| ViT5 full | `3e-5` |
| BARTpho full | `2e-5` |
| VietNews warm start | `1e-5` |
| ViT5 LoRA | `1e-4` |
| Qwen/DeepSeek LoRA | `1e-4` |

Lý do:

- Full fine-tuning thay đổi toàn bộ trọng số tốt đã học, nên cần bước nhỏ.
- LoRA chỉ học adapter mới, nên có thể dùng learning rate cao hơn.
- Warm start đã gần tác vụ, nên dùng learning rate thận trọng nhất.

## 7.3. Warmup và cosine decay

Các nhánh pretrained dùng:

```text
warmup → cosine learning-rate decay
```

Warmup giúp tránh cập nhật quá mạnh ở đầu train. Cosine decay giảm learning rate dần để tinh chỉnh ổn định về cuối.

## 7.4. Weight decay

Weight decay thường là `0.01`; VietNews warm start dùng `0.02`.

Mục tiêu:

- Hạn chế trọng số thay đổi quá cực đoan.
- Giảm overfitting.

## 7.5. Dropout

Dropout thường là `0.1`; VietNews warm start dùng `0.12`.

Dropout cao hơn ở warm start là một lựa chọn regularization để hạn chế checkpoint chuyên biệt cũ áp đặt quá mạnh lên dataset mới.

## 7.6. Early stopping

Các nhánh seq2seq sử dụng early stopping và chọn best checkpoint theo validation ROUGE-L.

Điều này tránh:

- Train quá lâu sau khi ROUGE dừng tăng.
- Overfit train.
- Dùng checkpoint cuối nhưng kém hơn checkpoint giữa quá trình.

## 7.7. Gradient checkpointing và FP16

Gradient checkpointing:

- Giảm VRAM.
- Đổi lại tăng thời gian tính toán.

FP16:

- Giảm bộ nhớ.
- Tăng tốc trên Tesla T4.

Đây là kỹ thuật vận hành, không trực tiếp làm model hiểu tốt hơn, nhưng cho phép train model lớn hơn hoặc context dài hơn.

## 8. Các mô hình được khảo sát

## 8.1. ViT5-base full fine-tuning

### Đặc điểm

- Kiến trúc: encoder-decoder T5.
- Ngôn ngữ: tiền huấn luyện chuyên biệt tiếng Việt.
- Quy mô: `227.720.448` tham số.
- Phương pháp: cập nhật toàn bộ tham số.
- Source prefix: `"summarize: "`.
- Source length: 768 token.
- Target length: 160 token.
- Decoding cuối: beam 4, length penalty 1.0, no-repeat 3-gram.

### Vì sao chọn làm mô hình chính?

- Kiến trúc phù hợp trực tiếp với mapping article → summary.
- Cross-attention giúp decoder truy vấn nguồn.
- Pretraining tiếng Việt phù hợp dataset.
- Quy mô đủ lớn nhưng vẫn dưới giới hạn 3 tỷ.

### Kết quả

| Split | ROUGE-1 | ROUGE-2 | ROUGE-L |
|---|---:|---:|---:|
| Full validation | **0.7417** | **0.4709** | **0.4924** |
| Full test | **0.7422** | **0.4675** | **0.4889** |

Đây là mô hình mạnh nhất đã hoàn thành.

## 8.2. ViT5-base LoRA rank 16

### Đặc điểm

- Cùng base model ViT5-base.
- Chỉ train adapter LoRA ở query/value projection.
- Trainable params: khoảng `1,77M`, tương đương `0,777%`.
- Rank 16, alpha 32, LoRA dropout 0.05.

### Kết quả

| Split | ROUGE-1 | ROUGE-2 | ROUGE-L |
|---|---:|---:|---:|
| Full validation | 0.7297 | 0.4484 | 0.4733 |
| Full test | 0.7262 | 0.4408 | 0.4663 |

### So với full fine-tuning

| Chỉ số | Chênh lệch LoRA so với full |
|---|---:|
| Validation ROUGE-L | `-0.0191` |
| Test ROUGE-L | `-0.0226` |

### Giải thích

LoRA giữ phần lớn năng lực ViT5 vì base model đã mạnh. Tuy nhiên, chỉ điều chỉnh một không gian hạng thấp ở một số module làm hạn chế khả năng thích nghi đầy đủ với phân phối summary mới.

Đổi lại, LoRA có hiệu quả tham số rất cao.

## 8.3. BARTpho-syllable full fine-tuning

### Đặc điểm

- Encoder-decoder tiếng Việt.
- Pretraining theo mục tiêu denoising kiểu BART.
- Không dùng source prefix.
- Learning rate `2e-5`.
- Source/target length 768/160.
- Full fine-tuning.

### Kết quả test

| ROUGE-1 | ROUGE-2 | ROUGE-L |
|---:|---:|---:|
| 0.7347 | 0.4617 | 0.4807 |

### Giải thích

BARTpho đạt gần ViT5 vì:

- Cũng là encoder-decoder tiếng Việt.
- Denoising pretraining giúp học cách tái tạo và tổ chức văn bản.
- Kiến trúc phù hợp với conditional generation.

BARTpho thấp hơn ViT5 `0.0082` ROUGE-L trên test. Khác biệt có thể đến từ:

- Mục tiêu pretraining khác.
- Tokenizer và biểu diễn âm tiết khác.
- Mức độ phù hợp giữa checkpoint và phong cách dataset.
- Cấu hình fine-tuning/decoding chưa tối ưu bằng ViT5.

## 8.4. ViT5 VietNews warm start

### Đặc điểm

- Bắt đầu từ checkpoint đã được fine-tune cho tóm tắt tin tức.
- Learning rate thấp `1e-5`.
- Weight decay `0.02`.
- Label smoothing `0.05`.
- Dropout `0.12`.
- Decode beam 6, length penalty 1.1.

### Kết quả test

| ROUGE-1 | ROUGE-2 | ROUGE-L |
|---:|---:|---:|
| 0.7161 | 0.4426 | 0.4728 |

### Vì sao warm start không thắng?

Một checkpoint đã biết tóm tắt không tự động phù hợp nhất. Các nguyên nhân hợp lý:

- Dataset mới có phong cách reference khác VietNews.
- Checkpoint cũ có bias về độ dài hoặc cách chọn ý.
- Fine-tuning trước đó làm mô hình kém linh hoạt hơn base checkpoint.
- Beam 6 và length penalty 1.1 có thể tạo output khác phân phối reference.
- Learning rate rất thấp có thể chưa thích nghi đủ.

Kết quả này cho thấy task-specific checkpoint phải được kiểm chứng, không nên mặc định là lựa chọn tốt nhất.

## 8.5. Qwen3-1.7B LoRA

### Đặc điểm

- Causal language model.
- Instruction + article + summary trong cùng một chuỗi.
- LoRA trên attention và MLP projections.
- Context 768 hoặc 1024 token.
- Target 128 token.
- Budget train dùng subset để phù hợp thời gian/GPU.

### Kết quả ablation trên tập con validation 500 mẫu

Các kết quả dưới đây không được so trực tiếp tuyệt đối với full-validation ViT5 vì khác số mẫu đánh giá.

| Thiết lập | Context | Rank | Train samples | ROUGE-1 | ROUGE-2 | ROUGE-L |
|---|---:|---:|---:|---:|---:|---:|
| Prompt cơ bản | 768 | 8 | 2.500 | 0.5938 | 0.2729 | 0.3379 |
| Prompt nghiêm ngặt | 768 | 8 | 2.500 | 0.6044 | 0.2682 | 0.3333 |
| Prompt nghiêm ngặt | 768 | 16 | 2.500 | 0.6565 | 0.2987 | 0.3611 |
| Prompt nghiêm ngặt | 1024 | 8 | 2.500 | 0.6652 | 0.3122 | **0.3792** |
| Thêm dữ liệu, beam 2 | 1024 | 16 | 7.000 | **0.6834** | **0.3213** | 0.3787 |

### Giải thích từng ablation

#### Prompt nghiêm ngặt

Prompt nghiêm ngặt tăng ROUGE-1 nhưng giảm nhẹ ROUGE-2/ROUGE-L trong thiết lập rank 8, context 768.

Điều này cho thấy:

- Chỉ dẫn có thể giúp model giữ nhiều từ khóa hơn.
- Prompt không thay thế được context hoặc năng lực thích nghi.
- Câu chữ prompt chặt hơn không bảo đảm chuỗi tổng thể gần reference hơn.

#### Tăng LoRA rank từ 8 lên 16

ROUGE tăng rõ rệt:

```text
RL: 0.3333 → 0.3611
```

Rank cao hơn cho adapter nhiều năng lực biểu diễn hơn. Tuy nhiên, rank cao cũng tăng tham số trainable và chi phí.

#### Tăng context từ 768 lên 1024

Đây là cải thiện ROUGE-L lớn nhất trong các run 300 step tương đồng:

```text
RL: 0.3333 → 0.3792
```

Nguyên nhân phù hợp với phân tích dataset:

- Một phần tư đầu article chỉ bao phủ `51,31%` unigram summary validation.
- Nửa đầu article chỉ bao phủ `73,51%`.
- Thông tin phía sau vẫn quan trọng.
- Causal LM còn mất context cho instruction/prompt.

#### Thêm dữ liệu và beam 2

Run 7.000 mẫu, rank 16, beam 2 đạt ROUGE-1/2 cao nhất nhưng ROUGE-L không vượt run context 1024 rank 8.

Điều này cho thấy:

- Thêm dữ liệu giúp giữ nhiều từ và cụm từ.
- Nhiều tham số hơn không tự động cải thiện cấu trúc chuỗi dài nhất.
- Decode và cách tổ chức summary vẫn là nút thắt.

## 8.6. DeepSeek-R1-Distill-Qwen-1.5B LoRA

### Đặc điểm

- Causal LM distilled từ họ reasoning model.
- LoRA rank 8.
- Context 768.
- Train trên 2.500 mẫu trong thí nghiệm kiểm soát.

### Kết quả tập con validation

| ROUGE-1 | ROUGE-2 | ROUGE-L |
|---:|---:|---:|
| 0.4481 | 0.1695 | 0.2619 |

### Vì sao thấp hơn Qwen3?

Các nguyên nhân hợp lý:

- Checkpoint distilled reasoning không nhất thiết phù hợp với tóm tắt ngắn bám nguồn.
- Model có thể thiên về sinh giải thích hoặc chuỗi suy luận.
- Context và budget train hạn chế.
- Prompt phải ngăn `<think>` và instruction leakage.
- Tiền huấn luyện/distillation objective có thể không phù hợp bằng Qwen3 cho tác vụ này.

Tên “reasoning model” không đồng nghĩa sẽ tốt hơn ở mọi tác vụ generation.

## 8.7. mT5-base

Repo có config cho `google/mt5-base`, nhưng không có kết quả cuối được đưa vào bảng báo cáo chính.

mT5 là baseline đa ngôn ngữ hợp lý, nhưng có thể:

- Dành năng lực cho nhiều ngôn ngữ thay vì riêng tiếng Việt.
- Nặng/chậm hơn so với lợi ích trên dataset tiếng Việt.
- Không cạnh tranh bằng ViT5 trong ngân sách hiện tại.

Không nên tự gán kết quả cho mT5 khi chưa có run hoàn chỉnh được xuất.

## 9. Bảng tổng hợp kết quả

## 9.1. Kết quả full validation có thể so sánh trực tiếp

| Mô hình | Phương pháp | ROUGE-1 | ROUGE-2 | ROUGE-L |
|---|---|---:|---:|---:|
| Transformer scratch | Train từ đầu | 0.5768 | 0.2087 | 0.3053 |
| ViT5-base | Full fine-tuning | **0.7417** | **0.4709** | **0.4924** |
| ViT5-base | LoRA rank 16 | 0.7297 | 0.4484 | 0.4733 |

## 9.2. Kết quả full test có thể so sánh trực tiếp

| Mô hình | Phương pháp | ROUGE-1 | ROUGE-2 | ROUGE-L |
|---|---|---:|---:|---:|
| Transformer scratch | Train từ đầu | 0.5747 | 0.2038 | 0.3045 |
| ViT5-base | Full fine-tuning | **0.7422** | **0.4675** | **0.4889** |
| ViT5-base | LoRA rank 16 | 0.7262 | 0.4408 | 0.4663 |
| BARTpho-syllable | Full fine-tuning | 0.7347 | 0.4617 | 0.4807 |
| ViT5 VietNews | Warm-start full fine-tuning | 0.7161 | 0.4426 | 0.4728 |

## 9.3. Causal-LM subset results

| Mô hình/run | Phương pháp | ROUGE-1 | ROUGE-2 | ROUGE-L |
|---|---|---:|---:|---:|
| Qwen3 context 1024, rank 8 | LoRA | 0.6652 | 0.3122 | **0.3792** |
| Qwen3 thêm dữ liệu, rank 16, beam 2 | LoRA | **0.6834** | **0.3213** | 0.3787 |
| DeepSeek-R1-Distill-Qwen-1.5B | LoRA | 0.4481 | 0.1695 | 0.2619 |

Các dòng này chỉ dùng để so sánh trong nhóm thí nghiệm subset tương ứng.

## 10. Vì sao ViT5 full fine-tuning đạt tốt nhất?

## 10.1. Kiến trúc phù hợp với tác vụ

Tóm tắt là bài toán conditional generation:

```text
nguồn dài → đích ngắn
```

Encoder-decoder tách rõ:

- Encoder đọc nguồn.
- Decoder sinh đích.
- Cross-attention truy vấn nguồn ở mỗi bước sinh.

Đây là inductive bias phù hợp hơn causal LM một luồng.

## 10.2. Pretraining chuyên biệt tiếng Việt

ViT5 đã học:

- Từ vựng tiếng Việt.
- Cấu trúc câu.
- Quan hệ ngữ nghĩa.
- Cách sinh văn bản.

Scratch phải học các khả năng này chỉ từ 10.775 mẫu.

## 10.3. Dataset bám nguồn mạnh

Trên validation:

- `92,51%` unigram reference xuất hiện trong article.
- `68,99%` bigram reference xuất hiện trong article.

Cross-attention của encoder-decoder rất phù hợp với việc lựa chọn và tổ chức lại cụm từ nguồn. Đây cũng là đặc trưng được ROUGE thưởng.

## 10.4. Ngân sách context tách biệt

ViT5 có:

```text
encoder budget: 768 source token
decoder budget: 160 target token
```

Causal LM phải dùng chung không gian cho prompt và article trước khi sinh target.

## 10.5. Full adaptation

Full fine-tuning cho phép mọi lớp ViT5 thích nghi với dataset, trong khi LoRA chỉ học cập nhật hạng thấp ở một số module.

## 10.6. Ngân sách train lớn hơn causal experiments

ViT5 full dùng toàn bộ 10.775 mẫu trong khoảng ba epoch thực tế.

Các causal ablation chủ yếu dùng:

- 2.500 hoặc 7.000 mẫu.
- 300 hoặc 700 step.
- Tập validation con.

Do đó, điểm causal thấp hơn phản ánh cả kiến trúc lẫn budget, không chỉ năng lực model.

## 11. Vì sao ROUGE-2 thể hiện khác biệt lớn?

ROUGE-2 yêu cầu giữ đúng cặp token liên tiếp. Nó nhạy với:

- Cụm từ quan trọng.
- Cách tổ chức câu.
- Mức bám sát reference.

ViT5 test ROUGE-2:

```text
0.4675
```

Scratch test ROUGE-2:

```text
0.2038
```

ViT5 hơn hơn hai lần, cho thấy:

- Không chỉ tìm đúng từ khóa.
- Mô hình giữ được cụm nội dung và cấu trúc cục bộ tốt hơn.
- Pretraining giúp sinh câu gần phong cách reference hơn.

## 12. Vì sao validation và test gần nhau?

| Mô hình | Validation RL | Test RL | Chênh lệch |
|---|---:|---:|---:|
| Scratch | 0.3053 | 0.3045 | -0.0008 |
| ViT5 full | 0.4924 | 0.4889 | -0.0035 |
| ViT5 LoRA | 0.4733 | 0.4663 | -0.0070 |

Điều này gợi ý:

- Validation đại diện tốt cho test.
- Model selection theo validation hợp lý.
- Không có dấu hiệu distribution shift lớn.

Điều này **không** có nghĩa được phép tune trên test.

## 13. Giới hạn của kết quả

## 13.1. Budget không hoàn toàn công bằng

ViT5 và causal LM không dùng cùng:

- Số sample train.
- Số step.
- Target length.
- Validation size.
- Chi phí tính toán.

Vì vậy không nên kết luận causal architecture tự thân luôn kém.

## 13.2. ROUGE chưa đo factuality đầy đủ

ROUGE cao không bảo đảm:

- Số đúng.
- Tên đúng.
- Ngày đúng.
- Không hallucination.

## 13.3. Decode config ảnh hưởng điểm

Beam size, length penalty và max length có thể thay đổi ROUGE mà không thay model weights.

## 13.4. Một run không thể hiện độ biến động

Nếu chỉ chạy một seed, chưa biết mức ổn định của mô hình. Lý tưởng nên chạy nhiều seed hoặc ít nhất ghi rõ hạn chế này.

## 14. Cách chọn phương pháp theo mục tiêu

| Mục tiêu | Phương pháp nên ưu tiên | Lý do |
|---|---|---|
| Chất lượng cao nhất | ViT5 full fine-tuning | Kết quả ROUGE tốt nhất |
| Giảm VRAM/checkpoint | ViT5 LoRA | Chỉ train 0,777% tham số |
| Mô hình seq2seq thay thế | BARTpho full | Gần ViT5, tiếng Việt tốt |
| Tận dụng checkpoint tóm tắt cũ | VietNews warm start | Có thể hội tụ nhanh nhưng phải kiểm chứng |
| Instruction/prompt linh hoạt | Qwen3 LoRA | Causal/instruction model linh hoạt |
| Học kiến trúc từ gốc | Scratch Transformer | Minh bạch và kiểm soát toàn bộ |

## 15. Phương pháp đánh giá fine-tuning nên dùng trong tương lai

Một quy trình mạnh hơn:

### Bước 1: Chọn checkpoint bằng validation

Đối với mọi model, ưu tiên generated validation ROUGE-L thay vì chỉ eval loss nếu đủ tài nguyên.

### Bước 2: Kiểm tra metric bổ sung

- ROUGE-1/2/L.
- Generated length.
- Repetition.
- Number support.
- Entity correctness.
- Date correctness.

### Bước 3: Đánh giá định tính cố định

Chọn một bộ mẫu validation cố định gồm:

- Article dài.
- Article nhiều số.
- Article nhiều thực thể.
- Summary trừu tượng cao.
- Summary dài.

### Bước 4: So sánh chi phí

Ghi:

- Trainable parameters.
- Peak VRAM.
- Train runtime.
- Generation speed.
- Checkpoint size.

### Bước 5: Chỉ dùng test một lần

Sau khi chốt model và decode config bằng validation, chạy test cuối.

## 16. Checklist đọc một kết quả fine-tuning

Trước khi tin một con số, hỏi:

1. Model base là gì?
2. Full fine-tuning hay LoRA?
3. Có warm start không?
4. Train bao nhiêu sample và step?
5. Validation là full split hay subset?
6. Source/target length bao nhiêu?
7. Checkpoint được chọn bằng loss hay ROUGE?
8. Decode config là gì?
9. ROUGE ở thang 0-1 hay 0-100?
10. Prediction có hallucination không?
11. So sánh có công bằng về budget không?
12. Test có được giữ độc lập không?

## 17. Liên kết code và config

### Pipeline sequence-to-sequence

- [`train.py`](../src/vn_summarization/train.py)
- [`data.py`](../src/vn_summarization/data.py)
- [`modeling.py`](../src/vn_summarization/modeling.py)
- [`metrics.py`](../src/vn_summarization/metrics.py)
- [`evaluate.py`](../src/vn_summarization/evaluate.py)

### Pipeline causal LM

- [`causal_data.py`](../src/vn_summarization/causal_data.py)
- [`train_causal_lm.py`](../src/vn_summarization/train_causal_lm.py)
- [`evaluate_causal_lm.py`](../src/vn_summarization/evaluate_causal_lm.py)

### Config chính

- [`vit5_base.yaml`](../configs/vit5_base.yaml)
- [`vit5_base_lora.yaml`](../configs/vit5_base_lora.yaml)
- [`bartpho_syllable.yaml`](../configs/bartpho_syllable.yaml)
- [`vit5_news_warmstart.yaml`](../configs/vit5_news_warmstart.yaml)
- [`qwen3_1_7b_lora.yaml`](../configs/qwen3_1_7b_lora.yaml)
- [`deepseek_r1_distill_qwen_1_5b_lora.yaml`](../configs/deepseek_r1_distill_qwen_1_5b_lora.yaml)

## 18. Kết luận

Kết quả dự án cho thấy kiến trúc encoder-decoder tiền huấn luyện chuyên biệt cho tiếng Việt là lựa chọn phù hợp nhất cho dataset này. ViT5-base full fine-tuning đạt kết quả cao nhất vì kết hợp được:

- Kiến thức tiếng Việt từ pretraining.
- Kiến trúc source-target có cross-attention.
- Khả năng thích nghi toàn bộ tham số.
- Context và target budget phù hợp.
- Chọn checkpoint bằng validation ROUGE-L.

ViT5 LoRA cho thấy sự đánh đổi hiệu quả giữa chi phí và chất lượng. BARTpho xác nhận rằng pretrained Vietnamese seq2seq nói chung rất mạnh. VietNews warm start chứng minh checkpoint gần tác vụ không bảo đảm tốt nhất. Các thí nghiệm Qwen3 cho thấy context nguồn là nút thắt quan trọng của causal LM, còn DeepSeek distilled reasoning không phù hợp bằng Qwen3 trong budget hiện tại.

Khi đánh giá fine-tuning, không nên chỉ nhìn loss hoặc một con số ROUGE. Quyết định đúng phải kết hợp chất lượng sinh, factuality, độ dài, chi phí train, phạm vi dữ liệu và tính công bằng của thí nghiệm.

