# 09. LoRA và causal language model

## 1. Tại sao cần LoRA?

Full fine-tuning cập nhật toàn bộ mô hình. Với model lớn, điều này tốn:

- VRAM cho tham số.
- VRAM cho gradient.
- VRAM cho trạng thái optimizer.
- Dung lượng checkpoint.

LoRA giữ nguyên trọng số gốc và chỉ học các ma trận cập nhật hạng thấp.

## 2. Công thức LoRA

Với ma trận gốc:

```math
W \in \mathbb{R}^{d_{out} \times d_{in}}
```

LoRA thay cập nhật đầy đủ `ΔW` bằng:

```math
W' = W + \frac{\alpha}{r}BA
```

Trong đó:

```math
A \in \mathbb{R}^{r \times d_{in}}
```

```math
B \in \mathbb{R}^{d_{out} \times r}
```

và `r` nhỏ hơn nhiều so với `d_in`, `d_out`.

## 3. Trực giác của rank

LoRA giả định thay đổi cần thiết cho tác vụ mới nằm trong một không gian chiều thấp.

- Rank nhỏ: ít tham số, nhanh, ít overfit nhưng năng lực thích nghi hạn chế.
- Rank lớn: nhiều tham số, linh hoạt hơn nhưng tốn bộ nhớ và có thể overfit hơn.

Trong ViT5 LoRA:

```text
r = 16
alpha = 32
dropout = 0.05
target modules = q, v
```

Chỉ khoảng 1,77 triệu tham số, tương đương 0,777% mô hình, được train.

## 4. LoRA target modules

Trong [`modeling.py`](../pretrained-summarization/src/vn_summarization/modeling.py):

- T5/mT5: LoRA áp vào `q` và `v`.
- BART/mBART: LoRA áp vào `q_proj` và `v_proj`.

Ý nghĩa: LoRA điều chỉnh cách attention truy vấn và lấy nội dung mà không thay toàn bộ mô hình.

Causal LM áp LoRA rộng hơn:

```text
q_proj, k_proj, v_proj, o_proj,
gate_proj, up_proj, down_proj
```

Ngoài attention, nó còn thích nghi các phép chiếu trong feed-forward/gated MLP.

## 5. Full fine-tuning so với LoRA

| Khía cạnh | Full fine-tuning | LoRA |
|---|---|---|
| Tham số cập nhật | Toàn bộ | Adapter hạng thấp |
| VRAM | Cao | Thấp hơn |
| Checkpoint | Lớn | Nhỏ |
| Khả năng thích nghi | Cao | Hạn chế hơn |
| Nguy cơ overfit | Có thể cao | Thường thấp hơn |
| Kết quả dự án | Tốt nhất | Thấp hơn một ít |

Kết quả ViT5:

| Phương pháp | Validation RL | Test RL |
|---|---:|---:|
| Full fine-tuning | 0.4924 | 0.4889 |
| LoRA rank 16 | 0.4733 | 0.4663 |

LoRA tiết kiệm đáng kể nhưng mất khoảng `0.0226` test ROUGE-L.

## 6. Causal language model là gì?

Causal LM dự đoán token tiếp theo từ các token đứng trước:

```math
P(x_1,\ldots,x_T)=\prod_tP(x_t|x_{<t})
```

Qwen và DeepSeek distilled Qwen thuộc nhóm này.

Khác encoder-decoder:

```text
encoder-decoder:
article → encoder
summary → decoder

causal LM:
[instruction + article + summary] → một luồng tự hồi quy
```

## 7. Biến tóm tắt thành bài toán causal LM

Prompt:

```text
Bạn là trợ lý tóm tắt tiếng Việt...

Văn bản:
{article}

Tóm tắt:
```

Khi train, nối thêm summary đúng:

```text
[prompt + article + "Tóm tắt:"] + [reference summary + EOS]
```

Mô hình về lý thuyết có thể học dự đoán mọi token. Nhưng ta chỉ muốn loss ở summary.

## 8. Label masking bằng `-100`

Trong [`causal_data.py`](../pretrained-summarization/src/vn_summarization/causal_data.py):

```python
input_ids = prompt_ids + target_ids
labels = [-100] * len(prompt_ids) + target_ids
```

Cross-entropy trong Transformers bỏ qua label `-100`.

Kết quả:

```text
prompt/article token: không tính loss
summary token: có tính loss
```

Nếu không mask prompt:

- Mô hình tốn năng lực học chép lại instruction và article.
- Loss không phản ánh riêng khả năng sinh summary.
- Tác vụ train lệch mục tiêu.

## 9. CausalDataCollator

[`CausalDataCollator`](../pretrained-summarization/src/vn_summarization/train_causal_lm.py) pad `input_ids` bằng pad token và pad `labels` bằng `-100`.

Tại sao label padding phải là `-100`?

Nếu label padding là pad token ID bình thường, mô hình sẽ bị thưởng/phạt vì dự đoán token padding ở cuối chuỗi, làm méo loss.

## 10. Train Qwen với LoRA

Quy trình:

```text
AutoTokenizer
→ đặt pad_token = eos_token nếu thiếu
→ AutoModelForCausalLM
→ tắt use_cache khi train
→ bật gradient checkpointing
→ gắn LoRA adapter
→ prompt + target tokenization
→ Trainer
→ lưu adapter
```

Config chính: [`qwen3_1_7b_lora.yaml`](../pretrained-summarization/configs/qwen3_1_7b_lora.yaml)

Giá trị quan trọng:

```yaml
data:
  max_source_length: 768
  max_target_length: 128
  max_length: 896
  max_train_samples: 2500
  max_eval_samples: 120

training:
  max_steps: 300
  learning_rate: 0.0001
  precision: fp16

lora:
  r: 8
  lora_alpha: 16
```

Các notebook thí nghiệm override context, rank, số mẫu và số step để tạo ablation.

## 11. Train và inference khác nhau ở causal LM

### Train

Input chứa cả summary đúng, nhưng label phần prompt bị mask:

```text
input:  [prompt][reference summary]
label:  [-100 ][reference summary]
```

### Inference

Input chỉ có prompt/article:

```text
input: [prompt]
model.generate() → summary continuation
```

Sau sinh, code chỉ lấy token sau độ dài input:

```python
gen_ids = output_ids[input_len:]
```

## 12. Vì sao causal LM thấp hơn seq2seq trong dự án?

### Kiến trúc không chuyên biệt cho mapping nguồn-đích

Encoder-decoder có cross-attention riêng để truy vấn article. Causal LM đặt tất cả trong một chuỗi.

### Chia sẻ ngân sách context

Causal LM phải chứa instruction và article trước khi sinh. Prompt và subword chiếm chỗ, làm giảm phần article thực tế được nhìn.

### Target ngắn hơn

Qwen dùng 128 target token, trong khi khoảng 15,05% summary validation đã dài hơn 128 **từ**.

### Ngân sách train thấp hơn

ViT5 dùng toàn bộ 10.775 mẫu trong ba epoch. Nhiều causal run chỉ dùng 2.500 hoặc 7.000 mẫu và 300/700 step.

### ROUGE ưu tiên độ trùng từ

Causal LM có thể diễn đạt lại trôi chảy nhưng dùng từ khác reference, dẫn đến ROUGE thấp hơn.

### Rủi ro ảo giác

Qwen có ví dụ đổi ngày 6/12 thành 12/11 dù đúng chủ đề.

## 13. Kết quả ablation Qwen

Trên tập con validation cố định 500 mẫu:

| Thiết lập | Context | Rank | Train samples | R1 | R2 | RL |
|---|---:|---:|---:|---:|---:|---:|
| Prompt cơ bản | 768 | 8 | 2.500 | 0.5938 | 0.2729 | 0.3379 |
| Prompt nghiêm ngặt | 768 | 8 | 2.500 | 0.6044 | 0.2682 | 0.3333 |
| Prompt nghiêm ngặt | 768 | 16 | 2.500 | 0.6565 | 0.2987 | 0.3611 |
| Prompt nghiêm ngặt | 1024 | 8 | 2.500 | 0.6652 | 0.3122 | **0.3792** |
| Thêm dữ liệu, beam 2 | 1024 | 16 | 7.000 | **0.6834** | **0.3213** | 0.3787 |

Kết luận:

- Tăng context tạo cải thiện ROUGE-L rõ nhất.
- Tăng rank và dữ liệu giúp R1/R2.
- Prompt nghiêm ngặt không tự động cải thiện mọi metric.
- Kiến trúc và budget phải được xét cùng nhau.

## 14. Khi nào nên dùng LoRA causal LM?

Phù hợp khi:

- Muốn tận dụng instruction-following.
- Cần thử nhiều tác vụ trên cùng base model.
- Full fine-tuning không vừa GPU.
- Muốn checkpoint adapter nhỏ.

Không phải lựa chọn đầu tiên khi:

- Bài toán là mapping nguồn-đích rõ ràng.
- Dữ liệu và metric ưu tiên bám nguồn.
- Context source dài.
- Cần độ trung thành cao về số liệu.

## 15. LoRA không phải phép màu

LoRA giảm chi phí thích nghi, nhưng không:

- Tự tăng context.
- Tự ngăn ảo giác.
- Tự làm dữ liệu tốt hơn.
- Bảo đảm thắng full fine-tuning.
- Biến causal LM thành encoder-decoder.

Chất lượng vẫn phụ thuộc vào base model, dữ liệu, prompt, rank, learning rate, số step và decoding.

