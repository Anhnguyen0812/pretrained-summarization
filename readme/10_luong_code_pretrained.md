# 10. Luồng code pretrained từ đầu đến cuối

Chương này đọc code theo thứ tự thực thi. Repo: [`../`](../)

## 1. Điểm vào chương trình

Train seq2seq:

```powershell
python -m vn_summarization.train --config configs/vit5_base.yaml
```

Train causal LM:

```powershell
python -m vn_summarization.train_causal_lm --config configs/qwen3_1_7b_lora.yaml
```

Evaluate seq2seq:

```powershell
python -m vn_summarization.evaluate `
  --config configs/vit5_base.yaml `
  --model_path outputs/vit5_base/best `
  --predictions_path outputs/vit5_base/predictions_valid.jsonl
```

## 2. Đọc config và override

[`utils.py`](../src/vn_summarization/utils.py) chịu trách nhiệm:

- Đọc YAML.
- Resolve đường dẫn.
- Áp dụng `--set key=value`.
- Đặt seed.
- Đếm tham số.
- Chọn precision.
- Lưu JSON.

Ví dụ override:

```powershell
python -m vn_summarization.train `
  --config configs/vit5_base.yaml `
  --set training.output_dir=outputs/vit5_lr2e5 `
  --set training.learning_rate=0.00002
```

Ưu điểm: không sửa config gốc và có thể chạy ablation rõ ràng.

## 3. `train.py`: hàm `main`

Luồng cuối file:

```python
args = parse_args()
config, config_path = load_yaml(args.config)
config = apply_overrides(config, args.overrides)
train(config, config_path)
```

`config_path` được giữ lại để resolve đường dẫn dataset tương đối với file config.

## 4. `train.py`: hàm `train`

Thứ tự:

```text
set_all_seeds()
log_runtime()
load_tokenizer_and_model()
apply_lora_if_enabled()
load_splits()
preprocess_dataset()
build_training_args()
DataCollatorForSeq2Seq()
EarlyStoppingCallback()
Seq2SeqTrainer()
trainer.train()
save model/tokenizer/state/metrics/config
trainer.evaluate()
```

Đây là xương sống của toàn bộ nhánh seq2seq.

## 5. Load dữ liệu

Trong [`data.py`](../src/vn_summarization/data.py):

```python
load_dataset(
    "parquet",
    data_files={
        "train": train_file,
        "validation": valid_file,
    },
)
```

Kết quả là `DatasetDict` có hai split.

`maybe_select` cho phép lấy tập con ngẫu nhiên có seed để smoke test hoặc ablation tiết kiệm.

## 6. Preprocess dữ liệu

`preprocess_dataset`:

1. Chọn subset nếu config yêu cầu.
2. Thêm prefix.
3. Làm sạch source/target.
4. Token hóa source với `max_source_length`.
5. Token hóa target với `max_target_length`.
6. Gắn target IDs vào `labels`.
7. Xóa cột text gốc khỏi dataset tokenized.

Việc xóa cột text giảm bộ nhớ khi train. Khi xuất prediction, code giữ một bản raw dataset riêng để ghép lại article/reference.

## 7. Load tokenizer/model

`load_tokenizer_and_model` trong [`modeling.py`](../src/vn_summarization/modeling.py):

### Bước 1: load config model

```python
model_config = AutoConfig.from_pretrained(name_or_path)
```

### Bước 2: load tokenizer

Code có fallback đặc biệt cho T5 SentencePiece để tránh lỗi phiên bản.

### Bước 3: chỉnh dropout trước khi tạo model

```python
_set_dropout(model_config, training_cfg.get("dropout"))
```

### Bước 4: load model

```python
AutoModelForSeq2SeqLM.from_pretrained(...)
```

### Bước 5: xử lý pad token/vocabulary

Nếu tokenizer không có pad token, dùng EOS. Nếu số token khác embedding size, resize embedding.

### Bước 6: tối ưu bộ nhớ

Nếu bật gradient checkpointing:

```text
enable gradient checkpointing
use_cache = False
```

Cache generation không phù hợp trong train có checkpointing.

### Bước 7: kiểm tra giới hạn tham số

Nếu model có từ 3 tỷ tham số trở lên, code ném lỗi.

## 8. Gắn LoRA

`apply_lora_if_enabled`:

```text
đọc lora.enabled
→ chọn target modules theo model type
→ tạo LoraConfig
→ get_peft_model
→ in số tham số trainable
```

Nếu LoRA tắt, model được trả về nguyên trạng và full fine-tuning diễn ra.

## 9. Tạo TrainingArguments

`build_training_args` chuyển YAML thành `Seq2SeqTrainingArguments`.

Một số trường quan trọng:

| Trường | Ý nghĩa |
|---|---|
| `predict_with_generate` | Sinh summary khi evaluate |
| `gradient_accumulation_steps` | Tạo batch hiệu dụng lớn |
| `label_smoothing_factor` | Regularization cho nhãn |
| `load_best_model_at_end` | Nạp checkpoint tốt nhất |
| `metric_for_best_model` | Metric chọn checkpoint |
| `generation_max_length` | Giới hạn sinh khi evaluate |
| `generation_num_beams` | Beam size khi evaluate |

Code kiểm tra signature để tương thích nhiều phiên bản Transformers có tên argument khác nhau.

## 10. Data collator

```python
DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    label_pad_token_id=-100,
    pad_to_multiple_of=8 if fp16_or_bf16 else None,
)
```

Đây là nơi batch được pad động và label padding được bỏ qua trong loss.

## 11. Trainer train và lưu kết quả

```python
train_result = trainer.train(...)
trainer.save_model(output_dir / "best")
tokenizer.save_pretrained(output_dir / "best")
trainer.save_metrics(...)
trainer.save_state()
```

Sau train:

```python
eval_metrics = trainer.evaluate(...)
```

Cuối cùng lưu `resolved_config.json`, rất quan trọng để biết run thực tế dùng override gì.

## 12. Tính ROUGE

Trong [`metrics.py`](../src/vn_summarization/metrics.py):

```text
pred token IDs + label token IDs
→ thay ID không hợp lệ/padding
→ batch_decode
→ strip text
→ evaluate.load("rouge")
→ rouge1, rouge2, rougeL
→ gen_len
```

Code nhân ROUGE với 100 khi ghi metric. Báo cáo chia lại cho 100 để trình bày trên thang 0-1.

## 13. Evaluate và xuất prediction

[`evaluate.py`](../src/vn_summarization/evaluate.py):

1. Nạp model hoặc LoRA adapter.
2. Load validation.
3. Token hóa.
4. `trainer.predict()` để sinh.
5. Decode prediction.
6. Ghi mỗi article/reference/prediction thành một dòng JSONL.

Nếu thư mục model có `adapter_config.json`, code biết đó là LoRA:

```text
load base model
→ PeftModel.from_pretrained(base, adapter_path)
```

## 14. Luồng causal LM

### `causal_data.py`

```text
article
→ build_prompt()
→ tokenize prompt
→ tokenize target + EOS
→ nối input
→ labels prompt = -100
```

### `train_causal_lm.py`

```text
AutoModelForCausalLM
→ LoRA
→ CausalDataCollator
→ Trainer
→ train/evaluate/save adapter
```

### `evaluate_causal_lm.py`

```text
prompt only
→ left padding
→ model.generate()
→ lấy token sau input_len
→ clean generation
→ ROUGE
→ predictions JSONL
```

Left padding khi generation giúp các prompt trong batch kết thúc ở cùng phía trước khi model sinh continuation.

## 15. Cách đọc một run khi có lỗi

Thứ tự kiểm tra:

1. `resolved_config.json`: run thực tế dùng gì?
2. `train_results.json`: train có chạy đủ không?
3. `eval_results.json`: loss/ROUGE/gen_len thế nào?
4. `trainer_state.json`: metric thay đổi theo step ra sao?
5. `predictions_valid.jsonl`: lỗi thực tế là gì?
6. Model/tokenizer trong `best/`: có đủ file không?

## 16. Những nguồn sự thật ưu tiên

Khi config, notebook và báo cáo khác nhau:

1. `resolved_config.json` của run.
2. Metric/prediction được xuất bởi run.
3. Notebook log.
4. Config YAML gốc.
5. Mô tả trong báo cáo.

Config YAML có thể đã bị notebook override, nên không phải lúc nào cũng phản ánh run cuối.
