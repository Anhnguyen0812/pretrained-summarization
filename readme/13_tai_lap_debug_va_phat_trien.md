# 13. Tái lập, debug và phát triển tiếp

## 1. Nguyên tắc tái lập

Một kết quả tái lập được cần ghi lại:

- Code version.
- Config thực tế.
- Model checkpoint gốc.
- Tokenizer.
- Seed.
- Phiên bản thư viện.
- Loại GPU.
- Dataset split.
- Metric code.
- Generation config.

Trong repo pretrained, `resolved_config.json` và các file metric là nền tảng tốt, nhưng nên bổ sung Git commit hash khi chạy.

## 2. Cài môi trường pretrained

```powershell
cd pretrained-summarization
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
```

Smoke test:

```powershell
python -m vn_summarization.train --config configs/smoke_mt5_small.yaml
```

Smoke test chỉ xác nhận pipeline chạy; nó không đánh giá chất lượng mô hình.

## 3. Kiểm tra tokenizer

```powershell
python scripts/check_tokenizer.py
```

Cần kiểm tra:

- Loại tokenizer.
- Vocabulary size.
- `pad_token_id`, `eos_token_id`, `unk_token_id`.
- Encode/decode tiếng Việt đúng.
- Không xuất hiện ký tự lỗi encoding.

## 4. Train ViT5

```powershell
python -m vn_summarization.train --config configs/vit5_base.yaml
```

Train ViT5 LoRA:

```powershell
python -m vn_summarization.train --config configs/vit5_base_lora.yaml
```

Kaggle T4x2:

```bash
python -m accelerate.commands.launch \
  --multi_gpu \
  --num_processes 2 \
  --mixed_precision fp16 \
  -m vn_summarization.train \
  --config configs/vit5_base_t4x2.yaml
```

## 5. Evaluate validation

```powershell
python -m vn_summarization.evaluate `
  --config configs/vit5_base.yaml `
  --model_path outputs/vit5_base/best `
  --predictions_path outputs/vit5_base/predictions_valid.jsonl
```

Sau đó kiểm tra:

- `validation_metrics.json`.
- `predictions_valid.jsonl`.
- Một số mẫu có tên/số/ngày.

## 6. Predict một article

```powershell
python -m vn_summarization.predict `
  --model_path outputs/vit5_base/best `
  --text "<văn bản tiếng Việt>"
```

Đây là kiểm tra nhanh hữu ích trước khi chạy evaluation toàn bộ.

## 7. Train causal LM

```powershell
python -m vn_summarization.train_causal_lm `
  --config configs/qwen3_1_7b_lora.yaml
```

Vì model lớn:

- Bắt đầu bằng subset và ít step.
- Theo dõi VRAM.
- Kiểm tra prompt encoding.
- Kiểm tra loss chỉ tính trên summary.
- Đọc prediction để phát hiện prompt leakage.

## 8. OOM: hết bộ nhớ GPU

Thử theo thứ tự:

1. Giảm `per_device_train_batch_size`.
2. Tăng `gradient_accumulation_steps` để giữ batch hiệu dụng.
3. Giảm `max_source_length`.
4. Giảm `max_target_length`.
5. Bật gradient checkpointing.
6. Dùng FP16/BF16.
7. Dùng LoRA.
8. Giảm eval batch size.

Không nên giảm context ngay nếu mục tiêu thí nghiệm là kiểm tra thông tin dài; có thể giảm batch trước.

## 9. Loss thành NaN hoặc train không ổn định

Kiểm tra:

- Learning rate có quá cao không?
- FP16 có overflow không?
- Có gradient clipping không?
- Label có toàn `-100` không?
- Input có rỗng không?
- Token ID có vượt vocabulary không?
- Batch có sample cực dài hoặc lỗi không?

Thử:

- Giảm learning rate.
- Chạy FP32 trên batch nhỏ để debug.
- In gradient norm.
- Kiểm tra một batch trước train.

## 10. Loss không giảm

Kiểm tra:

- Tham số có `requires_grad=True` không?
- Optimizer có nhận đúng tham số không?
- LoRA có gắn đúng target module không?
- Labels có đúng không?
- Decoder input/labels có shift đúng không?
- Learning rate có bằng 0 không?
- Scheduler/warmup có cấu hình sai không?

Bài test tốt: thử overfit 8-32 sample. Nếu mô hình không thể giảm loss mạnh trên tập nhỏ, pipeline có thể lỗi.

## 11. ROUGE thấp nhưng loss tốt

Nguyên nhân có thể:

- Teacher forcing tốt nhưng generation kém.
- Generation config không phù hợp.
- Prediction quá ngắn/dài.
- Tokenizer decode lỗi.
- Model học phong cách khác reference.
- Loss checkpoint và generated ROUGE checkpoint không cùng tiêu chí.

Hành động:

- Đọc prediction.
- So greedy và beam trên validation.
- Kiểm tra EOS/max length.
- Theo dõi `gen_len`.
- Chọn checkpoint theo ROUGE-L.

## 12. Prediction bị lặp

Kiểm tra:

- Model đã train đủ chưa?
- Beam size có quá lớn không?
- Repetition penalty/no-repeat n-gram.
- Target train có lặp không?
- EOS có đúng không?

Không chỉ tăng penalty; cần đọc xem lặp do decoding hay do model.

## 13. Prediction bị cắt

Dấu hiệu:

- Kết thúc giữa câu.
- Hầu hết output có cùng max length.
- `gen_len` sát `max_length`.

Giải pháp:

- Tăng max target/generation length.
- Kiểm tra tokenizer EOS.
- Kiểm tra target có bị truncation khi train.
- Điều chỉnh length penalty.

## 14. Prediction quá chung chung

Nguyên nhân:

- Model nhỏ hoặc underfit.
- Source bị truncation.
- Learning rate/regularization không hợp lý.
- Dataset có nhiều summary chung chung.
- Causal prompt chưa yêu cầu giữ chi tiết.

Giải pháp:

- Tăng source context.
- Train thêm nếu validation còn tăng.
- Dùng pretrained model mạnh hơn.
- Kiểm tra coverage và lỗi thực thể.

## 15. Lỗi tokenizer ViT5

Repo có fallback riêng vì một số phiên bản Transformers/tokenizers lỗi khi chuyển T5 SentencePiece sang fast tokenizer.

Giữ:

- `transformers>=4.51.0,<5`.
- `tokenizers>=0.22.0,<=0.23.0`.
- `use_fast_tokenizer: false` cho ViT5 nếu cần.

Nếu lỗi:

1. Chạy `scripts/check_tokenizer.py`.
2. Kiểm tra `spiece.model`.
3. Kiểm tra version.
4. Xóa cache model lỗi và tải lại nếu cần.

## 16. Thiết kế ablation đúng

Mỗi ablation chỉ thay một yếu tố:

```text
baseline
→ chỉ đổi context
→ so kết quả
```

Không nên đồng thời đổi context, rank, learning rate và số step rồi kết luận context tạo cải thiện.

Ghi cho mỗi run:

- Giả thuyết.
- Một thay đổi chính.
- Metric.
- Chi phí.
- Prediction mẫu.
- Kết luận.

## 17. Checklist trước khi báo cáo kết quả

- [ ] Run dùng đúng train/validation split.
- [ ] Không chọn config bằng test.
- [ ] Có `resolved_config.json`.
- [ ] Metric cùng thang điểm.
- [ ] Ghi rõ full split hay subset.
- [ ] Có prediction mẫu.
- [ ] Có kiểm tra tên/số/ngày.
- [ ] Có nêu trainable/total parameters.
- [ ] Có nêu giới hạn hoặc bất công budget.
- [ ] Có thể tìm lại checkpoint và tokenizer.

## 18. Hướng phát triển repo

Ưu tiên:

1. Bổ sung repo Transformer scratch.
2. Thêm unit test cho mask và attention.
3. Thêm script đo truncation theo tokenizer từng model.
4. Thêm metric kiểm tra number/entity faithfulness.
5. Thêm manifest run với Git commit và environment.
6. Tách decode tuning validation thành pipeline chuẩn.
7. Tạo bảng so sánh chất lượng, VRAM, thời gian và kích thước checkpoint.

