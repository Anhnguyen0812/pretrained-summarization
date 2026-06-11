# Cẩm nang dự án tóm tắt văn bản tiếng Việt

Thư mục này là tài liệu học và tài liệu kỹ thuật cho toàn bộ dự án. Mục tiêu không chỉ là hướng dẫn chạy code, mà còn giúp người đọc hiểu:

- Bài toán đang giải quyết là gì.
- Mô hình học từ dữ liệu bằng cách nào.
- Transformer hoạt động ra sao từ mức token đến attention.
- Khác biệt giữa mô hình huấn luyện từ đầu, mô hình tiền huấn luyện encoder-decoder và causal language model.
- Trong mỗi bước train, tensor nào đi vào mô hình, loss được tính ở đâu và tham số nào được cập nhật.
- Vì sao một cấu hình cho kết quả tốt hoặc kém.
- Cách đọc ROUGE, loss, prediction và phát hiện lỗi mô hình.

## Lộ trình đọc đề xuất

### Lộ trình A: Chưa biết học máy

Đọc tuần tự:

1. [01 - Nền tảng học máy](01_nen_tang_hoc_may.md)
2. [02 - Nền tảng học sâu](02_nen_tang_hoc_sau.md)
3. [03 - NLP và bài toán tóm tắt](03_nlp_va_bai_toan_tom_tat.md)
4. [04 - Dữ liệu của dự án](04_du_lieu_va_phan_tich.md)
5. [05 - Bản đồ toàn bộ dự án](05_ban_do_du_an.md)
6. [06 - Transformer từ đầu](06_transformer_scratch_kien_truc.md)
7. [07 - Huấn luyện Transformer từ đầu](07_transformer_scratch_huan_luyen.md)
8. [08 - Pretrained sequence-to-sequence](08_pretrained_seq2seq.md)
9. [09 - LoRA và causal language model](09_lora_va_causal_lm.md)
10. [10 - Luồng code pretrained](10_luong_code_pretrained.md)
11. [11 - Đánh giá và phân tích lỗi](11_danh_gia_va_phan_tich_loi.md)
12. [12 - Kết quả và cách diễn giải](12_ket_qua_va_dien_giai.md)
13. [13 - Chạy lại, debug và phát triển tiếp](13_tai_lap_debug_va_phat_trien.md)
14. [14 - Từ điển thuật ngữ và FAQ](14_thuat_ngu_va_faq.md)
15. [15 - Chiến lược fine-tuning, metric, model và kết quả](15_chien_luoc_finetune_metric_model_ket_qua.md)

### Lộ trình B: Đã biết deep learning, muốn hiểu dự án

Đọc theo thứ tự: `03 → 04 → 05 → 06 → 07 → 08 → 09 → 10 → 11 → 12 → 15`.

### Lộ trình C: Muốn chạy và sửa code pretrained

Đọc theo thứ tự: `05 → 08 → 09 → 10 → 15 → 13`.

## Hai hướng chính của đồ án

| Hướng | Ý tưởng | Điểm mạnh | Điểm yếu |
|---|---|---|---|
| Transformer từ đầu | Tự cài đặt encoder-decoder và học toàn bộ từ dữ liệu bài toán | Hiểu sâu kiến trúc; kiểm soát mọi thành phần; mô hình nhỏ | Không có kiến thức ngôn ngữ ban đầu; cần nhiều dữ liệu; chất lượng thấp hơn |
| Mô hình tiền huấn luyện | Bắt đầu từ ViT5/BARTpho/Qwen đã học ngôn ngữ trên dữ liệu lớn rồi tinh chỉnh | Chất lượng cao; hội tụ nhanh; sinh tiếng Việt tốt | Tốn bộ nhớ; phức tạp hơn; cần hiểu tokenizer/checkpoint/LoRA |

## Kết quả chính cần nhớ

| Mô hình | Validation ROUGE-1/2/L | Test ROUGE-1/2/L |
|---|---|---|
| Transformer từ đầu | 0.5768 / 0.2087 / 0.3053 | 0.5747 / 0.2038 / 0.3045 |
| ViT5-base full fine-tuning | **0.7417 / 0.4709 / 0.4924** | **0.7422 / 0.4675 / 0.4889** |
| ViT5-base LoRA | 0.7297 / 0.4484 / 0.4733 | 0.7262 / 0.4408 / 0.4663 |

ViT5 tốt hơn chủ yếu vì đã học tiếng Việt trước khi nhìn thấy tập dữ liệu của đồ án. Transformer từ đầu phải đồng thời học ngôn ngữ, kiến thức về cấu trúc văn bản và cách tóm tắt chỉ từ 10.775 mẫu train.

## Các tệp gốc quan trọng

- Báo cáo tiếng Việt: [`../reports_paper/ieee_abstractive_summarization_report_vi.tex`](../reports_paper/ieee_abstractive_summarization_report_vi.tex)
- Mã pretrained: [`../`](../)
- Phân tích train/validation: [`../../analyze_train_valid_dataset.py`](../../analyze_train_valid_dataset.py)
- Kết quả phân tích: [`../../train_valid_dataset_analysis.json`](../../train_valid_dataset_analysis.json)
- Notebook kết quả Kaggle: [`../result_notebook_kaggle/`](../result_notebook_kaggle/)

## Phạm vi và tính trung thực của tài liệu

Phần pretrained được giải thích trực tiếp từ mã nguồn hiện có trong `pretrained-summarization/`.

Mã nguồn Transformer scratch chưa có dưới dạng repo Python riêng trong workspace hiện tại. Vì vậy, các chương scratch mô tả đầy đủ kiến trúc, tensor, thuật toán và cấu hình đã được ghi nhận trong báo cáo: Transformer encoder-decoder 11,47 triệu tham số, bốn lớp encoder, bốn lớp decoder, `d_model=256`, tám attention head, Pre-LN, GELU, shared embedding, weight tying, label smoothing, warmup, beam search và chặn n-gram lặp. Khi repo scratch được bổ sung, nên cập nhật các liên kết mã nguồn tương ứng trong tài liệu.
