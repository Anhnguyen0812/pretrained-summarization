# 04. Dữ liệu và phân tích dataset

## 1. Cấu trúc dữ liệu

Mỗi mẫu gồm:

| Trường | Ý nghĩa |
|---|---|
| `article` | Văn bản nguồn cần tóm tắt |
| `summary` | Bản tóm tắt tham chiếu |

Các tập:

| Split | Số mẫu | Vai trò |
|---|---:|---|
| Train | 10.775 | Tính gradient và cập nhật mô hình |
| Validation | 1.349 | Chọn checkpoint, cấu hình và phân tích |
| Test | 1.344 | Đánh giá cuối cùng |

Phân tích mô tả trong dự án chỉ sử dụng train và validation. Không phân tích nội dung test.

## 2. Code phân tích

Script: [`../../analyze_train_valid_dataset.py`](../../analyze_train_valid_dataset.py)

Đầu ra: [`../../train_valid_dataset_analysis.json`](../../train_valid_dataset_analysis.json)

Chạy:

```powershell
python analyze_train_valid_dataset.py `
  --train dataset/train-00000-of-00001.parquet `
  --valid dataset/valid-00000-of-00001.parquet `
  --output train_valid_dataset_analysis.json
```

Script cố ý chỉ nhận train và validation để tránh sử dụng test trong phân tích.

## 3. Phân phối độ dài

| Chỉ số | Train | Validation |
|---|---:|---:|
| Số mẫu | 10.775 | 1.349 |
| Article trung bình | 463,0 từ | 471,2 từ |
| Article trung vị | 449 từ | 455 từ |
| Article P90 | 620 từ | 632 từ |
| Summary trung bình | 103,8 từ | 103,3 từ |
| Summary trung vị | 101 từ | 101 từ |
| Summary P90 | 135 từ | 135,2 từ |
| Tỷ lệ nén trung bình | 0,232 | 0,227 |

### Diễn giải

Một summary giữ khoảng 23% độ dài article. Bài toán không chỉ cần viết lại một câu đầu; mô hình phải chọn lọc nhiều thông tin trong văn bản dài.

Train và validation có phân phối gần nhau. Điều này làm validation đáng tin cậy hơn khi chọn mô hình.

## 4. Phân tích mức độ trừu tượng

| Chỉ số | Train | Validation |
|---|---:|---:|
| Recall unigram summary xuất hiện trong article | 92,26% | 92,51% |
| Recall bigram summary xuất hiện trong article | 68,14% | 68,99% |
| Tỷ lệ unigram mới trong summary | 6,43% | 6,22% |
| Tỷ lệ bigram mới trong summary | 31,06% | 30,23% |

### Ý nghĩa

Dataset mang tính trừu tượng nhưng bám nguồn mạnh:

- Phần lớn từ của summary có trong article.
- Nhiều cặp từ được tổ chức lại, thể hiện ở khoảng 30% bigram mới.
- ROUGE phù hợp để đo độ bảo toàn cụm từ, nhưng không đủ để đánh giá hoàn toàn tính đúng đắn.

Đặc trưng này có lợi cho encoder-decoder vì cross-attention giúp chọn và tổ chức lại cụm từ nguồn.

## 5. Thông tin quan trọng nằm ở đâu?

| Chỉ số | Train | Validation |
|---|---:|---:|
| Độ phủ summary bởi 25% đầu article | 50,34% | 51,31% |
| Độ phủ summary bởi 50% đầu article | 72,92% | 73,51% |

Nếu chỉ đọc một phần tư đầu bài, mô hình bỏ lỡ gần nửa nội dung từ vựng của summary. Vì vậy:

- Truncation quá mạnh gây mất thông tin.
- Tăng context từ 768 lên 1024 token có thể giúp causal LM.
- Chỉ dùng câu đầu hoặc lead-based extraction không đủ.

## 6. Độ dài đích và truncation

| Chỉ số | Train | Validation |
|---|---:|---:|
| Summary dài hơn 128 từ | 15,53% | 15,05% |
| Summary dài hơn 160 từ | 0,29% | 0,30% |

Lưu ý đây là số **từ**, còn mô hình giới hạn theo **subword token**. Một summary 100 từ có thể dài hơn 100 token.

Diễn giải:

- Đích 128 token có nguy cơ cắt đáng kể số mẫu.
- Đích 160 token hợp lý hơn, nhưng vẫn có thể cắt vì subword fragmentation.
- ViT5 dùng đích 160 token.
- Qwen thí nghiệm chính dùng đích 128 token, tạo bất lợi.

## 7. Số liệu và tính trung thành

Khoảng 96,93% số xuất hiện trong summary validation có mặt trong article tương ứng.

Điều này cho thấy:

- Summary thường bảo toàn số liệu nguồn.
- Sai ngày, số lượng hoặc tỷ lệ là lỗi nghiêm trọng.
- Nên bổ sung metric hoặc kiểm tra thủ công cho số, tên và ngày tháng.

## 8. `clean_text` làm gì?

Trong [`data.py`](../src/vn_summarization/data.py), `clean_text`:

1. Chuẩn hóa Unicode về NFC.
2. Thay non-breaking space và BOM bằng khoảng trắng thường.
3. Gom nhiều khoảng trắng thành một.
4. Xóa khoảng trắng đầu/cuối.

Việc này giúp tránh trường hợp hai chuỗi nhìn giống nhau nhưng có biểu diễn Unicode khác nhau.

## 9. Data leakage cần tránh

Data leakage xảy ra khi thông tin không nên có trong train lại ảnh hưởng đến mô hình hoặc quyết định cấu hình.

Các tình huống cần tránh:

- Dùng test để chọn beam size.
- Phân tích test rồi sửa preprocessing theo test.
- Trộn sample validation vào train nhưng vẫn báo validation cũ.
- Warm-start từ checkpoint đã train trực tiếp trên cùng dữ liệu mà không công bố.

## 10. Câu hỏi nên hỏi trước khi train

1. Article và summary có rỗng hoặc lỗi encoding không?
2. Tokenizer biến một article trung bình thành bao nhiêu token?
3. Bao nhiêu phần trăm article/summary bị truncation?
4. Train và validation có phân phối tương tự không?
5. Summary có xu hướng sao chép hay diễn đạt lại?
6. Số liệu, tên riêng và ngày tháng quan trọng đến mức nào?
