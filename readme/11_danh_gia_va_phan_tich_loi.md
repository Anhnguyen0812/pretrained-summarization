# 11. Đánh giá và phân tích lỗi

## 1. Đánh giá mô hình để trả lời câu hỏi gì?

Không có một con số duy nhất trả lời được mọi khía cạnh. Cần kết hợp:

- Loss: mô hình dự đoán token tốt đến đâu trong điều kiện teacher forcing?
- ROUGE: prediction trùng nội dung từ vựng với reference đến đâu?
- Độ dài: prediction có quá ngắn hoặc dài không?
- Phân tích định tính: prediction có đúng sự kiện, số và tên không?
- Chi phí: train/inference tốn bao nhiêu thời gian, VRAM và tham số?

## 2. ROUGE-1

ROUGE-1 đo độ trùng unigram giữa prediction và reference.

Ví dụ:

```text
Reference: "ViT5 đạt kết quả tốt trên tập validation"
Prediction: "ViT5 cho kết quả tốt ở validation"
```

Nhiều từ trùng nên ROUGE-1 cao.

ROUGE-1 phản ánh độ bao phủ từ khóa/nội dung cơ bản, nhưng không kiểm tra tốt thứ tự hay quan hệ giữa từ.

## 3. ROUGE-2

ROUGE-2 đo độ trùng bigram. Nó nghiêm ngặt hơn vì cần hai token liên tiếp giống nhau.

ROUGE-2 cao thường cho thấy:

- Cụm từ quan trọng được giữ.
- Cấu trúc cục bộ gần reference.
- Nội dung không chỉ trùng các từ rời rạc.

Trong dự án, ViT5 hơn gấp đôi scratch về ROUGE-2 trên test, cho thấy bảo toàn cụm từ tốt hơn nhiều.

## 4. ROUGE-L

ROUGE-L dựa trên dãy con chung dài nhất, cho phép có khoảng cách giữa token nhưng vẫn xét thứ tự.

Nó thường được dùng làm metric chính cho tóm tắt vì cân bằng giữa độ phủ và cấu trúc chuỗi.

Pipeline chọn checkpoint pretrained theo validation ROUGE-L.

## 5. Precision, recall và F1 trong ROUGE

Trực giác:

- Precision: phần prediction có bao nhiêu nội dung khớp reference?
- Recall: prediction bao phủ bao nhiêu nội dung reference?
- F1: cân bằng precision và recall.

Prediction rất dài có thể tăng recall nhưng giảm precision. Prediction quá ngắn có thể precision cao nhưng bỏ nhiều ý. Báo cáo dùng F1.

## 6. ROUGE không đo được gì?

ROUGE có thể bỏ sót:

- Hai câu đồng nghĩa dùng từ khác nhau.
- Sai thực thể nhưng cấu trúc câu giống.
- Sai số hoặc ngày tháng.
- Logic sai.
- Câu không tự nhiên nhưng sao chép nhiều từ.

Ví dụ:

```text
Reference: sự kiện diễn ra ngày 6/12
Prediction: sự kiện diễn ra ngày 12/11
```

Phần lớn từ vẫn trùng, nhưng thông tin quan trọng sai.

## 7. Vì sao cần prediction JSONL?

Prediction JSONL ghép ba phần:

```json
{
  "article": "...",
  "summary": "...",
  "prediction": "..."
}
```

Nó cho phép trả lời:

- Mô hình bỏ ý gì?
- Có thêm thông tin ngoài nguồn không?
- Có sai tên/số/ngày không?
- Có lặp không?
- Có chỉ sao chép câu đầu không?
- Độ dài có phù hợp không?

## 8. Taxonomy lỗi nên sử dụng

### Thiếu thông tin chính

Prediction bỏ thực thể, sự kiện, địa điểm hoặc số liệu quan trọng.

### Ảo giác

Prediction thêm hoặc thay đổi thông tin nguồn không hỗ trợ.

### Quá chung chung

Prediction đúng chủ đề nhưng không có thông tin cụ thể.

### Lặp

Lặp từ, cụm từ hoặc mệnh đề.

### Sai trọng tâm

Prediction chọn chi tiết phụ thay vì ý chính.

### Vấn đề độ dài

Prediction quá ngắn, quá dài hoặc bị cắt đột ngột.

### Lỗi ngôn ngữ

Câu không tự nhiên, sai ngữ pháp, lẫn ngôn ngữ hoặc rò rỉ prompt.

## 9. Quy trình đọc mẫu định tính

Với mỗi prediction:

1. Đọc reference để biết ý đích.
2. Đọc article để kiểm tra reference và prediction.
3. Gạch các thực thể, số, ngày, địa điểm trong prediction.
4. Kiểm tra từng chi tiết có nguồn hỗ trợ không.
5. Đánh dấu ý quan trọng bị bỏ.
6. Đánh giá độ mạch lạc và độ dài.
7. Gán một hoặc nhiều nhóm lỗi.

Không nên chỉ so prediction với reference vì reference cũng chỉ là một cách tóm tắt.

## 10. Ví dụ ViT5 trong dự án

Article nói về InterContinental Phu Quoc Long Beach Resort:

- Được vinh danh hạng mục gia đình ở châu Á năm thứ hai liên tiếp.
- Có thiết kế lấy cảm hứng từ đại dương.
- Có lưu trú, chăm sóc sức khỏe, giải trí và ẩm thực.

Prediction ViT5:

- Giữ đúng thực thể.
- Giữ giải thưởng, thiết kế, lưu trú và chăm sóc sức khỏe.
- Bỏ chi tiết “năm thứ hai liên tiếp”.
- Không có ảo giác rõ ràng.

Phân loại: **thiếu một chi tiết chính nhỏ**, nhưng nhìn chung trung thành và bao phủ nhiều khía cạnh.

## 11. Ví dụ Qwen trong dự án

Nguồn nói Miho Nakayama được phát hiện ngày `6/12`. Qwen prediction nói `12/11`.

Prediction:

- Mạch lạc.
- Đúng danh tính, địa điểm và chủ đề sự nghiệp.
- Sai ngày tháng.

Phân loại: **ảo giác số liệu**.

Đây là ví dụ vì sao một summary có vẻ tốt vẫn có thể không đáng tin.

## 12. Độ dài sinh

Theo dõi `gen_len` để phát hiện:

- Model luôn sinh sát max length: có thể chưa học dừng hoặc max length quá thấp.
- Model sinh quá ngắn: có thể length penalty, EOS hoặc train target gây vấn đề.
- Gen length khác xa reference distribution: ROUGE và chất lượng có thể giảm.

ViT5 sinh khoảng 125 token trên validation, khá gần giới hạn 160. Cần kiểm tra prediction có bị dài dòng hay cắt không.

## 13. Repetition

Scratch theo dõi tỷ lệ 3-gram lặp. Với no-repeat 3-gram, tỷ lệ gần 0.

Không nên chỉ kết luận mô hình không lặp vì metric bằng 0:

- Chặn n-gram có thể cưỡng chế không lặp.
- Mô hình vẫn có thể lặp ý bằng cách diễn đạt khác.
- Cần đọc mẫu.

## 14. Đánh giá công bằng giữa các run

Muốn so sánh công bằng:

- Cùng split validation.
- Cùng số sample.
- Cùng cách tính ROUGE.
- Cùng tiền xử lý.
- Cùng thang điểm 0-1 hoặc 0-100.
- Ghi rõ subset nếu dùng subset.
- Không so trực tiếp subset score với full-validation score.

Qwen ablation dùng tập con validation 500 mẫu, nên không được so trực tiếp tuyệt đối với bảng full validation.

## 15. Phân biệt model tuning và decode tuning

### Model tuning

Thay đổi trọng số bằng train:

- Learning rate.
- Epoch.
- LoRA rank.
- Dữ liệu.
- Dropout.

### Decode tuning

Giữ nguyên trọng số, thay cách sinh:

- Beam size.
- Length penalty.
- Max/min length.
- Repetition penalty.

Decode tuning nên thực hiện trên validation, không phải test.

## 16. Test protocol đúng

Quy trình:

```text
train nhiều run
→ chọn bằng validation
→ chốt model + decode config
→ chạy test một lần
→ báo cáo
```

Nếu điểm test thấp hơn dự kiến, ghi nhận kết quả; không tiếp tục chọn cấu hình theo test.

## 17. Metric bổ sung nên phát triển

ROUGE chưa đủ cho tính trung thành. Có thể bổ sung:

- Tỷ lệ số trong prediction được article hỗ trợ.
- Precision/recall thực thể tên riêng.
- Kiểm tra ngày tháng.
- BERTScore hoặc embedding similarity.
- Đánh giá con người theo rubric.
- LLM-as-judge, nhưng phải kiểm soát bias và không thay thế hoàn toàn đánh giá người.

## 18. Rubric đánh giá thủ công đề xuất

Chấm mỗi prediction từ 1 đến 5:

| Tiêu chí | Câu hỏi |
|---|---|
| Trung thành | Mọi thông tin có được nguồn hỗ trợ không? |
| Bao phủ | Có giữ các ý quan trọng nhất không? |
| Súc tích | Có bỏ chi tiết thừa và tránh lặp không? |
| Mạch lạc | Có đọc tự nhiên và logic không? |
| Số liệu/thực thể | Tên, số, ngày và địa điểm có đúng không? |

Nên có ít nhất hai người chấm một tập mẫu cố định và thảo luận các trường hợp bất đồng.

