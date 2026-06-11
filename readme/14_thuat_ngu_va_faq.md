# 14. Từ điển thuật ngữ và câu hỏi thường gặp

## A. Từ điển thuật ngữ

### Abstractive summarization

Tóm tắt hướng trừu tượng: sinh câu mới để tóm tắt nguồn, không chỉ lấy nguyên câu.

### Attention

Cơ chế cho phép một vị trí tính mức quan trọng của các vị trí khác và tổng hợp thông tin từ chúng.

### Autoregressive

Sinh tuần tự; token sau phụ thuộc các token trước.

### Backpropagation

Thuật toán tính gradient từ loss ngược qua mạng.

### Batch

Nhóm sample được xử lý cùng lúc.

### Beam search

Giải mã giữ nhiều chuỗi ứng viên thay vì chỉ một chuỗi tốt nhất cục bộ.

### BPE

Byte Pair Encoding; thuật toán học các đơn vị subword bằng cách gộp cặp thường gặp.

### Causal language model

Mô hình dự đoán token tiếp theo chỉ dựa trên token trước đó.

### Checkpoint

Trạng thái mô hình được lưu tại một thời điểm train.

### Cross-attention

Attention trong đó query từ decoder, key/value từ encoder.

### Cross-entropy

Loss phổ biến cho phân loại token; phạt mô hình khi xác suất token đúng thấp.

### Decoder

Thành phần sinh chuỗi đích.

### Dropout

Regularization ngẫu nhiên tắt một phần activation khi train.

### Embedding

Vector có thể học đại diện cho token.

### Encoder

Thành phần đọc và biểu diễn chuỗi nguồn.

### Epoch

Một lần đi qua toàn bộ tập train.

### Fine-tuning

Tiếp tục train mô hình tiền huấn luyện trên tác vụ/dataset mới.

### Full fine-tuning

Cập nhật toàn bộ tham số mô hình tiền huấn luyện.

### Gradient

Đạo hàm cho biết hướng và mức thay đổi loss theo tham số.

### Gradient accumulation

Cộng gradient nhiều micro-batch trước một optimizer step.

### Gradient checkpointing

Giảm VRAM bằng cách tính lại một phần activation khi backward.

### Hallucination

Mô hình sinh thông tin không được nguồn hỗ trợ.

### Label

Đáp án dùng để tính loss.

### Label smoothing

Làm mềm nhãn one-hot để giảm quá tự tin.

### LayerNorm

Chuẩn hóa đặc trưng trong mỗi token representation.

### Learning rate

Kích thước bước cập nhật tham số.

### Logits

Điểm chưa qua softmax cho từng token trong vocabulary.

### LoRA

Low-Rank Adaptation; tinh chỉnh bằng các ma trận cập nhật hạng thấp.

### Mask

Tensor đánh dấu vị trí mô hình được phép hoặc không được phép chú ý/tính loss.

### Overfitting

Mô hình tốt trên train nhưng kém trên dữ liệu mới.

### Padding

Token đệm để các chuỗi trong batch có cùng chiều dài.

### Parameter

Giá trị mô hình tự học, ví dụ weight và bias.

### Positional encoding

Thông tin vị trí được thêm vào token representation.

### Pretraining

Huấn luyện trước trên dữ liệu lớn để học biểu diễn tổng quát.

### Prompt

Chuỗi chỉ dẫn và ngữ cảnh đưa vào causal/instruction model.

### Residual connection

Kết nối cộng đầu vào khối vào đầu ra khối.

### ROUGE

Nhóm metric đo độ trùng giữa prediction và reference.

### Scheduler

Quy tắc thay đổi learning rate theo step.

### Self-attention

Attention trong cùng một chuỗi.

### Sequence-to-sequence

Mô hình ánh xạ một chuỗi nguồn sang một chuỗi đích.

### Step

Một lần optimizer cập nhật tham số.

### Subword

Đơn vị token nhỏ hơn hoặc bằng từ, giúp biểu diễn từ mới.

### Teacher forcing

Khi train, decoder nhận token đúng trước đó thay vì token tự dự đoán.

### Token

Đơn vị mà tokenizer và mô hình xử lý.

### Tokenizer

Thành phần biến văn bản thành token ID và ngược lại.

### Truncation

Cắt chuỗi vì vượt độ dài tối đa.

### Validation

Tập dùng để chọn mô hình/cấu hình, không cập nhật trọng số.

### Vocabulary

Tập token mô hình biết.

### Warmup

Giai đoạn tăng learning rate từ nhỏ trong đầu quá trình train.

### Weight decay

Regularization hạn chế trọng số tăng quá lớn.

### Weight tying

Dùng chung trọng số giữa embedding và output projection hoặc các thành phần liên quan.

## B. Câu hỏi thường gặp

### 1. Mô hình thực sự đang train cái gì?

Mô hình đang điều chỉnh các ma trận trọng số để tăng xác suất của token summary đúng trong điều kiện biết article và các token summary trước đó.

### 2. Mô hình có “hiểu” văn bản không?

Mô hình học biểu diễn và quan hệ thống kê rất mạnh, có thể thực hiện hành vi giống hiểu trong nhiều trường hợp. Tuy nhiên, nó không đảm bảo suy luận đúng hoặc trung thành; ảo giác vẫn xảy ra.

### 3. Tại sao cần tokenizer? Sao không đưa chữ trực tiếp?

Mạng nơ-ron làm việc với số. Tokenizer tạo một vocabulary hữu hạn và biến văn bản thành ID để tra embedding.

### 4. Tại sao Transformer cần positional encoding?

Self-attention không tự biết thứ tự. Nếu không có thông tin vị trí, chuỗi hoán đổi token có thể trông giống nhau với attention.

### 5. Vì sao decoder cần causal mask?

Để token hiện tại không nhìn thấy đáp án ở tương lai trong lúc train.

### 6. Vì sao scratch train loss giảm nhưng ROUGE vẫn thấp?

Loss đo dự đoán token khi có teacher forcing. Khi inference, mô hình dùng token tự sinh và lỗi có thể tích lũy. Ngoài ra scratch chưa có kiến thức ngôn ngữ tiền huấn luyện.

### 7. Vì sao ViT5 mạnh hơn scratch?

ViT5 đã học tiếng Việt từ corpus lớn, có nhiều tham số hơn và dùng context/target dài hơn.

### 8. LoRA có làm model nhỏ đi không?

Base model vẫn lớn như cũ khi inference. LoRA chủ yếu giảm số tham số cần train và kích thước adapter lưu trữ.

### 9. `-100` trong labels là gì?

Đó là giá trị mặc định được cross-entropy của Transformers bỏ qua. Nó dùng cho padding hoặc phần prompt không muốn tính loss.

### 10. Vì sao không dùng test để chọn cấu hình tốt nhất?

Vì khi đó test không còn là dữ liệu chưa thấy; điểm cuối sẽ lạc quan và không phản ánh generalization.

### 11. ROUGE cao có nghĩa summary đúng hoàn toàn không?

Không. Summary có thể sai tên hoặc số nhưng vẫn trùng nhiều từ với reference.

### 12. Beam search luôn tốt hơn greedy không?

Không. Beam search tìm kiếm rộng hơn nhưng có thể ưu tiên chuỗi chung chung hoặc dài không phù hợp. Cần kiểm tra trên validation.

### 13. Tại sao context dài hơn giúp Qwen?

Prompt và article dùng chung context; context ngắn cắt mất phần article sau, trong khi dữ liệu cho thấy thông tin quan trọng không chỉ nằm ở đầu bài.

### 14. Tại sao tăng rank LoRA không luôn tăng ROUGE-L?

Năng lực adapter chỉ là một yếu tố. Context, dữ liệu, số step, decoding và bản chất base model cũng giới hạn kết quả.

### 15. Nên nhìn metric nào trước?

ROUGE-L để chọn tổng quát, ROUGE-2 để xem bảo toàn cụm từ, `gen_len` để kiểm tra độ dài, sau đó bắt buộc đọc prediction mẫu.

### 16. Khi nào biết mô hình overfit?

Train loss tiếp tục giảm nhưng validation loss tăng hoặc validation ROUGE không tăng; prediction validation có thể kém trung thành hơn.

### 17. Khi nào nên dừng train?

Khi validation metric ngừng cải thiện qua nhiều lần đánh giá, hoặc early stopping kích hoạt.

### 18. Tại sao cùng config nhưng kết quả có thể hơi khác?

GPU operation, thứ tự dữ liệu, dropout và một số kernel có tính không hoàn toàn deterministic. Seed giảm biến động nhưng không luôn loại bỏ hoàn toàn.

### 19. `resolved_config.json` quan trọng thế nào?

Nó ghi config thực tế sau override. Khi notebook thay đổi config lúc chạy, file YAML gốc không đủ để biết run đã dùng gì.

### 20. Muốn hiểu dự án nhanh nhất nên bắt đầu đâu?

Đọc [05 - Bản đồ dự án](05_ban_do_du_an.md), sau đó [06 - Transformer scratch](06_transformer_scratch_kien_truc.md), [08 - Pretrained seq2seq](08_pretrained_seq2seq.md) và [10 - Luồng code pretrained](10_luong_code_pretrained.md).

