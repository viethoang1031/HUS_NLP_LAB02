# Error analysis — ví dụ minh họa

Mô hình trigram Laplace, `unk_cutoff=1`. Chia 1.000 documents theo document với seed 42: 80% train, 10% validation, 10% test. Mỗi context gồm hai từ; xác suất bên dưới là xác suất của từ mô hình dự đoán hoặc từ thực tế sau context. Đây là một số ví dụ từ test split, không phải kết quả của một phiên tương tác khác.

## Hai dự đoán đúng

1. **Context:** `clue is`  
   **Model prediction:** `a`  
   **Expected / actual:** `a`  
   **Probability:** `P(a | clue is) = 0.00007439`  
   **Nguyên nhân:** Mô hình đã thấy cách tiếp nối này trong dữ liệu train. Xác suất tuyệt đối nhỏ một phần vì vocabulary có 91.656 token và Laplace chia xác suất cho nhiều khả năng.

2. **Context:** `one type`  
   **Model prediction:** `of`  
   **Expected / actual:** `of`  
   **Probability:** `P(of | one type) = 0.00014877`  
   **Nguyên nhân:** Trigram khớp đúng tổ hợp context hai từ đã học. Context cụ thể mang lại thông tin hữu ích trong trường hợp này.

## Hai dự đoán sai

1. **Context:** `type of`  
   **Model prediction:** `person`  
   **Expected / actual:** `party`  
   **Probability:** `P(person | type of) = 0.00029665`; `P(party | type of) = 0.00007416`  
   **Nguyên nhân:** Corpus có thể chứa nhiều câu với `type of person`; trigram chọn phương án phổ biến trong dữ liệu train. Mô hình count-based không biết ý định cụ thể trong câu test.

2. **Context:** `of party`  
   **Model prediction:** `leaders`  
   **Expected / actual:** `ring`  
   **Probability:** `P(leaders | of party) = 0.00014877`; `P(ring | of party) = 0.00007438`  
   **Nguyên nhân:** Đây là context thưa; nhiều khả năng tổ hợp đầy đủ `of party ring` chưa thấy hoặc rất hiếm trong train. Laplace cho sự kiện chưa thấy xác suất dương nhưng không có thông tin ngữ nghĩa để ưu tiên đúng continuation.

## Nhận xét

Kết quả phụ thuộc corpus và split. Hai lỗi gợi ý sparsity và thiếu ngữ cảnh rộng hơn; smoothing tránh xác suất 0 nhưng không tự học nghĩa. Nên thử thêm contexts của bài, ghi actual next word sau khi xem dữ liệu, và kiểm tra lại nếu dùng seed hoặc cutoff khác.
