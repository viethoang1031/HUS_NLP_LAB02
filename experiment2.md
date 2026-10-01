# Phần 16 — MLE vs Laplace smoothing

## Cấu hình

- Corpus: `data/c4-train.00000-of-01024-30K.json`, toàn bộ 30.000 documents.
- Shuffle theo document với seed 42; train/validation/test = 80%/10%/10% (24.000/3.000/3.000 documents).
- Tách câu bằng English Punkt, lowercase, giữ từ/số Unicode và apostrophe nội bộ. Không nối n-gram qua ranh giới câu và không thêm BOS/EOS.
- Vocabulary chỉ học từ training. Từ có count <= 1 trong training được thay bằng `<UNK>`; từ OOV ở validation/test được ánh xạ về cùng token này. Mọi mô hình dùng cùng split, vocabulary và preprocessing.
- Từ đầu câu dùng unigram; từ thứ hai dùng bigram; các từ tiếp theo dùng bậc của mô hình. Không backoff khi gặp context chưa thấy.
- PPL = exp(-tổng log xác suất / tổng số token). Không lấy trung bình PPL của từng câu. Nếu có một sự kiện xác suất 0, corpus PPL là vô hạn.

## Kết quả

Số liệu dưới đây lấy từ lần chạy toàn corpus trong phiên làm việc, được lưu trong `results.csv` (làm tròn 6 chữ số thập phân). Vocabulary sau xử lý `<UNK>` có 91.656 token.

| Model | Smoothing | Train PPL | Validation PPL | Test PPL |
|---|---|---:|---:|---:|
| Bigram | MLE | 196.263959 | inf | inf |
| Bigram | Laplace | 3648.453327 | 4244.946203 | 4212.145647 |
| Trigram | MLE | 16.890113 | inf | inf |
| Trigram | Laplace | 16918.661581 | 26548.846895 | 26445.568152 |

## Trả lời: Tại sao kết quả lại như vậy?

MLE dùng trực tiếp tần suất quan sát. Các sự kiện trong training đều đã được đếm, nên không có zero probability trên training. Trigram sử dụng thêm một từ context, phân biệt các chuỗi tốt hơn và đạt training PPL thấp hơn bigram trong lần chạy này. Training PPL thấp chưa đủ để kết luận mô hình tốt trên dữ liệu mới.

Trên validation/test, các tổ hợp từ chưa từng xuất hiện trong training được MLE gán xác suất 0. Chỉ một sự kiện như vậy cũng làm tổng log probability bằng -inf và PPL bằng inf. `<UNK>` giải quyết từ ngoài vocabulary, nhưng không giải quyết mọi tổ hợp bigram/trigram chưa thấy.

Laplace cộng 1 cho mọi từ trong vocabulary và cộng V vào mẫu số, nên cả context chưa thấy cũng có phân phối xác suất dương. Vì OOV được ánh xạ sang `<UNK>`, validation/test PPL hữu hạn. Smoothing thay đổi toàn bộ phân phối, làm giảm xác suất của nhiều sự kiện phổ biến để dành xác suất cho sự kiện chưa thấy; do đó training PPL tăng.

Vocabulary lớn (V = 91.656) khiến add-one phân bổ nhiều khối lượng xác suất cho các khả năng chưa quan sát. Với trigram, context hai từ thường có ít quan sát hơn context một từ, nên phân phối bị kéo mạnh về phân phối đều. Điều này phù hợp với trigram Laplace có PPL cao hơn bigram Laplace trong cả ba split. Kết quả cho thấy context dài hơn và smoothing không bảo đảm PPL thấp hơn; hiệu quả phụ thuộc độ thưa dữ liệu, vocabulary và phương pháp smoothing.

## Chạy lại

Từ thư mục gốc repository:

```powershell
python WEEK2\implementation.py --experiment2 --seed 42 --unk-cutoff 1 --output WEEK2\results.csv
```

Chạy nhanh trên mẫu nhỏ (kết quả khác bảng toàn corpus):

```powershell
python WEEK2\implementation.py --experiment2 --max-documents 100
```

## AI assistance statement

- Tool: Codex.
- Purpose: bổ sung implementation, thực nghiệm MLE/Laplace và giải thích kết quả theo yêu cầu người dùng.
- What was generated: hàm PPL, chia split, thực nghiệm, xuất CSV và báo cáo.
- What was modified: `implementation.py`, `experiments.ipynb`; thêm `results.csv` và `experiment2.md`.
- Verification: ví dụ đếm tay, công thức PPL, OOV/UNK, chạy mẫu và chạy toàn corpus. Bảng số liệu lấy từ output của lần chạy toàn corpus.
