# LAB 02 — N-gram Language Models

## Nội dung

- `calculations.md`: tính toán tay các mục 7–11, 15 và 18.
- `prediction.md`: dự đoán trước thực nghiệm ở mục 12; tự sửa sau khi chạy nếu cần.
- `ngram_lm.py`: unigram, bigram, trigram; MLE/Laplace; log probability, perplexity; CLI cho các ứng dụng mục 20–21.
- `experiments.ipynb`: thống kê corpus (13), smoothing (16), perplexity (19), và context length (23).
- `results.csv`, `experiment2.md`: kết quả và giải thích thí nghiệm mục 16.
- `experiment3_results.csv`: kết quả mục 19.
- `context_length_results.csv`: kết quả đã chạy cho mục 23.
- `error_analysis.md`, `reflection.md`: cần hoàn thành độc lập theo yêu cầu môn học.

## Chạy từ thư mục gốc repository

```powershell
python WEEK2\ngram_lm.py --experiment2 --output WEEK2\results.csv
```

Chạy ứng dụng tương tác dự đoán từ tiếp theo và xếp hạng candidates:

```powershell
python WEEK2\ngram_lm.py --interactive --n 3 --smoothing laplace --max-documents 100
```

Bỏ `--max-documents 100` để dùng toàn bộ corpus 30K documents. Chế độ tương tác có menu dự đoán, xếp hạng câu tiếp nối và xem lịch sử. Muốn lưu lịch sử phiên chạy:

```powershell
python WEEK2\ngram_lm.py --interactive --n 3 --smoothing laplace --application-output WEEK2\application_results.csv
```

Mở `WEEK2/experiments.ipynb` và chạy các cell từ trên xuống. Cell Experiment 3 tái tạo `experiment3_results.csv`; cell Experiment 4 tạo `context_length_results.csv`.

## Quy ước

Preprocessing dùng English Punkt để tách câu, lowercase, giữ từ/số Unicode và apostrophe nội bộ. Không thêm BOS/EOS; không nối n-gram qua ranh giới câu. Các experiment học vocabulary trên train. Từ có count nhỏ hơn hoặc bằng cutoff được ánh xạ sang `<UNK>` để xử lý OOV. PPL là `exp(-tổng log probability / tổng token)`.

## AI assistance statement

- Tool: Codex.
- Purpose: hỗ trợ implementation các n-gram LM, log probability, perplexity, MLE/Laplace, ứng dụng CLI, thực nghiệm, error analysis và reflection.
- What was generated: một phần code trong `ngram_lm.py`, các cell Experiment 2–4, hướng dẫn chạy, ví dụ error analysis và câu trả lời reflection.
- What was modified: code được điều chỉnh để khớp preprocessing, vocabulary và phép chia dữ liệu; ví dụ error analysis được đo trên test split riêng.
- Verification: chạy Experiment 2–4 trên toàn corpus; các ví dụ error analysis được đo với trigram Laplace trên split từ 1.000 documents, seed 42.

