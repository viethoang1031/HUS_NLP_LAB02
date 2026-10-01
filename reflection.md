# Reflection

### 1. Nếu tăng n, mô hình nhận thêm thông tin gì?

N-gram bậc n dùng tối đa n−1 từ ngay trước từ cần dự đoán. Bigram nhìn một từ trước; trigram nhìn hai từ trước.

### 2. Tại sao tăng n làm sparsity tăng?

Context dài có nhiều tổ hợp khả dĩ hơn. Corpus hữu hạn không chứa đủ mọi tổ hợp, nên mỗi tổ hợp quan sát được thường có ít lần lặp hơn và nhiều n-gram chưa từng thấy hơn.

### 3. Tại sao smoothing cần thiết?

MLE gán xác suất 0 cho n-gram chưa xuất hiện. Khi đó cả câu chứa n-gram ấy có xác suất 0 và perplexity vô hạn. Smoothing phân phối một phần xác suất cho sự kiện chưa thấy.

### 4. Perplexity đo điều gì?

Perplexity đo mức độ bất ngờ trung bình của mô hình trên chuỗi token đã đánh giá. Thấp hơn thường nghĩa là mô hình gán xác suất cao hơn cho dữ liệu đó, nếu preprocessing và cách tính giống nhau.

### 5. Perplexity thấp hơn có luôn tạo văn bản tốt hơn với con người không?

Không. Perplexity đo khả năng dự đoán token, không trực tiếp đo tính đúng sự thật, mạch lạc dài hạn, hữu ích hay phù hợp với ý định người đọc. Nó cũng phụ thuộc domain và cách token hóa.

### 6. N-gram thất bại ở đâu khi so với cách con người hiểu ngôn ngữ?

N-gram chỉ giữ một context cục bộ cố định và dựa trên tần suất. Nó khó nắm phụ thuộc xa, nghĩa, kiến thức nền, ý định và cấu trúc của toàn văn bản; dữ liệu thưa cũng khiến nhiều câu hợp lý nhận điểm thấp.

### 7. Với context dài 100 từ, trigram có dùng thông tin của 97 từ đầu không?

Không. Trigram chỉ điều kiện theo hai từ ngay trước từ đang dự đoán; 97 từ xa hơn không đi vào xác suất trigram.
