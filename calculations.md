# Tính toán bằng tay - Mục 7 đến 11

## Quy ước

- Phân biệt chữ hoa và chữ thường.
- Không thêm token bắt đầu/kết thúc câu (`<s>`, `</s>`), vì đề bài không sử dụng các token này trong công thức.
- Với bigram MLE:

  $$P(w_i\mid w_{i-1})=\frac{C(w_{i-1},w_i)}{C(w_{i-1})}.$$

## 7. Bài tập tính toán

Corpus:

```text
the cat eats fish
the cat likes fish
the dog eats meat
```

### Bài 1 - Unigram

#### 1. Vocabulary

$$\mathcal{V}=\{\text{the, cat, eats, fish, likes, dog, meat}\}.$$

Do đó kích thước vocabulary là:

$$V=|\mathcal{V}|=7.$$

#### 2. Tổng số token

Mỗi câu có 4 token và corpus có 3 câu, nên:

$$N=4+4+4=12.$$

Bảng số lần xuất hiện:

| Từ $w$ | $C(w)$ | $P(w)=C(w)/N$ |
|---|---:|---:|
| `the` | 3 | $3/12=1/4=0.25$ |
| `cat` | 2 | $2/12=1/6\approx0.1667$ |
| `eats` | 2 | $2/12=1/6\approx0.1667$ |
| `fish` | 2 | $2/12=1/6\approx0.1667$ |
| `likes` | 1 | $1/12\approx0.0833$ |
| `dog` | 1 | $1/12\approx0.0833$ |
| `meat` | 1 | $1/12\approx0.0833$ |

#### 3. Các xác suất được yêu cầu

$$P(\text{the})=\frac{3}{12}=\frac14=0.25.$$

$$P(\text{cat})=\frac{2}{12}=\frac16\approx0.1667.$$

$$P(\text{fish})=\frac{2}{12}=\frac16\approx0.1667.$$

$$P(\text{dog})=\frac{1}{12}\approx0.0833.$$

#### 4. Kiểm tra tổng xác suất

$$
\begin{aligned}
\sum_{w\in\mathcal{V}}P(w)
&=\frac{3+2+2+2+1+1+1}{12}\\
&=\frac{12}{12}=1.
\end{aligned}
$$

Vậy phân phối unigram đã được chuẩn hóa.

### Bài 2 - Bigram

Các bigram trong corpus và số lần xuất hiện:

| Bigram | Count |
|---|---:|
| `the cat` | 2 |
| `cat eats` | 1 |
| `eats fish` | 1 |
| `cat likes` | 1 |
| `likes fish` | 1 |
| `the dog` | 1 |
| `dog eats` | 1 |
| `eats meat` | 1 |

Vì $C(\text{the})=3$ và $C(\text{cat})=2$:

$$P(\text{cat}\mid\text{the})
=\frac{C(\text{the cat})}{C(\text{the})}
=\frac23\approx0.6667.$$

$$P(\text{dog}\mid\text{the})
=\frac{C(\text{the dog})}{C(\text{the})}
=\frac13\approx0.3333.$$

$$P(\text{eats}\mid\text{cat})
=\frac{C(\text{cat eats})}{C(\text{cat})}
=\frac12=0.5.$$

$$P(\text{likes}\mid\text{cat})
=\frac{C(\text{cat likes})}{C(\text{cat})}
=\frac12=0.5.$$

Với context `the`, hai từ kế tiếp quan sát được là `cat` và `dog`. Vì mỗi lần `the` làm context đều phải có đúng một từ theo sau nên:

$$
\sum_{w\in\mathcal{V}}P(w\mid\text{the})
=\frac{\sum_w C(\text{the},w)}{C(\text{the})}
=\frac{C(\text{the})}{C(\text{the})}=1.
$$

Cụ thể:

$$P(\text{cat}\mid\text{the})+P(\text{dog}\mid\text{the})
=\frac23+\frac13=1,$$

còn các từ khác có xác suất bằng 0. Kết luận này giả sử mọi lần xuất hiện của context đều có một token kế tiếp và vocabulary chứa đầy đủ các khả năng đó.

### Bài 3 - Xác suất câu

Với câu `the cat eats fish`:

$$P(\text{the})=\frac{3}{12}=\frac14,$$

$$P(\text{cat}\mid\text{the})=\frac23,$$

$$P(\text{eats}\mid\text{cat})=\frac12,$$

$$P(\text{fish}\mid\text{eats})
=\frac{C(\text{eats fish})}{C(\text{eats})}
=\frac12.$$

Do đó:

$$
\begin{aligned}
P(\text{the cat eats fish})
&=P(\text{the})P(\text{cat}\mid\text{the})
  P(\text{eats}\mid\text{cat})P(\text{fish}\mid\text{eats})\\
&=\frac14\cdot\frac23\cdot\frac12\cdot\frac12\\
&=\frac1{24}\approx0.04167.
\end{aligned}
$$

Nếu nối thêm từ $w$ vào một câu $S$ thì:

$$P(S,w)=P(S)P(w\mid\text{từ cuối của }S).$$

Vì $0\le P(w\mid\cdot)\le1$, xác suất của chuỗi mới **không thể lớn hơn** xác suất của chuỗi cũ. Nó giảm nếu xác suất có điều kiện nhỏ hơn 1 và chỉ giữ nguyên nếu xác suất đó bằng 1. Vì vậy không nên dùng xác suất chuỗi thô để so sánh trực tiếp các câu có độ dài khác nhau.

### Bài 4 - Sentence ranking

Với $S_1=$ `the cat eats fish`:

$$P(S_1)=\frac1{24}.$$

Với $S_2=$ `the dog eats fish`:

$$P(\text{dog}\mid\text{the})=\frac13,$$

$$P(\text{eats}\mid\text{dog})
=\frac{C(\text{dog eats})}{C(\text{dog})}
=\frac11=1,$$

và $P(\text{fish}\mid\text{eats})=1/2$. Vì thế:

$$
\begin{aligned}
P(S_2)
&=\frac14\cdot\frac13\cdot1\cdot\frac12\\
&=\frac1{24}\approx0.04167.
\end{aligned}
$$

Kết luận: hai câu được mô hình bigram MLE xếp hạng **bằng nhau**:

$$P(S_1)=P(S_2)=\frac1{24}.$$

## 8. Zero Probability

Xét trường hợp giả định corpus không chứa bigram `cat eats`. Khi đó:

$$C(\text{cat eats})=0,$$

nên theo MLE:

$$P(\text{eats}\mid\text{cat})
=\frac{C(\text{cat eats})}{C(\text{cat})}=0.$$

Xác suất câu `the cat eats fish` sẽ chứa thừa số này:

$$
P(\text{the cat eats fish})
=P(\text{the})P(\text{cat}\mid\text{the})
\underbrace{P(\text{eats}\mid\text{cat})}_{0}
P(\text{fish}\mid\text{eats})=0.
$$

Chỉ một bigram chưa từng xuất hiện cũng làm xác suất của cả câu bằng 0. Đây là **zero-frequency problem**.

## 9. Bài tập suy luận trước khi smoothing

Corpus:

```text
I like NLP
I like AI
I study NLP
```

Ta có $C(\text{study})=1$ nhưng bigram `study AI` không xuất hiện.

1. Count:

   $$C(\text{study AI})=0.$$

2. Xác suất MLE:

   $$P_{\mathrm{MLE}}(\text{AI}\mid\text{study})
   =\frac{C(\text{study AI})}{C(\text{study})}
   =\frac01=0.$$

3. Mọi câu có chứa `study AI` sẽ có xác suất bigram MLE bằng 0, vì tích xác suất của câu chứa một thừa số bằng 0.

4. Kết quả này **không** có nghĩa mô hình biết rằng `study AI` chắc chắn không thể xảy ra. Nó chỉ có nghĩa bigram đó chưa xuất hiện trong corpus huấn luyện nhỏ. Nói cách khác, *không quan sát thấy* không đồng nghĩa với *bất khả thi*.

## 10. Smoothing

Laplace (add-one) smoothing được tính bởi:

$$
P_{\mathrm{Laplace}}(w\mid h)
=\frac{C(h,w)+1}{C(h)+V}.
$$

Ta cộng 1 cho **mọi** từ trong vocabulary, nên tổng count giả ở tử số tăng thêm $V$ và mẫu số phải đổi từ $C(h)$ thành $C(h)+V$. Nhờ đó một bigram chưa quan sát có xác suất dương:

$$P_{\mathrm{Laplace}}(w\mid h)=\frac1{C(h)+V}>0$$

khi $C(h,w)=0$. Việc chuẩn hóa vẫn được bảo toàn:

$$
\sum_{w\in\mathcal{V}}P_{\mathrm{Laplace}}(w\mid h)
=\frac{\sum_w(C(h,w)+1)}{C(h)+V}
=\frac{C(h)+V}{C(h)+V}=1.
$$

## 11. Bài tập tính smoothing

Cho:

$$C(\text{cat})=10,\qquad V=5.$$

### Trường hợp 1: $C(\text{cat eats})=0$

$$
\begin{aligned}
P_{\mathrm{Laplace}}(\text{eats}\mid\text{cat})
&=\frac{C(\text{cat eats})+1}{C(\text{cat})+V}\\
&=\frac{0+1}{10+5}\\
&=\frac1{15}\approx0.0667.
\end{aligned}
$$

So với MLE bằng 0, smoothing đã cấp một lượng xác suất dương cho bigram chưa quan sát.

### Trường hợp 2: $C(\text{cat eats})=3$

$$
\begin{aligned}
P_{\mathrm{Laplace}}(\text{eats}\mid\text{cat})
&=\frac{3+1}{10+5}\\
&=\frac4{15}\approx0.2667.
\end{aligned}
$$

Trong khi đó, xác suất MLE tương ứng là:

$$P_{\mathrm{MLE}}(\text{eats}\mid\text{cat})=\frac3{10}=0.3.$$

Vậy Laplace smoothing làm xác suất của bigram thường gặp này giảm từ $0.3$ xuống khoảng $0.2667$ để dành xác suất cho các khả năng khác, bao gồm cả những bigram chưa quan sát.

Tổng quát, nếu $C(\text{cat},w)=k$ thì:

$$P_{\mathrm{Laplace}}(w\mid\text{cat})=\frac{k+1}{15},$$

thay vì $P_{\mathrm{MLE}}(w\mid\text{cat})=k/10$. Vì add-one được áp dụng cho toàn bộ vocabulary, smoothing thay đổi **toàn bộ phân phối**, không chỉ các xác suất bằng 0. Xác suất của các bigram chưa thấy tăng từ 0 lên $1/15$; phần khối lượng xác suất đó được tái phân bổ từ các bigram đã quan sát, đặc biệt làm giảm xác suất của các bigram có count lớn.

## 18. Bài tập tính Perplexity

Chuỗi $W=(w_1,w_2,w_3)$ gồm $N=3$ token. Theo mô hình bigram:

$$P(W)=P(w_1)P(w_2\mid w_1)P(w_3\mid w_2),$$

$$PP(W)=P(W)^{-1/N}=P(W)^{-1/3}.$$

### Trường hợp 1: $P(w_2\mid w_1)=0.25$

Cho $P(w_1)=0.5$ và $P(w_3\mid w_2)=0.5$, ta có:

$$
P(W)=0.5\times0.25\times0.5
=0.0625=\frac1{16}.
$$

Do đó:

$$
PP(W)=\left(\frac1{16}\right)^{-1/3}
=\sqrt[3]{16}\approx2.5198.
$$

### Trường hợp 2: $P(w_2\mid w_1)=0.1$

Giữ nguyên hai xác suất còn lại:

$$
P(W)=0.5\times0.1\times0.5
=0.025=\frac1{40}.
$$

Do đó:

$$
PP(W)=\left(\frac1{40}\right)^{-1/3}
=\sqrt[3]{40}\approx3.4200.
$$

### Vì sao chỉ một xác suất nhỏ cũng có thể làm perplexity thay đổi đáng kể?

Xác suất chuỗi là tích của các xác suất thành phần. Khi một thành phần giảm từ $0.25$ xuống $0.1$, xác suất chuỗi giảm còn $0.1/0.25=0.4$ lần giá trị ban đầu. Perplexity là nghịch đảo của trung bình nhân các xác suất, nên xác suất chuỗi giảm làm perplexity tăng:

$$
\frac{PP_{\text{mới}}}{PP_{\text{cũ}}}
=\left(\frac{0.1}{0.25}\right)^{-1/3}
=\sqrt[3]{2.5}\approx1.3572.
$$

Vậy chỉ thay đổi một xác suất đã làm perplexity tăng khoảng $35.72\%$, từ $2.5198$ lên $3.4200$.

Trong công thức log tương đương:

$$
PP(W)=\exp\left(-\frac13\sum_{i=1}^{3}\log p_i\right),
$$

một xác suất rất nhỏ tạo ra giá trị $-\log p_i$ lớn, làm tăng mức độ bất ngờ trung bình của mô hình. Ảnh hưởng của một token được chia cho tổng số token $N$, nên trong chuỗi ngắn như bài này, thay đổi đó thể hiện rõ hơn. Nếu một xác suất bằng $0$, xác suất chuỗi bằng $0$ và perplexity là vô hạn.
