# Thiết kế v4, bản nháp bản ghi trước, viết ngày 04/10/2026, chưa duyệt, chưa tính số nào

Trạng thái: nháp. Chưa tính bất kỳ con số k-shot nào trên fold nội bộ, vì đó chính là dữ liệu của cổng. Khi tác giả duyệt, đổi tên file bỏ hậu tố `_DRAFT`, ghi sha256 vào FACTS, và từ đó không sửa.

Đây là chiến dịch thứ tư. Ba chiến dịch trước trên nhiệm vụ chuẩn của EngageNet (v1 21/09, v2 27/09, v3 02/10) có 13 thành phần ghi trước, không thành phần nào qua cổng. Bài viết từ v4 phải liệt kê đủ cả bốn vòng.

## 0. Khung bài: đây là một nhiệm vụ khác, không phải mô hình tốt hơn trên nhiệm vụ chuẩn

v4 giả định lúc triển khai có k clip của chính học sinh đó được giáo viên gán nhãn (một buổi ghi danh). Nhiệm vụ chuẩn của EngageNet không cho điều này. Vì vậy:

- Không so được với 67,61, 69,29, 73,58 hay bất kỳ số nào trên bảng xếp hạng. Không tuyên bố gì về bảng xếp hạng.
- So sánh duy nhất là với mô hình không hiệu chỉnh của chính nhóm, trên đúng cùng các clip truy vấn. So sánh này ghép cặp được và kiểm định được, vì nhóm có dự đoán từng clip.
- Tài liệu đi trước phải nêu: Churaev & Savchenko (AIED 2024, DOI 10.1007/978-3-031-64315-6_29) tinh chỉnh bộ phân loại nhị phân bằng một ít video có nhãn của từng người dùng, trên bộ dữ liệu riêng của họ, báo tăng 10–15 % độ chính xác. Alarefah, Kammoun Jarraya & Abuzinadah (Computers 2025, DOI 10.3390/computers14030109) dùng học ít mẫu kiểu mạng nguyên mẫu trên EngageNet và tự báo đạt mức cao nhất; chưa đọc được toàn văn (trang MDPI chặn truy cập tự động), nên chưa biết giao thức và con số.
- Phép hiệu chỉnh theo tiên nghiệm nhãn là kỹ thuật đã có trong tài liệu về dịch chuyển phân bố nhãn; tài liệu gốc phải tìm và kiểm trước khi viết.

## 1. Luận điểm và bằng chứng có trước

Phân bố nhãn khác nhau rất mạnh giữa các học sinh: sàn lớp đa số của ba fold nội bộ là 57,92 %, 43,91 % và 41,71 % vì các học sinh có nhãn gần như dồn vào một mức bị xếp chung fold; độ chính xác theo từng học sinh trải từ 0,00 đến 1,00 (FACTS mục 12.3, ghi 28/09, trước v3). Oracle dùng tiên nghiệm nhãn thật của từng học sinh cho +0,058 accuracy; bộ ước lượng không cần nhãn (phân bố dự đoán trung bình của học sinh) cho kết quả âm. Thông tin về tiên nghiệm của từng học sinh vì vậy phải đến từ nhãn. Nguồn rẻ nhất là vài clip giáo viên gán nhãn.

Oracle +0,058 dùng mọi clip của học sinh, kể cả các clip cùng một bản ghi với clip được chấm, mà nhãn liền kề trùng nhau 68 % (0,6834 / 0,6513 / 0,6923). Một phần oracle vì vậy có thể chỉ là tương quan nhãn theo thời gian trong cùng bản ghi. Giao thức dưới đây tách bản ghi ghi danh khỏi bản ghi truy vấn để loại phần đó.

## 2. Giao thức

Cấu trúc dữ liệu, đếm ngày 04/10 từ tên clip `subject_<id>_<session>_vid_<video>_<idx>`: train 91 học sinh, 247 bản ghi, 78 học sinh có từ hai bản ghi; test 26 học sinh, 75 bản ghi, 24 học sinh có từ hai bản ghi.

- Học sinh đủ điều kiện: có từ hai bản ghi. Học sinh chỉ có một bản ghi bị loại và được đếm, báo cáo.
- Bản ghi ghi danh: bản ghi đầu tiên theo thứ tự (session, video). Tập hỗ trợ: k clip rút ngẫu nhiên không lặp từ bản ghi ghi danh (ít hơn k clip thì lấy hết). Tập truy vấn: mọi clip thuộc các bản ghi còn lại của học sinh đó.
- R = 200 lần rút tập hỗ trợ cho mỗi học sinh, `numpy.random.default_rng(0)`, cố định trước khi tính bất cứ gì.
- k có cổng: k = 5. Đường cong k ∈ {1, 3, 5, 10, 20} được báo cáo, không có cổng.

Dự đoán nền:
- Fold nội bộ: nhánh C của v3 (CMS), trung bình xác suất ba seed cho mỗi clip, `preds/ft_v3_C_f<k>_s<s>_train.npz`.
- Test: tổ hợp ba seed SVFAP của v1, `preds/ft_g8_svfap_s{0,1,2}_trainonly_on_test_train.npz` cho giao thức train và `preds/ft_g8_svfap_s{0,1,2}_train-val.npz` cho giao thức train + val. Không dùng tổ hợp bốn thành viên (có MAE-DFER-B) của số chính v1, để mô hình nền ở test cùng loại với mô hình nền đã dùng để phát triển trên fold nội bộ.
- π_train: phân bố nhãn của tập huấn luyện đã sinh ra mô hình nền (hai fold còn lại cho fold nội bộ; train, hoặc train + val, cho test).

## 3. Các phương pháp, cố định trước

- M0, không hiệu chỉnh: argmax p(y | x).
- B1, đa số của tập hỗ trợ: gán nhãn xuất hiện nhiều nhất trong k clip hỗ trợ cho mọi clip truy vấn; hoà thì chọn nhãn có π_train lớn hơn.
- M1, mô hình đề xuất, hiệu chỉnh tiên nghiệm theo học sinh: p'(y | x) ∝ p(y | x) · π̂_s(y) / π_train(y), với π̂_s(y) = (n_s(y) + α · π_train(y)) / (k + α), n_s(y) là số clip hỗ trợ mang nhãn y. α → ∞ trả về M0; α → 0 tin hoàn toàn vào tập hỗ trợ.
- Chọn α trong lưới {1, 2, 5, 10, 20, 50, 100}: để chấm fold f, chọn α cho macro-F1 trung bình lớn nhất ở k = 5 trên hai fold còn lại (cùng giao thức rút); để chấm test, chọn trên cả ba fold nội bộ. Không dùng nhãn của fold đang chấm.

## 4. Thước đo

Với mỗi lần rút: macro-F1, UAR và accuracy gộp trên mọi clip truy vấn của các học sinh đủ điều kiện trong fold; lấy trung bình qua 200 lần rút. Hiệu số: M1 − M0 và M1 − B1. Khoảng tin cậy trong fold: bootstrap theo học sinh, B = 2 000, seed 0, các lần rút giữ cố định, mỗi mẫu bootstrap tính lại trung bình qua lần rút.

## 5. Cổng G-P, trên fold nội bộ, k = 5

M1 qua khi đủ bốn điều kiện:

1. Trung bình ba fold của macro-F1 (M1 − M0) ≥ +0,015.
2. Ở ít nhất hai fold, cận dưới của khoảng bootstrap theo học sinh cho macro-F1 (M1 − M0) > 0.
3. Trung bình ba fold của macro-F1 (M1 − B1) > 0.
4. Trung bình ba fold của UAR (M1 − M0) ≥ 0.

## 6. Các cách đọc, ghi trước

| Kết cục | Đọc là | Hệ quả |
|---|---|---|
| M1 qua G-P | tiên nghiệm của học sinh chuyển được giữa các bản ghi, và k = 5 clip đủ để ước lượng nó | chấm test một lần cho mỗi giao thức, mục 7 |
| M1 không hơn M0 | tiên nghiệm theo học sinh không chuyển được giữa các bản ghi; phần lớn oracle +0,058 là tương quan trong cùng bản ghi | không có bài mô hình |
| B1 bằng hoặc hơn M1 | nhãn của giáo viên tự nó mang thông tin, mô hình không thêm gì | không có bài mô hình |

## 7. Nếu G-P qua: đánh giá trên test

Áp M1 đã khoá (α chọn trên ba fold nội bộ) lên dự đoán test v1 của cả hai giao thức, cùng giao thức rút, k = 5, R = 200. Tuyên bố suy luận chính: khoảng bootstrap theo học sinh, ghép cặp, 95 %, cho accuracy và macro-F1 của (M1 − M0) trên các học sinh test đủ điều kiện, B = 2 000, seed 0. Báo kèm M1 − B1 và đường cong k.

Phải khai: bước này đọc nhãn test theo một giao thức mới (k clip hỗ trợ cho mỗi học sinh), sau khi nhãn test đã được dùng để chấm v1. Phương pháp và mọi tham số khoá trên fold nội bộ trước khi đọc.

So sánh phụ, chỉ khi G-P qua, theo kiểu Churaev & Savchenko: tinh chỉnh đầu phân loại trên embedding của k clip hỗ trợ. Việc này cần trích embedding bằng GPU vài giờ; báo cáo như một mô hình đối chứng, không có cổng.

DAiSEE: chuyển giao thuần, không cổng, không kỳ vọng; bản ghi lấy từ `stem[:7]`.

## 8. Dự báo, ghi trước

| Đại lượng | Dự báo | Xác suất |
|---|---|---|
| macro-F1 (M1 − M0), k = 5, trung bình ba fold | 0,000 đến +0,030 | qua G-P: 30 % |
| M1 > B1 | | 75 % |

Ghi kèm: 13 trên 13 thành phần trước trượt cổng; dự báo của v2 và v3 đều lệch về phía lạc quan.

## 9. Nếu G-P trượt

Không đọc nhãn test. Không có bài mô hình. Kết quả fold nội bộ ghi đầy đủ, kể cả âm. Không thử biến thể khác của M1 trên bộ này nếu không có bản ghi trước mới.

## 10. Chi phí

Toàn bộ phần fold nội bộ chạy trên CPU từ các file dự đoán đã có, khoảng một giờ. Phần test cũng chạy trên CPU. Chỉ so sánh phụ ở mục 7 cần GPU.
