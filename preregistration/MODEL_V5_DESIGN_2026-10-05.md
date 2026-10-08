# Thiết kế v5, bản ghi trước, viết ngày 05/10/2026 trước khi chạy bất cứ gì

Tác giả yêu cầu ngày 05/10/2026: "Tôi vẫn muốn bài mô hình, chạy hướng còn lại". Hướng còn lại là tiền huấn luyện tiếp backbone kiểu MAE ở cửa sổ CMS. File này cố định mọi lựa chọn và ngưỡng trước lần chạy đầu tiên; sha256 của nó được ghi vào FACTS trước khi chạy. Mọi thay đổi sau đó là một file bổ sung có ngày.

Đây là chiến dịch thứ năm. Bốn chiến dịch trước (v1, v2, v3, v4) có 14 thành phần ghi trước, không thành phần nào qua cổng. Bài viết từ v5 phải liệt kê đủ cả năm vòng.

## 1. Luận điểm

SVFAP-B được tiền huấn luyện trên VoxCeleb2 với 16 khung bước 4. Cấu hình tốt nhất trên EngageNet lại là 16 khung bước 20 (CMS), tức backbone được tinh chỉnh ở một cửa sổ dài gấp năm lần cửa sổ nó đã học, trên một miền khác (webcam học tập so với phỏng vấn người nổi tiếng). v3 cho thấy giữ cửa sổ tiền huấn luyện mà chia clip thành đoạn thì thua CMS ở cả ba fold, nên cửa sổ dài là thứ cần giữ. Câu hỏi của v5: cho backbone học lại biểu diễn tự giám sát ở chính cửa sổ CMS và trên chính dữ liệu đích (không dùng nhãn) thì mô hình tinh chỉnh sau đó có tốt hơn không.

Hướng này chính là bước tương lai thứ ba bài PeerJ đã viết trước ở Article S1, mục C.7: "Continue the self-supervised pretraining of the backbone on the target corpus at the frame budget the downstream task will use."

## 2. Mô hình, cố định trước

Tiền huấn luyện tiếp (nhánh P):
- Khởi tạo: checkpoint tiền huấn luyện SVFAP đã phát hành, gồm cả bộ mã hoá và bộ giải mã (`weights/svfap/svfap_base_voxceleb2_checkpoint-99.pth`; nạp khớp hoàn toàn vào `PretrainVisionTransformer` với bộ mã hoá 512 chiều, 8 đầu, bộ giải mã 384 chiều, 6 đầu, sâu 4, đã kiểm ngày 05/10).
- Dữ liệu: đúng tập clip huấn luyện của lần tinh chỉnh tương ứng, không dùng nhãn. Fold nội bộ f: các clip của hai fold còn lại, dựng bằng đúng mã chia fold của `ft_video.py`. Không clip nào của fold đang chấm, của validation hay của test đi vào tiền huấn luyện.
- Lấy mẫu: 16 khung bước 20, đúng bộ lấy mẫu huấn luyện của `ft_video.py` (nhánh dài: điểm bắt đầu ngẫu nhiên và nhiễu ±10 khung; nhánh ngắn: `range(0, n, 20)` đệm khung cuối); cắt ngẫu nhiên 0,7–1,0, lật ngang, chuẩn hoá ImageNet.
- Mặt nạ ống, tỉ lệ 0,9; mất mát MSE trên điểm ảnh đã chuẩn hoá theo từng patch của các token bị che, đúng `engine_for_pretraining.py` của SVFAP.
- AdamW β = (0,9; 0,95), weight decay 0,05, lr 1e-4, khởi động 2 epoch, cosine về 1e-6, batch 32 clip (có thể 16 × tích luỹ 2 nếu thiếu bộ nhớ), 30 epoch, seed 0, autocast fp16.
- Đầu ra: bộ mã hoá đã thích nghi, lưu đúng định dạng checkpoint cũ để `ft_video.py` nạp qua cờ mới `--init-ckpt`.

Tinh chỉnh: đúng công thức nhánh C của v3 (16 khung bước 20, LoRA hạng 16 trên q, k, v, 12 epoch, khởi động 2, EMA 0,999, lr 1e-3, batch 16 × 2, nhãn mềm 0,1, năm cửa sổ chấm), chỉ khác điểm khởi tạo. Seed 0, 1, 2 ở mỗi fold.

Đối chứng: nhánh C của v3, chín lần chạy đã có, cùng môi trường, cùng mã (v3 cho thấy chạy lại nhánh C trùng từng số với v1). Không chạy lại.

## 3. Cổng G-D, cổng duy nhất quyết định có bài mô hình hay không

Nhánh P qua khi đủ ba điều kiện, macro-F1 là chỉ số chọn:
1. Trung bình ba fold của hiệu (trung bình seed của P trừ trung bình seed của C) ≥ +0,015.
2. Ở ít nhất hai fold, hiệu đó dương và lớn hơn khoảng seed lớn hơn của hai nhánh ở fold đó.
3. Hiệu UAR trung bình ba fold không âm.

## 4. Các cách đọc, ghi trước

| Kết cục | Đọc là | Hệ quả |
|---|---|---|
| P qua G-D | thích nghi biểu diễn tự giám sát với miền và cửa sổ đích cải thiện mô hình tinh chỉnh | đánh giá chính thức, mục 5 |
| P không khác C | lệch miền hay lệch cửa sổ giữa tiền huấn luyện và tinh chỉnh không phải nút thắt trên bộ này | không có bài mô hình |
| P thua C | tiền huấn luyện tiếp trên khoảng 5 400 clip làm hỏng biểu diễn đã học từ VoxCeleb2 | không có bài mô hình |

## 5. Nếu G-D qua: đánh giá chính thức

Tiền huấn luyện tiếp trên toàn bộ train (cho giao thức train) và trên train + val (cho giao thức train + val), cùng công thức. Tinh chỉnh ba seed cho mỗi giao thức. Chấm validation một lần (giao thức train) và test một lần cho mỗi giao thức bằng `--open-official`. Số chính: tổ hợp ba seed của P, trung bình xác suất.

Tuyên bố suy luận chính: khoảng bootstrap theo người, ghép cặp, 95 %, cho accuracy và macro-F1 của (tổ hợp ba seed P − tổ hợp ba seed SVFAP của v1, `preds/ft_g8_svfap_s{0,1,2}_*`), cùng 2 256 clip và 26 người, B = 2 000, seed 0. Chênh với 67,61, 69,29 và 73,58 vẫn là chênh giữa ước lượng điểm, không kiểm định.

Phải khai: tập test EngageNet đã được nhóm chấm cho v1, mỗi giao thức một lần; lần chấm của v5 là lần thứ hai cho mỗi giao thức, với mô hình khoá trước khi chấm.

DAiSEE: chuyển giao thuần, không cổng, không kỳ vọng; tiền huấn luyện tiếp trên train (và train + val) của DAiSEE, cửa sổ 16 khung bước 19 như cấu hình DAiSEE của v1, rồi tinh chỉnh ba seed mỗi giao thức.

## 6. Nếu G-D trượt

Không mở split chính thức nào. Không có bài mô hình. Kết quả ghi đầy đủ, kể cả âm. Không thử biến thể khác (số epoch, lr, tỉ lệ mặt nạ, cửa sổ) nếu không có bản ghi trước mới.

## 7. Dự báo, ghi trước

| Đại lượng | Dự báo | Xác suất |
|---|---|---|
| P − C, macro-F1 trung bình ba fold | −0,010 đến +0,020 | qua G-D: 15 % |
| P − C âm | | 40 % |

Ghi kèm: 14 trên 14 thành phần trước trượt cổng; dự báo của v2 và v3 lệch về phía lạc quan.

## 8. Vận hành

Mỗi epoch tiền huấn luyện và tinh chỉnh đều lưu điểm kiểm tra và chạy tiếp được sau khi khởi động lại; hàng đợi có khoá, chờ đủ bộ nhớ GPU, tự khởi động lại bằng mục `@reboot` và tự gỡ mục đó khi xong. Kiểm thử khói trước: tiền huấn luyện 2 epoch trên 64 clip, ngắt sau điểm kiểm tra đầu rồi chạy tiếp; tinh chỉnh 1 epoch trên 64 clip từ checkpoint vừa tạo. Không đọc điểm của kiểm thử khói.
