# Thiết kế mô hình v3, bản ghi trước, viết ngày 02/10/2026 trước khi chạy bất cứ gì

Trạng thái: bản nháp chờ tác giả duyệt. Chưa chạy lần nào. Máy chủ 192.168.207.98 không kết nối được lúc viết (SSH hết thời gian chờ). Khi tác giả duyệt, ghi sha256 của file này vào `FACTS_MODEL_PAPER.md`; mọi sửa sau đó phải là một mục bổ sung có ngày, không sửa phần trên.

Đây là lần đề xuất mô hình thứ ba trên EngageNet. Hai lần trước là chiến dịch v1 (`MODEL_PAPER_DESIGN_2026-09-21.md`) và v2 (`MODEL_V2_DESIGN_2026-09-27.md`), với mười hai thành phần ghi trước, không thành phần nào qua cổng. Bài viết từ v3 phải liệt kê đủ cả ba vòng.

Chưa đặt tên mô hình. Trong file này mô hình là "nhánh S". Chỉ đặt tên nếu nhánh S qua cổng.

## 1. Luận điểm

Nhãn EngageNet mô tả cả clip 10 giây. Hai công thức đã chạy, mỗi cái chỉ đáp ứng một nửa yêu cầu đó.

Công thức kế thừa (16 khung, bước 4) giữ đúng cấu hình tiền huấn luyện của SVFAP-B, đã kiểm từ `scripts/voxceleb2/pretrain_svfap_base.sh` của kho SVFAP: `--num_frames 16 --sampling_rate 4`. Nhưng mỗi mẫu huấn luyện là một cửa sổ 61 khung mang nhãn của cả clip. Lúc chấm, năm cửa sổ trải khắp clip được lấy trung bình xác suất (`linspace(0, n − 61, 5)` trong `ft_video.py`; `tta_windows` = 5 trong `FT_g1_svfap_full_train.json`). Clip chỉ được phủ ở bước chấm, không bao giờ ở bước huấn luyện.

Công thức CMS (16 khung, bước 20) đưa cả clip vào một lượt tính, nên giám sát khớp với nhãn. Nhưng nó kéo cửa sổ ra gấp năm lần cấu hình tiền huấn luyện.

Nhánh S làm cả hai việc: chia clip thành K = 5 cửa sổ 16 khung bước 4, cho cùng một backbone chạy trên từng cửa sổ, lấy trung bình logit của năm cửa sổ rồi tính mất mát với nhãn clip. Đây là nguyên lý đồng thuận theo đoạn của TSN (Wang et al., 2016) áp lên một backbone video 3D tiền huấn luyện kiểu MAE. Điểm mới không nằm ở nguyên lý mà ở câu hỏi nó trả lời trên engagement.

Câu không được viết: nhánh S "giữ cùng thang thời gian tính bằng giây" với tiền huấn luyện. Tốc độ khung của VoxCeleb2 chưa kiểm. Chỉ viết "giữ cấu hình tiền huấn luyện, 16 khung bước 4".

## 2. Ba nhánh, cố định trước

Công thức chung cho cả ba nhánh, giữ nguyên công thức đã khoá: SVFAP-B, LoRA hạng 16 trên q, k, v, 12 epoch trong đó 2 khởi động, EMA 0,999, lr 1e-3, nhãn mềm 0,1, cắt ngẫu nhiên 0,7–1,0 và lật ngang, batch hiệu dụng 32 clip (không phải 32 cửa sổ). Ba fold nội bộ theo người như mọi chiến dịch trước (`default_rng(12345)`, `fold = i % 3`). Ba seed 0, 1, 2 cho mỗi nhánh ở mỗi fold. Mọi nhánh chạy trong cùng một môi trường (`venv/bin/python`), kể cả nhánh C đã có số cũ.

| Nhánh | Huấn luyện | Chấm |
|---|---|---|
| R, kế thừa | một cửa sổ 16 × 4 mỗi clip mỗi epoch, điểm bắt đầu ngẫu nhiên trong [0, n − 61], nhiễu ±2 khung | năm cửa sổ `linspace(0, n − 61, 5)`, trung bình xác suất |
| C, CMS | 16 khung bước 20, đúng như các lần chạy v1 | như v1 |
| S, đồng thuận theo đoạn | năm cửa sổ 16 × 4 mỗi clip; cửa sổ k bắt đầu ở `linspace(0, n − 61, 5)[k]` cộng một số nguyên đều trong ±⌊(n − 61)/8⌋, kẹp vào [0, n − 61], nhiễu ±2 khung; một vùng cắt và một lần lật dùng chung cho cả năm cửa sổ của một clip; logit năm cửa sổ lấy trung bình rồi mới tính mất mát | đúng năm cửa sổ của R, trung bình xác suất |

R và S chấm trên cùng cửa sổ với cùng cách gộp, nên hai nhánh chỉ khác nhau ở cách giám sát lúc huấn luyện. Đây là so sánh một yếu tố sạch nhất dự án từng có.

Clip có n < 61 khung đi nhánh ngắn hiện có của `ft_video.py` cho cả R và S.

Kiểm thử khói trước khi chạy thật: `--max-train 64`, một epoch, chỉ để kiểm bộ nhớ và hình tensor. Không đọc điểm của lần kiểm thử khói.

## 3. Cổng, ghi trước

Chỉ số chọn là macro-F1, chỉ số phụ là UAR. Nếu hai chỉ số không cùng chiều thì cấu hình không được chọn và bài báo cáo sự bất đồng đó. Accuracy chỉ để đặt cạnh bảng xếp hạng.

Cổng G-S là cổng duy nhất quyết định có bài mô hình hay không. Nhánh S qua khi đủ ba điều kiện:

1. Trung bình ba fold của hiệu (trung bình seed của S trừ trung bình seed của C) ≥ +0,015 macro-F1.
2. Ở ít nhất hai fold, hiệu trung bình seed dương và lớn hơn khoảng seed lớn hơn của hai nhánh ở fold đó (khoảng seed = lớn nhất trừ nhỏ nhất của ba seed). Độ lệch seed đã biết là 0,0152, sát ngưỡng ở điều kiện 1; điều kiện này giữ để một lần "qua" không phải nhiễu.
3. Hiệu UAR trung bình ba fold không âm.

Nhánh R không có cổng. Nó là phép chẩn đoán và luôn được báo cáo.

## 4. Các cách đọc, ghi trước

| Kết cục | Đọc là | Hệ quả |
|---|---|---|
| S qua G-S | giám sát cả clip ở cấu hình tiền huấn luyện tốt hơn kéo giãn cửa sổ | đi tiếp mục 6, mở split chính thức |
| S ≈ C (không qua G-S), cả hai hơn R | cái quyết định là giám sát theo clip, không phải cấu hình thời gian của từng lượt tính | không có bài mô hình; báo như một phát hiện cơ chế |
| S thua C | một cửa sổ thưa phủ cả clip trong một lượt tính tốt hơn đồng thuận trên các cửa sổ chuẩn; tương tác xa trong cùng lượt tính quan trọng hơn cấu hình tiền huấn luyện | CMS đứng; không có bài mô hình |

Trường hợp S không hơn R cũng phải báo: khi đó giám sát theo clip lúc huấn luyện không thêm gì so với trung bình năm cửa sổ lúc chấm.

## 5. Dự báo, ghi trước

| Đại lượng | Dự báo | Xác suất |
|---|---|---|
| S − C, macro-F1 trung bình ba fold | −0,010 đến +0,020 | qua G-S: 25 % |
| S − R, macro-F1 trung bình ba fold | +0,010 đến +0,040 | S > R: 65 % |
| C chạy lại ở môi trường mới so với số v1 cùng fold, cùng seed | chênh dưới 0,010 | 70 % |

Ghi kèm: dự báo của v2 cho ngân sách 32 khung sai cả chiều (+0,010 đến +0,030, thực tế −0,0437), và mười hai trên mười hai thành phần trước không qua cổng. Kết cục có khả năng nhất là không có bài mô hình.

## 6. Nếu G-S qua: đánh giá chính thức

Thành phần phụ có điều kiện, quyết định xong trước khi mở split: thay trung bình logit bằng một lớp gộp chú ý trên năm embedding cửa sổ (`forward_features` của SVFAP, kèm embedding thứ tự cửa sổ học được). Cổng: ≥ +0,010 macro-F1 so với trung bình logit, cùng điều kiện 2 và 3 của G-S, ba seed ba fold. Trượt thì mô hình là trung bình logit. Tài liệu đi trước cho kiểu gộp này phải tìm và kiểm trước khi viết.

Huấn luyện nhánh S đã khoá với ba seed trên train và ba seed trên train + val. Chấm validation một lần (giao thức train) và test một lần cho mỗi giao thức, bằng `--open-official`. Số chính là tổ hợp ba seed của S, trung bình xác suất.

Tuyên bố suy luận chính: khoảng bootstrap theo người, ghép cặp, cho hiệu (tổ hợp ba seed S trừ tổ hợp ba seed SVFAP của v1), trên cùng 2 256 clip và 26 người, B = 2 000, seed 0, cho accuracy và macro-F1. Đây là phép so sánh ghép cặp duy nhất làm được trên bộ này, vì chỉ có dự đoán từng clip của chính nhóm. Phải kiểm khi máy chủ có lại: các file `preds/ft_g8_svfap_s{0,1,2}*.npz` còn đủ. Chênh với 67,61, 69,29 và 73,58 vẫn là chênh giữa ước lượng điểm, không kiểm định. Nhánh C của v1 chạy ở môi trường cũ; độ lệch môi trường đo ở dòng ba của mục 5 phải báo cạnh khoảng ghép cặp.

Bài phải khai: tập test EngageNet đã được nhóm chấm cho v1, mỗi giao thức một lần; lần chấm của v3 là lần thứ hai cho mỗi giao thức, với một mô hình khoá trước khi chấm.

DAiSEE: chuyển giao thuần, công thức S đã khoá, ba seed, hai giao thức, không cổng, không kỳ vọng. Nhánh S không cần đổi bước cho DAiSEE vì cửa sổ 16 × 4 không phụ thuộc độ dài clip, nên sai khác nhánh lấy mẫu giữa hai bộ của v1 không còn.

## 7. Nếu G-S trượt

Không mở split chính thức nào. Không có bài mô hình. Kết quả ba nhánh được ghi đầy đủ và báo cáo, kể cả âm. Không thử biến thể khác của nhánh S trên bộ này nếu không có bản ghi trước mới.

## 8. Chi phí, đọc từ trường `seconds` của các file kết quả

Một lần chạy fold nội bộ 12 epoch ở T = 16 mất 3 131–3 144 s khi máy rỗi (v2) và 6 176–7 909 s khi máy dùng chung (g6, g7). Nhánh S ước khoảng năm lần, tức 4,4–11 giờ mỗi lần. Đây là giả định chưa đo; kiểm thử khói sẽ cho số đầu tiên.

| Giai đoạn | Số lần chạy | Ước tính |
|---|---|---|
| Fold nội bộ: R 9, C 9, S 9 | 27 | 2,5–6 ngày |
| Gộp chú ý, chỉ khi G-S qua | 9 | thêm 1,7–4 ngày |
| Chính thức EngageNet, S: 3 seed × 2 giao thức (một lần chạy chính thức v1 mất 4 596–4 831 s) | 6 | 1,5–2 ngày |
| DAiSEE, S: 3 seed × 2 giao thức | 6 | 1–2 ngày |

Tổng nếu đi hết: khoảng một đến hai tuần GPU trên một L4. Nếu G-S trượt thì dừng sau giai đoạn đầu.

## 9. Việc phải làm trước lần chạy đầu tiên

1. Tác giả duyệt file này; ghi sha256 vào FACTS.
2. Máy chủ kết nối lại; kiểm `preds/ft_g8_*` và đo phần clip EngageNet dưới 281 khung (ảnh hưởng đến câu "cửa sổ đều" của nhánh ngắn trong bài PeerJ).
3. Thêm cờ `--segments K` vào `ft_video.py` theo đúng mục 2, kèm kiểm thử khói. Không đổi hành vi khi không có `--segments`, để mọi lần chạy cũ vẫn tái lập được.
