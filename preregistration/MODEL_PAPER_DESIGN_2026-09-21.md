# Thiết kế bài mô hình, bản 2 — mục tiêu duy nhất: vượt SOTA trên EngageNet và DAiSEE

Viết ngày 21/09/2026, thay bản 1 cùng ngày (bản 1 lấy "bền vững theo khung hình" làm trục; tác giả đổi mục tiêu thành vượt SOTA, không giới hạn backbone và thời gian huấn luyện). Mọi dự đoán và ngưỡng ở mục 5 được ghi trước khi chạy. Kết quả nào cũng báo cáo, kể cả khi không vượt.

## 1. Mốc phải vượt (đã kiểm chứng)

**EngageNet, 4 mức, validation chính thức (11 người, 1 071 clip; test 26 người, 2 257 clip, nhãn do tác giả bộ dữ liệu cấp cho nhóm):**

| Phương pháp | Val | Test | Ghi chú |
|---|---|---|---|
| PriorNet (arXiv:2605.03615, 5/2026, tiền ấn phẩm, không mã) | **73,58** | — | SVFAP + Prior-LoRA hạng 16 ở mọi khối chẵn + placeholder mặt thiếu + mất mát evidential; 16 khung 224. Ablation của họ: SVFAP + CE 68,91; + placeholder 71,15; + LoRA 70,03; + objective 69,56; đủ 73,58 |
| Ordinal ST-GCN (arXiv:2403.17175) | 71,24 | — | landmark |
| ICMI 2023 (tác giả bộ dữ liệu) | 69,10 | 67,61 | OpenFace; train+val → test: 69,29 |
| VE-MD (arXiv:2604.02397) | 68,98 | — | |
| TCCT-Net | 68,91 | — | |
| Baseline MARLIN→Transformer (EmotiW 2023) | 65,54 | 62,00 | |
| Nhóm: MARLIN-L đóng băng + adapter (`LADDER_U4`) | 63,2 | — | 3 seed, KTC theo người [0,539; 0,731] |
| Nhóm (bài P1): MARLIN đóng băng + gộp thời gian, tổ hợp | 67,51 | 68,3 | |

**DAiSEE, 4 mức engagement, test chính thức 1 784 clip** (`P2-Identity-Invariant-Engagement/SOTA_DAISEE_KIEM_CHUNG.md`):

| Phương pháp | Test | Giao thức |
|---|---|---|
| ViBED-Net (arXiv:2510.18016, tiền ấn phẩm) | 73,43 | split chính thức; recall lớp thấp nhất 0,25 |
| PriorNet | 69,06 | train+val gộp |
| Selim 2022 (IEEE Access, bình duyệt) | 67,48 | split chính thức |
| Abedi & Khan ordinal TCN | 67,4 | |
| ResNet+TCN (có mã) | 63,9 | train+val gộp |
| Sàn lớp đa số | 49,4 | |

Cách đọc mốc: trên EngageNet, con số công bố là validation; test chỉ có số của tác giả bộ dữ liệu. Bài sẽ báo cả hai, tuyên bố "vượt" trên validation chỉ khi > 73,58, và trên test chỉ khi > 69,29. Trên DAiSEE, báo cả hai giao thức; tuyên bố "vượt" khi > 69,06 (gộp) và > 67,48 (chính thức); 73,43 của ViBED-Net ghi trong bảng với cờ tiền ấn phẩm.

## 2. Vì sao mô hình hiện tại kém 10 điểm và cái gì đóng được khoảng đó

Bộ mã hoá của nhóm đóng băng hoàn toàn (0,632). PriorNet cho thấy trên cùng bộ dữ liệu: fine-tune có LoRA + xử lý mặt thiếu + mất mát tốt hơn = +4,7 điểm trên chính backbone của họ. Backbone của họ (SVFAP, tiền huấn luyện tự giám sát trên VoxCeleb2, 160×160, 16 khung, bước 4) là backbone chuyên mặt-video, mạnh hơn MARLIN ở nhận dạng biểu cảm động (DFEW WR 74,8 so với MARLIN không công bố trên DFEW; MAE-DFER cùng nhóm 74,4). Ba đòn bẩy, theo thứ tự kỳ vọng:

1. **Fine-tune backbone** (không chỉ head): LoRA hạng 16 mọi khối, hoặc fine-tune toàn phần với layer-wise LR decay, EMA, drop-path; SVFAP-B là ứng viên đầu, MAE-DFER-B (VoxCeleb2, có trọng số Google Drive) thứ hai, MARLIN-L thứ ba, VideoMAEv2-giant (HF, CC-BY-NC) thứ tư nếu ba cái đầu chưa đủ.
2. **Phủ toàn clip**: mỗi clip 300 khung; SOTA dùng 16 khung. Huấn luyện với cửa sổ ngẫu nhiên, chấm bằng trung bình 5 cửa sổ × 2 crop (TTA), cộng một tầng gộp thời gian nhỏ trên embedding cửa sổ.
3. **Tổ hợp** nhiều backbone đã fine-tune, trung bình xác suất. Tổ hợp thường cho +1–3 điểm; đây là lý do chính để làm ba backbone chứ không phải một.

Thêm hai thành phần rẻ có tiền lệ: placeholder cho khung không phát hiện được mặt (PriorNet +2,2 trên EngageNet; nhóm tự viết lại, không dùng mã của họ) và nhãn mềm thứ bậc (LDL Gaussian σ = 0,5, đã có trong `train_core.py`).

Về DAiSEE: kiểm toán của nhóm cho thấy nhãn gần nhiễu (đồng thuận liền kề 0,58 so với 0,54 độc lập). Điểm cao trên DAiSEE một phần là khớp thiên lệch lớp; ViBED-Net 73,43 có recall lớp thấp nhất 0,25. Bài phải báo UAR và macro-F1 bên cạnh accuracy và nói rõ điều này, nhưng vẫn tối ưu accuracy vì đó là thước đo của bảng.

## 3. Giao thức, không đổi trong suốt chiến dịch

- EngageNet: train 91 người để chọn mọi thứ; validation 11 người chấm theo cấu hình đã chốt (mỗi cấu hình một lần); **test 26 người mở một lần cuối cùng** cho (a) mô hình huấn luyện trên train và (b) mô hình huấn luyện trên train + val, như tác giả bộ dữ liệu đã làm. Chọn siêu tham số bằng 3 fold độc lập người bên trong train, không dùng validation để chọn.
- DAiSEE: cùng cách, hai giao thức (chính thức; train+val gộp), test mở một lần.
- Chỉ số: accuracy (để so bảng), UAR, macro-F1, KTC 95 % bootstrap theo người, 3 seed, hiệu ghép cặp giữa các cấu hình.
- Không dùng nhãn hay thống kê của tập test ở bất kỳ bước nào, kể cả chuẩn hoá đặc trưng và chọn ngưỡng.
- Ghi cấu hình đầy đủ vào file kết quả; mọi lần chạy đều được liệt kê trong phụ lục, kể cả lần hỏng.

## 4. Kế hoạch chạy (theo giai đoạn, giai đoạn sau chỉ chạy nếu giai đoạn trước có lý do)

**G0. Dữ liệu.** Cache ảnh cắt mặt 224×224 JPEG cho mọi khung của mọi clip EngageNet (11 311) và DAiSEE (8 232): SCRFD trên mỗi khung thứ 4, nội suy hộp, nới 1,3×; khung không có mặt ghi cờ (dùng cho placeholder). Ước tính 50 GB, 4–6 giờ CPU với 40 tiến trình.

**G1. Tái lập mốc trên backbone mạnh nhất tải được.** SVFAP-B (nếu tải được; hiện link SharePoint trả 401, đang thử qua trình duyệt) hoặc MAE-DFER-B: fine-tune với công thức của chính họ cho DFEW (AdamW, lr 1e-3, 100 epoch, batch 16–32, 160×160, 16×4, mixup/label smoothing như mã gốc) trên EngageNet train, chấm val. Kỳ vọng: 67–71 (PriorNet báo SVFAP + CE = 68,91). Đây là điểm xuất phát; nếu dưới 66 thì pipeline có lỗi, không đi tiếp.

**G2. Từng đòn bẩy, mỗi cái một lần chạy 3 seed:** (a) placeholder mặt thiếu; (b) LDL thứ bậc thay CE; (c) TTA 5 cửa sổ; (d) LoRA hạng 16 so với fine-tune toàn phần; (e) tầng gộp thời gian trên 5 cửa sổ. Giữ cái nào cho > +0,5 điểm trên CV trong train.

**G3. Ba backbone với công thức đã chốt, rồi tổ hợp.** SVFAP-B, MAE-DFER-B, MARLIN-L (LoRA, 224). Tổ hợp bằng trung bình xác suất, trọng số chọn trên CV trong train.

**G4. Validation một lần, rồi test một lần** (train-only và train+val). Cùng lúc chạy DAiSEE với công thức đã chốt trên EngageNet (không chọn lại), cộng một lần chọn siêu tham số riêng trên train DAiSEE nếu công thức EngageNet cho kết quả dưới mốc 63,9 có mã.

Ngân sách: G1 khoảng 6–10 giờ/backbone/seed trên L4 (SVFAP-B 160² nhẹ hơn MARLIN-L 224² nhiều); toàn chiến dịch 150–250 giờ GPU, tức 1–2 tuần chạy liên tục. GPU đang trống.

## 5. Dự đoán và ngưỡng, ghi trước

| | Dự đoán của tôi (EngageNet val) | Ngưỡng dừng / bác bỏ |
|---|---|---|
| G1 fine-tune một backbone | 67–71 | < 66: sửa pipeline trước khi đi tiếp |
| G2 placeholder | +1 đến +2 | < +0,5 trên CV: bỏ |
| G2 LDL thứ bậc | 0 đến +1 | < +0,5: bỏ |
| G2 TTA 5 cửa sổ | +0,5 đến +1,5 | < +0,5: bỏ |
| G2 LoRA so với toàn phần | ngang nhau, LoRA ổn định hơn | chọn cái cao hơn trên CV |
| G3 tổ hợp ba backbone | +1,5 đến +3 so với backbone tốt nhất | < +0,5: báo cáo đơn backbone |
| **Kết quả cuối, val** | **72–75; xác suất vượt 73,58 tôi ước 40–50 %** | nếu < 73,58: bài báo test là số chính, val trong bảng, không tuyên bố SOTA val |
| Kết quả cuối, test (train+val) | 69–72 | so với 69,29 |
| DAiSEE, gộp | 67–71 | so với 69,06 |
| DAiSEE, chính thức | 66–70 | so với 67,48 |

Rủi ro lớn nhất ghi trước: validation EngageNet có 11 người, KTC rộng ±5 điểm; một mô hình 72 và một mô hình 74 không phân biệt được về thống kê trên tập đó. Bài sẽ nói điều này và đặt trọng tâm vào test 26 người. Nếu vượt SOTA chỉ trên một trong hai bộ dữ liệu thì tiêu đề và tóm tắt phải nói đúng như vậy.

## 6. Ranh giới trung thực

- Không có mã PriorNet; mọi thành phần viết lại từ mô tả; con số của họ là tự báo cáo trên tiền ấn phẩm.
- Trọng số SVFAP/MAE-DFER là của nhóm Sun và cộng sự; ghi nguồn và giấy phép.
- Bản v3 (giao thức có hiệu chuẩn trên OUC-CGE) làm chương đánh giá ngoài miền, không đổi kết luận của nó.
- Hai hội đồng phản biện mô phỏng và mã do AI viết dưới hướng dẫn của tác giả: khai báo đúng.

## Bổ sung ngày 21/09/2026, sau khi G1 chạy được 5 epoch và trước khi chọn bộ lever G2

Ba điểm dưới đây phát hiện khi xem 5 epoch đầu của `g1_svfap_full`; ghi lại trước khi bất kỳ lever nào chạy.

**1. Cửa sổ thời gian chỉ phủ 20 % clip. Đây là lever quan trọng nhất và chưa ai thử trên benchmark này.** Công thức `16 khung × bước 4` lấy từ công thức DFEW của chính nhóm SVFAP, mà clip DFEW ngắn nên 16×4 phủ gần hết. Clip EngageNet dài 10 giây, sau khi giảm mẫu còn ~304 khung, nên span = (16−1)×4+1 = 61 khung, tức **20 % clip**. Lúc chấm, 5 cửa sổ TTA có phủ hết clip nhưng là năm phán đoán 2 giây độc lập rồi lấy trung bình, không phải một phán đoán 10 giây; với một trạng thái biến thiên chậm như engagement hai thứ đó không tương đương. PriorNet ghi "16 khung 224×224" cho clip 10 giây, nhiều khả năng là lấy đều toàn clip. **Lever 1 của G2 là `--stride 20`** (span 301 ≈ 99 % clip), chạy trước placeholder / balanced / LDL / LoRA. Kèm theo: ở bước 20 phép tăng cường bằng dịch cửa sổ gần như biến mất, nên đã thêm nhiễu vị trí từng khung ±stride/2 vào `ft_video.py`.

**2. Bẫy so sánh giữa CV và validation. Mọi ngưỡng ghi trước ở mục 5 phải đọc lại theo đơn vị đúng.** Sàn lớp đa số của CV fold 0 là **0,5792**, của validation chính thức là **0,5313**. Ngưỡng G1 "67–71, dừng nếu < 66" viết theo đơn vị validation, nhưng mọi lần chạy G1/G2 báo số trên CV. Một con số 0,62 trên CV fold 0 **không phải** "kém SOTA 5 điểm"; nó là hơn sàn của fold đó 4 điểm, khác hẳn. Sửa: **chọn cấu hình bằng UAR và macro-F1 trên CV, không bằng accuracy**, vì hai chỉ số đó bền với mất cân bằng lớp và so được giữa các fold. `ft_video.py` giờ ghi `eval_majority_baseline`, `eval_class_counts`, `train_class_counts` và `span_frames` vào mọi file `FT_*.json` để không ai (kể cả phản biện) đọc nhầm một accuracy CV thành accuracy validation. Accuracy chỉ so được với 73,58 ở đúng một lần chạy validation cho mỗi cấu hình đã chốt.

**3. Chọn thân mô hình trước, mới chạy lever.** Bản kế hoạch đầu gán cứng các lever G2 cho SVFAP trước khi G1 nói backbone nào thắng. Đã sửa: `state/g2.sh` chờ cả hai lần chạy G1, chọn thân theo macro-F1, rồi mới chạy lever trên thân đó.

**4. Thêm lever lấy mẫu cân bằng lớp.** Phân bố lớp tập huấn luyện là [1213, 721, 1175, 2322]; ba epoch đầu mô hình dự đoán một lớp duy nhất. Tối ưu accuracy thì CE thường là đúng, nhưng lấy mẫu cân bằng nhiều khả năng rút ngắn pha đó và nâng UAR. Đã thêm cờ `--balanced`, đưa vào danh sách lever, **không** đổi cấu hình đang chạy.

**Ghi nhận diễn biến G1 (không phải sự cố):** epoch 0–2 chạy trong giai đoạn khởi động (`--warmup 3`) nên tốc độ học chưa đạt mức nền và mô hình dự đoán một lớp; epoch 3 là epoch đầu ở tốc độ học đầy đủ. Loss giảm đơn điệu 1,154 → 0,868 và phân bố dự đoán mở dần ra (1 lớp → 2 → 3 → 4). Epoch 4: acc 0,6254, UAR 0,3743, macro-F1 0,3613. Mốc kiểm tra tiếp theo là epoch 12 và 18; dấu hiệu hỏng thật sự sẽ là UAR vẫn ≤ 0,28 kèm dự đoán ≤ 2 lớp ở epoch 12 trở đi, hoặc loss chững trên 0,80.

## Bổ sung lần 5, ngày 22/09/2026 — ĐỔI KIẾN TRÚC: mô hình hai luồng. Ghi trước khi G3 kết thúc và trước mọi lần chạy validation.

Tác giả giao toàn quyền đổi mô hình, kiến trúc, backbone, miễn là đề xuất được một mô hình hoặc quy trình có khả năng thắng SOTA. Phần này ghi lại quyết định và, quan trọng hơn, **chốt trước các tham số hợp nhất** để con số validation một-lần không bị nhiễm chọn lọc.

### Vì sao đổi: tôi đã tự bó tay vào một luồng

Đọc lại bảng SOTA theo phương thức đầu vào, không theo điểm số: **mọi mục trong bảng trừ PriorNet đều là hình học, không phải pixel.** Ordinal ST-GCN 71,24 dùng 78 điểm mốc trên 300 khung; mốc ICMI 69,10 dùng gaze + head pose + AU của OpenFace; TCCT-Net 68,91 dùng đúng hai tín hiệu head-pose. Tôi thì huấn luyện một backbone pixel 16 khung và đạt 0,6544 trên fold CV. Trong khi đó bài P1 của nhóm đã có sẵn đặc trưng hình học 1 338 chiều cho cả 11 206 clip, xếp theo thứ tự tất định `sort_values(["subject","session","vid","chunk_idx"])`, khớp chính xác 7 879 / 1 071 / 2 256 dòng — tôi đã kiểm và căn khớp được.

Đo trên **đúng fold CV 0** mà chiến dịch pixel đang báo cáo:

| luồng | acc | UAR | macro-F1 |
|---|---|---|---|
| pixel (SVFAP fine-tune, phủ toàn clip) | 0,6544 | 0,5814 | 0,5863 |
| hình học (1 338 chiều, GBDT, huấn luyện vài giây) | 0,6712 | 0,5588 | 0,5554 |
| hợp nhất, trọng số đều | 0,6993 | 0,6042 | 0,6072 |

Hình học **một mình đã hơn** mô hình pixel về accuracy, với chi phí vài giây thay vì 11 giờ GPU; pixel hơn về macro-F1. Hai cái bổ sung cho nhau, và hợp nhất được thêm 4,5 điểm accuracy so với pixel.

### Chốt trước, ràng buộc, không được đảo sau khi thấy kết quả

1. **Trọng số hợp nhất `w_pixel = 0,5` được cố định từ trước** như một phép trung bình không trọng số. Lần quét w ∈ {0,2 … 0,8} trên fold 0 là **thăm dò** và **không** được dùng để chọn w. Lý do phải ghi rõ: đỉnh nằm đúng giữa lưới và cao hơn cả hai điểm kề 1,5 điểm (0,6846 → 0,6993 → 0,6895) trong một đường cong ngoài ra rất trơn, nên gần như chắc chắn là nhiễu của fold 0 chứ không phải cấu trúc. Vì vậy:
2. **Con số fold 0 trung thực của luồng hợp nhất là khoảng 0,685, không phải 0,6993.** Mọi dự báo và mọi ngưỡng phải dựng trên 0,685. Con số 0,6993 chỉ được nêu như cực đại thăm dò, kèm nguyên văn cảnh báo này.
3. Hợp nhất là trung bình xác suất. Hợp nhất theo hạng đã thử và **thua rõ** (0,6144), loại.
4. Không dùng stacking, không học trọng số hợp nhất trên tập đánh giá ở bất kỳ mức nào.

### Mô hình đề xuất

**Bộ nhận dạng engagement hai luồng, phủ toàn clip.**
- *Luồng A, diện mạo:* SVFAP-B fine-tune đầu-cuối, cửa sổ 16 khung bước 20 phủ 99 % clip, EMA, layer-wise lr decay. Đóng góp: động học diện mạo khuôn mặt.
- *Luồng B, hình học:* thống kê điểm mốc 3D và head pose trên toàn clip → cây tăng cường gradient. Đóng góp: hình học tư thế và hướng nhìn mà pixel ở 160 px làm mất.
- *Hợp nhất:* trung bình xác suất trọng số đều, cố định trước.
- *Tổ hợp:* nhiều seed và nhiều backbone ở luồng A, rồi mới hợp nhất với luồng B ở trọng số đều.

### Lever tiếp theo và phép thử rẻ để quyết định

Hình học hiện tại chỉ dùng **16 khung** lấy đều (`N_FRAMES = 16`, max nframes = 16), tức 1,6 fps: nhịp nháy mắt, vi chuyển động và phương sai tư thế đều vô hình. Vì phủ toàn clip ở luồng pixel đã cho +5,1 điểm, phủ toàn clip ở luồng hình học là lever lớn nhất còn lại.

Quyết định: **96 khung, không phải 300.** 300 khung tốn 19 lần chi phí gọi mô hình điểm mốc trên CPU, còn thông tin thêm trên khoảng 10 fps là nhỏ. 96 khung tốn 6 lần, cho khoảng 10 fps, đủ thấy nhịp nháy mắt (200–400 ms).

**Điều kiện quyết định thành bại của lever này không phải số khung mà là bộ thống kê.** Véc-tơ 1 338 chiều hiện tại là mean/std/min/max trên 16 mẫu; giữ nguyên cách tóm tắt đó rồi tăng lên 96 khung thì chỉ được các ước lượng ít nhiễu hơn của **cùng** đại lượng, gần như không thêm gì. Phần lợi nằm ở các thống kê **không định nghĩa được** ở 16 khung, nên bắt buộc thêm: nhịp nháy mắt và khoảng giữa hai lần nháy (từ điểm cắt không của tỉ lệ khía mắt); năng lượng ở dải tần thấp so với dải tần cao của yaw/pitch/roll; tỉ lệ khung có tốc độ góc đầu vượt ngưỡng và độ dài chuỗi bất động dài nhất; tự tương quan của tín hiệu tư thế ở 0,5 s và 1 s.

**Ngưỡng ghi trước:** thử ở 96 khung trên khoảng 800 clip trải cả phần train và phần eval của fold 0, dùng đúng GBDT đó, so với mốc hình học 0,6712. **Dưới +0,015** thì kết luận hình học 16 khung đã gần trần, dừng lever này, dồn thời gian còn lại cho seed và backbone. Từ +0,015 trở lên thì chạy đủ, và cân nhắc 192 khung ở vòng sau.

### Luồng thứ ba, gần như miễn phí

P1 đã cache sẵn đặc trưng đông cứng cùng thứ tự tất định: MAE-DFER 513 chiều, MARLIN 1 024 chiều, ArcFace 512 chiều. Thêm một luồng chỉ tốn một lần khớp GBDT. P1 đã thấy chúng bổ sung cho hình học. Thử trong cùng phép thăm dò; thêm được từ +0,01 là lấy.

### Dự báo cập nhật, nêu dạng khoảng

Fold 0 hợp nhất sau khi bỏ đỉnh nhiễu: khoảng 0,685. Huấn luyện trên 7 879 clip thay vì 5 431: +1 đến +3. Tổ hợp seed và backbone: +1 đến +2. Hình học phủ toàn clip: chưa biết, 0 đến +3. Tổng lại **0,70–0,74, với 73,58 nằm ở đỉnh khoảng chứ không phải giữa khoảng.** Đây là một cơ hội thật, nhưng đỉnh khoảng phụ thuộc vào việc lever hình học có trả lời được hay không. Hôm nay tôi đã hạ dự báo một lần dựa trên dữ liệu; nâng nó lên chỉ vì một đỉnh trên một fold sẽ là đúng cùng loại sai lầm theo chiều ngược lại.

## Bổ sung lần 6, ngày 22/09/2026 — kết quả nhiều luồng, và một lever của tôi đã thất bại

### Thăm dò mọi luồng đã cache, trên đúng fold CV 0

| luồng, một mình | chiều | acc | UAR | macro-F1 |
|---|---|---|---|---|
| hình học 16 khung | 1 338 | 0,6712 | 0,5588 | 0,5554 |
| OpenFace (gaze, head pose, AU — đúng bộ đặc trưng của mốc ICMI 69,10) | 295 | 0,6548 | 0,5526 | 0,5222 |
| pixel (SVFAP fine-tune, phủ toàn clip) | — | 0,6544 | 0,5814 | 0,5863 |
| OpenFace dạng chuỗi | 9 280 | 0,6197 | 0,4940 | 0,4343 |
| MAE-DFER đông cứng | 2 560 | 0,6160 | 0,4742 | 0,4405 |

Hợp nhất trung bình xác suất: `pixel+geom` 0,6993 acc và macro-F1 **0,6072** (cao nhất); `pixel+geom+ofseq` 0,7042 acc nhưng macro-F1 chỉ 0,5866. OpenFace và MAE-DFER thêm vào `pixel+geom` không cải thiện.

**Cảnh báo về chọn lọc, tự áp cho mình:** tôi vừa xem cực đại của 26 tổ hợp trên **một** fold. Chênh lệch giữa 0,6993 và 0,7042 là 0,5 điểm, nằm trong nhiễu của 2 448 clip. Chọn `pixel+geom+ofseq` vì nó đứng đầu bảng này sẽ là **đúng cùng loại sai lầm** mà tôi đã phê trong `03_integrity/PROTOCOL_AUDIT_2026-09-21.md` khi chỉ ra con số 63,2 là cực đại của lưới 16 ô trên validation. Vì vậy: **mô hình vẫn là `pixel+geom` hai luồng như đã chốt ở bổ sung 5**; bảng 26 tổ hợp là thăm dò; nếu muốn thêm luồng thứ ba thì phải chọn trên fold 1 và fold 2, là hai fold chưa dùng, rồi mới xác nhận.

### Lever hình học phủ toàn clip: giả thuyết của tôi sai, và phép thử của tôi thiếu lực

Đã viết `code/geom96.py`: dựng lại đúng véc-tơ 1 338 chiều ở 96 khung (≈9,6 Hz thay vì 1,6 Hz) bằng cách import chính `frame_descriptor` và `temporal_stats` của bài P1 nên 1 338 chiều đầu so sánh trực tiếp được, cộng một khối động học 152 chiều gồm nhịp lõm tín hiệu và độ sâu lõm, tỉ lệ năng lượng dải cao trên dải thấp, trọng tâm phổ, tỉ lệ khung chuyển động nhanh, chuỗi bất động dài nhất, và tự tương quan ở 0,5 s và 1,0 s.

Kết quả thử trên 919 clip:

| bộ đặc trưng | chiều | acc | so với (a) |
|---|---|---|---|
| (a) 16 khung, đã cache | 1 338 | 0,5174 | — |
| (b) 96 khung, cùng cách tóm tắt | 1 338 | 0,5283 | +0,0109 |
| (c) 96 khung, thêm khối động học | 1 490 | 0,5196 | **+0,0022** |

Ngưỡng ghi trước là +0,015 cho (c) so với (a). **(c) đạt +0,0022, trượt xa.** Đáng chú ý hơn: khối động học làm **tệ hơn** so với chỉ tăng số khung, tức 152 chiều tôi thêm vào là nhiễu chứ không phải tín hiệu. Giả thuyết của tôi rằng nhịp nháy mắt và phổ tần tư thế mang thông tin engagement, ở dạng tôi tính, là sai.

**Nhưng phép thử của tôi cũng có lỗi thiết kế, phải nói rõ:** tôi chia 919 clip thành 459 để khớp và 460 để chấm, nên sai số chuẩn của accuracy khoảng 0,023 — **lớn hơn cả ngưỡng 0,015 tôi đặt ra**. Phép thử đó không bao giờ đủ lực để phân biệt +0,015 với 0. Kết luận trung thực là **chưa có kết luận**, không phải "âm tính": nó chỉ loại trừ được khả năng lever này thắng lớn kiểu +5 điểm như lever pixel.

Vì CPU đang rỗi trong khi GPU chạy huấn luyện, trả lời cho đúng là **miễn phí**: đang chạy trích xuất đủ 11 206 clip ở 96 khung (khoảng một giờ), rồi chấm trên **toàn bộ** fold 0 với 2 448 clip, nơi sai số chuẩn khoảng 0,010 và ngưỡng +0,015 trở nên phân giải được. Phán quyết cuối ghi vào `results/G4_FUSED.json` theo đúng ngưỡng đã ghi trước, và nếu trượt thì lever bị loại, không được cứu bằng cách đổi ngưỡng.

### Hàng đợi hiện tại

`p2x_g2` lever LDL rồi LoRA → `p2x_g3` chốt công thức thắng, chạy ba backbone và ba seed rồi tổ hợp → `p2x_g4` hợp nhất pixel với hình học ở w = 0,5 cố định trước, và chấm phán quyết lever 96 khung. `p2x_geom96f` trích xuất hình học trên CPU song song. Con số G4 là con số tôi sẽ đem nộp.

## Bổ sung lần 7, ngày 24/09/2026 — chốt mô hình chính thức TRƯỚC khi mở validation và test

Ghi trước khi bất kỳ split chính thức nào bị chấm. Sau đoạn này không được đổi mô hình, số epoch, hệ số EMA, thành phần tổ hợp, hay cách hợp nhất.

### Bằng chứng ba fold nội bộ (G7, `results/G7_FUSED.json`)

| fold | sàn đa số | pixel (tổ hợp/đơn) | hình học | hợp nhất w=0,5 | hợp nhất − pixel, KTC theo người |
|---|---|---|---|---|---|
| 0 | 0,5792 | 0,7047 (4 thành viên) | 0,6712 | 0,6908 | −0,0139 [−0,0380; +0,0077] |
| 1 | 0,4391 | 0,6362 (đơn) | 0,5865 | 0,6051 | −0,0310 [−0,0723; +0,0067] |
| 2 | 0,4171 | 0,6555 (đơn) | 0,6716 | 0,6894 | +0,0340 [−0,0130; +0,0888] |

Ba điều rút ra:

1. **Hợp nhất với hình học ở w = 0,5 không còn tác dụng đo được.** Cả ba khoảng tin cậy của hiệu ghép cặp đều chứa 0, trung bình −0,004. Lý do nhìn thấy được: bổ sung 5 chốt w = 0,5 khi luồng pixel còn ở 0,6544 và hình học ở 0,6712, tức hình học mạnh hơn; sau khi sửa số epoch và EMA, luồng pixel lên 0,6985–0,7047 còn hình học vẫn đứng ở trần ~0,67 trên mọi fold, nên trọng số đều kéo tổ hợp về phía luồng yếu hơn. Đây đúng là việc CV nội bộ tồn tại để làm: kiểm một quyết định đã ghi trước và sửa nó **trước** khi chạm split chính thức.
2. **Ba fold rất khác nhau về phân bố lớp** (sàn 0,58 / 0,44 / 0,42) vì split theo người và nhiều người có nhãn lệch hẳn về một lớp. Toàn bộ sàng lọc lever ở G2–G6 làm trên fold 0, là fold dễ nhất; con số 0,70 của fold 0 là lạc quan so với trung bình ba fold 0,66. Validation (sàn 0,53) và test (sàn 0,50) nằm giữa.
3. **MAE-DFER ở 12 epoch thành thành viên đơn tốt nhất trên fold 0** (0,7087), ngược với xếp hạng ở 30 epoch (0,6058), tức lịch dài từng làm nó quá khớp nặng hơn SVFAP. MARLIN bị loại: hai lần hỏng đều do hết bộ nhớ GPU khi người dùng khác giữ 10 GB, không phải lỗi mô hình; hai backbone là đủ.

### Mô hình chính thức, cố định

- **Thành viên (4):** SVFAP-B LoRA-16 seed 0, 1, 2; MAE-DFER-B LoRA-16 seed 0. Cùng công thức: cửa sổ 16 khung bước 20, 12 epoch, khởi động 2, EMA 0,999, lr 1e-3, batch 16 × tích luỹ 2, layer-wise decay 0,75, TTA 5 cửa sổ.
- **Số chính (headline): tổ hợp pixel**, trung bình xác suất bốn thành viên.
- **Số phụ, báo cạnh số chính theo đúng bổ sung 5: hợp nhất w = 0,5** với luồng hình học 16 khung (GBDT), lấy từ **cùng các lượt truyền** nên không mở thêm split lần nào.
- Cả hai đều kèm khoảng tin cậy 95 % bootstrap theo người, UAR và macro-F1. Không dò w. Không chọn epoch trên split chính thức.

### Hai lần mở, mỗi split đúng một lần

1. Huấn luyện trên **train** (7 879 clip) → chấm **validation** một lần. Mục đích: đặt cạnh bảng đã công bố, vốn toàn số validation.
2. Huấn luyện trên **train + validation** (8 950 clip) → chấm **test** một lần. Mục đích: tuyên bố chính, đối chiếu với 69,29 của tác giả bộ dữ liệu theo đúng giao thức gộp mà họ dùng; và 67,61 cho biến thể chỉ train, lấy từ chính các thành viên của lần mở 1 chấm trên test cũng đúng một lần.

### Cam kết đọc kết quả, viết trước

- Cả bốn con số (val tổ hợp, val hợp nhất, test tổ hợp, test hợp nhất) được báo cáo nguyên vẹn, kể cả khi số phụ cao hơn số chính. Không đổi vai sau khi thấy số.
- Tuyên bố "vượt" chỉ đặt trên test, chỉ khi tổ hợp pixel > 69,29 **và** 69,29 nằm dưới khoảng tin cậy hoặc bài nói rõ nó nằm trong.
- Dự báo, nêu dạng khoảng, dựa trên trung bình ba fold 0,66 và độ khó của từng split: validation 0,68–0,71; **test 0,66–0,70, với 69,29 nằm giữa khoảng chứ không phải đáy**. Xác suất vượt 69,29 tôi ước 40–55 %. Nếu không vượt, bài không tuyên bố SOTA; đóng góp còn lại là mô hình phủ toàn clip cộng phát hiện rằng bảng validation 11 người không phân biệt nổi phương pháp, và quyết định công bố hay không là của tác giả.

## Bổ sung lần 8, ngày 25/09/2026 — KẾT QUẢ CHÍNH THỨC. Mỗi split mở đúng một lần, đúng mô hình đã khoá ở bổ sung 7.

Nguồn: `results/G8_OFFICIAL.json`, log `results/g8_official.log`. Bốn thành viên, 12 epoch, EMA 0,999, bước 20, LoRA-16. Khoảng tin cậy 95 % bootstrap theo người, 2 000 lần.

| Split | Huấn luyện trên | n / người | sàn | **Tổ hợp pixel (chính)** | Hợp nhất w=0,5 (phụ) | Hình học |
|---|---|---|---|---|---|---|
| Validation | train | 1 071 / 11 | 0,5313 | **0,7134** [0,6206; 0,7924] | 0,6863 [0,5933; 0,7744] | 0,6695 |
| Test | train | 2 256 / 26 | 0,5027 | **0,7021** [0,6210; 0,7765] | 0,6990 [0,6122; 0,7777] | 0,6817 |
| Test | train + val | 2 256 / 26 | 0,5027 | **0,7004** [0,6219; 0,7714] | **0,7026** [0,6174; 0,7823] | 0,6848 |

UAR / macro-F1 của tổ hợp pixel: val 0,6076 / 0,6289; test train-only 0,5625 / 0,5799; test train+val 0,5604 / 0,5766.

### Đối chiếu với các mốc đã kiểm chứng

| So sánh | Mốc | Ta | Chênh |
|---|---|---|---|
| Validation, so với tác giả bộ dữ liệu (ICMI) | 69,10 | 71,34 | **+2,24** |
| Validation, so với Ordinal ST-GCN | 71,24 | 71,34 | +0,10 |
| Validation, so với PriorNet (tiền ấn phẩm v1, không mã) | 73,58 | 71,34 | −2,24 |
| **Test, chỉ train, so với ICMI** | **67,61** | **70,21** | **+2,60** |
| **Test, train+val gộp, so với ICMI cùng giao thức** | **69,29** | **70,04** (pixel) / **70,26** (hợp nhất) | **+0,75 / +0,97** |

### Cách đọc, theo đúng cam kết viết trước ở bổ sung 7

1. **Trên test, điểm ước lượng vượt mọi mốc đối chiếu cùng giao thức**: +2,60 ở biến thể chỉ train và +0,75 đến +0,97 ở biến thể gộp. Đây là hai con số duy nhất trong lĩnh vực được so trên tập test 26 người ngoài chính tác giả bộ dữ liệu.
2. **69,29 nằm trong khoảng tin cậy của ta**, và ngược lại. Bài phải nói đúng như vậy: điểm ước lượng cao hơn, nhưng benchmark không tách được hai phương pháp về thống kê. Khoảng tin cậy trên test rộng 15,5 điểm; trên validation rộng 17,2 điểm.
3. **Trên validation, ta đứng thứ hai** trong bảng công bố, ngang ST-GCN, dưới PriorNet 2,24 điểm. Toàn bộ khoảng cách đó nằm gọn trong một khoảng tin cậy 17 điểm. Không tuyên bố vượt PriorNet; cũng không cần: con số đó chỉ tồn tại ở bản tiền ấn phẩm v1 trên 11 người.
4. **Số phụ cao hơn số chính 0,22 điểm ở test gộp.** Báo nguyên vẹn, không đổi vai; 0,22 điểm là nhiễu và ba fold nội bộ đã cho thấy hợp nhất không có tác dụng đo được.
5. Gộp validation vào train **không** cải thiện test (0,7004 so với 0,7021): 1 071 clip thêm từ 11 người không đổi được gì trong nhiễu. Đáng ghi vì ICMI thu được +1,68 từ cùng thao tác.

### Dự báo so với thực tế

Dự báo ghi trước: validation 0,68–0,71, test 0,66–0,70, xác suất vượt 69,29 là 40–55 %. Thực tế: validation 0,7134 (nhỉnh trên đỉnh khoảng), test 0,7004–0,7026 (đỉnh khoảng), vượt 69,29 về điểm ước lượng. Dự báo hơi thận trọng nhưng không sai hướng.

### Điều kiện công bố tự đặt ở `02_draft/DINH_VI_BAI_MO_HINH.md`: "vượt 69,29 trên test kèm khoảng tin cậy". Đã đạt về điểm ước lượng. Tuyên bố đúng mức là: *cao hơn mọi mốc cùng giao thức về điểm ước lượng, và là bài đầu tiên báo khoảng tin cậy theo người, qua đó chứng minh benchmark này không phân biệt nổi các phương pháp đang xếp trên nó.*

## Bổ sung lần 9, ngày 25/09/2026 — DAiSEE, ghi trước khi mở bất kỳ split chính thức nào của nó

**Mẫu số đã khớp chính thức.** 146 clip test và 506 clip train "thiếu" hoá ra đều có trên đĩa dưới dạng `.mp4`; script cache chỉ quét `.avi`. Đã sửa và cache nốt. Không còn lý do để báo số DAiSEE trên mẫu số lệch. Đối tượng lấy từ `ClipID[:6]` cho 109 mã; chỉ dùng để nhóm bootstrap nên lệch so với con số 112 trong tài liệu không ảnh hưởng điểm số, sẽ ghi chú.

**Mô hình: chuyển nguyên công thức đã khoá ở bổ sung 7, không chọn lại gì trên DAiSEE**, đúng như G4 của kế hoạch gốc. Bốn thành viên, 12 epoch, EMA 0,999, LoRA-16. Một khác biệt kỹ thuật để giữ hành vi tương đương: clip DAiSEE đúng 300 khung nên bước 20 (span 301) rơi vào nhánh clip-ngắn của `windows()` và mất nhiễu vị trí khi huấn luyện; dùng **bước 19** (span 286, phủ 95 %) để đi cùng nhánh với EngageNet. Đây là giữ công thức, không phải dò tham số. Không có luồng hình học cho DAiSEE (cache của P1 chỉ có EngageNet), nên chỉ có số tổ hợp pixel.

**Ba lần mở, mỗi split một lần cho mỗi cấu hình:** train → validation; chính các mô hình đó → test (giao thức split chính thức, mốc Selim 67,48 "như tác giả công bố, chưa kiểm được", và mốc có mã ResNet+TCN 63,9); train + validation → test (giao thức gộp, mốc PriorNet 69,06). Kèm KTC theo người, UAR, macro-F1, và phân bố dự đoán để thấy recall lớp thiểu số — điểm yếu mà ViBED-Net 73,43 bị ghi nhận (recall lớp thấp nhất 0,25).

**Dự báo ghi trước:** kiểm toán nhãn DAiSEE của nhóm cho thấy nhãn gần nhiễu (đồng thuận liền kề 0,58 so với 0,54 độc lập), nên tôi kỳ vọng thấp hơn EngageNet: test chính thức 0,63–0,68, test gộp 0,64–0,69. Xác suất vượt 69,06 tôi ước 25–35 %; vượt 67,48 khoảng 45 %. Nếu công thức EngageNet cho dưới 63,9 thì mới được phép chọn siêu tham số riêng trên train DAiSEE bằng CV nội bộ, như kế hoạch gốc cho phép.

## Bổ sung lần 10, ngày 25/09/2026 — cải tiến thêm: luồng chuỗi điểm mốc. Ghi trước khi trích dữ liệu và trước khi huấn luyện.

### Bằng chứng về chỗ còn dư địa (`state/headroom.py`, chỉ trên fold nội bộ)

| Hướng | Số đo | Kết luận |
|---|---|---|
| Làm mượt theo clip liền kề, cửa sổ ±1/±2/±4 | −1,2 … −6,3 điểm trên cả ba fold | **Loại.** Dự đoán từng clip tốt hơn bản làm mượt, dù nhãn liền kề đồng thuận 68 % |
| Hiệu chỉnh theo người, bản không nhãn | −3,2 / +0,1 / −2,5 | **Loại** cách này; trần oracle biết phân bố nhãn thật là +3,3 / +5,9 / +8,3, tức dư địa có thật nhưng chưa có phương pháp không nhãn với tới. Ghi vào bài như một trần đo được, không dùng |
| Đa dạng tổ hợp | tương quan lỗi giữa 4 thành viên **0,79**; oracle "chọn đúng thành viên" 0,783 so với trung bình 0,705 | Thêm seed cùng công thức là vô ích; cần thành viên **khác đầu vào** |

### Hướng chọn và vì sao

Ordinal ST-GCN đạt 71,24 trên validation **chỉ bằng chuỗi 78 điểm mốc trên 300 khung** — ngang tổ hợp pixel của tôi, bằng một đầu vào hoàn toàn khác. Luồng hình học hiện có của tôi là thống kê tóm tắt 16 khung đưa vào cây (0,67); phép thử 96 khung thất bại là thất bại của **cách tóm tắt**, không phải của phương thức. Một mô hình chuỗi trên điểm mốc theo từng khung là thứ chưa thử, có tiền lệ công bố ở mức 71, và có kỳ vọng lỗi ít tương quan với pixel hơn là seed với seed.

### Thiết kế, cố định trước

- **Dữ liệu:** chuỗi điểm mốc theo từng khung, **đủ 300 khung** (không lấy mẫu thưa), mỗi khung 204 toạ độ chuẩn hoá + 19 đại lượng gồm yaw/pitch/roll; cùng detector và cùng `frame_descriptor` với luồng hình học hiện có. Khung không có mặt ghi cờ và điền bằng khung gần nhất.
- **Hai kiến trúc thử, chọn bằng CV ba fold nội bộ:** (i) GRU hai chiều hai tầng, ẩn 256, gộp chú ý; (ii) transformer thời gian 4 tầng, d = 256, 4 đầu, mã hoá vị trí học được, gộp trung bình. Cả hai: dropout 0,3, AdamW, 40 epoch, EMA, chọn epoch trên fold nội bộ. Nhỏ, huấn luyện vài phút, có thể chạy CPU nếu GPU bận DAiSEE.
- **Hợp nhất:** trung bình xác suất với tổ hợp pixel bốn thành viên, **w = 0,5 cố định trước** (không dò, cùng lý do như bổ sung 5).

### Cổng, ghi trước — không mở split chính thức nếu không qua

1. Mô hình chuỗi một mình phải đạt ≥ 0,68 trung bình ba fold nội bộ (mốc hình học cũ 0,64–0,67).
2. Hợp nhất với tổ hợp pixel phải cho **≥ +0,015 accuracy trung bình ba fold và không âm ở ít nhất hai fold** so với tổ hợp pixel một mình. Đây là ngưỡng cao hơn lever hình học trước (+0,015 trên một fold) vì lần này mỗi lần mở là một chi phí về sức tuyên bố.
3. Qua cổng thì mới mở validation và test **một lần nữa**, cho đúng cấu hình đã khoá. Bài sẽ báo **cả hai cấu hình** đã mở (bổ sung 7 là "mô hình nền", cấu hình này là "mô hình đủ"), nói rõ mỗi split được chấm hai lần, mỗi lần một cấu hình khoá trước. Không có lần mở thứ ba.

### Dự báo, ghi trước

Mô hình chuỗi một mình: 0,66–0,70 ba fold (ST-GCN gợi ý đỉnh khoảng là khả thi). Hợp nhất: +0,01 đến +0,025 nếu tương quan lỗi với pixel dưới 0,6; ≈ 0 nếu trên 0,7. Xác suất qua cổng 2 tôi ước 40 %. Nếu không qua, kết quả chính thức ở bổ sung 8 là kết quả cuối, và phần này vào bài như một thí nghiệm âm có kiểm soát.

## Bổ sung lần 11, ngày 25/09/2026 — PHÁN QUYẾT CỔNG luồng chuỗi điểm mốc: KHÔNG QUA. Kết quả chính thức ở bổ sung 8 là kết quả cuối của EngageNet.

Nguồn: `results/SEQ_CV.json`, log `results/seq_cv.log`. Ba fold nội bộ, epoch chọn theo đường cong trung bình (epoch 18 cho cả hai kiến trúc), hợp nhất w = 0,5 cố định.

| Kiến trúc | Chuỗi một mình (3 fold) | TB | Cổng 1 ≥ 0,68 | Lợi hợp nhất so với tổ hợp pixel (3 fold) | TB | ≥ 0 | Cổng 2: TB ≥ +0,015 và ≥ 2 fold ≥ 0 |
|---|---|---|---|---|---|---|---|
| GRU hai chiều | 0,6356 / 0,6106 / 0,6392 | 0,6284 | không | −0,0004 / +0,0151 / +0,0212 | +0,0120 | 2/3 | **không** |
| Transformer thời gian | 0,6761 / 0,6001 / 0,6660 | 0,6474 | không | +0,0004 / +0,0043 / +0,0360 | **+0,0136** | **3/3** | **không, thiếu 0,0014** |

Tương quan lỗi giữa luồng chuỗi và tổ hợp pixel: 0,38–0,55, thấp hơn hẳn 0,79 giữa các thành viên pixel. Tính đa dạng mà bổ sung 10 kỳ vọng **là có thật**.

### Cách đọc, theo đúng luật đã ghi

1. **Cổng không qua, nên không mở split chính thức lần hai.** `seq_official.py` tự đọc file này và từ chối chạy; đã kiểm. Kết quả chính thức ở bổ sung 8 — test 70,04 / 70,26 so với 69,29 — là kết quả cuối của EngageNet.
2. **Transformer trượt cổng 2 đúng 0,0014.** Lợi ích dương ở cả ba fold, lớn nhất ở fold khó nhất (+3,6 điểm ở fold 2). Đây là bằng chứng *gợi ý*, không phải bằng chứng đủ theo tiêu chuẩn tôi tự đặt trước khi chạy — và tiêu chuẩn được đặt chính xác để một kết quả gợi ý không tiêu tốn một lần mở split. Hạ ngưỡng xuống 0,013 sau khi thấy 0,0136 là đúng cái tôi đã cấm ở bổ sung 6: "trượt thì loại, không cứu bằng cách đổi ngưỡng".
3. **Luồng chuỗi một mình chỉ đạt 0,65**, thấp hơn nhiều so với 71,24 của Ordinal ST-GCN. Ba giải thích có thể, chưa phân biệt được: kiến trúc tích chập đồ thị trên cấu trúc điểm mốc của họ mang thông tin mà GRU/transformer phẳng không có; 78 điểm mốc MediaPipe của họ khác 68 điểm mốc 3D InsightFace của tôi; hoặc con số của họ lạc quan do chọn trên validation 11 người. Không kết luận cái nào.
4. **Vào bài như một thí nghiệm âm có kiểm soát**, với đủ ba số trên: ghép hai họ đầu vào mạnh nhất của bảng xếp hạng cho lợi ích +0,012 đến +0,014 trên ba fold nội bộ, dưới mức thiết kế phân giải được, dù hai họ mắc lỗi khác nhau rõ rệt.

### Nếu tác giả muốn đi tiếp hướng này sau khi nộp

Chỉ với một bản ghi trước mới, và bài phải liệt kê đủ mọi vòng đã thử. Ứng viên có căn cứ: tích chập đồ thị đúng theo ST-GCN thay vì mô hình chuỗi phẳng, và số epoch ngắn hơn (đường cong trung bình đạt đỉnh ở 18/40, tức 40 epoch đã quá khớp). Mỗi vòng thêm là một lần "thử tới khi qua", làm yếu tuyên bố; phải cân nhắc có đáng không khi lợi ích kỳ vọng vẫn nằm trong khoảng tin cậy ±7,5 điểm của tập test.

## Bổ sung lần 12, ngày 25/09/2026 — DAiSEE: công thức EngageNet thất bại, kích hoạt điều khoản dự phòng của bổ sung 9. Ghi trước khi sweep.

### Tín hiệu từ lần mở chính thức đang chạy (G9, chưa có tổ hợp)

Từng thành viên, train → test: acc 0,530 / 0,570 / 0,536 / 0,517; UAR 0,27–0,30 (ngẫu nhiên 0,25); chỉ dự đoán lớp 2 và 3; sàn test 0,4944. Train → validation: 0,596–0,600 so với sàn 0,5689. Loss huấn luyện còn giảm ở epoch 11 (0,905 → 0,896): **chưa học xong**, khác hẳn EngageNet cùng công thức (loss 0,63 ở epoch 11). Dữ liệu đã kiểm: mặt tìm được 100 %, 300 khung/clip, mặt 167 px, ảnh cắt đúng. Nhãn train [34, 213, 2 617, 2 494]: 95 % dữ liệu ở hai lớp, nên accuracy gần như là bài toán 2-so-với-3 với prior 51/49.

Điều khoản bổ sung 9: "nếu công thức EngageNet cho dưới 63,9 thì mới được phép chọn siêu tham số riêng trên train DAiSEE bằng CV nội bộ". Đã dưới. Lưu ý thêm: pipeline cũ của nhóm (đặc trưng đóng băng, `NEGCTRL`) cũng cho DAiSEE 0,488, **dưới cả sàn** — DAiSEE là nơi cả hai thế hệ pipeline đều thất bại, và kiểm toán nhãn của nhóm (đồng thuận liền kề 0,58 so với 0,54 ngẫu nhiên, κ ≈ 0,09) nói nhãn gần nhiễu. Kỳ vọng phải thấp.

### Sweep, cố định trước

Ba fold theo người (`ClipID[:6]`) trên train DAiSEE, sàng lọc trên fold 0 rồi xác nhận cấu hình tốt nhất trên fold 1 và 2. Bốn giả thuyết, mỗi cái một lần chạy:

| | Giả thuyết | Cấu hình |
|---|---|---|
| A | chưa học xong | công thức khoá nhưng **36 epoch** |
| B | LoRA-16 quá chật cho miền khó hơn | **fine-tune toàn phần**, 12 epoch, lr 5e-4, layer-decay 0,75 |
| C | cả hai | fine-tune toàn phần, 24 epoch |
| D | tín hiệu 2-so-với-3 nằm ở động học mịn | công thức khoá nhưng **bước 4** (cửa sổ 2 s) |

Giữ mọi thứ khác nguyên: EMA 0,999, batch 16 × 2, TTA 5 cửa sổ, nhãn mềm 0,1.

### Cổng cho lần mở DAiSEE thứ hai, ghi trước

Cấu hình thắng phải đạt **≥ 0,60 accuracy trung bình ba fold nội bộ** (sàn train 0,488, tức lift +11, tương đương lift của các mốc công bố) **và hơn công thức khoá ≥ 0,03 trên cùng ba fold**. Không qua thì DAiSEE vào bài với đúng con số của G9, kèm khoảng tin cậy và phân bố dự đoán, như một kết quả âm được báo đầy đủ; bài không tuyên bố gì trên DAiSEE. Qua thì mở đúng một lần nữa, báo cả hai cấu hình.

Song song, trích chuỗi điểm mốc 300 khung cho DAiSEE trên CPU rỗi: trên EngageNet luồng này trượt cổng 0,0014, nhưng ở DAiSEE luồng pixel yếu hơn nhiều nên vai trò có thể khác; chỉ dùng nếu qua cùng loại cổng như bổ sung 10.

## Bổ sung lần 13, ngày 25/09/2026 — KẾT QUẢ CHÍNH THỨC DAiSEE, cấu hình khoá ở bổ sung 7, mỗi split mở đúng một lần. Ghi đúng con số, không giải thích thêm ngoài những gì đã ghi trước ở bổ sung 12.

Nguồn: `results/G9_DAISEE_OFFICIAL.json`, log `logs/g9.log` trên máy chủ. Bốn thành viên (SVFAP seed 0/1/2, MAE-DFER seed 0), tổ hợp trung bình xác suất, TTA 5 cửa sổ. Khoảng tin cậy 95 % bootstrap theo người, 2 000 lần.

| Cấu hình | n / người | Sàn | Tổ hợp | KTC 95 % | UAR | macro-F1 | Phân bố dự đoán [0,1,2,3] | Thành viên |
|---|---|---|---|---|---|---|---|---|
| train → validation | 1 429 / 19 | 0,5689 | **0,6004** | [0,5626; 0,6339] | 0,296 | 0,277 | [0, 0, 1 197, 232] | 0,600 / 0,600 / 0,598 / 0,596 |
| train → test | 1 784 / 21 | 0,4944 | **0,5331** | [0,4911; 0,5757] | 0,279 | 0,271 | [0, 0, 1 064, 720] | 0,530 / 0,570 / 0,536 / 0,517 |
| train + validation → test | 1 784 / 21 | 0,4944 | **0,5723** | [0,5319; 0,6122] | 0,297 | 0,282 | [0, 0, 1 299, 485] | 0,572 / 0,578 / 0,563 / 0,548 |

Mốc đã kiểm chứng trên test DAiSEE: ResNet + TCN 63,9; Selim 67,48 (split theo lời tác giả, chưa kiểm được); PriorNet 69,06 (gộp train + val). **Cấu hình khoá thua mọi mốc**, kể cả mốc thấp nhất, ở cả hai giao thức: −6,7 điểm so với 63,9 và −11,8 so với 69,06. Khoảng tin cậy không chạm 63,9.

Ba điều đọc được từ đúng con số, không hơn:

1. Hai lớp thấp không bao giờ được dự đoán ở cả ba lần mở; UAR 0,28–0,30 gần mức ngẫu nhiên 0,25. Mô hình thực chất học một ngưỡng giữa lớp 2 và lớp 3.
2. Tổ hợp bốn thành viên trên test **không cao hơn** thành viên tốt nhất (0,5723 so với 0,5779; 0,5331 so với 0,5701): các thành viên mắc lỗi giống nhau, khác hẳn EngageNet nơi tổ hợp cộng thêm.
3. Gộp validation vào train cộng +3,9 điểm trên test, cùng chiều và lớn hơn mức +1,7 của mốc ICMI trên EngageNet; ở đây thêm 1 429 clip, tức +32 % dữ liệu, cho một mô hình còn chưa học xong (bổ sung 12).

Dự báo ở bổ sung 9 ("công thức EngageNet, không chỉnh gì") không ghi con số cụ thể; tín hiệu sớm ở bổ sung 12 (0,52–0,57 từng thành viên) đã đúng. Điều khoản dự phòng đang chạy (G11). Nếu G11 không qua cổng, đây là con số DAiSEE vào bài.

### Tín hiệu đầu tiên của G11, ghi lúc 09:33 UTC (giả thuyết A, fold 0, chưa xong)

Công thức khoá kéo dài 36 epoch trên fold 0: accuracy đỉnh 0,5782 ở epoch 20 rồi giảm đơn điệu về 0,5568 ở epoch 33 trong khi loss huấn luyện tiếp tục giảm 0,896 → 0,838. **Huấn luyện dài hơn không chữa được**: mô hình quá khớp trước khi hai lớp thấp được học. Luật sàng lọc lấy epoch cuối, nên A sẽ thua tham chiếu 12 epoch. Ghi lại để không ai đọc đỉnh 0,5782 như kết quả của A.

## Bổ sung lần 14, ngày 25/09/2026 — luồng chuỗi điểm mốc trên DAiSEE. Ghi trước khi huấn luyện, sau khi trích xong 9 068 chuỗi (0 lỗi).

Vì sao vẫn thử dù trên EngageNet luồng này trượt cổng: ở EngageNet nó phải cộng vào một luồng pixel đã đạt 0,69; ở DAiSEE luồng pixel chỉ 0,57 và bốn thành viên mắc lỗi giống nhau (tổ hợp không cộng thêm, bổ sung 13), nên một luồng có lỗi khác kiểu có dư địa lớn hơn. Dữ liệu đã có sẵn, mô hình nhỏ, không tốn thêm lần mở nào.

**Thiết kế, cố định:** `code/seqmodel_daisee.py`. Cùng hai kiến trúc, cùng siêu tham số, cùng augment, cùng cách chọn epoch từ đường cong trung bình như bổ sung 10 — không chỉnh gì cho DAiSEE. Ba fold theo người **trùng đúng** với fold của `ft_video.py --dataset daisee --cv-folds 3` (cùng seed 12345, cùng cách lọc clip theo crop). Hợp nhất w = 0,5 cố định với (i) công thức khoá trên cùng fold (các lần chạy `g11_ref_locked*`) và (ii) cấu hình thắng sweep G11 nếu có.

**Cổng, ghi trước:** dùng **cổng của bổ sung 12**, không phải bổ sung 10, vì bất kỳ cấu hình nào muốn mở DAiSEE lần hai đều phải qua cùng một cửa: hợp nhất trên ba fold đạt **trung bình ≥ 0,60 và hơn luồng pixel cùng fold ≥ 0,03**. Có một lần mở thứ hai duy nhất cho DAiSEE; nếu cả sweep G11 lẫn hợp nhất chuỗi cùng qua, chọn cái có trung bình ba fold cao hơn, ghi rõ cả hai. Không qua thì luồng chuỗi DAiSEE vào bài như kết quả âm có kiểm soát, cùng cách như bổ sung 11.

**Dự báo, ghi trước:** chuỗi một mình 0,55–0,60 ba fold (sàn train 0,488). Lợi hợp nhất +0,01 đến +0,04. Xác suất qua cổng: 20 %.

## Bổ sung lần 15, ngày 26/09/2026 — PHÁN QUYẾT DAiSEE: cả sweep G11 lẫn luồng chuỗi đều KHÔNG QUA cổng. Không mở DAiSEE lần hai. Con số ở bổ sung 13 là con số DAiSEE cuối cùng của bài. Chiến dịch thực nghiệm kết thúc.

Nguồn: `results/G11_DAISEE_SWEEP.json`, `results/SEQD_CV.json`, log `logs/g11.log`, `logs/seqd_cv.log`, `logs/seqd_fuse.log`. Ba fold theo người của train DAiSEE (1 823 / 1 665 / 1 870 clip, 23 người mỗi fold), sàn từng fold 0,4931 / 0,5327 / 0,4872.

### Sweep G11 (bổ sung 12), epoch cuối, không chọn đỉnh

Sàng lọc fold 0: B fine-tune toàn phần 12 epoch **0,5897**; C toàn phần 24 epoch 0,5826; tham chiếu khoá 0,5754; D cửa sổ 2 s 0,5639; A khoá 36 epoch 0,5535. Bốn giả thuyết, đúng một cái vượt tham chiếu quá 1 điểm. A thấp nhất: kéo dài huấn luyện làm hại (đỉnh 0,578 ở epoch 20, tụt về 0,554). D thấp hơn tham chiếu: tín hiệu không nằm ở động học mịn.

Xác nhận B trên ba fold: **0,5897 / 0,5772 / 0,5251, trung bình 0,5640**; tham chiếu khoá 0,5754 / 0,5682 / 0,5144, trung bình 0,5527; chênh **+0,0113**. Cổng đòi ≥ 0,60 và ≥ +0,03: **không qua** cả hai điều kiện. B vẫn chỉ dự đoán lớp 2 và 3, UAR 0,30.

### Luồng chuỗi điểm mốc (bổ sung 14)

| | Chuỗi một mình, 3 fold | TB | Hợp nhất với khoá: lợi 3 fold | TB | Hợp nhất với B: lợi 3 fold | TB |
|---|---|---|---|---|---|---|
| GRU | 0,5496 / 0,5357 / 0,5332 | 0,5395 | −0,0022 / −0,0162 / +0,0091 | **−0,0031** | −0,0154 / −0,0132 / +0,0171 | **−0,0038** |
| Transformer | 0,5568 / 0,5333 / 0,5380 | 0,5427 | +0,0044 / −0,0150 / +0,0086 | **−0,0007** | +0,0077 / −0,0060 / +0,0070 | **+0,0029** |

Chuỗi một mình chỉ hơn sàn 3,5–4 điểm, và cũng chỉ dự đoán lớp 2 và 3. Tương quan lỗi với pixel 0,29–0,50, tức hai luồng sai khác kiểu thật, nhưng hợp nhất không cộng thêm gì: bốn tổ hợp cho −0,004 đến +0,003. Dự báo ghi trước (chuỗi 0,55–0,60, lợi +0,01 đến +0,04) **sai về phía lạc quan** ở cả hai vế.

### Cách đọc, theo đúng luật đã ghi

1. DAiSEE không được mở lần hai. Con số vào bài là bổ sung 13: test 0,533 (train) và 0,572 (train + val), KTC [0,532; 0,612], dưới mọi mốc đã kiểm chứng. Bài không tuyên bố gì trên DAiSEE.
2. Bài phải kể đủ mọi thứ đã thử trên DAiSEE để không ai đọc 0,572 là "chưa cố": công thức khoá; bốn giả thuyết sweep; hai kiến trúc luồng chuỗi và bốn tổ hợp hợp nhất. Tất cả trên fold nội bộ của train, không đụng test.
3. Một quan sát nhất quán qua mọi cấu hình, đáng ghi vào phần thảo luận với đúng mức độ: **không cấu hình nào, pixel hay hình học, LoRA hay toàn phần, học được hai lớp thấp**, và mọi cấu hình đều dừng ở 0,55–0,59 trên fold nội bộ. Kết hợp với kiểm toán nhãn của nhóm (κ ≈ 0,09 giữa clip liền kề), đây là bằng chứng gián tiếp, không phải kết luận, rằng phần lớn khoảng cách tới 63,9–69,06 không nằm ở mô hình. Không được viết mạnh hơn thế, và phải nói rõ ta không tái lập được các mốc đó nên không loại trừ lỗi ở phía ta.
4. Chiến dịch thực nghiệm kết thúc. Không thử thêm cấu hình nào trên cả hai bộ. Việc còn lại là viết.
