# Thiết kế mô hình v2 — bản ghi trước, viết 27/09/2026 TRƯỚC khi chạy bất cứ gì

Tác giả yêu cầu: bài phải đề xuất một **mô hình**, không phải một bài đo lường. File này thiết kế lại mô hình để cơ chế là đóng góp, và ghi trước mọi ngưỡng trước khi chạy. Chiến dịch v1 đã đóng ở bổ sung 15 của `MODEL_PAPER_DESIGN_2026-09-21.md`; đây là bản ghi trước mới, đúng theo điều kiện đã đặt ở bổ sung 11 ("chỉ với một bản ghi trước mới, và bài phải liệt kê đủ mọi vòng đã thử").

Số liệu nền: `FACTS_MODEL_PAPER.md`. Mọi con số trong file này lấy từ đó hoặc từ hai phép đo mới ghi ở mục 2.

---

## 1. Luận điểm của mô hình, và vì sao nó là cơ chế chứ không phải mẹo

**Engagement là một trạng thái biến thiên chậm, được quan sát qua một cảm biến không đáng tin.** Hai hệ quả, mỗi hệ quả một thành phần, và mỗi hệ quả đã có bằng chứng đo được từ chiến dịch v1.

**Hệ quả 1 — phải lấy mẫu cả episode, và có một điểm gãy giữa phạm vi và độ phân giải.** Chiến dịch v1 đo được: giữ nguyên 16 khung mà mở phạm vi cửa sổ từ 61 khung (20 % clip) lên toàn clip cho **+0,0514 macro-F1**, lớn khoảng tám lần đòn bẩy đơn kế tiếp. Đó là trục **phạm vi** ở ngân sách khung cố định. Trục còn lại — **ngân sách khung ở phạm vi cố định** — chưa ai đo, kể cả nhóm. Hai trục cùng thuộc một tham số thiết kế; đo cả hai cho một phát biểu chuyển được sang bài khác: với trạng thái biến thiên chậm, phủ hết episode quan trọng hơn lấy dày, và lợi ích của lấy dày tắt sau một điểm gãy.

**Hệ quả 2 — mức hữu ích của từng phương thức thay đổi theo chất lượng quan sát, nên phép hợp nhất phải có điều kiện.** Bằng chứng v1: lợi ích hợp nhất ở trọng số cố định w = 0,5 biến động **−0,0139 / −0,0310 / +0,0340** qua ba fold, cả ba khoảng tin cậy ghép cặp đều chứa 0; tương quan lỗi giữa luồng chuỗi điểm mốc và tổ hợp pixel là 0,38–0,55, so với 0,79 giữa các thành viên pixel; và **13 % clip EngageNet có khung không tìm được mặt** (tỉ lệ trung bình 0,919 mỗi clip, chỉ 87,1 % clip đạt 100 %). Đọc đúng: **phương thức thứ hai không sai, phép kết hợp sai.** Một trọng số hằng không thể vừa đúng cho clip mặt sạch vừa đúng cho clip mất mặt một phần.

Đây là điểm tôi phải nói thẳng: v1 báo hợp nhất và tổ hợp là hai thành phần **không trả được cái giá của chúng**. Mô hình v2 không xoá kết quả đó; nó lấy chính kết quả đó làm động cơ, và nếu v2 thắng thì câu chuyện là *cơ chế kết hợp*, không phải *thêm phương thức*.

---

## 2. Hai phép đo mới, làm ngày 27/09/2026, quyết định tính khả thi

**Positional embedding của SVFAP là sin-cos cố định, không phải tham số học được** (`get_sinusoid_encoding_table` ở `repos/SVFAP/modeling_finetune.py:289`, dùng ở dòng 343, và `detach()` ở dòng 463; bảng được sinh từ `num_patches` suy ra từ `all_frames`). Hệ quả: **đổi ngân sách khung là đổi một tham số hàm khởi tạo**, bảng vị trí được sinh lại bằng giải tích, không lệch checkpoint, không cần nội suy. Trục ngân sách khung tốn đúng chi phí tính toán, không tốn công cài đặt.

**Cache mặt đã lưu đủ tín hiệu độ tin cậy, không cần trích xuất thêm gì.** Mỗi file `crops_face/engagenet/<key>.npz` có `found` (300, bool mỗi khung), `box` (300×4, float32 mỗi khung), `fps`, `hw`. Toàn bộ véc-tơ độ tin cậy ở mục 3.3 tính được từ đây với chi phí bằng không.

Máy chủ: L4 23 GB, 0 MiB đang dùng, 0 người đăng nhập.

---

## 3. Mô hình, cố định trước

### 3.1. Luồng diện mạo và **lưới hai trục thời gian**

SVFAP-B, LoRA hạng 16 trên q, k, v — giữ nguyên vì đã đo (+0,0233 accuracy ở lịch 30 epoch). Thay đổi: ngân sách khung `T` và bước `s` được chọn sao cho **span nằm trong 280–290 khung ở cả ba cấu hình**, tức phạm vi giữ gần như không đổi ở khoảng 95 % clip, và **chỉ ngân sách khung thay đổi**:

| Cấu hình | T | s | span = (T−1)s+1 | phủ | tần số trong cửa sổ |
|---|---|---|---|---|---|
| đối chứng | 16 | 19 | 286 | ~95 % | 1,6 Hz |
| B32 | 32 | 9 | 280 | ~93 % | 3,3 Hz |
| B48 | 48 | 6 | 283 | ~94 % | 5,1 Hz |

**Chọn bước như vậy là một quyết định có chủ ý, không phải mặc định.** Ở v1, bước 20 cho span 301 > 300 khung nên **hơn 92 % clip EngageNet rơi vào nhánh clip-ngắn**: một cửa sổ tất định, không nhiễu vị trí, và năm cửa sổ TTA giống hệt nhau nên TTA vô tác dụng. Ba cấu hình trên có span ≤ 286 ≤ 300 nên **toàn bộ đi nhánh clip-dài**: điểm bắt đầu ngẫu nhiên, nhiễu vị trí ±s/2, và năm cửa sổ TTA khác nhau thật. Điều này cũng đưa EngageNet về **cùng nhánh với DAiSEE** (span 286 ở bước 19), tức xoá luôn sai lệch C8 mà v1 phải khai.

Phụ phẩm miễn phí: so đối chứng T=16/s=19 với con số v1 T=16/s=20 cho **phép đo tác động của nhánh lấy mẫu**, tức trả lời chính câu hỏi C8 để ngỏ.

Mọi thứ khác giữ nguyên công thức đã khoá: 12 epoch, khởi động 2, EMA 0,999, lr 1e-3, layer-decay 0,75, nhãn mềm 0,1, TTA 5 cửa sổ. Batch giảm và tích luỹ tăng khi T lớn để giữ batch hiệu dụng 32.

### 3.2. Hai luồng còn lại, đã có sẵn, chi phí bằng không

Hình học 1 338 chiều + GBDT (một mình 0,6712 trên fold 0) và luồng chuỗi điểm mốc 300 khung × 223 chiều với transformer thời gian (một mình 0,6474 trung bình ba fold, tương quan lỗi với pixel 0,38–0,55). Cả hai đã trích xuất xong cho toàn bộ clip EngageNet và DAiSEE.

### 3.3. **Cổng độ tin cậy** — thành phần chính thứ hai

Véc-tơ độ tin cậy `r` mỗi clip, mười chiều, tính từ `found` và `box` đã cache:

1. tỉ lệ khung tìm được mặt; 2. độ dài chuỗi mất mặt dài nhất chia 300; 3. diện tích mặt trung bình chia diện tích khung; 4. độ lệch chuẩn của diện tích mặt; 5. dịch chuyển tâm hộp trung bình giữa hai khung liền kề; 6. phân vị 95 của dịch chuyển đó; 7. tỉ lệ khung hộp chạm biên khung hình; 8. fps của clip (14 người có clip 1 000 fps là một nhóm thật); 9. số khung giải mã được; 10. tỉ lệ khung mà diện tích mặt dưới nửa trung vị của clip.

Cổng `g`: MLP 10 → 16 → 3, softmax cho ba trọng số luồng. Xác suất cuối là tổ hợp lồi ba luồng với trọng số `g(r)`.

**Ràng buộc chống rò rỉ, và một thay đổi so với cam kết cũ phải khai.** Bổ sung 5 của v1 ghi "không dùng stacking, không học trọng số hợp nhất trên tập đánh giá ở bất kỳ mức nào". Cổng độ tin cậy **là** một bộ kết hợp học được, nên đây là một thay đổi so với cam kết đó và tôi khai rõ. Lý do chấp nhận được: cái mà luật cũ bảo vệ là *không dò trọng số trên tập đánh giá*, và ràng buộc đó được giữ nguyên tuyệt đối. Cổng chỉ được khớp trên **dự đoán ngoài fold của tập train**, theo sơ đồ lồng độc lập theo người: với mỗi fold nội bộ k, ba luồng huấn luyện trên hai fold còn lại, sinh dự đoán cho fold k, và cổng khớp trên hợp của các dự đoán ngoài fold đó. Không một tham số nào của cổng nhìn thấy validation hay test. Khi chạy chính thức, cổng được khớp trên dự đoán ngoài fold của toàn bộ train rồi đóng băng.

Kèm theo, **một biến thể đơn giản hơn phải báo cạnh nó**: định tuyến cứng theo một ngưỡng duy nhất (`nếu tỉ lệ tìm được mặt < τ thì dùng hình học, ngược lại dùng diện mạo`), τ chọn trên fold nội bộ. Nếu định tuyến cứng đạt gần bằng cổng học được thì cơ chế là *điều kiện hoá*, không phải *dung lượng của cổng*, và bài phải nói vậy.

### 3.4. Hai đòn bẩy phụ, **chỉ chạy nếu hai thành phần chính qua cổng**

- **Đầu ra thứ bậc CORAL** thay softmax bốn lớp. Động cơ đo được: khoảng cách UAR 0,5625 so với accuracy 0,7021 trên test EngageNet là lớn, và trên DAiSEE không cấu hình nào dự đoán hai lớp thấp với UAR đứng ở 0,28–0,30. Nhãn mềm LDL Gauss đã thử và trượt (−0,0016 macro-F1), nhưng LDL **không phải** hồi quy thứ bậc nhất quán theo hạng; đây là thứ khác và chưa thử.
- **Bộ gộp thời gian nhỏ** trên embedding các đoạn, hai lớp, để mô hình hoá tính bền của trạng thái thay vì gộp trung bình.

---

## 4. Cổng, ghi trước. Chọn bằng **UAR và macro-F1**, không bằng accuracy

Ba fold nội bộ độc lập theo người của train, đúng cách dựng fold của v1 (`rng = default_rng(12345)`, `fold = i % 3`). Trung bình ba fold. Epoch cố định 12, không chọn epoch theo đường cong.

**Điểm này là sửa một sai lệch của v1.** Ở v1 tôi ghi trước luật chọn theo UAR và macro-F1 nhưng khi chọn lịch huấn luyện lại chọn theo accuracy (bổ sung C7). Lần này luật là luật: **mọi quyết định chọn cấu hình dùng macro-F1 làm chỉ số chính và UAR làm chỉ số phụ; nếu hai chỉ số đó không đồng thuận thì cấu hình không được chọn và bài báo cáo sự không đồng thuận đó.** Accuracy chỉ dùng để đặt cạnh bảng xếp hạng, không dùng để chọn.

| Cổng | Điều kiện | Nếu trượt |
|---|---|---|
| **G-A, ngân sách khung** | cấu hình ngân sách tốt nhất hơn đối chứng T=16/s=19 **≥ +0,015 macro-F1** trung bình ba fold **và** không âm ở ≥ 2 fold | trục ngân sách vào bài như thí nghiệm âm; mô hình v2 chỉ còn cổng độ tin cậy |
| **G-B, cổng độ tin cậy** | hợp nhất có cổng hơn **luồng đơn tốt nhất ≥ +0,015 macro-F1** trung bình ba fold **và** hơn **hợp nhất w = 0,5 cố định ≥ +0,010** | cổng vào bài như thí nghiệm âm |
| **G-C, mở split chính thức** | **cả G-A và G-B đều qua** | không mở lần thứ ba; kết quả v1 là kết quả cuối và bài giữ trục cũ |
| **G-D, đòn bẩy phụ** | mỗi cái ≥ +0,010 macro-F1 trung bình và không âm ở ≥ 2 fold, chạy sau khi G-A và G-B đã xong | loại, báo đủ số |

**DAiSEE: chuyển giao thuần, không tuyên bố gì, không có cổng.** Chạy đúng cấu hình đã khoá của v2, báo con số kèm khoảng tin cậy và phân bố dự đoán. Nhãn DAiSEE có κ ≈ 0,09 và v1 thua mốc thấp nhất 6,7 điểm; không đặt kỳ vọng.

---

## 5. Dự báo, ghi trước

| | Dự báo của tôi | Xác suất qua ngưỡng |
|---|---|---|
| Ngân sách 16 → 32 khung ở phạm vi cố định | +0,010 đến +0,030 macro-F1 | qua +0,015: **50 %** |
| Ngân sách 48 khung so với 32 | ≤ 32 khung (lợi ích tắt, và 7 879 clip dễ quá khớp) | 48 > 32: **30 %** |
| Cổng độ tin cậy so với w = 0,5 cố định | +0,010 đến +0,030 | qua cả hai ngưỡng: **45 %** |
| Nhánh lấy mẫu, s=19 so với s=20 ở T=16 | −0,005 đến +0,015 (nhánh dài có tăng cường thật) | — |
| CORAL, nếu chạy | accuracy 0 đến +0,010; **UAR +0,02 đến +0,05** | qua +0,010 macro-F1: 40 % |
| **EngageNet test nếu cả G-A và G-B qua** | **71 đến 73** | vượt 69,29 về điểm ước lượng: **70 %** |

**Một điều phải nói trước và không được quên khi viết bài: không thay đổi mô hình nào làm biên số trở thành có ý nghĩa thống kê trên benchmark này.** Độ rộng khoảng tin cậy do 26 người trong tập test quyết định, không do mô hình; nó sẽ vẫn khoảng 15 điểm. Xác suất chặn dưới của khoảng tin cậy vượt 69,29 tôi ước **dưới 5 %**. Nghĩa là bài có thể đặt mô hình làm tuyên bố chính, nhưng phần khoảng tin cậy **không được bỏ** — nó chuyển từ tuyên bố chính sang phần hạn chế, và câu "điểm ước lượng cao hơn nhưng benchmark không tách được về thống kê" vẫn phải có trong bài.

---

## 6. Chi phí và giá phải trả

Ước tính trên L4 đang rỗi: lưới ngân sách 3 cấu hình × 3 fold ≈ 45 giờ; cổng độ tin cậy dùng lại dự đoán đã lưu nên gần như miễn phí; hai đòn bẩy phụ ≈ 30 giờ nếu chạy; lần mở chính thức 4 thành viên × 2 giao thức ≈ 40 giờ; DAiSEE ≈ 20 giờ. Tổng khoảng **115–135 giờ GPU, tức năm đến bảy ngày**.

**Giá phải trả về tính chính trực, và đây là quyết định của tác giả.** Tập test EngageNet đã được chấm hai lần (một lần cho mỗi giao thức huấn luyện) với mô hình v1. Mở lần thứ ba cho v2 nghĩa là bài **phải liệt kê cả ba lần mở** và người đọc phải thấy rõ mô hình v2 đã được khoá trước lần mở đó. Điều này làm yếu đi câu "mỗi split mở đúng một lần" và thêm một mục lịch sử phát triển mô hình vào bài. Đổi lại, bài có một mô hình để đề xuất. Không có cách nào tránh cái giá này ngoài việc không mở lần thứ ba.

## 7. Nếu cổng không qua

Nếu G-A và G-B đều trượt thì **không mở split lần thứ ba**, kết quả v1 là kết quả cuối, và bài buộc phải quay về trục đo lường — vì khi đó ta không có cơ chế nào chứng minh được. Tôi ghi điều này trước để không có chuyện sau khi thấy số rồi mới đi tìm cách kể.

---

## 8. Kết quả tạm, ghi 27/09/2026 lúc 14:00 UTC — sáu trong chín lần chạy đã xong. **Trục ngân sách khung TRƯỢT, và trượt rõ.**

Nguồn: `results/FT_v2_T{16,32}_f{0,1,2}_train.json`, log `logs/v2a.log`, xem bằng `code/peek_v2a.py` (kịch bản này **không** ghi `GATE_V2A.json`, để một lần xem tạm không bị nhầm thành phán quyết).

| Cấu hình | fold 0 | fold 1 | fold 2 | TB macro-F1 | TB UAR | TB acc |
|---|---|---|---|---|---|---|
| đối chứng T=16, s=19, span 286 | 0,5939 | 0,5535 | 0,6008 | **0,5827** | 0,5658 | 0,6696 |
| B32, T=32, s=9, span 280 | 0,5740 | 0,4721 | 0,5711 | **0,5391** | 0,5321 | 0,6437 |

Hiệu số B32 so với đối chứng: macro-F1 **−0,0199 / −0,0814 / −0,0297, trung bình −0,0437, không âm ở 0/3 fold**. UAR −0,0337. Accuracy −0,0259. Ngưỡng ghi trước là ≥ +0,015 và không âm ở ≥ 2 fold: **trượt, và trượt xa gấp khoảng ba lần ngưỡng về phía sai**.

**Dự báo của tôi sai hẳn hướng.** Tôi ghi trước "+0,010 đến +0,030, xác suất qua cổng 50 %". Thực tế −0,0437. Đây là lần dự báo sai nặng nhất của tôi trong cả hai chiến dịch và phải ghi đúng như vậy trong bài.

### Đã loại trừ một lời giải thích tầm thường trước khi diễn giải

Nghi vấn đầu tiên phải kiểm là **trọng số tiền huấn luyện có bị âm thầm bỏ khi đổi số khung hay không**. Kiến trúc SVFAP có `temporal_seq_len = num_frames / tubelet_size` (8 ở 16 khung, 16 ở 32 khung), ba tầng với `Conv1d` hạ mẫu thời gian hệ số 2 giữa các tầng, và bộ tóm tắt bottleneck cố định 8 token. Nếu tham số nào có hình phụ thuộc độ dài thời gian thì việc nạp sẽ lệch hình.

Kiểm: `ft_video.py:169` dùng `load_state_dict(sd, strict=False)` **không lọc theo hình**, và PyTorch vẫn báo lỗi khi lệch hình dù `strict=False`. Lần chạy 32 khung không lỗi và báo đúng 4 khoá thiếu (`fc_norm`, `head`) giống hệt ở 16 khung. **Kết luận: toàn bộ trọng số tiền huấn luyện đã nạp đủ; kết quả không phải hiện vật của việc nạp.**

### Ba cách giải thích, chưa phân biệt được, không chọn cái nào

1. **Lệch giữa tiền huấn luyện và tinh chỉnh về thang thời gian.** SVFAP tiền huấn luyện ở 16 khung, nên các `Conv1d` hạ mẫu và bộ bottleneck học cách làm việc trên 8 → 4 → 2 token thời gian. Ở 32 khung chúng thấy 16 → 8 → 4. Trọng số tương thích về hình nhưng thang cấu trúc thời gian chúng đã học không còn khớp. Kiểm được, nhưng đòi tiền huấn luyện lại.
2. **Ngân sách thích nghi cố định.** LoRA hạng 16 trên 39 khối cho 1 058 820 tham số huấn luyện được, giữ nguyên ở mọi ngân sách khung, nhưng phải thích nghi trên số token gấp đôi trong cùng 12 epoch.
3. **Dư thừa cộng quá khớp.** Với một trạng thái biến thiên chậm, khung thêm vào ở 3,3 Hz gần như trùng thông tin với khung liền kề ở 1,6 Hz, nên chúng tốn dung lượng mà không thêm thông tin.

### Điều này nói gì về luận điểm của mô hình

Cách đọc số này mà **không** phóng đại: hệ quả 1 của luận điểm ở mục 1 nói phải lấy mẫu cả episode. v1 đo được **phạm vi** có giá trị lớn (+0,0514 macro-F1). v2 vừa đo được **độ phân giải** trong cùng phạm vi **không** có giá trị, và ở 3,3 Hz thì có hại. Hai nửa hợp lại thành một phát biểu sắc hơn cả hai: *với engagement, phủ hết episode là điều quan trọng, còn lấy dày hơn 1,6 Hz thì không, và vượt một mức nào đó thì tốn dung lượng mà không thêm thông tin.* Đây là một phát hiện đầy đủ hơn phiên bản v1, và nó có được chính vì trục thứ hai đã được đo và trượt.

Nhưng nó là **một phát hiện, không phải một thành phần mô hình**. Trục ngân sách khung không cho ta cái gì để đề xuất. Vì vậy sức nặng của bài mô hình giờ dồn toàn bộ vào cổng độ tin cậy (G-B).

### Hai phép đo phụ, miễn phí, đều có giá trị

**Nhánh lấy mẫu (câu hỏi C8 mà v1 để ngỏ).** Đối chứng T=16/s=19 (span 286, nhánh clip-dài, có nhiễu vị trí, năm cửa sổ TTA khác nhau thật) so với v1 T=16/s=20 (span 301, nhánh clip-ngắn, không nhiễu vị trí, TTA vô tác dụng), cùng ba fold: macro-F1 −0,0022 / −0,0072 / +0,0172, **trung bình +0,0026**; accuracy trung bình +0,0062. Tức **nhánh lấy mẫu gần như không ảnh hưởng**. Hạn chế C8 mà bản thảo v1 phải khai giờ có một phép đo đi kèm, và phép đo nói nó nhỏ. Lưu ý khi báo: seed không ghép cặp và các lần chạy v1 là một seed, nên đây là ước lượng thô.

**Tái lập.** Đối chứng mới đạt TB macro-F1 0,5827 so với 0,5801 của v1 trên cùng ba fold, với một bước khác và một đường mã khác. Pipeline tái lập được chính nó.

### Quyết định về T=48: **chạy tiếp cho xong**

Đang chạy fold 0, khoảng 1 500 giây mỗi epoch, ba fold xong khoảng 04:30 UTC ngày 28. Dừng bây giờ vì đã "thấy rõ" chính là dừng tuỳ ý — đúng loại lỗi mà bản ghi trước tồn tại để ngăn, và tôi đã ghi "không dừng sớm" vào chính kịch bản. Thêm nữa, một lưới ba điểm cho thấy suy giảm đơn điệu theo ngân sách khung là bằng chứng mạnh hơn nhiều so với hai điểm.

### Một khiếm khuyết trong thiết kế cổng của chính tôi, phải nói ra

Mục 4 ghi cổng G-C (mở split chính thức) đòi **cả G-A và G-B đều qua**. Viết như vậy là gộp hai thành phần độc lập vào một điều kiện, và giờ nó khoá luôn khả năng dùng một thành phần còn sống. Tôi **không** tự sửa điều kiện sau khi thấy G-A trượt — đó là dịch chuyển ngưỡng. Việc đúng: chạy G-B theo đúng ngưỡng đã ghi trước của nó, rồi trình cho tác giả quyết định có sửa điều kiện liên kết hay không, và nếu sửa thì bài phải khai rõ đây là một tu chính **làm sau khi đã thấy G-A trượt**.

---

## 9. PHÁN QUYẾT CỔNG G-B: **KHÔNG QUA**. Cả hai thành phần chính của v2 đều trượt. Không mở split chính thức lần thứ ba.

Nguồn: `results/GATE_V2B.json`, log `logs/v2b.log`. Ba luồng ngoài fold trên cùng ba fold nội bộ: pixel (đối chứng v2 T=16/s=19), hình học 1 338 chiều + GBDT khớp lại từng fold, chuỗi điểm mốc 300 khung (transformer). Bộ định tuyến là MLP 10 → 16 → 3 trên véc-tơ độ tin cậy, khớp trên dự đoán ngoài fold của hai fold còn lại.

| | fold 0 | fold 1 | fold 2 | TB macro-F1 |
|---|---|---|---|---|
| pixel một mình | 0,5939 | 0,5535 | 0,6008 | **0,5828** |
| hình học một mình | — | 0,5293 | 0,5862 | — |
| chuỗi một mình | — | 0,5461 | 0,5658 | — |
| hợp nhất đều ba luồng | — | 0,5447 | 0,6044 | — |
| hợp nhất w=0,5 pixel+hình học | — | 0,5419 | 0,6059 | — |
| **định tuyến theo clip** | 0,5852 | 0,5494 | 0,6111 | **0,5819** |
| định tuyến + tổng hợp theo người | 0,5874 | 0,5548 | 0,5987 | 0,5803 |

Định tuyến theo clip so với **luồng đơn tốt nhất: −0,0009**, cần ≥ +0,015. So với **hợp nhất trọng số cố định tốt nhất: +0,0033**, cần ≥ +0,010. Không âm ở **1/3 fold**. Biến thể có tổng hợp theo người: −0,0024. **Cả hai trượt.**

Bộ định tuyến **không** suy biến: trọng số trung bình (pixel, hình học, chuỗi) là 0,571 / 0,057 / 0,372 ở fold 0 và 0,699 / 0,071 / 0,230 ở fold 1, với độ lệch chuẩn 0,10–0,24 nên nó thật sự thay đổi theo từng clip. Nó học được điều đúng — gần như bỏ hình học, dựa vào pixel và một phần chuỗi — và điều đó vẫn không hơn pixel một mình.

### Cách đọc, theo đúng luật đã ghi ở mục 4 và mục 7

G-A trượt và G-B trượt, nên theo cổng G-C **không mở split chính thức lần thứ ba**. Kết quả v1 là kết quả cuối của EngageNet. Không có tu chính nào được đề xuất: điều khoản ở mục 4 mà tôi tự phê là "gộp hai thành phần độc lập" giờ không còn ý nghĩa thực tế, vì thành phần thứ hai cũng đã chết theo đúng ngưỡng của riêng nó.

### Điều quan trọng nhất của cả hai chiến dịch, nói thành một câu

**Mười một thành phần đã được thử và không cái nào trả được cái giá của nó**, tất cả đều có ngưỡng ghi trước trước khi chạy: placeholder khung mất mặt (+0,0035), nhãn mềm thứ bậc (−0,0016), lấy mẫu cân bằng lớp (−0,0077), hình học 96 khung kèm khối động học (−0,0004), luồng chuỗi điểm mốc trên EngageNet (+0,0136, thiếu 0,0014), hợp nhất hình học ở w = 0,5 (−0,004), tổ hợp bốn thành viên (hơn thành viên trung bình, không hơn thành viên tốt nhất), ngân sách 32 khung (−0,0437), định tuyến theo độ tin cậy theo clip (−0,0009) và có tổng hợp theo người (−0,0024), cộng hai nhánh dự phòng DAiSEE.

**Ba thứ đã hoạt động, và cả ba đều không phải là thêm dung lượng:** phạm vi cửa sổ thời gian (+0,0514 macro-F1), lịch huấn luyện 12 epoch với EMA 0,999 (+0,0208 accuracy), và LoRA hạng 16 (+0,0233 accuracy, đo ở lịch 30 epoch). Cả ba là chuyện *quan sát cái gì* và *tối ưu thế nào*, không phải *mô hình lớn thêm bao nhiêu*.

Và **ba trên ba bộ ước lượng không cần nhãn đều âm**: làm trơn theo clip liền kề, prior theo người từ phân bố dự đoán trung bình, và bộ định tuyến theo độ tin cậy. Oracle theo người cho +0,058 và oracle chọn thành viên cho +0,078, nhưng không bộ ước lượng thực nào chạm tới được. Đây là tỉ lệ nền để đánh giá mọi đề xuất tiếp theo trong không gian này.

### Hai việc không dừng

48 khung chạy tiếp cho xong (khoảng 04:30 UTC ngày 28) để đường cong ngân sách có ba điểm. Dự đoán hình học từng fold giờ đã được cache ở `preds/geom_gbdt_fold{0,1,2}.npz` nên mọi lần chạy lại của G-B là vài phút thay vì hai giờ.

---

## 10. ĐÓNG CHIẾN DỊCH v2, ngày 28/09/2026 theo quyết định của tác giả

Tác giả chọn: nhận kết quả test đã có, không chạy thêm chiến dịch nào, tập trung viết. Không mở split chính thức lần thứ ba. Kết quả chính thức của v1 (bổ sung 8 và 13 của `MODEL_PAPER_DESIGN_2026-09-21.md`) là kết quả cuối của cả hai bộ dữ liệu.

**Nhưng chiến dịch v2 không vô ích, và đây là điều phải đưa vào bài.** Nó biến tuyên bố cơ chế của bài từ một mặt thành hai mặt:

- v1 đo **phạm vi** cửa sổ ở ngân sách khung cố định: từ 20 % clip lên toàn clip cho **+0,0514 macro-F1**, khoảng tám lần đòn bẩy đơn kế tiếp.
- v2 đo **ngân sách khung** ở phạm vi cố định (span 280–286 cả ba cấu hình): 16 → 32 → 48 khung cho **suy giảm đơn điệu**, fold 0 là 0,5939 → 0,5740 → 0,5426 và fold 1 là 0,5535 → 0,4721 → 0,4450.

Hợp lại thành một nguyên lý thiết kế có điểm gãy đo được: *với một trạng thái biến thiên chậm, phủ hết episode là điều quan trọng; lấy dày hơn khoảng 1,6 Hz thì tốn dung lượng mà không thêm thông tin.* Chưa ai đo trục nào trong hai trục này. Đây là đóng góp cơ chế của bài, và nó là một nguyên lý chứ không phải một module.

v2 còn cho ba thứ nữa, cả ba đều vào bài:

1. **Định tuyến theo độ tin cậy trượt cổng** (−0,0009 so với luồng đơn tốt nhất): kết hợp có điều kiện nhúc nhích hơn trọng số hằng (+0,0033) nhưng không hơn một luồng duy nhất. Bộ định tuyến không suy biến — trọng số trung bình pixel 0,57–0,70, chuỗi 0,23–0,37, hình học 0,06–0,08, độ lệch chuẩn 0,10–0,24 — nên nó học đúng cái cần học và vẫn không đủ.
2. **Phép đo nhánh lấy mẫu, giải quyết hạn chế C8 của v1**: chênh lệch giữa nhánh clip-dài (s=19) và nhánh clip-ngắn (s=20) chỉ **+0,0026 macro-F1** trung bình ba fold. Hạn chế đó giờ có số đi kèm và số nói nó nhỏ.
3. **Kiểm tra tái lập**: đối chứng 16 khung mới đạt TB macro-F1 0,5827 so với 0,5801 của v1 trên cùng ba fold, bằng một bước khác và một đường mã khác.

Còn một ô số duy nhất chưa có: **48 khung fold 2**, đang chạy, xong khoảng 05:50 UTC ngày 28. Kịch bản cổng tự tính và ghi `GATE_V2A.json` khi nó xong. Phán quyết không thể đổi: 32 khung đã trượt −0,0437 với 0/3 fold không âm, và 48 khung còn thấp hơn ở cả hai fold đã có.

**Tổng kết hai chiến dịch, một câu.** Mười hai thành phần đã thử với ngưỡng ghi trước, không cái nào trả được cái giá của nó. Ba thứ hoạt động — phạm vi cửa sổ, lịch huấn luyện, LoRA — không cái nào là thêm dung lượng. Ba trên ba bộ ước lượng không cần nhãn đều âm, dù oracle theo người cho +0,058 và oracle chọn thành viên cho +0,078. Bài báo phải kể đủ cả ba dòng này.

---

## 11. Bổ sung ngày 29/09/2026 — nơi lưu bản ghi trước của phép đối chứng tách biến

Vòng audit dữ liệu chỉ ra một khuyết điểm về **nơi lưu**: bản thảo nói ba cách đọc kết quả của phép đối chứng được ghi trước, nhưng người đọc tìm trong hai file thiết kế thì không thấy — vì tôi ghi chúng vào `FACTS_MODEL_PAPER.md` mục 13, là một tài liệu làm việc bị sửa liên tục, chứ không vào một bản ghi trước có ngày tháng. Ghi nhận đúng như vậy, và chép lại đây để bản ghi nằm đúng chỗ người đọc sẽ tìm.

**Ghi trước ngày 28/09/2026, trước khi chạy T=16/s=9, ba cách đọc:**

| Kết cục | Đọc là | Tuyên bố của bài trở thành |
|---|---|---|
| T=16/s=9 ≈ T=32/s=9 (khoảng 0,54) | ngân sách khung không quan trọng ở bước cố định; **bước là biến quyết định** | "lấy mẫu thưa"; bước là đòn bẩy duy nhất đáng kể |
| T=16/s=9 ≈ T=16/s=19 (khoảng 0,58) | bước không quan trọng ở ngân sách cố định; **ngân sách khung là biến quyết định** | "giữ ngân sách khung nhỏ"; bước không phải đòn bẩy |
| nằm giữa hai mức | cả hai đóng góp | bài phân tách hai phần và nêu tỉ lệ |

Dự báo ghi kèm: 60 % rơi vào dòng một, 20 % dòng hai, 20 % dòng ba. **Kết cục thực tế: dòng ba** — B = 0,5517 nằm giữa, gần C hơn A. Đúng chiều, sai dòng.

Bản đầu của bảng này dán nhãn hai dòng đầu bằng "tần số" và "độ phủ" và **dòng hai bị đảo**; lỗi đã ghi ở `FACTS_MODEL_PAPER.md` mục 13.3. Không ảnh hưởng quyết định nào vì kết cục rơi vào dòng ba, và dòng ba viết đúng từ đầu.
