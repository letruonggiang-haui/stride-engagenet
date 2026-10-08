# Thiết kế v6, vòng xác nhận của v5, bản ghi trước, viết ngày 06/10/2026 trước khi chạy bất cứ gì

Tác giả yêu cầu ngày 06/10/2026: "Chạy một vòng xác nhận v6 vì tôi muốn có bài mô hình". File này cố định mọi lựa chọn trước lần chạy đầu tiên; sha256 của nó được ghi vào FACTS trước khi chạy. Mọi thay đổi sau đó là một file bổ sung có ngày.

## 0. Vì sao có v6, và điều phải khai

v5 (`MODEL_V5_DESIGN_2026-10-05.md`, FACTS mục 45) trượt cổng G-D sát ngưỡng: P − C = +0,0164 macro-F1 trung bình ba fold (ngưỡng +0,015), dương ở cả ba fold, UAR +0,0190, nhưng chỉ 1/3 fold có hiệu vượt khoảng seed (cần 2). v6 được làm vì v5 sát ngưỡng, và bài phải nói đúng như vậy. Đây là chiến dịch thứ sáu; năm chiến dịch trước có 15 thành phần ghi trước, không thành phần nào qua cổng. Bài viết từ v6 phải liệt kê đủ cả sáu vòng.

Ba ràng buộc để v6 không thành "thử tới khi qua":
1. Quyết định chỉ dựa trên các lần chạy mới của v6. Không gộp số v5 vào cổng, không chọn giữa v5 và v6.
2. Cổng giữ nguyên luật G-D của v5, không nới, không đổi thước đo.
3. Nếu v6 trượt thì dừng hẳn hướng này trên EngageNet. Không có v7 cho cùng mô hình.

## 1. Mô hình, giữ nguyên v5

Mô hình đề xuất (nhánh P) đúng như mục 2 của bản ghi trước v5: tiền huấn luyện tiếp SVFAP-B kiểu MAE ở cửa sổ CMS (16 khung, bước 20, mặt nạ ống 0,9, 30 epoch, lr 1e-4, batch 32) trên các clip huấn luyện của fold, không dùng nhãn, rồi tinh chỉnh bằng công thức CMS (LoRA 16, 12 epoch, EMA 0,999). Mã không đổi so với v5: `pretrain_v5.py` sha256 cef6ff9b…, `ft_video.py` sha256 65da774c….

## 2. Lần chạy mới

- Tiền huấn luyện: một lần mới cho mỗi fold với seed 1 (v5 dùng seed 0), nhãn `v6_pre_f<k>`.
- Tinh chỉnh: seed 3, 4, 5 cho cả hai nhánh ở mỗi fold (v1, v3, v5 dùng 0, 1, 2).
  - P: `v6_P_f<k>_s<s>`, khởi tạo từ `weights/svfap_v5/v6_pre_f<k>.pth`.
  - C (đối chứng CMS, chạy mới): `v6_C_f<k>_s<s>`, khởi tạo từ checkpoint SVFAP phát hành.
  - P và C cùng seed tinh chỉnh nên dùng chung thứ tự dữ liệu, tăng cường và khởi tạo LoRA.
- Tổng: 3 lần tiền huấn luyện, 18 lần tinh chỉnh. Không lần nào mở split chính thức.

## 3. Cổng G-D6

Đúng luật G-D của v5, áp trên 9 lần chạy P và 9 lần chạy C mới:
1. Trung bình ba fold của hiệu (trung bình seed của P trừ trung bình seed của C) ≥ +0,015 macro-F1.
2. Ở ít nhất hai fold, hiệu đó dương và lớn hơn khoảng seed lớn hơn của hai nhánh ở fold đó.
3. Hiệu UAR trung bình ba fold không âm.

## 4. Các cách đọc, ghi trước

| Kết cục | Đọc là | Hệ quả |
|---|---|---|
| G-D6 qua | hiệu ứng của v5 lặp lại trên seed tiền huấn luyện và seed tinh chỉnh mới, ở mức vượt nhiễu seed | đánh giá chính thức theo mục 5 của bản ghi trước v5, không đổi |
| G-D6 trượt, hiệu trung bình dương | tiền huấn luyện tiếp có thể giúp một chút nhưng thiết kế này không tách được khỏi nhiễu seed | dừng; báo v5 và v6 như một kết quả dương chưa xác nhận; không có bài mô hình |
| G-D6 trượt, hiệu trung bình ≤ 0 | +0,0164 của v5 là nhiễu | dừng; không có bài mô hình |

Báo cáo phụ, không thuộc cổng: hiệu P − C gộp v5 và v6 (6 seed mỗi nhánh mỗi fold), kèm câu rằng con số gộp không dùng để quyết định.

## 5. Dự báo, ghi trước

| Đại lượng | Dự báo | Xác suất |
|---|---|---|
| P − C, macro-F1 trung bình ba fold | 0,000 đến +0,025 | qua G-D6: 30 % |
| hiệu trung bình dương | | 75 % |

## 6. Vận hành

Như v5: điểm kiểm tra mỗi epoch, chạy tiếp sau khởi động lại, khoá, chờ bộ nhớ GPU, mục `@reboot` tự gỡ khi xong. Không cần kiểm thử khói vì mã không đổi so với v5 và v5 đã qua kiểm thử khói. Thứ tự: mỗi fold, tiền huấn luyện trước, rồi lần lượt C và P cho từng seed. Ước tính khi GPU không bị chia: khoảng 18 giờ.
