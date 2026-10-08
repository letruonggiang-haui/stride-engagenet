# Bổ sung cho bản ghi trước v5, ngày 05/10/2026, trước lần chạy thật đầu tiên

Bản ghi trước: `MODEL_V5_DESIGN_2026-10-05.md`, sha256 c13e32416f4df296273040b4692d73d1f91695761bf3c942a2a804de90103994, không sửa.

Kiểm thử khói lần đầu (06:29 UTC) dừng hàng đợi vì một lỗi của chính kiểm thử: với 64 clip, hai epoch tiền huấn luyện xong trong 5 giây, trước khi bước "ngắt sau điểm kiểm tra đầu" kịp chen vào, nên không có lần chạy tiếp nào để kiểm. Tiền huấn luyện tự nó chạy đúng: nạp checkpoint khớp hoàn toàn, hai epoch, lưu điểm kiểm tra, ghi trọng số, bộ nhớ đỉnh 7,14 GB ở batch 32.

Thay đổi duy nhất, thuần vận hành: kiểm thử khói tiền huấn luyện dùng 640 clip thay cho 64 và kiểm điểm kiểm tra mỗi 2 giây thay cho 15 giây; nhãn kiểm thử đổi thành `v5smoke2_*` để không lẫn với lần đầu. Không đổi gì ở mô hình, dữ liệu, cổng hay cách đọc. Không có số kết quả nào trước dòng này.
