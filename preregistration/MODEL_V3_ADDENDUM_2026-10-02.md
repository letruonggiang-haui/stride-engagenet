# Bổ sung cho bản ghi trước v3, viết ngày 02/10/2026 trước lần chạy đầu tiên

Bản ghi trước được duyệt: `MODEL_V3_DESIGN_2026-10-02.md`, sha256 6843bb9613b99ebb7310657e95cd46faf290c6ce69939241651aa12e2705ec70. File đó không sửa. Bốn điểm dưới đây là thay đổi về vận hành, khai trước khi có bất kỳ số nào. Không điểm nào đụng tới nhánh, cổng, ngưỡng hay cách đọc.

1. Trình thông dịch. Mục 2 của bản ghi trước ghi `venv/bin/python`. Ghi như vậy là sai sự thật: kiểm trên máy chủ, các chiến dịch v1 và v2 chạy bằng `~/venvs/engagenet/bin/python` (Python 3.13.13, torch 2.6.0+cu124, timm 1.0.27; `state/v2a_grid.sh`, `state/g7.sh`, `state/g8.sh`). Mục đích của câu trong bản ghi trước là mọi nhánh chạy cùng một môi trường và nhánh C so được với số v1. Vì vậy v3 dùng `~/venvs/engagenet/bin/python` cho cả ba nhánh. Phép kiểm môi trường ở dòng ba mục 5 vẫn giữ nguyên.

2. Lưu trạng thái để chịu được khởi động lại. Tác giả yêu cầu lưu trạng thái trong lúc chạy. `ft_video.py` có thêm cờ `--resume`: sau mỗi epoch lưu điểm kiểm tra (trọng số học được, EMA, bộ tối ưu, bộ co giãn, bước, nhật ký, trạng thái bộ sinh số ngẫu nhiên) vào `ckpt/`, ghi nguyên tử, và chạy tiếp từ đó nếu tiến trình bị ngắt. Mọi lần chạy v3 dùng cờ này. Một lần chạy đã chạy tiếp sau khi bị ngắt không trùng từng bit với một lần chạy liền mạch, vì trạng thái ngẫu nhiên của các tiến trình nạp dữ liệu không khôi phục được. File kết quả ghi rõ các epoch đã chạy tiếp ở trường `resumed_at_epochs`, và bài phải báo các lần chạy đó.

3. Chờ bộ nhớ GPU. Máy chủ dùng chung. Hàng đợi chờ đủ bộ nhớ trống trước mỗi lần chạy để không làm hỏng việc khác đang chạy trên cùng card. Việc này chỉ ảnh hưởng thời gian.

4. Kiểm thử khói. Chạy nhánh S trên 64 clip của fold 0 trong hai epoch, ngắt sau điểm kiểm tra đầu tiên rồi chạy tiếp, để kiểm bộ nhớ, đường mã đoạn và đường chạy tiếp cùng một lúc. Kích thước batch của nhánh S chọn ở đây (4 × 8, lùi về 2 × 16 nếu thiếu bộ nhớ), giữ batch hiệu dụng 32 clip. Không đọc điểm của lần kiểm thử khói.
