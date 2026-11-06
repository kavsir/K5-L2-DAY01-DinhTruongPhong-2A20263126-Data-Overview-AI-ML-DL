
# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:** 11/6/2026

**Runtime Colab:** gpu t4

**Python / PyTorch / Ultralytics:** 

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation:** `0.35` (cần xác nhận với cấu hình notebook).

**Thay đổi so với notebook nguồn:** không đổi

## 1. Phân loại ảnh

### 1. Record hạng 1 của ảnh `traffic`

- `class_id`: `468` 
- `class_name`: `cab`
- `rank`: `1`
- `score`: `0.510915`
- `taxonomy_name`: `ImageNet-1K`

### 2. Đơn vị dự đoán

Đây là dự đoán cho **toàn ảnh**, không phải từng phương tiện. Nhãn `cab` không có nghĩa mọi xe trong ảnh đều là taxi.

### 3. Nguồn danh sách lớp

Danh sách lớp thuộc taxonomy/checkpoint đã huấn luyện; mô hình chọn trong danh sách này, không tạo lớp mới.

### 4. Vì sao cần ghi đủ ba trường?

`class_id` là mã lớp, `class_name` là tên dễ đọc, `taxonomy_name` xác định bộ nhãn. Ghi đủ giúp tránh nhầm giữa các taxonomy.

### 5. Guideline cho ảnh có nhiều vật thể

Guideline cần nêu cách chọn chủ thể chính và xử lý khi có nhiều chủ thể ngang nhau. Ca không rõ cần chuyển Lab Coach.

### 6. Không dùng model score làm ground truth

Score chỉ thể hiện mức độ tin của mô hình, không chứng minh nhãn đúng. Ground truth phải do người gán và kiểm tra theo guideline; dự đoán `cab` cần được đối chiếu với quy tắc gán nhãn.

## 2. Phát hiện vật thể

### 1. Record được chọn

Chọn người bên phải ảnh `kitchen`:

- `class_name`: `person`
- `score`: `0.912625`
- `bbox_xyxy`: `385.33,69.24,498.92,348.92`
- `bbox_width`: `113.58`
- `bbox_height`: `279.68`

### 2. Giải thích bounding box

Với `bbox_xyxy = [x1, y1, x2, y2]`, `(x1, y1)` là góc trên-trái, `(x2, y2)` là góc dưới-phải. Tọa độ tính bằng pixel từ góc trên-trái ảnh; x tăng sang phải, y tăng xuống dưới. Box bao quanh người từ đầu đến chân.

### 3. So sánh hai ngưỡng

| Ngưỡng | Số prediction trên ảnh `kitchen` |
| --- | --- |
| `0.20` | Ít nhất `11`; cần ô so sánh trong notebook để biết số chính xác |
| `0.60` | `6` |

JSON ngưỡng `0.35` có `11` prediction cho `kitchen`; ở `0.60` còn `6`. Ở `0.20` có thể có thêm prediction dưới `0.35`, nên cần lấy số chính xác từ notebook Colab.

### 4. Công việc của reviewer khi hạ ngưỡng

Ngưỡng thấp giữ thêm hộp score thấp. Reviewer cần kiểm tra dự đoán nhầm, hộp trùng, sai lớp, box chưa khít và vật thể bị bỏ sót.

### 5. Quy tắc box đề xuất

Vẽ box nhỏ nhất bao quanh phần nhìn thấy; không suy đoán phần bị che và giữ tọa độ trong ảnh.

### 6. Vật thể bị cắt mép

Ở mép trái phía dưới ảnh có người bị cắt, prediction `person` có score khoảng `0.61`. Guideline nên quy định chỉ gán nhãn khi phần nhìn thấy đủ nhận diện và vẽ box quanh phần đó; trường hợp chưa rõ thì hỏi Lab Coach.

## 3. Phân đoạn theo từng đối tượng

### 1. Record được chọn

Chọn người bên phải ảnh `kitchen`:

- `instance_id`: `kitchen-001`
- `class_name`: `person`
- `score`: `0.899318`
- `polygon_point_count`: `348`
- 5 điểm đầu của `polygon_xy`: `[[446, 70], [445, 71], [444, 71], [443, 72], [442, 72]]`

### 2. Polygon bổ sung thông tin gì?

Box là hình chữ nhật bao quanh vật thể; polygon bám sát đường biên, phân biệt vật thể với nền.

### 3. Ý nghĩa của `instance_id`

`instance_id` phân biệt các đối tượng trong ảnh. Hai người cùng lớp `person` có thể có ID khác nhau. ID này không biểu thị lớp và không theo dõi đối tượng qua nhiều khung hình.

### 4. Quy tắc vẽ biên mask đề xuất

Vẽ phần nhìn thấy của vật thể, loại nền và vật thể khác; không tự vẽ phần bị che.

### 5. Chỗ khó quyết định

Các bát gần nhau trên bàn nên khó xác định ranh giới. Tách từng bát theo phần nhìn thấy; nếu biên không rõ, hỏi Lab Coach thay vì sao chép mask dự đoán.

## 4. Vòng đời và kiểm tra chất lượng

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

Lab này chỉ xem prediction; chưa tạo ground truth hay huấn luyện mô hình.

| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định thấy trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | Một nhãn theo taxonomy và guideline | Ảnh có nhiều loại xe; `cab` đứng đầu nhưng cần đối chiếu guideline | Chọn lớp theo chủ thể/toàn cảnh đã quy định | Lớp có đúng nội dung và guideline không |
| Phát hiện vật thể | Một lớp và box cho mỗi vật thể | Người bị cắt ở mép; bát/cốc gần nhau | Vẽ box phần nhìn thấy, ghi nhận ca khó | Sai lớp, bỏ sót, hộp trùng hoặc lệch |
| Instance segmentation | Một lớp và mask/polygon cho mỗi vật thể | Biên bát và mặt bàn khó phân biệt | Tách mask theo từng vật thể nhìn thấy | Biên, vùng che khuất, gộp/tách nhầm instance |

## 5. An toàn dữ liệu

1. Chỉ dùng dữ liệu trong phạm vi lab; không đưa thông tin cá nhân hoặc dữ liệu nhạy cảm vào báo cáo/output.
2. Nếu dữ liệu ngoài phạm vi, dừng xử lý và báo Lab Coach/giảng viên.

## 6. Checklist trước khi nộp link repo

Kiểm tra các mục sau trong thư mục `report/` trên GitHub:

- [ ] `REPORT.md` đã điền đủ, không còn trường `[Cần bổ sung]`
- [ ] `day1_lab_outputs/classification_predictions.json`
- [ ] `day1_lab_outputs/detection_predictions.json`
- [ ] `day1_lab_outputs/segmentation_predictions.json`
- [ ] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [ ] `day1_lab_outputs/visuals/classification_top5.png`
- [ ] `day1_lab_outputs/visuals/detection_predictions.png`
- [ ] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [ ] Ô validation cuối notebook báo `PASS`
- [ ] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [ ] Đã nộp **link repository** trên LMS