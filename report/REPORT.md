# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:** 06/11/2026

**Runtime Colab:** GPU (`Device: cuda`)

**Python / PyTorch / Ultralytics:** `3.13.16 / 2.11.0+cu130 / 8.4.145`

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Kiểm tra checkpoint:** Kết quả setup đã báo `PASS` cho mã SHA-256 của cả ba checkpoint.

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35`, được xác nhận từ kết quả detection và segmentation đã chạy.

**Thay đổi so với notebook nguồn:** Không. Giữ nguyên mã nguồn, checkpoint và các ngưỡng theo notebook được cung cấp.

> Các quy tắc gán nhãn trong báo cáo là đề xuất cho bài tập; trường hợp chưa có quy định cần được Lab Coach hoặc giảng viên thống nhất trước khi áp dụng.

## 1. Phân loại ảnh

**Nguồn đối chiếu:** `day1_lab_outputs/classification_predictions.json` và `day1_lab_outputs/visuals/classification_top5.png`.

**Ảnh sử dụng:** `sample_id = traffic`; chọn record có `rank = 1`.

### 1.1. Record hạng 1

| Trường | Giá trị |
| --- | --- |
| `sample_id` | `traffic` |
| `class_id` | `468` |
| `class_name` | `cab` |
| `rank` | `1` |
| `score` | `0.510915` |
| `taxonomy_name` | `ImageNet-1K` |

Record thuộc ảnh COCO `210273`, kích thước `640 × 428` pixel, được dự đoán bằng `yolo11n-cls.pt` với Ultralytics `8.4.145`.

**Các lớp được xếp hạng cho cùng ảnh:**

| Hạng | Mã lớp | Tên lớp | Model score |
| --- | --- | --- | --- |
| 1 | 468 | `cab` | 0.510915 |
| 2 | 654 | `minibus` | 0.164284 |
| 3 | 734 | `police_van` | 0.085848 |
| 4 | 757 | `recreational_vehicle` | 0.05411 |
| 5 | 829 | `streetcar` | 0.048194 |

Mô hình xếp `cab` (taxi) ở hạng cao nhất. Năm record là năm lớp được xếp hạng cho cùng một ảnh, không phải năm vật thể đã được định vị. Score `0.510915` không có nghĩa là chất lượng nhãn đạt 51.0915% hoặc chứng minh nhãn `cab` đúng; cần đối chiếu hình và guideline.

**Đối chiếu với hình:** Ảnh `traffic` là cảnh đường phố có nhiều xe buýt và ô tô; các xe buýt màu xanh ở phía trái và giữa ảnh khá nổi bật. Biểu đồ xếp `cab` ở hạng 1, khớp score `0.510915` trong JSON, nhưng ảnh không chỉ có một loại phương tiện. Vì vậy, chọn một nhãn cho toàn ảnh là trường hợp cần guideline xác định tiêu chí chủ thể chính. Không thể chỉ dựa vào biểu đồ để kết luận các ô tô trong ảnh là taxi hoặc nhãn `cab` là ground truth; cần đối chiếu đặc điểm nhìn thấy và taxonomy.

### 1.2. Phạm vi của record

Record này mô tả một lớp dự đoán cho toàn bộ bức ảnh, không định vị một vật thể riêng. Tác vụ phân loại xếp hạng các lớp cho ảnh đầu vào; record không có hộp tọa độ hay đa giác để xác định vị trí vật thể. `rank = 1` chỉ cho biết lớp có model score cao nhất trong kết quả của ảnh đó.

### 1.3. Nguồn của danh sách lớp

Danh sách lớp đi kèm taxonomy và checkpoint đã được huấn luyện, không do mô hình tự tạo tên mới khi dự đoán. Checkpoint `yolo11n-cls.pt` trong bài dùng ImageNet-1K. Vì ảnh đầu vào là cảnh COCO còn bộ lớp phân loại là ImageNet-1K, kết quả hạng 1 có thể không trùng với cách người xem mô tả cảnh.

### 1.4. Vì sao cần ghi đủ mã lớp, tên lớp và taxonomy?

`class_id` là mã dùng trong xử lý dữ liệu; `class_name` giúp người đọc hiểu ý nghĩa của lớp; `taxonomy_name` cho biết bộ lớp mà mã và tên đó thuộc về. Một mã số không có ý nghĩa thống nhất giữa mọi bộ dữ liệu. Ghi đủ ba trường giúp tra cứu đúng lớp và tránh nhầm khi so sánh các mô hình hoặc bộ dữ liệu khác nhau.

### 1.5. Quy tắc khi ảnh có nhiều vật thể hoặc chủ đề

Guideline cần xác định rõ tiêu chí chọn một lớp cho cả ảnh: dựa trên chủ thể chính hay loại cảnh, thứ tự ưu tiên nếu có nhiều lớp phù hợp và cách xử lý khi không đủ thông tin. Các tiêu chí phải được áp dụng nhất quán giữa người gán nhãn. Nếu trường hợp chưa được quy định, cần hỏi Lab Coach thay vì tự chọn lớp theo cảm tính hoặc sao chép top-1 của mô hình.

### 1.6. Vì sao model score không phải ground truth?

Model score là giá trị mô hình dùng để xếp hạng dự đoán; nó không tự chứng minh dự đoán đúng và không phải điểm chất lượng nhãn do con người tạo. Mô hình vẫn có thể cho score cao với lớp sai. Ground truth cần được gán theo guideline và kiểm tra chất lượng, không lấy trực tiếp từ score hay prediction chưa được xác minh.

## 2. Phát hiện vật thể

**Nguồn đối chiếu:** `day1_lab_outputs/detection_predictions.json` và `day1_lab_outputs/visuals/detection_predictions.png`.

**Kết quả chạy tổng hợp:** `yolo11n.pt` tạo 53 prediction ở ngưỡng `0.35` trên cả ba ảnh mẫu. Đây là tổng số dự đoán của mô hình, không phải số vật thể thực tế đã được xác minh hoặc số prediction riêng của ảnh `kitchen`.

**Ảnh sử dụng:** `sample_id = kitchen`; chọn một record.

### 2.1. Record được chọn

| Trường | Giá trị |
| --- | --- |
| `sample_id` | `kitchen` |
| `class_name` | `person` |
| `score` | `0.912625` |
| `bbox_xyxy` | `[385.33, 69.24, 498.92, 348.92]` |
| `bbox_width` | `113.58` pixel |
| `bbox_height` | `279.68` pixel |

Record này tương ứng với người đứng ở phía phải ảnh bếp, có score hiển thị làm tròn là `0.91` trên PNG. Các giá trị trong bảng được chép từ JSON, không ước lượng từ ảnh chụp màn hình.

### 2.2. Giải thích hộp tọa độ

`bbox_xyxy = [x_min, y_min, x_max, y_max]` mô tả hình chữ nhật bao quanh vật thể. Gốc tọa độ `(0, 0)` ở góc trên bên trái ảnh; trục x tăng sang phải, trục y tăng xuống dưới. Đơn vị là pixel.

Đối với record đã chọn:

- Góc trên bên trái: `(385.33, 69.24)`, tức cách mép trái ảnh 385.33 pixel và mép trên 69.24 pixel.
- Góc dưới bên phải: `(498.92, 348.92)`.
- Chiều rộng theo JSON: `113.58` pixel.
- Chiều cao theo JSON: `279.68` pixel, bằng `348.92 - 69.24`.

Chiều rộng được tính từ tọa độ gốc trước khi làm tròn. Phép trừ hai tọa độ đã xuất là `498.92 - 385.33 = 113.59`, lệch `0.01` pixel so với `bbox_width = 113.58` do làm tròn riêng từng trường; báo cáo giữ nguyên giá trị JSON.

### 2.3. So sánh ngưỡng

Lấy số lượng trực tiếp từ output ô so sánh threshold của ảnh `kitchen`, không suy ra từ tổng record của cả ba ảnh.

| Ngưỡng | Số prediction |
| --- | --- |
| `0.20` | 17 |
| `0.35` | 11 |
| `0.60` | 6 |

**Nhận xét thực tế:** Khi tăng ngưỡng từ `0.20` lên `0.60`, số prediction giảm từ 17 xuống 6, tức giảm 11 dự đoán. Ngưỡng `0.35` dùng để tạo JSON giữ 11 dự đoán riêng cho ảnh `kitchen`. Ở ngưỡng `0.60`, danh sách còn các lớp `person`, `bowl`, `oven`; ở `0.20` có thêm các lớp như `cup`, `spoon`, `potted plant`, `dining table` và `bottle`. Ngưỡng thấp giữ nhiều ứng viên hơn để kiểm tra, nhưng không có nghĩa tất cả đều đúng. Các con số này không phải số vật thể thực tế đã được xác minh.

### 2.4. Công việc của reviewer khi hạ ngưỡng

Hạ ngưỡng có thể giữ lại thêm các dự đoán có score thấp. Reviewer phải kiểm tra thêm các hộp để xác định vật thể có thực sự tồn tại, lớp có đúng, hộp có bám sát vật thể hay bị trùng/lệch. Đồng thời cần nhìn toàn ảnh để phát hiện vật thể bị bỏ sót. Không được coi mọi hộp giữ lại là nhãn chuẩn hoặc dùng threshold làm quy tắc bỏ qua vật thể khi tạo ground truth.

### 2.5. Đề xuất một quy tắc box

Với quy ước gán nhãn phần nhìn thấy: mỗi vật thể thuộc taxonomy được gán một hộp chữ nhật nhỏ nhất bao kín phần nhìn thấy của chính vật thể đó, không gộp hai vật thể vào một hộp và không suy đoán phần bị che. Nếu guideline của dự án yêu cầu hộp bao toàn bộ vật thể kể cả phần bị che, phải tuân theo quy định đó; trường hợp chưa rõ cần hỏi Lab Coach.

### 2.6. Trường hợp che khuất hoặc cắt mép trong ảnh kitchen

- Vật thể và vị trí trên ảnh: phần người/tay ở mép trái ảnh, ngay phía trên bàn, trong vùng mô hình gán `person` với score hiển thị khoảng `0.61`.
- Dấu hiệu bị che khuất/cắt mép: chỉ thấy một phần nhỏ cơ thể; phần còn lại nằm ngoài khung hình. Hộp dự đoán chạm mép trái ảnh trong `detection_predictions.png`.
- Đối chiếu JSON: record `person` này có `score = 0.610931` và `bbox_xyxy = [0.12, 263.18, 61.55, 310.78]`; đây là record khác với người đứng ở phía phải được chọn tại mục 2.1.
- Cách xử lý đã chọn: đưa lên Lab Coach (escalation) để xác nhận trường hợp chỉ còn một phần cơ thể có thuộc phạm vi gán nhãn `person` hay không và quy ước box cần áp dụng.
- Lý do: score của mô hình không thay thế quy tắc gán nhãn. Khi chưa có quy định cụ thể về mức độ nhìn thấy tối thiểu, không tự suy đoán toàn bộ người hoặc kéo hộp ra ngoài ảnh. Có thể đề xuất quy tắc box cho phần nhìn thấy ở mục 2.5 để GV xem xét thống nhất.

Nguyên tắc lựa chọn: áp dụng guideline khi quy định đã rõ; nếu chưa có quy định hoặc chưa đủ căn cứ để xác định lớp/phạm vi hộp, đưa trường hợp lên Lab Coach. Đề xuất quy tắc mới cần được thống nhất trước khi áp dụng rộng rãi.

## 3. Phân đoạn theo từng đối tượng

**Nguồn đối chiếu:** `day1_lab_outputs/segmentation_predictions.json` và `day1_lab_outputs/visuals/segmentation_prediction.png`.

**Kết quả chạy tổng hợp:** `yolo11n-seg.pt` tạo 49 instance dự đoán ở ngưỡng `0.35` trên cả ba ảnh mẫu; JSON có 11 instance riêng cho ảnh `kitchen`. Đây là số dự đoán, không phải số vật thể thực tế đã được xác minh.

**Ảnh sử dụng:** `sample_id = kitchen`; chọn một record.

### 3.1. Record được chọn

| Trường | Giá trị |
| --- | --- |
| `sample_id` | `kitchen` |
| `instance_id` | `kitchen-005` |
| `class_name` | `dining table` |
| `score` | `0.578416` |
| `polygon_point_count` | `598` |
| 3–5 điểm đầu của `polygon_xy` | `[[251.0, 399.0], [252.0, 400.0], [252.0, 401.0], [253.0, 402.0], [254.0, 402.0]]` |

Record được chọn là chiếc bàn ở phía dưới bên trái ảnh `kitchen` (ảnh có kích thước `640 × 427` pixel). Score hiển thị trên PNG được làm tròn thành `0.58`. Số `598` là số điểm của polygon, không phải số vật thể hoặc diện tích mask.

### 3.2. Thông tin polygon bổ sung so với box

Box chỉ biểu diễn phạm vi bằng một hình chữ nhật và có thể chứa nhiều vùng nền xung quanh vật thể. Polygon là chuỗi điểm mô tả đường biên, giúp thể hiện hình dạng và vùng mà vật thể chiếm trong ảnh chi tiết hơn. Polygon dự đoán vẫn có thể lệch, thiếu hoặc lấn sang vật thể khác nên cần đối chiếu bằng mắt.

### 3.3. Ý nghĩa của instance_id

`instance_id` phân biệt từng đối tượng trong output của bài lab. Hai vật thể cùng lớp vẫn cần có ID và polygon riêng. `class_id` cho biết loại vật thể, còn `instance_id` cho biết một đối tượng cụ thể trong output. ID này không phải tracking ID: nó không bảo đảm nhận diện cùng một vật thể qua các khung hình video hoặc các lần chạy khác nhau.

### 3.4. Đề xuất một quy tắc vẽ biên mask

Với quy ước chỉ gán nhãn phần nhìn thấy: vẽ biên mask sát ranh giới nhìn thấy của từng vật thể, loại phần nền và phần thuộc vật thể khác, đồng thời giữ các instance tách biệt dù cùng lớp hoặc chạm nhau. Không tự nối vùng bị che bằng đường biên tưởng tượng; nếu không xác định được ranh giới theo guideline, đánh dấu để hỏi Lab Coach.

### 3.5. Trường hợp biên mờ, tiếp xúc hoặc bị che trong ảnh kitchen

- Vật thể và vị trí: chiếc bàn ở nửa dưới bên trái ảnh `kitchen`, lớp `dining table`, `instance_id = kitchen-005`, score trong JSON là `0.578416` (hiển thị khoảng `0.58` trên PNG).
- Vấn đề quan sát được: mặt bàn bị các bát/khuôn và chai đặt phía trên che khuất một phần; vùng quanh đáy và mép các vật này cần được phân biệt với phần bàn nhìn thấy. Đường bao hình chữ nhật không thể biểu diễn hết các ranh giới đó.
- Bằng chứng: trong `segmentation_prediction.png`, vùng bàn được tô xanh, các bát/khuôn phía trái có vùng tô hồng và chai đứng gần giữa bàn. Đối chiếu record `kitchen-005` trong JSON: polygon có 598 điểm, hộp bao là `[0.97, 234.74, 345.86, 423.87]`, phù hợp vùng bàn phía dưới bên trái. Đây là vùng cần kiểm tra ranh giới giữa các đối tượng; màu phủ trên ảnh chưa đủ để kết luận mọi điểm biên đều chính xác.
- Cách xử lý: đề xuất áp dụng quy tắc mask cho phần nhìn thấy ở mục 3.4, tách vùng bàn khỏi các vật đặt trên bàn và không tự lấp phần bị che. Cần hỏi Lab Coach nếu guideline chưa quy định cách xử lý vùng bị che hoặc định dạng polygon không thể biểu diễn vùng loại trừ cần thiết. Reviewer đối chiếu ảnh gốc và mask trước khi chấp nhận đường biên.

## 4. Vòng đời và kiểm tra chất lượng

Vòng đời production được sử dụng để liên hệ các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

Guideline thống nhất cách gán nhãn; annotator tạo ground truth; dữ liệu được kiểm tra và sử dụng trong quá trình xây dựng mô hình. Prediction là đầu ra của mô hình, cần được đối chiếu và kiểm tra khi sử dụng. Các lỗi phát hiện được là căn cứ cho bước chỉnh sửa dữ liệu hoặc quy trình phù hợp. Trong bài lab này, hoạt động thực hiện là quan sát prediction của mô hình có sẵn, không huấn luyện lại mô hình.

| Tác vụ | Ground truth trông như thế nào? | Lỗi hoặc chỗ khó quyết định thực tế | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | Một lớp cho toàn bộ ảnh theo taxonomy và quy tắc single-label của bài. | Ảnh `traffic` có nhiều xe buýt và ô tô, nhưng top-1 là `cab` (`0.510915`); cần quy tắc chọn chủ thể chính, không mặc định top-1 là nhãn chuẩn. Đối chiếu `classification_top5.png` và record hạng 1. | Chọn lớp theo tiêu chí chủ thể/cảnh đã thống nhất; báo ca nhiều lớp khó phân định. | Kiểm tra nhãn có phù hợp nội dung ảnh, đúng taxonomy và nhất quán với guideline; không duyệt chỉ theo score. |
| Phát hiện vật thể | Một lớp và một hộp cho mỗi vật thể thuộc phạm vi cần gán nhãn. | Trong `detection_predictions.png` của `kitchen`, vùng `person` ở mép trái chỉ có một phần cơ thể, score hiển thị khoảng `0.61`; chưa rõ tiêu chí chấp nhận người bị cắt mép. | Xác định đủ vật thể và vẽ hộp theo quy ước; báo trường hợp che khuất/cắt mép chưa có quy định. | Kiểm tra lớp, hộp lệch/trùng, vật thể bị sót và cách xử lý che khuất; yêu cầu sửa khi không đạt guideline. |
| Instance segmentation | Một lớp và mask/polygon riêng cho từng instance. | Trong `segmentation_prediction.png` của `kitchen`, bàn (`dining table`, score hiển thị khoảng `0.58`) bị các bát/khuôn và chai che một phần; cần kiểm tra mask tại các ranh giới bàn–vật thể. | Vẽ biên từng instance theo guideline, tách các vật thể và báo vùng không rõ. | Kiểm tra độ bám biên, phần thiếu/thừa, vùng lấn sang vật khác và việc tách instance. |

## 5. An toàn dữ liệu

### 5.1. Quy tắc bảo vệ dữ liệu

Chỉ sử dụng ba ảnh COCO công khai có sẵn trong notebook, giữ thông tin nguồn ảnh và không đưa ảnh cá nhân, thông tin định danh không cần thiết, dữ liệu khách hàng hoặc dữ liệu nội bộ lên Colab/GitHub.

### 5.2. Xử lý dữ liệu ngoài phạm vi

Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, tôi sẽ dừng sử dụng/upload dữ liệu đó và báo Lab Coach hoặc giảng viên để được hướng dẫn, không tự thay ảnh mẫu hoặc tiếp tục chia sẻ dữ liệu.

## 6. Checklist trước khi nộp link repo

Chỉ đánh dấu sau khi đã kiểm tra thực tế. Các file sau phải nằm trong thư mục `report/` của repository bài làm:

- [x] Đã điền đầy đủ nội dung, đối chiếu kết quả và lưu thành `REPORT.md`.
- [x] `REPORT.md` có số liệu từ cả ba JSON và nhận xét khớp đúng ảnh.
- [ ] `day1_lab_outputs/classification_predictions.json`
- [ ] `day1_lab_outputs/detection_predictions.json`
- [ ] `day1_lab_outputs/segmentation_predictions.json`
- [ ] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [ ] `day1_lab_outputs/visuals/classification_top5.png`
- [ ] `day1_lab_outputs/visuals/detection_predictions.png`
- [ ] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [x] Ô validation cuối notebook báo `PASS`: đủ 3 JSON, 3 PNG, attribution và `instance_id` duy nhất.
- [x] Có ít nhất một lỗi hoặc điểm mơ hồ được đối chiếu thực tế trên ảnh/JSON.
- [ ] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output.
- [ ] `REPORT.md` và `day1_lab_outputs/` nằm trực tiếp dưới `report/`, không nằm trong thư mục bọc thêm.
- [ ] Đã commit các file lên GitHub và kiểm tra lại nội dung.
- [ ] Đã nộp link repository trên LMS; không nộp file ZIP.

**Tài liệu theo bài:** [Notebook và repository nguồn](https://github.com/VinUni-AI20k/K5-L2-Day1-Data-Overview-AI-ML-DL-Student/tree/v1.0.3).
