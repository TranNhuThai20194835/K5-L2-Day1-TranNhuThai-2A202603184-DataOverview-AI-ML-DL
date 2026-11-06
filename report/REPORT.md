# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy: 06/11/2026**

**Runtime Colab:T4 GPU** 

**Python 3**

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35` (chỉ đổi nếu GV cho phép; nếu đổi hãy ghi rõ)

**Thay đổi so với notebook nguồn:*Không*

> **Cách dùng mẫu này:** copy/điền trên máy, lưu thành `REPORT.md`, rồi **upload** vào panel Files
> của Colab (notebook không tự tạo file này). Ô cuối Colab sẽ đóng ZIP cùng `day1_lab_outputs/`.
> Giải nén và đưa vào `report/` trên GitHub, rồi nộp **link repo** trên LMS.
> Không ghi họ tên, MSSV, email, số điện thoại hoặc dữ liệu cá nhân khác vào báo cáo/output;
> họ tên/MSSV chỉ nằm trong tên repository.

### Từ cần nhớ (đọc nhanh trước khi viết)

| Từ | Nghĩa ngắn |
| --- | --- |
| Prediction | Kết quả model vừa đoán ra; chưa phải nhãn đúng |
| Ground truth | Nhãn do người gán theo quy tắc |
| Taxonomy | Danh sách tên lớp có sẵn kèm model (vd. ImageNet-1K, COCO-80) |
| Checkpoint | File model `.pt` đã huấn luyện sẵn |
| Guideline | Quy tắc gán nhãn do người viết |
| Annotator | Người gán nhãn |
| Reviewer / QC | Người kiểm tra nhãn |
| Escalation | Ca khó → hỏi Lab Coach, không tự bịa |
| Model score | Số “tin” của model; không phải điểm chất lượng nhãn người |

## 1. Phân loại ảnh

**Mở file:** `day1_lab_outputs/classification_predictions.json` và `visuals/classification_top5.png`.  
Chỉ dùng ảnh `sample_id = traffic`, lấy dòng `rank = 1`.
    
1. Chép record hạng 1: `class_id`=468 ; `class_name`="cab" ; `rank`=1 ; `score`=0.510915 ; `taxonomy_name`="ImageNet-1K"
2. Record này nói về **cả bức ảnh**. Vì :
- task là image_classification, nên đơn vị dự đoán là một ảnh.
- Record chỉ có image_width, image_height và coco_image_id, không có bounding box, mask hay toạ độ nào. Nếu nói về một vật thể riêng thì phải có vị trí của vật thể đó.
- Biểu đồ hiển thị cả ảnh và xếp hạng top-5 lớp cho toàn ảnh.
3. Danh sách tên lớp (class list) do đâu mà có: đi kèm taxonomy/checkpoint, model không tự nghĩ ra
4. Vì sao cần ghi cả `class_id`, `class_name` và `taxonomy_name`, không chỉ ghi mỗi tên lớp?
Vì:
- class_id là định danh máy đọc, ổn định và gọn. Nó dùng để join, tính metric, so sánh giữa các lần chạy.
- class_name giúp người đọc hiểu ngay, nhưng tên có thể đổi cách viết, có đồng nghĩa hoặc khác nhau giữa các bản dịch.
- taxonomy_name cho biết class_id thuộc bảng nào. Cùng số 468 ở một taxonomy khác (COCO, Open Images, taxonomy tự xây) sẽ là một lớp hoàn toàn khác. Thiếu thông tin này thì class_id mất nghĩa.
Ghi đủ cả ba giúp dữ liệu tự mô tả, truy vết được và tái lập được. Phối hợp với model_file, model_sha256 và ultralytics_version, người khác có thể tái tạo đúng kết quả.
5. Nếu trong ảnh có nhiều vật / nhiều chủ đề, guideline cần nói rõ điều gì để người gán nhãn chọn **một** lớp?
Guideline phải quy định tiêu chí chọn đúng một lớp. Cụ thể:
- Chọn theo vật thể/chủ đề chính hay theo bối cảnh tổng thể (ví dụ ảnh này là “đường phố giao thông” hay “xe buýt”)?
- Quy tắc ưu tiên khi nhiều thứ cùng xuất hiện: vật lớn nhất, nằm giữa khung, nổi bật nhất, hay là chủ thể mà người chụp rõ ràng hướng tới.
- Cách xử lý ảnh mơ hồ hoặc không có chủ thể nổi bật (ví dụ gán nhãn “không xác định”, loại khỏi tập, hoặc chuyển sang bài toán multi-label).
- Có cho phép nhiều nhãn hay không. Nếu bài toán là single-label thì phải nói rõ là chỉ một.
- Ví dụ minh hoạ và các trường hợp biên, để các người gán nhãn quyết định giống nhau.- 
6. Vì sao không được lấy `score` của model làm ground truth?
Vì: 
- score là độ tin cậy của chính model, phản ánh model “nghĩ” gì, không phải sự thật khách quan. Ground truth phải đến từ người gán nhãn theo guideline, hoặc từ nguồn đã được kiểm chứng.
- Model có thể sai nhưng vẫn tự tin. Trong ví dụ này, cab chỉ đạt khoảng 0.51 và bức ảnh thực tế là cảnh giao thông có nhiều xe buýt, nên top-1 chưa chắc đã đúng.
- Điểm số bị ảnh hưởng bởi taxonomy giới hạn: khi không có lớp phù hợp, model vẫn phải chia xác suất cho các lớp “gần giống”, nên điểm không nói lên độ đúng.
- Nếu lấy dự đoán của model làm nhãn rồi đem đánh giá chính model đó thì sẽ thành vòng luẩn quẩn, metric cao giả tạo, và lỗi của model bị nhân lên trong dữ liệu.

## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.  
Số lượng prediction theo ngưỡng: xem output ô so sánh `0.20` / `0.35` / `0.60` trong notebook.

1. Chép một record: `class_name`="person" ; `score`=0.912625 ; `bbox_xyxy`=[
      385.33,
      69.24,
      498.92,
      348.92
    ] ; `bbox_width`=113.58 ; `bbox_height`=279.68
2. Giải thích box bằng lời thường: góc trên-trái ở đâu, góc dưới-phải ở đâu, rộng/cao bao nhiêu pixel?
Vì bbox_format là xyxy, nên bốn số là [x_min, y_min, x_max, y_max]. Gốc toạ độ nằm ở góc trên-trái của ảnh, x tăng sang phải, y tăng xuống dưới, đơn vị là pixel (coordinate_unit: pixel).
- Góc trên-trái: (385.33, 69.24), tức cách mép trái khoảng 385 px và cách mép trên khoảng 69 px.
- Góc dưới-phải: (498.92, 348.92), tức cách mép trái khoảng 499 px và cách mép trên khoảng 349 px.
- Rộng: 498.92 − 385.33 ≈ 113.6 px.
- Cao: 348.92 − 69.24 ≈ 279.7 px.
3. Chọn hai ngưỡng (vd. 0.20 và 0.60). Mỗi ngưỡng có bao nhiêu prediction?
- Ngưỡng 0.20: giữ 17 prediction.
- Ngưỡng 0.6: giữ 6 prediction.
4. Khi hạ ngưỡng (giữ nhiều hộp hơn), người **reviewer** phải làm thêm việc gì?
Từ 0.60 xuống 0.20 là thêm 11 hộp, và những hộp thêm vào có score thấp nên khả năng sai cao hơn. Reviewer phải:
- Kiểm tra từng hộp có thật sự là vật thể không, và xoá hộp nhầm (ví dụ potted plant, bottle, spoon chập chờn).
- Kiểm tra lớp có đúng không, ví dụ cup hay bowl.
- Loại hộp trùng lặp hoặc chồng lên nhau trên cùng một vật.
- Chỉnh lại box cho khít với vật thể, vì box score thấp thường lệch.
5. Viết **một quy tắc box** để người gán nhãn làm theo (áp dụng được cho nhiều ảnh, không chỉ ảnh này): 
- Mỗi vật thể thuộc danh sách lớp có đúng một box, ôm sát toàn bộ phần nhìn thấy được của vật thể: cạnh box chạm vào mép ngoài của vật thể, không chừa nền thừa và không cắt mất phần nào của vật
6. Trên ảnh `kitchen`, chỉ ra **một** vật bị che khuất hoặc cắt mép. Bạn sẽ viết thành quy tắc trong guideline, hay đưa lên hỏi Lab Coach (escalation)? Vì sao?
Mình sẽ đưa lên hỏi Lab Coach (escalation), vì:
- Câu hỏi ở đây không phải cách vẽ box mà là định nghĩa của lớp: chỉ thấy một bàn tay thì có tính là person không? Câu trả lời thay đổi ground truth và cả metric, nên người gán nhãn không nên tự quyết.
- Quy tắc ôm sát phần nhìn thấy (câu 5) chưa trả lời được câu hỏi này, tức guideline hiện tại chưa phủ trường hợp đó.
- Nếu mỗi người tự chọn một kiểu, nhãn sẽ không nhất quán giữa các ảnh.
Sau khi Lab Coach trả lời, mình mới viết kết quả thành quy tắc trong guideline.
## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.

1. Chép một record: `instance_id`="kitchen-001" ; `class_name`="person" ; `score`=0.899318 ; số điểm polygon=`polygon_point_count`=348 ; 3–5 điểm đầu của `polygon_xy`=[
      [
        446.0,
        70.0
      ],
      [
        445.0,
        71.0
      ],
      [
        444.0,
        71.0
      ],
      [
        443.0,
        72.0
      ],
      [
        442.0,
        72.0
      ]
]
2. So với box, polygon cho thêm thông tin gì về hình dạng vật thể?
- Box chỉ cho biết vật nằm trong một hình chữ nhật (113 × 282 px). Polygon cho biết đúng pixel nào thuộc vật thể: đường viền của đầu, vai, tay chống hông, tạp dề và chân. Trên ảnh, mask xanh chỉ phủ người đàn ông, còn tường, nồi chảo và quầy phía sau box thì không nằm trong mask.
- Nhờ đó ta tính được diện tích thật, hình dạng và tư thế, tách vật khỏi nền, và xử lý vật không phải hình chữ nhật (như cái bàn chéo ở bên trái). Đổi lại, polygon tốn công gán nhãn hơn nhiều (348 điểm so với 4 số) và có nhiều chỗ để hai người vẽ khác nhau.
3. `instance_id` dùng để làm gì? Nó **không** phải class ID và **không** phải tracking ID — bạn hiểu câu này thế nào?
- `instance_id`là định danh của từng cá thể (instance) trong một ảnh, để phân biệt “người thứ nhất” với “người thứ hai” cùng lớp person. Nhờ vậy mỗi mask, polygon và nhãn gắn đúng vào một vật cụ thể, và có thể review, sửa hoặc xoá riêng từng cái.
- Câu “không phải class ID, không phải tracking ID” mình hiểu như sau:
+ Không phải class ID: class_id (0 = person) chỉ loại vật, và mọi người đều có cùng giá trị này. instance_id chỉ từng cá thể, nên khác nhau giữa hai người dù cùng lớp.
+ Không phải tracking ID: tracking ID nối cùng một vật qua nhiều khung hình video. instance_id ở đây chỉ có nghĩa trong một ảnh tĩnh, không nói gì về việc vật đó có xuất hiện ở ảnh khác hay không.
4. Viết **một quy tắc vẽ biên mask** để người gán nhãn làm theo.
- Đặt các điểm polygon dọc theo mép ngoài của vật thể, sao cho mask bao trọn phần nhìn thấy của vật và không lấn sang nền hay sang vật khác; ở những đoạn cong thì đặt điểm đủ dày để đường viền không bị gãy khúc.
5. Trên ảnh `kitchen`, chỉ ra **một** chỗ biên mờ / hai vật chạm nhau / bị che. Bạn dùng guideline hay escalation? Vì sao?
- Mình chọn khu vực bên trái: cái bàn (dining table 0.58) và các bowl chồng lên nhau. Mask bàn màu xanh nằm đè lên các bowl 0.70, 0.53, 0.74, nên ranh giới giữa bàn và bát, và giữa các bát với nhau, rất mờ. Ngoài ra bàn còn bị cắt ở mép ảnh và đồ vật trên bàn che mất một phần.
- Mình sẽ chia làm hai phần:
+ Dùng guideline cho phần đã có đáp án rõ: bát đặt trên bàn thì mask bát tách riêng, mask bàn không bao gồm vùng của bát, và mỗi bát là một instance riêng. Đây là loại tình huống lặp lại ở rất nhiều ảnh, nên ghi thành quy tắc kèm ảnh minh hoạ.
+ Escalation lên Lab Coach cho phần guideline chưa trả lời: ví dụ khi hai bát chồng lên nhau mà không nhìn ra biên, hoặc khi một vật bị che gần hết, thì nên vẽ phần nhìn thấy, vẽ cả phần đoán là bị che, hay bỏ qua? Câu trả lời ảnh hưởng tới định nghĩa nhãn và metric (IoU), nên người gán nhãn không nên tự quyết.

## 4. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

Điền bảng (viết ngắn, có dẫn chứng từ JSON/PNG khi nói về lỗi):

| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh |1 nhãn cho cả ảnh, từ taxonomy cố định (class_id, class_name, taxonomy_name) |traffic: top-1 cab 0.51 nhưng ảnh là nhiều xe buýt. kitchen: top-1 gong 0.42, sai rõ | Chọn đúng 1 lớp theo guideline; ảnh mơ hồ thì escalate |Nhãn có khớp guideline không; ưu tiên ảnh score thấp hoặc top-1/top-2 sát nhau |
| Phát hiện vật thể |1 box bbox_xyxy (pixel) cho mỗi vật |person 0.61 chỉ là bàn tay ở mép ảnh; các bowl chồng nhau; số box đổi theo ngưỡng (17/11/6) |Vẽ box ôm sát phần nhìn thấy, xoá box thừa/trùng, sửa lớp  |Thiếu/thừa box, nhầm lớp, box lệch; ca đã escalate có áp dụng nhất quán không  |
| Instance segmentation |1 polygon polygon_xy + instance_id cho mỗi cá thể |1 polygon polygon_xy + instance_id cho mỗi cá thể |Vẽ biên theo mép ngoài, tách mask khi vật chạm nhau; ca che khuất/biên mờ thì escalate | Mask có lấn nền/vật khác, có lỗ, chồng nhau không; lớp và instance_id đúng chưa |

## 5. An toàn dữ liệu

1. Quy tắc bảo vệ dữ liệu của bạn cho bài lab này là gì? (viết 1 câu)
- Chỉ dùng ảnh và dữ liệu được cấp cho bài lab này, chỉ xử lý trong môi trường lab được phép, không tải lên hay chia sẻ ra ngoài, và không thêm ảnh hoặc thông tin nhận dạng cá nhân nào ngoài phạm vi.
2. Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, bạn sẽ dừng và báo cho ai?
Mình sẽ dừng ngay việc gán nhãn hoặc xử lý ảnh đó và báo cho Lab Coach. Mình sẽ không tự quyết định giữ, sửa hay xoá.
## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

- [ ] `REPORT.md` (file này đã điền)
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
