# 03. Kiểm toán đánh giá mô hình

**Phạm vi.** Cấu hình `level=mini, mode=full, test_policy=subset`, seed được ghi là 59. Kiểm toán đọc các artifact đã lưu và tính lại metric từ dự đoán. **Verified** là đối chiếu trực tiếp trên artifact hoặc mã nguồn. **Inferred** là diễn giải có giới hạn. **Missing** là thông tin chưa có để xác minh. Không phát hiện mâu thuẫn trong các đối chiếu số liệu đánh giá thực hiện được.

## 1. Giao thức, checkpoint và preprocessing

| Phát hiện | Trạng thái, độ tin cậy | Bằng chứng |
| --- | --- | --- |
| Lượt đánh giá ghi checkpoint `/content/outputs/models/full_mini.pt`, epoch 3, mode `full`, seed 59. Summary ghi cùng checkpoint và best epoch. | Verified, cao cho metadata được lưu | `outputs/results/full_mini_test_metrics.json`, `outputs/results/summary.csv`. |
| Đường đánh giá dùng checkpoint schema 1 qua `torch.load(weights_only=True)`. Mã khôi phục tokenizer từ artifact, đối chiếu vocabulary checkpoint với vocabulary tái dựng từ train, rồi nạp model state với `strict=True`. | Verified, cao cho mã hiện tại | `src/evaluate.py:116` và `176`–`198`, `src/train.py:166`. |
| Pipeline kiểm tra hash CSV và đối chiếu manifest của ảnh, split, tokenizer và preprocessing trước khi gọi loader test. Eval transform resize bicubic kích thước 248, crop giữa 224×224 và normalize mean/std đều 0,5 mỗi kênh. | Verified, cao cho mã và transform ghi trong validation manifest | `src/evaluate.py:183`–`198`, `src/train.py:185`–`209`, `outputs/results/data_validation_mini_subset.json`, khóa `data_manifest.eval_transform`. |
| Bản CSV hiện đọc được có SHA-256 trùng JSON đánh giá và validation manifest. Vocabulary phục hồi từ thứ tự nhãn confusion có 11 mục. SHA-256 JSON của mapping và SHA-256 của bốn file xử lý dữ liệu hiện tại đều trùng manifest. | Verified, cao | `data/source_inspection/PlantVillageVQA.csv` (`a96f8b5306bac41e171ad36af7ed3b2c0d0fd4ca4d3b2eacc06617ef8fafc06d`), `data_manifest.answer_vocab_sha256`, `data_manifest.data_code_sha256`. |
| Nhãn và thành viên test nhất quán với annotation nguồn. Đối chiếu đa tập trên (image_id chuẩn hóa, question_type, chỉ số đáp án) cho toàn bộ 197 ảnh trong predictions và các hàng factoid tương ứng từ CSV nguồn cho 425 hàng ở mỗi bên, không có hàng từ split khác và 0 hàng lệch ở cả hai chiều. | Verified, cao cho tập hàng đã xuất | `data/source_inspection/PlantVillageVQA.csv`, `outputs/results/full_mini_test_predictions.csv` và thứ tự label trong JSON. |
| JSON đánh giá ghi đường dẫn và epoch, thiếu hash trọng số checkpoint hoặc manifest đầy đủ ở đầu ra đánh giá. Đối chiếu metadata và bảng kết quả chưa xác thực được mối liên hệ giữa từng dự đoán với nội dung trọng số hiện có ở file local. | Missing, giới hạn của provenance | `src/evaluate.py:199`–`206`, các khóa cấp gốc của `full_mini_test_metrics.json`. |
| Phần đánh giá này sử dụng các dự đoán đã lưu. Việc mở thư mục ảnh theo revision bằng công cụ Node.js trả về `EPERM`. | Missing ở phạm vi kiểm toán này, chưa đối chiếu lại suy luận | SHA-256 ảnh ghi trong manifest và kết quả tái chạy được kiểm tra riêng nếu có môi trường phù hợp. |

Baseline chọn đáp án phổ biến theo loại câu hỏi trên train cuối cùng, xử lý hòa bằng thứ tự từ điển. Trong dự đoán đã xuất, baseline dùng `Yes` cho Detailed Verification, `No` cho General Health Assessment và Visual Attribute Grounding, `Tomato` cho Plant Species Identification. Hai phương pháp được đánh giá trên cùng 425 dòng test. Cross-entropy tính trung bình trên nhãn có trong vocabulary. Exact match dùng toàn bộ dòng, coi mọi `true_label=-1` là sai. Nguồn: `src/evaluate.py:28`–`47`, `50`–`111` và predictions CSV.

## 2. Tái tính kết quả trên test mini

**Verified (cao).** Từ `full_mini_test_predictions.csv`, tính hit bằng `true_label >= 0 and prediction == true_label`, cho cả `predicted_label` và `baseline_label`. Exact match mô hình **354/425 = 83,29%**, baseline **250/425 = 58,82%**, chênh **24,47 điểm phần trăm** trên test mini. Có **5** nhãn ngoài vocabulary (420/425 có nhãn trong vocabulary, **98,82%**). Cả hai confusion matrix tái tính trùng nội dung JSON, và confusion matrix mô hình trùng bản CSV. Không có hàng lệch giữa `predicted_answer` và mapping nhãn, hoặc giữa cờ `correct` và tiêu chí hit. Tổng hợp theo question type trùng JSON và CSV.

**Verified (cao cho giá trị lưu, chưa tái tính).** `metrics.loss_known_answers` ở JSON là **0,3938767609330604**. Predictions CSV chỉ có nhãn phân lớp và thiếu logits hoặc probability để tái tính loss.

Macro lấy trung bình trên toàn bộ 12 nhãn phân tích (11 đáp án train và bucket unknown, gồm Blueberry không xuất hiện trên test). Weighted lấy trọng số là support test của từng nhãn, micro gộp đếm trước khi chia. Mẫu số bằng 0 cho kết quả 0. Bucket unknown có TP bằng 0. Nguồn của mọi ô là confusion matrix tái dựng từ predictions CSV.

| Phương pháp | Trung bình | Precision | Recall | F1 |
| --- | --- | ---: | ---: | ---: |
| ViT+BERT | Macro | 53,08% | 63,27% | 56,46% |
| ViT+BERT | Weighted | 83,00% | 83,29% | 82,91% |
| ViT+BERT | Micro | 83,29% | 83,29% | 83,29% |
| Baseline | Macro | 14,74% | 17,39% | 14,01% |
| Baseline | Weighted | 62,89% | 58,82% | 56,73% |
| Baseline | Micro | 58,82% | 58,82% | 58,82% |

Nếu lấy macro trên 11 đáp án train (vẫn tính Blueberry, bỏ bucket unknown khỏi phép lấy trung bình, vẫn tính nhầm từ các dòng unknown), F1 của mô hình là **61,59%**. Nếu chỉ lấy macro trên 10 nhãn train có support test dương, giá trị là **67,75%**. Mọi đối chiếu F1 cần cố định quy ước.

| Loại câu hỏi | Tổng QA | Unknown | Đúng mô hình | Đúng baseline | Chênh accuracy (điểm %) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Detailed Verification | 64 | 0 | 63 (98,44%) | 46 (71,88%) | +26,56 |
| General Health Assessment | 140 | 0 | 121 (86,43%) | 93 (66,43%) | +20,00 |
| Plant Species Identification | 146 | 5 | 116 (79,45%) | 46 (31,51%) | +47,95 |
| Visual Attribute Grounding | 75 | 0 | 54 (72,00%) | 65 (86,67%) | −14,67 |

## 3. Phân tích theo lớp đáp án

| Nhãn thật | Support | Số được dự đoán | TP | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Apple | 3 | 0 | 0 | 0,00% | 0,00% | 0,00% |
| Blueberry | 0 | 0 | 0 | 0,00% | 0,00% | 0,00% |
| Corn | 3 | 3 | 3 | 100,00% | 100,00% | 100,00% |
| Grape | 3 | 4 | 3 | 75,00% | 100,00% | 85,71% |
| No | 211 | 200 | 176 | 88,00% | 83,41% | 85,64% |
| Orange | 9 | 9 | 9 | 100,00% | 100,00% | 100,00% |
| Pepper bell | 1 | 2 | 0 | 0,00% | 0,00% | 0,00% |
| Potato | 5 | 8 | 5 | 62,50% | 100,00% | 76,92% |
| Soybean | 3 | 8 | 3 | 37,50% | 100,00% | 54,55% |
| Tomato | 46 | 44 | 43 | 97,73% | 93,48% | 95,56% |
| Yes | 136 | 147 | 112 | 76,19% | 82,35% | 79,15% |
| &lt;UNKNOWN&gt; | 5 | 0 | 0 | 0,00% | 0,00% | 0,00% |

- **Verified, cao.** Sai lệch thường gặp nhất là `No→Yes` (35) và `Yes→No` (24), tổng cộng 59/71 lỗi. `Apple` không có TP trên ba mẫu thật (dự đoán Soybean hai lần và Potato một lần). Mẫu thật `Pepper bell` duy nhất bị dự đoán là Soybean. Blueberry thiếu support. Với ít mẫu, một lỗi có thể làm thay đổi mạnh tỷ lệ của lớp. Nguồn: predictions CSV và confusion matrix tái tính.
- **Verified, cao.** Visual Attribute Grounding có 54/75 đúng so với 65/75 ở baseline. Đối chiếu từng hàng thấy 53 câu cả hai đúng, 9 cả hai sai, 1 chỉ mô hình đúng, 12 chỉ baseline đúng. Nguồn: predictions CSV.
- **Verified, cao.** Năm nhãn unknown đều ở Plant Species Identification. Một số câu trong nhóm này có đáp án Yes/No, nên accuracy của nhóm bao gồm cả các câu xác nhận loài. Nguồn: predictions CSV và annotation nguồn.
- **Inferred, trung bình.** Kết quả mô tả tập câu hỏi factoid con lấy từ test gốc (`test_policy=subset`). Khả năng khái quát sang toàn bộ test gốc hoặc ảnh thực địa chưa được xác lập. Các câu hỏi cùng ảnh có thể tương quan, nên đơn vị lấy mẫu phù hợp cho khoảng bất định là ảnh.

## 4. Hướng xác minh tiếp theo

- Lưu SHA-256 của best checkpoint, manifest đầy đủ, lệnh đánh giá, phiên bản gói và thiết bị cùng artifact đánh giá. Nếu cần tái tính loss, lưu logits trên vocabulary đã chốt và định danh câu hỏi để ghép chính xác.
- Chỉ chạy đánh giá lại khi ảnh và CSV theo đúng revision, môi trường dependency và best checkpoint hợp lệ đều sẵn sàng. Dùng thư mục mới, ví dụ `python -m src.evaluate --ckpt outputs/models/full_mini.pt --data_dir <dataset_dir> --csv <source_csv> --output_dir <new_audit_dir>`.
- Nếu cần khoảng bất định, đặt trước giao thức lấy mẫu lại theo cụm `image_id` và giữ ghép cặp với baseline. Chỉ chọn checkpoint hoặc thay đổi siêu tham số dựa trên validation đã quy định.
