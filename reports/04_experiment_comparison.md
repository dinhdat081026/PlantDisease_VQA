# 04 — So sánh thực nghiệm

## 4.1 Phạm vi và tiêu chí so sánh

**Verified · độ tin cậy cao trong workspace.** Có một cấu hình học sâu có đủ bằng chứng hoàn thành, `level=mini`, `mode=full`, seed 59. Ma trận cấu hình từ CLI là ba mức dữ liệu × bốn mode, tức 12 tổ hợp. Các mode `frozen`, `question_only`, `image_only` ở mức mini và cả bốn mode ở mỗi mức class/full chưa có artifact kết quả trong phạm vi đã kiểm tra. Tình trạng chạy bên ngoài workspace là **Missing**. Baseline tần suất huấn luyện theo loại câu hỏi dùng để đối chiếu trên cùng test và không phải một lượt huấn luyện backbone. [Cấu hình](../src/config.py), [run marker](../outputs/results/full_mini_run.json), [lịch sử](../outputs/results/full_mini_history.csv), [summary gốc](../outputs/results/summary.csv), [bảng tổng hợp](../outputs/results/audit_experiment_summary.csv).

Tiêu chí lựa chọn đã định trong code là exact-match accuracy trên validation, giữ epoch sớm nhất khi hòa. Với `mini/full`, epoch 3 đạt cao nhất, **86,50%** (282/326) và nằm trong best checkpoint `full_mini.pt`. File `full_mini_last.pt` lưu trạng thái resume sau epoch 4. Không có đủ run để xếp hạng giữa các chiến lược. [Mã chọn checkpoint](../src/train.py#L422), [lịch sử theo epoch](../outputs/results/full_mini_history.csv), [metadata kiểm tra](checkpoint_metadata.json).

## 4.2 Đối chiếu ghép cặp trên cùng tập test

**Verified · độ tin cậy cao cho dự đoán đã lưu.** Đối chiếu gồm 425 QA thuộc 197 ảnh, với `test_policy=subset`. So khớp nhãn tại từng dòng và tính mọi nhãn thật ngoài vocabulary là sai đối với cả hai bên. Bảng dùng exact-match accuracy, còn chênh lệch được tính từ số câu đúng chia cho cùng mẫu số. Baseline chọn đáp án phổ biến theo `question_type` từ hàng train đã giữ lại và có fallback từ phân bố train tổng thể. [Dự đoán theo dòng](../outputs/results/full_mini_test_predictions.csv), [quy tắc baseline](../src/evaluate.py#L28), [số liệu theo loại câu hỏi](../outputs/results/full_mini_test_question_types.csv).

| Nhóm câu hỏi | QA | ViT+BERT đúng | Baseline đúng | Chênh lệch (điểm %) | Chỉ ViT+BERT đúng | Chỉ baseline đúng |
|---|---:|---:|---:|---:|---:|---:|
| Tổng | 425 | 354 (83,29%) | 250 (58,82%) | +24,47 | 132 | 28 |
| Detailed Verification | 64 | 63 (98,44%) | 46 (71,88%) | +26,56 | 17 | 0 |
| General Health Assessment | 140 | 121 (86,43%) | 93 (66,43%) | +20,00 | 41 | 13 |
| Plant Species Identification | 146 | 116 (79,45%) | 46 (31,51%) | +47,95 | 73 | 3 |
| Visual Attribute Grounding | 75 | 54 (72,00%) | 65 (86,67%) | −14,67 | 1 | 12 |

Phép tính độc lập từ CSV dự đoán có thể chạy từ thư mục gốc với Python chuẩn.

```python
from collections import Counter, defaultdict
from csv import DictReader

counts = defaultdict(Counter)
with open("outputs/results/full_mini_test_predictions.csv", newline="", encoding="utf-8") as file:
    predictions = list(DictReader(file))
for row in predictions:
    truth, prediction, baseline = map(int, (row["true_label"], row["predicted_label"], row["baseline_label"]))
    pair = truth >= 0 and prediction == truth, truth >= 0 and baseline == truth
    counts[row["question_type"]][pair] += 1
    counts["all"][pair] += 1
for group, c in counts.items():
    n = sum(c.values())
    model_correct = c[True, True] + c[True, False]
    baseline_correct = c[True, True] + c[False, True]
    print(group, n, model_correct, baseline_correct, 100 * (model_correct - baseline_correct) / n,
          c[True, False], c[False, True])
print("images", len({row["image_id"] for row in predictions}))
```

**Inferred · diễn giải giới hạn.** Lợi thế tổng thể so với baseline tập trung mạnh ở nhận diện loài trên mẫu này. Visual Attribute Grounding cho kết quả ngược chiều. Trong nhóm đó, ở 12 QA chỉ baseline đoán đúng và ở một QA chỉ mô hình đoán đúng. QA dùng chung ảnh nên các dòng không độc lập. Chưa có kiểm định ghép cặp theo cụm ảnh hoặc tập ngoài mẫu để suy rộng. Kết quả cũng chưa định lượng đóng góp ảnh, câu hỏi hay pretrained weights vì các đối chứng đó chưa được quan sát. [Kiểm toán đánh giá](03_evaluation_audit.md).

## 4.3 Kiến trúc và đánh đổi chi phí

| Chiến lược | Kiến trúc theo code, Verified | Chi phí và ý nghĩa so sánh, Inferred khi chưa đo |
|---|---|---|
| `full` | ViT và BERT pretrained được cập nhật cùng projection và fusion head | Có gradient qua hai encoder. Lượt mini/full có log T4 và AMP float16 nhưng thiếu giờ GPU, thời gian mỗi epoch, bộ nhớ đỉnh và latency |
| `frozen` | Giữ hai encoder ở chế độ eval, forward trong no-grad; projection và fusion head vẫn học | Giảm backward và trạng thái optimizer cho encoder. Suy luận vẫn cần hai encoder. Chưa đo tốc độ, bộ nhớ hay mức thay đổi accuracy |
| `question_only` | Bỏ ViT và projection ảnh, điền vector 0 tại fusion | Kiểm tra tín hiệu dự đoán từ câu hỏi và khuôn mẫu. Tốn ít phép tính encoder hơn theo kiến trúc. Pipeline hiện vẫn có thể đọc/biến đổi ảnh trước model, nên cần đo chi phí toàn đường chạy |
| `image_only` | Bỏ BERT và projection văn bản, điền vector 0 tại fusion | Kiểm tra tín hiệu từ ảnh mà không phân biệt được câu hỏi trên cùng ảnh qua model. Pipeline hiện vẫn có thể tokenize trước forward. Chưa đo chi phí toàn đường chạy |
| Baseline tần suất train | Tra bảng theo loại câu hỏi và fallback | Không có ViT/BERT hay cập nhật gradient; dùng nhãn loại câu hỏi rõ ràng, trong khi backbone nhận nội dung câu hỏi và ảnh. Chi phí tương đối được suy từ thuật toán |

Nguồn kiến trúc và pipeline là [model](../src/models.py#L46), [train CLI](../src/train.py#L498), [dataset](../src/data.py), [mã baseline](../src/evaluate.py#L28) và [notebook Colab](../notebooks/train_colab.ipynb). Lượt chạy đã lưu dùng batch 4, tích lũy gradient 4, learning rate encoder 0,00002, head 0,001 và AdamW theo đường chạy hiện tại. Một cặp learning rate và một lịch học cố định không cung cấp đối chứng để quy hiệu năng cho từng siêu tham số. Giữ pretrained ở cả hai mode `full`/`frozen` khi so sánh sẽ cô lập quyết định mở gradient encoder tốt hơn. Muốn biết giá trị của pretrained initialization, cần một đối chứng khởi tạo ngẫu nhiên và quy định ngân sách huấn luyện trước. [Run marker](../outputs/results/full_mini_run.json), [kiểm toán cấu hình](01_configuration_audit.md).

## 4.4 Điều kiện cho so sánh tiếp theo

1. Chốt cùng manifest train/val/test, vocabulary, preprocessing, quy tắc unknown và tiêu chí chọn checkpoint cho từng cặp cần đối chiếu. Các mức mini/class/full có thể có test view khác nhau. Muốn đo lợi ích quy mô dữ liệu thì cần một test view giữ cố định và báo cáo mức phủ vocabulary riêng khi cần.
2. Nếu có ngân sách và được phép, chạy các đối chứng `frozen`, `image_only`, `question_only` ở cùng một mức với cấu hình hai nhánh, ghi toàn bộ hash artifact và thông số phần mềm, lưu số tham số trainable, thời gian epoch, GPU memory đỉnh và latency dưới một batch/thiết bị cố định.
3. Đặt trước tiêu chí đánh giá validation, chỉ dùng test sau khi chốt. Ghi song song exact-match accuracy, F1 theo lớp và theo loại câu hỏi, phân tầng theo độ phủ nhãn. Nếu báo độ bất định trên test, dùng đơn vị cụm ảnh. Tuân thủ quy tắc một lần chạy cho mỗi cấu hình với seed 59 trong repository. [Quy tắc project](../AGENTS.md).
