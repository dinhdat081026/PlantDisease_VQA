# Phân tích thực nghiệm Plant Disease VQA

Cuộc kiểm toán bao phủ **12 cấu hình mà CLI hỗ trợ** (3 mức dữ liệu × 4 chế độ mô hình). Có **1 cấu hình hoàn thành được xác nhận trong workspace**, `mini/full`. Hai file checkpoint vượt qua kiểm tra ZIP/CRC và đối chiếu metadata. Một file best đã được nạp vào ViT+BERT trên CPU với `strict=True`, file thứ hai là trạng thái resume. Các số liệu test được tái tính từ bản ghi dự đoán. Các ảnh của revision cần chạy replay chưa truy cập được tại workspace, nên không chạy lại suy luận test.

Các nhãn bằng chứng dùng trong báo cáo gồm **Verified** (đã đọc hoặc tái tính từ nguồn), **Inferred** (diễn giải có giới hạn), **Missing** (thiếu bằng chứng) và **Inconsistent** (các mô tả hoặc giá trị không nhất quán). Mức tin cậy cao áp dụng cho việc đối chiếu artifact; suy luận về hành vi mô hình ngoài dữ liệu ghi nhận có mức thấp hơn.

## Chương 1. Tổng quan thực nghiệm

Bài toán là trả lời câu hỏi trên ảnh lá bằng bộ phân loại đáp án. PlantVillageVQA nguồn có 193.609 cặp QA trên 50.073 ảnh tham chiếu. View factoid gồm 96.825 QA trên 45.206 ảnh, giữ Detailed Verification, General Health Assessment, Plant Species Identification và Visual Attribute Grounding. Đây là thống kê nguồn; lần chạy thực tế chỉ dùng view mini và vocabulary 11 nhãn xây từ train cuối. Nguồn: [dataset card](../docs/dataset_card.md), [verification](../outputs/results/verification.json), [validation manifest](../outputs/results/data_validation_mini_subset.json).

Kiến trúc đối chiếu là timm `vit_base_patch16_224` và Transformers `bert-base-uncased`. Từng encoder cho vector 768 chiều, chiếu về 512 chiều và hợp nhất qua đầu phân loại `1024 → 512 → số nhãn`. Nhánh ngôn ngữ dùng trạng thái CLS cuối. Đường CLI huấn luyện yêu cầu trọng số pretrained. Chế độ `full` fine-tune hai encoder, `frozen` cố định hai encoder, và `question_only` / `image_only` bỏ một nhánh. Baseline đối chứng dự đoán đáp án phổ biến theo loại câu hỏi từ train cuối. Chỉ có một cấu hình mạng đã được đo; baseline không phải một lần huấn luyện ViT+BERT nữa. Nguồn: [models.py](../src/models.py), [train.py](../src/train.py), [evaluate.py](../src/evaluate.py).

## Chương 2. Thiết lập thực nghiệm

| Hạng mục | Thiết lập đã ghi nhận | Nguồn |
|---|---|---|
| Nơi chạy | Notebook lưu Python 3.13.16, PyTorch 2.14.1+cu130, CUDA 13.0, Tesla T4 14,56 GiB. Best checkpoint xác nhận bản PyTorch | [Notebook](../notebooks/train_colab.ipynb), [metadata](checkpoint_metadata.json) |
| Cấu hình | Mini, mode full, seed 59, 4 epoch, CUDA và AMP float16, microbatch 4 × tích lũy 4, workers 2, token limit 64 | [run marker](../outputs/results/full_mini_run.json), [notebook](../notebooks/train_colab.ipynb), [code AMP](../src/evaluate.py) |
| Tối ưu | AdamW, encoder LR 2×10⁻⁵, head LR 10⁻³, weight decay 0,01 cho nhóm phù hợp, gradient clipping 1,0. `LambdaLR` hệ số 1; LR cuối trong resume state khớp LR gốc | [run marker](../outputs/results/full_mini_run.json), [train.py](../src/train.py), [metadata](checkpoint_metadata.json) |
| Chia dữ liệu | Theo `image_id`, validation 15% tổng ảnh được lấy từ train gốc, giữ membership gốc trong sampled test; loại QA train có nhãn tần suất <5 | [data.py](../src/data.py), [manifest](../outputs/results/data_validation_mini_subset.json), [mã tái kiểm](scripts/verify_audit_data.py) |
| Tiền xử lý eval | Resize cạnh ngắn 248 với bicubic, crop giữa 224×224, chuẩn hóa từng kênh mean/std 0,5; tokenizer, độ dài và mapping khôi phục từ checkpoint | [manifest](../outputs/results/data_validation_mini_subset.json), [evaluate.py](../src/evaluate.py) |
| Quy tắc chọn | Tối đa exact-match validation accuracy, giữ epoch sớm nhất nếu hòa. Nhãn chưa biết tính sai trong accuracy; loss validation/test chỉ trên nhãn đã biết | [train.py](../src/train.py), [evaluate.py](../src/evaluate.py) |

Mini ban đầu có 1.000 ảnh / 2.141 QA. Sau lọc đáp án hiếm ở train, các tập có 646 ảnh / 1.375 QA (train), 150 / 326 (validation), 197 / 425 (test), tổng 993 ảnh / 2.126 QA. Không có giao `image_id` chuẩn hóa giữa các tập. Tái dựng từ CSV đã xác nhận SHA-256 của ba split và vocabulary, cùng SHA-256 của CSV, bốn file code dữ liệu và hai file test ghi trong manifest. Baseline được fit lại từ train và khớp với toàn bộ cặp nhãn thật/baseline trong bản xuất test khi ghép theo ảnh, loại câu hỏi và nhãn. Kiểm tra này không xác nhận trùng lặp nội dung giữa ảnh khác ID. Nguồn: [kiểm toán cấu hình](01_configuration_audit.md), [mã tái kiểm](scripts/verify_audit_data.py).

Log cổng dữ liệu lưu **36 pass, 0 fail/error/skip, 3 warning** và ghi nhận đã decode ảnh ở môi trường gốc. Lần kiểm toán này kiểm chứng lại hash và dữ liệu bảng; trạng thái decode ảnh là bằng chứng lịch sử. Phiên bản PyTorch Colab khác pin 2.6.0 trong requirements do notebook giữ bản CUDA có sẵn. Thiếu manifest đầy đủ của runtime, commit của mã đã chạy và revision/hash trọng số pretrained để tái lập chính xác. Nguồn: [validation log](../outputs/results/data_validation_mini_subset.pytest.log), [notebook](../notebooks/train_colab.ipynb), [requirements](../requirements.txt).

## Chương 3. Kết quả thực nghiệm

| Mức | full | frozen | question_only | image_only |
|---|---|---|---|---|
| mini | Hoàn thành 4 epoch; có test | Missing | Missing | Missing |
| class | Missing | Missing | Missing | Missing |
| full | Missing | Missing | Missing | Missing |

Nguồn: [CSV tổng hợp đủ các cấu hình](../outputs/results/audit_experiment_summary.csv). Missing biểu thị thiếu kết quả trong workspace; trạng thái của các run ngoài workspace chưa biết.

### Lịch sử huấn luyện

| Epoch | Train loss ghi log | Train accuracy | Validation loss trên nhãn đã biết | Validation accuracy |
|---:|---:|---:|---:|---:|
| 1 | 3,215700 | 70,98% | 0,553426 | 80,37% |
| 2 | 2,102134 | 82,18% | 0,427822 | 82,21% |
| **3** | **1,722248** | **85,45%** | **0,326043** | **86,50%** |
| 4 | 1,663278 | 85,60% | 0,679766 | 80,98% |

Verified, cao. Bốn epoch trong CSV khớp lịch sử bên trong `full_mini_last.pt`. Epoch 3 đạt 282/326 trên validation (3 unknown) và khớp `summary.csv`, metadata `full_mini.pt` và epoch mà test JSON ghi nhận. Resume checkpoint ở epoch 4 và tự ghi best epoch 3. Kiểm tra ZIP đọc 363/363 entry best và 1.430/1.430 entry resume với CRC hợp lệ. Best checkpoint còn nạp thành công qua `load_checkpoint` và `ViTBERT.load_state_dict(strict=True)` trên Windows, Python 3.13.2, PyTorch 2.8.0+cpu. Kiểm tra resume dừng ở toàn vẹn file và metadata. Nguồn: [history](../outputs/results/full_mini_history.csv), [summary](../outputs/results/summary.csv), [metadata](checkpoint_metadata.json), [kiểm toán training](02_training_audit.md).

[Training loss](figures/training_loss.pdf) và [training accuracy](figures/training_accuracy.pdf) chỉ hiển thị bốn điểm đã ghi và đánh dấu epoch được chọn. Train loss giữ giá trị gốc. Phép so khoảng cách giữa train loss và validation loss bị giới hạn bởi cách ghi loss hiện tại (Chương 4).

### Test mini

| Chỉ số | ViT+BERT | Baseline phổ biến theo loại câu |
|---|---:|---:|
| Exact-match accuracy | 354/425 (83,29%) | 250/425 (58,82%) |
| Macro precision | 53,08% | 14,74% |
| Macro recall | 63,27% | 17,39% |
| Macro F1 | 56,46% | 14,01% |
| Weighted F1 | 82,91% | 56,73% |

Verified, cao trên các QA đã ghi. Test là **425 QA trên 197 ảnh** từ view factoid mini. Có 5 nhãn ngoài vocabulary, tất cả tính sai. Macro lấy trung bình 11 đáp án cộng nhãn unknown, chỉ số không xác định quy về 0. Weighted F1 theo số mẫu thật từng lớp. Test CE trên nhãn đã biết là 0,393877 theo JSON; không xuất logits để tái tính CE. Nguồn: [predictions](../outputs/results/full_mini_test_predictions.csv), [metrics](../outputs/results/full_mini_test_metrics.json), [kiểm toán đánh giá](03_evaluation_audit.md).

| Loại câu hỏi | ViT+BERT đúng | Baseline đúng |
|---|---:|---:|
| Detailed Verification | 63/64 | 46/64 |
| General Health Assessment | 121/140 | 93/140 |
| Plant Species Identification | 116/146 | 46/146 |
| Visual Attribute Grounding | 54/75 | 65/75 |

Xem [so sánh accuracy trên test mini](figures/test_accuracy_baseline.pdf), [confusion matrix theo số QA](figures/test_confusion_matrix.pdf), [precision/recall/F1 theo đáp án](figures/test_metrics_by_class.pdf) và [bảng CSV theo lớp](../outputs/results/audit_full_mini_per_class.csv). Mã [tạo hình và bảng](scripts/build_audit_figures.py) tái tính hai ma trận từ dự đoán, đối chiếu với JSON và không điền dữ liệu cho các cấu hình thiếu.

## Chương 4. Phân tích và thảo luận

**Quá trình học.** Train accuracy tăng qua bốn epoch. Đến epoch 3, validation accuracy tăng và loss known giảm; ở epoch 4, validation accuracy giảm 5,52 điểm phần trăm và loss known tăng từ 0,326043 lên 0,679766. Đây là dấu hiệu suy giảm trên validation ở cuối run, phù hợp với khả năng bắt đầu quá khớp hoặc dao động của một run ngắn. Bốn điểm trên một split chưa đủ để xác định nguyên nhân, độ ổn định, mức hội tụ hay tình trạng underfitting. Mã hiện tại tại `src/train.py` dùng CE chưa chia trong biểu thức ghi log nhưng lại nhân accumulation. Nếu đúng phiên bản đó đã tạo lịch sử, train loss cao gấp 4 so với CE trung bình của các batch đã ghi. Không tự sửa lịch sử vì thiếu phiên bản được gắn trực tiếp với run. Nguồn: [history](../outputs/results/full_mini_history.csv), [train.py](../src/train.py#L115), [kiểm toán training](02_training_audit.md).

**So sánh trên cùng mẫu.** ViT+BERT cao hơn baseline 24,47 điểm phần trăm trên toàn view test mini. Kết quả giảm 14,67 điểm phần trăm ở Visual Attribute Grounding. Có 35 lỗi `No→Yes` và 24 lỗi `Yes→No`, chiếm 59/71 lỗi. `Apple` không có dự đoán đúng trên ba mẫu, `Pepper bell` sai ở mẫu duy nhất và `Blueberry` không có mẫu thật. Các lớp ít mẫu chỉ cho tín hiệu mô tả. QA có tương quan theo ảnh, nên phép suy luận thống kê tương lai phải giữ cụm ảnh và ghép cặp baseline. Nguồn: [kiểm toán đánh giá](03_evaluation_audit.md), [phân tích so sánh](04_experiment_comparison.md).

**Ảnh hưởng của pretrained và chi phí.** Run thực tế dùng đường fine-tuning pretrained, nhưng không có đối chứng frozen, init ngẫu nhiên hay ablation trên cùng split. Không thể tách hiệu ứng của kiến trúc, transfer learning và siêu tham số từ một đối chiếu với majority baseline. Mã cho thấy frozen giảm phần cần giữ gradient và optimizer state, còn ablation chỉ chạy encoder cần dùng. Không có số đo wall time, peak VRAM, throughput hay latency để định lượng đánh đổi. Nguồn: [phân tích so sánh](04_experiment_comparison.md).

**Vấn đề và giới hạn.**

| Mức | Trạng thái | Phát hiện |
|---|---|---|
| Critical khi công bố benchmark tham chiếu | Inconsistent | Số mini trước lọc 1.000/2.141 so với yêu cầu 995/2.102, class kiểm tra pipeline 8.000/17.076 so với 7.999/17.064. Đây là khác biệt giao thức ở phạm vi đã ghi. Không có manifest/sampler tham chiếu để truy nguyên đến danh sách ID từng ảnh. |
| Major | Inconsistent | Công thức log train loss trong mã hiện tại lệch thang khi gradient accumulation > 1. Ghi chú phiên bản chạy trước khi diễn giải giá trị tuyệt đối. |
| Major | Missing | Test JSON có đường dẫn và epoch nhưng không có hash trọng số được đánh giá. Nhãn thật, baseline, metric, epoch và nguồn nhất quán; chưa xác thực bằng replay ảnh hoặc định danh checkpoint theo byte. |
| Major | Inferred | Resume lưu RNG global nhưng không lưu generator riêng của DataLoader. Resume có khả năng khác thứ tự shuffle so với run liên tục. Không có dấu hiệu run mini đã dùng resume. |
| Major | Missing | Chưa có kết quả các cấu hình còn lại, đo runtime đầy đủ hoặc kiểm tra trùng ảnh theo nội dung. Test mini cũng không đại diện mặc định cho ảnh thực địa. |
| Minor | Inconsistent | README có mô tả hiện trạng cũ. Ưu tiên log, notebook và artifact có thể kiểm tra. |

Nguồn: [kiểm toán cấu hình](01_configuration_audit.md), [kiểm toán training](02_training_audit.md), [kiểm toán đánh giá](03_evaluation_audit.md), [phản biện khoa học](05_scientific_review.md). Không thấy bằng chứng lựa chọn checkpoint bằng test hoặc trùng `image_id` ở split đã tái dựng.

## Chương 5. Kết luận

Epoch 3 là **checkpoint được chọn của cấu hình mini/full**, theo validation accuracy 86,50%. Checkpoint này có test accuracy lưu và tái tính được là 83,29% trên view mini, so với 58,82% của baseline cùng mẫu. Độ tin cậy cao đối với đối chiếu log, split và metric từ artifact, có giới hạn đối với việc gắn dự đoán với chính xác byte trọng số. Chưa có căn cứ xếp hạng 12 cấu hình hoặc kết luận về ưu thế của fine-tuning trên cùng benchmark.

Các bước bổ sung theo thứ tự ưu tiên:

1. Lấy manifest split/sampler tham chiếu và xác định giao thức chính thức trước khi gộp mini/class vào bảng benchmark. Giữ kết quả đang có dưới nhãn sampled mini.
2. Thu thập ảnh theo revision, commit mã, runtime freeze, GPU/driver, revision pretrained, lệnh và fingerprint trọng số. Lưu metadata nhận dạng checkpoint/manifest khi xuất dự đoán lần tới. Chỉ replay khi đủ điều kiện.
3. Khi được phép, chạy các đối chứng frozen, question_only và image_only trên cùng split đã chốt. Bổ sung đối chứng init ngẫu nhiên nếu mục tiêu là đo transfer learning. Chọn cấu hình theo validation đã định trước rồi mới tính test.
4. Thu thập epoch time, peak VRAM và optimizer step AMP hợp lệ. Rà soát công thức log loss và khả năng resume liên tục trong bản thay đổi riêng.
5. Kiểm tra ảnh trùng/near-duplicate xuyên tập và phân tích các lỗi Yes/No theo câu hỏi. Nếu cần khoảng bất định cho test hiện tại, dùng phép lấy mẫu theo cụm ảnh với giao thức định trước và báo cáo như phân tích sau thực nghiệm.

Danh sách artifact, thư mục Colab/Drive và tên các file cần lấy nằm trong [danh mục thực nghiệm](experiment_inventory.md).
