# Danh mục thực nghiệm trong workspace

Phạm vi là repository Plant Disease VQA và đầu ra Colab đã lưu trong đó. Không khảo sát trực tiếp Google Drive hoặc phiên Colab hiện tại. Không chạy lại huấn luyện hay suy luận.

| Nhóm | File và bằng chứng | Trạng thái |
|---|---|---|
| Mã nguồn | `src/config.py`, `src/download.py`, `src/data.py`, `src/models.py`, `src/train.py`, `src/evaluate.py`, `src/eda.py`, `src/verify_numbers.py`, `src/app.py`. Có thêm `src/training.py` và `src/evaluation.py` | Verified — hiện diện |
| Notebook / môi trường | `notebooks/train_colab.ipynb`, `requirements.txt`, `docs/colab_training.md`. Notebook lưu output kiểm tra GPU, lệnh, smoke test, đường dẫn lưu best và Drive | Verified — bằng chứng lịch sử, không xác nhận trạng thái Drive hiện tại |
| Dataset | `docs/dataset_card.md`, `data/source_inspection/PlantVillageVQA.csv`, `outputs/results/source_provenance.json`, `outputs/results/archive_inventory.json`, `outputs/results/verification.json` | Verified — CSV/metadata có sẵn. Không thể liệt kê cây ảnh tại `data/PlantVillageVQA/a025a438308eff9a7289231ebc09e27e8ec6678c` do EPERM |
| Huấn luyện mini/full | `outputs/results/full_mini_run.json`, `full_mini_history.csv`, `summary.csv` cùng thư mục | Verified — có log và dòng kết quả của một cấu hình |
| Cổng dữ liệu | `outputs/results/data_validation_mini_subset.json` và `.pytest.log`, `.pytest.xml` cùng tiền tố | Verified — có báo cáo kiểm tra lịch sử theo hash |
| Best checkpoint | `outputs/models/full_mini.pt` | Verified — file hiện diện; metadata và CRC ở báo cáo 02 |
| Resume checkpoint | `outputs/models/full_mini_last.pt` | Verified — file hiện diện; metadata và CRC ở báo cáo 02 |
| Test mini/full | `outputs/results/full_mini_test_metrics.json`, `full_mini_test_predictions.csv`, `full_mini_test_question_types.csv`, `full_mini_test_confusion_matrix.csv` | Verified — có thể tính lại metric từ dòng dự đoán |
| EDA / báo cáo triển khai | `outputs/figures/{answer_lengths,question_types}.pdf`; `outputs/results/{eda,observed_numbers,data_pipeline_inspection,models_implementation_report,train_evaluate_implementation_report}.json` | Verified — tách EDA và báo cáo kiểm thử khỏi kết quả thực nghiệm |
| Các cấu hình khác | CLI hỗ trợ tích Descartes 3 mức × 4 mode. Chỉ một cặp có artifact kết quả trong cây được đọc. Không thấy file `.pth`, `.ckpt` hoặc TensorBoard event trong cây được đọc | Missing — trạng thái training ngoài workspace chưa biết |

Để kiểm toán các run khác nếu đã chạy trên Colab, cần lấy từ `/content/outputs` hoặc bản sao `/content/drive/MyDrive/PlantDiseaseVQA`. Với từng cặp mode/level, cần `models/{mode}_{level}.pt`, `models/{mode}_{level}_last.pt` nếu có, `results/{mode}_{level}_run.json`, `results/{mode}_{level}_history.csv`, `results/summary.csv`, bộ `results/{mode}_{level}_test_{metrics.json,predictions.csv,question_types.csv,confusion_matrix.csv}` và bộ `results/data_validation_{level}_{test_policy}.*`. Để đối chiếu thực thi cần commit hoặc snapshot mã, runtime versions, thời gian train/chi phí nếu có, và hash checkpoint gắn với evaluation. Để replay cần đủ ảnh của revision và CSV có hash tương ứng. Thiếu file chỉ giới hạn kiểm toán.

Các đường dẫn viết tắt ở trên được tính từ thư mục đã nêu ở ô cùng hàng. Xem bảng CSV tổng hợp sau khi phân tích, trong `outputs/results/audit_experiment_summary.csv`.
