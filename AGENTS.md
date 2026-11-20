
# Plant Disease VQA

Bài toán hỏi đáp trên ảnh lá cây (visual question answering, VQA) dạng phân loại: mô hình nhận một ảnh lá và một câu hỏi, rồi chọn một đáp án trong tập đáp án xây từ train. Baseline là ViT + BERT (bert-base-uncased) trên PlantVillageVQA. Chỉ giữ câu hỏi factoid (đáp án tối đa 3 từ). Đọc docs/dataset_card.md trước khi viết bất kỳ đoạn code nào liên quan đến dữ liệu.

## Cấu trúc
src/{config,download,data,models,train,evaluate,analysis,eda,app}.py,
tests/, outputs/{figures,models,results}/, data/ (không commit),
.claude/{hooks,skills,agents}.

## Lệnh chính
- python -m src.download --data_dir data/PlantVillageVQA
  (hoặc đặt PLANTVQA_DATA_DIR)
- python -m src.train --level class --mode full (mức: mini, class, full)
- python .claude/skills/vqa-benchmark/scripts/run_experiments.py \
    --gpus 1,2
- python -m src.evaluate --ckpt outputs/models/full_class.pt
- streamlit run src/app.py
- pytest

## Ràng buộc bắt buộc
- SEED = 59 chỉ khai báo một lần trong src/config.py, mọi seed khác suy ra từ SEED. Không dùng 42.
- Chia train/val/test theo image_id, mọi cặp QA của một ảnh nằm trong cùng một tập. Phép chia chỉ viết trong src/data.py.
- Tập đáp án và mọi thống kê chuẩn hóa chỉ xây từ train.
- Mỗi cấu hình thí nghiệm chỉ chạy một lần với SEED, không báo cáo mean và std, không kết luận về chênh lệch nhỏ.
- Chạy trên Google Colab T4: dùng AMP torch.float16, không dùng bf16, không dùng flash_attention_2. Ghim phiên bản trong requirements.txt.
- Chọn checkpoint theo val accuracy, không bao giờ dò theo test.
- Không huấn luyện trong app. App chỉ nạp checkpoint trong outputs/models/.
- Không copy dữ liệu vào repo và không commit data/.

## Quy ước
- Hình lưu PDF, dpi=150, bbox_inches="tight", font serif họ Times, nhãn tiếng Anh.
- Bảng kết quả lưu thành file .csv trong outputs/results/.
- Mọi con số báo cáo phải có đoạn code tính ra.

## Manual tests
Các con số dưới đây đã được kiểm chứng bằng code (src/verify_numbers.py). Nếu một thay đổi làm lệch bất kỳ con số nào thì thay đổi đó sai, không phải con số sai.
- PlantVillageVQA.csv có 193609 cặp QA, 50073 ảnh, 6 cột, split gốc gồm train (40066 ảnh) và test (10007 ảnh), không có image_id nào ở hơn một split.
- Chín question_type. Bốn loại được giữ: Detailed Verification (2 đáp án), General Health Assessment (2), Plant Species Identification (16), Visual Attribute Grounding (2).
- Sau khi lọc còn 96825 cặp QA và 45206 ảnh, đáp án dài nhất 2 từ.
- Mức class (target 8000 ảnh): 7999 ảnh, 17064 cặp QA, 16 đáp án. Train 5196 ảnh và 11129 cặp, val 1200 ảnh và 2544 cặp, test 1603 ảnh và 3391 cặp.
- Mức mini: 995 ảnh, 2102 cặp QA, 11 đáp án. Mức full: 45206 ảnh, 96825 cặp QA, 16 đáp án.
- summary.csv có đúng một dòng mỗi cấu hình, cột seed bằng 59 ở mọi dòng.
