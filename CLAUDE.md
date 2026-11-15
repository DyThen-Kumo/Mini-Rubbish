# CLAUDE.md

## PRODUCTION
Hệ thống RAG hỏi đáp tiếng Việt về môn CS106 – Artificial Intelligence.

- **Nguồn tri thức:** 9 file `.txt` (UTF-8, tiếng Việt, ~300 KB) trong `OriginalDocs/`: Introduction, Uninformed/Informed Search, CSP & Local Search, Minimax, Expectimax, MDPs, Reinforcement Learning, Deep RL.
- **Người dùng:** sinh viên ôn tập. Họ hỏi bằng tiếng Việt và nhận câu trả lời dựa trên tài liệu, kèm trích dẫn nguồn (tên file + mục).
- **Luồng chính:** câu hỏi → tìm các đoạn liên quan trong tài liệu → LLM trả lời chỉ dựa trên các đoạn đó → hiển thị câu trả lời + nguồn.
- **Ngoài phạm vi:** khi tài liệu không có thông tin, hệ thống nói "không tìm thấy trong tài liệu" thay vì bịa câu trả lời.
- **Tiêu chí xong:** trả lời đúng ≥ 80% bộ câu hỏi mẫu trong `eval/`, mỗi câu trả lời đều có nguồn.

## ARCHITECTURE
```
OriginalDocs/        # Tài liệu gốc – CHỈ ĐỌC, không sửa
data/index/          # Vector index đã build (sinh tự động, có thể xoá và build lại)
src/
  ingest/            # Đọc file → tách chunk theo mục "=====" → embed → lưu vào data/index
  retrieval/         # Nhận câu hỏi → tìm top-k chunk liên quan
  generation/        # Ghép prompt (câu hỏi + chunk) → gọi LLM → trả lời kèm nguồn
app/                 # Giao diện (CLI hoặc web) gọi retrieval + generation
eval/                # Bộ câu hỏi/đáp án mẫu + script chấm điểm
tests/               # Unit test
```
Luồng dữ liệu: `OriginalDocs → ingest → data/index → retrieval → generation → app`

## DESIGN

## RULES
- KHÔNG BAO GIỜ sửa file trong `OriginalDocs/`
- KHÔNG BAO GIỜ run các file .ipynb và .py, cùng với các ngôn ngữ lập trình khác mà không có sự đồng ý.
- KHÔNG BAO GIỜ xoá các file trong `.gitignore`
- Mọi lệnh chạy trong conda env `rag`; thêm thư viện mới thì ghi vào `requirements.txt` kèm version.
- Đọc/ghi file luôn dùng `encoding="utf-8"` (tài liệu là tiếng Việt).
- Không hardcode API key: đọc từ biến môi trường / file `.env`, và không commit `.env`.
- Dùng đường dẫn tương đối từ gốc project (`pathlib.Path`), không dùng đường dẫn tuyệt đối.
- Code Python theo PEP 8, đặt tên rõ nghĩa, có type hint cho hàm public.
- Mỗi hàm làm một việc; không thêm abstraction/config khi chưa cần.
- Tham số hay chỉnh (chunk size, k, tên model) để ở một chỗ duy nhất, không rải khắp code.
- Xử lý lỗi rõ ràng (file thiếu, API lỗi), không `except: pass`.
- Logic mới (chunking, retrieval) phải có test trong `tests/`.
- Không sửa code ngoài phạm vi task đang làm.
- Làm xong một bước trong `TASK.md` thì đánh dấu `[x]`.
