<p align="center">
  <img src="assets/images/LOGO_Mini_Rubbish_Stelle.png" alt="Mini Rubbish logo" width="360">
</p>

<h1 align="center">MINI RUBBISH</h1>

Hệ thống hỏi đáp (RAG) về môn CS106 – Artificial Intelligence, dựa trên các tài liệu tóm tắt bài giảng trong `OriginalDocs/`.

## Problem
Sinh viên cần ôn tập 9 bài giảng CS106 (Search, CSP, Minimax, Expectimax, MDPs, Reinforcement Learning, Deep RL). Đọc lại toàn bộ tài liệu để tìm một khái niệm thì tốn thời gian, còn hỏi một chatbot chung chung thì câu trả lời có thể sai hoặc không khớp với nội dung bài giảng.

Bài toán là xây dựng hệ thống nhận câu hỏi bằng ngôn ngữ tự nhiên, tìm đúng đoạn liên quan trong tài liệu môn học, rồi sinh câu trả lời **chỉ dựa trên các đoạn đó** và kèm nguồn.

## Input
- **Kho tài liệu:** 9 file `.txt` trong `OriginalDocs/`, mỗi file tóm tắt một bài giảng.
- **Câu hỏi:** một câu hỏi dạng văn bản của người dùng, ví dụ: *"Sự khác nhau giữa SARSA và Q-Learning là gì?"*

## Output
- **Câu trả lời:** văn bản tiếng Việt trả lời trực tiếp câu hỏi.
- **Nguồn:** danh sách đoạn tài liệu đã dùng (tên file + tên mục).
- Nếu tài liệu không có thông tin: một thông báo rõ ràng *"Không tìm thấy trong tài liệu"*.

## Constraints
**Kho tài liệu:**
- Là văn bản thuần, mã hoá UTF-8, chủ yếu viết bằng tiếng Việt và giữ nguyên nhiều thuật ngữ tiếng Anh (agent, heuristic, Q-value…).
- Mỗi file được chia mục bằng các dòng tiêu đề kẻ `=====`.
- Có công thức viết dạng text (Bellman equation, TD update…).
- Là dữ liệu chỉ đọc: hệ thống không được sửa `OriginalDocs/`.

**Câu hỏi:**
- Có thể viết bằng tiếng Việt, tiếng Anh hoặc trộn cả hai.
- Có thể ngắn, mơ hồ, sai chính tả hoặc không có dấu.
- Có thể nằm ngoài phạm vi môn học.
- Có thể cần thông tin từ nhiều bài cùng lúc (ví dụ so sánh Minimax và Expectimax).

## Requirements
- **Trung thực với tài liệu:** chỉ dùng thông tin có trong tài liệu, không bịa thêm kiến thức.
- **Có nguồn:** mọi câu trả lời đều chỉ ra được đoạn tài liệu đã dùng.
- **Biết từ chối:** khi tài liệu không có thông tin, hệ thống nói rõ là không tìm thấy thay vì đoán.
- **Đúng trọng tâm:** trả lời thẳng vào câu hỏi, ngắn gọn, không lan man.
- **Ngôn ngữ:** trả lời bằng tiếng Việt và giữ nguyên thuật ngữ tiếng Anh như trong tài liệu.
- **Nhất quán:** cùng một câu hỏi thì cho câu trả lời có nội dung tương đương.
- **Đo được:** đạt ≥ 80% câu trả lời đúng trên bộ câu hỏi mẫu trong `eval/`.

## Reference
Bản BibTeX đầy đủ: [`references.bib`](references.bib).

1. CS106 Artificial Intelligence – Lecture Summaries 01–10. Thư mục `OriginalDocs/`.
2. P. Lewis et al. *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.* NeurIPS 2020. [arXiv:2005.11401](https://arxiv.org/abs/2005.11401)
3. S. Robertson, H. Zaragoza. *The Probabilistic Relevance Framework: BM25 and Beyond.* Foundations and Trends in Information Retrieval, 3(4), 2009. [doi:10.1561/1500000019](https://doi.org/10.1561/1500000019)
4. G. van Rossum, B. Warsaw, A. Coghlan. *PEP 8 – Style Guide for Python Code.* 2001. [peps.python.org/pep-0008](https://peps.python.org/pep-0008/)
