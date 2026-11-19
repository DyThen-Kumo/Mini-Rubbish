<p align="center">
  <img src="assets/images/LOGO_Mini_Rubbish_Stelle.png" alt="Mini Rubbish logo" width="240">
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

## Usage
Chạy trong conda env `rag`, cần file `.env` có `GOOGLE_API_KEY` và index đã build (`python -m src.ingest.build_index`).

- **CLI:** `python -m app.cli`
- **Web:** `uvicorn api.main:app`, rồi mở `http://127.0.0.1:8000` (khung chat + mục lục bài giảng; bấm một mục để điền sẵn câu hỏi).
- **API:** `uvicorn api.main:app`, rồi gửi câu hỏi:
  ```bash
  curl -X POST http://127.0.0.1:8000/ask -H "Content-Type: application/json" -d '{"question": "A* là gì?"}'
  ```
  Trả về `{"answer": "...", "sources": ["file > mục", ...]}`. Lỗi: 422 (câu hỏi rỗng / dài quá 1000 ký tự), 429 (Gemini hết quota), 502 (Gemini lỗi), 503 (thiếu index / API key).

Không chạy CLI và API cùng lúc: Qdrant local mode chỉ cho một tiến trình mở `data/index`.

### Docker Compose (Qdrant server + API)
Cần Docker Desktop và file `.env`. Lần đầu:
```bash
docker compose up -d qdrant                                        # bật Qdrant server
docker compose run --rm api python -m src.ingest.build_index       # build image, embed tài liệu vào Qdrant (một lần)
docker compose up -d --build                                       # bật API
```
Mở `http://127.0.0.1:8000` (web) hoặc `http://localhost:6333/dashboard` (Qdrant). Tắt: `docker compose down`; `docker compose down -v` xoá luôn index và bge-m3 đã tải.

Docker Compose đặt biến `QDRANT_URL`, nên API dùng Qdrant server. Chạy trực tiếp trên máy (không có biến này) thì vẫn dùng local `data/index` như trên.

## Reference
Bản BibTeX đầy đủ: [`references.bib`](references.bib).

1. CS106 Artificial Intelligence – Lecture Summaries 01–10. Thư mục `OriginalDocs/`.
2. P. Lewis et al. *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.* NeurIPS 2020. [arXiv:2005.11401](https://arxiv.org/abs/2005.11401)
3. S. Robertson, H. Zaragoza. *The Probabilistic Relevance Framework: BM25 and Beyond.* Foundations and Trends in Information Retrieval, 3(4), 2009. [doi:10.1561/1500000019](https://doi.org/10.1561/1500000019)
4. G. van Rossum, B. Warsaw, A. Coghlan. *PEP 8 – Style Guide for Python Code.* 2001. [peps.python.org/pep-0008](https://peps.python.org/pep-0008/)
5. V. Karpukhin et al. *Dense Passage Retrieval for Open-Domain Question Answering.* EMNLP 2020. [arXiv:2004.04906](https://arxiv.org/abs/2004.04906)
6. N. Reimers, I. Gurevych. *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks.* EMNLP-IJCNLP 2019. [arXiv:1908.10084](https://arxiv.org/abs/1908.10084)
7. J. Chen et al. *M3-Embedding: Multi-Linguality, Multi-Functionality, Multi-Granularity Text Embeddings Through Self-Knowledge Distillation.* Findings of ACL 2024. [arXiv:2402.03216](https://arxiv.org/abs/2402.03216)
8. Yu. A. Malkov, D. A. Yashunin. *Efficient and Robust Approximate Nearest Neighbor Search Using Hierarchical Navigable Small World Graphs.* IEEE TPAMI, 42(4), 2020. [doi:10.1109/TPAMI.2018.2889473](https://doi.org/10.1109/TPAMI.2018.2889473)
9. A. Radford et al. *Learning Transferable Visual Models From Natural Language Supervision.* ICML 2021. [arXiv:2103.00020](https://arxiv.org/abs/2103.00020)
10. LangChain. *LangChain Documentation.* [python.langchain.com](https://python.langchain.com/)
11. Google. *Gemini API Documentation.* [ai.google.dev/gemini-api/docs](https://ai.google.dev/gemini-api/docs)
12. Qdrant. *Qdrant Documentation.* [qdrant.tech/documentation](https://qdrant.tech/documentation/)
13. S. Ramírez. *FastAPI Documentation.* [fastapi.tiangolo.com](https://fastapi.tiangolo.com/)
14. Anthropic. *Introducing Contextual Retrieval.* 2024. [anthropic.com/news/contextual-retrieval](https://www.anthropic.com/news/contextual-retrieval)
15. S. Xiao et al. *C-Pack: Packed Resources For General Chinese Embeddings.* SIGIR 2024. [arXiv:2309.07597](https://arxiv.org/abs/2309.07597)
16. S. Es, J. James, L. Espinosa-Anke, S. Schockaert. *RAGAs: Automated Evaluation of Retrieval Augmented Generation.* EACL 2024 System Demonstrations. [arXiv:2309.15217](https://arxiv.org/abs/2309.15217)
17. P. Rajpurkar, R. Jia, P. Liang. *Know What You Don't Know: Unanswerable Questions for SQuAD.* ACL 2018. [arXiv:1806.03822](https://arxiv.org/abs/1806.03822)
18. N. F. Liu et al. *Lost in the Middle: How Language Models Use Long Contexts.* TACL, 12, 2024. [arXiv:2307.03172](https://arxiv.org/abs/2307.03172)
19. L. Zheng et al. *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena.* NeurIPS 2023 Datasets and Benchmarks. [arXiv:2306.05685](https://arxiv.org/abs/2306.05685)
20. A. Panickssery, S. R. Bowman, S. Feng. *LLM Evaluators Recognize and Favor Their Own Generations.* NeurIPS 2024. [arXiv:2404.13076](https://arxiv.org/abs/2404.13076)
21. S. Min et al. *FActScore: Fine-grained Atomic Evaluation of Factual Precision in Long Form Text Generation.* EMNLP 2023. [arXiv:2305.14251](https://arxiv.org/abs/2305.14251)
22. C. Dwork et al. *The reusable holdout: Preserving validity in adaptive data analysis.* Science, 349(6248), 2015. [doi:10.1126/science.aaa9375](https://doi.org/10.1126/science.aaa9375)
23. E. M. Voorhees. *The TREC-8 Question Answering Track Report.* TREC-8, 1999. [trec.nist.gov](https://trec.nist.gov/pubs/trec8/papers/qa_report.pdf)
