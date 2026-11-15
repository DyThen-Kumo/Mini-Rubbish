# TASK

1. [ ] Chọn stack: LLM, embedding model (cần hỗ trợ tiếng Việt), vector store, framework → ghi vào DESIGN trong `CLAUDE.md`
2. [ ] Tạo `requirements.txt`, cài vào conda env `rag`
3. [ ] Ingest: đọc 9 file UTF-8, tách chunk theo tiêu đề mục, gắn metadata (file, tên mục)
4. [ ] Embed các chunk và lưu index vào `data/index/`
5. [ ] Retrieval: hàm `search(question, k)` trả về top-k chunk + metadata
6. [ ] Generation: prompt bắt LLM chỉ dùng context và trích nguồn; nếu không có thông tin thì nói không biết
7. [ ] App: CLI hỏi đáp (sau đó mới làm web UI nếu cần)
8. [ ] Eval: viết 20–30 câu hỏi mẫu trải đều 9 bài, đo độ chính xác và tỉ lệ có nguồn
9. [ ] Tinh chỉnh: kích thước chunk, k, prompt, có thể thêm hybrid search (BM25) hoặc rerank
10. [ ] Tests cho ingest và retrieval
