<p align="center">
  <img src="assets/images/LOGO_Mini_Rubbish_Stelle.png" alt="Mini Rubbish logo" width="240">
</p>

<h1 align="center">MINI RUBBISH</h1>
<p align="center"><b>Hệ thống hỏi đáp RAG tiếng Việt cho môn CS106 – Artificial Intelligence</b></p>

---

## 1. Tóm tắt

**Mục đích:** xây dựng hệ thống trả lời câu hỏi tiếng Việt về môn CS106. Câu trả lời chỉ được dựa trên 9 tài liệu tóm tắt bài giảng, phải kèm nguồn, và hệ thống phải từ chối khi tài liệu không có thông tin.

**Phương pháp:** dùng pipeline RAG [2]. Tài liệu được tách chunk theo mục và embed bằng `BAAI/bge-m3` [7], sau đó lưu vào Qdrant [12]. Với mỗi câu hỏi, hệ thống lấy top-4 chunk rồi đưa cho Gemini 3.5 Flash Lite [11] sinh câu trả lời có trích nguồn. Hệ thống được chấm bằng LLM-as-judge [19] (ChatGPT, tiêu chí chặt) trên bộ dev 40 câu và bộ test held-out 30 câu.

**Kết quả:** trên bộ test, độ chính xác đạt 18/30 (60%) và 26/26 câu trả lời có nguồn (100%). Hệ thống từ chối đúng 3/3 câu ngoài phạm vi, retrieval hit@4 đạt 27/27.

**Kết luận:** hệ thống đạt tiêu chí "mọi câu trả lời có nguồn" nhưng chưa đạt tiêu chí độ chính xác ≥ 80%. Retrieval gần như luôn lấy đúng đoạn tài liệu, còn nút thắt nằm ở model sinh câu trả lời: câu trả lời thường thiếu ý so với đáp án mẫu, hiếm khi sai.

---

## 2. Giới thiệu và mục tiêu

### 2.1 Bối cảnh và vấn đề
Sinh viên cần ôn tập 9 bài giảng CS106: Introduction, Uninformed/Informed Search, CSP & Local Search, Minimax, Expectimax, MDPs, Reinforcement Learning và Deep RL. Đọc lại toàn bộ tài liệu để tìm một khái niệm thì tốn thời gian. Hỏi một chatbot chung chung thì câu trả lời có thể sai hoặc không khớp với nội dung bài giảng.

Bài toán đặt ra là xây dựng một hệ thống nhận câu hỏi bằng ngôn ngữ tự nhiên, tìm đúng đoạn liên quan trong tài liệu môn học, rồi sinh câu trả lời **chỉ dựa trên các đoạn đó** và kèm nguồn.

### 2.2 Đầu vào và đầu ra
- **Đầu vào:**
  - Kho tài liệu gồm 9 file `.txt` trong `OriginalDocs/` (UTF-8, ~300 KB).
  - Một câu hỏi dạng văn bản, ví dụ *"Sự khác nhau giữa SARSA và Q-Learning là gì?"*
- **Đầu ra:**
  - Câu trả lời tiếng Việt, giữ nguyên thuật ngữ tiếng Anh như trong tài liệu.
  - Danh sách nguồn dạng `tên file > tên mục`.
  - Khi tài liệu không có thông tin, đầu ra là đúng câu *"Không tìm thấy trong tài liệu"*.

### 2.3 Ràng buộc
- **Tài liệu:**
  - Văn bản thuần, chủ yếu tiếng Việt, có nhiều thuật ngữ tiếng Anh.
  - Chia mục bằng dòng kẻ `=====`, công thức viết dạng text.
  - Chỉ đọc: hệ thống không được sửa `OriginalDocs/`.
- **Câu hỏi:**
  - Có thể viết bằng tiếng Việt, tiếng Anh hoặc trộn cả hai.
  - Có thể ngắn, mơ hồ, sai chính tả hoặc không dấu.
  - Có thể nằm ngoài phạm vi môn học, hoặc cần thông tin từ nhiều bài cùng lúc.

### 2.4 Mục tiêu và tiêu chí xong
1. **Trung thực:** chỉ dùng thông tin có trong tài liệu.
2. **Có nguồn:** mọi câu trả lời đều chỉ ra đoạn tài liệu đã dùng.
3. **Biết từ chối:** nói "không tìm thấy" thay vì đoán.
4. **Đo được:** đúng ≥ 80% trên bộ câu hỏi mẫu, và 100% câu trả lời có nguồn.

### 2.5 Câu hỏi nghiên cứu và giả thuyết
- **Câu hỏi:** một pipeline RAG đơn giản, miễn phí (embedding chạy local và LLM free tier) có đạt được tiêu chí trên với tài liệu tiếng Việt hay không?
- **Giả thuyết H1:** tách chunk theo mục có sẵn của tài liệu và dùng embedding đa ngôn ngữ thì retrieval sẽ lấy đúng mục chứa đáp án ở phần lớn câu hỏi.
- **Giả thuyết H2:** khi context đã đúng, prompt buộc chỉ dùng context sẽ cho câu trả lời đúng, có nguồn và không bịa.

---

## 3. Cơ sở lý thuyết

Chi tiết lý thuyết và lý do của từng quyết định có trong [`THEORY.md`](THEORY.md). Phần này chỉ tóm tắt.

- **Retrieval-Augmented Generation (RAG) [2]:**
  - LLM sinh câu trả lời dựa trên các đoạn văn được truy xuất từ kho tài liệu, thay vì chỉ dựa vào kiến thức đã học.
  - Nhờ vậy câu trả lời bám tài liệu và trích được nguồn.
- **Dense retrieval [5][6]:**
  - Câu hỏi và chunk được mã hoá độc lập bằng cùng một bi-encoder thành vector.
  - Độ liên quan đo bằng cosine giữa hai vector.
  - BGE-M3 [7] (kế thừa họ BGE [15]) là model đa ngôn ngữ 1024 chiều, có hỗ trợ tiếng Việt và không cần prefix cho câu hỏi.
- **Tìm kiếm lân cận:**
  - Qdrant server dùng HNSW [8], tìm gần đúng với độ phức tạp $O(\log N)$.
  - Local mode tìm vét cạn $O(N)$, đủ nhanh với 248 chunk.
- **Hạn chế của dense retrieval:**
  - Khớp ký hiệu như "A*" kém hơn tìm kiếm theo từ khoá BM25 [3].
  - LLM khai thác context dài kém, nhất là thông tin nằm ở giữa context [18], nên `k` cần nhỏ.
- **Câu không trả lời được [17]:** hệ thống chặn ở hai lớp:
  - Ngưỡng điểm retrieval.
  - Prompt yêu cầu từ chối khi context không có thông tin.
- **Đánh giá:**
  - LLM-as-judge [19] tiện lợi nhưng có thiên lệch tự ưu ái khi judge và hệ thống cùng một model [20].
  - Vì vậy judge tách đáp án thành các ý chính rồi kiểm tra từng ý, theo hướng chấm theo từng sự kiện nguyên tử [21][16].
- **Tách dev/test:** chỉnh hệ thống nhiều lần trên cùng một holdout làm holdout mất tính độc lập [22], nên bộ test chỉ được chạy một lần.

---

## 4. Vật liệu và phương pháp

### 4.1 Dữ liệu và công cụ

**Bảng 1.** Thành phần hệ thống.

| Thành phần | Lựa chọn | Ghi chú |
|---|---|---|
| Tài liệu | 9 file `.txt`, CS106 Lecture Summaries [1] | 248 chunk sau khi tách |
| Framework | LangChain [10] | Chỉ dùng Document, embeddings, vector store, chat model |
| Embedding | `BAAI/bge-m3` [7] | Chạy local, 1024 chiều, `normalize_embeddings=True` |
| Vector store | Qdrant [12] | Local mode `data/index`; có `QDRANT_URL` thì dùng server mode |
| LLM | `gemini-3.5-flash-lite` [11] | Free tier, 10 request/phút; model bỏ qua `temperature=0` |
| API / Web | FastAPI [13] | `POST /ask`, `GET /toc`, giao diện chat tại `GET /` |
| Triển khai | Docker Compose | Qdrant server + API |
| Judge | ChatGPT (chấm tay qua giao diện web) | Khác model với hệ thống, để tránh thiên lệch [20] |

### 4.2 Sơ đồ hệ thống

```
OriginalDocs (9 file .txt)
  → ingest: tách chunk theo "=====", gắn metadata (file, mục)
  → bge-m3: chunk → vector 1024 chiều (chuẩn hoá)
  → Qdrant: lưu vector + payload                          [data/index]

câu hỏi → bge-m3 → Qdrant top-4 (cosine)
  → top-1 < MIN_SCORE ? → "Không tìm thấy trong tài liệu"
  → prompt (câu hỏi + 4 chunk đánh số [1]..[4]) → Gemini
  → câu trả lời + nguồn (code ghép từ metadata của chunk được trích)
```

### 4.3 Các bước tiến hành
1. **Ingest (`src/ingest/chunking.py`):**
   - Tách mỗi file theo các tiêu đề mục `=====`.
   - Mục dài hơn `CHUNK_SIZE = 1500` ký tự thì cắt tiếp, ưu tiên ranh giới đoạn văn, rồi dòng, rồi từ.
   - Thêm dòng `[tên file > tên mục]` vào đầu mỗi chunk (contextual header [14]).
2. **Index (`src/ingest/build_index.py`):** embed toàn bộ chunk bằng bge-m3 rồi ghi vào collection `cs106` (cosine). Mỗi lần build sẽ tạo lại collection từ đầu.
3. **Retrieval (`src/retrieval/search.py`):** `search(question, k)` embed câu hỏi bằng cùng model, rồi trả về top-k chunk kèm điểm và metadata.
4. **Generation (`src/generation/answer.py`):**
   - Chunk tốt nhất có điểm < `MIN_SCORE = 0.5` thì từ chối ngay, không gọi LLM.
   - Ngược lại, prompt yêu cầu chỉ dùng context, nêu đủ định nghĩa, công thức, điều kiện; ghi số nguồn `[n]` sau mỗi ý; không có thông tin thì trả lời đúng câu từ chối.
   - Danh sách nguồn do code tạo từ metadata, nên LLM không thể bịa tên file hay tên mục.
5. **Ứng dụng:** CLI (`app/cli.py`), API và web (`api/main.py`, `app/web/index.html`), Docker Compose.
6. **Kiểm thử:** 49 unit test cho chunking, retrieval, generation, API, CLI và script eval (`pytest tests/`).

### 4.4 Thiết kế thí nghiệm

**Bảng 2.** Các biến của thí nghiệm.

| Loại biến | Biến |
|---|---|
| Độc lập (thay đổi giữa các lần chạy) | Judge (Gemini dễ → Gemini chặt → ChatGPT chặt); prompt ("ngắn gọn" → "đầy đủ"); `TOP_K` (4, 6, 8, 10 khi đo retrieval) |
| Phụ thuộc (đo) | Độ chính xác, tỉ lệ có nguồn, nguồn đúng, từ chối đúng/oan, trung thực, hit@k, MRR |
| Kiểm soát (giữ cố định) | Tài liệu, `CHUNK_SIZE = 1500`, bge-m3, `MIN_SCORE = 0.5`, LLM `gemini-3.5-flash-lite` |

**Bộ câu hỏi:**
- **Dev** (`eval/questions/`, 40 câu q01–q40), dùng để tinh chỉnh:
  - 30 câu thường, trải đều 9 bài.
  - 10 câu khó: không dấu, tiếng Anh, trộn Anh-Việt, sai chính tả, ngắn/mơ hồ, nhiều bài.
  - Trong đó có 4 câu ngoài phạm vi.
- **Test** (`eval/test/`, 30 câu q101–q130):
  - Phân bố giống dev và lấy từ các mục mà dev chưa hỏi.
  - Chỉ chạy **một lần** ở bước cuối, không tinh chỉnh sau khi xem kết quả.

### 4.5 Thu thập và xử lý dữ liệu
1. Chạy `python -m eval.run_eval` (dev) hoặc `--split test`. Hệ thống trả lời từng câu và ghi vào `eval/results/<run>.jsonl`.
2. Dán `<run>.judge_prompt.txt` và upload `<run>.judge_input.json` lên ChatGPT, tải về `verdicts.json`.
3. Judge trả về `missing_points` (ý thiếu), `wrong_points` (ý sai) và `faithful` cho mỗi câu. **Code** quyết định đúng/sai, không để judge tự quyết:

$$\text{correct} = (\text{missing\_points} = \varnothing) \land (\text{wrong\_points} = \varnothing)$$

$$\text{Accuracy} = \frac{\#\text{câu đúng}}{\#\text{câu}}$$

4. Độ đo retrieval (`python -m eval.retrieval_hits`, không gọi LLM):
   - **Section hit@k:** mọi cặp (file, mục) trong nhãn đều nằm trong top-k.
   - **MRR@k** [23]: với $\text{rank}_i$ là hạng của chunk đúng mục đầu tiên ($1/\text{rank}_i = 0$ nếu không có trong top-k),

$$\text{MRR@}k = \frac{1}{|Q|}\sum_{i=1}^{|Q|} \frac{1}{\text{rank}_i}$$

---

## 5. Kết quả

**Bảng 3.** Kết quả trên bộ dev (40 câu) qua các lần chạy, và trên bộ test (30 câu).

| Lần | Thay đổi | Bộ | Độ chính xác | Có nguồn | Từ chối đúng / oan | Trung thực | Độ dài TB |
|---|---|---|---|---|---|---|---|
| 1 | Baseline, judge Gemini dễ | dev | 39/40 (98%) | 35/35 | 4/4 / 1 | 35/35 | 330 |
| 2 | Judge Gemini chặt | dev | 23/40 (58%) | 35/35 | 4/4 / 1 | 35/35 | 326 |
| 3 | Judge ChatGPT chặt | dev | 14/40 (35%) | 35/35 | 4/4 / 1 | 35/35 | 338 |
| 4 | Prompt "đầy đủ" | dev | 23/40 (58%) | 35/35 | 4/4 / 1 | 35/35 | 814 |
| **5** | **Không đổi (held-out)** | **test** | **18/30 (60%)** | **26/26** | **3/3 / 1** | **25/26** | **799** |

*Chú thích:* "Có nguồn" và "Trung thực" chỉ tính các câu hệ thống trả lời, không tính câu từ chối. Độ dài TB tính bằng ký tự. Lần 1–3 giữ nguyên cấu hình hệ thống, chỉ khác judge.

**Bảng 4.** Retrieval trên bộ dev (36 câu trong phạm vi), dense only.

| | hit@4 | hit@6 | hit@8 | hit@10 |
|---|---|---|---|---|
| File đúng | 94% | 94% | 94% | 97% |
| Mục đúng | 92% | 92% | 92% | 94% |

MRR@10 theo mục là **0.907**. Trên bộ test, retrieval hit@4 (file) đạt **27/27**.

**Bảng 5.** Độ chính xác theo loại câu hỏi.

| Loại | Dev (Lần 4) | Test (Lần 5) |
|---|---|---|
| Thường | 17/30 | 13/21 |
| Không dấu | 3/3 | 1/2 |
| Tiếng Anh | 0/1 | 0/1 |
| Trộn Anh-Việt | 1/1 | 1/1 |
| Sai chính tả | 0/1 | 1/1 |
| Ngắn, mơ hồ | 1/1 | 1/1 |
| Nhiều bài | 1/3 | 1/3 |

**Bảng 6.** Phân loại 12 câu sai của bộ test.

| Nhóm | Câu |
|---|---|
| Thiếu 1 ý | q102, q108, q110, q115, q116, q121 |
| Thiếu 2–5 ý | q107, q112, q117, q127 |
| Thiếu ý, có ý sai, không trung thực | q125 |
| Từ chối oan (câu trong phạm vi) | q119 |

Số liệu chi tiết của từng lần chạy có trong [`RESULTS.md`](RESULTS.md).

---

## 6. Thảo luận

### 6.1 Đối chiếu với giả thuyết
- **H1 được ủng hộ:**
  - Section hit@4 đạt 92% trên dev, và MRR = 0.907 cho thấy chunk đúng gần như luôn đứng hạng 1.
  - Tăng `k` từ 4 lên 8 không thêm câu nào đúng, nên giữ `TOP_K = 4`. Kết quả này phù hợp với lập luận về context dài [18].
  - Vì retrieval không phải nút thắt nên không cần làm hybrid BM25 [3] hay rerank.
- **H2 chỉ đúng một phần:**
  - Hệ thống trung thực (25/26 câu trên test, 35/35 trên dev) và luôn có nguồn.
  - Nhưng câu trả lời thường **thiếu ý**: 10/11 câu sai được chấm chỉ thiếu ý, không có ý sai. Ở 8 câu dev được kiểm tra tay, ý bị thiếu đều có trong context.
  - Nghĩa là giới hạn nằm ở model sinh câu trả lời (Flash Lite), không nằm ở tài liệu hay retrieval.

### 6.2 Đối chiếu với tài liệu
- **Thiên lệch của judge:**
  - Trên cùng câu trả lời, judge Gemini cùng model chấm 58%, còn ChatGPT chấm 35%.
  - Điều này phù hợp với hiện tượng judge ưu ái câu trả lời của chính model mình [19][20].
  - Lần 1 (98%) cho thấy judge dễ dãi làm số liệu lạc quan, nên từ Lần 2 hệ thống dùng tiêu chí chặt theo từng ý [21].
- **Bộ test so với dev:**
  - Test đạt 60%, dev đạt 58%.
  - Hai con số gần nhau nên việc tinh chỉnh trên dev không gây overfit đáng kể [22].

### 6.3 Nguồn sai số
- **Sai số ngẫu nhiên:**
  - Gemini Flash Lite bỏ qua `temperature=0`, nên câu trả lời có thể đổi giữa các lần chạy.
  - Mỗi loại câu khó chỉ có 1–3 câu, nên tỉ lệ theo loại dao động mạnh (ví dụ "sai chính tả": 0/1 trên dev, 1/1 trên test).
- **Sai số hệ thống:**
  - Đáp án mẫu do người làm project viết, có câu nhiều ý hơn mức cần thiết, nên judge chặt có xu hướng chấm thiếu ý.
  - Judge vẫn là một LLM: chấm tay qua giao diện web và không được lặp lại nhiều lần.
  - Bộ test do cùng người tinh chỉnh viết, nên vẫn có rò rỉ nhẹ.

### 6.4 Hạn chế
- `MIN_SCORE` không tách được câu ngoài phạm vi: top-1 cosine của câu ngoài phạm vi (0.402–0.533) chồng lên câu trong phạm vi (thấp nhất 0.519). Lớp chặn chính vì vậy là prompt.
- Vẫn còn 1 câu từ chối oan ở mỗi bộ (dev q40, test q119). Ở test có 1 câu không trung thực (q125).
- Câu "nhiều bài" và câu so sánh là nhóm yếu nhất (1/3 ở cả hai bộ).
- Docker Compose mới chạy được Qdrant server. API chưa chạy end-to-end trong Docker vì máy thiếu RAM.

### 6.5 Hướng cải thiện
1. Thay model sinh câu trả lời bằng model mạnh hơn. Đây là thay đổi có khả năng tăng độ chính xác nhiều nhất, vì nút thắt nằm ở bước này.
2. Cho prompt liệt kê đủ ý từ context trước, rồi mới viết câu trả lời.
3. Với câu nhiều bài, có thể tách câu hỏi thành các câu con trước khi truy xuất.
4. Tăng số câu mỗi loại khó, và chấm nhiều lần để giảm sai số ngẫu nhiên.

Mọi cải thiện phải đánh giá lại trên dev, vì bộ test đã được dùng.

---

## 7. Kết luận
- Hệ thống RAG tiếng Việt cho CS106 đã hoạt động đầy đủ qua CLI, web, API và Docker (chỉ mới chạy được phần Qdrant).
- **Đạt:**
  - 100% câu trả lời có nguồn.
  - Từ chối đúng mọi câu ngoài phạm vi.
  - Retrieval lấy đúng mục ở 92% câu dev và đúng file ở 27/27 câu test.
- **Chưa đạt:** độ chính xác trên bộ test là 60%, thấp hơn mục tiêu 80%.
- **Phát hiện chính:**
  - Khi retrieval đã tốt, chất lượng câu trả lời bị giới hạn bởi model sinh câu trả lời: câu trả lời đúng nhưng thiếu ý.
  - Kết quả đánh giá phụ thuộc mạnh vào judge: cùng câu trả lời được chấm 98% hay 35% tuỳ judge và tiêu chí.

---

## 8. Tài liệu tham khảo

Kiểu trích dẫn IEEE. Bản BibTeX đầy đủ: [`references.bib`](references.bib).

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
24. R. A. Day, B. Gastel. *How to Write and Publish a Scientific Paper.* 7th ed. Santa Barbara, CA: Greenwood, 2011. ISBN 978-0-313-39195-8. [libcat.library.tamu.edu](https://libcat.library.tamu.edu/Record/in00002754875)
25. Council of Science Editors. *Scientific Style and Format: The CSE Manual for Authors, Editors, and Publishers.* 8th ed. Chicago: University of Chicago Press, 2014. ISBN 978-0-226-11649-5. [scientificstyleandformat.org](https://www.scientificstyleandformat.org/)

Bố cục báo cáo này theo cấu trúc IMRaD (Introduction – Methods – Results – Discussion) [24][25].

---

## 9. Phụ lục

### A. Hướng dẫn chạy
Chạy trong conda env `rag`. Cần file `.env` có `GOOGLE_API_KEY`, và index đã được build bằng `python -m src.ingest.build_index`.

- **CLI:** `python -m app.cli`
- **Web:** chạy `uvicorn api.main:app`, rồi mở `http://127.0.0.1:8000`.
  - Giao diện gồm khung chat và mục lục bài giảng.
  - Bấm một mục để điền sẵn câu hỏi; mở một bài để tải file `.txt` gốc qua `/files/<tên file>`.
- **API:** chạy `uvicorn api.main:app`, rồi gửi câu hỏi:
  ```bash
  curl -X POST http://127.0.0.1:8000/ask -H "Content-Type: application/json" -d '{"question": "A* là gì?"}'
  ```
  - Trả về `{"answer": "...", "sources": ["file > mục", ...]}`.
  - Mã lỗi: 422 (câu hỏi rỗng hoặc dài quá 1000 ký tự), 429 (Gemini hết quota), 502 (Gemini lỗi), 503 (thiếu index hoặc API key).
- **Test:** `pytest tests/`

Không chạy CLI và API cùng lúc, vì Qdrant local mode chỉ cho một tiến trình mở `data/index`.

**Docker Compose (Qdrant server + API).** Cần Docker Desktop và file `.env`. Lần đầu chạy:
```bash
docker compose up -d qdrant                                        # bật Qdrant server
docker compose run --rm api python -m src.ingest.build_index       # build image, embed tài liệu vào Qdrant (một lần)
docker compose up -d --build                                       # bật API
```
- Sau đó mở `http://127.0.0.1:8000` (web) hoặc `http://localhost:6333/dashboard` (Qdrant).
- Tắt bằng `docker compose down`. Lệnh `docker compose down -v` xoá luôn index và bge-m3 đã tải.
- Docker Compose đặt biến `QDRANT_URL`, nên API dùng Qdrant server. Chạy trực tiếp trên máy, không có biến này, thì vẫn dùng local `data/index`.

### B. Chạy đánh giá
```bash
python -m eval.run_eval                                    # bộ dev
python -m eval.run_eval --resume eval/results/<run>.jsonl --verdicts eval/results/<run>.verdicts.json
python -m eval.retrieval_hits                              # hit@k, MRR (không gọi LLM)
```

### C. Tham số (`src/config.py`)

| Tham số | Giá trị |
|---|---|
| `CHUNK_SIZE` | 1500 ký tự |
| `EMBEDDING_MODEL` | `BAAI/bge-m3` |
| `COLLECTION_NAME` | `cs106` |
| `TOP_K` | 4 |
| `LLM_MODEL` | `gemini-3.5-flash-lite` |
| `GEMINI_RPM` | 10 |
| `MIN_SCORE` | 0.5 |

### D. Số liệu thô và tài liệu bổ sung
- `eval/results/`: câu trả lời từng câu (`.jsonl`), đầu vào cho judge và verdicts của mỗi lần chạy. Bộ test nằm ở `test-20261119-170246.*`.
- [`RESULTS.md`](RESULTS.md): kết quả chi tiết Lần 1–5.
- [`THEORY.md`](THEORY.md): lý thuyết và lý do của từng quyết định.
- [`PROGRESS.md`](PROGRESS.md): nhật ký từng phiên làm việc.

### E. Cấu trúc thư mục
```
OriginalDocs/   tài liệu gốc (chỉ đọc)
data/index/     vector index (sinh tự động)
src/            ingest/, retrieval/, generation/, config.py
app/            cli.py, web/index.html
api/            main.py (FastAPI)
eval/           questions/ (dev), test/ (held-out), results/, run_eval.py, retrieval_hits.py
tests/          unit test
assets/         images/, videos/
```
