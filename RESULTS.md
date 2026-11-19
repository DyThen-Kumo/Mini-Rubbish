# RESULTS

Kết quả chấm hệ thống trên bộ câu hỏi `eval/questions/` (40 câu). Mỗi lần chạy thêm một mục mới ở cuối file để so sánh trước/sau khi tinh chỉnh (Task 9).

---

## Lần 1: baseline sau Task 7 (2026-11-18)

**Lệnh:** `python -m eval.run_eval`
**Dữ liệu chi tiết:** `eval/results/20261118-010234.jsonl` (từng câu), `eval/results/20261118-010234.summary.txt`

### Cấu hình

| Tham số | Giá trị |
|---|---|
| LLM (trả lời + judge) | `gemini-3.5-flash-lite` (bỏ qua `temperature=0`) |
| Embedding | `BAAI/bge-m3`, dense only |
| `CHUNK_SIZE` | 1500 ký tự |
| `TOP_K` | 4 |
| `MIN_SCORE` | 0.5 |
| Bộ câu hỏi | 30 câu thường (q01–q30) + 10 câu khó (q31–q40); 4 câu ngoài phạm vi |

### Chỉ số chính

| Chỉ số | Kết quả | Mục tiêu | Đạt? |
|---|---|---|---|
| **Độ chính xác** | **39/40 (98%)** | ≥ 80% | ✅ |
| Nguồn đúng (câu trong phạm vi) | 34/36 (94%) | — | ✅ (xem q38, q40) |
| Câu trả lời có nguồn | 35/35 (100%) | 100% | ✅ |
| Từ chối đúng (câu ngoài phạm vi) | 4/4 (100%) | — | ✅ |
| Từ chối oan (câu trong phạm vi) | 1 câu (q40) | 0 | ❌ |
| Trung thực (judge) | 35/35 (100%) | — | ✅ |
| Retrieval hit@4 | 34/36 (94%) | — | ✅ |
| Độ dài trả lời trung bình | 330 ký tự | — | — |

### Độ chính xác theo loại câu hỏi

| Loại | Kết quả |
|---|---|
| Thường (q01–q30) | 30/30 (100%) |
| Không dấu | 3/3 (100%) |
| Tiếng Anh | 1/1 (100%) |
| Trộn Anh-Việt | 1/1 (100%) |
| Sai chính tả | 1/1 (100%) |
| Ngắn, mơ hồ | 1/1 (100%) |
| **Nhiều bài** | **2/3 (67%)** |

### Câu sai và câu cần chú ý

**q40 (sai, từ chối oan)**: "Expected utility ở bài Introduction liên hệ thế nào với nguyên tắc MEU ở bài Expectimax?"
- Hệ thống: "Không tìm thấy trong tài liệu".
- Đáp án mẫu: Introduction nêu rational agent chọn hành động có expected utility cao nhất; Expectimax hình thức hoá thành EU(a) = Σ P(o|a)·U(o), chọn argmax.
- Nguyên nhân: top-4 không chứa đủ hai mục cần thiết (`retrieval_hit = false`). Chưa rõ bị chặn ở ngưỡng `MIN_SCORE` hay do LLM từ chối vì context thiếu; kiểm tra bằng `python -m app.cli --debug`.

**q38 (đúng, nhưng chấm "nguồn sai")**: "So sánh Minimax và Expectimax"
- Hệ thống trả lời đầy đủ, lấy từ `07_Expectimax_Summary.txt > 23. SO SÁNH MINIMAX, EXPECTIMAX VÀ EXPECTIMINIMAX`.
- Đáp án mẫu yêu cầu cả file 06 và 07, nhưng bài 07 đã có sẵn mục so sánh. Đây là lỗi nhãn của bộ eval (quá chặt), không phải lỗi hệ thống. Có thể sửa `source_file` của q38 thành chỉ file 07.

### Nhận xét

1. **Đạt tiêu chí xong của project** (≥ 80% đúng, mọi câu trả lời đều có nguồn).
2. **Điểm yếu thật: câu cần nhiều bài.** Dense retrieval với k = 4 dễ lấy hết chunk từ một bài. Hướng cho Task 9: tăng k, hybrid BM25, hoặc rerank.
3. **Điểm có thể cao hơn thực tế:**
   - Cùng một model vừa trả lời vừa chấm (self-preference bias), nên judge có thể dễ dãi. Ví dụ q34 bỏ phần α/β và điều kiện cắt vẫn được chấm đúng ("thiếu chi tiết phụ vẫn tính là đúng").
   - Câu khó mới có 10 câu, mỗi loại chỉ 1–3 câu: chưa đủ để kết luận về từng loại.
   - Model bỏ qua `temperature`, nên chạy lại có thể chênh 1–2 câu.
4. **Ngưỡng `MIN_SCORE = 0.5` an toàn:** cả 4 câu ngoài phạm vi đều được từ chối (qua prompt), chỉ 1 câu trong phạm vi bị từ chối oan.

### Cần duyệt tay

- [x] q40: xác nhận là sai (từ chối oan).
- [x] Xem nhanh 3–5 câu được chấm đúng (q34, q39): **judge dễ dãi** → làm lại judge ở Lần 2.

---

## Lần 2: judge chặt (2026-11-18)

**Lệnh:** `python -m eval.run_eval`
**Dữ liệu chi tiết:** `eval/results/20261118-011704.jsonl`, `eval/results/20261118-011704.summary.txt`

**Thay đổi so với Lần 1:** chỉ đổi judge, hệ thống giữ nguyên cấu hình. Judge tách đáp án mẫu thành các ý chính, trả về `missing_points` (ý thiếu) và `wrong_points` (ý sai). Code tính **đúng khi cả hai danh sách đều rỗng**, không để judge tự quyết đúng/sai. Lý do: duyệt tay Lần 1 thấy judge dễ dãi (ví dụ q34 bỏ α/β và điều kiện cắt vẫn được chấm đúng).

### So sánh

| Chỉ số | Lần 1 (judge dễ) | Lần 2 (judge chặt) |
|---|---|---|
| **Độ chính xác** | 39/40 (98%) | **23/40 (58%)** ❌ |
| Câu có ý **sai** (`wrong_points`) | — | **0/35** |
| Nguồn đúng | 34/36 (94%) | 34/36 (94%) |
| Câu trả lời có nguồn | 35/35 (100%) | 35/35 (100%) |
| Từ chối đúng / từ chối oan | 4/4 / 1 | 4/4 / 1 |
| Trung thực | 35/35 (100%) | 35/35 (100%) |
| Retrieval hit@4 | 34/36 (94%) | 34/36 (94%) |
| Độ dài trả lời TB | 330 ký tự | 326 ký tự |

### Độ chính xác theo loại câu hỏi

| Loại | Lần 1 | Lần 2 |
|---|---|---|
| Thường (q01–q30) | 30/30 | 17/30 |
| Không dấu | 3/3 | 3/3 |
| Tiếng Anh | 1/1 | 0/1 |
| Trộn Anh-Việt | 1/1 | 1/1 |
| Sai chính tả | 1/1 | 0/1 |
| Ngắn, mơ hồ | 1/1 | 1/1 |
| Nhiều bài | 2/3 | 1/3 |

### 17 câu sai: thiếu ý gì

Không câu nào có ý sai; tất cả đều sai vì **thiếu ý**, trừ q40 (từ chối oan, như Lần 1).

| Nhóm | Câu | Ý bị thiếu (theo judge) |
|---|---|---|
| Thiếu 1 ý phụ / sắc thái | q01, q02, q03, q08, q10, q19, q24, q27 | Ví dụ q01 thiếu "rationality quan tâm quyết định, không phải quá trình suy nghĩ"; q19 thiếu công thức P(s' \| s, a, history) = P(s' \| s, a); q24 thiếu ε/\|A\| = 0.05 |
| Thiếu 2 ý | q06, q09, q12, q14, q26 | Ví dụ q14 thiếu khởi tạo α = -∞, β = +∞ và "không đổi minimax value"; q26 thiếu "tái sử dụng experience", "mini-batch đa dạng" |
| Trả lời quá ngắn, thiếu phần lớn | q34, q36, q39 | q34 thiếu cả định nghĩa α/β lẫn điều kiện cắt; q36 thiếu miền 0 ≤ γ ≤ 1, γ^2, γ gần 1; q39 thiếu cả hai công thức |
| Từ chối oan | q40 | Retrieval thiếu một trong hai bài |

### Nhận xét

1. **Hệ thống không nói sai, nhưng nói thiếu.** 0/35 câu có `wrong_points`, 35/35 trung thực. Lỗi chính là câu trả lời quá ngắn: prompt yêu cầu "ngắn gọn" nên LLM bỏ công thức, điều kiện, giá trị cụ thể.
2. **Điểm thật nằm giữa 58% và 98%.** Judge chặt chấm kiểu "tất cả hoặc không": thiếu một sắc thái phụ (q01, q02, q03) bị tính sai như thiếu toàn bộ (q34, q39). Có thể nói khoảng 8 câu ở nhóm "thiếu 1 ý phụ" là chấp nhận được với sinh viên ôn tập.
3. **Theo judge chặt thì chưa đạt tiêu chí ≥ 80%.** Từ giờ lấy judge chặt làm thước đo chính cho Task 9, vì nó tách được "sai" và "thiếu", và chỉ ra cụ thể chỗ cần sửa.
4. **Hướng cho Task 9 (theo thứ tự đáng thử):**
   - Sửa prompt: yêu cầu nêu đủ định nghĩa, công thức, điều kiện có trong tài liệu, thay vì "ngắn gọn". Tác động lớn nhất, rẻ nhất.
   - Câu nhiều bài (q39, q40): tăng `TOP_K`, hybrid BM25, hoặc rerank.
   - Thêm chỉ số độ phủ ý (số ý có / tổng số ý) để thấy tiến bộ từng phần, không chỉ đúng/sai.
5. **Giới hạn vẫn còn:** judge và hệ thống cùng một model (Flash Lite); đáp án mẫu do mình viết, có câu nhiều ý hơn mức cần thiết; mỗi loại câu khó chỉ 1–3 câu.

---

## Lần 3–4: judge ChatGPT, sửa prompt (2026-11-19)

**Thay đổi chung:** judge chuyển từ Gemini sang **ChatGPT** (chấm tay qua giao diện web) để tránh self-preference khi judge và hệ thống cùng một model. Cùng tiêu chí chặt của Lần 2 (`missing_points`/`wrong_points`, code quyết định đúng/sai).

**Lệnh:** `python -m eval.run_eval` → dán `<run>.judge_prompt.txt` + upload `<run>.judge_input.json` lên ChatGPT → `python -m eval.run_eval --resume <run>.jsonl --verdicts <run>.verdicts.json`

| Lần | Thay đổi hệ thống | Dữ liệu |
|---|---|---|
| 3 | Không đổi (prompt "ngắn gọn"), chỉ đổi judge | `eval/results/20261119-022701.*` |
| 4 | Prompt mới: bỏ "ngắn gọn", yêu cầu đủ định nghĩa/công thức/điều kiện/ví dụ số, câu so sánh nêu đủ từng vế, không thêm ý ngoài tài liệu, có một phần thông tin thì trả lời phần đó | `eval/results/20261119-024609.*` |

### So sánh

| Chỉ số | Lần 2 (judge Gemini) | Lần 3 (judge ChatGPT) | Lần 4 (prompt mới) |
|---|---|---|---|
| **Độ chính xác** | 23/40 (58%) | 14/40 (35%) | **23/40 (58%)** |
| Nguồn đúng | 34/36 (94%) | 34/36 (94%) | 34/36 (94%) |
| Câu trả lời có nguồn | 35/35 (100%) | 35/35 (100%) | 35/35 (100%) |
| Từ chối đúng / từ chối oan | 4/4 / 1 | 4/4 / 1 | 4/4 / 1 |
| Trung thực | 35/35 (100%) | 35/35 (100%) | 35/35 (100%) |
| Độ dài trả lời TB | 326 ký tự | 338 ký tự | 814 ký tự |

| Loại | Lần 3 | Lần 4 |
|---|---|---|
| Thường (q01–q30) | 9/30 | 17/30 |
| Không dấu | 3/3 | 3/3 |
| Tiếng Anh | 0/1 | 0/1 |
| Trộn Anh-Việt | 1/1 | 1/1 |
| Sai chính tả | 0/1 | 0/1 |
| Ngắn, mơ hồ | 1/1 | 1/1 |
| Nhiều bài | 0/3 | 1/3 |

### Retrieval (`python -m eval.retrieval_hits`, không gọi LLM, nhãn q38 cũ)

| | hit@4 | hit@6 | hit@8 | hit@10 |
|---|---|---|---|---|
| File đúng | 94% | 94% | 94% | 97% |
| Mục đúng | 92% | 92% | 92% | 94% |

- MRR@10 (theo mục): **0.907**.
- Trượt: q38 (nhãn quá chặt), q39 (đúng bài 09 nhưng thiếu mục "PHẦN 2"), q40 (bài 01 ở hạng 9).
- Top-1 score: câu trong phạm vi thấp nhất 0.519 (q36); câu ngoài phạm vi 0.402–0.533. Hai nhóm chồng nhau nên `MIN_SCORE = 0.5` chỉ chặn được q33; từ chối đúng chủ yếu nhờ prompt.

### Nhận xét

1. **ChatGPT chấm chặt hơn Gemini** (35% so với 58% trên cùng câu trả lời), phù hợp giả thuyết self-preference của judge cùng model.
2. **Sửa prompt tăng 35% → 58%**, câu trả lời dài gấp ~2.4 lần, vẫn 0 câu có ý sai và 4/4 từ chối đúng.
3. **Nút thắt là model generate, không phải retrieval.** 16/17 câu sai chỉ thiếu 1–2 ý nhỏ, và ở 8 câu kiểm tra thì ý bị thiếu đều có trong context. Tăng `TOP_K` không giúp (hit@4 = hit@8), nên giữ `TOP_K = 4`; không làm BM25/rerank.
4. **Sau Lần 4 sửa nhãn q38** thành chỉ file 07, mục "2. WORST-CASE VS. AVERAGE-CASE" (bài 07 đã tự so sánh Minimax/Expectimax). Các số ở trên dùng nhãn cũ.
5. Với Gemini Flash Lite, tiêu chí ≥ 80% (judge chặt) nhiều khả năng không đạt; khi báo cáo nên tách retrieval (92% đúng mục) và generation (58%).

---

## Lần 5: held-out test, Task 13 (2026-11-19)

**Chạy một lần duy nhất** trên `eval/test/` (30 câu, q101–q130), cấu hình giữ nguyên như Lần 4 (prompt mới, `TOP_K = 4`, Gemini Flash Lite), judge ChatGPT với tiêu chí chặt. Không tinh chỉnh sau khi xem kết quả.

**Lệnh:** `python -m eval.run_eval --split test` → chấm ChatGPT → `python -m eval.run_eval --resume eval/results/test-20261119-170246.jsonl --verdicts eval/results/test-20261119-170246.verdicts.json`

**Dữ liệu:** `eval/results/test-20261119-170246.*`

### Chỉ số chính

| Chỉ số | Dev (Lần 4) | **Test (Lần 5)** | Tiêu chí |
|---|---|---|---|
| **Độ chính xác** | 23/40 (58%) | **18/30 (60%)** | ≥ 80% ❌ |
| Câu trả lời có nguồn | 35/35 (100%) | **26/26 (100%)** | 100% ✅ |
| Nguồn đúng | 34/36 (94%) | 25/27 (93%) | — |
| Từ chối đúng / từ chối oan | 4/4 / 1 | 3/3 / 1 (q119) | — |
| Trung thực | 35/35 (100%) | 25/26 (96%) | — |
| Retrieval hit@k | — | 27/27 (100%) | — |
| Độ dài trả lời TB | 814 ký tự | 799 ký tự | — |

| Loại | Test |
|---|---|
| Thường | 13/21 (62%) |
| Không dấu | 1/2 |
| Tiếng Anh | 0/1 |
| Trộn Anh-Việt | 1/1 |
| Sai chính tả | 1/1 |
| Ngắn, mơ hồ | 1/1 |
| Nhiều bài | 1/3 |

### 12 câu sai

| Nhóm | Câu | Ghi chú |
|---|---|---|
| Thiếu 1 ý | q102, q108, q110, q115, q116, q121 | ví dụ: thiếu alignment (q102), thiếu cách ước lượng R (q115), thiếu công thức Q tuyến tính (q121) |
| Thiếu 2–5 ý | q107, q112, q117, q127 | q112 (Allais paradox) bỏ toàn bộ ví dụ lottery; q107 thiếu bản chất local search của Min-Conflicts |
| Thiếu ý + sai + không trung thực | q125 | so sánh policy evaluation (MDP) với passive RL: bỏ vế MDP và khẳng định sai là tài liệu không phân biệt |
| Từ chối oan | q119 | câu trong phạm vi nhưng trả lời "không tìm thấy trong tài liệu" |

### Nhận xét

1. **Không đạt tiêu chí độ chính xác (60% < 80%), đạt tiêu chí nguồn (100%).**
2. **Test khớp dev (60% so với 58%)**, nên tinh chỉnh ở Task 9 không overfit bộ dev.
3. **Nút thắt vẫn là generation:** retrieval đạt 27/27, nhưng câu trả lời thiếu ý (10/11 câu sai được chấm chỉ thiếu ý, không có ý sai). Lần đầu xuất hiện một câu không trung thực (q125).
4. Hướng cải thiện (không làm vì đã xem test): model generate mạnh hơn, hoặc prompt yêu cầu liệt kê đủ ý từ context trước khi trả lời; câu "nhiều bài" và câu so sánh là nhóm yếu nhất.
