# Báo Cáo Cá Nhân — Lab 7: Embedding & Vector Store

**Họ tên:** Nguyễn Mai Hoàng Thiện
**Nhóm:** Nhóm 38
**Ngày:** 19/9/2026

> **Nộp 1 bản / sinh viên.** Phần nhóm (lựa chọn tài liệu, thiết kế chiến lược, bộ câu hỏi đánh giá, demo) nộp chung 1 bản trong `REPORT_NHOM.md`. Chi tiết thang điểm: `docs/SCORING.md`.

**Tổng điểm phần cá nhân: 60** = Khởi động (5) + Hướng tiếp cận (10) + Hoàn thiện code (30) + Dự đoán độ tương tự (5) + Kết quả truy xuất của tôi (10).

---

## 1. Khởi động (Warm-up) — Cá nhân (5 điểm)

### Độ tương tự Cosine (Cosine Similarity) (Bài tập 1.1)

**Độ tương tự cosine cao (High cosine similarity) nghĩa là gì?**

> Hai vector đại diện cho hai đoạn văn bản đang trỏ về gần cùng một hướng trong không gian đa chiều, điều này chỉ ra rằng chúng có ý nghĩa/ngữ nghĩa rất giống nhau (điểm càng gần 1 càng tương đồng).

**Ví dụ có độ tương tự CAO:**

- Câu A: Con mèo đang ngủ ngon lành trên ghế sofa.
- Câu B: Chú mèo cưng nhắm mắt nằm nghỉ trên chiếc ghế dài.
- Tại sao tương đồng: Dù dùng từ vựng khác nhau, cả hai câu đều miêu tả cùng một hành động của một con mèo đang ngủ trên ghế.

**Ví dụ có độ tương tự THẤP:**

- Câu A: Công nghệ AI đang phát triển mạnh mẽ.
- Câu B: Bà ngoại tôi rất thích nấu ăn.
- Tại sao khác: Hai câu nói về hai chủ đề hoàn toàn khác biệt (công nghệ so với gia đình/nấu ăn), không có điểm chung về ngữ nghĩa.

**Tại sao độ tương tự cosine (cosine similarity) được ưu tiên hơn khoảng cách Euclid (Euclidean distance) cho text embeddings?**

> Khoảng cách Euclidean bị ảnh hưởng lớn bởi độ dài của văn bản (magnitude của vector), trong khi cosine similarity chỉ quan tâm đến góc (hướng/ngữ nghĩa). Do đó, cosine đánh giá chính xác độ tương đồng ngữ nghĩa dù một câu ngắn và một câu dài.

### Bài toán tính toán Chunking (Bài tập 1.2)

**Tài liệu 10,000 ký tự, chunk\_size=500, overlap=50. Bao nhiêu chunks?**

> *Trình bày phép tính:* Bước nhảy (stride) = chunk\_size - overlap = 500 - 50 = 450. Số chunk = ceil((10000 - 500) / 450) + 1 = ceil(21.11) + 1 = 22 + 1 = 23 chunks.
> *Đáp án:* 23 chunks.

**Nếu độ chồng chéo (overlap) tăng lên 100, số lượng chunk thay đổi thế nào? Tại sao muốn độ chồng chéo nhiều hơn?**

> Bước nhảy giảm xuống còn 400, số chunk sẽ tăng lên thành ceil((10000 - 500) / 400) + 1 = 25 chunks. Việc tăng độ chồng chéo giúp đảm bảo ngữ cảnh ở phần giao giữa hai chunk không bị cắt đứt đột ngột, giữ trọn vẹn ý nghĩa cho các câu vắt ngang.

---

## 2. Hướng tiếp cận của tôi (My Approach) — Cá nhân (10 điểm)

Giải thích cách tiếp cận của bạn khi lập trình (implement) các phần chính trong gói `src`.

### Các hàm chia nhỏ (Chunking Functions)

**SentenceChunker.chunk** — hướng tiếp cận:

> Sử dụng biểu thức chính quy `re.split(r'(?<=[.!?])\s+|\n+', text)` để cắt câu an toàn (giữ lại dấu câu ở cuối). Ngoại lệ như các câu rỗng hay chuỗi khoảng trắng được filter loại bỏ (bằng `s.strip()`) trước khi gom lại thành các chunk không vượt quá `max_sentences_per_chunk`.

**RecursiveChunker.chunk / \_split** — hướng tiếp cận:

> Dùng đệ quy cắt chuỗi. Base case là đoạn văn bản <= `chunk_size` thì trả về. Nếu đoạn vẫn lớn hơn, thử chia bằng ký tự phân cách hiện tại, đoạn nào còn vượt quá kích thước sẽ tiếp tục đệ quy `_split` với dấu phân cách nhỏ hơn tiếp theo, sau đó gộp các phần tử lại một cách tối ưu nhất sát mức max size.

### Lớp EmbeddingStore

**add\_documents + search** — hướng tiếp cận:

> Lưu trữ bằng danh sách list of dicts (in-memory) hoặc ChromaDB (nếu có thư viện). Với `search` (in-memory), chạy vòng lặp tính độ tương tự cosine (`_dot()`) giữa vector câu hỏi với embedding của từng tài liệu, sắp xếp giảm dần và lấy `top_k`.

**search\_with\_filter + delete\_document** — hướng tiếp cận:

> Lọc (filter) danh sách TRƯỚC theo điều kiện metadata khớp hoàn toàn, sau đó mới duyệt mảng đã lọc để tính cosine tìm kiếm top\_k. Để xóa, dùng list comprehension tạo danh sách mới loại bỏ những record có `id` khớp với `doc_id`.

### Tác tử KnowledgeBaseAgent

**answer** — hướng tiếp cận:

> Gọi phương thức `search` từ store để lấy top chunks phù hợp. Duyệt mảng kết quả lấy `hit['content']` và nối chuỗi tạo thành một Context Block rõ ràng (có chú thích Rank và Score), đưa vào template Prompt cùng với câu hỏi, rồi đẩy vào hàm `_llm_fn` để lấy câu trả lời cuối cùng.

---

## 3. Hoàn thiện code (Core Implementation) — Cá nhân (30 điểm)

Vượt qua bộ kiểm thử là điều kiện tính điểm phần này.

### Kết Quả Kiểm Thử (Test Results)

```text
============================= test session starts =============================
platform win32 -- Python 3.10.11, pytest-9.1.1, pluggy-1.6.0
rootdir: E:\dev.nmht.ai\K4-DAY07-NguyenMaiHoangThien-2A202602912
plugins: anyio-4.13.0
collecting ... collected 42 items

tests/test_solution.py::TestProjectStructure::test_root_main_entrypoint_exists PASSED [  2%]
... (40 tests omitted for brevity) ...
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_true_for_existing_doc PASSED [100%]

============================= 42 passed in 0.46s ==============================
```

**Số lượng bài test vượt qua (pass):** 42 / 42

---

## 4. Dự đoán độ tương tự (Similarity Predictions) — Cá nhân (5 điểm)

| Cặp | Câu A | Câu B | Dự đoán    | Điểm thực tế | Đúng? |
| --- | ----- | ----- | ---------- | ------------ | ----- |
| 1   |       |       | cao / thấp |              |       |
| 2   |       |       | cao / thấp |              |       |
| 3   |       |       | cao / thấp |              |       |
| 4   |       |       | cao / thấp |              |       |
| 5   |       |       | cao / thấp |              |       |

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn ý nghĩa?**

> *Viết 2-3 câu:*

---

## 5. Kết quả truy xuất của tôi (Competition Results) — Cá nhân (10 điểm)

Chạy **5 câu hỏi đánh giá của nhóm** trên mã nguồn cá nhân của bạn trong gói `src`. **5 câu hỏi này phải trùng với các thành viên cùng nhóm** (xem `REPORT_NHOM.md`).

| # | Câu hỏi (Query) | Top-1 Chunk truy xuất được (tóm tắt) | Điểm Score | Có liên quan không? (Relevant) | Câu trả lời của Agent (tóm tắt) |
| - | --------------- | ------------------------------------ | ---------- | ------------------------------ | ------------------------------- |
| 1 |                 |                                      |            |                                |                                 |
| 2 |                 |                                      |            |                                |                                 |
| 3 |                 |                                      |            |                                |                                 |
| 4 |                 |                                      |            |                                |                                 |
| 5 |                 |                                      |            |                                |                                 |

**Bao nhiêu câu hỏi trả về chunk có liên quan trong top-3?** \_\_ / 5

**Điều hay nhất tôi học được từ thành viên khác / nhóm khác (qua demo):**

> *Viết 2-3 câu:*

---

## Tự Đánh Giá (Phần Cá Nhân)

| Tiêu chí                                        | Điểm tự đánh giá |
| ----------------------------------------------- | ---------------- |
| Khởi động (Warm-up)                             | / 5              |
| Hướng tiếp cận của tôi (My Approach)            | / 10             |
| Hoàn thiện code (Core Implementation — tests)   | / 30             |
| Dự đoán độ tương tự (Similarity Predictions)    | / 5              |
| Kết quả truy xuất của tôi (Competition Results) | / 10             |
| **Tổng phần cá nhân**                           | **/ 60**         |
