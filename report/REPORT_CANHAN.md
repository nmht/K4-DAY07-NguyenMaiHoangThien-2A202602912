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

| Cặp | Câu A                                                 | Câu B                                                     | Dự đoán | Điểm thực tế | Đúng? |
| --- | ----------------------------------------------------- | --------------------------------------------------------- | ------- | ------------ | ----- |
| 1   | Con mèo đang ngủ ngon lành trên ghế sofa.             | Chú mèo cưng nhắm mắt nằm nghỉ trên chiếc ghế dài.        | cao     | 0.302        | Khá   |
| 2   | Công nghệ AI đang phát triển mạnh mẽ.                 | Bà ngoại tôi rất thích nấu ăn.                            | thấp    | 0.000        | Đúng  |
| 3   | Đại học Bách Khoa Hà Nội là trường kỹ thuật hàng đầu. | HUST là một trong những trường đại học kỹ thuật tốt nhất. | cao     | 0.500        | Đúng  |
| 4   | Tôi muốn đăng ký vào ký túc xá năm nay.               | Làm sao để xin một suất ở khu nội trú sinh viên?          | cao     | 0.000        | Sai   |
| 5   | Hôm nay trời mưa to quá.                              | Lịch học ngày mai bắt đầu từ 7h sáng.                     | thấp    | 0.136        | Đúng  |

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn ý nghĩa?**

> Bất ngờ nhất là cặp số 4 có điểm tương đồng bằng 0.0 mặc dù ý nghĩa của hai câu gần như tương đương nhau. Nguyên nhân là do LexicalHashEmbedder trong bài test (dựa trên tần suất từ vựng) không hiểu được ngữ nghĩa (semantic) mà chỉ so khớp mặt chữ. Từ đó cho thấy các mô hình nhúng (embedding models) thực thụ (như OpenAI/Gemini) rất cần thiết vì chúng ánh xạ ý nghĩa câu vào vector thay vì chỉ mã hóa từng từ riêng lẻ.

---

## 5. Kết quả truy xuất của tôi (Competition Results) — Cá nhân (10 điểm)

Chạy **5 câu hỏi đánh giá của nhóm** trên mã nguồn cá nhân của bạn trong gói `src`. **5 câu hỏi này phải trùng với các thành viên cùng nhóm** (xem `REPORT_NHOM.md`).

| # | Câu hỏi (Query)                                                          | Top-1 Chunk truy xuất được (tóm tắt)                                                                         | Điểm Score | Có liên quan không? (Relevant)                                      | Câu trả lời của Agent (tóm tắt)                                                                                                             |
| - | ------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------ | ---------- | ------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| 1 | Ký túc xá Đại học Bách khoa Hà Nội có bao nhiêu dãy nhà...               | (0.503) hust-dormitory-overview chunk=0: Ký túc xá sinh viên Bách Khoa được giới thiệu...                    | 0.503      | Có (Một phần, câu trả lời chính xác nằm ở chunk=1)                  | Dựa vào ngữ cảnh (chunk 1), Ký túc xá Bách Khoa có 10 dãy nhà, 435 phòng ở và đón khoảng 4.200 sinh viên.                                   |
| 2 | Cơ sở vật chất và lệ phí nhà X1, nhà X2 cho tân sinh viên K71...         | (0.346) hust-dormitory-overview chunk=1: Cơ sở vật chất - Ký túc xá gồm 10 dãy nhà...                        | 0.346      | Không (Top 1 là của HUST, nhưng Top 2 chứa thông tin đúng của HUCE) | Theo thông tin từ HUCE, Nhà X1: 2.700.000 đồng/kỳ; Nhà X2: 3.950.000 đồng/kỳ (đã bao gồm tiền cọc).                                         |
| 3 | PTIT bố trí bao nhiêu chỗ ở tại KTX B1, B2 và cơ sở Ngọc Trục...         | (0.430) dorm-slot chunk=0: Bố trí chỗ ở nội trú cho sinh viên khóa 2025 PTIT tại Hà Nội...                   | 0.430      | Có (Thông tin chi tiết ở chunk=1)                                   | PTIT bố trí KTX B1 có 40 chỗ, KTX B2 có 460 chỗ, và cơ sở Ngọc Trục có 340 chỗ.                                                             |
| 4 | Sinh viên ĐH Thương mại đăng ký ở KTX cơ sở Hà Nội theo quy trình nào... | (0.706) tmu-dormitory-registration-hanoi chunk=0: Cách thức đăng ký ở Ký túc xá cơ sở Hà Nội...              | 0.706      | Có                                                                  | Sinh viên truy cập biểu mẫu Google Forms từ 25/08 đến 30/08/2023, xem ưu tiên và đăng ký. Sinh viên đủ điều kiện sẽ nhận tin nhắn xác nhận. |
| 5 | Mức giá điện nước tại khu nội trú được ban hành ngày nào...              | (0.546) tmu-dormitory-electric-water-fees chunk=0: Điều chỉnh mức giá điện nước tại Khu nội trú sinh viên... | 0.546      | Có                                                                  | Văn bản điều chỉnh được ban hành ngày 05/08/2024, đính kèm file dieu-chinh-gia-dien-nuoc-kntpdf-1727328575.pdf.                             |

**Bao nhiêu câu hỏi trả về chunk có liên quan trong top-3?** 5 / 5

**Điều hay nhất tôi học được từ thành viên khác / nhóm khác (qua demo):**

> Mình nhận ra rằng kích thước chunk (chunk\_size) ảnh hưởng rất lớn đến độ nhiễu khi search. Nếu chunk quá dài, nó sẽ bao hàm quá nhiều thông tin không cần thiết khiến thuật toán tìm kiếm bị loãng (chẳng hạn như việc chunk của HUST lại lọt top ở câu hỏi về HUCE do có trùng nhiều từ vựng chung). Đồng thời, việc gán thêm metadata filter giúp loại bỏ hẳn các document không liên quan từ sớm, cải thiện độ chính xác rõ rệt.

---

## Tự Đánh Giá (Phần Cá Nhân)

| Tiêu chí                                        | Điểm tự đánh giá |
| ----------------------------------------------- | ---------------- |
| Khởi động (Warm-up)                             | 5 / 5              |
| Hướng tiếp cận của tôi (My Approach)            | 10 / 10             |
| Hoàn thiện code (Core Implementation — tests)   | 30 / 30             |
| Dự đoán độ tương tự (Similarity Predictions)    | 5 / 5              |
| Kết quả truy xuất của tôi (Competition Results) | 10 / 10             |
| **Tổng phần cá nhân**                           | **60 / 60**         |
