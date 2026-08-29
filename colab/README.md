# `colab/` — phòng thí nghiệm TTS chạy trên GPU của Google Colab

Thư mục này là **toàn bộ phần chạy trên GPU**. Nó phục vụ hai việc tách rời nhau:

| | Việc gì | Chạy bằng |
|---|---|---|
| **Benchmark** | So sánh 7 engine trên cùng một bộ câu, sinh bảng số để đưa vào luận văn | `benchmark.py` |
| **Serve** | Mở endpoint HTTP phục vụ **cả ba stage nặng** cho pipeline dưới máy | `server.py` + `stages.py` |

Cả hai dùng chung một registry engine, nên **con số bạn đo được chính là con số
pipeline sẽ nhận**. Đây là lý do hai việc này nằm chung một thư mục thay vì hai
notebook rời.

## Vì sao lại là Colab

**Lý do là dung lượng đĩa, không chỉ tốc độ.** Chạy pipeline dưới máy một lần là
tải về 4.0 GB trọng số:

| Model | Dung lượng | Stage |
|---|---|---|
| nllb-200-distilled-600M | 2.3 GB | dịch |
| faster-whisper-medium | 1.4 GB | ASR |
| viXTTS / F5-TTS | 2.5–3 GB | synthesis (nếu bật) |

Và máy Mac này **không có CUDA**: whisper rơi về CPU int8, kernel của coqui và F5
không có bản Metal. Tức là 4 GB đĩa cho những model chạy chậm. Đẩy hết lên Colab
free tier (T4 16 GB) thì máy dưới **không lưu trọng số nào**.

`server.py` phục vụ cả ba stage:

```
POST /transcribe   whisper trên GPU      -> tiết kiệm 1.4 GB
POST /translate    NLLB / SeamlessM4T    -> tiết kiệm 2.3 GB
POST /synthesize   7 engine TTS          -> tiết kiệm 2.5-3 GB
```

### Đánh đổi phải nói rõ

**Dịch** chỉ gửi text — phụ đề cả phim vài chục KB. Không có đánh đổi gì.

**ASR upload audio.** Đây là thay đổi thật so với thiết kế ban đầu và không nên
giấu: bật `REMOTE_ASR_ENABLED` là dải tiếng nói của video đi lên mạng. Nó được nén
Opus 24 kbps trước (phim 10 phút ≈ **1.7 MB** thay vì 19 MB wav), và **video vẫn
không bao giờ rời máy** — nhưng câu "chỉ có text rời khỏi máy" không còn đúng. Dub
tư liệu không được phép chia sẻ thì để tắt và chấp nhận tải whisper về.

**Không có Colab thì không chạy được.** Trước đây pipeline luôn chạy được local. Giờ
nếu tunnel chết, ASR và dịch rơi về model local — tức là **sẽ tải model về máy** đúng
lần đó. Muốn máy tuyệt đối sạch thì tắt AI service khi Colab chết.

### Còn hai model nhỏ vẫn ở local

`demucs` (~80 MB, tách nhạc nền) và `pyannote` (~31 MB, diarization) vẫn tải về máy
khi chạy job. Cả hai chạy tốt trên MPS và trả về file lớn nếu đưa lên mạng — Demucs
phải upload audio rồi tải về **hai** stem. 111 MB đổi lấy một round-trip nặng là
đánh đổi tệ, nên tạm để nguyên. Nếu bạn muốn máy tuyệt đối không có model nào thì
đây là việc tiếp theo.

## Bảng engine

| id | Clone giọng | Ngôn ngữ | Chạy ở đâu | Vai trò trong so sánh |
|---|---|---|---|---|
| `mms` | không | ~1100 | GPU (0.5 GB) | **Sàn.** Đây là thứ pipeline đang dùng khi fallback |
| `xtts_v2` | có | 17, **không có vi** | GPU (2.5 GB) | Chuẩn tham chiếu cho cloning đa ngôn ngữ |
| `vixtts` | có | vi | GPU (2.5 GB) | Cùng kiến trúc, fine-tune tiếng Việt |
| `f5_vi` | có | vi | GPU (3 GB) | Flow-matching, fine-tune ~1000h ViVoice |
| `f5_base` | có | en, zh | GPU (3 GB) | Cùng kiến trúc F5 trên dữ liệu gốc |
| `piper` | không | 50+ | **CPU** | **Trần tốc độ.** Cho biết chất lượng đắt cỡ nào |
| `edge` | không | 100+ | cloud, free | **Trần chất lượng.** Mức thương mại, nhưng không clone được |

Bộ này được chọn để **trả lời được câu hỏi**, không phải để dài. Bốn cặp đối
chứng nằm sẵn trong đó:

* `xtts_v2` ↔ `vixtts` — fine-tune tiếng Việt đổi được gì trên kiến trúc AR?
* `f5_base` ↔ `f5_vi` — **câu hỏi đó lặp lại trên kiến trúc thứ hai.** Nếu hai
  cặp cùng dịch chuyển một hướng thì kết luận là về fine-tune; nếu ngược nhau
  thì kết luận là về kiến trúc. Một engine đơn lẻ không phân biệt được hai điều
  này — đó là lý do `f5_base` xứng đáng có chỗ dù nó không nói được tiếng Việt.
* `vixtts` ↔ `f5_vi` — AR (XTTS) so với flow-matching (F5) trên cùng dữ liệu.
* mọi engine ↔ `edge` — khoảng cách còn lại tới chất lượng thương mại.

## Chỉ số đo được

`metrics.py` đo ba thứ, mỗi thứ bắt một kiểu hỏng khác nhau:

| Chỉ số | Đo cái gì | Đọc thế nào |
|---|---|---|
| **RTF** | compute / audio | `< 1.0` là nhanh hơn thời gian thực. Phim 10 phút (~6 phút thoại): RTF 0.3 ≈ 2 phút synth |
| **WER / CER** | Whisper nghe lại xem có đúng chữ không | **Càng thấp càng tốt, 0 là hoàn hảo.** Chỉ số quyết định |
| **SECS** | Cosine similarity ECAPA với giọng gốc | `>0.80` chắc chắn cùng người · `0.65–0.80` nhận ra được · `<0.45` không clone |
| **MOS** | UTMOS dự đoán điểm người nghe (1–5) | Chỉ để sàng lọc thô. Model fit trên tiếng Anh, dùng cho tiếng Việt là ngoài miền |

**WER là chỉ số quyết định.** Model card của viXTTS ghi thẳng: *"Subpar
performance for input sentences under 10 words in Vietnamese."* Phụ đề phim thì
đa số dưới 10 từ. Nên bộ câu trong `sentences.py` cố ý xếp từ 1 từ đến 28 từ, và
báo cáo tách WER theo độ dài câu. Một engine đẹp ở câu dài mà rơi ở câu ngắn thì
**không dùng được cho pipeline này**, dù demo có hay đến đâu.

Cả bốn chỉ số **không đo được ngữ điệu**. Chúng thu hẹp danh sách xuống hai ba
engine đáng nghe; chọn ra engine thắng thì vẫn phải nghe.

## Dùng

### 1. Benchmark

```bash
python benchmark.py --language vi --reference giong_mau.wav
```

Kết quả nằm trong `results/`:

```
audio/<engine>/line_00.wav   từng câu, để nghe đối chứng
results.json                 dữ liệu thô, một record mỗi (engine, câu)
results.csv                  bản phẳng cho Excel / pandas
report.md                    bảng đã format, dán thẳng vào luận văn
```

Vài cờ hay dùng:

```bash
python benchmark.py --engines vixtts,f5_vi --language vi --reference ref.wav
python benchmark.py --language vi --reference ref.wav --no-metrics   # chỉ đo tốc độ
python benchmark.py --language en --reference ref.wav                # bộ câu tiếng Anh
```

Engine được nạp **lần lượt và unload giữa các lần**. T4 có 16 GB, mỗi model
cloning ăn 2.5–3 GB; giữ cả bốn cùng lúc là cách biến benchmark thành một
traceback OOM trông giống lỗi model. Thời gian nạp được báo riêng và **không
tính vào RTF** — pipeline thật chỉ trả chi phí đó một lần mỗi phiên.

### 2. Serve

```bash
AUTH_TOKEN=... DEFAULT_ENGINE=vixtts uvicorn server:app --host 0.0.0.0 --port 8000
```

Contract đúng bằng thứ `ai-service/app/services/tts/remote.py` nói:

```
GET  /health      -> catalogue: engine nào có, nói được gì
GET  /engines     -> cùng danh sách, không kèm field của server
POST /synthesize  -> audio/wav
     form: text, language, speed, engine?, speaker_id?, reference_id?
     file: reference?        (chỉ gửi lần đầu mỗi giọng)
     409 {"error":"missing_reference"} -> client tự upload lại
```

**Chọn engine theo từng request.** `engine=vixtts` ở call này và `engine=f5_vi`
ở call sau, nghĩa là A/B trên một job thật chỉ tốn hai lần chạy đổi một biến môi
trường. Engine thực sự phát âm trả về ở header `X-TTS-Model`, nên manifest ghi
lại được từng segment và một job chạy trộn engine vẫn đọc được về sau.

**Mỗi lúc chỉ một model nằm trong VRAM.** Đổi engine thì model cũ bị unload
trước. Trên A100 thì đặt `ALLOW_MULTI_RESIDENT=1` để giữ chúng nóng.

**Cache reference.** Pipeline gọi một (đôi khi hai) lần mỗi segment và mọi câu
của cùng một nhân vật dùng chung một file wav. File lên một lần rồi được địa chỉ
hoá bằng sha1. Runtime restart thì `REF_DIR` rỗng nhưng client vẫn giữ hash —
server trả 409 và client upload lại đúng những reference còn đang dùng.

### 3. Nối vào pipeline dưới máy

Notebook in ra sẵn bốn dòng để dán vào `ai-service/.env`:

```bash
REMOTE_TTS_URL=https://<ngau-nhien>.trycloudflare.com
REMOTE_TTS_TOKEN=<token bạn đặt>
REMOTE_TTS_ENGINES=vixtts,f5_vi        # mỗi id thành một model chọn được
REMOTE_TTS_LANGUAGES=vi
```

Rồi khởi động lại AI service. Router sẽ thấy `remote:vixtts` và `remote:f5_vi`
như hai model riêng biệt. Chúng **xuất hiện ngay trong dropdown "Voice / TTS model"**
trên web UI — chọn engine cho job tiếp theo chỉ là một cú click, không phải sửa file.

Hoặc ép bằng `force_model` khi gọi API:

```bash
curl -X POST localhost:8000/speech/synthesize \
  -H 'content-type: application/json' \
  -d '{"job_id":"<id>","force_model":"remote:f5_vi"}'
```

**URL trycloudflare đổi mỗi lần chạy lại cell.** Khi tunnel chết,
`RemoteAdapter` thử lại 3 lần rồi để router rơi xuống MMS-TTS — job không hỏng,
chỉ là các segment sau đó nghe tệ hơn, và cột `tts_model` trong manifest cho
biết chính xác segment nào rơi vào trường hợp đó.

## Thêm một engine

Ba bước, không file nào khác phải sửa:

1. Viết `engines/tts_<ten>.py`, kế thừa `TTSEngine`, cài `load()` và `speak()`.
   Mọi import nặng đặt **bên trong** hai hàm đó — nhờ vậy một runtime chỉ cài
   dependency của một engine vẫn import được registry và trả lời `/health` cho
   cả bảy.
2. Thêm vào `_CLASSES` trong `engines/__init__.py`.
3. Xong. Benchmark và server tự thấy; client phát hiện qua `GET /health`.

## Giấy phép — đọc trước khi đưa vào luận văn

| Model | Giấy phép | Ý nghĩa |
|---|---|---|
| XTTS-v2 / viXTTS | Coqui Public Model License | **Phi thương mại.** Nghiên cứu, luận văn thì được |
| F5-TTS | CC-BY-NC-4.0 | Phi thương mại, phải ghi nguồn |
| MMS-TTS | CC-BY-NC-4.0 | Phi thương mại |
| Piper | MIT (code), giấy phép theo từng giọng | Đa số giọng cho phép thương mại |
| Edge-TTS | Không có giấy phép công khai | Đây là endpoint nội bộ của trình duyệt Edge. **Chỉ dùng làm mốc tham chiếu**, đừng đưa vào sản phẩm |

Clone giọng của người thật cần sự đồng ý của họ. Với clip phim dùng để thử
nghiệm thì đó là fair use trong phạm vi nghiên cứu; phát hành thì không.
