# Hệ thống Dịch và Lồng tiếng Video Đa ngôn ngữ

Pipeline lồng tiếng tự động, **100% model mã nguồn mở, chạy local, không dùng API trả phí**
(không ElevenLabs, không OpenAI TTS, không SaaS).

* **n8n Community Edition** (self-host, Docker) làm **workflow orchestrator** — GUI kéo-thả,
  execution history, retry, branch, webhook.
* **FastAPI AI service** chạy toàn bộ model nặng. n8n chỉ gọi HTTP.
* **MinIO** giữ toàn bộ media. Giữa các node n8n chỉ truyền `job_id` + object key —
  video 500 MB không bao giờ đi xuyên workflow.
* **PostgreSQL** vừa là DB của n8n vừa là job registry để hiển thị tiến độ từng stage.
* **React UI** để upload, xem/sửa bản dịch, preview và tải kết quả.

```text
Frontend / Form / Manual Trigger
              │
              ▼
            n8n  ──── Workflow Orchestrator (GUI)
              │  HTTP (job_id + object keys)
              ▼
      FastAPI AI Service
              │
   ┌──────┬───┴────┬──────────────┬─────────────────┐
 FFmpeg  Demucs  faster-whisper  pyannote      HuggingFace / PyTorch
                                               (NLLB, SeamlessM4T,
                                                XTTS-v2, MMS-TTS)
              │
              ▼
         MinIO / Local Storage
              │
              ▼
       Final Dubbed Video
```

---

## Bản đồ tài liệu

| File | Trả lời câu hỏi |
|---|---|
| `README.md` (file này) | Hệ thống là gì, kiến trúc, cách dựng lần đầu |
| `COMMANDS.md` | Lệnh cần gõ, theo thứ tự — mở file này khi đang làm việc |
| `docs/architecture.md` | Vì sao chia tầng như vậy, ranh giới network, ràng buộc phiên bản |
| `docs/pipeline.md` | Chi tiết 13 stage: input/output, thuật toán đồng bộ |
| `docs/quality.md` | Đọc và cải thiện chất lượng bản lồng tiếng |
| `docs/troubleshooting.md` | Gặp lỗi thì tra ở đây trước |
| `colab/README.md` | **So sánh 7 engine TTS trên GPU Colab, và nối nó vào pipeline** |
| `n8n/README.md` | Import workflow, ý nghĩa từng node |
| `docs/ui-redesign-prompt.md` | Prompt dùng một lần để redesign web UI (không phải tài liệu hệ thống) |

Quy ước khi sửa code: `make check` phải xanh trước khi commit — nó chạy ruff, 145 unit
test, và một lần verify end-to-end trên clip thật.

---

## 1. Chạy toàn bộ hệ thống

Mặc định là **hybrid**: hạ tầng chạy Docker, AI service chạy native.

```bash
./scripts/setup-native.sh     # 1 lần: tạo .venv + cài torch (MPS) và các model lib
./scripts/start.sh            # dựng n8n + Postgres + MinIO + web UI, import workflow
./scripts/run-native.sh       # chạy AI service (terminal thứ hai)
```

### Vì sao AI service không chạy trong Docker

Docker Desktop trên macOS chạy trong VM Linux và **không có Metal passthrough**. Container
không thấy GPU của máy, nên toàn bộ Demucs / XTTS / NLLB / pyannote bị kẹt trên CPU. Chạy
native thì torch dùng được backend `mps`:

| Stage | Docker (CPU) | Native (MPS) |
|---|---|---|
| Demucs, clip 1 phút | 2–5 phút | ~40 giây |
| XTTS, 1 câu | 3–10 giây | ~1.5 giây |
| NLLB, 30 câu | ~60 giây | ~15 giây |
| Whisper medium | 1–3 phút | 1–3 phút (\*) |

(\*) faster-whisper chạy trên CTranslate2 — thư viện này **không có backend Metal**, nên
ASR luôn ở CPU. `resolve_device_for("asr")` tự map `mps → cpu` thay vì để nó crash lúc
load model.

Kiểm tra sau khi chạy: `make device` hoặc <http://localhost:47800/health/device>.

### Vẫn muốn mọi thứ trong Docker

Vẫn giữ nguyên đường chạy cũ (hữu ích khi nộp bài hoặc chạy trên Linux/GPU):

```bash
./scripts/start.sh --docker-ai          # tất cả trong Docker (CPU trên macOS)
./scripts/start.sh --docker-ai --gpu    # Linux + NVIDIA GPU
```

`ai-service` nằm trong profile `docker-ai`, nên `docker compose up -d` thường chỉ dựng
hạ tầng. Script `start.sh` tự sửa `AI_SERVICE_BASE_URL` trong `.env` cho khớp chế độ
(`host.docker.internal:47800` hay `ai-service:8000`).

Sau khi các container `healthy`:

| Dịch vụ | URL | Chạy ở đâu |
|---|---|---|
| Web UI | <http://localhost:47300> | Docker |
| n8n | <http://localhost:47678> | Docker |
| FastAPI (Swagger) | <http://localhost:47800/docs> | Native (hoặc Docker nếu bật profile) |
| MinIO Console | <http://localhost:47901> (`minioadmin` / `minioadmin123`) | Docker |

> Port nằm trong dải **473xx–479xx** thay vì 3000/5432/5678/8000/9000, để không đụng các
> app khác trên máy. Đổi được bằng `*_PORT` trong `.env` (`FRONTEND_PORT`, `N8N_PORT`,
> `AI_SERVICE_PORT`, `MINIO_API_PORT`, `MINIO_CONSOLE_PORT`, `POSTGRES_PORT`) rồi
> `docker compose up -d`. Port bên trong container giữ nguyên mặc định.

Import workflow vào n8n:

```bash
make import-workflow
# docker compose exec n8n n8n import:workflow --input=/workflows/multilingual-dubbing-pipeline.json
```

> **Các pin không được tự ý nâng:** `demucs==4.0.1` (4.1.x kéo theo `sphn`, không có
> wheel arm64) và `torch/torchaudio 2.11.0 + torchcodec 0.11.0` (bộ đầu tiên có wheel
> arm64 mà `pyannote.audio 4.x` chấp nhận). Chi tiết trong `docs/architecture.md`.

> **Lần build đầu mất khá lâu** (torch + transformers + demucs + coqui-tts ≈ 6–8 GB).
> Lần chạy đầu của mỗi model còn tải weights về volume `model-cache`, nên hãy chạy
> `make test-video` một lần để "làm nóng" trước khi demo.

### Bật speaker diarization

Dùng **`pyannote/speaker-diarization-community-1`** (pyannote.audio 4.x) — chỉ cần
accept **một** repo, vì community-1 là pipeline hoàn chỉnh (segmentation + embedding +
PLDA + VBx clustering nằm sẵn trong checkpoint), không còn tải `segmentation-3.0` riêng.

1. Tạo token tại <https://hf.co/settings/tokens>
2. Bấm *Agree* ở <https://hf.co/pyannote/speaker-diarization-community-1>
3. Điền `HUGGINGFACE_TOKEN=hf_...` vào `.env`, rồi `docker compose restart ai-service`

Vì sao chọn community-1 thay cho `speaker-diarization-3.1` (benchmark chính thức
pyannote, cập nhật 09/2025 — DER càng thấp càng tốt):

| Dataset | `3.1` legacy | **`community-1`** | `precision-2` (trả phí) |
|---|---:|---:|---:|
| AISHELL-4 | 12.2 | **11.7** | 11.4 |
| AliMeeting | 24.5 | **20.3** | 15.2 |
| AMI (IHM) | 18.8 | **17.0** | 12.9 |
| AMI (SDM) | 22.7 | **19.9** | 15.6 |
| DIHARD 3 | 21.4 | **20.2** | 14.7 |

`precision-2` mạnh hơn nhưng là dịch vụ cloud trả phí của pyannoteAI → loại, vì đề bài
yêu cầu free/local. Muốn quay lại model cũ: đổi `DIARIZATION_MODEL` trong `.env`, code
hỗ trợ cả hai output format.

Không có token hệ thống **vẫn chạy**: nó tự degrade về giả định 1 speaker
(stage hiện `skipped`/`enabled: false` chứ không fail cả job).

---

## 2. Demo nhanh

Repo đã có sẵn clip cắt từ `data/test/The-Avengers.mp4`:

| File | Độ dài | Dùng khi |
|---|---|---|
| `data/samples/avengers_20s.mp4` | 20 s, 480p | Chạy thử lần đầu — nhanh nhất |
| `data/samples/avengers_60s.mp4` | 60 s, 720p | Demo với giáo viên |
| `data/test/The-Avengers.mp4` | 5 phút 19 s, 1080p | Chỉ chạy khi có GPU |

Cắt lại đoạn khác: `./scripts/make-clip.sh data/test/The-Avengers.mp4 150 45`

```bash
make verify                                                # pipeline thật, model giả (~20s)
./scripts/smoke-test.sh data/samples/avengers_20s.mp4 vi   # end-to-end với model thật
make test-video                                            # clip tổng hợp 2 giọng
```

`make verify` chạy toàn bộ ffmpeg + logic đồng bộ trên clip 60 giây thật và chỉ giả lập
ba lời gọi model (ASR / dịch / TTS). Nó kiểm tra 40 điều kiện trên **file đầu ra thật** —
kể cả việc đo lại từ chính audio rằng mỗi câu nằm đúng timestamp gốc và khoảng trống
giữa các câu không có tiếng lồng. Chạy trước khi tốn hàng chục phút cho model thật.

> **Đừng chạy nguyên file 5 phút trên CPU cho lần đầu.** Demucs + Whisper medium +
> XTTS trên 5 phút audio mất khoảng 1–2 giờ trên Apple Silicon. Clip 20 giây mất
> ~5–10 phút (đã tính cả thời gian tải model lần đầu).

Hoặc demo bằng GUI:

1. Mở <http://localhost:47300>, kéo `data/samples/avengers_20s.mp4` vào, chọn ngôn ngữ
   đích → **Upload & run pipeline** (UI gọi `POST /webhook/dubbing/start` của n8n).
   *Workflow trong n8n phải đang **Active** thì URL `/webhook/` mới tồn tại.*
2. Mở <http://localhost:47678> → **Executions** để giáo viên nhìn workflow chạy từng node.
3. Quay lại UI: tab **transcript** để sửa bản dịch, tab **result** để preview và tải về.

Muốn demo hoàn toàn trong n8n: mở node **Form: Upload Video**, copy *Form URL*, upload ở đó.

---

## 3. Cấu trúc thư mục

```text
.
├── docker-compose.yml            # n8n · ai-service · postgres · minio · frontend
├── docker-compose.gpu.yml        # override cho NVIDIA GPU
├── .env.example                  # toàn bộ tham số (không hard-code ở đâu khác)
├── Makefile
│
├── ai-service/
│   ├── Dockerfile                # 2 target: cpu | gpu
│   ├── requirements*.txt
│   └── app/
│       ├── main.py               # FastAPI app, error handler, request-id
│       ├── core/                 # config · logging · errors · device · languages
│       ├── storage/              # abstraction: MinIO | local, layout object key
│       ├── jobs/                 # SQLAlchemy models + stage tracking
│       ├── schemas/              # pydantic contracts (chính là body n8n gửi)
│       ├── services/
│       │   ├── ffmpeg.py         # wrapper duy nhất gọi ffmpeg/ffprobe
│       │   ├── media.py          # extract audio
│       │   ├── separation.py     # Demucs
│       │   ├── asr.py            # faster-whisper
│       │   ├── segmentation.py   # cắt window của Whisper thành utterance (thuần data)
│       │   ├── diarization.py    # pyannote.audio
│       │   ├── alignment.py      # merge transcript + speaker + voice reference
│       │   ├── translation/      # base · nllb · seamless · router · duration_aware
│       │   ├── tts/              # base · xtts · mms · f5 · router · service
│       │   ├── sync.py           # time-stretch + đặt audio lên timeline gốc
│       │   ├── mixing.py         # trộn với background stem
│       │   ├── subtitles.py      # .srt / .vtt
│       │   └── render.py         # mux FFmpeg
│       └── api/routes/           # health · jobs · media · speech · translation · audio · subtitle · video
│
├── colab/                        # MỌI MODEL NẶNG CHẠY Ở ĐÂY - xem colab/README.md
│   ├── engines/                  # base · mms · xtts · f5 · piper · edge (registry tự khám phá)
│   ├── stages.py                 # whisper (ASR) + NLLB (dịch) - 3.7 GB không nằm ở máy
│   ├── benchmark.py              # 7 engine × cùng bộ câu -> results/report.md
│   ├── metrics.py                # RTF · WER/CER · speaker similarity · MOS
│   ├── sentences.py              # bộ câu cố định, xếp từ 1 từ đến 28 từ
│   ├── server.py                 # endpoint multi-engine cho RemoteAdapter gọi vào
│   └── tts_lab.ipynb             # notebook điều khiển, không chứa logic
│
├── frontend/                     # React + Vite, nginx proxy /api và /webhook
├── n8n/workflows/                # workflow JSON import lại được
├── infra/                        # init script cho postgres và minio
├── scripts/                      # start · run-native · make-clip · smoke-test · reset
├── docs/                         # architecture · pipeline · quality · troubleshooting
│
└── data/                         # KHÔNG commit (.gitignore), toàn bộ là runtime
    ├── test/                     # video nguồn dài, tự bỏ vào
    ├── samples/                  # clip demo do ./scripts/make-clip.sh cắt ra
    ├── output/                   # kết quả smoke-test.sh tải về, xem được ngay
    ├── verify/                   # output GIẢ của verify_pipeline.py (sine tone)
    └── jobs/_scratch/            # scratch từng job — dọn bằng ./scripts/reset.sh
```

> `data/verify/SYNTHETIC_no_real_voice.mp4` **không phải bản lồng tiếng.** Nó do
> `make verify` sinh ra với ASR/dịch/TTS bị thay bằng stub (giọng = sine 180 Hz,
> "bản dịch" = tiếng Anh cộng từ đệm). Nó chỉ chứng minh phần ffmpeg + đồng bộ +
> trộn tiếng là đúng. Bản dub thật nằm ở `data/output/<job_id>/`.

---

## 4. Pipeline

Workflow n8n được chia thành 6 nhóm (mỗi nhóm một sticky note trên canvas):

| Nhóm | Node | Endpoint |
|---|---|---|
| 1. Input & media preprocessing | Create Job → Upload Video → Extract Audio → Separate | `/jobs`, `/media/extract-audio`, `/media/separate` |
| 2. Speech understanding | Speech-to-Text → Diarization (community-1) → Merge | `/speech/transcribe`, `/speech/diarize`, `/speech/merge` |
| 3. Multilingual translation | Translate → *(lỗi)* Fallback SeamlessM4T | `/translation/translate` |
| 4. Speech generation | Generate Speech → IF duration | `/speech/synthesize` |
| 5. Temporal synchronization | Adapt → Re-generate ⟲ · Place on timeline | `/translation/adapt`, `/audio/synchronize` |
| 6. Mixing & rendering | Mix → Subtitle → Render | `/audio/mix`, `/subtitle/generate`, `/video/render` |

### Đồng bộ thời gian — phần quan trọng nhất

Mỗi câu giữ nguyên `start`, `end`, `target_duration` của bản gốc. Sau khi sinh tiếng:

```text
ratio = generated_duration / original_duration

0.90 ≤ ratio ≤ 1.10        → accept
0.80–0.90 hoặc 1.10–1.20   → time-stretch nhẹ (ffmpeg atempo, không đổi cao độ)
ratio < 0.80 hoặc > 1.20   → viết lại bản dịch → sinh tiếng lại
```

Vòng lặp viết lại có trần (`SYNC_MAX_RETRANSLATE_ATTEMPTS`, mặc định 2); hết lượt thì
segment bị hạ xuống `stretch` nên vòng lặp **luôn kết thúc**.

Hệ thống **không** nối các file TTS lại với nhau. Nó tạo một "audio canvas" im lặng dài
đúng bằng video, rồi **overlay từng segment tại `position = segment.start`**. Khoảng trống
giữa các câu vẫn là background gốc.

### Giữ nhạc nền

```text
Original Audio → Demucs → speech.wav + background.wav
                              │            │
                            ASR/dịch/TTS    │
                              │            │
                          Dubbed Voice ────┴── Mixing → Final Audio
```

Soundtrack gốc (nhạc, tiếng động, ambience) **không bị xoá** — chỉ giọng nói bị thay.

### Multi-speaker

Diarization tách từng speaker, `alignment.py` cắt một đoạn voice reference riêng cho mỗi
người (`speakers/speaker_01.wav`), rồi TTS router clone đúng giọng tương ứng nếu model hỗ
trợ ngôn ngữ đó.

**Exclusive diarization.** Timestamp của Whisper và của diarization không bao giờ khớp
tuyệt đối:

```text
Whisper       10.20 → 12.70   "We need to solve this."
Diarization   10.15 → 11.95   SPEAKER_00
              11.90 → 12.78   SPEAKER_01   ← chồng lấn, gán ai?
```

`community-1` trả thêm `output.exclusive_speaker_diarization`, trong đó **mỗi thời điểm
chỉ thuộc về đúng một speaker**. `alignment.py` ưu tiên dùng bản này
(`DIARIZATION_USE_EXCLUSIVE=true`) nên việc quy từng từ về đúng người không còn mơ hồ.
Trường `turn_source` trong `segments.json` cho biết đã dùng `exclusive` hay `regular`.

### Router / fallback của TTS

```python
if language_supported_by_voice_clone and voice_reference_available:
    XTTS-v2          # 17 ngôn ngữ, clone giọng gốc
else:
    MMS-TTS          # facebook/mms-tts-<iso3>, ~1100 ngôn ngữ
```

`F5-TTS` có sẵn adapter, bật bằng `F5_TTS_ENABLED=true`.

#### Mọi model nặng chạy trên Colab — máy này không lưu trọng số nào

Chạy pipeline hoàn toàn dưới máy là tải về **4.0 GB** trọng số, cho những model mà
máy Mac này chạy chậm (không có CUDA: whisper rơi về CPU int8, kernel coqui/F5
không có bản Metal). Nên cả ba stage nặng đi lên **một** endpoint Colab:

| Stage | Model | Nếu chạy local | Gửi cái gì lên |
|---|---|---|---|
| ASR | whisper-medium | 1.4 GB | **audio** (Opus ~24 kbps, phim 10 phút ≈ 1.7 MB) |
| Dịch | nllb-200-distilled-600M | 2.3 GB | text (vài chục KB) |
| TTS | viXTTS / F5-TTS | 2.5–3 GB | text + vài giây audio tham chiếu |

```bash
REMOTE_URL=https://<ngau-nhien>.trycloudflare.com
REMOTE_TOKEN=<token>
REMOTE_ASR_ENABLED=true
REMOTE_TRANSLATION_ENABLED=true
REMOTE_TTS_ENGINES=vixtts,f5_vi
```

Kiểm tra máy có sạch không:

```bash
curl -s localhost:47800/health/models | python3 -m json.tool
```

`local_model_cache` phải là `0.00 GB`. Con số đó tồn tại vì fallback về model local
tải hàng GB một cách âm thầm — không có gì khác báo cho bạn biết.

**Ba điều phải nói rõ.** (1) Bật ASR từ xa nghĩa là **dải tiếng nói đi lên mạng** —
video thì vẫn không, nhưng "chỉ text rời khỏi máy" không còn đúng. (2) Tunnel chết
thì ASR/dịch rơi về model local, **tức là sẽ tải model về** đúng lần đó. (3) `demucs`
(~80 MB) và `pyannote` (~31 MB) vẫn ở local — chúng chạy tốt trên MPS và Demucs phải
trả về hai stem nếu đưa lên mạng, nên 111 MB đổi lấy round-trip nặng là đánh đổi tệ.

#### Chạy model trên GPU Colab

Adapter `remote` đẩy **riêng bước synthesis** sang một máy có CUDA. Lý do rất hẹp:
XTTS và F5 có kernel không có bản Metal nên trên máy Mac chúng âm thầm rơi về CPU và
một câu thoại mất lâu hơn chính đoạn phim chứa nó. Demucs, Whisper, NLLB, pyannote
vẫn chạy MPS tại chỗ. Thứ đi qua mạng chỉ là một dòng text và vài giây audio tham
chiếu — **video không bao giờ rời máy**.

**Chọn model ngay trên web UI.** Panel "New dubbing job" có dropdown *Voice / TTS
model*. Nó hỏi `GET /speech/tts-models?language=<code>` nên danh sách đổi theo ngôn
ngữ đích, và **engine không dùng được vẫn hiện, kèm lý do** — `XTTS-v2 — XTTS-v2's 17
languages do not include 'vi'`. Đây là chủ ý: một dropdown chỉ chứa MMS-TTS làm hệ
thống trông như chưa bao giờ có lựa chọn nào khác.

Lựa chọn được lưu vào `job.options.tts_model`, **không đi qua request của từng
stage**. Nhờ vậy n8n không phải mang tham số đó qua 13 lời gọi HTTP, và mọi đường
chạy — n8n, `smoke-test.sh`, hay một lệnh curl tay — đều tôn trọng nó mà không cần
biết nó tồn tại.

`REMOTE_TTS_ENGINES=vixtts,f5_vi` biến một endpoint thành nhiều model chọn được:

```bash
curl -X POST localhost:47800/speech/synthesize \
  -H 'content-type: application/json' \
  -d '{"job_id":"<id>","force_model":"remote:f5_vi"}'
```

Đây là điều làm phép so sánh trở nên khả thi trên **job thật** chứ không chỉ trên câu
rời: chạy cùng một job hai lần, đổi đúng một tham số, rồi so `synthesis_manifest`.
Cột `duration_ratio` cho biết engine nào khớp slot thời gian tốt hơn — thứ mà benchmark
câu rời không đo được vì nó không có slot nào để khớp. Cột `tts_model` ghi engine đã
thực sự phát âm từng segment, nên một job rơi fallback giữa chừng vẫn đọc được về sau.

Bảy engine, bộ chỉ số khách quan (RTF · WER/CER · speaker similarity · MOS) và lý do
chọn đúng bảy cái đó: **`colab/README.md`**.

---

## 5. Layout object trong MinIO

```text
jobs/{job_id}/
    source/input.mp4
    audio/{original,speech,background,dubbed_speech,final_mix}.wav
    transcript/{source,diarization,segments}.json
    translation/{lang}.json
    speakers/speaker_01.wav
    generated/segment_0001.wav …
    subtitles/{lang}.srt, {lang}.vtt
    output/dubbed_{lang}.mp4
```

Giữa các node n8n chỉ đi qua:

```json
{ "job_id": "abc123", "target_language": "vi" }
```

---

## 6. Xử lý lỗi

Mọi lỗi trả về JSON có `code` ổn định để n8n branch được:

```json
{ "error": { "code": "no_speech_detected", "message": "...", "retryable": false, "details": {} } }
```

| Tình huống | Hành vi |
|---|---|
| Video không có audio | `invalid_input`, dừng sớm với thông báo rõ |
| Không có speech | `no_speech_detected` |
| Audio quá nhỏ / nhiễu | Vẫn chạy, `peak_dbfs` + cờ `quiet_warning` trong stage output |
| Không nhận diện được ngôn ngữ | `language_detection_failed` (confidence < 0.35) |
| Diarization lỗi / thiếu token | Fallback 1 speaker, job vẫn chạy |
| TTS không hỗ trợ ngôn ngữ | Router chuyển sang MMS-TTS; hết cách → `unsupported_language` |
| Audio sinh ra quá dài/ngắn | Vòng IF adapt → re-generate → time-stretch |
| FFmpeg render fail | Tự retry bằng re-encode trước khi báo lỗi |
| Thiếu GPU | `DEVICE=auto` tự về CPU |
| Model load fail | `model_load_failed` (retryable), n8n retry 3 lần |
| Translation engine lỗi | Error output → nhánh SeamlessM4T |

Nhánh lỗi trong n8n đổ về node **Report Pipeline Failure** → ghi mã lỗi vào Postgres →
UI hiện stage đó màu đỏ, thay vì treo ở trạng thái *running*.

---

## 7. Cấu hình đáng chú ý (`.env`)

| Biến | Mặc định | Ý nghĩa |
|---|---|---|
| `DEVICE` | `auto` | `auto` \| `cpu` \| `cuda` \| `mps` (auto → cuda > mps > cpu) |
| `ASR_DEVICE`, `TTS_DEVICE`, … | `auto` | Ép riêng từng stage khi một backend lỗi trên Metal |
| `WHISPER_MODEL` | `medium` | `tiny`…`large-v3` (GPU nên dùng `large-v3`) |
| `DEMUCS_ENABLED` | `true` | Tắt để chạy nhanh khi debug |
| `DIARIZATION_MODEL` | `pyannote/speaker-diarization-community-1` | Đổi về `speaker-diarization-3.1` nếu cần |
| `DIARIZATION_USE_EXCLUSIVE` | `true` | Dùng exclusive diarization để ghép với timestamp ASR |
| `TRANSLATION_PRIMARY` / `_FALLBACK` | `nllb` / `seamless` | Đổi thứ tự để benchmark |
| `TTS_PREFER_VOICE_CLONE` | `true` | `false` → luôn dùng MMS-TTS |
| `SYNC_RATIO_*` | 0.90/1.10/0.80/1.20 | Ngưỡng accept / stretch |
| `ASR_AUDIO_SOURCE` | `original` | Whisper nghe bản mix gốc, **không** nghe stem Demucs |
| `ASR_MAX_SEGMENT_SECONDS` | `12.0` | Trần độ dài một segment lồng tiếng |
| `TTS_TRIM_SILENCE` | `true` | Cắt ~0.7 s im lặng TTS gói quanh mỗi take |
| `TTS_MAX_SPEAKING_RATE` | `1.45` | Đọc nhanh hơn trước khi phải time-stretch |
| `BACKGROUND_GAIN_DB` | `-6.0` | Mức nhạc nền trong bản mix |
| `MIX_DUCK_ENABLED` | `true` | Ducking sidechain: nền tự lùi khi bản dub nói |
| `SPEECH_AUTO_GAIN` | `true` | Cân mức giọng dub về `-3 dBFS` trước khi trộn |
| `STORAGE_BACKEND` | `minio` | `local` để dùng thư mục `data/jobs` |

Vì sao từng tham số ở giá trị đó — kèm số đo trên clip mẫu: [docs/quality.md](docs/quality.md).

Không có đường dẫn nào hard-code trong workflow: node n8n dùng
`{{ $env.AI_SERVICE_BASE_URL }}`, service dùng `app/storage/layout.py`.

---

## 8. Lệnh hay dùng

```bash
make up              # hybrid: hạ tầng Docker (= ./scripts/start.sh)
make native-setup    # tạo .venv + cài torch/model lib
make native-run      # chạy AI service native (dùng GPU Apple)
make device          # xem từng stage đang chạy trên device nào
make up-docker       # tất cả trong Docker (CPU trên macOS)
make test            # 76 unit test logic thuần, không cần model
make lint            # ruff
make clip            # cắt clip demo có thoại từ data/test/
make logs-ai         # xem log ai-service
make import-workflow # import/cập nhật workflow n8n
make verify          # chạy toàn bộ ffmpeg + đồng bộ, GIẢ LẬP model (~20 s)
make smoke           # chạy pipeline THẬT end-to-end bằng curl -> data/output/
make down            # dừng (giữ volume)
make nuke            # xoá cả volume: postgres, minio, n8n, model cache
./scripts/reset.sh   # xoá job + dọn scratch (scratch KHÔNG tự dọn)
```

---

## 9. Chất lượng bản dub

Một pipeline "chạy xong không lỗi" vẫn có thể cho ra file chỉ có nhạc nền và không ai nói.
Sáu nguyên nhân độc lập gây ra triệu chứng đó — cắt segment của Whisper, chọn nguồn audio
cho ASR, ngân sách ký tự, im lặng thừa của TTS, ducking, và `-shortest` cắt cụt video —
được ghi lại kèm số đo ở [docs/quality.md](docs/quality.md).

Hai giới hạn đã biết, **không phải bug**:

* **Tiếng Việt không nhân bản được giọng.** XTTS-v2 chỉ hỗ trợ 17 ngôn ngữ, không có `vi`,
  nên job tiếng Việt dùng MMS-TTS: một giọng tổng hợp đơn, ngữ điệu phẳng.
  `GET /languages` trả cờ `voice_cloning` cho từng ngôn ngữ.
* **`ja` / `zh` / `ko` cần extra của coqui-tts.** Ba ngôn ngữ này nằm trong 17 ngôn ngữ
  của XTTS nhưng cần tokenizer riêng (`cutlet`, `pypinyin`, g2p tiếng Hàn), và MMS-TTS
  **không** đỡ được vì `facebook/mms-tts-jpn` không tồn tại. `requirements.txt` đã ghim
  `coqui-tts[languages]`; môi trường cũ chạy
  `.venv/bin/pip install 'coqui-tts[languages]==0.27.5'`.
* **`data/verify/SYNTHETIC_no_real_voice.mp4` không phải bản dub** — xem mục 3.

---

Xem thêm: [docs/architecture.md](docs/architecture.md) ·
[docs/pipeline.md](docs/pipeline.md) · [docs/quality.md](docs/quality.md) ·
[docs/troubleshooting.md](docs/troubleshooting.md) · [n8n/README.md](n8n/README.md)
# MultilingualVideoDubbingSystem
