# Kiến trúc

## Nguyên tắc thiết kế

1. **n8n chỉ điều phối.** Không có model nào chạy trong Code node. Mọi thứ nặng nằm sau
   HTTP trong `ai-service`. Đổi model = sửa một file Python, workflow không đổi.
2. **Không truyền binary qua workflow.** Chỉ node *Upload Video to Storage* chạm vào file.
   Sau đó mỗi node gửi/nhận đúng `{ "job_id": "..." }`.
3. **Mọi artifact có object key xác định.** `app/storage/layout.py` là nơi duy nhất biết
   đường dẫn; không stage nào tự ghép chuỗi path.
4. **Mỗi stage tự report.** `jobs.track(job_id, stage)` bọc quanh mỗi endpoint: ghi
   `running → completed/failed`, đo thời gian, bắt lỗi. Đó là nguồn dữ liệu cho thanh
   tiến độ trong UI.
5. **Degrade thay vì crash.** Thiếu HF token → 1 speaker. Demucs tắt → speech = audio gốc,
   background = im lặng. NLLB lỗi → SeamlessM4T.
6. **Model có thể thay mà không sửa workflow.** `diarization.py` đọc được cả output của
   `community-1` (object có `.speaker_diarization` + `.exclusive_speaker_diarization`) lẫn
   của `speaker-diarization-3.1` (Annotation trần), nên đổi model chỉ là đổi một dòng
   `.env`.

## Các thành phần

| Thành phần | Vai trò | Cổng | Chạy ở đâu |
|---|---|---|---|
| `n8n` | Workflow orchestrator (Community Edition, self-host) | 47678 | Docker |
| `ai-service` | FastAPI: ffmpeg + toàn bộ model AI | 47800 | **Native** (profile `docker-ai` nếu muốn Docker) |
| `postgres` | DB cho n8n (`n8n`) + job registry (`dubbing`) | 47432 | Docker |
| `minio` | Object storage cho media | 47900 / 47901 | Docker |
| `minio-init` | Tạo bucket + lifecycle rồi thoát | — | Docker |
| `frontend` | React build, nginx proxy `/api` và `/webhook` | 47300 | Docker |

## Vì sao hybrid

Docker Desktop trên macOS chạy container trong VM Linux **không có Metal passthrough**.
Model nào cũng bị kẹt trên CPU dù máy có GPU. Nên `ai-service` — thành phần duy nhất cần
GPU — chạy native, còn ba service hạ tầng (không dùng GPU, cài native thì phiền) ở lại
Docker. Ranh giới network:

```text
  host                                    docker network "mvds"
  ────                                    ─────────────────────
  ai-service (venv, :47800)  ◄─────────── n8n        (host.docker.internal:47800)
        │                                frontend   (nginx proxy /api)
        ├──► postgres  localhost:47432 ──► postgres
        └──► minio     localhost:47900 ──► minio
```

`.env` giữ endpoint phía **native** (`localhost:47432`, `localhost:47900`);
`docker-compose.yml` ghi đè bằng tên service (`postgres:5432`, `minio:9000`) khi
`ai-service` chạy trong profile `docker-ai`.

### Ports

Toàn bộ port **publish ra host** nằm trong dải `473xx-479xx` để không đụng các cổng phổ
biến (3000, 5432, 5678, 8000, 9000) mà máy có thể đang dùng cho việc khác. Port **bên
trong container** giữ nguyên mặc định của từng image — đổi port host không ảnh hưởng gì
tới cấu hình nội bộ.

| Dịch vụ | Host | Trong container |
|---|---|---|
| Frontend | 47300 | 80 |
| Postgres | 47432 | 5432 |
| n8n | 47678 | 5678 |
| AI service | 47800 | 8000 (chỉ khi bật profile `docker-ai`) |
| MinIO API | 47900 | 9000 |
| MinIO Console | 47901 | 9001 |

Đổi port: sửa `*_PORT` trong `.env` rồi `docker compose up -d`. Link trên header của web
UI được inject lúc build (`VITE_*` build args) nên cũng tự đi theo.
`scripts/start.sh` tự sửa `AI_SERVICE_BASE_URL` / `AI_SERVICE_UPSTREAM` theo chế độ, còn
nginx của frontend dùng template + envsubst nên một image chạy được cả hai hướng.

### Device theo từng stage

`core/device.py` phân giải device **cho từng component** chứ không phải một biến toàn cục,
vì không phải backend nào cũng nói được mọi device. Đáng nhớ nhất: faster-whisper chạy
trên CTranslate2, thư viện này không có backend Metal, nên `asr` luôn bị map `mps → cpu`.
Mỗi stage vẫn ép riêng được bằng `ASR_DEVICE`, `TTS_DEVICE`, … khi cần.

## Lớp trong `ai-service`

```text
api/routes/*        HTTP boundary - chỉ validate, gọi service, bọc jobs.track()
schemas/            pydantic contract, cũng chính là body n8n gửi
services/           logic thật, không biết gì về HTTP
services/ffmpeg.py  wrapper duy nhất gọi binary; lỗi -> FFmpegError kèm stderr
storage/            MinIO | local đằng sau cùng một interface
jobs/               SQLAlchemy: Job + JobStage
core/               config (env-driven), logging (JSON + correlation id),
                    errors (mã lỗi ổn định), device (cpu/cuda), languages
```

### Model registry

`services/models_registry.py` giữ model warm giữa các HTTP call (n8n gọi mỗi stage một
request riêng). Load được bảo vệ bằng lock nên hai job song song không nạp trùng model.
`DELETE /health/models` giải phóng RAM/VRAM khi cần.

### Vì sao endpoint đồng bộ

Mỗi stage chặn cho tới khi xong, n8n chờ. Điều này giữ workflow đúng như hình vẽ trong đề
bài (mũi tên tuần tự, thấy được trên GUI) và giữ retry/branch của n8n có ý nghĩa. Tiến độ
vẫn quan sát được vì stage ghi thẳng vào Postgres, UI poll `/jobs/{id}`.

Node n8n vì thế đặt timeout rộng (Demucs/TTS trên CPU có thể vài giờ). Nếu muốn scale thật,
bước tiếp theo là đổi các endpoint sang trả `202 Accepted` + queue — nhưng khi đó workflow
sẽ mất tính "nhìn là hiểu" khi trình bày.

## Ràng buộc phiên bản đáng nhớ

`pyannote.audio 4.x` kéo theo một chuỗi ràng buộc cứng:

```text
pyannote.audio 4.0.7  →  torch >= 2.8, torchcodec >= 0.7
torchcodec 0.11       ↔  torch 2.11        (bảng tương thích chính thức)
torchcodec < 0.11     →  KHÔNG có wheel linux/arm64
```

Vì Docker trên Apple Silicon chạy `linux/arm64`, bộ pin duy nhất build được trên **cả**
Mac và Linux/GPU là **torch 2.11.0 + torchaudio 2.11.0 + torchcodec 0.11.0**. Đó là lý do
`requirements-cpu.txt` và `requirements-gpu.txt` pin đúng ba dòng đó, và vì sao image GPU
dùng base CUDA 12.8 (khớp với `--extra-index-url .../cu128`).

Một ràng buộc nữa đi ngược chiều:

```text
demucs >= 4.1.0  →  sphn (Rust)  →  KHÔNG có wheel linux/arm64 ở mọi phiên bản
                                 →  compile từ source → libopus qua CMake → hỏng trên CMake 4
```

Nên `demucs` bị **pin ở 4.0.1** (bản cuối không dùng sphn). Đã kiểm chứng 4.0.1 vẫn tương
thích torch/torchaudio 2.11: `torchaudio.save()` còn nguyên tham số `encoding` và
`bits_per_sample`, `torchaudio.load()` còn nguyên chữ ký, và mọi cờ CLI mà
`services/separation.py` dùng (`-n -o -d --shifts --segment --two-stems --filename`) đều
tồn tại trong 4.0.1. Code demucs 4.0.1 cũng không dùng alias numpy đã bị xoá ở numpy 2.

Các package còn lại đều thoả: `coqui-tts 0.27.5` (torch>=2.2, transformers>=4.57),
`transformers 4.57.6`, `numpy 2.2.6` (giao của `numba<2.6` và `librosa>=2.1`).

> Bài học chung: mỗi lần nâng một package AI, kiểm tra **wheel có tồn tại cho
> linux/arm64 hay không** trước khi kiểm tra ràng buộc phiên bản. Ràng buộc phiên bản
> pip báo rõ ràng; thiếu wheel thì pip im lặng chuyển sang compile và hỏng sau 5 phút.

## Ranh giới ngôn ngữ

`core/languages.py` là nơi duy nhất chuyển đổi mã ngôn ngữ:

| Model | Định dạng | Ví dụ |
|---|---|---|
| Whisper / UI | ISO 639-1 | `vi` |
| NLLB-200 | FLORES-200 | `vie_Latn` |
| SeamlessM4T, MMS-TTS | ISO 639-3 | `vie` |
| XTTS-v2 | danh sách riêng | `zh-cn` |

## Bảo mật / phạm vi

Đây là stack học tập chạy local: n8n basic auth tắt mặc định, MinIO dùng credential dev,
CORS mở. Trước khi đưa ra mạng thật cần: bật `N8N_BASIC_AUTH_ACTIVE`, đổi
`N8N_ENCRYPTION_KEY` và credential MinIO, siết CORS, đặt sau reverse proxy có TLS.
