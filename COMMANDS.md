# Lệnh chạy tay

Thư mục dự án:

```bash
cd ~/Personal/master/MultilingualVideoDubbingSystem
```

---

## 0. Chuẩn bị (chỉ 1 lần)

`.venv` **đã cài xong rồi** (torch 2.11 macOS arm64 + toàn bộ model lib). Bỏ qua bước này
trừ khi bạn xoá `.venv`.

```bash
./scripts/setup-native.sh
```

Bật Docker Desktop và **đợi icon con cá voi trên menu bar hết dấu chấm than**:

```bash
open -a Docker
# đợi ~30-60 giây rồi kiểm tra:
docker info | head -5
```

---

## 1. Terminal A — hạ tầng (Docker)

```bash
cd ~/Personal/master/MultilingualVideoDubbingSystem
./scripts/start.sh
```

Script này: build/pull image → dựng `postgres`, `minio`, `n8n`, `frontend` → chờ health →
import workflow vào n8n. Lần đầu mất khoảng 3–5 phút.

Kiểm tra:

```bash
docker compose ps
```

---

## 2. Terminal B — AI service (native, dùng GPU Apple)

```bash
cd ~/Personal/master/MultilingualVideoDubbingSystem
./scripts/run-native.sh
```

Cửa sổ này phải **để mở**. Ctrl-C để dừng.

Xác nhận GPU được dùng:

```bash
curl -s localhost:47800/health/device | python3 -m json.tool
```

Mong đợi `"resolved": "mps"`, và `per_component` có `asr: cpu` (đúng — CTranslate2 không
có backend Metal), còn `separation` / `tts` / `translation` / `diarization` là `mps`.

---

## 3. Bật workflow trong n8n

`./scripts/start.sh` đã tự làm bước này. Kiểm tra lại:

```bash
curl -s -o /dev/null -w '%{http_code}\n' -X POST \
  http://localhost:47678/webhook/dubbing/start -H 'Content-Type: application/json' -d '{}'
```

`200` là xong. Nếu ra `404` thì workflow chưa active:

```bash
./scripts/activate-workflow.sh
```

Không active thì URL `/webhook/` **không tồn tại**: web UI upload xong sẽ đứng ở 2/13
stage (15.4%) mãi mãi, vì không ai gọi stage tiếp theo. Làm tay: mở
<http://localhost:47678> → mở workflow → **Save** → gạt **Active**.

---

## 4. Chạy thử

### Cắt clip demo (1 lần)

```bash
./scripts/make-clip.sh
```

Sinh `data/samples/dialogue_37s.mp4` và `dialogue_60s.mp4` — cắt từ những đoạn **có
thoại**. Chọn nhầm đoạn hành động là hỏng cả buổi demo: `avengers_20s.mp4` cũ chỉ có đúng
một câu ("Send the rest.") trong 20 giây, nên bản dub của nó nghe ra đúng là "chỉ có nhạc
nền, không ai nói" — trong khi pipeline hoàn toàn bình thường.

### Cách A — bản dub thật, qua curl, không qua n8n (khuyến nghị)

```bash
./scripts/smoke-test.sh data/samples/dialogue_37s.mp4 vi
```

Chạy tuần tự 13 stage kèm vòng adapt, in kết quả từng bước, rồi **tải kết quả về
`data/output/<job_id>/`**:

```text
data/output/<job_id>/dubbed_vi.mp4      <- mở lên xem được ngay
data/output/<job_id>/dubbed_vi.srt
```

Lần đầu mất ~15-30 phút vì phải tải model (whisper medium 1.5 GB, NLLB 2.5 GB, pyannote,
demucs, mms-tts). Các lần sau ~2 phút cho clip 37 giây.

### Cách B — qua web UI

Mở <http://localhost:47300>, kéo `data/samples/dialogue_37s.mp4` vào, chọn ngôn ngữ đích
`vi`, bấm **Upload & run pipeline**.

### Cách C — verify code mà không cần model (nhanh, ~20 giây)

```bash
make verify
```

Chạy thật mọi bước ffmpeg + toàn bộ logic đồng bộ trên clip thật, **giả lập ASR/dịch/TTS**.

> Output ở `data/verify/SYNTHETIC_no_real_voice.mp4` **không phải bản lồng tiếng**: giọng
> là sine 180 Hz, "bản dịch" là tiếng Anh cộng từ đệm. Nó chỉ chứng minh phần ffmpeg +
> đồng bộ + trộn tiếng là đúng. Đừng đánh giá chất lượng dub bằng file này.

### Kiểm tra chất lượng bản dub

```bash
# xem phụ đề dịch
cat data/output/<job_id>/dubbed_vi.srt

# xem số liệu đồng bộ của job (ratio, segment bị đè, stretch)
curl -s localhost:47800/jobs/<job_id> | python3 -m json.tool
```

`duration_ratio` trung bình nên quanh 1.0-1.4, `overlapping_segments` nên rỗng.
Giải thích từng con số: [docs/quality.md](docs/quality.md).

---

## Các URL

| Dịch vụ | URL |
|---|---|
| Web UI | <http://localhost:47300> |
| n8n | <http://localhost:47678> |
| FastAPI Swagger | <http://localhost:47800/docs> |
| MinIO Console | <http://localhost:47901> (`minioadmin` / `minioadmin123`) |

---

## Lệnh hay dùng

```bash
# trạng thái + log
docker compose ps
docker compose logs -f n8n
make device                      # device của từng stage

# clip demo
./scripts/make-clip.sh                                       # 2 clip mặc định
./scripts/make-clip.sh data/test/The-Avengers.mp4 150 45 my_clip   # cắt đoạn khác

# test + lint
make test                                                    # unit test, không cần model
make lint

# dọn dẹp
./scripts/reset.sh               # xoá hết job (DB + object + scratch)
./scripts/reset.sh --orphans     # chỉ quét scratch mồ côi (scratch không tự dọn!)
docker compose down              # dừng hạ tầng, giữ dữ liệu
docker compose down -v           # xoá luôn volume (postgres, minio, n8n)
```

---

## Khi có sự cố

```bash
# Docker chưa chạy
open -a Docker && sleep 45 && docker info | head -3

# kiểm tra cache model ghi được không (lỗi hay gặp khi chạy native)
curl -s localhost:47800/health/ready | python3 -m json.tool

# webhook n8n lỗi -> xem nguyên nhân thật
docker compose logs --tail=80 n8n

# n8n không gọi được AI service
docker compose exec n8n env | grep AI_SERVICE
# phải là http://host.docker.internal:47800

# AI service không thấy Postgres/MinIO
nc -z localhost 47432 && echo "postgres ok"
nc -z localhost 47900 && echo "minio ok"

# Metal lỗi ở một stage cụ thể -> ép về CPU trong .env rồi chạy lại run-native.sh
#   DIARIZATION_DEVICE=cpu
#   TTS_DEVICE=cpu

# xem tất cả artifact của một job
curl -s localhost:47800/jobs/<job_id>/artifacts | python3 -m json.tool
```

Chi tiết hơn: [docs/troubleshooting.md](docs/troubleshooting.md)
