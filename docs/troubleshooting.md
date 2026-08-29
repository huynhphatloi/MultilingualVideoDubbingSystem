# Troubleshooting

## "Job đứng ở WAITING, 2/13 stage, 15.4% và không nhúc nhích"

Nguyên nhân gần như luôn là **workflow n8n chưa Active**. Import workflow **không** tự
activate nó, mà workflow chưa active thì **không có URL production** — `POST
/webhook/dubbing/start` trả 404. Web UI upload video xong (2 stage đầu là `create_job` +
`store_video`, do AI service tự làm), rồi gọi webhook, webhook 404, và **không ai gọi
stage 3 nữa**. Job nằm đó vĩnh viễn.

Kiểm tra trong 1 giây:

```bash
curl -s -o /dev/null -w '%{http_code}\n' -X POST \
  http://localhost:47678/webhook/dubbing/start \
  -H 'Content-Type: application/json' -d '{}'
```

`404` = chưa active. `200` = ổn.

Sửa:

```bash
./scripts/activate-workflow.sh
```

Script này activate workflow rồi **restart container n8n** — bắt buộc, vì `n8n
update:workflow --active` báo thẳng *"Activation will not take effect if n8n is
running"*. Làm tay cũng được: mở <http://localhost:47678>, mở workflow, bấm **Save**, gạt
**Active**.

Job đã lỡ kẹt thì không cần upload lại — video còn nguyên trong MinIO, chỉ cần đá lại:

```bash
curl -X POST http://localhost:47678/webhook/dubbing/start \
  -H 'Content-Type: application/json' \
  -d '{"job_id":"<job_id>","target_language":"vi"}'
```

Hoặc chạy nốt bằng tay, không cần n8n: `./scripts/smoke-test.sh`.

> `./scripts/start.sh` giờ tự activate sau khi import, nên chuyện này chỉ còn xảy ra khi
> bạn import workflow bằng tay hoặc tự tắt Active trong n8n.

---

## "Speech generation ✓ nhưng Timeline synchronization đỏ: No generated audio to synchronise"

Hai stage này mâu thuẫn nhau vì stage tổng hợp **đã từng nói dối**: nó bắt lỗi từng
segment, ghi vào `failures`, rồi vẫn trả 200 với `synthesized: 0`. UI tô xanh, job đi
tiếp, và stage sau mới chết — với một câu đúng sự thật nhưng chỉ sai địa chỉ.

Đã sửa: không tạo được segment nào thì `/speech/synthesize` **trả lỗi ngay**, kèm
`distinct_errors` (gộp 32 lỗi giống nhau thành `32x ...`) và một `hint` chỉ đúng thuốc.
Nó cũng dừng sau `_FAIL_FAST_AFTER = 3` segment hỏng liên tiếp thay vì cày hết 32 cái —
lỗi cấu hình thì segment nào cũng hỏng như nhau.

Nguyên nhân gốc hay gặp nhất — **coqui-tts cài thiếu extra ngôn ngữ**:

| target | lỗi thật | cài |
|---|---|---|
| `ja` | `No module named 'cutlet'` | `pip install 'coqui-tts[ja]'` |
| `zh` | thiếu `pypinyin` / `spacy-pkuseg` | `pip install 'coqui-tts[zh]'` |
| `ko` | thiếu g2p | `pip install 'coqui-tts[ko]'` |

`ai-service/requirements.txt` giờ ghim `coqui-tts[languages]`, nên cài mới là đủ. Môi
trường cũ thì chạy:

```bash
.venv/bin/pip install 'coqui-tts[languages]==0.27.5'
```

**Vì sao không có fallback:** router TTS thử `[xtts, mms]`. Với `ja`/`ko`, MMS-TTS cũng
không cứu được — `facebook/mms-tts-jpn` **không tồn tại** (MMS nhắm vào ngôn ngữ ít tài
nguyên; jpn/kor không nằm trong bộ VITS đó). XTTS là engine duy nhất, thiếu tokenizer là
hỏng cả job.

Trước đây thông báo lỗi chỉ hiện engine **cuối cùng** trong chuỗi, tức là lỗi MMS — chỉ
sai hẳn engine. Giờ nó liệt kê mọi engine đã thử:

```
All TTS engines failed for segment 0
  attempts: ["xtts_v2: No module named 'cutlet'",
             "mms_tts: No MMS-TTS checkpoint for 'facebook/mms-tts-jpn'"]
```

---

## "Mở file ra chỉ có nhạc nền, không có giọng, phụ đề thì sai"

Triệu chứng kinh điển. Không có bug nào tên như vậy — nó là nhiều nguyên nhân độc lập
cộng lại. Kiểm tra theo đúng thứ tự này, dừng ở cái đầu tiên khớp:

**1. Bạn đang mở nhầm file.** `data/verify/SYNTHETIC_no_real_voice.mp4` (trước đây là
`data/samples/verify_output.mp4`) là output của `make verify`, trong đó ASR / dịch / TTS
đều bị **thay bằng stub**: "giọng" là sine 180 Hz, "bản dịch" là tiếng Anh cộng từ đệm
tiếng Việt. Nó chỉ để chứng minh phần ffmpeg + đồng bộ + trộn tiếng đúng. Bản dub thật ở
`data/output/<job_id>/`.

**2. Clip nguồn gần như không có thoại.** Kiểm tra `segment_count` ở stage `transcribe`:

```bash
curl -s localhost:47800/jobs/<job_id> | python3 -m json.tool | grep -A3 transcribe
```

`avengers_20s.mp4` cũ chỉ chứa đúng một câu ("Send the rest.") trong 20 giây hành động —
bản dub của nó *đúng* là gần như chỉ có nhạc. Dùng `./scripts/make-clip.sh` để cắt clip
có thoại (`dialogue_37s.mp4`, `dialogue_60s.mp4`).

**3. `segment_count` thấp bất thường so với lượng thoại nghe thấy.** Nghĩa là ASR mất
thoại. Kiểm tra `ASR_AUDIO_SOURCE`: giá trị đúng là `original`. Cho Whisper nghe stem
`vocals` của Demucs làm chất lượng nhận dạng tệ đi rõ rệt (đo được: 18 câu sạch → 11 câu
dính liền không dấu câu).

**4. Bản dịch sai / cụt.** Xem cả `source_text` lẫn `translated_text`:

```bash
curl -s localhost:47800/jobs/<job_id>/segments \
  | python3 -c "import json,sys; [print(s['source_text'],'->',s['translated_text']) for s in json.load(sys.stdin)['segments']]"
```

Nếu `source_text` đã sai thì lỗi ở ASR, không phải ở dịch — quay lại bước 3.

**5. Video ra ngắn hơn video gốc.** Đã sửa: `ffmpeg.mux()` từng dùng `-shortest`, mà
`-shortest` xét cả track phụ đề nhúng (kết thúc ở cue cuối). Clip 20 giây có câu cuối ở
giây 17.5 ra file 17.5 giây. Giờ output được ghim bằng `-t` theo độ dài video nguồn.

**6. Có giọng nhưng bị nhạc át.** Xem response của stage `mix`: phải có
`ducking_applied: true` và `speech_auto_gain_db` khác 0 khi cần. Chỉnh
`BACKGROUND_GAIN_DB`, `MIX_DUCK_RATIO`, `MIX_DUCK_THRESHOLD`.

Giải thích đầy đủ từng nguyên nhân và số đo: [quality.md](quality.md).

---

## Chất lượng đầu ra

**Video ra chỉ có tiếng lồng, mất hết nhạc nền và tiếng động**
Xảy ra khi `DEMUCS_ENABLED=false` hoặc Demucs lỗi. Trước đây nhánh dự phòng đặt
background = **im lặng tuyệt đối**, nên bản mix chỉ còn giọng lồng trôi trên nền trống.
Giờ nhánh này giữ nguyên audio gốc và hạ nhỏ (`VOICEOVER_DUCK_DB`, mặc định -4 dB) —
kiểu **voice-over/lektor**: nhạc và tiếng động còn nguyên, thoại gốc còn nghe thấy lờ mờ
dưới giọng lồng. Stage `separate` báo `mode: "voiceover"` và
`original_dialogue_present: true` để UI nói đúng sự thật.

Muốn xoá hẳn thoại gốc thì **bắt buộc phải có Demucs** — không tách nguồn thì không thể
gỡ giọng ra khỏi bản trộn.

**`background_used: true` nhưng vẫn không nghe thấy nhạc nền**
Đã sửa: trước đây `mixing` chỉ kiểm tra stem có tồn tại và dài > 0.1s, nên một file im
lặng hoàn toàn vẫn được báo là "đã dùng". Giờ nó đo peak; dưới -60 dBFS thì báo
`background_used: false` kèm `background_reason`.

## Chế độ hybrid (native + Docker)

**`OSError: [Errno 30] Read-only file system: '/models'`**
Đã sửa. `.env` trước đây để `HF_HOME=/models/huggingface`, `TORCH_HOME=/models/torch`,
`XDG_CACHE_HOME=/models/cache` — đó là đường dẫn **trong container** (volume
`model-cache`). Chạy native trên macOS thì `/models` không tồn tại và `/` là read-only,
nên torch.hub không tạo được thư mục cache. Triệu chứng đánh lừa: lỗi nổ ra ở **stage đầu
tiên cần tải model** (Demucs) nên trông như Demucs hỏng.

Giờ `.env` giữ đường dẫn native (`./models/...`), `docker-compose.yml` ghi đè bằng
`/models/...`, `run-native.sh` chuyển sang đường dẫn tuyệt đối (subprocess của Demucs
không thừa kế cwd của ta), và `app/core/paths.py` tự kiểm tra quyền ghi lúc khởi động —
không ghi được thì fallback sang `~/.cache/mvds` kèm cảnh báo trong log.

Kiểm tra trước khi chạy job:
```bash
curl -s localhost:47800/health/ready | python3 -m json.tool
```
Các dòng `cache:HF_HOME`, `cache:TORCH_HOME`, `cache:XDG_CACHE_HOME` phải là `ok (...)`.

**`separation_failed: Demucs ...` / `UnpicklingError: Weights only load failed`**
Đã sửa. Nguyên nhân không liên quan tới Metal: demucs 4.0.1 nạp checkpoint bằng
`torch.load(path, 'cpu')`, mà **từ torch 2.6 tham số `weights_only` mặc định là `True`**
nên nó từ chối unpickle object `klass` mà demucs lưu trong package. Ta buộc phải dùng
torch 2.11 (pyannote.audio 4.x yêu cầu >= 2.8) nên không thể lùi torch.

Upstream sửa đúng một chỗ ở demucs 4.1.0: `torch.load(path, 'cpu', weights_only=False)`.
Nhưng 4.1.0 kéo theo `sphn` (không có wheel linux/arm64), nên project dùng
`app/services/demucs_runner.py` — một shim áp đúng bản vá đó trong subprocess mà không
cần đổi dependency. Checkpoint chỉ tải từ URL release chính thức của Meta nên việc nới
`weights_only` ở đây không mở ra rủi ro thực tế.

Chạy tay để kiểm chứng:
```bash
source .venv/bin/activate
python ai-service/app/services/demucs_runner.py -n htdemucs --two-stems vocals \
  -d cpu -o /tmp/demucs-check data/jobs/_scratch/<job_id>/audio/original.wav
```

**`ForeignKeyViolation: Key (job_id)=(...) is not present in table "jobs"` khi tạo job**
Đã sửa. Nguyên nhân: `create_job` mới `flush()` chứ chưa `commit()`, trong khi
`complete_stage("create_job")` lại mở **session khác** — transaction thứ hai không nhìn
thấy hàng `jobs` chưa commit nên chèn `job_stages` bị chặn bởi khoá ngoại. Giờ stage đầu
tiên được đóng ngay trong cùng transaction và `create_job` commit trước khi trả về.
Có regression test ở `ai-service/tests/test_job_registry.py` (chạy trên SQLite thật, bật
`PRAGMA foreign_keys=ON`, mỗi session một connection để mô phỏng đúng isolation của
Postgres).

**`run-native.sh: line NN: EXTRA[@]: unbound variable`**
Đã sửa. Nguyên nhân: macOS vẫn dùng **bash 3.2** làm mặc định, và bash < 4.4 coi việc
mở rộng một **mảng rỗng** dưới `set -u` là lỗi unbound. Script giờ không dùng mảng nữa.
Nếu tự viết thêm script cho dự án này, nhớ tránh `"${arr[@]}"` khi mảng có thể rỗng —
hoặc dùng `"${arr[@]:+"${arr[@]}"}"`.

**`postgres is not reachable on localhost:47432` khi chạy `run-native.sh`**
Hạ tầng chưa lên. Chạy `docker compose up -d` trước.

**AI service native chạy nhưng n8n báo `ECONNREFUSED`**
Node HTTP đang trỏ sai. Kiểm tra: `docker compose exec n8n env | grep AI_SERVICE`.
Ở chế độ hybrid phải là `http://host.docker.internal:47800`. `scripts/start.sh` tự set;
nếu sửa `.env` bằng tay thì nhớ `docker compose up -d n8n` cho nó đọc lại.

**Web UI báo lỗi gọi `/api/...`**
nginx trong container frontend đang proxy sai upstream. `AI_SERVICE_UPSTREAM` phải là
`host.docker.internal:47800` (hybrid) hoặc `ai-service:8000` (profile `docker-ai`), rồi
`docker compose up -d frontend`.

**`/health/device` báo `"resolved": "cpu"` dù đang chạy native trên Mac M-series**
Torch cài từ index CPU thay vì wheel macOS mặc định. Cài lại:
`pip install -r ai-service/requirements-native-macos.txt --force-reinstall`.
Kiểm tra nhanh: `python -c "import torch; print(torch.backends.mps.is_available())"`.

**`NotImplementedError: ... not implemented for MPS`**
Một op của pyannote/coqui chưa có kernel Metal. `run-native.sh` đã export
`PYTORCH_ENABLE_MPS_FALLBACK=1`; nếu vẫn lỗi thì ép riêng stage đó về CPU, ví dụ
`DIARIZATION_DEVICE=cpu` trong `.env`.

**Kết quả trên MPS khác trên CPU (giọng méo, transcript lạ)**
Ép stage nghi ngờ về `cpu` bằng biến `*_DEVICE` rồi so sánh. Đây là cách khoanh vùng
nhanh nhất xem lỗi do Metal hay do model.

## Build & khởi động

**`ai-service` build rất lâu / hết dung lượng**
Image CPU ≈ 6 GB (torch, transformers, demucs, coqui-tts). Cần ≥ 25 GB trống cho image +
model cache. Kiểm tra: `docker system df`, dọn: `docker builder prune`.

**`ai-service` restart liên tục**
`docker compose logs ai-service`. Thường gặp:
* `Could not initialise database` → postgres chưa sẵn sàng; service tự retry 10 lần × 3 s.
* `Cannot reach MinIO` → xem `docker compose logs minio-init`.

**Postgres init không chạy**
Script trong `infra/postgres/init/` chỉ chạy khi volume còn trống. Đã lỡ tạo volume rồi
thì: `docker compose down -v && docker compose up -d`.

## n8n

**Import workflow báo lỗi permission**
```bash
docker compose exec n8n n8n import:workflow --input=/workflows/multilingual-dubbing-pipeline.json
```
Nếu vẫn lỗi, import thủ công: n8n UI → *Workflows* → *Import from File*.

**Node HTTP báo `ECONNREFUSED`**
Trong container phải gọi `http://ai-service:8000`, không phải `localhost:47800`.
Kiểm tra `AI_SERVICE_BASE_URL` trong `docker compose exec n8n env`.

**`$env` trả về undefined**
Cần `N8N_BLOCK_ENV_ACCESS_IN_NODE=false` (đã set sẵn trong compose). Sau khi đổi phải
`docker compose up -d n8n`.

**Node timeout (`ETIMEDOUT`)**
Demucs/TTS trên CPU rất chậm. Tăng *Options → Timeout* của node đó, hoặc giảm
`WHISPER_MODEL`, hoặc `DEMUCS_ENABLED=false` khi chỉ cần demo luồng.

**Webhook trả 500**
Xem lỗi thật trước: `docker compose logs --tail=80 n8n`, hoặc n8n UI → **Executions**.
Một nguyên nhân đã gặp và đã sửa: node Webhook đặt `responseMode: onReceived` kèm
`responseData: allEntries` — hai tham số này loại trừ nhau (`allEntries` chỉ hợp lệ khi
`responseMode: lastNode`), n8n đi nhầm nhánh và trả 500. Nếu bạn import workflow từ trước
lần sửa này, import lại: `make import-workflow`, rồi Save + Active lại trong n8n.

**Webhook trả 404**
Workflow chưa *active*. Đây là nguyên nhân số 1 của "job đứng ở 2/13 stage" — xem mục đầu
tài liệu này. Chạy `./scripts/activate-workflow.sh`, hoặc bấm **Execute workflow** rồi gọi
`/webhook-test/dubbing/start` (URL test dùng được đúng một lần).

Lưu ý: activate bằng CLI **không ăn** nếu n8n đang chạy — n8n in ra *"Activation or
deactivation will not take effect if n8n is running. Please restart n8n"*. Phải
`docker restart mvds-n8n` sau đó (script đã làm sẵn).

## Model

**`model_load_failed` ở stage diarize**
Chưa có `HUGGINGFACE_TOKEN`, hoặc chưa *Agree* điều khoản của
<https://hf.co/pyannote/speaker-diarization-community-1>. Chỉ cần accept **một** repo này
(community-1 là pipeline hoàn chỉnh, không dùng `segmentation-3.0` riêng nữa). Không có
token thì hệ thống tự chạy chế độ 1 speaker và job vẫn hoàn tất.

**`has_exclusive: false` dù đã dùng community-1**
Nghĩa là đang chạy model cũ. Kiểm tra `DIARIZATION_MODEL` trong `.env` và log của stage
`diarize`. Với `speaker-diarization-3.1` thì không có exclusive diarization — pipeline
vẫn chạy nhưng dùng `turns` thường (`turn_source: "regular"` trong `segments.json`).

**`torchcodec` báo lỗi không tìm thấy FFmpeg**
`pyannote.audio 4.x` dùng `torchcodec` để đọc audio, và torchcodec cần thư viện chia sẻ
FFmpeg 4–7. Image đã cài gói `ffmpeg` của Debian nên bình thường là đủ; nếu tự build
image khác thì phải cài `ffmpeg` (không chỉ binary tĩnh).

**Build chết ở `Failed building wheel for sphn` / `audiopus_sys` / `CMake < 3.5`**
Đây là `demucs 4.1.x`, không phải lỗi của bạn. Từ 4.1.0 demucs phụ thuộc cứng vào `sphn`
(Rust extension) — và `sphn` **không có wheel linux/arm64 ở bất kỳ phiên bản nào**. Trên
Apple Silicon pip buộc phải compile từ source, kéo theo Rust toolchain rồi build libopus
bằng CMake, và chết vì CMake 4 đã bỏ hỗ trợ `cmake_minimum_required(<3.5)`.

Cách xử lý: giữ nguyên `demucs==4.0.1` trong `requirements.txt` (đã pin sẵn). Bản 4.0.1
không dùng sphn, và `ta.load` / `ta.save` mà nó gọi vẫn còn trong torchaudio 2.11 kèm
đúng các tham số `encoding` / `bits_per_sample`. **Đừng nâng lên 4.1.x.**

Nếu buộc phải dùng 4.1.x: image đã cài sẵn `pkg-config` + `libopus-dev` và đặt
`CMAKE_POLICY_VERSION_MINIMUM=3.5`, nên audiopus_sys sẽ link libopus hệ thống thay vì
build bằng CMake. Build vẫn chạy được nhưng chậm hơn nhiều vì phải compile Rust.

**`pip` không tìm thấy wheel `torchcodec` khi build trên Apple Silicon**
Wheel linux-aarch64 của torchcodec chỉ có từ `0.11.0` trở lên, và bản đó ghép với
`torch 2.11`. Đừng hạ `torch` xuống 2.8/2.9 — build sẽ hỏng trên máy ARM. Bộ pin đúng
nằm trong `requirements-cpu.txt`.

**XTTS báo lỗi license**
`COQUI_TOS_AGREED=1` (đã set trong Dockerfile và `.env.example`).

**`unsupported_language` ở TTS**
Ngôn ngữ đích không nằm trong 17 ngôn ngữ của XTTS **và** không có checkpoint
`facebook/mms-tts-<iso3>`. Kiểm tra mã ISO-639-3 tại `GET /languages`.

**Giọng nhân bản nghe tệ**
Voice reference quá ngắn hoặc lẫn nhạc. Bật Demucs để reference được cắt từ `speech.wav`.
Độ dài reference do `MAX_SPEAKER_REFERENCE_SECONDS` (mặc định 12 s) quyết định — tăng nó
nếu mỗi nhân vật có đủ thoại sạch.

## Chất lượng đầu ra

**Lời thoại lệch so với hình**
Xem `ratio_stats` trong `GET /jobs/{id}`. `mean` lớn hơn 1 nhiều nghĩa là ngôn ngữ đích
"dài" hơn: siết `SYNC_RATIO_ACCEPT_MAX` hoặc tăng `SYNC_MAX_RETRANSLATE_ATTEMPTS`.

**Giọng nghe như robot / bị méo**
Time-stretch quá mạnh. Hạ trần trong `services/sync.py` (`_MAX_EXTRA_STRETCH`) và để
vòng adapt làm việc nhiều hơn. Với tiếng Việt thì còn một lý do nữa: XTTS-v2 (engine nhân
bản giọng) **không hỗ trợ `vi`**, nên job tiếng Việt rơi về MMS-TTS — một giọng tổng hợp
đơn, không clone, ngữ điệu phẳng. Đó là thiết kế chứ không phải lỗi; `GET /languages` trả
cờ `voice_cloning` cho từng ngôn ngữ.

**`ratio_stats.mean` quá cao (take dài hơn khe thời gian nhiều)**
Pipeline đã tự xử theo ba mức: đọc nhanh hơn (`TTS_MAX_SPEAKING_RATE`) → time-stretch →
dịch lại ngắn hơn. Nếu vẫn cao, kiểm tra `TTS_TRIM_SILENCE=true`: MMS-TTS gói mỗi take
trong ~0.7 giây im lặng, không cắt đi thì mọi con số ratio đều sai.

**Vòng adapt chạy mà không cải thiện gì**
Bình thường với câu ngắn: NLLB trả về đúng câu cũ dù ngân sách ký tự có siết. Response của
`/translation/adapt` liệt kê những segment đó trong trường `identical` — chúng được chuyển
thẳng sang time-stretch thay vì đốt thêm một lượt model.

**Mất nhạc nền**
`background_used: false` ở stage mix → Demucs bị tắt hoặc fail. Xem output stage
`separate_sources`.

**Chỉ nhận ra 1 speaker dù video nhiều người**
Diarization đang fallback (`enabled: false`) hoặc các giọng quá giống nhau. Ép số người:
`POST /speech/diarize {"min_speakers": 2, "max_speakers": 4}`.

## Hiệu năng

| Việc | CPU (M-series / 8 core) | GPU (RTX 3060) |
|---|---|---|
| Demucs, clip 1 phút | 2–5 phút | ~20 giây |
| Whisper medium, 1 phút | 1–3 phút | ~10 giây |
| XTTS, 1 câu | 3–10 giây | ~1 giây |

Mẹo demo: chuẩn bị sẵn clip 30–60 giây, chạy trước một lần để model đã nằm trong
`model-cache`, và giữ `WHISPER_MODEL=small` khi trình bày trên CPU.
