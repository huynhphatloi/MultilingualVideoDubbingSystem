# Pipeline chi tiết

Mỗi mục = một node n8n = một endpoint FastAPI. Tất cả input/output đều là object key.

## 1. `POST /jobs` — Create Processing Job
Sinh `job_id`, tạo 13 bản ghi stage ở trạng thái `pending`, chốt prefix `jobs/{job_id}/`.

## 2. `POST /jobs/{id}/upload` — Store Video
Node duy nhất nhận binary. Ghi `source/input.mp4`, probe bằng ffprobe, lưu duration.

## 3. `POST /media/extract-audio`
FFmpeg → `audio/original.wav` (mono, 16 kHz, PCM). Kiểm tra có audio stream không, đo
`peak_dbfs` để cảnh báo audio quá nhỏ.

## 4. `POST /media/separate`
Demucs `--two-stems vocals`:

```text
original.wav ─► speech.wav      (chỉ giọng nói, 16 kHz mono → diarization + voice ref)
             └► background.wav  (nhạc, tiếng động, ambience, 48 kHz stereo)
```

Tắt Demucs (`DEMUCS_ENABLED=false`) → passthrough kiểu **voice-over**: background là bản
mix gốc đã hạ `VOICEOVER_DUCK_DB`, nên nhạc và tiếng động vẫn còn, đổi lại thoại gốc còn
nghe thấy lờ mờ dưới bản dub. Response nói rõ điều đó (`mode: "voiceover"`,
`original_dialogue_present: true`).

## 5. `POST /speech/transcribe`
faster-whisper, `vad_filter=true`, `word_timestamps=true`.

**Nghe bản mix gốc, không nghe stem Demucs.** `ASR_AUDIO_SOURCE=original` là mặc định:
Whisper vốn được huấn luyện trên audio nhiễu, còn stem `vocals` mang artefact pha mà nó
chưa từng gặp. Đo trên `data/samples/dialogue_37s.mp4`: mix gốc → 18 câu sạch đúng dấu
câu; stem Demucs → 11 câu dính liền không dấu câu. Chi tiết: [quality.md](quality.md).

Whisper trả về **decoder window**, không phải câu thoại — một window có thể trùm cả đoạn
im lặng mà nó chưa từng được nói (`[61.38-115.05] "Call it, Captain."`). Lồng tiếng chia
cho khoảng đó, nên `segmentation.build_utterances()` cắt lại window thành utterance bằng
word timestamp trước khi bất kỳ stage nào khác nhìn thấy:

```text
window thô   [ 61.38 - 115.05]  "Call it, Captain."      (54.0 s)
utterance    [114.37 - 115.05]  "Call it, Captain."      ( 0.7 s)
```

```json
{ "start": 10.2, "end": 12.7, "duration": 2.5,
  "source_text": "We need to solve this problem." }
```

Auto-detect ngôn ngữ; confidence < 0.35 → `language_detection_failed`.

## 6. `POST /speech/diarize`
`pyannote/speaker-diarization-community-1` (pyannote.audio 4.x) → **hai** danh sách turn:

* `turns` — diarization thường, có thể chồng lấn khi hai người nói cùng lúc;
* `exclusive_turns` — mỗi thời điểm chỉ thuộc một speaker.

Tuỳ chọn: `num_speakers` (biết chính xác số người), hoặc `min_speakers`/`max_speakers`.

```json
{
  "diarization_key": "jobs/abc123/transcript/diarization.json",
  "speaker_count": 3,
  "speakers": ["SPEAKER_00", "SPEAKER_01", "SPEAKER_02"],
  "has_exclusive": true,
  "model": "pyannote/speaker-diarization-community-1"
}
```

Community-1 là pipeline hoàn chỉnh nên không cần gọi `segmentation-3.0` riêng như kiến
trúc 3.1 cũ:

```text
kiến trúc cũ (3.1)                     kiến trúc mới (community-1)
segmentation-3.0                       speaker-diarization-community-1
      ↓                                          ↓
speaker embeddings                     (segmentation + embedding + PLDA + VBx
      ↓                                 đã nằm trong config.yaml của checkpoint)
clustering                                       ↓
      ↓                                turns + exclusive_turns
speaker-diarization-3.1
```

## 7. `POST /speech/merge`
Gộp ASR + diarization thành **working document** `transcript/segments.json`:

```json
{
  "segment_id": 12,
  "speaker_id": "SPEAKER_01",
  "start": 10.2, "end": 12.7, "duration": 2.5,
  "source_language": "en", "target_language": "vi",
  "source_text": "We need to solve this problem.",
  "voice_reference": "jobs/abc123/speakers/speaker_01.wav"
}
```

Gán speaker theo độ chồng lấn thời gian, **ưu tiên `exclusive_turns`**. Đây là lý do
chính chọn community-1: với diarization thường, đoạn chồng lấn thuộc về cả hai speaker
nên câu hỏi "từ này của ai" không có đáp án xác định; với exclusive thì có.

Nếu có word timestamp và speaker đổi giữa câu (bên thiểu số chiếm ≥ 25% số từ) thì
**tách segment** — nhờ vậy hội thoại nhanh không bị gán nhầm.

Đồng thời cắt voice reference: chọn các đoạn dài nhất của từng speaker từ `speech.wav`,
nối lại tới ~12 giây, lưu `speakers/speaker_XX.wav`.

## 8. `POST /translation/translate`
NLLB-200 với **char budget** cho từng segment:

```text
budget = duration × chars_per_second(target)  ×  hệ_số_tốc_độ_người_nói
```

`chars_per_second` tính từ tốc độ âm tiết của ngôn ngữ nhân hệ số ký tự/âm tiết theo hệ
chữ (CJK ≈ 1.05, abugida ≈ 2.4, Latin ≈ 2.75). Hệ số tốc độ đo từ chính câu gốc nên người
nói nhanh/chậm đều được tính đúng.

Engine sinh nhiều ứng viên với `length_penalty` khác nhau rồi chọn ứng viên có số ký tự
gần budget nhất. Lỗi → error output của n8n rẽ sang SeamlessM4T.

## 9. `POST /speech/synthesize`
Router chọn adapter, sinh audio, **đo duration thật**, tính `ratio` và gắn nhãn:

| ratio | `sync_action` |
|---|---|
| 0.90 – 1.10 | `accept` |
| 0.80 – 0.90, 1.10 – 1.20 | `stretch` |
| < 0.80 hoặc > 1.20 | `retranslate` |

Trả về `needs_adaptation: [segment_id, …]` — chính là điều kiện của node IF.

## 10. `POST /translation/adapt`
Chỉ dịch lại các segment bị lệch, với budget **hiệu chỉnh bằng số đo thực tế**:

```text
budget_mới = len(bản_dịch_hiện_tại) × (1 / ratio)
```

Take dài 24% → lần sau ngắn đi ~24%. Chính xác hơn nhiều so với bảng tra tĩnh.
Quá `SYNC_MAX_RETRANSLATE_ATTEMPTS` lần → hạ xuống `stretch`, vòng lặp dừng.

## 10b. Vòng adapt (node *Duration Within Tolerance?* của n8n)
Một take quá dài có ba cách sửa, xếp theo mức thiệt hại tăng dần — pipeline thử đúng thứ
tự này:

1. **đọc nhanh hơn** (ngay trong `/speech/synthesize`): `speaking_rate` của VITS, tối đa
   `TTS_MAX_SPEAKING_RATE`. Không mất gì về chất lượng;
2. **time-stretch** (trong `/audio/synchronize`): `atempo`, tối đa 1.35× (1.6× khi cần
   tránh đè lên câu kế tiếp);
3. **dịch lại ngắn hơn** (`/translation/adapt`): tốn thêm một vòng model, và với câu ngắn
   NLLB thường trả về **đúng câu cũ**. `adapt()` phát hiện và trả về trong trường
   `identical` — segment đó chuyển thẳng sang time-stretch, không đốt lượt thử.

Ngân sách ký tự tính theo tốc độ đọc **đo được của chính engine trên job này**
(`tts_chars_per_second`), không phải theo tốc độ của diễn viên gốc.

## 11. `POST /audio/synchronize`
* Tạo canvas im lặng dài đúng bằng video.
* Với mỗi take: nếu `stretch` thì atempo `factor = duration / slot` (giữ nguyên cao độ).
* Ràng buộc cứng: không tràn sang lượt của người kế tiếp — nếu tràn thì nén thêm, tối đa
  1.35× (dưới ngưỡng đó giọng bắt đầu méo).
* Overlay từng take tại `adelay = start × 1000` ms.

```text
segment 1: 0.0 → 2.5
segment 2: 3.1 → 5.0
segment 3: 7.4 → 10.2
```

Khoảng trống giữ nguyên background gốc.

## 12. `POST /audio/mix`
Ba việc, theo thứ tự:

1. **auto-gain giọng** — các engine TTS chênh nhau cả chục dB, nên track dub được đưa về
   đỉnh `SPEECH_TARGET_PEAK_DBFS` (-3 dBFS) trước, giới hạn ±`SPEECH_AUTO_GAIN_LIMIT_DB`;
2. **ducking bằng sidechain** — `sidechaincompress` nén nhạc nền *trong lúc* bản dub đang
   nói rồi trả lại ngay sau đó (`MIX_DUCK_*`). Đo trên clip mẫu: nền ở -27 dB trong khoảng
   nghỉ, -36 dB khi có thoại — tức ~9 dB ducking, còn giọng cao hơn nền ~8 dB;
3. **loudnorm** EBU R128 (`I=-16, TP=-1.5, LRA=11`).

Hạ đều `BACKGROUND_GAIN_DB` cho cả phim không thay được ducking: hoặc nhạc át lời, hoặc
mất nhạc. Response trả `ducking_applied`, `speech_auto_gain_db`, `speech_peak_dbfs`.

## 13. `POST /subtitle/generate`
`.srt` + `.vtt` cho bản dịch và cả bản gốc. Wrap ≤ 42 ký tự/dòng, tối đa 3 dòng, tự sửa
cue chồng nhau.

## 14. `POST /video/render`
Mux `final_mix.wav` lên video stream gốc (`-c:v copy`). `burn_subtitles=true` → re-encode
kèm filter `subtitles`. Nếu stream-copy fail (codec không hợp MP4) → tự re-encode H.264.

Độ dài output được ghim bằng `-t` đúng bằng độ dài video nguồn, **không dùng `-shortest`**:
`-shortest` xét mọi stream kể cả track phụ đề nhúng, mà track đó kết thúc ở cue cuối cùng.
Một clip 20 giây có câu thoại cuối ở giây 17.5 từng ra file 17.5 giây — mất hẳn phần cuối.

Kết quả: `output/dubbed_{lang}.mp4`, job chuyển `completed`.
