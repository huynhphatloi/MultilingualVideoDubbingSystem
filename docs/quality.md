# Chất lượng bản dub — những quyết định thật sự quyết định kết quả

Tài liệu này ghi lại các chỗ mà một pipeline lồng tiếng "chạy xong không lỗi" vẫn cho ra
file không nghe được, và cách hệ thống này xử lý. Mọi con số đều đo trên
`data/samples/dialogue_37s.mp4` (37 giây, cảnh briefing của The Avengers, ~18 câu thoại),
dịch `en → vi`.

Triệu chứng kinh điển của một bản dub hỏng: **mở file lên chỉ nghe nhạc nền, không có
giọng, phụ đề thì sai**. Không có bug nào tên như vậy — nó là hệ quả của 6 nguyên nhân
độc lập bên dưới cộng lại.

---

## 1. Whisper trả về *decoder window*, không phải câu thoại

`faster-whisper` với `vad_filter=true` bỏ khoảng lặng đi rồi map timestamp ngược lại. Một
segment vì thế có thể trùm lên cả đoạn im lặng mà nó chưa từng được nói:

```text
[ 61.38 - 115.05]  "Call it, Captain."      <- 3 từ, 54 giây
```

Với phụ đề thì chỉ là xấu. Với lồng tiếng thì hỏng hẳn:

* `duration_ratio = generated / (end - start)` ≈ 0.02 → segment bị xếp `retranslate`,
  đốt cả hai lượt adapt, rồi bị ép time-stretch: một câu 1 giây bị kéo giãn để lấp 54 giây;
* take được đặt tại `start`, ~53 giây còn lại chỉ còn nhạc nền. **Đó chính là hiện tượng
  "có nhạc mà không có ai nói".**

**Xử lý** — [`app/services/segmentation.py`](../ai-service/app/services/segmentation.py):
window thô không bao giờ đi tiếp. Chúng được cắt lại thành utterance bằng word timestamp
mà Whisper vốn đã trả về:

* siết `start`/`end` về đúng từ đầu / từ cuối;
* tách khi khoảng nghỉ giữa hai từ ≥ `ASR_SPLIT_GAP_SECONDS` (0.7 s);
* tách thêm ở dấu câu khi utterance đã đủ dài; cắt cứng nếu vẫn vượt
  `ASR_MAX_SEGMENT_SECONDS` (12 s);
* bỏ window rỗng / lặp (vòng lặp hallucination của Whisper).

Đây là module thuần dữ liệu — không model, không I/O — nên test rất rẻ:
`ai-service/tests/test_segmentation.py`.

## 2. Đừng cho Whisper nghe stem của Demucs

Trực giác nói: tách giọng ra rồi mới nhận dạng thì chính xác hơn. Đo thì ngược lại.
Whisper được huấn luyện trên audio nhiễu; còn stem `vocals` của Demucs mang artefact pha
mà nó chưa từng thấy. Cùng một clip 37 giây:

| Nguồn audio cho ASR | Kết quả |
|---|---|
| `audio/speech.wav` (stem Demucs) | 11 câu dính liền, không dấu câu: *"eyes call it captain all right listen up until we can close that porta"* |
| `audio/original.wav` (mix gốc) | 18 câu sạch, đúng dấu câu: *"Call it, Captain." / "All right, listen up." / "Until we can close that portal, our priority is containment."* |

Mọi thứ phía sau đều dịch và đọc lại đúng cái mà stage này sinh ra. Transcript sai thì
bản dub sai, dù phần còn lại của pipeline có hoàn hảo.

**Xử lý** — `ASR_AUDIO_SOURCE=original` (mặc định). Diarization và voice reference vẫn
dùng stem Demucs: ở đó việc tách giọng thật sự có ích.

## 3. `no_speech_prob` một mình không đủ để bỏ một câu

Whisper chỉ coi một window là im lặng khi **cả hai** điều kiện đúng: `no_speech_prob` cao
**và** `avg_logprob` thấp. Lọc chỉ bằng `no_speech_prob` sẽ xoá mất thoại thật — trên stem
Demucs, những câu thoại rõ ràng vẫn có `no_speech_prob ≈ 0.84`. Một lần chỉnh sai ngưỡng
đã ném đi 6 trong 9 window của cả cảnh.

**Xử lý** — `segmentation._is_noise()` yêu cầu đủ cả hai tín hiệu.
Test: `test_confident_speech_survives_a_high_no_speech_prob`.

## 4. Ngân sách ký tự phải tính theo tốc độ của **máy đọc**, không phải của diễn viên

`estimate_budget()` từng nhân ngân sách với tốc độ nói thật của diễn viên trong clip. Diễn
viên đọc "Call it, Captain." trong 0.86 giây đẩy hệ số lên kịch trần 1.6, nên bộ dịch được
bảo là "20 ký tự sẽ vừa". Nhưng người đọc bản dub không phải diễn viên đó — mà là một
giọng TTS có tốc độ cố định, cần ~1.2 giây cho 20 ký tự đó.

Hai chỉnh sửa:

* hệ số theo tốc độ nguồn bị siết còn 0.85–1.15;
* sau lượt tổng hợp đầu tiên, `measured_chars_per_second()` đo **tốc độ thật của engine
  trên chính job này** (MMS-TTS tiếng Việt: 16.9 ký tự/giây, bảng tra nói 14.3) và ghi vào
  `tts_chars_per_second` trong working document. Vòng adapt nhắm vào con số đo được đó.

## 5. TTS gói mỗi take trong một lớp im lặng

Mỗi take MMS-TTS (VITS) có ~0.41 giây im lặng ở đầu và ~0.28 giây ở cuối — gần 0.7 giây
trống trên một câu mà khe thời gian trong phim có khi chỉ 0.86 giây. Phần trống đó bị tính
là tiếng nói, làm `duration_ratio` phình lên và kéo theo một lệnh time-stretch phá chất
lượng của chính những từ có thật.

**Xử lý** — `ffmpeg.trim_silence()` chạy cho mọi take, giữ lại 40 ms mỗi đầu để không cắt
mất phụ âm.

## 6. Ba cách sửa một câu quá dài, theo thứ tự thiệt hại tăng dần

Khi take vượt khe thời gian, pipeline thử theo đúng thứ tự này:

1. **đọc nhanh hơn** — `speaking_rate` của VITS (`length_scale = 1 / speaking_rate`),
   tối đa `TTS_MAX_SPEAKING_RATE` (1.45). Không tốn gì về chất lượng;
2. **time-stretch sau khi tổng hợp** — `atempo`, tối đa 1.35 (1.6 khi cần tránh đè lên
   câu kế tiếp: hai giọng chồng nhau khó nghe hơn một giọng hơi nhanh);
3. **dịch lại ngắn hơn** — tốn thêm một vòng model, và với câu ngắn NLLB thường trả về
   **đúng câu cũ**. `adapt()` phát hiện trường hợp đó, không đốt lượt thử, không tổng hợp
   lại, mà chuyển thẳng segment sang time-stretch.

Để lượt dịch lại có cái để chọn, NLLB sinh 4 `length_penalty` × 4 beam = 16 ứng viên rồi
lấy ứng viên gần ngân sách nhất (hoà thì lấy ngắn hơn).

## 7. Nghe được ≠ trộn đúng

Ba việc quyết định bản dub có nghe rõ hay không:

* **auto-gain giọng.** Các engine TTS chênh nhau cả chục dB. Track dub được đưa về đỉnh
  `SPEECH_TARGET_PEAK_DBFS` (-3 dBFS) trước khi trộn (`SPEECH_AUTO_GAIN`);
* **ducking bằng sidechain.** Nhạc nền bị nén xuống *trong lúc* bản dub đang nói
  (`sidechaincompress`, `MIX_DUCK_*`) rồi trả lại ngay sau đó. Hạ đều `background_gain_db`
  cho cả phim không làm được việc này: hoặc nhạc át lời, hoặc mất nhạc;
* **loudnorm** ở cuối (I=-16, TP=-1.5) cho mức phát chuẩn.

## 8. `-shortest` cắt mất phần cuối video

`ffmpeg -shortest` xét **mọi** stream, kể cả track phụ đề nhúng — mà track phụ đề kết thúc
ở cue cuối cùng. Một clip 20 giây có câu thoại cuối ở giây 17.5 ra file **17.5 giây**:
phần cuối biến mất, và thoại càng thưa thì mất càng nhiều.

**Xử lý** — `ffmpeg.mux()` ghim output theo đúng độ dài video nguồn bằng `-t`.

---

## Đo lại trên clip mẫu

| | trước | sau |
|---|---|---|
| segment ASR | 3 (11 window bị lọc nhầm) | 18 |
| chất lượng transcript | dính liền, không dấu câu | đúng câu, đúng dấu |
| `duration_ratio` trung bình | 2.34 | 1.32 |
| segment đè lên câu kế tiếp | 10 | 0–1 |
| độ dài output | 17.5 s / 20.0 s nguồn | đúng bằng nguồn |

## 9. Một giọng cho tất cả nhân vật ("trộn giọng tùm lum")

MMS-TTS chỉ có **đúng một giọng cho mỗi ngôn ngữ**. Với `vi` (XTTS không hỗ trợ), mọi
nhân vật đều do cùng một người đọc — một cảnh hai người thành ra như một người tự nói
chuyện với mình. Đây không phải lỗi đồng bộ; đo trên clip mẫu thì nền nhạc sạch (không
rò thoại gốc) và chỉ có 2/18 câu bị đè lên nhau.

**Xử lý** — `TTS_SPEAKER_PITCH_ENABLED`: mỗi speaker được dịch cao độ một chút, xếp theo
thời lượng thoại (người nói nhiều nhất giữ giọng gốc, offset 0). Đo lại trên chính output
của MMS: yêu cầu -2.5 nửa cung → đo được -2.5 nửa cung, độ dài giữ nguyên 2.24 s.

Không có `rubberband` trong ffmpeg bản thường, nên dùng cách kinh điển: `asetrate` để đổi
cao độ (kéo theo độ dài), rồi `atempo` **nghịch đảo** để trả lại độ dài. Dùng nhầm `atempo`
cùng chiều thì hiệu ứng bị bình phương — 2.0 s ra 3.17 s.

## 10. Nén giọng quá tay — nguyên nhân thật của "nghe méo"

Stage đồng bộ từng nén mỗi take cho vừa **slot gốc** (khoảng thời gian diễn viên nói),
thay vì vừa **window** (khoảng trống trước khi câu kế tiếp bắt đầu). Đo trên clip 37 giây:
**9/18 take bị nén mạnh hơn mức cần thiết**, trong đó có câu bị nén 1.32× dù ngay sau nó
là 3.26 giây im lặng không ai dùng.

Lồng tiếng thật làm ngược lại: vào đúng nhịp, rồi cho câu chạy lấn vào khoảng nghỉ phía
sau. Take vẫn được đặt tại `start` gốc nên vẫn khớp hình, chỉ là mượn khoảng lặng đang bỏ
trống.

| | trước | sau |
|---|---|---|
| nén > 1.35× (nghe rõ là vội) | 9/18 | 6/18 |
| không nén gì (1.00×) | 2/18 | 7/18 |

Sáu câu còn lại nằm sát nhau thật, không mượn được chỗ nào — đó là lúc một model dịch
gọn hơn mới giúp được (mục 11).

## 11. Ngân sách ký tự từng **xoá nội dung**

`char_budget` từng là mục tiêu tuyệt đối, và nó phá nát những câu ngắn:

| gốc | budget | ra | vấn đề |
|---|---|---|---|
| "All right, listen up." | 11 | "Được rồi." | mất hẳn "listen up" |
| "Smash." | 22 | "Đúng rồi." | nghĩa là "đúng vậy" |

Cả hai đều ngắn hơn. Không cái nào là bản dịch.

**Xử lý** — `TRANSLATION_MIN_KEEP_RATIO` (0.75): lấy bản dịch tự nhiên làm mốc, nếu nó đã
vừa khe (trong ngưỡng 1.25×) thì dùng luôn; nếu không thì chỉ xét những ứng viên dài ít
nhất 75% bản mốc. Câu quá dài còn cứu được (đọc nhanh hơn, nén lại); nghĩa đã bị xoá thì
không.

Kết quả: `"All right, listen up." → "Được rồi, nghe đi."`

## 12. Tên riêng: vấn đề hẹp hơn tưởng

Trực giác nói phải "khoá" mọi danh từ riêng. Đo thì ngược lại — NLLB **giữ được 15/16**
tên trong bộ thử (David Malan, CS50, Harvard, edX, YouTube, Apple TV, Scratch, Barton,
Stark, Legolas, Thor, Hulk, Fury, Romanoff, Rogers).

Và khoá tất cả thì **tệ hơn**. Cùng một câu, thay tên bằng marker rồi trả lại:

```text
thường  : "Được rồi, hãy cất cánh đi, Legolas."
có khoá : "Được rồi, tốt hơn là clench lên, @@0@@."     <- "clench" không được dịch
thường  : "Tên tôi là David Malan, và đây là CS50, Đại học Harvard."
có khoá : "Tên tôi là @@0@@, và đây là @@1@@, @@2@@ của trường đại học"
```

Marker cắt mất ngữ cảnh của model, và ngữ pháp xung quanh hỏng theo.

Cái thật sự hỏng là một nhóm rất hẹp: **từ vừa là danh từ chung vừa là tên** trong đúng
tài liệu đó. "Captain" là ví dụ chuẩn — là quân hàm thì đúng là "Đại úy", nhưng khi đó là
cách người ta gọi Steve Rogers thì phải giữ "Captain". Model không thể tự biết phim này
dùng nghĩa nào; chỉ người vận hành biết.

**Xử lý** — `config/glossary.json`, **danh sách ngắn**, chỉ những từ như vậy:

```json
{ "en": { "Captain": "Captain", "Cap": "Cap" } }
```

Marker `@@N@@` được chọn bằng đo đạc: trong các dạng thử (`@@N@@`, `{N}`, `XNX`, `NNN`,
`[N]`, `#N#`, `%N%`) thì `@@N@@`, `XNX`, `[N]` sống sót 100%, còn `#N#` và `%N%` thì
không. NLLB thỉnh thoảng nhả thừa dấu (`@@1@@@`) nên hàm khôi phục cố tình khớp lỏng.

Nếu một term biến mất hẳn, hệ thống **không** chắp nó lại vào cuối câu — chuỗi này sắp
được **đọc thành tiếng**, và "Và Hulk Captain Smash" nghe còn tệ hơn là thiếu một cái tên.
Nó ghi cảnh báo vào log thay vì vậy.

## 13. Giới hạn của chính bộ dịch

Sau khi mọi thứ trên đã đúng, chất lượng câu chữ là giới hạn của NLLB-200-distilled-**600M**
— bản nhỏ nhất trong họ NLLB. Trên clip mẫu nó vẫn dịch sai vài chỗ:

| gốc | ra | đúng ra là |
|---|---|---|
| "Call out patterns and strays." | "kêu gọi các mô hình và người lạc lối" | "báo các đợt và những con đi lẻ" |
| "You've got the perimeter." | "anh có đường biên giới" | "anh lo vòng ngoài" |

Đó không phải lỗi pipeline. Ba cách xử lý, tuỳ nhu cầu:

1. `NLLB_MODEL=facebook/nllb-200-distilled-1.3B` (~5 GB) hoặc `nllb-200-3.3B` (~17 GB);
2. `TRANSLATION_PRIMARY=seamless` để so sánh với SeamlessM4T;
3. **sửa tay** — `POST /jobs/{id}/review` với `[{segment_id, translated_text}]`. Segment
   được sửa sẽ được đánh dấu `edited_by_user`, xoá audio cũ và tổng hợp lại. Đây là đường
   dành cho bản dub thật sự dùng được, và cũng là lý do UI có tab sửa bản dịch.

## Việc còn lại (đã biết, chưa làm)

* **Tiếng Việt không nhân bản được giọng — trên máy này.** XTTS-v2 chỉ hỗ trợ 17 ngôn
  ngữ và không có `vi`, nên chạy thuần local thì `vi` rơi về MMS-TTS: một giọng đơn,
  không clone, ngữ điệu phẳng. Đúng như thiết kế, không phải lỗi. `GET /languages` trả
  cờ `voice_cloning` cho từng ngôn ngữ.

  **Đường ra gần nhất là Edge-TTS**, bật sẵn và không cần GPU. Nó không clone được
  giọng gốc, nhưng có **2 giọng mỗi ngôn ngữ** nên hai nhân vật vẫn nghe ra hai
  người — thứ MMS-TTS không làm được. Đo trên clip mẫu: `duration_ratio` từ 1.481
  (MMS) xuống 1.097 (Edge).

  **Muốn giữ đúng giọng diễn viên gốc thì phải clone, và đó là adapter `remote`.** Các checkpoint cộng đồng có clone tiếng Việt
  (viXTTS, F5-TTS-Vietnamese) tồn tại, nhưng kernel của chúng không có bản Metal nên
  trên Mac chúng rơi về CPU và chậm hơn cả thời gian thực. Đặt `REMOTE_TTS_URL` +
  `REMOTE_TTS_ENGINES` là riêng bước synthesis đi sang GPU Colab, phần còn lại vẫn chạy
  MPS tại chỗ. Xem `colab/README.md`.

  **Chọn engine nào thì cần đo, không đoán.** Model card của viXTTS ghi rõ nó yếu ở câu
  dưới 10 từ, mà phụ đề phim đa số dưới 10 từ — nên "engine này nghe hay hơn" trên một
  đoạn demo dài không kết luận được gì cho pipeline này. `colab/benchmark.py` chạy 7
  engine trên cùng bộ câu xếp từ 1 đến 28 từ và tách WER theo độ dài câu; đó mới là bảng
  quyết định. So sánh trên job thật thì chạy hai lần với `force_model="remote:<id>"` khác
  nhau rồi đối chiếu `duration_ratio` trong `synthesis_manifest`.
* **Diarization gộp nhiều nhân vật.** Trên cảnh mẫu, pyannote gộp Cap/Stark/Hawkeye thành
  `SPEAKER_01`. Nhạc nền lớn là nguyên nhân. Có thể ép bằng `num_speakers` khi biết trước.
