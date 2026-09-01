# Google Colab TTS service

This folder is an application service, not a benchmark lab. It exposes one API:

```text
POST /synthesize -> audio/wav
```

Open `tts_service.ipynb` in Google Colab, run all cells, then copy the printed
`COLAB_TTS_URL` and `COLAB_TTS_TOKEN` values into the project's `.env` file.

MMS-TTS provides the same 50 application languages shown in the frontend. The
service keeps only one language model in GPU memory at a time.
