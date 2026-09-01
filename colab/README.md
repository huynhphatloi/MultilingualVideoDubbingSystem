# Google Colab AI service

This folder is an application service, not a research or benchmark lab. It exposes:

```text
POST /transcribe -> timestamped text
POST /translate  -> translated text
POST /synthesize -> audio/wav
```

Open `ai_service.ipynb` in Google Colab, select a GPU runtime, and run all cells.
Copy the printed `COLAB_API_URL`, `COLAB_API_TOKEN`, and `COLAB_API_TIMEOUT`
values into the project's `.env` file.

The service runs Whisper, NLLB-200, and MMS-TTS for the same 50 languages shown
in the frontend. The local application has no model dependencies or fallback.
