# Audio Transcription Script

A lightweight command‑line utility to batch transcribe all `.mp3` files in a folder to text using the Hugging Face implementation of OpenAI's Whisper model (`openai/whisper-base`). Each MP3 gets a matching `.txt` file with the transcription.

---

## ✨ Features
* Batch processes every `*.mp3` file in a target directory
* Automatic device selection (CUDA GPU if available, else CPU)
* In‑memory MP3 → WAV conversion (no temp files written)
* Sample rate normalization (resamples to 16kHz for Whisper)
* Deterministic inference mode (`model.eval()` + `torch.no_grad()`)
* Graceful error handling per file (continues after failures)
* Minimal, readable code with type hints

---

## 📂 Repository Contents
| File | Purpose |
|------|---------|
| `audio_transcriber.py` | Main transcription script (CLI) |
| `README.md` | Documentation (you're here) |

---

## 🧩 How It Works (High Level)
1. Loads Whisper processor + model (`openai/whisper-base`).
2. Detects GPU: uses `cuda` if `torch.cuda.is_available()` else falls back to CPU.
3. For each `.mp3` file in the provided folder:
	* Decodes with `pydub` into memory.
	* Converts to WAV bytes (still in memory) and loads via `torchaudio`.
	* Resamples (if needed) to 16 kHz.
	* Extracts log-mel features with the processor.
	* Generates text with the model.
	* Writes `<original>.txt` alongside the audio.

---

## ✅ Requirements
* Python 3.9+ (tested with 3.9/3.10/3.11)
* FFmpeg (required by `pydub` to decode MP3)
* pip packages:
  * `torch` & `torchaudio`
  * `transformers`
  * `pydub`
  * (optional) `accelerate` if you later extend optimization

### Install FFmpeg
Ubuntu / Debian:
```bash
sudo apt update && sudo apt install -y ffmpeg
```
macOS (Homebrew):
```bash
brew install ffmpeg
```
Windows (Chocolatey):
```powershell
choco install ffmpeg
```

### Install Python Dependencies
You can install dependencies directly (no `requirements.txt` is provided by default):
```bash
pip install --upgrade pip
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121  # for CUDA (adjust version as needed)
pip install transformers pydub
```
For CPU-only install (no GPU):
```bash
pip install torch torchaudio transformers pydub
```

> If you run into CUDA mismatch issues, visit: https://pytorch.org/get-started/locally/

---

## 🚀 Usage
Basic command:
```bash
python audio_transcriber.py /path/to/folder/with/mp3s
```

Example:
```bash
python audio_transcriber.py ./samples
```

Output: For each `track.mp3`, a `track.txt` file appears in the same folder.

### Sample Output (Truncated)
```
Loading Whisper model and processor...
Using device: cuda

Found 3 MP3 file(s) to transcribe.

Processing file: interview_intro.mp3
✅ Transcription saved to: interview_intro.txt
```

---

## 🧪 Verifying Install
1. Put a short MP3 (5–10 sec) into a folder, e.g. `test/hello.mp3`.
2. Run:
	```bash
	python audio_transcriber.py test
	```
3. Confirm `hello.txt` appears with transcription text.

---

## 🛠 Troubleshooting
| Issue | Cause | Fix |
|-------|-------|-----|
| `ModuleNotFoundError: pydub` | Dependency missing | Install with `pip install pydub` |
| `Couldn’t find ffmpeg` | FFmpeg not installed/in PATH | Install FFmpeg (see above) |
| CUDA not used | GPU drivers / CUDA not found | Check `nvidia-smi`; install proper PyTorch CUDA build |
| Slow inference | CPU fallback | Use a GPU or smaller model (modify script to use `tiny` / `small`) |
| Memory error | Large model + low VRAM | Switch to smaller Whisper model |

---

## 🔧 Customization Ideas
You can easily extend the script:
* Add support for recursive folder traversal (`rglob('*.mp3')`).
* Allow choosing model size via CLI arg (`--model openai/whisper-small`).
* Add language forcing (`forced_decoder_ids`).
* Export JSON with timestamps (requires modification & use of `return_timestamps`).
* Parallelize CPU preprocessing for large batches.

---

## ♻️ Changing the Model
Edit inside `setup_model_and_processor()`:
```python
processor = WhisperProcessor.from_pretrained("openai/whisper-small")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small").to(device)
```
Smaller models (`tiny`, `base`) are faster; larger (`small`, `medium`, `large`) are more accurate.

---

## 📊 Performance Tips
* Prefer GPU (even a modest one) for long audio batches.
* Batch multiple clips by concatenating them only if logically safe (not recommended for unrelated files).
* Avoid very long MP3s (>30–60 min) without chunking; consider splitting before transcription.
* Pin dependency versions for reproducibility in production.

---

## 🔐 Privacy & Data Note
All transcription happens locally—no API calls to OpenAI's hosted Whisper or other SaaS. Model weights are downloaded once from Hugging Face and cached in `~/.cache/huggingface`.

---

## 🧭 Roadmap (Optional Enhancements)
* [ ] CLI flags: `--model`, `--language`, `--verbose`
* [ ] Progress bar (e.g. `tqdm`)
* [ ] Optional diarization integration
* [ ] Timestamped subtitle output (`.srt` / `.vtt`)
* [ ] Dockerfile for reproducible runtime
* [ ] `requirements.txt` / `pyproject.toml`

---

## 📄 License
Specify a license (e.g. MIT) here. (Currently unspecified.)

---

## 🙌 Acknowledgements
* OpenAI for Whisper
* Hugging Face Transformers & community
* PyTorch / Torchaudio teams

---

## 💬 Feedback
Issues & suggestions welcome—open an issue or submit a PR.

---

Happy transcribing! 🎧
