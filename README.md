# Voice Agent (Pipecat + Whisper + Coqui TTS) — CPU-only starter

## Requirements

-   Python 3.10 or 3.11
-   CPU with enough RAM (Whisper and torch on CPU can be memory-heavy)
-   PortAudio (sounddevice) — often installed automatically on Windows/macOS; on Linux you may need `libportaudio` packages.

## Install (recommended)

1. Create virtual env:
   python -m venv venv
   source venv/bin/activate # macOS/Linux
   venv\Scripts\activate # Windows

2. Install CPU PyTorch wheels first (recommended) — example:

    # Linux / Windows CPU

    python -m pip install --upgrade pip
    python -m pip install "torch>=2.0.0" --index-url https://download.pytorch.org/whl/cpu
    python -m pip install torchaudio --index-url https://download.pytorch.org/whl/cpu

    (If the above fails, see https://pytorch.org for the correct CPU wheel for your OS.)

    # Nothing to be done if on MacOS

3. Install other requirements:
   python -m pip install uv
   uv pip install -r requirements.txt

4. Download Piper Voice
   python3 -m piper.download_voices en_US-lessac-medium

5. Run Piper TTS as a server
   python -m piper.http_server --model en_US-lessac-medium --port 5002

7. Run:
   uv run voice_agent.py

## Notes & tips

## Sources
-   Pipecat docs / quickstart. :contentReference[oaicite:5]{index=5}
-   Piper TTS installation and API. :contentReference[oaicite:6]{index=6}
-   Whisper (openai-whisper) details. :contentReference[oaicite:7]{index=7}
