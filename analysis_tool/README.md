# Analysis Tool – Quick Start

1. **Prereqs**: Python 3.10+, `llama-cpp-python`, `fpdf`, and a `models.json` pointing to your GGUF files.

2. **VC pitch analysis**
   ```powershell
   python analysis.py ..\sample_pitch.txt --model LFM2-1.2B -o out.pdf -t dark
   ```
3. **Add summaries to an existing analysis JSON**
   ```powershell
   python generate_summaries.py "file_path.json"
   ```
4. Creates: `output/.../VC_Pitch_Analysis_<timestamp>.pdf` + `<name>_analysis.json` + `_summaries.txt`.
5. Omit `--model` to use the default; logs print to console.
