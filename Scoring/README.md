This system evaluates venture capital pitches across 10 key criteria using a local GGUF model, providing numerical scores (1-10) for each section and a comprehensive summary. It includes fallback mechanisms for model failures.

### Installation

1.  **Install dependencies:**
    ```bash
    pip install llama-cpp-python
    ```
2.  **Download model:**
    See `models.json` for model location. 

### How to Run

**1. Command Line Usage:**

To score a pitch from a text file:

```bash
python run_scorer.py "P:\path\to\your\pitch.txt"
```

### Output

Results are saved as JSON files (e.g., `pitch_score_YYYYMMDD_HHMMSS.json`) in the `scoring_results/` directory, detailing scores for each criterion and overall metrics.

### Troubleshooting

If you encounter "llama_decode returned -1" errors, ensure the model file is correctly placed and you have sufficient RAM. The system will automatically fall back to keyword-based scoring if the LLM fails.