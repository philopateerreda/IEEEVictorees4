# Venture Pitch Scoring System

A comprehensive scoring system for evaluating venture capital pitches across 10 key criteria using local GGUF models.

## Overview

This system evaluates business pitches using 10 critical venture capital criteria, providing numerical scores (1-10) for each section. It uses a local GGUF model for analysis but includes fallback mechanisms when the model encounters issues.

## Features

- **10-Criteria Evaluation**: Scores pitches across all key VC evaluation areas
- **Local Model Support**: Works with local GGUF models (tested with LFM2-1.2B-Q8_0.gguf)
- **Fallback Mechanisms**: Keyword-based scoring when LLM fails
- **JSON Output**: Structured results for easy integration
- **Simple API**: Easy-to-use functions for scoring text or files
- **Fast Processing**: Individual criterion evaluation for quick results

## Installation

1. Ensure you have the required dependencies:
   ```bash
   pip install llama-cpp-python
   ```

2. Download the LFM2-1.2B-Q8_0.gguf model and place it in:
   `P:\progs\vsCode\VSProjects\NLP\models\liquid\LFM2-1.2B-Q8_0.gguf`

## Usage

### Simple Command Line Usage

```bash
python run_scorer.py "P:\path\to\your\pitch.txt"
```

Example:
```bash
python run_scorer.py "P:\progs\vsCode\VSProjects\NLP\IEEEVic\sample_pitch.txt"
```

### Programmatic Usage

```python
from pitch_scorer import score_pitch_from_file, print_scoring_summary

# Score a pitch from file
results = score_pitch_from_file("path/to/pitch.txt")
print_scoring_summary(results)

# Access individual scores
if "scores" in results:
    problem_score = results["scores"]["problem"]["score"]
    solution_score = results["scores"]["solution"]["score"]
    # ... access other scores
    
# Get overall metrics
average_score = results["average_score"]
total_score = results["total_score"]
```

## Scoring Criteria

1. **Problem**: Does it solve a real, urgent, and valuable problem?
2. **Solution**: Is it a 10x improvement with clear advantages?
3. **Market**: Large, growing market with winnable entry point?
4. **Product & Tech**: Scalable technology with proprietary moat?
5. **Go-to-Market**: Validated, repeatable customer acquisition plan?
6. **Team**: Unique expertise and proven execution ability?
7. **Traction**: Tangible proof of momentum and key metrics?
8. **Competition**: Nuanced understanding and credible win plan?
9. **Defensibility**: Advantages that deepen over time?
10. **Financials**: Funding tied to value-creating milestones?

## File Structure

```
Scoring/
├── scoring_config.py       # Configuration and scoring criteria
├── pitch_scorer.py         # Main scoring engine and functions
├── run_scorer.py          # Simple command-line runner
├── scoring_results/       # Output directory for JSON results
│   ├── pitch_score_20250902_223507.json
│   └── pitch_score_YYYYMMDD_HHMMSS.json
└── README.md              # This file
```

## Output Format

Results are saved as JSON with unique timestamped filenames in the format:
`pitch_score_YYYYMMDD_HHMMSS.json`

Example: `pitch_score_20250902_223507.json`

JSON structure:

```json
{
  "pitch_length": 1861,
  "total_criteria": 10,
  "scored_criteria": 10,
  "total_score": 52,
  "average_score": 5.2,
  "scores": {
    "problem": {
      "name": "Problem",
      "description": "Does it solve a real, urgent, and valuable problem...",
      "score": 7
    },
    // ... other criteria
  },
  "timestamp": "2025-09-02 22:24:08"
}
```

## Troubleshooting

If you encounter "llama_decode returned -1" errors:
1. Ensure your model file is correctly placed
2. Check that you have sufficient RAM (at least 4GB recommended)
3. The system will automatically fall back to keyword-based scoring

## License

This project is licensed under the MIT License.