# VC Pitch Analysis Tool - Terminal Version

A powerful, terminal-based VC pitch analysis system that uses your local LLM to analyze business pitches with a 10-epoch approach for high-quality analysis and generates professional PDF reports with JSON output.

## Files

- **analysis.py** - Main VC pitch analyzer with 10-epoch analysis system (one question per epoch)
- **llm_summarization.py** - LLM interface for local GGUF models
- **pdf_generator.py** - PDF report generation with light/dark themes
- **run_analysis.py** - Simple wrapper script
- **sample_pitch.txt** - Example pitch for testing
- **output/** - Directory where all analysis results are stored

## Key Features

✅ **10-Epoch Individual Question Processing**: Each of the 10 VC checklist questions is analyzed separately to minimize LLM hallucinations  
✅ **Unique Output Naming**: All outputs automatically get unique timestamps and are organized in the `output/` folder  
✅ **LLM Context Reset**: Complete context reset between epochs prevents contamination  
✅ **Dual Output Format**: Generates both PDF reports and structured JSON files  
✅ **Quality Assessment**: LLM evaluates question clarity and information availability  
✅ **Response Cleaning**: Removes unwanted instruction text from LLM outputs  
✅ **Professional PDF Generation**: Clean reports with light/dark theme support  
✅ **Terminal-Only Execution**: No web servers or APIs required

## Requirements

- Python 3.8+
- llama-cpp-python
- fpdf2
- A local GGUF model file

## Usage

### Direct Analysis
```bash
python analysis.py <pitch_file> [-o output.pdf] [-t light|dark]
```

### Using the Wrapper Script
```bash
python run_analysis.py <pitch_file> [-o output.pdf] [-t light|dark]
```

**Note**: All outputs are automatically placed in the `output/` folder with unique timestamps, regardless of the specified filename.

## Examples

```bash
# Basic analysis with automatic unique naming
# Creates: ./output/VC_Analysis_YYYYMMDD_HHMMSS/VC_Pitch_Analysis_YYYYMMDD_HHMMSS.pdf
# Plus JSON: ./output/VC_Analysis_YYYYMMDD_HHMMSS/VC_Pitch_Analysis_YYYYMMDD_HHMMSS_analysis.json
python analysis.py sample_pitch.txt

# Custom filename with unique timestamp (still goes to output/ folder)
# Creates: ./output/VC_Analysis_YYYYMMDD_HHMMSS/my_analysis_YYYYMMDD_HHMMSS.pdf
python analysis.py sample_pitch.txt -o my_analysis.pdf -t dark

# All outputs are automatically organized in timestamped folders
python analysis.py sample_pitch.txt -o report.pdf -t light

# Using wrapper script (same behavior)
python run_analysis.py sample_pitch.txt -o custom_report.pdf -t dark
```

### Output Structure
Every analysis creates a unique folder structure:
```
output/
└── VC_Analysis_20250126_143052/
    ├── VC_Pitch_Analysis_20250126_143052.pdf
    └── VC_Pitch_Analysis_20250126_143052_analysis.json
```

## Analysis Process

The system uses a **10-epoch approach** for superior analysis quality:

1. **Epoch Processing**: Each of the 10 VC checklist questions is analyzed individually
2. **Context Reset**: LLM context is completely reset between epochs to prevent hallucinations
3. **Progress Tracking**: Terminal displays "First epoch finished", "Second epoch finished", etc.
4. **Quality Assessment**: LLM evaluates if questions are too vague or if pitch lacks required information
5. **Response Cleaning**: Unwanted instruction text is automatically removed
6. **Structured Output**: Results are compiled into professional PDF and JSON formats

### 10-Point VC Checklist Questions:
1. Problem Identification and Severity
2. Solution Efficacy
3. Market Opportunity
4. Product and Technology Viability
5. Monetization Strategy
6. Competitive Analysis and Differentiation
7. Go-to-Market Strategy & Market Validation
8. Team Assessment
9. Traction and Progress
10. Funding Request and Capital Allocation

## Model Configuration

The system expects your GGUF model at:
```
P:\progs\vsCode\VSProjects\NLP\models\liquid\LFM2-1.2B-Q8_0.gguf
```

You can modify the model path in `llm_summarization.py` if needed.

## Output

The system generates comprehensive analysis results:

### 1. Terminal Progress
```
=== STARTING VC PITCH ANALYSIS (10-EPOCH APPROACH) ===
📋 Preparing EPOCH 1: Question 1
🚀 STARTING EPOCH 1 ANALYSIS...
First epoch finished
📋 Preparing EPOCH 2: Question 2
🚀 STARTING EPOCH 2 ANALYSIS...
Second epoch finished
...
🎉 VC PITCH ANALYSIS COMPLETED SUCCESSFULLY (10 EPOCHS)
```

### 2. PDF Report
- Professional VC analysis with structured sections
- Light/dark theme support
- Formatted headings and bullet points
- Clean, readable layout

### 3. JSON Analysis
- Machine-readable structured data
- Individual question results with metadata
- Timestamps and analysis method information
- Perfect for data processing and archival

### JSON Structure
```json
{
  "analysis_metadata": {
    "timestamp": "2025-01-26T14:30:52",
    "total_questions": 10,
    "analysis_method": "10-epoch individual question processing",
    "llm_model": "Local GGUF Model"
  },
  "questions": [
    {
      "question_number": 1,
      "question_title": "Problem Identification and Severity",
      "full_question": "Does the problem represent a critical...",
      "analysis": "The pitch identifies...",
      "timestamp": "2025-01-26T14:30:52",
      "epoch": 1
    }
  ]
}
```

---

**Simple, powerful, terminal-based VC pitch analysis with no web servers, APIs, or complex dependencies!**