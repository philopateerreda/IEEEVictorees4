

### 10-Epoch NER Analysis System

A high-accuracy Named Entity Recognition system for VC pitch documents using **multi-pass focused extraction**.

#### Key Features
*   **10 focused epochs** targeting specific business sections
*   **Context reset** between epochs to prevent data contamination
*   **Reduced hallucinations** through stateless extraction
*   **Single merged output** file with comprehensive results

#### Epoch Sections
1. **Problem Extraction** - Core business problems and pain points
2. **Solution Extraction** - Proposed solutions and value propositions
3. **Funding Information** - Funding requirements and financial projections
4. **Market Analysis** - Industry analysis and target markets
5. **Company Profile** - Company overview and mission
6. **Product & Technology** - Technology stack and product features
7. **Personnel** - Team members and advisors
8. **Business Performance & Traction** - Financial performance and metrics
9. **Strategy & Goals** - Business strategy and objectives
10. **Mentioned Organizations** - External organizations and partners

#### How to Run

**Command Line:**
```bash
python ner_analysis.py input_pitch.txt -o output_directory
```

**Required Arguments:**
- `input_file`: Path to text file containing pitch content

**Optional Arguments:**
- `-o, --output_dir`: Output directory (default: ./10_Epoch_NER/)
- `-m, --model_path`: Custom GGUF model path

**API Usage:**
```python
from ner_analysis import analyze_pitch_10_epochs

results = analyze_pitch_10_epochs(
    pitch_text="Your pitch content here",
    output_dir="./results/",
    model_path="./models/custom.gguf"
)
```

#### Output
Creates `final_extraction_result.json` containing all extracted entities from all 10 sections in a single merged file.