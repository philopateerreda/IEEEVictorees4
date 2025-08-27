

### 8-Epoch NER Analysis System

A system for high-accuracy Named Entity Recognition (NER) on VC pitch documents using a **multi-pass focused extraction** method.

It processes a document in eight separate passes ("epochs"), each targeting a specific section. By resetting the LLM's context between each pass, it minimizes errors and cross-section "hallucinations."

#### Key Features
*   **Focused Extraction**: Each epoch targets one of eight sections (e.g., Personnel, Market Analysis, Product & Tech).
*   **Context Reset**: The LLM's memory is wiped between epochs to ensure high accuracy and prevent data contamination.
*   **Reduced Hallucinations**: The focused, stateless approach yields more reliable results than single-pass methods.
*   **Modular Output**: Creates separate JSON files for each section before merging them into a final comprehensive output.

#### Workflow
1.  **Initialize**: Load the input document.
2.  **Iterate & Extract**: For each of the 8 sections, generate a focused prompt and extract the relevant data.
3.  **Reset & Save**: Clear the LLM context and save the section's results to a unique JSON file.
4.  **Merge**: Combine all 8 JSON files into a single, final result.

#### Usage
The system can be run via the command line or imported as a Python module.

**CLI Example:**
```bash
python ner_8_epoch_analysis.py input_pitch.txt -o output_dir
```

**API Example:**
```python
from ner_8_epoch_analysis import analyze_pitch_8_epochs
results = analyze_pitch_8_epochs(pitch_text="...")
```