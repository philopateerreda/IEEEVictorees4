# Pitch Feedback System - Usage Guide

## Overview
The pitch feedback system analyzes your venture pitches across 10 key criteria using your local LLM model and provides very brief, concise feedback on what's wrong or missing.

## Quick Start

### Method 1: Interactive Analysis (PowerShell)
```powershell
Set-Location "P:\progs\vsCode\VSProjects\NLP\IEEEVic\Scoring"
python .\feedback\analyze_pitch.py
# Then paste your pitch text and press Ctrl+Z then Enter
```

### Method 2: One-Liner (PowerShell)
```powershell


# analyze a text file (path is relative to the *feedback* folder)
python .\feedback\analyze_pitch.py "..\..\sample_pitch.txt"
```


## What It Analyzes

The system checks these 10 areas from `scoring_config.py`:

1. **Problem Definition & Market Pain** - Clear, urgent problem?
2. **Solution & Value Proposition** - Transformative improvement?
3. **Market Strategy & Sizing** - Credible market analysis?
4. **Technology & Product Excellence** - Competitive advantage?
5. **Go-to-Market & Customer Acquisition** - Scalable acquisition plan?
6. **Team & Founder-Market Fit** - Domain expertise?
7. **Traction & Momentum** - Evidence of product-market fit?
8. **Competition & Positioning** - Understanding of landscape?
9. **Defensibility & Scalability** - Compounding advantages?
10. **Financials & Capital Strategy** - Path to profitability?

## Output Format

Each section gets:
- ✅ **Brief feedback** (1-2 sentences)
- 🔴 **❌ Issues** - What's missing/wrong
- 🟢 **Strengths** - What works well
- 💾 **Report** - Saved JSON with details

## Model Used
- **Path**: `P:\progs\vsCode\VSProjects\NLP\models\liquid\LFM2-1.2B-Q8_0.gguf`
- **Type**: Local GGUF model (no internet required)
- **Speed**: ~5-10 seconds per section

## Example Output
```
📝 Problem Definition & Market Pain: Lacks clear pain point quantification and urgency demonstration
📝 Solution & Value Proposition: Solution unclear, needs specific benefits and differentiation
📝 Market Strategy & Sizing: Missing TAM/SAM analysis and beachhead market definition
```

## Tips for Better Pitches
- **Quantify everything** - Numbers, percentages, dollar amounts
- **Be specific** - Avoid vague terms like "huge market" or "great team"
- **Show evidence** - Traction metrics, pilot results, customer feedback
- **Address competition** - Never say "no competitors"
- **Clear ask** - Specific funding amount tied to milestones