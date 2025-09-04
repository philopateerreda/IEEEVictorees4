"""
Configuration and prompts for the venture pitch scoring system.
Contains the 10 scoring criteria and their evaluation prompts.
"""

import os
import json

def get_model_path(model_name: str = None) -> str:
    """Load model path from the central models.json config file."""
    # Correctly locate models.json relative to this file
    # __file__ -> scoring_config.py
    # os.path.dirname(__file__) -> .../Scoring
    # os.path.join(..., '..') -> .../IEEEVic
    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models.json'))
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Model config not found at {config_path}")

    with open(config_path, 'r') as f:
        config = json.load(f)

    if model_name is None:
        model_name = config.get("default_model")
        if not model_name:
            raise ValueError("No default model specified in models.json")

    model_info = config.get("models", {}).get(model_name)
    if not model_info or 'path' not in model_info:
        raise ValueError(f"Model '{model_name}' not found or path is missing in models.json")

    return model_info['path']

# Model configuration - dynamically loaded
MODEL_PATH = get_model_path() # Load the default model

# Output configuration
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "scoring_results")

# Scoring criteria with their evaluation prompts
SCORING_CRITERIA = {
    "problem": {
        "name": "Problem Definition & Market Pain",
        "description": "Does the pitch identify a critical, underserved problem with quantifiable impact?",
        "prompt": """Rate "Problem Definition & Market Pain" (1-10).

9-10: Exceptional: A massive, urgent "painkiller" for a specific customer, backed by compelling data or storytelling. A clear "why now?" is articulated. [10, 18, 22]
6-8: Strong: A clear, significant problem with strong evidence of market need for a well-defined customer segment. [1, 9]
4-5: Adequate: A real problem is identified, but its urgency or value is unclear (a "vitamin"). The target customer is too broad.
1-3: Weak: An unconvincing or vague problem. It feels like a solution in search of a problem.

Pitch: {pitch_text}

Score:"""
    },
    
    "solution": {
        "name": "Solution & Value Proposition",
        "description": "Does the solution offer a transformative improvement with clear, measurable benefits?",
        "prompt": """Rate "Solution & Value Proposition" (1-10).

9-10: Exceptional: An elegant, 10x solution that is clearly articulated and demonstrably better than existing alternatives. The value proposition is powerful and obvious. [47]
6-8: Strong: A well-explained and feasible solution that offers a significant improvement over the status quo. [37]
4-5: Adequate: An incremental improvement. The solution is logical but lacks strong differentiation or a compelling "wow" factor.
1-3: Weak: A marginal, overly complex, or unconvincing solution that doesn't clearly address the stated problem.

Pitch: {pitch_text}

Score:"""
    },
    
    "market": {
        "name": "Market Strategy & Sizing",
        "description": "Is the market analysis credible, with a logical strategy to capture a significant share?",
        "prompt": """Rate "Market Strategy & Sizing" (1-10).

9-10: Exceptional: Bottom-up TAM/SAM/SOM shows a massive opportunity. The beachhead is specific, winnable ($100M+), and a clear stepping stone to larger market expansion. [19, 32]
6-8: Strong: Credible TAM >$1B with logical sizing. A well-defined initial market is identified with a clear expansion plan.
4-5: Adequate: TAM is plausible but may be top-down or poorly justified. The beachhead is too niche (<$50M) or overly broad and unfocused.
1-3: Weak: Unrealistic or absent market sizing. No clear go-to-market or customer segment identified.

Pitch: {pitch_text}

Score:"""
    },
    
    "product_tech": {
        "name": "Technology & Product Excellence",
        "description": "Does the technology create a genuine competitive advantage with demonstrable results?",
        "prompt": """Rate "Technology & Product Excellence" (1-10).

9-10: Exceptional: Proprietary technology or a unique application of tech that creates a strong, defensible moat. Backed by proven metrics or a compelling demo.
6-8: Strong: Solid technology with evidence of validation (e.g., successful pilots, strong user feedback). Shows promising early results.
4-5: Adequate: Competent execution using standard technology. The product works but has no significant technical differentiation.
1-3: Weak: Vague technical claims ("AI-powered") with no substance, validation, or a clear product roadmap.

Pitch: {pitch_text}

Score:"""
    },
    
    "go_to_market": {
        "name": "Go-to-Market & Customer Acquisition",
        "description": "Is there a validated, scalable, and capital-efficient path to acquiring customers?",
        "prompt": """Rate "Go-to-Market & Customer Acquisition" (1-10).

9-10: Exceptional: A clear, multi-channel strategy with evidence of early success (e.g., favorable CAC/LTV). Shows a repeatable, scalable, and capital-efficient sales process. [26, 41]
6-8: Strong: A well-defined plan for reaching the initial customer segment with a realistic understanding of the sales cycle and costs. [40]
4-5: Adequate: A generic or single-channel plan (e.g., "paid ads") with little validation or understanding of unit economics.
1-3: Weak: No credible plan beyond "we will do sales and marketing." No understanding of customer acquisition costs.

Pitch: {pitch_text}

Score:"""
    },
    
    "team": {
        "name": "Team & Founder-Market Fit",
        "description": "Does the team have the unique domain expertise, skills, and resilience to win?",
        "prompt": """Rate "Team & Founder-Market Fit" (1-10).

9-10: Exceptional: Obvious founder-market fit with deep, unique insights into the industry. A complete team with a proven track record of execution. [4, 5, 14]
6-8: Strong: Relevant domain and startup experience. The team has the core skills required, though some gaps may exist. [2, 29]
4-5: Adequate: Passionate team but lacks direct experience in the market or key functional areas.
1-3: Weak: Incomplete team, no relevant experience, or clear skills mismatch for the problem they are solving.

Pitch: {pitch_text}

Score:"""
    },
    
    "traction": {
        "name": "Traction & Momentum",
        "description": "Is there compelling evidence of product-market fit and accelerating growth?",
        "prompt": """Rate "Traction & Momentum" (1-10).

9-10: Exceptional: Strong, accelerating metrics that prove product-market fit (e.g., >20% MoM growth, low churn, high engagement). Happy, paying customers are referenceable. [1, 20]
6-8: Strong: Clear evidence of market validation: paying customers, successful pilots with ROI, or strong strategic partnerships. Positive unit economics are emerging. [6, 11]
4-5: Adequate: Early indicators like LOIs, a growing waitlist, or high user sign-ups. Lacks definitive proof of value or willingness to pay.
1-3: Weak: No meaningful traction or relies on vanity metrics (e.g., social media followers). No customer validation.

Pitch: {pitch_text}

Score:"""
    },
    
    "competition": {
        "name": "Competitive Landscape & Positioning",
        "description": "Do they show a deep understanding of the competitive landscape and a winning strategy?",
        "prompt": """Rate "Competitive Dynamics & Positioning" (1-10).

MANDATORY: If no competitors are mentioned, score 1.

9-10: Exceptional: Sophisticated analysis of the competitive landscape, showing a deep understanding of why incumbents fail and how their approach provides a sustainable advantage. [23, 35]
6-8: Strong: Good awareness of direct and indirect competitors with clear, compelling points of differentiation. [27]
4-5: Adequate: Acknowledges competitors but differentiation is weak or unsustainable. Analysis is superficial.
2-3: Weak: Dismissive of competition ("we have no competitors") or demonstrates a poor understanding of the market. [24]

Pitch: {pitch_text}

Score:"""
    },
    
    "defensibility": {
        "name": "Defensibility & Scalability",
        "description": "Does the business model create compounding advantages and increasing returns to scale?",
        "prompt": """Rate "Scalability & Defensive Moats" (1-10).

9-10: Exceptional: Clear evidence of compounding advantages (e.g., network effects, proprietary data, high switching costs) that strengthen with scale.
6-8: Strong: At least one significant and developing moat is identified. A good understanding of how defensibility will emerge over time.
4-5: Adequate: Some defensible elements, but they are primarily based on execution speed rather than structural advantages.
1-3: Weak: No clear defensibility. The business model is linear, easy to replicate, or lacks scaling advantages.

Pitch: {pitch_text}

Score:"""
    },
    
    "financials": {
        "name": "Financials & Capital Strategy",
        "description": "Is there a credible, milestone-driven path to sustainable unit economics and profitability?",
        "prompt": """Rate "Financials & Capital Strategy" (1--10).

9-10: Exceptional: A specific funding request tied to clear, value-inflecting milestones. Demonstrates strong grasp of unit economics and a credible path to profitability. [48]
6-8: Strong: A reasonable funding request with a sensible use-of-funds plan. Key metrics and burn rate are understood. [28]
4-5: Adequate: A generic funding request with vague milestones. Limited understanding of the key drivers of the business.
1-3: Weak: An unrealistic funding request or no clear plan for the capital. No demonstrated path to a sustainable business.

Pitch: {pitch_text}

Score:"""
    }
}

# System prompt for the scoring model
SYSTEM_PROMPT = """You are an expert VC analyst. Score the pitch on the given criterion from 1-10. 

Scoring scale:
1-3: Weak/Missing
4-5: Adequate
6-8: Strong  
9-10: Exceptional

Respond with ONLY a single number (1-10)."""