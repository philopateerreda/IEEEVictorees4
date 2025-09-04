"""
Consolidated Venture Pitch Scoring System.
Evaluates venture pitches across 10 key criteria using local GGUF models.
"""

import os
import logging
import json
import re
from typing import Optional, Dict, Any
from datetime import datetime
from scoring_config import get_model_path, OUTPUT_DIR, SCORING_CRITERIA, SYSTEM_PROMPT

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class VenturePitchScorer:
    """
    Venture pitch scoring system using local GGUF model.
    Evaluates pitches across 10 key venture capital criteria.
    """
    
    def __init__(self, model_name: Optional[str] = None):
        """Initialize the scoring engine."""
        self.model_name = model_name or "default"
        self.model_path = get_model_path(model_name)
        self.llm = None
        self.is_available = False
        
        # LLM configuration for fast, consistent scoring
        self.n_ctx = 2048  # Increased for longer detailed prompts
        self.n_gpu_layers = 0
        self.n_threads = 4
        
        self._initialize_llm()
    
    def _initialize_llm(self):
        """Initialize the LLM model for scoring."""
        try:
            from llama_cpp import Llama
            
            if not os.path.exists(self.model_path):
                logging.error(f"Model not found at {self.model_path}")
                return
            
            self.llm = Llama(
                model_path=self.model_path,
                chat_format="chatml",
                n_gpu_layers=self.n_gpu_layers,
                n_ctx=self.n_ctx,
                n_threads=self.n_threads,
                use_mmap=True,
                use_mlock=False,
                verbose=False
            )
            self.is_available = True
            logging.info("Venture scoring LLM initialized successfully")
            
        except ImportError:
            logging.error("llama-cpp-python not installed. Cannot initialize scoring engine.")
        except Exception as e:
            logging.error(f"Failed to initialize LLM: {e}")
    
    def score_criterion(self, pitch_text: str, criterion_key: str) -> int:
        """Score a pitch on a specific criterion."""
        if not self.is_available:
            logging.error("LLM not available for scoring")
            return 0
        
        if criterion_key not in SCORING_CRITERIA:
            logging.error(f"Unknown criterion: {criterion_key}")
            return 0
        
        try:
            # Reinitialize LLM for each criterion to prevent decode errors
            self._reinitialize_llm()
            
            if not self.llm:
                logging.error("LLM reinitialization failed")
                return self._fallback_score(pitch_text[:500], criterion_key)
            
            criterion = SCORING_CRITERIA[criterion_key]
            
            # Truncate pitch text to prevent context overflow
            max_pitch_length = 300  # Reduced to accommodate longer prompts
            truncated_pitch = pitch_text[:max_pitch_length] if len(pitch_text) > max_pitch_length else pitch_text
            
            prompt = criterion["prompt"].format(pitch_text=truncated_pitch.strip())
            simple_prompt = f"{SYSTEM_PROMPT}\n\n{prompt}"
            
            # Generate score with conservative parameters
            response = self.llm(
                simple_prompt,
                max_tokens=3,
                temperature=0.0,
                top_p=0.9,
                stop=["\n", " ", "."],
                echo=False
            )
            
            # Extract score from response
            if isinstance(response, dict) and 'choices' in response and response['choices']:
                content = response['choices'][0]['text'].strip()
            else:
                content = str(response).strip() if response else ""
            
            # Extract numerical score using regex
            score_match = re.search(r'\b([1-9]|10)\b', content)
            if score_match:
                score = int(score_match.group(1))
                logging.info(f"Scored {criterion['name']}: {score}/10")
                return score
            else:
                logging.warning(f"Could not extract score from: {content}")
                return self._fallback_score(truncated_pitch, criterion_key)
                
        except Exception as e:
            if "llama_decode" in str(e) or "-1" in str(e):
                logging.warning(f"Decode error for {criterion_key}, using fallback scoring")
                return self._fallback_score(pitch_text[:500], criterion_key)
            else:
                logging.error(f"Error scoring {criterion_key}: {e}")
                return 0
    
    def _reinitialize_llm(self):
        """Reinitialize LLM to prevent context issues."""
        old_llm = getattr(self, 'llm', None)
        
        try:
            if hasattr(self, 'llm') and self.llm:
                del self.llm
            
            from llama_cpp import Llama
            
            self.llm = Llama(
                model_path=self.model_path,
                n_gpu_layers=self.n_gpu_layers,
                n_ctx=1024,  # Increased for longer prompts
                n_threads=self.n_threads,
                use_mmap=True,
                use_mlock=False,
                verbose=False
            )
        except Exception as e:
            logging.error(f"Failed to reinitialize LLM: {e}")
            self.llm = old_llm
    
    def _fallback_score(self, pitch_text: str, criterion_key: str) -> int:
        """Fallback scoring based on keyword analysis when LLM fails."""
        try:
            text_lower = pitch_text.lower()
            
            fallback_scores = {
                "problem": self._score_problem_keywords(text_lower),
                "solution": self._score_solution_keywords(text_lower),
                "market": self._score_market_keywords(text_lower),
                "product_tech": self._score_tech_keywords(text_lower),
                "go_to_market": self._score_gtm_keywords(text_lower),
                "team": self._score_team_keywords(text_lower),
                "traction": self._score_traction_keywords(text_lower),
                "competition": self._score_competition_keywords(text_lower),
                "defensibility": self._score_defensibility_keywords(text_lower),
                "financials": self._score_financials_keywords(text_lower)
            }
            
            score = fallback_scores.get(criterion_key, 5)
            logging.info(f"Fallback scored {criterion_key}: {score}/10")
            return score
            
        except Exception as e:
            logging.error(f"Fallback scoring failed for {criterion_key}: {e}")
            return 5
    
    def _score_problem_keywords(self, text: str) -> int:
        """Score problem based on keywords."""
        positive_keywords = ['urgent', 'critical', 'pain', 'problem', 'inefficient', 'costly', 'waste']
        negative_keywords = ['maybe', 'might', 'could', 'nice to have']
        
        positive_count = sum(1 for word in positive_keywords if word in text)
        negative_count = sum(1 for word in negative_keywords if word in text)
        
        base_score = 5 + positive_count - negative_count
        return max(1, min(10, base_score))
    
    def _score_solution_keywords(self, text: str) -> int:
        """Score solution based on keywords."""
        positive_keywords = ['ai', 'algorithm', 'improvement', 'reduce', 'optimize', 'efficient', 'proprietary']
        
        positive_count = sum(1 for word in positive_keywords if word in text)
        percentage_match = re.search(r'(\d+)%', text)
        
        base_score = 5 + positive_count
        if percentage_match and int(percentage_match.group(1)) > 20:
            base_score += 2
        
        return max(1, min(10, base_score))
    
    def _score_market_keywords(self, text: str) -> int:
        """Score market based on keywords."""
        market_indicators = ['billion', 'million', 'growing', 'market', 'opportunity', '%']
        
        indicator_count = sum(1 for word in market_indicators if word in text)
        base_score = 4 + indicator_count
        
        return max(1, min(10, base_score))
    
    def _score_tech_keywords(self, text: str) -> int:
        """Score technology based on keywords."""
        tech_keywords = ['ai', 'iot', 'algorithm', 'ml', 'machine learning', 'scalable', 'platform']
        
        tech_count = sum(1 for word in tech_keywords if word in text)
        base_score = 4 + tech_count
        
        return max(1, min(10, base_score))
    
    def _score_gtm_keywords(self, text: str) -> int:
        """Score go-to-market based on keywords."""
        gtm_keywords = ['pilot', 'customers', 'sales', 'strategy', 'partnerships', 'expansion']
        
        gtm_count = sum(1 for word in gtm_keywords if word in text)
        base_score = 4 + gtm_count
        
        return max(1, min(10, base_score))
    
    def _score_team_keywords(self, text: str) -> int:
        """Score team based on keywords."""
        team_keywords = ['ceo', 'cto', 'experience', 'years', 'expert', 'former', 'mba']
        
        team_count = sum(1 for word in team_keywords if word in text)
        base_score = 4 + team_count
        
        return max(1, min(10, base_score))
    
    def _score_traction_keywords(self, text: str) -> int:
        """Score traction based on keywords."""
        traction_keywords = ['revenue', 'customers', 'growth', 'pilot', 'mrr', 'users']
        
        traction_count = sum(1 for word in traction_keywords if word in text)
        
        revenue_match = re.search(r'\$(\d+)', text)
        customer_match = re.search(r'(\d+)\s*(customers|pilots)', text)
        
        base_score = 4 + traction_count
        if revenue_match:
            base_score += 2
        if customer_match:
            base_score += 1
        
        return max(1, min(10, base_score))
    
    def _score_competition_keywords(self, text: str) -> int:
        """Score competition based on keywords."""
        comp_keywords = ['competitor', 'competitive', 'advantage', 'differentiate', 'unique']
        
        comp_count = sum(1 for word in comp_keywords if word in text)
        base_score = 3 + comp_count * 2
        
        return max(1, min(10, base_score))
    
    def _score_defensibility_keywords(self, text: str) -> int:
        """Score defensibility based on keywords."""
        defense_keywords = ['proprietary', 'patent', 'network effects', 'data', 'moat', 'barrier']
        
        defense_count = sum(1 for word in defense_keywords if word in text)
        base_score = 3 + defense_count * 2
        
        return max(1, min(10, base_score))
    
    def _score_financials_keywords(self, text: str) -> int:
        """Score financials based on keywords."""
        financial_keywords = ['funding', 'profitability', 'revenue', 'investment', 'break-even', 'arr']
        
        financial_count = sum(1 for word in financial_keywords if word in text)
        funding_match = re.search(r'\$(\d+)\s*million', text)
        
        base_score = 4 + financial_count
        if funding_match:
            base_score += 2
        
        return max(1, min(10, base_score))
    
    def score_pitch(self, pitch_text: str) -> Dict[str, Any]:
        """Score a complete venture pitch across all 10 criteria."""
        if not pitch_text or not isinstance(pitch_text, str):
            return {"error": "Invalid pitch text provided"}
        
        if not self.is_available:
            return {"error": "Scoring engine not available"}
        
        logging.info("=== STARTING VENTURE PITCH SCORING ===")
        logging.info(f"Pitch length: {len(pitch_text)} characters")
        
        scores = {}
        total_score = 0
        scored_criteria = 0
        
        # Score each criterion individually
        for criterion_key, criterion_info in SCORING_CRITERIA.items():
            logging.info(f"Scoring: {criterion_info['name']}")
            
            score = self.score_criterion(pitch_text, criterion_key)
            if score > 0:
                scores[criterion_key] = {
                    "name": criterion_info["name"],
                    "description": criterion_info["description"],
                    "score": score
                }
                total_score += score
                scored_criteria += 1
            else:
                scores[criterion_key] = {
                    "name": criterion_info["name"],
                    "description": criterion_info["description"],
                    "score": 0,
                    "error": "Scoring failed"
                }
        
        # Calculate average score
        average_score = total_score / scored_criteria if scored_criteria > 0 else 0
        
        # Create comprehensive results
        results = {
            "pitch_length": len(pitch_text),
            "total_criteria": len(SCORING_CRITERIA),
            "scored_criteria": scored_criteria,
            "total_score": total_score,
            "average_score": round(average_score, 2),
            "scores": scores,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        logging.info(f"=== SCORING COMPLETE: {average_score:.1f}/10 average ===")
        return results
    
    def cleanup(self):
        """Clean up resources."""
        if hasattr(self, 'llm') and self.llm:
            del self.llm
            self.llm = None
        self.is_available = False


def score_pitch_from_file(file_path: str, model_name: Optional[str] = None) -> Dict[str, Any]:
    """
    Score a venture pitch from a text file and save results to JSON.
    
    Args:
        file_path: Path to text file containing the pitch
        
    Returns:
        Dictionary containing scores and analysis
    """
    try:
        # Read pitch from file
        with open(file_path, 'r', encoding='utf-8') as f:
            pitch_text = f.read().strip()
        
        if not pitch_text:
            return {"error": f"No content found in file: {file_path}"}
        
        logging.info(f"Loading pitch from: {file_path}")
        
        # Initialize scorer
        scorer = VenturePitchScorer(model_name)
        
        if not scorer.is_available:
            return {"error": "Scoring system not available. Please check model installation."}
        
        try:
            # Score the pitch
            results = scorer.score_pitch(pitch_text)
            
            if "error" not in results:
                # Ensure output directory exists
                os.makedirs(OUTPUT_DIR, exist_ok=True)
                
                # Generate unique filename with timestamp
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_file = os.path.join(OUTPUT_DIR, f"pitch_score_{timestamp}.json")
                
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
                
                logging.info(f"Results saved to: {output_file}")
                results["output_file"] = output_file
            
            return results
            
        except Exception as e:
            error_msg = f"Scoring failed: {str(e)}"
            logging.error(error_msg)
            return {"error": error_msg}
        
        finally:
            # Clean up resources
            scorer.cleanup()
        
    except FileNotFoundError:
        return {"error": f"File not found: {file_path}"}
    except Exception as e:
        return {"error": f"Failed to read file {file_path}: {str(e)}"}


def print_scoring_summary(results: Dict[str, Any]):
    """Print a formatted summary of scoring results."""
    if "error" in results:
        print(f"❌ Error: {results['error']}")
        return
    
    print("\n" + "="*60)
    print("🎯 VENTURE PITCH SCORING RESULTS")
    print("="*60)
    
    print(f"📊 Overall Score: {results['average_score']:.1f}/10")
    print(f"📝 Total Points: {results['total_score']}/{results['total_criteria']*10}")
    print(f"✅ Criteria Evaluated: {results['scored_criteria']}/{results['total_criteria']}")
    print(f"⏰ Timestamp: {results['timestamp']}")
    
    print(f"\n📋 DETAILED SCORES:")
    print("-" * 60)
    
    for criterion_key, score_data in results["scores"].items():
        score = score_data["score"]
        name = score_data["name"]
        
        # Add emoji based on score
        if score >= 8:
            emoji = "🟢"
        elif score >= 6:
            emoji = "🟡"
        elif score >= 4:
            emoji = "🟠"
        else:
            emoji = "🔴"
        
        print(f"{emoji} {name}: {score}/10")
        
        if "error" in score_data:
            print(f"   ⚠️  {score_data['error']}")
    
    print("-" * 60)
    
    # Add interpretation
    if results['average_score'] >= 8:
        print("🚀 Excellent pitch - Strong investment opportunity!")
    elif results['average_score'] >= 6:
        print("✅ Good pitch - Some areas for improvement")
    elif results['average_score'] >= 4:
        print("⚠️ Average pitch - Significant work needed")
    else:
        print("❌ Weak pitch - Major issues to address")
    
    print("="*60 + "\n")