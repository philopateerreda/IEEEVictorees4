"""
Pitch Feedback System
Provides concise, actionable feedback on pitch sections using local GGUF model.
"""

import os
import logging
import json
import sys
# Ensure parent directory is on the Python path so we can import scoring_config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from typing import Dict, List, Optional
from datetime import datetime
from scoring_config import get_model_path, SCORING_CRITERIA

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class PitchFeedbackGenerator:
    """Generates brief, actionable feedback for pitch sections using local model."""
    
    def __init__(self, model_name: Optional[str] = None):
        """Initialize the feedback generator."""
        self.model_name = model_name or "default"
        self.model_path = get_model_path(model_name)
        self.llm = None
        self.is_available = False
        
        # Optimized for brief feedback generation
        self.n_ctx = 1024
        self.n_gpu_layers = 0
        self.n_threads = 4
        
        self._initialize_llm()
    
    def _initialize_llm(self):
        """Initialize the LLM model for feedback generation."""
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
            logging.info("Feedback LLM initialized successfully")
            
        except ImportError:
            logging.error("llama-cpp-python not installed. Cannot initialize feedback system.")
        except Exception as e:
            logging.error(f"Failed to initialize LLM: {e}")
    
    def generate_section_feedback(self, pitch_text: str, section: str) -> str:
        """Generate brief feedback for a specific pitch section."""
        if not self.is_available:
            return f"❌ Model unavailable - Cannot analyze {section} section"
        
        if section not in SCORING_CRITERIA:
            return f"❌ Unknown section: {section}"
        
        try:
            self._reinitialize_llm()
            
            criterion = SCORING_CRITERIA[section]
            
            # Create focused feedback prompt
            feedback_prompt = self._create_feedback_prompt(pitch_text, criterion, section)
            
            # Generate feedback
            response = self.llm(
                feedback_prompt,
                max_tokens=80,  # Very brief responses
                temperature=0.1,
                top_p=0.9,
                stop=["\n\n", "Score:", "Rating:"],
                echo=False
            )
            
            # Extract feedback
            if isinstance(response, dict) and 'choices' in response and response['choices']:
                feedback = response['choices'][0]['text'].strip()
            else:
                feedback = str(response).strip() if response else ""
            
            # Clean and format feedback
            feedback = self._clean_feedback(feedback, section)
            
            if not feedback or len(feedback) < 10:
                return self._fallback_feedback(pitch_text, section)
            
            return f"📝 {criterion['name']}: {feedback}"
            
        except Exception as e:
            logging.error(f"Error generating feedback for {section}: {e}")
            return self._fallback_feedback(pitch_text, section)
    
    def _create_feedback_prompt(self, pitch_text: str, criterion: Dict, section: str) -> str:
        """Create a focused prompt for generating brief feedback."""
        # Truncate pitch for context limits
        max_pitch_length = 200
        truncated_pitch = pitch_text[:max_pitch_length] if len(pitch_text) > max_pitch_length else pitch_text
        
        base_prompt = f"""Analyze this pitch section for "{criterion['name']}".

Give 1-2 sentence feedback on what's missing or wrong:

Pitch: {truncated_pitch.strip()}

Feedback:"""
        
        return base_prompt
    
    def _clean_feedback(self, feedback: str, section: str) -> str:
        """Clean and format the generated feedback."""
        # Remove common artifacts
        feedback = feedback.replace("Feedback:", "")
        feedback = feedback.replace("Analysis:", "")
        feedback = feedback.strip()
        
        # Ensure it's not too long
        if len(feedback) > 150:
            sentences = feedback.split(". ")
            feedback = sentences[0] + ("..." if len(sentences) > 1 else "")
        
        return feedback
    
    def _reinitialize_llm(self):
        """Reinitialize LLM to prevent context issues."""
        try:
            if hasattr(self, 'llm') and self.llm:
                del self.llm
            
            from llama_cpp import Llama
            
            self.llm = Llama(
                model_path=self.model_path,
                n_gpu_layers=self.n_gpu_layers,
                n_ctx=self.n_ctx,
                n_threads=self.n_threads,
                use_mmap=True,
                use_mlock=False,
                verbose=False
            )
        except Exception as e:
            logging.error(f"Failed to reinitialize LLM: {e}")
    
    def _fallback_feedback(self, pitch_text: str, section: str) -> str:
        """Generate fallback feedback when LLM fails."""
        text_lower = pitch_text.lower()
        
        fallback_messages = {
            "problem": "❌ Missing clear pain point or market problem statement",
            "solution": "❌ Solution unclear or benefits not quantified", 
            "market": "❌ Market size or target customer not specified",
            "product_tech": "❌ No technical differentiation or demo mentioned",
            "go_to_market": "❌ Customer acquisition strategy missing",
            "team": "❌ Team experience or domain expertise not highlighted",
            "traction": "❌ No metrics, customers, or validation mentioned",
            "competition": "❌ Competitive analysis missing" if "compet" not in text_lower else "⚠️ Competition mentioned but differentiation unclear",
            "defensibility": "❌ No moats or competitive advantages identified",
            "financials": "❌ Financial projections or funding needs unclear"
        }
        
        return fallback_messages.get(section, "❌ Section needs improvement")
    
    def generate_comprehensive_feedback(self, pitch_text: str) -> Dict[str, str]:
        """Generate feedback for all scoring sections."""
        feedback_results = {}
        
        print("🔍 Analyzing pitch sections...\n")
        
        for section_key, criterion in SCORING_CRITERIA.items():
            print(f"Analyzing: {criterion['name']}...")
            feedback = self.generate_section_feedback(pitch_text, section_key)
            feedback_results[section_key] = feedback
            print(feedback)
            print()
        
        return feedback_results
    
    def save_feedback_report(self, pitch_text: str, feedback_results: Dict[str, str], filename: Optional[str] = None) -> str:
        """Save feedback results to a file."""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"pitch_feedback_{timestamp}.json"
        
        # Create feedback output directory structure
        feedback_dir = os.path.join(os.path.dirname(__file__), "results")
        os.makedirs(feedback_dir, exist_ok=True)
        
        report_path = os.path.join(feedback_dir, filename)
        
        report_data = {
            "timestamp": datetime.now().isoformat(),
            "pitch_preview": pitch_text[:200] + "..." if len(pitch_text) > 200 else pitch_text,
            "feedback_by_section": feedback_results,
            "model_name": self.model_name,
            "model_path": self.model_path
        }
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        
        print(f"📄 Feedback report saved to: {report_path}")
        return report_path

def analyze_pitch_feedback(pitch_text: str, model_name: Optional[str] = None) -> Dict[str, str]:
    """Convenience function to analyze a pitch and get feedback."""
    generator = PitchFeedbackGenerator(model_name)
    return generator.generate_comprehensive_feedback(pitch_text)

def analyze_pitch_from_file(file_path: str, model_name: Optional[str] = None) -> Dict[str, str]:
    """Analyze a pitch from a text file."""
    try:
        # Handle relative paths from Scoring directory
        if not os.path.isabs(file_path):
            scoring_dir = os.path.dirname(__file__)
            file_path = os.path.join(scoring_dir, file_path)
        
        # Read pitch text from file
        with open(file_path, 'r', encoding='utf-8') as f:
            pitch_text = f.read().strip()
        
        if not pitch_text:
            raise ValueError(f"File {file_path} is empty or contains only whitespace")
        
        print(f"📄 Reading pitch from: {file_path}")
        print(f"📊 Pitch length: {len(pitch_text)} characters")
        
        return analyze_pitch_feedback(pitch_text, model_name)
        
    except FileNotFoundError:
        print(f"❌ File not found: {file_path}")
        return {}
    except Exception as e:
        print(f"❌ Error reading file {file_path}: {e}")
        return {}

def main():
    """Example usage of the feedback system."""
    sample_pitch = """
    Our startup revolutionizes the way people think about productivity.
    We use AI to help teams work better together.
    The market is huge and we have a great team.
    """
    
    print("🚀 Pitch Feedback Generator\n")
    print("Sample pitch analysis:\n")
    
    feedback_results = analyze_pitch_feedback(sample_pitch)
    
    print("\n" + "="*50)
    print("📊 FEEDBACK SUMMARY")
    print("="*50)
    
    for section, feedback in feedback_results.items():
        print(feedback)
    
    # Save report
    generator = PitchFeedbackGenerator()
    generator.save_feedback_report(sample_pitch, feedback_results)

if __name__ == "__main__":
    main()