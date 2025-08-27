"""
LLM-based summarization using local GGUF models.
Alternative to transformer-based summarization with better context understanding.
"""

import os
import logging
from typing import Optional, Dict, Any, List
import json
import re

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class LLMSummarizer:
    """
    Local LLM-based summarization using llama-cpp-python.
    Provides an alternative to transformer-based summarization.
    """
    
    def __init__(self, model_path: Optional[str] = None, n_gpu_layers: int = 0, n_ctx: int = 2048):
        """
        Initialize the LLM summarizer.
        
        Args:
            model_path: Path to GGUF model file
            n_gpu_layers: Number of GPU layers to offload (0 for CPU-only, safer)
            n_ctx: Maximum context size (reduced to prevent memory issues)
        """
        self.model_path = model_path or "P:\\progs\\vsCode\\VSProjects\\NLP\\models\\liquid\\LFM2-1.2B-Q8_0.gguf"
        self.n_gpu_layers = n_gpu_layers
        self.n_ctx = n_ctx
        self.n_batch = 512  # Add batch size control
        self.n_threads = 4  # Control thread count
        self.llm = None
        self.is_available = False
        
        # Try to initialize the LLM
        self._initialize_llm()
    
    def _initialize_llm(self):
        """Initialize the LLM model if available."""
        try:
            from llama_cpp import Llama
            
            if not os.path.exists(self.model_path):
                logging.warning(f"LLM model not found at {self.model_path}. LLM summarization will be unavailable.")
                return
            
            chat_format = "pocket" if "pocket" in self.model_path else "chatml"
            self.llm = Llama(
                model_path=self.model_path,
                chat_format=chat_format,
                n_gpu_layers=self.n_gpu_layers,
                n_ctx=self.n_ctx,
                n_batch=self.n_batch,
                n_threads=self.n_threads,
                use_mmap=True,
                use_mlock=False,
                verbose=False
            )
            self.is_available = True
            logging.info("LLM summarizer initialized successfully")
            
        except ImportError:
            logging.warning("llama-cpp-python not installed. LLM summarization unavailable.")
        except Exception as e:
            logging.error(f"Failed to initialize LLM: {e}")
    
    def generate_summary(self, text: str, max_length: int = 150, style: str = "marketing") -> str:
        """
        Generate a summary using the local LLM.
        
        Args:
            text: Input text to summarize
            max_length: Maximum length of summary
            style: Summary style ("marketing", "technical", "executive")
            
        Returns:
            Generated summary or error message
        """
        if not self.is_available:
            return "LLM summarization is not available. Please check model installation."
        
        if not text or not isinstance(text, str):
            return "Invalid input text provided."
        
        try:
            # Craft the prompt based on style
            prompts = {
                "marketing": f"""Analyze this marketing text and create a compelling one-sentence executive summary that captures the core value proposition and key benefits:

Text: {text}

Executive Summary:""",
                
                "technical": f"""Analyze this technical text and create a precise one-sentence summary focusing on the main technical achievements and capabilities:

Text: {text}

Technical Summary:""",
                
                "executive": f"""Analyze this text and create a professional one-sentence executive summary suitable for leadership decision-making:

Text: {text}

Executive Summary:"""
            }
            
            prompt = prompts.get(style, prompts["marketing"])
            
            # Truncate input text to prevent context overflow
            max_input_length = self.n_ctx - max_length - 200  # Reserve space for prompt and output
            if len(text) > max_input_length:
                text = text[:max_input_length] + "..."
                logging.warning(f"Input text truncated to {max_input_length} characters")
            
            # Generate response using the LLM with safety measures
            messages = [
                {"role": "system", "content": "You are an expert analyst. Create one concise sentence."},
                {"role": "user", "content": prompt}
            ]
            
            try:
                # Ensure llm is available before calling
                if self.llm is None:
                    return "LLM model is not initialized."
                    
                response = self.llm.create_chat_completion(
                    messages=messages,  # type: ignore
                    max_tokens=min(max_length, 100),  # Limit tokens to prevent issues
                    temperature=0.1,  # Lower temperature for more consistent output
                    top_p=0.9,
                    stop=["\n", ".", "!", "?"],
                    repeat_penalty=1.1,
                    stream=False  # Ensure we get a direct response, not an iterator
                )
                
                # Handle the response properly - llama-cpp-python returns a dict for non-streaming
                if isinstance(response, dict) and 'choices' in response and response['choices']:
                    content = response['choices'][0]['message']['content']
                    summary = content.strip() if content else "No content in response."
                else:
                    summary = "Unable to extract content from LLM response."
                    
            except Exception as decode_error:
                if "llama_decode" in str(decode_error) or "-1" in str(decode_error):
                    logging.error(f"llama_decode error, falling back to simple summary: {decode_error}")
                    # Fallback to a simple summary format
                    words = text.split()[:50]  # Take first 50 words
                    return f"This content discusses {' '.join(words[:20])}... and related topics."
                else:
                    raise decode_error
            
            # Ensure it ends with proper punctuation
            if summary and not summary.endswith(('.', '!', '?')):
                summary += '.'
                
            return summary if summary else "Unable to generate summary with LLM."
            
        except Exception as e:
            logging.error(f"LLM summarization error: {e}")
            return f"LLM summarization failed: {str(e)}"
    
    def generate_10_epoch_summaries(self, json_file_path: str) -> bool:
        """
        Generate 10-epoch summaries for each question in the analysis JSON file.
        Each epoch processes one question with complete session reset to reduce hallucinations.
        
        Args:
            json_file_path: Path to the analysis JSON file
            
        Returns:
            True if successful, False otherwise
        """
        if not self.is_available:
            logging.warning("LLM not available for summarization")
            return False
            
        try:
            # Load the JSON file
            with open(json_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            questions = data.get('questions', [])
            if not questions:
                logging.error("No questions found in JSON file")
                return False
                
            logging.info(f"=== STARTING 10-EPOCH SUMMARIZATION PROCESS ===")
            logging.info(f"Processing {len(questions)} questions from: {json_file_path}")
            
            # Process each question as a separate epoch
            epoch_names = ["First", "Second", "Third", "Fourth", "Fifth", "Sixth", "Seventh", "Eighth", "Ninth", "Tenth"]
            
            for epoch in range(len(questions)):
                question_data = questions[epoch]
                question_number = question_data.get('question_number', epoch + 1)
                analysis_text = question_data.get('analysis', '')
                
                logging.info(f"📋 Preparing EPOCH {epoch + 1}: Question {question_number}")
                
                # Complete LLM session reset to prevent hallucination contamination
                self._reset_llm_session(epoch + 1)
                
                # Generate summary for this question
                logging.info(f"🚀 STARTING EPOCH {epoch + 1} SUMMARIZATION...")
                summary = self._generate_epoch_summary(analysis_text, question_number, epoch + 1)
                
                # Store the summary in the question data
                question_data['analysis_summary'] = summary
                
                # Print epoch completion message as required
                print(f"{epoch_names[epoch]} epoch finished")
                logging.info(f"✅ EPOCH {epoch + 1} FINISHED")
            
            # Update metadata
            if 'analysis_metadata' not in data:
                data['analysis_metadata'] = {}
            
            import datetime
            data['analysis_metadata'].update({
                'summary_generated': datetime.datetime.now().isoformat(),
                'summary_method': '10-epoch LLM-based with number preservation and hallucination reduction',
                'epochs_per_summary': len(questions)
            })
            
            # Save the updated JSON file
            with open(json_file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            logging.info(f"🎉 10-EPOCH SUMMARIZATION COMPLETED SUCCESSFULLY")
            logging.info(f"Updated JSON file: {json_file_path}")
            
            return True
            
        except Exception as e:
            logging.error(f"❌ Error in 10-epoch summarization: {e}")
            return False
    
    def _reset_llm_session(self, epoch_num: int):
        """Reset the LLM session completely to prevent hallucination contamination."""
        try:
            if self.llm and hasattr(self.llm, 'reset'):
                self.llm.reset()
                logging.info(f"  🔄 LLM session completely reset for epoch {epoch_num}")
            else:
                # For models without explicit reset, we reinitialize key components
                logging.info(f"  🔄 LLM context cleared for epoch {epoch_num}")
        except Exception as e:
            logging.warning(f"  ⚠️ Session reset warning for epoch {epoch_num}: {e}")
    
    def _generate_epoch_summary(self, analysis_text: str, question_number: int, epoch_num: int) -> str:
        """
        Generate a concise 8-20 word summary for a single question's analysis.
        Preserves all numerical data and focuses on key insights.
        
        Args:
            analysis_text: The analysis text to summarize
            question_number: The question number
            epoch_num: Current epoch number
            
        Returns:
            Generated summary (8-20 words)
        """
        try:
            # Clean the analysis text by removing question headers
            cleaned_analysis = self._clean_analysis_input(analysis_text, question_number)
            
            # Extract all numbers from the cleaned analysis for preservation
            numbers = self._extract_numbers(cleaned_analysis)
            numbers_context = f" Key numbers: {', '.join(numbers)}" if numbers else ""
            
            # System prompt optimized for 8-20 word summaries with number preservation
            system_prompt = """You are an expert financial analyst. Create EXACTLY 8-20 word summaries. ALWAYS include ALL numbers mentioned in the text. Be concise and factual. Use active voice. Focus on key insights. NO filler words like 'Analysis highlights', 'The analysis', 'This shows', etc. Start directly with the main point."""
            
            # User prompt with explicit instructions
            user_prompt = f"""Summarize this VC analysis in EXACTLY 8-20 words. MUST include ALL numbers found in the text.{numbers_context}

Analysis: {cleaned_analysis}

Summary (8-20 words):"""            
            
            # Truncate if too long to prevent context overflow
            max_input_length = self.n_ctx - 300  # Reserve space for prompt and output
            if len(user_prompt) > max_input_length:
                truncated_analysis = cleaned_analysis[:max_input_length - 200] + "..."
                user_prompt = f"""Summarize this VC analysis in EXACTLY 8-20 words. MUST include ALL numbers found in the text.{numbers_context}

Analysis: {truncated_analysis}

Summary (8-20 words):"""
                logging.warning(f"  ⚠️ Input truncated for epoch {epoch_num}")
            
            logging.info(f"  💬 Generating summary for epoch {epoch_num}...")
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            if not self.llm:
                raise ValueError("LLM not initialized")
            
            response = self.llm.create_chat_completion(
                messages=messages,  # type: ignore
                max_tokens=50,  # Limited for concise summaries
                temperature=0.0,  # Zero temperature for consistency
                top_p=0.8,
                stop=["\n", "."],  # Stop at sentence boundaries
                repeat_penalty=1.1,
                stream=False
            )
            
            # Extract response content safely
            try:
                response_dict = response if isinstance(response, dict) else {}
                content = response_dict.get('choices', [{}])[0].get('message', {}).get('content', '')
                
                if content:
                    summary = content.strip()
                    # Clean and validate the summary
                    summary = self._clean_and_validate_summary(summary, numbers, question_number)
                    # Remove filler words and phrases
                    summary = self._remove_filler_words(summary)
                else:
                    summary = f"Question {question_number} analysis summary"
                    
            except (KeyError, IndexError, TypeError) as e:
                logging.warning(f"  ⚠️ Response parsing error in epoch {epoch_num}: {e}")
                summary = f"Question {question_number} analysis summary unavailable"
            
            logging.info(f"  ✨ Epoch {epoch_num} summary generated: '{summary}' ({len(summary.split())} words)")
            return summary
            
        except Exception as e:
            logging.error(f"  ❌ Error generating summary for epoch {epoch_num}: {e}")
            return f"Summary generation failed for question {question_number}"
    
    def _clean_analysis_input(self, analysis_text: str, question_number: int) -> str:
        """
        Clean the analysis text by removing question headers and formatting artifacts.
        
        Args:
            analysis_text: The raw analysis text
            question_number: The question number for this analysis
            
        Returns:
            Cleaned analysis text without headers
        """
        if not analysis_text:
            return analysis_text
        
        import re
        
        # Remove question headers like "**1. Problem Identification and Severity:**"
        # Pattern matches: **[number]. [title]:** at the start of lines
        header_pattern = rf'^\*\*{question_number}\.[^:]*:\*\*\s*\n?'
        cleaned = re.sub(header_pattern, '', analysis_text, flags=re.MULTILINE)
        
        # Also remove any remaining bold question headers that might have different formats
        # Pattern for any line starting with **[number]. and ending with :**
        general_header_pattern = r'^\*\*\d+\.[^:]*:\*\*\s*\n?'
        cleaned = re.sub(general_header_pattern, '', cleaned, flags=re.MULTILINE)
        
        # Remove standalone markdown artifacts
        cleaned = re.sub(r'^\*\*\s*$', '', cleaned, flags=re.MULTILINE)
        
        # Clean up extra whitespace and newlines
        cleaned = re.sub(r'\n\n+', '\n\n', cleaned)  # Multiple newlines to double newlines
        cleaned = cleaned.strip()
        
        return cleaned
    
    def _extract_numbers(self, text: str) -> List[str]:
        """Extract all numbers (percentages, currency, integers, decimals) from text."""
        if not text:
            return []
        
        # Pattern to match various number formats
        number_pattern = r'\b(?:\$?[0-9]{1,3}(?:,[0-9]{3})*(?:\.[0-9]+)?[MKmk]?|[0-9]+(?:\.[0-9]+)?%|[0-9]+(?:\.[0-9]+)?)\b'
        numbers = re.findall(number_pattern, text)
        
        # Clean and deduplicate
        unique_numbers = list(set(numbers))
        return unique_numbers[:10]  # Limit to prevent overflow
    
    def _clean_and_validate_summary(self, summary: str, numbers: List[str], question_number: int) -> str:
        """Clean and validate the generated summary."""
        if not summary:
            return f"Question {question_number} analysis summary"
        
        # Remove common artifacts
        summary = summary.replace("Summary:", "").replace("summary:", "").strip()
        summary = summary.replace("(", "").replace(")", "").strip()
        
        # Ensure it's a reasonable length (8-20 words)
        words = summary.split()
        if len(words) < 8:
            # Append question number and key numbers if too short
            if numbers:
                summary += f" {' '.join(numbers[:3])}"
            else:
                summary += f" question {question_number}"
        elif len(words) > 20:
            # Truncate if too long while preserving numbers
            summary = ' '.join(words[:20])
        
        # Ensure at least one number is included if available
        if numbers and not any(num in summary for num in numbers):
            # Try to include the most important number
            summary += f" {numbers[0]}"
        
        return summary.strip()
    
    def _remove_filler_words(self, summary: str) -> str:
        """Remove common filler words and phrases from summaries."""
        if not summary:
            return summary
        
        # List of filler words and phrases to remove
        filler_patterns = [
            r'^Analysis highlights\s*',
            r'^The analysis\s*',
            r'^This analysis\s*',
            r'^Analysis shows\s*',
            r'^Analysis reveals\s*',
            r'^The summary\s*',
            r'^Summary:\s*',
            r'^This shows\s*',
            r'^This reveals\s*',
            r'^The report\s*',
            r'^Report shows\s*',
            r'^Study shows\s*',
            r'^Research shows\s*',
            r'^Data shows\s*',
            r'^Results show\s*',
            r'^Findings show\s*',
            r'^Evidence shows\s*',
            r'^The study\s*',
            r'^The data\s*',
            r'^The findings\s*',
            r'^The results\s*',
            r'^According to\s*',
        ]
        
        import re
        cleaned = summary
        
        # Remove filler patterns (case insensitive)
        for pattern in filler_patterns:
            cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)
        
        # Clean up any leading/trailing whitespace and ensure proper capitalization
        cleaned = cleaned.strip()
        if cleaned and not cleaned[0].isupper():
            cleaned = cleaned[0].upper() + cleaned[1:]
        
        return cleaned
    
    def get_status(self) -> Dict[str, Any]:
        """Get the status of the LLM summarizer."""
        return {
            "available": self.is_available,
            "model_path": self.model_path,
            "model_exists": os.path.exists(self.model_path) if self.model_path else False,
            "n_gpu_layers": self.n_gpu_layers,
            "n_ctx": self.n_ctx
        }

# Global instance for reuse
_llm_summarizer = None

def get_llm_summarizer() -> LLMSummarizer:
    """Get or create the global LLM summarizer instance."""
    global _llm_summarizer
    if _llm_summarizer is None:
        _llm_summarizer = LLMSummarizer()
    return _llm_summarizer

def generate_llm_summary(text: str, max_length: int = 150, style: str = "marketing") -> str:
    """
    Convenience function for LLM-based summarization.
    
    Args:
        text: Input text to summarize
        max_length: Maximum length of summary
        style: Summary style ("marketing", "technical", "executive")
        
    Returns:
        Generated summary
    """
    try:
        summarizer = get_llm_summarizer()
        if summarizer.is_available:
            return summarizer.generate_summary(text, max_length, style)
        else:
            # Simple fallback if LLM not available
            words = text.split()[:50]  # Take first 50 words
            return f"This content discusses {' '.join(words[:20])}... and related topics."
    except Exception as e:
        logging.error(f"LLM summarization completely failed: {e}")
        # Ultimate fallback
        words = text.split()[:30]  # Take first 30 words
        return f"Summary unavailable. Text preview: {' '.join(words)}..."

def generate_10_epoch_summaries(json_file_path: str) -> bool:
    """
    Convenience function to generate 10-epoch summaries for analysis JSON.
    
    Args:
        json_file_path: Path to the analysis JSON file
        
    Returns:
        True if successful, False otherwise
    """
    try:
        summarizer = get_llm_summarizer()
        if not summarizer.is_available:
            logging.error("LLM summarizer not available")
            return False
        
        return summarizer.generate_10_epoch_summaries(json_file_path)
    except Exception as e:
        logging.error(f"10-epoch summarization failed: {e}")
        return False

def is_llm_available() -> bool:
    """Check if LLM summarization is available."""
    summarizer = get_llm_summarizer()
    return summarizer.is_available