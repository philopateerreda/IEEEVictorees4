"""
LLM-based summarization using local GGUF models.
Alternative to transformer-based summarization with better context understanding.
"""

import os
import logging
from typing import Optional, Dict, Any
import json

def get_model_path(model_name: str = None) -> str:
    """Load model path from the central models.json config file."""
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

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class LLMSummarizer:
    """
    Local LLM-based summarization using llama-cpp-python.
    Provides an alternative to transformer-based summarization.
    """
    
    def __init__(self, model_name: Optional[str] = None, n_gpu_layers: int = 0, n_ctx: int = 2048):
        """
        Initialize the LLM summarizer.
        
        Args:
            model_name: The name of the model to use (from models.json).
            n_gpu_layers: Number of GPU layers to offload (0 for CPU-only, safer).
            n_ctx: Maximum context size (reduced to prevent memory issues).
        """
        self.model_path = get_model_path(model_name)
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
    
    def get_status(self) -> Dict[str, Any]:
        """Get the status of the LLM summarizer."""
        return {
            "available": self.is_available,
            "model_path": self.model_path,
            "model_exists": os.path.exists(self.model_path) if self.model_path else False,
            "n_gpu_layers": self.n_gpu_layers,
            "n_ctx": self.n_ctx
        }

# Global instances for reuse, keyed by model name
_llm_summarizers: Dict[str, LLMSummarizer] = {}

def get_llm_summarizer(model_name: Optional[str] = None) -> LLMSummarizer:
    """Get or create a global LLM summarizer instance for a specific model."""
    global _llm_summarizers
    
    # Use default model if none is provided
    key = model_name or "default"

    if key not in _llm_summarizers:
        _llm_summarizers[key] = LLMSummarizer(model_name=model_name)
        
    return _llm_summarizers[key]

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

def is_llm_available() -> bool:
    """Check if LLM summarization is available."""
    summarizer = get_llm_summarizer()
    return summarizer.is_available