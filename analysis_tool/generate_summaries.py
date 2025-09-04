"""
VC Analysis Summary Generator
Generates concise summaries for analysis questions and updates JSON files.
"""

import json
import os
import re
import logging
import datetime
import sys
from typing import Dict, Any, List, Optional
from llm_summarization import get_llm_summarizer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class AnalysisSummarizer:
    """Generates concise summaries for VC analysis questions using the same LLM."""
    
    def __init__(self, model_name: Optional[str] = None):
        """Initialize the summarizer with LLM access."""
        self.llm_summarizer = get_llm_summarizer(model_name=model_name)
        
    def extract_numbers_from_text(self, text: str) -> List[str]:
        """Extract all numbers, percentages, and monetary values from text."""
        # Pattern to match various number formats including percentages, money, decimals
        number_patterns = [
            r'\$[\d,]+(?:\.\d+)?[MmBbKk]?',  # Money: $5M, $2.2M, $75k, etc.
            r'\d+(?:\.\d+)?%',                # Percentages: 15%, 35%, 60%
            r'\d+(?:,\d{3})*(?:\.\d+)?',      # Regular numbers: 50, 3, 75,000
        ]
        
        numbers = []
        for pattern in number_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            numbers.extend(matches)
        
        return numbers
    
    def generate_summary_10_epoch(self, question_title: str, analysis_text: str) -> str:
        """
        Generate an 8-15 word summary using 3 epochs to reduce hallucinations.
        Each epoch generates a summary candidate, then the best one is selected.
        
        Args:
            question_title: The title of the question
            analysis_text: The full analysis text
            
        Returns:
            Best summary from 3 epochs with preserved numbers
        """
        if not self.llm_summarizer.is_available:
            # Fallback summary if LLM not available
            numbers = self.extract_numbers_from_text(analysis_text)
            number_text = " ".join(numbers) if numbers else ""
            return f"{question_title}: Key issues identified {number_text}".strip()[:15]
        
        # Extract all numbers from the analysis
        numbers = self.extract_numbers_from_text(analysis_text)
        numbers_context = f" Important numbers to preserve: {', '.join(numbers)}" if numbers else ""
        
        # Staging area to collect all epoch summaries
        epoch_summaries = []
        epoch_names = ["First", "Second", "Third"]
        
        logging.info(f"🔄 Starting 3-epoch summarization for: {question_title}")
        
        # Process 3 epochs for better summary quality
        for epoch in range(3):
            logging.info(f"📋 EPOCH {epoch + 1}: Generating summary candidate...")
            
            try:
                # Execute single epoch with context reset
                summary_candidate = self._execute_summary_epoch(
                    question_title, analysis_text, numbers, numbers_context, epoch_num=epoch + 1
                )
                
                # Store candidate
                epoch_summaries.append(summary_candidate)
                
                # Print epoch completion
                print(f"{epoch_names[epoch]} summary epoch finished")
                logging.info(f"✅ SUMMARY EPOCH {epoch + 1} FINISHED: {summary_candidate}")
                
            except Exception as e:
                logging.warning(f"⚠️ Epoch {epoch + 1} failed: {e}")
                # Create fallback for this epoch
                fallback = self._create_fallback_summary(question_title, analysis_text, numbers)
                epoch_summaries.append(fallback)
        
        # Select the best summary from all epochs
        logging.info("🎯 Selecting best summary from 3 epoch candidates...")
        best_summary = self._select_best_summary(epoch_summaries, numbers)
        
        logging.info(f"🏆 Final selected summary: {best_summary}")
        return best_summary
    
    def _execute_summary_epoch(self, question_title: str, analysis_text: str, numbers: List[str], numbers_context: str, epoch_num: int = 0) -> str:
        """
        Execute a single summary epoch with complete context reset to reduce hallucinations.
        """
        try:
            logging.info(f"  🤖 Initializing LLM for summary epoch {epoch_num}...")
            
            # Complete LLM context/session reset to prevent hallucinations
            if self.llm_summarizer.llm and hasattr(self.llm_summarizer.llm, 'reset'):
                self.llm_summarizer.llm.reset()
                logging.info(f"  🔄 LLM context completely reset for summary epoch {epoch_num}")
            
            # Create focused prompt for summary generation
            prompt = f"""Create a precise 8-15 word summary of this analysis that MUST include ALL specific numbers, percentages, and dollar amounts mentioned.

Question: {question_title}

Analysis: {analysis_text}

{numbers_context}

Requirements:
- Exactly 8-15 words
- Include ALL numbers: {', '.join(numbers) if numbers else 'none'}
- Start with question topic
- Be specific and actionable
- Focus on key insights and findings

Summary:"""

            from typing import cast, Any
            
            messages = [
                {"role": "system", "content": "You are an expert at creating precise, number-inclusive summaries. Always preserve ALL numerical data. Response must be exactly 8-15 words. Be direct and specific."},
                {"role": "user", "content": prompt}
            ]
            
            logging.info(f"  ⏳ Processing summary epoch {epoch_num} with fresh LLM context...")
            if self.llm_summarizer.llm is None:
                raise ValueError("LLM not initialized")
            
            response = self.llm_summarizer.llm.create_chat_completion(
                messages=cast(Any, messages),  # Type cast to avoid strict typing issues
                max_tokens=30,  # Slightly more tokens for 8-15 words
                temperature=0.1,  # Low temperature for consistent output
                top_p=0.9,
                stop=["\n", ".", "Summary:", "Analysis:"],
                stream=False
            )
            
            # Handle response safely with type casting
            try:
                response_dict = cast(Dict[str, Any], response)
                content = response_dict['choices'][0]['message']['content']
                if content:
                    result = content.strip()
                    # Clean unwanted instruction text that might appear in output
                    result = self._clean_summary_response(result)
                else:
                    result = ''
            except (KeyError, IndexError, TypeError) as e:
                logging.warning(f"  ⚠️ Unexpected response format in summary epoch {epoch_num}: {e}")
                result = f"Summary epoch {epoch_num} error"
            
            # Ensure all numbers are included and validate word count
            validated_summary = self._validate_and_fix_summary(result, question_title, numbers)
            
            logging.info(f"  ✨ Summary epoch {epoch_num} response: {validated_summary}")
            return validated_summary
            
        except Exception as e:
            logging.error(f"  ❌ Error in summary epoch {epoch_num} execution: {e}")
            return self._create_fallback_summary(question_title, analysis_text, numbers)
    
    def _clean_summary_response(self, response: str) -> str:
        """Clean the LLM summary response to remove unwanted instruction text."""
        if not response:
            return response
            
        # Remove common instruction patterns
        unwanted_patterns = [
            "Summary:",
            "Analysis:", 
            "Create a precise",
            "Requirements:",
            "Question:"
        ]
        
        cleaned = response
        for pattern in unwanted_patterns:
            cleaned = re.sub(r'^' + re.escape(pattern), '', cleaned, flags=re.IGNORECASE).strip()
        
        return cleaned.strip()
    
    def _validate_and_fix_summary(self, summary: str, question_title: str, numbers: List[str]) -> str:
        """Validate and fix summary to ensure it meets requirements."""
        if not summary:
            return self._create_fallback_summary(question_title, "", numbers)
        
        # Clean up the summary
        summary = re.sub(r'^(Summary:|Analysis:)', '', summary).strip()
        
        # Ensure all numbers are included
        missing_numbers = []
        for num in numbers:
            if num not in summary:
                missing_numbers.append(num)
        
        if missing_numbers:
            # Add missing numbers to summary
            summary = f"{summary} {' '.join(missing_numbers)}"
        
        # Ensure word count is within range (8-15 words)
        words = summary.split()
        if len(words) > 15:
            summary = ' '.join(words[:15])
        elif len(words) < 8:
            # Pad with context if too short
            topic = question_title.split(':')[0] if ':' in question_title else question_title
            summary = f"{topic}: {summary}"
            words = summary.split()
            if len(words) > 15:
                summary = ' '.join(words[:15])
        
        return summary
    
    def _select_best_summary(self, epoch_summaries: List[str], numbers: List[str]) -> str:
        """Select the best summary from all epoch candidates based on quality criteria."""
        if not epoch_summaries:
            return "No summary generated"
        
        # Score each summary based on multiple criteria
        best_summary = ""
        best_score = -1
        
        for summary in epoch_summaries:
            score = self._score_summary(summary, numbers)
            logging.info(f"  📊 Summary candidate: '{summary}' (score: {score})")
            
            if score > best_score:
                best_score = score
                best_summary = summary
        
        return best_summary if best_summary else epoch_summaries[0]
    
    def _score_summary(self, summary: str, numbers: List[str]) -> float:
        """Score a summary based on quality criteria."""
        if not summary:
            return 0.0
        
        score = 0.0
        words = summary.split()
        
        # Word count criteria (8-15 is optimal)
        word_count = len(words)
        if 8 <= word_count <= 15:
            score += 3.0
        elif 6 <= word_count <= 17:
            score += 1.0
        
        # Numbers preservation (critical)
        numbers_included = sum(1 for num in numbers if num in summary)
        if numbers:
            numbers_ratio = numbers_included / len(numbers)
            score += numbers_ratio * 4.0  # High weight for number preservation
        else:
            score += 2.0  # Bonus if no numbers needed
        
        # Content quality indicators
        quality_words = ['issues', 'concerns', 'lacks', 'missing', 'needs', 'requires', 'weak', 'strong', 'identified']
        quality_score = sum(1 for word in quality_words if word.lower() in summary.lower())
        score += min(quality_score * 0.5, 2.0)
        
        # Penalize very short or very long summaries
        if word_count < 5:
            score -= 2.0
        elif word_count > 20:
            score -= 1.0
        
        return score
    
    def generate_summary(self, question_title: str, analysis_text: str) -> str:
        """
        Fallback method: Generate a simple summary (kept for compatibility).
        For production use, prefer generate_summary_10_epoch for better quality.
        """
        logging.warning("Using fallback single-epoch summarization. Consider using generate_summary_10_epoch for better quality.")
        
        if not self.llm_summarizer.is_available:
            numbers = self.extract_numbers_from_text(analysis_text)
            number_text = " ".join(numbers) if numbers else ""
            return f"{question_title}: Key issues identified {number_text}".strip()[:15]
        
        # Use single epoch as fallback
        numbers = self.extract_numbers_from_text(analysis_text)
        numbers_context = f" Important numbers to preserve: {', '.join(numbers)}" if numbers else ""
        
        try:
            return self._execute_summary_epoch(question_title, analysis_text, numbers, numbers_context, epoch_num=1)
        except Exception as e:
            logging.warning(f"Single-epoch summary generation failed: {e}")
            return self._create_fallback_summary(question_title, analysis_text, numbers)
    
    def _create_fallback_summary(self, question_title: str, analysis_text: str, numbers: List[str]) -> str:
        """Create a fallback summary when LLM fails."""
        # Extract key words from analysis (excluding common words)
        common_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'been', 'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those'}
        
        words = re.findall(r'\b\w+\b', analysis_text.lower())
        key_words = [w for w in words if w not in common_words and len(w) > 3][:5]
        
        # Create summary
        topic = question_title.split(':')[0] if ':' in question_title else question_title
        numbers_text = ' '.join(numbers[:3])  # Limit to first 3 numbers
        key_text = ' '.join(key_words[:3])
        
        summary = f"{topic}: {key_text} {numbers_text}".strip()
        
        # Ensure 8-15 words
        words = summary.split()
        if len(words) > 15:
            summary = ' '.join(words[:15])
        elif len(words) < 8:
            summary = f"{summary} needs improvement and validation"
            words = summary.split()
            if len(words) > 15:
                summary = ' '.join(words[:15])
        
        return summary
    
    def process_analysis_json(self, json_path: str) -> tuple[str, str]:
        """
        Process the analysis JSON file to add summaries and create text output.
        
        Args:
            json_path: Path to the analysis JSON file
            
        Returns:
            Tuple of (updated_json_path, text_summary_path)
        """
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"JSON file not found: {json_path}")
        
        logging.info(f"Processing analysis JSON: {json_path}")
        
        # Load the JSON data
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if 'questions' not in data:
            raise ValueError("Invalid JSON structure: 'questions' key not found")
        
        # Generate summaries for each question using 3-epoch approach
        summaries = []
        logging.info("🚀 Generating 3-epoch summaries for all questions...")
        
        for i, question in enumerate(data['questions']):
            question_title = question.get('question_title', f"Question {i+1}")
            analysis_text = question.get('analysis', '')
            
            logging.info(f"📋 Starting 3-epoch summarization for: {question_title}")
            
            # Generate summary using 3-epoch approach
            summary = self.generate_summary_10_epoch(question_title, analysis_text)
            
            # Add summary to the question data
            question['analysis_summary'] = summary
            
            # Store for text file
            summaries.append({
                'title': question_title,
                'summary': summary,
                'number': question.get('question_number', i+1)
            })
            
            logging.info(f"    Summary: {summary}")
        
        # Update metadata
        data['analysis_metadata']['summary_generated'] = datetime.datetime.now().isoformat()
        data['analysis_metadata']['summary_method'] = "3-epoch LLM-based with number preservation and hallucination reduction"
        data['analysis_metadata']['epochs_per_summary'] = 3
        
        # Save updated JSON
        updated_json_path = json_path  # Overwrite original
        with open(updated_json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        logging.info(f"Updated JSON saved: {updated_json_path}")
        
        # Create text summary file
        text_summary_path = self._create_text_summary(json_path, summaries)
        
        return updated_json_path, text_summary_path
    
    def _create_text_summary(self, json_path: str, summaries: List[Dict[str, Any]]) -> str:
        """Create a text file with all summaries."""
        # Generate text file path
        base_path = os.path.splitext(json_path)[0]
        text_path = f"{base_path}_summaries.txt"
        
        # Create text content
        content_lines = [
            "VC PITCH ANALYSIS - QUESTION SUMMARIES",
            "=" * 50,
            f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Source: {os.path.basename(json_path)}",
            "",
        ]
        
        # Add each summary with question title first, then analysis summary
        for summary_data in summaries:
            content_lines.extend([
                f"{summary_data['number']}. {summary_data['title']}",
                f"   {summary_data['summary']}",
                ""
            ])
        
        # Write text file
        with open(text_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(content_lines))
        
        logging.info(f"Text summary saved: {text_path}")
        return text_path

def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate high-quality summaries for VC analysis JSON files using 3-epoch approach')
    parser.add_argument('json_path', help='Path to the analysis JSON file')
    parser.add_argument('--output-dir', help='Output directory (defaults to same as input)')
    parser.add_argument('--single-epoch', action='store_true', help='Use single-epoch fallback instead of 3-epoch approach')
    parser.add_argument('--model', type=str, default=None,
                        help='Name of the model to use for summarization (uses default if not specified)')
    
    args = parser.parse_args()
    
    try:
        summarizer = AnalysisSummarizer(model_name=args.model)
        
        if not summarizer.llm_summarizer.is_available:
            print("⚠️ Warning: LLM not available. Using fallback summary generation.")
        
        print(f"📊 Processing: {args.json_path}")
        
        # Store original method reference
        original_method = summarizer.generate_summary_10_epoch
        
        if args.single_epoch:
            print("🔄 Using single-epoch summarization (fallback mode)")
            # Temporarily override the method for single-epoch mode
            summarizer.generate_summary_10_epoch = summarizer.generate_summary
        else:
            print("🚀 Using 3-epoch summarization for optimal quality and speed")
        
        updated_json, text_summary = summarizer.process_analysis_json(args.json_path)
        
        if args.single_epoch:
            # Restore original method
            summarizer.generate_summary_10_epoch = original_method
        
        print(f"✅ Updated JSON: {updated_json}")
        print(f"📄 Text summary: {text_summary}")
        print("🎉 Summary generation completed successfully!")
        
        # Ensure program exits cleanly
        logging.info("Summary generation process completed. Exiting.")
        sys.exit(0)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        logging.error(f"Summary generation failed: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()