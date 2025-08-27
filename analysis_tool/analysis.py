"""VC Pitch Analysis Module
Terminal-based VC Pitch Analysis with structured output format.
"""

from llm_summarization import get_llm_summarizer
from typing import Dict, Any, List, Optional
import logging
import os
import datetime
import json

# Configure logging for terminal output
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class VCPitchAnalyzer:
    """Simplified VC Pitch Analyzer with clean structured output."""
    
    def __init__(self):
        """Initialize the VC analyzer with 10-epoch approach: one question per epoch."""
        self.llm_summarizer = get_llm_summarizer()
        self.question_results = []  # Store individual question results for JSON output
        
    def get_vc_checklist_questions(self) -> List[str]:
        """Return the 10-point VC checklist questions."""
        return [
            " Problem Identification and Severity: Does the problem represent a critical, high-priority need for the target customer?",
            " Solution Efficacy: Does the solution address an essential need or provide an incremental benefit?", 
            " Market Opportunity: Is the market large enough to support a company with significant, venture-scale growth potential?",
            " Product and Technology Viability: Is the product concept clear and is its existence demonstrable?",
            " Monetization Strategy: What is the strategy for generating revenue?",
            " Competitive Analysis and Differentiation: Who are the key competitors, and what is the company's sustainable advantage?",
            " Go-to-Market Strategy & Market Validation: What is the plan for reaching and acquiring customers, and is there quantifiable evidence of market acceptance and operational progress?",
            " Your Team Assessment: Does the team possess the requisite expertise and capabilities to execute the business plan successfully?",
            " Traction and Progress: What measurable progress has been made and what metrics demonstrate market validation?",
            " Funding Request and Capital Allocation: What is the requested funding amount and how will it be utilized?"
        ]
    
    def analyze_pitch(self, pitch_text: str) -> str:
        """
        Analyze the pitch using 10 epochs (1 question each) with LLM context reset to reduce hallucinations.
        
        Args:
            pitch_text: The business pitch text to analyze
            
        Returns:
            Formatted string with VC analysis in the specified structure
        """
        if not self.llm_summarizer.is_available:
            logging.warning("LLM not available for analysis")
            return "Error: LLM not available for analysis"
        
        logging.info("=== STARTING VC PITCH ANALYSIS (10-EPOCH APPROACH) ===")
        logging.info(f"Pitch text length: {len(pitch_text)} characters")
        
        questions = self.get_vc_checklist_questions()
        
        # Introduction text
        introduction = "You are tasked with analyzing a business pitch using the 10-Point Venture Capital Checklist, providing a critical yet constructive review that identifies weak or missing areas, explains why they fail to meet investor standards, and offers concrete recommendations for improvement, then concluding with an overall verdict on the pitch's investment readiness, while keeping all company or founder references anonymized with generic placeholders."
        
        # Staging area to collect all epoch results
        epoch_results = []
        self.question_results = []  # Reset for this analysis
        
        # Process 10 epochs of 1 question each
        epoch_names = ["First", "Second", "Third", "Fourth", "Fifth", "Sixth", "Seventh", "Eighth", "Ninth", "Tenth"]
        
        for epoch in range(10):
            question_idx = epoch
            epoch_question = [questions[question_idx]]  # Single question
            current_question = questions[question_idx]
            
            logging.info(f"📋 Preparing EPOCH {epoch + 1}: Question {question_idx + 1}")
            
            # Create prompt for this epoch (single question)
            epoch_prompt = self._create_epoch_prompt(
                introduction, 
                pitch_text, 
                epoch_question, 
                question_idx + 1, 
                question_idx + 1
            )
            
            # Execute epoch with context reset
            logging.info(f"🚀 STARTING EPOCH {epoch + 1} ANALYSIS...")
            epoch_response = self._execute_llm_epoch(epoch_prompt, epoch_num=epoch + 1)
            
            # Store result in staging area
            epoch_results.append(epoch_response)
            
            # Extract question title and store individual result for JSON
            question_parts = current_question.split(': ', 1)
            if len(question_parts) >= 2:
                question_title = question_parts[0].strip()
                full_question = current_question
            else:
                question_title = f"Question {question_idx + 1}"
                full_question = current_question
            
            # Store individual question result for JSON output
            question_result = {
                "question_number": question_idx + 1,
                "question_title": question_title,
                "full_question": full_question,
                "analysis": epoch_response,
                "timestamp": datetime.datetime.now().isoformat(),
                "epoch": epoch + 1
            }
            self.question_results.append(question_result)
            
            # Print the specific epoch finished message as required by specifications
            print(f"{epoch_names[epoch]} epoch finished")
            logging.info(f"✅ EPOCH {epoch + 1} FINISHED")
        
        # Combine all epoch results into final structured output
        logging.info("🔄 Combining all epoch results into final analysis...")
        final_result = self._format_epoch_results(epoch_results)
        
        logging.info("🎉 VC PITCH ANALYSIS COMPLETED SUCCESSFULLY (10 EPOCHS)")
        logging.info(f"Final output length: {len(final_result)} characters")
        
        return final_result
    
    def _create_epoch_prompt(self, introduction: str, pitch_text: str, questions: List[str], start_num: int, end_num: int) -> str:
        """Create a prompt for an epoch of questions."""
        
        # Extract just the question title from the single question
        question = questions[0]  # Single question per epoch
        question_parts = question.split(': ', 1)
        if len(question_parts) >= 2:
            title = question_parts[0].strip()
            question_detail = question_parts[1].strip()
            formatted_question = f"**{title}**\n{question_detail}"
            question_title = title
        else:
            formatted_question = f"**{start_num}. {question}**"
            question_title = f"Question {start_num}"
        
        prompt = f"""{introduction}

**PITCH TO ANALYZE:**
{pitch_text}

**QUESTION TO ANALYZE:**
{formatted_question}

Provide a concise, critical analysis for this question in this exact format:

**{start_num}. {question_title}:**
[Your concise analysis here - Write a concise analysis in 2-3 short, focused sentences. Be direct, specific, and avoid filler words.]
[IMPORTANT: Also mention if this question is too vague to answer properly, or if the required information is missing/unclear in the pitch]

Begin your analysis now:"""
        
        return prompt
    
    def _execute_llm_epoch(self, prompt: str, epoch_num: int = 0) -> str:
        """Execute a single LLM epoch with complete context reset to reduce hallucinations."""
        try:
            logging.info(f"  🤖 Initializing LLM for epoch {epoch_num}...")
            
            # Complete LLM context/session reset to prevent hallucinations
            if self.llm_summarizer.llm and hasattr(self.llm_summarizer.llm, 'reset'):
                self.llm_summarizer.llm.reset()
                logging.info(f"  🔄 LLM context completely reset for epoch {epoch_num}")
            
            logging.info(f"  💬 Sending fresh prompt to LLM (epoch {epoch_num})...")
            logging.debug(f"  Prompt length: {len(prompt)} characters")
            
            from typing import cast, Any
            
            messages = [
                {"role": "system", "content": "You are a seasoned VC analyst. Provide concise, critical analysis. Be direct and specific. Maximum 3-4 sentences per response. Always assess if the question is vague or if the pitch lacks sufficient information to answer properly. Follow the exact format requested. Do not repeat instructions or meta-commentary."},
                {"role": "user", "content": prompt}
            ]
            
            logging.info(f"  ⏳ Processing epoch {epoch_num} with fresh LLM context...")
            if not self.llm_summarizer.llm:
                raise ValueError("LLM not initialized")
                
            response = self.llm_summarizer.llm.create_chat_completion(
                messages=cast(Any, messages),  # Type cast to avoid strict typing issues
                max_tokens=150,  # Reduced for concise responses (single question)
                temperature=0.1,  # Low temperature for consistent formatting
                top_p=0.9,
                stream=False  # Ensure direct response
            )
            
            # Handle response safely with type casting
            try:
                response_dict = cast(Dict[str, Any], response)
                result = response_dict['choices'][0]['message']['content']
                if result:
                    result = result.strip()
                    # Clean unwanted instruction text that might appear in output
                    result = self._clean_llm_response(result)
                else:
                    result = ''
            except (KeyError, IndexError, TypeError) as e:
                logging.warning(f"  ⚠️ Unexpected response format in epoch {epoch_num}: {e}")
                result = f"Error processing epoch {epoch_num} response"
                
            logging.info(f"  ✨ Epoch {epoch_num} response received ({len(result)} characters)")
            
            return result
            
        except Exception as e:
            logging.error(f"  ❌ Error in LLM epoch {epoch_num} execution: {e}")
            return f"Error in epoch {epoch_num} analysis: {str(e)}"
    
    def _clean_llm_response(self, response: str) -> str:
        """Clean the LLM response to remove unwanted instruction text and formatting issues."""
        if not response:
            return response
            
        # Remove common instruction patterns that might appear in output
        unwanted_patterns = [
            "INSTRUCTIONS:",
            "For each question above, provide your analysis",
            "Begin your analysis:",
            "Begin your analysis now:", 
            "Provide your analysis for each question",
            "Follow the exact format requested",
            "This is a fresh analysis session"
        ]
        
        cleaned = response
        for pattern in unwanted_patterns:
            # Remove the pattern and any text on the same line
            lines = cleaned.split('\n')
            cleaned_lines = []
            for line in lines:
                if pattern.lower() not in line.lower():
                    cleaned_lines.append(line)
            cleaned = '\n'.join(cleaned_lines)
        
        # Remove excessive whitespace while preserving structure
        cleaned = '\n'.join(line.strip() for line in cleaned.split('\n') if line.strip())
        
        return cleaned.strip()

    def generate_json_output(self, output_path: str) -> str:
        """Generate a JSON file containing all analysis results."""
        try:
            # Create JSON structure with all questions
            json_data = {
                "analysis_metadata": {
                    "timestamp": datetime.datetime.now().isoformat(),
                    "total_questions": len(self.question_results),
                    "analysis_method": "10-epoch individual question processing",
                    "llm_model": "Local GGUF Model"
                },
                "questions": self.question_results
            }
            
            # Generate JSON filename based on PDF path
            pdf_dir = os.path.dirname(output_path)
            pdf_name = os.path.splitext(os.path.basename(output_path))[0]
            json_filename = f"{pdf_name}_analysis.json"
            json_path = os.path.join(pdf_dir, json_filename)
            
            # Write JSON file
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=2, ensure_ascii=False)
            
            logging.info(f"📊 JSON analysis saved: {json_path}")
            print(f"📊 JSON analysis generated: {json_path}")
            
            return json_path
            
        except Exception as e:
            logging.error(f"❌ JSON generation failed: {e}")
            print(f"❌ JSON generation failed: {e}")
            return ""
    
    def _format_epoch_results(self, epoch_results: List[str]) -> str:
        """Format the final output by combining all epoch results while maintaining structure."""
        
        # Clean up all epoch responses
        cleaned_results = []
        for i, result in enumerate(epoch_results):
            # Remove any potential end tokens and clean whitespace
            cleaned = result.replace(f"<end {i+1}>", "").strip()
            if cleaned:
                cleaned_results.append(cleaned)
        
        # Combine all epoch results into the same final structure
        combined_analysis = "\n\n".join(cleaned_results)
        
        # Create the final formatted output with the same structure as before
        final_output = f"""### **Analysis of the Pitch**

{combined_analysis}"""
        
        return final_output

def generate_output_path(custom_output: Optional[str] = None) -> str:
    """Generate a unique output path with folder creation within the project directory."""
    # Get the directory where this script is located (project directory)
    project_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Always create output in the 'output' folder with unique timestamped subfolder
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if custom_output:
        # Extract filename from custom output and make it unique
        if os.path.isabs(custom_output):
            custom_filename = os.path.basename(custom_output)
        else:
            custom_filename = custom_output
        
        # Remove extension and add timestamp for uniqueness
        base_name = os.path.splitext(custom_filename)[0]
        extension = os.path.splitext(custom_filename)[1] or '.pdf'
        unique_filename = f"{base_name}_{timestamp}{extension}"
        
        # Create unique folder in output directory
        folder_name = f"VC_Analysis_{timestamp}"
        output_dir = os.path.join(project_dir, "output", folder_name)
        output_path = os.path.join(output_dir, unique_filename)
    else:
        # Generate default unique folder and filename
        folder_name = f"VC_Analysis_{timestamp}"
        output_dir = os.path.join(project_dir, "output", folder_name)
        output_filename = f"VC_Pitch_Analysis_{timestamp}.pdf"
        output_path = os.path.join(output_dir, output_filename)
    
    # Create the output directory
    os.makedirs(output_dir, exist_ok=True)
    print(f"📁 Created output directory: {output_dir}")
    
    return output_path

# Main function to run VC analysis from terminal
def main():
    """
    Main function to run VC pitch analysis from the terminal.
    """
    import argparse
    import os
    
    parser = argparse.ArgumentParser(description='VC Pitch Analysis Tool')
    parser.add_argument('pitch_file', help='Path to text file containing the pitch')
    parser.add_argument('-o', '--output', default='VC_Analysis_Report.pdf', 
                       help='Output PDF filename (will be made unique with timestamp and placed in output/ folder. Default: auto-generated unique filename)')
    parser.add_argument('-t', '--theme', choices=['light', 'dark'], default='light',
                       help='PDF theme (default: light)')
    
    args = parser.parse_args()
    
    # Read pitch text from file
    if not os.path.exists(args.pitch_file):
        print(f"Error: File '{args.pitch_file}' not found.")
        return
    
    try:
        with open(args.pitch_file, 'r', encoding='utf-8') as f:
            pitch_text = f.read().strip()
        
        if not pitch_text:
            print(f"Error: File '{args.pitch_file}' is empty.")
            return
        
        print(f"📖 Read pitch from '{args.pitch_file}' ({len(pitch_text)} characters)")
        
        # Create analyzer and run analysis
        analyzer = VCPitchAnalyzer()
        print("🚀 Starting VC pitch analysis...")
        
        analysis_results = analyzer.analyze_pitch(pitch_text)
        
        # Generate appropriate output path - always create unique names in output folder
        if args.output == 'VC_Analysis_Report.pdf':
            # Use default behavior with unique naming
            final_output_path = generate_output_path()
        else:
            # Use custom output but ensure it's unique and in output folder
            final_output_path = generate_output_path(args.output)
        
        # Generate PDF report
        try:
            from pdf_generator import create_vc_analysis_pdf
            create_vc_analysis_pdf(
                pitch_text=pitch_text,
                vc_analysis=analysis_results,
                output_filename=final_output_path,
                theme=args.theme
            )
            print(f"✅ PDF report generated: {final_output_path}")
            
            # Generate JSON output alongside PDF
            json_path = analyzer.generate_json_output(final_output_path)
            if json_path:
                print(f"📊 JSON analysis available: {json_path}")
                
                # Generate 10-epoch summaries for each question
                print("🔄 Generating 10-epoch summaries for each question...")
                try:
                    from llm_summarization import generate_10_epoch_summaries
                    if generate_10_epoch_summaries(json_path):
                        print(f"✅ Question summaries generated and added to JSON: {json_path}")
                    else:
                        print(f"⚠️ Summary generation failed, but JSON analysis is still available")
                except Exception as sum_error:
                    print(f"⚠️ Summary generation error: {sum_error}")
                    print(f"📊 Original JSON analysis still available: {json_path}")
                
        except Exception as e:
            print(f"❌ PDF generation failed: {e}")
            # Still show the analysis results
            print("\n📋 Analysis Results:")
            print("=" * 50)
            print(analysis_results)
        
    except Exception as e:
        print(f"❌ Error processing pitch: {e}")

if __name__ == '__main__':
    main()