"""
10-Epoch Named Entity Recognition (NER) Analysis for VC Pitch Text
Using Multi-Pass Focused Extraction with LLM Context Resets

This module implements the 10-epoch approach for NER extraction, where each epoch
focuses on one specific section with complete LLM context reset between epochs.
"""

import json
import logging
import os
import datetime
import re
from typing import Dict, Any, List, Optional
from llm_summarization import get_llm_summarizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class TenEpochNERAnalyzer:
    """10-Epoch Named Entity Recognition analyzer using focused extraction with LLM context resets."""
    
    def __init__(self, model_name: Optional[str] = None):
        """Initialize the 8-epoch NER analyzer."""
        self.llm_summarizer = get_llm_summarizer(model_name)
        self.epoch_configs = [
            {
                "epoch_number": 1,
                "section_name": "problem_extraction",
                "schema": {
                    "problem_extraction": {
                        "core_problem": "string",
                        "problem_description": "string",
                        "target_audience_affected": ["string"],
                        "pain_points": ["string"],
                        "market_gap_identified": "string"
                    }
                }
            },
            {
                "epoch_number": 2,
                "section_name": "solution_extraction",
                "schema": {
                    "solution_extraction": {
                        "proposed_solution": "string",
                        "solution_description": "string",
                        "key_features": ["string"],
                        "competitive_advantage": "string",
                        "implementation_approach": "string"
                    }
                }
            },
            {
                "epoch_number": 3,
                "section_name": "funding_information",
                "schema": {
                    "funding_information": {
                        "funding_ask": {
                            "amount": "string",
                            "round": "string"
                        },
                        "use_of_funds": ["string"],
                        "financial_projections": [
                            {
                                "value": "string",
                                "context": "string"
                            }
                        ]
                    }
                }
            },
            {
                "epoch_number": 4,
                "section_name": "market_analysis",
                "schema": {
                    "market_analysis": {
                        "industry_sector": "string",
                        "target_geographies": ["string"],
                        "target_customer_segments": ["string"],
                        "market_metrics": [
                            {
                                "value": "string",
                                "context": "string",
                                "metric_type": "Market Size | Growth Rate"
                            }
                        ]
                    }
                }
            },
            {
                "epoch_number": 5,
                "section_name": "company_profile",
                "schema": {
                    "company_profile": {
                        "company_name": "string",
                        "company_description": "string",
                        "founding_year": "string",
                        "headquarters_location": "string",
                        "mission_statement": "string"
                    }
                }
            },
            {
                "epoch_number": 6,
                "section_name": "product_and_technology",
                "schema": {
                    "product_and_technology": {
                        "core_technology_stack": ["string"],
                        "products_and_features": [
                            {
                                "name": "string",
                                "description": "string"
                            }
                        ]
                    }
                }
            },
            {
                "epoch_number": 7,
                "section_name": "personnel", 
                "schema": {
                    "personnel": {
                        "key_team_members": [
                            {
                                "name": "string",
                                "role": "string",
                                "background_summary": "string"
                            }
                        ],
                        "advisory_board": "string"
                    }
                }
            },
            {
                "epoch_number": 8,
                "section_name": "business_performance_and_traction",
                "schema": {
                    "business_performance_and_traction": {
                        "pilot_program_results": [
                            {
                                "value": "string",
                                "context": "string"
                            }
                        ],
                        "current_financials": [
                            {
                                "value": "string",
                                "context": "string"
                            }
                        ],
                        "sales_pipeline": [
                            {
                                "value": "string",
                                "context": "string"
                            }
                        ]
                    }
                }
            },
            {
                "epoch_number": 9,
                "section_name": "strategy_and_goals",
                "schema": {
                    "strategy_and_goals": {
                        "go_to_market_summary": "string",
                        "short_term_goals": [
                            {
                                "target": "string",
                                "timeline": "string"
                            }
                        ]
                    }
                }
            },
            {
                "epoch_number": 10,
                "section_name": "mentioned_organizations",
                "schema": {
                    "mentioned_organizations": [
                        {
                            "name": "string",
                            "context": "Affiliation or background of a team member, partner, etc."
                        }
                    ]
                }
            }
        ]
        
    def _get_epoch_configurations(self) -> List[Dict[str, Any]]:
        """Get the 8 epoch configurations with section names and schemas."""
        return [
            {
                "epoch_number": 1,
                "section_name": "company_profile",
                "schema": {
                    "company_profile": {
                        "name": "string",
                        "mission_statement": "string"
                    }
                }
            },
            {
                "epoch_number": 2,
                "section_name": "personnel", 
                "schema": {
                    "personnel": {
                        "key_team_members": [
                            {
                                "name": "string",
                                "role": "string",
                                "background_summary": "string"
                            }
                        ],
                        "advisory_board": "string"
                    }
                }
            },
            {
                "epoch_number": 3,
                "section_name": "market_analysis",
                "schema": {
                    "market_analysis": {
                        "industry_sector": "string",
                        "target_geographies": ["string"],
                        "target_customer_segments": ["string"],
                        "market_metrics": [
                            {
                                "value": "string",
                                "context": "string",
                                "metric_type": "Market Size | Growth Rate"
                            }
                        ]
                    }
                }
            },
            {
                "epoch_number": 4,
                "section_name": "product_and_technology",
                "schema": {
                    "product_and_technology": {
                        "core_technology_stack": ["string"],
                        "products_and_features": [
                            {
                                "name": "string",
                                "description": "string"
                            }
                        ]
                    }
                }
            },
            {
                "epoch_number": 5,
                "section_name": "business_performance_and_traction",
                "schema": {
                    "business_performance_and_traction": {
                        "pilot_program_results": [
                            {
                                "value": "string",
                                "context": "string"
                            }
                        ],
                        "current_financials": [
                            {
                                "value": "string",
                                "context": "string"
                            }
                        ],
                        "sales_pipeline": [
                            {
                                "value": "string",
                                "context": "string"
                            }
                        ]
                    }
                }
            },
            {
                "epoch_number": 6,
                "section_name": "strategy_and_goals",
                "schema": {
                    "strategy_and_goals": {
                        "go_to_market_summary": "string",
                        "short_term_goals": [
                            {
                                "target": "string",
                                "timeline": "string"
                            }
                        ]
                    }
                }
            },
            {
                "epoch_number": 7,
                "section_name": "funding_information",
                "schema": {
                    "funding_information": {
                        "funding_ask": {
                            "amount": "string",
                            "round": "string"
                        },
                        "use_of_funds": ["string"],
                        "financial_projections": [
                            {
                                "value": "string",
                                "context": "string"
                            }
                        ]
                    }
                }
            },
            {
                "epoch_number": 8,
                "section_name": "mentioned_organizations",
                "schema": {
                    "mentioned_organizations": [
                        {
                            "name": "string",
                            "context": "Affiliation or background of a team member, partner, etc."
                        }
                    ]
                }
            },
            {
                "epoch_number": 9,
                "section_name": "problem_extraction",
                "schema": {
                    "problem_extraction": {
                        "core_problem": "string",
                        "problem_description": "string",
                        "target_audience_affected": ["string"],
                        "pain_points": ["string"],
                        "market_gap_identified": "string"
                    }
                }
            },
            {
                "epoch_number": 10,
                "section_name": "solution_extraction",
                "schema": {
                    "solution_extraction": {
                        "proposed_solution": "string",
                        "solution_description": "string",
                        "key_features": ["string"],
                        "competitive_advantage": "string",
                        "implementation_approach": "string"
                    }
                }
            }
        ]
    
    def run_10_epoch_analysis(self, pitch_text: str, output_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Run the complete 10-epoch NER analysis with focused extraction.
        
        Args:
            pitch_text: The business pitch text to analyze
            output_dir: Custom output directory for epoch files
            
        Returns:
            Dictionary containing analysis results and file paths
        """
        if not self.llm_summarizer.is_available:
            logging.warning("LLM not available for 8-epoch NER analysis")
            return {
                "error": "LLM not available for 8-epoch NER analysis",
                "timestamp": datetime.datetime.now().isoformat()
            }
        
        logging.info("=== STARTING 10-EPOCH NER ANALYSIS ===")
        logging.info(f"Pitch text length: {len(pitch_text)} characters")
        
        # Setup output directory - create distinct folder for each pitch analysis
        if output_dir is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            
            # Create a unique identifier for this pitch analysis
            import hashlib
            pitch_hash = hashlib.md5(pitch_text.encode('utf-8')).hexdigest()[:8]
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Create output directory in output_directory folder with pitch-specific name
            output_dir = os.path.join(base_dir, 'output_directory', f'pitch_analysis_{pitch_hash}_{timestamp}')
        
        os.makedirs(output_dir, exist_ok=True)
        logging.info(f"Output directory: {output_dir}")
        
        # Store epoch results and file paths
        epoch_results = []
        epoch_files = []
        
        epoch_names = ["Problem", "Solution", "Funding", "Market Analysis", "Company Profile", "Product & Tech", "Personnel", "Business Performance", "Strategy & Goals", "Mentioned Organizations"]
        
        # Process each epoch
        for config in self.epoch_configs:
            epoch_num = config["epoch_number"]
            section_name = config["section_name"]
            schema = config["schema"]
            
            logging.info(f" Preparing EPOCH {epoch_num}: {section_name}")
            
            # Complete LLM session reset to prevent context contamination
            self._reset_llm_session(epoch_num)
            
            # Generate the master prompt for this epoch
            prompt = self._create_master_prompt(epoch_num, section_name, schema, pitch_text)
            
            # Execute the focused extraction
            logging.info(f" STARTING EPOCH {epoch_num} EXTRACTION...")
            extraction_result = self._execute_epoch_extraction(prompt, epoch_num, section_name)
            
            # Save individual epoch result
            epoch_filename = f"epoch_{epoch_num}_{section_name}.json"
            epoch_file_path = os.path.join(output_dir, epoch_filename)
            
            with open(epoch_file_path, 'w', encoding='utf-8') as f:
                json.dump(extraction_result, f, indent=2, ensure_ascii=False)
            
            epoch_results.append(extraction_result)
            epoch_files.append(epoch_file_path)
            
            # Print epoch completion message as required
            print(f"{epoch_names[epoch_num - 1]} epoch finished")
            logging.info(f" EPOCH {epoch_num} FINISHED - Saved to {epoch_filename}")
        
        # Merge all epoch results into final comprehensive JSON
        final_result = self._merge_epoch_results(epoch_results, output_dir)
        
        # Create summary
        summary = {
            "analysis_type": "10-Epoch Focused NER Extraction",
            "timestamp": datetime.datetime.now().isoformat(),
            "input_text_length": len(pitch_text),
            "output_directory": output_dir,
            "epoch_files": epoch_files,
            "final_merged_file": final_result["final_file_path"],
            "total_sections_extracted": len(self.epoch_configs),
            "model_info": {
                "model_path": self.llm_summarizer.model_path,
                "model_available": self.llm_summarizer.is_available
            },
            "extraction_summary": final_result["extraction_summary"]
        }
        
        logging.info(" 10-EPOCH NER ANALYSIS COMPLETED SUCCESSFULLY")
        logging.info(f"Final merged result: {final_result['final_file_path']}")
        
        return summary
    
    def _reset_llm_session(self, epoch_num: int):
        """Reset the LLM session completely to prevent context contamination."""
        try:
            if self.llm_summarizer.llm and hasattr(self.llm_summarizer.llm, 'reset'):
                self.llm_summarizer.llm.reset()
                logging.info(f"  LLM session completely reset for epoch {epoch_num}")
            else:
                # For models without explicit reset, we reinitialize key components
                logging.info(f"  LLM context cleared for epoch {epoch_num}")
        except Exception as e:
            logging.warning(f"  Session reset warning for epoch {epoch_num}: {e}")
    
    def _create_master_prompt(self, epoch_num: int, section_name: str, schema: Dict[str, Any], pitch_text: str) -> str:
        """Create the master prompt template for the specific epoch."""
        schema_json = json.dumps(schema, indent=2)
        
        prompt = f"""**### ROLE AND GOAL ###**
You are a specialist data extraction system performing a highly focused analysis. Your function is to parse a business document and extract information for ONE specific section, mapping it precisely to a given JSON schema.

**### CURRENT TASK: EPOCH {epoch_num}/8 - EXTRACTING {section_name.upper()} ###**
Your SOLE objective in this task is to find and extract information related ONLY to the **`{section_name}`** section of the business pitch. Ignore all other information in the text that is not relevant to this specific section.

**### CRITICAL INSTRUCTIONS & CONSTRAINTS ###**
1.  **EXTREME FOCUS:** Only extract data relevant to the `{section_name}` schema provided below. If a piece of text (e.g., a dollar amount) belongs in a different section, you MUST ignore it for this task.
2.  **MANDATORY NULL:** If you cannot find any information for a field within the `{section_name}` section, you MUST use the JSON value `null`. For arrays, use an empty array `[]`. Do not omit any keys from the provided schema.
3.  **NO HALLUCINATIONS:** Extract information ONLY from the provided text. Do not invent facts or infer information not explicitly stated.
4.  **PRISTINE JSON OUTPUT:** Your output MUST be ONLY a single, valid JSON object containing the `{section_name}` key and its populated data. Do not include any introductory text, explanations, or markdown code fences (` ```json `).

**### JSON SCHEMA FOR THIS EPOCH ###**
*Reminder: Your entire output must be a JSON object with only ONE top-level key: `{section_name}`.*

{schema_json}

**### BUSINESS PITCH TEXT TO ANALYZE ###**
{pitch_text}

**### YOUR TASK ###**
Extract ONLY information related to `{section_name}` from the above text and format it according to the provided schema. Output valid JSON only."""

        return prompt
    
    def _execute_epoch_extraction(self, prompt: str, epoch_num: int, section_name: str) -> Dict[str, Any]:
        """Execute the focused extraction for a single epoch."""
        try:
            logging.info(f"  Processing epoch {epoch_num} extraction for {section_name}...")
            
            from typing import cast, Any
            
            messages = [
                {
                    "role": "system", 
                    "content": "You are a specialist data extraction system. Extract information for ONE specific section only. Return valid JSON format with no additional text. Be precise and focused."
                },
                {"role": "user", "content": prompt}
            ]
            
            if not self.llm_summarizer.llm:
                raise ValueError("LLM not initialized")
            
            # Truncate prompt if too long to prevent context overflow
            max_input_length = self.llm_summarizer.n_ctx - 500  # Reserve space for response
            if len(prompt) > max_input_length:
                # Truncate the pitch text portion while preserving the schema and instructions
                lines = prompt.split('\n')
                pitch_start = -1
                for i, line in enumerate(lines):
                    if "### BUSINESS PITCH TEXT TO ANALYZE ###" in line:
                        pitch_start = i + 1
                        break
                
                if pitch_start > 0:
                    before_pitch = '\n'.join(lines[:pitch_start])
                    after_pitch = '\n'.join(lines[pitch_start + 1:]) if pitch_start + 1 < len(lines) else ""
                    
                    available_space = max_input_length - len(before_pitch) - len(after_pitch) - 100
                    if available_space > 0:
                        pitch_text = lines[pitch_start][:available_space] + "..."
                        prompt = before_pitch + '\n' + pitch_text + '\n' + after_pitch
                        logging.warning(f"  Prompt truncated for epoch {epoch_num}")
            
            response = self.llm_summarizer.llm.create_chat_completion(
                messages=cast(Any, messages),
                max_tokens=800,  # Adequate for structured extraction
                temperature=0.0,  # Zero temperature for consistency
                top_p=0.8,
                stream=False
            )
            
            # Handle response safely
            response_dict = cast(Dict[str, Any], response)
            result = response_dict['choices'][0]['message']['content']
            
            if result:
                result = result.strip()
                # Clean the response
                cleaned_result = self._clean_extraction_response(result)
                # Parse and validate JSON
                parsed_result = self._parse_and_validate_json(cleaned_result, section_name, epoch_num)
                logging.info(f"  Epoch {epoch_num} extraction completed successfully")
                return parsed_result
            else:
                return self._create_empty_section_result(section_name, "Empty response from LLM")
                
        except Exception as e:
            logging.error(f"Epoch {epoch_num} extraction error: {e}")
            return self._create_empty_section_result(section_name, f"Extraction failed: {str(e)}")
    
    def _clean_extraction_response(self, response: str) -> str:
        """Clean the LLM extraction response to extract valid JSON."""
        # Remove markdown code blocks
        if "```json" in response:
            start = response.find("```json") + 7
            end = response.find("```", start)
            if end != -1:
                response = response[start:end].strip()
        elif "```" in response:
            start = response.find("```") + 3
            end = response.find("```", start)
            if end != -1:
                response = response[start:end].strip()
        
        # Remove common prefixes and suffixes
        response = response.strip()
        prefixes_to_remove = [
            "Here is the extracted information:",
            "Here is the JSON output:",
            "The extracted data is:",
            "JSON output:",
            "Result:",
            "Output:"
        ]
        
        for prefix in prefixes_to_remove:
            if response.startswith(prefix):
                response = response[len(prefix):].strip()
        
        return response
    
    def _parse_and_validate_json(self, response: str, section_name: str, epoch_num: int) -> Dict[str, Any]:
        """Parse and validate the JSON response for the specific section."""
        try:
            parsed = json.loads(response)
            
            # Validate that it's a dict and contains the expected section
            if isinstance(parsed, dict):
                if section_name in parsed:
                    return parsed
                else:
                    # If the section name is not found, try to wrap the content
                    logging.warning(f"Section '{section_name}' not found in epoch {epoch_num} response, attempting to wrap")
                    return {section_name: parsed}
            else:
                logging.warning(f"Response is not a dict for epoch {epoch_num}")
                return self._create_empty_section_result(section_name, "Response is not a valid JSON object")
                
        except json.JSONDecodeError as e:
            logging.warning(f"Failed to parse JSON for epoch {epoch_num}: {e}")
            # Try to fix incomplete JSON
            fixed_response = self._try_fix_incomplete_json(response)
            if fixed_response:
                try:
                    parsed = json.loads(fixed_response)
                    if isinstance(parsed, dict) and section_name in parsed:
                        return parsed
                except json.JSONDecodeError:
                    pass
            
            # Return empty result with error information
            return self._create_empty_section_result(section_name, f"JSON parsing failed: {str(e)}")
    
    def _try_fix_incomplete_json(self, response: str) -> Optional[str]:
        """Try to fix incomplete JSON structures."""
        try:
            response = response.strip()
            
            # If it starts with { but doesn't end with }, try to close it
            if response.startswith('{') and not response.endswith('}'):
                # Count open and close braces
                open_braces = response.count('{')
                close_braces = response.count('}')
                
                if open_braces > close_braces:
                    # Add missing closing braces
                    response += '}' * (open_braces - close_braces)
            
            # Similar logic for arrays
            if '[' in response:
                open_brackets = response.count('[')
                close_brackets = response.count(']')
                
                if open_brackets > close_brackets:
                    response += ']' * (open_brackets - close_brackets)
            
            return response
        except Exception:
            return None
    
    def _create_empty_section_result(self, section_name: str, error_message: str) -> Dict[str, Any]:
        """Create an empty result structure for a section with error information."""
        # Create a minimal structure based on the section schema
        config = next((c for c in self.epoch_configs if c["section_name"] == section_name), None)
        
        if config:
            schema = config["schema"]
            section_schema = schema.get(section_name, {})
            
            # Create empty structure based on schema
            empty_section = {}
            for key, value_type in section_schema.items():
                if isinstance(value_type, list):
                    empty_section[key] = []
                elif isinstance(value_type, dict):
                    empty_section[key] = None
                else:
                    empty_section[key] = None
            
            return {
                section_name: empty_section,
                "_extraction_error": error_message,
                "_extraction_timestamp": datetime.datetime.now().isoformat()
            }
        else:
            return {
                section_name: None,
                "_extraction_error": error_message,
                "_extraction_timestamp": datetime.datetime.now().isoformat()
            }
    
    def _merge_epoch_results(self, epoch_results: List[Dict[str, Any]], output_dir: str) -> Dict[str, Any]:
        """Merge all epoch results into a final comprehensive JSON object."""
        master_json = {}
        extraction_summary = {}
        
        sections = [
            "company_profile",
            "personnel", 
            "market_analysis",
            "product_and_technology",
            "business_performance_and_traction",
            "strategy_and_goals",
            "funding_information",
            "mentioned_organizations",
            "problem_extraction",
            "solution_extraction"
        ]
        
        for i, section in enumerate(sections):
            if i < len(epoch_results):
                result = epoch_results[i]
                
                # Extract the main section data
                if section in result:
                    master_json[section] = result[section]
                    extraction_summary[section] = "Successfully extracted"
                    logging.info(f"Successfully loaded and merged epoch_{i+1}_{section}")
                else:
                    master_json[section] = None
                    extraction_summary[section] = "No data found"
                    logging.warning(f"No data found for {section} in epoch {i+1}")
                
                # Log any extraction errors
                if "_extraction_error" in result:
                    extraction_summary[section] += f" (Error: {result['_extraction_error']})"
                    logging.warning(f"Extraction error for {section}: {result['_extraction_error']}")
            else:
                master_json[section] = None
                extraction_summary[section] = "Epoch not processed"
                logging.warning(f"Epoch not processed for {section}")
        
        # Add metadata
        master_json["_analysis_metadata"] = {
            "analysis_type": "10-Epoch Focused NER Extraction",
            "timestamp": datetime.datetime.now().isoformat(),
            "total_epochs": 10,
            "extraction_method": "Multi-pass focused extraction with LLM context resets",
            "extraction_summary": extraction_summary
        }
        
        # Save the final merged JSON
        final_filename = "final_extraction_result.json"
        final_file_path = os.path.join(output_dir, final_filename)
        
        with open(final_file_path, 'w', encoding='utf-8') as f:
            json.dump(master_json, f, indent=2, ensure_ascii=False)
        
        logging.info(f"All sections merged into {final_filename}")
        
        return {
            "final_file_path": final_file_path,
            "extraction_summary": extraction_summary,
            "merged_sections": len([s for s in extraction_summary.values() if "Successfully" in s])
        }

def analyze_pitch_10_epochs(pitch_text: str, output_dir: Optional[str] = None, model_name: Optional[str] = None) -> Dict[str, Any]:
    """
    Convenience function to run 10-epoch NER analysis on pitch text.
    
    Args:
        pitch_text: The business pitch text to analyze
        output_dir: Custom output directory
        model_name: The name of the model to use for analysis.
        
    Returns:
        Analysis results with file paths
    """
    analyzer = TenEpochNERAnalyzer(model_name=model_name)
    return analyzer.run_10_epoch_analysis(pitch_text, output_dir)

def main():
    """Main function for standalone execution."""
    import argparse
    
    parser = argparse.ArgumentParser(description='10-Epoch NER Analysis for VC Pitches')
    parser.add_argument('input_file', help='Path to text file containing the pitch')
    parser.add_argument('-o', '--output', help='Custom output directory for results')
    parser.add_argument('--model', help='Name of the model to use for analysis')
    
    args = parser.parse_args()
    
    # Ensure output_directory folder exists
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_directory = os.path.join(base_dir, 'output_directory')
    os.makedirs(output_directory, exist_ok=True)
    
    # Read input file
    try:
        with open(args.input_file, 'r', encoding='utf-8') as f:
            pitch_text = f.read()
    except FileNotFoundError:
        print(f"Error: Input file '{args.input_file}' not found.")
        return
    except Exception as e:
        print(f"Error reading input file: {e}")
        return
    
    print(f" Reading pitch from: {args.input_file}")
    print(f" Pitch length: {len(pitch_text)} characters")
    print(" Starting 10-Epoch NER Analysis...")
    
    # Perform 10-epoch NER analysis
    results = analyze_pitch_10_epochs(
        pitch_text=pitch_text,
        output_dir=args.output,
        model_name=args.model
    )
    
    # Display results
    print("\n" + "="*60)
    print(" 10-EPOCH NER ANALYSIS RESULTS")
    print("="*60)
    
    if "error" in results:
        print(f"❌ Error: {results['error']}")
        return
    
    print(f"📊 Total sections processed: {results['total_sections_extracted']}")
    print(f"⏰ Analysis completed at: {results['timestamp']}")
    print(f"📁 Output directory: {results['output_directory']}")
    print(f"📄 Final merged file: {results['final_merged_file']}")
    
    # Display extraction summary
    extraction_summary = results['extraction_summary']
    print(f"\n📈 EXTRACTION SUMMARY:")
    for section, status in extraction_summary.items():
        status_icon = "✅" if "Successfully" in status else "⚠️"
        print(f"  {status_icon} {section}: {status}")
    
    # Show individual epoch files
    print(f"\n📋 INDIVIDUAL EPOCH FILES:")
    for i, file_path in enumerate(results['epoch_files'], 1):
        filename = os.path.basename(file_path)
        print(f"  {i}. {filename}")
    
    print("\n✅ 10-Epoch NER Analysis Complete!")

if __name__ == "__main__":
    main()