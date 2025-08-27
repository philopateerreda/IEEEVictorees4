"""
Named Entity Recognition (NER) Analysis for VC Pitch Text
Using Local LLM for Entity Extraction and Classification

This module provides standalone NER analysis without running the full VC analysis workflow.
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

class NERAnalyzer:
    """Named Entity Recognition analyzer using local LLM."""
    
    def __init__(self):
        """Initialize the NER analyzer."""
        self.llm_summarizer = get_llm_summarizer()
        
    def get_ner_system_prompt(self) -> str:
        """Return the NER system prompt with entity type definitions."""
        return """You are a sharp, objective Venture Analyst AI. Your purpose is to extract actionable intelligence, ignoring all promotional hype.
Your task is to analyze the following business pitch. Perform a precise Named Entity Recognition (NER) and label all entities using *only* the types provided below.

---

**Instruction:** From the provided text, extract and label all entities that match the following types.

#### **Core Entities**
*   **`ORGANIZATION`**: A company, competitor, or partner.
    *   *Examples*: "Nvidia", "OpenAI", "Acme Corp."
*   **`PRODUCT`**: A specific commercial good, service, or software.
    *   *Examples*: "iPhone 15", "ChatGPT-4", "Azure Cloud Services"
*   **`PERSON`**: Key executives, founders, or influential figures.
    *   *Examples*: "Satya Nadella", "the CEO"
*   **`LOCATION`**: A geographic market, country, or region.
    *   *Examples*: "the EU market", "Silicon Valley", "China"

#### **Quantitative & Market Data**
*   **`MONEY`**: Financial figures, valuations, or investment amounts.
    *   *Examples*: "$1.2 billion", "a €50M funding round"
*   **`PERCENT`**: Percentage-based metrics for growth, market share, or change.
    *   *Examples*: "15% year-over-year growth", "a 30% stake"
*   **`INDUSTRY_SECTOR`**: A specific industry or vertical.
    *   *Examples*: "SaaS", "FinTech", "AI research", "Cybersecurity"
*   **`MARKET_SEGMENT`**: A specific customer group or demographic.
    *   *Examples*: "SMBs", "enterprise clients", "Gen Z consumers"

#### **Actions & Concepts**
*   **`EVENT`**: A significant business action or occurrence.
    *   *Examples*: "acquisition", "product launch", "partnership", "funding round"
*   **`TECHNOLOGY`**: A specific technology or technical field.
    *   *Examples*: "Large Language Models", "Blockchain", "API"
*   **`REGULATION`**: A specific law, policy, or legal challenge.
    *   *Examples*: "GDPR", "Digital Services Act", "antitrust lawsuit\""""

    def analyze_ner(self, pitch_text: str) -> Dict[str, Any]:
        """
        Perform NER analysis on the pitch text using the local LLM.
        
        Args:
            pitch_text: The business pitch text to analyze
            
        Returns:
            Dictionary containing extracted entities and metadata
        """
        if not self.llm_summarizer.is_available:
            logging.warning("LLM not available for NER analysis")
            return {
                "error": "LLM not available for NER analysis",
                "timestamp": datetime.datetime.now().isoformat()
            }
        
        logging.info("=== STARTING NER ANALYSIS ===")
        logging.info(f"Pitch text length: {len(pitch_text)} characters")
        
        # Create the NER prompt
        ner_prompt = self._create_ner_prompt(pitch_text)
        
        # Execute NER analysis
        logging.info("🚀 Analyzing entities with LLM...")
        entities_result = self._execute_ner_analysis(ner_prompt)
        
        # Parse the LLM response into structured format
        parsed_entities = self._parse_ner_response(entities_result)
        
        # Create final result structure
        result = {
            "analysis_type": "Named Entity Recognition (NER)",
            "timestamp": datetime.datetime.now().isoformat(),
            "input_text_length": len(pitch_text),
            "raw_llm_response": entities_result,
            "extracted_entities": parsed_entities,
            "entity_counts": self._count_entities_by_type(parsed_entities),
            "total_entities": len(parsed_entities.get("entities", [])),
            "model_info": {
                "model_path": self.llm_summarizer.model_path,
                "model_available": self.llm_summarizer.is_available
            }
        }
        
        logging.info("✅ NER ANALYSIS COMPLETED SUCCESSFULLY")
        logging.info(f"Total entities extracted: {result['total_entities']}")
        
        return result
    
    def _create_ner_prompt(self, pitch_text: str) -> str:
        """Create the NER analysis prompt."""
        system_prompt = self.get_ner_system_prompt()
        
        prompt = f"""{system_prompt}

**BUSINESS PITCH TO ANALYZE:**
{pitch_text}

**INSTRUCTIONS FOR OUTPUT:**
Extract all entities from the text and format them as a JSON structure like this:

```json
{{
    "entities": [
        {{"text": "entity_text", "label": "ENTITY_TYPE", "context": "surrounding context"}},
        {{"text": "another_entity", "label": "ENTITY_TYPE", "context": "surrounding context"}}
    ]
}}
```

**IMPORTANT RULES:**
1. Only use the entity types listed above (ORGANIZATION, PRODUCT, PERSON, LOCATION, MONEY, PERCENT, INDUSTRY_SECTOR, MARKET_SEGMENT, EVENT, TECHNOLOGY, REGULATION)
2. Include the exact text as it appears in the pitch
3. Provide brief context showing where the entity appears
4. Be precise and only extract clear, unambiguous entities
5. Return valid JSON format

Begin NER extraction:"""
        
        return prompt
    
    def _execute_ner_analysis(self, prompt: str) -> str:
        """Execute NER analysis using the LLM."""
        try:
            logging.info("  🤖 Initializing LLM for NER analysis...")
            
            # Reset LLM context for fresh analysis
            if self.llm_summarizer.llm and hasattr(self.llm_summarizer.llm, 'reset'):
                self.llm_summarizer.llm.reset()
                logging.info("  🔄 LLM context reset for NER analysis")
            
            from typing import cast, Any
            
            messages = [
                {
                    "role": "system", 
                    "content": "You are an expert NER analyst. Extract entities precisely using only the specified types. Return valid JSON format only. Be thorough but accurate."
                },
                {"role": "user", "content": prompt}
            ]
            
            logging.info("  ⏳ Processing NER analysis with LLM...")
            if not self.llm_summarizer.llm:
                raise ValueError("LLM not initialized")
            
            response = self.llm_summarizer.llm.create_chat_completion(
                messages=cast(Any, messages),
                max_tokens=1200,  # Increased for comprehensive entity extraction
                temperature=0.1,  # Low temperature for consistent extraction
                top_p=0.9,
                stream=False
            )
            
            # Handle response safely
            response_dict = cast(Dict[str, Any], response)
            result = response_dict['choices'][0]['message']['content']
            
            if result:
                result = result.strip()
                # Clean any unwanted prefixes/suffixes
                result = self._clean_ner_response(result)
                logging.info(f"  📝 NER analysis completed. Response length: {len(result)} characters")
                return result
            else:
                return '{"entities": [], "error": "Empty response from LLM"}'
                
        except Exception as e:
            logging.error(f"NER analysis error: {e}")
            return f'{{"entities": [], "error": "NER analysis failed: {str(e)}"}}'
    
    def _clean_ner_response(self, response: str) -> str:
        """Clean the LLM response to extract valid JSON."""
        # Remove any markdown code blocks
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
        
        # Clean common prefixes
        response = response.strip()
        if response.startswith("Here is the NER analysis:"):
            response = response[len("Here is the NER analysis:"):].strip()
        
        return response
    
    def _parse_ner_response(self, response: str) -> Dict[str, Any]:
        """Parse the LLM NER response into structured format."""
        try:
            # Try to parse as JSON
            parsed = json.loads(response)
            if isinstance(parsed, dict) and "entities" in parsed:
                return parsed
            else:
                return {"entities": [], "parse_error": "Invalid JSON structure"}
        except json.JSONDecodeError as e:
            logging.warning(f"Failed to parse NER response as JSON: {e}")
            # Try to fix incomplete JSON
            fixed_response = self._try_fix_incomplete_json(response)
            if fixed_response:
                try:
                    parsed = json.loads(fixed_response)
                    if isinstance(parsed, dict) and "entities" in parsed:
                        return parsed
                except json.JSONDecodeError:
                    pass
            
            # Fallback: try to extract entities from text format
            return self._fallback_parse_entities(response)
    
    def _try_fix_incomplete_json(self, response: str) -> Optional[str]:
        """Try to fix incomplete JSON by closing open structures."""
        try:
            # Remove markdown code blocks first
            cleaned = response.strip()
            if "```json" in cleaned:
                start = cleaned.find("```json") + 7
                end = cleaned.find("```", start)
                if end == -1:
                    # No closing ```, take everything after ```json
                    cleaned = cleaned[start:].strip()
                else:
                    cleaned = cleaned[start:end].strip()
            elif "```" in cleaned:
                start = cleaned.find("```") + 3
                end = cleaned.find("```", start)
                if end == -1:
                    cleaned = cleaned[start:].strip()
                else:
                    cleaned = cleaned[start:end].strip()
            
            # Try to fix common incomplete JSON patterns
            if cleaned.startswith('{') and not cleaned.endswith('}'):
                # Find the last complete entity
                last_complete = cleaned.rfind('}')
                if last_complete > 0:
                    # Find if we're in an incomplete entity
                    after_last = cleaned[last_complete + 1:].strip()
                    if after_last.startswith(','):
                        # Remove the incomplete part and close properly
                        cleaned = cleaned[:last_complete + 1]
                        # Check if we need to close the entities array
                        if '"entities":[' in cleaned and not cleaned.endswith(']}'):
                            cleaned += ']'
                        if not cleaned.endswith('}'):
                            cleaned += '}'
            
            return cleaned
        except Exception:
            return None
    
    def _fallback_parse_entities(self, response: str) -> Dict[str, Any]:
        """Fallback parser for non-JSON responses."""
        entities = []
        
        # Try to extract entity patterns from the text
        import re
        
        # Look for JSON-like entity patterns
        entity_pattern = r'\{"text":\s*"([^"]+)",\s*"label":\s*"([^"]+)",\s*"context":\s*"([^"]*?)"\}'
        matches = re.findall(entity_pattern, response)
        
        for match in matches:
            entity = {
                "text": match[0],
                "label": match[1],
                "context": match[2] if len(match) > 2 else ""
            }
            entities.append(entity)
        
        # If regex didn't work, try line-by-line parsing
        if not entities:
            lines = response.split('\n')
            current_entity = {}
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # Look for entity field patterns
                if '"text":' in line:
                    # Extract text value
                    text_match = re.search(r'"text":\s*"([^"]+)"', line)
                    if text_match:
                        current_entity['text'] = text_match.group(1)
                
                elif '"label":' in line:
                    # Extract label value
                    label_match = re.search(r'"label":\s*"([^"]+)"', line)
                    if label_match:
                        current_entity['label'] = label_match.group(1)
                
                elif '"context":' in line:
                    # Extract context value
                    context_match = re.search(r'"context":\s*"([^"]*?)"', line)
                    if context_match:
                        current_entity['context'] = context_match.group(1)
                
                # If we have text and label, save the entity
                if 'text' in current_entity and 'label' in current_entity:
                    entities.append(current_entity.copy())
                    current_entity = {}
        
        logging.info(f"Fallback parser extracted {len(entities)} entities")
        
        return {
            "entities": entities,
            "fallback_parsed": True,
            "extraction_method": "regex_and_line_parsing",
            "original_response": response
        }
    
    def _count_entities_by_type(self, parsed_entities: Dict[str, Any]) -> Dict[str, int]:
        """Count entities by their type."""
        counts = {}
        entities = parsed_entities.get("entities", [])
        
        for entity in entities:
            entity_type = entity.get("label", "UNKNOWN")
            counts[entity_type] = counts.get(entity_type, 0) + 1
        
        return counts
    
    def save_ner_results(self, results: Dict[str, Any], output_dir: Optional[str] = None, filename: Optional[str] = None) -> str:
        """
        Save NER results to JSON file.
        
        Args:
            results: NER analysis results
            output_dir: Output directory (defaults to output folder in current project)
            filename: Custom filename (defaults to timestamp-based name)
            
        Returns:
            Path to saved file
        """
        # Set default output directory
        if output_dir is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            output_dir = os.path.join(base_dir, 'output', f'NER_Analysis_{timestamp}')
        
        # Create directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Set default filename
        if filename is None:
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'NER_Analysis_{timestamp}.json'
        
        # Ensure .json extension
        if not filename.endswith('.json'):
            filename += '.json'
        
        file_path = os.path.join(output_dir, filename)
        
        # Save results
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logging.info(f"NER results saved to: {file_path}")
        return file_path

def analyze_pitch_ner(pitch_text: str, save_results: bool = True, output_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    Convenience function to analyze pitch text for named entities.
    
    Args:
        pitch_text: The business pitch text to analyze
        save_results: Whether to save results to JSON file
        output_dir: Custom output directory
        
    Returns:
        NER analysis results
    """
    analyzer = NERAnalyzer()
    results = analyzer.analyze_ner(pitch_text)
    
    if save_results:
        saved_path = analyzer.save_ner_results(results, output_dir)
        results["saved_to"] = saved_path
    
    return results

def main():
    """Main function for standalone execution."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Named Entity Recognition Analysis for VC Pitches')
    parser.add_argument('input_file', help='Path to text file containing the pitch')
    parser.add_argument('-o', '--output', help='Output directory for JSON results')
    parser.add_argument('--no-save', action='store_true', help='Do not save results to file')
    
    args = parser.parse_args()
    
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
    
    print(f"📄 Reading pitch from: {args.input_file}")
    print(f"📏 Pitch length: {len(pitch_text)} characters")
    
    # Perform NER analysis
    results = analyze_pitch_ner(
        pitch_text=pitch_text,
        save_results=not args.no_save,
        output_dir=args.output
    )
    
    # Display results
    print("\n" + "="*60)
    print("🎯 NER ANALYSIS RESULTS")
    print("="*60)
    
    if "error" in results:
        print(f"❌ Error: {results['error']}")
        return
    
    print(f"📊 Total entities extracted: {results['total_entities']}")
    print(f"⏰ Analysis completed at: {results['timestamp']}")
    
    # Display entity counts by type
    entity_counts = results['entity_counts']
    if entity_counts:
        print("\n📈 ENTITY BREAKDOWN BY TYPE:")
        for entity_type, count in sorted(entity_counts.items()):
            print(f"  • {entity_type}: {count}")
    
    # Display extracted entities
    entities = results['extracted_entities'].get('entities', [])
    if entities:
        print(f"\n🏷️  EXTRACTED ENTITIES:")
        for i, entity in enumerate(entities[:10], 1):  # Show first 10
            text = entity.get('text', 'N/A')
            label = entity.get('label', 'N/A')
            context = entity.get('context', 'N/A')
            print(f"  {i:2d}. [{label}] {text}")
            if context and context != 'N/A':
                print(f"      Context: {context}")
        
        if len(entities) > 10:
            print(f"      ... and {len(entities) - 10} more entities")
    
    # Show save location
    if "saved_to" in results:
        print(f"\n💾 Results saved to: {results['saved_to']}")
    
    print("\n✅ NER Analysis Complete!")

if __name__ == "__main__":
    main()