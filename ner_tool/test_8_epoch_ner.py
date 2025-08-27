"""
Test script for the 8-Epoch NER Analysis System
Demonstrates the multi-pass focused extraction workflow
"""

import os
import sys
from pathlib import Path

# Add the current directory to Python path for imports
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

from ner_8_epoch_analysis import analyze_pitch_8_epochs

def create_sample_pitch() -> str:
    """Create a sample VC pitch for testing."""
    return """
TechFlow Solutions: Revolutionizing Enterprise Data Management

Company Overview:
TechFlow Solutions is a B2B SaaS company founded in 2022 by Sarah Chen (former Microsoft data architect) and Dr. Marcus Rodriguez (PhD in Machine Learning from Stanford). Our mission is to democratize enterprise data analytics through intelligent automation.

Market Analysis:
The global enterprise data management market is valued at $89.6 billion and growing at 12.3% CAGR. We target mid-market companies (500-5,000 employees) in North America and Europe, specifically in the financial services, healthcare, and manufacturing sectors. Current market penetration is only 23%, indicating massive growth opportunity.

Product & Technology:
Our core platform leverages advanced machine learning algorithms, natural language processing, and cloud-native architecture. Key products include DataFlow Analytics Suite and AutoInsight Dashboard. We utilize AWS infrastructure, Python-based ML models, and React-powered frontend interfaces.

Business Performance:
We've completed successful pilot programs with 3 Fortune 500 companies, showing 40% reduction in data processing time and $2.3M in cost savings annually. Current ARR stands at $850K with 15 enterprise clients. Our sales pipeline includes 47 qualified prospects worth $12.8M in potential revenue.

Strategy & Goals:
Our go-to-market strategy focuses on direct enterprise sales and strategic partnerships. Short-term goals include reaching $5M ARR by Q4 2024 and expanding to 50 enterprise clients within 18 months.

Funding Information:
We are raising a $15M Series A round to accelerate product development and market expansion. Use of funds: 40% engineering team expansion, 30% sales and marketing, 20% international expansion, 10% operational infrastructure. Financial projections show $25M ARR by 2026 with 35% gross margins.

Team & Advisory:
Our advisory board includes Jennifer Walsh (former VP at Salesforce) and Robert Kim (ex-CTO of Snowflake). Key team members include Lisa Thompson (Head of Engineering, former Google), and David Park (VP Sales, ex-Oracle).

Partnerships:
We have established partnerships with Amazon Web Services, Databricks, and Tableau. Our technology integrates with existing enterprise systems including SAP, Oracle, and Microsoft Dynamics.
"""

def test_8_epoch_analysis():
    """Test the 8-epoch NER analysis system."""
    print("🧪 TESTING 8-EPOCH NER ANALYSIS SYSTEM")
    print("="*50)
    
    # Create sample pitch
    sample_pitch = create_sample_pitch()
    print(f"📝 Sample pitch length: {len(sample_pitch)} characters")
    
    # Create test output directory
    test_output_dir = os.path.join(os.path.dirname(__file__), 'output', 'test_8_epoch')
    
    print(f"📁 Test output directory: {test_output_dir}")
    print("\n🚀 Starting 8-Epoch Analysis...")
    print("-" * 30)
    
    try:
        # Run the 8-epoch analysis
        results = analyze_pitch_8_epochs(
            pitch_text=sample_pitch,
            output_dir=test_output_dir
        )
        
        # Display results
        print("\n✅ ANALYSIS COMPLETED!")
        print("="*50)
        
        if "error" in results:
            print(f"❌ Error: {results['error']}")
            return
        
        print(f"📊 Sections processed: {results['total_sections_extracted']}")
        print(f"⏰ Completed at: {results['timestamp']}")
        print(f"📁 Output: {results['output_directory']}")
        print(f"📄 Final file: {results['final_merged_file']}")
        
        # Show extraction summary
        extraction_summary = results['extraction_summary']
        print(f"\n📈 EXTRACTION SUMMARY:")
        successful_extractions = 0
        
        for section, status in extraction_summary.items():
            if "Successfully" in status:
                status_icon = "✅"
                successful_extractions += 1
            else:
                status_icon = "⚠️"
            print(f"  {status_icon} {section}: {status}")
        
        print(f"\n🎯 SUCCESS RATE: {successful_extractions}/{len(extraction_summary)} sections")
        
        # Show epoch files created
        print(f"\n📋 EPOCH FILES CREATED:")
        for i, file_path in enumerate(results['epoch_files'], 1):
            filename = os.path.basename(file_path)
            file_size = os.path.getsize(file_path) if os.path.exists(file_path) else 0
            print(f"  {i}. {filename} ({file_size} bytes)")
        
        # Show sample of final merged result if it exists
        final_file = results['final_merged_file']
        if os.path.exists(final_file):
            print(f"\n📖 SAMPLE FROM FINAL MERGED RESULT:")
            print("-" * 30)
            try:
                import json
                with open(final_file, 'r', encoding='utf-8') as f:
                    final_data = json.load(f)
                
                # Show first few sections with data
                shown_sections = 0
                for section_name, section_data in final_data.items():
                    if not section_name.startswith('_') and section_data and shown_sections < 3:
                        print(f"\n🏷️  {section_name.upper()}:")
                        if isinstance(section_data, dict):
                            for key, value in list(section_data.items())[:2]:  # Show first 2 fields
                                if value is not None:
                                    value_preview = str(value)[:100] + "..." if len(str(value)) > 100 else str(value)
                                    print(f"    {key}: {value_preview}")
                        shown_sections += 1
                
                print(f"\n📊 Total sections in final file: {len([k for k in final_data.keys() if not k.startswith('_')])}")
                
            except Exception as e:
                print(f"Error reading final file: {e}")
        
        print(f"\n🎉 8-EPOCH NER ANALYSIS TEST COMPLETED SUCCESSFULLY!")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Main test function."""
    print("8-EPOCH NER ANALYSIS - TEST SUITE")
    print("="*60)
    print("This test demonstrates the 8-epoch focused extraction system.")
    print("Each epoch processes one specific section with LLM context reset.\n")
    
    # Check if LLM is available
    try:
        from llm_summarization import get_llm_summarizer
        llm = get_llm_summarizer()
        
        if llm.is_available:
            print("✅ LLM is available for testing")
            print(f"📍 Model path: {llm.model_path}")
            print(f"🧠 Context size: {llm.n_ctx}")
        else:
            print("⚠️  LLM not available - test will show error handling")
        
    except Exception as e:
        print(f"⚠️  LLM check failed: {e}")
    
    print("\n" + "-"*60)
    
    # Run the test
    test_8_epoch_analysis()

if __name__ == "__main__":
    main()