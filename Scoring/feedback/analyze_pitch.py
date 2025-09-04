"""
Interactive Pitch Feedback CLI
Analyze pitch text and get concise feedback using local GGUF model.
"""

import sys
import os
from Feedback import PitchFeedbackGenerator, analyze_pitch_from_file

def analyze_pitch_from_input():
    """Interactive CLI for pitch analysis."""
    print("🚀 Pitch Feedback Analyzer")
    print("Using model: P:\\progs\\vsCode\\VSProjects\\NLP\\models\\liquid\\LFM2-1.2B-Q8_0.gguf")
    print("=" * 60)
    
    # Get pitch text or file path
    if len(sys.argv) > 1:
        input_arg = " ".join(sys.argv[1:])
        
        # Check if it's a file path
        if input_arg.endswith('.txt') or '/' in input_arg or '\\' in input_arg:
            print(f"Analyzing pitch from file: {input_arg}")
            feedback_results = analyze_pitch_from_file(input_arg)
            
            if not feedback_results:
                return
            
            # Read the file again to get pitch text for report
            try:
                if not os.path.isabs(input_arg):
                    scoring_dir = os.path.dirname(__file__)
                    input_arg = os.path.join(scoring_dir, input_arg)
                
                with open(input_arg, 'r', encoding='utf-8') as f:
                    pitch_text = f.read().strip()
            except Exception as e:
                print(f"⚠️ Could not re-read file for report: {e}")
                pitch_text = "File content unavailable"
        else:
            # Use as direct pitch text
            pitch_text = input_arg
            print(f"Analyzing pitch from command line...")
            
            print(f"\n📊 Pitch Preview: {pitch_text[:100]}{'...' if len(pitch_text) > 100 else ''}")
            print("\n🔍 Analyzing sections...\n")
            
            # Generate feedback
            generator = PitchFeedbackGenerator()
            feedback_results = generator.generate_comprehensive_feedback(pitch_text)
    else:
        # Interactive input
        print("Enter your pitch text (press Ctrl+Z then Enter when done):")
        print("OR enter a file path (e.g., 'sample_pitch.txt' or '../sample_pitch.txt'):")
        print("-" * 40)
        try:
            user_input = sys.stdin.read().strip()
        except KeyboardInterrupt:
            print("\n\n❌ Analysis cancelled.")
            return
        
        if not user_input:
            print("❌ No input provided.")
            return
        
        # Check if input looks like a file path
        if user_input.endswith('.txt') or '/' in user_input or '\\' in user_input:
            print(f"\nTreating input as file path: {user_input}")
            feedback_results = analyze_pitch_from_file(user_input)
            
            if not feedback_results:
                return
            
            # Read the file to get pitch text for report
            try:
                if not os.path.isabs(user_input):
                    scoring_dir = os.path.dirname(__file__)
                    user_input = os.path.join(scoring_dir, user_input)
                
                with open(user_input, 'r', encoding='utf-8') as f:
                    pitch_text = f.read().strip()
            except Exception as e:
                print(f"⚠️ Could not re-read file for report: {e}")
                pitch_text = "File content unavailable"
        else:
            # Use as direct pitch text
            pitch_text = user_input
            print(f"\n📊 Pitch Preview: {pitch_text[:100]}{'...' if len(pitch_text) > 100 else ''}")
            print("\n🔍 Analyzing sections...\n")
            
            # Generate feedback
            generator = PitchFeedbackGenerator()
            feedback_results = generator.generate_comprehensive_feedback(pitch_text)
    
    # Print summary
    print("\n" + "=" * 60)
    print("📋 FEEDBACK SUMMARY")
    print("=" * 60)
    
    issues_found = []
    strengths_found = []
    
    for section, feedback in feedback_results.items():
        if "❌" in feedback or "⚠️" in feedback:
            issues_found.append(feedback)
        else:
            strengths_found.append(feedback)
    
    if issues_found:
        print(f"\n🔴 Issues Found ({len(issues_found)}):")
        for issue in issues_found:
            print(f"  • {issue}")
    
    if strengths_found:
        print(f"\n🟢 Positive Areas ({len(strengths_found)}):")
        for strength in strengths_found:
            print(f"  • {strength}")
    
    # Save report
    try:
        report_path = generator.save_feedback_report(pitch_text, feedback_results)
        print(f"\n💾 Full report saved: {report_path}")
    except Exception as e:
        print(f"\n⚠️ Could not save report: {e}")
    
    print(f"\n✅ Analysis complete! Found {len(issues_found)} areas for improvement.")

def main():
    """Main CLI entry point."""
    try:
        analyze_pitch_from_input()
    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")
        return 1
    return 0

if __name__ == "__main__":
    sys.exit(main())