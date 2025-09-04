#!/usr/bin/env python3
"""
Simple runner script for the Venture Pitch Scoring System.
Usage: python run_scorer.py path/to/pitch.txt
"""

import sys
import os
from pitch_scorer import score_pitch_from_file, print_scoring_summary


def main():
    """Main function to run the pitch scorer."""
    if len(sys.argv) != 2:
        print("Usage: python run_scorer.py <path_to_pitch.txt>")
        print("Example: python run_scorer.py P:\\progs\\vsCode\\VSProjects\\NLP\\IEEEVic\\sample_pitch.txt")
        sys.exit(1)
    
    pitch_file_path = sys.argv[1]
    
    # Check if file exists
    if not os.path.exists(pitch_file_path):
        print(f"❌ Error: File not found: {pitch_file_path}")
        sys.exit(1)
    
    print(f"🎯 Scoring venture pitch from: {pitch_file_path}")
    print("=" * 60)
    
    # Score the pitch
    results = score_pitch_from_file(pitch_file_path)
    
    # Print results
    print_scoring_summary(results)
    
    # Show JSON output location if saved
    if "output_file" in results:
        print(f"💾 Full JSON results saved to: {results['output_file']}")


if __name__ == "__main__":
    main()