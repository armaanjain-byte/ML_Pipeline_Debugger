import argparse
import json
import sys
import os
from app.pipeline.pipeline_runner import PipelineRunner

def main():
    """Main entry point for the ML Pipeline Debugger CLI."""
    
    # 1. Setup the Command Line Interface (CLI)
    parser = argparse.ArgumentParser(
        description="ML Pipeline Debugger - Analyze data and prevent pipeline failures."
    )
    
    parser.add_argument('--file', '-f', required=True, type=str,
                        help='Path to the dataset (CSV format)')
    
    parser.add_argument('--target', '-t', required=True, type=str,
                        help='The target column you want to predict')
    
    parser.add_argument('--task', choices=['regression', 'classification'], 
                        default='regression',
                        help='Type of ML task (default: regression)')

    args = parser.parse_args()

    # 2. Validate the file exists before running
    if not os.path.exists(args.file):
        print(f"\n❌ ERROR: Could not find the file at '{args.file}'")
        sys.exit(1)

    # 3. Print a nice header
    print("="*60)
    print("🚀 ML PIPELINE DEBUGGER INITIALIZED")
    print("="*60)
    print(f"📁 Dataset: {args.file}")
    print(f"🎯 Target:  {args.target}")
    print(f"⚙️ Task:   {args.task.capitalize()}")
    print("-" * 60)

    # 4. Run YOUR actual backend pipeline
    try:
        runner = PipelineRunner(
            file_path=args.file,
            target_column=args.target,
            task_type=args.task
        )

        output = runner.run()

        # 5. Output the Recommendations
        print("\n" + "="*60)
        print("🔍 DIAGNOSTIC RECOMMENDATIONS")
        print("="*60)
        
        if output.get("recommendations"):
            print(json.dumps(output["recommendations"], indent=4))
        else:
            print("✅ No critical issues found in the data.")

    except Exception as e:
        print(f"\n❌ PIPELINE FAILED: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()