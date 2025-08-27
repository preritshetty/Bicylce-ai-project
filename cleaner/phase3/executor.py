"""
Simple Code Executor for Phase 3 Detection Codes

Clean pipeline: CSV in → Run detection codes → CSV out with flags → Summary stats
"""

import pandas as pd
import json
import os
from datetime import datetime


def load_data(csv_path):
    """
    Read the Phase 2 cleaned CSV file
    Add flag_status column initialized to 0
    Return the DataFrame
    """
    df = pd.read_csv(csv_path)
    df['flag_status'] = 0
    return df


def load_detection_codes(codes_json_path):
    """
    Read the generated detection codes JSON from code_generator
    Parse into list of detection code strings
    Return the codes ready to execute
    """
    with open(codes_json_path, 'r') as f:
        codes_data = json.load(f)
    
    detection_codes = codes_data.get('detection_codes', [])
    
    # Extract just the code strings
    code_strings = []
    for code_info in detection_codes:
        code_strings.append(code_info['detection_code'])
    
    return code_strings


def execute_detection_codes(df, detection_codes):
    """
    Loop through each detection code string
    Execute: exec(code_string) on the DataFrame
    Each code applies its flag using bitwise OR: df.loc[condition, 'flag_status'] |= flag_value
    """
    for code_string in detection_codes:
        exec(code_string)
    
    return df


def save_flagged_data(df, output_path):
    """
    Save the DataFrame with flag_status column to CSV
    Return path to saved file
    """
    df.to_csv(output_path, index=False)
    return output_path


def generate_summary(df):
    """
    Count total rows flagged
    Count rows for each flag value (1, 2, 4, 8, etc.)
    Show flag_status distribution
    Return summary dict
    """
    total_rows = len(df)
    flagged_rows = (df['flag_status'] > 0).sum()
    clean_rows = total_rows - flagged_rows
    
    # Count individual flags
    flag_counts = {}
    for flag_value in [1, 2, 4, 8, 16, 32, 64, 128]:
        count = (df['flag_status'] & flag_value).sum() // flag_value
        if count > 0:
            flag_counts[flag_value] = count
    
    # Flag status distribution
    status_distribution = df['flag_status'].value_counts().to_dict()
    
    summary = {
        'total_rows': total_rows,
        'flagged_rows': flagged_rows,
        'clean_rows': clean_rows,
        'flagged_percentage': round(flagged_rows / total_rows * 100, 2),
        'individual_flag_counts': flag_counts,
        'flag_status_distribution': status_distribution,
        'timestamp': datetime.now().isoformat()
    }
    
    return summary


def main():
    """
    Main execution flow:
    Load data → Load codes → Execute codes → Save results → Generate summary
    """
    print("🚀 Starting Code Executor")
    
    # File paths
    csv_path = "phase outputs/phase1_cleaned_data.csv"
    codes_path = "phase outputs/detection_codes_20250824_220220.json"
    output_path = f"phase outputs/flagged_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    
    # 1. Load data
    print(f"📊 Loading data from: {csv_path}")
    df = load_data(csv_path)
    print(f"   Loaded {len(df)} rows")
    
    # 2. Load codes
    print(f"🤖 Loading detection codes from: {codes_path}")
    detection_codes = load_detection_codes(codes_path)
    print(f"   Loaded {len(detection_codes)} detection codes")
    
    # 3. Execute codes
    print("⚡ Executing detection codes on data...")
    df = execute_detection_codes(df, detection_codes)
    
    # 4. Save results
    print(f"💾 Saving flagged data to: {output_path}")
    save_flagged_data(df, output_path)
    
    # 5. Generate summary
    print("📈 Generating summary...")
    summary = generate_summary(df)
    
    print("\n✅ Execution Complete!")
    print(f"📊 Total rows: {summary['total_rows']:,}")
    print(f"🚩 Flagged rows: {summary['flagged_rows']:,} ({summary['flagged_percentage']}%)")
    print(f"✅ Clean rows: {summary['clean_rows']:,}")
    
    if summary['individual_flag_counts']:
        print("\n🔍 Individual Flag Counts:")
        for flag_value, count in summary['individual_flag_counts'].items():
            print(f"   Flag {flag_value}: {count:,} rows")
    
    return summary


if __name__ == "__main__":
    main()
