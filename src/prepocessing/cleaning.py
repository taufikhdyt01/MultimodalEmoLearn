import pandas as pd
import os
from datetime import datetime
import shutil

# Function to extract time from timestamp (handles both string and pandas Timestamp)
def extract_time_from_timestamp(timestamp):
    try:
        # Check if it's already a pandas Timestamp
        if isinstance(timestamp, pd.Timestamp):
            dt = timestamp
        else:
            # Try different formats
            try:
                dt = datetime.strptime(timestamp, "%d/%m/%Y %H:%M:%S")
            except ValueError:
                try:
                    dt = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
                except ValueError:
                    # If all parsing attempts fail, try to convert to string and parse
                    dt = pd.to_datetime(timestamp)
        
        # Return formatted time string for matching with frame filenames
        return f"{dt.hour:02d}_{dt.minute:02d}_{dt.second:02d}"
    except Exception as e:
        print(f"Error parsing timestamp {timestamp}: {e}")
        return None

# Function to process a single sample without problem separation
def process_sample(sample_num, user_id, df_all, base_dir, output_base_dir):
    # Filter dataframe for this user_id
    df_sample = df_all[df_all['user_id'] == user_id].copy()
    
    if df_sample.empty:
        print(f"No data found for Sample {sample_num} (user_id: {user_id})")
        return pd.DataFrame(), []
    
    # Define source directory for frames
    frames_dir = os.path.join(base_dir, f"Sample {sample_num}", "frames")
    
    if not os.path.exists(frames_dir):
        print(f"Frames directory not found for Sample {sample_num}: {frames_dir}")
        return pd.DataFrame(), []
    
    # Get all frame files in the directory
    frame_files = {}
    for filename in os.listdir(frames_dir):
        if filename.startswith('frame_') and filename.endswith('.jpg'):
            # Extract timestamp from filename (format: frame_11_54_14.jpg)
            time_parts = filename.replace('frame_', '').replace('.jpg', '')
            frame_files[time_parts] = filename
    
    # Create output directory for this sample (without problem separation)
    sample_output_dir = os.path.join(output_base_dir, f"Sample {sample_num}", "cleaned_frames")
    os.makedirs(sample_output_dir, exist_ok=True)
    
    # Process all records for this sample
    matched_indices = []
    matched_times = set()
    unmatched_frames = []
    
    print(f"\n  Processing all records for Sample {sample_num}...")
    
    # Process all records without grouping by problem
    for index, row in df_sample.iterrows():
        time_key = extract_time_from_timestamp(row['timestamp'])
        if not time_key:
            continue
        
        # Check if we have a matching frame
        if time_key in frame_files:
            # Check confidence (if confidence columns exist)
            emotion_columns = ['neutral', 'happy', 'sad', 'angry', 'fearful', 'disgusted', 'surprised']
            
            # Check if emotion columns exist in the dataframe
            if all(col in df_sample.columns for col in emotion_columns):
                max_confidence = max(
                    float(row['neutral']), float(row['happy']), float(row['sad']), 
                    float(row['angry']), float(row['fearful']), 
                    float(row['disgusted']), float(row['surprised'])
                )
                
                is_valid = (max_confidence >= 0.5 and 
                           (pd.isna(row.get('Classification', '')) or row.get('Classification', '') != 'Low Confidence'))
            else:
                # If emotion columns don't exist, just include the record
                is_valid = True
            
            if is_valid:
                # Keep this record and copy the corresponding frame
                matched_indices.append(index)
                matched_times.add(time_key)
                
                src_path = os.path.join(frames_dir, frame_files[time_key])
                dst_path = os.path.join(sample_output_dir, frame_files[time_key])
                shutil.copy2(src_path, dst_path)
    
    # Find frames that don't match any record
    for time_key, filename in frame_files.items():
        if time_key not in matched_times:
            unmatched_frames.append(filename)
    
    # Create DataFrame with all matched records
    matched_df = df_sample.loc[matched_indices].copy() if matched_indices else pd.DataFrame()
    
    # Save sample-specific Excel
    if not matched_df.empty:
        sample_excel_dir = os.path.join(output_base_dir, f"Sample {sample_num}")
        os.makedirs(sample_excel_dir, exist_ok=True)
        sample_excel_path = os.path.join(sample_excel_dir, f"cleaned_data.xlsx")
        matched_df.to_excel(sample_excel_path, index=False)
    
    # Print summary for this sample
    print(f"\nSample {sample_num} (user_id: {user_id}) Summary:")
    print(f"  - Total records in Excel: {len(df_sample)}")
    print(f"  - Total valid matched records: {len(matched_df)}")
    print(f"  - Total frames without valid records: {len(unmatched_frames)}")
    print(f"  - Records processed without problem filtering")
    
    return matched_df, unmatched_frames

# Main function to process all samples
def process_all_samples(excel_path, base_dir, output_base_dir, sample_mapping):
    # Load the Excel file containing all samples
    print(f"Loading data from {excel_path}...")
    try:
        df_all = pd.read_excel(excel_path)
        print(f"Loaded {len(df_all)} records.")
        
        # Convert timestamp column to datetime if it's not already
        if 'timestamp' in df_all.columns and not pd.api.types.is_datetime64_any_dtype(df_all['timestamp']):
            df_all['timestamp'] = pd.to_datetime(df_all['timestamp'], errors='coerce')
            
        # Print the column names to help identify available columns
        print("\nAvailable columns:", df_all.columns.tolist())
            
    except Exception as e:
        print(f"Error loading Excel file: {e}")
        return
    
    # Create output base directory
    os.makedirs(output_base_dir, exist_ok=True)
    
    # Process each sample
    all_matched_dfs = []
    all_stats = {}
    
    for sample_num, user_id in sample_mapping.items():
        print(f"\nProcessing Sample {sample_num} (user_id: {user_id})...")
        matched_df, unmatched_frames = process_sample(
            sample_num, user_id, df_all, base_dir, output_base_dir
        )
        
        if not matched_df.empty:
            all_matched_dfs.append(matched_df)
        
        all_stats[sample_num] = {
            'user_id': user_id,
            'matched_records': len(matched_df),
            'unmatched_frames': len(unmatched_frames)
        }
    
    # Combine all matched records into a single DataFrame
    if all_matched_dfs:
        combined_df = pd.concat(all_matched_dfs, ignore_index=True)
        # Save the cleaned data
        output_excel = os.path.join(output_base_dir, "all_cleaned_data.xlsx")
        combined_df.to_excel(output_excel, index=False)
        print(f"\nSaved cleaned data to {output_excel}")
    else:
        print("\nNo valid records found across all samples.")

    # Save detailed summary report
    summary_data = []
    for sample_num, stats in all_stats.items():
        summary_data.append({
            'Sample': sample_num,
            'User ID': stats['user_id'],
            'Total Matched Records': stats['matched_records'],
            'Total Unmatched Frames': stats['unmatched_frames']
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_path = os.path.join(output_base_dir, "processing_summary.xlsx")
    summary_df.to_excel(summary_path, index=False)
    print(f"Saved processing summary to {summary_path}")
    
    # Print summary
    print("\n===== SUMMARY =====")
    total_matched = 0
    total_unmatched = 0
    
    for sample_num, stats in all_stats.items():
        matched = stats['matched_records']
        unmatched = stats['unmatched_frames']
        user_id = stats['user_id']
        total_matched += matched
        total_unmatched += unmatched
        print(f"Sample {sample_num} (user_id: {user_id}): {matched} matched records, {unmatched} unmatched frames")
    
    print(f"\nTOTAL: {total_matched} matched records, {total_unmatched} unmatched frames")

# Example usage
if __name__ == "__main__":
    # Configuration
    excel_path = "D:/research/2025_iris_taufik/MultimodalEmoLearn/data/raw/all_samples_data.xlsx"  # Path to your Excel file with all samples
    base_dir = "D:/research/2025_iris_taufik/MultimodalEmoLearn/data/raw"                         # Base directory containing all sample folders
    output_base_dir = "D:/research/2025_iris_taufik/MultimodalEmoLearn/data/processed"          # Where to save cleaned data
    
    # Define mapping between Sample number and user_id
    # This needs to be filled in manually based on your data
    sample_to_user_id = {
        1: 97,   
        2: 117,    
        3: 99,
        4: 100,
        5: 101,
        6: 103,
        7: 102,
        8: 118,
        9: 104,
        10: 106,
        11: 107,
        12: 108,
        13: 109,
        14: 110,
        15: 111,
        16: 112,
        17: 114,
        18: 113,
        19: 115, 
        20: 116
    }
    
    # Run the processing
    process_all_samples(excel_path, base_dir, output_base_dir, sample_to_user_id)