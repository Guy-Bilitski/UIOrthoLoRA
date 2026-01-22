import json
import sys
import os
from collections import defaultdict

def main():
    # Use the filename from your example as default, or take from command line args
    default_filename = "google_gemma-3-12b-it_scores-final.jsonl"
    
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    elif os.path.exists(default_filename):
        file_path = default_filename
    else:
        print(f"Usage: python count_scores.py <path_to_jsonl_file>")
        print(f"Could not find default file: {default_filename}")
        return

    print(f"Processing file: {file_path}")
    
    # Initialize dictionary to store counts
    score_counts = defaultdict(int)
    total_samples = 0
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                try:
                    data = json.loads(line)
                    # Extract the score from the nested structure
                    # Structure: row -> base_eval -> score
                    base_eval = data.get("base_eval", {})
                    score = base_eval.get("score")
                    
                    if score is not None:
                        # Round to 1 decimal place to group 0.999... or floating point artifacts
                        rounded_score = round(float(score), 1)
                        score_counts[rounded_score] += 1
                        total_samples += 1
                        
                except json.JSONDecodeError:
                    print("Skipping invalid JSON line")
                    continue
                except ValueError:
                    print("Skipping entry with invalid score format")
                    continue

        # Print the Results
        print("\n" + "="*40)
        print(f"{'Score':<10} | {'Count':<10} | {'Percentage':<10}")
        print("="*40)
        
        # Sort by score (0.0 to 1.0)
        # We explicitly iterate 0.0 to 1.0 to show zeros for missing buckets if needed,
        # or just iterate the found keys. 
        # Ideally, we show all 0.0-1.0 buckets for completeness.
        all_possible_scores = [round(x * 0.1, 1) for x in range(11)]
        
        for score in all_possible_scores:
            count = score_counts[score]
            percentage = (count / total_samples * 100) if total_samples > 0 else 0
            print(f"{score:<10.1f} | {count:<10} | {percentage:<9.1f}%")
            
        print("="*40)
        print(f"Total Samples: {total_samples}")

    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
