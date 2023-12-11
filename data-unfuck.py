import json
import os
import pandas as pd
from datetime import datetime
import shutil

def convert_date_format(date_str):
    """Convert date to DD-MM-YYYY format."""
    try:
        return datetime.strptime(date_str, '%Y%m%d').strftime('%d-%m-%Y')
    except ValueError:
        return date_str  # Return original if format is incorrect

def process_json_file(file_path, to_csv=False):
    """Process the JSON file to fix the DATE field and optionally convert to CSV."""
    try:
        # Read JSON file
        with open(file_path, 'r') as file:
            data = json.load(file)

        # Process each entry
        for entry in data:
            if 'DATE' in entry:
                entry['DATE'] = convert_date_format(entry['DATE'])

        # Convert to CSV if requested
        if to_csv:
            df = pd.DataFrame(data)
            csv_file_path = file_path.rsplit('.', 1)[0] + '.csv'
            df.to_csv(csv_file_path, index=False)
            print(f"Converted JSON to CSV: {csv_file_path}")
        else:
            # Write the modified JSON back to a file
            with open(file_path, 'w') as file:
                json.dump(data, file)
            print(f"Processed JSON file with updated dates: {file_path}")

    except Exception as e:
        print(f"An error occurred: {e}")

def main():
    # Print welcome message center-aligned
    term_width = shutil.get_terminal_size((80, 20)).columns
    print("Welcome to Data Un-Fucker".center(term_width))

    # Ask user for file path
    file_path = input("Enter the path to the JSON file: ").strip()
    
    # Check if file exists
    if not os.path.exists(file_path):
        print("File does not exist. Please check the path.")
        return

    # Ask user if they want to convert to CSV
    convert_to_csv = input("Do you want to convert the JSON to CSV? (yes/no): ").strip().lower() == 'yes'

    # Process the file
    process_json_file(file_path, convert_to_csv)

if __name__ == "__main__":
    main()
