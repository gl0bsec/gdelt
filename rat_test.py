#%%
import os
import urllib.request
from zipfile import ZipFile
from datetime import datetime, timedelta
import pandas as pd
import json

def gen_dates(input_date):
    try:
        # Parse the input date string into a datetime object
        input_date = datetime.strptime(input_date, "%d/%m/%Y")
        
        # Initialize lists to store past and succeeding 15 days
        past_dates = []
        succeeding_dates = []

        # Generate past dates
        for i in range(15, 0, -1):
            past_date = input_date - timedelta(days=i)
            past_dates.append(past_date.strftime("%Y%m%d"))  # Format as yyyymmdd

        # Generate succeeding dates
        for i in range(1, 16):
            succeeding_date = input_date + timedelta(days=i)
            succeeding_dates.append(succeeding_date.strftime("%Y%m%d"))  # Format as yyyymmdd

        return past_dates, succeeding_dates
    except ValueError:
        # Handle the case when the input date is not in the correct format
        return [], []

def download_and_filter_gdelt_data(output_file_path, input_date, locations_regex, themes_regex):
    try:
        # Generate the date range using the input date
        past_dates, succeeding_dates = gen_dates(input_date)

        # Set the directory where the script is located as the working directory
        script_directory = os.path.dirname(os.path.abspath(__file__))
        os.chdir(script_directory)

        # Create an empty list to store the merged data
        merged_data = []

        # Create a directory to store temporary JSON files
        temp_dir = "temp_json"
        os.makedirs(temp_dir, exist_ok=True)

        # Download, filter, and save GDELT data
        for day in past_dates + succeeding_dates:
            day_str = day.replace("/", "")
            url = "http://data.gdeltproject.org/gkg/" + day_str + ".gkg.csv.zip"
            file_name = f"GEvents1_{day_str}.zip"

            # Download the file
            urllib.request.urlretrieve(url, file_name)
            print("Downloaded " + file_name)

            # Extract the file
            with ZipFile(file_name, "r") as zipObj:
                zipObj.extractall()
            os.remove(file_name)

            # Construct the full path to the CSV file
            csv_file_name = os.path.join(script_directory, f"gkg.{day_str}.gkg.csv")

            # Check if the CSV file exists
            if not os.path.exists(csv_file_name):
                print(f"CSV file {csv_file_name} not found for {day_str}. Skipping.")
                continue

            # Load the CSV data into a pandas DataFrame
            df = pd.read_csv(csv_file_name, delimiter="\t")

            # Apply optional filters for locations using contains and regex
            if locations_regex is not None:
                df = df[df['LOCATIONS'].str.contains(locations_regex, case=False, na=False, regex=True)]

            # Apply optional filters for themes using contains and regex
            if themes_regex is not None:
                df = df[df['THEMES'].str.contains(themes_regex, case=False, na=False, regex=True)]

            # Save the filtered data as a temporary JSON file
            temp_json_file = os.path.join(temp_dir, f'output_{day_str}.json')
            df.to_json(temp_json_file, orient='records')
            print(f'Saved filtered data for {day_str} to {temp_json_file}')

        # Merge all the temporary JSON files into one
        merged_json_data = []
        for day in past_dates + succeeding_dates:
            day_str = day.replace("/", "")
            temp_json_file = os.path.join(temp_dir, f'output_{day_str}.json')
            if os.path.exists(temp_json_file):
                with open(temp_json_file, 'r') as json_file:
                    merged_json_data.extend(json.load(json_file))

        # Save the merged and filtered data as a single JSON file
        with open(output_file_path, "w") as output_file:
            json.dump(merged_json_data, output_file)

        print(f"Merged and filtered data saved to {output_file_path}")

    except ValueError:
        print("Invalid date format. Please use 'dd/mm/yyyy'.")

# Example usage:
output_file_path = 'gdelt_tkm.json'
input_date = "26/12/2018"
locations_regex = 'Turkmenistan'  # Adjust the regex pattern as needed
themes_regex = 'HUMAN_RIGHTS|HUMAN_RIGHTS_'  # Adjust the regex pattern as needed
download_and_filter_gdelt_data(output_file_path, input_date, locations_regex, themes_regex)

# %%
