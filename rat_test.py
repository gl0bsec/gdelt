#%%
import os
import urllib.request
from zipfile import ZipFile
from datetime import datetime, timedelta
import pandas as pd
import json
import nbformat
from nbconvert import MarkdownExporter
import sys
from tqdm import tqdm  

#%%
# Set parameters
output_file_path = 'gdelt_mlt.json'
input_date = "06/06/2014"
locations_regex = 'Malta'  # Adjust the regex pattern as needed

#%% 
#Download files 
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
    
def dates_from(delta,number):
    end_date = datetime.today() - delta
    start_date = end_date - number

    print("Downloading GDELT GkG 1.0 files")
    def date_range(start, end):
        delta = end - start  # as timedelta
        days = [start + timedelta(days=i) for i in range(delta.days + 1)]
        return days

    days = [str(day.strftime("%Y%m%d")) for day in date_range(start_date, end_date)]
    return

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
        for day in tqdm(past_dates + succeeding_dates, desc="Downloading GDELT data"):
            # Download the file
            day_str = day.replace("/", "")
            url = "http://data.gdeltproject.org/gkg/" + day_str + ".gkg.csv.zip"
            
            urllib.request.urlretrieve(url, temp_dir+"/" + "GEvents1" + day_str + ".zip",)

            # Extract the file
            with ZipFile(temp_dir+"/" + "GEvents1" + day_str + ".zip", "r") as zipObj:
                zipObj.extractall(temp_dir+"/")
            os.remove(temp_dir+"/" + "GEvents1" + day_str + ".zip")

            # Load the CSV data into a pandas DataFrame
            df = pd.read_csv(temp_dir+"/" + day_str + ".gkg.csv", delimiter="\t")
            
            # # Apply optional filters for themes using contains and regex
            # if themes_regex is not None:
            #     df = df[df['THEMES'].str.contains(themes_regex, case=False, na=False, regex=True)]

            # Apply optional filters for locations using contains and regex
            if locations_regex is not None:
                df = df[df['LOCATIONS'].str.contains(locations_regex, case=False, na=False, regex=True)]

            # Save the filtered data as a temporary JSON file
            temp_json_file = os.path.join(temp_dir, f'output_{day_str}.json')
            df.to_json(temp_json_file, orient='records')
            os.remove(temp_dir+"/" + day_str + ".gkg.csv")

            # Append the data to the merged list
            with open(temp_json_file, 'r') as json_file:
                merged_data.extend(json.load(json_file))
            os.remove(temp_json_file)  # Remove the temporary JSON file

        # Save the merged and filtered data as a single JSON file
        with open(output_file_path, "w") as output_file:
            json.dump(merged_data, output_file)

        print(f"Merged and filtered data saved to {output_file_path}")

    except ValueError:
        print("Invalid date format. Please use 'dd/mm/yyyy'.") 
    return

# themes_regex = 'HUMAN_RIGHTS|HUMAN_RIGHTS_'  # Adjust the regex pattern as needed
download_and_filter_gdelt_data(output_file_path, input_date, locations_regex, None)