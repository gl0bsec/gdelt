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
import data_helpers as el
import glob
n = 25
# k = 1 not in use 
input_date = None
locations_regex = None
# Israel|Iraq|Saudi Arabia|Yemen|'  # Adjust the regex pattern as needed

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

        return past_dates, input_date + succeeding_dates
    except ValueError:
        # Handle the case when the input date is not in the correct format
        return []+[]

def dates_from(delta,number):
    end_date = datetime.today() - timedelta(delta)
    start_date = end_date - timedelta(number)

    print("Downloading GDELT GkG 1.0 files")
    def date_range(start, end):
        delta = end - start  # as timedelta
        days = [start + timedelta(days=i) for i in range(delta.days + 1)]
        return days

    days = [str(day.strftime("%Y%m%d")) for day in date_range(start_date, end_date)]
    return days


if n != None:
    date_range = dates_from(1,n)
else:
    past_dates, succeeding_dates = gen_dates(input_date)
    date_range = past_dates + succeeding_dates

# Set the directory where the script is located as the working directory
script_directory = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_directory)

# Create an empty list to store the merged data
merged_data = []

# Create a directory to store temporary JSON files
temp_dir = 'big_dump/'
os.makedirs(temp_dir, exist_ok=True)

# Download, filter, and save GDELT data
for day in tqdm(date_range, desc="Downloading GDELT data"):
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

print('donwloaded all files')

n_days = list(range(1,n+1))
filenames = glob.glob("big_dump/*.json")
for path in filenames:
    print('loading data: '+path)
    el.create_and_load_es_index(9200, path, 'gulf')
    print('loaded')
    print(' ')

# %%
