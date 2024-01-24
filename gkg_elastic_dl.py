import argparse
import os
import urllib.request
from zipfile import ZipFile
from datetime import datetime, timedelta
import pandas as pd
from tqdm import tqdm
import data_helpers as el
import glob

# Functions
def gen_dates(input_date):
    try:
        input_date = datetime.strptime(input_date, "%d/%m/%Y")
        past_dates = [input_date - timedelta(days=i) for i in range(15, 0, -1)]
        succeeding_dates = [input_date + timedelta(days=i) for i in range(1, 16)]
        return [d.strftime("%Y%m%d") for d in past_dates + succeeding_dates]
    except ValueError:
        return []

def dates_from(delta, number):
    end_date = datetime.today() - timedelta(delta)
    start_date = end_date - timedelta(number)
    days = [start_date + timedelta(days=i) for i in range(number)]
    return [day.strftime("%Y%m%d") for day in days]

# Command line argument parsing
parser = argparse.ArgumentParser(description="GDELT Data Processor")
parser.add_argument('--n', type=int, help='Number of days to download', required=True)
parser.add_argument('--k', type=int, default=1, help='Days back from today')
parser.add_argument('--input_date', type=str, help='Specific date to start from (dd/mm/yyyy)')
parser.add_argument('--locations_regex', type=str, help='Regex pattern for filtering locations')
parser.add_argument('--themes_regex', type=str, help='Regex pattern for filtering locations')
args = parser.parse_args()

# Assigning arguments to variables
n = args.n
k = args.k
input_date = args.input_date
locations_regex = args.locations_regex

# Generate date range
if input_date:
    date_range = gen_dates(input_date)
else:
    date_range = dates_from(k, n)

# Set working directory and create temp directory
script_directory = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_directory)
temp_dir = 'big_dump/'
os.makedirs(temp_dir, exist_ok=True)

# Download and process data
for day in tqdm(date_range, desc="Downloading GDELT data"):
    url = f"http://data.gdeltproject.org/gkg/{day}.gkg.csv.zip"
    urllib.request.urlretrieve(url, os.path.join(temp_dir, f"GEvents1{day}.zip"))

    with ZipFile(os.path.join(temp_dir, f"GEvents1{day}.zip"), "r") as zipObj:
        zipObj.extractall(temp_dir)
    os.remove(os.path.join(temp_dir, f"GEvents1{day}.zip"))

    df = pd.read_csv(os.path.join(temp_dir, f"{day}.gkg.csv"), delimiter="\t", error_bad_lines=False)

    if locations_regex:
        df = df[df['LOCATIONS'].str.contains(locations_regex, case=False, na=False, regex=True)]

    df.to_json(os.path.join(temp_dir, f'output_{day}.json'), orient='records')
    os.remove(os.path.join(temp_dir, f"{day}.gkg.csv"))

print('Downloaded all files')

# Load data to Elasticsearch
filenames = glob.glob(os.path.join(temp_dir, "*.json"))
for path in filenames:
    print(f'Loading data: {path}')
    el.create_and_load_es_index(9200, path, '3week_test')
    print('Loaded')

if __name__ == "__main__":
    # Add any additional code to be executed if needed
    pass
