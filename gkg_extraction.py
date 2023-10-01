import pandas as pd 
import os
import glob
import urllib.request
from zipfile import ZipFile
from datetime import datetime, timedelta, date
import pandas as pd
from datetime import datetime
import json


end_date = datetime.today() - timedelta(days=int(input("timedelta: ")))
start_date = end_date - timedelta(days= int(input("how many days? ")))

print("Downloading GDELT GkG 1.0 files")

def date_range(start, end):
    delta = end - start  # as timedelta
    days = [start + timedelta(days=i) for i in range(delta.days + 1)]
    return days

days = [str(day.strftime("%Y%m%d")) for day in date_range(start_date, end_date)]

for day in days:
    urllib.request.urlretrieve("http://data.gdeltproject.org/gkg/" + day + ".gkg.csv.zip", "gdelt_extraction/dump/" + "GEvents1" + day + ".zip",)
    print("donwloaded "+"GEvents1" + day + ".zip")
    

print("Unzipping files")
test_dir = os.listdir("gdelt_extraction/dump")
for n in test_dir:
    with ZipFile("gdelt_extraction/dump/" + n, "r") as zipObj:
        zipObj.extractall("gdelt_extraction/results")

gdelt_extraction_dir = glob.glob("gdelt_extraction/dump/*")
for f in gdelt_extraction_dir:
    os.remove(f) 
    
results_dir = os.listdir("gdelt_extraction/results")
for result in results_dir:
    print(result)
    test = pd.read_csv('gdelt_extraction/results/'+result,delimiter="\t")
    test.to_json('outputs/'+result[:8]+'.json', orient='records')
    print('saved '+result+" to outputs")   

file_paths  = os.listdir("outputs")

merged_data = []

# Read each file, extract the data, and merge
for file_path in file_paths:
    with open('outputs/'+file_path, "r") as file:
        data = json.load(file)
        for d in data:
            merged_data.append(d)

gdelt_extraction_dir = glob.glob("outputs/*")
for f in gdelt_extraction_dir:
    os.remove(f) 
# Save the merged data as a single JSON file
output_file_path = "test_.json"  # Replace with your desired output file path
with open(output_file_path, "w") as output_file:
    json.dump(merged_data, output_file)

print(f"Merged data saved to {output_file_path}")

import matplotlib.pyplot as plt
# Load the dataset
data = pd.read_json('test_.json', orient='records', lines=True)
formatted_data = pd.json_normalize(data.iloc[0])

# Filter the dataset for migration and refugees related entries
migration_refugee_data = formatted_data[formatted_data['THEMES'].str.contains('MIGRATION|REFUGEE', case=False, na=False, regex=True)]
migration_refugee_data.to_csv("sample_migration_data2.csv", index=False)