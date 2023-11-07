from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
import urllib
import json
from zipfile import ZipFile
from collections import Counter
import os
import glob

# Download and filter data
def download_data(start_date, end_date, download_dir="dump/"):
    print(f"Downloading GDELT GkG 1.0 files for {start_date} to {end_date}")
    days = [day.strftime("%Y%m%d") for day in date_range(start_date, end_date)]
    for day in days:
        urllib.request.urlretrieve(
            f"http://data.gdeltproject.org/gkg/{day}.gkg.csv.zip",
            f"{download_dir}GEvents1{day}.zip",
        )
        print(f"downloaded GEvents1{day}.zip")
    print("Unzipping files")
    for n in os.listdir(download_dir):
        with ZipFile(f"{download_dir}{n}", "r") as zipObj:
            zipObj.extractall("results")

def clean_up_files(file_paths):
    for f in file_paths:
        os.remove(f)

# Read and merge data from JSON files
def merge_data_from_json(file_dir):
    file_paths = os.listdir(file_dir)
    merged_data = []
    for file_path in file_paths:
        with open(f'{file_dir}{file_path}', "r") as file:
            data = json.load(file)
            merged_data.extend(data)
    return merged_data

# Save data to a JSON file
def save_data_to_json(data, output_file_path):
    with open(output_file_path, "w") as output_file:
        json.dump(data, output_file)
    print(f"Merged data saved to {output_file_path}")

# Create a date range
def date_range(start, end):
    delta = end - start
    return [start + timedelta(days=i) for i in range(delta.days + 1)]

# Filter and count locations
def filter_and_count_locations(df):
    location_counter = Counter()
    for locations_str in df['LOCATIONS']:
        locations_list = locations_str.split(';')
        for location in locations_list:
            location_name = location.split('#')[1]
            location_counter[location_name] += 1
    return location_counter

# Plot minimalist map
def plot_minimalist_map(counter, title, color_map='Greens', power=0.5):
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    locations_df = pd.DataFrame.from_dict(counter, orient='index', columns=['Count']).reset_index()
    locations_df = locations_df.rename(columns={'index': 'name'})
    locations_df['Adjusted_Count'] = np.power(locations_df['Count'], power)
    world = world.merge(locations_df, how='left', left_on='name', right_on='name')
    fig, ax = plt.subplots(1, 1, figsize=(15, 10))
    ax.axis('off')
    world.plot(column='Adjusted_Count', ax=ax, cmap=color_map, missing_kwds={'color': 'lightgrey'}, linewidth=0)
    plt.title(title, fontdict={'fontsize': 16})
    return fig

# Generate visualizations
def generate_visualizations(df, top_n):
    # This function needs to be adapted to your specific visualization needs.
    # The code you provided for visualization is a starting point.
    # You can split it into smaller functions if needed.
    pass

# Prepare data for the heatmap
def prepare_heatmap_data(df, themes_to_exclude, top_n):
    df['THEMES'] = df['THEMES'].apply(lambda x: str(x).split(';') if pd.notnull(x) else [])
    theme_counts = aggregate_theme_counts(df)
    theme_tones = aggregate_theme_tone(df)
    theme_counts_df = pd.DataFrame.from_dict(theme_counts, orient='index').fillna(0)
    theme_tone_df = pd.DataFrame.from_dict(theme_tones, orient='index').fillna(0)
    theme_counts_df = theme_counts_df.drop(columns=themes_to_exclude, errors='ignore')
    theme_tone_df = theme_tone_df.drop(columns=themes_to_exclude, errors='ignore')
    top_themes = theme_counts_df.sum().sort_values(ascending=False).head(top_n).index.tolist()
    theme_counts_df = theme_counts_df[top_themes]
    theme_tone_df = theme_tone_df[top_themes]
    return theme_counts_df, theme_tone_df

# Plot side by side heatmaps
def plot_side_by_side_heatmaps(theme_counts_df, theme_tone_df, date_labels):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(30, 10), sharey=True)
    sns.heatmap(theme_counts_df.T, cmap='coolwarm', cbar=True, annot=False, ax=ax1)
    sns.heatmap(theme_tone_df.T, cmap='coolwarm_r', cbar=True, annot=False, ax=ax2)
    ax1.set_title('Counts of Top Themes Over Dates')
    ax1.set_xticklabels(date_labels, rotation=45)
    ax2.set_title('Average Tone of Top Themes Over Dates')
    ax2.set_xticklabels(date_labels, rotation=45)
    plt.suptitle('Side by Side Heatmaps of Theme Counts and Average Tone Over Dates')
    return fig

# Additional helper functions
def aggregate_theme_counts(df):
    theme_date_dict = {}
    for _, row in df.iterrows():
        date = row['DATE']
        themes = row['THEMES']
        if date not in theme_date_dict:
            theme_date_dict[date] = Counter()
        theme_date_dict[date].update(themes)
    return theme_date_dict

def aggregate_theme_tone(df):
    theme_tone_date_dict = {}
    for _, row in df.iterrows():
        date = row['DATE']
        themes = row['THEMES']
        avg_tone = float(str(row['TONE']).split(',')[0]) if pd.notnull(row['TONE']) else 0.0
        if date not in theme_tone_date_dict:
            theme_tone_date_dict[date] = Counter()
        for theme in themes:
            if theme not in theme_tone_date_dict[date]:
                theme_tone_date_dict[date][theme] = {'sum_tone': 0.0, 'count': 0}
            theme_tone_date_dict[date][theme]['sum_tone'] += avg_tone
            theme_tone_date_dict[date][theme]['count'] += 1
    return {date: {theme: data['sum_tone'] / data['count'] for theme, data in theme_data.items()} for date, theme_data in theme_tone_date_dict.items()}
