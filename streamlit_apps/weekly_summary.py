#%% 
# Import dependencies 
from datetime import datetime, timedelta, date
import matplotlib.pyplot as plt
import matplotlib.style as style
from collections import Counter
from zipfile import ZipFile
import networkx as nx
import seaborn as sns
import pandas as pd
import numpy as np
import urllib
import json
import glob 
# import viz 
import os 


#%% 
# Download and filter data
end_date = datetime.today() - timedelta(days=2)
start_date = end_date - timedelta(days= 7)

print("Downloading GDELT GkG 1.0 files for"+str(start_date)+" to "+str(end_date))

def date_range(start, end):
    delta = end - start  # as timedelta
    days = [start + timedelta(days=i) for i in range(delta.days + 1)]
    return days

days = [str(day.strftime("%Y%m%d")) for day in date_range(start_date, end_date)]

for day in days:
    urllib.request.urlretrieve("http://data.gdeltproject.org/gkg/" + day + ".gkg.csv.zip", "dump/" + "GEvents1" + day + ".zip",)
    print("donwloaded "+"GEvents1" + day + ".zip")
    

print("Unzipping files")
test_dir = os.listdir("dump")
for n in test_dir:
    with ZipFile("dump/" + n, "r") as zipObj:
        zipObj.extractall("results")

gdelt_extraction_dir = glob.glob("dump/*")
for f in gdelt_extraction_dir:
    os.remove(f) 
    
results_dir = os.listdir("results")
for result in results_dir:
    test = pd.read_csv('results/'+result,delimiter="\t")
    test.to_json('outputs/'+result[:8]+'.json', orient='records')
    print('saved '+result+" to outputs")   

gdelt_extraction_dir = glob.glob("results/*")
for f in gdelt_extraction_dir:
    os.remove(f) 

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
output_file_path = 'gkg1.0_7day'+".json"  # Replace with your desired output file path
with open(output_file_path, "w") as output_file:
    json.dump(merged_data, output_file)

print(f"Merged data saved to {output_file_path}")


#%% 
# Assoc locations map
df = pd.read_json('gkg1.0_7day.json')

import geopandas as gpd
def filter_and_count_locations(df):
    """
    Filters the DataFrame for entries related to a given country and counts the occurrences of each location.
    """
    location_counter = Counter()
    
    for locations_str in df['LOCATIONS']:
        locations_list = locations_str.split(';')
        for location in locations_list:
            location_name = location.split('#')[1]
            location_counter[location_name] += 1
            
    return location_counter

def plot_minimalist_map(counter, title, color_map='Greens', power=0.5):
    """
    Plot locations on a world map color-coded by their counts, in a minimalist style.
    """
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    locations_df = pd.DataFrame.from_dict(counter, orient='index', columns=['Count']).reset_index()
    locations_df = locations_df.rename(columns={'index': 'name'})
    locations_df['Adjusted_Count'] = np.power(locations_df['Count'], power)
    world = world.merge(locations_df, how='left', left_on='name', right_on='name')
    
    fig, ax = plt.subplots(1, 1, figsize=(15, 10))
    ax.axis('off')
    world.plot(column='Adjusted_Count', ax=ax, cmap=color_map, missing_kwds={'color': 'lightgrey'}, linewidth=0)
    plt.title(title, fontdict={'fontsize': 16})
    plt.show()
    
plot_minimalist_map(filter_and_count_locations(df), 'Country heatmap', color_map='Greens', power=0.5)


# %%
#Attribute counts
from collections import Counter
d = 0
def generate_visualizations(df, top_n):

    # Calculate the overall tone for each entry related to the selected country
    df['OVERALL_TONE'] = df['TONE'].apply(lambda x: float(x.split(',')[0]) if pd.notnull(x) else None)

    # Themes
    all_themes = ';'.join(df['THEMES'].dropna()).split(';')
    theme_counter = Counter(all_themes)
    theme_names_n = [theme[0] for theme in theme_counter.most_common(top_n)]
    theme_counts_n = [theme[1] for theme in theme_counter.most_common(top_n)]
    theme_tones = {}
    for theme in theme_names_n:
        theme_df = df[df['THEMES'].str.contains(theme, case=False, na=False)]
        theme_tones[theme] = theme_df['OVERALL_TONE'].mean()
    theme_colors_reversed = [sns.color_palette("coolwarm_r", as_cmap=True)(0.5 + tone / 10) for tone in [theme_tones[theme] for theme in theme_names_n]]
    
    # Organizations
    all_orgs = ';'.join(df['ORGANIZATIONS'].dropna()).split(';')
    org_counter = Counter(all_orgs)
    org_names_n = [org[0] for org in org_counter.most_common(top_n)]
    org_counts_n = [org[1] for org in org_counter.most_common(top_n)]
    org_tones = {}
    for org in org_names_n:
        org_df = df[df['ORGANIZATIONS'].str.contains(org, case=False, na=False)]
        org_tones[org] = org_df['OVERALL_TONE'].mean()
    org_colors_reversed = [sns.color_palette("coolwarm_r", as_cmap=True)(0.5 + tone / 10) for tone in [org_tones[org] for org in org_names_n]]

    # Plots
    ## Themes
    plt.figure(figsize=(15, 8))
    barplot_themes_reversed = sns.barplot(x=theme_counts_n, y=theme_names_n, palette=theme_colors_reversed)
    for index, p in enumerate(barplot_themes_reversed.patches):
        avg_tone = theme_tones[theme_names_n[index]]
        barplot_themes_reversed.annotate(f'{avg_tone:.2f}', (p.get_width() / 2, p.get_y() + p.get_height() / 2), ha='center', va='center', color='white')
    plt.title(f'Top {top_n} Themes')
    plt.show()

    ## Organizations
    plt.figure(figsize=(15, 8))
    barplot_orgs_reversed = sns.barplot(x=org_counts_n, y=org_names_n, palette=org_colors_reversed)
    for index, p in enumerate(barplot_orgs_reversed.patches):
        avg_tone = org_tones[org_names_n[index]]
        barplot_orgs_reversed.annotate(f'{avg_tone:.2f}', (p.get_width() / 2, p.get_y() + p.get_height() / 2), ha='center', va='center', color='white')
    plt.title(f'Top {top_n} Organizations')
    plt.show()
    
generate_visualizations(df, 30-d)

# Viz countries
# Calculate the overall tone for each entry related to Iran
df['OVERALL_TONE'] = df['TONE'].apply(lambda x: float(x.split(',')[0]) if pd.notnull(x) else None)

# Calculate the overall tone for the entire dataset
df['OVERALL_TONE'] = df['TONE'].apply(lambda x: float(x.split(',')[0]) if pd.notnull(x) else None)

### Themes
all_themes = ';'.join(df['THEMES'].dropna()).split(';')
theme_counter = Counter(all_themes)
theme_names_20 = [theme[0] for theme in theme_counter.most_common(N-1) if theme[0] != '']
theme_counts_20 = [theme[1] for theme in theme_counter.most_common(N-1) if theme[0] != '']
theme_tones = {}
for theme in theme_names_20:
    theme_df = df[df['THEMES'].str.contains(theme, case=False, na=False)]
    theme_tones[theme] = theme_df['OVERALL_TONE'].mean()
theme_colors_reversed = [sns.color_palette("coolwarm_r", as_cmap=True)(0.5 + tone / 10) for tone in [theme[1] for theme in sorted(theme_tones.items(), key=lambda x: x[1])]]

### Countries
country_tones = []
for _, row in df.iterrows():
    if pd.isna(row['LOCATIONS']) or pd.isna(row['OVERALL_TONE']):
        continue
    locations = row['LOCATIONS'].split(';')
    for location in locations:
        parts = location.split('#')
        if len(parts) < 2:
            continue
        country_name = parts[1]
        tone = row['OVERALL_TONE']
        country_tones.append((country_name, tone))
df_country_tones = pd.DataFrame(country_tones, columns=['Country', 'Tone'])
df_country_avg_tone = df_country_tones.groupby('Country')['Tone'].mean().reset_index().sort_values(by='Tone')
df_country_count_top20 = df_country_tones['Country'].value_counts().reset_index()
df_country_count_top20.columns = ['Country', 'Count']
df_country_count_top20 = df_country_count_top20.nlargest(N-1, 'Count')
df_country_count_top20 = df_country_count_top20.merge(df_country_avg_tone, on='Country', how='left').sort_values(by='Count')
country_colors_reversed = [sns.color_palette("coolwarm_r", as_cmap=True)(0.5 + tone / 10) for tone in df_country_count_top20['Tone']]

plt.figure(figsize=(15, 8))
barplot_countries_reversed = sns.barplot(x='Count', y='Country', data=df_country_count_top20, palette=country_colors_reversed)
for index, p in enumerate(barplot_countries_reversed.patches):
    avg_tone = df_country_count_top20.iloc[index]['Tone']
    barplot_countries_reversed.annotate(f'{avg_tone:.2f}', (p.get_width() / 2, p.get_y() + p.get_height() / 2), ha='center', va='center', color='white')
plt.show()


# %%
# Timeline heatmap
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
    return theme_tone_date_dict

def plot_side_by_side_heatmaps(theme_counts_df, theme_tone_df, top_N_themes, date_labels):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(30, 10), sharey=True)
    sns.heatmap(theme_counts_df.T, cmap='coolwarm', cbar=True, annot=False, ax=ax1)
    ax1.set_title('Counts of Top 20 Themes Over Dates')
    ax1.set_xlabel('Dates (Day-Month-Year)')
    ax1.set_xticklabels(date_labels, rotation=45)
    sns.heatmap(theme_tone_df.T, cmap='coolwarm_r', cbar=True, annot=False, ax=ax2)  # Reversed color map
    ax2.set_title('Average Tone of Top 20 Themes Over Dates')
    ax2.set_xlabel('Dates (Day-Month-Year)')
    ax2.set_xticklabels(date_labels, rotation=45)
    plt.suptitle('Side by Side Heatmaps of Theme Counts and Average Tone Over Dates')
    plt.ylabel('Top 20 Themes')
    plt.show()

# Convert the DATE column to a more readable date format
df['DATE'] = pd.to_datetime(df['DATE'], format='%Y%m%d')

# Clean up the THEMES column
df['THEMES'] = df['THEMES'].apply(lambda x: str(x).split(';') if pd.notnull(x) else [])

# Aggregate theme counts and tones
theme_date_dict = aggregate_theme_counts(df)
theme_tone_date_dict = aggregate_theme_tone(df)

# Convert to DataFrames
theme_counts_df = pd.DataFrame.from_dict({
    date: theme_data for date, theme_data in theme_date_dict.items()
}, orient='index').fillna(0)

theme_tone_df = pd.DataFrame.from_dict({
    date: {theme: data['sum_tone'] / data['count'] for theme, data in theme_data.items()}
    for date, theme_data in theme_tone_date_dict.items()
}, orient='index').fillna(0)

# Filter out the specified themes and limit to top N themes
themes_to_exclude = ["TAX_FNCACT", "", 'CRISISLEX_CRISISLEXREC']
theme_counts_df_filtered = theme_counts_df.drop(columns=themes_to_exclude, errors='ignore')
theme_tone_df_filtered = theme_tone_df.drop(columns=themes_to_exclude, errors='ignore')

N = 30
top_N_themes = theme_counts_df_filtered.sum().sort_values(ascending=False).head(N).index.tolist()
theme_counts_df_top_N = theme_counts_df_filtered[top_N_themes]
theme_tone_df_top_N = theme_tone_df_filtered[top_N_themes]

# Sort the DataFrames by date for better visualization
theme_counts_df_top_N.sort_index(inplace=True)
theme_tone_df_top_N.sort_index(inplace=True)

# Generate example plot
date_labels = theme_counts_df_top_N.index.strftime('%d-%m-%Y')
plot_side_by_side_heatmaps(theme_counts_df_top_N, theme_tone_df_top_N, top_N_themes, date_labels)

