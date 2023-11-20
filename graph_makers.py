import pandas as pd
import json
from itertools import combinations

# Load the JSON file
file_path = 'gdelt_ISR.json'  # Update with the path to your JSON file
with open(file_path, 'r') as file:
    data = json.load(file)

# Convert the data to a DataFrame
df = pd.DataFrame(data)

# Function to extract countries and themes from each row
def extract_countries_themes(row):
    countries = []
    themes = row['THEMES'].split(';') if row['THEMES'] else []

    if row['LOCATIONS']:
        locations = row['LOCATIONS'].split(';')
        for loc in locations:
            parts = loc.split('#')
            if parts[0] == '1':  # Country type location
                country = parts[1]
                countries.append(country)
    
    return countries, themes

# Applying the function to each row
df['Extracted'] = df.apply(extract_countries_themes, axis=1)

# Splitting the tuple into two separate columns
df[['Countries', 'Themes']] = pd.DataFrame(df['Extracted'].tolist(), index=df.index)

# Dropping the original and temporary columns
df.drop(columns=['Extracted', 'LOCATIONS', 'THEMES'], inplace=True)

# Function to generate co-occurrences of countries
def generate_co_occurrences(row):
    co_occurrences = list(combinations(row['Countries'], 2))
    return [(*co_occurrence, row['Themes'], row['DATE'], row['TONE'], row['SOURCES']) for co_occurrence in co_occurrences]

# Generate a list of tuples for all rows
co_occurrences_list = []
for _, row in df.iterrows():
    co_occurrences_list.extend(generate_co_occurrences(row))

# Create a new DataFrame from the list of tuples
co_occurrences_df = pd.DataFrame(co_occurrences_list, columns=['Country1', 'Country2', 'Themes', 'Date', 'Tone', 'Sources'])

# Splitting themes into separate columns
co_occurrences_df = co_occurrences_df.join(co_occurrences_df['Themes'].str.join('|').str.get_dummies())

# Dropping the original 'Themes' column
co_occurrences_df.drop(columns=['Themes'], inplace=True)

# Saving the co-occurrences DataFrame to a CSV file
output_file_path = 'country_co_occurrences.csv'  # Update with your desired file path
co_occurrences_df.to_csv(output_file_path, index=False)

# %%
# Updated script to also group CAMEOEVENTIDS and ORGANIZATIONS

# Function to extract countries, themes, cameoeventids, and organizations from each row
def extract_countries_themes_organizations_cameo(row):
    countries = []
    themes = row['THEMES'].split(';') if row['THEMES'] else []
    cameoeventids = row['CAMEOEVENTIDS'].split(';') if row['CAMEOEVENTIDS'] else []
    organizations = row['ORGANIZATIONS'].split(';') if row['ORGANIZATIONS'] else []

    if row['LOCATIONS']:
        locations = row['LOCATIONS'].split(';')
        for loc in locations:
            parts = loc.split('#')
            if parts[0] == '1':  # Country type location
                country = parts[1]
                countries.append(country)
    
    return countries, themes, cameoeventids, organizations

# Applying the function to each row
df['Extracted'] = df.apply(extract_countries_themes_organizations_cameo, axis=1)

# Splitting the tuple into separate columns
df[['Countries', 'Themes', 'CameoEventIDs', 'Organizations']] = pd.DataFrame(df['Extracted'].tolist(), index=df.index)

# Dropping the original and temporary columns
df.drop(columns=['Extracted', 'LOCATIONS', 'THEMES', 'CAMEOEVENTIDS', 'ORGANIZATIONS'], inplace=True)

# Function to generate co-occurrences of countries with grouped themes, cameoeventids, and organizations
def generate_co_occurrences_with_grouped_data(row):
    co_occurrences = list(combinations(row['Countries'], 2))
    themes = '|'.join(row['Themes'])  # Grouping themes
    cameoeventids = '|'.join(row['CameoEventIDs'])  # Grouping cameoeventids
    organizations = '|'.join(row['Organizations'])  # Grouping organizations
    return [(*co_occurrence, themes, cameoeventids, organizations, row['DATE'], row['TONE'], row['SOURCES']) for co_occurrence in co_occurrences]

# Generate a list of tuples for all rows with grouped data
co_occurrences_list_grouped_all = []
for _, row in df.iterrows():
    co_occurrences_list_grouped_all.extend(generate_co_occurrences_with_grouped_data(row))

# Create a new DataFrame from the list of tuples
co_occurrences_df_grouped_all = pd.DataFrame(co_occurrences_list_grouped_all, 
                                             columns=['Country1', 'Country2', 'Grouped_Themes', 'Grouped_CameoEventIDs', 
                                                      'Grouped_Organizations', 'Date', 'Tone', 'Sources'])

# Saving the new DataFrame to a CSV file
output_file_path_grouped_all = '/data/country_co_occurrences_grouped_all.csv'
co_occurrences_df_grouped_all.to_csv(output_file_path_grouped_all, index=False)

output_file_path_grouped_all