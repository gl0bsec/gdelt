import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import json
import os
import tempfile
import urllib.request
from collections import Counter
import numpy as np
import zipfile

# Assuming the existence of these directories
TEMP_JSON_DIR = tempfile.mkdtemp()
GDELT_DATA_DIR = "gdelt_data"
os.makedirs(GDELT_DATA_DIR, exist_ok=True)

# Function to generate past and succeeding dates from a given date
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
    

# Function to download GDELT data
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
            
            # Apply optional filters for themes using contains and regex
            if themes_regex is not None:
                df = df[df['THEMES'].str.contains(themes_regex, case=False, na=False, regex=True)]

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
        

# Function to generate visualizations
def generate_visualizations(df, country_name, top_n):
    if country_name is None: 
        country_df = df 
    else: 
        country_df = df[df['LOCATIONS'].str.contains(country_name, case=False, na=False)].reset_index(drop=True)

    # Calculate the overall tone for each entry related to the selected country
    country_df['OVERALL_TONE'] = country_df['TONE'].apply(lambda x: float(x.split(',')[0]) if pd.notnull(x) else None)

    # Themes
    all_themes = ';'.join(country_df['THEMES'].dropna()).split(';')
    theme_counter = Counter(all_themes)
    theme_names_n = [theme[0] for theme in theme_counter.most_common(top_n)]
    theme_counts_n = [theme[1] for theme in theme_counter.most_common(top_n)]
    theme_tones = {}
    for theme in theme_names_n:
        theme_df = country_df[country_df['THEMES'].str.contains(theme, case=False, na=False)]
        theme_tones[theme] = theme_df['OVERALL_TONE'].mean()
    theme_colors_reversed = [sns.color_palette("coolwarm_r", as_cmap=True)(0.5 + tone / 10) for tone in [theme_tones[theme] for theme in theme_names_n]]
    
    # Organizations
    all_orgs = ';'.join(country_df['ORGANIZATIONS'].dropna()).split(';')
    org_counter = Counter(all_orgs)
    org_names_n = [org[0] for org in org_counter.most_common(top_n)]
    org_counts_n = [org[1] for org in org_counter.most_common(top_n)]
    org_tones = {}
    for org in org_names_n:
        org_df = country_df[country_df['ORGANIZATIONS'].str.contains(org, case=False, na=False)]
        org_tones[org] = org_df['OVERALL_TONE'].mean()
    org_colors_reversed = [sns.color_palette("coolwarm_r", as_cmap=True)(0.5 + tone / 10) for tone in [org_tones[org] for org in org_names_n]]

    # Plots
    ## Themes
    plt.figure(figsize=(15, 8))
    barplot_themes_reversed = sns.barplot(x=theme_counts_n, y=theme_names_n, palette=theme_colors_reversed)
    for index, p in enumerate(barplot_themes_reversed.patches):
        avg_tone = theme_tones[theme_names_n[index]]
        barplot_themes_reversed.annotate(f'{avg_tone:.2f}', (p.get_width() / 2, p.get_y() + p.get_height() / 2), ha='center', va='center', color='white')
    plt.title(f'Top {top_n} Themes Related to {country_name}')
    plt.show()

    ## Organizations
    plt.figure(figsize=(15, 8))
    barplot_orgs_reversed = sns.barplot(x=org_counts_n, y=org_names_n, palette=org_colors_reversed)
    for index, p in enumerate(barplot_orgs_reversed.patches):
        avg_tone = org_tones[org_names_n[index]]
        barplot_orgs_reversed.annotate(f'{avg_tone:.2f}', (p.get_width() / 2, p.get_y() + p.get_height() / 2), ha='center', va='center', color='white')
    plt.title(f'Top {top_n} Organizations Related to {country_name}')
    plt.show()
    
    # Countries
    country_tones = []
    for _, row in filtered_df.iterrows():
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
    df_country_count_top20 = df_country_count_top20.nlargest(top_n-1, 'Count')
    df_country_count_top20 = df_country_count_top20.merge(df_country_avg_tone, on='Country', how='left').sort_values(by='Count')
    country_colors_reversed = [sns.color_palette("coolwarm_r", as_cmap=True)(0.5 + tone / 10) for tone in df_country_count_top20['Tone']]

    plt.figure(figsize=(15, 8))
    barplot_countries_reversed = sns.barplot(x='Count', y='Country', data=df_country_count_top20, palette=country_colors_reversed)
    for index, p in enumerate(barplot_countries_reversed.patches):
        avg_tone = df_country_count_top20.iloc[index]['Tone']
        barplot_countries_reversed.annotate(f'{avg_tone:.2f}', (p.get_width() / 2, p.get_y() + p.get_height() / 2), ha='center', va='center', color='white')
    plt.show()
    
    return
# Function to plot data from DataFrame
def plot_from_dataframe(df, shade_date_str):
    # Process the DataFrame to calculate the number of entries per day
    df['Date'] = pd.to_datetime(df['DATE'], format='%Y%m%d')
    entries_per_day = df.groupby('Date').size()

    # Process the DataFrame to calculate the average sentiment per day
    df['Sentiment'] = df['TONE'].apply(lambda x: float(x.split(',')[1]) if x is not None else None)
    sentiment_per_day = df.groupby('Date')['Sentiment'].mean()


    # Combine the entries and sentiment into one DataFrame for plotting
    combined_data = pd.DataFrame({'Entries': entries_per_day, 'Sentiment': sentiment_per_day}).sort_index()

    # Convert the shading date from string to datetime
    shade_date = datetime.strptime(shade_date_str, '%Y%m%d')
    shade_start = shade_date - pd.Timedelta(days=1)
    shade_end = shade_date + pd.Timedelta(days=1)

    # Create two subplots, with the sentiment plot below the counts plot, and add wider vertical shading
    fig, ax = plt.subplots(2, 1, figsize=(20, 14), sharex=True)

    # Plot number of entries in the first subplot
    ax[0].plot(combined_data.index, combined_data['Entries'], marker='o', linestyle='-', color='blue')
    ax[0].axvspan(shade_start, shade_end, color='grey', alpha=0.5)
    ax[0].set_title('Number of Entries Per Day', fontsize=16)
    ax[0].set_ylabel('Number of Entries', fontsize=14)

    # Plot average sentiment in the second subplot
    ax[1].plot(combined_data.index, combined_data['Sentiment'], marker='x', linestyle='-', color='red')
    ax[1].axvspan(shade_start, shade_end, color='grey', alpha=0.5)
    ax[1].set_title('Average Sentiment Per Day', fontsize=16)
    ax[1].set_ylabel('Average Sentiment', fontsize=14)
    ax[1].set_xlabel('Date', fontsize=14)

    # Rotate the date labels for better visibility
    plt.xticks(combined_data.index, combined_data.index.strftime('%Y-%m-%d'), rotation=90)

    # Use a tight layout to adjust spacing
    plt.tight_layout()

    # Show the plot
    plt.show()


# Streamlit app main function
def main():
    st.title('GDELT Data Analysis and Visualization')

    # Sidebar for user inputs
    st.sidebar.header('User Inputs')
    input_date_str = st.sidebar.text_input('Enter date (dd/mm/yyyy)', '06/06/2014')
    location_regex_input = st.sidebar.text_input('Location Regex (Optional)', '')
    theme_regex_input = st.sidebar.text_input('Theme Regex (Optional)', '')
    country_name_input = st.sidebar.text_input('Country Name for Visualization (Optional)', '')
    top_n_input = st.sidebar.number_input('Top N items to visualize', 5, 100, 30)

    # Button to download and filter data
    if st.sidebar.button('Download and Filter GDELT Data'):
        with st.spinner('Downloading and processing data...'):
            try:
                output_file_path = os.path.join(GDELT_DATA_DIR, 'gdelt_mta.json')
                download_and_filter_gdelt_data(output_file_path, input_date_str, location_regex_input, theme_regex_input)
                st.success('Data downloaded and filtered successfully!')
            except Exception as e:
                st.error(f'An error occurred: {e}')

    # Load data and visualize
    if st.sidebar.button('Load Data and Generate Visualizations'):
        with st.spinner('Loading data and generating visualizations...'):
            try:
                filtered_df = pd.read_json(os.path.join(GDELT_DATA_DIR, 'gdelt_mta.json'))
                # Generate and display visualizations
                generate_visualizations(filtered_df, country_name_input, top_n_input)
                # Plot from dataframe
                plot_from_dataframe(filtered_df, input_date_str.replace('/', ''))
                st.success('Visualizations generated successfully!')
            except Exception as e:
                st.error(f'An error occurred: {e}')

# Run the main function
if __name__ == "__main__":
    main()
