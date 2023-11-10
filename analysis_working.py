#%%
from networkx.algorithms import community as nx_community
from datetime import datetime, timedelta
import matplotlib.style as style
from collections import Counter
import matplotlib.pyplot as plt
from zipfile import ZipFile  
import seaborn as sns
from tqdm import tqdm
import urllib.request
import networkx as nx
import pandas as pd
import numpy as np
import json
import os

filtered_df = pd.read_json('gdelt_mta.json')
df = filtered_df
# Themes, Orgs and Countries

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

# Call the function with the DataFrame and a specified date to test it

plot_from_dataframe(filtered_df, '20140606')
generate_visualizations(filtered_df, None, 30)

#%%



