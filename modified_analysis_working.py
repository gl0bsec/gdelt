
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('ggplot')
standard_palette = sns.color_palette('muted')
from itertools import combinations
import ipysigma
import ipysigma as Sigma
from networkx.algorithms import community as nx_community
from datetime import datetime, timedelta
import matplotlib.style as style
from collections import Counter
import matplotlib.pyplot as plt
from zipfile import ZipFile  
import matplotlib.cm as cm
import seaborn as sns
from tqdm import tqdm
import networkx as nx
import pandas as pd
import numpy as np

# Themes, Orgs and Countries

def generate_countrycounts(df, country_name, top_n):    
    # Countries
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
    df_country_avg_tone = df_country_tones.groupby('Country')['Tone'].mean().reset_index().sort_values(by='Tone',ascending=True)
    df_country_count_top20 = df_country_tones['Country'].value_counts().reset_index()
    df_country_count_top20.columns = ['Country', 'Count']
    df_country_count_top20 = df_country_count_top20.nlargest(top_n-1, 'Count')
    df_country_count_top20 = df_country_count_top20.merge(df_country_avg_tone, on='Country', how='left').sort_values(by='Count')
    country_colors_reversed = [sns.color_palette("coolwarm_r", as_cmap=True)(0.5 + tone / 10) for tone in df_country_count_top20['Tone']]

    plt.figure(figsize=(10, 6))
    barplot_countries_reversed = sns.barplot(x='Count', y='Country', data=df_country_count_top20, palette=country_colors_reversed)
    for index, p in enumerate(barplot_countries_reversed.patches):
        avg_tone = df_country_count_top20.iloc[index]['Tone']
        barplot_countries_reversed.annotate(f'{avg_tone:.2f}', (p.get_width() / 2, p.get_y() + p.get_height() / 2), ha='center', va='center', color='white')
    plt.show()
    
    return


def filter_themes(df, excluded_themes):
    if not excluded_themes:
        return df
    for theme in excluded_themes:
        df = df[~df['THEMES'].str.contains(theme, case=False, na=False)]
    return df

def generate_visualizations2(df, country_name, top_n, excluded_themes=None):
    if country_name is None: 
        country_df = df 
    else: 
        country_df = df[df['LOCATIONS'].str.contains(country_name, case=False, na=False)].reset_index(drop=True)

    # Apply theme filter
    country_df = filter_themes(country_df, excluded_themes)

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

    # Side-by-side plots
    fig, axes = plt.subplots(1, 2, figsize=(30, 8))

    # Themes plot
    barplot_themes_reversed = sns.barplot(x=theme_counts_n, y=theme_names_n, palette=theme_colors_reversed, ax=axes[0])
    if country_name == None: 
        axes[0].set_title(f'Top {top_n} Themes')
    else:     
        axes[0].set_title(f'Top {top_n} Themes Related to {country_name}')
    # Add annotations for themes plot
    for index, p in enumerate(barplot_themes_reversed.patches):
        avg_tone = theme_tones[theme_names_n[index]]
        barplot_themes_reversed.annotate(f'{avg_tone:.2f}', (p.get_width(), p.get_y() + p.get_height() / 2), ha='right', va='center', color='black')

    # Organizations plot
    barplot_orgs_reversed = sns.barplot(x=org_counts_n, y=org_names_n, palette=org_colors_reversed, ax=axes[1])
    if country_name == None: 
        axes[1].set_title(f'Top {top_n} Organisations')
    else: 
        axes[1].set_title(f'Top {top_n} Organizations Related to {country_name}')
    # Add annotations for organizations plot
    for index, p in enumerate(barplot_orgs_reversed.patches):
        avg_tone = org_tones[org_names_n[index]]
        barplot_orgs_reversed.annotate(f'{avg_tone:.2f}', (p.get_width(), p.get_y() + p.get_height() / 2), ha='right', va='center', color='black')

    plt.tight_layout()
    plt.show()

    return


def plot_side_by_side_organization_tones(positive_df, negative_df):
    """
    Plot bar charts for positive and negative tone organizations side by side.
    """
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    # Plot for positive tones
    tone_min_pos = positive_df['Avg_Tone'].min()
    tone_max_pos = positive_df['Avg_Tone'].max()
    normalized_tones_pos = (positive_df['Avg_Tone'] - tone_min_pos) / (tone_max_pos - tone_min_pos)
    bars_pos = axes[0].barh(positive_df['Organization'][::-1], positive_df['Count'][::-1], color='green')
    for bar, tone in zip(bars_pos, normalized_tones_pos[::-1]):
        bar.set_alpha(tone)
    axes[0].set_title('Top Organizations with Positive Tone by Count')
    axes[0].set_xlabel('Count')
    axes[0].set_ylabel('Organization Name')

    # Plot for negative tones
    tone_min_neg = negative_df['Avg_Tone'].min()
    tone_max_neg = negative_df['Avg_Tone'].max()
    normalized_tones_neg = (negative_df['Avg_Tone'] - tone_min_neg) / (tone_max_neg - tone_min_neg)
    bars_neg = axes[1].barh(negative_df['Organization'][::-1], negative_df['Count'][::-1], color='red')
    for bar, tone in zip(bars_neg, normalized_tones_neg[::-1]):
        bar.set_alpha(tone)
    axes[1].set_title('Top Organizations with Negative Tone by Count')
    axes[1].set_xlabel('Count')

    plt.tight_layout()
    plt.show()

# Usage:
# plot_side_by_side_organization_tones(positive_organizations_df, negative_organizations_df)


def plot_filtered_side_by_side_shaded_bars_reversed_color(df1, df2, filter_themes=None, title1='Organizations', title2='Themes'):
    """
    Plot side-by-side bar charts for two dataframes, shaded by average tone.
    The color coding is reversed to be consistent with the numerical value of the average tone,
    using a color map that transitions from blue (positive) to red (negative).
    """
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))

    # Function to map tone to color, reversed
    def tone_to_color(tone, tone_min, tone_max):
        # Normalize tone
        normalized_tone = (tone - tone_min) / (tone_max - tone_min)
        # Map tone to color: blue (positive) to red (negative), reversed from RdBu
        return cm.RdBu_r(normalized_tone)

    # Find global min and max tones for scaling
    global_tone_min = min(df1['Avg_Tone'].min(), df2['Avg_Tone'].min())
    global_tone_max = max(df1['Avg_Tone'].max(), df2['Avg_Tone'].max())

    # Filtering themes if specified
    if filter_themes:
        df2 = df2[~df2['Name'].isin(filter_themes)]

    # Organization plot
    colors_org = df1['Avg_Tone'].apply(lambda x: tone_to_color(x, global_tone_min, global_tone_max))
    bars_org = axes[0].barh(df1['Name'][::-1], df1['Count'][::-1], color=colors_org)
    axes[0].set_title(title1)
    axes[0].set_xlabel('Count')

    # Adding average tone values as text on the bars for organizations
    for bar, tone in zip(bars_org, df1['Avg_Tone'][::-1]):
        axes[0].text(bar.get_width(), bar.get_y() + bar.get_height() / 2, f'{tone:.2f}', va='center')

    # Theme plot
    colors_theme = df2['Avg_Tone'].apply(lambda x: tone_to_color(x, global_tone_min, global_tone_max))
    bars_theme = axes[1].barh(df2['Name'][::-1], df2['Count'][::-1], color=colors_theme)
    axes[1].set_title(title2)
    axes[1].set_xlabel('Count')

    # Adding average tone values as text on the bars for themes
    for bar, tone in zip(bars_theme, df2['Avg_Tone'][::-1]):
        axes[1].text(bar.get_width(), bar.get_y() + bar.get_height() / 2, f'{tone:.2f}', va='center')

    plt.tight_layout()
    plt.show()
 
def plot_side_by_side_organization_tones(positive_df, negative_df):
    """
    Plot bar charts for positive and negative tone organizations side by side.
    """
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    # Plot for positive tones
    tone_min_pos = positive_df['Avg_Tone'].min()
    tone_max_pos = positive_df['Avg_Tone'].max()
    normalized_tones_pos = (positive_df['Avg_Tone'] - tone_min_pos) / (tone_max_pos - tone_min_pos)
    bars_pos = axes[0].barh(positive_df['Organization'][::-1], positive_df['Count'][::-1], color='green')
    for bar, tone in zip(bars_pos, normalized_tones_pos[::-1]):
        bar.set_alpha(tone)
    axes[0].set_title('Top Organizations with Positive Tone by Count')
    axes[0].set_xlabel('Count')
    axes[0].set_ylabel('Organization Name')

    # Plot for negative tones
    tone_min_neg = negative_df['Avg_Tone'].min()
    tone_max_neg = negative_df['Avg_Tone'].max()
    normalized_tones_neg = (negative_df['Avg_Tone'] - tone_min_neg) / (tone_max_neg - tone_min_neg)
    bars_neg = axes[1].barh(negative_df['Organization'][::-1], negative_df['Count'][::-1], color='red')
    for bar, tone in zip(bars_neg, normalized_tones_neg[::-1]):
        bar.set_alpha(tone)
    axes[1].set_title('Top Organizations with Negative Tone by Count')
    axes[1].set_xlabel('Count')

    plt.tight_layout()
    plt.show()

# Usage:
# plot_side_by_side_organization_tones(positive_organizations_df, negative_organizations_df)

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

def create_orgs_and_themes_dataframes_ordered_by_count(df, top_n=30):
    """
    Creates two separate DataFrames for organizations and themes, ordered by count and shaded by the average tone.
    """
    org_tone_sum = {}
    org_counter = Counter()
    theme_tone_sum = {}
    theme_counter = Counter()

    # Processing each row for tone and count for both organizations and themes
    for _, row in df.iterrows():
        if pd.notna(row['TONE']):
            tone_values = row['TONE'].split(',')

            try:
                tone = float(tone_values[0])  # Extracting the first tone value
            except ValueError:
                continue  # Skip if tone value is not convertible to float

            # Processing organizations
            if pd.notna(row['ORGANIZATIONS']):
                organizations_list = row['ORGANIZATIONS'].split(';')
                for organization in organizations_list:
                    org_counter[organization] += 1
                    if organization in org_tone_sum:
                        org_tone_sum[organization] += tone
                    else:
                        org_tone_sum[organization] = tone

            # Processing themes
            if pd.notna(row['THEMES']):
                themes_list = row['THEMES'].split(';')
                for theme in themes_list:
                    theme_counter[theme] += 1
                    if theme in theme_tone_sum:
                        theme_tone_sum[theme] += tone
                    else:
                        theme_tone_sum[theme] = tone

    # Creating DataFrames
    org_df = pd.DataFrame(
        [(org, org_tone_sum[org] / org_counter[org], org_counter[org]) for org in org_counter],
        columns=['Name', 'Avg_Tone', 'Count']
    ).sort_values(by='Count', ascending=False).head(top_n)

    theme_df = pd.DataFrame(
        [(theme, theme_tone_sum[theme] / theme_counter[theme], theme_counter[theme]) for theme in theme_counter],
        columns=['Name', 'Avg_Tone', 'Count']
    ).sort_values(by='Count', ascending=False).head(top_n)

    return org_df, theme_df


# Re-applying the revised approach for organizations
def create_organization_tone_dataframes_ordered_by_count(df, top_n=30):
    organization_tone_sum = {}
    organization_counter = Counter()

    # Processing each row for tone and count, focusing on organizations
    for _, row in df.iterrows():
        if pd.notna(row['ORGANIZATIONS']) and pd.notna(row['TONE']):
            organizations_list = row['ORGANIZATIONS'].split(';')
            tone_values = row['TONE'].split(',')

            try:
                tone = float(tone_values[0])  # Extracting the first tone value
            except ValueError:
                continue  # Skip if tone value is not convertible to float

            for organization in organizations_list:
                organization_counter[organization] += 1

                if organization in organization_tone_sum:
                    organization_tone_sum[organization] += tone
                else:
                    organization_tone_sum[organization] = tone

    # Calculating average tone for each organization
    organization_avg_tone = {org: organization_tone_sum[org] / organization_counter[org] for org in organization_counter}

    # Separating positive and negative tones, and ordering by count
    positive_organizations = [(org, organization_avg_tone[org], count) for org, count in organization_counter.items() if organization_avg_tone[org] > 0]
    negative_organizations = [(org, organization_avg_tone[org], count) for org, count in organization_counter.items() if organization_avg_tone[org] < 0]

    # Sorting by count and selecting top 'n'
    top_positive_organizations = sorted(positive_organizations, key=lambda x: x[2], reverse=True)[:top_n]
    top_negative_organizations = sorted(negative_organizations, key=lambda x: x[2], reverse=True)[:top_n]

    # Creating DataFrames
    positive_df = pd.DataFrame(top_positive_organizations, columns=['Organization', 'Avg_Tone', 'Count'])
    negative_df = pd.DataFrame(top_negative_organizations, columns=['Organization', 'Avg_Tone', 'Count'])

    return positive_df, negative_df



# Plotting function
def plot_organization_tone_locations_shaded_by_tone(df, tone_type='positive'):
    tone_min = df['Avg_Tone'].min()
    tone_max = df['Avg_Tone'].max()
    normalized_tones = (df['Avg_Tone'] - tone_min) / (tone_max - tone_min)

    color = 'green' if tone_type == 'positive' else 'red'

    plt.figure(figsize=(10, 8))
    bars = plt.barh(df['Organization'][::-1], df['Count'][::-1], color=color)

    # Adjusting bar colors based on normalized tone values
    for bar, tone in zip(bars, normalized_tones[::-1]):
        bar.set_alpha(tone)

    plt.xlabel('Count')
    plt.ylabel('Organization Name')
    plt.title(f'Top Organizations with Most {tone_type.capitalize()} Tone by Count')
    plt.show()








# Loading the newly provided dataset

def create_source_tone_dataframes_ordered_by_count(df):
    """
    Creates a DataFrame for sources, ordered by count and shaded by the average tone.
    """
    source_tone_sum = {}
    source_counter = Counter()

    # Processing each row for tone and count, focusing on sources
    for _, row in df.iterrows():
        if pd.notna(row['SOURCES']) and pd.notna(row['TONE']):
            sources_list = row['SOURCES'].split(';')
            tone_values = row['TONE'].split(',')

            try:
                tone = float(tone_values[0])  # Extracting the first tone value
            except ValueError:
                continue  # Skip if tone value is not convertible to float

            for source in sources_list:
                source_counter[source] += 1

                if source in source_tone_sum:
                    source_tone_sum[source] += tone
                else:
                    source_tone_sum[source] = tone
    # Calculating average tone for each source
    source_avg_tone = {src: source_tone_sum[src] / source_counter[src] for src in source_counter}

    # Combining counts and average tones
    combined_sources = [(src, source_avg_tone[src], count) for src, count in source_counter.items()]
    combined_sources_df = pd.DataFrame(combined_sources, columns=['Source', 'Avg_Tone', 'Count'])

    return combined_sources_df

def plot_source_counts_shaded_by_tone(df, top_n):
    """
    Plot a bar chart of the top sources ordered by count, shaded by the average tone.
    """
    sorted_df = df.sort_values(by='Count', ascending=False).head(top_n)

    tone_min = sorted_df['Avg_Tone'].min()
    tone_max = sorted_df['Avg_Tone'].max()
    normalized_tones = (sorted_df['Avg_Tone'] - tone_min) / (tone_max - tone_min)

    plt.figure(figsize=(10, 8))
    bars = plt.barh(sorted_df['Source'][::-1], sorted_df['Count'][::-1], color='blue')

    # Adjusting bar colors based on normalized tone values
    for bar, tone in zip(bars, normalized_tones[::-1]):
        bar.set_alpha(tone)

    plt.xlabel('Count')
    plt.ylabel('Source')
    plt.title('Top Sources by Count (Shaded by Average Tone)')
    plt.show()

def create_org_network(gdelt_df, centrality_type='degree', country_codes=None):
    """
    Creates a network graph of co-occurring organizations in the GDELT data, 
    represented as a pandas DataFrame, filtered by specified country codes if provided.
    Nodes are sized based on the specified centrality measure and colored in a muted dark yellow.

    Parameters:
    gdelt_df (DataFrame): Pandas DataFrame containing the GDELT data.
    centrality_type (str): Type of centrality measure ('degree', 'betweenness', 'closeness', 'eigenvector').
    country_codes (list): List of country codes to filter the data. If None, no filtering is applied.
    """
    # Create a NetworkX graph
    G = nx.Graph()

    # Filter DataFrame by country codes if provided
    if country_codes:
        mask = gdelt_df['LOCATIONS'].apply(lambda x: any(country_code in x for country_code in country_codes))
        filtered_df = gdelt_df[mask]
    else:
        filtered_df = gdelt_df

    # Iterate over the DataFrame rows
    for _, row in filtered_df.iterrows():
        organizations = row['ORGANIZATIONS']
        if pd.notna(organizations):
            orgs = organizations.split(';')
            for org1, org2 in combinations(orgs, 2):
                if G.has_edge(org1, org2):
                    G[org1][org2]['weight'] += 1
                else:
                    G.add_edge(org1, org2, weight=1)

    # Calculate specified centrality
    if centrality_type == 'betweenness':
        centrality = nx.betweenness_centrality(G)
    elif centrality_type == 'closeness':
        centrality = nx.closeness_centrality(G)
    elif centrality_type == 'eigenvector':
        centrality = nx.eigenvector_centrality(G, max_iter=1000)
    else:  # default to degree centrality
        centrality = nx.degree_centrality(G)

    # Normalize centrality values for sizing nodes
    max_centrality = max(centrality.values())
    sizes = [1000 * centrality[node] / max_centrality for node in G.nodes()]  # Adjust node size scale as needed

    # Node color - muted dark yellow
    node_color = "#DAA520"

    # Visualize the graph using ipysigma with sized nodes
    sigma = ipysigma.Sigma(G, start_layout=True, node_color=node_color, node_size=sizes)
    return sigma



def visualize_top_fncact_tones(gdelt_data, N):
    """
    Visualizes the top N functional activities in the GDELT data, color-coded by the sum of the first value of the TONE field.
    
    Parameters:
    gdelt_data (DataFrame): Pandas DataFrame, each row representing a GDELT entry.
    N (int): Number of top functional activities to display.
    """
    # Initializing counters for functional activities and tone sums
    fncact_counts = Counter()
    tone_sums = {}

    for _, row in gdelt_data.iterrows():
        themes = row['THEMES']
        tone = row['TONE']
        
        if pd.notna(themes) and pd.notna(tone):
            try:
                first_tone_value = float(tone.split(',')[0])
            except ValueError:
                continue

            for theme in themes.split(';'):
                if theme.startswith("TAX_FNCACT") and theme != "TAX_FNCACT":
                    fncact_counts[theme] += 1
                    tone_sums[theme] = tone_sums.get(theme, 0) + first_tone_value

    # Sorting and selecting top N activities
    sorted_activities = [activity for activity, _ in fncact_counts.most_common(N)]
    counts = [fncact_counts[activity] for activity in sorted_activities]
    tones = [tone_sums.get(activity, 0) for activity in sorted_activities]

    # Normalizing tone values for color mapping
    min_tone, max_tone = min(tones), max(tones)
    normalized_tones = [(tone - min_tone) / (max_tone - min_tone) if max_tone - min_tone else 0.5 for tone in tones]

    # Creating the color-coded horizontal bar chart
    plt.figure(figsize=(10, 6))
    plt.barh(sorted_activities, counts, color=[plt.cm.RdYlBu(tone) for tone in normalized_tones])
    plt.xlabel('Frequency')
    plt.ylabel('Functional Activities')
    plt.title(f'Top {N} Functional Activities with Tone-Color Coding in GDELT Data')
    plt.gca().invert_yaxis()  # Highest count at the top
    plt.show()

def visualize_top_fncact_tones_by_top_countries(gdelt_data, N, top_n_countries):
    """
    Produces a grid of visualize_top_fncact_tones plots for the top N countries based on the number of entries in the GDELT data.

    Parameters:
    gdelt_data (DataFrame): Pandas DataFrame, each row representing a GDELT entry.
    N (int): Number of top functional activities to display for each country.
    top_n_countries (int): Number of top countries to display based on the number of entries.
    """
    # Extracting country codes
    gdelt_data['COUNTRY_CODES'] = gdelt_data['LOCATIONS'].str.extractall(r'1#.*?#([A-Z]{2})#').groupby(level=0).first()[0]

    # Filter out rows without a country code
    gdelt_data = gdelt_data.dropna(subset=['COUNTRY_CODES'])

    # Counting entries per country and selecting the top N countries
    country_counts = gdelt_data['COUNTRY_CODES'].value_counts()
    top_countries = country_counts.nlargest(top_n_countries).index

    # Setting up the plot grid
    num_countries = len(top_countries)
    num_cols = 2  # Adjust the number of columns as needed
    num_rows = int(np.ceil(num_countries / num_cols))

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, num_rows * 5))
    axes = axes.flatten()

    for i, country_code in enumerate(top_countries):
        country_data = gdelt_data[gdelt_data['COUNTRY_CODES'] == country_code]

        # Following the same logic as in the original visualize_top_fncact_tones function
        fncact_counts = Counter()
        tone_sums = {}

        for _, row in country_data.iterrows():
            themes = row['THEMES']
            tone = row['TONE']
            
            if pd.notna(themes) and pd.notna(tone):
                try:
                    first_tone_value = float(tone.split(',')[0])
                except ValueError:
                    continue

                for theme in themes.split(';'):
                    if theme.startswith("TAX_FNCACT") and theme != "TAX_FNCACT":
                        fncact_counts[theme] += 1
                        tone_sums[theme] = tone_sums.get(theme, 0) + first_tone_value

        # Sorting and selecting top N activities
        sorted_activities = [activity for activity, _ in fncact_counts.most_common(N)]
        counts = [fncact_counts[activity] for activity in sorted_activities]
        tones = [tone_sums.get(activity, 0) for activity in sorted_activities]

        # Normalizing tone values for color mapping
        min_tone, max_tone = min(tones), max(tones)
        normalized_tones = [(tone - min_tone) / (max_tone - min_tone) if max_tone - min_tone else 0.5 for tone in tones]

        # Creating the color-coded horizontal bar chart for each country
        sns.barplot(x=counts, y=sorted_activities, ax=axes[i], palette=plt.cm.RdYlBu(normalized_tones))
        axes[i].set_title(f'Top {N} Functional Activities in {country_code}')
        axes[i].set_xlabel('Frequency')
        axes[i].set_ylabel('Functional Activities')

    # Adjust layout
    plt.tight_layout()
    plt.show()

# Usage example:
# visualize_top_fncact_tones_by_top_countries(gdelt_data, N=5, top_n_countries=10)

def visualize_fncact_network(gdelt_data, country_code):
    """
    Visualizes a network of functional actors for a given country code using ipysigma.

    Parameters:
    gdelt_data (DataFrame): Pandas DataFrame, each row representing a GDELT entry.
    country_code (str): The country code to filter the data.
    """

    if country_code != None: 
        # Extracting country codes
        gdelt_data['COUNTRY_CODES'] = gdelt_data['LOCATIONS'].str.extractall(r'1#.*?#([A-Z]{2})#').groupby(level=0).first()[0]

        # Filter for the specified country code
        country_data = gdelt_data[gdelt_data['COUNTRY_CODES'] == country_code]
    else:
        country_data = gdelt_data

    # Creating a network graph
    G = nx.Graph()

    for _, row in country_data.iterrows():
        themes = row['THEMES']
        if pd.notna(themes):
            # Extracting functional actors
            fncacts = [theme for theme in themes.split(';') if theme.startswith("TAX_FNCACT") and theme != "TAX_FNCACT"]
            
            # Adding nodes and edges to the graph
            for fncact in fncacts:
                G.add_node(fncact)
                for other_fncact in fncacts:
                    if fncact != other_fncact:
                        G.add_edge(fncact, other_fncact)

    # Visualizing the network graph
    sigma = Sigma(G)
    return sigma