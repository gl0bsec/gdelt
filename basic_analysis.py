# Initialize and Set Parameters 
#%%
from networkx.algorithms import community as nx_community
from datetime import datetime, timedelta
import matplotlib.style as style
from collections import Counter
import matplotlib.pyplot as plt
from zipfile import ZipFile
import matplotlib as plt
import networkx as nx
import seaborn as sns
import pandas as pd
import numpy as np
import json

## What Country: 
country = 'Iran'
country_list = []
# Number of entities (for bar chat):
N = 33
csv_name = 'iran_data.csv'

# Load Data 
#%%
file = 'gkg_30_days.json'
# Read data from JSON file
with open(file, 'r') as f:
    data = json.load(f)

# Convert to DataFrame and filter for Iran-related entries
df = pd.DataFrame(data)

## country filtering?
iran_df = df[df['LOCATIONS'].str.contains('Kosovo', case=False, na=False)].reset_index(drop=True)
df = pd.DataFrame(data)

gdelt_df = pd.read_csv('gdelt_iran.csv')

# Output to CSV?
gdelt_df.to_csv(csv_name)

# Data Viz 
#%% 
## Assoc locations map

def filter_and_count_locations(df, country):
    """
    Filters the DataFrame for entries related to a given country and counts the occurrences of each location.
    """
    filtered_df = df[df['LOCATIONS'].str.contains(country, na=False, case=False)]
    location_counter = Counter()
    
    for locations_str in filtered_df['LOCATIONS']:
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
    
plot_minimalist_map(filter_and_count_locations(df, 'Iran'), 'Countries associated with Iran', color_map='Greens', power=0.5)

#%% 
# Themes, Orgs and Countries
d = 0
def generate_visualizations(df, country_name, top_n):
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
    
generate_visualizations(df, 'Iran', N-d)

# Viz countries
# Calculate the overall tone for each entry related to Iran
gdelt_df['OVERALL_TONE'] = gdelt_df['TONE'].apply(lambda x: float(x.split(',')[0]) if pd.notnull(x) else None)

# Calculate the overall tone for the entire dataset
df['OVERALL_TONE'] = df['TONE'].apply(lambda x: float(x.split(',')[0]) if pd.notnull(x) else None)

### Themes
all_themes = ';'.join(gdelt_df['THEMES'].dropna()).split(';')
theme_counter = Counter(all_themes)
theme_names_20 = [theme[0] for theme in theme_counter.most_common(N-1) if theme[0] != '']
theme_counts_20 = [theme[1] for theme in theme_counter.most_common(N-1) if theme[0] != '']
theme_tones = {}
for theme in theme_names_20:
    theme_df = gdelt_df[gdelt_df['THEMES'].str.contains(theme, case=False, na=False)]
    theme_tones[theme] = theme_df['OVERALL_TONE'].mean()
theme_colors_reversed = [sns.color_palette("coolwarm_r", as_cmap=True)(0.5 + tone / 10) for tone in [theme[1] for theme in sorted(theme_tones.items(), key=lambda x: x[1])]]

### Countries
country_tones = []
for _, row in gdelt_df.iterrows():
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

### Tone distribution
plt.figure(figsize=(15, 6))
sns.histplot(df['OVERALL_TONE'], bins=50, color='gray', kde=True, label='All Data', alpha=0.5)
ax2 = plt.twinx()
sns.histplot(gdelt_df['OVERALL_TONE'], bins=50, color='salmon', kde=True, label='Iran Data', ax=ax2)
ax2.legend(loc='upper left')
plt.legend(loc='upper right')
plt.show()

#%%
# Interactive networks
from collections import defaultdict
from itertools import combinations
from pyvis.network import Network

# Function to create a co-occurrence matrix for a given column (either 'PERSONS' or 'ORGANIZATIONS')
def create_co_occurrence_matrix(data, column_name):
    co_occurrence = defaultdict(int)
    for items in data[column_name].dropna():
        items_list = items.split(';')
        for i, j in combinations(items_list, 2):
            pair = tuple(sorted([i, j]))
            co_occurrence[pair] += 1
    return co_occurrence

# Function to visualize a network graph using pyvis from a co-occurrence DataFrame
def visualize_network_with_pyvis(df, col1, col2, title, scale_factor=1.0):
    net = Network(notebook=True)
    for _, row in df.iterrows():
        net.add_node(row[col1])
        net.add_node(row[col2])
        net.add_edge(row[col1], row[col2], value=row['Count'] * scale_factor)
    net.show(title + ".html")

# Load the dataset
data = pd.read_csv('path_to_your_dataset.csv')

# Filter rows based on the presence of the theme 'TAX_FNCACT'
tax_fncact_data = data[data['THEMES'].str.contains('TAX_FNCACT', na=False)]

# Create co-occurrence matrices for persons and organizations
persons_co_occurrence = create_co_occurrence_matrix(tax_fncact_data, 'PERSONS')
organizations_co_occurrence = create_co_occurrence_matrix(tax_fncact_data, 'ORGANIZATIONS')

# Convert dictionaries to DataFrames
persons_df = pd.DataFrame.from_dict(persons_co_occurrence, orient='index', columns=['Count']).reset_index()
persons_df.columns = ['Person1', 'Person2', 'Count']
organizations_df = pd.DataFrame.from_dict(organizations_co_occurrence, orient='index', columns=['Count']).reset_index()
organizations_df.columns = ['Organization1', 'Organization2', 'Count']

# Select the top N co-occurring pairs based on their counts for both persons and organizations
N = 50
top_persons_df = persons_df.nlargest(N, 'Count')
top_organizations_df = organizations_df.nlargest(N, 'Count')

# Visualize the network graphs using pyvis
visualize_network_with_pyvis(top_persons_df, 'Person1', 'Person2', 'Top N Co-occurrences Among Persons', scale_factor=4.0)
visualize_network_with_pyvis(top_organizations_df, 'Organization1', 'Organization2', 'Top N Co-occurrences Among Organizations', scale_factor=4.0)

#%%
# Networks
import matplotlib.pyplot as plt
import random

def generate_pastel_colors(n):
    pastel_colors = []
    for _ in range(n):
        red = random.uniform(0.6, 1) * 255
        green = random.uniform(0.6, 1) * 255
        blue = random.uniform(0.6, 1) * 255
        color = "#{:02x}{:02x}{:02x}".format(int(red), int(green), int(blue))
        pastel_colors.append(color)
    return pastel_colors

def get_theme_counts_by_country(df, country,var):
    """Return the theme counts for a specified country."""
    country_data = df[df['LOCATIONS'].str.contains(country, na=False, case=False)]
    themes = country_data[var].str.split(';').explode()
    theme_counts = themes.value_counts()
    return theme_counts

def generate_network_graph(df, top_themes, country,var):
    """Generate a network graph for co-occurring themes for a specified country."""
    country_data = df[df['LOCATIONS'].str.contains(country, na=False, case=False)]
    G = nx.Graph()
    for _, row in country_data.iterrows():
        themes_in_row = set(str(row[var]).split(';')) & set(top_themes)
        for theme1 in themes_in_row:
            for theme2 in themes_in_row:
                if theme1 != theme2:
                    if G.has_edge(theme1, theme2):
                        G[theme1][theme2]['weight'] += 1
                    else:
                        G.add_edge(theme1, theme2, weight=1)
    return G

def normalize_node_sizes_enhanced(theme_counts, min_size=100, max_size=2000):
    """Enhanced normalization of node sizes to accentuate differences."""
    min_count_val = min(theme_counts.values)
    max_count_val = max(theme_counts.values)
    normalized_sizes = {}
    for theme, count in theme_counts.items():
        normalized_size = min_size + (max_size - min_size) * ((count - min_count_val)**1.5 / (max_count_val - min_count_val)**1.5)
        normalized_sizes[theme] = normalized_size
    return normalized_sizes

def remove_blank_node(G):
    """Remove the blank node from the graph."""
    if '' in G:
        G.remove_node('')

def plot_normalized_sized_network_graph(G, theme_counts, country, N, d):
    """Plot the network graph with normalized node sizing based on counts."""
    plt.figure(figsize=(18, 12), dpi=150)
    pos = nx.spring_layout(G, k=0.8, iterations=50)
    normalized_sizes = normalize_node_sizes_enhanced(theme_counts.nlargest(N))
    node_sizes = [normalized_sizes[theme] for theme in G.nodes()]
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=generate_pastel_colors(N-d))
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')
    nx.draw_networkx_edges(G, pos, alpha=0.5, width=0.5)
    plt.title(f'Normalized Sized Network Graph of Top {N} {var} for {country}', fontsize=16)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

#%%
# Net1:
countries = ['Iran']
var = "THEMES"
for country in countries:
    N = 30
    theme_counts = get_theme_counts_by_country(df, country,"THEMES")
    top_themes = theme_counts.head(N).index
    G = generate_network_graph(df, top_themes, country,"THEMES")
    remove_blank_node(G)
    plot_normalized_sized_network_graph(G, theme_counts, country, N, 1)
    
    
#%%
def generate_network_graphs_for_theme(df, theme, top_n):
    # Filter the data to include only rows with the specified theme
    df_theme = df[df['THEMES'].apply(lambda x: theme in str(x).split(';'))]
    
    # Function to get top N entities based on frequency
    def get_top_n_entities(df, column_name, n):
        entity_counter = Counter()
        for entities in df[column_name].dropna().str.split(';'):
            entity_counter.update(entities)
        return [item[0] for item in entity_counter.most_common(n)], entity_counter

    # Get top N persons and organizations
    top_n_persons, person_counter = get_top_n_entities(df_theme, 'PERSONS', top_n)
    top_n_organizations, organization_counter = get_top_n_entities(df_theme, 'ORGANIZATIONS', top_n)

    # Function to create and draw network graph
    def draw_network_graph(entities, entity_counter, title, color):
        G = nx.Graph()
        for _, row in df_theme.iterrows():
            entity_list = str(row[entities]).split(';')
            for entity in entity_list:
                if entity in entity_counter:
                    G.add_node(entity, size=entity_counter[entity])
                    for other_entity in entity_list:
                        if other_entity in entity_counter and other_entity != entity:
                            if G.has_edge(entity, other_entity):
                                G[entity][other_entity]['weight'] += 1
                            else:
                                G.add_edge(entity, other_entity, weight=1)
        
        # Adjust node sizes and draw the graph
        min_size = min([G.nodes[node]['size'] for node in G.nodes])
        max_size = max([G.nodes[node]['size'] for node in G.nodes])
        sizes = [(G.nodes[node]['size'] - min_size + 1) * 3000 / (max_size - min_size + 1) + 100 for node in G.nodes]
        pos = nx.spring_layout(G, seed=42)
        edges = nx.draw_networkx_edges(G, pos, alpha=0.3)
        nodes = nx.draw_networkx_nodes(G, pos, node_color=color, node_size=sizes)
        labels = nx.draw_networkx_labels(G, pos, font_size=10, font_color='black')
        plt.title(title)
        plt.show()

    # Draw the network graphs
    draw_network_graph('PERSONS', person_counter, f"Top {top_n} Persons Associated with '{theme}' Theme", 'skyblue')
    draw_network_graph('ORGANIZATIONS', organization_counter, f"Top {top_n} Organizations Associated with '{theme}' Theme", 'lightgreen')

generate_network_graphs_for_theme(gdelt_df, 'LEADER', 10)

#%%
# Usage:
countries = ['Iran']
var = "ORGANIZATIONS"
for country in countries:
    N = 40
    theme_counts = get_theme_counts_by_country(df, country,var)
    top_themes = theme_counts.head(N).index
    G = generate_network_graph(df, top_themes, country,var)
    remove_blank_node(G)
    plot_normalized_sized_network_graph(G, theme_counts, country, N)

#%%
# Timeline analysis 
import matplotlib.pyplot as plt

def load_and_preprocess_data(file_path):
    """
    Load the dataset and preprocess it.
    """
    df = pd.read_csv(file_path)
    df = df.dropna(subset=['THEMES'])
    df['DATE'] = pd.to_datetime(df['DATE'], format='%Y%m%d')
    return df

def get_top_n_themes(df, n):
    """
    Get the top N themes based on frequency.
    """
    themes_counter = Counter()
    for themes in df['THEMES'].str.split(';'):
        themes_counter.update(themes)
    top_n_themes = [item[0] for item in themes_counter.most_common(n)]
    return top_n_themes

def generate_heatmaps(df, top_n_themes):
    """
    Generate heatmaps for frequency and sentiment of top N themes.
    """
    # Initialize matrices
    matrix_df = pd.DataFrame(columns=top_n_themes, index=pd.date_range(start=df['DATE'].min(), end=df['DATE'].max()))
    sentiment_matrix_df = pd.DataFrame(columns=top_n_themes, index=pd.date_range(start=df['DATE'].min(), end=df['DATE'].max()))
    
    for date, group in df.groupby('DATE'):
        themes_counter = Counter()
        themes_sentiment = {theme: [] for theme in top_n_themes}
        
        for idx, row in group.iterrows():
            themes = row['THEMES'].split(';')
            sentiment = row['OVERALL_TONE']
            
            themes_counter.update(themes)
            for theme in themes:
                if theme in top_n_themes:
                    themes_sentiment[theme].append(sentiment)
                    
        for theme, count in themes_counter.items():
            if theme in top_n_themes:
                matrix_df.at[date, theme] = count
        
        for theme, sentiments in themes_sentiment.items():
            if sentiments:
                sentiment_matrix_df.at[date, theme] = np.mean(sentiments)

    # Replace NaN with 0
    matrix_df.fillna(0, inplace=True)
    sentiment_matrix_df.fillna(0, inplace=True)
    
    # Generate heatmaps
    fig, ax = plt.subplots(1, 2, figsize=(30, 12))
    
    date_labels = [date.strftime('%Y-%m-%d') for date in matrix_df.index]
    
    sns.heatmap(matrix_df.astype(float).T, cmap="YlGnBu", linewidths=.5, xticklabels=date_labels, ax=ax[0])
    ax[0].set_title("Frequency of Top N Themes")
    ax[0].set_xlabel("Date")
    ax[0].set_ylabel("Themes")
    
    sns.heatmap(sentiment_matrix_df.astype(float).T, cmap="coolwarm", linewidths=.5, xticklabels=date_labels, ax=ax[1])
    ax[1].set_title("Average Sentiment of Top N Themes")
    ax[1].set_xlabel("Date")
    ax[1].set_ylabel("Themes")
    
    for axes in ax:
        plt.setp(axes.get_xticklabels(), rotation=45)
        
    plt.tight_layout()
    plt.show()

def main(file_path, n):
    """
    Main function to run the analysis.
    """
    df = load_and_preprocess_data(file_path)
    top_n_themes = get_top_n_themes(df, n)
    generate_heatmaps(df, top_n_themes)

# Example usage
main('gdelt_iran.csv', 30)  # Replace '/path/to/your/dataset.csv' with the actual path and N with the number of top themes you want.
