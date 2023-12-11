
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('ggplot')
standard_palette = sns.color_palette('muted')

import pandas as pd 
import matplotlib.pyplot as plt
from collections import Counter
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt

def filter_and_count_locations(df, country):
    filtered_df = df[df['LOCATIONS'].str.contains(f'#{country}#', na=False, case=False)]
    location_counter = Counter()
    
    for locations_str in filtered_df['LOCATIONS']:
        if locations_str:
            locations_list = locations_str.split(';')
            for location in locations_list:
                location_details = location.split('#')
                if len(location_details) > 2:
                    location_name = location_details[1]
                    location_counter[location_name] += 1
            
    return location_counter

def plot_country_tones(data, location_column='LOCATIONS', tone_column='TONE'):
    def extract_first_tone(tone_str):
        try:
            return float(tone_str.split(',')[0])
        except:
            return 0

    def extract_country_names(location_str):
        if not location_str:
            return []
        locations = location_str.split(';')
        country_names = set()
        for loc in locations:
            parts = loc.split('#')
            if len(parts) > 1:
                country_names.add(parts[1])  # The second part is the country name
        return list(country_names)

    # Extract country names and first tone value
    data['CountryNames'] = data[location_column].apply(extract_country_names)
    data['FirstTone'] = data[tone_column].apply(extract_first_tone)
    # Aggregate the average tone for each country
    exploded_data = data.explode('CountryNames')
    country_tone_avg = exploded_data.groupby('CountryNames')['FirstTone'].mean()
    # Load the world map
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    # Merge with world map data
    merged = world.set_index('name').join(country_tone_avg)
    merged = merged.fillna(0)  # Replace NaN values with 0
    # Plotting
    fig, ax = plt.subplots(1, 1, figsize=(15, 10))
    merged.plot(column='FirstTone', ax=ax, cmap='coolwarm', edgecolor='black')
    ax.set_title('Average Tone of Countries in the Query')
    ax.set_axis_off()  # Remove axis for minimalist style
    return

def plot_country_frequencies(data, location_column='LOCATIONS', contrast_scale=1.0):
    def extract_country_names(location_str):
        if not location_str:
            return []
        locations = location_str.split(';')
        country_names = set()
        for loc in locations:
            parts = loc.split('#')
            if len(parts) > 1:
                country_names.add(parts[1])  # The second part is the country name
        return list(country_names)

    # Extract country names
    data['CountryNames'] = data[location_column].apply(extract_country_names)
    # Explode the DataFrame on the 'CountryNames' column to have one country per row
    exploded_data = data.explode('CountryNames')
    # Count the frequency of each country name
    country_name_frequency = exploded_data['CountryNames'].value_counts()
    # Adjusting the frequency data for contrast
    adjusted_frequency = country_name_frequency ** contrast_scale
    # Load the world map
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    # Merge with world map data
    merged = world.set_index('name').join(adjusted_frequency.rename('Frequency'))
    merged = merged.fillna(0)  # Replace NaN values with 0
    # Plotting
    fig, ax = plt.subplots(1, 1, figsize=(15, 10))
    merged.plot(column='Frequency', ax=ax,
                cmap='Oranges', edgecolor='black')

    ax.set_title('Frequency of Countries in the Dataset')
    ax.set_axis_off()  # Remove axis for minimalist style
    return

def plot_top_subnational_locations(df, top_n=30):
    # Extracting all sub-national locations
    all_locations = []
    for locations_str in df['LOCATIONS']:
        if locations_str:
            locations_list = locations_str.split(';')
            for location in locations_list:
                # Splitting by '#' and checking if it's a sub-national location (type 4)
                location_details = location.split('#')
                if len(location_details) > 1 and location_details[0] == '4':
                    location_name = location_details[1]
                    all_locations.append(location_name)

    # Counting occurrences
    location_counter = Counter(all_locations)

    # Selecting top 'n' locations
    top_locations = location_counter.most_common(top_n)

    # Extracting names and counts for plotting
    names = [loc[0] for loc in top_locations]
    counts = [loc[1] for loc in top_locations]

    # Plotting
    plt.figure(figsize=(10, 6))10, 8)
    plt.barh(names[::-1], counts[::-1])  # Reversing to have the highest count on top
    plt.xlabel('Count')
    plt.ylabel('Sub-national Location Name')
    plt.title(f'Top {top_n} Sub-national Locations by Occurrence')
    plt.show()

def create_country_tone_dataframes_ordered_by_count(df, top_n=30):
    country_tone_sum = {}
    country_counter = Counter()

    # Processing each row for tone and count, focusing only on country mentions (type 1)
    for _, row in df.iterrows():
        if row['LOCATIONS'] and pd.notna(row['TONE']):
            locations_list = row['LOCATIONS'].split(';')
            tone_values = row['TONE'].split(',')

            try:
                tone = float(tone_values[0])  # Extracting the first tone value
            except ValueError:
                continue  # Skip if tone value is not convertible to float

            for location in locations_list:
                location_details = location.split('#')
                if len(location_details) > 2 and location_details[0] == '1':  # Check for country type (1)
                    country_name = location_details[1]
                    country_counter[country_name] += 1

                    if country_name in country_tone_sum:
                        country_tone_sum[country_name] += tone
                    else:
                        country_tone_sum[country_name] = tone

    # Calculating average tone for each country
    country_avg_tone = {loc: country_tone_sum[loc] / country_counter[loc] for loc in country_counter}

    # Separating positive and negative tones, and ordering by count
    positive_countries = [(loc, country_avg_tone[loc], count) for loc, count in country_counter.items() if country_avg_tone[loc] > 0]
    negative_countries = [(loc, country_avg_tone[loc], count) for loc, count in country_counter.items() if country_avg_tone[loc] < 0]

    # Sorting by count and selecting top 'n'
    top_positive_countries = sorted(positive_countries, key=lambda x: x[2], reverse=True)[:top_n]
    top_negative_countries = sorted(negative_countries, key=lambda x: x[2], reverse=True)[:top_n]

    # Creating DataFrames
    positive_df = pd.DataFrame(top_positive_countries, columns=['Country', 'Avg_Tone', 'Count'])
    negative_df = pd.DataFrame(top_negative_countries, columns=['Country', 'Avg_Tone', 'Count'])

    return positive_df, negative_df


def create_tone_dataframes_ordered_by_count(df, top_n=30):
    location_tone_sum = {}
    location_counter = Counter()

    for _, row in df.iterrows():
        if row['LOCATIONS'] and pd.notna(row['TONE']):
            locations_list = row['LOCATIONS'].split(';')
            tone_values = row['TONE'].split(',')

            try:
                tone = float(tone_values[0])  # Extracting the first tone value
            except ValueError:
                continue  # Skip if tone value is not convertible to float

            for location in locations_list:
                location_details = location.split('#')
                if len(location_details) > 2:
                    location_name = location_details[1]
                    location_counter[location_name] += 1

                    if location_name in location_tone_sum:
                        location_tone_sum[location_name] += tone
                    else:
                        location_tone_sum[location_name] = tone

    location_avg_tone = {loc: location_tone_sum[loc] / location_counter[loc] for loc in location_counter}

    positive_locations = [(loc, location_avg_tone[loc], count) for loc, count in location_counter.items() if location_avg_tone[loc] > 0]
    negative_locations = [(loc, location_avg_tone[loc], count) for loc, count in location_counter.items() if location_avg_tone[loc] < 0]

    top_positive = sorted(positive_locations, key=lambda x: x[2], reverse=True)[:top_n]
    top_negative = sorted(negative_locations, key=lambda x: x[2], reverse=True)[:top_n]

    positive_df = pd.DataFrame(top_positive, columns=['Location', 'Avg_Tone', 'Count'])
    negative_df = pd.DataFrame(top_negative, columns=['Location', 'Avg_Tone', 'Count'])

    return positive_df, negative_df

def plot_tone_locations_shaded_by_tone(df, tone_type='positive'):
    tone_min = df['Avg_Tone'].min()
    tone_max = df['Avg_Tone'].max()
    normalized_tones = (df['Avg_Tone'] - tone_min) / (tone_max - tone_min)

    color = 'green' if tone_type == 'positive' else 'red'

    plt.figure(figsize=(10, 6))10, 8))
    bars = plt.barh(df['Location'][::-1], df['Count'][::-1], color=color)

    for bar, tone in zip(bars, normalized_tones[::-1]):
        bar.set_alpha(tone)

    plt.xlabel('Count')
    plt.ylabel('Location Name')
    plt.title(f'Top Locations with Most {tone_type.capitalize()} Tone by Count')
    plt.show()



def plot_side_by_side_tone_locations(negative_df, positive_df):

    def plot_tone(df, ax, tone_type):
        tone_min = df['Avg_Tone'].min()
        tone_max = df['Avg_Tone'].max()
        normalized_tones = (df['Avg_Tone'] - tone_min) / (tone_max - tone_min)

        color = 'green' if tone_type == 'positive' else 'red'
        bars = ax.barh(df['Location'][::-1], df['Count'][::-1], color=color)

        for bar, tone in zip(bars, normalized_tones[::-1]):
            bar.set_alpha(tone)

        ax.set_xlabel('Count')
        ax.set_ylabel('Location Name')
        ax.set_title(f'Top Locations with Most {tone_type.capitalize()} Tone by Count')

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    # Plot for negative tones
    plot_tone(negative_df, axes[0], 'negative')

    # Plot for positive tones
    plot_tone(positive_df, axes[1], 'positive')

    plt.tight_layout()
    plt.show()


# Plotting the bar chart for the top 30 sub-national locations

def plot_country_tone_locations_shaded_by_tone(df, tone_type):
    tone_min = df['Avg_Tone'].min()
    tone_max = df['Avg_Tone'].max()
    normalized_tones = (df['Avg_Tone'] - tone_min) / (tone_max - tone_min)

    color = 'green' if tone_type == 'positive' else 'red'

    plt.figure(figsize=(10, 6))10, 8))
    bars = plt.barh(df['Country'][::-1], df['Count'][::-1], color=color)

    # Adjusting bar colors based on normalized tone values
    for bar, tone in zip(bars, normalized_tones[::-1]):
        bar.set_alpha(tone)

    plt.xlabel('Count')
    plt.ylabel('Country Name')
    plt.title(f'Top Countries with Most {tone_type.capitalize()} Tone by Count')
    plt.show()

# Plotting the bar charts for the top countries with negative and positive tones, shaded by average tone


def filter_and_count_locations(df, country):
    # Filter rows where LOCATIONS column contains the country
    filtered_df = df[df['LOCATIONS'].str.contains(f'#{country}#', na=False, case=False)]
    location_counter = Counter()
    
    # Iterate over filtered rows to count locations
    for locations_str in filtered_df['LOCATIONS']:
        if locations_str:
            locations_list = locations_str.split(';')
            for location in locations_list:
                # Splitting by '#' and extracting the location name (2nd element after split)
                location_details = location.split('#')
                if len(location_details) > 2:
                    location_name = location_details[1]
                    location_counter[location_name] += 1
            
    return location_counter

def plot_minimalist_map(counter, title, color_map='Greens', power=0.5):

    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    locations_df = pd.DataFrame.from_dict(counter, orient='index', columns=['Count']).reset_index()
    locations_df = locations_df.rename(columns={'index': 'name'})

    # Adjusting counts for visualization purposes
    locations_df['Adjusted_Count'] = np.power(locations_df['Count'], power)

    # Merging with the world map data (this might require additional location matching logic)
    world = world.merge(locations_df, how='left', left_on='name', right_on='name')

    # Plotting
    fig, ax = plt.subplots(1, 1, figsize=(15, 10))
    ax.axis('off')
    world.plot(column='Adjusted_Count', ax=ax, cmap=color_map, missing_kwds={'color': 'lightgrey'}, linewidth=0)
    plt.title(title, fontdict={'fontsize': 16})
    plt.show()

# Example usage:
# counter = filter_and_count_locations(df, 'Malta')
# plot_minimalist_map(counter, "Location Counts in Malta")
# Note: Uncomment above lines to run the example. The actual country code should be adjusted according to the dataset.

def create_source_tone_dataframes_ordered_by_count(df):

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

def plot_source_counts_shaded_by_tone(df, top_n=30):

    sorted_df = df.sort_values(by='Count', ascending=False).head(top_n)

    tone_min = sorted_df['Avg_Tone'].min()
    tone_max = sorted_df['Avg_Tone'].max()
    normalized_tones = (sorted_df['Avg_Tone'] - tone_min) / (tone_max - tone_min)

    plt.figure(figsize=(10, 6))10, 8))
    bars = plt.barh(sorted_df['Source'][::-1], sorted_df['Count'][::-1], color='blue')

    # Adjusting bar colors based on normalized tone values
    for bar, tone in zip(bars, normalized_tones[::-1]):
        bar.set_alpha(tone)

    plt.xlabel('Count')
    plt.ylabel('Source')
    plt.title('Top Sources by Count (Shaded by Average Tone)')
    plt.show()



def create_location_tone_dataframes_ordered_by_count(df, top_n=30, location_type='1'):

    location_tone_sum = {}
    location_counter = Counter()

    # Processing each row for tone and count
    for _, row in df.iterrows():
        if pd.notna(row['LOCATIONS']) and pd.notna(row['TONE']):
            locations_list = row['LOCATIONS'].split(';')
            tone_values = row['TONE'].split(',')

            try:
                tone = float(tone_values[0])  # Extracting the first tone value
            except ValueError:
                continue  # Skip if tone value is not convertible to float

            for location in locations_list:
                location_details = location.split('#')
                if len(location_details) > 2 and location_details[0] == location_type:
                    location_name = location_details[1]
                    location_counter[location_name] += 1

                    if location_name in location_tone_sum:
                        location_tone_sum[location_name] += tone
                    else:
                        location_tone_sum[location_name] = tone

    # Calculating average tone for each location
    location_avg_tone = {loc: location_tone_sum[loc] / location_counter[loc] for loc in location_counter}

    # Separating positive and negative tones, and ordering by count
    positive_locations = [(loc, location_avg_tone[loc], count) for loc, count in location_counter.items() if location_avg_tone[loc] > 0]
    negative_locations = [(loc, location_avg_tone[loc], count) for loc, count in location_counter.items() if location_avg_tone[loc] < 0]

    # Sorting by count and selecting top 'n'
    top_positive_locations = sorted(positive_locations, key=lambda x: x[2], reverse=True)[:top_n]
    top_negative_locations = sorted(negative_locations, key=lambda x: x[2], reverse=True)[:top_n]

    # Creating DataFrames
    positive_df = pd.DataFrame(top_positive_locations, columns=['Location', 'Avg_Tone', 'Count'])
    negative_df = pd.DataFrame(top_negative_locations, columns=['Location', 'Avg_Tone', 'Count'])

    return positive_df, negative_df

def plot_side_by_side_location_tones(positive_df, negative_df, location_type='Locations'):
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    # Plot for positive tones
    tone_min_pos = positive_df['Avg_Tone'].min()
    tone_max_pos = positive_df['Avg_Tone'].max()
    normalized_tones_pos = (positive_df['Avg_Tone'] - tone_min_pos) / (tone_max_pos - tone_min_pos)
    bars_pos = axes[0].barh(positive_df['Location'][::-1], positive_df['Count'][::-1], color='green')
    for bar, tone in zip(bars_pos, normalized_tones_pos[::-1]):
        bar.set_alpha(tone)
    axes[0].set_title(f'Top {location_type} with Positive Tone by Count')
    axes[0].set_xlabel('Count')
    axes[0].set_ylabel(f'{location_type} Name')

    # Plot for negative tones
    tone_min_neg = negative_df['Avg_Tone'].min()
    tone_max_neg = negative_df['Avg_Tone'].max()
    normalized_tones_neg = (negative_df['Avg_Tone'] - tone_min_neg) / (tone_max_neg - tone_min_neg)
    bars_neg = axes[1].barh(negative_df['Location'][::-1], negative_df['Count'][::-1], color='red')
    for bar, tone in zip(bars_neg, normalized_tones_neg[::-1]):
        bar.set_alpha(tone)
    axes[1].set_title(f'Top {location_type} with Negative Tone by Count')
    axes[1].set_xlabel('Count')

    plt.tight_layout()
    plt.show()

