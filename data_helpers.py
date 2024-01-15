#%% 
from elasticsearch import Elasticsearch, helpers
import json
from datetime import datetime

def create_and_load_es_index(port, file_path, index_name):
    # Connect to Elasticsearch instance
    es = Elasticsearch(
        ["https://localhost:" + str(port)],
        basic_auth=('elastic', '_b2x4M4+wjlfiJVTUPLI'),
        verify_certs=False
    )

    # Define the mapping for the index
    mapping = {
        "mappings": {
            "properties": {
                "DATE": {"type": "date"},
                "NUMARTS": {"type": "integer"},
                "COUNTS": {"type": "text"},
                "THEMES": {"type": "keyword"},
                "LOCATIONS": {
                    "type": "nested",
                    "properties": {
                        "type": {"type": "keyword"},
                        "name": {"type": "text"},
                        "country_code": {"type": "keyword"},
                        "adm1_code": {"type": "keyword"},
                        "lat": {"type": "float"},
                        "long": {"type": "float"},
                        "feature_id": {"type": "keyword"}
                    }
                },
                "PERSONS": {"type": "text"},
                "ORGANIZATIONS": {"type": "text"},
                "TONE": {
                    "type": "object",
                    "properties": {
                        "tone1": {"type": "float"},
                        "tone2": {"type": "float"},
                        # Add additional tone fields as required
                    }
                },
                "CAMEOEVENTIDS": {"type": "text"},
                "SOURCES": {"type": "keyword"},
                "SOURCEURLS": {"type": "keyword"}
            }
        }
    }

    # Check if the index already exists, and create it with the mapping if it doesn't
    if not es.indices.exists(index=index_name):
        es.indices.create(index=index_name, body=mapping)

    # Function to parse TONE field
    def parse_tone(tone_str):
        tone_values = tone_str.split(',') if tone_str else []
        return {f'tone{index + 1}': float(value) for index, value in enumerate(tone_values)}

    # Process and load data
    with open(file_path, 'r') as file:
        try:
            data = json.load(file)
        except json.JSONDecodeError as e:
            print("Error reading JSON file:", e)
            return

    actions = []

    for entry in data:
        try:
            # Format DATE field
            entry['DATE'] = datetime.strptime(str(entry['DATE']), '%Y%m%d').isoformat()

            # Split THEMES into an array
            entry['THEMES'] = entry['THEMES'].split(';') if entry['THEMES'] else []

            # Process LOCATIONS
            locations = []
            if entry['LOCATIONS']:
                for loc in entry['LOCATIONS'].split(';'):
                    loc_parts = loc.split('#')
                    if len(loc_parts) == 7:
                        location = {
                            'type': loc_parts[0],
                            'name': loc_parts[1],
                            'country_code': loc_parts[2],
                            'adm1_code': loc_parts[3],
                            'lat': float(loc_parts[4]) if loc_parts[4] else None,
                            'long': float(loc_parts[5]) if loc_parts[5] else None,
                            'feature_id': loc_parts[6]
                        }
                        locations.append(location)
            entry['LOCATIONS'] = locations

            # Process TONE
            entry['TONE'] = parse_tone(entry['TONE'])

            action = {
                "_index": index_name,
                "_source": entry
            }
            actions.append(action)
        except Exception as e:
            print(f"Error processing entry: {e}")

    # Bulk index the data
    try:
        helpers.bulk(es, actions)
    except Exception as e:
        print(f"Error in bulk indexing: {e}")

file_path = 'gdelt_mlt.json'
index_name = 'mlt_test2'
create_and_load_es_index(9200, file_path, index_name)


#%%
def load_data_into_existing_es_index(port,file_path, index_name):
    # Connect to Elasticsearch instance
    es = Elasticsearch()

    # Ensure the index exists
    if not es.indices.exists(index=index_name):
        raise ValueError(f"Index '{index_name}' does not exist.")

    # Process and load data
    with open(file_path, 'r') as file:
        data = json.load(file)
        actions = []

        for entry in data:
            # Split THEMES into an array
            if entry['THEMES'] is not None:
                entry['THEMES'] = entry['THEMES'].split(';')

            # Process LOCATIONS
            if entry['LOCATIONS'] is not None:
                locations = []
                for loc in entry['LOCATIONS'].split(';'):
                    loc_parts = loc.split('#')
                    if len(loc_parts) == 7:
                        locations.append({
                            'type': loc_parts[0],
                            'name': loc_parts[1],
                            'country_code': loc_parts[2],
                            'adm1_code': loc_parts[3],
                            'lat': loc_parts[4],
                            'long': loc_parts[5],
                            'feature_id': loc_parts[6]
                        })
                entry['LOCATIONS'] = locations

            # Split TONE into an array of numbers
            if entry['TONE'] is not None:
                entry['TONE'] = [float(tone) for tone in entry['TONE'].split(',')]

            action = {
                "_index": index_name,
                "_source": entry
            }
            actions.append(action)

        # Bulk index the data
        helpers.bulk(es, actions)

# Usage
# file_path = '/path/to/your/gdelt_mlt.json'
# index_name = 'existing_index_name'
# load_data_into_existing_es_index(file_path, index_name)

#%%
from elasticsearch import Elasticsearch, helpers
import json
from datetime import datetime

def create_and_load_es_index(port, file_path, index_name):
    # Connect to Elasticsearch instance
    es = Elasticsearch(
        ["https://localhost:" + str(port)],
        basic_auth=('elastic', '_b2x4M4+wjlfiJVTUPLI'),
        verify_certs=False
    )

    # Define the mapping for the index
    mapping = {
        "mappings": {
            "properties": {
                "DATE": {"type": "date"},
                "NUMARTS": {"type": "integer"},
                "COUNTS": {"type": "intiger"},
                "THEMES": {"type": "keyword"},
                "LOCATIONS": {
                    "type": "nested",
                    "properties": {
                        "type": {"type": "keyword"},
                        "name": {"type": "keyword"},
                        "country_code": {"type": "keyword"},
                        "adm1_code": {"type": "keyword"},
                        "lat": {"type": "float"},
                        "long": {"type": "float"},
                        "feature_id": {"type": "keyword"}
                    }
                },
                "PERSONS": {"type": "keyword"},  # Changed to keyword
                "ORGANIZATIONS": {"type": "keyword"},  # Changed to keyword
                "TONE": {
                    "type": "object",
                    "properties": {
                        "tone1": {"type": "float"},
                        "tone2": {"type": "float"},
                        # Add additional tone fields as required
                    }
                },
                "CAMEOEVENTIDS": {"type": "keyword"},
                "SOURCES": {"type": "keyword"},
                "SOURCEURLS": {"type": "keyword"}
            }
        }
    }

    # Check if the index already exists, and create it with the mapping if it doesn't
    if not es.indices.exists(index=index_name):
        es.indices.create(index=index_name, body=mapping)

    # Function to parse TONE field
    def parse_tone(tone_str):
        tone_values = tone_str.split(',') if tone_str else []
        return {f'tone{index + 1}': float(value) for index, value in enumerate(tone_values)}

    # Process and load data
    with open(file_path, 'r') as file:
        try:
            data = json.load(file)
        except json.JSONDecodeError as e:
            print("Error reading JSON file:", e)
            return

    actions = []

    for entry in data:
        try:
            # Format DATE field
            entry['DATE'] = datetime.strptime(str(entry['DATE']), '%Y%m%d').isoformat()

            # Split THEMES into an array
            entry['THEMES'] = entry['THEMES'].split(';') if entry['THEMES'] else []

            # Split PERSONS into an array
            entry['PERSONS'] = entry['PERSONS'].split(';') if entry['PERSONS'] else []

            # Split ORGANIZATIONS into an array
            entry['ORGANIZATIONS'] = entry['ORGANIZATIONS'].split(';') if entry['ORGANIZATIONS'] else []

            # Process LOCATIONS
            locations = []
            if entry['LOCATIONS']:
                for loc in entry['LOCATIONS'].split(';'):
                    loc_parts = loc.split('#')
                    if len(loc_parts) == 7:
                        location = {
                            'type': loc_parts[0],
                            'name': loc_parts[1],
                            'country_code': loc_parts[2],
                            'adm1_code': loc_parts[3],
                            'lat': float(loc_parts[4]) if loc_parts[4] else None,
                            'long': float(loc_parts[5]) if loc_parts[5] else None,
                            'feature_id': loc_parts[6]
                        }
                        locations.append(location)
            entry['LOCATIONS'] = locations

            # Process TONE
            entry['TONE'] = parse_tone(entry['TONE'])

            action = {
                "_index": index_name,
                "_source": entry
            }
            actions.append(action)
        except Exception as e:
            print(f"Error processing entry: {e}")

    # Bulk index the data
    try:
        helpers.bulk(es, actions)
    except Exception as e:
        print(f"Error in bulk indexing: {e}")

file_path = 'gdelt_mlt.json'
index_name = 'mlt_test4bruh2'
create_and_load_es_index(9200, file_path, index_name)


# %%
