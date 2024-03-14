#%% 
from elasticsearch import Elasticsearch, helpers
import json
from datetime import datetime
import pandas as pd 

def create_and_load_es_index(address,port, file_path, index_name,pwd):
    es = Elasticsearch(
        ["https://"+address+":" + str(port)],
        basic_auth=('elastic', pwd),
        verify_certs=False
    )

    # Updated mapping with a geo_point field
    mapping = {
        "mappings": {
            "properties": {
                "DATE": {"type": "date"},
                "NUMARTS": {"type": "integer"},
                "COUNTS": {"type": "integer"},
                "THEMES": {"type": "keyword"},
                "location_type": {"type": "keyword"},
                "location_name": {"type": "keyword"},
                "country_code": {"type": "keyword"},
                "adm1_code": {"type": "keyword"},
                "latitude": {"type": "float"},
                "longitude": {"type": "float"},
                "geo_location": {"type": "geo_point"},  # Added geo_point field
                "feature_id": {"type": "keyword"},
                "PERSONS": {"type": "keyword"},
                "ORGANIZATIONS": {"type": "keyword"},
                "TONE": {
                    "type": "object",
                    "properties": {
                        "tone1": {"type": "float"},
                        "tone2": {"type": "float"},
                    }
                },
                "CAMEOEVENTIDS": {"type": "keyword"},
                "SOURCES": {"type": "keyword"},
                "SOURCEURLS": {"type": "keyword"}
            }
        }
    }

    if not es.indices.exists(index=index_name):
        es.indices.create(index=index_name, body=mapping)

    def parse_tone(tone_str):
        tone_values = tone_str.split(',') if tone_str else []
        return {f'tone{index + 1}': float(value) for index, value in enumerate(tone_values)}

    with open(file_path, 'r') as file:
        try:
            data = json.load(file)
        except json.JSONDecodeError as e:
            print("Error reading JSON file:", e)
            return

    actions = []

    for entry in data:
        try:
            entry['DATE'] = datetime.strptime(str(entry['DATE']), '%Y%m%d').isoformat()
            entry['THEMES'] = entry['THEMES'].split(';') if entry['THEMES'] else []
            entry['PERSONS'] = entry['PERSONS'].split(';') if entry['PERSONS'] else []
            entry['ORGANIZATIONS'] = entry['ORGANIZATIONS'].split(';') if entry['ORGANIZATIONS'] else []

            entry['location_type'] = []
            entry['location_name'] = []
            entry['country_code'] = []
            entry['adm1_code'] = []
            entry['latitude'] = []
            entry['longitude'] = []
            entry['feature_id'] = []
            entry['geo_location'] = []  # Initialize the geo_location field

            if entry['LOCATIONS']:
                for loc in entry['LOCATIONS'].split(';'):
                    loc_parts = loc.split('#')
                    if len(loc_parts) == 7:
                        lat = float(loc_parts[4]) if loc_parts[4] else None
                        lon = float(loc_parts[5]) if loc_parts[5] else None
                        if lat is not None and lon is not None:
                            entry['geo_location'].append({"lat": lat, "lon": lon})
                        
                        # Other location fields
                        entry['location_type'].append(loc_parts[0])
                        entry['location_name'].append(loc_parts[1])
                        entry['country_code'].append(loc_parts[2])
                        entry['adm1_code'].append(loc_parts[3])
                        entry['latitude'].append(lat)
                        entry['longitude'].append(lon)
                        entry['feature_id'].append(loc_parts[6])

            del entry['LOCATIONS']
            entry['TONE'] = parse_tone(entry['TONE'])

            action = {
                "_index": index_name,
                "_source": entry
            }
            actions.append(action)
        except Exception as e:
            print(f"Error processing entry: {e}")

    try:
        helpers.bulk(es, actions)
    except Exception as e:
        print(f"Error in bulk indexing: {e}")

# file_path = 'gdelt_mlt.json'
# index_name = 'mlt_test2'
# create_and_load_es_index(9200, file_path, index_name)

def batch_create_and_load_es_index(port, file_path, pwd, index_name,batch_size=1000):
    # Connect to Elasticsearch instance
    es = Elasticsearch(
        ["https://localhost:" + str(port)],
        basic_auth=('elastic', pwd),
        verify_certs=False
    )

    # Define the mapping for the index
    mapping = {
        "mappings": {
            "properties": {
                "DATE": {"type": "date"},
                "NUMARTS": {"type": "integer"},
                "COUNTS": {"type": "integer"},  # Corrected the type from "intiger" to "integer"
                "THEMES": {"type": "keyword"},
                "location_type": {"type": "keyword"},
                "location_name": {"type": "keyword"},
                "country_code": {"type": "keyword"},
                "adm1_code": {"type": "keyword"},
                "latitude": {"type": "float"},
                "longitude": {"type": "float"},
                "feature_id": {"type": "keyword"},
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

    def load_json_in_batches(file_path, batch_size):
        with open(file_path, 'r') as file:
            # Try to parse the whole file as a JSON array
            try:
                data = json.load(file)
                for i in range(0, len(data), batch_size):
                    yield data[i:i + batch_size]
            except json.JSONDecodeError:
                # If that fails, assume each line is a separate JSON object
                file.seek(0)  # Reset file read position
                batch = []
                for line in file:
                    batch.append(json.loads(line))
                    if len(batch) == batch_size:
                        yield batch
                        batch = []
                if batch:
                    yield batch

    total_entries = 0

    for batch in load_json_in_batches(file_path, batch_size):
        actions = []
        for entry in batch:
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

        print(f"Total entries loaded: {total_entries}")


def download_es_index_to_csv(port, passwd, index_name, output_file):
    # Connect to the Elasticsearch instance
    es = Elasticsearch(
        ["https://localhost:" + str(port)],
        basic_auth=('elastic', passwd),
        verify_certs=False
    )

    # Prepare a scan query to retrieve all documents from the specified index
    query = {"query": {"match_all": {}}}
    
    # Use the scan helper to retrieve all documents
    results = helpers.scan(es, query=query, index=index_name)
    
    # Initialize a list to hold the documents
    documents = []

    # Iterate over the scan results and store each document
    for result in results:
        documents.append(result['_source'])

    # Convert the list of documents into a DataFrame
    df = pd.DataFrame(documents)
    
    # Save the DataFrame to a CSV file
    df.to_csv(output_file, index=False)
    print(f"Saved {len(df)} documents to '{output_file}'")

# Example usage
# es_instance_url = 9200  # Change this to your Elasticsearch instance URL
# index_name = '3week_test'  # Specify the index name
# output_file = 'es_index_data.csv'  # Specify the path to the output CSV file

# download_es_index_to_csv(es_instance_url, index_name, output_file)


def download_es_data(port, index_name, output_format, query=None, output_file=None):
    """
    Download data from an Elasticsearch instance based on a query and store it as specified.
    
    Parameters:
    es_instance_url (str): URL of the Elasticsearch instance.
    index_name (str): Name of the index to download data from.
    output_format (str): Format to store the data ('csv', 'json', or 'dataframe').
    query (dict): Elasticsearch query for filtering data. If None, downloads entire index.
    output_file (str): Path to the output file. Required if output_format is 'csv' or 'json'.
    
    Returns:
    pandas.DataFrame: If output_format is 'dataframe', returns the data as a DataFrame.
    """
    # Connect to the Elasticsearch instance
    es = Elasticsearch(
        ["https://localhost:" + str(port)],
        basic_auth=('elastic', '_b2x4M4+wjlfiJVTUPLI'),
        verify_certs=False
    )
    # Use a match_all query if none is provided
    if query is None:
        query = {"query": {"match_all": {}}}
    
    # Use the scan helper to retrieve all documents based on the query
    results = helpers.scan(es, query=query, index=index_name)
    
    # Initialize a list to hold the documents
    documents = []

    # Iterate over the scan results and store each document
    for result in results:
        documents.append(result['_source'])

    # Depending on the desired output format, process the documents accordingly
    if output_format.lower() == 'csv' or output_format.lower() == 'json':
        if output_file is None:
            raise ValueError("output_file is required when output_format is 'csv' or 'json'")
        
        # Convert the list of documents into a DataFrame
        df = pd.DataFrame(documents)

        # Save to the specified format
        if output_format.lower() == 'csv':
            df.to_csv(output_file, index=False)
        elif output_format.lower() == 'json':
            df.to_json(output_file, orient='records')
        
        print(f"Saved {len(df)} documents to '{output_file}' in {output_format.upper()} format.")
    
    elif output_format.lower() == 'dataframe':
        # Return the documents as a DataFrame
        return pd.DataFrame(documents)
    else:
        raise ValueError("output_format must be one of 'csv', 'json', or 'dataframe'")

# # Example usage:
# es_instance_url = 9200  # Change this to your Elasticsearch instance URL
# index_name = 'your_index_name'  # Specify the index name

# # To download the entire index as a CSV
# download_es_data(es_instance_url, index_name, 'csv', output_file='es_data.csv')

# # To download filtered data as a DataFrame
# query = {
#     "query": {
#         "match": {
#             "your_field": "your_value"
#         }
#     }
# }
# df = download_es_data(es_instance_url, index_name, 'dataframe', query=query)
# print(df.head())

# file_path = '7day_gkg.json'
# index_name = '7day_gkg_test_bruh2'
# create_and_load_es_index(9200, file_path, index_name)
