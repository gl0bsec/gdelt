#%%
#Import common  
import json
import pandas as pd
import sqlite3
import os 

# Deserialize and dump
#%% 
def add_unique_id(df):
    """
    Add a unique ID column to the DataFrame.
    """
    df['unique_id'] = range(1, len(df) + 1)
    return df

def load_data_df(df,db_name,table_name):
    # Step 3: Create SQLite DB
    conn = sqlite3.connect(db_name)
    
    # Step 4: Create Table and Insert Data
    df.to_sql(table_name, conn, if_exists='replace', index=False)
    print(f"Successfully loaded data into {db_name}, table name: {table_name}")
    conn.close()
    return 

def load_data(filename,db_name,table_name):
    with open(filename, 'r', errors='ignore') as f:
        json_data = json.load(f)
    df = pd.DataFrame(json_data)
    
    # Step 2: Add Unique ID
    df = add_unique_id(df)
    
    # Step 3: Create SQLite DB
    conn = sqlite3.connect(db_name)
    
    # Step 4: Create Table and Insert Data
    df.to_sql(table_name, conn, if_exists='replace', index=False)
    print(f"Successfully loaded data into {db_name}, table name: {table_name}")
    conn.close()
    return 

def mass_load(filenames,db_name,table_name): 
    for name in filenames:
        if os.path.exists('loaded/'+name):
            print("exists")
        else: 
            load_data(name,db_name,table_name) 
    return
# %%
