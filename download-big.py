#%%
import rat_test as dl 
import data_helpers as el
# k = 1
output_file_dir = 'big_dump/'
input_date = "06/06/2014"
n = 90
n_days = list(range(1,n+1))
locations_regex = 'Israel|Occupied Palestenian Territory|Iraq|Saudi Arabia|Yemen|'  # Adjust the regex pattern as needed

for k in n_days:
    print('trying')
    dl.download_and_filter_gdelt_data(k,1,output_file_dir+'gkg_'+str(k)+'.json', None, locations_regex, None)
    print('it works!')





# %%
n = 35
n_days = list(range(1,n+1))
filenames = [output_file_dir+'gkg_'+str(k)+'.json' for k in n_days]
for path in filenames:
    print('loading data')
    el.create_and_load_es_index(9200, path, 'gulf_30day')
    print('loaded')
    print(' ')
# %%
