#Only run script on Amazon AWS
#saves concatenated dataframe as pkl 

import pandas as pd 
import os, sys 

def readable_df(file): 
    pass 

if __name__=='__main__':
#create empty dataframe for loop to work without case for first item
    total_df = pd.DataFrame()

    for root, dirs, files in os.walk(os.getcwd()):
        for file in files:
            #ignore this script, swp files, etc.
            if file.endswith('.json'):
                df = pd.read_json(file)
                #add data_df to list
                total_df = total_df.append(data_df) 
                print total_df.shape
    
    #Save to pickle
    df.to_pickle('total_dataset.pkl')
