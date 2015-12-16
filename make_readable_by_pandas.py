#Script reads in json, fixes format to that it is readable by pandas
#Then appends each to a list

import pandas as pd 
import os, sys 

def readable_df(file): 
    #returns df that is readable by pandas
    with open(file, 'rb') as f:
        data = f.readlines()
        #remove the trailing '\n' from each line
        data = map(lambda x: x.rstrip(), data)
        #each element of 'data' is a JSON object
        #converting to 'array' of JSON objects by adding brackets, comma
        data_json_str = '[' + ','.join(data) + ']'
        #can load into dataframe 
        data_df = pd.read_json(data_json_str)
        return data_df  

if __name__=='__main__':
#create empty dataframe for loop to work without case for first item
    total_df = pd.DataFrame()

    for root, dirs, files in os.walk(os.getcwd()):
        for file in files:
            #ignore this script, swp files, etc.
            if file.endswith('.json'):
                data_df = readable_df(file)
                #add data_df to list
                total_df = total_df.append(data_df) 
                print total_df.shape
