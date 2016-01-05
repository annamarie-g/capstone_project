import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns 


def read_data(): 
    df = pd.read_pickle('dataframe_for_eda.pkl')
    return df 

def age_histogram(df_age):

def num_images_by_age(df_age):

def post_length_by_age(df): 



if __name__ == '__main__':
    df = read_data()
    df_age = df[df['age'].notnull(), :] 
    age_histogram(df)
