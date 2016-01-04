from flask import Flask
import pandas as pd 
import numpy as np 
import text_preprocessing as tp 
import random 
import cPickle 
import string
from flask_bootstrap import Bootstrap

app = Flask(__name__)

def get_random_post(df): 
    smut_words = ['oral', 'load', 'unload','blow', 'cut', 'inches', 'ass', 'fuck', 'dick', 'tits', 'breasts', 'rim', 'hole', 'pussy', 'wet', 'lick', 'suck', 'cock', 'hung', 'balls']
    smut_pattern = '|'.join(smut_words)   
    post = pd.Series(['ass', 'ass'])
    while post[:2].str.lower().str.contains(smut_pattern).any():
        #return random posting
        idx = random.randint(0, df.shape[0])
        post = df.ix[idx, ['title', 'total_text', 'age']]
    return post

def load_data():
    df = pd.read_pickle('df_age_predict_edited.pkl')
    #filter out smutty
    return df 

def posting_text(text): 
    text = tp.custom_preprocessor(text)
    text = ' '.join([word for word in text.replace("\\","").split(' ') if word != ''])
    return text 

def posting(df): 
    post = get_random_post(df)
    post[['title','total_text']] = post[['title', 'total_text']].map(posting_text)
    return post

@app.route('/')
def run():
    df = load_data()
    post = posting(df)
    # post['title'] 
    #print "\n"
    return post['total_text']

if __name__ == '__main__':
    app.debug = True
    app.run()
