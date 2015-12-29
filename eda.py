import modeling as md
import numpy as np 
import scipy as sc
import pandas as pd 
import cPickle 

from sklearn.decomposition import NMF 




def get_data():
    with open('d.pkl', 'rb') as fid:
	df = cPickle.load(fid)
    return df

def reduce_dimensions(total_mat, n_topics):
    #input is data matrix, shape (n_samples, n_features)
    #returns W array, shape (n_samples, n_components)
    
    nmf = NMF(n_components = n_topics, random_state=42)
    nmf.fit_transform(total_mat)
    return nmf 

if __name__=='__main__':
    df = get_data()
    text_mat, text_features = md.tfidf_matrix(df['total_text'])
    
    
