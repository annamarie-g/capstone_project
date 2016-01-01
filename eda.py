import modeling as md
import numpy as np 
import scipy as sc
import pandas as pd 
import cPickle 
import modeling as md
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer


def print_top_tokens(model, feature_names, n_top_words, category):

   for topic_idx, topic in enumerate(model.components_):
	with open('results.txt', 'a') as fid:
	   fid.write(category)
           fid.write('\n')
	   fid.write("Topic #%d:" % topic_idx)
	   fid.write(' '.join([feature_names[i] for i in topic.argsort()[:-n_top_words -1:-1]]))

def get_data():
    with open('dataframe_for_eda.pkl', 'rb') as fid:
	df = cPickle.load(fid)
    return df

def reduce_dimensions(total_mat, n_topics):
    #input is data matrix, shape (n_samples, n_features)
    #returns W array, shape (n_samples, n_components)
    
    nmf = NMF(n_components = n_topics, random_state=42)
    nmf.fit(total_mat)
    return nmf 

if __name__=='__main__':
    df = get_data()

    #Use tfidf features for NMF
    for category in df.category_code.unique.tolist():
    	#df_cat = df.ix[df['category_code']==category, :]
    	text_mat, text_features = md.tfidf_matrix(df.ix[df['category_code'] == category, 'total_text'])
    
    	#Fit NMF
    	n_samples = text_mat.shape[0]
	n_features = text_mat.shape[1]
	n_topics = 5
	n_top_words = 20
	print category 
	print 'Fitting the NMF model with tf-idf features'
	nmf = reduce_dimensions(text_mat, n_topics) 

	print_top_tokens(nmf,text_features,n_top_words, category)

