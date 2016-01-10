import modeling as md
import numpy as np 
import scipy as sc
import pandas as pd 
import cPickle 
import modeling as md
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from PIL import Image
from wordcloud import WordCloud
import matplotlib.pyplot as plt 
import text_preprocessing_eda as tp
import random

def print_top_tokens(model, feature_names, n_top_words, category):
   with open('results_m4w_4.txt', 'wb') as fid: 
       for topic_idx, topic in enumerate(model.components_):
           fid.write('\n')
	   fid.write(category)
	   fid.write('\n')
	   fid.write("Topic #%d:" % topic_idx)
	   fid.write(' '.join([feature_names[i] for i in topic.argsort()[:-n_top_words -1:-1]]))
	   #fid.write('\n')
	   #fid.write( for i in topic.argsort()[:-n_top_words -1:-1]])
	   fid.write('\n')

def topic_word_freq(topics, idx, feature_names): 
	freq_sum = np.sum(topics[idx]) 	
	frequencies =  [val/freq_sum for val in topics[idx]] 
	return zip(feature_names, frequencies) 

def get_data():
    with open('dataframe_for_eda.pkl', 'rb') as fid:
	df = cPickle.load(fid)
    return df

def reduce_dimensions(total_mat, n_topics):
    #input is data matrix, shape (n_samples, n_features)
    #returns W array, shape (n_samples, n_components)
    
    nmf = NMF(n_components = n_topics, random_state=42, alpha=.2,  l1_ratio=0.5)
    nmf.fit(total_mat)
    X = nmf.transform(total_mat) 
    w = nmf.components_ 
    return nmf 

def grey_color_func(word, font_size, position, orientation, random_state=None, **kwargs):
    return "hsl(0, 0%%, %d%%)" % random.randint(70, 100)

if __name__=='__main__':
    df = get_data()
    mask_path = False
    #Use tfidf features for NMF
    for category in df.category_code.unique().tolist():
        if category == 'mis':
            #df_cat = df.ix[df['category_code']==category, :]
            text_mat, text_features = md.tfidf_matrix(df.ix[df['category_code'] == category, 'total_text'])
            #Fit NMF
            n_samples = text_mat.shape[0]
            n_features = text_mat.shape[1]
            n_topics = 10 
            n_top_words = 500
            print category 
            print 'Fitting the NMF model with tf-idf features'
            nmf = reduce_dimensions(text_mat, n_topics) 
            #print_top_tokens(nmf,text_features,n_top_words, category)

            word_freq = topic_word_freq(nmf.components_, 2, text_features)
            wc = WordCloud(stopwords=tp.custom_stop_words(), background_color='black', max_words=n_top_words, width=2000, height=1800)
            wc.fit_words(word_freq)
            plt.figure()
            plt.imshow(wc)
            #wc.recolor(color_func=grey_color_func, random_state=3)
            #wc.to_file('background.png')
            plt.axis('off')
