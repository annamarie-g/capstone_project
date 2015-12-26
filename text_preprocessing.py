import numpy as np 
import pandas as pd 
import re 
from nltk.corpus import stopwords 
from nltk.tokenize import TweetTokenizer
from nltk.tokenize.mwe import MWETokenizer
from nltk.collocations import BigramAssocMeasures
from nltk.collocations import TrigramAssocMeasures

#df.apply is for operations on rows/columns
#df.applymap is for applying a function elementwise on dataframe 
#series.map is for applying function elementwise on series 

def word_tokenize(df):
    #tokenizes using nltk tweet tokenizer
    tokenizer = TweetTokenizer(reduce_len = True, preserve_case = False)
    df[['title_word_tokens', 'total_text_word_tokens']] = df[['title', 'total_text']].applymap(tokenizer.tokenize)
    return df 

def custom_tokenizer(text):
    #spell 420
    text = re.sub('420', 'fourtwenty', text)
    #remove age
    text = re.sub('[0-9]', ' ', text)
    tokenizer = TweetTokenizer(reduce_len = True, preserve_case = False)
    tokens = tokenizer.tokenize(text)
    return tokens

def add_pos_usage(df):
    #find percentage of verbs, adverbs, nouns, etc. 
    pass

def find_collocations(text_series): 
    #use nltk.collocations to find the most commonly occuring bigrams and trigrams
    bigram_measures = BigramAssocMeasures()
    trigram_measures = TrigramAssocMeasures()
    tokens = text_series.tolist() 
    finder_bigrams = BigramCollocationFinder.from_words(tokens,window_size =20)
    finder_trigrams = TrigramCollocationFinder.from_words(tokens, window_size =20)
    #finder = BigramCollocationFinder.from_words(
    #returns list of MWEs
    #return MWEs 
    return bigram_MWEs, trigram_MWEs 

def mwe_tokenize(text_series):
    #Retokenizes tokenized text to combine MWEs from list of most common
    MWEs  = _find_collocations(text_series)
    tokenizer = MWETokenizer(MWEs, separator='+')
    text_series.map(tokenizer)
    return text_series 

def normalize(): 
    #add number of corrections as a feature
    #identify SMS language
	pass 

def custom_stop_words():
    stop_words = stopwords.words('english')
    #Remove anything that is age-related 
    relation = ['mother', 'husband', 'wife', 'daughter', 'dad', 'father', 'daddy', 'son']
    age_status = ['old', 'young', 'retired', 'year', 'youth', 'youthful', 'older', 'younger', 'mature', 'lady', 'girl', 'boy']
    stop_words.extend(relation)
    stop_words.extend(age_status)
    return stop_words 

'''
def change_420(df): 
    #change four twenty to text 
    df.ix[:,['title', 'total_text']] = df.ix[:, ['title', 'total_text']].applymap(lambda x: re.sub('420', 'fourtwenty', x))
    return df 

def remove_age(df):
    #remove all mentions of age
    #make sure that only removing 2 consecutive numbers 
    df.ix[:, ['title','total_text']] = df.ix[:, ['title', 'total_text']].applymap(lambda x: re.sub('\s[0-9]{2}', ' ', x))
    return df 
'''

def is_09nyms(df):
    #find all numeronyms and count number of occurences-- use as feature
    return num_09nyms

def add_num_09nyms(df): 
    #add the number of numeronyms as a feature 
    df['num_09nyms'] = df.ix[:, ['total_text','title']].map(is_09nyms).apply(sum, axis=0)  
    return df 

def remove_09nyms(df):
    pass  


if __name__=='__main__':
    df = pd.read_pickle('df_age_predict.pkl')
    df[['title', 'total_text']] = df[['title', 'total_text']].applymap(lambda x: x.decode('ISO-8859-1'))
    #fix fourtwenty so it's not removed 
    df = change_420(df)
    #remove mentions of age   
    df = remove_age(df)
    #identify 09nyms, then remove
    
    #tokenize words in total text and title 
    df = word_tokenize(df)
    #spell correct - account for number of corrections as feature

    #find collocations among total_text

    #find collocations among titles
    #MWE tokenizer -- apply for total_text 
    #MWE tokenizer -- apply for title 
