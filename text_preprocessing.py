import numpy as np 
import cPickle
import pandas as pd 
import re 
import string 
from string import translate
from nltk.corpus import stopwords 
from nltk.tokenize import TweetTokenizer
from nltk.tokenize.mwe import MWETokenizer
from nltk.collocations import BigramAssocMeasures, BigramCollocationFinder
from nltk.collocations import TrigramAssocMeasures, TrigramCollocationFinder

#df.apply is for operations on rows/columns
#df.applymap is for applying a function elementwise on dataframe 
#series.map is for applying function elementwise on series 

def tokenize(series):
    #tokenizes using nltk tweet tokenizer 
    token_series = series.map(custom_tokenizer)
    return token_series

def custom_preprocessor(text):
    #remove breaklines 
    text = re.sub('\n', ' ', text)
    #spell 420
    text = re.sub('420', 'fourtwenty', text)
    #remove all numbers 
    text = re.sub('[0-9]', ' ', text)
    return text 

def custom_tokenizer(text):
    tokenizer = TweetTokenizer(reduce_len = True, preserve_case = False)
    tokens = tokenizer.tokenize(text)

    return tokens

def remove_escape_sequences(text): 
    #removes escape sequences by only returning printable characters from string
    #allords = [i for i in xrange(256)]
    #allchars = ''.join(chr(i) for i in allords)
    printableords = [ord(ch) for ch in string.printable]
    deletechars  = ''.join(chr(i) for i in xrange(256) if i not in printableords)
    trans_table = dict.fromkeys(deletechars, None)
    clean_text = text.translate(trans_table) 
    return clean_text 

def add_pos_usage(df):
    #find percentage of verbs, adverbs, nouns, etc. 
    pass

def find_collocations(text_series): 
    #use nltk.collocations to find the most commonly occuring bigrams and trigrams
    bigram_measures = BigramAssocMeasures()
    trigram_measures = TrigramAssocMeasures()
    tokens = [token for token_list in text_series for token in token_list]
    bigrams = BigramCollocationFinder.from_words(tokens)
    trigrams  = TrigramCollocationFinder.from_words(tokens)
    scored_bigrams = bigrams.score_ngrams(bigram_measures.likelihood_ratio)
    scored_trigrams = trigrams.score_ngrams(trigram_measures.likelihood_ratio)
    #save to pickle
    with open('bigrams.pkl', 'wb') as fid:
        cPickle.dump(scored_bigrams,fid)
    with open('trigrams.pkl', 'wb') as fid:
        cPickle.dump(scored_trigrams, fid)

def mwe_tokenize(text_series):
    #Retokenizes tokenized text to combine MWEs from list of most common
    with open('trigrams.pkl', 'rb') as fid:
        trigrams = cPickle.load(fid) 

    tokenizer = MWETokenizer(trigrams, separator='+')
    text_series.map(tokenizer)
    return text_series 

def normalize(): 
    #add number of corrections as a feature
    #identify SMS language
	pass 

def custom_stop_words():
    stop_words = stopwords.words('english')
    #Remove anything that is age-related 
    randoms = ["i'm", 'im']
    relation = ['mother', 'husband', 'wife', 'daughter', 'dad', 'father', 'daddy', 'son']
    age_status = ['old', 'young', 'retired', 'year', 'youth', 'youthful', 'older', 'younger', 'mature', 'lady', 'girl', 'boy']
    stop_words.extend(randoms)
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
    expr = '([a-zA-Z])([0-9])'
    return num_09nyms

def add_num_09nyms(df): 
    #add the number of numeronyms as a feature 
    df['num_09nyms'] = df.ix[:, ['total_text','title']].map(is_09nyms).apply(sum, axis=0)  
    return df 

if __name__=='__main__':
    df = pd.read_pickle('df_age_predict.pkl')
    df[['title', 'total_text']] = df[['title', 'total_text']].applymap(lambda x: x.decode('ISO-8859-1'))
    df[['title', 'total_text']] = df[['title', 'total_text']].applymap(remove_escape_sequences)
    df = df.to_pickle('df_age_predict.pkl')
    #identify 09nyms, then add as feature
    #spell correct - account for number of corrections as feature
    #find collocations among total_text
    #find collocations among titles
    #MWE tokenizer -- apply for total_text 
    #MWE tokenizer -- apply for title 
