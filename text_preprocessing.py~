#Embedded file name: text_preprocessing.py
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
from nltk.stem.snowball import SnowballStemmer

def tokenize(series):
    token_series = series.map(custom_tokenizer)
    return token_series


def custom_preprocessor(text):
    text = re.sub('\n', ' ', text)
    text = re.sub('420', 'fourtwenty', text)
    text = re.sub('[0-9]', ' ', text)
    return text


def custom_tokenizer(text, bigrams = None):
    chunks = text.split('-')
    tokenizer = TweetTokenizer(reduce_len=True, preserve_case=False)
    tokens = tokenizer.tokenize(text)
    tokens = [ subchunk for chunk in chunks for subchunk in tokenizer.tokenize(chunk) ]
    tokens = [ token for token in tokens if token.isalpha() ]
    if bigrams:
        tokens = mwe_tokenize(tokens, bigrams)
    stemmer = SnowballStemmer('english', ignore_stopwords=True)
    tokens = [ stemmer.stem(token) for token in tokens ]
    return tokens


def remove_escape_sequences(text):
    printableords = [ ord(ch) for ch in string.printable ]
    deletechars = ''.join((chr(i) for i in xrange(256) if i not in printableords))
    trans_table = dict.fromkeys(deletechars, None)
    clean_text = text.translate(trans_table)
    return clean_text


unicodeToAsciiMap = {u'\u2019': "'",
 u'\u2018': '`'}

def unicodeToAscii(inStr):
    try:
        return str(inStr)
    except:
        pass

    outStr = ''
    for i in inStr:
        try:
            outStr = outStr + str(i)
        except:
            if unicodeToAsciiMap.has_key(i):
                outStr = outStr + unicodeToAsciiMap[i]
            else:
                try:
                    i
                except:
                    i

                outStr = outStr + '_'

    return outStr


def add_pos_usage(df):
    pass


def find_collocations(text_series):
    bigram_measures = BigramAssocMeasures()
    trigram_measures = TrigramAssocMeasures()
    tokens = [ token for token_list in text_series for token in token_list ]
    bigrams = BigramCollocationFinder.from_words(tokens)
    trigrams = TrigramCollocationFinder.from_words(tokens)
    scored_bigrams = bigrams.score_ngrams(bigram_measures.likelihood_ratio)
    scored_trigrams = trigrams.score_ngrams(trigram_measures.likelihood_ratio)
    with open('bigrams.pkl', 'wb') as fid:
        cPickle.dump(scored_bigrams, fid)
    with open('trigrams.pkl', 'wb') as fid:
        cPickle.dump(scored_trigrams, fid)


def mwe_tokenize(tokens, bigrams):
    tokenizer = MWETokenizer(mwes=bigrams[:100], separator='+')
    tokens = tokenizer.tokenize(tokens)
    return tokens


def normalize():
    pass


def custom_stop_words():
    stop_words = stopwords.words('english')
    randoms = ["i'm", 'im']
    relation = ['mother',
     'husband',
     'wife',
     'daughter',
     'dad',
     'father',
     'daddy',
     'son']
    age_status = ['old',
     'young',
     'retired',
     'year',
     'youth',
     'youthful',
     'older',
     'younger',
     'mature',
     'lady',
     'girl',
     'boy']
    stop_words.extend(randoms)
    stop_words.extend(relation)
    stop_words.extend(age_status)
    return stop_words


def is_09nyms(df):
    expr = '([a-zA-Z])([0-9])'
    return num_09nyms


def add_num_09nyms(df):
    df['num_09nyms'] = df.ix[:, ['total_text', 'title']].map(is_09nyms).apply(sum, axis=0)
    return df


if __name__ == '__main__':
    df = pd.read_pickle('dataframe_for_eda.pkl')
    df[['title', 'total_text']] = df[['title', 'total_text']].applymap(lambda x: x.decode('ISO-8859-1'))
    df[['title', 'total_text']] = df[['title', 'total_text']].applymap(remove_escape_sequences)
    df = df.to_pickle('dataframe_for_eda_edited.pkl')
