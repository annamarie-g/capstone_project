#Steps needed for preprocessing 
import pandas as pd
import numpy as np 
#for detecting langugage
import nltk
nltk.download()
from nltk.corpus import stopwords
from nltk import wordpunct_tokenize
import re 

#df.apply is for operations on rows/columns
#df.applymap is for applying a function elementwise on dataframe 
#series.map is for applying function elementwise on series 

def create_age_groups(df):
    #get rid of all ages over 91 
    df = df.ix[df.age < 91,:]
    df['age_group'] = pd.cut(df.ix[:,'age'], range(5, 95,10), right=False,labels= [ "{0} - {1}".format(i, i + 9) for i in range(5, 85, 10)])
    return df 

def create_total_text(df):
    #adds all the text from attributes together and creates column total text 
    attributes = df[['weight', 'status', 'kids, want', 'general', 'body', 'body art', 'diet', 'dislikes', 'education', 'ethnicity', 'eye color', 'facial hair', 'fears', 'hair', 'stds', 'interests', 'kids, have', 'likes', 'native language', 'occupation', 'personality','pets', 'politics', 'religion', 'resembles']].fillna('')    
    #create series of just spaces and periods 
    space = pd.Series(' ', index=np.arange(len(attributes)))
    attributes['total'] = space 
    attributes['total'] = attributes['total'].fillna(' ') #for some reason wasn't filling whole column  
    for col in attributes.columns.values.tolist()[:-1]:
        attributes['total'] = attributes['total'].str.cat(space)
        attributes['total'] = attributes['total'].str.cat(attributes[col])
    #add to existing df  
    df['total_text'] = df['posting_body'].str.cat(attributes['total'])
    df.drop(['weight', 'status', 'kids, want', 'general', 'body', 'body art', 'diet', 'dislikes', 'education', 'ethnicity', 'eye color', 'facial hair', 'fears', 'hair', 'stds', 'interests', 'kids, have', 'likes', 'native language', 'occupation', 'personality','pets', 'politics', 'religion', 'resembles', 'posting_body'], axis =1, inplace=True)
    return df 
    
def _calculate_languages_ratios(text): 
    """
    Calculate probability of given text to be written in several languages and
    return a dictionary that looks like {'french': 2, 'spanish': 4, 'english': 0}
    
    @param text: Text whose language want to be detected
    @type text: str
    
    @return: Dictionary with languages and unique stopwords seen in analyzed text
    @rtype: dict
    """

    languages_ratios = {}

    '''
    nltk.wordpunct_tokenize() splits all punctuations into separate tokens
    '''

    tokens = wordpunct_tokenize(text)
    words = [word.lower() for word in tokens]

    # Compute per language included in nltk number of unique stopwords appearing in analyzed text
    for language in stopwords.fileids():
        stopwords_set = set(stopwords.words(language))
        words_set = set(words)
        common_elements = words_set.intersection(stopwords_set)

        languages_ratios[language] = len(common_elements) # language "score"

    return languages_ratios

def detect_language(text):
    """
    Calculate probability of given text to be written in several languages and
    return the highest scored.
    
    It uses a stopwords based approach, counting how many unique stopwords
    are seen in analyzed text.
    
    @param text: Text whose language want to be detected
    @type text: str
    
    @return: Most scored language guessed
    @rtype: str
    """

    ratios = _calculate_languages_ratios(text)
    most_rated_language = max(ratios, key=ratios.get)

    return most_rated_language

def get_english_posts(df):
    df['language'] = df['total_text'].map(detect_language)
    #only look at english posts 
    df = df.ix[df['language']=='english', :]
    return df 

def num_attributes(df):
    #sum of the number of user supplied attributes
    attributes = df[['zodiac', 'weight', 'status', 'kids, want', 'general', 'body', 'body art', 'diet', 'dislikes', 'education', 'ethnicity', 'eye color', 'facial hair', 'fears', 'hair', 'stds', 'interests', 'kids, have', 'likes', 'native language', 'occupation', 'personality','pets', 'politics', 'religion', 'resembles', 'drugs', 'drinks']].notnull()    
    df['num_attributes'] = attributes.apply(sum, axis=1)
    return df 

def drop_stuff(df):
    df.columns = [col.rstrip() for col in df.columns.values.tolist()]
    #drops duplicate urls
    df.drop_duplicates('url', inplace=True)
    #drops anything that's a duplicate posting body and title pair  
    df.drop_duplicates(['posting_body'], inplace=True)
    df.drop_duplicates(['title', 'location'], inplace=True)
    df.drop('_id', axis = 1, inplace= True) 
    df.drop('Unnamed: 0', axis = 1, inplace = True)
    df.drop('ATTR_ :', axis=1, inplace=True)
    df.columns  = [col[:-2] if col.endswith(' :') else col for col in df.columns.values.tolist()] 
    df.columns = [col[5:] if col.startswith('ATTR_') else col for col in df.columns.values.tolist()]
    df.rename(columns={'age:':'age', '':'general'}, inplace=True)
    df.drop('address', axis =1, inplace=True)
    df.drop('kids, ', axis=1, inplace=True)
    df.drop('bod art', axis=1, inplace=True)
    df.drop('like', axis=1, inplace=True)
    return df 

def combine_stds(df):
    df.drop(['/hsv/hpv','hiv/', 'hiv//hpv', 'hiv/hsv/'],axis=1, inplace = True )
    df.rename(columns={'hiv/hsv/hpv':'stds'}, inplace=True)
    return df

def time_str_to_datetime(df): 
    df['post_time'] = df['post_time'].apply(dateutil.parser.parse) 
    #for updated time, need to apply only to non null values 
#    df['post_updated_time'] = df['post_updated_time'].apply(dateutil.parser.parse) 
    df['post_time_of_day'] = df['post_time'].apply(lambda x: x.time())
 #   df['post_updated_time_of_day'] = df['post_updated_time'].apply(lambda x: x.time())
    return df

if __name__=='__main__':
    df = pd.read_pickle('raw_dataframe.pkl')
    pd.set_option('max_colwidth', pd.util.terminal.get_terminal_size()[0])
    df = drop_stuff(df) 
    df = combine_stds(df)
    df = num_attributes(df) 
    df = create_age_groups(df)
    #df = df.ix[df.age.notnull(), :]
    df = create_total_text(df)
    df.drop(['smokes','drinks', 'drugs', 'height', 'area', 'notices', 'post_id', 'repost_of', 'scrape_time'], axis=1, inplace=True)
    #df = get_english_posts(df)
    df = df.set_index(np.arange(df.shape[0]))
    df.to_pickle('dataframe_for_eda.pkl')
