import requests 
from bs4 import BeautifulSoup 
#import shelve
import json 
import os 

'''
gillan.anna@gmail.com
Script scrapes US craiglist locations, save to file 
Also scrapes categories and saves to file 
'''

def us_scrape(): 
    #Saves to file list of string hrefs associated with US locations  
    resp = requests.get('https://www.craigslist.org/about/sites')
    soup = BeautifulSoup(resp.content)
    locations = {}
    states = soup.h1.findNextSibling().find_all('h4')
    for state in states: 
        location_block = state.findNextSibling().find_all('a', href=True) 
        locations[state.text] = [(loc.text, loc['href']) for loc in location_block]

    with open('locations.json', 'wb') as fp:
        json.dump(locations, fp) 
    
''''        
    #delete file if already exists
    try: 
        os.remove('locations.db')
    except OSError: 
        pass 

    #open file for locations dictionary
    shelf = shelve.open('locations')
    #serializing 
    shelf.update(locations) 
    shelf.close() 
'''

def categories_dict():
    '''
    Saves to file list of craigslist categories 
    categories = {3 letter key: category name} 
    '''

    #first scrape for sale by owner categories 
    categories = categories_scrape('sso', u' - by owner')
    
    #then scrape for sale by dealer categories 
    categories.update(categories_scrape('ssq', u' - by dealer'))

    with open('categories.json', 'wb') as fp: 
        json.dump(categories, fp)

'''
    #delete file if already exists 
    try: 
        os.remove('categories.db')
    except OSError: 
        pass 
    
    #open file for categories dictionary
    shelf = shelve.open('categories')
    #serializing 
    shelf.update(categories)
    shelf.close() 
'''
    
def categories_scrape(search_key, u_sellertype):
    #first scrape for sale by owner 
    categories = {} 
    url = 'https://denver.craigslist.org/search/{}'.format(search_key)
    resp = requests.get(url)
    soup = BeautifulSoup(resp.content)    
    cat_by_owner = soup.find_all('a', {'class':'category'}, href=True) 
    for cat in cat_by_owner: 
        three_letter_key = cat['href'][-3:]
        categories[three_letter_key] = cat.text + u_sellertype
    return categories

if __name__=='__main__':
    us_scrape() 
    categories_dict()

