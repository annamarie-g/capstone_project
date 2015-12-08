import requests 
import threading as th
from multiprocessing import Pool 
from multiprocessing.dummy import Pool as ThreadPool 
from multiprocessing import cpu_count
import itertools 
from bs4 import BeautifulSoup 
from pymongo import MongoClient 
import pip 
import shelve
import copy 
import Queue
import json

try:    
    import stem 
except: 
    #installTor controller 
    pip.main(['install', 'stem'])

'''
WORKFLOW:
scrape_locations_concurrent > use multiprocessing to scrape locations
    scrape_categories > scrape categories iteratively
        scrape_category >  
            < return item_hrefs list for category page 
        scrape_hrefs_parallel -> scrape hrefs from page 
            >scrape_posting 
                >export to mongo 
'''
'''
def access_dicts(): 
    #import pdb; pdb.set_trace()
    locations_serialized = shelve.open('locations.db', flag='r')
    categories_serialized = shelve.open('categories.db', flag='r') 
    locations = copy.deepcopy(dict(locations_serialized))
    categories = copy.deepcopy(dict(categories_serialized))
    #locations_serialized.close()
    #categories_serialized.close() 
    return locations, categories 
'''

def access_dicts(): 
    with open('categories.json') as fp:
        categories = json.load(fp)
    with open('locations.json') as fp:
        locations = json.load(fp) 
    return locations, categories 

def scrape_locations_concurrent(locations, categories): 
    pool  = Pool(cpu_count())
    location_tuples = [(k,i) for k,v in locations.iteritems() for i in v]
    pool.map(scrape_categories, itertools.izip(location_tuples, itertools.repeat(categories)))
    pool.close()
    pool.join()

def scrape_categories(location_tuple, categories):
    category_tuples = [(k,v) for k,v in categories.iteritems()]
    for category_tuple in category_tuples:
        scrape_category(location_tuple, category_tuple) 
    
def scrape_category(location_tuple, category_tuple): 
    location_href = location_tuple[1][1]
    base_url = '{}/search/{}'.format(location_href, category_tuple[0]) 
    #Build list of urls 
    urls = ['{}?s={}'.format(base_url, page_index) for page_index in range(2400, -100, -100)]
    for url in urls: 
        item_hrefs = scrape_category_page(url)
        scrape_hrefs_concurrent(location_tuple, category_tuple, item_hrefs) 

def scrape_hrefs_concurrent(location_tuple, category_tuple, item_hrefs): 
    location_href = location_tuple[1][1] 
    item_urls = [location_href + href for href in item_hrefs]
    #Build worker pool 
    thread_pool = ThreadPool(cpu_count())
    thread_pool.map(scrape_posting, itertools.izip(itertools.repeat(location_tupe),itertools.repeat(category_tuple), item_urls))
        
def scrape_posting(location_tuple, category_tuple, url): 
    resp = requests.get(url) 
    soup = BeautifulSoup(resp.content).find('section', {'class':'body'})

    post_dict = {}

    post_dict['state'] = location_tuple[0]
    post_dict['location'] = location_tuple[1][0]
    post_dict['category_code'] = category_tuple[0]
    post_dict['category_title'] = category_tuple[1] 
    post_dict['title'] = soup.find('span', {'class':'postingtitletext'}).text 
    post_dict['neighborhood'] = soup.find('span', {'class':'postingtitletext'}).findNext().text 
    post_dict['num_thumbnails'] = len(soup.find('div', {'id':'thumbs'}).find_all('a'))
    post_dict['latitude'] = float(soup.find('div', {'id':'map'})['data-latitude'])
    post_dict['longitude'] = float(soup.find('div', {'id':'map'})['data-longitude'])
    post_dict['user_id'] = soup.find('span', {'class':'otherpostings'}).find('a')['href'][19:]
    post_dict['post_body'] = soup.find('section', {'id':'postingbody'}).text
    post_dict['notices'] = [notice.text for notice in soup.find('ul', {'class':'notices'}).find_all('li')]
   
'''
def category_scrape_sequential(location, category): 
    queue = Queue.Queue() 
    location_href = location[1]
    item_hrefs_category = []
    base_url = '{}/search/{}'.format(location_href, category) 
    for page_index in range(2400, -100, -100): 
        url = '{}?s={}'.format(base_url, page_index) 
        scrape_category_page(url, queue)
    while not queue.empty():
        item_hrefs_category.extend(queue.get())
''' 

def threadpool_category_scrape_concurrent(location, category): 
    '''collects all item links in given location and category
    '''
    item_hrefs_category = []
    location_href = location[1]
    base_url = '{}/search/{}'.format(location_href, category) 
    #Build list of urls 
    urls = ['{}?s={}'.format(base_url, page_index) for page_index in range(2400, -100, -100)]
    #Build worker pool 
    thread_pool = ThreadPool(cpu_count())
    items_hrefs_category = thread_pool.map(scrape_category_page, urls) 
    return item_hrefs_category

def scrape_category_page(url):
    resp = requests.get(url) 
    soup = BeautifulSoup(resp.content)
    items = soup.find_all('a', {'class':'i'}, href=True)
    item_hrefs_page  = [item['href'] for item in items]
    return item_hrefs_page

def export_to_mongo(): 
    pass 

if __name__=='__main__':

   
    #locations = {u'State': [(u'location name', u'location url')...]
    #categories = {'xxx': u'category name - by owner(dealer)'}}
    import pdb; pdb.set_trace()
    locations, categories = access_dicts()
    threadpool_category_scrape_concurrent(locations.values()[10][0], categories.keys()[0])
    
