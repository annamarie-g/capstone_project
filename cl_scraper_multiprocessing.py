import requests
import pip 
import json
from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool
from multiprocessing import cpu_count
import itertools 
from collections import defaultdict
from bs4 import BeautifulSoup 
from pymongo import MongoClient 
from pymongo.errors import DuplicateKeyError, CollectionInvalid
import datetime as dt 
from sys import exit   
from functools import partial 
import time 
import os

'''
try: 
    import stem
except: 
    #install Tor controller
    pip.main(['install', 'stem'])
'''

def requests_get_trycatch(url):

    try:
        r = requests.get(url) 
        #not a valid url 
        if r.status_code == 404:    
            return False 
        if r.status_code == 403: 
            reconfigure_ip()
    except: 
        #pause/prompt terminal to continue
        print 'Connection interrupted' 
        cont = raw_input('Enter Y/N to continue: ')
        if cont in ['Y', 'y']:
            r = requests_get_trycatch(url)
        else:
            exit()

    return r 


def reconfigure_ip(): 
    #If interval request = True wait a minute to continue     
    print 'you are blocked' 
    cont = raw_input('Enter Y/N to continue: ')
    if cont in ['Y', 'y']:
       r = requests_get_trycatch(url)
    pass 
        

def load_dicts(location_dict = 'locations.json', category_dict = 'categories.json'): 
    with open(category_dict) as fp:
        categories = json.load(fp)
    with open(location_dict) as fp:
        locations = json.load(fp) 

	location_tuples = [(k,i[0], i[1]) for k,v in locations.iteritems() for i in v]
    category_tuples = [(k,v) for k,v in categories.iteritems()]

    return location_tuples, category_tuples  

def select_region():
    regions = []
    for f in os.listdir(os.getcwd() + '/regions'):
        if f.endswith('.json'):
            regions.append(f[:-5])

    region = ''
    while region not in regions: 
        print 'Regions:' 
        print regions
        region = raw_input('Select Region:')
    return region 

def select_category():
    categories = []
    for f in os.listdir(os.getcwd() + '/categories'):
        if f.endswith('.json'):
            categories.append(f[:-5])

    category  = ''
    while category not in categories:
        print 'Categories:'
        print categories 
        category = raw_input('Select Category:')
    return category

def category_page_urls(location_tuple, category_tuple):
    #Returns list of category page urls for location and category
    location_href = location_tuple[2]
    base_url = '{}/search/{}'.format(location_href, category_tuple[0]) 
    page_urls = ['{}?s={}'.format(base_url, page_index) for page_index in range(2400, -100, -100)]
    return page_urls 

def posting_urls(location_tuple, category_tuple, item_hrefs):
	#Returns list of posting urls given list of item hrefs from category page scrape 
	location_href = location_tuple[2]
	item_urls = [location_href + href for href in item_hrefs]
	return item_urls 

def scrape_category_page(url):
    resp = requests_get_trycatch(url) 
    if not resp: #URL not valid  
        return []
    soup = BeautifulSoup(resp.content)
    items = soup.find_all('a', {'class':'i'}, href=True)
    #if href contains craigslist them it is a redirect to a posting at another location 
    item_hrefs = [item['href'] for item in items if 'craigslist.org' not in item['href']]
    
    return item_hrefs

def scrape_posting((location_tuple, category_tuple, url)): 
    time.sleep(1)
    post_dict = defaultdict()
    resp = requests_get_trycatch(url)
        #check if valid URL 
    if not resp: 
        return post_dict
            
    soup = BeautifulSoup(resp.content)

    repost_index = soup.text.find('repost_of = ')
    if repost_index !=-1:#it is a repost 
        repost_value_index = repost_index + len('repost_of = ')
        post_dict['repost_of'] = soup.text[repost_value_index:repost_value_index+10]

    soup = soup.find('section', {'class':'body'})
    post_dict['url'] = url 
    post_dict['state'] = location_tuple[0]
    post_dict['location'] = location_tuple[1]
    post_dict['category_code'] = category_tuple[0]
    post_dict['category_title'] = category_tuple[1] 

    #Post title 
    title_block  = soup.find('span', {'class':'postingtitletext'})
    if title_block:
        post_dict['title'] = title_block.text 
        
        #if there is a subtitle
        subtitle = title_block.find('small')
        if subtitle:
            post_dict['subtitle'] = subtitle.text
             #if there is a subtitle, remove it from the title and replace title
            post_dict['title'] = post_dict['title'].replace(subtitle.text, '')
        #if price listed 
        price = title_block.find('span', {'class':'price'})
        if price: 
            post_dict['price'] = price.text 
            #if there is a price in title, remove it from the title and replace title 
            post_dict['title'] = post_dict['title'].replace(price.text, '')

    #Number of images 
    num_images = len(soup.find_all('img'))
    if num_images ==1: 
        post_dict['num_images'] = 1
    else: 
        post_dict['num_images'] = num_images - 1  

    #Latitud:e and Longitude 
    map_attributes = soup.find('div', {'id':'map'})
    if map_attributes: 
        post_dict['latitude'] = soup.find('div', {'id':'map'})['data-latitude']
        post_dict['longitude'] = soup.find('div', {'id':'map'})['data-longitude']

	#Post address
	address = soup.find('div', {'class':'mapaddress'})
	if address:
		post_dict['address'] = address.text 
	
	#Other postings by user
	otherpostings = soup.find('span', {'class':'otherpostings'})
	if otherpostings: 
	    post_dict['user_id'] = otherpostings.find('a')['href'][19:]
        	
	#Attributes
    attribute_groups = soup.find_all('p', {'class':'attrgroup'})
    if attribute_groups:
        attributes = [attribute_group.find_all('span') for attribute_group in attribute_groups]
        #flatten
        chain = itertools.chain(*attributes)
        attributes = list(chain)
        for attribute in attributes: 
            #only add attribute to dictionary is there is a value given  
            attribute_value = attribute.find('b')
            #only if attribute value given create attribute name
            if attribute_value: 
                attribute_name = attribute.text.replace(attribute_value.text, '')  
                #add attribute to dictionary 
                post_dict['ATTR_' + attribute_name] = attribute_value.text 

    #Post Text		
    postingbody = soup.find('section', {'id':'postingbody'})
    if postingbody: 
        post_dict['postbody_text'] = postingbody.text
        post_dict['postbody_contents'] = postingbody.encode('utf8')

    #Additional Notices 
    notices = soup.find('ul', {'class':'notices'})
    if notices: 
        notice_list = notices.find_all('li')
        post_dict['notices'] = [notice.text for notice in notice_list]

    #Posting Info 
    postinginfo = soup.find('div', {'class':'postinginfos'})
    if postinginfo: 
        post_dict['post_id'] = postinginfo.find('p', {'class':'postinginfo'}).text.split()[2]
        post_times = postinginfo.find_all('time')
        post_dict['post_time'] = post_times[0]['datetime']
          
        if len(post_times) > 1: #means post was updated
            post_dict['post_updated_time'] = post_times[1]['datetime']

    post_dict['scrape_time'] = dt.datetime.utcnow().isoformat()
    
    table.insert_one(post_dict) 

    #print '{}, {} : {} : {}'.format(post_dict['location'], post_dict['state'], post_dict['category_title'], post_dict.get('title', '').encode('utf8'))

def scrape_category_pages_concurrent(cat_page_urls):
    #creates thread for each of 24 category pages and scrapes item hrefs concurrently
    cat_threadpool = ThreadPool(4)
    results = cat_threadpool.map(scrape_category_page, cat_page_urls)
    cat_threadpool.close()
    cat_threadpool.join() 
    #nested list (each list is 100 hrefs)
    return results 

def scrape_hrefs_concurrent(location_tuple, category_tuple, posting_urls):
    #takes list of posting urls and scrapes each

    #create list of argument tuples 
    args = list(itertools.izip(itertools.repeat(location_tuple), itertools.repeat(category_tuple),posting_urls))
    post_threadpool = ThreadPool(4) 
    results = post_threadpool.map(scrape_posting,args)  
    post_threadpool.close()
    post_threadpool.join() 
    #scrape_posting inserts into mongo, no need to return 


def scrape_concurrent(location_tuples, category_tuples):
    #Scrapes all postings for each location/category pair
	#threadpool for scraping 24 category pages 
	#threadpool for scraping 100 hrefs from category pages

    for location_tuple in location_tuples:
	print location_tuple 
	for category_tuple in category_tuples:
	    print category_tuple 
	    cat_page_urls = category_page_urls(location_tuple, category_tuple)
	    #scrape_category_pages_concurrent gives a nested list [[hrefs pg 1], [hrefs pg 5], [hrefs pg 12], etc] 
	    cat_item_hrefs = scrape_category_pages_concurrent(cat_page_urls)
	    for page_of_hrefs in cat_item_hrefs:
            post_urls = posting_urls(location_tuple, category_tuple, page_of_hrfs)
		    #creates threadpool for page of posts and scrapes posting
	    	scrape_hrefs_concurrent(location_tuple, category_tuple, post_urls) 
		    print 'number of postings: ' + str(table.count())
	
if __name__=='__main__':
	
    #prompt user to select region
    region = select_region()
    category = select_category() 

    #Define the MongoDB database and table 
    db_client = MongoClient()
    db_name = 'cl_scrape' 
    db = db_client[db_name]
    table_name = '_'.join([region, category, dt.date.today().isoformat()[-5:]]) 
    table = db[table_name]

    region_dict_fp = 'regions/' + region + '.json' 
    category_dict_fp = 'categories/' + category + '.json' 
    location_tuples, category_tuples = load_dicts(location_dict = region_dict_fp, category_dict = category_dict_fp)     
    scrape_concurrent(location_tuples, category_tuples) 	

    #export table to mongo after scrape
    output_fp = 'data/loc_cat_scrape/table_name' + '.json'
    os.system('mongoexport --db {} --collection {} --jsonArray --out {}'.format(db_name, table_name, output_fp))
