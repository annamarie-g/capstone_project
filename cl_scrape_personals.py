import requests 
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


def requests_get_trycatch(url):
    """returns requests.get(url) if valid page"""
    try:
	r = requests.get(url) 
	if r.status_code == 404:    
    #not a valid url
	    print 'not a valid url: {}'.format(url)
	elif r.status_code == 403: 
	    print 'You are blocked on this IP'
	    exit()
    except:  
        exit()
    return r 

def write_remaining_to_dict(locations):
    """creates json of locations that have not been scraped. Can restart scraper
       overwrites existing file
    """   
    with open('regions/remaining_{}.json'.format(table_name), 'wb') as fp:
        json.dump(locations, fp) 

def load_dicts(location_dict, category_dict): 
    with open(category_dict) as fp:
        categories = json.load(fp)
    with open(location_dict) as fp:
        locations = json.load(fp) 

	location_tuples = [(k,i[0], i[1]) for k,v in locations.iteritems() for i in v]
    category_tuples = [(k,v) for k,v in categories.iteritems()]

    return locations, location_tuples, category_tuples  

def select_region():
    """prompts user to select region from regions/ directory
       returns region name (str) selected by user.
    """
    regions = []
    for f in os.listdir(os.getcwd() + '/regions'):
        if f.endswith('.json'):
            regions.append(f[:-5])
   
   #prompt user
    region = ''
    while region not in regions: 
        print 'Regions:' 
        print regions
        region = raw_input('Select Region:')
    return region 

def select_category():
    """prompts user to select category from categories/personals/ directory
       returns category name (str) selected by user.
    """
    categories = []
    for f in os.listdir(os.getcwd() + '/categories/personals/'):
        if f.endswith('.json'):
            categories.append(f[:-5])
    
    #prompt user
    category  = ''
    while category not in categories:
        print 'Categories:'
        print categories 
        category = raw_input('Select Category:')
    return category

def category_page_urls(location_tuple, category_tuple):
    """Returns list of category page urls for location and category"""
    location_href = location_tuple[2]
    base_url = '{}/search/{}'.format(location_href, category_tuple[0]) 

    #get total page count 
    resp = requests_get_trycatch(base_url)
    soup = BeautifulSoup(resp.content)
    total_posted = soup.find('span', {'class':'totalcount'})
    if total_posted: 
    	total = int(soup.find('span', {'class':'totalcount'}).text)
    	#calculate upper bound of postings
    	upperbnd = total - (total%100)
    else: 
	#upperbnd == 2400 means there are no postings
        return []

    page_urls = ['{}?s={}'.format(base_url, page_index) for page_index in range(upperbnd, -100, -100)]
    return page_urls 

def posting_urls(location_tuple, category_tuple, item_hrefs):
	"""Returns list of posting urls given list of item hrefs from category page scrape"""
    location_href = location_tuple[2]
    item_urls = [location_href + href for href in item_hrefs if ('craigslist.ca' not in href) and ('craigslist.mx' not in href)] 
    return item_urls 

def scrape_category_page(url):
    """Scrapes category page given url. Returns list of item hrefs """
    resp = requests_get_trycatch(url) 
    if not resp: #r is false,    
        return []
    soup = BeautifulSoup(resp.content)
    items = soup.find_all('a', {'class':'i'}, href=True)
    #if href contains craigslist them it is a redirect to a posting at another location 
    item_hrefs = [item['href'] for item in items if 'craigslist.' not in item['href']]
    
    return item_hrefs

def scrape_personals_posting((location_tuple, category_tuple, url)):
    """Scrapes personals posting given location tuple, category tuple, and url.
       Stores scraped information to dictionary. 
       Inserts dictionary to globally defined mongodb table
    """
    #slight pause 
    time.sleep(1)
    post_dict = defaultdict() 
    resp = requests_get_trycatch(url)
        #check if valid URL 
    if not resp: 
        return post_dict

    soup = BeautifulSoup(resp.text)

    #posting was removed
    if soup.find('div', {'class':'removed'}):
        return []

    #record if repost
    repost_index = soup.text.find('repost_of = ')
    if repost_index !=-1: #repost 
        repost_value_index = repost_index + len('repost_of = ')
        post_dict['repost_of'] = soup.text[repost_value_index:repost_value_index+10]

    #Posting Body 
    post_dict['posting_body'] = soup.find('section', {'id':'postingbody'}).text 
    #check if post was deleted
    if ('This posting has been deleted by its author.' or 'This posting has been flagged for removal.') in post_dict['posting_body']:
        return post_dict 

    soup = soup.find('section', {'class':'body'})
    post_dict['url'] = url 
    post_dict['state'] = location_tuple[0]
    post_dict['location'] = location_tuple[1]
    post_dict['category_code'] = category_tuple[0]
    post_dict['category_title'] = category_tuple[1] 

    #Title
    title_block  = soup.find('span', {'class':'postingtitletext'})
    if title_block:
	post_dict['title'] = title_block.text 	
    
    	subtitle = title_block.find('small')
    	if subtitle:
            post_dict['area'] = subtitle.text
	    post_dict['title'] = post_dict['title'].replace(subtitle.text, '')
	    
    #Number of images 
    num_images = len(soup.find_all('img'))
    if num_images == 0: 
        post_dict['num_images'] = 0
    elif num_images ==1: 
        post_dict['num_images'] = 1
    else: 
        post_dict['num_images'] = num_images - 1  

	#Post address
	address = soup.find('div', {'class':'mapaddress'})
	if address:
		post_dict['address'] = address.text 
        	
	#Attributes
    attribute_groups = soup.find_all('p', {'class':'attrgroup'})
    if attribute_groups:
        attributes = [attribute_group.find_all('span') for attribute_group in attribute_groups]
        #flatten
        chain = itertools.chain(*attributes)
        attributes = list(chain)
        for attribute in attributes: 
            attribute_value = attribute.find('b')
            #only if attribute value given create attribute name
            if attribute_value: 
                attribute_name = attribute.text.replace(attribute_value.text, '')  
                #add attribute to dictionary 
                post_dict['ATTR_' + attribute_name] = attribute_value.text 

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


def scrape_category_pages_concurrent(cat_page_urls):
    """Creates thread for each of 24 category pages and scrapes item hrefs concurrently.
       Returns nested list of hrefs. 
    """
    cat_threadpool = ThreadPool(4)
    results = cat_threadpool.map(scrape_category_page, cat_page_urls)
    cat_threadpool.close()
    cat_threadpool.join() 
    #each list is 100 hrefs
    return results 

def scrape_hrefs_concurrent(location_tuple, category_tuple, posting_urls):
    """Takes list of posting urls and scrapes each concurrently"""
    args = list(itertools.izip(itertools.repeat(location_tuple), itertools.repeat(category_tuple),posting_urls))
    post_threadpool = ThreadPool(4) 
    #scrape_personals_posting inserts to mongodb table
    results = post_threadpool.map(scrape_personals_posting,args)  
    post_threadpool.close()
    post_threadpool.join() 

def scrape_concurrent(locations, location_tuples, category_tuples):
    """Scrapes all postings for each location/category pair.
       Calls scrape_category_pages_concurrent to scrape hrefs from category pages. 
       Removes location tuple as location is scraped
   """

    for location_tuple in location_tuples:
        print location_tuple 
        
        for category_tuple in category_tuples:
            print category_tuple 
            cat_page_urls = category_page_urls(location_tuple, category_tuple)
            #scrape_category_pages_concurrent gives a nested list [[hrefs pg 1], [hrefs pg 5], [hrefs pg 12], etc] 
            cat_item_hrefs = scrape_category_pages_concurrent(cat_page_urls)
            for page_of_hrefs in cat_item_hrefs:
                post_urls = posting_urls(location_tuple, category_tuple, page_of_hrefs)
                #creates threadpool for page of posts and scrapes posting
                scrape_hrefs_concurrent(location_tuple, category_tuple, post_urls) 
            print 'number of postings: ' + str(table.count())

        #remove location_tuple from list if finished scraping location 
        locations[location_tuple[0]].remove([location_tuple[1], location_tuple[2]])
    	write_remaining_to_dict(locations)	

def export_table():
    """Called when scrape is finished, or when scrape gets interrupted 
       export table to mongo after finished 
    """
    output_fp = 'data/loc_cat_scrape/{}'.format(table_name) + '.json'
    os.system('mongoexport --db {} --collection {} --jsonArray --out {}'.format(db_name, table_name, output_fp))

if __name__=='__main__':
	
    #prompt user to select region
    region = select_region()
    category = select_category() 

    #Define the MongoDB database and table 
    db_client = MongoClient()
    db_name = 'cl_scrape' 
    db = db_client[db_name]
    global table_name #global so that mongo can export in function, this table name will never change during execution of script 
    table_name = ''.join([region.replace('_', '').replace(category,''), category.replace('_',  ''), dt.date.today().isoformat()[-5:].replace('-', '')]) 
    table = db[table_name]
	
    region_dict_fp = 'regions/' + region + '.json' 
    category_dict_fp = 'categories/personals/' + category + '.json' 
    locations, location_tuples, category_tuples = load_dicts(location_dict = region_dict_fp, category_dict = category_dict_fp)     
    
    scrape_concurrent(locations, location_tuples, category_tuples) 	
    os.remove('regions/remaining_{}.json'.format(table_name))
    os.remove('urls_visited.txt')
