import requests 
from bs4 import BeautifulSoup 
from pymongo import MongoClient 


def cl_categories():
    #returns list of craigslist categories 
    resp = requests.get('https://denver.craigslist.org/')
    soup = BeautifulSoup(resp.content) 
    #grab all posting categories 
    cat = soup.find_all('div', {'class':'cats'})
    forsale_hrefs = cat[4].find_all('a', href=True)
    category_overview = []
    #categories = [forsale_cat['href'] if forsale_cat['href'].startswith('/search')  for forsale_cat in forsale_hrefs]


#all owner/dealer ---> href="/search/sss'
#all owner ---> href="/search/sso'
#all dealer ---> href='/search/ssq'

def scrape_cl(city_href, category=None):
    '''Input: city (str), category (str) 
       Output:list of PIDs 
    ''' 

    #all sections will only display up to last 2500 items but if you query and there aren't 2500 listings it will just return an empty list

    #reposts taken off city site-- but need to compare to what is already stored and remove what is already stored  
    
    item_pids = {}
    for i in range(0,2500,100):
        page_index = i
        url = '{}/search/sss?s={}'.format(city_href, page_index)
        print "Retrieiving PIDs from {}".format(city_href)
        print url

        resp = requests.get(url)
        soup = BeautifulSoup(resp.content)
        #gets all items 
        item_soup = soup.findAll('p', {'class':'row'})
        #gets data-pid's - but also need to check for reposts 
        item_pids[page_index] =[item['data-pid'] for item in item_soup]
    
    return item_pids

def scrape_cl_concurrent(locations): 
    jobs = []
    #Initiate threads
    for location in locations:
        t = threading.Thread(target=scrape_cl, args=(location,))
        jobs.append(t)
        t.start()

    #wait until thread terminates
    results = []
    for t in jobs: 
        t.join()
        results.append(t.result) 
    
    return results 

def item_prices(items):
    prices = [item.find('span','price').text if item.find('span','price') is not None else 0 for item in items]       

    #need to fig
    return prices

def scrape_pids(hrefs):
    #Input: list of posting ids 
    #Output: tbd
    pass 

if __name__ == '__main__': 

    locations = us_scrape() 
    scrape = scrape_cl_concurrent(locations)
