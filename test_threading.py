import requests
from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool
from multiprocessing import cpu_count
import itertools 
  
url = 'https://resume.creddle.io/'

def test_threadpool():

	pool = ThreadPool(4)	


def do_something(url):
	return requests.get(url)
