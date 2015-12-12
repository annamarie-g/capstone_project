import requests
from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool
from multiprocessing import cpu_count
import itertools 


def test_threadpool():
    pool = ThreadPool(4)	
    results = pool.map(do_something, urls)
    pool.close()
    pool.join()

def do_something(url):
	return requests.get(url)


if __name__=='__main__':
    urls = ['', '', '', '', '', '', '', ''] 
    test_threadpool(urls) 
