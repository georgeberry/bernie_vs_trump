#from urllib2 import urlopen
from bs4 import BeautifulSoup
from urllib2 import URLError
from time import sleep
import csv

#This code is necessary to pull from Javascrip
from selenium import webdriver
driver = webdriver.PhantomJS()

#Website urls
bernie = "https://berniesanders.com/press-release/page/"

number_of_pages_bernie = 34

min_bernie_len_url = 47

i=1 #index value set to one (so begins on first page)

all_pr_urls = []

for i in range(1, number_of_pages_bernie+1):
    press_release_urls = []
    url = bernie + str(i) #Concatenate url with index value
    driver.get(url)  #Get the webpage
    soup = BeautifulSoup(driver.page_source) #Convert it to a BS object - "soup"
    #page = urlopen(url).read()
    #soup = BeautifulSoup(page)
    
    for link in soup.findAll('a', href=True): #finds all hyperlinks
        L = link['href'] #gets the link string as L
        if "https://berniesanders.com/press-release/" in L and "mailto" not in L and "facebook.com" not in L and "twitter.com" not in L and len(L) > min_bernie_len_url:
            #then we have a valid press release link!
            #so we append it to our list
            print "SECOND", L
            #press_release_urls.append(L)
    i+=1