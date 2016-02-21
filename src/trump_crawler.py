#from urllib2 import urlopen
from bs4 import BeautifulSoup
from urllib2 import URLError
from time import sleep
import csv

#This code is necessary to pull from Javascrip
from selenium import webdriver
driver = webdriver.PhantomJS()

#Website urls
trump = "https://www.donaldjtrump.com/press-releases/P"

number_of_pages_trump = 90

min_trump_len_url = 50

i=1 #index value set to one (so begins on first page)
all_pr_urls = []

while i <= number_of_pages_trump:
    press_release_urls = []
    url = trump + str(i) #Concatenate url with index value
    driver.get(url)  #Get the webpage
    soup = BeautifulSoup(driver.page_source) #Convert it to a BS object - "soup"
    #page = urlopen(url).read()
    #soup = BeautifulSoup(page)
    
    for link in soup.findAll('a', href=True): #finds all hyperlinks
        L = link['href'] #gets the link string as L
        #print url
        if "press-release" in L and len(L) > min_trump_len_url:
            #then we have a valid press release link!
            #so we append it to our list
            press_release_urls.append(L)
    for pr in press_release_urls:
        if pr not in all_pr_urls:
            sleep(1) #limit calls to 1 per second
            all_pr_urls.append(pr)
            driver.get(pr)
            soup = BeautifulSoup(driver.page_source)
            content = soup.find_all('p')
            print len(content), "START OF NEW PRESS RELEASE"
            Paragraphs = []
            for c in content:
                text = c.getText()
                Paragraphs.append(text)   
            Paragraphs.pop(0) #Removes first element
            
            Fragments = []
            for p in Paragraphs:
                p_frag = p[:12]
                Fragments.append(p_frag)
            #print Fragments
            end = Fragments.index("Next Release")
            
            Paragraphs = Paragraphs[:end]
            print Paragraphs
            print pr
            #Find text "Next release" or "paid for"
            #for paragraph in text:
            #    print paragraph.getText(), "END OF PARAGRAPH"
                
    i+=1
    
 