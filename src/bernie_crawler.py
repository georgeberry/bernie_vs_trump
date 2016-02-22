from bs4 import BeautifulSoup
import time
import json

# This code is necessary to pull from Javascript
from selenium import webdriver
driver = webdriver.PhantomJS()

# Website url
base_url_bernie = "https://berniesanders.com/press-release/page/"

# Constants
# Number of pages of press releases on site
NUMBER_OF_PAGES_BERNIE = 34
# Min length for url for an individual press release
MIN_BERNIE_LEN_URL = 47

# This is where our output data will go
OUTPUT_PATH = '../data/press_releases.json'

# This set will contain all visited urls
press_release_url_set = set()

# Main Body

"""
The intuition behind this is simple:
    1) We get the press release urls from each page of press releases
    2) We then go through the press release urls and get press release text
    3) We append them to a newline-delimited json file
"""

# CRAWLING THROUGH WEBPAGES
for i in range(1, NUMBER_OF_PAGES_BERNIE+1):
    press_release_urls = [] #empty list to contain urls
    
    # READING WEBPAGES
    url = base_url_bernie + str(i) #Concatenate url with index value
    driver.get(url)  #Get the webpage
    soup = BeautifulSoup(driver.page_source) #Convert it to a BS object - "soup"
    
    # FINDING INDIVIDUAL PRESS RELEASES
    for link in soup.findAll('a', href=True): #finds html objects containing hyperlinks
        candidate_link = link['href'] #gets the link string as L
        
        # if the link is press-relase, does not contain the other strings in lines 50/51,
        # as exceeds the minimum length, then include it
        if "https://berniesanders.com/press-release/" in candidate_link:
            if "mailto" not in candidate_link:
                if "facebook.com" not in candidate_link and "twitter.com" not in candidate_link:
                    if len(candidate_link) > MIN_BERNIE_LEN_URL:
            #so we append it to our list
                        press_release_urls.append(candidate_link)
    
    # PROCESSING PRESS RELASES                 
    for pr_url in press_release_urls:
        # if it is not in the set of visited links
        if pr_url not in press_release_url_set:
            # add it to the set and visit it
            press_release_url_set.add(pr_url)
            time.sleep(1) #limit calls to 1 per second
            driver.get(pr_url)
            soup = BeautifulSoup(driver.page_source)
            content = soup.find_all('p')
            # print([x.getText() for x in content][-5:])
            print (
                "START OF NEW PRESS RELEASE WITH LENGTH {}!".format(len(content))
            )
            paragraphs = []
            for c in content:
                c_text = c.getText()
                paragraphs.append(c_text)
            # we don't need the last 3 elements
            # so we slice them out
            trimmed_paragraphs = paragraphs[:-3]
            # we join them back together into a string
            press_release_text = "".join(trimmed_paragraphs) #trimmed_

            # print press_release_text
            
            # CREATING DICTIONARY
            press_release_dict = {
                "text": press_release_text,
                "url": pr_url,
                "author": "Bernie",
            }
            
            # WRITING THE DICTIONARY TO JSON FILE
            with open(OUTPUT_PATH, 'a') as f:
                #turns dict into valid json string on 1 line
                j = json.dumps(press_release_dict) + '\n'
                #writes j to file f
                f.write(j)
      
    i+=1 # increment index by 1 