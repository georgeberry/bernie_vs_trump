
from bs4 import BeautifulSoup
from urllib2 import URLError
from time import sleep
import csv
import json

# This code is necessary to pull from Javascript
from selenium import webdriver
driver = webdriver.PhantomJS()

# Website urls
base_url_trump = "https://www.donaldjtrump.com/press-releases/P"

# Constants
#Number of pages of press releases
NUMBER_OF_PAGES_TRUMP = 90

#Min length of a valid press release url

MIN_TRUMP_URL_LEN = 50

#Where we save the data output
OUTPUT_PATH = '../data/press_releases.json'

#This set will contain all visited press release urls
press_release_url_set = set()

# Main Body

"""
The intuition behind this is simple:
    1) We get the press release urls from each page of press releases
    2) We then go through the press release urls and get press release text
    3) We append them to a newline-delimited json file
"""

for i in range(1, NUMBER_OF_PAGES_TRUMP + 1):

    press_release_urls = [] #empty list to store urls
    url = base_url_trump + str(i) # Concatenate url with index value
    driver.get(url)  # Get the webpage
    # Convert it to a BS object - "soup"
    soup = BeautifulSoup(driver.page_source)

    # iterate through links, store them
    for link in soup.findAll('a', href=True):
        candidate_link = link['href']
        # two simple criteria for determining if this is a press release url
        if "press-release" in candidate_link:
            if len(candidate_link) > MIN_TRUMP_URL_LEN:
                press_release_urls.append(candidate_link)
    for pr_url in press_release_urls:
        if pr_url not in press_release_url_set:

            sleep(1) #limit calls to 2 per second
            press_release_url_set.add(pr_url)
            driver.get(pr_url)
            soup = BeautifulSoup(driver.page_source)
            content = soup.find_all('p')
            #print([x.getText() for x in content][-5:])
            print (
                "START OF NEW PRESS RELEASE WITH LENGTH {}!".format(len(content))
            )
            paragraphs = []
            for c in content:
                c_text = c.getText()
                paragraphs.append(c_text)

            # we don't need the first or last 5 elements
            # so we slice them out
            trimmed_paragraphs = paragraphs[1:-5]
            press_release_text = "".join(trimmed_paragraphs)

            press_release_dict = {
                "text": press_release_text,
                "url": pr_url,
                "author": "Trump",
            }

            with open(OUTPUT_PATH, 'a') as f:
                #turns dict into valid json string on 1 line
                j = json.dumps(press_release_dict) + '\n'
                #writes j to file f
                f.write(j)

            # alternate way of doing this
            # search for the element that begins with "Next Release"
            # and drop that one and everything after it
            #fragments = []
            #for p in paragraphs:
            #    p_frag = p[:12]
            #    fragments.append(p_frag)
            #print Fragments
            #end = fragments.index("Next Release")
            #paragraphs = paragraphs[:end]
            #print paragraphs
            #print pr
            #Find text "Next release" or "paid for"
            #for paragraph in text:
            #    print paragraph.getText(), "END OF PARAGRAPH"