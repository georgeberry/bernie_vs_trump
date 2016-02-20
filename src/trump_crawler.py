from bs4 import BeautifulSoup
from urllib2 import URLError
from time import sleep
import csv

# This code is necessary to pull from Javascript
from selenium import webdriver
driver = webdriver.PhantomJS()

# Website urls
base_url_trump = "https://www.donaldjtrump.com/press-releases/P"
base_url_bernie = "https://berniesanders.com/press-release/page/"

# Constants
# how are these numbers determined?
NUMBER_OF_PAGES_TRUMP = 90
NUMBER_OF_PAGES_BERNIE = 34
MIN_TRUMP_URL_LEN = 50

# Functions

def clean_text(text):
    """
    Give this paragraph text before joining to one string statement
    Cleans up the unicode and newline junk


    """
    pass

# this is a set since we're doing existence tests
press_release_url_set = set()

# Main Body

for i in range(1, NUMBER_OF_PAGES_TRUMP + 1):
    press_release_urls = []
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
            # Question: why sleep here?
            sleep(1) #limit calls to 1 per second
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
            press_release = "".join(trimmed_paragraphs)
            print(press_release)

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
