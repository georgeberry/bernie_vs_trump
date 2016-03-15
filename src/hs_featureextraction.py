import string
import nltk
import sklearn
import re
import json
import pandas
from textstat.textstat import textstat
from vaderSentiment.vaderSentiment import sentiment as VS

"""This program extracts features from the labelled hate speech tweets
    and returns a pandas dataframe."""

##Defining stopwords list
stopwords = nltk.corpus.stopwords.words("english")
#Adding unique stopwords for Twitter
other_exclusions = ["#FF", "#ff"] #note the exclusions vary depending if we stem
stopwords.extend(other_exclusions)

stemmer = nltk.stem.porter.PorterStemmer()
tokenizer = nltk.tokenize.TweetTokenizer(reduce_len=True)

def tokenize(tweet):
    tokens = tokenizer.tokenize(tweet)
    #Remove stopwords
    tokens = [t for t in tokens if t not in stopwords]
    #stem tokens
    stems = [stemmer.stem(t) for t in tokens]
    return stems

def other_features(tweet):

    ##SENTIMENT
    sentiment = VS(tweet)

    ##READABILITY

    #See https://pypi.python.org/pypi/textstat/
    flesch = round(textstat.flesch_reading_ease(tweet),3)
    flesch_kincaid = round(textstat.flesch_kincaid_grade(tweet),3)
    gunning_fog = round(textstat.gunning_fog(tweet),3)
    ##TEXT-BASED
    length = len(tweet)
    num_terms = len(tweet.split())
    ##TWITTER SPECIFIC TEXT FEATURES
    hashtag_count = tweet.count("#")
    mention_count = tweet.count("@")
    url_count = tweet.count("http")
    retweet = 0
    if tweet.lower().startswith("rt") is True:
        retweet = 1
    #Checking if RT is in the tweet
    words = tweet.lower().split()
    if "rt" in words or "#rt" in words:
        retweet = 1
    features = [sentiment['compound'],flesch, flesch_kincaid,
                gunning_fog, length, num_terms,
                hashtag_count, mention_count,
                url_count, retweet]
    return features


def hate_dict_features(hate_words, tweet):
    """This function creates a feature vector based on the terms
    in the hate words lexicon

    This function takes a tweet and checks for matching terms in
    the hate words list. It assigns a vector to the tweet
    of the length of list, where nth entry == 1 iff nth term
    is present in the tweet, else 0."""

    ##Amend this to make format a data frame that can be extended
    df = []
    tweet = tweet.lower()
    for term in hate_words:
        #if term is in tweet + a space (to prevent over counting)
        if term.lower()+' ' in tweet or ' '+term.lower() in tweet:
            df.append(1)
        #if term is not in tweet we assign a 0
        else:
            df.append(0)
    return df

if __name__ == '__main__':
    D = pickle.load(open("hate_dict.p", 'rb'))
    tweets = D.keys()
    hate_words = []
    file = 'd_cf_dict.csv'
    f = open(file, 'rU')
    reader = csv.reader(f)
    for row in reader:
        hate_words.append(row[0])
