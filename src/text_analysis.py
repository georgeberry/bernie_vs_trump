import nltk
import string
import sklearn
import json
import re
from sklearn.feature_extraction.text import TfidfVectorizer

"""
This program analyzes the texts we collected using the bag-of-words (BoW) approach.

We treat each set of texts (e.g. Tweets made by Trump) as a BoW by combining
them all together. We then compare these BoW to assess the similarity between each set.
    (1) We use NLTK and string methods to clean up the data and simplify
    the strings to make a BoW.
    (2) After constructing each BoW we use scikit-learn to represent the BoW as a numerical vector using TF-IDF weighting.
    (3) We compute the cosine similarity scores of each vector to assess their similarity.
"""

##RESOURCES USED:
##GENERAL INFO: http://stackoverflow.com/questions/8897593/similarity-between-two-text-documents
##ON COSINE SIMILARIY: http://nlp.stanford.edu/IR-book/html/htmledition/dot-products-1.html,
##http://stackoverflow.com/questions/12118720/python-tf-idf-cosine-to-find-document-similarity
##TF-IDF WEIGHTS: http://nlp.stanford.edu/IR-book/html/htmledition/tf-idf-weighting-1.html


#Defining stopwords
stopwords = nltk.corpus.stopwords.words("english")
other_exclusions = ["ff", "rt"]
stopwords.extend(other_exclusions)

stemmer = nltk.stem.porter.PorterStemmer()

def Preprocess(S):
    """
    Takes a string, converts it to lowercase and removes punctuation.
        Returns a the edited string
    """
    #Coverts uppercase to lower
    giant_url_regex = ('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|'
        '[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    S = S.lower()
    S = re.sub(r'\s+', ' ', S)
    S = re.sub(giant_url_regex, '', S)
    #Removes all punctuation
    punct = set(string.punctuation)
    S = ''.join(ch for ch in S if ch not in punct)
    return S

"""
Takes a string, tokenizes the string and stems the tokens.
    Returns a list of stems
"""
def Tokenize(text):
    tokens = nltk.word_tokenize(text)
    stems = [stemmer.stem(t) for t in tokens]
    return stems

"""
Converts each string into a TF-IDF weighted numerical vector.
    Returns a data frame containing vectors as columns.
"""
def Vectorize(*texts):
    vectorizer = TfidfVectorizer(tokenizer=Tokenize, stop_words=stopwords)
    tfidf = vectorizer.fit_transform(texts)
    return tfidf

def freq_count(text):
    count_dict = {}
    words = text.split()
    for word in words:
        if word not in count_dict:
            count_dict[word] = 0
        count_dict[word] += 1
    return sorted(count_dict.items(), reverse=True, key=lambda x: x[1])


if __name__ == '__main__':
    ##INITIALIZING BAG OF WORDS OBJECTS AS EMPTY STRINGS
    trump_BOW = []
    bernie_BOW = []
    trump_tweets_BOW = []
    bernie_tweets_BOW = []
    trumpf_tweets_BOW = []
    bernief_tweets_BOW = []

    ##READING IN PRESS RELEASE FILE
    with open("../data/press_releases.json", 'rb') as f:
        for line in f:
            press_release = json.loads(line)
            author = press_release['author']
            text = press_release['text']
            if author == "Trump":
                trump_BOW.append(text)
            else:
                bernie_BOW.append(text)


    ##Code below reads tweet file but file too larger
    ##Instead we ran create_slim_dataset.py to extract only the
    ##Relevant information from the json file and created a new file
    ##that is only 6% of the size

    ##READING IN TWEET FILE
    #    with open("../data/tweets.json", 'rb') as f:
    #        for line in f:
    #            tweet = json.loads(line)
    #            #list_tweets.append(j)
    #            if tweet['luminary_followed'] == None:
    #            #Tweet is bernie or trump
    #                if tweet['user']['screen_name'] == "BernieSanders":
    #                    #tweet is by Bernie
    #                    bernie_tweets_BOW += " " + tweet['text']
    #                else:
    #                    #Tweet is by trume
    #                    trump_tweets_BOW += " " + tweet['text']
    #            #Tweet belongs to a follower
    #            elif tweet['luminary_followed'] == "realDonaldTrump":
    #                trumpf_tweets_BOW += " " + tweet['text']
    #                #Tweet is by trump follower
    #            elif tweet['luminary_followed'] == "BernieSanders":
    #                #Tweet is by bernie follower
    #                bernief_tweets_BOW += " " + tweet['text']

    ##READING IN TWEET FILE (Shortened file)
    with open("../data/tweet_text.json", 'rb') as f:
        for line in f:
            tweet = json.loads(line)
            if tweet['author_status'] == "Bernie":
                bernie_tweets_BOW.append(tweet['text'])
            elif tweet['author_status'] == "Trump":
                trump_tweets_BOW.append(tweet['text'])
            elif tweet['author_status'] == "Bernie follower":
                trumpf_tweets_BOW.append(tweet['text'])
            elif tweet['author_status'] == "Trump follower":
                bernief_tweets_BOW.append(tweet['text'])

    # join the lists into big strings
    trump_BOW = ' '.join(trump_BOW)
    bernie_BOW = ' '.join(bernie_BOW)
    trump_tweets_BOW = ' '.join(trump_tweets_BOW)
    bernie_tweets_BOW = ' '.join(bernie_tweets_BOW)
    trumpf_tweets_BOW = ' '.join(trumpf_tweets_BOW)
    bernief_tweets_BOW = ' '.join(bernief_tweets_BOW)

    ##CHECKING LENGTH OF EACH STRING (Number of characters)
    print "Bernie tweets len: ", len(bernie_tweets_BOW)
    print "Trump tweets len: ", len(trump_tweets_BOW)
    print "Bernie follower tweets len: ", len(bernief_tweets_BOW)
    print "Trump follower tweets len: ", len(trumpf_tweets_BOW)

    ##ADDING BOW STRINGS TO A LIST
    texts = [
        trump_BOW,
        bernie_BOW,
        trump_tweets_BOW,
        bernie_tweets_BOW,
        trumpf_tweets_BOW,
        bernief_tweets_BOW,
    ]

    #PREPROCESSING EACH BoW
    print "Preprocessing"
    texts = [Preprocess(x) for x in texts]

    ###COMPUTING TFIDF VECTORS
    print "Vectorizing"
    vectors = Vectorize(*texts)

    from sklearn.metrics.pairwise import cosine_similarity
    cs = cosine_similarity(vectors[0:6], vectors)
    print cs

    ##COMPUTING COSINE SIMILARITY SCORES
    print "Comparing Trump and Bernie PR"
    print ((vectors * vectors.T))[0,1]

    print "Comparing Trump and Bernie Tweets"
    print ((vectors * vectors.T))[2,3]

    print "Comparing Trump PR and tweets"
    print ((vectors * vectors.T))[0,2]

    print "Comparing Bernie PR and tweets"
    print ((vectors * vectors.T))[1,3]

    print "Comparing Trump and Bernie Followers' Tweets"
    print ((vectors * vectors.T))[4,5]

    print "Comparing Trump tweets and his followers' Tweets"
    print ((vectors * vectors.T))[2,4]

    print "Comparing Bernie tweets and his followers' Tweets"
    print ((vectors * vectors.T))[3,5]
