import string
import nltk
import sklearn
import math
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

"""This is a simple demo to compute the cosine similarity between two documents, Trump and Bernie's
    press releases on Justice Scalia's death. We draw upon nltk and sklearn but try to illustrate
    process using our own functions."""

##Setting strings
T = "I would like to offer my sincerest condolences to the Scalia family after the passing of Justice Scalia. Justice Scalia was a remarkable person and a brilliant Supreme Court Justice, one of the best of all time. His career was defined by his reverence for the Constitution and his legacy of protecting Americans' most cherished freedoms. He was a Justice who did not believe in legislating from the bench and he is a person whom I held in the highest regard and will always greatly respect his intelligence and conviction to uphold the Constitution of our country. My thoughts and prayers are with his family during this time."
B = "U.S. Sen. Bernie Sanders issued the following statement on Saturday ont he passing of U.S. Supreme Court Justice Antonin Scalia. While I differed with Justice Scalia's views and jurisprudence, he was a brilliant, colorful and outspoken member of the Supreme Court. My thoughts and prayers are with his family and his colleagues on the court who mourn his passing."

def lower(s):
    s = s.lower()
    return s

def remove_punct(s):
    punct = set(string.punctuation)
    s = ''.join(ch for ch in s if ch not in punct)
    return s

def tokens(s):
    tokens = nltk.word_tokenize(s)
    return tokens

stopwords = nltk.corpus.stopwords.words("english")

def words(tokens):
    words = [word for word in tokens if not word in stopwords]
    return words

def stems(words):
    stemmer = nltk.stem.porter.PorterStemmer()
    stems = [stemmer.stem(word) for word in words]
    return stems

def freq_count(text):
    count_dict = {}
    words = text.split()
    for word in words:
        if word not in count_dict:
            count_dict[word] = 0
        count_dict[word] += 1
    return sorted(count_dict.items(), reverse=True, key=lambda x: x[1])

def tfidf(freq_all, freq_b, freq_t):
    N = 2
    #Convert both freq lists to dict
    fb = dict(freq_b)
    ft = dict(freq_t)
    idf_vector = []
    #For each of the term/freq pairs
    #Calculate idf weight
    for i in freq_all:
        word = i[0]
        tf = i[1]
        if tf == 1:
            idf = math.log(1 + N/1)
        #term occurs more than once in corpus
        elif word in fb and word in ft:
            #if term is in both docs
            idf = math.log(1 + N/2)
            #term is only in one doc
        else: #word is only in one document but occurs
            #more than once
            idf = math.log(1 + N/1)
        t = (word,idf)
        idf_vector.append(t)
    #convert the list of tuples to a dict
    idf_dict = dict(idf_vector)
    #create two empty dicts for each text
    tfidf_T = {}
    tfidf_B = {}
    #Get a list of all terms
    all_terms = idf_dict.keys()
    #For each term
    for term in all_terms:
        #if word is in Bernie dict as a key
        if term in fb:
            #Then add the term to bernie's dict
            #and the value is the idf score from idf dict
            #multipled by freq score from the original dict
            tfidf_B[term] = idf_dict[term]*fb[term]
        if term not in fb:
            tfidf_B[term] = 0
        if term in ft:
            tfidf_T[term] = idf_dict[term]*ft[term]
        if term not in ft:
            tfidf_T[term] = 0
    #Return dicts as sorted lists of tuples
    return (sorted(idf_dict.items(), reverse=True, key=lambda x: x[1]),
            sorted(tfidf_T.items(), reverse=True, key=lambda x: x[1]),
            sorted(tfidf_B.items(), reverse=True, key=lambda x: x[1]))

def cosine_sim(vector1,vector2):
    #Converting lists of tuples to arrays
    v1 = []
    v2 = []
    #Converting both to arrays
    for i in vector1:
        v1.append(i[1])
    for i in vector2:
        v2.append(i[1])
    v1 = np.asarray(v1)
    v2 = np.asarray(v2)
    #Computing the dot product of two arrays
    numerator = np.dot(v1,v2)
    #Computing length of each vector
    sum_v1_sq = 0
    sum_v2_sq = 0
    for i in v1:
        k = i**2
        sum_v1_sq += k
    length_v1 = math.sqrt(sum_v1_sq)
    for i in v2:
        k = i**2
        sum_v2_sq += k
    length_v2 = math.sqrt(sum_v2_sq)
    denominator = abs(length_v1)*abs(length_v2)
    cosine_sim = numerator/denominator
    return cosine_sim

if __name__ == '__main__':

    t = lower(T)
    t = remove_punct(T)
    T2 = t
    T = tokens(T)
    T = stems(T)
    T = ' '.join(T)
    Tf = freq_count(T)
    B = lower(B)
    B = remove_punct(B)
    B2 = B
    B = tokens(B)
    B = stems(B)
    B = ' '.join(B)
    Bf = freq_count(B)
    All = B+T
    Allf = freq_count(All)
    tfidf1 = tfidf(Allf,Bf,Tf)
    cossim = cosine_sim(tfidf1[1],tfidf1[0])
    print cossim

vectorizer = TfidfVectorizer(tokenizer=tokens, stop_words=stopwords)
tfidf = vectorizer.fit_transform([T,B])
print ((tfidf * tfidf.T))[0,1]
from sklearn.metrics.pairwise import cosine_similarity
cs = cosine_similarity(tfidf[0:1], tfidf)
print cs
