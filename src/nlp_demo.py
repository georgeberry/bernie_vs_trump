import string
import nltk
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer

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

if __name__ == '__main__':
    T1 = lower(T)
    T2 = remove_punct(T1)
    B1 = lower(B)
    B2 = remove_punct(B1)
    vectorizer = TfidfVectorizer(tokenizer=tokens, stop_words=stopwords)
    tfidf = vectorizer.fit_transform([T2,B2])
    print tfidf
    print ((tfidf * tfidf.T))[0,1]
