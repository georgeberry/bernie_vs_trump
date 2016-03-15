"""
Use NLTK + SciKit learn to classify tweets as "Bernie" or "Trump"
    Bernie = 1
    Trump = 0

We can use a range of models:
    SVM
    Naive Bayes
    Logistic Regression
    Lasso
    Decision Tree
    Decision Forest
    GBDT

This script has two parts:
    1) Feature Creation
    2) Model Fitting

Part #1 is more important for NLP work

We use AUC curves to evaluate model performance
These easily display the precision/recall at various classificastion thresolds

F2 measures are also useful
"""
import re
import json
import pandas as pd
from sklearn import tree
from sklearn import metrics
from sklearn import cross_validation
from sklearn import naive_bayes
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.tokenize import TweetTokenizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textstat.textstat import textstat
from ggplot import *

vadr = SentimentIntensityAnalyzer()
tweet_tokenizer = TweetTokenizer(reduce_len=True)

stemmer = nltk.stem.porter.PorterStemmer()
stopwords = nltk.corpus.stopwords.words("english")
other_exclusions = ["ff", "rt"]
stopwords.extend(other_exclusions)

## Feature Creation ##

### Feature Creation Functions ###

def preprocess(text_string):
    """
    Accepts a text string and replaces:
    1) urls with URLHERE
    2) lots of whitespace with one instance
    3) mentions with MENTIONHERE

    This allows us to get standardized counts of urls and mentions
    Without caring about specific people mentioned
    """
    space_pattern = '\s+'
    giant_url_regex = ('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|'
        '[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    mention_regex = '@[\w\-]+'
    parsed_text = text_string.lower()
    parsed_text = re.sub(space_pattern, ' ', text_string)
    parsed_text = re.sub(giant_url_regex, 'URLHERE', parsed_text)
    parsed_text = re.sub(mention_regex, 'MENTIONHERE', parsed_text)
    return parsed_text

def tokenizer(text_string):
    """
    A soup-to-nuts tokenizer
    Does all of the processing we want
    """
    preprocessed_text = preprocess(text_string)
    tokens = tweet_tokenizer(preprocessed_text)
    return tokens

def stemming_tokenizer(text_string):
    """
    Exactly like the tokenizer function but with stemmer
    """
    preprocessed_text = preprocess(text_string)
    tokens = tweet_tokenizer(preprocessed_text)
    stems = [stemmer.stem(t) for t in tokens]
    return stems

def extra_features(text_string):
    """
    Sometimes we want to hand craft features, this does that
    Should create that preprocess + tokenizer can't
    Takes a text string, returns counts of features

    We return a named dict for each row
    These are easy to convert to pandas DF later

    The textstat measures return numbers
    The sentiment is a named dict
    """
    output_dict = {}
    # try:
    #     output_dict['flesch'] = textstat.flesch_reading_ease(text_string)
    # except:
    #     output_dict['flesch'] = 0
    sentiment_dict = vadr.polarity_scores(text_string)
    for k, v in sentiment_dict.items():
        output_dict[k] = v
    return output_dict

if __name__ == '__main__':
    list_of_dicts = []
    with open('../data/tweet_text.json', 'rb') as f:
        for line in f:
            j = json.loads(line)
            author_status = j['author_status']
            if author_status == 'Trump':
                j['author_status'] = 0
                list_of_dicts.append(j)
            elif author_status == 'Bernie':
                j['author_status'] = 1
                list_of_dicts.append(j)
            elif author_status == 'Trump follower':
                j['author_status'] = 2
                list_of_dicts.append(j)
            elif author_status == 'Bernie follower':
                j['author_status'] = 3
                list_of_dicts.append(j)
            else:
                # keep only bernie/trump tweets for now
                continue

    # Use scipy data frame because it plays nice with numpy/scipy
    # Also handles strings nicely
    df = pd.DataFrame(list_of_dicts)
    # This is the syntax to pull out columns of the df

    # Trump v Bernie
    # Select rows that correspond to Bernie or Trump tweets only
    trump_bernie_df = df[df['author_status'] <= 1].copy()

    vectorizer = TfidfVectorizer(
        tokenizer=tokenizer,
        ngram_range=(1, 2),
        stop_words=stopwords,
    )
    # Split into features and target
    y = trump_bernie_df['author_status']
    X_strings = trump_bernie_df['text']
    vectorizer.fit(X_strings) # fit vectorizer here
    # X is our sparse matrix of predictors
    X = vectorizer.transform(X_strings)

    X_train, X_test, y_train, y_test = cross_validation.train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    bnb = naive_bayes.BernoulliNB()
    bnb.fit(X_train, y_train)
    y_pred = bnb.predict(X_test)

    # Evaluate this classifier! #
    # Precision: tp / (tp + fp)
    # Recall: tp / (tp + fn)
    # F_1: 2 * (precision * recall) / precision + recall
    #   e.g. harmonic mean of precision and recall
    print(
        'Baseline guessing is {}'.format(float(sum(y_train)) / len(y_train))
    )
    print(
        'The precision is {}'.format(metrics.precision_score(y_test, y_pred))
    )
    print(
        'The recall is {}'.format(metrics.recall_score(y_test, y_pred))
    )
    print(
        'The f score is {}'.format(metrics.f1_score(y_test, y_pred))
    )

    # AUC-ROC
    # area under receiver operator curve
    y_pred_prob = bnb.predict_proba(X_test)[:,1]
    fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_prob)
    roc_df = pd.DataFrame(dict(fpr=fpr, tpr=tpr))
    '''
    g = ggplot(roc_df, aes(x='fpr',y='tpr')) +\
        geom_line() +\
        geom_abline(linetype='dashed') +\
        ylim(low=-0.1, high=1.1) +\
        xlim(low=-0.1, high=1.1)
    print(g)
    '''
    #############################
    ## Follower classification ##
    follower_df = df[df['author_status'] >= 2].copy().sample(n=10000)
    # follower_df = follower_df.iloc[0:100,:]
    follower_df['author_status'] -= 2

    y_follower = follower_df['author_status']
    X_follower = vectorizer.transform(follower_df['text'])
    y_follower_pred_prob = bnb.predict_proba(X_follower)[:,1]
    y_follower_pred = bnb.predict(X_follower)
    # summary measures here
    print(
        'Baseline guessing is {}'.format(
            float(sum(y_follower)) / len(y_follower)
        )
    )
    print(
        'The precision is {}'.format(
            metrics.precision_score(y_follower, y_follower_pred)
        )
    )
    print(
        'The recall is {}'.format(
            metrics.recall_score(y_follower, y_follower_pred)
        )
    )
    print(
        'The f score is {}'.format(
            metrics.f1_score(y_follower, y_follower_pred)
        )
    )

    fpr, tpr, _ = metrics.roc_curve(y_follower, y_follower_pred_prob)
    roc_df = pd.DataFrame(dict(fpr=fpr, tpr=tpr))
    '''
    g = ggplot(roc_df, aes(x='fpr',y='tpr')) +\
        geom_line() +\
        geom_abline(linetype='dashed') +\
        ylim(low=-0.1, high=1.1) +\
        xlim(low=-0.1, high=1.1)
    print(g)
    '''
    #############################
    # Retrain this on followers #
    vectorizer_f = TfidfVectorizer(
        tokenizer=tokenizer,
        ngram_range=(1, 2),
        stop_words=stopwords,
    )
    vectorizer_f.fit(follower_df['text'])
    X_follower = vectorizer_f.transform(follower_df['text'])
    Xf_train, Xf_test, yf_train, yf_test = cross_validation.train_test_split(
        X_follower,
        y_follower,
        test_size=0.2,
        random_state=42,
    )
    bnbf = naive_bayes.BernoulliNB()
    bnbf.fit(Xf_train, yf_train)
    yf_pred_prob = bnbf.predict_proba(Xf_test)[:,1]
    yf_pred = bnbf.predict(Xf_test)
    # summary measures here
    print(
        'Baseline guessing is {}'.format(float(sum(yf_train)) / len(yf_train))
    )
    print(
        'The precision is {}'.format(metrics.precision_score(yf_test, yf_pred))
    )
    print(
        'The recall is {}'.format(metrics.recall_score(yf_test, yf_pred))
    )
    print(
        'The f score is {}'.format(metrics.f1_score(yf_test, yf_pred))
    )

    fpr, tpr, _ = metrics.roc_curve(yf_test, yf_pred_prob)
    roc_df = pd.DataFrame(dict(fpr=fpr, tpr=tpr))
    '''
    g = ggplot(roc_df, aes(x='fpr',y='tpr')) +\
        geom_line() +\
        geom_abline(linetype='dashed') +\
        ylim(low=-0.1, high=1.1) +\
        xlim(low=-0.1, high=1.1)
    print(g)
    '''
    ##########################################
    # Now for something completely different #

    # We have 2 models, bnb and bnbf
    # Let's use these to generate features
    # Let's apply them to our more difficult problem, categorizing followers

    feature_list_of_dicts = [extra_features(x) for x in follower_df['text']]
    X_lum = vectorizer.transform(follower_df['text'])
    X_f = vectorizer_f.transform(follower_df['text'])
    predicted_luminary = bnb.predict_proba(X_lum)[:,1]
    predicted_follower = bnbf.predict_proba(X_f)[:,1]

    X_all = pd.DataFrame(feature_list_of_dicts)
    X_all['luminary'] = predicted_luminary
    X_all['follower'] = predicted_follower
    y_all = follower_df['author_status']

    Xa_train, Xa_test, ya_train, ya_test = cross_validation.train_test_split(
        X_all,
        y_all,
        test_size=0.2,
        random_state=42,
    )

    dt = tree.DecisionTreeClassifier()
    dt.fit(Xa_train, ya_train)
    ya_pred = dt.predict(Xa_test)

    print(
        'Baseline guessing is {}'.format(float(sum(ya_train)) / len(ya_train))
    )
    print(
        'The precision is {}'.format(metrics.precision_score(ya_test, ya_pred))
    )
    print(
        'The recall is {}'.format(metrics.recall_score(ya_test, ya_pred))
    )
    print(
        'The f score is {}'.format(metrics.f1_score(ya_test, ya_pred))
    )
