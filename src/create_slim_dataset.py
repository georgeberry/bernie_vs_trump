"""
This file takes the raw tweets stored in tweets.json and slims them down

We don't need all of the tweet data for our planned analysis

We output a line-delimited json file of format:
{
    'text': tweet_text,
    'author_status': author_status
}

author_status can take on one of four values:
    'Bernie',
    'Trump',
    'Bernie follower',
    'Trump follower'

We store the output in tweet_text.json
"""
import json

INPUT_PATH = '../data/tweets.json'
OUTPUT_PATH = '../data/tweet_text.json'

def slim_tweet(raw_tweet):
    text = raw_tweet['text']
    author = raw_tweet['user']['screen_name']
    luminary_status = raw_tweet['luminary_followed']
    if luminary_status == 'BernieSanders':
        author_status = 'Bernie follower'
    elif luminary_status == 'realDonaldTrump':
        author_status = 'Trump follower'
    elif author == 'BernieSanders':
        author_status = 'Bernie'
    elif author == 'realDonaldTrump':
        author_status = 'Trump'
    else:
        raise ValueError('Could not determine author_status')
    return {'text': text, 'author_status': author_status}

with open(INPUT_PATH, 'rb') as f:
    with open(OUTPUT_PATH, 'wb') as g:
        for line in f:
            j = json.loads(line)
            slim_j = json.dumps(slim_tweet(j)) + '\n'
            g.write(slim_j)
