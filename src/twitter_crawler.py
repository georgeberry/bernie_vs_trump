import tweepy
import json
import sleep

"""
We start with bernie and trump's twitter.

We get the last 3200 messages by paging back.

We save these messages.

We then call the GetFollowersIds API.


"""

# Constants
OUTPUT_PATH = '../data/tweets.json'
LUMINARIES = [
    '@BernieSanders',
    '@realDonaldTrump',
]

# Global variables
followers = {
    'trump': [],
    'bernie': [],
}

# API globals
api = twitter.Api(
    consumer_key='consumer_key',
    consumer_secret='consumer_secret',
    access_token_key='access_token',
    access_token_secret='access_token_secret',
)

# Functions
def write_tweets(tweet_list):
    with open(OUTPUT_PATH, 'a') as f:
        for tweet in tweet_list:
            j = json.dumps(tweet) + '\n'
            f.write(j)

def crawl_user(api, user_id=None, screen_name=None):
    """
    This function crawls all of a user's publicly available tweets
    It uses a while true loop plus a break condition
        We do this since we don't have prior information
        about how far back we need to go
        We also want to avoid a hacky "initialization call"

    The API here returns a list of dicts
    """
    if user_id and screen_name:
        raise ValueError
    if not user_id and not screen_name:
        raise ValueError
    max_id = None
    user_tweets = []
    while True:
        statuses = api.GetUserTimeline(
            screen_name=screen_name,
            user_id=user_id,
            max_id=max_id,
        )
        user_tweets.extend(statuses)
        # we take the minimum of the observed statuses
        max_id = min([status['id'] for status in statuses])

        # stopping condition is simple: if we see less than a full cache
        if len(statuses) < 200:
            break
    write_tweets(user_tweets)
