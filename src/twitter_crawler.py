import twitter
import json
import time
import random

"""
We start with bernie and trump's twitter.

We get the last 3200 messages by paging back.

We save these messages.

We then call the GetFollowersIds API.
"""

# Constants
OUTPUT_PATH = '../data/tweets.json'
LUMINARIES = [
    'BernieSanders',
    'realDonaldTrump',
]
SAMPLE_FOLLOWER_NUM = 3

# Global variables
followers_dict = {
    x: [] for x in LUMINARIES
}

# Functions
def write_tweets(tweet_list_of_dicts, output_path):
    with open(output_path, 'a') as f:
        for tweet in tweet_list_of_dicts:
            j = json.dumps(tweet) + '\n'
            f.write(j)

def check_inputs(user_id=None, screen_name=None):
    """
    We want to accept either user_id or screen_name, not both

    This basically checks that we have an XOR of (user_id, screen_name)
        XOR returns True if A OR B is True, False otherwise
        XOR in python like this: A ^ B

    Raises ValueError otherwise
    """
    user_id_bool = bool(user_id)
    screen_name_bool = bool(screen_name)
    if user_id_bool ^ screen_name_bool:
        # continue silently if we meet the condition
        return None
    else:
        raise ValueError(
            'Exactly one of screen_name and user_id must be present'
        )

def get_rate_lim_info(api, endpoint):
    """
    Give this the api instance and an endpoint
    Endpoint is format:
        '/category/endpoint_name'
    Returns a dict of form:
        {
            'reset': unixtime,
            'limit': int,
            'remaining': int,
        }
    We use the following endpoints
        '/statuses/user_timeline'
    """
    rate_lim_statuses = api.GetRateLimitStatus()
    split_endpoint = endpoint.split('/')
    category = split_endpoint[1]
    rate_lim_info = \
        rate_lim_statuses['resources'][category][endpoint]
    return rate_lim_info

def get_user_timeline(
    api,
    user_id=None,
    screen_name=None,
    output_path=None,
    luminary_name=None
    ):
    """
    This function crawls all of a user's publicly available tweets
    It uses a while true loop plus a break condition
        We do this since we don't have prior information
            about how far back we need to go
        We also want to avoid a hacky "initialization call"

    Read up on paging through timelines here:
        https://dev.twitter.com/rest/public/timelines

    The API here returns a list of dicts
    """
    # Make sure inputs are valid
    check_inputs(user_id, screen_name)
    # Create an instance of our RateLimHandler
    # ratelim_handler = RateLimitHandler(api)
    rate_lim_info = get_rate_lim_info(
        api,
        '/statuses/user_timeline'
    )
    # Variables that we'll use to crawl the whole timeline
    max_id = None
    user_tweets = []
    while True:
        statuses = api.GetUserTimeline(
            screen_name=screen_name,
            user_id=user_id,
            max_id=max_id,
            count=200,
        )
        user_tweets.extend(statuses)
        # We take the minimum of the observed statuses
        # This is the max status we want for the NEXT call
        max_id = min([status._id for status in statuses]) - 1
        # Stopping condition is simple: if we see less than a full cache
        if len(statuses) < 150:
            print('Only found {} statuses for {}'.format(
                    len(statuses),
                    screen_name,
                )
            )
            break
        else:
            print('found 200 statuses for {}'.format(screen_name))
        rate_lim_info['remaining'] -= 1
        print(rate_lim_info)
        if rate_lim_info['remaining'] == 0:
            to_wait = max(0, rate_lim_info['reset'] - time.time())
            print('sleeping for {}'.format(to_wait))
            time.sleep(to_wait)
        # Our ratelim class makes it easy to wait for api reset
        # ratelim_handler.user_timeline_call()
    # we use this when we're crawling the followers of either bernie or trump
    user_tweets_as_dict = [x.AsDict() for x in user_tweets]
    for tweet in user_tweets_as_dict:
        if luminary_name:
            tweet['luminary_followed'] = luminary_name
        else:
            tweet['luminary_followed'] = None
    write_tweets(user_tweets_as_dict, output_path)

def sample_followers(
    api,
    user_id=None,
    screen_name=None,
    total_count=25000,
    ):
    """
    This function takes an api instance and user_id or screen_name
    Gets number of pages specified by "pages" argument

    We use a cursor, read up here:
        https://dev.twitter.com/overview/api/cursoring

    Returns a list of follower ids for the given user
    """
    check_inputs(user_id, screen_name)
    res = api.GetFollowerIDs(
        user_id=user_id,
        screen_name=screen_name,
        total_count=total_count,
    )
    return res


if __name__ == '__main__':
    with open('your_credentials.json', 'rb') as f:
        config = json.loads(f.read())
    # API globals
    api = twitter.Api(
        **config
    )
    # go through bernie and trump
    for luminary in LUMINARIES:
        get_user_timeline(
            api,
            screen_name=luminary,
            output_path=OUTPUT_PATH,
        )
        #followers = sample_followers(
        #    api,
        #    screen_name=luminary,
        #)
        #followers_dict[luminary].extend(followers)
    followers_to_crawl = []
    '''
    for luminary, followers in followers_dict.items():
        follower_sample = random.sample(followers, SAMPLE_FOLLOWER_NUM)
        for follower in follower_sample:
            get_user_timeline(
                api,
                user_id=follower,
                output_path=OUTPUT_PATH,
                luminary_name=luminary,
            )
    '''
