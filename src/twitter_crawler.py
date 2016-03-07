import twitter
import json
import time
import random

"""
We start with bernie and trump's twitter.

We get the last 3200 messages by paging back.

We save these messages.

We then call the GetFollowersIds API to get 5000 followers of both
    Randomly sample a few followers
"""

# Functions
def write_tweets(tweet_list_of_dicts, output_path):
    '''
    Takes data of format:
        [{tweet}, {tweet}, ...]
    Writes to path specified
    '''
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
    rate_lim_info = get_rate_lim_info(api, '/statuses/user_timeline')
    if rate_lim_info['remaining'] == 0:
        to_wait = max(0, rate_lim_info['reset'] - time.time())
        print('Sleeping for {}'.format(to_wait))
        time.sleep(to_wait)
        rate_lim_info = get_rate_lim_info(api, '/statuses/user_timeline')
    # Variables that we'll use to crawl the whole timeline
    max_id = None #None since we start at top of the timeline
    user_tweets = []
    # We do this loop until we see "break"
    # Has to be this format because we don't know how far back we have to go
    while True:
        # This except block handles querying private accounts
        # We get a TwitterError when we make a call to a private acct
        try:
            statuses = api.GetUserTimeline(
                screen_name=screen_name,
                user_id=user_id,
                max_id=max_id,
                count=200,
            )
        except twitter.TwitterError as e:
            print('Could not query user, got Twitter Error: {}'.format(e))
            break
        user_tweets.extend(statuses)
        # Handle EXIT CONDITION here
        # The API is supposed to return exactly 200 Tweets
        # In practice, we usually get 190+
        # 100 seems like a safe cutoff, this is hand tuned
        if len(statuses) < 100:
            print('Only found {} statuses'.format(len(statuses)))
            break
        else:
            print('Found {} statuses'.format(len(statuses)))
        # We handling PAGING here
        # This is the max status we want for the NEXT call
        # See paging docs: https://dev.twitter.com/rest/public/timelines
        max_id = min([status._id for status in statuses]) - 1 #Sets max as the minimum id from the previous call
        # We handle RATE LIMITING here
        # We generated rate_lim_info above
        # We decrement the number of calls remaining
        # If there are 0 calls left, we wait until API window resets
        rate_lim_info['remaining'] -= 1
        print(rate_lim_info)
        if rate_lim_info['remaining'] == 0:
            to_wait = max(0, rate_lim_info['reset'] - time.time())
            print('Sleeping for {}'.format(to_wait))
            time.sleep(to_wait)
            rate_lim_info = get_rate_lim_info(api, '/statuses/user_timeline')
        else:
            time.sleep(1.0) #This helps space out calls; good practice
    # After we've gotten the whole timeline
    # Convert tweets to dict
    user_tweets_as_dict = [x.AsDict() for x in user_tweets]
    # Add a field to this tweet dict indicating if a user follows a luminary
    #   e.g. Bernie or Trump
    for tweet in user_tweets_as_dict:
        if luminary_name:
            tweet['luminary_followed'] = luminary_name
        else:
            tweet['luminary_followed'] = None
    # Write tweets by appending them to a big file
    write_tweets(user_tweets_as_dict, output_path)

def sample_followers(
    api,
    user_id=None,
    screen_name=None,
    total_count=5000,
    ):
    """
    This function takes an api instance and user_id or screen_name
    Gets number of pages specified by "pages" argument

    We use a cursor, read up here:
        https://dev.twitter.com/overview/api/cursoring

    Returns a list of follower ids for the given user
    """
    check_inputs(user_id, screen_name)
    rate_lim_info = get_rate_lim_info(
        api,
        '/followers/ids',
    )
    # Handle rate limit issues here
    # If we have 0 calls left, wait until reset
    print(rate_lim_info)
    if rate_lim_info['remaining'] == 0:
        to_wait = max(0, rate_lim_info['reset'] - time.time())
        print('sleeping for {}'.format(to_wait))
        time.sleep(to_wait)
    res = api.GetFollowerIDs(
        user_id=user_id,
        screen_name=screen_name,
        total_count=total_count,
    )
    return res


if __name__ == '__main__':
    # Constants
    OUTPUT_PATH = '../data/tweets3.json'
    LUMINARIES = [
        'BernieSanders',
        'realDonaldTrump',
       # 'tedcruz','marcorubio',
       # 'HillaryClinton'
    ]
    SAMPLE_FOLLOWER_NUM = 250

    # Global variables
    followers_dict = {
        x: [] for x in LUMINARIES
    }
    '''
    Read in your secret API credentials here
    File looks like this:
        {
            "consumer_key": consumer_key,
            "consumer_secret": consumer_secret,
            "access_token_key": access_token_key,
            "access_token_secret": access_token_secret,
        }
    '''
    with open('credentials.json', 'rb') as f:
        config = json.loads(f.read())
    # Authenticate an API instance
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
        followers = sample_followers(
            api,
            screen_name=luminary,
        )
        followers_dict[luminary].extend(followers)
    # for both Bernie and Trump, crawl a small sample of followers
    for luminary, followers in followers_dict.items():
        follower_sample = random.sample(followers, SAMPLE_FOLLOWER_NUM)
        for follower in follower_sample:
            print('Crawling follower {}'.format(follower))
            get_user_timeline(
                api,
                user_id=follower,
                output_path=OUTPUT_PATH,
                luminary_name=luminary,
            )
