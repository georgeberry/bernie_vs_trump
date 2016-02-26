import twitter
import json
import time

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

with open('your_credentials.json', 'rb') as f:
    config = json.loads(f.read())

# API globals
api = twitter.Api(
    **config
)

# Class to wrap API ratelimit handling
class RateLimitHandler:
    """
    Hand this class the api instance
    Gets the rate limit statuses

    Lets you know how many calls you have left, and when it resetes
    """
    def __init__(self, api):
        self.api = api
        api_ratelim_output = self.api.GetRateLimitStatus()
        self.rate_limits = self.flatten_api_output(api_ratelim_output)

    def endpoint_current_ratelim(self, endpoint):
        """
        Returns a dict of form:
        {
            'reset': unixtime,
            'limit': int,
            'remaining': int,
        }
        For the endpoint requested
        For instance, can return this dict for '/users/statuses'
        """
        self.refresh()
        return self.rate_limits[endpoint]

    def user_timeline_call(self):
        """
        Every time we make a statuses/user_timeline call, we call this
        It keeps track of ratelim for us, and sleeps/refeshes automatically
        """
        user_timeline_lim = self.rate_limits['/statuses/user_timeline']
        user_timeline_lim['remaining'] -= 1
        calls_remaining = user_timeline_lim['remaining']
        if calls_remaining > 0:
            print('{} calls remaining'.format(remaining))
        else:
            wait_time = self.reset_wait_time(user_timeline_lim['reset'])
            print('Need to sleep for {} seconds'.format(wait_time))
            time.sleep(wait_time + 1)
            self.refresh()

    def refresh(self):
        """
        If we can make calls to the rate-limit-checking API
            Then reset the rate limit status
        If we can't make calls to the rate-limit-checking API
            Then wait and recursively call this function
        """
        rate_limit_lim = self.rate_limits['/application/rate_limit_status']
        wait_time = self.reset_wait_time(rate_limit_lim['reset'])
        calls_remaining = rate_limit_lim['remaining']

        if calls_remaining > 0 or wait_time <= 0:
            api_ratelim_output = self.api.GetRateLimitStatus()
            self.rate_limits = self.flatten_api_output(api_ratelim_output)
        else:
            print('Need to sleep for {} seconds'.format(wait_time))
            time.sleep(wait_time + 1)
            self.refresh()

    @staticmethod
    def reset_wait_time(reset_time):
        wait_time = reset_time - time.time()
        return wait_time

    @staticmethod
    def flatten_api_output(api_ratelim_output):
        """
        The API returns somethign that looks like:
            {'resources': {category: {endpoint: {limit_info}}}}
        We translate to:
            {endpoint: {limit_info}}
        """
        flattened_rate_limits = {}
        for category, endpoints in api_ratelim_output['resources'].items():
            for endpoint, limit_info in endpoints.items():
                flattened_rate_limits[endpoint] = limit_info
        return flattened_rate_limits


# Functions
def write_tweets(tweet_list):
    with open(OUTPUT_PATH, 'a') as f:
        for tweet in tweet_list:
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

def get_user_timeline(api, user_id=None, screen_name=None):
    """
    This function crawls all of a user's publicly available tweets
    It uses a while true loop plus a break condition
        We do this since we don't have prior information
            about how far back we need to go
        We also want to avoid a hacky "initialization call"

    The API here returns a list of dicts
    """
    # Make sure inputs are valid
    check_inputs(user_id, screen_name)
    # Create an instance of our RateLimHandler
    ratelim_handler = RateLimitHandler(api)
    # Variables that we'll use to crawl the whole timeline
    max_id = None
    user_tweets = []
    while True:
        statuses = api.GetUserTimeline(
            screen_name=screen_name,
            user_id=user_id,
            max_id=max_id,
        )
        user_tweets.extend(statuses)
        # We take the minimum of the observed statuses
        max_id = min([status['id'] for status in statuses])
        # Stopping condition is simple: if we see less than a full cache
        if len(statuses) < 200:
            break
        # Our ratelim class makes it easy to wait for api reset
        ratelim_handler.user_timeline_call()
    write_tweets(user_tweets)

def sample_follower_tweets(api, user_id=None, screen_name=None):
    """
    This function takes
    """
    if user_id and screen_name:
        raise ValueError
    if not user_id and not screen_name:
        raise ValueError
    api.GetFollowersIds()



if __name__ == '__main__':
    pass
