import time
import twitter

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
        The API returns something that looks like:
            {
                'resources': {
                    category: {
                        endpoint: {limit_info}
                    }
                }
            }
        We translate to:
            {endpoint: {limit_info}}
        """
        flattened_rate_limits = {}
        for category, endpoints in api_ratelim_output['resources'].items():
            for endpoint, limit_info in endpoints.items():
                flattened_rate_limits[endpoint] = limit_info
        return flattened_rate_limits
