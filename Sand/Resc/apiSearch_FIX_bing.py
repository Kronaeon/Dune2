





class BingSearchAPI(SearchEngine):
    """Search engine implementation for Bing using the official API.
    
    Attributes:
        api_key (str): Your Bing Search API key.
        rate_limiter (RateLimiter): Rate limiting instance.
        name (str): Name of the search engine.
    """
    def __init__(self, api_key, rate_limiter):
        self.api_key = api_key
        self.rate_limiter = rate_limiter
        self.name = 'Bing API'
        self.endpoint = 'https://api.bing.microsoft.com/v7.0/search'

    def search(self, query, max_results):
        """Fetches search results from Bing API.

        Args:
            query (str): Search query.
            max_results (int): Maximum number of results to return.

        Returns:
            list: List of dicts with 'url', 'title', 'snippet', 'rank', 'search_engine'.
        """
        results = []
        offset = 0
        count = min(50, max_results)  # Bing API allows up to 50 results per request
        
        headers = {
            'Ocp-Apim-Subscription-Key': self.api_key,
            'Accept': 'application/json'
        }
        
        while len(results) < max_results:
            params = {
                'q': query,
                'count': count,
                'offset': offset,
                'mkt': 'en-US',
                'responseFilter': 'Webpages'
            }
            
            self.rate_limiter.wait()
            
            try:
                response = requests.get(self.endpoint, headers=headers, params=params)
                response.raise_for_status()  # Raise an exception for non-200 status codes
                
                data = response.json()
                
                if 'webPages' not in data or 'value' not in data['webPages']:
                    logging.warning("No web pages found in Bing API response")
                    break
                
                pages = data['webPages']['value']
                
                if not pages:
                    break
                
                for i, page in enumerate(pages):
                    rank = len(results) + i + 1
                    results.append({
                        'url': page['url'],
                        'title': page['name'],
                        'snippet': page['snippet'],
                        'rank': rank,
                        'search_engine': self.name
                    })
                
                # Check if we need to get more results
                if len(pages) < count or len(results) >= max_results:
                    break
                    
                offset += count
                
            except requests.exceptions.RequestException as e:
                logging.error(f"Error fetching Bing API results: {e}")
                break
                
        return results[:max_results]


