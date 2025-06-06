import requests
from bs4 import BeautifulSoup
from flask import Flask, jsonify, request
import urllib.parse
import time
import random
from typing import List, Dict, Optional

# imports from the apiSearch Google.
import logging
import hashlib
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class BingSearchScraper:
    def __init__(self):
        self.base_url = "https://www.bing.com/search"
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }
        self.session = requests.Session()
        self.session.headers.update(self.headers)

    def search(self, query: str, count: int = 10, offset: int = 0) -> Dict:
        """
        Perform a Bing search and return structured results
        
        Args:
            query: Search query string
            count: Number of results to return (max 50)
            offset: Offset for pagination
            
        Returns:
            Dictionary containing search results and metadata
        """
        try:
            # Prepare search parameters
            params = {
                'q': query,
                'count': min(count, 50),  # Bing typically shows max 50 per page
                'first': offset + 1 if offset > 0 else 1
            }
            
            # Add random delay to avoid being blocked
            time.sleep(random.uniform(0.5, 1.5))
            
            # Make the request
            response = self.session.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            
            # Parse the HTML
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract search results
            results = self._extract_results(soup)
            
            # Extract metadata
            total_results = self._extract_total_results(soup)
            
            return {
                'query': query,
                'total_results': total_results,
                'results_count': len(results),
                'offset': offset,
                'results': results
            }
            
        except requests.RequestException as e:
            return {
                'error': f'Request failed: {str(e)}',
                'query': query,
                'results': []
            }
        except Exception as e:
            return {
                'error': f'Parsing failed: {str(e)}',
                'query': query,
                'results': []
            }

    def _extract_results(self, soup: BeautifulSoup) -> List[Dict]:
        """Extract search results from the parsed HTML"""
        results = []
        
        # Bing uses different selectors for search results
        result_containers = soup.find_all('li', class_='b_algo')
        
        for container in result_containers:
            try:
                # Extract title and URL
                title_elem = container.find('h2')
                if not title_elem:
                    continue
                    
                link_elem = title_elem.find('a')
                if not link_elem:
                    continue
                
                title = title_elem.get_text(strip=True)
                url = link_elem.get('href', '')
                
                # Clean up URL if it's a Bing redirect
                if url.startswith('/'):
                    url = 'https://www.bing.com' + url
                
                # Extract description
                desc_elem = container.find('p') or container.find('div', class_='b_caption')
                description = desc_elem.get_text(strip=True) if desc_elem else ''
                
                # Extract displayed URL (cite)
                cite_elem = container.find('cite')
                display_url = cite_elem.get_text(strip=True) if cite_elem else url
                
                if title and url:
                    results.append({
                        'title': title,
                        'url': url,
                        'description': description,
                        'display_url': display_url
                    })
                    
            except Exception as e:
                # Skip problematic results but continue processing
                continue
        
        return results

    def _extract_total_results(self, soup: BeautifulSoup) -> Optional[str]:
        """Extract total number of results"""
        try:
            # Look for results count in various possible locations
            count_selectors = [
                '.sb_count',
                '.b_count',
                '[id*="count"]'
            ]
            
            for selector in count_selectors:
                count_elem = soup.select_one(selector)
                if count_elem:
                    return count_elem.get_text(strip=True)
            
            return None
        except:
            return None
    
    def fetch_page_content(self, url, timeout=10):
        """Fetch the content of a webpage.
        
        Args:
            url (str): The URL to fetch.
            timeout (int): Request timeout in seconds.
            
        Returns:
            str: The HTML content of the page, or None if the request failed.
        """
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        try:
            response = requests.get(url, headers=headers, timeout=timeout)
            response.raise_for_status()
            return response.text
        except requests.exceptions.RequestException as e:
            logging.error(f"Error fetching page content from {url}: {e}")
            return None

    def extract_text_from_html(self, html):
        """Extract main text content from HTML.
        
        Args:
            html (str): HTML content to parse.
            
        Returns:
            str: Extracted text content.
        """
        if not html:
            return ""
        
        try:
            soup = BeautifulSoup(html, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style", "nav", "footer", "header"]):
                script.extract()
                
            # Get text
            text = soup.get_text(separator='\n')
            
            # Break into lines and remove leading and trailing space on each
            lines = (line.strip() for line in text.splitlines())
            
            # Break multi-headlines into a line each
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            
            # Remove blank lines
            text = '\n'.join(chunk for chunk in chunks if chunk)
            
            return text
        except Exception as e:
            logging.error(f"Error extracting text from HTML: {e}")
            return ""
    
    def save_content_to_file(self, result, content, output_dir="search_results"):
        """Save content to a text file.
        
        Args:
            result (dict): Search result dictionary.
            content (str): Text content to save.
            output_dir (str): Directory to save files in.
            
        Returns:
            str: Path to the saved file, or None if save failed.
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # Create a filename based on the title or URL
        if result.get('title'):
            # Use title for filename but ensure it's valid
            base_filename = ''.join(c if c.isalnum() or c in [' ', '_', '-'] else '_' for c in result['title'])
            base_filename = base_filename[:50]  # Limit length
        else:
            # Use a hash of the URL if no title
            base_filename = hashlib.md5(result['url'].encode()).hexdigest()
            
        # Add rank to ensure uniqueness - use rank if available, otherwise use a counter
        rank = result.get('rank', 0)
        filename = f"{rank:03d}_{base_filename}.txt"
        filepath = os.path.join(output_dir, filename)
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                # Write URL and title as metadata at the top
                f.write(f"URL: {result['url']}\n")
                f.write(f"Title: {result['title']}\n")
                f.write(f"Rank: {rank}\n")
                f.write("-" * 80 + "\n\n")
                
                # Write the actual content
                f.write(content)
                
            logging.info(f"Saved content to {filepath}")
            return filepath
        except Exception as e:
            logging.error(f"Error saving content to file: {e}")
            return None
    
    def download_and_save_results(self, results, output_dir="search_results"):
        """Download content from search results and save to files.
        
        Args:
            results (list): List of search result dictionaries.
            output_dir (str): Directory to save files in.
            
        Returns:
            list: List of successful save paths.
        """
        saved_files = []
        
        for i, result in enumerate(results, 1):
            # Add rank field to result if not present (fix for Bing results structure)
            if 'rank' not in result:
                result['rank'] = i
                
            logging.info(f"Processing: [{result['rank']}] {result['title']} - {result['url']}")
            
            # Fetch HTML content
            html = self.fetch_page_content(result['url'])
            if not html:
                logging.warning(f"Could not fetch content from {result['url']}")
                continue
                
            # Extract text from HTML
            text_content = self.extract_text_from_html(html)
            if not text_content:
                logging.warning(f"Could not extract text from {result['url']}")
                continue
                
            # Save content to file
            filepath = self.save_content_to_file(result, text_content, output_dir)
            if filepath:
                saved_files.append(filepath)
                
            # Simple rate limiting to be nice to servers
            time.sleep(1)
            
        return saved_files

# Flask API wrapper
app = Flask(__name__)
scraper = BingSearchScraper()

@app.route('/search', methods=['GET'])
def search_endpoint():
    """
    Search endpoint
    
    Parameters:
        q: Query string (required)
        count: Number of results (optional, default=10, max=50)
        offset: Offset for pagination (optional, default=0)
    
    Returns:
        JSON response with search results
    """
    query = request.args.get('q')
    if not query:
        return jsonify({
            'error': 'Query parameter "q" is required'
        }), 400
    
    try:
        count = int(request.args.get('count', 10))
        offset = int(request.args.get('offset', 0))
    except ValueError:
        return jsonify({
            'error': 'Count and offset must be integers'
        }), 400
    
    if count > 50:
        return jsonify({
            'error': 'Count cannot exceed 50'
        }), 400
    
    if offset < 0:
        return jsonify({
            'error': 'Offset cannot be negative'
        }), 400
    
    results = scraper.search(query, count, offset)
    return jsonify(results)

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'Bing Search Scraper API'
    })

@app.route('/', methods=['GET'])
def index():
    """API documentation endpoint"""
    return jsonify({
        'name': 'Bing Search Scraper API',
        'version': '1.0.0',
        'endpoints': {
            '/search': {
                'method': 'GET',
                'description': 'Search Bing and return results',
                'parameters': {
                    'q': 'Search query (required)',
                    'count': 'Number of results (optional, default=10, max=50)',
                    'offset': 'Offset for pagination (optional, default=0)'
                },
                'example': '/search?q=python%20programming&count=5'
            },
            '/health': {
                'method': 'GET',
                'description': 'Health check endpoint'
            }
        }
    })


# Example usage as a standalone script
def example_usage():
    """Example of how to use the scraper directly"""
    scraper = BingSearchScraper()
    
    # Search for Python programming
    search_results = scraper.search("Python programming", count=5)
    
    output_dir = "search_content"
    
    print(f"Query: {search_results['query']}")
    print(f"Total results indicator: {search_results.get('total_results', 'N/A')}")
    print(f"Retrieved: {search_results['results_count']} results")
    print("\nResults:")
    
    for i, result in enumerate(search_results['results'], 1):
        print(f"\n{i}. {result['title']}")
        print(f"   URL: {result['url']}")
        print(f"   Description: {result['description'][:100]}...")
    
    if search_results['results']:
        print(f"\nDownloading and saving content to '{output_dir}' directory...")
        # Download and save content from results
        saved_files = scraper.download_and_save_results(search_results['results'], output_dir)
        print(f"\nSaved {len(saved_files)} files to '{output_dir}' directory:")
        for file in saved_files:
            print(f"- {file}")
    else:
        print("No results to download content from.")


if __name__ == '__main__':
    # For development only
    # app.run(debug=True, host='0.0.0.0', port=5000)

    # scraper = BingSearchScraper()
    # results = scraper.search("Python Programming", count=5)
    # print(results, "\n")
    example_usage()