#!/usr/bin/env python3
"""
FINAL COMPLETE BING SCRAPER SOLUTION
====================================

This script provides a complete solution for scraping Bing search results and downloading content.
It replaces the previous bingSearch4.py script and addresses all known issues:
1. Bypasses Bing bot detection using browser automation.
2. Successfully scrapes and downloads content from search results.
3. Works with the existing searchFile.txt configuration.
4. Creates an organized folder structure under search_content/ for storing downloaded files.
5. Ensures functionality where the original script failed.

INSTALLATION:
To install the required dependencies, run:
pip install selenium beautifulsoup4 requests webdriver-manager

USAGE:
To execute the script, run:
python final_complete_solution.py
"""

import os
import sys
import time
import random
import logging
from typing import List, Dict, Optional
import traceback
import re

# Check and install dependencies
def check_dependencies():
    """Check if required packages are installed"""
    required_packages = {
        'selenium': 'selenium',
        'bs4': 'beautifulsoup4', 
        'requests': 'requests',
        'webdriver_manager': 'webdriver-manager'
    }
    
    missing_packages = []
    for module, package in required_packages.items():
        try:
            __import__(module)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"Missing required packages: {', '.join(missing_packages)}")
        print(f"Install with: pip install {' '.join(missing_packages)}")
        return False
    return True

if not check_dependencies():
    sys.exit(1)

# Now import after checking dependencies
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.common.exceptions import TimeoutException, WebDriverException
from bs4 import BeautifulSoup
import requests

try:
    from webdriver_manager.chrome import ChromeDriverManager
    AUTO_DRIVER = True
except ImportError:
    AUTO_DRIVER = False

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class FinalBingScraper:
    def __init__(self, headless=True):
        """Initialize the final working Bing scraper"""
        self.headless = headless
        self.driver = None
        self.wait = None
        
    def setup_driver(self):
        """Setup Chrome driver with robust configuration"""
        chrome_options = Options()
        
        # Essential anti-detection settings
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        chrome_options.add_argument('--disable-blink-features=AutomationControlled')
        chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
        chrome_options.add_experimental_option('useAutomationExtension', False)
        chrome_options.add_argument('--disable-web-security')
        chrome_options.add_argument('--allow-running-insecure-content')
        chrome_options.add_argument('--disable-extensions')
        
        # Realistic browser settings
        chrome_options.add_argument('--window-size=1920,1080')
        chrome_options.add_argument('--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')
        
        if self.headless:
            chrome_options.add_argument('--headless')
            
        # Performance optimizations
        chrome_options.add_argument('--disable-images')
        chrome_options.add_argument('--disable-javascript')  # JavaScript not needed for search results
        
        try:
            if AUTO_DRIVER:
                service = Service(ChromeDriverManager().install())
                self.driver = webdriver.Chrome(service=service, options=chrome_options)
            else:
                self.driver = webdriver.Chrome(options=chrome_options)
                
            # Anti-detection JavaScript
            self.driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
            
            self.wait = WebDriverWait(self.driver, 20)
            logging.info("Chrome driver setup successful")
            return True
            
        except Exception as e:
            logging.error(f"Chrome driver setup failed: {e}")
            print("\nDRIVER SETUP TROUBLESHOOTING:")
            print("1. Install Chrome browser: https://www.google.com/chrome/")
            print("2. Install ChromeDriver: pip install webdriver-manager")
            print("3. Or download manually: https://chromedriver.chromium.org/")
            return False
    
    def search_and_download(self, query: str, output_dir: str = None, max_results: int = 10) -> Dict:
        """Complete search and download process"""
        if not self.setup_driver():
            return {'success': False, 'error': 'Driver setup failed'}
            
        try:
            logging.info(f"Starting search for: {query}")
            
            # Step 1: Navigate to Bing
            self.driver.get("https://www.bing.com")
            time.sleep(random.uniform(2, 4))
            
            # Step 2: Perform search
            search_box = self.wait.until(EC.presence_of_element_located((By.NAME, "q")))
            search_box.clear()
            
            # Type query naturally
            for char in query:
                search_box.send_keys(char)
                time.sleep(random.uniform(0.05, 0.15))
                
            search_box.submit()
            time.sleep(random.uniform(3, 6))
            
            # Step 3: Extract results
            results = self._extract_all_results(max_results)
            
            if not results:
                return {'success': False, 'error': 'No search results found'}
                
            logging.info(f"Found {len(results)} search results")
            
            # Step 4: Download content
            saved_files = self._download_all_content(results, output_dir)
            
            return {
                'success': True,
                'query': query,
                'results_count': len(results),
                'results': results,
                'saved_files': saved_files,
                'downloaded_count': len(saved_files),
                'output_directory': output_dir
            }
            
        except Exception as e:
            logging.error(f"Search process failed: {e}")
            return {'success': False, 'error': str(e)}
        
        finally:
            if self.driver:
                self.driver.quit()
    
    def _extract_all_results(self, max_results: int) -> List[Dict]:
        """Extract all search results using multiple strategies"""
        results = []
        
        try:
            # Wait for results to load
            self.wait.until(EC.presence_of_element_located((By.ID, "b_results")))
            
            # Strategy 1: Standard Bing selectors
            selectors = [
                'li.b_algo',
                'li[class*="algo"]', 
                '.b_algo',
                'div.b_algo'
            ]
            
            result_elements = []
            for selector in selectors:
                try:
                    elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
                    if elements:
                        result_elements = elements
                        logging.info(f"Found {len(elements)} results with selector: {selector}")
                        break
                except:
                    continue
            
            # Strategy 2: Fallback to any reasonable containers
            if not result_elements:
                logging.info("Trying fallback extraction...")
                all_containers = self.driver.find_elements(By.CSS_SELECTOR, '#b_results li, #b_results div')
                result_elements = [elem for elem in all_containers if self._is_valid_result(elem)]
                logging.info(f"Fallback found {len(result_elements)} results")
            
            # Extract data from each element
            for i, element in enumerate(result_elements[:max_results]):
                try:
                    result = self._extract_single_result(element, i + 1)
                    if result:
                        results.append(result)
                except Exception as e:
                    logging.warning(f"Failed to extract result {i+1}: {e}")
                    continue
                    
            # Save page source for debugging
            with open('final_scraper_results.html', 'w', encoding='utf-8') as f:
                f.write(self.driver.page_source)
            logging.info("Saved page source to final_scraper_results.html")
            
        except TimeoutException:
            logging.error("Timeout waiting for search results")
        except Exception as e:
            logging.error(f"Result extraction failed: {e}")
            
        return results
    
    def _is_valid_result(self, element) -> bool:
        """Check if element is a valid search result"""
        try:
            text = element.text.strip()
            links = element.find_elements(By.TAG_NAME, 'a')
            
            # Must have text and at least one external link
            has_text = len(text) > 30
            has_external_link = any(
                link.get_attribute('href') and 
                link.get_attribute('href').startswith('http') and
                'bing.com' not in link.get_attribute('href')
                for link in links
            )
            
            return has_text and has_external_link
        except:
            return False
    
    def _extract_single_result(self, element, rank: int) -> Optional[Dict]:
        """Extract data from a single result element"""
        try:
            title = ""
            url = ""
            description = ""
            
            # Extract title and URL from links
            links = element.find_elements(By.TAG_NAME, 'a')
            for link in links:
                href = link.get_attribute('href')
                text = link.text.strip()
                
                if (href and href.startswith('http') and 
                    'bing.com' not in href and text and len(text) > 5):
                    title = text
                    url = href
                    break
            
            if not title or not url:
                return None
            
            # Extract description from element text
            element_text = element.text.strip()
            if len(element_text) > len(title) + 20:
                description = element_text.replace(title, '').strip()
                # Clean up description
                description = ' '.join(description.split())[:300]
            
            return {
                'rank': rank,
                'title': title.strip(),
                'url': url.strip(),
                'description': description.strip()
            }
            
        except Exception as e:
            logging.warning(f"Error extracting result {rank}: {e}")
            return None
    
    def _download_all_content(self, results: List[Dict], output_dir: str = None) -> List[str]:
        """Download content from all result URLs"""
        saved_files = []
        
        # Use provided output directory or default
        if output_dir is None:
            output_dir = "ajax_tocco_content"
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            logging.info(f"Created output directory: {output_dir}")
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        }
        
        for result in results:
            try:
                rank = result['rank']
                title = result['title']
                url = result['url']
                
                logging.info(f"Downloading [{rank}]: {title[:50]}...")
                
                # Download page content
                response = requests.get(url, headers=headers, timeout=15)
                response.raise_for_status()
                
                # Extract text content
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Remove unwanted elements
                for element in soup(['script', 'style', 'nav', 'footer', 'header']):
                    element.decompose()
                
                # Get clean text
                text_content = soup.get_text(separator='\n')
                lines = (line.strip() for line in text_content.splitlines())
                text_content = '\n'.join(line for line in lines if line)
                
                # Save to file
                safe_title = ''.join(c if c.isalnum() or c in [' ', '_', '-'] else '_' for c in title)
                safe_title = safe_title[:50]
                filename = f"{rank:03d}_{safe_title}.txt"
                filepath = os.path.join(output_dir, filename)
                
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(f"RANK: {rank}\n")
                    f.write(f"TITLE: {title}\n")
                    f.write(f"URL: {url}\n")
                    f.write(f"DESCRIPTION: {result.get('description', '')}\n")
                    f.write("=" * 80 + "\n\n")
                    f.write(text_content)
                
                saved_files.append(filepath)
                logging.info(f"Saved: {filepath}")
                
                # Rate limiting
                time.sleep(random.uniform(1, 3))
                
            except Exception as e:
                logging.error(f"Failed to download {url}: {e}")
                continue
        
        return saved_files


class SearchConfiguration:
    """Handle search configuration from searchFile.txt"""
    def __init__(self, filename="searchFile.txt"):
        self.topic = ""
        self.keyword_include = []
        self.keyword_exclude = []
        self.domains = []
        self._load_config(filename)
    
    def _load_config(self, filename):
        """Load configuration from file"""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                content = f.read()
            
            current_section = None
            for line in content.split('\n'):
                line = line.strip()
                if not line:
                    continue
                
                if line.endswith(':'):
                    current_section = line[:-1].lower()
                else:
                    if current_section == "topic":
                        self.topic = line
                    elif current_section == "keywordinclude":
                        self.keyword_include.append(line)
                    elif current_section == "keywordexclude":
                        self.keyword_exclude.append(line)
                    elif current_section == "domains":
                        self.domains.append(line)
            
            logging.info(f"Loaded configuration from {filename}")
            
        except FileNotFoundError:
            logging.error(f"Configuration file {filename} not found")
            # Use default configuration
            self.topic = "Ajax TOCCO Magnethermic"
            self.keyword_include = ["Production", "Hydrogen"]
        except Exception as e:
            logging.error(f"Error loading configuration: {e}")
    
    def build_query(self) -> str:
        """Build search query from configuration"""
        query = self.topic
        
        # Add include keywords
        for keyword in self.keyword_include:
            query += f" {keyword}"
        
        # Add exclude keywords
        for keyword in self.keyword_exclude:
            query += f" -{keyword}"
        
        return query.strip()
    
    def get_safe_topic_name(self) -> str:
        """Get filesystem-safe version of topic name"""
        # Remove or replace problematic characters
        safe_name = re.sub(r'[<>:"/\\|?*]', '_', self.topic)
        safe_name = re.sub(r'\s+', '_', safe_name)  # Replace spaces with underscores
        safe_name = safe_name.strip('_')  # Remove leading/trailing underscores
        
        # Ensure it's not empty
        if not safe_name:
            safe_name = "search_results"
            
        return safe_name
    
    def get_output_directory(self) -> str:
        """Get the output directory path for this search"""
        base_dir = "search_content"
        topic_dir = self.get_safe_topic_name()
        return os.path.join(base_dir, topic_dir)
    
    def display(self):
        """Display current configuration"""
        print("SEARCH CONFIGURATION:")
        print("-" * 25)
        print(f"Topic: {self.topic}")
        print(f"Include Keywords: {self.keyword_include}")
        print(f"Exclude Keywords: {self.keyword_exclude}")
        print(f"Domains: {self.domains}")
        print(f"Final Query: {self.build_query()}")
        print(f"Output Directory: {self.get_output_directory()}")


def main():
    """Main execution function"""
    print("FINAL COMPLETE BING SCRAPER SOLUTION")
    print("=" * 50)
    print("This script replaces the previous bingSearch4.py with a working solution.")
    print()
    
    try:
        # Load search configuration
        config = SearchConfiguration("searchFile.txt")
        config.display()
        print()
        
        # Initialize scraper
        scraper = FinalBingScraper(headless=True)  # Set False to see browser
        
        # Build query and get output directory
        query = config.build_query()
        output_dir = config.get_output_directory()
        
        print(f"Executing search: {query}")
        print(f"Output directory: {output_dir}")
        print()
        
        # Perform complete search and download
        result = scraper.search_and_download(query, output_dir, max_results=10)
        
        if result['success']:
            print("COMPLETE SUCCESS!")
            print(f"   Found {result['results_count']} search results")
            print(f"   Downloaded {result['downloaded_count']} files")
            print(f"   Files saved to: {result['output_directory']}/")
            print()
            
            print("DOWNLOADED FILES:")
            for filepath in result['saved_files']:
                filename = os.path.basename(filepath)
                print(f"   {filename}")
            
            print()
            print("SEARCH RESULTS FOUND:")
            for i, res in enumerate(result['results'][:5], 1):
                print(f"   {i}. {res['title'][:60]}...")
                print(f"      {res['url'][:70]}...")
            
            if len(result['results']) > 5:
                print(f"   ... and {len(result['results']) - 5} more results")
                
        else:
            print(f"SEARCH FAILED: {result.get('error', 'Unknown error')}")
            print()
            print("TROUBLESHOOTING STEPS:")
            print("1. Ensure Chrome browser is installed")
            print("2. Check internet connection")
            print("3. Try with headless=False to see what's happening")
            print("4. Check logs above for specific errors")
    
    except Exception as e:
        logging.error(f"Fatal error: {e}")
        print(f"\nFATAL ERROR: {e}")
        print("\nFull traceback:")
        traceback.print_exc()
        
    finally:
        print("\nScraper execution completed")


if __name__ == "__main__":
    main()