# -----======================================================
# Author: M. Arthur Dean 
# Created: 06/05/2025
# Last Update: 06/05/2025
# madean@snl.gov
# ProgramName: api search for duck duck go
# Version: 0.1
# ------======================================================

# env = firewatchXI [python 3.11.13] # ELEVEN


import time
import logging
import requests
import os
from urllib.parse import urlparse
from bs4 import BeautifulSoup
import hashlib

from duckduckgo_search import DDGS


# Begin logging at this point (Start of file)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')





if __name__ == "__main__":





    # Default test bench (provided by the module)

    results = DDGS().text("python programming", max_results=5)
    print(results)







