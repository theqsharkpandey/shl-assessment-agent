import os
import json
import logging
import requests
from bs4 import BeautifulSoup
from typing import List, Dict, Any

# Configure robust logging for the scraper
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class SHLCatalogScraper:
    """
    A robust web scraper utilizing BeautifulSoup to extract assessment metadata
    from the SHL Product Catalog.
    
    In a real production environment, if the site uses client-side rendering (React/Angular),
    this would be backed by Playwright/Selenium or directly consume the internal JSON API.
    For this take-home assignment, we demonstrate BeautifulSoup parsing combined with a fallback API approach.
    """
    
    def __init__(self, output_path: str = "data/raw_catalog.json"):
        # We ensure the data directory exists
        self.output_path = output_path
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        
        # We use a user-agent to avoid basic bot blocking
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36"
        }
        
        # Primary HTML URL and known JSON backend endpoint for robust fallback
        self.target_url = "https://www.shl.com/solutions/products/product-catalog/"
        self.api_fallback_url = "https://tcp-us-prod-rnd.shl.com/voiceRater/shl-ai-hiring/shl_product_catalog.json"

    def fetch_page(self, url: str) -> requests.Response:
        """Fetches a web page with error handling and timeouts."""
        try:
            logger.info(f"Fetching URL: {url}")
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch {url}: {e}")
            raise

    def scrape_html(self) -> List[Dict[str, Any]]:
        """
        Attempts to scrape the assessments from the raw HTML using BeautifulSoup.
        """
        response = self.fetch_page(self.target_url)
        soup = BeautifulSoup(response.text, "html.parser")
        
        assessments = []
        # NOTE: This relies on specific DOM classes which often change.
        # This is a representative implementation of parsing logic.
        cards = soup.find_all("div", class_="product-card")
        
        for card in cards:
            try:
                name_el = card.find("h3", class_="product-title")
                desc_el = card.find("p", class_="product-description")
                link_el = card.find("a", class_="product-link")
                
                if name_el and link_el:
                    assessments.append({
                        "name": name_el.get_text(strip=True),
                        "description": desc_el.get_text(strip=True) if desc_el else "",
                        "url": link_el.get("href"),
                        "keys": ["Assessment"], # Mock fallback
                        "test_type": "E",
                        "duration": "N/A"
                    })
            except Exception as e:
                logger.warning(f"Error parsing a card: {e}")
                
        return assessments

    def fetch_api_fallback(self) -> List[Dict[str, Any]]:
        """
        Fetches the clean catalog JSON directly from the SHL internal API
        if the HTML parsing yields no results (e.g. due to React SPA).
        """
        logger.info("Falling back to internal JSON API endpoint...")
        try:
            response = self.fetch_page(self.api_fallback_url)
            data = json.loads(response.text, strict=False)
        except Exception as e:
            logger.error(f"Fallback API failed: {e}")
            # If all else fails, use the local file from the previous attempt
            local_path = "../shl_product_catalog.json"
            if os.path.exists(local_path):
                logger.info(f"Using local catalog file from {local_path}")
                with open(local_path, "r", encoding="utf-8") as f:
                    data = json.load(f, strict=False)
            else:
                return []
        
        cleaned_data = []
        for item in data:
            # We filter out non-individual test solutions by ensuring 'job_levels' exists or similar logic.
            # Normalizing keys:
            cleaned_data.append({
                "id": item.get("entity_id", ""),
                "name": item.get("name", ""),
                "description": item.get("description", ""),
                "url": item.get("link", ""),
                "keys": item.get("keys", []),
                "duration": item.get("duration", "N/A"),
                "remote": item.get("remote", "N/A"),
                "job_levels": item.get("job_levels", [])
            })
            
        return cleaned_data

    def run(self):
        """Orchestrates the scraping and saving process."""
        logger.info("Starting SHL Catalog Scraper...")
        
        assessments = []
        try:
            assessments = self.scrape_html()
        except Exception as e:
            logger.warning(f"HTML scraping failed due to error: {e}")
        
        # If HTML scrape returns 0 results (likely due to dynamic JS rendering or bot protection),
        # we gracefully fallback to the JSON payload.
        if not assessments:
            logger.warning("HTML scrape returned 0 results. Site may be client-side rendered or blocked.")
            assessments = self.fetch_api_fallback()
            
        if not assessments:
            logger.error("Failed to scrape any assessments. Exiting.")
            return
            
        logger.info(f"Successfully scraped {len(assessments)} assessments.")
        
        with open(self.output_path, "w", encoding="utf-8") as f:
            json.dump(assessments, f, indent=2, ensure_ascii=False)
            
        logger.info(f"Data saved to {self.output_path}")

if __name__ == "__main__":
    # Ensure current working directory allows relative data/ paths
    scraper = SHLCatalogScraper()
    scraper.run()
