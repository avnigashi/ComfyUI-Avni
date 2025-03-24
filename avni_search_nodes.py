import json
import requests
from io import BytesIO
from PIL import Image
import numpy as np
import torch
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import random
from urllib.parse import quote, urlencode
from typing import Dict, List, Tuple, Optional, Any

class PROMPT:
    def __init__(self, prompt_type: str, content: str):
        self.prompt_type = prompt_type
        self.content = content

    def to_dict(self) -> Dict[str, str]:
        return {
            "prompt_type": self.prompt_type,
            "content": self.content
        }

class BaseSearchNode:
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Dict[str, Any]]:
        return {
            "required": {
                "search_query": ("STRING", {"default": ""}),
                "search_engine": (["Google", "Bing", "DuckDuckGo", "Baidu"], {"default": "Google"}),
                "language": (["de-de", "en-us", "fr-fr", "es-es", "it-it", "zh-cn"], {"default": "en-us"}),
                "max_results": ("INT", {"default": 10, "min": 1, "max": 50}),
                "depth": ("INT", {"default": 1, "min": 1, "max": 5}),
            },
            "optional": {
                "random_seed": ("INT", {"default": -1, "min": -1, "max": 0xffffffffffffffff}),
                "skip_result": ("INT", {"default": 0, "min": 0}),
                "preprompt": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "IMAGE")
    RETURN_NAMES = ("results_json", "urls", "images")
    FUNCTION = "crawl_web"
    CATEGORY = "Search"

    def setup_webdriver(self) -> webdriver.Chrome:
        """Setup and return a configured Chrome WebDriver."""
        options = Options()
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--headless')  # Run in headless mode
        service = Service()
        return webdriver.Chrome(service=service, options=options)

    def generate_search_url(self, search_engine: str, content_type: str, query: str,
                            language: str, **kwargs) -> str:
        """Generate search URL based on engine and content type."""
        base_urls = {
            "Google": {
                "images": f"https://www.google.com/search?tbm=isch&hl={language}",
                "news": f"https://news.google.com/search?hl={language}&gl=US&ceid=US:{language}",
                "text": f"https://www.google.com/search?hl={language}",
            },
            "Bing": {
                "images": f"https://www.bing.com/images/search?setlang={language}",
                "news": f"https://www.bing.com/news/search?setlang={language}",
                "text": f"https://www.bing.com/search?setlang={language}",
            },
            "DuckDuckGo": {
                "images": "https://duckduckgo.com/?t=h_&iax=images&ia=images",
                "news": "https://duckduckgo.com/?t=h_&iar=news&ia=news",
                "text": "https://duckduckgo.com/",
            },
            "Baidu": {
                "images": "https://image.baidu.com/search/index?tn=baiduimage",
                "news": "https://news.baidu.com/ns",
                "text": "https://www.baidu.com/s",
            }
        }

        if search_engine == "Baidu":
            return self.generate_baidu_url(content_type, query, language, **kwargs)

        base_url = base_urls.get(search_engine, {}).get(content_type, "")
        params = {"q": query}

        # Add safe search parameter if enabled
        if kwargs.get("safe_mode"):
            params.update(self.get_safe_search_param(search_engine))

        return f"{base_url}&{urlencode(params)}"

    def get_safe_search_param(self, search_engine: str) -> Dict[str, str]:
        """Get safe search parameters for different search engines."""
        safe_search_params = {
            "Google": {"safe": "active"},
            "Bing": {"safesearch": "strict"},
            "DuckDuckGo": {"kp": "1"},
            "Baidu": {"safe": "1"}
        }
        return safe_search_params.get(search_engine, {})

    def generate_baidu_url(self, content_type: str, query: str, language: str, **kwargs) -> str:
        """Generate Baidu-specific search URLs."""
        if content_type == "images":
            url = f"https://image.baidu.com/search/index?tn=baiduimage&word={quote(query)}"
        elif content_type == "news":
            url = f"https://news.baidu.com/ns?word={quote(query)}&tn=news"
        else:  # text search
            url = f"https://www.baidu.com/s?wd={quote(query)}"

        # Add language parameter for Baidu
        if language == "zh-cn":
            url += "&ie=utf-8"
        else:
            url += "&ie=utf-8&lang=en"  # Force English results for non-Chinese queries

        return url

    @classmethod
    def IS_CHANGED(cls, search_query: str, max_results: int,
                   depth: int, **kwargs) -> Tuple[str, int, int, int, int]:
        """Track changes to determine when to re-run the node."""
        return (
            search_query,
            max_results,
            depth,
            kwargs.get('random_seed', -1),
            kwargs.get('skip_result', 0))

class SearchImagesNode(BaseSearchNode):
    def crawl_web(self, search_query: str, search_engine: str, language: str, max_results: int, depth: int, **kwargs) -> Tuple[str, str, torch.Tensor]:
        """Crawl web for images."""
        content_type = "images"
        try:
            driver = self.setup_webdriver()
            query_url = self.generate_search_url(search_engine, content_type, search_query, language, **kwargs)
            driver.get(query_url)

            results = self.extract_images(driver, search_engine)
            driver.quit()

            # Apply random seed if specified
            if kwargs.get('random_seed', -1) != -1:
                random.seed(kwargs['random_seed'])
                random.shuffle(results)

            # Apply skip and limit
            start = kwargs.get('skip_result', 0)
            end = start + max_results
            results = results[start:end]

            # Process results
            formatted_results = [
                PROMPT(
                    prompt_type=content_type,
                    content=json.dumps(result)
                ).to_dict()
                for result in results
            ]

            # Extract URLs
            urls = ",".join([
                result.get('link', result.get('url', ''))
                for result in results
                if isinstance(result, dict)
            ])

            # Handle images
            image_tensors = []
            for result in results:
                if isinstance(result, dict) and "url" in result:
                    img_tensor = self.download_and_process_image(result["url"])
                    if img_tensor is not None:
                        image_tensors.append(img_tensor)

            if image_tensors:
                try:
                    # Concatenate all image tensors along batch dimension
                    batch_tensor = torch.cat(image_tensors, dim=0)
                    return (json.dumps(formatted_results), urls, batch_tensor)
                except Exception as e:
                    print(f"Error concatenating image tensors: {str(e)}")
                    # Return first valid image if concatenation fails
                    return (json.dumps(formatted_results), urls, image_tensors[0])

            # Return empty tensor with correct dimensions if no images were successfully processed
            return (json.dumps(formatted_results), urls, torch.zeros((1, 512, 512, 3)))

        except Exception as e:
            print(f"Error in crawl_web: {str(e)}")
            # Return empty results with correctly dimensioned tensor
            return (json.dumps([]), "", torch.zeros((1, 512, 512, 3)))

    def download_and_process_image(self, url: str) -> Optional[torch.Tensor]:
        """Download and process an image from URL into a tensor."""
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                img = Image.open(BytesIO(response.content))
                img = img.convert('RGB')

                # Resize image to a standard size to ensure consistent tensor dimensions
                target_size = (512, 512)  # You can adjust this size as needed
                img = img.resize(target_size, Image.Resampling.LANCZOS)

                img_np = np.array(img)
                # Convert to torch tensor and normalize to 0-1 range
                img_tensor = torch.from_numpy(img_np).float() / 255.0
                # Rearrange dimensions to match ComfyUI format (B, H, W, C)
                img_tensor = img_tensor.unsqueeze(0)
                return img_tensor
            return None
        except Exception as e:
            print(f"Error downloading image {url}: {str(e)}")
            return None

    def extract_images(self, driver: webdriver.Chrome, search_engine: str) -> List[Dict[str, str]]:
        """Extract image results based on search engine."""
        extractors = {
            "Google": self.extract_google_images,
            "Bing": self.extract_bing_images,
            "DuckDuckGo": self.extract_duckduckgo_images,
            "Baidu": self.extract_baidu_images
        }

        extractor = extractors.get(search_engine)
        if not extractor:
            return []

        return extractor(driver)

    def extract_google_images(self, driver: webdriver.Chrome) -> List[Dict[str, str]]:
        """Extract images from Google image search."""
        try:
            WebDriverWait(driver, 10).until(
                EC.presence_of_all_elements_located((By.CSS_SELECTOR, 'img.rg_i'))
            )
            images = []
            for img in driver.find_elements(By.CSS_SELECTOR, 'img.rg_i'):
                try:
                    img.click()
                    large_img = WebDriverWait(driver, 2).until(
                        EC.presence_of_element_located((By.CSS_SELECTOR, 'img.n3VNCb'))
                    )
                    url = large_img.get_attribute('src')
                    if url and url.startswith('http'):
                        title = img.get_attribute('alt') or ''
                        images.append({
                            "url": url,
                            "title": title,
                            "source": driver.current_url
                        })
                except Exception:
                    continue
            return images
        except Exception as e:
            print(f"Error extracting Google images: {str(e)}")
            return []

    def extract_bing_images(self, driver: webdriver.Chrome) -> List[Dict[str, str]]:
        """Extract images from Bing image search."""
        try:
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CLASS_NAME, 'mimg'))
            )
            images = []
            for img in driver.find_elements(By.CLASS_NAME, 'mimg'):
                try:
                    url = img.get_attribute('src') or img.get_attribute('data-src')
                    if url and url.startswith('http'):
                        title = img.get_attribute('alt') or ''
                        images.append({
                            "url": url,
                            "title": title,
                            "source": img.get_attribute('data-sourceurl') or driver.current_url
                        })
                except Exception:
                    continue
            return images
        except Exception as e:
            print(f"Error extracting Bing images: {str(e)}")
            return []

    def extract_duckduckgo_images(self, driver: webdriver.Chrome) -> List[Dict[str, str]]:
        """Extract images from DuckDuckGo image search."""
        try:
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CLASS_NAME, 'tile--img__img'))
            )
            images = []
            for img in driver.find_elements(By.CLASS_NAME, 'tile--img__img'):
                try:
                    url = img.get_attribute('src') or img.get_attribute('data-src')
                    if url and url.startswith('http'):
                        title = img.get_attribute('alt') or ''
                        images.append({
                            "url": url,
                            "title": title,
                            "source": driver.current_url
                        })
                except Exception:
                    continue
            return images
        except Exception as e:
            print(f"Error extracting DuckDuckGo images: {str(e)}")
            return []

    def extract_baidu_images(self, driver: webdriver.Chrome) -> List[Dict[str, str]]:
        """Extract images from Baidu image search."""
        try:
            WebDriverWait(driver, 10).until(
                EC.presence_of_all_elements_located((By.CSS_SELECTOR, '.imgitem img'))
            )
            images = []
            for img in driver.find_elements(By.CSS_SELECTOR, '.imgitem img'):
                try:
                    url = img.get_attribute('data-imgurl') or img.get_attribute('src')
                    if url and url.startswith('http'):
                        title = img.get_attribute('alt') or ''
                        images.append({
                            "url": url,
                            "title": title,
                            "source": driver.current_url
                        })
                except Exception:
                    continue
            return images
        except Exception as e:
            print(f"Error extracting Baidu images: {str(e)}")
            return []

class SearchNewsNode(BaseSearchNode):
    def crawl_web(self, search_query: str, search_engine: str, language: str, max_results: int, depth: int, **kwargs) -> Tuple[str, str, torch.Tensor]:
        """Crawl web for news articles."""
        content_type = "news"
        try:
            driver = self.setup_webdriver()
            query_url = self.generate_search_url(search_engine, content_type, search_query, language, **kwargs)
            driver.get(query_url)

            results = self.extract_news(driver, search_engine, kwargs.get('preprompt', ''))
            driver.quit()

            # Apply random seed if specified
            if kwargs.get('random_seed', -1) != -1:
                random.seed(kwargs['random_seed'])
                random.shuffle(results)

            # Apply skip and limit
            start = kwargs.get('skip_result', 0)
            end = start + max_results
            results = results[start:end]

            # Process results
            formatted_results = [
                PROMPT(
                    prompt_type=content_type,
                    content=result.get('content', json.dumps(result))
                ).to_dict()
            for result in results
            ]

            # Extract URLs
            urls = ",".join([
                result.get('link', '')
                for result in results
                if isinstance(result, dict)
            ])

            # Return results and empty image tensor
            return (json.dumps(formatted_results), urls, torch.zeros((1, 512, 512, 3)))

        except Exception as e:
            print(f"Error in crawl_web: {str(e)}")
            # Return empty results with correctly dimensioned tensor
            return (json.dumps([]), "", torch.zeros((1, 512, 512, 3)))

    def extract_news(self, driver: webdriver.Chrome, search_engine: str, preprompt: str = "") -> List[Dict[str, str]]:
        """Extract news results based on search engine."""
        extractors = {
            "Google": self.extract_google_news,
            "Bing": self.extract_bing_news,
            "DuckDuckGo": self.extract_duckduckgo_news,
            "Baidu": self.extract_baidu_news
        }

        extractor = extractors.get(search_engine)
        if not extractor:
            return []

        return extractor(driver, preprompt)

    def extract_google_news(self, driver: webdriver.Chrome, preprompt: str = "") -> List[Dict[str, str]]:
        """Extract news from Google News search."""
        try:
            WebDriverWait(driver, 10).until(
                EC.presence_of_all_elements_located((By.TAG_NAME, 'article'))
            )
            news_items = []
            for article in driver.find_elements(By.TAG_NAME, 'article'):
                try:
                    title_elem = article.find_element(By.TAG_NAME, 'h3')
                    title = title_elem.text
                    link = title_elem.find_element(By.TAG_NAME, 'a').get_attribute('href')
                    time_elem = article.find_element(By.TAG_NAME, 'time')
                    source_elem = article.find_element(By.CSS_SELECTOR, 'div:last-child > div:first-child')
                    snippet_elem = article.find_element(By.CSS_SELECTOR, 'div:last-child > div:last-child')

                    news_items.append({
                        "title": title,
                        "link": link,
                        "time": time_elem.text if time_elem else "",
                        "source": source_elem.text if source_elem else "",
                        "snippet": snippet_elem.text if snippet_elem else "",
                        "content": f"{preprompt} {title} {snippet_elem.text if snippet_elem else ''}"
                    })
                except Exception:
                    continue
            return news_items
        except Exception as e:
            print(f"Error extracting Google news: {str(e)}")
            return []

    def extract_bing_news(self, driver: webdriver.Chrome, preprompt: str = "") -> List[Dict[str, str]]:
        """Extract news from Bing News search."""
        try:
            WebDriverWait(driver, 10).until(
                EC.presence_of_all_elements_located((By.CSS_SELECTOR, 'div.news-card'))
            )
            news_items = []
            for article in driver.find_elements(By.CSS_SELECTOR, 'div.news-card'):
                try:
                    title_elem = article.find_element(By.CSS_SELECTOR, 'a.title')
                    title = title_elem.text
                    link = title_elem.get_attribute('href')
                    source_elem = article.find_element(By.CSS_SELECTOR, 'div.source')
                    time_elem = article.find_element(By.CSS_SELECTOR, 'span.time')
                    snippet_elem = article.find_element(By.CSS_SELECTOR, 'div.snippet')

                    news_items.append({
                        "title": title,
                        "link": link,
                        "time": time_elem.text if time_elem else "",
                        "source": source_elem.text if source_elem else "",
                        "snippet": snippet_elem.text if snippet_elem else "",
                        "content": f"{preprompt} {title} {snippet_elem.text if snippet_elem else ''}"
                    })
                except Exception:
                    continue
            return news_items
        except Exception as e:
            print(f"Error extracting Bing news: {str(e)}")
            return []

    def extract_duckduckgo_news(self, driver: webdriver.Chrome, preprompt: str = "") -> List[Dict[str, str]]:
        """Extract news from DuckDuckGo News search."""
        try:
            WebDriverWait(driver, 10).until(
                EC.presence_of_all_elements_located((By.CSS_SELECTOR, 'div.result'))
            )
            news_items = []
            for article in driver.find_elements(By.CSS_SELECTOR, 'div.result'):
                try:
                    title_elem = article.find_element(By.CSS_SELECTOR, 'a.result__a')
                    title = title_elem.text
                    link = title_elem.get_attribute('href')
                    snippet_elem = article.find_element(By.CSS_SELECTOR, 'a.result__snippet')
                    source_elem = article.find_element(By.CSS_SELECTOR, 'span.result__extras__url')

                    news_items.append({
                        "title": title,
                        "link": link,
                        "source": source_elem.text if source_elem else "",
                        "snippet": snippet_elem.text if snippet_elem else "",
                        "content": f"{preprompt} {title} {snippet_elem.text if snippet_elem else ''}"
                    })
                except Exception:
                    continue
            return news_items
        except Exception as e:
            print(f"Error extracting DuckDuckGo news: {str(e)}")
            return []

    def extract_baidu_news(self, driver: webdriver.Chrome, preprompt: str = "") -> List[Dict[str, str]]:
        """Extract news from Baidu News search."""
        try:
            WebDriverWait(driver, 10).until(
                EC.presence_of_all_elements_located((By.CSS_SELECTOR, 'div.result'))
            )
            news_items = []
            for article in driver.find_elements(By.CSS_SELECTOR, 'div.result'):
                try:
                    title_elem = article.find_element(By.CSS_SELECTOR, 'h3 a')
                    title = title_elem.text
                    link = title_elem.get_attribute('href')
                    snippet_elem = article.find_element(By.CSS_SELECTOR, 'div.c-summary')
                    source_elem = article.find_element(By.CSS_SELECTOR, 'p.c-author')

                    news_items.append({
                        "title": title,
                        "link": link,
                        "source": source_elem.text if source_elem else "",
                        "snippet": snippet_elem.text if snippet_elem else "",
                        "content": f"{preprompt} {title} {snippet_elem.text if snippet_elem else ''}"
                    })
                except Exception:
                    continue
            return news_items
        except Exception as e:
            print(f"Error extracting Baidu news: {str(e)}")
            return []

class SearchNode(BaseSearchNode):
    def crawl_web(self, search_query: str, search_engine: str, language: str, max_results: int, depth: int, **kwargs) -> Tuple[str, str, torch.Tensor]:
        """Crawl web for general text content."""
        content_type = "text"
        try:
            driver = self.setup_webdriver()
            query_url = self.generate_search_url(search_engine, content_type, search_query, language, **kwargs)
            driver.get(query_url)

            results = self.extract_text(driver, search_engine, kwargs.get('preprompt', ''))
            driver.quit()

            # Apply random seed if specified
            if kwargs.get('random_seed', -1) != -1:
                random.seed(kwargs['random_seed'])
                random.shuffle(results)

            # Apply skip and limit
            start = kwargs.get('skip_result', 0)
            end = start + max_results
            results = results[start:end]

            # Process results
            formatted_results = [
                PROMPT(
                    prompt_type=content_type,
                    content=result.get('content', json.dumps(result))
                ).to_dict()
            for result in results
            ]

            # Extract URLs
            urls = ",".join([
                result.get('link', '')
                for result in results
                if isinstance(result, dict)
            ])

            # Return results and empty image tensor
            return (json.dumps(formatted_results), urls, torch.zeros((1, 512, 512, 3)))

        except Exception as e:
            print(f"Error in crawl_web: {str(e)}")
            # Return empty results with correctly dimensioned tensor
            return (json.dumps([]), "", torch.zeros((1, 512, 512, 3)))

    def extract_text(self, driver: webdriver.Chrome, search_engine: str, preprompt: str = "") -> List[Dict[str, str]]:
        """Extract text results based on search engine."""
        extractors = {
            "Google": self.extract_google_text,
            "Bing": self.extract_bing_text,
            "DuckDuckGo": self.extract_duckduckgo_text,
            "Baidu": self.extract_baidu_text
        }

        extractor = extractors.get(search_engine)
        if not extractor:
            return []

        return extractor(driver, preprompt)

    def extract_google_text(self, driver: webdriver.Chrome, preprompt: str = "") -> List[Dict[str, str]]:
        """Extract text results from Google search."""
        try:
            WebDriverWait(driver, 10).until(
                EC.presence_of_all_elements_located((By.CSS_SELECTOR, 'div.g'))
            )
            results = []
            for result in driver.find_elements(By.CSS_SELECTOR, 'div.g'):
                try:
                    title_elem = result.find_element(By.TAG_NAME, 'h3')
                    title = title_elem.text
                    link = title_elem.find_element(By.XPATH, '..').get_attribute('href')
                    snippet_elem = result.find_element(By.CLASS_NAME, 'IsZvec')
                    snippet = snippet_elem.text

                    results.append({
                        "title": title,
                        "link": link,
                        "snippet": snippet,
                        "content": f"{preprompt} {title} {snippet}"
                    })
                except Exception:
                    continue
            return results
        except Exception as e:
            print(f"Error extracting Google text results: {str(e)}")
            return []

    def extract_bing_text(self, driver: webdriver.Chrome, preprompt: str = "") -> List[Dict[str, str]]:
        """Extract text results from Bing search."""
        try:
            WebDriverWait(driver, 10).until(
                EC.presence_of_all_elements_located((By.CSS_SELECTOR, 'li.b_algo'))
            )
            results = []
            for result in driver.find_elements(By.CSS_SELECTOR, 'li.b_algo'):
                try:
                    title_elem = result.find_element(By.TAG_NAME, 'h2')
                    title = title_elem.text
                    link = title_elem.find_element(By.TAG_NAME, 'a').get_attribute('href')
                    snippet_elem = result.find_element(By.CLASS_NAME, 'b_caption')
                    snippet = snippet_elem.text

                    results.append({
                        "title": title,
                        "link": link,
                        "snippet": snippet,
                        "content": f"{preprompt} {title} {snippet}"
                    })
                except Exception:
                    continue
            return results
        except Exception as e:
            print(f"Error extracting Bing text results: {str(e)}")
            return []

    def extract_duckduckgo_text(self, driver: webdriver.Chrome, preprompt: str = "") -> List[Dict[str, str]]:
        """Extract text results from DuckDuckGo search."""
        try:
            WebDriverWait(driver, 10).until(
                EC.presence_of_all_elements_located((By.CSS_SELECTOR, 'div.result'))
            )
            results = []
            for result in driver.find_elements(By.CSS_SELECTOR, 'div.result'):
                try:
                    title_elem = result.find_element(By.CSS_SELECTOR, 'a.result__a')
                    title = title_elem.text
                    link = title_elem.get_attribute('href')
                    snippet_elem = result.find_element(By.CSS_SELECTOR, 'a.result__snippet')
                    snippet = snippet_elem.text

                    results.append({
                        "title": title,
                        "link": link,
                        "snippet": snippet,
                        "content": f"{preprompt} {title} {snippet}"
                    })
                except Exception:
                    continue
            return results
        except Exception as e:
            print(f"Error extracting DuckDuckGo text results: {str(e)}")
            return []

    def extract_baidu_text(self, driver: webdriver.Chrome, preprompt: str = "") -> List[Dict[str, str]]:
        """Extract text results from Baidu search."""
        try:
            WebDriverWait(driver, 10).until(
                EC.presence_of_all_elements_located((By.CSS_SELECTOR, 'div.result'))
            )
            results = []
            for result in driver.find_elements(By.CSS_SELECTOR, 'div.result'):
                try:
                    title_elem = result.find_element(By.CSS_SELECTOR, 'h3 a')
                    title = title_elem.text
                    link = title_elem.get_attribute('href')
                    snippet_elem = result.find_element(By.CLASS_NAME, 'c-abstract')
                    snippet = snippet_elem.text

                    results.append({
                        "title": title,
                        "link": link,
                        "snippet": snippet,
                        "content": f"{preprompt} {title} {snippet}"
                    })
                except Exception:
                    continue
            return results
        except Exception as e:
            print(f"Error extracting Baidu text results: {str(e)}")
            return []

# Register the nodes with ComfyUI
NODE_CLASS_MAPPINGS = {
    "SearchImagesNode": SearchImagesNode,
    "SearchNewsNode": SearchNewsNode,
    "SearchNode": SearchNode
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "SearchImagesNode": "Search Images",
    "SearchNewsNode": "Search News",
    "SearchNode": "Search"
}
