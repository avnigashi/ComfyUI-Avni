import json
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import random
import time
from nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

class PROMPT:
    def __init__(self, prompt_type, content):
        self.prompt_type = prompt_type
        self.content = content

    def to_dict(self):
        return {
            "prompt_type": self.prompt_type,
            "content": self.content
        }
class WebcrawlerNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "search_query": ("STRING", {"default": ""}),
                "search_type": (["news", "text"], {"default": "text"}),
                "language": (["de-de", "en-us"], {"default": "de-de"}),
                "max_results": ("INT", {"default": 10, "min": 1, "max": 50}),
                "depth": ("INT", {"default": 1, "min": 1, "max": 5}),
            },
            "optional": {
                "random_seed": ("INT", {"default": -1, "min": -1, "max": 0xffffffffffffffff}),
                "skip_result": ("INT", {"default": 0, "min": 0}),
                "preprompt": ("STRING", {"default": ""})
            }
        }

    RETURN_TYPES = ("PROMPT", "STRING")
    RETURN_NAMES = ("results_json", "urls")
    FUNCTION = "crawl_web"
    CATEGORY = "avni/Search"

    def crawl_web(self, search_query, search_type, max_results, depth, language, random_seed=-1, skip_result=0, preprompt=""):
        def recursive_crawl(query, search_type, results_limit, current_depth, max_depth):
            if current_depth > max_depth:
                return []

            try:
                options = Options()
                service = Service()
                driver = webdriver.Chrome(service=service, options=options)

                if search_type == "news":
                    url = f"https://news.google.com/search?q={query}&hl={language}&gl=US&ceid=US:{language}"
                else:
                    url = f"https://www.google.com/search?q={query}&hl={language}"

                driver.get(url)
                WebDriverWait(driver, 10).until(
                    EC.presence_of_all_elements_located((By.CSS_SELECTOR, 'body'))
                )
                page_source = driver.page_source

                soup = BeautifulSoup(page_source, 'html.parser')
                if search_type == "news":
                    articles = soup.find_all('article')
                else:
                    articles = soup.find_all('div', class_='g')

                extracted_results = []

                if search_type == "news":
                    for article in articles[skip_result:skip_result+results_limit]:
                        try:
                            title = article.find('h3').text
                            link = article.find('a')['href']
                            if link.startswith('./'):
                                link = 'https://news.google.com' + link[1:]
                            source = article.find('time').parent.text
                            extracted_results.append({"title": title, "link": link, "source": source, "content": f"{preprompt} {article.text}"})
                            extracted_results.extend(
                                recursive_crawl(title, search_type, results_limit, current_depth + 1, max_depth)
                            )
                        except:
                            continue
                else:
                    for result in articles[skip_result:skip_result+results_limit]:
                        title = result.find('h3')
                        link = result.find('a', href=True)
                        if title and link:
                            driver.get(link['href'])
                            page_soup = BeautifulSoup(driver.page_source, 'html.parser')
                            for script in page_soup(["script", "style"]):
                                script.decompose()
                            full_content = page_soup.get_text(separator=' ', strip=True)
                            extracted_results.append({
                                "title": f"Describe the Article: {title.text}",
                                "link": link['href'],
                                "content": f"{preprompt} {full_content}"
                            })
                            extracted_results.extend(
                                recursive_crawl(title.text, search_type, results_limit, current_depth + 1, max_depth)
                            )

                driver.quit()

                if random_seed != -1:
                    random.seed(random_seed)
                    random.shuffle(extracted_results)

                return extracted_results

            except Exception as e:
                print(f"Error in WebcrawlerNode: {str(e)}")
                return []

        final_results = recursive_crawl(search_query, search_type, max_results, 1, depth)
        urls = ",".join([result['link'] for result in final_results])
        return (json.dumps(final_results), urls)

    @classmethod
    def IS_CHANGED(cls, search_query, search_type, max_results, depth, random_seed, skip_result):
        return (search_query, search_type, max_results, depth, random_seed, skip_result)

NODE_CLASS_MAPPINGS["WebcrawlerNode"] = WebcrawlerNode
NODE_DISPLAY_NAME_MAPPINGS["WebcrawlerNode"] = "Web Crawler"
