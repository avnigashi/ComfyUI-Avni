import json
from urllib.parse import quote
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup

class ImageCrawlerNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "search_query": ("STRING", {"default": ""}),
                "max_results": ("INT", {"default": 10, "min": 1, "max": 50}),
                "search_engine": (["Google", "Bing", "Baidu"], {"default": "Google"}),
            },
            "optional": {
                "skip_result": ("INT", {"default": 0, "min": 0}),
                "color": ("STRING", {"default": ""}),
                "image_type": ("STRING", {"default": ""}),
                "safe_mode": ("BOOLEAN", {"default": False}),
                "face_only": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("results_json", "image_urls")
    FUNCTION = "crawl_images"
    CATEGORY = "Search"

    def crawl_images(self, search_query, max_results, search_engine, skip_result=0, color="", image_type="", safe_mode=False, face_only=False):
        try:
            # Setup Selenium WebDriver options
            options = Options()
            options.add_argument('--headless')
            options.add_argument('--no-sandbox')
            options.add_argument('--disable-dev-shm-usage')
            service = Service()
            driver = webdriver.Chrome(service=service, options=options)

            # Choose search engine and perform the search
            if search_engine == "Google":
                query_url = self.google_gen_query_url(search_query, face_only, safe_mode, image_type, color)
                driver.get(query_url)
                image_urls = self.google_image_url_from_webpage(driver)
            elif search_engine == "Bing":
                query_url = self.bing_gen_query_url(search_query, face_only, safe_mode, image_type, color)
                driver.get(query_url)
                image_urls = self.bing_image_url_from_webpage(driver)
            elif search_engine == "Baidu":
                query_url = self.baidu_gen_query_url(search_query, face_only, safe_mode, color)
                driver.get(query_url)
                image_urls = self.baidu_image_url_from_webpage(driver)

            driver.quit()

            # Limit results and prepare extracted results
            extracted_results = [{"url": url} for url in image_urls[skip_result:skip_result + max_results]]
            return (json.dumps(extracted_results), ",".join(image_urls[skip_result:skip_result + max_results]))

        except Exception as e:
            print(f"Error in ImageCrawlerNode: {str(e)}")
            return ("", "")

    @staticmethod
    def google_gen_query_url(keywords, face_only=False, safe_mode=False, image_type=None, color=None):
        base_url = "https://www.google.com/search?tbm=isch"
        keywords_str = "&q=" + quote(keywords)
        query_url = base_url + keywords_str

        params = []
        if face_only:
            params.append("itp:face")
        if image_type:
            params.append(f"itp:{image_type.lower()}")
        if color:
            params.append(f"ic:{color.lower()}")
        if params:
            query_url += "&tbs=" + ",".join(params)
        if safe_mode:
            query_url += "&safe=active"

        return query_url

    @staticmethod
    def google_image_url_from_webpage(driver):
        try:
            WebDriverWait(driver, 10).until(
                EC.presence_of_all_elements_located((By.CSS_SELECTOR, 'body'))
            )
            page_source = driver.page_source

            soup = BeautifulSoup(page_source, 'html.parser')
            img_tags = soup.find_all('img')

            image_urls = []
            for img in img_tags:
                src = img.get('src', '')
                if src and src.startswith('http') and not src.startswith('https://www.google.com'):
                    image_urls.append(src)

            return image_urls

        except Exception as e:
            print(f"Error extracting Google images: {str(e)}")
            return []

    @staticmethod
    def bing_gen_query_url(keywords, face_only=False, safe_mode=False, image_type=None, color=None):
        base_url = "https://www.bing.com/images/search?"
        keywords_str = "&q=" + quote(keywords)
        query_url = base_url + keywords_str
        filter_url = "&qft="

        if face_only:
            filter_url += "+filterui:face-face"
        if image_type:
            filter_url += "+filterui:photo-{}".format(image_type)
        if color:
            filter_url += f"+filterui:color2-{color.lower() if color.lower() == 'bw' else 'color'}"

        query_url += filter_url
        return query_url

    @staticmethod
    def bing_image_url_from_webpage(driver):
        image_urls = []
        driver.implicitly_wait(5)
        image_elements = driver.find_elements(By.CLASS_NAME, "iusc")
        for elem in image_elements:
            m_json_str = elem.get_attribute("m")
            try:
                m_json = json.loads(m_json_str)
                image_urls.append(m_json["murl"])
            except json.JSONDecodeError:
                continue
        return image_urls

    @staticmethod
    def baidu_gen_query_url(keywords, face_only=False, safe_mode=False, color=None):
        base_url = "https://image.baidu.com/search/index?tn=baiduimage"
        keywords_str = "&word=" + quote(keywords)
        query_url = base_url + keywords_str
        if face_only:
            query_url += "&face=1"
        color_code = {
            "white": 1024, "bw": 2048, "black": 512, "pink": 64, "blue": 16,
            "red": 1, "yellow": 2, "purple": 32, "green": 4, "teal": 8,
            "orange": 256, "brown": 128
        }
        if color:
            query_url += f"&ic={color_code.get(color.lower(), '')}"
        return query_url

    @staticmethod
    def baidu_image_url_from_webpage(driver):
        driver.implicitly_wait(5)
        image_elements = driver.find_elements(By.CLASS_NAME, "imgitem")
        return [elem.get_attribute("data-objurl") for elem in image_elements]

# Register the node with ComfyUI
NODE_CLASS_MAPPINGS = {"ImageCrawlerNode": ImageCrawlerNode}
NODE_DISPLAY_NAME_MAPPINGS = {"ImageCrawlerNode": "Image Crawler"}