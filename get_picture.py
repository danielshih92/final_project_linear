################################(version 1)#########################################
# from selenium import webdriver
# from selenium.webdriver.common.keys import Keys
# from selenium.webdriver.edge.service import Service
# from selenium.webdriver.common.by import By
# from concurrent.futures import ThreadPoolExecutor
# import time
# import requests
# import os

# # # def fetch_image_urls(query, max_links_to_fetch, headers):
# # #     search_url = f"https://www.google.com/search?q={query}&tbm=isch"
# # #     response = requests.get(search_url, headers=headers)
# # #     soup = BeautifulSoup(response.text, 'html.parser')
# # #     image_urls = []
    
# # #     for img in soup.find_all('img', limit=max_links_to_fetch):
# # #         img_url = img.get('src')
# # #         if img_url and img_url.startswith('http'):
# # #             image_urls.append(img_url)
    
# # #     return image_urls

# def fetch_image_urls(query, num_images, driver_path):
#     # 設置EdgeDriver的路徑
#     options = webdriver.EdgeOptions()
#     options.add_argument('--headless')  # 啟用無頭模式
#     service = Service(executable_path=driver_path)
#     driver = webdriver.Edge(service=service, options=options)

#     # 打開Google圖片搜索
#     search_url = f"https://www.google.com/search?q={query}&tbm=isch"
#     driver.get(search_url)

#     # 模擬滾動以加載更多圖片
#     elem = driver.find_element(By.TAG_NAME, "body")
#     while len(driver.find_elements(By.CSS_SELECTOR, "img.mimg.vimgld")) < num_images:
#         elem.send_keys(Keys.PAGE_DOWN)
#         time.sleep(0.2)  # 稍等片刻讓頁面加載

#     # 提取圖片URL
#     image_elements = driver.find_elements(By.CSS_SELECTOR, "img.mimg.vimgld")
#     image_urls = [img.get_attribute('src') for img in image_elements[:num_images]]

#     driver.quit()
#     return image_urls

# def download_image(url, save_path):
#     try:
#         response = requests.get(url, timeout=10)  # 增加超時設置
#         if response.status_code == 200:
#             with open(save_path, 'wb') as f:
#                 f.write(response.content)
#         else:
#             print(f"錯誤: 無法下載 {url}，狀態碼 {response.status_code}")
#     except requests.RequestException as e:
#         print(f"錯誤: 下載時發生異常 {url}，異常信息 {e}")

# def download_images(names, urls, save_directory):
#     with ThreadPoolExecutor(max_workers=5) as executor:
#         for name, url in zip(names, urls):
#             save_path = f"{save_directory}/{name}.jpg"
#             executor.submit(download_image, url, save_path)

# # # 設定搜索查詢和下載目標
# if __name__ == "__main__":
#     player_name_for_search = "Lebron James"
#     player_name_for_file = "aaa"
#     num_images = 30
#     save_directory = "nba_players/Lebron_James"
#     driver_path = "D:\\numerical\\final_project_linear\\edgedriver_win64\\msedgedriver.exe"


#     # # 設置請求頭以模擬瀏覽器(this part is for BeautifulSoup)
#     # headers = {'User-Agent': 'Mozilla/5.0'} 

#     # 獲取圖片鏈接並下載圖片
#     image_urls = fetch_image_urls(player_name_for_search, num_images, driver_path)
#     print(image_urls)
#     download_images(player_name_for_file, image_urls, save_directory)
################################(version 2)#########################################
# from selenium import webdriver
# from selenium.webdriver.edge.service import Service
# from selenium.webdriver.common.by import By
# from selenium.webdriver.common.keys import Keys
# import time
# import os
# import requests

# def fetch_image_urls(query, max_links_to_fetch, wd, sleep_between_interactions=1):
#     def scroll_to_end(wd):
#         wd.execute_script("window.scrollTo(0, document.body.scrollHeight);")
#         time.sleep(sleep_between_interactions)

#     search_url = f"https://www.google.com/search?q={query}&tbm=isch"
#     wd.get(search_url)

#     image_urls = set()
#     image_count = 0
#     results_start = 0

#     while image_count < max_links_to_fetch:
#         scroll_to_end(wd)

#         thumbnail_results = wd.find_elements(By.CSS_SELECTOR, "img.Q4LuWd")
#         number_results = len(thumbnail_results)

#         for img in thumbnail_results[results_start:number_results]:
#             try:
#                 img.click()
#                 time.sleep(sleep_between_interactions)
#             except Exception:
#                 continue

#             images = wd.find_elements(By.CSS_SELECTOR, "img.n3VNCb")
#             for image in images:
#                 if image.get_attribute('src') and 'http' in image.get_attribute('src'):
#                     image_urls.add(image.get_attribute('src'))

#             image_count = len(image_urls)

#             if len(image_urls) >= max_links_to_fetch:
#                 break
#         else:
#             results_start = len(thumbnail_results)

#     return list(image_urls)

# def download_images(player_name, image_urls, save_folder):
#     if not os.path.exists(save_folder):
#         os.makedirs(save_folder)
    
#     for i, url in enumerate(image_urls):
#         try:
#             response = requests.get(url)
#             with open(os.path.join(save_folder, f'{player_name}{i+1}.jpg'), 'wb') as file:
#                 file.write(response.content)
#             print(f"Downloaded {url}")
#         except Exception as e:
#             print(f"Failed to download {url} - {e}")

# if __name__ == "__main__":
#     player_name_for_search = "Lebron James"
#     player_name_for_file = "Lebron_James"
#     num_images = 50
#     save_directory = "nba_players/Lebron_James"
    
#     # 指定WebDriver的路径
#     driver_path = 'D:\\numerical\\final_project_linear\\edgedriver_win64\\msedgedriver.exe' 
#     service = Service(executable_path=driver_path)
#     wd = webdriver.Edge(service=service)

#     try:
#         image_urls = fetch_image_urls(player_name_for_search, num_images, wd)
#         print(f"\n\nHere is your urls: {image_urls}")
#         download_images(player_name_for_file, image_urls, save_directory)
#     finally:
#         wd.quit()
################################(version 3)#########################################
# from selenium import webdriver
# from selenium.webdriver.edge.service import Service
# from selenium.webdriver.common.by import By
# from selenium.webdriver.support.ui import WebDriverWait
# from selenium.webdriver.support import expected_conditions as EC
# from bs4 import BeautifulSoup
# import time
# import os
# import requests

# def scroll_to_end(wd, sleep_between_interactions=1):
#     wd.execute_script("window.scrollTo(0, document.body.scrollHeight);")
#     time.sleep(sleep_between_interactions)

# def fetch_image_urls(query, max_links_to_fetch, wd, sleep_between_interactions=1):
#     search_url = f"https://www.google.com/search?q={query}&tbm=isch"
#     wd.get(search_url)

#     image_urls = set()
#     seen_urls = set()
#     image_count = 0

#     while image_count < max_links_to_fetch:
#         scroll_to_end(wd, sleep_between_interactions)
#         time.sleep(sleep_between_interactions)
        
#         page_source = wd.page_source
#         soup = BeautifulSoup(page_source, 'html.parser')

#         for img in soup.find_all('img'):
#             img_url = img.get('src')
#             if img_url and img_url.startswith('http') and img_url not in seen_urls:
#                 seen_urls.add(img_url)
#                 image_urls.add(img_url)
#                 image_count = len(image_urls)
#                 if image_count >= max_links_to_fetch:
#                     break

#         try:
#             load_more_button = wd.find_element(By.CSS_SELECTOR, ".mye4qd")
#             if load_more_button:
#                 wd.execute_script("document.querySelector('.mye4qd').click();")
#         except Exception as e:
#             print("No more load button found or unable to click:", e)
#             break

#     return list(image_urls)

# def download_images(player_name, image_urls, save_folder):
#     if not os.path.exists(save_folder):
#         os.makedirs(save_folder)
    
#     for i, url in enumerate(image_urls):
#         try:
#             response = requests.get(url)
#             with open(os.path.join(save_folder, f'{player_name}{i+1}.jpg'), 'wb') as file:
#                 file.write(response.content)
#             print(f"Downloaded {url}")
#         except Exception as e:
#             print(f"Failed to download {url} - {e}")

# if __name__ == "__main__":  
#     player_name_for_search = "Lebron James"
#     player_name_for_file = "aa"
#     num_images = 80
#     save_directory = "nba_players/Lebron_James"
    
#     # 指定WebDriver的路径
#     driver_path = 'D:\\numerical\\final_project_linear\\edgedriver_win64\\msedgedriver.exe'  # 替换为你的msedgedriver路径
#     service = Service(executable_path=driver_path)
#     wd = webdriver.Edge(service=service)

#     try:
#         image_urls = fetch_image_urls(player_name_for_search, num_images, wd)
#         print(f"Found {len(image_urls)} image URLs")
#         print(image_urls)
#         download_images(player_name_for_file, image_urls, save_directory)
#     finally:
#         wd.quit()
################################(version 4)#########################################
from selenium import webdriver
from selenium.webdriver.edge.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import time
import os
import requests

def scroll_to_end(wd, sleep_between_interactions=1):
    wd.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(sleep_between_interactions)

def fetch_image_urls(query, max_links_to_fetch, wd, min_width=100, min_height=100, sleep_between_interactions=1):
    search_url = f"https://www.google.com/search?q={query}&tbm=isch"
    wd.get(search_url)

    image_urls = set()
    seen_urls = set()
    image_count = 0

    while image_count < max_links_to_fetch:
        scroll_to_end(wd, sleep_between_interactions)
        time.sleep(sleep_between_interactions)
        
        page_source = wd.page_source
        soup = BeautifulSoup(page_source, 'html.parser')

        for img in soup.find_all('img'):
            try:
                img_url = img.get('src')
                width = int(img.get('width'))
                height = int(img.get('height'))
                
                if img_url and img_url.startswith('http') and img_url not in seen_urls and width >= min_width and height >= min_height:
                    seen_urls.add(img_url)
                    image_urls.add(img_url)
                    image_count = len(image_urls)
                    if image_count >= max_links_to_fetch:
                        break
            except Exception as e:
                print(f"Skipping an element due to error: {e}")
                continue

        try:
            load_more_button = wd.find_element(By.CSS_SELECTOR, ".mye4qd")
            if load_more_button:
                wd.execute_script("document.querySelector('.mye4qd').click();")
        except Exception as e:
            print("No more load button found or unable to click:", e)
            break

    return list(image_urls)

def download_images(player_name, image_urls, save_folder):
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    
    for i, url in enumerate(image_urls):
        try:
            response = requests.get(url)
            with open(os.path.join(save_folder, f'{player_name}{i+1}.jpg'), 'wb') as file:
                file.write(response.content)
            print(f"Downloaded {url}")
        except Exception as e:
            print(f"Failed to download {url} - {e}")

if __name__ == "__main__":
    player_name_for_search = "triangle"
    player_name_for_file = "aa"
    num_images = 150
    save_directory = "shape/triangle"
    
    # 指定WebDriver的路径
    driver_path = 'D:\\numerical\\final_project_linear\\edgedriver_win64\\msedgedriver.exe'  # 替换为你的msedgedriver路径
    service = Service(executable_path=driver_path)
    wd = webdriver.Edge(service=service)

    try:
        image_urls = fetch_image_urls(player_name_for_search, num_images, wd)
        print(f"Found {len(image_urls)} image URLs")
        print(image_urls)
        download_images(player_name_for_file, image_urls, save_directory)
    finally:
        wd.quit()

