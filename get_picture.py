import requests
import os

# 設定API Key和搜索引擎ID
api_key = 'YOUR_API_KEY'
search_engine_id = 'YOUR_SEARCH_ENGINE_ID'

def fetch_image_urls(query, num_images, api_key, search_engine_id):
    search_url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "q": query,
        "cx": search_engine_id,
        "key": api_key,
        "searchType": "image",
        "num": num_images
    }
    response = requests.get(search_url, params=params)
    results = response.json()
    image_urls = [item['link'] for item in results['items']]
    return image_urls

def download_images(image_urls, save_folder):
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    
    for i, url in enumerate(image_urls):
        try:
            response = requests.get(url)
            with open(os.path.join(save_folder, f'image_{i+1}.jpg'), 'wb') as file:
                file.write(response.content)
            print(f"Downloaded {url}")
        except Exception as e:
            print(f"Failed to download {url} - {e}")

# 設定搜索查詢和下載目標
player_name = "LeBron James"
num_images = 10
save_directory = "nba_players/lebron_james"

# 獲取圖片鏈接並下載圖片
image_urls = fetch_image_urls(player_name, num_images, api_key, search_engine_id)
download_images(image_urls, save_directory)
