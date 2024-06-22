import requests
from bs4 import BeautifulSoup
import os

def fetch_image_urls(query, max_links_to_fetch, headers):
    search_url = f"https://www.google.com/search?q={query}&tbm=isch"
    response = requests.get(search_url, headers=headers)
    soup = BeautifulSoup(response.text, 'html.parser')
    image_urls = []
    
    for img in soup.find_all('img', limit=max_links_to_fetch):
        img_url = img.get('src')
        if img_url and img_url.startswith('http'):
            image_urls.append(img_url)
    
    return image_urls

def download_images(player_name,image_urls, save_folder):
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

# # 設定搜索查詢和下載目標
if __name__ == "__main__":
    player_name_for_search = "Lebron James"
    player_name_for_file = "Lebron_James"
    num_images = 150
    save_directory = "nba_players/Lebron_James"

    # 設置請求頭以模擬瀏覽器
    headers = {'User-Agent': 'Mozilla/5.0'} 

    # 獲取圖片鏈接並下載圖片
    image_urls = fetch_image_urls(player_name_for_search, num_images, headers)
    print(image_urls)
    download_images(player_name_for_file, image_urls, save_directory)


