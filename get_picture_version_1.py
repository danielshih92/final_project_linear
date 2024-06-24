from bs4 import BeautifulSoup
import requests
import os

def fetch_image_urls(query, max_links_to_fetch, headers):
    search_url = f"https://www.google.com/search?q={query}&tbm=isch"
    image_urls = []
    seen_urls = set()

    while len(image_urls) < max_links_to_fetch:
        response = requests.get(search_url, headers=headers)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        for img in soup.find_all('img'):
            img_url = img.get('src')
            if img_url and img_url.startswith('http') and img_url not in seen_urls:
                seen_urls.add(img_url)
                image_urls.append(img_url)
                if len(image_urls) >= max_links_to_fetch:
                    break
        
        next_page_element = soup.find('a', {'aria-label': 'Next page'})
        if next_page_element:
            search_url = 'https://www.google.com' + next_page_element['href']
        else:
            break
    
    return image_urls

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

# # 設定搜索查詢和下載目標
if __name__ == "__main__":
    player_name_for_search = "Lebron James"
    player_name_for_file = "aaa"
    num_images = 50
    save_directory = "nba_players/Lebron_James"

    # # 設置請求頭以模擬瀏覽器(this part is for BeautifulSoup)
    headers = {'User-Agent': 'Mozilla/5.0'}

    # 獲取圖片鏈接並下載圖片
    image_urls = fetch_image_urls(player_name_for_search, num_images, headers)
    print(image_urls)
    download_images(player_name_for_file, image_urls, save_directory)


