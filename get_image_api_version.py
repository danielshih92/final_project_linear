from pexels_api import API
import requests
import os

PEXELS_API_KEY = 'api_key'  

def fetch_image_urls_from_pexels(query, max_links_to_fetch):
    headers = {
        'Authorization': PEXELS_API_KEY
    }
    search_url = f"https://api.pexels.com/v1/search?query={query}&per_page={max_links_to_fetch}"
    response = requests.get(search_url, headers=headers)
    data = response.json()
    image_urls = [photo['src']['original'] for photo in data['photos']]
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

if __name__ == "__main__":
    player_name_for_search = "female"
    player_name_for_file = "female"
    num_images = 130
    save_directory = "Human_beings/female"

    image_urls = fetch_image_urls_from_pexels(player_name_for_search, num_images)
    print(image_urls)
    download_images(player_name_for_file, image_urls, save_directory)
