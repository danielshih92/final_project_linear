from PIL import Image, ImageOps, ImageEnhance
import os
import random
from concurrent.futures import ThreadPoolExecutor

def random_transform(image):
    """對圖片進行隨機變換操作"""
    operations = [lambda img: ImageOps.mirror(img),
                  lambda img: img.rotate(random.randint(-30, 30)),
                  lambda img: ImageEnhance.Color(img).enhance(random.uniform(0.5, 1.5)),
                  lambda img: ImageEnhance.Brightness(img).enhance(random.uniform(0.5, 1.5))]
    
    # 隨機選擇並應用操作
    for _ in range(random.randint(1, len(operations))):  # 隨機選擇操作次數
        operation = random.choice(operations)
        image = operation(image)
    
    # 隨機裁切
    if random.choice([True, False]):
        width, height = image.size
        new_width, new_height = int(width * random.uniform(0.7, 0.9)), int(height * random.uniform(0.7, 0.9))
        left = (width - new_width) // 2
        top = (height - new_height) // 2
        image = image.crop((left, top, left + new_width, top + new_height))
    
    return image

def process_image(filename, source_folder, target_folder):
    """處理單張圖片並儲存"""
    image_path = os.path.join(source_folder, filename)
    image = Image.open(image_path)
    image = random_transform(image)
    target_path = os.path.join(target_folder, filename)
    image.save(target_path)

def process_images(source_folder, target_folder, max_workers=4):
    """處理資料夾中的圖片並儲存到新的資料夾，使用多線程加速"""
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for filename in os.listdir(source_folder):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                executor.submit(process_image, filename, source_folder, target_folder)

# 設定原始和目標資料夾路徑
source_folder = 'Human_beings/male'
target_folder = 'Human_beings/male_changed'

# 處理圖片
process_images(source_folder, target_folder)




