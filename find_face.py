# import cv2
# import numpy as np

# # 定義一個函數來檢測並裁剪面部，並標示出面部區域
# def detect_and_crop_face(image_path):
#     image = cv2.imread(image_path)
    
#     if image is None:
#         print(f"Failed to load image: {image_path}")
#         return None
    
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
#     faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
#     # 如果檢測到面部，標示出面部區域並裁剪第一個檢測到的面部
#     for (x, y, w, h) in faces:
#         face = image[y:y+h, x:x+w]
#         # 標示出面部區域
#         cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
#         # 顯示帶有標示的圖像
#         cv2.imshow('Detected Face', image)
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()
        
#         return face
    
#     # 如果未檢測到面部，返回原始圖像並打印消息
#     print("No face detected.")
#     return image

# # 測試檢測並裁剪面部的函數
# image_path = 'D:\\numerical\\final_project_linear\\nba_players\Stephen_Curry\Stephen_Curry44.jpg'
# face_image = detect_and_crop_face(image_path)

# if face_image is not None:
#     # 顯示裁剪後的面部圖像
#     cv2.imshow('Cropped Face', face_image)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
# else:
#     print("Failed to detect and crop face.")
import os
import cv2
import shutil

# 定義一個函數來檢測並裁剪面部，並標示出面部區域
def detect_and_crop_face(image_path):
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"Failed to load image: {image_path}")
        return None
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    # 如果檢測到面部，標示出面部區域並裁剪第一個檢測到的面部
    for (x, y, w, h) in faces:
        face = image[y:y+h, x:x+w]
        # 標示出面部區域
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        # 顯示帶有標示的圖像
        cv2.imshow('Detected Face', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        return face
    
    # 如果未檢測到面部，返回None
    return None

# 定義一個函數來處理資料夾中的圖片
def process_images_in_folder(folder_path, discard_folder_path):
    if not os.path.exists(discard_folder_path):
        os.makedirs(discard_folder_path)
    
    for file_name in os.listdir(folder_path):
        image_path = os.path.join(folder_path, file_name)
        
        if not os.path.isfile(image_path):
            continue
        
        face_image = detect_and_crop_face(image_path)
        
        if face_image is None:
            discard_path = os.path.join(discard_folder_path, file_name)
            shutil.move(image_path, discard_path)
            print(f"Moved {file_name} to discard folder.")

# 具體路徑設置
image_folder_path = 'D:\\numerical\\final_project_linear\\nba_players\\Luka_Doncic'
discard_folder_path = 'D:\\numerical\\final_project_linear\\nba_players\\discard'

# 處理資料夾中的圖片
process_images_in_folder(image_folder_path, discard_folder_path)

