import os
import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 定義每個類別的圖像資料夾和標籤
categories = {
    'landscape': 0,
    'animal': 1,
    'human': 2
}

# 初始化圖像路徑和標籤列表
image_paths = []
labels = []

# 生成圖像路徑和標籤
data_dir = 'images'
for category in categories.keys():
    dir_path = os.path.join(data_dir, category)
    for file_name in os.listdir(dir_path):
        image_path = os.path.join(dir_path, file_name)
        image_paths.append(image_path)
        labels.append(categories[category])

# 初始化 SIFT
sift = cv2.SIFT_create()

# 提取特徵
def extract_sift_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    return descriptors

# 從圖像路徑列表中提取特徵
def extract_features(image_paths):
    all_descriptors = []
    for path in image_paths:
        img = cv2.imread(path)
        if img is not None:
            descriptors = extract_sift_features(img)
            if descriptors is not None:
                all_descriptors.append(descriptors)
    return all_descriptors

# 提取特徵
all_descriptors = extract_features(image_paths)

# 將所有特徵堆疊成一個數組
X = np.vstack(all_descriptors)
y = np.array(labels)

# 切分訓練集和測試集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 訓練 SVM 模型
clf = SVC(kernel='linear')
clf.fit(X_train, y_train)

# 預測
y_pred = clf.predict(X_test)

# 評估模型
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

# # 分類新圖片
# def classify_image(image_path):
#     img = cv2.imread(image_path)
#     descriptors = extract_sift_features(img)
#     if descriptors is not None:
#         prediction = clf.predict(descriptors)
#         return prediction
#     return None

# # 測試新圖片
# test_image_path = 'path_to_new_image.jpg'
# result = classify_image(test_image_path)
# if result is not None:
#     if result[0] == 0:
#         print("這張圖片中沒有動物或人類")
#     elif result[0] == 1:
#         print("這張圖片中有動物")
#     elif result[0] == 2:
#         print("這張圖片中有人類")
# else:
#     print("無法識別特徵")

