# import cv2
# import numpy as np
# from sklearn.decomposition import PCA
# from sklearn.neighbors import KNeighborsClassifier

# # 定義一個函數來提取SIFT特徵
# def extract_sift_features(image_path):
#     # 加載圖像並轉換為灰度圖
#     image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
#     if image is None:
#         raise FileNotFoundError(f"Image file not found: {image_path}")
    
#     # 初始化SIFT檢測器
#     sift = cv2.SIFT_create()
    
#     # 檢測關鍵點和計算描述子
#     keypoints, descriptors = sift.detectAndCompute(image, None) 
    
#     return descriptors

# # 提取多張圖像的特徵
# descriptors1 = extract_sift_features('curry1.jpg')
# descriptors2 = extract_sift_features('LeBron_James2.jpg')

# # 使用PCA進行降維
# pca = PCA(n_components=50)
# reduced_descriptors1 = pca.fit_transform(descriptors1)
# reduced_descriptors2 = pca.fit_transform(descriptors2)

# # 特徵組合
# combined_features1 = np.sum(reduced_descriptors1, axis=0)
# combined_features2 = np.sum(reduced_descriptors2, axis=0)

# # 標籤數據（假設有兩個類別，0 和 1）
# labels = np.array([0, 1])

# # 特徵數據
# features = np.array([combined_features1, combined_features2])

# # 訓練KNN分類器
# knn = KNeighborsClassifier(n_neighbors=1)
# knn.fit(features, labels)

# # 測試新圖像
# new_descriptors = extract_sift_features('LeBron_James1.jpg')
# reduced_new_descriptors = pca.transform(new_descriptors)
# combined_new_features = np.sum(reduced_new_descriptors, axis=0)

# # 預測
# predicted_label = knn.predict([combined_new_features])
# print("Predicted Label:", predicted_label)

import cv2
import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 定義一個函數來提取SIFT特徵
def extract_sift_features(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image, None)
    return descriptors

# 定義每個人的圖像數量和標籤
people = {
    'Stephen_Curry': 6,
    'Lebron_James': 5
}
label_map = {'Stephen_Curry': 0, 'Lebron_James': 1}

# 初始化圖像路徑和標籤列表
image_paths = []
labels = []

# 生成圖像路徑和標籤
for person, num_images in people.items():
    for i in range(1, num_images + 1):
        image_path = f'nba_players/{person}/{person}{i}.jpg'
        image_paths.append(image_path)
        labels.append(label_map[person])

print(image_paths)
print(labels)

# 提取所有圖像的特徵
all_descriptors = []
for image_path in image_paths:
    descriptors = extract_sift_features(image_path)
    all_descriptors.append(descriptors)

print(all_descriptors)

# 使用PCA進行降維
pca = PCA(n_components=50)
reduced_descriptors = [pca.fit_transform(descriptor) for descriptor in all_descriptors]

# 特徵組合
combined_features = [np.sum(descriptor, axis=0) for descriptor in reduced_descriptors]

# 分割訓練集和測試集
X_train, X_test, y_train, y_test = train_test_split(combined_features, labels, test_size=0.2, random_state=42)

# 訓練KNN分類器
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# # 預測
# y_pred = knn.predict(X_test)

# # 計算準確率
# accuracy = accuracy_score(y_test, y_pred)
# print("Accuracy:", accuracy)


##################################################################
# # 測試新圖像
# new_descriptors = extract_sift_features('test.jpg')
# reduced_new_descriptors = pca.transform(new_descriptors)
# combined_new_features = np.sum(reduced_new_descriptors, axis=0)

# # 預測
# predicted_label = knn.predict([combined_new_features])
# print("Predicted Label:", predicted_label)
