import os
import cv2
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

# 定義一個函數來檢測並裁剪面部
def detect_and_crop_face(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    # 如果檢測到面部，裁剪並返回第一個檢測到的面部
    for (x, y, w, h) in faces:
        face = image[y:y+h, x:x+w]
        return face
    
    # 如果未檢測到面部，返回原始圖像
    return image

# 定義一個函數來提取SIFT特徵
def extract_sift_features(image):
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image, None)
    return descriptors

# 定義每個人的圖像數量和標籤
players_name = {
    'Lebron_James': 68,
    'Stephen_Curry': 50,
    'Luka_Doncic': 49,
}
label_map = {'Lebron_James': 0, 'Stephen_Curry': 1, 'Luka_Doncic': 2}

# 初始化圖像路徑和標籤列表
image_paths = []
labels = []

# 生成圖像路徑和標籤
for player in players_name.keys():
    dir_path = f'nba_players/{player}'
    for file_name in os.listdir(dir_path):
        image_path = os.path.join(dir_path, file_name)
        image_paths.append(image_path)
        labels.append(label_map[player])

# 提取所有圖像的特徵
all_descriptors = []
augmented_labels = []
for image_path, label in zip(image_paths, labels):
    image = cv2.imread(image_path)
    
    # 檢查圖像是否讀取成功
    if image is None:
        print(f"Failed to load image: {image_path}")
        continue
    
    # 檢測並裁剪面部
    face_image = detect_and_crop_face(image)
    
    # 轉換為灰度圖像
    gray_face = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)

    descriptors = extract_sift_features(gray_face)
    if descriptors is not None:
        all_descriptors.append(descriptors)
        augmented_labels.append(label)

# 檢查提取的特徵是否為空
if len(all_descriptors) == 0:
    raise ValueError("No features extracted. Check image paths and feature extraction.")

# 使用PCA進行降維
pca = PCA(n_components=50)
reduced_descriptors = []
for descriptor in all_descriptors:
    if descriptor.shape[0] >= 50:  # 如果樣本數已經足夠，直接使用
        reduced_descriptor = pca.fit_transform(descriptor[:50])
    else:
        padded_descriptor = np.pad(descriptor, ((0, 50 - descriptor.shape[0]), (0, 0)), 'constant')
        reduced_descriptor = pca.fit_transform(padded_descriptor)
    reduced_descriptors.append(reduced_descriptor)

# 確保所有特徵形狀一致
combined_features = []
for descriptor in reduced_descriptors:
    combined_features.append(np.mean(descriptor, axis=0))

# 檢查特徵組合的結果
if len(combined_features) == 0:
    raise ValueError("No combined features. Check feature combination.")

# 數據標準化
scaler = StandardScaler()
combined_features = scaler.fit_transform(combined_features)

# 檢查標準化結果
if combined_features is None or len(combined_features) == 0:
    raise ValueError("Standardization failed. Check feature scaling.")

# 分割訓練集和測試集
X_train, X_test, y_train, y_test = train_test_split(combined_features, augmented_labels, test_size=0.2, random_state=6)

# 檢查訓練集和測試集大小
print(f"Training set size: {len(X_train)}, Testing set size: {len(X_test)}")

# 使用GridSearchCV來調整KNN模型的參數
param_grid = {'n_neighbors': [1, 3, 5, 7, 9], 'metric': ['euclidean', 'manhattan']}
grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5)
grid_search.fit(X_train, y_train)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best cross-validation score: {grid_search.best_score_}")

# 使用最佳參數重新訓練模型
knn = KNeighborsClassifier(n_neighbors=grid_search.best_params_['n_neighbors'],
                           metric=grid_search.best_params_['metric'])
knn.fit(X_train, y_train)

# 預測
y_pred = knn.predict(X_test)

# 計算準確率
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# 使用SVM進行分類
svm = SVC(kernel='linear')
svm.fit(X_train, y_train)

# 預測
y_pred_svm = svm.predict(X_test)

# 計算準確率
accuracy_svm = accuracy_score(y_test, y_pred_svm)
print("SVM Accuracy:", accuracy_svm)

# 保存模型
# import joblib
# joblib.dump(knn, 'knn_model.pkl')
# joblib.dump(scaler, 'scaler.pkl')
# joblib.dump(pca, 'pca_model.pkl')

# # 測試新圖像
# def classify_new_image(image_path):
#     image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
#     new_descriptors = extract_sift_features(image)
    
#     if new_descriptors is None:
#         print(f"No features extracted from {image_path}")
#         return None
    
#     if new_descriptors.shape[0] >= 50:
#         reduced_new_descriptors = pca.transform(new_descriptors[:50])
#     else:
#         padded_new_descriptors = np.pad(new_descriptors, ((0, 50 - new_descriptors.shape[0]), (0, 0)), 'constant')
#         reduced_new_descriptors = pca.transform(padded_new_descriptors)
    
#     combined_new_features = np.mean(reduced_new_descriptors, axis=0)
#     combined_new_features = scaler.transform([combined_new_features])
    
#     predicted_label = svm.predict(combined_new_features)
#     return predicted_label





