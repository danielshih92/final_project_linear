###version1
import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

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
    if descriptors is None:
        return np.zeros((128,))  # 如果沒有檢測到特徵，返回一個零向量
    return np.mean(descriptors, axis=0)

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
        labels.append(player)

# 提取所有圖像的特徵
features = []
for image_path in image_paths:
    image = cv2.imread(image_path)
    
    # 檢查圖像是否讀取成功
    if image is None:
        print(f"Failed to load image: {image_path}")
        continue
    
    # 檢測並裁剪面部
    face_image = detect_and_crop_face(image)
    
    # 轉換為灰度圖像
    gray_face = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
    
    # 提取SIFT特徵
    feature_vector = extract_sift_features(gray_face)
    features.append(feature_vector)

# 將標籤轉換為數字
le = LabelEncoder()
labels_encoded = le.fit_transform(labels)

# 將特徵和標籤轉換為numpy數組
features = np.array(features)
labels_encoded = np.array(labels_encoded)

# 分割訓練集和測試集
X_train, X_test, y_train, y_test = train_test_split(features, labels_encoded, test_size=0.2, random_state=38)

# 定義CNN模型
model = models.Sequential()
model.add(layers.Input(shape=(128,)))
model.add(layers.Reshape((128, 1, 1)))
model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(layers.MaxPooling2D((2, 2), padding='same'))
model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(layers.MaxPooling2D((2, 2), padding='same'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(len(players_name), activation='softmax'))

# 編譯模型
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 訓練模型
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# 評估模型
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f'Test accuracy: {test_acc}')

# 預測
predictions = model.predict(X_test)
predicted_labels = np.argmax(predictions, axis=1)

# 計算準確率
accuracy = np.mean(predicted_labels == y_test)
print(f'Prediction accuracy: {accuracy}')











##################################################################
# # 測試新圖像
# new_descriptors = extract_sift_features('test.jpg')
# reduced_new_descriptors = pca.transform(new_descriptors)
# combined_new_features = np.sum(reduced_new_descriptors, axis=0)

# # 預測
# predicted_label = knn.predict([combined_new_features])
# print("Predicted Label:", predicted_label)
# import os
# import cv2
# import numpy as np
# from keras_vggface.vggface import VGGFace
# from keras_vggface.utils import preprocess_input
# from sklearn.preprocessing import LabelEncoder
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
# from tensorflow.keras import layers, models

# # 定義一個函數來檢測並裁剪面部
# def detect_and_crop_face(image):
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
#     faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
#     # 如果檢測到面部，裁剪並返回第一個檢測到的面部
#     for (x, y, w, h) in faces:
#         face = image[y:y+h, x:x+w]
#         return face
    
#     # 如果未檢測到面部，返回原始圖像
#     return image

# # 定義每個人的圖像數量和標籤
# players_name = {
#     'Lebron_James': 68,
#     'Stephen_Curry': 50,
# }
# label_map = {'Lebron_James': 0, 'Stephen_Curry': 1}

# # 初始化圖像路徑和標籤列表
# image_paths = []
# labels = []

# # 生成圖像路徑和標籤
# for player in players_name.keys():
#     dir_path = f'nba_players/{player}'
#     for file_name in os.listdir(dir_path):
#         image_path = os.path.join(dir_path, file_name)
#         image_paths.append(image_path)
#         labels.append(player)

# # 提取所有圖像的特徵
# features = []
# for image_path in image_paths:
#     image = cv2.imread(image_path)
    
#     # 檢查圖像是否讀取成功
#     if image is None:
#         print(f"Failed to load image: {image_path}")
#         continue
    
#     # 檢測並裁剪面部
#     face_image = detect_and_crop_face(image)
    
#     # 調整面部圖像大小為224x224（VGGFace輸入要求）
#     face_image = cv2.resize(face_image, (224, 224))
    
#     # 預處理圖像
#     face_image = face_image.astype('float32')
#     face_image = preprocess_input(face_image, version=2)
    
#     features.append(face_image)

# # 將標籤轉換為數字
# le = LabelEncoder()
# labels_encoded = le.fit_transform(labels)

# # 將特徵和標籤轉換為numpy數組
# features = np.array(features)
# labels_encoded = np.array(labels_encoded)

# # 分割訓練集和測試集
# X_train, X_test, y_train, y_test = train_test_split(features, labels_encoded, test_size=0.2, random_state=38)

# # 加載VGGFace模型
# base_model = VGGFace(include_top=False, input_shape=(224, 224, 3), pooling='avg')

# # 定義新模型
# model = models.Sequential()
# model.add(base_model)
# model.add(layers.Dense(256, activation='relu'))
# model.add(layers.Dropout(0.5))
# model.add(layers.Dense(len(players_name), activation='softmax'))

# # 編譯模型
# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# # 訓練模型
# model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# # 評估模型
# test_loss, test_acc = model.evaluate(X_test, y_test)
# print(f'Test accuracy: {test_acc}')

# # 預測
# predictions = model.predict(X_test)
# predicted_labels = np.argmax(predictions, axis=1)

# # 計算準確率
# accuracy = np.mean(predicted_labels == y_test)
# print(f'Prediction accuracy: {accuracy}')


