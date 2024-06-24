# import tkinter as tk
# from tkinter import filedialog, messagebox
# from PIL import Image, ImageTk
# import cv2
# import numpy as np
# import joblib

# # 定義一個函數來提取SIFT特徵並顯示在圖片上
# def extract_and_display_sift(image_path):
#     # Load the image
#     image = cv2.imread(image_path)
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale

#     # Initialize SIFT detector
#     sift = cv2.SIFT_create()
#     keypoints, descriptors = sift.detectAndCompute(gray, None)

#     # Draw keypoints
#     img_with_keypoints = cv2.drawKeypoints(
#         image,
#         keypoints,
#         None,
#         flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
#         color=(0, 255, 0)  # Green color for keypoints
#     )

#     # Load Haar Cascade for face detection
#     face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

#     # Detect faces
#     faces = face_cascade.detectMultiScale(gray, 1.1, 4)

#     # Draw rectangles around each face
#     for (x, y, w, h) in faces:
#         cv2.rectangle(img_with_keypoints, (x, y), (x+w, y+h), (255, 0, 0), 2)  # Blue color for faces

#     return img_with_keypoints, keypoints, descriptors

# # Function to match SIFT features between two images
# def match_sift_features(desc1, desc2):
#     bf = cv2.BFMatcher()
#     matches = bf.knnMatch(desc1, desc2, k=2)
    
#     # Apply ratio test
#     good_matches = []
#     for m, n in matches:
#         if m.distance < 0.75 * n.distance:
#             good_matches.append(m)
    
#     return good_matches

# # Function to resize image while maintaining aspect ratio
# def resize_image(image, max_size):
#     w, h = image.size
#     if w > h:
#         new_w = max_size
#         new_h = int(h * (max_size / w))
#     else:
#         new_h = max_size
#         new_w = int(w * (max_size / h))
#     return image.resize((new_w, new_h), Image.Resampling.LANCZOS)

# # Function to open an image file and display it with SIFT keypoints
# def open_image(image_num):
#     file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.jpeg *.png")])
#     if file_path:
#         img_with_keypoints, keypoints, descriptors = extract_and_display_sift(file_path)
        
#         # 將OpenCV影像轉換為PIL影像以顯示在Tkinter中
#         img = cv2.cvtColor(img_with_keypoints, cv2.COLOR_BGR2RGB)
#         img = Image.fromarray(img)
        
#         # 將圖片調整到適合顯示的大小
#         img = resize_image(img, 400)
#         img_with_keypoints_tk = ImageTk.PhotoImage(img)

#         # 讀取原圖
#         original_image = Image.open(file_path)
#         original_image = resize_image(original_image, 400)
#         original_image_tk = ImageTk.PhotoImage(original_image)
        
#         if image_num == 1:
#             panel1.config(image=img_with_keypoints_tk)
#             panel1.image = img_with_keypoints_tk
#             panel1.config(width=img_with_keypoints_tk.width(), height=img_with_keypoints_tk.height())
#             original_panel1.config(image=original_image_tk)
#             original_panel1.image = original_image_tk
#             original_panel1.config(width=original_image_tk.width(), height=original_image_tk.height())
#             global descriptors1, keypoints1
#             descriptors1 = descriptors
#             keypoints1 = keypoints
#         else:
#             panel2.config(image=img_with_keypoints_tk)
#             panel2.image = img_with_keypoints_tk
#             panel2.config(width=img_with_keypoints_tk.width(), height=img_with_keypoints_tk.height())
#             original_panel2.config(image=original_image_tk)
#             original_panel2.image = original_image_tk
#             original_panel2.config(width=original_image_tk.width(), height=original_image_tk.height())
#             global descriptors2, keypoints2
#             descriptors2 = descriptors
#             keypoints2 = keypoints

# def compare_images():
#     if descriptors1 is not None and descriptors2 is not None:
#         matches = match_sift_features(descriptors1, descriptors2)
#         num_good_matches = len(matches)
        
#         # Calculate similarity percentage
#         min_features = min(len(keypoints1), len(keypoints2))
#         similarity_percentage = (num_good_matches / min_features) * 100
        
#         if num_good_matches > 50:  # 假設匹配點數量超過50個即認為圖片相似
#             result_text.set(f"The images are similar with {num_good_matches} good matches ({similarity_percentage:.2f}%).")
#         else:
#             result_text.set(f"The images are not similar with only {num_good_matches} good matches ({similarity_percentage:.2f}%).")
#     else:
#         result_text.set("Please load both images first.")

# # Function to classify the player in the image using KNN model
# def classify_player(image_path):
#     image = cv2.imread(image_path)
#     if image is None:
#         messagebox.showerror("Error", "Failed to load image.")
#         return
    
#     face_image = detect_and_crop_face(image)
#     gray_face = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
#     descriptors = extract_sift_features(gray_face)
    
#     if descriptors is not None:
#         if descriptors.shape[0] >= 50:
#             reduced_descriptor = pca.transform(descriptors[:50])
#         else:
#             padded_descriptor = np.pad(descriptors, ((0, 50 - descriptors.shape[0]), (0, 0)), 'constant')
#             reduced_descriptor = pca.transform(padded_descriptor)
#         combined_feature = np.mean(reduced_descriptor, axis=0)
#         combined_feature = scaler.transform([combined_feature])
#         prediction = knn_model.predict(combined_feature)
#         return prediction[0]
#     else:
#         messagebox.showerror("Error", "No features extracted.")
#         return None

# # Load the trained KNN model and preprocessing tools
# knn_model = joblib.load('knn_model.pkl')
# scaler = joblib.load('scaler.pkl')
# pca = joblib.load('pca_model.pkl')
# label_map = {'Lebron_James': 0, 'Stephen_Curry': 1, 'Luka_Doncic': 2}
# reverse_label_map = {v: k for k, v in label_map.items()}

# # 定義面部檢測和裁剪函數
# def detect_and_crop_face(image):
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
#     faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
#     for (x, y, w, h) in faces:
#         face = image[y:y+h, x:x+w]
#         return face
    
#     return image

# # 定義SIFT特徵提取函數
# def extract_sift_features(image):
#     sift = cv2.SIFT_create()
#     keypoints, descriptors = sift.detectAndCompute(image, None)
#     return descriptors

# def open_image(image_num):
#     file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.jpeg *.png")])
#     if file_path:
#         img_with_keypoints, keypoints, descriptors = extract_and_display_sift(file_path)
        
#         # 將OpenCV影像轉換為PIL影像以顯示在Tkinter中
#         img = cv2.cvtColor(img_with_keypoints, cv2.COLOR_BGR2RGB)
#         img = Image.fromarray(img)
        
#         # 將圖片調整到適合顯示的大小
#         img = resize_image(img, 400)
#         img_with_keypoints_tk = ImageTk.PhotoImage(img)

#         # 讀取原圖
#         original_image = Image.open(file_path)
#         original_image = resize_image(original_image, 400)
#         original_image_tk = ImageTk.PhotoImage(original_image)
        
#         if image_num == 1:
#             panel1.config(image=img_with_keypoints_tk)
#             panel1.image = img_with_keypoints_tk
#             panel1.config(width=img_with_keypoints_tk.width(), height=img_with_keypoints_tk.height())
#             original_panel1.config(image=original_image_tk)
#             original_panel1.image = original_image_tk
#             original_panel1.config(width=original_image_tk.width(), height=original_image_tk.height())
#             global descriptors1, keypoints1
#             descriptors1 = descriptors
#             keypoints1 = keypoints
#             global image_path1
#             image_path1 = file_path
#         else:
#             panel2.config(image=img_with_keypoints_tk)
#             panel2.image = img_with_keypoints_tk
#             panel2.config(width=img_with_keypoints_tk.width(), height=img_with_keypoints_tk.height())
#             original_panel2.config(image=original_image_tk)
#             original_panel2.image = original_image_tk
#             original_panel2.config(width=original_image_tk.width(), height=original_image_tk.height())
#             global descriptors2, keypoints2
#             descriptors2 = descriptors
#             keypoints2 = keypoints
#             global image_path2
#             image_path2 = file_path

# def classify_image():
#     if image_path1:
#         prediction = classify_player(image_path1)
#         if prediction is not None:
#             player_name = reverse_label_map[prediction]
#             result_text.set(f"Prediction: {player_name}")
#         else:
#             result_text.set("Could not classify the player.")
#     else:
#         result_text.set("Please load the image first.")

# # Create the main window
# root = tk.Tk()
# root.title("SIFT Feature Extractor and Comparator")

# # Step 1: Create a Canvas and a Scrollbar
# canvas = tk.Canvas(root)
# canvas.grid(row=0, column=0, sticky="news")

# scrollbar = tk.Scrollbar(root, orient="vertical", command=canvas.yview)
# scrollbar.grid(row=0, column=1, sticky="ns")

# # Configure the canvas to use the scrollbar
# canvas.configure(yscrollcommand=scrollbar.set)

# # Step 2: Create a frame that will be scrolled
# scrollable_frame = tk.Frame(canvas)

# # Add the scrollable frame to a window in the canvas
# canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")

# # Step 3: Update the scrollregion whenever the size of the scrollable_frame changes
# def on_frame_configure(event):
#     canvas.configure(scrollregion=canvas.bbox("all"))

# scrollable_frame.bind("<Configure>", on_frame_configure)


# # Initialize global variables to store descriptors and keypoints
# descriptors1 = None
# descriptors2 = None
# keypoints1 = None
# keypoints2 = None

# # Add frames to organize the layout
# frame1 = tk.Frame(scrollable_frame)
# frame1.grid(row=1, column=0, padx=10, pady=10)

# frame2 = tk.Frame(scrollable_frame)
# frame2.grid(row=0, column=0, padx=10, pady=10)

# frame3 = tk.Frame(scrollable_frame)
# frame3.grid(row=1, column=1, padx=10, pady=10)

# frame4 = tk.Frame(scrollable_frame)
# frame4.grid(row=0, column=1, padx=10, pady=10)

# root.grid_rowconfigure(0, weight=1)
# root.grid_columnconfigure(0, weight=1)

# # Set a minimum size for the window, if desired
# root.minsize(800, 600)

# # Add panels to display the images
# panel1 = tk.Label(frame1, text="Please insert the image", relief="solid", width=50, height=25)
# panel1.pack(padx=10, pady=10)

# original_panel1 = tk.Label(frame2, text="Original Image", relief="solid", width=50, height=25)
# original_panel1.pack(padx=10, pady=10)

# panel2 = tk.Label(frame3, text="Please insert the image", relief="solid", width=50, height=25)
# panel2.pack(padx=10, pady=10)

# original_panel2 = tk.Label(frame4, text="Original Image", relief="solid", width=50, height=25)
# original_panel2.pack(padx=10, pady=10)

# # Add buttons to load images
# load_button1 = tk.Button(frame1, text="Load Image 1", command=lambda: open_image(1))
# load_button1.pack(pady=5)

# classify_button1 = tk.Button(frame1, text="Classify Player1", command=lambda: classify_image(1))
# classify_button1.pack(pady=5)

# load_button2 = tk.Button(frame3, text="Load Image 2", command=lambda: open_image(2))
# load_button2.pack(pady=5)

# classify_button2 = tk.Button(frame3, text="Classify Player2", command=lambda: classify_image(2))
# classify_button2.pack(pady=5)

# # Add a button to start comparing images
# compare_button = tk.Button(root, text="Start Comparing", command=compare_images)
# compare_button.grid(row = 2 , column=0, columnspan=4, pady=10)

# # Add a label to display the results
# result_text = tk.StringVar()
# result_label = tk.Label(root, textvariable=result_text, font=("Arial", 14))
# result_label.grid(row=3, column= 0, columnspan = 4, pady = 20)

# # Run the GUI
# root.mainloop()

import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import joblib

# 定義一個函數來提取SIFT特徵並顯示在圖片上
def extract_and_display_sift(image_path):
    # Load the image
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale

    # Initialize SIFT detector
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray, None)

    # Draw keypoints
    img_with_keypoints = cv2.drawKeypoints(
        image,
        keypoints,
        None,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
        color=(0, 255, 0)  # Green color for keypoints
    )

    # Load Haar Cascade for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # Draw rectangles around each face
    for (x, y, w, h) in faces:
        cv2.rectangle(img_with_keypoints, (x, y), (x+w, y+h), (255, 0, 0), 2)  # Blue color for faces

    return img_with_keypoints, keypoints, descriptors

# Function to match SIFT features between two images
def match_sift_features(desc1, desc2):
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(desc1, desc2, k=2)
    
    # Apply ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)
    
    return good_matches

# Function to resize image while maintaining aspect ratio
def resize_image(image, max_size):
    w, h = image.size
    if w > h:
        new_w = max_size
        new_h = int(h * (max_size / w))
    else:
        new_h = max_size
        new_w = int(w * (max_size / h))
    return image.resize((new_w, new_h), Image.Resampling.LANCZOS)

# Function to open an image file and display it with SIFT keypoints
def open_image(image_num):
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.jpeg *.png")])
    if file_path:
        img_with_keypoints, keypoints, descriptors = extract_and_display_sift(file_path)
        
        # 將OpenCV影像轉換為PIL影像以顯示在Tkinter中
        img = cv2.cvtColor(img_with_keypoints, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        
        # 將圖片調整到適合顯示的大小
        img = resize_image(img, 400)
        img_with_keypoints_tk = ImageTk.PhotoImage(img)

        # 讀取原圖
        original_image = Image.open(file_path)
        original_image = resize_image(original_image, 400)
        original_image_tk = ImageTk.PhotoImage(original_image)
        
        if image_num == 1:
            panel1.config(image=img_with_keypoints_tk)
            panel1.image = img_with_keypoints_tk
            panel1.config(width=img_with_keypoints_tk.width(), height=img_with_keypoints_tk.height())
            original_panel1.config(image=original_image_tk)
            original_panel1.image = original_image_tk
            original_panel1.config(width=original_image_tk.width(), height=original_image_tk.height())
            global descriptors1, keypoints1, image_path1
            descriptors1 = descriptors
            keypoints1 = keypoints
            image_path1 = file_path
        else:
            panel2.config(image=img_with_keypoints_tk)
            panel2.image = img_with_keypoints_tk
            panel2.config(width=img_with_keypoints_tk.width(), height=img_with_keypoints_tk.height())
            original_panel2.config(image=original_image_tk)
            original_panel2.image = original_image_tk
            original_panel2.config(width=original_image_tk.width(), height=original_image_tk.height())
            global descriptors2, keypoints2, image_path2
            descriptors2 = descriptors
            keypoints2 = keypoints
            image_path2 = file_path


def compare_images():
    if descriptors1 is not None and descriptors2 is not None:
        matches = match_sift_features(descriptors1, descriptors2)
        num_good_matches = len(matches)
        
        # Calculate similarity percentage
        min_features = min(len(keypoints1), len(keypoints2))
        similarity_percentage = (num_good_matches / min_features) * 100
        
        if num_good_matches > 50:  # 假設匹配點數量超過50個即認為圖片相似
            result_text.set(f"The images are similar with {num_good_matches} good matches ({similarity_percentage:.2f}%).")
        else:
            result_text.set(f"The images are not similar with only {num_good_matches} good matches ({similarity_percentage:.2f}%).")
    else:
        result_text.set("Please load both images first.")

# Function to classify the player in the image using KNN model
def classify_player(image_path):
    image = cv2.imread(image_path)
    if image is None:
        messagebox.showerror("Error", "Failed to load image.")
        return
    
    face_image = detect_and_crop_face(image)
    gray_face = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
    descriptors = extract_sift_features(gray_face)
    
    if descriptors is not None:
        if descriptors.shape[0] >= 50:
            reduced_descriptor = pca.transform(descriptors[:50])
        else:
            padded_descriptor = np.pad(descriptors, ((0, 50 - descriptors.shape[0]), (0, 0)), 'constant')
            reduced_descriptor = pca.transform(padded_descriptor)
        combined_feature = np.mean(reduced_descriptor, axis=0)
        combined_feature = scaler.transform([combined_feature])
        prediction = knn_model.predict(combined_feature)
        return prediction[0]
    else:
        messagebox.showerror("Error", "No features extracted.")
        return None
    
def classify_image(image_num):
    if image_num == 1 and image_path1:
        prediction = classify_player(image_path1)
    elif image_num == 2 and image_path2:
        prediction = classify_player(image_path2)
    else:
        messagebox.showerror("Error", "Please load the image first.")
        return

    if prediction is not None:
        player_name = reverse_label_map[prediction]
        if image_num == 1:
            result_text1.set(f"Prediction: {player_name}")
        else:
            result_text2.set(f"Prediction: {player_name}")
    else:
        if image_num == 1:
            result_text1.set("Could not classify the player.")
        else:
            result_text2.set("Could not classify the player.")

# Load the trained KNN model and preprocessing tools
knn_model = joblib.load('knn_model.pkl')
scaler = joblib.load('scaler.pkl')
pca = joblib.load('pca_model.pkl')
label_map = {'Lebron_James': 0, 'Stephen_Curry': 1, 'Luka_Doncic': 2}
reverse_label_map = {v: k for k, v in label_map.items()}

# 定義面部檢測和裁剪函數
def detect_and_crop_face(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    for (x, y, w, h) in faces:
        face = image[y:y+h, x:x+w]
        return face
    
    return image

# 定義SIFT特徵提取函數
def extract_sift_features(image):
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image, None)
    return descriptors

# Create the main window
root = tk.Tk()
root.title("SIFT Feature Extractor and Comparator")

# Step 1: Create a Canvas and a Scrollbar
canvas = tk.Canvas(root)
canvas.grid(row=0, column=0, sticky="news")

scrollbar = tk.Scrollbar(root, orient="vertical", command=canvas.yview)
scrollbar.grid(row=0, column=1, sticky="ns")

# Configure the canvas to use the scrollbar
canvas.configure(yscrollcommand=scrollbar.set)

# Step 2: Create a frame that will be scrolled
scrollable_frame = tk.Frame(canvas)

# Add the scrollable frame to a window in the canvas
canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")

# Step 3: Update the scrollregion whenever the size of the scrollable_frame changes
def on_frame_configure(event):
    canvas.configure(scrollregion=canvas.bbox("all"))

scrollable_frame.bind("<Configure>", on_frame_configure)

# Initialize global variables to store descriptors and keypoints
descriptors1 = None
descriptors2 = None
keypoints1 = None
keypoints2 = None
image_path1 = None
image_path2 = None

# Add frames to organize the layout
frame1 = tk.Frame(scrollable_frame)
frame1.grid(row=1, column=0, padx=10, pady=10)

frame2 = tk.Frame(scrollable_frame)
frame2.grid(row=0, column=0, padx=10, pady=10)

frame3 = tk.Frame(scrollable_frame)
frame3.grid(row=1, column=1, padx=10, pady=10)

frame4 = tk.Frame(scrollable_frame)
frame4.grid(row=0, column=1, padx=10, pady=10)

root.grid_rowconfigure(0, weight=1)
root.grid_columnconfigure(0, weight=1)

# Set a minimum size for the window, if desired
root.minsize(800, 600)

# Add panels to display the images
panel1 = tk.Label(frame1, text="Please insert the image", relief="solid", width=50, height=25)
panel1.pack(padx=10, pady=10)

original_panel1 = tk.Label(frame2, text="Original Image", relief="solid", width=50, height=25)
original_panel1.pack(padx=10, pady=10)

panel2 = tk.Label(frame3, text="Please insert the image", relief="solid", width=50, height=25)
panel2.pack(padx=10, pady=10)

original_panel2 = tk.Label(frame4, text="Original Image", relief="solid", width=50, height=25)
original_panel2.pack(padx=10, pady=10)

# Add buttons to load images
load_button1 = tk.Button(frame1, text="Load Image 1", command=lambda: open_image(1))
load_button1.pack(pady=5)

classify_button1 = tk.Button(frame1, text="Classify Player 1", command=lambda: classify_image(1))
classify_button1.pack(pady=5)

result_text1 = tk.StringVar()
result_label1 = tk.Label(frame1, textvariable=result_text1, font=("Arial", 12))
result_label1.pack(pady=5)

load_button2 = tk.Button(frame3, text="Load Image 2", command=lambda: open_image(2))
load_button2.pack(pady=5)

classify_button2 = tk.Button(frame3, text="Classify Player 2", command=lambda: classify_image(2))
classify_button2.pack(pady=5)

result_text2 = tk.StringVar()
result_label2 = tk.Label(frame3, textvariable=result_text2, font=("Arial", 12))
result_label2.pack(pady=5)

# Add a button to start comparing images
compare_button = tk.Button(root, text="Start Comparing", command=compare_images)
compare_button.grid(row=2, column=0, columnspan=4, pady=10)

# Add a label to display the results
result_text = tk.StringVar()
result_label = tk.Label(root, textvariable=result_text, font=("Arial", 14))
result_label.grid(row=3, column=0, columnspan=4, pady=20)

# Run the GUI
root.mainloop()








