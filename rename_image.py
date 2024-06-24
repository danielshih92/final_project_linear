import os
from PIL import Image

# player_name = "triangle"
# folder_path = f'shape/triangle'

# player_name = "rectangle"
# folder_path = f'shape/rectangle'

# player_name = "circle_or_ellipse"
# folder_path = f'shape/circle_or_ellipse'

player_name = "Luka_Doncic"
folder_path = f'nba_players/Luka_Doncic'

# List all files in the folder
files = os.listdir(folder_path)

# Filter out non-image files if necessary (optional)
image_files = [file for file in files if file.endswith(('.png', '.jpg', '.jpeg'))]

# Sort the files if needed (optional)
image_files.sort()

# Define a new name pattern for the images
new_name_pattern = f"{player_name}{{}}.jpg"  # Set the extension to .jpg

# Loop through each image file, rename it, and convert to .jpg
for index, file in enumerate(image_files, start=1):
    # Construct the new file name
    new_name = new_name_pattern.format(index)
    # Construct the full old and new file paths
    old_file_path = os.path.join(folder_path, file)
    new_file_path = os.path.join(folder_path, new_name)
    # Open the image
    with Image.open(old_file_path) as img:
        # Check if image mode is 'P' and convert to 'RGBA' first if necessary
        if img.mode == 'P':
            img = img.convert('RGBA')
        # Then convert to 'RGB' and save as .jpg
        img.convert('RGB').save(new_file_path, 'JPEG')
    # If the new file name is different from the old one, remove the old file
    if old_file_path != new_file_path:
        os.remove(old_file_path)