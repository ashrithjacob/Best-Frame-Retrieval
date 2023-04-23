import os
import cv2
import json

path = './DATA'

# Get the names of all files in the directory and its subdirectories
jpg_filenames = [os.path.join(root, filename) for root, _, filenames in os.walk(path) for filename in filenames if filename.lower().endswith('.jpg')]

# Create a list to store the filenames of grayscale images
grayscale_images = []

assert(len(jpg_filenames) == 30607)

for filename in (jpg_filenames):
    image = cv2.imread(filename,-1)
    # Check if the file is a grayscale image
    is_grey = (len(image.shape) == 2)
    if image is not None and is_grey:
            print(filename)
            grayscale_images.append({
                "file": filename,
                "width": image.shape[1],
                "height": image.shape[0]
            })

# Save the grayscale images to a JSON file
with open("grayscale_images.json", "w") as f:
    json.dump(grayscale_images, f, indent=4)
