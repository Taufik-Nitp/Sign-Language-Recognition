import os
import cv2
from image_processing import func
data_path="/home/taufik/Desktop/SignLanguage/Sign-Language-to-Text/data/train"
for (dirpath, dirnames, filenames) in os.walk(data_path):
    for dirname in dirnames:
        print(dirname)
        train_folder = "/home/taufik/Desktop/SignLanguage/Sign-Language-to-Text/data/train/"+dirname
        # print("dirname========>"+train_folder)
        file_list = os.listdir(train_folder)
        # Filter out non-image files
        image_files = [file for file in file_list if file.endswith(('jpg', 'jpeg', 'png'))]

# Loop over the image files and read each on
        for image_file in image_files:
            # Read the image using OpenCV
            image_path = os.path.join(train_folder, image_file)
            # image = cv2.imread(image_path)
            print("dkfajsjfkajsdf")
            print(image_path)
            new_image=func(image_path)
            cv2.imwrite(image_path,new_image)
            # Do something with the image, like process it or display it
            # ...
        
            # Release the image
            cv2.destroyAllWindows()

