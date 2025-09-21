# splits files within each Pneumonia folder (train, test, val) into Bacterial and Virus folders for each set.

import os
import shutil

BASE_DIRECTORY = "../data/raw/chest_xray"
splits = ["test", "train", "val"]

for split in splits:
    pneumonia_dir = os.path.join(BASE_DIRECTORY, split, "PNEUMONIA")
    bacterial_dir = os.path.join(BASE_DIRECTORY, split, "BACTERIAL")
    viral_dir = os.path.join(BASE_DIRECTORY, split, "VIRAL")

    os.makedirs(bacterial_dir, exist_ok = True)
    os.makedirs(viral_dir, exist_ok = True)

    for file in os.listdir(pneumonia_dir):
        lower = file.lower()
        src_path = os.path.join(pneumonia_dir, file)

        #bacterial vs. viral indicated in file name of each image
        # e.g. "person96_bacteria_466.jpeg" 
        if "bacteria" in lower:
            shutil.move(src_path, os.path.join(bacterial_dir, file)) #keep original name
        elif "virus" in lower:
            shutil.move(src_path, os.path.join(viral_dir, file))
        else:
            print(f"ERROR: the file '{file}' doesn't have 'bacteria' or 'virus' in name.")
    
    #remove pneumonia folder (optional)
    if not os.listdir(pneumonia_dir):
        os.rmdir(pneumonia_dir)

print("Process successful. Pneumonia images have been split into Bacteria and Viral Folders.") #
input() #just to keep the prompt open