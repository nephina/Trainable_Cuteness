# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 10:42:08 2021

@author: Cupcake

file to check to make sure that all the files are unique and if they aren't changing
the name of the file 
"""
# %%
# import string
# import random

# np.random.seed(1234)

# string.ascii_lowercase + string.ascii_uppercase + string.digits
# https://stackoverflow.com/questions/2257441/random-string-generation-with-upper-case-letters-and-digits
# https://stackoverflow.com/questions/18383384/python-copy-files-to-a-new-directory-and-rename-if-file-name-already-exists
# %%
import os
import uuid
import csv
import shutil
import urllib.request

#path = "/home/alexa/Desktop/Trainable_Cuteness" 
path = os.getcwd()

data_path = os.path.join(path,'Data')
temp_data_path = os.path.join(data_path,'Temp')
# %%
"""
For data download, add an option to delete downloaded folder
Can I os.walk through a zip folder and copy files out?
Do I need a check for mac, linux, windows? Can this be done automatically

Start with prompt for where to download/pull files from
Add line to select where to download to

https://stackoverflow.com/questions/20540274/most-pythonic-way-to-do-input-validation
"""
# Check os type
import platform
if platform.system() == "Windows":
    Windows = True
elif platform.system() == "Linux":
    Linux = True
elif platform.system() == "Darwin":
    Mac = True
else:
    computer = input("What Operating system are you running (Windows, Linux, or Mac)? ")
    if computer.lower() == "windows":
        Windows = True
    elif computer.lower() == "linux":
        Linux = True
    elif computer.lower() == "darwin":
        Mac = True

# Pull files from Website or data base
# need to move json file to C:\Users\Cupcake\.kaggle
# need to check to see if the root folder has .kaggle folder then copy the json file to it
#https://stackoverflow.com/questions/49386920/download-kaggle-dataset-by-using-python
if input("Download files (Yes or No)? ").lower() == "yes":
    down_files = True
    import kaggle
    kaggle.api.authenticate()
    kaggle.api.dataset_download_files('alessiocorrado99/animals10', path=temp_data_path, unzip=True, quiet=False)
else:
    down_files = False
        
# Ask if user wants to delete downloaded files when done
if input("Delete Downloaded files (Yes or No)? ").lower() == "yes":
    delete_files = True
else:
    delete_files = False
# %%
# search for all file types in the folder and then exclude py and ipynb files
ListFiles = os.walk(temp_data_path)
FileTypes = []
for walk_output in os.walk(data_path):
    for file_name in walk_output[-1]:
        FileTypes.append(file_name.split(".")[-1])

ext = list(set(FileTypes)) #set turns it into a dict with each type of file, then list turns it into a list, 
exclude_ext = ["py", "ipynb"]
ext = ['.' + i for i in ext if not any(j in i for j in exclude_ext)] # removes excluded and adds . to extension type

# copy files to new path
ImageFiles = []
if not os.path.exists(os.path.join(data_path,'raw-img')):
    os.makedirs(os.path.join(data_path,'raw-img'))
newdir = os.path.join(data_path,'raw-img')
for root, dirs, files in os.walk(temp_data_path):
    for filename in files:
        if filename.endswith(tuple(ext)):
            ImageFiles.append(filename)
            # I use absolute path, case you want to move several dirs.
            old_name = os.path.join(os.path.abspath(root), filename)
        
            # Separate base from extension
            base, extension = os.path.splitext(filename)
            new_name = os.path.join(newdir, filename)

            if not os.path.exists(new_name):  # file does not exist
                shutil.copy(old_name, new_name)
            else:  # file exists
                while True:
                    new_name = os.path.join(newdir, uuid.uuid4().hex + "-" + base + extension)
                    shutil.copy(old_name, new_name)
                    # print("Copied", old_name, "as", new_name)
                    break   


#os.chdir(path) # changing to the file directory we want

# create ImageList.csv with image list in new directory
if os.path.exists('Data/ImageList.csv'):
    os.remove('Data/ImageList.csv')
    
headers = ['ImageFile', 'Rating']
with open('Data/ImageList.csv', 'wt') as file:
    write = csv.writer(file, delimiter=',', lineterminator = '\n')
    write.writerow(i for i in headers)
    for row in ImageFiles:
        write.writerow([row, 1])
    file.close()

os.rmdir(temp_data_path)
# Have it bring up file explorer? For now have it prompt for directory path
# have the code prompt the user for both data paths, the one to copy from and the one to copy to