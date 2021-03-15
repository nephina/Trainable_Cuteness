import pandas as pd
from numpy import diff
import random
from itertools import combinations
import sys
import os
from PyQt5.QtWidgets import (QApplication, QDialog, QGridLayout, QPushButton,
                            QHBoxLayout, QProgressBar, QLabel)
from PyQt5.QtGui import QPixmap
from PyQt5 import QtCore
from trainer import trainer
import csv
from sampling_utils import *
import yaml
from yaml import safe_load as load

os.environ['LRU_CACHE_CAPACITY']='20'
settings = load(open('settings.yaml','r'))

#path = "F:/Cuteness AI/Code/Trainable_Cuteness-main/"
#os.chdir(path) # changing to the file directory we want

class PairwisePrompt(QDialog):
    '''Generate a window that displays two images and selection buttons'''

    def __init__(self, parent=None):
        super(PairwisePrompt, self).__init__(parent)

        self.left_image = QLabel()
        self.left_image.setFixedSize(settings['window_imsize'],
                                     settings['window_imsize'])
        self.right_image = QLabel()
        self.right_image.setFixedSize(settings['window_imsize'],
                                      settings['window_imsize'])

        self.image_preference = image_preference
        self.LeftSelectionButton = QPushButton('Select Image 1')
        self.RightSelectionButton = QPushButton('Select Image 2')
        self.LeftSelectionButton.setDefault(True)
        self.RightSelectionButton.setDefault(True)
        self.SkipButton = QPushButton('Skip')
        self.ProgressBar = QProgressBar()
        self.ProgressBar.setRange(0, settings['choices_per_sess'])
        self.ProgressBar.setValue(0)
        self.StatusText = QLabel('Currently in: user entry mode')

        topLayout = QHBoxLayout()
        mainLayout = QGridLayout()
        mainLayout.addLayout(topLayout, 5, 2, 1, 1)
        mainLayout.addWidget(self.left_image, 0, 0)
        mainLayout.addWidget(self.right_image, 0, 1)
        mainLayout.addWidget(self.LeftSelectionButton,1,0)
        mainLayout.addWidget(self.RightSelectionButton,1,1)
        mainLayout.addWidget(self.SkipButton,2,0,1,2)
        mainLayout.addWidget(self.ProgressBar,4,0,1,2)
        mainLayout.addWidget(self.StatusText,3,0,1,2)
        self.setLayout(mainLayout)

        self.setWindowTitle("AI training example generation")


        self.LeftSelectionButton.clicked.connect(left_pref)
        self.RightSelectionButton.clicked.connect(right_pref)
        self.SkipButton.clicked.connect(no_pref)


def read_image_list():
    '''Read the csv containing the names and ranks of the full image data'''

    image_list = pd.read_csv('Data/ImageList.csv') #Read all image_list
    n_total_images = len(image_list)
    return image_list, n_total_images

def update_single_image(preference):
    '''Update one of the two window images'''

    new_display_image,image_index = get_random_image(image_list)
    image_location = 'Data/raw-img/'+new_display_image['ImageFile']
    image_pix_map = QPixmap(image_location)
    image_pix_map = image_pix_map.scaled(settings['window_imsize'],
                                         settings['window_imsize'],
                                         QtCore.Qt.KeepAspectRatio)
    if preference == 1: #if the right job was preferred
        window.left_image.setPixmap(image_pix_map)
    if preference == 0: #if the left job was preferred
        window.right_image.setPixmap(image_pix_map)
    return (image_index)

def update_both_image(refinement=False):
    '''Update both of the window images'''

    def write_both_to_app(left_display_image,right_display_image):
        image_location_L = 'Data/raw-img/'+left_display_image['ImageFile']
        image_location_R = 'Data/raw-img/'+right_display_image['ImageFile']
        image_pix_map_L = QPixmap(image_location_L)
        image_pix_map_L = image_pix_map_L.scaled(settings['window_imsize'],
                                            settings['window_imsize'],
                                            QtCore.Qt.KeepAspectRatio)
        image_pix_map_R = QPixmap(image_location_R)
        image_pix_map_R = image_pix_map_R.scaled(settings['window_imsize'],
                                            settings['window_imsize'],
                                            QtCore.Qt.KeepAspectRatio)
        window.left_image.setPixmap(image_pix_map_L)
        window.right_image.setPixmap(image_pix_map_R)

    if refinement:
        sampletype = 0 #random.randint(0,1)
        if sampletype == 0:
            return_tuple = get_closest_pair(image_list)
            left_display_image = return_tuple[0]
            right_display_image = return_tuple[1]
            list_indices = return_tuple[2]
        elif sampletype == 1:
            return_tuple = get_topend_close_pair(image_list)
            left_display_image = return_tuple[0]
            right_display_image = return_tuple[1]
            list_indices = return_tuple[2]
        write_both_to_app(left_display_image,right_display_image)
    else:
        return_tuple = get_random_image_pair(image_list)
        left_display_image = return_tuple[0]
        right_display_image = return_tuple[1]
        list_indices = return_tuple[2]
        write_both_to_app(left_display_image,right_display_image)

    return list_indices

def run_ai_training():
    '''Save the most recent ranked images, and start the AI training'''

    global image_list, n_total_images, pairwise_ranked_images
    # remove_conflicts broken right now
    #pairwise_ranked_images = remove_conflicts(pairwise_ranked_images)
    final_ranked_pairs = pd.DataFrame(pairwise_ranked_images)
    final_ranked_pairs.columns = ['ImageFile','Rating']
    final_ranked_pairs['SortKey'] = range(len(final_ranked_pairs))
    final_ranked_pairs.to_csv('Data/RankedPairs.csv',index=False)
    trainer(window, image_list)
    window.ProgressBar.setRange(0, settings['choices_per_sess'])
    window.ProgressBar.setValue(0)
    window.StatusText.setText('Currently in: user entry mode')
    image_list, n_total_images = read_image_list()
    first_prompt_run[0] = 0

def update_step_state(preference):
    '''Store the images and preference, sample new image(s) to display'''

    global pairwise_ranked_images, total_selection_count
    left_image_filename = image_list['ImageFile'][list_indices[0]]
    right_image_filename = image_list['ImageFile'][list_indices[1]]
    if preference == 0:
        pairwise_ranked_images.append([left_image_filename,1])
        pairwise_ranked_images.append([right_image_filename,0])
    else:
        pairwise_ranked_images.append([left_image_filename,0])
        pairwise_ranked_images.append([right_image_filename,1])
    total_selection_count[0] += 1
    window.ProgressBar.setValue(total_selection_count[0])
    if total_selection_count[0] >= settings['choices_per_sess']:
        window.left_image.setPixmap(QPixmap())
        window.right_image.setPixmap(QPixmap())
        run_ai_training()
        total_selection_count[0] = 0 #reset the counter

def left_pref():
    '''Upon user selection of the left image, update viewer and store info'''

    global list_indices

    update_step_state(preference=0)

    if first_prompt_run[0] == 1:
        right_select_count[0] = 0
        left_select_count[0] += 1
        if left_select_count[0] < 25:
            right_image_index = update_single_image(0)
            list_indices[1] = right_image_index
        else:
            left_image_index = update_single_image(1)
            list_indices[0] = left_image_index
            left_select_count[0] = 0

    else:
        list_indices = update_both_image(refinement=False)

def right_pref():
    '''Upon user selection of the right image, update viewer and store info'''

    global list_indices

    update_step_state(preference=1)

    if first_prompt_run[0] == 1:
        left_select_count[0] = 0
        right_select_count[0] += 1
        if right_select_count[0] < 25:
            left_image_index = update_single_image(1)
            list_indices[0] = left_image_index

        else:
            right_image_index = update_single_image(0)
            list_indices[1] = right_image_index
            right_select_count[0] = 0

    else:
        list_indices = update_both_image(refinement=False)

def no_pref():
    '''Upon user selection of the skip button, update viewer'''

    global list_indices
    if first_prompt_run[0] == 1:
        list_indices = update_both_image(refinement=False)
    else:
        list_indices = update_both_image(refinement=False)

def remove_conflicts(pairwise_ranked_images):
    '''Search through the ranked images and remove any duplicates/conflicts
    
    This is a significantly annoying process that I feel should have an
    algorithmically optimal answer, but right now I simply nest IF statements
    to determine whether any pair exists twice in the ranked pairs, and if it
    does, whether it exists as a duplicate, or a conflict, regardless of the
    order of the individual elements in the two pairs. Right now this is a
    >O(n!) piece of code, so this will kill the feasability of larger datasets.
    
    
    The BREAK statements must be used because removing one of the pairs changes
    the index of all pairs coming after it, so the process must be run again.
    When a conflict does exist, the code assumes the user's more recent choices
    are more true and deletes all but the last instance.
    '''

    def get_combs(pairwise_ranked_images):
        '''Return all possible combinations of pairs in the ranked list'''

        combs = combinations(list(range(int(len(pairwise_ranked_images)/2))),2)
        combs = list(combs)
        return combs
    combs = get_combs(pairwise_ranked_images)
    remove_pairs = True
    while remove_pairs:
        for comb in combs:
            remove_pairs = False
            pair_one = pairwise_ranked_images[int(comb[0]*2):int(comb[0]*2)+2]
            pair_two = pairwise_ranked_images[int(comb[1]*2):int(comb[1]*2)+2]
            pair_one = pair_one.to_numpy()
            pair_two = pair_two.to_numpy()
            if (pair_one[0][0] == pair_two[0][0] and
                pair_one[1][0] == pair_two[1][0]):
                if pair_one[0][1] == pair_two[0][1]:
                    print('You had a duplicate pair: lines '+
                          str((comb[0]+1)*2)+
                          ' and '+
                          str((comb[1]+1)*2))
                    remove_pairs = [comb[0],comb[1]]
                    break
                elif pair_one[0][1] != pair_two[0][1]:
                    print('You had a conflict between pairs: lines '+
                          str((comb[0]+1)*2)+
                          ' and '+
                          str((comb[1]+1)*2))
                    remove_pairs = [comb[0],comb[1]]
                    break

            elif (pair_one[0][0] == pair_two[1][0] and 
                 pair_one[1][0] == pair_two[0][0]):
                if pair_one[0][1] == pair_two[1][1]:
                    print('You had a duplicate pair: lines '+
                          str((comb[0]+1)*2)+
                          ' and '+
                          str((comb[1]+1)*2))
                    remove_pairs = [comb[0],comb[1]]
                    break
                elif pair_one[0][1] != pair_two[1][1]:
                    print('You had a conflict between pairs: lines '+
                          str((comb[0]+1)*2)+
                          ' and '
                          +str((comb[1]+1)*2))
                    remove_pairs = [comb[0],comb[1]]
                    break

        if remove_pairs:
            pairwise_ranked_images.pop(int(remove_pairs[0]*2))
            pairwise_ranked_images.pop(int((remove_pairs[0]*2)))
            combs = get_combs(pairwise_ranked_images)
    return pairwise_ranked_images



image_preference=[]
pairwise_ranked_images = []
left_select_count = [0]
right_select_count = [0]
total_selection_count = [0]
first_prompt_run = [1]



app = QApplication(sys.argv)
window = PairwisePrompt()
window.show()
image_list, n_total_images = read_image_list()
if os.path.exists('Data/RankedPairs.csv'):
    first_prompt_run = [0]
    pairwise_ranked_images = pd.read_csv('Data/RankedPairs.csv')
    pairwise_ranked_images = pairwise_ranked_images.drop(['SortKey'],axis=1)
    pairwise_ranked_images = pairwise_ranked_images.values.tolist()
else:
    headers = ['ImageFile', 'Rating', 'SortKey']
    with open('Data/RankedPairs.csv', 'wt') as file:
        write = csv.writer(file, delimiter=',')
        write.writerow(i for i in headers)
    file.close()
list_indices = update_both_image(refinement=False)

sys.exit(app.exec_())
