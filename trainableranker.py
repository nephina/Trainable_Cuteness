import pandas as pd
from itertools import combinations
import random
import sys
import os
from PyQt5.QtWidgets import (QApplication, QDialog, QGridLayout, QPushButton, QHBoxLayout, QProgressBar, QLabel)
from PyQt5.QtGui import QPixmap
from PyQt5 import QtCore
from trainer import trainer
from sampling_utils import *

os.environ['LRU_CACHE_CAPACITY']='20'
image_window_size = 800

class PairwisePrompt(QDialog):
    def __init__(self, parent=None):
        super(PairwisePrompt, self).__init__(parent)

        self.left_image = QLabel()
        self.left_image.setFixedSize(image_window_size,image_window_size)
        self.right_image = QLabel()
        self.right_image.setFixedSize(image_window_size, image_window_size)

        self.image_preference = image_preference
        self.LeftSelectionButton = QPushButton('Select Image 1')
        self.RightSelectionButton = QPushButton('Select Image 2')
        self.LeftSelectionButton.setDefault(True)
        self.RightSelectionButton.setDefault(True)
        self.SkipButton = QPushButton('Skip')
        self.ProgressBar = QProgressBar()
        self.ProgressBar.setRange(0, 50)
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


def read_ranking_file():
    image_list = pd.read_csv('Data/ImageList.csv') #Read all image_list
    n_total_images = len(image_list)
    return image_list, n_total_images

def update_single_image(preference):
    new_display_image,image_index = get_random_image(image_list)
    if preference == 1: #if the right job was preferred
        window.left_image.setPixmap(QPixmap('Data/raw-img/'+new_display_image['ImageFile']).scaled(image_window_size, image_window_size, QtCore.Qt.KeepAspectRatio))
    if preference == 0: #if the left job was preferred
        window.right_image.setPixmap(QPixmap('Data/raw-img/'+new_display_image['ImageFile']).scaled(image_window_size, image_window_size, QtCore.Qt.KeepAspectRatio))
    return (image_index)

def update_both_image(refinement=False):
    def write_both_to_app(left_display_image,right_display_image):
        window.left_image.setPixmap(QPixmap('Data/raw-img/'+left_display_image['ImageFile']).scaled(image_window_size, image_window_size, QtCore.Qt.KeepAspectRatio))
        window.right_image.setPixmap(QPixmap('Data/raw-img/'+right_display_image['ImageFile']).scaled(image_window_size, image_window_size, QtCore.Qt.KeepAspectRatio))

    if refinement:
        sampletype = random.randint(0,1)
        if sampletype == 0:
            left_display_image,right_display_image,list_indices =  get_closest_pair(image_list)
        elif sampletype == 1:
            left_display_image,right_display_image,list_indices = get_topend_close_pair(image_list)
        write_both_to_app(left_display_image,right_display_image)
    else:
        left_display_image,right_display_image,list_indices = get_random_image_pair(image_list)
        write_both_to_app(left_display_image,right_display_image)

    return list_indices

def run_ai_training():
    global image_list, n_total_images, pairwise_ranked_images
    if len(pairwise_ranked_images) > 2:
        pairwise_ranked_images = remove_conflicts(pairwise_ranked_images)
    final_ranked_pairs = pd.DataFrame(pairwise_ranked_images)
    final_ranked_pairs.columns = ['ImageFile','Rating']
    final_ranked_pairs['SortKey'] = range(len(final_ranked_pairs))
    final_ranked_pairs.to_csv('Data/RankedPairs.csv',index=False)
    trainer(window, image_list)
    window.ProgressBar.setRange(0, 50)
    window.ProgressBar.setValue(0)
    window.StatusText.setText('Currently in: user entry mode')
    image_list, n_total_images = read_ranking_file()
    first_prompt_run[0] = 0

def update_step_state(preference):
    global pairwise_ranked_images, total_selection_count
    if preference == 0:
        pairwise_ranked_images.append([image_list['ImageFile'][list_indices[0]],1])
        pairwise_ranked_images.append([image_list['ImageFile'][list_indices[1]],0])
    else:
        pairwise_ranked_images.append([image_list['ImageFile'][list_indices[0]],0])
        pairwise_ranked_images.append([image_list['ImageFile'][list_indices[1]],1])
    total_selection_count[0] += 1
    window.ProgressBar.setValue(total_selection_count[0])
    if total_selection_count[0] >= 2:
        window.left_image.setPixmap(QPixmap())
        window.right_image.setPixmap(QPixmap())
        run_ai_training()
        total_selection_count[0] = 0 #reset the counter

def left_pref():
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
    global list_indices
    if first_prompt_run[0] == 1:
        list_indices = update_both_image(refinement=False)
    else:
        list_indices = update_both_image(refinement=False)

def remove_conflicts(pairwise_ranked_images):

    def get_combs(pairwise_ranked_images):
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
            if pair_one[0][0] == pair_two[0][0] and pair_one[1][0] == pair_two[1][0]:
                if pair_one[0][1] == pair_two[0][1]:
                    print('You had a duplicate pair: lines '+str((comb[0]+1)*2)+' and '+str((comb[1]+1)*2))
                    remove_pairs = [comb[0],comb[1]]
                    break
                elif pair_one[0][1] != pair_two[0][1]:
                    print('You had a conflict between pairs: lines '+str((comb[0]+1)*2)+' and '+str((comb[1]+1)*2))
                    remove_pairs = [comb[0],comb[1]]
                    break

            elif pair_one[0][0] == pair_two[1][0] and pair_one[1][0] == pair_two[0][0]:
                if pair_one[0][1] == pair_two[1][1]:
                    print('You had a duplicate pair: lines '+str((comb[0]+1)*2)+' and '+str((comb[1]+1)*2))
                    remove_pairs = [comb[0],comb[1]]
                    break
                elif pair_one[0][1] != pair_two[1][1]:
                    print('You had a conflict between pairs: lines '+str((comb[0]+1)*2)+' and '+str((comb[1]+1)*2))
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
image_list, n_total_images = read_ranking_file()
if os.path.exists('Data/RankedPairs.csv'):
    first_prompt_run = [0]
    pairwise_ranked_images = pd.read_csv('Data/RankedPairs.csv').drop(['SortKey'],axis=1).values.tolist()
    if len(pairwise_ranked_images) > 2:
        pairwise_ranked_images = remove_conflicts(pairwise_ranked_images)
list_indices = update_both_image(refinement=False)

sys.exit(app.exec_())
