
import random
import pandas as pd
from numpy import diff

def get_random_image(image_list):
    image_index = random.sample(range(0,len(image_list)-1),2)
    image = image_list.iloc[image_index[0]]
    return image,image_index[0]

def get_random_image_pair(image_list):
    listing_pair_indices = random.sample(range(0,len(image_list)-1),2)
    Listing1 = image_list.iloc[listing_pair_indices[0]]
    Listing2 = image_list.iloc[listing_pair_indices[1]]
    return Listing1,Listing2,listing_pair_indices

def get_closest_pair(image_list):
    image_list.sort_values(by=['Rating'],inplace=True,ascending=False)
    #Toplistings = image_list[0:int(n_total_images*0.2)]
    deltas = abs(diff(image_list['Rating']))
    deltas = [deltas.tolist()]
    deltas.append([index for index in range(len(deltas[0]))])
    deltas = pd.DataFrame(deltas)
    deltas.sort_values(by=[0],inplace=True,axis=1)
    deltas = deltas.transpose()
    deltas = deltas.reset_index(drop=True)
    small_delta_random_sample_index = random.sample(range(0,int((len(deltas)*0.2))),1)
    list_indices = [int(deltas[1][small_delta_random_sample_index]), int(deltas[1][small_delta_random_sample_index])+1]
    return image_list.iloc[list_indices[0]],image_list.iloc[list_indices[1]],list_indices

def get_adjacent_random_pair(image_list):
    image_list.sort_values(by=['Rating'],inplace=True,ascending=False)
    list_indices = [None,None]
    list_indices[0] = random.randint(0,len(image_list)-1)
    list_indices[1] = list_indices[0]+1
    return image_list.iloc[list_indices[0]],image_list.iloc[list_indices[1]],list_indices

def get_topend_close_pair(image_list):
    image_list.sort_values(by=['Rating'],inplace=True,ascending=False)
    list_indices = [None,None]
    list_indices[0] = random.randint(0,200)
    list_indices[1] = list_indices[0]+1
    return image_list.iloc[list_indices[0]],image_list.iloc[list_indices[1]],list_indices