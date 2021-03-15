
import random
import pandas as pd
from numpy import diff

def get_random_image(image_list):
    '''Return a single random image'''

    image_index = random.sample(range(0,len(image_list)-1),2)
    image = image_list.iloc[image_index[0]]
    return image,image_index[0]

def get_random_image_pair(image_list):
    '''Return two images from a random location in the list'''

    listing_pair_indices = random.sample(range(0,len(image_list)-1),2)
    Listing1 = image_list.iloc[listing_pair_indices[0]]
    Listing2 = image_list.iloc[listing_pair_indices[1]]
    return Listing1,Listing2,listing_pair_indices

def get_closest_pair(image_list):
    '''Return a pair of images that are very close in rank

    Sort images by rank, make list of the rank deltas between consecutive images
    From these image rank deltas, extract smallest 20% and their assoc. images
    Return a random pair from this 20% set
    '''

    image_list.sort_values(by=['Rating'],inplace=True,ascending=False)
    deltas = abs(diff(image_list['Rating']))
    deltas = [deltas.tolist()]
    deltas.append([index for index in range(len(deltas[0]))])
    deltas = pd.DataFrame(deltas)
    deltas.sort_values(by=[0],inplace=True,axis=1)
    deltas = deltas.transpose()
    deltas = deltas.reset_index(drop=True)
    num_samples = int((len(deltas)*0.2))
    small_delta_random_sample_index = random.sample(range(0,num_samples),1)
    list_indices = [int(deltas[1][small_delta_random_sample_index]),
                    int(deltas[1][small_delta_random_sample_index])+1]
    return (image_list.iloc[list_indices[0]],
           image_list.iloc[list_indices[1]],
           list_indices)

def get_adjacent_random_pair(image_list):
    '''Return a consecutive pair of images from a random location in the list'''

    image_list.sort_values(by=['Rating'],inplace=True,ascending=False)
    list_indices = [None,None]
    list_indices[0] = random.randint(0,len(image_list)-1)
    list_indices[1] = list_indices[0]+1
    return (image_list.iloc[list_indices[0]],
           image_list.iloc[list_indices[1]],
           list_indices)

def get_topend_close_pair(image_list):
    '''Return a pair of images right next to each other from a random location
    in the top 200 list entries
    '''

    image_list.sort_values(by=['Rating'],inplace=True,ascending=False)
    list_indices = [None,None]
    list_indices[0] = random.randint(0,200)
    list_indices[1] = list_indices[0]+1
    return (image_list.iloc[list_indices[0]],
           image_list.iloc[list_indices[1]],
           list_indices)