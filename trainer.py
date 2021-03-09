import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import numpy as np
import pandas as pd
from model import *
from dataloader import ImageDataSet
from loss import pairwise_loss
import logging

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def train(model, iterator, optimizer, loss_type = 'order'):
    '''Run the training loop on a training dataset'''

    epoch_loss = 0
    epoch_order_loss = 0
    epoch_mean = 0
    epoch_std = 0

    model.train()
    for batch in iterator:
        images = batch[0].to(device,non_blocking=True)
        ranks = batch[1].to(device,non_blocking=True)
        predictions, log_vars = model(images)
        loss, order_loss = pairwise_loss(predictions,
                                        log_vars,
                                        ranks,
                                        loss_type)

        loss.backward()
        optimizer.step()
        model.zero_grad()

        epoch_mean += torch.mean(predictions).cpu().item()
        epoch_std += torch.std(predictions).item()
        epoch_order_loss += order_loss.item()
        epoch_loss += loss.item()
    epoch_loss       /= len(iterator)
    epoch_order_loss /= len(iterator)
    epoch_mean       /= len(iterator)
    epoch_std        /= len(iterator)
    return epoch_loss, epoch_order_loss, epoch_mean, epoch_std

def test(window, model, iterator):
    '''Run the test loop on a test dataset in eval mode'''

    test_result = []
    model.eval()
    with torch.no_grad():
        for iteration, batch in enumerate(iterator):
            images = batch[0].to(device)
            predictions, log_vars = model(images)
            test_result.extend(torch.flatten(predictions).to(torch.device('cpu')).tolist())
            window.ProgressBar.setValue(iteration+1)
    return test_result

def trainer(window, Listings):
    '''Sets up the train/full datasets and runs train/prediction loops
    
    First it sets up some basic things like batch sizes, datasets, optimizer,
    learning rate scheduler, and tensorboard. Then it enters a while loop
    where it will keep running through epochs until it reaches a state where all
    the iterations in the epoch came back with all images ranked correctly.
    At repeated intervals in this loop it will also re-rank the full dataset
    and write the results to the ImageList.csv file.
    '''

    #Define the training characteristics
    train_batch_size = 300
    full_set_batch_size = 50
    image_size = 256

    window.StatusText.setText('Building Training Dataset')
    window.ProgressBar.setRange(0, 200)
    window.ProgressBar.setValue(0)
    
    pd_csv = pd.read_csv('Data/RankedPairs.csv').drop(['SortKey'],axis=1)
    trainset = ImageDataSet(pd_csv,image_size)
    trainloader = DataLoader(trainset,
                            batch_size=train_batch_size,
                            num_workers=0,
                            shuffle=False,
                            pin_memory=True)

    model = resnet18(pretrained=False)
    try:
        model.load_state_dict(torch.load('RankPrediction-model.pkl',
                                         map_location='cpu'))
    except:
        print('no previously existing trained model')
    model = model.to(device)
    
    optimizer = torch.optim.AdamW(
        [param for param in model.parameters() if param.requires_grad == True],
                                lr=0.02, 
                                betas=(0.9, 0.999), 
                                eps=1e-08, 
                                weight_decay=0.01, 
                                amsgrad=False)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
                                            T_0=5,
                                            T_mult=2,
                                            eta_min=1e-12,
                                            last_epoch=-1,
                                            verbose=False)

    writer = SummaryWriter(flush_secs=15) #tensorboardX writer

    def rerank_images(window):
        '''Run through the full dataset and save the ranking for all entries'''
        
        window.StatusText.setText('Ranking Images')
        dataset = ImageDataSet(pd.read_csv('Data/ImageList.csv'),
                                            image_size,device,
                                            do_augmentation=False)
        dataloader = DataLoader(dataset,
                                batch_size=full_set_batch_size,
                                num_workers=0,
                                shuffle=False,
                                pin_memory=True)
        window.ProgressBar.setRange(0, len(dataloader))
        window.ProgressBar.setValue(0)
        
        predictions = test(window,model,dataloader)

        test_result_df = pd.read_csv('Data/ImageList.csv')
        test_result_df['Rating']=predictions
        test_result_df.sort_values(by=['Rating'],inplace=True,ascending=False)
        test_result_df.reset_index(drop=True,inplace=True)
        test_result_df.to_csv('Data/ImageList.csv',index=False)
        return window

    def shuffle_ranked_pairs(trainset):
        '''Return a list of ranked pairs shuffled to a different orders

        The list of ranked pairs needs to have the pairs kept together, so most
        of this code is just concerned with reshaping to two columns before
        shuffling, then reshaping back to one column for the dataloader
        '''
        # resort the paired examples to mix up the data
        train_csv = pd.read_csv('Data/RankedPairs.csv')
        sort_key = [x for x in range(len(trainset))]
        sort_key = np.reshape(sort_key,(-1,2))
        np.random.shuffle(sort_key)
        sort_key = np.reshape(sort_key,(-1))
        train_csv['SortKey'] = sort_key
        train_csv.sort_values(by=['SortKey'],inplace=True,ascending=True)
        train_csv.reset_index(drop=True,inplace=True)
        pd_csv = pd.read_csv('Data/RankedPairs.csv').drop(['SortKey'],axis=1)
        trainset = ImageDataSet(pd_csv,image_size,device)
        trainloader = DataLoader(trainset,
                                batch_size=train_batch_size,
                                num_workers=0,
                                shuffle=False,
                                pin_memory=True)
        return trainset,trainloader

    window.StatusText.setText('Training the Neural Net')
    window.ProgressBar.setRange(0, 100)
    window.ProgressBar.setValue(0)
    epoch = 0
    train_loss = 10e10
    train_order_loss = 10e10
    train_std = 0

    while (train_order_loss != 0) or (1-train_std > 0.01): # or (abs(train_mean) > 0.01):
        
        trainset,trainloader = shuffle_ranked_pairs(trainset)
        train_loss, train_order_loss, train_mean, train_std = train(model, trainloader, optimizer, loss_type='orderandvariational')
        scheduler.step(epoch)
        print(train_loss/(train_std+1.0e-25),train_order_loss,train_mean,train_std)
        writer.add_scalar('Loss over STD', train_loss/(train_std+1.0e-25), epoch)
        writer.add_scalar('Order loss', train_order_loss, epoch)
        writer.add_scalar('Mean', train_mean, epoch)
        writer.add_scalar('STD', train_std, epoch)
        writer.add_scalar('Learning rate',scheduler._last_lr[0],epoch)
        epoch += 1
        
        if epoch % 1000 == 0:
            print('Reranking images')
            rerank_images(window)
            torch.save(model.state_dict(), 'RankPrediction-model.pkl')
            window.StatusText.setText('Training the Neural Net')
            window.ProgressBar.setRange(0, 100)
            window.ProgressBar.setValue(0)

    torch.save(model.state_dict(), 'RankPrediction-model.pkl')
    rerank_images(window)
    writer.close()
