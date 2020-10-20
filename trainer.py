import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import numpy as np
import pandas as pd
from model import CNNSingleValueRanker
from dataloader import ImageDataSet
from loss import pairwise_loss

def train(model, iterator, optimizer, criterion, loss_type = 'order'):

    epoch_loss = 0
    epoch_order_loss = 0
    epoch_mean = 0
    epoch_std = 0

    model.train()
    for batch in iterator:
        images = batch[0]
        ranks = batch[1]
        predictions = model(images)

        loss, order_loss = pairwise_loss(predictions,
                                        ranks,
                                        criterion,
                                        'orderandstdev')

        loss.backward()
        optimizer.step()
        model.zero_grad()

        epoch_mean += torch.mean(predictions).item()
        epoch_std += torch.std(predictions).item()
        epoch_order_loss += order_loss.item()
        epoch_loss += loss.item()
    epoch_loss       /= len(iterator)
    epoch_order_loss /= len(iterator)
    epoch_mean       /= len(iterator)
    epoch_std        /= len(iterator)
    return epoch_loss, epoch_order_loss, epoch_mean, epoch_std

def test(window, model, iterator):
    test_result = []
    model.eval()
    with torch.no_grad():
        for iteration, batch in enumerate(iterator):
            images = batch[0]
            predictions = torch.flatten(model(images))
            test_result.extend(predictions.tolist())
            window.ProgressBar.setValue(iteration+1)
    return test_result

def trainer(window, Listings):
    
    #Define the training characteristics
    train_batch_size = 100
    full_set_batch_size = 100
    image_size = 256

    window.StatusText.setText('Building Training Dataset')
    window.ProgressBar.setRange(0, 100)
    window.ProgressBar.setValue(0)
    
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    pd_csv = pd.read_csv('Data/RankedPairs.csv').drop(['SortKey'],axis=1)
    trainset = ImageDataSet(pd_csv,image_size,device)
    trainloader = DataLoader(trainset,
                            batch_size=train_batch_size,
                            num_workers=0,
                            shuffle=False,
                            pin_memory=True)


    model = CNNSingleValueRanker(image_size=image_size)
    #try:
    #    model.load_state_dict(torch.load('RankPrediction-model.pkl'))
    #except:
    #    print('no previously existing trained model')
    model = model.to(device)
    
    optimizer = torch.optim.Adam([param for param in model.parameters() if param.requires_grad == True],lr=0.001)

    criterion = nn.MSELoss()#KLDivLoss(reduction='batchmean')
    criterion.to(device)

    writer = SummaryWriter(flush_secs=15) #tensorboardX writer

    def rerank_images(window):
        window.StatusText.setText('Ranking Images')
        dataset = ImageDataSet(pd.read_csv('Data/ImageList.csv'),image_size,device,do_augmentation=False)
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

    while (train_order_loss != 0) or (1-train_std > 0.01):
        
        trainset,trainloader = shuffle_ranked_pairs(trainset)

        train_loss, train_order_loss, train_mean, train_std = train(model, trainloader, optimizer, criterion,loss_type='orderandstdev')
        print(train_loss/(train_std+1.0e-25),train_order_loss,train_mean,train_std)

        torch.save(model.state_dict(), 'RankPrediction-model.pkl')
        writer.add_scalar('Loss over STD', train_loss/(train_std+1.0e-25), epoch)
        writer.add_scalar('Order loss', train_order_loss, epoch)
        writer.add_scalar('Mean', train_mean, epoch)
        writer.add_scalar('STD', train_std, epoch)
        epoch += 1
        
        if epoch % 5 == 0:
            print('Reranking images')
            window  = rerank_images(window)
            window.StatusText.setText('Training the Neural Net')
            window.ProgressBar.setRange(0, 100)
            window.ProgressBar.setValue(0)
        
        #window.ProgressBar.setValue(epoch)

    rerank_images(window)
    writer.close()