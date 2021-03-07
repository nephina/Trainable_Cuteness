import torch

def pair_order_loss(output,target):
    '''Returns the pairwise loss of a series of image pair rankings

    Pairs off couples of sequential prediction instances in the batch, and then 
    computes a loss based on whether they are in the right order or not. If they
    are, loss is 0. If not, loss is proportional to how far apart they are.
    '''
    # This reshaping takes the pairwise batch data in format: 
    # [[example1left][example1right][example2left][example2right]] 
    # and reshapes it to 
    # [[example1left,example1right],[example2left,example2right]]
    n_pairs = int(torch.true_divide(len(output),2).item())
    reshaped_prediction = output.view(n_pairs,2)
    reshaped_target = target.view(n_pairs,2)
    
    # This function takes all of the pairwise comparisons and 
    # outputs a positive num if wrong order, negative num if correct order
    prediction_factor = (reshaped_prediction[:,0]-reshaped_prediction[:,1])
    target_factor = (reshaped_target[:,0]-reshaped_target[:,1])
    x = -(prediction_factor*target_factor) 

    # This function punishes any incorrectly ranked items, while allowing 
    # correctly-ranked items to move around at will in the ranking system.
    loss = torch.mean(torch.relu(x).pow(2))
    
    return(loss)

def pairwise_loss(predictions,log_vars,ranks,criterion,loss_type):
    '''Returns the loss calculated from both pairwise and variational methods

    Pair loss isn't quite enough usually. There is a way out of that loss 
    function, and that is to rank everything exactly the same. So to mitigate 
    this, we can force the net to maintain a certain spread of ranking results 
    with another loss function. Standard deviation seems to work fairly well, 
    although the beta distribution loss can be used to try to approximate a 
    top-heavy training distribution. In my implementation the top of the ranks 
    are preferentially sampled in order to fine-tune on only the things I care 
    about, but to be honest standard deviation loss is still what I use most of 
    the time anyway. Suprisingly, it is still possible to do SGD, except in our 
    case the example is actually a minibatch of size 2. I've found the standard 
    deviation loss still works, although training isn't as stable.
    '''
    order_loss = pair_order_loss(predictions,ranks)
    if loss_type == 'order':
        loss = order_loss

    elif loss_type == 'orderandstdev':
        stdev_loss = (1-torch.std(predictions)).pow(2)
        if torch.isnan(stdev_loss):
            loss = order_loss
        else:
            loss = order_loss+stdev_loss

    elif loss_type == 'orderandcenterednormal':
        
        sorted_softlog_predictions = torch.log_softmax(predictions.sort().values,dim=0)
        normal_dist_softlog = torch.log_softmax(torch.randn_like(predictions).to(torch.device('cuda:0')).sort().values,dim=0)
        kldiv = torch.nn.KLDivLoss(reduction='batchmean', log_target=True)
        centered_normal_loss = kldiv(
                                    sorted_softlog_predictions, 
                                    normal_dist_softlog)
        if torch.isnan(centered_normal_loss).any():
            loss = order_loss
        else:
            loss = order_loss+centered_normal_loss

    elif loss_type == 'orderandbeta':
        distribution = torch.distributions.beta.Beta(
            torch.tensor([1.2],dtype=torch.float32),
            torch.tensor([4],dtype=torch.float32))
        distribution.sample(predictions.size()).squeeze(1)
        distribution = 1-distribution
        beta_dist_loss = criterion(torch.sort(predictions).values,
            torch.sort(distribution).values)
        if torch.isnan(beta_dist_loss).any():
            loss = order_loss
        else:
            loss = order_loss+beta_dist_loss

    elif loss_type == 'orderandvariational':
        std = log_vars.mul(0.5).exp_()
        epsilon = torch.randn_like(std)
        rankings = predictions + (std * epsilon)

        order_loss = pair_order_loss(rankings,ranks)
        kldivloss = -0.5 * torch.sum(1 + log_vars - predictions**2 - log_vars.exp())
        loss = order_loss + (1*kldivloss)

    else:
        loss = order_loss
        
    return loss, order_loss

'''
|  ||

|| |_
'''