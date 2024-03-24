
import numpy as np
import torch
import torch.nn as nn 
import sklearn.metrics as metrics
from torch_geometric.loader import DataLoader

from utils import custom_const_target


def make_null_split(dataset, null_size, test_size, out_label=1, sparsity=0.1, train_calib_ratio=0.5):
    """
    dataset: torch-geometric Dataset object  
    """
    labels=np.array([x.y.numpy()[0] for x in dataset]).astype(float)
    outlr_idx = np.arange(len(dataset))[labels==out_label]
    inlr_idx = np.arange(len(dataset))[labels!=out_label]
    outlr= dataset[outlr_idx]
    inlr= dataset[inlr_idx]
    
    n_test1 = int( sparsity* test_size )
    n_test0 = test_size - n_test1

    if (null_size + n_test0 > len(inlr)) or (n_test1 > len(outlr)):
        raise ValueError("Review sample sizes: Num of outliers {} Num of inliers {}".format(len(outlr), len(inlr)))

    
    #make null sample and test sample
    outlr=outlr.shuffle()
    test1 = outlr[:n_test1]

    inlr=inlr.shuffle()
    test0 = inlr[:n_test0]

    null_dataset = inlr[n_test0:n_test0+null_size]
    test_dataset = test0 + test1 #is of type ConcatDataset

    #split randomly null sample to make training sample 
    null_dataset = null_dataset.shuffle()
    k= int (null_size * train_calib_ratio)

    null_train_dataset = null_dataset[:k]
    null_calib_dataset = null_dataset[k:]

    return null_train_dataset, null_calib_dataset, test0, test1, test_dataset


def make_loaders(test0, test1, test_dataset, null_train_dataset, null_calib_dataset, batch_size): 
    
    train_dataset = torch.utils.data.ConcatDataset([custom_const_target(null_train_dataset, 0)] 
                    + [custom_const_target(dt, 1) for dt in [null_calib_dataset, test0, test1]]
                    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    calib_loader = DataLoader(null_calib_dataset, batch_size=batch_size, shuffle=False) 
    
    test_dataset = torch.utils.data.ConcatDataset([custom_const_target(test0, 0), custom_const_target(test1, 1)])
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)  
    
    return train_loader, test_loader, calib_loader


def make_train_test_loaders(dataset, 
                        null_size, test_size, out_label=1, sparsity=0.1, train_calib_ratio=0.5, 
                        batch_size=32):
    """
    dataset: torch-geometric Dataset object  
    """
    null_train_dataset, null_calib_dataset, test0, test1, test_dataset = make_null_split(dataset, 
                        null_size=null_size, test_size=test_size, out_label=out_label, sparsity=sparsity, train_calib_ratio=train_calib_ratio)

    return make_loaders(test0, test1, test_dataset, null_train_dataset, null_calib_dataset, batch_size)




def evaluate(data_loader, model):
    model.eval()

    preds=[]
    ypreds=[]
    labels=[]
    for data in data_loader:
        pred = model(data)
        preds.append(pred.detach().numpy())
        _, indices = torch.max(pred, 1)
        ypreds.append(indices.detach().numpy())
        labels.append(data.y.numpy())

    labels = np.hstack(labels)
    preds = np.vstack(preds)
    ypreds = np.hstack(ypreds)

    return preds, labels, metrics.accuracy_score(labels, ypreds)#, metrics.f1_score(labels, ypreds)

def train(dataset, model,
        val_dataset=None,
        clip=0.0, num_epochs=1000, lr=0.01,
        log=False, num_log_epochs=10):
    """
    dataset: object of type DataLoader 
    """
    
    optimizer = torch.optim.Adam(filter(lambda p : p.requires_grad, model.parameters()), lr=lr)
    
    for epoch in range(num_epochs):

        model.train()

        avg_loss = 0.0

        for batch_idx, data in enumerate(dataset):
            
            optimizer.zero_grad()
            output = model(data)

            loss = model.loss(output, data.y.long())
            loss.backward()
            if clip is not None: nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()

            avg_loss += loss.item()

        avg_loss /= batch_idx + 1
        if log==True:
            if epoch==0 or (epoch+1)%num_log_epochs==0: 

                if val_dataset is not None:
                    _, _, val_acc = evaluate(val_dataset, model)
                    print('Epoch: ', epoch+1, '; Avg loss: ', avg_loss, "; Val acc:", val_acc)
                else: print('Epoch: ', epoch+1, '; Avg loss: ', avg_loss)

    return model





