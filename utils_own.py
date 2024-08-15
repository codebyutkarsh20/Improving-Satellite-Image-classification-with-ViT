import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import random
import numpy as np
# rewrite all using torch.flip

def horizontal_flip(x):
    x1 = torch.zeros_like(x)
    for k in range(x.shape[-3]):
        x1[:,:,x.shape[-3] - 1 - k,:,:] = x[:,:,k,:,:]
        

    return x1

def vertical_flip(x):
    x1 = torch.zeros_like(x)
    
    for k in range(x.shape[-4]):
        x1[:,x.shape[-4] - 1 - k,:,:,:] = x[:,k,:,:,:]
        # x1[:,:,:,:,x.shape[-2] - 1 - k,:] = x[:,:,:,:,k,:]

    return x1

def horizontal_flip_target(x):
    
    x1 = torch.zeros_like(x)
    for k in range(x.shape[-1]):
        x1[:,:,:,:,x.shape[-1] - 1 - k] = x[:,:,:,:,k]
        

    return x1

def vertical_flip_target(x):
    x1 = torch.zeros_like(x)
    
    for k in range(x.shape[-2]):
        x1[:,:,:,x.shape[-2] - 1 - k,:] = x[:,:,:,k,:]

    return x1

def flip(x, type):
    if type == 'horizontal':
        return horizontal_flip_target(horizontal_flip(x))
    else:
        return vertical_flip_target(vertical_flip(x))


def get_acc(y_true, y_pred):
    return sum(y_true == y_pred)/len(y_true)

def get_results(output, labels, verbose = True):
    y_true = labels.cpu().detach().numpy()
    y_pred = output.softmax(dim=-1).argmax(dim = -1).cpu().detach().numpy()
    precision = precision_score(y_true=y_true, y_pred=y_pred,average="macro")
    recall = recall_score(y_true=y_true, y_pred=y_pred,average="macro")
    f1 = f1_score(y_true=y_true, y_pred=y_pred,average="macro")
    acc = accuracy_score(y_true=y_true, y_pred=y_pred)
    # acc = get_acc(y_true=y_true, y_pred=y_pred)
    
    if verbose:
        print(f'Precision: {precision}')
        print(f'Recall: {recall}')
        print(f'F1 Score: {f1}')
        print(f'Accuracy: {acc}')
    return precision, recall, f1, acc

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False