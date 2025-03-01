import torch
import numpy as np
import os
from ..configs.config import get_config

config = get_config()

# function to load the cc12m data
def load_retrieval_data(fil_pth):
    pass

class run_retrieval(object):
    def __init__(self):
        super(run_retrieval,self).__init__()
        self.device = torch.device('cuda')
        self.Xr = np.load(config["coco_train_data"])
        self.Yr = np.load(config["coco_train_data"])
        self.Xr = torch.from_numpy(self.Xr).float().to(self.device)
        self.Yr = torch.from_numpy(self.Yr).float().to(self.device)

    def retrieve_X(self,q,k):
        q = torch.from_numpy(q).float().to(self.device)
        ans = torch.matmul(self.Xr,q)
        _,ind = torch.topk(ans, k)
        ind = ind.cpu().detach().numpy()
        return self.Xr[ind], self.Yr[ind]

    def retrieve_Y(self,q,k):
        q = torch.from_numpy(q).float().to(self.device)
        ans = torch.matmul(self.Yr,q)
        _,ind = torch.topk(ans, k)
        ind = ind.cpu().detach().numpy()
        return self.Xr[ind], self.Yr[ind]  
    
class find_retrieved_items(object):
    def __init__(self):
        super(find_retrieved_items, self).__init__()
        self.ret_obj = run_retrieval()

    def get_array(self, X, k):
        a = []; b=[]
        for i in range(X.shape[0]):
            Xr, Yr = self.ret_obj.retrieve_Y(X[i,:], k)
            a.append(Xr); b.append(Yr)
        return np.asarray(a).reshape(X.shape[0], self.k, -1), np.asarray(b).reshape(X.shape[0], self.k, -1)

