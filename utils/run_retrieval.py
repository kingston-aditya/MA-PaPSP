import torch
import numpy as np
import os

def load_retrieval_data(fil_pth):
    fl = os.listdir()

class run_retrieval(object):
    def __init__(self, fil_pth):
        super(run_retrieval,self).__init__()
        self.device = torch.device('cuda')
        self.Xr, self.Yr = load_retrieval_data(fil_pth)
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
        return ind  
