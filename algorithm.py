import numpy as np
import torch
from munkres import Munkres

def get_match(D,device):
    P = torch.zeros_like(D)
    size = D.shape[0]
    index_S = [i for i in range(size)]
    index_S_hat = [i for i in range(size)]
    for i in range(size):
        cur_size = D.shape[0]
        argmin = torch.argmin(D.to(device)).item()
        r = argmin // cur_size
        c = argmin % cur_size
        P[index_S[r]][index_S_hat[c]] = 1
        index_S.remove(index_S[r])
        index_S_hat.remove(index_S_hat[c])
        D = D[torch.arange(D.size(0)) != r] #remove the row
        D = D.t()[torch.arange(D.t().size(0)) != c].t() #remove the col
    return P.t()#rows of P cor to S_hat

def hungarian(D):
    P = torch.zeros_like(D)
    matrix = D.tolist()
    m = Munkres()
    indexes = m.compute(matrix)
    total = 0
    for r,c in indexes:
        P[r][c] = 1
        total += matrix[r][c]
    #print("total cost " + str(total))
    return P.t()
