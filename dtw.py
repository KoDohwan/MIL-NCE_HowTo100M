import torch
import torch.nn as nn
import math
  
class DTW(nn.Module):
    def __init__(self, use_cuda=True):
        super(DTW, self).__init__()
        self.use_cuda = use_cuda
    
    def _cosine_dist_func(self, x, y):
        """
        Calculates the Cosine distance between each element in x and y per timestep
        """
        n = x.size(1)
        m = y.size(1)
        d = x.size(2)
        x = x.unsqueeze(2).expand(-1, n, m, d)
        y = y.unsqueeze(1).expand(-1, n, m, d)
        distance = 1 - torch.nn.functional.cosine_similarity(x, y, dim=3)
        return distance
    
    def forward(self, x, y):
        cost = self._cosine_dist_func(x, y).double().cuda()

        row = cost.shape[1]
        col = cost.shape[2]
        
        if self.use_cuda:
            tc = torch.ones_like(cost).double().cuda() * math.inf
        else:
            tc = torch.ones_like(cost) * math.inf
        path = torch.zeros_like(cost).double().cuda()
        path[:, row-1, col-1] = 1
    
        tc[:, 0, 0] = cost[:, 0, 0]
    
        # Initialize first column of total cost(tc) array
        # for i in range(1, m + 1):
        #     tc[i][0] = tc[i-1][0] + cost[i][0]
        for i in range(1, row):
            tc[:, i, 0] = tc[:, i-1, 0] + cost[:, i, 0]
    
        # Initialize first row of tc array
        # for j in range(1, n + 1):
        #     tc[0][j] = tc[0][j-1] + cost[0][j]
        for j in range(1, col):
            tc[:, 0, j] = tc[:, 0, j-1] + cost[:, 0, j]
    
        # Construct rest of the tc array
        for i in range(1, row):
            for j in range(1, col):
                val = torch.min(torch.min(tc[:, i-1, j-1], tc[:, i-1, j]), tc[:, i, j-1]).detach()
                tc[:, i, j] = val + cost[:, i, j]
        
        
        for b in range(cost.shape[0]):
            i = row - 1
            j = col - 1
            while not (i == 0 or j == 0):
                if (tc[b, i, j] - cost[b, i, j]).item() == tc[b, i-1, j-1].item():
                    path[b, i-1, j-1] = 1
                    i -= 1
                    j -= 1
                elif (tc[b, i, j] - cost[b, i, j]).item() == (tc[b, i-1, j]).item():
                    path[b, i-1, j] = 1
                    i -= 1
                elif (tc[b, i, j] - cost[b, i, j]).item() == tc[b, i, j-1].item():
                    path[b, i, j-1] = 1
                    j -= 1
                else:
                    print('error')
            path[b, 0, 0] = 1
        pos = torch.logsumexp(torch.sum(cost * path, dim=1), dim=1)
        neg = torch.logsumexp(torch.sum(cost, dim=1), dim=1)
        return pos - neg
    
  
# Driver program to test above functions
if __name__ == '__main__':
    x = torch.rand((4, 8, 512)).cuda()
    y = torch.rand((4, 8, 512)).cuda()
    dtw = DTW(use_cuda=True)
    loss = dtw(x, y)
    print(loss)
    loss.mean().backward()