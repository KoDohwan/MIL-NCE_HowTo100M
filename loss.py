import torch as th
import torch.nn as nn
from soft_dtw_cuda import SoftDTW
from dtw import DTW

class MILNCELoss(th.nn.Module):
    def __init__(self):
        super(MILNCELoss, self).__init__()

    def forward(self, video_embd, text_embd):
        x = th.matmul(video_embd, text_embd.t())
        x = x.view(video_embd.shape[0], video_embd.shape[0], -1)
        nominator = x * th.eye(x.shape[0])[:,:,None].cuda()
        nominator = nominator.sum(dim=1)
        nominator = th.logsumexp(nominator, dim=1)
        denominator = th.cat((x, x.permute(1,0,2)), dim=1).view(x.shape[0], -1)
        denominator = th.logsumexp(denominator, dim=1)
        return th.mean(denominator - nominator)

class CDTW(th.nn.Module):
    def __init__(self, args):
        super(CDTW, self).__init__()
        self.args = args

    def forward(self, video_embd, text_embd):
        sdtw = SoftDTW(use_cuda=True, gamma=1e-5, dist_func='cosine')
        # sdtw = DTW(use_cuda=True)
        device = self.args.rank
        pos = sdtw(video_embd[device].unsqueeze(0), text_embd[device].unsqueeze(0))
        neg = sdtw(video_embd[device].repeat(8, 1, 1), text_embd)
        loss = (pos - th.logsumexp(neg, 0)).unsqueeze(0)
        return loss

class SDTW_CIDM(th.nn.Module):
    def __init__(self, args):
        super(SDTW_CIDM, self).__init__()
        self.args = args
        self.sdtw = SoftDTW(use_cuda=True, gamma=1e-1, dist_func='cosine')

    def _cosine_dist_func(self, x, y):
        n = x.size(1)
        m = y.size(1)
        d = x.size(2)
        x = x.unsqueeze(2).expand(-1, n, m, d)
        y = y.unsqueeze(1).expand(-1, n, m, d)
        distance = 1 - th.nn.functional.cosine_similarity(x, y, dim=3)
        return distance
    
    def _pairwise_dist_func(self, x, y):
        n = x.size(1)
        m = y.size(1)
        x = x.unsqueeze(2).expand(-1, n, m)
        y = y.unsqueeze(1).expand(-1, n, m)
        return th.abs(x - y)

    def forward(self, video_embd, text_embd, start, end):
        lam = 1.
        sigma = 10
        distance = self._pairwise_dist_func(start, start)
        y = th.where(distance > sigma, 1., 0.)
        w_ = distance + 1
        w = 1 / w_
        D_x = self._cosine_dist_func(video_embd, video_embd)
        D_y = self._cosine_dist_func(text_embd, text_embd)
        I_x = (y * w_ * nn.ReLU()(lam - D_x) + (1 - y) * w * D_x).sum(1).sum(1)
        I_y = (y * w_ * nn.ReLU()(lam - D_y) + (1 - y) * w * D_y).sum(1).sum(1)
        dtw = self.sdtw(video_embd, text_embd)
        return th.mean(I_x + I_y + dtw)

class SDTW_negative(th.nn.Module):
    def __init__(self, args):
        super(SDTW_negative, self).__init__()
        self.args = args
        self.sdtw = SoftDTW(use_cuda=True, gamma=1e-1, dist_func='cosine')
        self.device = self.args.rank

    def forward(self, video_embd, text_embd):
        sdtw_loss = self.sdtw(video_embd, text_embd)
        pairwise = th.matmul(video_embd.view(-1, 512), text_embd.view(-1, 512).t())

        pairwise = th.chunk(pairwise, 160, 0)
        pairwise = th.cat(pairwise, 1)
        mask = [1288 * i + j for i in range(160) for j in range(8)]
        pairwise[:, mask] = 0.
        pairwise = th.chunk(pairwise, 160, 1)
        pairwise = th.cat(pairwise, 0)

        negative_loss = th.exp(pairwise).sum(1).view(160, 8).sum(1)

        loss = th.mean(sdtw_loss + negative_loss / 159)
        return loss

class SDTW_3(th.nn.Module):
    def __init__(self, args):
        super(SDTW_3, self).__init__()
        self.args = args
        self.sdtw = SoftDTW(use_cuda=True, gamma=1e-1, dist_func='negative_dot')
        self.device = self.args.rank

    def video_video(self, video_embd):
        b, n, d = video_embd.shape
        pos = -self.sdtw(video_embd, video_embd)
        video_embd_row = video_embd.unsqueeze(0).expand(b, b, n ,d).reshape(-1, n ,d)
        video_embd_col = video_embd.unsqueeze(1).expand(b, b, n ,d).reshape(-1, n, d)
        neg = -self.sdtw(video_embd_row, video_embd_col).reshape(b, b)
        neg = th.logsumexp(neg, 1)
        loss = th.mean(neg - pos)
        return loss

    def video_text(self, video_embd, text_embd):
        b, n, d = video_embd.shape
        pos = -self.sdtw(video_embd, text_embd)
        video_embd_row = video_embd.unsqueeze(0).expand(b, b, n ,d).reshape(-1, n ,d)
        text_embd_col = text_embd.unsqueeze(1).expand(b, b, n ,d).reshape(-1, n, d)
        neg = -self.sdtw(video_embd_row, text_embd_col).reshape(b, b)
        neg = th.logsumexp(neg, 1)
        loss = th.mean(neg - pos)
        return loss

    def text_text(self, text_embd):
        b, n, d = text_embd.shape
        pos = -self.sdtw(text_embd, text_embd)
        text_embd_row = text_embd.unsqueeze(0).expand(b, b, n ,d).reshape(-1, n ,d)
        text_embd_col = text_embd.unsqueeze(1).expand(b, b, n ,d).reshape(-1, n, d)
        neg = -self.sdtw(text_embd_row, text_embd_col).reshape(b, b)
        neg = th.logsumexp(neg, 1)
        loss = th.mean(neg - pos)
        return loss

    def forward(self, video_embd, text_embd):
        loss1 = self.video_video(video_embd)
        loss2 = self.video_text(video_embd, text_embd)
        loss3 = self.text_text(text_embd)
        return loss1, loss2, loss3
       
# class CIDM(th.nn.Module):
#     def __init__(self, args):
#         super(CIDM, self).__init__()
#         self.args = args
#         self.sdtw = SoftDTW(use_cuda=True, gamma=1e-1, dist_func='cosine')
#         self.dtw = DTW(use_cuda=True)
#         self.sigma = 10
#         self.device = self.args.rank


#     def _cosine_dist_func(self, x, y):
#         n = x.size(1)
#         m = y.size(1)
#         d = x.size(2)
#         x = x.unsqueeze(2).expand(-1, n, m, d)
#         y = y.unsqueeze(1).expand(-1, n, m, d)
#         distance = 1 - th.nn.functional.cosine_similarity(x, y, dim=3)
#         return distance
    
#     def _pairwise_dist_func(self, x, y):
#         n = x.size(1)
#         m = y.size(1)
#         x = x.unsqueeze(2).expand(-1, n, m)
#         y = y.unsqueeze(1).expand(-1, n, m)
#         return th.abs(x - y)
    
#     def forward_video_video(self, video_embd, y):
#         small_video_embd = video_embd[4*self.device:4*self.device+4]
#         D = self._cosine_dist_func(small_video_embd, small_video_embd)
#         # pos = th.exp(((1 - y) * (1 - D))).sum((1, 2))
#         pos = th.exp(1 - D).sum((1, 2))

#         input_video_embd1 = video_embd[4*self.device].repeat(159, 1, 1)
#         target_video_embd1 = th.cat([video_embd[:4*self.device], video_embd[4*self.device+1:]], dim=0)
#         neg1 = 1 - self._cosine_dist_func(input_video_embd1, target_video_embd1)
#         neg1 = th.exp(neg1).sum()

#         input_video_embd2 = video_embd[4*self.device+1].repeat(159, 1, 1)
#         target_video_embd2 = th.cat([video_embd[:4*self.device+1], video_embd[4*self.device+2:]], dim=0)
#         neg2 = 1 - self._cosine_dist_func(input_video_embd2, target_video_embd2)
#         neg2 = th.exp(neg2).sum()

#         input_video_embd3 = video_embd[4*self.device+2].repeat(159, 1, 1)
#         target_video_embd3 = th.cat([video_embd[:4*self.device+2], video_embd[4*self.device+3:]], dim=0)
#         neg3 = 1 - self._cosine_dist_func(input_video_embd3, target_video_embd3)
#         neg3 = th.exp(neg3).sum()

#         input_video_embd4 = video_embd[4*self.device+3].repeat(159, 1, 1)
#         target_video_embd4 = th.cat([video_embd[:4*self.device+3], video_embd[4*self.device+4:]], dim=0)
#         neg4 = 1 - self._cosine_dist_func(input_video_embd4, target_video_embd4)
#         neg4 = th.exp(neg4).sum()

#         loss = th.log(pos[0] + neg1) - th.log(pos[0]) + th.log(pos[1] + neg2) - th.log(pos[1]) + th.log(pos[2] + neg3) - th.log(pos[2]) + th.log(pos[3] + neg4) - th.log(pos[3])
#         return loss.unsqueeze(0) / 4
    
#     def forward_text_text(self, text_embd, y):
#         small_text_embd = text_embd[4*self.device:4*self.device+4]
#         D = self._cosine_dist_func(small_text_embd, small_text_embd)
#         pos = th.exp(-(1 - y) * D).sum((1, 2))

#         input_text_embd1 = text_embd[4*self.device].repeat(159, 1, 1)
#         target_text_embd1 = th.cat([text_embd[:4*self.device], text_embd[4*self.device+1:]], dim=0)
#         neg1 = th.exp(-self._cosine_dist_func(input_text_embd1, target_text_embd1)).sum()

#         input_text_embd2 = text_embd[4*self.device+1].repeat(159, 1, 1)
#         target_text_embd2 = th.cat([text_embd[:4*self.device+1], text_embd[4*self.device+2:]], dim=0)
#         neg2 = th.exp(-self._cosine_dist_func(input_text_embd2, target_text_embd2)).sum()

#         input_text_embd3 = text_embd[4*self.device+2].repeat(159, 1, 1)
#         target_text_embd3 = th.cat([text_embd[:4*self.device+2], text_embd[4*self.device+3:]], dim=0)
#         neg3 = th.exp(-self._cosine_dist_func(input_text_embd3, target_text_embd3)).sum()

#         input_text_embd4 = text_embd[4*self.device+3].repeat(159, 1, 1)
#         target_text_embd4 = th.cat([text_embd[:4*self.device+3], text_embd[4*self.device+4:]], dim=0)
#         neg4 = th.exp(-self._cosine_dist_func(input_text_embd4, target_text_embd4)).sum()

#         loss = th.log(neg1 + pos[0]) - th.log(pos[0]) + th.log(neg2 + pos[1]) - th.log(pos[1]) + th.log(neg3 + pos[2]) - th.log(pos[2]) + th.log(neg4 + pos[3]) - th.log(pos[3])
#         return loss.unsqueeze(0) / 4

#     def forward_video_text(self, video_embd, text_embd):
#         small_video_embd = video_embd[4*self.device:4*self.device+4]
#         small_text_embd = text_embd[4*self.device:4*self.device+4]
#         loss = self.sdtw(small_video_embd, small_text_embd)
#         return th.mean(loss).unsqueeze(0)

#     def forward(self, video_embd, text_embd, start, end):
#         distance = self._pairwise_dist_func(start[4*self.device:4*self.device+4], start[4*self.device:4*self.device+4])
#         y = th.where(distance > self.sigma, 1., 0.)
#         loss1 = self.forward_video_video(video_embd, y)
#         # loss2 = self.forward_video_text(video_embd, text_embd)
#         loss3 = self.forward_text_text(text_embd, y)
#         # print(loss1.item(), loss3.item())
#         return loss1 + loss3 * 1e-10