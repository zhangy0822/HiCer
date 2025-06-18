import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class ContrastiveLoss(nn.Module):
    """
    Compute contrastive loss (max-margin based)
    """

    def __init__(self, opt, margin=0, max_violation=False):
        super(ContrastiveLoss, self).__init__()
        self.opt = opt
        self.margin = margin
        self.max_violation = max_violation
        self.l_alpha = opt.mu
        self.l_ep = opt.gama
        # self.max_violation = True
    def max_violation_on(self):
        self.max_violation = True
        print('Use VSE++ objective.')

    def max_violation_off(self):
        self.max_violation = False
        print('Use VSE0 objective.')


    def forward(self, im, s):
        bsize = im.size()[0]
        scores = get_sim(im, s)
        
        tmp  = torch.eye(bsize).cuda()
        s_diag = tmp * scores
        scores_ = scores - s_diag
        S_ = torch.exp(self.l_alpha * (scores_ - self.l_ep))
        
        loss_diag = - torch.log(1 + F.relu(s_diag.sum(0)))

        loss = torch.sum( torch.log(1 + S_.sum(0)) / self.l_alpha + torch.log(1 + S_.sum(1)) / self.l_alpha + loss_diag ) / bsize

        return loss

    # def forward(self, im, s):
    #     # compute image-sentence score matrix
    #     scores = get_sim(im, s)
    #     diagonal = scores.diag().view(im.size(0), 1)
    #     d1 = diagonal.expand_as(scores)
    #     d2 = diagonal.t().expand_as(scores)

    #     # compare every diagonal score to scores in its column
    #     # caption retrieval
    #     cost_s = (self.margin + scores - d1).clamp(min=0)
    #     # compare every diagonal score to scores in its row
    #     # image retrieval
    #     cost_im = (self.margin + scores - d2).clamp(min=0)

    #     # clear diagonals
    #     mask = torch.eye(scores.size(0)) > .5
    #     I = Variable(mask)
    #     if torch.cuda.is_available():
    #         I = I.cuda()
    #     cost_s = cost_s.masked_fill_(I, 0)
    #     cost_im = cost_im.masked_fill_(I, 0)

    #     # keep the maximum violating negative for each query
    #     if self.max_violation:
    #         cost_s = cost_s.max(1)[0]
    #         cost_im = cost_im.max(0)[0]

    #     return cost_s.sum() + cost_im.sum()




def get_sim(images, captions):
    similarities = images.mm(captions.t())
    return similarities


class infoNCELoss(nn.Module):
    """
    Compute infoNCE loss
    """
    def __init__(self, tau=1):
        super(infoNCELoss, self).__init__()
        self.tau = tau

    def forward(self, im, s):                                             #  scores: (bsize, bsize)
        scores = get_sim(im, s)
        bsize, bsize = scores.size()
        scores = self.tau * scores.clamp(min=-1e10)
        d1 = F.log_softmax(scores, dim=1)                                  #  文本为正例，图片为负例的loss
        d2 = F.log_softmax(scores, dim=0)                                  #  图片为正例，文本为负例的loss

        loss_s = torch.sum(d1.diag())
        loss_im = torch.sum(d2.diag())
        loss_infoNCE = -1 * (loss_s + loss_im) / bsize                     #  infoNCE Loss

        return loss_infoNCE

def infonce_loss(
    v_pos,
    v_neg,
    t_pos,
    t_neg,
    T=0.07,
):
    v_logits = torch.cat([v_pos, v_neg], dim=1) / T
    t_logits = torch.cat([t_pos, t_neg], dim=1) / T
    labels = torch.zeros(v_logits.shape[0], dtype=torch.long).cuda()
    loss = F.cross_entropy(v_logits, labels) + F.cross_entropy(t_logits, labels)
    return loss


class HALLoss(nn.Module):
    def __init__(self, opt):
        super(HALLoss, self).__init__()
        self.opt = opt
        self.l_alpha = opt.mu
        self.l_ep = opt.gama

    def forward(self, im, s ):

        bsize = im.size()[0]
        scores = get_sim(im, s)    # 128*128, 一个batch中的相似度
    
        tmp  = torch.eye(bsize).cuda()   #
        s_diag = tmp * scores        # 保留对角线，其余权威0
        scores_ = scores - s_diag      # 对角线全为0
        S_ = torch.exp(self.l_alpha * (scores_ - self.l_ep))
    
        loss_diag_1 = - torch.log(1 + F.relu(s_diag.sum(0)))

        # loss_diag_2 = - torch.log(1 + F.relu(s_diag.sum(1)))

        # loss_1 = torch.sum( torch.log(1 + S_.sum(0)) / self.l_alpha + loss_diag_1 ) / bsize

        # loss_2 = torch.sum( torch.log(1 + S_.sum(1)) / self.l_alpha + loss_diag_2 ) / bsize

        # loss = loss_1 + loss_2

        loss = torch.sum( torch.log(1 + S_.sum(0)) / self.l_alpha + torch.log(1 + S_.sum(1)) / self.l_alpha + loss_diag_1) / bsize

        return loss
    def moco_forward(self, v_q, t_k, t_q, v_k, v_queue, t_queue, neg_idx):

        # v positive logits: Nx1
        v_pos = torch.einsum("nc,nc->n", [v_q, t_k]).unsqueeze(-1)
        # v negative logits: NxK
        t_queue = t_queue.clone().detach()
        # t_queue = t_queue[:,neg_idx] 
        v_neg = torch.einsum("nc,ck->nk", [v_q, t_queue])

        # t positive logits: Nx1
        t_pos = torch.einsum("nc,nc->n", [t_q, v_k]).unsqueeze(-1)
        # t negative logits: NxK
        v_queue = v_queue.clone().detach()
        # v_queue = v_queue[:,neg_idx]
        t_neg = torch.einsum("nc,ck->nk", [t_q, v_queue])

        v_pos_diag = torch.diag_embed(v_pos.squeeze(-1))
        v_bsize = v_pos_diag.size()[0]

        # if self.opt.use_mod:
        #     sim_i2t_m = torch.einsum("nc,ck->nk", [v_k, t_queue])
        #     v_pos_diag = self.opt.mod_alpha * sim_i2t_m + (1-self.opt.mod_alpha)*v_pos_diag

        v_loss_diag = - torch.log(1 + F.relu(v_pos_diag.sum(0)))
        v_S_ = torch.exp((v_neg - self.l_ep) * self.l_alpha)
        v_S_T = v_S_.T
        v_loss = torch.sum( torch.log(1 + v_S_T.sum(0)) / self.l_alpha + v_loss_diag) / v_bsize

        t_pos_diag = torch.diag_embed(t_pos.squeeze(-1))
        t_bsize = t_pos_diag.size()[0]

        # if self.opt.use_mod:
        #     sim_t2i_m = torch.einsum("nc,ck->nk", [t_k, v_queue])
        #     t_pos_diag = self.opt.mod_alpha * sim_t2i_m + (1-self.opt.mod_alpha)*t_pos_diag

        t_loss_diag = - torch.log(1 + F.relu(t_pos_diag.sum(0)))
        t_S_ = torch.exp((t_neg - self.l_ep) * self.l_alpha)
        t_S_T = t_S_.T
        t_loss = torch.sum( torch.log(1 + t_S_T.sum(0)) / self.l_alpha + t_loss_diag) / t_bsize

        return v_loss + t_loss


class AngularLoss(nn.Module):
    """
    Angular loss
    Wang, Jian. "Deep Metric Learning with Angular Loss," ICCV, 2017
    https://arxiv.org/pdf/1708.01682.pdf
    """

    def __init__(self, l2_reg=0.02, angle_bound=1., lambda_ang=2, opt=None):
        super(AngularLoss, self).__init__()
        self.l2_reg = l2_reg
        self.angle_bound = angle_bound
        self.lambda_ang = lambda_ang
        self.softplus = nn.Softplus()

    def forward(self, anchors, positives, negatives):

        # anchors = embeddings[n_pairs[:, 0]]  # (n, embedding_size)
        # positives = embeddings[n_pairs[:, 1]]  # (n, embedding_size)
        # negatives = embeddings[n_negatives]  # (n, n-1, embedding_size)

        losses = self.angular_loss(anchors, positives, negatives, self.angle_bound) \
                 + self.l2_reg * self.l2_loss(anchors, positives)

        return losses

    @staticmethod
    def angular_loss(anchors, positives, negatives, angle_bound=1.):
        """
        Calculates angular loss
        :param anchors: A torch.Tensor, (n, embedding_size)
        :param positives: A torch.Tensor, (n, embedding_size)
        :param negatives: A torch.Tensor, (n, n-1, embedding_size)
        :param angle_bound: tan^2 angle
        :return: A scalar
        """
        anchors = torch.unsqueeze(anchors, dim=1)  # (n, 1, embedding_size)
        positives = torch.unsqueeze(positives, dim=1)  # (n, 1, embedding_size)

        x = 4. * angle_bound * torch.matmul((anchors + positives), negatives.transpose(1, 2)) \
            - 2. * (1. + angle_bound) * torch.matmul(anchors, positives.transpose(1, 2))  # (n, 1, n-1)

        # Preventing overflow
        with torch.no_grad():
            t = torch.max(x, dim=2)[0]

        x = torch.exp(x - t.unsqueeze(dim=1))
        x = torch.log(torch.exp(-t) + torch.sum(x, 2))
        loss = torch.mean(t + x)

        return loss
    
    
    @staticmethod
    def l2_loss(anchors, positives):
        """
        Calculates L2 norm regularization loss
        :param anchors: A torch.Tensor, (n, embedding_size)
        :param positives: A torch.Tensor, (n, embedding_size)
        :return: A scalar
        """
        return torch.sum(anchors ** 2 + positives ** 2) / anchors.shape[0]