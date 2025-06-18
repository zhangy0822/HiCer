"""VSE model"""
from turtle import forward
import numpy as np

import torch
import torch.nn as nn
import torch.nn.init
import torch.backends.cudnn as cudnn
from torch.nn.utils import clip_grad_norm_

from lib.encoders import get_image_encoder, get_text_encoder, Fusion_net, VisualSA
from lib.modules.resnet import ResnetFeatureExtractor

from lib.loss import ContrastiveLoss, HALLoss, get_sim, AngularLoss

import logging
import copy
import torch.nn.functional as F
from lib.modules.mlp import MLPHead
logger = logging.getLogger(__name__)


# class VSEModel(object):
class VSEModel(nn.Module):
    """
        The standard VSE model
    """
    def __init__(self, opt):
        super().__init__()
        # Build Models
        self.grad_clip = opt.grad_clip
        self.img_enc = get_image_encoder(no_imgnorm=opt.no_imgnorm, opt = opt)

        self.txt_enc = get_text_encoder(opt.embed_size, no_txtnorm=opt.no_txtnorm)

        if opt.use_moco:
            self.K = opt.moco_M
            self.m = opt.moco_r
            self.v_encoder_k = copy.deepcopy(self.img_enc)
            self.t_encoder_k = copy.deepcopy(self.txt_enc)
            for param in self.v_encoder_k.parameters():
                param.requires_grad = False
            for param in self.t_encoder_k.parameters():
                param.requires_grad = False
            self.register_buffer("t_queue", torch.rand(opt.embed_size, self.K))
            self.t_queue = F.normalize(self.t_queue, dim=0)
            self.register_buffer("v_queue", torch.rand(opt.embed_size, self.K))
            self.v_queue = F.normalize(self.v_queue, dim=0)
            self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
            self.register_buffer("id_queue", -torch.ones((1, self.K), dtype=torch.long))


        if torch.cuda.is_available():
            self.img_enc.cuda()
            self.txt_enc.cuda()
            if opt.use_moco:
                self.v_encoder_k.cuda()
                self.t_encoder_k.cuda()
                self.t_queue = self.t_queue.cuda()
                self.v_queue = self.v_queue.cuda()
                self.queue_ptr = self.queue_ptr.cuda()
                self.id_queue = self.id_queue.cuda()
            cudnn.benchmark = True


        # Loss and Optimizer

        self.hal_loss = HALLoss(opt=opt)
        
        if opt.use_angle_loss:
            self.angular_loss = AngularLoss(opt=opt)
        

        params = list(self.txt_enc.parameters())
        params += list(self.img_enc.parameters())

        self.params = params

        self.opt = opt

        # Set up the lr for different parts of the VSE model
        decay_factor = 1e-4
        if self.opt.optim == 'adam':
            all_text_params = list(self.txt_enc.parameters())
            bert_params = list(self.txt_enc.bert.parameters())
            bert_params_ptr = [p.data_ptr() for p in bert_params]
            text_params_no_bert = list()
            for p in all_text_params:
                if p.data_ptr() not in bert_params_ptr:
                    text_params_no_bert.append(p)
            all_img_params = list(self.img_enc.parameters())

            self.optimizer = torch.optim.AdamW(
                [
                {'params': text_params_no_bert, 'lr': opt.learning_rate},
                {'params': bert_params, 'lr': opt.learning_rate * 0.1},
                {'params': all_img_params, 'lr': opt.learning_rate}
                ],
                lr=opt.learning_rate, weight_decay=decay_factor)
        elif self.opt.optim == 'sgd':
            self.optimizer = torch.optim.SGD(self.params, lr=opt.learning_rate, momentum=0.9)
        else:
            raise ValueError('Invalid optim option {}'.format(self.opt.optim))

        logger.info('Use {} as the optimizer, with init lr {}'.format(self.opt.optim, opt.learning_rate))

        self.Eiters = 0
        self.data_parallel = False

    @staticmethod
    def regression_loss(x, y):
        x = F.normalize(x, dim=1)
        y = F.normalize(y, dim=1)
        return 2 - 2 * (x * y).sum(dim=-1)

    def state_dict(self):
        state_dict = [self.img_enc.state_dict(), self.txt_enc.state_dict()]

        return state_dict

    def load_state_dict(self, state_dict):
        self.img_enc.load_state_dict(state_dict[0], strict=False)
        self.txt_enc.load_state_dict(state_dict[1], strict=False)

    def train_start(self):
        """switch to train mode
        """
        self.img_enc.train()
        self.txt_enc.train()


    def val_start(self):
        """switch to evaluate mode
        """
        self.img_enc.eval()
        self.txt_enc.eval()

    # def make_data_parallel(self):
    #     self.img_enc = nn.DataParallel(self.img_enc)
    #     self.txt_enc = nn.DataParallel(self.txt_enc)
    #     self.data_parallel = True
    #     logger.info('All Nets is data paralleled now.')
        
    def get_pairs(self, sample1, sample2):
        anchors, positives = sample1, sample2
        negtives = []
        
        for i in range(len(sample1)):
            negtive_sample = torch.cat((sample2[:i,:], sample2[i+1:,:]), dim=0)
            negtives.append(negtive_sample)
        
        negtives = torch.stack(negtives, dim=0)
        
        return anchors, positives, negtives

    @property
    def is_data_parallel(self):
        return self.data_parallel

    def forward_emb(self, detections, rg_lengths, boxes, grids, masks, captions, lengths, img_ids=None, is_train = False):
        """Compute the image and caption embeddings
        """
        # Set mini-batch dataset
        # print(detections.size(), grids.size())
        img_ids = torch.tensor(img_ids).long()
        if torch.cuda.is_available():
            detections = detections.cuda()
            captions = captions.cuda()
            boxes = boxes.cuda()
            grids = grids.cuda()
            masks = masks.cuda()
            lengths = lengths.cuda()
            rg_lengths = rg_lengths.cuda()
            img_ids = img_ids.cuda()

        img_emb = self.img_enc(detections, grids, boxes, masks, rg_lengths = rg_lengths)
        cap_emb = self.txt_enc(captions, lengths)

        if is_train and self.opt.use_moco:
            N = detections.shape[0]
            with torch.no_grad():
                self._momentum_update_key_encoder()
                v_embed_k = self.v_encoder_k(detections, grids, boxes, masks, rg_lengths = rg_lengths)
                t_embed_k = self.t_encoder_k(captions, lengths)

            # regard same instance ids as positive sapmles, we need filter them out
            pos_idx = (
                self.id_queue.expand(N, self.K)
                .eq(img_ids.unsqueeze(-1))
                .nonzero(as_tuple=False)[:, 1]
            )
            unique, counts = torch.unique(
                torch.cat([torch.arange(self.K).long().cuda(), pos_idx]),
                return_counts=True,
            )
            neg_idx = unique[counts == 1]

            loss_moco1 = self.hal_loss.moco_forward(img_emb, t_embed_k, cap_emb, v_embed_k, self.v_queue, self.t_queue, neg_idx)

            self._dequeue_and_enqueue(v_embed_k, t_embed_k, img_ids)
            
            if self.opt.use_angle_loss:
                pairs_v = self.get_pairs(img_emb, cap_emb)
                pairs_t = self.get_pairs(cap_emb, img_emb)

                loss_angle_batch_v = self.angular_loss(pairs_v[0], pairs_v[1], pairs_v[2])
                loss_angle_batch_t = self.angular_loss(pairs_t[0], pairs_t[1], pairs_t[2])
                
                v_queue = self.v_queue.clone().detach().transpose(0,1)
                t_queue = self.t_queue.clone().detach().transpose(0,1)
                
                v_queue = v_queue.repeat(img_emb.size(0), 1, 1)
                t_queue = t_queue.repeat(img_emb.size(0), 1, 1)
                
                loss_angle_queue_t = self.angular_loss(img_emb, t_embed_k, t_queue)
                loss_angle_queue_v = self.angular_loss(cap_emb, v_embed_k, v_queue)
                
                loss_angle = ( loss_angle_batch_v + loss_angle_batch_t + loss_angle_queue_t + loss_angle_queue_v ) * self.opt.angle_loss_ratio / 4
                
                return img_emb, cap_emb, loss_moco1, loss_angle

            return img_emb, cap_emb, loss_moco1

        return img_emb, cap_emb

    def forward_loss(self, img_emb, cap_emb):
        """Compute the loss given pairs of image and caption embeddings
        """
        # loss = self.criterion(img_emb, cap_emb)

        loss = self.hal_loss(img_emb, cap_emb)*self.opt.loss_lamda

        # self.logger.update('Le', loss.data.item(), img_emb.size(0))
        # self.logger.update('Le_info', loss_info.data.item(), img_emb.size(0))

        return loss

    def train_emb(self, detections, rg_lengths, boxes, grids, masks, captions, lengths, img_ids):
        """One training step given images and captions.
        """
        self.Eiters += 1
        self.logger.update('Eit', self.Eiters)
        self.logger.update('lr', self.optimizer.param_groups[0]['lr'])

        # # compute the embeddings
        if self.opt.use_moco:
            
            if self.opt.use_angle_loss:
                img_emb, cap_emb, loss_moco1, loss_angualr = self.forward_emb(detections, rg_lengths, boxes, grids, masks, captions, lengths, img_ids,
                                                is_train=True)
                self.logger.update('L_angular', loss_angualr.data.item(), img_emb.size(0))   
            else:
                img_emb, cap_emb, loss_moco1 = self.forward_emb(detections, rg_lengths, boxes, grids, masks, captions, lengths, img_ids, 
                                                is_train=True)
                
            self.logger.update('L_queue', loss_moco1.data.item(), img_emb.size(0))
        
            loss_encoder = self.forward_loss(img_emb, cap_emb)
            
            self.logger.update('L_batch', loss_encoder.data.item(), img_emb.size(0))

            loss = loss_encoder + loss_moco1
            # loss = loss_encoder
            # loss = loss_moco1
            
            if self.opt.use_angle_loss:
                loss = loss + loss_angualr
              
        else:
            img_emb, cap_emb = self.forward_emb(detections, rg_lengths, boxes, grids, masks, captions, lengths,
                                                is_train=True)
            loss = self.forward_loss(img_emb, cap_emb)

        # measure accuracy and record loss
        self.optimizer.zero_grad()

        # if warmup_alpha is not None:
        #     loss = loss * warmup_alpha

        # compute gradient and update
        loss.backward()
        if self.grad_clip > 0:
            clip_grad_norm_(self.params, self.grad_clip)

        self.optimizer.step()

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.img_enc.parameters(), self.v_encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)
        for param_q, param_k in zip(self.txt_enc.parameters(), self.t_encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)

        
    @torch.no_grad()
    def _dequeue_and_enqueue(self, v_keys, t_keys, id_keys):
        batch_size = v_keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.v_queue[:, ptr : ptr + batch_size] = v_keys.T
        self.t_queue[:, ptr : ptr + batch_size] = t_keys.T
        self.id_queue[:, ptr : ptr + batch_size] = id_keys.unsqueeze(-1).T
        
        ptr = (ptr + batch_size) % self.K  # move pointer 
        self.queue_ptr[0] = ptr


def compute_sim(images, captions):
    similarities = np.matmul(images, np.matrix.transpose(captions))
    return similarities


def l2norm(X):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=1, keepdim=True).sqrt()
    X = torch.div(X, norm)
    return X