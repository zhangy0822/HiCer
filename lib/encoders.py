"""VSE modules"""

from turtle import forward
import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict

from transformers import BertModel

from lib.modules.aggr.gpo import GPO
from lib.modules.mlp import MLP
from lib.modules.attention_nets import GridAugmentedEncoder

import logging

import torch.nn.functional as F

from models.COME.encoders import TransformerEncoder
from models.COME.attention import ScaledDotProductAttention

logger = logging.getLogger(__name__)

def l1norm(X, dim, eps=1e-8):
    """L1-normalize columns of X
    """
    norm = torch.abs(X).sum(dim=dim, keepdim=True) + eps
    X = torch.div(X, norm)
    return X


def l2norm(X, dim, eps=1e-8):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X


def maxk_pool1d_var(x, dim, k, lengths):
    results = list()
    lengths = list(lengths.cpu().numpy())
    lengths = [int(x) for x in lengths]
    for idx, length in enumerate(lengths):
        k = min(k, length)
        max_k_i = maxk(x[idx, :length, :], dim - 1, k).mean(dim - 1)
        results.append(max_k_i)
    results = torch.stack(results, dim=0)
    return results


def maxk_pool1d(x, dim, k):
    max_k = maxk(x, dim, k)
    return max_k.mean(dim)


def maxk(x, dim, k):
    index = x.topk(k, dim=dim)[1]
    return x.gather(dim, index)


def get_text_encoder(embed_size, no_txtnorm=False):
    return EncoderText(embed_size, no_txtnorm=no_txtnorm)


def get_image_encoder(no_imgnorm=False, opt = None):
    """A wrapper to image encoders. Chooses between an different encoders
    that uses precomputed image features.
    """

    img_enc = TransformerEncoder(opt.n_layer, 0, attention_module=ScaledDotProductAttention,
                                 d_in=opt.img_dim,
                                 d_k=opt.d_k,
                                 d_v=opt.d_v,
                                 h=opt.head,
                                 d_model=opt.d_model,
                                 opt = opt)
    

    return img_enc


class EncoderImageAggr(nn.Module):
    def __init__(self, img_dim, embed_size, precomp_enc_type='basic', no_imgnorm=False, clip_dim = None, opt = None):
        super(EncoderImageAggr, self).__init__()
        self.embed_size = embed_size
        self.no_imgnorm = no_imgnorm
        # self.fc = nn.Linear(img_dim, embed_size)
        self.precomp_enc_type = precomp_enc_type
        self.clip_dim = clip_dim
        self.opt = opt
        # if precomp_enc_type == 'basic':
        #     self.mlp = MLP(img_dim, embed_size // 2, embed_size, 2)

        self.gpool = GPO(32, 32)

        # self.init_weights()

    def init_weights(self):
        """Xavier initialization for the fully connected layer
        """
        # r = np.sqrt(6.) / np.sqrt(self.fc.in_features +
        #                           self.fc.out_features)
        # self.fc.weight.data.uniform_(-r, r)
        # self.fc.bias.data.fill_(0)

    def forward(self, images, image_lengths, res_emb=None):
        """Extract image feature vectors."""
        # features = self.fc(images)

        # if self.precomp_enc_type == 'basic':
        #     features = self.mlp(images) + features
        features = images
        features, pool_weights = self.gpool(features, image_lengths)

        
        if self.opt.use_clip:
            if not self.no_imgnorm:
                features = l2norm(features, dim=-1)
            if self.opt.use_residual and res_emb:
                features = features + res_emb 
                return features
            else:
                return features
        else:
            if not self.no_imgnorm:
                features = l2norm(features, dim=-1)
                return features
        
class EncoderImageGrid_Region(nn.Module):
    def __init__(self, img_dim, embed_size, backbone_cnn=None,  no_imgnorm=False, opt=None):
        super(EncoderImageGrid_Region, self).__init__()
        self.img_dim = img_dim
        self.embed_size = embed_size
        self.clip_dim = opt.clip_embed_size
        self.opt = opt

        self.image_encoder = EncoderImageAggr(img_dim, embed_size, \
                                              precomp_enc_type=opt.precomp_enc_type, \
                                              no_imgnorm=opt.no_imgnorm, \
                                              opt=opt)


        self.attention = GridAugmentedEncoder()
        # self.init_weights()

    # def init_weights(self):
    #     """Xavier initialization for the fully connected layer
    #     """
    #     r = np.sqrt(6.) / np.sqrt(self.pooling_layer.in_features +
    #                             self.pooling_layer.out_features)
    #     self.pooling_layer.weight.data.uniform_(-r, r)
    #     self.pooling_layer.bias.data.fill_(0)

    def forward(self, images_regions, image_lengths, clip_images, images):
        """Extract image feature vectors."""

        if not self.opt.use_clip and self.opt.use_grid:
            
            if self.opt.finetune_bkb:
                base_features = self.backbone(images)
                base_features = base_features.permute(0,2,1)
                base_features_pooler = self.pooling_layer(base_features).permute(0,2,1)
                base_features_pooler = l2norm(base_features_pooler, -1)
                out_feature = self.attention(images_regions, base_features_pooler)
                features = self.image_encoder(out_feature, image_lengths)

            else:
                # base_features = images
                # base_features = base_features.permute(0,2,1)
                # base_features_pooler = self.pooling_layer(base_features).permute(0,2,1)
                # base_features = l2norm(base_features, -1)
                base_features = torch.mean(images_regions, dim=1, keepdim=True)
                out_feature = self.attention(images_regions, base_features)
                features = self.image_encoder(out_feature, image_lengths)
            
            return features

        elif self.opt.use_clip:
            clip_emb = self.clip_enc(clip_images)
            clip_emb_res = self.clip_enc_res(clip_images)
            # images = self.region_attention(images_regions, clip_emb)
            out_feature = self.attention(images_regions, clip_emb)
            features = self.image_encoder(out_feature, image_lengths, clip_emb_res)
            return features
        
        elif self.opt.use_gcn:
            features, gcn_img_emd = self.r_gcn(images_regions)
            out_feature = self.attention(images_regions, features.unsqueeze(1))
            final_features = self.image_encoder(out_feature, image_lengths)

            return features, final_features

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
        logger.info('Backbone freezed.')

    def unfreeze_backbone(self, fixed_blocks):
        for param in self.backbone.parameters():  # open up all params first, then adjust the base parameters
            param.requires_grad = True
        self.backbone.set_fixed_blocks(fixed_blocks)
        self.backbone.unfreeze_base()
        logger.info('Backbone unfreezed, fixed blocks {}'.format(self.backbone.get_fixed_blocks()))


    def region_attention(self, images, clip_emb):
        features_t = torch.transpose(images, 1, 2).contiguous()
        attn = torch.matmul(clip_emb.unsqueeze(1), features_t)
        attn_softmax = F.softmax(attn*self.opt.attention_lamda, dim=2)
        attn_softmax = l2norm(attn_softmax, -1)
        features = images + attn_softmax.permute(0,2,1)*(clip_emb.unsqueeze(1))

        return features


# Language Model with BERT
class EncoderText(nn.Module):
    def __init__(self, embed_size, no_txtnorm=False):
        super(EncoderText, self).__init__()
        self.embed_size = embed_size
        self.no_txtnorm = no_txtnorm

        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.linear = nn.Linear(768, embed_size)
        self.gpool = GPO(32, 32)

    def forward(self, x, lengths):
        """Handles variable size captions
        """
        # Embed word ids to vectors
        bert_attention_mask = (x != 0).float()
        # print(bert_attention_mask)
        bert_emb = self.bert(x, bert_attention_mask)[0]  # B x N x D
        cap_len = lengths

        cap_emb = self.linear(bert_emb)
        # cap_emb_raw = cap_emb.clone().detach()

        pooled_features, pool_weights = self.gpool(cap_emb, cap_len.to(cap_emb.device))

        # normalization in the joint embedding space
        if not self.no_txtnorm:
            pooled_features = l2norm(pooled_features, dim=-1)

        # return pooled_features, cap_emb_raw
        return pooled_features



class VisualSA(nn.Module):
    """
    Build global image representations by self-attention.
    Args: - local: local region embeddings, shape: (batch_size, 36, 1024)
          - raw_global: raw image by averaging regions, shape: (batch_size, 1024)
    Returns: - new_global: final image by self-attention, shape: (batch_size, 1024).
    """
    def __init__(self, embed_dim, dropout_rate, num_region):
        super(VisualSA, self).__init__()
        self.num_region = num_region
        self.embedding_local = nn.Sequential(nn.Linear(embed_dim, embed_dim),
                                            #  nn.BatchNorm1d(self.num_region),
                                            nn.LayerNorm(embed_dim),
                                             nn.Tanh(), nn.Dropout(dropout_rate))
        self.embedding_global = nn.Sequential(nn.Linear(embed_dim, embed_dim),
                                            #   nn.BatchNorm1d(embed_dim),
                                            nn.LayerNorm(embed_dim),
                                              nn.Tanh(), nn.Dropout(dropout_rate))
        self.embedding_common = nn.Sequential(nn.Linear(embed_dim, 1))

        self.init_weights()
        self.softmax = nn.Softmax(dim=1)

    def init_weights(self):
        for embeddings in self.children():
            for m in embeddings:
                if isinstance(m, nn.Linear):
                    r = np.sqrt(6.) / np.sqrt(m.in_features + m.out_features)
                    m.weight.data.uniform_(-r, r)
                    m.bias.data.fill_(0)
                elif isinstance(m, nn.BatchNorm1d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()

    def forward(self, local, raw_global):
        # compute embedding of local regions and raw global image
        l_emb = self.embedding_local(local)
        g_emb = self.embedding_global(raw_global)

        # compute the normalized weights, shape: (batch_size, 36)
        g_emb = g_emb.unsqueeze(1).repeat(1, l_emb.size(1), 1)
        common = l_emb.mul(g_emb)
        weights = self.embedding_common(common).squeeze(2)
        weights = self.softmax(weights)

        # compute final image, shape: (batch_size, 1024)
        # new_global = (weights.unsqueeze(2) * local)
        new_global = (weights.unsqueeze(2) * local).sum(dim=1)
        new_global = l2norm(new_global, dim=-1)

        return new_global


class Fusion_net(nn.Module):
    def __init__(self, embed_size):
        super(Fusion_net, self).__init__()
        self.embed_size = embed_size
        self.cat_trans = nn.Linear(self.embed_size*2, self.embed_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_l, x_g):
        x = torch.cat((x_l, x_g), dim=1)
        g = self.cat_trans(x)
        g_sig = self.sigmoid(g)
        out = g_sig * x_l + (1-g_sig) * x_g
        out = l2norm(out, dim=-1)
    
        return out
    
        
        