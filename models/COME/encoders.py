from torch.nn import functional as F
from models.COME.utils import PositionWiseFeedForward
import torch
from torch import nn
from models.COME.attention import MultiHeadBoxAttention as MultiHeadAttention
from ..relative_embedding import BoxRelationalEmbedding, GridRelationalEmbedding, AllRelationalEmbedding
from .. import position_encoding as pe
from lib.modules.aggr.gpo import GPO

def l2norm(X, dim, eps=1e-8):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X

class SelfAtt(nn.Module):
    def __init__(self, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1, identity_map_reordering=False,
                 attention_module=None, attention_module_kwargs=None):
        super(SelfAtt, self).__init__()
        self.identity_map_reordering = identity_map_reordering
        self.mhatt = MultiHeadAttention(d_model, d_k, d_v, h, dropout, identity_map_reordering=identity_map_reordering,
                                        attention_module=attention_module,
                                        attention_module_kwargs=attention_module_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.lnorm = nn.LayerNorm(d_model)
        self.pwff = PositionWiseFeedForward(d_model, d_ff, dropout, identity_map_reordering=identity_map_reordering)

    def forward(self, queries, keys, values, relative_geometry_weights, attention_mask=None, attention_weights=None,
                pos=None):
        # print('-' * 50)
        # print('layer input')
        # print(queries[11])
        q = queries + pos
        k = keys + pos
        values = values + pos # zy add
        att = self.mhatt(q, k, values, relative_geometry_weights, attention_mask, attention_weights)
        # print('mhatt outpout')
        # print(att[11])
        att = self.lnorm(queries + self.dropout(att))
        # print('norm out')
        # print(att[11])
        ff = self.pwff(att)
        # print('ff out')
        # print(ff[11])
        # print('-' * 50)
        return ff


class LCCA(nn.Module):
    def __init__(self, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1, identity_map_reordering=False,
                 attention_module=None, attention_module_kwargs=None):
        super(LCCA, self).__init__()
        self.identity_map_reordering = identity_map_reordering
        self.mhatt = MultiHeadAttention(d_model, d_k, d_v, h, dropout, identity_map_reordering=identity_map_reordering,
                                        attention_module=attention_module,
                                        attention_module_kwargs=attention_module_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.lnorm = nn.LayerNorm(d_model)
        self.pwff = PositionWiseFeedForward(d_model, d_ff, dropout, identity_map_reordering=identity_map_reordering)

    def forward(self, queries, keys, values, relative_geometry_weights, attention_mask=None, attention_weights=None,
                pos_source=None, pos_cross=None):
        # print('-' * 50)
        # print('layer input')
        # print(queries[11])
        q = queries + pos_source
        k = keys + pos_cross
        values = values + pos_cross # zy add
        att = self.mhatt(q, k, values, relative_geometry_weights, attention_mask, attention_weights)
        # print('mhatt outpout')
        # print(att[11])
        att = self.lnorm(queries + self.dropout(att))
        # print('norm out')
        # print(att[11])
        ff = self.pwff(att)
        # print('ff out')
        # print(ff[11])
        # print('-' * 50)
        return ff


class MultiLevelEncoder(nn.Module):
    def __init__(self, N, padding_idx, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1,
                 identity_map_reordering=False, attention_module=None, attention_module_kwargs=None,
                 opt=None):
        super(MultiLevelEncoder, self).__init__()
        self.opt= opt
        self.d_model = d_model
        self.dropout = dropout
        self.layers_region = nn.ModuleList([SelfAtt(d_model, d_k, d_v, h, d_ff, dropout,
                                                         identity_map_reordering=identity_map_reordering,
                                                         attention_module=attention_module,
                                                         attention_module_kwargs=attention_module_kwargs)
                                            for _ in range(N)])
        self.layers_grid = nn.ModuleList([SelfAtt(d_model, d_k, d_v, h, d_ff, dropout,
                                                       identity_map_reordering=identity_map_reordering,
                                                       attention_module=attention_module,
                                                       attention_module_kwargs=attention_module_kwargs)
                                          for _ in range(N)])
        if self.opt.train_mode == 'r2g':
            self.region2grid = nn.ModuleList([LCCA(d_model, d_k, d_v, h, d_ff, dropout,
                                                            identity_map_reordering=identity_map_reordering,
                                                            attention_module=attention_module,
                                                            attention_module_kwargs=attention_module_kwargs)
                                          for _ in range(N)])
        else:
            assert self.opt.train_mode == 'g2r'
            self.grid2region = nn.ModuleList([LCCA(d_model, d_k, d_v, h, d_ff, dropout,
                                                            identity_map_reordering=identity_map_reordering,
                                                            attention_module=attention_module,
                                                            attention_module_kwargs=attention_module_kwargs)
                                          for _ in range(N)])
        
        self.padding_idx = padding_idx

        self.WGs = nn.ModuleList([nn.Linear(128, 1, bias=True) for _ in range(h)])


        self.gpool = GPO(opt.gpo_step, opt.gpo_step)
        

    def forward(self, regions, grids, boxes, aligns, attention_weights=None, region_embed=None, grid_embed=None, rg_lengths = None):
        # input (b_s, seq_len, d_in)
        attention_mask_region = (torch.sum(regions == 0, -1) != 0).unsqueeze(1).unsqueeze(1)  # (b_s, 1, 1, seq_len)
        attention_mask_grid = (torch.sum(grids == 0, -1) != 0).unsqueeze(1).unsqueeze(1)  # (b_s, 1, 1, seq_len)

        # box embedding
        relative_geometry_embeddings = AllRelationalEmbedding(boxes)
        flatten_relative_geometry_embeddings = relative_geometry_embeddings.view(-1, 128)
        box_size_per_head = list(relative_geometry_embeddings.shape[:3])
        box_size_per_head.insert(1, 1)
        relative_geometry_weights_per_head = [l(flatten_relative_geometry_embeddings).view(box_size_per_head) for l in
                                              self.WGs]
        relative_geometry_weights = torch.cat((relative_geometry_weights_per_head), 1)
        relative_geometry_weights = F.relu(relative_geometry_weights)

        n_regions = regions.shape[1]  # 50
        n_grids = grids.shape[1]  # 49

        region2region = relative_geometry_weights[:, :, :n_regions, :n_regions]
        grid2grid = relative_geometry_weights[:, :, n_regions:, n_regions:]
        # region2grid = relative_geometry_weights[:, :, :n_regions, n_regions:]
        # grid2region = relative_geometry_weights[:, :, n_regions:, :n_regions]
        region2all = relative_geometry_weights[:,:,:n_regions,:]
        grid2all = relative_geometry_weights[:, :, n_regions:, :]

        bs = regions.shape[0]

        outs = []
        out_region = regions
        out_grid = grids
        aligns = aligns.unsqueeze(1)  # bs * 1 * n_regions * n_grids

        tmp_mask = torch.eye(n_regions, device=out_region.device).unsqueeze(0).unsqueeze(0)
        tmp_mask = tmp_mask.repeat(bs, 1, 1, 1)  # bs * 1 * n_regions * n_regions
        region_aligns = (torch.cat([tmp_mask, aligns], dim=-1) == 0) # bs * 1 * n_regions *(n_regions+n_grids)

        tmp_mask = torch.eye(n_grids, device=out_region.device).unsqueeze(0).unsqueeze(0)
        tmp_mask = tmp_mask.repeat(bs, 1, 1, 1)  # bs * 1 * n_grids * n_grids
        grid_aligns = (torch.cat([aligns.permute(0, 1, 3, 2), tmp_mask], dim=-1)==0) # bs * 1 * n_grids *(n_grids+n_regions)

        pos_cross = torch.cat([region_embed,grid_embed],dim=-2)
        
        if self.opt.train_mode == 'r2g':
            
            multi_layer_output = []
            
            for l_region, l_grid, l_r2g in zip(self.layers_region, self.layers_grid, self.region2grid):

                out_region = l_region(out_region, out_region, out_region, region2region, attention_mask_region,
                                attention_weights, pos=region_embed)
                
                out_grid = l_grid(out_grid, out_grid, out_grid, grid2grid, attention_mask_grid, attention_weights,
                            pos=grid_embed)

                out_all = torch.cat([out_region, out_grid], dim=1)


                out_region = l_r2g(out_region, out_all, out_all, region2all, region_aligns, attention_weights,
                                pos_source=region_embed, pos_cross=pos_cross)
                
                multi_layer_output.append(out_region)
                                  
            features = 0
            for layer in multi_layer_output:
                features_1, pool_weights_1 = self.gpool(layer, rg_lengths)
                features = features + features_1

            # features, pool_weights_1 = self.gpool(multi_layer_output[-1], rg_lengths)

            features = l2norm(features, dim=-1)
            
            return features

        else:
            assert self.opt.train_mode == 'g2r'
            multi_layer_output = []
            
            for l_region, l_grid, l_g2r in zip(self.layers_region, self.layers_grid, self.grid2region):

                out_region = l_region(out_region, out_region, out_region, region2region, attention_mask_region,
                                    attention_weights, pos=region_embed)
                
                out_grid = l_grid(out_grid, out_grid, out_grid, grid2grid, attention_mask_grid, attention_weights,
                                pos=grid_embed)

                out_all = torch.cat([out_region, out_grid], dim=1)



                out_grid = l_g2r(out_grid, out_all, out_all, grid2all, grid_aligns,
                                 attention_weights, pos_source=grid_embed, pos_cross=pos_cross)
                
                multi_layer_output.append(out_grid)
                
            features = 0
            lengths = [49]*bs
            lengths = torch.tensor(lengths).cuda()
            for layer in multi_layer_output:
                features_1, pool_weights_1 = self.gpool(layer, lengths)
                features = features + features_1
            
            # features, pool_weights_1 = self.gpool(multi_layer_output[-1], lengths)

            features = l2norm(features, dim=-1)
            
            return features
    

class TransformerEncoder(MultiLevelEncoder):
    def __init__(self, N, padding_idx, d_in=2048, **kwargs):
        super(TransformerEncoder, self).__init__(N, padding_idx, **kwargs)
        self.fc_region = nn.Linear(d_in, self.d_model)
        self.dropout_region = nn.Dropout(p=self.dropout)
        self.layer_norm_region = nn.LayerNorm(self.d_model)

        self.fc_grid = nn.Linear(d_in, self.d_model)
        self.dropout_grid = nn.Dropout(p=self.dropout)
        self.layer_nrom_grid = nn.LayerNorm(self.d_model)

        self.grid_embedding = pe.PositionEmbeddingSine(self.d_model/2, normalize=True)
        self.box_embedding = nn.Linear(4, self.d_model)

    def get_pos_embedding(self, boxes, grids,split=False):
        bs = boxes.shape[0]
        region_embed = self.box_embedding(boxes)
        grid_embed = self.grid_embedding(grids.view(bs, 7, 7, -1))

        return region_embed, grid_embed


    def forward(self, regions, grids, boxes, aligns, attention_weights=None, region_embed=None, grid_embed=None, rg_lengths = None):

        region_embed, grid_embed = self.get_pos_embedding(boxes, grids, split=True)
        
        mask_regions = (torch.sum(regions, dim=-1) == 0).unsqueeze(-1)
        mask_grids = (torch.sum(grids, dim=-1) == 0).unsqueeze(-1)
        # print('\ninput', input.view(-1)[0].item())
        out_region = F.relu(self.fc_region(regions))
        out_region = self.dropout_region(out_region)
        out_region = self.layer_norm_region(out_region)
        out_region = out_region.masked_fill(mask_regions, 0)

        out_grid = F.relu(self.fc_grid(grids))
        out_grid = self.dropout_grid(out_grid)
        out_grid = self.layer_nrom_grid(out_grid)
        out_grid = out_grid.masked_fill(mask_grids, 0)

        # print('out4',out[11])
        return super(TransformerEncoder, self).forward(out_region, out_grid, boxes, aligns,
                                                       attention_weights=attention_weights,
                                                       region_embed=region_embed, grid_embed=grid_embed,
                                                       rg_lengths = rg_lengths)
        
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
        # out = l2norm(out, dim=-1)
    
        return out
