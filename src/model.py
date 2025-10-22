import torch
from torch import nn
from layers import *
import clip

class TemporalModule(nn.Module):
    '''
    Temporal Module 
    return the v_feature
    '''
    def __init__(self, cfg,d_model,n_heads, dropout_rate, gamma, bias, device,norm=None):
        super(TemporalModule, self).__init__()
        self.n_heads = n_heads
        self.self_attn = GATMultiHeadLayer(512,512//self.n_heads,dropout=dropout_rate,alpha=cfg.alpha,nheads=self.n_heads,concat=True)
        # self.self_attn2 = GATMultiHeadLayer(512,512//self.n_heads,dropout=dropout_rate,alpha=0.2,nheads=self.n_heads,concat=True)
        self.linear2 = nn.Linear(512,512)
        #self.linear1 = nn.Conv1d(d_model, 512, kernel_size=1) #512,the same as t_input
        self.dropout = nn.Dropout(dropout_rate)
        self.norm = nn.LayerNorm(d_model)
        #self.norm = RMSNorm(d_model)
        self.device = device
        self.loc_adj = DistanceAdj(gamma, bias,self.device)
        # self.alpha = nn.Parameter(torch.tensor(0.))
        self.mask_rate = cfg.mask_rate
    def forward(self, x, seq_len=None):
        adj = self.loc_adj(x.shape[0], x.shape[1])#disadj:two version
        #simadj = self.adj(x, seq_len) #simadj 
         # mask the adj
        feats = x
        #print(feats.shape)
        feat_magnitudes = torch.norm(feats, p=2, dim=2)
        #print(feat_magnitudes.shape)
        k = int(self.mask_rate*feats.shape[1])# 0.4
        topk = feat_magnitudes.topk(k, dim=-1).indices
        mask = torch.zeros_like(adj)
        for ix,i in enumerate(topk):
           mask[ix] =  mask[ix].index_fill(1,i,1)
           mask[ix] =  mask[ix].index_fill(0,i,1)    
        mask = mask.bool()
        adj = adj.masked_fill(~mask,0)

        tmp = self.self_attn(x, adj)
        # tmp_f = self.self_attn2(x,simadj)
        
      
        # tmp = self.alpha * tmp_f + (1 - self.alpha) * tmp_t
        
        if self.norm:
            tmp = torch.sqrt(F.relu(tmp)) - torch.sqrt(F.relu(-tmp))  # power norm
            tmp = F.normalize(tmp)  # l2 norm

        x = x + self.linear2(tmp)
        
      
        x = self.norm(x).permute(0, 2, 1)
        # x = self.dropout1(F.gelu(self.linear1(x)))            
        return x

class FeatureMapTemporalModule(nn.Module):
    '''
    Temporal Module for feature maps [B, T, 512, H, W]
    Applies temporal dependency while preserving spatial relationships.
    '''
    def __init__(self, cfg, d_model, n_heads, dropout_rate, gamma, bias, device, norm=None):
        super(FeatureMapTemporalModule, self).__init__()
        self.n_heads = n_heads
        self.self_attn = GATMultiHeadLayer(
            512, 512 // self.n_heads,
            dropout=dropout_rate,
            alpha=cfg.alpha,
            nheads=self.n_heads,
            concat=True
        )
        self.linear2 = nn.Linear(512, 512)
        self.dropout = nn.Dropout(dropout_rate)
        self.norm = nn.LayerNorm(d_model)
        self.device = device
        self.loc_adj = DistanceAdj(gamma, bias, self.device)
        self.mask_rate = cfg.mask_rate

    def forward(self, x, seq_len=None):
        # x: [B, T, 512, H, W]
        B, T, C, H, W = x.shape

        # Flatten spatial dims → [B, T, C, N], where N = H*W
        x = x.view(B, T, C, -1)  # [B, T, 512, N]
        N = x.shape[-1]

        # Rearrange → [B*N, T, C]
        x = x.permute(0, 3, 1, 2).contiguous().view(B * N, T, C)

        # Create adjacency for temporal graph
        adj = self.loc_adj(x.shape[0], x.shape[1])  # [B*N, T, T]

        # Magnitude for masking
        feat_magnitudes = torch.norm(x, p=2, dim=2)  # [B*N, T]
        k = int(self.mask_rate * T)
        topk = feat_magnitudes.topk(k, dim=-1).indices
        mask = torch.zeros_like(adj)
        for ix, i in enumerate(topk):
            mask[ix] = mask[ix].index_fill(1, i, 1)
            mask[ix] = mask[ix].index_fill(0, i, 1)
        mask = mask.bool()
        adj = adj.masked_fill(~mask, 0)

        # Temporal self-attention
        tmp = self.self_attn(x, adj)

        # Normalization
        if self.norm:
            tmp = torch.sqrt(F.relu(tmp)) - torch.sqrt(F.relu(-tmp))  # power norm
            tmp = F.normalize(tmp)

        x = x + self.linear2(tmp)
        x = self.norm(x).permute(0, 2, 1)  # [B*N, C, T]

        # Reshape back → [B, T, C, H, W]
        x = x.view(B, N, C, T).permute(0, 3, 2, 1).contiguous()  # [B, T, C, N]
        x = x.view(B, T, C, H, W)

        return x

class ModelDL(nn.Module):
    def __init__(self, cfg, device='cuda'):
        super(ModelDL, self).__init__()
        self.TM = TemporalModule(cfg,cfg.feat_dim,cfg.head_num, cfg.dropout_gat,cfg.gamma, cfg.bias,device)
        self.fmtm = FeatureMapTemporalModule(cfg,cfg.feat_dim,cfg.head_num, cfg.dropout_gat,cfg.gamma, cfg.bias, device)
        
        self.device = device
        self.pre_temporal_spatial_head = nn.Sequential(
            nn.Conv2d(768, 512, kernel_size = 1),
            nn.ReLU(),
            nn.Dropout2d(p = 0.4)
        )
        # self.post_temporal_spatial_head = nn.Sequential(
        #     nn.Conv2d(512, 1, kernel_size = 1),
        #     nn.Dropout2d(p=0.2)
        # )
        # nn.init.constant_(self.post_temporal_spatial_head[0].bias, -2.0)
        self.post_temporal_spatial_head = nn.Conv2d(512, 1, kernel_size = 1)
        nn.init.constant_(self.post_temporal_spatial_head.bias, 0.0)
        nn.init.xavier_uniform_(self.post_temporal_spatial_head.weight, gain = 0.1)
        
        # notice to frozen its parameters
        self.clipmodel, _ = clip.load('ViT-B/16', device=self.device, jit=False) 
        for paramclip in self.clipmodel.parameters():
            paramclip.requires_grad = False
        # detector to [b,seqlen,1]
        self.classifier = nn.Sequential(
                nn.Conv1d(512, cfg.cls_hidden, kernel_size=1, padding=0),
                nn.GELU(),
                nn.Dropout(p=0.3),
                nn.Conv1d(cfg.cls_hidden,1,kernel_size=1,padding=0)
        ) 

        self.mlp1 = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(512, 512 * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(512 * 4, 512))
        ]))   
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / cfg.temp))
       
        # self.apply(weight_init)
        self.has_feature_input = cfg.has_feature_input
        self.temporal = cfg.temporal
        self.norm = nn.LayerNorm(512)
        self.std_init = cfg.std_init


    def forward(self, x, seq_len=None):
        
        #video feature
        B, T, C, H, W = x.shape
        D = 512
        if self.has_feature_input:
            x_v = x
        else: 
            x_v, feature_map = self.clipmodel.encode_image(x.view(B * T, C, H, W))
            x_v = x_v.view(B, T, D)
            ano_map = self.pre_temporal_spatial_head(feature_map) #In: [B * T, 768, 14, 14]; Out: [B * T, 512, 14, 14]
            ano_map = ano_map.contiguous().view(B, T, D, 14, 14) #In: [B * T, 512, 14, 14]; Out: [B, T, 512, 14, 14]
            
        if self.temporal:
            x_v = self.TM(x_v, seq_len) #in:[b,t,512];out:[b,512,t]
            ano_map = self.fmtm(ano_map) #In: [B, T, 512, 14, 14]; Out: [B, T, 512, 14, 14]
            # print(f"Feature stats: mean={ano_map.mean():.3f}, std={ano_map.std():.3f}")
            # print(f"Feature range: [{ano_map.min():.3f}, {ano_map.max():.3f}]")
            ano_map = self.post_temporal_spatial_head(ano_map.contiguous().view(B * T, D, 14, 14)) #In: [B * T, 512, 14, 14]; Out: [B * T, 1, 14, 14]
            # print(f"Heatmap logits: mean={ano_map.mean():.3f}, std={ano_map.std():.3f}")
            # print(f"Heatmap range: [{ano_map.min():.3f}, {ano_map.max():.3f}]")
            # print(f"Positive ratio: {(ano_map > 0).float().mean():.3f}")
           
            ####ablation experiment：test transformer#####
            # x_v = self.temporalModelling(x_v).permute(0,2,1)
            ################################
        else:
            x_v = x_v.permute(0,2,1)        #In: [B, T, 512]; Out: [B, 512, T]
            ano_map = self.post_temporal_spatial_head(ano_map.contiguous().view(B * T, D, 14, 14)) #In: [B * T, 512, 14, 14]; Out: [B * T, 1, 14, 14]
        ano_map = ano_map.view(B, T, 1, 14, 14)
        logits = self.classifier(x_v)       #In: [B, 512, T]; Out: [B, 1, T]    --> Raw Logits
        logits = logits.permute(0, 2, 1)    #In: [B, 1, T]; Out: [B, T, 1]      --> Raw Logits
        
        #probs = torch.sigmoid(logits)      #In: [B, T, 1]; Out: [B, T, 1]      --> Sigmoid Probabilities

        return logits, ano_map