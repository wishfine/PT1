import torch
import torch.nn as nn
from timm.models.vision_transformer import Attention, Mlp

class WindowAttBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0):
        super().__init__()
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.nnorm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.nattn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, attn_drop=0.1, proj_drop=0.1)
        self.nnorm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.nmlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=0.1)
        self.snorm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.sattn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, attn_drop=0.1, proj_drop=0.1)
        self.snorm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.smlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=0.1)

    def forward(self, x):
        # x: [B, T, N, D]
        B, T, N, D = x.shape
        # depth attention
        qkv = self.snorm1(x.reshape(B*T,N,D))
        x = x + self.sattn(qkv).reshape(B,T,N,D)
        x = x + self.smlp(self.snorm2(x))
        # breadth attention
        qkv = self.nnorm1(x.transpose(1,2).reshape(B*N,T,D))
        x = x + self.nattn(qkv).reshape(B,N,T,D).transpose(1,2)
        x = x + self.nmlp(self.nnorm2(x))
        return x

class PatchSTG(nn.Module):
    def __init__(self, output_len, node_num, layers, input_dims, node_dims, time_dims):
        super().__init__()
        self.node_num = node_num
        dims = input_dims + 6 * time_dims + node_dims
        self.input_st_fc = nn.Conv2d(in_channels=1, out_channels=input_dims, kernel_size=(1, 1), stride=(1, 1), bias=True)
        self.node_emb = nn.Parameter(torch.empty(node_num, node_dims))
        nn.init.xavier_uniform_(self.node_emb)
        self.weekday_emb = nn.Embedding(7, time_dims)
        self.hour_emb = nn.Embedding(24, time_dims)
        self.minute_emb = nn.Embedding(60, time_dims)
    # day_type: 0-8 （你确认有 9 类）
    self.daytype_emb = nn.Embedding(9, time_dims)
        self.day_emb = nn.Embedding(31, time_dims)
        self.month_emb = nn.Embedding(12, time_dims)
        self.spa_encoder = nn.ModuleList([
            WindowAttBlock(dims, 1, mlp_ratio=1) for _ in range(layers)
        ])
        self.regression_conv = nn.Conv2d(in_channels=dims, out_channels=output_len, kernel_size=(1, 1), bias=True)

    def forward(self, x, te, mask=None):
        # x: [B,T,N,1] input traffic
        # te: [B,T,N,6] time features: weekday, hour, minute, day_type, day, month
        embeded_x = self.embedding(x, te)  # [B,T,N,D]
        out = embeded_x
        for block in self.spa_encoder:
            out = block(out)
        pred_y = self.regression_conv(out.permute(0,3,1,2))  # [B, out_dim, T, N]
        pred_y = pred_y.permute(0,2,3,1)  # [B, T, N, out_dim]
        if mask is not None:
            pred_y = pred_y * mask.unsqueeze(1).unsqueeze(-1)  # mask: [B,N] -> [B,1,N,1]
        return pred_y

    def embedding(self, x, te):
        b, t, n, _ = x.shape
        # te: [B,T,N,6] 0:weekday,1:hour,2:minute,3:day_type,4:day,5:month
        weekday = te[..., 0].long()
        hour = te[..., 1].long()
        minute = te[..., 2].long()
        daytype = te[..., 3].long()
        day = te[..., 4].long()
        month = te[..., 5].long()
        # input traffic
        input_data = self.input_st_fc(x.permute(0,3,1,2)).permute(0,2,3,1)  # [B,T,N,input_dims]
        # 防护性截断/映射，避免越界（weekday 0-6, hour 0-23, minute 0-59, daytype 0-8, day 0-30, month 0-11）
        weekday = weekday % 7
        hour = hour % 24
        minute = minute % 60
        daytype = daytype.clamp(min=0, max=8)
        day = day.clamp(min=0, max=30)
        month = month.clamp(min=0, max=11)

        # 拼接所有时间 embedding
        input_data = torch.cat([
            input_data,
            self.weekday_emb(weekday),
            self.hour_emb(hour),
            self.minute_emb(minute),
            self.daytype_emb(daytype),
            self.day_emb(day),
            self.month_emb(month)
        ], dim=-1)
        # 拼接空间 embedding
        node_emb = self.node_emb.unsqueeze(0).unsqueeze(0).expand(b, t, n, -1)
        input_data = torch.cat([input_data, node_emb], -1)
        return input_data
