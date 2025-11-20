import torch
import torch.nn as nn
from timm.models.vision_transformer import Attention, Mlp

class WindowAttBlock(nn.Module):
    """
    时空窗口注意力块 (Window Attention Block)
    结合了深度注意力 (Depth Attention) 和广度注意力 (Breadth Attention) 来捕获时空依赖。
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0):
        """
        初始化 WindowAttBlock
        :param hidden_size: 隐藏层维度
        :param num_heads: 注意力头数
        :param mlp_ratio: MLP 扩展比例
        """
        super().__init__()
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        
        # 广度注意力 (Breadth Attention) 相关层
        self.nnorm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.nattn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, attn_drop=0.1, proj_drop=0.1)
        self.nnorm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.nmlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=0.1)
        
        # 深度注意力 (Depth Attention) 相关层
        self.snorm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.sattn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, attn_drop=0.1, proj_drop=0.1)
        self.snorm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.smlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=0.1)

    def forward(self, x):
        """
        前向传播
        :param x: 输入张量 [B, T, N, D] (Batch, Time, Node, Dim)
        :return: 输出张量 [B, T, N, D]
        """
        # x: [B, T, N, D]
        B, T, N, D = x.shape
        
        # 1. 深度注意力 (Depth Attention) - 关注时间维度或特征维度内的交互
        # 将 B 和 T 合并，对每个时间步内的 N 个节点进行注意力计算 (或者理解为在 N 维度上进行注意力)
        # 这里 reshape 成 (B*T, N, D)，意味着 Attention 是在 N 维度上进行的 (Spatial Attention)
        qkv = self.snorm1(x.reshape(B*T,N,D))
        x = x + self.sattn(qkv).reshape(B,T,N,D)
        x = x + self.smlp(self.snorm2(x))
        
        # 2. 广度注意力 (Breadth Attention) - 关注空间维度或跨时间步的交互
        # 转置为 [B, N, T, D]，然后 reshape 为 [B*N, T, D]
        # 意味着 Attention 是在 T 维度上进行的 (Temporal Attention)
        qkv = self.nnorm1(x.transpose(1,2).reshape(B*N,T,D))
        x = x + self.nattn(qkv).reshape(B,N,T,D).transpose(1,2)
        x = x + self.nmlp(self.nnorm2(x))
        
        return x

class PatchSTG(nn.Module):
    """
    PatchSTG 模型
    用于大规模交通预测的时空图神经网络模型。
    """
    def __init__(self, output_len, node_num, layers, input_dims, node_dims, time_dims):
        """
        初始化 PatchSTG
        :param output_len: 输出时间步长度
        :param node_num: 节点数量
        :param layers: 编码器层数
        :param input_dims: 输入特征维度
        :param node_dims: 节点嵌入维度
        :param time_dims: 时间特征嵌入维度
        """
        super().__init__()
        self.node_num = node_num
        # 总特征维度 = 输入维度 + 6个时间特征 * 时间维度 + 节点维度
        dims = input_dims + 6 * time_dims + node_dims
        
        # 输入投影层: 将输入通道映射到 input_dims
        self.input_st_fc = nn.Conv2d(in_channels=1, out_channels=input_dims, kernel_size=(1, 1), stride=(1, 1), bias=True)
        
        # 节点嵌入 (可学习参数)
        self.node_emb = nn.Parameter(torch.empty(node_num, node_dims))
        nn.init.xavier_uniform_(self.node_emb)
        
        # 时间特征嵌入层
        self.weekday_emb = nn.Embedding(7, time_dims)   # 星期 (0-6)
        self.hour_emb = nn.Embedding(24, time_dims)     # 小时 (0-23)
        self.minute_emb = nn.Embedding(60, time_dims)   # 分钟 (0-59)
        self.daytype_emb = nn.Embedding(9, time_dims)   # 日期类型 (0-8)
        self.day_emb = nn.Embedding(31, time_dims)      # 日 (0-30)
        self.month_emb = nn.Embedding(12, time_dims)    # 月 (0-11)
        
        # 空间编码器 (由多个 WindowAttBlock 组成)
        self.spa_encoder = nn.ModuleList([
            WindowAttBlock(dims, 1, mlp_ratio=1) for _ in range(layers)
        ])
        
        # 回归预测层: 将高维特征映射回输出长度
        self.regression_conv = nn.Conv2d(in_channels=dims, out_channels=output_len, kernel_size=(1, 1), bias=True)

    def forward(self, x, te, mask=None):
        """
        前向传播
        :param x: 输入交通流量 [B, T, N, 1]
        :param te: 时间特征 [B, T, N, 6] (weekday, hour, minute, day_type, day, month)
        :param mask: 掩码 [B, N] (可选)
        :return: 预测结果 [B, T, N, output_len] -> 注意这里代码最后 permute 成了 [B, T, N, out_dim]
        """
        # x: [B,T,N,1] input traffic
        # te: [B,T,N,6] time features
        
        # 1. 嵌入层: 将输入和时间特征转换为高维表示
        embeded_x = self.embedding(x, te)  # [B,T,N,D]
        out = embeded_x
        
        # 2. 编码器层: 通过多层 WindowAttBlock 提取时空特征
        for block in self.spa_encoder:
            out = block(out)
            
        # 3. 预测层
        # out: [B, T, N, D] -> permute to [B, D, T, N] for Conv2d
        pred_y = self.regression_conv(out.permute(0,3,1,2))  # [B, out_dim, T, N]
        pred_y = pred_y.permute(0,2,3,1)  # [B, T, N, out_dim]
        
        # 4. 应用掩码 (如果有)
        if mask is not None:
            pred_y = pred_y * mask.unsqueeze(1).unsqueeze(-1)  # mask: [B,N] -> [B,1,N,1]
            
        return pred_y

    def embedding(self, x, te):
        """
        特征嵌入处理
        """
        b, t, n, _ = x.shape
        # te: [B,T,N,6] 0:weekday, 1:hour, 2:minute, 3:day_type, 4:day, 5:month
        weekday = te[..., 0].long()
        hour = te[..., 1].long()
        minute = te[..., 2].long()
        daytype = te[..., 3].long()
        day = te[..., 4].long()
        month = te[..., 5].long()
        
        # 处理输入流量数据: [B, T, N, 1] -> [B, T, N, input_dims]
        input_data = self.input_st_fc(x.permute(0,3,1,2)).permute(0,2,3,1)
        
        # 防护性截断/映射，避免索引越界
        weekday = weekday % 7
        hour = hour % 24
        minute = minute % 60
        daytype = daytype.clamp(min=0, max=8)
        day = day.clamp(min=0, max=30)
        month = month.clamp(min=0, max=11)

        # 拼接所有时间嵌入
        input_data = torch.cat([
            input_data,
            self.weekday_emb(weekday),
            self.hour_emb(hour),
            self.minute_emb(minute),
            self.daytype_emb(daytype),
            self.day_emb(day),
            self.month_emb(month)
        ], dim=-1)
        
        # 拼接节点嵌入 (广播到 B, T)
        node_emb = self.node_emb.unsqueeze(0).unsqueeze(0).expand(b, t, n, -1)
        input_data = torch.cat([input_data, node_emb], -1)
        
        return input_data
