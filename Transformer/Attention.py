import torch
import torch.nn as nn


def singlehead_attention(e_in: torch.Tensor, dim_att: int, p_dropout: float = 0.0) -> torch.Tensor:
    """
    注意力机制
    Args:
        e_in (torch.Tensor):输入 embedding，形状 (B, T, E)。
        dim_att (int): 注意力投影维度。
        dropout (float): softmax 后的 dropout概率

    Returns:
        torch.Tensor: 注意力输出，与输入的 batch/序列维匹配。
    """
    e_batch, e_len, e_dim = e_in.shape
    # (E,E_att)
    w_key = torch.randn(e_dim, dim_att)
    w_query = torch.randn(e_dim, dim_att)
    w_value = torch.randn(e_dim, dim_att)
    # (E_att,E)
    w_out = torch.randn(dim_att, e_dim)
    # (B,T,E_att)
    key = e_in @ w_key
    query = e_in @ w_query
    value = e_in @ w_value
    # relevant(i,j)表示的是第i个token对于第j个toke询问的相关性结果
    relevant = (query@key.transpose(1, 2))/torch.sqrt(dim_att)
    mask = torch.triu(torch.ones_like(relevant),
                      diagonal=1).bool()  # 上三角(不含对角)为 True
    # 要做掩码，因为是自回归输入。i必须大于等于j，否则相当于看见了未来
    relevant = relevant.masked_fill(mask, -float("inf"))
    # 按query即行做softmax
    p_att = relevant.softmax(dim=-1)
    # 以p的概率对这些权重进行dropout
    p_att = torch.dropout(p_att, p_dropout, train=True)
    # 这边得到的维度是(B,T,E_att) 因为之后还会乘一个wout回到(B,T,E)
    delta_e = p_att @ value @ w_out
    # 如果返回p_att @ value就是返回注意力了
    return delta_e


def singlehead_attention_official(e_in, dim_att, p_dropout=0.0):
    """
    官方单头注意力示例，内部仍用 MultiheadAttention，但 num_heads=1。
    e_in: (B, T, E) 输入
    dim_att: 特征维度 E
    p_dropout: dropout概率
    """
    mha = nn.MultiheadAttention(
        embed_dim=dim_att, num_heads=1, dropout=p_dropout, batch_first=True)
    t = e_in.size(1)
    causal_mask = torch.triu(torch.ones(
        t, t, device=e_in.device), diagonal=1).bool()
    attn_mask = causal_mask.masked_fill(causal_mask, -float("inf"))

    output, attn_weights = mha(e_in, e_in, e_in, attn_mask=attn_mask)
    return output, attn_weights


def multihead_attention_official(e_in, dim_att, num_heads=8, p_dropout=0.0):
    """
    示例：调用官方 MultiheadAttention（batch_first=True）。
    e_in: (B, T, E) 输入
    dim_att: 特征维度 E
    num_heads: 头数
    p_dropout: dropout概率
    """
    mha = nn.MultiheadAttention(
        embed_dim=dim_att, num_heads=num_heads, dropout=p_dropout, batch_first=True)
    # 因果掩码：上三角为 -inf
    t = e_in.size(1)
    causal_mask = torch.triu(torch.ones(
        t, t, device=e_in.device), diagonal=1).bool()
    attn_mask = causal_mask.masked_fill(causal_mask, -float("inf"))

    output, attn_weights = mha(e_in, e_in, e_in, attn_mask=attn_mask)
    return output, attn_weights


class Attention(nn.Module):
    def __init__(self, args, is_causal: bool = False):
        super().__init__()
        # w*为三个embedding matrix投影到注意力隐空间的投影矩阵
        self.wq = nn.Linear(args.dim, args.dim_att, bias=False)
        self.wk = nn.Linear(args.dim, args.dim_att, bias=False)
        self.wv = nn.Linear(args.dim, args.dim_att, bias=False)
        # 掩码不需要进行训练，但是需要记录为模型参数
        self.is_causal = is_causal
        if self.is_causal:
            mask = torch.full(((1, args.max_len, args.max_len)), -float("inf"))
            mask = torch.triu(mask, diagonal=1)
            self.register_buffer("mask", mask)
        self.att_dropout = nn.Dropout(args.att_dropout)  # 注意力权重dropout
        self.wo = nn.Linear(args.dim_att, args.dim, bias=False)
        self.out_dropout = nn.Dropout(args.out_dropout)  # 残差dropout

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        # 输入的是embedding
        # 自注意力就是(x,x,x);交叉注意力就是(x,y,y)
        # 经过w*矩阵的投射计算后才得到后续计算的q,k,v
        q = self.wq(Q)
        k = self.wk(K)
        v = self.wv(V)
        att = torch.matmul(q, k.transpose(1, 2))/torch.sqrt(q.size(-1))
        if self.is_causal:
            T = att.size(-1)
            att = att + self.mask[:, :T, :T]
        p_att = att.softmax(dim=-1)
        p_att = self.att_dropout(p_att)
        att = torch.matmul(p_att, v)
        output = self.wo(att)
        output = self.out_dropout(output)
        return output  # 返回残差输出，即会加回到Atteion(x,,)的x上


class MultiAttention(nn.Module):
    def __init__(self, args, is_causal: bool = False, num_head: int = 1):
        super().__init__()
        self.num_head = num_head
        assert args.dim_att % self.num_head == 0 
        self.dim_head = args.dim_att//self.num_head
        # w*为三个embedding matrix投影到注意力隐空间的投影矩阵
        self.wq = nn.Linear(args.dim, self.dim_head*self.num_head, bias=False)
        self.wk = nn.Linear(args.dim, self.dim_head*self.num_head, bias=False)
        self.wv = nn.Linear(args.dim, self.dim_head*self.num_head, bias=False)
        # 掩码不需要进行训练，但是需要记录为模型参数
        self.is_causal = is_causal
        if self.is_causal:
            # 多一个维度因为有多个注意力头
            mask = torch.full(
                ((1, 1, args.max_len, args.max_len)), -float("inf"))
            mask = torch.triu(mask, diagonal=1)
            self.register_buffer("mask", mask)
        self.att_dropout = nn.Dropout(args.att_dropout)  # 注意力权重dropout
        self.wo = nn.Linear(self.dim_head*self.num_head, args.dim, bias=False)
        self.out_dropout = nn.Dropout(args.out_dropout)  # 残差dropout

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        # 输入的是embedding
        # 自注意力就是(x,x,x);交叉注意力就是(x,y,y)
        # 经过w*矩阵的投射计算后才得到后续计算的q,k,v
        batch_size, len_embedding, _ = Q.shape
        q = self.wq(Q)
        k = self.wk(K)
        v = self.wv(V)
        # 拆分成多头，即注意力头也要占一个维度
        q = q.view(batch_size, len_embedding, self.num_head, self.dim_head).transpose(1,2)
        k = k.view(batch_size, len_embedding, self.num_head, self.dim_head).transpose(1,2)
        v = v.view(batch_size, len_embedding, self.num_head, self.dim_head).transpose(1,2)
        att = torch.matmul(q, k.transpose(2, 3))/torch.sqrt(self.dim_head)
        if self.is_causal:
            T = att.size(-1)
            att = att + self.mask[:, :, :T, :T]
        p_att = att.softmax(dim=-1)
        p_att = self.att_dropout(p_att)
        att = torch.matmul(p_att,v)
        att = att.transpose(1,2).contiguous().view(batch_size, len_embedding, self.num_head*self.dim_head)
        output = self.wo(att)
        output = self.out_dropout(output)
        return output  # 返回残差输出，即会加回到Atteion(x,,)的x上
