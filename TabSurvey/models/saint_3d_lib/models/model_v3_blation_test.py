import torch
import torch.nn.functional as F
from torch import nn, einsum
import numpy as np
from einops import rearrange


# helpers

def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def ff_encodings(x, B):
    x_proj = (2. * np.pi * x.unsqueeze(-1)) @ B.t()
    return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


# classes

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


# attention

class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim)
        )

    def forward(self, x, **kwargs):
        return self.net(x)


class Attention(nn.Module):
    def __init__(
            self,
            dim,
            heads=8,
            dim_head=16,
            dropout=0.
    ):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        _, a2, __ = x.shape
        scale = a2 ** -0.5
        h = self.heads
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v))
        sim = einsum('b h i d, b h j d -> b h i j', q, k) * scale
        # sim = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = sim.softmax(dim=-1)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)', h=h)
        return self.to_out(out)


class RowColTransformer(nn.Module):
    def __init__(self, num_tokens, dim, nfeats, depth, heads, dim_head, attn_dropout, ff_dropout, style='col',
                 device=None):
        super().__init__()
        self.device = device
        self.embeds = nn.Embedding(num_tokens, dim)
        self.layers = nn.ModuleList([])
        self.layers_mirror = nn.ModuleList([])
        self.mask_embed = nn.Embedding(nfeats, dim)
        max_length = 10
        self.pos_embedding = nn.Embedding(max_length, int(dim * nfeats / 5))
        self.scale = torch.sqrt(torch.FloatTensor([dim * nfeats / 5]).to(device))
        self.style = style
        for _ in range(depth):
            if self.style == 'colrow':
                self.layers.append(nn.ModuleList([
                    PreNorm(dim, Residual(Attention(dim, heads=1, dropout=attn_dropout))),
                    PreNorm(dim, Residual(FeedForward(dim, dropout=ff_dropout))),
                    PreNorm(dim * nfeats // 5,
                            Residual(Attention(dim * nfeats // 5, heads=nfeats // 5, dropout=attn_dropout))),
                    PreNorm(dim * nfeats // 5, Residual(FeedForward(dim * nfeats // 5, dropout=ff_dropout))),

                    PreNorm(int(dim * nfeats / 5),
                            Residual(Attention(int(dim * nfeats / 5), heads=int(nfeats / 5), dropout=attn_dropout))),
                    PreNorm(int(dim * nfeats / 5), Residual(FeedForward(int(dim * nfeats / 5), dropout=ff_dropout))),
                ]))
                self.layers_mirror.append(nn.ModuleList([
                    PreNorm(dim, Residual(Attention(dim, heads=1, dropout=attn_dropout))),
                    PreNorm(dim, Residual(FeedForward(dim, dropout=ff_dropout))),
                    PreNorm(dim * nfeats // 5,
                            Residual(Attention(dim * nfeats // 5, heads=nfeats // 5, dropout=attn_dropout))),
                    PreNorm(dim * nfeats // 5, Residual(FeedForward(dim * nfeats // 5, dropout=ff_dropout)))

                ]))
                self.freeze_weights(self.layers_mirror)
            else:
                self.layers.append(nn.ModuleList([
                    PreNorm(dim * nfeats,
                            Residual(Attention(dim * nfeats, heads=heads, dim_head=64, dropout=attn_dropout))),
                    PreNorm(dim * nfeats, Residual(FeedForward(dim * nfeats, dropout=ff_dropout))),
                ]))

    def freeze_weights(self, module_list):
        for module in module_list:
            if isinstance(module, nn.ModuleList):
                self.freeze_weights(module)
            else:
                for param in module.parameters():
                    param.requise_grad = False

    def copy_datas(self, module_list_source, module_list_target):
        if isinstance(module_list_source, nn.ModuleList):
            for module_source, module_target in zip(module_list_source, module_list_target):
                if isinstance(module_source, nn.ModuleList):
                    self.copy_datas(module_source, module_target)
                else:
                    for param_source, param_target in zip(module_source.parameters(), module_target.parameters()):
                        param_target.data.copy_(param_source.data)
        else:
            for param_source, param_target in zip(module_list_source.parameters(), module_list_target.parameters()):
                param_target.data.copy_(param_source.data)
    def forward(self, x, x_cont=None, mask=None,blation_test_id=11):
        each_day_feature_num = 39
        each_day_cat_feature_num = 3
        if x_cont is not None:
            x_new = []
            for i in range(5):
                x_new.append(torch.cat((x[:, i * 3:(i + 1) * each_day_cat_feature_num, :], x_cont[:, i * (
                        each_day_feature_num - each_day_cat_feature_num):(i + 1) * (
                        each_day_feature_num - each_day_cat_feature_num), :]), dim=1))
            x = torch.cat(x_new, dim=1)
        else:
            print(f'x_cont is {x_cont}')
        batch, n, _ = x.shape
        if self.style == 'colrow':
            for (attn1, ff1, attn2, ff2, attn3, ff3), (attn1_mirror, ff1_mirror, attn2_mirror, ff2_mirror) in zip(
                    self.layers, self.layers_mirror):
                self.copy_datas(attn1, attn1_mirror)
                self.copy_datas(ff1, ff1_mirror)
                self.copy_datas(attn2, attn2_mirror)
                self.copy_datas(ff2, ff2_mirror)
                x = self.blation_test(blation_test_id,x, attn1,attn1_mirror, ff1,ff1_mirror, attn2,attn2_mirror, ff2,ff2_mirror, attn3, ff3, batch, n)
        else:
            for attn1, ff1 in self.layers:
                x = rearrange(x, 'b n d -> 1 b (n d)')
                x = attn1(x)
                x = ff1(x)
                x = rearrange(x, '1 b (n d) -> b n d', n=n)

        x = x[:, 0:38, :]
        # x = rearrange(x, 'b n d -> b (n d)')
        return x

    def blation_test(self,blation_test_id,x, attn1,attn1_mirror, ff1,ff1_mirror, attn2,attn2_mirror, ff2,ff2_mirror, attn3, ff3, batch, n):
        """
        列维度0
        行维度1
        时间维度2
        列+行维度3
        行+列维度6
        列+时间维度4
        时间+列维度7
        行+时间维度5
        时间+行维度8
        行+列+时间维度9
        行+时间+列维度10
        列+行+时间维度11
        列+时间+行维度12
        时间+行+列维度13
        时间+列+行维度14

        串行网络 结果相加
        1	1	1           15
        1	0.1	0.1         16
        0.1	1	0.1         17
        0.1	0.1	1           18

        并行网络 结果相加
        1	1	1           19
        1	0.1	0.1         20
        0.1	1	0.1         21
        0.1	0.1	1           22


        """
        if blation_test_id==6:#r+c
            x1 = self.r_attention(x, attn1,attn1_mirror, ff1,ff1_mirror, attn2,attn2_mirror, ff2,ff2_mirror, attn3, ff3, batch, n)
            x2 = self.c_attention(x1, attn1,attn1_mirror, ff1,ff1_mirror, attn2,attn2_mirror, ff2,ff2_mirror, attn3, ff3, batch, n)
            return x2
        if blation_test_id==7:#t+c
            x1 = self.t_attention(x, attn1,attn1_mirror, ff1,ff1_mirror, attn2,attn2_mirror, ff2,ff2_mirror, attn3, ff3, batch, n)
            x2 = self.c_attention(x, attn1,attn1_mirror, ff1,ff1_mirror, attn2,attn2_mirror, ff2,ff2_mirror, attn3, ff3, batch, n)
            return x2
        if blation_test_id == 8:#t+r
            x1 = self.t_attention(x, attn1,attn1_mirror, ff1,ff1_mirror, attn2,attn2_mirror, ff2,ff2_mirror, attn3, ff3, batch, n)
            x2 = self.r_attention(x1, attn1,attn1_mirror, ff1,ff1_mirror, attn2,attn2_mirror, ff2,ff2_mirror, attn3, ff3, batch, n)
            return x2
        if blation_test_id == 9:  # r+c+t
            x1 = self.r_attention(x, attn1,attn1_mirror, ff1,ff1_mirror, attn2,attn2_mirror, ff2,ff2_mirror, attn3, ff3, batch, n)
            x2 = self.c_attention(x1, attn1,attn1_mirror, ff1,ff1_mirror, attn2,attn2_mirror, ff2,ff2_mirror, attn3, ff3, batch, n)
            x3 = self.t_attention(x2, attn1,attn1_mirror, ff1,ff1_mirror, attn2,attn2_mirror, ff2,ff2_mirror, attn3, ff3, batch, n)
            return x3
        if blation_test_id == 10:  # r+t+c
            x1 = self.r_attention(x, attn1,attn1_mirror, ff1,ff1_mirror, attn2,attn2_mirror, ff2,ff2_mirror, attn3, ff3, batch, n)
            x2 = self.t_attention(x1, attn1,attn1_mirror, ff1,ff1_mirror, attn2,attn2_mirror, ff2,ff2_mirror, attn3, ff3, batch, n)
            x3 = self.c_attention(x2, attn1,attn1_mirror, ff1,ff1_mirror, attn2,attn2_mirror, ff2,ff2_mirror, attn3, ff3, batch, n)
            return x3
        if blation_test_id == 11:  # c+r+t
            x1 = self.c_attention(x, attn1,attn1_mirror, ff1,ff1_mirror, attn2,attn2_mirror, ff2,ff2_mirror, attn3, ff3, batch, n)
            x2 = self.r_attention(x1, attn1,attn1_mirror, ff1,ff1_mirror, attn2,attn2_mirror, ff2,ff2_mirror, attn3, ff3, batch, n)
            x3 = self.t_attention(x2, attn1,attn1_mirror, ff1,ff1_mirror, attn2,attn2_mirror, ff2,ff2_mirror, attn3, ff3, batch, n)
            return x3
        if blation_test_id == 12:  # c+t+r
            x1 = self.c_attention(x, attn1,attn1_mirror, ff1,ff1_mirror, attn2,attn2_mirror, ff2,ff2_mirror, attn3, ff3, batch, n)
            x2 = self.t_attention(x1, attn1,attn1_mirror, ff1,ff1_mirror, attn2,attn2_mirror, ff2,ff2_mirror, attn3, ff3, batch, n)
            x3 = self.r_attention(x2, attn1,attn1_mirror, ff1,ff1_mirror, attn2,attn2_mirror, ff2,ff2_mirror, attn3, ff3, batch, n)
            return x3
        if blation_test_id == 13:  # t+r+c
            x1 = self.t_attention(x, attn1,attn1_mirror, ff1,ff1_mirror, attn2,attn2_mirror, ff2,ff2_mirror, attn3, ff3, batch, n)
            x2 = self.r_attention(x1, attn1,attn1_mirror, ff1,ff1_mirror, attn2,attn2_mirror, ff2,ff2_mirror, attn3, ff3, batch, n)
            x3 = self.c_attention(x2, attn1,attn1_mirror, ff1,ff1_mirror, attn2,attn2_mirror, ff2,ff2_mirror, attn3, ff3, batch, n)
            return x3
        if blation_test_id == 14:  # t+c+r
            x1 = self.t_attention(x, attn1,attn1_mirror, ff1,ff1_mirror, attn2,attn2_mirror, ff2,ff2_mirror, attn3, ff3, batch, n)
            x2 = self.c_attention(x, attn1,attn1_mirror, ff1,ff1_mirror, attn2,attn2_mirror, ff2,ff2_mirror, attn3, ff3, batch, n)
            x3 = self.r_attention(x, attn1,attn1_mirror, ff1,ff1_mirror, attn2,attn2_mirror, ff2,ff2_mirror, attn3, ff3, batch, n)
            return x3
        if blation_test_id in [19,20,21,22]:  # t+c+r
            x1 = self.t_attention(x, attn1,attn1_mirror, ff1,ff1_mirror, attn2,attn2_mirror, ff2,ff2_mirror, attn3, ff3, batch, n)
            x2 = self.c_attention(x, attn1,attn1_mirror, ff1,ff1_mirror, attn2,attn2_mirror, ff2,ff2_mirror, attn3, ff3, batch, n)
            x3 = self.r_attention(x, attn1,attn1_mirror, ff1,ff1_mirror, attn2,attn2_mirror, ff2,ff2_mirror, attn3, ff3, batch, n)
            x1_param = 0
            x2_param = 0
            x3_param = 1.0
            if blation_test_id == 19:
                x1_param = 1.0
                x2_param = 1.0
                x3_param = 1.0
            if blation_test_id == 20:
                x1_param = 1.0
                x2_param = 0.1
                x3_param = 0.1
            if blation_test_id == 21:
                x1_param = 0.1
                x2_param = 1.0
                x3_param = 0.1
            if blation_test_id == 22:
                x1_param = 0.1
                x2_param = 0.1
                x3_param = 1.0
            x = x1_param * x1 + x2_param * x2 + x3_param * x3
            return x
        x1 = self.c_attention(x,attn1,attn1_mirror, ff1,ff1_mirror, attn2,attn2_mirror, ff2,ff2_mirror, attn3, ff3, batch, n)
        if blation_test_id==0: # c
            return x1

        if blation_test_id==1:# r
            x1 = x
        if blation_test_id == 5:  # r+t
            x1 = x
        x2 = self.r_attention(x1,attn1,attn1_mirror, ff1,ff1_mirror, attn2,attn2_mirror, ff2,ff2_mirror, attn3, ff3, batch, n)
        if blation_test_id == 1:# r
            return x2

        if  blation_test_id == 3: # c+r
            return x2

        if  blation_test_id == 4: # c+t
            x2 = x1

        if blation_test_id == 2:# t
            x2 = x
        x3 = self.t_attention(x2,attn1,attn1_mirror, ff1,ff1_mirror, attn2,attn2_mirror, ff2,ff2_mirror, attn3, ff3, batch, n)
        if blation_test_id == 2:# t
            return x3
        if blation_test_id == 4:  # c+t
            return x3
        if blation_test_id == 5:  # r+t
            return x3

        x1_param=0
        x2_param=0
        x3_param=1.0
        if blation_test_id == 15:
            x1_param = 1.0
            x2_param = 1.0
            x3_param = 1.0
        if blation_test_id == 16:
            x1_param = 1.0
            x2_param = 0.1
            x3_param = 0.1
        if blation_test_id == 17:
            x1_param = 0.1
            x2_param = 1.0
            x3_param = 0.1
        if blation_test_id == 18:
            x1_param = 0.1
            x2_param = 0.1
            x3_param = 1.0
        x = x1_param * x1 + x2_param * x2 + x3_param * x3
        return x


    def c_attention(self, x,attn1,attn1_mirror, ff1,ff1_mirror, attn2,attn2_mirror, ff2,ff2_mirror, attn3, ff3, batch, n):
        each_day_feature_num = 39
        each_day_cat_feature_num = 3
        x1 = []
        i = 0
        _x1 = attn1(x[:, i * each_day_feature_num:(i + 1) * each_day_feature_num, :])
        _x1 = ff1(_x1)
        x1.append(_x1)
        for j in range(1, 5):
            _x1 = attn1_mirror(x[:, j * each_day_feature_num:(j + 1) * each_day_feature_num, :])
            _x1 = ff1_mirror(_x1)
            x1.append(_x1)
        x1 = torch.cat(x1, dim=1)
        return x1

    def r_attention(self, x1,attn1,attn1_mirror, ff1,ff1_mirror, attn2,attn2_mirror, ff2,ff2_mirror, attn3, ff3, batch, n):
        each_day_feature_num = 39
        each_day_cat_feature_num = 3
        i = 0
        x2 = []
        x2_ = rearrange(x1, 'b n d -> 1 b (n d)')
        _x2 = attn2(x2_[:, :, i * each_day_feature_num * 8:(i + 1) * each_day_feature_num * 8])
        _x2 = ff2(_x2)
        x2.append(_x2)
        for j in range(1, 5):
            _x2 = attn2_mirror(x2_[:, :, j * each_day_feature_num * 8:(j + 1) * each_day_feature_num * 8])
            _x2 = ff2_mirror(_x2)
            x2.append(_x2)
        x2 = torch.cat(x2, dim=2)
        x2 = rearrange(x2, '1 b (n d) -> b n d', n=n)
        return x2

    def t_attention(self, x2,attn1,attn1_mirror, ff1,ff1_mirror, attn2,attn2_mirror, ff2,ff2_mirror, attn3, ff3, batch, n):
        each_day_feature_num = 39
        each_day_cat_feature_num = 3
        x3 = rearrange(x2, 'b (d_1 d_2) d -> b d_1 d_2 d', d_1=5)
        x3 = rearrange(x3, 'b d_1 d_2 d -> b d_1 (d_2 d)')
        pos = torch.arange(5, 0, -1).unsqueeze(0).repeat(batch, 1).to(self.device)
        x3 = x3 * self.scale + self.pos_embedding(pos)
        x3 = attn3(x3)
        x3 = ff3(x3)
        x3 = rearrange(x3, 'b d_1 (d_2 d) -> b d_1 d_2 d', d=8)
        x3 = rearrange(x3, 'b d_1 d_2 d -> b (d_1 d_2) d')
        return x3

    def forward1(self, x, x_cont=None, mask=None):
        if x_cont is not None:
            x = torch.cat((x, x_cont), dim=1)
        batch, n, _ = x.shape
        if self.style == 'colrow':
            for attn1, ff1, attn2, ff2, attn3, ff3 in self.layers:
                x1 = attn1(x)
                x1 = ff1(x1)

                x2 = rearrange(x1, 'b n d -> 1 b (n d)')
                x2 = attn2(x2)
                x2 = ff2(x2)
                x2 = rearrange(x2, '1 b (n d) -> b n d', n=n)

                # x3 = rearrange(x2, 'b n d -> b n d_1 d_2', d_2=5)
                # x3 = rearrange(x3, 'b n d_1 d_2 -> b d_2 (n d_1)')
                # x3 = attn3(x3)
                # x3 = ff3(x3)
                # x3 = rearrange(x3, 'b d_2 (n d_1) -> b n d_1 d_2', d_2=5)
                # x3 = rearrange(x3, 'b n d_1 d_2 -> b n d')
                x3 = rearrange(x2, 'b (d_1 d_2) d -> b d_1 d_2 d', d_1=5)
                x3 = rearrange(x3, 'b d_1 d_2 d -> b d_1 (d_2 d)')
                pos = torch.arange(0, 5).unsqueeze(0).repeat(batch, 1).to(self.device)
                x3 = x3 * self.scale + self.pos_embedding(pos)
                x3 = attn3(x3)
                x3 = ff3(x3)
                x3 = rearrange(x3, 'b d_1 (d_2 d) -> b d_1 d_2 d', d=8)
                x3 = rearrange(x3, 'b d_1 d_2 d -> b (d_1 d_2) d')

                x = 0.6 * x1 + 0.3 * x2 + 0.1 * x3

        else:
            for attn1, ff1 in self.layers:
                x = rearrange(x, 'b n d -> 1 b (n d)')
                x = attn1(x)
                x = ff1(x)
                x = rearrange(x, '1 b (n d) -> b n d', n=n)
        return x


# transformer
class Transformer(nn.Module):
    def __init__(self, num_tokens, dim, depth, heads, dim_head, attn_dropout, ff_dropout):
        super().__init__()
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Residual(Attention(dim, heads=heads, dim_head=dim_head, dropout=attn_dropout))),
                PreNorm(dim, Residual(FeedForward(dim, dropout=ff_dropout))),
            ]))

    def forward(self, x, x_cont=None):
        if x_cont is not None:
            x = torch.cat((x, x_cont), dim=1)
        for attn, ff in self.layers:
            x = attn(x)
            x = ff(x)
        return x


# mlp
class MLP(nn.Module):
    def __init__(self, dims, act=None):
        super().__init__()
        dims_pairs = list(zip(dims[:-1], dims[1:]))
        layers = []
        for ind, (dim_in, dim_out) in enumerate(dims_pairs):
            is_last = ind >= (len(dims) - 1)
            linear = nn.Linear(dim_in, dim_out)
            layers.append(linear)

            if is_last:
                continue
            if act is not None:
                layers.append(act)

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class simple_MLP(nn.Module):
    def __init__(self, dims):
        super(simple_MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(dims[0], dims[1]),
            nn.ReLU(),
            nn.Linear(dims[1], dims[2])
        )

    def forward(self, x):
        if len(x.shape) == 1:
            x = x.reshape(x.size(0), -1)
        x = self.layers(x)
        return x


# main class

class TabAttention(nn.Module):
    def __init__(
            self,
            *,
            categories,
            num_continuous,
            dim,
            depth,
            heads,
            dim_head=16,
            dim_out=1,
            mlp_hidden_mults=(4, 2),
            mlp_act=None,
            num_special_tokens=1,
            continuous_mean_std=None,
            attn_dropout=0.,
            ff_dropout=0.,
            lastmlp_dropout=0.,
            cont_embeddings='MLP',
            scalingfactor=10,
            attentiontype='col'
    ):
        super().__init__()
        assert all(map(lambda n: n > 0, categories)), 'number of each category must be positive'

        # categories related calculations
        self.num_categories = len(categories)
        self.num_unique_categories = sum(categories)

        # create category embeddings table

        self.num_special_tokens = num_special_tokens
        self.total_tokens = self.num_unique_categories + num_special_tokens

        # for automatically offsetting unique category ids to the correct position in the categories embedding table
        categories_offset = F.pad(torch.tensor(list(categories)), (1, 0), value=num_special_tokens)
        categories_offset = categories_offset.cumsum(dim=-1)[:-1]

        self.register_buffer('categories_offset', categories_offset)

        self.norm = nn.LayerNorm(num_continuous)
        self.num_continuous = num_continuous
        self.dim = dim
        self.cont_embeddings = cont_embeddings
        self.attentiontype = attentiontype

        if self.cont_embeddings == 'MLP':
            self.simple_MLP = nn.ModuleList([simple_MLP([1, 100, self.dim]) for _ in range(self.num_continuous)])
            input_size = (dim * self.num_categories) + (dim * num_continuous)
            nfeats = self.num_categories + num_continuous
        else:
            print('Continous features are not passed through attention')
            input_size = (dim * self.num_categories) + num_continuous
            nfeats = self.num_categories

            # transformer
        if attentiontype == 'col':
            self.transformer = Transformer(
                num_tokens=self.total_tokens,
                dim=dim,
                depth=depth,
                heads=heads,
                dim_head=dim_head,
                attn_dropout=attn_dropout,
                ff_dropout=ff_dropout
            )
        elif attentiontype in ['row', 'colrow']:
            self.transformer = RowColTransformer(
                num_tokens=self.total_tokens,
                dim=dim,
                nfeats=nfeats,
                depth=depth,
                heads=heads,
                dim_head=dim_head,
                attn_dropout=attn_dropout,
                ff_dropout=ff_dropout,
                style=attentiontype
            )

        l = input_size // 8
        hidden_dimensions = list(map(lambda t: l * t, mlp_hidden_mults))
        all_dimensions = [input_size, *hidden_dimensions, dim_out]

        self.mlp = MLP(all_dimensions, act=mlp_act)
        self.embeds = nn.Embedding(self.total_tokens, self.dim)  # .to(device)

        cat_mask_offset = F.pad(torch.Tensor(self.num_categories).fill_(2).type(torch.int8), (1, 0), value=0)
        cat_mask_offset = cat_mask_offset.cumsum(dim=-1)[:-1]

        con_mask_offset = F.pad(torch.Tensor(self.num_continuous).fill_(2).type(torch.int8), (1, 0), value=0)
        con_mask_offset = con_mask_offset.cumsum(dim=-1)[:-1]

        self.register_buffer('cat_mask_offset', cat_mask_offset)
        self.register_buffer('con_mask_offset', con_mask_offset)

        self.mask_embeds_cat = nn.Embedding(self.num_categories * 2, self.dim)
        self.mask_embeds_cont = nn.Embedding(self.num_continuous * 2, self.dim)

    def forward(self, x_categ, x_cont, x_categ_enc, x_cont_enc):
        device = x_categ.device
        if self.attentiontype == 'justmlp':
            if x_categ.shape[-1] > 0:
                flat_categ = x_categ.flatten(1).to(device)
                x = torch.cat((flat_categ, x_cont.flatten(1).to(device)), dim=-1)
            else:
                x = x_cont.clone()
        else:
            if self.cont_embeddings == 'MLP':
                x = self.transformer(x_categ_enc, x_cont_enc.to(device))
            else:
                if x_categ.shape[-1] <= 0:
                    x = x_cont.clone()
                else:
                    flat_categ = self.transformer(x_categ_enc).flatten(1)
                    x = torch.cat((flat_categ, x_cont), dim=-1)
        flat_x = x.flatten(1)
        return self.mlp(flat_x)
