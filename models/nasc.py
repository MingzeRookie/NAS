from models.diffattention.multihead_flashdiff import MultiheadFlashDiff
import torch
import torch.nn as nn
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
from torch import einsum
import torch.nn.functional as F

def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def cache_fn(f):
    cache = dict()

    @wraps(f)
    def cached_fn(*args, _cache=True, key=None, **kwargs):
        if not _cache:
            return f(*args, **kwargs)
        nonlocal cache
        if key in cache:
            return cache[key]
        result = f(*args, **kwargs)
        cache[key] = result
        return result

    return cached_fn

class NASCSingle(nn.Module):
    def __init__(self,
        input_dim=384,
        num_class=3,
        embed_dim=384,
        depth=5, # current layer index
        diff_num_heads=8,
        diff_num_kv_heads=None,
        dropout=0.0
        ):
        super().__init__()
        
        self._cos_cached = None
        self._sin_cached = None
        self.input_fc = nn.Sequential(nn.Linear(input_dim,embed_dim),nn.GELU())
        self.cls_token = nn.Parameter(torch.randn(1,embed_dim))
        self.cls_token.requires_grad_(True)
        self.layers = nn.ModuleList()
        for i in range(depth):
            self.layers.append(nn.Sequential(nn.LayerNorm(embed_dim),
                                            MultiheadFlashDiff(embed_dim,i+1,diff_num_heads,diff_num_kv_heads),
                                            nn.Dropout(dropout)))
        self.output_fc = nn.Sequential(nn.LayerNorm(embed_dim),nn.Linear(embed_dim, num_class))
        
        self.cross_attn = PreNorm(input_dim,Attention(input_dim,heads=1,dropout=dropout,),context_dim=input_dim)
        self.cross_ff = PreNorm(input_dim, FeedForward(input_dim, dropout=dropout))
        self.to_logits = nn.Sequential(
            Rearrange("b n d -> b (n d)"), nn.Linear(input_dim, num_class)
        )
    # def _update_cos_sin_tables(self, coords):
    #     seq_len = coords.max().item()
    #     if seq_len > self.max_coords:
    #         self.max_coords = seq_len
    #         t = torch.arange(self.max_coords + 1).to(
    #             coords
    #         )  # Arange(max) returns [0,..., max-1], we need [0,..., max]
    #         if self.freqs_for == "pixel":
    #             t = t / self.max_freq
    #         freqs = torch.einsum("i,j->ij", t, self.inv_freq)
    #         emb = torch.repeat_interleave(freqs, 2, -1)
    #         self._cos_cached = emb.cos()  # (seq_len, n_dim/2)
    #         self._sin_cached = emb.sin()  # (seq_len, n_dim/2)

    #     return self._cos_cached, self._sin_cached
    
    def forward(self, feats: torch.Tensor):
        """
        Args:
            features (torch.Tensor): (..., T, dim)
            coords (torch.Tensor): (... T, 2)
        """       
        # self._cos_cached, self._sin_cached = self._update_cos_sin_tables(coords)
        bs,_,_ = feats.shape
        cls_token_in = repeat(self.cls_token,'n d -> b n d', b=bs)
        h = torch.concat([cls_token_in,feats],dim=1)
        h = self.input_fc(h)  # [b,n,d]
        for layer in self.layers:
            h_ = h
            h = layer[0](h)
            h = layer[1](h)
            h = layer[2](h)
            h = h + h_
        # h_ = h
        # h_ = self.layers(h_)
        # h = h + h_
        # cls_token_out = h[:, 0]
        
        # ----mean head----
        # instance_pred = self.output_fc(h)
        # bag_pred = reduce(instance_pred, 'b n d -> b d', reduction='mean')
        
        # ---cross attention head---
        x, attn_map = self.cross_attn(cls_token_in, context=h)
        x = x + cls_token_in
        x = self.cross_ff(x) + x
        bag_pred = self.to_logits(x)
        
        return bag_pred

class PreNorm(nn.Module):
    def __init__(self, dim, fn, context_dim=None):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(context_dim) if exists(context_dim) else None

    def forward(self, x, **kwargs):
        x = self.norm(x)

        if exists(self.norm_context):
            context = kwargs["context"]
            normed_context = self.norm_context(context)
            kwargs.update(context=normed_context)

        return self.fn(x, **kwargs)


class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)


class FeedForward(nn.Module):
    def __init__(self, dim, mult=1, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Linear(dim * mult, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)
    
class Attention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = query_dim
        self.scale = query_dim**-0.5
        self.heads = heads
        self.to_q = nn.Linear(query_dim, query_dim, bias=False)
        context_dim = default(context_dim, query_dim)
        self.to_kv = nn.Linear(context_dim, query_dim * 2, bias=False)
        self.to_v = nn.Linear(context_dim, query_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.Linear(query_dim, query_dim)

    def forward(self, x, context=None):
        h = self.heads
        # q = self.to_q(x)
        q = x
        context = default(context, x)
        k, v = self.to_kv(context).chunk(2, dim=-1)
        # k, v = context, self.to_v(context)
        # k, v = context, context
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> (b h) n d", h=h), (q, k, v))
        sim = einsum("b i d, b j d -> b i j", q, k) * self.scale
        attn = sim.softmax(dim=-1)
        # attn = attn.detach()
        attn = self.dropout(attn)
        out = einsum("b i j, b j d -> b i d", attn, v)
        out = rearrange(out, "(b h) n d -> b n (h d)", h=h)

        # return out, sim
        return self.to_out(out), sim
    

        
if __name__ == '__main__':
    model = NASCSingle(input_dim=1536,
        num_class=4,
        embed_dim=256).cuda().half()
    x = torch.randn(1,23000,1536).cuda().half()
    print(x)
    y = model(x)
    print(y)
        
        
        
        