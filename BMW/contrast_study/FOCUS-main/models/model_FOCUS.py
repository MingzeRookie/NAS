# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import logging
from os.path import join as pjoin
from .model_utils import *
logger = logging.getLogger(__name__)
import torch
import torch.nn as nn
from torch.nn import functional as F
import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
_tokenizer = _Tokenizer()

from conch.open_clip_custom import create_model_from_pretrained, get_tokenizer, tokenize

class TextEncoder(nn.Module):
    def __init__(self, conch_model):
        super().__init__()
        self.transformer = conch_model.text.transformer
        self.positional_embedding = conch_model.text.positional_embedding
        self.ln_final = conch_model.text.ln_final
        self.text_projection = conch_model.text.text_projection
        # Get dtype from one of the model's parameters
        self.dtype = next(conch_model.parameters()).dtype

    def forward(self, prompts, tokenized_prompts):
        # Rest of the code remains the same
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x).type(self.dtype)
        x = x[:, 0] @ self.text_projection
        return x

class PromptLearner(nn.Module):
    def __init__(self, classnames, conch_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = 16
        ctx_init = ""
        dtype = next(conch_model.parameters()).dtype
        ctx_dim = conch_model.text.ln_final.weight.shape[0]
        
        # Get the tokenizer
        self.tokenizer = get_tokenizer()

        if ctx_init:
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            # Use the correct tokenize function with both arguments
            prompt = tokenize(self.tokenizer, [ctx_init])
            with torch.no_grad():
                embedding = conch_model.text.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            if False:
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        self.ctx = nn.Parameter(ctx_vectors)

        # Process class names
        classnames = [name.replace("_", " ") for name in classnames]
        prompts = [name for name in classnames]
        
        # Use the correct tokenize function with both arguments
        tokenized_prompts = tokenize(self.tokenizer, prompts)
        
        with torch.no_grad():
            embedding = conch_model.text.token_embedding(tokenized_prompts).type(dtype)

        self.register_buffer("token_prefix", embedding[:, :1, :])
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts
        # Use the tokenizer's encode method for getting lengths
        self.name_lens = [len(self.tokenizer.encode(name, 
                                                   max_length=127,
                                                   truncation=True)) 
                         for name in classnames]
        self.class_token_position = "end"

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,
                    ctx,
                    suffix,
                ],
                dim=1,
            )
        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i_half1 = ctx[i : i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i : i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,
                        ctx_i_half1,
                        class_i,
                        ctx_i_half2,
                        suffix_i,
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)
        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i = ctx[i : i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,
                        class_i,
                        ctx_i,
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)
        else:
            raise ValueError
        return prompts

def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.
    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)
    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor

def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)

class FOCUS(nn.Module):
    def __init__(self, config, num_classes=3):
        super(FOCUS, self).__init__()
        self.loss_ce = nn.CrossEntropyLoss()
        self.num_classes = num_classes
        self.window_size = config.window_size
        self.sim_threshold = config.sim_threshold
        
        # Feature dimensions
        self.L = config.input_size
        # self.D = config.input_size
        self.D = 512
        self.L_max = config.max_context_length
        
        conch_model_cfg = 'conch_ViT-B-16'
        conch_checkpoint_path = '/remote-home/share/zhangyuanyuan/CONCH/ckpt/pytorch_model2.bin'
        conch_model, preprocess = create_model_from_pretrained(conch_model_cfg, conch_checkpoint_path)
        _ = conch_model.eval()
        
        self.feature_dim = conch_model.text.text_projection.shape[1] # 512
        
        self.prompt_learner = PromptLearner(config.text_prompt, conch_model.float())
        self.text_encoder = TextEncoder(conch_model.float())
        
        # Feature encoder with projection
        self.feature_encoder = nn.Sequential(
            nn.Linear(self.L, self.D), # 512 -> 512
            nn.LayerNorm(self.D),
            nn.ReLU(),
            nn.Dropout(0.25)
        )
        
        # Cross attention components
        num_heads = 8
        self.head_dim = self.feature_dim // num_heads
        self.q_proj = nn.Linear(self.feature_dim, self.feature_dim)
        self.k_proj = nn.Linear(self.feature_dim, self.feature_dim)
        self.v_proj = nn.Linear(self.feature_dim, self.feature_dim)
        self.o_proj = nn.Linear(self.feature_dim, self.feature_dim)
        self.num_heads = num_heads
        
        # Classifier
        self.classifier = nn.Linear(self.feature_dim, num_classes)
        
    def cross_attention(self, queries, keys, values, attention_mask=None):
        bsz, q_len, _ = queries.size()
        _, kv_len, _ = keys.size()
        
        # Linear projections
        query_states = self.q_proj(queries)
        key_states = self.k_proj(keys)
        value_states = self.v_proj(values)
        
        # Reshape for multi-head attention
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, kv_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, kv_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        attn_output = F.scaled_dot_product_attention(
            query_states, key_states, value_states,
            attn_mask=attention_mask
        )
        
        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.feature_dim)
        attn_output = self.o_proj(attn_output)
        
        return attn_output

    def compute_patch_similarity(self, x, window_size):
        """Compute similarity between patches within sliding windows"""
        N, D = x.shape
        x_norm = F.normalize(x, p=2, dim=-1)
        
        similarities = []
        selected_indices = []
        
        for i in range(0, N, window_size):
            window = x_norm[i:i+window_size]
            
            # Skip if window is too small
            if len(window) < 2:
                selected_indices.append(torch.arange(i, min(i+window_size, N), device=x.device))
                continue
                
            # Local similarity computation
            window_sim = torch.mm(window, window.t())
            
            # Adaptive thresholding
            # Only compute std if we have enough samples
            if window_sim.numel() > 1:
                threshold = window_sim.mean() + window_sim.std(unbiased=False)  # use biased std
            else:
                threshold = window_sim.mean()  # fallback to just mean if not enough samples
            
            # Select non-redundant patches
            redundant = window_sim.mean(1) > threshold
            keep_indices = torch.where(~redundant)[0] + i
            
            # If all patches are marked as redundant, keep at least one
            if len(keep_indices) == 0:
                keep_indices = torch.tensor([i], device=x.device)
                
            selected_indices.append(keep_indices)
            similarities.append(window_sim)
        
        # Handle case where no indices were selected
        if not selected_indices:
            return [], torch.arange(N, device=x.device)
            
        return similarities, torch.cat(selected_indices)

    def adaptive_token_selection(self, features, text_features):
        """Select tokens based on text relevance and local structure"""
        N, D = features.shape
        
        # Compute similarities and get initial selection
        similarities, indices = self.compute_patch_similarity(features, self.window_size)
        
        # Project features to match text_features dimension if needed
        if features.shape[-1] != text_features.shape[-1]:
            projection = nn.Linear(features.shape[-1], text_features.shape[-1], device=features.device)
            features_projected = projection(features)
        else:
            features_projected = features
        
        # Text-guided importance scoring
        text_relevance = torch.matmul(features_projected, text_features.T).mean(-1)
        
        # Create importance mask
        importance_mask = torch.zeros(N, device=features.device)
        importance_mask[indices] = text_relevance[indices]
        
        # Select top tokens
        num_tokens = min(self.L_max, N)
        _, selected_indices = torch.topk(importance_mask, num_tokens)
        selected_indices, _ = torch.sort(selected_indices)  # maintain sequence order
        
        selected_features = features[selected_indices]
        
        return selected_features, selected_indices

    def spatial_token_compression(self, features, text_features):
        """Compress tokens while preserving important information"""
        N, D = features.shape
        
        # Process in chunks like LongVU
        chunk_size = 8  # Similar to LongVU's implementation
        compressed_chunks = []
        
        for i in range(0, N, chunk_size):
            chunk = features[i:i+chunk_size]
            if len(chunk) == 1:
                compressed_chunks.append(chunk)
                continue
                
            # Compute chunk similarities
            chunk_norm = F.normalize(chunk, p=2, dim=-1)
            sim = F.cosine_similarity(
                chunk_norm[:-1],
                chunk_norm[1:],
                dim=-1
            )
            
            # Keep first token and dissimilar tokens
            keep_mask = sim < self.sim_threshold
            kept_tokens = torch.cat([
                chunk[:1],
                chunk[1:][keep_mask]
            ])
            compressed_chunks.append(kept_tokens)
        
        compressed_features = torch.cat(compressed_chunks)
        
        # Ensure we don't exceed max length
        if len(compressed_features) > self.L_max:
            compressed_features = compressed_features[:self.L_max]
            
        return compressed_features

    def forward(self, x_s, x_l, label):
        # Get text features
        prompts = self.prompt_learner()
        text_features = self.text_encoder(prompts, self.prompt_learner.tokenized_prompts)[self.num_classes:]
        
        # Encode and project features
        features = self.feature_encoder(x_l.float())
        
        # Apply token selection and compression
        selected_features, _ = self.adaptive_token_selection(features, text_features)
        compressed_features = self.spatial_token_compression(selected_features, text_features)
        
        # Prepare for attention
        compressed_features = compressed_features.unsqueeze(0)  # Add batch dimension
        text_features = text_features.unsqueeze(0)
        
        # Cross attention
        attended_features = self.cross_attention(
            text_features,
            compressed_features,
            compressed_features
        )
        
        # Classification
        final_features = attended_features.mean(1)
        logits = self.classifier(final_features)
        
        # Compute loss and predictions
        loss = self.loss_ce(logits, label)
        Y_prob = F.softmax(logits, dim=1)
        Y_hat = torch.topk(Y_prob, 1, dim=1)[1]
        
        return Y_prob, Y_hat, loss