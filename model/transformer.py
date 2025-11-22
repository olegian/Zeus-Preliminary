from typing import Dict
import einops
import torch
import torch.nn as nn
import timeit

from model.causal_multi_head_attention import CausalMultiHeadAttentionLayer
from model.rms_norm import RMSNormLayer
from model.swiglu import SwiGluLayer


class TransformerBlock(nn.Module):
    def __init__(self, 
            d_model: int, 
            num_heads: int,
            d_ff: int,
            theta: float,
            max_seq_len: int,
            eps: float = 1e-5,
            device: torch.device | None = None,
            dtype: torch.dtype | None = None
        ):

        super().__init__()

        self.rms_attn = RMSNormLayer(d_model, eps=eps, device=device, dtype=dtype)
        self.attn_layer = CausalMultiHeadAttentionLayer(
            d_model,
            num_heads, 
            mask=True, 
            theta=theta,
            max_seq_len=max_seq_len,
            device=device,
            dtype=dtype
        )

        self.rms_ff = RMSNormLayer(d_model, eps=eps, device=device, dtype=dtype)
        self.ff = SwiGluLayer(d_model, d_ff, device=device, dtype=dtype)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:    
        *_, seq_len, _ = x.size()
        token_positions = torch.arange(seq_len, device=x.device, dtype=torch.long)

        attn_layer_rms = self.rms_attn(x)
        attn_layer_fwd = self.attn_layer(attn_layer_rms, token_positions)
        attn = x + attn_layer_fwd
        ff_layer_rms = self.rms_ff(attn)
        ff_layer_fwd = self.ff(ff_layer_rms)

        return attn + ff_layer_fwd

    def load(self, state_dict: Dict[str, torch.Tensor]):
        self.rms_attn.load_state_dict({
            "gains": state_dict['ln1.weight']
        })
        self.rms_ff.load_state_dict({
            "gains": state_dict['ln2.weight']
        })

        attn_W = torch.hstack((state_dict['attn.q_proj.weight'], state_dict['attn.k_proj.weight'], state_dict['attn.v_proj.weight']))
        self.attn_layer.load_state_dict({
            "W_o": state_dict['attn.output_proj.weight'],
            "W": attn_W
        })
        
        self.ff.load_state_dict({
            "w1": state_dict['ffn.w1.weight'],
            "w2": state_dict['ffn.w2.weight'],
            "w3": state_dict['ffn.w3.weight'],
        })

