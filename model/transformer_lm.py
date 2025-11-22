from typing import Dict
import torch
import torch.nn as nn

from model.embedding import EmbeddingLayer
from model.linear import LinearLayer
from model.rms_norm import RMSNormLayer
from model.transformer import TransformerBlock

class TransformerLM(nn.Module):
    def __init__(self, 
            d_model: int, 
            num_heads: int,
            d_ff: int,
            theta: float,
            max_seq_len: int,
            vocab_size: int,
            context_length: int,
            num_layers: int,
            eps: float = 1e-5,
            device: torch.device | None = None,
            dtype: torch.dtype | None = None
        ):
        super().__init__()

        self.embedding_layer = EmbeddingLayer(
            num_embeddings=vocab_size,
            embedding_dim=d_model,
            dtype=dtype
        )

        self.transformer_layers = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                theta=theta,
                max_seq_len=max_seq_len,
                eps=eps,
                device=device,
                dtype=dtype
            )
            for _ in range(num_layers)
        ])

        self.norm_layer = RMSNormLayer(
            d_model=d_model,
            eps=eps,
            dtype=dtype
        )

        self.output_layer = LinearLayer(
            in_features=d_model,
            out_features=vocab_size,
            device=device,
            dtype=dtype
        )
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:    
        x = self.embedding_layer(x)

        for transformer_layer in self.transformer_layers:
            x = transformer_layer(x)

        x = self.norm_layer(x)
        
        return self.output_layer(x)
        

    def load(self, state_dict: Dict[str, torch.Tensor]):
        self.embedding_layer.load_state_dict(
            {
                "embeddings": state_dict['token_embeddings.weight']
            }
        )

        for i, transformer_layer in enumerate(self.transformer_layers):
            transformer_state_dict = {
                ".".join(k.split(".")[2:]): v
                for k, v in state_dict.items() if k.startswith(f"layers.{i}")
            }
            transformer_layer.load(transformer_state_dict)

        self.norm_layer.load_state_dict({
            "gains": state_dict['ln_final.weight']
        })

        self.output_layer.load_state_dict({
            'w': state_dict['lm_head.weight']
        })
