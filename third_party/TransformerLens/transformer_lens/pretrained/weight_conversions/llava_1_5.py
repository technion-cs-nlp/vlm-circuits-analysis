import einops
import torch

from transformer_lens.HookedTransformerConfig import HookedTransformerConfig


def convert_llava_1_5_weights(llava, cfg: HookedTransformerConfig):
    state_dict = {}

    state_dict["embed.W_E"] = llava.language_model.model.embed_tokens.weight

    assert cfg.d_mlp is not None  # keep mypy happy

    for l in range(cfg.n_layers):
        # HACK to get the positional embeddings from HF model (theres currently a bug in the TLens implementation)
        pos_embed = llava.language_model.model.layers[l].self_attn.rotary_emb(
            torch.ones(1, device=llava.device, dtype=cfg.dtype),
            torch.arange(cfg.n_ctx, device=llava.device).unsqueeze(0),
        )
        cos, sin = pos_embed
        state_dict[f"blocks.{l}.attn.rotary_cos"] = cos.squeeze(0).to(cfg.device)
        state_dict[f"blocks.{l}.attn.rotary_sin"] = sin.squeeze(0).to(cfg.device)

        state_dict[f"blocks.{l}.ln1.w"] = llava.language_model.model.layers[
            l
        ].input_layernorm.weight

        W_Q = llava.language_model.model.layers[l].self_attn.q_proj.weight
        W_K = llava.language_model.model.layers[l].self_attn.k_proj.weight
        W_V = llava.language_model.model.layers[l].self_attn.v_proj.weight
        W_Q = einops.rearrange(W_Q, "(n h) m->n m h", n=cfg.n_heads)
        W_K = einops.rearrange(W_K, "(n h) m->n m h", n=cfg.n_heads)
        W_V = einops.rearrange(W_V, "(n h) m->n m h", n=cfg.n_heads)

        state_dict[f"blocks.{l}.attn.W_Q"] = W_Q
        state_dict[f"blocks.{l}.attn.W_K"] = W_K
        state_dict[f"blocks.{l}.attn.W_V"] = W_V

        # NO BIAS IN LLAVA
        state_dict[f"blocks.{l}.attn.b_Q"] = torch.zeros(cfg.n_heads, cfg.d_head, dtype=cfg.dtype)
        state_dict[f"blocks.{l}.attn.b_K"] = torch.zeros(cfg.n_heads, cfg.d_head, dtype=cfg.dtype)
        state_dict[f"blocks.{l}.attn.b_V"] = torch.zeros(cfg.n_heads, cfg.d_head, dtype=cfg.dtype)

        W_O = llava.language_model.model.layers[l].self_attn.o_proj.weight
        W_O = einops.rearrange(W_O, "m (n h)->n h m", n=cfg.n_heads)
        state_dict[f"blocks.{l}.attn.W_O"] = W_O

        state_dict[f"blocks.{l}.attn.b_O"] = torch.zeros(cfg.d_model, dtype=cfg.dtype)

        state_dict[f"blocks.{l}.ln2.w"] = llava.language_model.model.layers[
            l
        ].post_attention_layernorm.weight

        state_dict[f"blocks.{l}.mlp.W_in"] = llava.language_model.model.layers[
            l
        ].mlp.up_proj.weight.T
        state_dict[f"blocks.{l}.mlp.W_gate"] = llava.language_model.model.layers[
            l
        ].mlp.gate_proj.weight.T
        state_dict[f"blocks.{l}.mlp.b_in"] = torch.zeros(cfg.d_mlp, dtype=cfg.dtype)

        state_dict[f"blocks.{l}.mlp.W_out"] = llava.language_model.model.layers[
            l
        ].mlp.down_proj.weight.T
        state_dict[f"blocks.{l}.mlp.b_out"] = torch.zeros(cfg.d_model, dtype=cfg.dtype)

    state_dict["ln_final.w"] = llava.language_model.model.norm.weight

    state_dict["unembed.W_U"] = llava.language_model.lm_head.weight.T
    state_dict["unembed.b_U"] = torch.zeros(cfg.d_vocab, dtype=cfg.dtype)

    return state_dict
