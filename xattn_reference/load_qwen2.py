from typing import Optional, Tuple
import torch
from transformers import AutoTokenizer
from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2ForCausalLM,
    Qwen2Attention,
    apply_rotary_pos_emb,
    repeat_kv,
)
import logging
from transformers.cache_utils import Cache
from flash_attn import flash_attn_func
import time
import types

# from xattn.threshold.llama_threshold import llama_fuse_4, llama_fuse_8, llama_fuse_16
try:
    from xattn.src.Xattention import Xattention_prefill
except:
    print("Xattention Import Fail")
try:
    from xattn.src.Minference import Minference_prefill
except:
    print("Minference Prefill Import Fail")
try:
    from xattn.src.Fullprefill import Full_prefill
except:
    print("Full Prefill Import Fail")
try:
    from xattn.src.Flexprefill import Flexprefill_prefill
except:
    print("Flex Prefill Import Fail")
try:
    from xattn.src.XFlex import XFlex_prefill
except:
    print("XFlex Import Fail")
# from xattn.src.Tattention import Tattention_prefill
from xattn.src.utils import *

logger = logging.getLogger(__name__)


def qwen2_forward_eval(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Qwen2 attention forward with XAttention support.
    
    Adapted from Qwen2Attention.forward to support various sparse attention mechanisms.
    """
    if self.fastprefillconfig.print_detail:
        start_time = time.time()
    
    bsz, q_len, _ = hidden_states.size()

    # Linear projections
    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    # Reshape for attention computation
    query_states = query_states.view(bsz, q_len, self.config.num_attention_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.config.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.config.num_key_value_heads, self.head_dim).transpose(1, 2)

    if self.fastprefillconfig.print_detail:
        torch.cuda.synchronize()
        reshape_time = time.time() - start_time
        print(f"     Reshape took: {reshape_time:.6f} seconds")

    # Handle KV cache and position embeddings
    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        if self.layer_idx is None:
            raise ValueError(
                f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                "with a layer index."
            )
        kv_seq_len += past_key_value.get_seq_length(self.layer_idx)

    # Apply rotary positional embeddings
    if position_embeddings is None:
        # Compute RoPE embeddings if not provided
        if cache_position is not None:
            rotary_seq_len = max(kv_seq_len, cache_position[-1].max().item()) + 1
        else:
            rotary_seq_len = max(kv_seq_len, position_ids[:, -1].max().item()) + 1
        cos, sin = self.rotary_emb(value_states, position_ids)
    else:
        # Use provided position embeddings
        cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

    if self.fastprefillconfig.print_detail:
        start_time = time.time()

    # Update KV cache if needed
    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

    # Determine if we're in decoding mode
    _, _, k_len, _ = key_states.shape
    _, _, q_len, _ = query_states.shape
    decoding = (q_len != k_len and q_len == 1)

    # Expand key/value for multi-head attention (only for prefill)  
    if not decoding:
        key_states = repeat_kv(key_states, self.num_key_value_groups).to("cuda")
        value_states = repeat_kv(value_states, self.num_key_value_groups).to("cuda")

    if self.fastprefillconfig.print_detail:
        torch.cuda.synchronize()
        past_kv_time = time.time() - start_time
        print(f"     Past KV update and repeat took: {past_kv_time:.6f} seconds")

    if self.fastprefillconfig.print_detail:
        start_time = time.time()
        print(f"q length: {q_len} k length: {k_len}")

    stride = self.fastprefillconfig.stride
    if not decoding:
        if self.fastprefillconfig.metric == "flex":
            gamma = getattr(self.fastprefillconfig, 'gamma', 0.9)
            tau = getattr(self.fastprefillconfig, 'tau', 0.1)
            attn_output = Flexprefill_prefill(query_states.transpose(1, 2), key_states.transpose(1, 2), value_states.transpose(1, 2), gamma=gamma, tau=tau).transpose(1, 2)
        elif self.fastprefillconfig.metric == "xattn":
            # 支持简化注意力模式
            use_simple = getattr(self.fastprefillconfig, 'use_simple_attention', False)
            if isinstance(self.fastprefillconfig.threshold, torch.Tensor):
                attn_output = Xattention_prefill(query_states, key_states, value_states, stride, norm=1, threshold=self.fastprefillconfig.threshold[self.layer_idx], use_triton=True, use_simple_attention=use_simple, layer_idx=self.layer_idx)
            else:
                attn_output = Xattention_prefill(query_states, key_states, value_states, stride, norm=1, threshold=self.fastprefillconfig.threshold, use_triton=True, use_simple_attention=use_simple, layer_idx=self.layer_idx)
        # elif self.fastprefillconfig.metric == "tattn":
        #     if isinstance(self.fastprefillconfig.threshold, torch.Tensor):
        #         attn_output = Tattention_prefill(
        #             query_states, key_states, value_states, stride, 
        #             norm=1, 
        #             threshold=self.fastprefillconfig.threshold[self.layer_idx],
        #             use_triton=True, 
        #             score_ratio=getattr(self.fastprefillconfig, 'score_ratio', 0.9),
        #         )
        #     else:
        #         attn_output = Tattention_prefill(
        #             query_states, key_states, value_states, stride, 
        #             norm=1, 
        #             threshold=self.fastprefillconfig.threshold,
        #             use_triton=True, 
        #             score_ratio=getattr(self.fastprefillconfig, 'score_ratio', 0.9),
        #         )
        elif self.fastprefillconfig.metric == "xflex":
            if isinstance(self.fastprefillconfig.threshold, torch.Tensor):
                attn_output = XFlex_prefill(
                    query_states, key_states, value_states, 
                    stride=stride, 
                    norm=1, 
                    threshold=self.fastprefillconfig.threshold[self.layer_idx],
                    use_triton=True, 
                    score_ratio=getattr(self.fastprefillconfig, 'score_ratio', 0.95),
                ).transpose(1, 2)
            else:
                attn_output = XFlex_prefill(
                    query_states, key_states, value_states, 
                    stride=stride, 
                    norm=1, 
                    threshold=self.fastprefillconfig.threshold,
                    use_triton=True, 
                    score_ratio=getattr(self.fastprefillconfig, 'score_ratio', 0.95),
                ).transpose(1, 2)
        elif self.fastprefillconfig.metric == "full":
            attn_output = Full_prefill(query_states, key_states, value_states, attention_mask=attention_mask, layer=self.layer_idx)
        elif self.fastprefillconfig.metric == "minfer":
            attn_output = Minference_prefill(query_states, key_states, value_states, adaptive_budget=0.3)
    else:
        # Decoding: use flash_attn
        if key_states.device != query_states.device:
            key_states = key_states.to(query_states.device)
        if value_states.device != query_states.device:
            value_states = value_states.to(query_states.device)

        # Reshape for flash_attn
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)
        
        attn_output = flash_attn_func(
            query_states, key_states, value_states,
            dropout_p=0.0, softmax_scale=None, causal=True
        )
        attn_output = attn_output.transpose(1, 2)

    if self.fastprefillconfig.print_detail:
        torch.cuda.synchronize()
        attn_time = time.time() - start_time
        print(f"     Attention computation took: {attn_time:.6f} seconds")

    if self.fastprefillconfig.print_detail:
        start_time = time.time()

    # Output projection and reshaping
    if attn_output.size() != (bsz, self.config.num_attention_heads, q_len, self.head_dim):
        raise ValueError(
            f"`attn_output` should be of size {(bsz, self.config.num_attention_heads, q_len, self.head_dim)}, but is"
            f" {attn_output.size()}"
        )
    
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, q_len, -1)
    attn_output = self.o_proj(attn_output)

    if self.fastprefillconfig.print_detail:
        torch.cuda.synchronize()
        post_attn_time = time.time() - start_time
        print(f"     Post-attention processing took: {post_attn_time:.6f} seconds")

    return attn_output, None, past_key_value


class FastPrefillConfig(dict):
    """
    Configuration class for FastPrefill, which provides flexible settings for optimizing 
    prefill computations in transformer models.

    Attributes:
    - threshold (float or torch.Tensor, optional): The threshold for selecting relevant attention blocks.
    - print_detail (bool): Whether to print detailed timing and debugging information.
    - stride (int): Determines the level of fused attention computation (e.g., 16, 8, or 4).
    - metric (str): Defines the type of prefill mechanism used ('xattn', 'full', 'minfer', 'flex').
    """

    def __init__(
        self,
        threshold: float = None,
        print_detail: bool = False,
        stride: int = 16,
        metric: str = "xattn",
        gamma: float = 0.9,
        tau: float = 0.1,
        score_ratio: float = 0.9,
        use_simple_attention = False,
    ):
        """
        Initialize the configuration with default or user-provided values.
        
        Args:
            use_simple_attention: Simple attention mode. Supports:
                - False/0: Disabled  
                - True/1/"v1": V1 version (standard)
                - 2/"v2": V2 version
                - 3/"v3": V3 version 
                - 4/"v4": V4 version (dynamic scale + middle query position)
                - 5/"v5": V5 version (mixed query components)
                - 6/"v6": V6 version (golden ratio selection + temperature)
        """
        super().__init__()
        self.print_detail = print_detail
        self.metric = metric
        self.stride = stride
        self.gamma = gamma
        self.tau = tau
        self.score_ratio = score_ratio
        self.use_simple_attention = use_simple_attention

        if threshold is not None:
            print("I'm using the fixed threshold.")
            self.threshold = torch.ones((28, 28)).to("cuda") * threshold
        else:
            # For full attention or other modes that don't need threshold
            self.threshold = torch.ones((28, 28)).to("cuda") * 0.9  # Default fallback
        # else:
            # # Load appropriate threshold based on stride (use llama thresholds)
            # from xattn.threshold.llama_threshold import llama_fuse_4, llama_fuse_8, llama_fuse_16
            # if stride == 16:
            #     self.threshold = torch.tensor(llama_fuse_16)
            # elif stride == 8:
            #     self.threshold = torch.tensor(llama_fuse_8)
            # elif stride == 4:
            #     self.threshold = torch.tensor(llama_fuse_4)
        self.threshold = self.threshold.to("cuda")

def load_qwen2_model(fastprefillconfig=None, name_or_path=""):
    """
    Loads a Qwen2 model with FastPrefill optimizations applied.

    This function initializes the model, applies the FastPrefill configuration to attention 
    layers, and loads the tokenizer.

    Parameters:
    - fastprefillconfig (FastPrefillConfig, optional): The configuration for FastPrefill optimizations.
    - name_or_path (str): The model path or identifier for loading the pre-trained model.

    Returns:
    - model: The loaded Qwen2 model with FastPrefill optimizations.
    - tokenizer: The tokenizer associated with the model.
    """
    if fastprefillconfig is None:
        fastprefillconfig = FastPrefillConfig()
    
    # Load the Qwen2 model
    model = Qwen2ForCausalLM.from_pretrained(
        name_or_path,
        device_map="balanced",
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2"
    )
    model.eval()
    
    # Replace attention forward functions in all layers
    for layer in model.model.layers:
        if hasattr(layer, 'self_attn') and isinstance(layer.self_attn, Qwen2Attention):
            layer.self_attn.fastprefillconfig = fastprefillconfig
            # Set reference to rotary embedding
            layer.self_attn.rotary_emb = model.model.rotary_emb
            layer.self_attn.forward = types.MethodType(qwen2_forward_eval, layer.self_attn)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(name_or_path)
    
    print(f"✅ Successfully loaded Qwen2 model with {fastprefillconfig.metric} attention")
    print(f"📋 Model: {name_or_path}")
    print(f"🎯 Attention metric: {fastprefillconfig.metric}")
    
    return model, tokenizer# For backward compatibility, create an alias
load_model = load_qwen2_model



