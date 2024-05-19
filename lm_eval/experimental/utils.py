import os
import torch
import math
from typing import Optional, Tuple
import torch.nn.functional as F
from enum import Enum, auto
import numpy as np
import pandas as pd
from torch.autograd import Function


abs_bias = lambda tensor1, tensor2: torch.mean((tensor1.type(torch.float32) - tensor2.type(torch.float32))).abs().item()
be = lambda tensor1, tensor2: (tensor1 - tensor2).abs().sum().item() == 0.0
mse = lambda tensor1, tensor2: torch.mean((tensor1.type(torch.float32) - tensor2.type(torch.float32))**2).item()

def cs(tensor1:torch.Tensor, tensor2:torch.Tensor, return_angle=False) -> float:
    tensor1 = tensor1.type(torch.float32)
    tensor2 = tensor2.type(torch.float32)
    if mse(tensor1, tensor2) == 0:
        if return_angle:
            return 0
        else:
            return 1.0
    cs_ = (tensor1.flatten() / torch.norm(tensor1)) @ (tensor2.flatten() / torch.norm(tensor2))
    cs_ = torch.clamp(cs_, -1.0, 1.0)
    if return_angle:
        return (torch.arccos(cs_) / torch.pi * 180).item()
    else:
        return cs_.item()


# TODO: Use torch.softmax instead.
def softmax(input_:torch.Tensor, dim:int) -> torch.Tensor:
    exps = input_.exp()
    sum_exps = exps.sum(dim=dim, keepdim=True)
    softmax_output = exps / sum_exps
    return softmax_output


class CLIP_TYPE(Enum):
    NONE = auto()
    GRID = auto()
    GAUSS = auto()
    LAPLACE = auto()
    ACIQ_GAUSS = auto()
    ACIQ_LAPLACE = auto()
    AVG = auto()


def quant_dequant(t:torch.Tensor, bitwidth:int, min_value:float, clip_value:Optional[float]=None) -> torch.Tensor:
    # NOTE: This operation assumes that t <= 0.
    min_ = min_value
    max_ = 0.0
    if clip_value is not None:
        min_ = max(min_, clip_value)
    
    t = t.clip(min_, max_)
    act_scale = (2 ** bitwidth - 1) / (max_ - min_)
    t = (t - min_) * act_scale
    t = t.round()
    t = (t / act_scale) + min_
    return t


class MinEstimator:
    def __init__(self, calibrate:bool) -> None:
        self.calibrate = calibrate
        self.calibrate_steps = 25
        self.min_values = []

    def get_min_value(self, t:torch.Tensor) -> float:
        if self.calibrate and (len(self.min_values) >= self.calibrate_steps):
            return np.array(self.min_values).mean()
        
        min_value = t.min().item()

        if self.calibrate:
            self.min_values.append(min_value)
        
        return min_value
        

class ClipEstimator:
    def __init__(self, bitwidth:int, clip_type:CLIP_TYPE.NONE, use_optimal_clipping_values:bool, calibrate:bool) -> None:
        self.bitwidth = bitwidth
        self.clip_type = clip_type
        self.use_optimal_clipping_values = use_optimal_clipping_values
        self.calibrate = calibrate
        self.calibrate_steps = 25
        self.clip_values = []

        if self.use_optimal_clipping_values:
            optimal_clipping_values_df = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'optimal_clipping_values.csv'))
            self.std_list = optimal_clipping_values_df['std'].values
            self.clip_list = optimal_clipping_values_df[f'int{self.bitwidth}'].values


    def get_clip_value(self, t:torch.Tensor) -> float:
        if self.clip_type == CLIP_TYPE.NONE:
            return None

        if self.calibrate and (len(self.clip_values) >= self.calibrate_steps):
            return np.array(self.clip_values).mean()
        
        if self.clip_type == CLIP_TYPE.GRID:
            clip_value = self.__grid(t)
        if self.clip_type == CLIP_TYPE.GAUSS:
            clip_value = self.__gauss(t)
        if self.clip_type == CLIP_TYPE.LAPLACE:
            clip_value = self.__laplace(t)
        if self.clip_type == CLIP_TYPE.ACIQ_GAUSS:
            clip_value = self.__aciq_gauss(t)
        if self.clip_type == CLIP_TYPE.ACIQ_LAPLACE:
            clip_value = self.__aciq_laplace(t)
        if self.clip_type == CLIP_TYPE.AVG:
            clip_value = self.__avg(t)
 
        
        if self.calibrate:
            self.clip_values.append(clip_value)
        
        return clip_value
    
    
    def __grid(self, t:torch.Tensor) -> float:
        clip_value_list = np.linspace(t.min().item(), 0, 100)[:-1]
        mse_list = []
        # ref = t.to(torch.bfloat16).exp()
        # for clip_value in clip_value_list:
        #     mse_list.append(mse(ref, quant_dequant(t, bitwidth=self.bitwidth, clip_value=clip_value).to(torch.bfloat16).exp()))
        
        # TODO: Experimental
        ref = softmax(t.original_t.to(torch.bfloat16), dim=-1)
        for clip_value in clip_value_list:
            tmp = quant_dequant(t, bitwidth=self.bitwidth, min_value=t.min().item(), clip_value=clip_value)
            t.original_t.flatten()[t.original_mask.flatten()] = tmp
            tmp = softmax(t.original_t.to(torch.bfloat16), dim=-1)
            mse_list.append(mse(ref, tmp))

        clip_value = clip_value_list[np.array(mse_list).argmin()]
        return clip_value

    
    def __gauss(self, t:torch.Tensor) -> float:
        std = t.std().item()

        if self.use_optimal_clipping_values:
            clip_value = np.interp(std, self.std_list, self.clip_list)
        else:
            if self.bitwidth == 2:
                clip_value = -1.66 * std - 1.85
            if self.bitwidth == 3:
                clip_value = -1.75 * std - 2.06
            if self.bitwidth == 4:
                clip_value = -1.83 * std - 2.27

        return clip_value
    
    def __laplace(self, t:torch.Tensor) -> float:
        sigma = (t - t.mean()).abs().mean().item()
        if self.bitwidth == 2:
            clip_value = -4.01 * sigma - 1.77
        if self.bitwidth == 4:
            clip_value = -6.82 * sigma - 0.15
        return clip_value
    
    def __aciq_gauss(self, t:torch.Tensor) -> float:
        alpha_gauss = {2: 1.71, 3: 2.15, 4: 2.55, 5: 2.93, 6: 3.28, 7: 3.61, 8: 3.92}
        gauss_const = (0.5 * 0.35) * (1 + (math.pi * math.log(4)) ** 0.5)
        min_val = t.min()
        max_val = t.max()
        std_val = ((max_val - min_val) * gauss_const) / ((2 * math.log(t.view(-1).size()[0])) ** 0.5)
        alpha_val = alpha_gauss[self.bitwidth] * std_val
        mean_val = t.mean()
        delta = 2 * alpha_val
        clip_value = max(min_val, mean_val - delta / 2).item()
        return clip_value
    
    def __aciq_laplace(self, t:torch.Tensor) -> float:
        alpha_laplace = {2: 2.83, 3: 3.89, 4: 5.03, 5: 6.2, 6: 7.41, 7: 8.64, 8: 9.89}
        b = torch.mean(torch.abs(t - t.mean()))
        alpha_val = alpha_laplace[self.bitwidth] * b
        mean_val = t.mean()
        min_val = t.min()
        delta = 2 * alpha_val
        clip_value = max(min_val, mean_val - delta / 2).item()
        return clip_value
    
    def __avg(self, t:torch.Tensor) -> float:
        original_t = t.original_t
        original_mask = t.original_mask
        B = original_t.size(0)
        min_list = []
        for i in range(B):
            tmp_t = original_t[i]
            tmp_mask = original_mask[i]
            min_list.append(tmp_t.flatten()[tmp_mask.flatten()].min().item())
        clip_value = np.mean(min_list)
        return clip_value


class Cast:
    def __init__(self, quantize:bool, cast_dtype:torch.dtype, bitwidth:int, clip_type:CLIP_TYPE.NONE, 
                 use_optimal_clipping_values:bool, calibrate:bool) -> None:
        self.quantize = quantize
        self.cast_dtype = cast_dtype
        self.bitwidth = bitwidth
        self.min_estimator = MinEstimator(calibrate)
        self.clip_estimator = ClipEstimator(bitwidth, clip_type, use_optimal_clipping_values, calibrate)

    def __call__(self, t:torch.Tensor, mask:torch.Tensor) -> torch.Tensor:
        if self.quantize:
            t_dtype = t.dtype
            t = t.to(self.cast_dtype)

            masked_t = t.flatten()[mask.flatten()]
            # -------------------------------------------------------------------------------
            # TODO: The following is a quick workaround, and should be removed in the future.
            masked_t.original_t = t.clone()
            masked_t.original_mask = mask.clone()
            # -------------------------------------------------------------------------------
            min_value = self.min_estimator.get_min_value(masked_t)
            clip_value = self.clip_estimator.get_clip_value(masked_t)
            masked_t = quant_dequant(masked_t, self.bitwidth, min_value, clip_value)
            t.flatten()[mask.flatten()] = masked_t
            
            return t.to(t_dtype)
        else:
            return t
        

class SoftmaxFunction(Function):
    @staticmethod
    def forward(ctx, input_:torch.Tensor, dim:int, mask:torch.Tensor, cast:Cast, layer_idx:int, call_idx:int) -> torch.Tensor:
        # Find the maximum value along the specified dimension for numerical stability
        max_value = input_.max(dim=dim, keepdim=True)[0]
        stabilized_input = input_ - max_value

        if os.getenv('DUMP', 'false').lower() == 'true':
            if call_idx < 1:
                model_ = 'llama2_7b'
                torch.save(mask.detach().cpu(), f'/root/tmp/{model_}/mask_layer_{layer_idx}__call_{call_idx}')
                torch.save(stabilized_input.detach().cpu(), f'/root/tmp/{model_}/stabilized_input_layer_{layer_idx}__call_{call_idx}')


        stabilized_input = cast(stabilized_input, mask) # Int2

        # Compute the softmax on the stabilized input
        softmax_output = softmax(stabilized_input, dim)

        # Save for backward pass
        ctx.save_for_backward(softmax_output)
        ctx.dim = dim

        return softmax_output
    
    @staticmethod
    def backward(ctx, grad_output:torch.Tensor) -> torch.Tensor:
        softmax_output, = ctx.saved_tensors
        dim = ctx.dim

        # Compute gradient of softmax
        grad_input = softmax_output * (grad_output - (softmax_output * grad_output).sum(dim=dim, keepdim=True))

        return grad_input, None, None, None, None, None


class Softmax:
    def __init__(self, quantize:bool, cast_dtype:torch.dtype, bitwidth:int, clip_type:CLIP_TYPE.NONE, 
                 use_optimal_clipping_values:bool, calibrate:bool, layer_idx:int) -> None:
        self.cast = Cast(quantize, cast_dtype, bitwidth, clip_type, use_optimal_clipping_values, calibrate)
        self.layer_idx = layer_idx
        self.call_idx = 0

    def __call__(self, input_:torch.Tensor, dim:int, mask:torch.Tensor) -> torch.Tensor:
        output_ = SoftmaxFunction.apply(input_, dim, mask, self.cast, self.layer_idx, self.call_idx)
        self.call_idx += 1
        return output_


class SDPA:
    def __init__(self, 
                 quantize:bool=False, 
                 cast_dtype:torch.dtype=torch.float32, 
                 bitwidth:int=2,
                 clip_type:CLIP_TYPE=CLIP_TYPE.NONE,
                 use_optimal_clipping_values:bool=False,
                 calibrate:bool=False,
                 layer_idx:int=0) -> None:
        self.softmax = Softmax(quantize, cast_dtype, bitwidth, clip_type, use_optimal_clipping_values, calibrate, layer_idx)

    def __call__(self, query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None) -> torch.Tensor:
        # Efficient implementation equivalent to the following:
        B, L, S = query.size(0), query.size(-2), key.size(-2)
        scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
        attn_bias = torch.zeros(B, 1, L, S, dtype=query.dtype, device=query.device)
        if is_causal:
            assert attn_mask is None
            temp_mask = torch.ones(B, 1, L, S, dtype=torch.bool).tril(diagonal=0).to(query.device)
            attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
            attn_bias.to(query.dtype)

        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
            else:
                attn_bias += attn_mask
        attn_weight = query @ key.transpose(-2, -1) * scale_factor
        attn_weight += attn_bias
        mask = (torch.zeros_like(attn_weight) + attn_bias) == 0
        attn_weight = self.softmax(attn_weight, dim=-1, mask=mask)
        attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
        return attn_weight @ value



# NOTE: transformers==4.40.0
from transformers.cache_utils import Cache
from transformers.models.llama.modeling_llama import logger, apply_rotary_pos_emb, repeat_kv
def LlamaSdpaAttention_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    if output_attentions:
        assert False

    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    cos, sin = self.rotary_emb(value_states, position_ids)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    # In case static cache is used, it is an instance attribute.
    past_key_value = getattr(self, "past_key_value", past_key_value)

    if past_key_value is not None:
        # sin and cos are specific to RoPE models; position_ids needed for the static cache
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    causal_mask = attention_mask
    if attention_mask is not None:
        causal_mask = causal_mask[:, :, :, : key_states.shape[-2]]

    # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
    # Reference: https://github.com/pytorch/pytorch/issues/112577.
    if query_states.device.type == "cuda" and causal_mask is not None:
        query_states = query_states.contiguous()
        key_states = key_states.contiguous()
        value_states = value_states.contiguous()

    attn_output = self.sdpa(
        query_states,
        key_states,
        value_states,
        attn_mask=causal_mask,
        dropout_p=self.attention_dropout if self.training else 0.0,
        is_causal=causal_mask is None and q_len > 1,
    )

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.view(bsz, q_len, self.hidden_size)

    attn_output = self.o_proj(attn_output)

    return attn_output, None, past_key_value


# NOTE: transformers==4.40.0
import warnings
def LlamaDecoderLayer_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
    """
    Args:
        hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
        attention_mask (`torch.FloatTensor`, *optional*):
            attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
            query_sequence_length, key_sequence_length)` if default attention is used.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under
            returned tensors for more detail.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
            (see `past_key_values`).
        past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
    """
    if "padding_mask" in kwargs:
        warnings.warn(
            "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
        )
    if hidden_states.device != self.input_layernorm.weight.device:
        hidden_states = hidden_states.to(self.input_layernorm.weight.device)
    if attention_mask is not None:
        if attention_mask.device != self.input_layernorm.weight.device:
            attention_mask = attention_mask.to(self.input_layernorm.weight.device)
    if position_ids is not None:
        if position_ids.device != self.input_layernorm.weight.device:
            position_ids = position_ids.to(self.input_layernorm.weight.device)
    if cache_position is not None:
        if cache_position.device != self.input_layernorm.weight.device:
            cache_position = cache_position.to(self.input_layernorm.weight.device)

    residual = hidden_states

    hidden_states = self.input_layernorm(hidden_states)

    # Self Attention
    hidden_states, self_attn_weights, present_key_value = self.self_attn(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_value=past_key_value,
        output_attentions=output_attentions,
        use_cache=use_cache,
        cache_position=cache_position,
        **kwargs,
    )
    hidden_states = residual + hidden_states

    # Fully Connected
    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = self.mlp(hidden_states)
    hidden_states = residual + hidden_states

    outputs = (hidden_states,)

    if output_attentions:
        outputs += (self_attn_weights,)

    if use_cache:
        outputs += (present_key_value,)

    return outputs


def Embedding_forward(self, input: torch.Tensor) -> torch.Tensor:
    if input.device != self.weight.device:
        input = input.to(self.weight.device)

    return F.embedding(
        input, self.weight, self.padding_idx, self.max_norm,
        self.norm_type, self.scale_grad_by_freq, self.sparse)