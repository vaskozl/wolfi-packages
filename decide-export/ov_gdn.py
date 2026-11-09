#  Copyright 2026 The HuggingFace Team. All rights reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

# Vendored from optimum-intel 2.2.0 (optimum/exporters/openvino/_ov_ops.py and
# model_patcher.py): optimum-intel cannot be installed next to transformers 5.19.
# The code below is unchanged so the OpenVINO CPU plugin keeps fusing the Loop
# into its GatedDeltaNet op.
import numpy as np
import openvino as ov
import openvino.opset14 as ops
import torch
import torch.nn.functional as F


def convert_recurrent_attention_cell(context):
    query = context.get_input(0)
    key = context.get_input(1)
    value = context.get_input(2)
    g = context.get_input(3)
    beta = context.get_input(4)
    last_recurrent_state_old = context.get_input(5)

    value_shape = ops.shape_of(value)
    const_zero = ops.constant(0, dtype=np.float32)
    core_attn_out = ops.broadcast(const_zero, value_shape)
    const_two_out = ops.constant(2, dtype=np.int32)
    const_zero_out = ops.constant(0, dtype=np.int32)
    seq_len = ops.gather(value_shape, const_two_out, const_zero_out)

    timestep_param = ops.parameter([], np.int32, "timestep")
    q_t_param = ops.parameter([-1, -1, 1, -1], np.float32, "q_t")
    k_t_param = ops.parameter([-1, -1, 1, -1], np.float32, "k_t")
    v_t_param = ops.parameter([-1, -1, 1, -1], np.float32, "v_t")
    g_t_param = ops.parameter([-1, -1, 1], np.float32, "g_t")
    beta_t_param = ops.parameter([-1, -1, 1], np.float32, "beta_t")
    last_recurrent_state_t = ops.parameter([-1, -1, -1, -1], np.float32, "last_recurrent_state_t")
    core_attn_out_t = ops.parameter([-1, -1, -1, -1], np.float32, "core_attn_out_t")

    const_two = ops.constant(2, dtype=np.int32)
    q_t = ops.squeeze(q_t_param, const_two)
    k_t = ops.squeeze(k_t_param, const_two)
    v_t = ops.squeeze(v_t_param, const_two)
    const_minus_one = ops.constant(-1, dtype=np.int32)
    g_t = ops.unsqueeze(ops.exp(g_t_param), const_minus_one)
    beta_t = beta_t_param

    last_recurrent_state_in = ops.multiply(last_recurrent_state_t, g_t)
    const_minus_two = ops.constant(-2, dtype=np.int32)
    kv_mem = ops.multiply(last_recurrent_state_in, ops.unsqueeze(k_t, const_minus_one))
    kv_mem = ops.reduce_sum(kv_mem, const_minus_two, False)
    delta = ops.multiply(ops.subtract(v_t, kv_mem), beta_t)
    last_recurrent_state_delta = ops.multiply(
        ops.unsqueeze(k_t, const_minus_one), ops.unsqueeze(delta, const_minus_two)
    )
    last_recurrent_state_in = ops.add(last_recurrent_state_in, last_recurrent_state_delta)
    core_attn_update = ops.multiply(last_recurrent_state_in, ops.unsqueeze(q_t, const_minus_one))
    core_attn_update = ops.reduce_sum(core_attn_update, const_minus_two, True)
    const_zero = ops.constant(0, dtype=np.int32)
    timestep = ops.unsqueeze(timestep_param, const_zero)

    core_attn_out_res = ops.scatter_update(core_attn_out_t, timestep, core_attn_update, const_two)
    last_recurrent_state_res = last_recurrent_state_in

    body_cond = ops.constant([True], dtype=bool)

    body_model = ov.Model(
        [body_cond, last_recurrent_state_res, core_attn_out_res],
        [
            timestep_param,
            q_t_param,
            k_t_param,
            v_t_param,
            g_t_param,
            beta_t_param,
            last_recurrent_state_t,
            core_attn_out_t,
        ],
        "body_model",
    )

    seq_len = ops.convert(seq_len, "i32")
    loop = ops.loop(seq_len, ops.constant(True, dtype="bool"))
    loop.set_function(body_model)

    loop.set_sliced_input(q_t_param, query, 0, 1, 1, -1, 2)
    loop.set_sliced_input(k_t_param, key, 0, 1, 1, -1, 2)
    loop.set_sliced_input(v_t_param, value, 0, 1, 1, -1, 2)
    loop.set_sliced_input(g_t_param, g, 0, 1, 1, -1, 2)
    loop.set_sliced_input(beta_t_param, beta, 0, 1, 1, -1, 2)
    loop.set_merged_input(last_recurrent_state_t, last_recurrent_state_old, last_recurrent_state_res.output(0))
    loop.set_merged_input(core_attn_out_t, core_attn_out.output(0), core_attn_out_res.output(0))
    loop.set_special_body_ports([0, 0])

    core_attn_out_new = loop.get_iter_value(core_attn_out_res.output(0), -1)
    last_recurrent_state_new = loop.get_iter_value(last_recurrent_state_res.output(0), -1)

    flatten_shape = ops.constant([-1], dtype=np.int32)
    core_attn_out_new = ops.reshape(core_attn_out_new, flatten_shape, False)
    last_recurrent_state_new = ops.reshape(last_recurrent_state_new, flatten_shape, False)

    final_output = ops.concat([core_attn_out_new, last_recurrent_state_new], 0)

    return [final_output.output(0)]


# Unified torch representation of the CausalConv1d operation
# This representation is used across all models with CausalConv1d.
# The resulting OV graph for this function can be fused into the internal CausalConv1d operation.
def ov_causal_conv1d(conv_state, input_embeds, weight, bias):
    _, hidden_size, seq_len = input_embeds.shape
    _, w_in_channels, _ = weight.shape
    state_len = conv_state.shape[-1]
    groups = hidden_size // w_in_channels

    input_embeds_new = torch.cat([conv_state, input_embeds.to(conv_state.dtype)], dim=-1)
    conv_out = F.conv1d(input_embeds_new.to(weight.dtype), weight, bias, padding=0, groups=groups)
    conv_out = conv_out[:, :, -seq_len:]

    new_conv_state = input_embeds_new[:, :, -state_len:]

    return conv_out, new_conv_state



# Patched implementation of the gated delta rule in recurrent form.
# Adapted from:
# https://github.com/huggingface/transformers/blob/v4.57-release/src/transformers/models/qwen3_next/modeling_qwen3_next.py#L522
#
# To represent the for-loop that generates output embeddings, we use a module
# and the conversion extension mechanism. This is necessary because there is
# no known vectorized form of this loop that would allow it to be correctly
# traced with torch.jit.trace
def patched_recurrent_gated_delta_rule(
    self, query, key, value, g, beta, initial_state, output_final_state, use_qk_l2norm_in_kernel=False
):
    def l2norm(x: torch.FloatTensor, dim: int = -1, eps: float = 1e-6):
        """This function is intended to align with the l2norm implementation in the FLA library."""
        inv_norm = torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)
        return x * inv_norm

    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, beta, g)
    ]

    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    scale = 1 / (query.shape[-1] ** 0.5)
    query = query * scale

    last_recurrent_state = (
        torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim).to(value)
        if initial_state is None
        else initial_state.to(value)
    )

    output_cell = self.recurrent_attention_cell(
        query,  # (B, H, T, D1)
        key,  # (B, H, T, D1)
        value,  # (B, H, T, D2)
        g,  # (B, H, T)
        beta,  # (B, H, T)
        last_recurrent_state,  # (B, H, D1, D2)
    )

    num_elems = value.numel()
    core_attn_out = output_cell[:num_elems].reshape(value.shape)
    last_recurrent_state = output_cell[num_elems:].reshape(last_recurrent_state.shape)

    if not output_final_state:
        last_recurrent_state = None
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state


# This torch.nn.Module represents the GatedDeltaNet layer in its recurrent form.
# It is required for converting the GatedDeltaNet layer with OpenVINO using the ModuleExtension mechanism.
class RecurrentAttentionCell(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        query,  # (B, H, T, D1)
        key,  # (B, H, T, D1)
        value,  # (B, H, T, D2)
        g,  # (B, H, T)
        beta,  # (B, H, T)
        last_recurrent_state,  # (B, H, D1, D2)
    ):
        _, _, sequence_length, _ = key.shape
        core_attn_out = torch.zeros_like(value)

        for i in range(sequence_length):
            q_t = query[:, :, i]
            k_t = key[:, :, i]
            v_t = value[:, :, i]
            g_t = g[:, :, i].exp().unsqueeze(-1).unsqueeze(-1)
            beta_t = beta[:, :, i].unsqueeze(-1)

            last_recurrent_state = last_recurrent_state * g_t
            kv_mem = (last_recurrent_state * k_t.unsqueeze(-1)).sum(dim=-2)
            delta = (v_t - kv_mem) * beta_t
            last_recurrent_state = last_recurrent_state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
            core_attn_out[:, :, i] = (last_recurrent_state * q_t.unsqueeze(-1)).sum(dim=-2)

        # This is a workaround to ensure a single output from the torch.nn.Module.
        # The OpenVINO ModuleExtension mechanism has a limitation and expects
        # the module to produce only one output.
        output_cell = torch.cat([core_attn_out.flatten(), last_recurrent_state.flatten()], dim=0)
        return output_cell

