from collections import defaultdict
from typing import Dict, List, Set
import pandas as pd
from pytorch_callstack_analysis.utils.musa_basic_kernel_info import calculate_CheckpointWithoutOutputFunction, calculate_groupedlinear_tflops_or_bw, calculate_linear_tflops_or_bw, calculate_scaled_dot_product_attention_flash_musa_flops

DUP_LABEL = "@dup@"
SHAPE_LABEL = "@shape@"
ANNOTATE_LABEL = [DUP_LABEL, SHAPE_LABEL] 

SHAPE_POSITION = {
    # CheckpointWithoutOutputFunction is RMSNorm_2's parents
    "nn.Module: RMSNorm_\d+": {
        "type": "BW",
        "ShapeFrom": r"CheckpointWithoutOutputFunction",
        "formula": calculate_CheckpointWithoutOutputFunction,
    },
    "transformer_engine/pytorch/cpp_extensions/gemm.py\(\d+\): general_grouped_gemm": {
        "type": "TFLOPS",
        "ShapeFrom": r'_GroupedLinear', #  r"nn.Module: TE(Row|Column)ParallelGroupedLinear_0"
        "formula": calculate_groupedlinear_tflops_or_bw,
    },
    "transformer_engine/pytorch/cpp_extensions/gemm.py\(\d+\): general_gemm": {
        "type": "TFLOPS",
        "ShapeFrom": r"(_Linear|_LayerNormLinear|RouterGatingLinearFunction)", # _Linear
        "formula": calculate_linear_tflops_or_bw,
    },
    "aten::_scaled_dot_product_attention_flash_musa": {
        "type": "TFLOPS",
        "ShapeFrom": r"aten::_scaled_dot_product_attention_flash_musa",
        "formula": calculate_scaled_dot_product_attention_flash_musa_flops,
    },
    "LinearWithGradAccumulationAndAsyncCommunication": {
        "type": "TFLOPS",
        "ShapeFrom": r"LinearWithGradAccumulationAndAsyncCommunication", # _Linear
        "formula": calculate_linear_tflops_or_bw,
    },
    "aten::mm": {
        "type": "TFLOPS",
        "ShapeFrom": r"LinearWithGradAccumulationAndAsyncCommunicationBackward", # _Linear
        "formula": calculate_linear_tflops_or_bw,
    }
}
SHAPE_TO_FLOPS_FUNC =  {
    "transformer_engine/pytorch/cpp_extensions/gemm.py\(\d+\): general_grouped_gemm": "fused_multi_quantize",
}

SHAPE_TO_VOLUME_FUNC = {
    "nn.Module: RMSNorm_\d+": "RMSNorm forward",
}

output_template_to_file = r"""
pretrain_deepseekv2.py\(\d+\): forward_step
    pretrain_deepseekv2.py\(\d+\): get_batch
    nn.Module: LanguageModelEmbedding_0
    
    moe:
    nn.Module: TransformerLayer_0
        nn.Module: RMSNorm_0
        nn.Module: MLASelfAttention_0
        nn.Module: RMSNorm_1
        nn.Module: MoELayer_0
            nn.Module: TopKRouter_0
                megatron/core/transformer/moe/router.py\(\d+\): gating
                megatron/core/transformer/moe/router.py\(\d+\): routing
                    musa_patch/moe_utils.py\(\d+\): topk_softmax_with_capacity
                        megatron/core/transformer/moe/moe_utils.py\(\d+\): group_limited_topk
                    <built-in method softmax of type object at 0x\w+>
                    megatron/core/transformer/moe/router.py\(\d+\): apply_load_balancing_loss
            megatron/core/transformer/moe/token_dispatcher.py\(\d+\): token_permutation
                megatron/core/transformer/moe/token_dispatcher.py\(\d+\): preprocess
                megatron/core/transformer/moe/moe_utils.py\(\d+\): permute
                megatron/core/tensor_parallel/mappings.py\(\d+\): all_to_all @dup@
                    _AllToAll @dup@ @shape@
                megatron/core/transformer/moe/moe_utils.py\(\d+\): sort_chunks_by_idxs @dup@
            nn.Module: TEGroupedMLP_0
                nn.Module: TEColumnParallelGroupedLinear_0
                    transformer_engine/pytorch/module/grouped_linear.py\(\d+\): forward @dup@
                        _GroupedLinear @dup@ @shape@
                            <built-in method fused_multi_quantize of PyCapsule object at 0x\w+> @dup@
                            transformer_engine/pytorch/cpp_extensions/gemm.py\(\d+\): general_grouped_gemm @dup@ @shape@
                megatron/core/fusions/fused_bias_swiglu.py\(\d+\): bias_swiglu_impl
                nn.Module: TERowParallelGroupedLinear_0
                    transformer_engine/pytorch/module/grouped_linear.py\(\d+\): forward @dup@
                        _GroupedLinear @dup@ @shape@
                            <built-in method fused_multi_quantize of PyCapsule object at 0x\w+> @dup@
                            transformer_engine/pytorch/cpp_extensions/gemm.py\(\d+\): general_grouped_gemm @dup@ @shape@
            megatron/core/transformer/moe/token_dispatcher.py\(\d+\): token_unpermutation
                megatron/core/transformer/moe/moe_utils.py\(\d+\): sort_chunks_by_idxs @dup@
                megatron/core/tensor_parallel/mappings.py\(\d+\): all_to_all @dup@
                    _AllToAll @dup@ @shape@
                megatron/core/transformer/moe/moe_utils.py\(\d+\): unpermute
            nn.Module: SharedExpertMLP_0
        megatron/core/fusions/fused_bias_dropout.py\(\d+\): _bias_dropout_add

    nn.Module: RMSNorm_3

    loss:
    nn.Module: ColumnParallelLinear_0
    megatron/core/models/common/language_module/language_module.py\(\d+\): compute_language_model_loss
"""

output_template_to_file_dsv3 = r"""
pretrain_deepseekv2.py\(\d+\): forward_step
    pretrain_deepseekv2.py\(\d+\): get_batch
    nn.Module: LanguageModelEmbedding_0
    
    moe:
    nn.Module: TransformerLayer_0
        nn.Module: RMSNorm_0
        nn.Module: MLASelfAttention_0
        nn.Module: RMSNorm_1
        nn.Module: MoELayer_0
            megatron/core/transformer/moe/moe_layer.py\(\d+\): router_and_preprocess
            megatron/core/transformer/moe/moe_layer.py\(\d+\): dispatch
            megatron/core/transformer/moe/moe_layer.py\(\d+\): experts_compute
                nn.Module: TEGroupedMLP_0
            megatron/core/transformer/moe/moe_layer.py\(\d+\): combine
    nn.Module: RMSNorm_2
    nn.Module: ColumnParallelLinear_0
    megatron/core/models/common/language_module/language_module.py\(\d+\): compute_language_model_loss
"""

output_template_to_file_debug = r"""
nn.Module: TEDotProductAttention_0
    nn.Module: FlashAttention_0
        aten::_scaled_dot_product_attention_flash_musa @shape@
"""

output_template_to_file_kimi = r"""
pretrain_kimi.py\(\d+\): <module>
    musa_patch/training.py\(\d+\): train_step
        musa_patch/core_pipeline_parallel_schedules.py\(\d+\): forward_step
            pretrain_kimi.py\(\d+\): forward_step
                nn.Module: DistributedDataParallel_0
                    megatron/core/models/gpt/gpt_model.py\(\d+\): _preprocess
                        nn.Module: LanguageModelEmbedding_0
                    nn.Module: TransformerBlock_0
                        megatron/core/transformer/transformer_layer.py\(\d+\): __call__
                            nn.Module: TransformerLayer_\d+
                                megatron/core/transformer/transformer_layer.py\(\d+\): _forward_attention
                                    megatron/core/tensor_parallel/random.py\(\d+\): checkpoint
                                        nn.Module: RMSNorm_\d+ @shape@
                                    nn.Module: MLASelfAttention_\d+
                                        megatron/core/tensor_parallel/random.py\(\d+\): checkpoint @dup@
                                    megatron/core/fusions/fused_bias_dropout.py\(\d+\): _bias_dropout_add @dup@
                                megatron/core/transformer/transformer_layer.py\(\d+\): _forward_mlp
                                    nn.Module: MLP_\d+
                                        nn.Module: TELayerNormColumnParallelLinear_\d+
                                        megatron/core/fusions/fused_bias_swiglu.py\(\d+\): bias_swiglu_impl
                                        nn.Module: TERowParallelLinear_1
                                    megatron/core/tensor_parallel/random.py\(\d+\): checkpoint @dup@
                                    nn.Module: MoELayer_0
                                        megatron/core/transformer/moe/moe_layer.py\(\d+\): custom_forward
                                            megatron/core/transformer/moe/moe_layer.py\(\d+\): router_and_preprocess
                                                nn.Module: TopKRouter_0
                                                megatron/core/transformer/moe/token_dispatcher.py: dispatch_preprocess
                                            megatron/core/transformer/moe/moe_layer.py\(\d+\): dispatch
                                            megatron/core/transformer/moe/moe_layer.py\(\d+\): experts_compute
                                                nn.Module: SharedExpertMLP_0
                                                megatron/core/transformer/moe/token_dispatcher.py\(\d+\): dispatch_postprocess
                                                nn.Module: TEGroupedMLP_0
                                                    nn.Module: TEColumnParallelGroupedLinear_0
                                                        transformer_engine/pytorch/cpp_extensions/gemm.py\(\d+\): general_grouped_gemm @dup@ @shape@
                                                    megatron/core/transformer/moe/experts.py\(\d+\): bias_act_func
                                                    nn.Module: TERowParallelGroupedLinear_0
                                                        transformer_engine/pytorch/cpp_extensions/gemm.py\(\d+\): general_grouped_gemm @dup@ @shape@
                                                megatron/core/transformer/moe/token_dispatcher.py\(\d+\): combine_preprocess
                                            megatron/core/transformer/moe/moe_layer.py\(\d+\): combine
                                    megatron/core/fusions/fused_bias_dropout.py\(\d+\): _bias_dropout_add @dup@
                    megatron/core/models/gpt/gpt_model.py\(\d+\): _postprocess
                        nn.Module: ColumnParallelLinear_0
                        megatron/core/models/common/language_module/language_module.py\(\d+\): compute_language_model_loss
            megatron/core/pipeline_parallel/schedules.py\(\d+\): forward_step_calc_loss
        megatron/core/pipeline_parallel/p2p_communication.py\(\d+\): send_backward_recv_forward
        megatron/core/pipeline_parallel/p2p_communication.py\(\d+\): send_forward_recv_backward
        megatron/core/pipeline_parallel/p2p_communication.py\(\d+\): send_forward
        megatron/core/pipeline_parallel/p2p_communication.py\(\d+\): send_backward
        megatron/core/pipeline_parallel/p2p_communication.py\(\d+\): recv_forward
        megatron/core/pipeline_parallel/p2p_communication.py\(\d+\): recv_backward
"""

output_template_to_file_kimi_epoverlap = r"""
pretrain_kimi.py\(\d+\): <module>
    musa_patch/training.py\(\d+\): train_step
        megatron/core/pipeline_parallel/combined_1f1b.py\(\d+\): combined_forward_backward_step
            megatron/core/models/common/model_chunk_schedule_plan.py\(\d+\): run
                megatron/core/models/gpt/fine_grained_callables.py\(\d+\): submodule_attn_forward
                    megatron/core/transformer/transformer_layer.py\(\d+\): _forward_attention
                        megatron/core/tensor_parallel/random.py\(\d+\): checkpoint @dup@
                            nn.Module: RMSNorm_0 @shape@
                        nn.Module: MLASelfAttention_\d+
                        megatron/core/fusions/fused_bias_dropout.py\(\d+\): _bias_dropout_add @dup@
                megatron/core/models/gpt/fine_grained_callables.py\(\d+\): submodule_post_attn_forward
                    megatron/core/tensor_parallel/random.py\(\d+\): checkpoint @dup@
                        nn.Module: RMSNorm_1 @shape@
                    megatron/core/transformer/moe/moe_layer.py\(\d+\): router_and_preprocess
                        nn.Module: TopKRouter_0
                        megatron/core/transformer/moe/token_dispatcher.py\(\d+\): dispatch_preprocess
                megatron/core/models/gpt/fine_grained_callables.py\(\d+\): submodule_dispatch_forward
                    FusedDispatch
                megatron/core/models/gpt/fine_grained_callables.py\(\d+\): submodule_moe_forward
                    megatron/core/transformer/moe/moe_layer.py\(\d+\): experts_compute
                        nn.Module: SharedExpertMLP_0
                        megatron/core/transformer/moe/token_dispatcher.py\(\d+\): dispatch_postprocess
                        nn.Module: TEGroupedMLP_0
                            nn.Module: TEColumnParallelGroupedLinear_0
                                transformer_engine/pytorch/cpp_extensions/gemm.py\(\d+\): general_grouped_gemm @dup@ @shape@
                            megatron/core/tensor_parallel/random.py\(\d+\): checkpoint @dup@
                            nn.Module: TERowParallelGroupedLinear_0
                                transformer_engine/pytorch/cpp_extensions/gemm.py\(\d+\): general_grouped_gemm @dup@ @shape@
                        megatron/core/transformer/moe/token_dispatcher.py\(\d+\): combine_preprocess
                        transformer_engine/pytorch/cpu_offload.py\(\d+\): wait_offload
                megatron/core/models/gpt/fine_grained_callables.py\(\d+\): submodule_combine_forward
                    megatron/core/transformer/moe/moe_layer.py\(\d+\): combine
                    megatron/core/fusions/fused_bias_dropout.py\(\d+\): _bias_dropout_add @dup@
            megatron/core/pipeline_parallel/schedules.py\(\d+\): forward_step_calc_loss
        megatron/core/pipeline_parallel/p2p_communication.py\(\d+\): send_backward_recv_forward
        megatron/core/pipeline_parallel/p2p_communication.py\(\d+\): send_forward_recv_backward
        megatron/core/pipeline_parallel/p2p_communication.py\(\d+\): send_forward
        megatron/core/pipeline_parallel/p2p_communication.py\(\d+\): send_backward
        megatron/core/pipeline_parallel/p2p_communication.py\(\d+\): recv_forward
        megatron/core/pipeline_parallel/p2p_communication.py\(\d+\): recv_backward
"""

# Function to count leading spaces or tabs
def count_leading_whitespace(line):
    count = 0
    for char in line:
        if char == ' ' or char == '\t':
            count += 1
        else:
            break
    return count

# transform call stack to list
def call_stack_with_ancestors_to_list(stack, call_stack_list, level=0):
    for entry in stack:
        call_stack_list.append((entry['name'], entry['ancestors']))
        call_stack_with_ancestors_to_list(entry['children'], call_stack_list, level + 1)

def extract_dup_or_shape_func_name_from_template(output_template: str) -> (Set[str], Set[str]):
    # Split the output_template by newlines and strip leading/trailing whitespace
    lines = [line for line in output_template.splitlines() if line.strip() and line.strip() not in ['dense:', 'moe:', 'loss:'] ]

    dup_func_name = set()
    need_shape_func_name = set()
    for line in lines:
        Has_DUP_LABEL = False
        if DUP_LABEL in line:
            line = line.replace(DUP_LABEL, '')
            Has_DUP_LABEL = True

        Has_SHAPE_LABEL = False
        if SHAPE_LABEL in line:
            line = line.replace(SHAPE_LABEL, '')
            Has_SHAPE_LABEL = True

        func_name = line.strip()
        if Has_DUP_LABEL:
            dup_func_name.add(func_name)
        if Has_SHAPE_LABEL:
            need_shape_func_name.add(func_name)
    return dup_func_name, need_shape_func_name

def extract_func_name_from_template(output_template: str) -> List[str]:
    # Split the output_template by newlines and strip leading/trailing whitespace
    lines = [line for line in output_template.splitlines() if line.strip() and line.strip() not in ['dense:', 'moe:', 'loss:'] ]

    dup_func_name: Dict[str, int] = defaultdict(int)
    # Parse the call stack
    call_stack = []
    stack = []

    for line in lines:
        indent_level = count_leading_whitespace(line)
        for annotate in ANNOTATE_LABEL:
            if annotate in line:
                line = line.replace(annotate, '')
        func_name = line.strip()+'@'+ str(dup_func_name[line.strip()])
        dup_func_name[line.strip()] += 1
        entry = {'name': func_name, 'children': [], 'ancestors': []}

        while stack and stack[-1]['indent'] >= indent_level:
            stack.pop()

        if stack:
            stack[-1]['node']['children'].append(entry)
            entry['ancestors'] = stack[-1]['node']['ancestors'] + [stack[-1]['node']['name']]
        else:
            call_stack.append(entry)

        stack.append({'node': entry, 'indent': indent_level})
    # Print the parsed call stack with ancestors
    call_stack_list = []
    call_stack_with_ancestors_to_list(call_stack, call_stack_list)
    return call_stack_list

# Function to print the call stack with ancestors
def print_call_stack_with_ancestors(stack, level=0):
    for forward_func_name, func_ancestors in stack:
        print('    ' * level + forward_func_name)
        print('    ' * (level + 1) + 'Ancestors: ' + ' -> '.join(func_ancestors))
        # print_call_stack_with_ancestors(entry['children'], level + 1)
    
def set_pandas_display_options():
    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_colwidth", None)
    pd.set_option("display.width", None)
    pd.set_option("display.float_format", "{:.2f}".format)


if __name__ == '__main__':
    print(extract_func_name_from_template(output_template_to_file))
    dup_func_name, need_shape_func_name = extract_dup_or_shape_func_name_from_template(output_template_to_file)
    print(dup_func_name)
    print(need_shape_func_name)