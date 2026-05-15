from functools import reduce
import operator

BYTES_DICT = {
    'c10::Half': 2.0,
    'c10::Float': 4.0,
    'c10::Int': 1.0,
    'c10::BFloat16': 2.0,
    'long int': 8.0,
    'float': 4.0,
    'unsigned char': 1.0,
}

def drop_empty_arrays(arr):
    """去掉数组中的空数组"""
    return [x for x in arr if x]

def _prod(shape):
    return reduce(operator.mul, shape, 1)

def _cal_macs(m, n, k):
    macs = m * n * k * 2
    return macs

def _cal_comm_volume(m, n, k, input_type, cal_phrase):
    volume = 0.0
    if cal_phrase == 'fwd-0':
        volume = m * k * 2 * get_num_of_bytes(input_type) # read and write
    elif cal_phrase == 'bwd-0':
        volume = m * n * 2 * get_num_of_bytes(input_type) # read and write
    elif cal_phrase == 'bwd-1':
        volume = m * k * 2 * get_num_of_bytes(input_type) # read and write
    return volume

def _cal_shape(M, N, K, cal_phrase):
    if cal_phrase =='bwd-0':
        return [[M, N], [N, K]]
    elif cal_phrase == 'bwd-1':
        return [[M, N], [M, K]]
    else:
        return [[M, K], [N, K]]
    
# kernel span(unit: s)
def calculate_groupedlinear_tflops_or_bw(input_dims, input_type, kernel_span, shape_from_func, calculate_type, cal_phrase):
    if len(input_dims) < 14:
        if "TERowParallelGroupedLinear_0" in shape_from_func: # corresponding to grouped gemm 1
            input_dims = [[32768, 2048], [], [], [], [], [], [], [], [], [], [], [], [], [7168, 2048]]
        elif "TEColumnParallelGroupedLinear_0" in shape_from_func: # corresponding to grouped gemm 0 
            input_dims = [[32768, 7168], [], [], [], [], [], [], [], [], [], [], [], [], [4096, 7168]]
        else:
            return 0
    dims = drop_empty_arrays(input_dims)
    #print(f'cal_phrase: {cal_phrase}, shape_from_func: {shape_from_func} \n dims:{dims}')
    M = dims[0][0]
    N, K = dims[1][0], dims[1][1]
    #print(f'M: {M}, N: {N}, K:{K}')
    if calculate_type.upper() == "TFLOPS":
        return _cal_shape(M, N, K, cal_phrase), M * N * K * 2/ 1e12 / kernel_span
    elif calculate_type.upper() == "GB/S":
        volume = _cal_comm_volume(M, N, K, input_type, cal_phrase)
        return _cal_shape(M, N, K, cal_phrase), volume / (1024.0 ** 3) / kernel_span # GB
    return [[M, K], [N, K]], 0.0

def calculate_linear_tflops_or_bw(input_dims, input_type, kernel_span, shape_from_func, calculate_type, cal_phrase):
    # print(f"input_dims {input_dims}, shape_from_func: {shape_from_func}")
    dims = drop_empty_arrays(input_dims)
    M, N, K = 0, 0, 0
    if shape_from_func.startswith("_LayerNormLinear"):
        M = dims[0][0]
        N, K = dims[2][0], dims[2][1]
    elif shape_from_func.startswith("_Linear"):
        M, K = dims[1][0], dims[1][2]
        N = dims[0][0]
    elif shape_from_func.startswith("RouterGatingLinearFunction") or shape_from_func.startswith("LinearWithGradAccumulationAndAsyncCommunication"):
        M, K = dims[0][0], dims[0][2]
        N = dims[1][0]
    if calculate_type.upper() == "TFLOPS":
        return _cal_shape(M, N, K, cal_phrase), M * N * K * 2 / 1e12 / kernel_span
    elif calculate_type.upper() == "GB/S":
        volume = _cal_comm_volume(M, N, K, input_type, cal_phrase)
        return _cal_shape(M, N, K, cal_phrase), volume / (1024 ** 3) / kernel_span # GB
    return [[M, K], [N, K]], 0.0


def calculate_scaled_dot_product_attention_flash_musa_flops(input_dims, input_type, kernel_span, shape_from_func, calculate_type, cal_phrase):
    dims = drop_empty_arrays(input_dims)
    # Todo: 20260127 only one flash attention gemm kernel
    # print(f'func name: {shape_from_func}, dims: {dims[:3]}')
    batch_size = dims[0][0]
    nheads = dims[0][1]
    seq_len = dims[0][2]
    macs = 0
    if cal_phrase == 'fwd-0':
        qk_head_dim = dims[0][3]
        v_head_dim = dims[2][3]
        #[[batch=1, nheads=64, seq_q_len=4096, qk_head_dim=192], [1, 64, seq_kv_len 4096, qk_head_dim=192], [1, 64, 4096, v_head_dim=128]]
        macs = batch_size * nheads * (seq_len**2) * (qk_head_dim + v_head_dim)
    elif cal_phrase == 'bwd-0':
        qk_head_dim = dims[2][3]
        v_head_dim = dims[0][3]
        # [[1, 64, 4096, 128], [1, 64, 4096, 192], [1, 64, 4096, 192]]
        macs = batch_size * nheads * (seq_len**2) * (3*qk_head_dim + 2*v_head_dim)
    return dims[:3], macs / 1e12 / kernel_span
   


def get_num_of_bytes(input_type):
    if input_type not in BYTES_DICT: 
        print(f'type {input_type} not in BYTES_DICT, is set to 1B')
    return BYTES_DICT.get(input_type, 1.0)

def calculate_CheckpointWithoutOutputFunction(input_dims, input_type):
    volume = _prod(input_dims[0]) * get_num_of_bytes(input_type) * 2 # read and write
    dims = input_dims[:1]
    return dims, volume / (1024 ** 4)  # TB
