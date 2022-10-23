import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.allow_tf32 = True
data = torch.randn([16, 3, 1280, 1280], dtype=torch.half, device='cuda', requires_grad=True)
net = torch.nn.Conv2d(3, 32, kernel_size=[3, 3], padding=[1, 1], stride=[1, 1], dilation=[1, 1], groups=1)
net = net.cuda().half()
out = net(data)
out.backward(torch.randn_like(out))
torch.cuda.synchronize()
"""
ConvolutionParams
    data_type = CUDNN_DATA_HALF
    padding = [1, 1, 0]
    stride = [1, 1, 0]
    dilation = [1, 1, 0]
    groups = 1
    deterministic = false
    allow_tf32 = true
input: TensorDescriptor 0000021303B48670
    type = CUDNN_DATA_HALF
    nbDims = 4
    dimA = 16, 3, 1280, 1280,
    strideA = 4915200, 1638400, 1280, 1,
output: TensorDescriptor 0000021303B48750
    type = CUDNN_DATA_HALF
    nbDims = 4
    dimA = 16, 32, 1280, 1280,
    strideA = 52428800, 1638400, 1280, 1,
weight: FilterDescriptor 00000212F46C3B80
    type = CUDNN_DATA_HALF
    tensor_format = CUDNN_TENSOR_NCHW
    nbDims = 4
    dimA = 32, 3, 3, 3,
Pointer addresses:
    input: 000000093D400000
    output: 0000000962C00000
    weight: 00000009049FF200"""
