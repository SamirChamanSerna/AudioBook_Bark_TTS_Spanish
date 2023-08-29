import torch

torch.set_default_tensor_type(torch.cuda.FloatTensor)
print(torch.rand(10).device)
