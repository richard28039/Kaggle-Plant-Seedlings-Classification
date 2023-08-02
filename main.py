import torch
device = torch.device("cuda:0")
print(torch.cuda.get_device_properties(device))
