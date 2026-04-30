print('Hello world!')

import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
test_tensor = torch.tensor([0., 1., 0.], device=device)

print(test_tensor)
print(test_tensor.device)
