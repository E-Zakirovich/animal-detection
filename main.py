from model.cnn import CNN
import torch

model = CNN()
dummy = torch.randn(32, 3, 128, 128)
out = model(dummy)
print(out.shape)