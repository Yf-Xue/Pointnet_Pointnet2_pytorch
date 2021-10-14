import torch
import numpy 
import time

t0 = time.time()
matrix1 = torch.randn(10000, 1000)
matrix2 = torch.randn(1000, 10000)
device = torch.device('cuda')
matrix1 = matrix1.to(device)
matrix2 = matrix2.to(device)
t1 = time.time()

while t1 - t0 < 90:
    for i in range(100):
        c = torch.matmul(matrix1, matrix2)
    t1 = time.time()

print(matrix1.device, matrix2.device)