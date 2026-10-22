"""
第一个练习：三个最小实验，建立手感。

运行方式：
    python 01-first-steps.py

不用先看视频。先跑完，再回去看讲解，会顺很多。
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # 某些环境里 numpy 和 torch 各带一份 OpenMP，同进程加载会直接退出

import time
import torch


print("=" * 52)
print("Part 1  一个完整的训练循环")
print("=" * 52)

torch.manual_seed(0)
x = torch.randn(1000, 1)
y = 3 * x + 2 + 0.1 * torch.randn(1000, 1)   # 真实规律是 y = 3x + 2

model = torch.nn.Linear(1, 1)
loss_fn = torch.nn.MSELoss()
opt = torch.optim.SGD(model.parameters(), lr=0.1)

print(f"初始参数: w={model.weight.item():.3f}  b={model.bias.item():.3f}")

for step in range(200):
    pred = model(x)          # 前向：拿当前参数算预测
    loss = loss_fn(pred, y)  # 算预测和真实的差距
    opt.zero_grad()          # 清掉上一轮的梯度
    loss.backward()          # 反向：算出每个参数该往哪调
    opt.step()               # 按梯度更新参数
    if step % 40 == 0:
        print(f"  step {step:3d}   loss {loss.item():.4f}")

print(f"训练后:   w={model.weight.item():.3f}  b={model.bias.item():.3f}")
print("应该接近 w=3, b=2")


print()
print("=" * 52)
print("Part 2  同样的矩阵乘法，CPU 和 GPU 差多少")
print("=" * 52)

if not torch.cuda.is_available():
    print("没检测到可用的 CUDA 设备，Part 2 和 Part 3 跳过。")
    print("这两部分是为了看 GPU 和 CPU 的差异，需要在有 NVIDIA 显卡的机器上跑。")
    raise SystemExit(0)

n = 4096
a = torch.randn(n, n)
b = torch.randn(n, n)

t0 = time.perf_counter()
a @ b
cpu_ms = (time.perf_counter() - t0) * 1000
print(f"CPU FP32:      {cpu_ms:.1f} ms")

a_gpu, b_gpu = a.cuda(), b.cuda()
a_gpu @ b_gpu                      # 先跑一次，把初始化开销排除掉
torch.cuda.synchronize()
t0 = time.perf_counter()
a_gpu @ b_gpu
torch.cuda.synchronize()           # 等 GPU 真的算完，不然计时是假的
gpu_ms = (time.perf_counter() - t0) * 1000
print(f"GPU FP32:      {gpu_ms:.1f} ms")

a16, b16 = a.cuda().half(), b.cuda().half()
a16 @ b16
torch.cuda.synchronize()
t0 = time.perf_counter()
a16 @ b16
torch.cuda.synchronize()
fp16_ms = (time.perf_counter() - t0) * 1000
print(f"GPU FP16:      {fp16_ms:.1f} ms")

print()
print(f"GPU 相对 CPU（同为 FP32）: {cpu_ms / gpu_ms:.1f} 倍")
print(f"GPU 换半精度相对 FP32:     {gpu_ms / fp16_ms:.1f} 倍")
print("FP16 那行走的是 Tensor Core。这个单元只认低精度输入、只做矩阵乘，")
print("算力远高于普通运算单元，所以同一个模型换个精度就能快出好几倍。")



print()
print("=" * 52)
print("Part 3  张量变小以后，GPU 的优势去哪了")
print("=" * 52)

small = torch.randn(16, 16)
small_gpu = small.cuda()
N = 10000

t0 = time.perf_counter()
for _ in range(N):
    small @ small
small_cpu_ms = (time.perf_counter() - t0) * 1000

small_gpu @ small_gpu
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(N):
    small_gpu @ small_gpu
torch.cuda.synchronize()
small_gpu_ms = (time.perf_counter() - t0) * 1000

print(f"16x16 矩阵乘 {N} 次")
print(f"CPU: {small_cpu_ms:.0f} ms")
print(f"GPU: {small_gpu_ms:.0f} ms")
print()
print("对照 Part 2 的加速比看这个结果。")
print("如果差距远没有那么大，甚至 GPU 更慢，那就是 kernel 启动的固定开销吃掉了收益。")
print("运算规模越小，这笔固定开销摊得越薄，GPU 越不划算。")
