import torch
import time

# Check device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on: {device}")

# Simulate a Covariance Matrix (N x N)
N = 10000  # 10k points (usually kills a CPU)
print(f"Creating {N}x{N} matrix...")

# Create a random positive-definite matrix (like a kernel matrix)
A = torch.randn(N, N, device=device)
K = A @ A.T + 1e-3 * torch.eye(N, device=device)  # K = A*A' + nugget

# Vector y
y = torch.randn(N, 1, device=device)

# Benchmark Cholesky Decomposition (The bottleneck of GPs)
print("Starting Cholesky decomposition...")
torch.cuda.synchronize()
start = time.time()

L = torch.linalg.cholesky(K)
alpha = torch.cholesky_solve(y, L)

torch.cuda.synchronize()
end = time.time()

print(f"Done! Time taken: {end - start:.4f} seconds")