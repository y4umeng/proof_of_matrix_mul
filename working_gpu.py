"""
Tiny POMM round-trip
GPU: one 4x4 FP16 matmul + SHA-256 digests via Numba-CUDA
CPU verifier: recompute & hash, compare
"""

import hashlib, random, numpy as np, torch, torch.utils.dlpack as dlpack
from numba import cuda, uint32

# ── SHA‑256 constants ──────────────────────────────────────────────────
K = np.array([
    0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
    0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
    0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
    0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
    0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
    0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
    0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
    0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2
], dtype=np.uint32)

# ── SHA‑256 kernel (1 thread ↔ 1 block) ────────────────────────────────
@cuda.jit(device=True, inline=True)
def _rotr(x, n):  # rotate right
    return ((x >> n) | (x << (32 - n))) & 0xffffffff

# # don't really need to understand this
# # it's just the implementation of SHA-256
@cuda.jit
def sha256_kernel(words, digests):
    i = cuda.grid(1)
    if i >= words.shape[0]:
        return
    w = cuda.local.array(64, uint32)
    for t in range(16):
        w[t] = words[i, t]
    for t in range(16, 64):
        s0 = _rotr(w[t-15],7)^_rotr(w[t-15],18)^(w[t-15]>>3)
        s1 = _rotr(w[t- 2],17)^_rotr(w[t- 2],19)^(w[t- 2]>>10)
        w[t] = (w[t-16]+s0+w[t-7]+s1)&0xffffffff
    a,b,c,d,e,f,g,h = (
        0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,
        0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19)
    for t in range(64):
        S1 = _rotr(e,6)^_rotr(e,11)^_rotr(e,25)
        ch = (e&f)^((~e)&g)
        tmp1 = (h+S1+ch+K[t]+w[t]) & 0xffffffff
        S0 = _rotr(a,2)^_rotr(a,13)^_rotr(a,22)
        maj = (a&b)^(a&c)^(b&c)
        tmp2 = (S0+maj) & 0xffffffff
        h,g,f,e,d,c,b,a = g,f,e,(d+tmp1)&0xffffffff, c,b,a,(tmp1+tmp2)&0xffffffff
    digests[i,0]=(0x6a09e667+a)&0xffffffff; digests[i,1]=(0xbb67ae85+b)&0xffffffff
    digests[i,2]=(0x3c6ef372+c)&0xffffffff; digests[i,3]=(0xa54ff53a+d)&0xffffffff
    digests[i,4]=(0x510e527f+e)&0xffffffff; digests[i,5]=(0x9b05688c+f)&0xffffffff
    digests[i,6]=(0x1f83d9ab+g)&0xffffffff; digests[i,7]=(0x5be0cd19+h)&0xffffffff

def gpu_sha256(t):
    """Return SHA-256(t) - no extra Merkle root when nblocks == 1."""
    byte_flat = t.contiguous().view(torch.uint8).flatten().cpu()
    nblocks = (byte_flat.numel() + 64) // 64  # Ensure at least one extra block. don't know wtf this does
    padded    = torch.zeros(nblocks*64, dtype=torch.uint8)
    padded[: byte_flat.numel()] = byte_flat
    padded[byte_flat.numel()]   = 0x80
    bit_len = (byte_flat.numel()*8).to_bytes(8,'big')
    padded[-8:] = torch.tensor(list(bit_len), dtype=torch.uint8)

    words  = np.frombuffer(padded.numpy(), dtype='>u4').astype(np.uint32).reshape(nblocks,16)
    d_out  = cuda.device_array((nblocks,8), dtype=np.uint32)
    sha256_kernel[((nblocks+127)//128), 128](cuda.to_device(words), d_out)

    digests = d_out.copy_to_host()

    if nblocks == 1:                       # <<< skip the extra hash
        return ''.join(f'{x:08x}' for x in digests[0])
    else:
        return hashlib.sha256(digests.tobytes()).hexdigest()



# ── prover: one multiply, three GPU hashes ─────────────────────────────
A = torch.randn(1000,5000, dtype=torch.float16, device='cuda')
B = torch.randn(5000,1000, dtype=torch.float16, device='cuda')
C = torch.matmul(A, B)

log = [(0, gpu_sha256(A), gpu_sha256(B), gpu_sha256(C))]
print("Prover log:", log[0])

# ── verifier: CPU recompute & hash check ───────────────────────────────
_, shaA, shaB, shaC = log[0]
A_cpu, B_cpu, C_cpu = A.cpu(), B.cpu(), C.cpu()
verified = (
    hashlib.sha256(A_cpu.numpy().tobytes()).hexdigest() == shaA and
    hashlib.sha256(B_cpu.numpy().tobytes()).hexdigest() == shaB and
    hashlib.sha256(C_cpu.numpy().tobytes()).hexdigest() == shaC and
    torch.allclose(C_cpu, A_cpu @ B_cpu, atol=1e-3) # ERROR TOLERANCE VALUES
)
print("Verifier all-clear:", verified)


# ── verifier: digests first, then matmul accuracy ──────────────────────
sha_ok = (
    hashlib.sha256(A_cpu.numpy().tobytes()).hexdigest() == shaA and
    hashlib.sha256(B_cpu.numpy().tobytes()).hexdigest() == shaB and
    hashlib.sha256(C_cpu.numpy().tobytes()).hexdigest() == shaC
)

# float16 matmul can differ by 1–2 × 10‑2 on the CPU
matmul_ok = torch.allclose(
    C_cpu, (A_cpu @ B_cpu).to(dtype=C_cpu.dtype),
    atol=2e-2, rtol=2e-2
)

print(f"digests match: {sha_ok}   matmul match: {matmul_ok}")

import time

# Function to time operations with proper CUDA synchronization
def timeit(fn, *args, iters=50, **kwargs):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        result = fn(*args, **kwargs)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000 / iters, result

# Define matrix sizes for testing
sizes = [(100, 100, 100), (1000, 1000, 1000)]

print("Timing comparison: regular matmul vs. matmul with SHA-256 hashing")
print("=" * 70)

for m, k, n in sizes:
    print(f"\nMatrix sizes: A({m}×{k}), B({k}×{n})")
    
    # Create matrices
    A = torch.randn(m, k, dtype=torch.float16, device='cuda')
    B = torch.randn(k, n, dtype=torch.float16, device='cuda')
    
    # Time regular matrix multiplication
    reg_time, C = timeit(torch.matmul, A, B, iters=10)
    
    # Time matrix multiplication with hashing
    def matmul_with_hash():
        C = torch.matmul(A, B)
        _ = gpu_sha256(A)
        _ = gpu_sha256(B)
        _ = gpu_sha256(C)
        return C
    
    hash_time, _ = timeit(matmul_with_hash, iters=10)
    
    # Print results
    print(f"Regular matmul:       {reg_time:.3f} ms")
    print(f"Matmul with hashing:  {hash_time:.3f} ms")
    print(f"Overhead:             {(hash_time-reg_time)/reg_time*100:.1f}%")
