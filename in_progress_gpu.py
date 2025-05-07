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

@cuda.jit('void(uint8[:], int64, uint32[:,:])') # Explicit type signature
def pad_kernel(byte_in, nbytes, words_out):
    widx = cuda.grid(1)
    total = words_out.size # words_out.size is total number of elements
    if widx >= total: # Each thread computes one word
        return

    base = widx * 4
    word = uint32(0)

    for i in range(4): # Iterate through 4 bytes to form one 32-bit word
        b_ofs = base + i
        shift = (3 - i) * 8 # Big-endian: byte 0 is MSB
        if b_ofs < nbytes:                        # If byte is part of original data
            word |= uint32(byte_in[b_ofs]) << shift # Access scalar, cast to uint32, shift, and OR
        elif b_ofs == nbytes:                     # If byte is the 0x80 padding marker
            word |= uint32(0x80) << shift

    # SHA-256 padding: last 64 bits (2 words) are the message length in bits
    # The widx here corresponds to the absolute word index in the entire padded message.
    if widx == total - 2:                         # Second to last word (big-endian length, high bits)
        # Assuming message length fits in 64 bits, high 32 bits of length are 0 for practical purposes
        word = uint32(0)
    elif widx == total - 1:                       # Last word (big-endian length, low bits)
        word = uint32(nbytes * 8)                 # Length of original message in bits

    # Store the constructed word into the correct block and position within that block
    words_out[widx // 16, widx % 16] = word

# ───────────────── SHA‑256 compression kernel (unchanged) ──────────────
@cuda.jit
def sha256_kernel(words, digests):
    i = cuda.grid(1)
    if i >= words.shape[0]:                     # one thread per 512‑bit block
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
        tmp2 = (S0+maj)&0xffffffff
        h,g,f,e,d,c,b,a = g,f,e,(d+tmp1)&0xffffffff, c,b,a,(tmp1+tmp2)&0xffffffff
    digests[i,0]=(0x6a09e667+a)&0xffffffff; digests[i,1]=(0xbb67ae85+b)&0xffffffff
    digests[i,2]=(0x3c6ef372+c)&0xffffffff; digests[i,3]=(0xa54ff53a+d)&0xffffffff
    digests[i,4]=(0x510e527f+e)&0xffffffff; digests[i,5]=(0x9b05688c+f)&0xffffffff
    digests[i,6]=(0x1f83d9ab+g)&0xffffffff; digests[i,7]=(0x5be0cd19+h)&0xffffffff

# ───────────────── GPU SHA‑256 helper ──────────────────────────────────
def gpu_sha256(t):
    """Pure-GPU SHA-256 of any contiguous torch CUDA tensor."""
    t_flat  = t.contiguous().view(torch.uint8)                     # still on GPU
    nbytes  = t_flat.numel()
    nblocks = (nbytes + 63) // 64

    # device views (no copy) via DLPack
    d_bytes = cuda.as_cuda_array(t_flat)
    d_words = cuda.device_array((nblocks,16), dtype=np.uint32)
    pad_kernel[(nblocks*16 + 127)//128, 128](d_bytes, nbytes, d_words)

    d_digests = cuda.device_array((nblocks,8), dtype=np.uint32)
    sha256_kernel[(nblocks + 127)//128, 128](d_words, d_digests)

    digests = d_digests.copy_to_host()          # tiny (≤ 256 B)

    if nblocks == 1:                            # single block → direct digest
        return ''.join(f'{x:08x}' for x in digests[0])
    return hashlib.sha256(digests.tobytes()).hexdigest()  # one‑level Merkle

# ───────────────── toy POMM round ──────────────────────────────────────
A = torch.randn(4,4, dtype=torch.float32, device='cuda')
B = torch.randn(4,4, dtype=torch.float32, device='cuda')
C = A @ B

log = [(0, gpu_sha256(A), gpu_sha256(B), gpu_sha256(C))]
print("Prover log:", log[0])

_, hA, hB, hC = log[0]
A_cpu, B_cpu, C_cpu = A.cpu(), B.cpu(), C.cpu()

print("digests match:",
      hashlib.sha256(A_cpu.numpy().tobytes()).hexdigest() == hA and
      hashlib.sha256(B_cpu.numpy().tobytes()).hexdigest() == hB and
      hashlib.sha256(C_cpu.numpy().tobytes()).hexdigest() == hC)


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
