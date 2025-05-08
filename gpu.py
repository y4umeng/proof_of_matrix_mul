import contextlib
import torch
import random
import time
import types
import hashlib
import numpy as np
import torch.nn as nn
from functools import wraps
from numba import cuda, uint64, uint8 
from tabulate import tabulate 

# --- FNV-1a Hashing Constants ---
FNV_OFFSET_BASIS_64 = np.uint64(0xcbf29ce484222325)
FNV_PRIME_64 = np.uint64(0x00000100000001B3)

# --- Numba CUDA Kernel for FNV-1a Hash ---
@cuda.jit
def gpu_fnv1a_hash_kernel(data_ptr, data_len, out_hash_ptr):
    hval = uint64(FNV_OFFSET_BASIS_64)
    for i in range(data_len):
        byte_as_uint64 = uint64(data_ptr[i]) 
        hval = hval ^ byte_as_uint64
        hval = hval * uint64(FNV_PRIME_64)
    out_hash_ptr[0] = hval

# --- Python Wrapper for GPU Fast Hash ---
_gpu_fast_hash_data_path_printed_once = False
_internal_debug_prints_enabled = False # Set to True for verbose internal timings

def gpu_fast_hash(t_gpu: torch.Tensor):
    global _gpu_fast_hash_data_path_printed_once, _internal_debug_prints_enabled
    
    if not t_gpu.is_cuda:
        raise TypeError("gpu_fast_hash expects a CUDA tensor.")

    if _internal_debug_prints_enabled: print(f"DEBUG gpu_fast_hash: Called for tensor shape {t_gpu.shape}, dtype {t_gpu.dtype}")
    
    # Stage 1: Tensor Preparation
    if torch.cuda.is_available(): torch.cuda.synchronize()
    t_prep_start = time.perf_counter()
    
    gpu_byte_view_flat = t_gpu.contiguous().view(torch.uint8).flatten()
    data_len_bytes = gpu_byte_view_flat.numel()

    if torch.cuda.is_available(): torch.cuda.synchronize()
    t_prep_end = time.perf_counter()
    prep_time_ms = (t_prep_end - t_prep_start) * 1000
    if _internal_debug_prints_enabled: print(f"DEBUG gpu_fast_hash: Prep (contiguous, view, flatten) took: {prep_time_ms:.4f} ms")

    if data_len_bytes == 0:
        if _internal_debug_prints_enabled: print(f"DEBUG gpu_fast_hash: Empty tensor, returning FNV offset basis.")
        return format(FNV_OFFSET_BASIS_64, '016x') 

    # Stage 2: CuPy/Numba Bridging
    if torch.cuda.is_available(): torch.cuda.synchronize()
    t_bridge_start = time.perf_counter()
    data_gpu_numba = None
    try:
        import cupy
        cupy_array = cupy.asarray(gpu_byte_view_flat) # PyTorch GPU tensor to CuPy array
        data_gpu_numba = cuda.as_cuda_array(cupy_array) # CuPy array to Numba device array view
        if not _gpu_fast_hash_data_path_printed_once:
            # print("INFO: gpu_fast_hash using CuPy path.")
            _gpu_fast_hash_data_path_printed_once = True
    except ImportError:
        if not _gpu_fast_hash_data_path_printed_once:
            # print("INFO: CuPy not found. gpu_fast_hash attempting direct PyTorch tensor to Numba.")
            _gpu_fast_hash_data_path_printed_once = True
        # Fallback to direct PyTorch -> Numba if CuPy is not installed
        try:
            data_gpu_numba = cuda.as_cuda_array(gpu_byte_view_flat)
        except TypeError as e_direct:
            raise RuntimeError(f"Direct PyTorch->Numba tensor conversion (cuda.as_cuda_array) failed: {e_direct}. Install CuPy or check Numba/PyTorch compatibility.")
        except Exception as e_other_direct:
             raise RuntimeError(f"An unexpected error occurred during direct PyTorch->Numba conversion: {e_other_direct}")
    except Exception as e_cupy_general:
        raise RuntimeError(f"CuPy path failed: {e_cupy_general}. Ensure CuPy is installed correctly for your CUDA version.")

    if data_gpu_numba is None: # Should not happen if exceptions are raised correctly
        raise RuntimeError("Failed to convert PyTorch tensor to Numba device array.")

    if torch.cuda.is_available(): torch.cuda.synchronize()
    t_bridge_end = time.perf_counter()
    bridge_time_ms = (t_bridge_end - t_bridge_start) * 1000
    if _internal_debug_prints_enabled: print(f"DEBUG gpu_fast_hash: Bridge (CuPy/Numba) took: {bridge_time_ms:.4f} ms")

    # Stage 3: Kernel Execution & Result Copy
    if torch.cuda.is_available(): torch.cuda.synchronize()
    t_kernel_start = time.perf_counter()

    out_hash_gpu = cuda.device_array(1, dtype=np.uint64) # Allocate output on GPU
    gpu_fnv1a_hash_kernel[1, 1](data_gpu_numba, data_len_bytes, out_hash_gpu) # Launch kernel
    if torch.cuda.is_available(): torch.cuda.synchronize() # Ensure kernel finishes
    
    final_hash_cpu = out_hash_gpu.copy_to_host() # Copy result to CPU
    
    if torch.cuda.is_available(): torch.cuda.synchronize() # Ensure copy finishes
    t_kernel_end = time.perf_counter()
    kernel_time_ms = (t_kernel_end - t_kernel_start) * 1000
    if _internal_debug_prints_enabled: print(f"DEBUG gpu_fast_hash: Kernel exec + D2H copy took: {kernel_time_ms:.4f} ms")
    
    total_time_ms = prep_time_ms + bridge_time_ms + kernel_time_ms
    if _internal_debug_prints_enabled: print(f"DEBUG gpu_fast_hash: Total internal time: {total_time_ms:.4f} ms")

    return format(final_hash_cpu[0], '016x')

# --- SHA256 CPU (for comparison) ---
def sha256_cpu(x):
    h = hashlib.sha256()
    h.update(x.detach().cpu().numpy().tobytes())
    return h.hexdigest()

# --- _record (slightly simplified for debugging focus) ---
def _record(tag, a, b, out, cfg):
    if random.random() > cfg.sample_rate: return
    
    t0_hash_start = time.perf_counter()
    h_in, h_out = None, None
    
    if torch.is_tensor(a):
        h_in = gpu_fast_hash(a) if a.is_cuda and cfg.use_gpu_hash else sha256_cpu(a)
    if torch.is_tensor(out):
        h_out = gpu_fast_hash(out) if out.is_cuda and cfg.use_gpu_hash else sha256_cpu(out)
    
    h_ms = (time.perf_counter() - t0_hash_start) * 1000 
    cfg.records.append(dict(tag=tag, shape_a=tuple(a.shape) if torch.is_tensor(a) else None, 
                           hash_in=h_in, hash_out=h_out, hash_ms=h_ms))

# --- _wrap_matmul ---
def _wrap_matmul(fn, name, cfg):
    @wraps(fn)
    def wrapper(*args, **kw):
        out = fn(*args, **kw); _record(name, args[0], args[1], out, cfg); return out
    return wrapper

# --- verification context manager ---
@contextlib.contextmanager
def verification(sample_rate=0.2, hash_weights=False, use_gpu_hash=True): # hash_weights not used in this simplified _record
    cfg = types.SimpleNamespace(sample_rate=sample_rate, records=[], use_gpu_hash=use_gpu_hash and torch.cuda.is_available())
    patched = []
    for name in ("mm", "matmul", "bmm"):
        if hasattr(torch, name):
            orig = getattr(torch, name)
            if callable(orig): setattr(torch, name, _wrap_matmul(orig, name, cfg)); patched.append((torch, name, orig))
    
    fwd_hook_handle, bwd_hook_handle = None, None
    try:
        from torch.nn.modules.module import register_module_forward_hook, register_module_full_backward_hook
        def fwd_hook(mod, inp, out):
            if isinstance(mod, nn.Linear): _record("linear_fw", inp[0], mod.weight.t(), out, cfg)
        fwd_hook_handle = register_module_forward_hook(fwd_hook)
        # Simplified bwd_hook for now, focusing on gpu_fast_hash performance
        # def bwd_hook(mod, grad_in, grad_out):
        #     if isinstance(mod, nn.Linear): # ... record call ...
        # bwd_hook_handle = register_module_full_backward_hook(bwd_hook)
    except ImportError: pass
    try: yield cfg.records
    finally:
        for tgt, n, orig in patched: setattr(tgt, n, orig)
        if fwd_hook_handle: fwd_hook_handle.remove()
        if bwd_hook_handle: bwd_hook_handle.remove()

# --- Main ---
if __name__ == '__main__':
    _internal_debug_prints_enabled = True # Enable verbose prints from gpu_fast_hash

    def timeit_fn(fn, *args, iters=50, warmup_iters=5, **kw):
        for _ in range(warmup_iters): fn(*args, **kw)
        if torch.cuda.is_available(): torch.cuda.synchronize()
        
        all_results = []
        t_start_total = time.perf_counter()
        for i in range(iters):
            # print(f"timeit_fn iter {i+1}/{iters}") # Verbose iter count
            iter_t_start = time.perf_counter()
            res = fn(*args, **kw)
            if torch.cuda.is_available(): torch.cuda.synchronize()
            iter_t_end = time.perf_counter()
            all_results.append((iter_t_end - iter_t_start) * 1000)
        
        t_end_total = time.perf_counter()
        avg_time = sum(all_results) / len(all_results) if all_results else 0
        # print(f"Individual times for {fn.__name__}: {all_results}") # Print all individual times
        return avg_time, res # res is from the last iteration

    print("--- Testing gpu_fast_hash ---")
    if torch.cuda.is_available():
        print("INFO: Explicitly compiling Numba kernel for gpu_fast_hash (if not already done)...")
        # Using a very small tensor for initial JIT. The actual type matters more than size for compilation.
        compile_tensor = torch.randn(2, 2, device="cuda", dtype=torch.float16) 
        try:
            gpu_fast_hash(compile_tensor) 
            torch.cuda.synchronize()
            print("INFO: Numba kernel compilation call complete.")
        except Exception as e_compile: print(f"ERROR during explicit compile: {e_compile}")
        
        _gpu_fast_hash_data_path_printed_once = False # Reset for actual tests
        _internal_debug_prints_enabled = True # Keep True for the first detailed test run

        print("\n--- Benchmarking 1000x1000 fp16 tensor ---")
        large_tensor = torch.randn(1000, 1000, device="cuda", dtype=torch.float16)
        # More warmup, fewer iters for the main timing to reduce log spam if debug prints are on
        hash_large_time, hash_large_val = timeit_fn(gpu_fast_hash, large_tensor, iters=20, warmup_iters=10) 
        print(f"GPU FNV1a hash (1000x1000 fp16): {hash_large_val} (took {hash_large_time:.4f} ms on avg)")

        print("\n--- Benchmarking 128x128 fp16 tensor ---")
        small_tensor = torch.randn(128, 128, device="cuda", dtype=torch.float16)
        hash_small_time, hash_small_val = timeit_fn(gpu_fast_hash, small_tensor, iters=50, warmup_iters=10)
        print(f"GPU FNV1a hash (128x128 fp16): {hash_small_val} (took {hash_small_time:.4f} ms on avg)")

        # CPU SHA256 for comparison (on large tensor)
        cpu_sha_time, _ = timeit_fn(sha256_cpu, large_tensor, iters=10, warmup_iters=1)
        print(f"CPU SHA256 hash (1000x1000 fp16): (took {cpu_sha_time:.4f} ms on avg)")
        
        _internal_debug_prints_enabled = False # Disable for model tests to reduce log spam

    else:
        print("CUDA not available, skipping gpu_fast_hash direct test.")

    # --- Tiny Model Test (Simplified, focusing on gpu_fast_hash behavior) ---
    print("\n--- Running verification on a simple NN (Tiny) ---")
    class Tiny(nn.Module):
        def __init__(self, in_f=128, h=64, out_f=32): super().__init__(); self.l1=nn.Linear(in_f,h); self.l2=nn.Linear(h,out_f)
        def forward(self, x): return self.l2(torch.relu(self.l1(x)))    

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    model_tiny = Tiny().to(device)
    x_tiny = torch.randn(16, 128, device=device) 

    if device == "cuda":
        print("INFO: Warming up gpu_fast_hash for Tiny model test...")
        _internal_debug_prints_enabled = True # Enable for one call
        try:
            # Create a tensor similar to what the model will process
            temp_hidden = model_tiny.l1(x_tiny) # (16, 64)
            gpu_fast_hash(x_tiny) # (16, 128)
            gpu_fast_hash(temp_hidden) 
            torch.cuda.synchronize()
            print("INFO: Warmup for Tiny model test complete.")
        except Exception as e_tiny_warmup: print(f"ERROR during Tiny model warmup: {e_tiny_warmup}")
        _internal_debug_prints_enabled = False # Disable for the actual timed run
        _gpu_fast_hash_data_path_printed_once = False


    base_tiny_ms, _ = timeit_fn(model_tiny, x_tiny, iters=100, warmup_iters=20)
    print(f"Baseline Tiny model forward: {base_tiny_ms:.4f} ms")

    with verification(sample_rate=1.0, use_gpu_hash=True) as recs_tiny:
        pomm_tiny_ms, _ = timeit_fn(model_tiny, x_tiny, iters=100, warmup_iters=0) # No separate warmup for pomm run
    
    print(f"Tiny model forward with verification: {pomm_tiny_ms:.4f} ms")
    if base_tiny_ms > 1e-9 : 
        overhead_tiny = (pomm_tiny_ms - base_tiny_ms) / base_tiny_ms * 100
        print(f"Overhead: {overhead_tiny:.2f}%")
    else: print("Base time too small for reliable overhead calculation.")

    if recs_tiny:
        print(f"Number of records from Tiny: {len(recs_tiny)}")
        # Simplified keys for brevity
        keys_to_show = ['tag', 'hash_in', 'hash_out', 'hash_ms']
        filtered_recs_tiny = [ {k: r.get(k) for k in keys_to_show} for r in recs_tiny[:5]]
        if filtered_recs_tiny: print(tabulate(filtered_recs_tiny, headers="keys", tablefmt="grid"))
    else: print("No records captured for Tiny model.")
    
    # MNIST part can be re-enabled later once gpu_fast_hash is confirmed fast.
    print("\n--- MNIST training loop section omitted for this debugging run ---")

