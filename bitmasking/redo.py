import torch
import threading
import types
import contextlib
from functools import wraps
from torch import nn
from torch.nn import Module
from torch.nn.modules.module import register_module_forward_hook
import random
import time
import socket  # TCP sockets
import struct # pack/unpack binary headers
import torch.nn.functional as F
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM


# seed setup
torch.manual_seed(42)
random.seed(42)
torch.set_default_dtype(torch.float64)

# WORK FROM PROVER

# B is known to both parties
# A is only known to the prover
A = torch.randn(4000,4000) # activations
B = torch.randn(4000,4000) # public weight matrix

# full matmul
C = A@B

print("C.shape: ",C.shape)

n_rows = C.shape[0]
n_cols=C.shape[1]

# amount of rows and columns to sample
n_row_samples = int(n_rows*0.001)
n_col_samples = int(n_cols * 0.01)

# sampling random row and column indices
row_indices=torch.randperm(n_rows)[:n_row_samples]
col_indices = torch.randperm(n_cols)[:n_col_samples]

# sampling the rows of matrices to be sent over
sampled_A = A[row_indices]
sampled_C = C[row_indices][:,col_indices] # VALUES TO BE CHECKED.


print("shape of sampled_A: ",sampled_A.shape)
print("sampled_C.shape: ",sampled_C.shape)

# VERIFIER SIDE RECOMPUTATION
sampled_B = B[:,col_indices]

mat = sampled_A @ sampled_B
print("shape of matrix recomputation: ",mat.shape)

passed = torch.allclose(mat, sampled_C)
max_diff = (mat - sampled_C).abs().max()
print("max_diff: ",max_diff)
print(passed)


HOST = "127.0.0.1"
PORT = 1123

def prepare_packet(sampled_A: torch.Tensor, sampled_C: torch.Tensor) -> bytes:
  """
  1. copy gpu tensors to cpu pinned memory
  2. extract shapes to include a tiny header
  3. return (header_bytes, raw_bytes_A, raw_bytes_C)
  """
  A_cpu = sampled_A.detach().cpu().pin_memory()
  C_cpu = sampled_C.detach().cpu().pin_memory()

  n_rows, k_dim = A_cpu.shape
  _, n_cols = C_cpu.shape

  header = struct.pack("<III",n_rows,n_cols,k_dim)

  raw_A = A_cpu.numpy().tobytes()
  raw_C = C_cpu.numpy().tobytes()

  return header, raw_A, raw_C

# A = torch.randn(10, 16, device="cuda")
# B = torch.randn(16, 8,  device="cuda")

# C = A @ B
# # sample a few rows/cols for demo
# rows = torch.arange(3)
# cols = torch.arange(4)
# sampled_A = A[rows]
# sampled_C = C[rows][:, cols]

# hdr, bufA, bufC = prepare_packet(sampled_A, sampled_C)
# print("Streaming random matrices")
# print("Header bytes:", len(hdr), "=> shapes", struct.unpack("<III", hdr))
# print("bufA:", len(bufA), "bytes;", "bufC:", len(bufC), "bytes")

# don't know that well what this function does
def send_packet(header: bytes, raw_A: bytes, raw_C: bytes, chunk_size: int = 1048576):
  """
  open tcp connection to (host, port), then
  1. send 12-byte header
  2. stream raw_A in chunk_size-byte slices
  3. stream raw_C in chunk_size-byte slices
  """

  conn = socket.create_connection((HOST,PORT))
  try:
    conn.sendall(header)
    total_A = len(raw_A)
    offset = 0
    while offset < total_A:
      end = offset + chunk_size
      conn.sendall(raw_A[offset:end])
      offset = end

    total_C = len(raw_C)
    offset = 0
    while offset < total_C:
      end = offset + chunk_size
      conn.sendall(raw_C[offset:end])
      offset = end
  finally:
    conn.close()


# WRAPPER FUNCTION
def new(A,B):
  C = A @ B

  A2d = A.reshape(-1,A.shape[-1])
  C2d = C.reshape(-1,C.shape[-1])

  n_rows, k_dim = A2d.shape
  n_cols = C2d.shape[1]

  # amount of rows and columns to sample
  # entire rows of A have to be streamed over so it should be lower
  n_row_samples = int(n_rows*0.01)
  n_col_samples = int(n_cols * 0.01)

  # sampling random row and column indices
  row_indices=torch.randperm(n_rows)[:n_row_samples]
  col_indices = torch.randperm(n_cols)[:n_col_samples]

  # sampling the rows of matrices to be sent over
  sampled_A = A2d[row_indices] # ENTIRE ROWS
  sampled_C = C2d[row_indices][:,col_indices] # VALUES TO BE CHECKED.


  # ALL STREAMING LOGIC
  t_prep_start = time.perf_counter()
  header, bufA, bufC = prepare_packet(sampled_A, sampled_C)
  prep_ms = (time.perf_counter() - t_prep_start) * 1000
  # print(f"prepare_packet: {prep_ms:.2f} ms")

  t_send_start = time.perf_counter()
  send_packet(header, bufA, bufC)
  send_ms = (time.perf_counter() - t_send_start) * 1000
  # print(f"send_packet: {send_ms:.2f} ms")
  print(f"total network overhead: {(prep_ms + send_ms):.2f} ms")

# testing out the wrapper function
A = torch.randn(4000,4000) # activations
B = torch.randn(4000,4000) # public weight matrix

def time_matmuls(iters: int) -> float:
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for i in range(iters):
      # original matmul
      _ = A @ B
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000.0   # ms

def time_sampling_matmuls(iters: int) -> float:
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for i in range(iters):
      # new wrapper function
      new(A,B)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000.0   # ms

# print("\n")
# baseline = time_matmuls(10)
# print("baseline (normal matmul) time: ", baseline, "ms")

# print("\n")
# new_time = time_sampling_matmuls(10)
# print("NEW TIME (WITH STREAMING) time: ",new_time, "ms")

# overhead = new_time - baseline
# print(f"overhead (%): {overhead/baseline:}%")

# NETWORK SETUP
import threading, socket, struct

HOST, PORT = "127.0.0.1", 1123

def robust_server():
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((HOST, PORT))
    srv.listen(1)
    # print("[server] listening on", HOST, PORT)

    while True:
        conn, addr = srv.accept()
        # print("[server] connection from", addr)
        try:
            # 1) read header (we packed rows, cols, k_dim)
            hdr = conn.recv(12)
            n_rows, n_cols, k_dim = struct.unpack("<III", hdr)
            # print(f"[server] header → rows={n_rows}, cols={n_cols}, k_dim={k_dim}")

            # 2) read exactly A_bytes then C_bytes
            # 8 if float64
            # 4 if float32
            A_bytes = n_rows * k_dim * 8
            C_bytes = n_rows * n_cols * 8

            buf = bytearray()
            while len(buf) < A_bytes + C_bytes:
                chunk = conn.recv((A_bytes + C_bytes) - len(buf))
                if not chunk:
                    raise RuntimeError("connection closed early asdf")
                buf.extend(chunk)
            # print(f"[server] received payload: {len(buf)} bytes")

            # (we're just discarding it here; a real verifier would reshape & check)

        except Exception as e:
            print("[server] error during handling:", e)
        finally:
            conn.close()
            print("[server] closed connection")

# start it once, in daemon mode
threading.Thread(target=robust_server, daemon=True).start()


_THREAD = threading.local()

# saving original matrix multiplication functions
_orig_matmul         = torch.matmul
_orig_tensor_matmul  = torch.Tensor.__matmul__
_orig_tensor_rmatmul = torch.Tensor.__rmatmul__


def _make_wrapper(orig_function):
  """
  returns a wrapper function around orig_function that:
  1. bypasses hooking if _THREAD.no_hook is True
  2. sets flag to avoid recursive hooks.
  3. calls orig_function to get the real results
  4. invokes new(a,b) streaming logic
  """
  @wraps(orig_function)
  def wrapper(a,b,*args,**kwargs):
    # if already inside a hook, just do the raw operaetion
    if getattr(_THREAD, "no_hook", False):
      return orig_function(a,b,*args,**kwargs)

    # raise flag so nested matmuls aren't hooked
    _THREAD.no_hook = True
    try:
      out = orig_function(a,b,*args,**kwargs)
      new(a,b)

    finally:
      _THREAD.no_hook = False

    return out
  return wrapper

def _linear_forward_hook(module: Module, inputs: tuple, output: torch.Tensor):
  """
  called after every nn.linear.forward.
  1. skip if inside another hook
  2. pull out the inpute activations and weight matrix and call new() to stream the sampled slice
  """
  if getattr(_THREAD,"no_hook",False):
    return


  if not isinstance(module, nn.Linear):
    return

  _THREAD.no_hook = True
  try:
    inp = inputs[0]
    weight = module.weight.t()
    new(inp,weight)
  finally:
    _THREAD.no_hook = False


@contextlib.contextmanager
def streaming_audit():
  torch.matmul = _make_wrapper(torch.matmul)
  torch.Tensor.__matmul__ = _make_wrapper(torch.Tensor.__matmul__)
  torch.Tensor.__rmatmul__ = _make_wrapper(torch.Tensor.__rmatmul__)

  hook_handle = register_module_forward_hook(_linear_forward_hook)

  try:
    yield # returns to user code with hooks active
  finally:
    # unpatch everything
    torch.matmul              = _orig_matmul
    torch.Tensor.__matmul__   = _orig_tensor_matmul
    torch.Tensor.__rmatmul__  = _orig_tensor_rmatmul

    # 4) Remove the forward-hook
    hook_handle.remove()

# testing out the wrapper function
A = torch.randn(4000,4000) # activations
B = torch.randn(4000,4000) # public weight matrix

def time_matmuls(iters: int) -> float:
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for i in range(iters):
      # original matmul
      _ = A @ B
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000.0   # ms

def time_sampling_matmuls(iters: int) -> float:
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with streaming_audit():
      # Each of these operations will:
      # 1) compute the result normally on GPU
      # 2) call new(A, B) under the hood to stream sample
      for i in range(iters):
        # new wrapper function
        new(A,B)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000.0   # ms

# print("\n")
# # baseline = time_matmuls(10)
# print("baseline (normal matmul) time: ", baseline, "ms")

# print("\n")
# # new_time = time_sampling_matmuls(10)
# print("NEW TIME (WITH STREAMING) time: ",new_time, "ms")

# # overhead = new_time - baseline
# print(f"overhead (%): {overhead/baseline:}%")


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(10000, 5000)
        self.l2 = nn.Linear(5000, 1000)
    def forward(self, x):
        return self.l2(F.relu(self.l1(x)))

model = Tiny().cuda()
x = torch.randn(1000, 10000, device="cuda")

@torch.no_grad()
def time_matmuls(iters: int) -> float:
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for i in range(iters):
      # original matmul
      _ = model(x)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000.0   # ms

@torch.no_grad()
def time_sampling_matmuls(iters: int) -> float:
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with streaming_audit():
      for i in range(iters):
        _ = model(x)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000.0   # ms

print("\n")
baseline = time_matmuls(3)
print("baseline (normal matmul) time: ", baseline, "ms")

print("\n")
new_time = time_sampling_matmuls(3)
print("NEW TIME (WITH STREAMING) time: ",new_time, "ms")

overhead = new_time - baseline
print(f"overhead (%): {overhead/baseline:}%")


device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
model.eval()

BATCH   = 16
SEQ_LEN = 32

prompt_text = "The quick brown fox jumps over the lazy dog. " * 4
tokens  = tokenizer(prompt_text, return_tensors="pt")["input_ids"][0][:SEQ_LEN]
inputs  = tokens.unsqueeze(0).repeat(BATCH, 1).to(device)

@torch.no_grad()
def time_matmuls(iters: int) -> float:
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for i in range(iters):
      # original matmul
      _ = model(inputs)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000.0   # ms

@torch.no_grad()
def time_sampling_matmuls(iters: int) -> float:
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with streaming_audit():
      # Each of these operations will:
      # 1) compute the result normally on GPU
      # 2) call new(A, B) under the hood to stream sample
      for i in range(iters):
        # new wrapper function
        _ = model(inputs)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000.0   # ms

iters = 20

print("\n")
baseline = time_matmuls(iters)
print("baseline (normal matmul) time: ", baseline, "ms")

print("\n")
new_time = time_sampling_matmuls(iters)
print("NEW TIME (WITH STREAMING) time: ",new_time, "ms")

overhead = new_time - baseline
print(f"overhead (%): {overhead/baseline:}%")