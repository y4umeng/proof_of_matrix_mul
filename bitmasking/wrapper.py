import socket, struct, threading, time, random, types, contextlib
import torch, numpy as np, torch.nn as nn
from functools import wraps
from torch.nn.modules.module import register_module_forward_hook



HOST, PORT_BASE, GLOBAL_SEED = "127.0.0.1", 11234, 42
random.seed(GLOBAL_SEED); torch.manual_seed(GLOBAL_SEED)

orig_matmul = torch.matmul
orig_tensor_matmul = torch.Tensor.__matmul__
orig_tensor_rmatmul = torch.Tensor.__rmatmul__



def _verifier_server(B_public, m, n, layer_idx, n_rows, n_cols):
    """Bare-bones TCP listener; exits after one blob."""
    def recvall(sock, n):
        buf = bytearray()
        while len(buf) < n:
            chunk = sock.recv(n - len(buf))
            if not chunk:
                raise RuntimeError("socket closed")
            buf.extend(chunk)
        return bytes(buf)

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.bind((HOST, PORT_BASE + layer_idx))
    srv.listen(1)
    conn, _ = srv.accept()

    hdr = recvall(conn, 8)
    layer, rows, cols = struct.unpack("<HHH2x", hdr)
    print(f"[Verifier] Received header: layer_idx={layer}, n_rows={rows}, n_cols={cols}")

    k = B_public.shape[0]
    bytes_A = rows * k * 4
    bytes_C = rows * cols * 4
    payload = recvall(conn, bytes_A + bytes_C)
    print(f"[Verifier] Received payload: {len(payload)} bytes (A: {bytes_A}, C: {bytes_C})")

    A_rows = np.frombuffer(payload[:bytes_A], dtype=np.float32).reshape(rows, k)
    C_vals = np.frombuffer(payload[bytes_A:], dtype=np.float32).reshape(rows, cols)

    rng = random.Random(GLOBAL_SEED)
    row_idx_v = torch.tensor(rng.sample(range(m), n_rows))
    col_idx_v = torch.tensor(rng.sample(range(n), n_cols))

    print(f"[Verifier] row_idx shape: {row_idx_v.shape}, first few: {row_idx_v[:5]}")
    print(f"[Verifier] col_idx shape: {col_idx_v.shape}, first few: {col_idx_v[:5]}")

    # disable hooks **inside verifier** to avoid recursion
    _THREAD.no_hook = True
    try:
        B_sub      = B_public[:, col_idx_v]                    # k × cols
        recomputed = orig_tensor_matmul(torch.from_numpy(A_rows.copy()), B_sub)
    finally:
        _THREAD.no_hook = False

    diff = torch.abs(recomputed - torch.from_numpy(C_vals.copy()))
    ok = torch.allclose(recomputed, torch.from_numpy(C_vals), atol=1e-3, rtol=1e-3)

    print(f"[Verifier] Max diff: {diff.max().item()}, mean diff: {diff.mean().item()}")
    print(f"[Verifier] layer {layer} passed? {ok}\n")
    conn.close(); srv.close()


def _prover_send(sampled_A, sampled_C, layer_idx):
    n_rows = sampled_A.shape[0]
    n_cols = sampled_C.shape[1]
    print(f"[Prover] Sending: n_rows={n_rows}, n_cols={n_cols}")

    buf_A = sampled_A.cpu().numpy().astype(np.float32).tobytes()
    buf_C = sampled_C.cpu().numpy().astype(np.float32).tobytes()
    hdr = struct.pack("<HHH2x", layer_idx, n_rows, n_cols)

    with socket.create_connection((HOST, PORT_BASE + layer_idx)) as s:
        s.sendall(hdr + buf_A + buf_C)
        print(f"[Prover] Sent {len(hdr)+len(buf_A)+len(buf_C)} total bytes\n")


def audit_protocol(A, B, layer_idx):
    if getattr(_THREAD,"in_audit",False):
        return


    _THREAD.in_audit = True
    try:
        # use the *un-patched* implementation exactly once
        C = orig_matmul(A, B)

        m, _ = A.shape
        n    = B.shape[1]

        rng = random.Random(GLOBAL_SEED)
        row_idx = torch.tensor(rng.sample(range(m), max(1, int(m * 0.001))))
        col_idx = torch.tensor(rng.sample(range(n), max(1, int(n * 0.01))))

        sampled_A = A[row_idx]
        sampled_C = C[row_idx][:, col_idx]

        print(f"[Prover] row_idx shape: {row_idx.shape}, first few: {row_idx[:5]}")
        print(f"[Prover] col_idx shape: {col_idx.shape}, first few: {col_idx[:5]}")
        print(f"[Prover] sampled_A shape: {sampled_A.shape}")
        print(f"[Prover] sampled_C shape: {sampled_C.shape}")

        th = threading.Thread(
            target=_verifier_server,
            args=(B, m, n, layer_idx,
                  sampled_A.shape[0], sampled_C.shape[1]),
            daemon=True)
        th.start()
        time.sleep(0.05)
        _prover_send(sampled_A, sampled_C, layer_idx)
        th.join()
    finally:
        _THREAD.in_audit = False

# -----------------  monkey-patch + public context  ------------------
_THREAD = threading.local()

def _wrap_fn(fn, op_name, cfg):
    @wraps(fn)
    def wrapper(*args, **kw):
        if getattr(_THREAD, "no_hook", False):
            return fn(*args, **kw)          # bypass while flag is set
        out = fn(*args, **kw)
        if random.random() <= cfg.sample_rate:
            cfg.counter += 1
            audit_protocol(args[0], args[1], cfg.counter)
        return out
    return wrapper


@contextlib.contextmanager
def verification(sample_rate=0.1):
    cfg = types.SimpleNamespace(sample_rate=sample_rate, counter=0)
    _THREAD.records = []

    patched = []
    for name in ("mm", "matmul", "bmm"):
        orig = getattr(torch, name)
        setattr(torch, name, _wrap_fn(orig, name, cfg))
        patched.append((torch, name, orig))

     # ------------ patch tensor @-operator methods ------------------
    def _make_tensor_patch(orig_meth):
        @wraps(orig_meth)
        def _tensor_mm(self, other):
            if getattr(_THREAD, "no_hook", False):
                return orig_meth(self, other)
            out = orig_meth(self, other)
            if random.random() <= cfg.sample_rate:
                cfg.counter += 1
                audit_protocol(self, other, cfg.counter)
            return out
        return _tensor_mm
 
    for meth_name, orig_meth in (("__matmul__", orig_tensor_matmul),
                                 ("__rmatmul__", orig_tensor_rmatmul)):
        setattr(torch.Tensor, meth_name, _make_tensor_patch(orig_meth))
        patched.append((torch.Tensor, meth_name, orig_meth))

    def _linear_hook(module, inputs, output):
        if isinstance(module, nn.Linear) and random.random() <= cfg.sample_rate:
            cfg.counter += 1
            audit_protocol(inputs[0], module.weight.t(), cfg.counter)

    hook_handle = register_module_forward_hook(_linear_hook)

    try:
        yield
    finally:
        for tgt, name, orig in patched:
            setattr(tgt, name, orig)
        hook_handle.remove()


# ----------------------------  demo  --------------------------------
if __name__ == "__main__":
    A = torch.randn(10000, 10000)
    B = torch.randn(10000, 1000)

    with verification(sample_rate=1.0):
        _ = A @ B
        _ = torch.matmul(A,B)
        torch.matmul(torch.randn(10000,10000), torch.randn(10000,1000))