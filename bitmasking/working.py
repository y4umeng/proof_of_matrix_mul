"""
Toy prover ↔ verifier stream for the long-context audit scheme.

 * One TCP connection per sampled matmul.
 * Payload contains ONLY
       - sampled_A  (fp16 rows)
       - sampled_C  (fp16 scalars)
       - layer_idx  (uint16 for now)
 * Indices are *not* sent; the verifier regenerates them with the same PRNG.

Run this file as-is: first it starts the verifier thread, then the prover
connects, pushes its bytes, and you should see "verification passed? True".
"""

import socket, struct, threading, time, random, torch
import numpy as np

# ---------------- 0.  GLOBAL CONSTANTS & SEED  --------------------
HOST, PORT       = "127.0.0.1", 11234 # loopback → no firewall friction
GLOBAL_SEED      = 42                      # shared PRNG seed
LAYER_IDX        = 3                       # hard-coded for this toy demo

torch.manual_seed(GLOBAL_SEED)
random.seed(GLOBAL_SEED)

# ---------------- 1.  PROVER CALCULATES ITS SAMPLES  -------------
d = 4000
A = torch.randn(d, d)          # activations (private to prover)
B = torch.randn(d, d)          # weights (public / verifier also has)

C = A @ B                      # full matmul

# mask:   0.1 % rows , 1 % cols
rng = random.Random(GLOBAL_SEED)
row_idx = torch.tensor(rng.sample(range(d), int(d*0.001)))
col_idx = torch.tensor(rng.sample(range(d), int(d*0.01)))
# row_idx = torch.randperm(d)[: int(d * 0.001)]      # 4 rows
# col_idx = torch.randperm(d)[: int(d * 0.01)]       # 40 columns

print(f"[Prover] row_idx shape: {row_idx.shape}, first few: {row_idx[:5]}")
print(f"[Prover] col_idx shape: {col_idx.shape}, first few: {col_idx[:5]}")

sampled_A = A[row_idx]                             # 4 × 4000
sampled_C = C[row_idx][:, col_idx]                 # 4 × 40

print(f"[Prover] sampled_A shape: {sampled_A.shape}")
print(f"[Prover] sampled_C shape: {sampled_C.shape}")

# ---------------- 2.  VERIFIER SERVER THREAD  --------------------
def verifier_server(B_public, d):
    """
    Bare-bones TCP listener.
    Waits for one blob, checks it, prints result, then exits.
    """

    # ---- helper: recv exactly n bytes or die ----
    def recvall(sock, n):
        buf = bytearray()
        while len(buf) < n:
            chunk = sock.recv(n - len(buf))
            if not chunk:              # connection closed too early
                raise RuntimeError("socket closed")
            buf.extend(chunk)
        return bytes(buf)

    # ---- server main ----
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.bind((HOST, PORT))
    srv.listen(1)
    conn, _ = srv.accept()

    # 2.1 read fixed 8-byte header   (uint16 layer | uint16 n_rows | uint16 n_cols | padding)
    hdr = recvall(conn, 8)
    layer_idx, n_rows, n_cols = struct.unpack("<HHH2x", hdr)
    print(f"[Verifier] Received header: layer_idx={layer_idx}, n_rows={n_rows}, n_cols={n_cols}")

    # 2.2 compute how many bytes to expect for the two tensors
    bytes_A = n_rows * d * 4              # fp32 → 4 B each
    bytes_C = n_rows * n_cols * 4
    payload = recvall(conn, bytes_A + bytes_C)
    print(f"[Verifier] Received payload: {len(payload)} bytes (A: {bytes_A}, C: {bytes_C})")

    # 2.3 slice + reshape
    A_rows = np.frombuffer(payload[:bytes_A], dtype=np.float32).reshape(n_rows, d)
    C_vals = np.frombuffer(payload[bytes_A:], dtype=np.float32).reshape(n_rows, n_cols)
    print(f"[Verifier] A_rows shape: {A_rows.shape}, C_vals shape: {C_vals.shape}")

    # 2.4 regenerate PRNG mask (must match prover's)
    rng = random.Random(GLOBAL_SEED)                  # same seed
    row_idx_v = torch.tensor(rng.sample(range(d), n_rows))
    col_idx_v = torch.tensor(rng.sample(range(d), n_cols))
    
    print(f"[Verifier] row_idx shape: {row_idx_v.shape}, first few: {row_idx_v[:5]}")
    print(f"[Verifier] col_idx shape: {col_idx_v.shape}, first few: {col_idx_v[:5]}")
    
    # Check if indices match between prover and verifier
    indices_match = True
    try:
        # These globals should be accessible from the verifier thread
        if not torch.all(row_idx_v == row_idx):
            indices_match = False
            print(f"[Verifier] ERROR: row indices don't match!")
        if not torch.all(col_idx_v == col_idx):
            indices_match = False
            print(f"[Verifier] ERROR: column indices don't match!")
        print(f"[Verifier] Indices match with prover: {indices_match}")
    except NameError:
        print("[Verifier] Can't check if indices match - prover indices not accessible")

    # 2.5 recompute C = A @ B  for the masked entries
    B_sub = B_public[:, col_idx_v]                  # 4000 × 40
    print(f"[Verifier] B_sub shape: {B_sub.shape}")
    
    # Create a writeable copy to avoid the warning
    A_rows_writeable = A_rows.copy()
    recomputed = torch.from_numpy(A_rows_writeable.astype(np.float32)) @ B_sub  # 4 × 40
    print(f"[Verifier] recomputed shape: {recomputed.shape}")
    
    # Convert C_vals to a writeable copy
    C_vals_writeable = C_vals.copy()
    C_tensor = torch.from_numpy(C_vals_writeable)
    
    # Check a few individual elements
    print(f"[Verifier] First few values of recomputed: {recomputed.flatten()[:5]}")
    print(f"[Verifier] First few values of C_vals: {C_tensor.flatten()[:5]}")
    
    # Calculate difference between values
    # recomputed_half = recomputed.half()
    diff = torch.abs(recomputed - C_tensor)
    print(f"[Verifier] Max difference: {diff.max().item()}, Mean difference: {diff.mean().item()}")

    ok = torch.allclose(recomputed, C_tensor, atol=1e-3, rtol=1e-3)
    print(f"[Verifier] layer {layer_idx}  passed? {ok}")
    conn.close()
    srv.close()

# launch verifier thread
threading.Thread(target=verifier_server, args=(B, d), daemon=True).start()
time.sleep(0.1)          # give the listener a moment

# ---------------- 3.  PROVER SENDS ITS BLOB  ----------------------
def prover_send(sampled_A, sampled_C):
    n_rows, _ = sampled_A.shape
    _,      n_cols   = sampled_C.shape
    
    print(f"[Prover] Sending: n_rows={n_rows}, n_cols={n_cols}")

    # 3.1 flatten to raw bytes (little-endian fp32)
    buf_A = sampled_A.cpu().numpy().astype(np.float32).tobytes()
    buf_C = sampled_C.cpu().numpy().astype(np.float32).tobytes()
    
    print(f"[Prover] buf_A size: {len(buf_A)} bytes, buf_C size: {len(buf_C)} bytes")

    # 3.2 8-byte header: layer | n_rows | n_cols
    hdr = struct.pack("<HHH2x", LAYER_IDX, n_rows, n_cols)

    with socket.create_connection((HOST, PORT)) as s:
        s.sendall(hdr + buf_A + buf_C)    # one shot; OS chunks as needed
        print(f"[Prover] Sent {len(hdr) + len(buf_A) + len(buf_C)} total bytes")

prover_send(sampled_A, sampled_C)

# tiny sleep so the verifier thread can print before script exits
time.sleep(0.5)  # Increased sleep time to ensure all debug messages are printed