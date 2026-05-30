import torch
import hashlib
from itertools import combinations
import time

# Original function for comparison
def hash_array_original(arr):
    s = ','.join(map(str, arr))
    return hashlib.md5(s.encode()).hexdigest()

def simulate_original(arr, max_steps=1000, max_len=10**6):
    seen_exact = set()
    seen_sorted = set()
    for step in range(max_steps):
        if len(arr) == 1:
            return {"status": "converged", "step": step, "value": arr[0], "length": 1}
        key_exact = tuple(arr)
        key_sorted = tuple(sorted(arr))
        if key_exact in seen_exact or key_sorted in seen_sorted:
            return {"status": "loop_detected", "step": step, "value": arr, "length": len(arr)}
        seen_exact.add(key_exact)
        seen_sorted.add(key_sorted)
        if len(set(arr)) == 1 and step > 0:
            return {"status": "constant_value_loop", "step": step, "value": arr[0], "length": len(arr)}
        if len(arr) > max_len:
            return {"status": "too_large", "step": step, "length": len(arr)}
        
        # This is the bottleneck
        next_arr = []
        for a, b in combinations(arr, 2):
            next_arr.append(a ^ b)
        arr = next_arr
    return {"status": "max_steps_reached", "step": max_steps, "length": len(arr)}

# --- PyTorch Accelerated Version ---

def simulate_pytorch(arr, max_steps=1000, max_len=10**6, device="cuda"):
    """
    Accelerated version of the simulation using PyTorch for vectorized XOR operations.
    """
    # Use GPU if available, otherwise fallback to CPU
    if device == "cuda" and torch.cuda.is_available():
        dev = torch.device("cuda")
        # print("Using CUDA GPU for acceleration.")
    else:
        dev = torch.device("cpu")
        # print("Using CPU.")

    seen_exact = set()
    seen_sorted = set()

    # The main loop structure remains the same
    for step in range(max_steps):
        if len(arr) == 1:
            return {"status": "converged", "step": step, "value": arr[0], "length": 1}

        # Loop detection logic is unchanged
        key_exact = tuple(arr)
        key_sorted = tuple(sorted(arr))
        if key_exact in seen_exact or key_sorted in seen_sorted:
            return {"status": "loop_detected", "step": step, "value": arr, "length": len(arr)}
        seen_exact.add(key_exact)
        seen_sorted.add(key_sorted)
        if len(set(arr)) == 1 and step > 0:
            return {"status": "constant_value_loop", "step": step, "value": arr[0], "length": len(arr)}
        if len(arr) > max_len:
            return {"status": "too_large", "step": step, "length": len(arr)}

        # --- Core PyTorch Acceleration ---
        n = len(arr)
        # 1. Convert list to a PyTorch tensor on the chosen device.
        #    Use `torch.long` for integer bitwise operations.
        t = torch.tensor(arr, dtype=torch.long, device=dev)

        # 2. Vectorized XOR Operation using Broadcasting
        #    t.view(-1, 1) creates a column vector.
        #    PyTorch broadcasts `t` across the columns and `t.view(-1, 1)` across the rows,
        #    creating an n x n matrix of all pair-wise XORs.
        #    e.g., for t = [a, b, c], this creates:
        #    [[a^a, a^b, a^c],
        #     [b^a, b^b, b^c],
        #     [c^a, c^b, c^c]]
        xor_matrix = t.view(-1, 1) ^ t

        # 3. Extract the unique pairs.
        #    This is equivalent to `itertools.combinations`. We take the upper triangle
        #    of the matrix, excluding the diagonal (k=1).
        indices = torch.triu_indices(n, n, offset=1)
        next_t = xor_matrix[indices[0], indices[1]]
        
        # 4. Convert tensor back to a Python list for the next iteration's logic.
        arr = next_t.cpu().tolist()
        # --- End of PyTorch section ---

    return {
        "status": "max_steps_reached",
        "step": max_steps,
        "length": len(arr)
    }

def simulate_pytorch_advanced(arr, max_steps=1000, max_len=10**6, device="cuda"):
    if device == "cuda" and torch.cuda.is_available():
        dev = torch.device("cuda")
    else:
        dev = torch.device("cpu")

    # seen_sorted 現在儲存的是 bytes hash key
    seen_sorted_hashes = set()

    current_t = torch.tensor(arr, dtype=torch.long, device=dev)

    for step in range(max_steps):
        n = current_t.shape[0]
        if n == 1:
            return {"status": "converged", "step": step, "value": current_t.item(), "length": 1}
        if n > max_len:
            return {"status": "too_large", "step": step, "length": n}
        if len(torch.unique(current_t)) == 1 and step > 0:
             return {"status": "constant_value_loop", "step": step, "value": current_t[0].item(), "length": n}

        # --- 優化的循環偵測 ---
        # 1. 在 GPU 上排序
        sorted_t = torch.sort(current_t).values

        # 2. 將排序後的 Tensor 轉為 bytes 並計算 hash，這比 tolist() 和 tuple() 快得多
        # 我們只偵測 sorted array 的循環，因為這已包含 exact array 的情況
        key_bytes = sorted_t.cpu().numpy().tobytes()
        hash_key = hashlib.md5(key_bytes).hexdigest()

        if hash_key in seen_sorted_hashes:
            return {
                "status": "loop_detected",
                "step": step,
                "value": current_t.cpu().tolist(), # 最後才轉回 list
                "length": n
            }
        seen_sorted_hashes.add(hash_key)
        # --- 結束 ---

        # 核心計算 (可結合 JIT)
        xor_matrix = current_t.view(-1, 1) ^ current_t
        rows, cols = torch.triu_indices(n, n, 1, device=dev)
        current_t = xor_matrix[rows, cols]

    return {
        "status": "max_steps_reached",
        "step": max_steps,
        "length": current_t.shape[0]
    }

if __name__ == '__main__':
    # Initial array for testing. The array size grows very quickly.
    # A small initial array is sufficient to show the performance difference.
    initial_arr = list(range(10)) 
    
    print(f"--- Testing with initial array of length {len(initial_arr)} ---")

    # Time the original function
    start_time_orig = time.time()
    result_orig = simulate_original(initial_arr.copy(), max_steps=10)
    end_time_orig = time.time()
    print(f"\nOriginal function result: {result_orig}")
    print(f"Original function time: {end_time_orig - start_time_orig:.6f} seconds")

    print("-" * 20)

    # Time the PyTorch function
    start_time_pt = time.time()
    result_pt = simulate_pytorch(initial_arr.copy(), max_steps=10)
    end_time_pt = time.time()
    print(f"\nPyTorch function result: {result_pt}")
    print(f"PyTorch function time: {end_time_pt - start_time_pt:.6f} seconds")

    speedup = (end_time_orig - start_time_orig) / (end_time_pt - start_time_pt)
    print(f"\nSpeedup: {speedup:.2f}x")

    # --- Testing with advanced PyTorch function ---
    start_time_pt_advanced = time.time()
    result_pt_advanced = simulate_pytorch_advanced(initial_arr.copy(), max_steps=10)
    end_time_pt_advanced = time.time()
    print(f"\nAdvanced PyTorch function result: {result_pt_advanced}")
    print(f"Advanced PyTorch function time: {end_time_pt_advanced - start_time_pt_advanced:.6f} seconds")

    speedup_advanced = (end_time_orig - start_time_orig) / (end_time_pt_advanced - start_time_pt_advanced)
    print(f"\nSpeedup: {speedup_advanced:.2f}x")
