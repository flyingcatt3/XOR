from tqdm import tqdm
from itertools import combinations
import torch
import hashlib
import numpy as np
import time

def simulate_original_with_progress(arr, max_steps=1000, max_len=10**6):
    # 1. 用 tqdm 包裝迴圈
    pbar = tqdm(range(max_steps), desc="Original Simulation", unit=" step", ncols=100)
    for step in pbar:
        # 初始狀態顯示
        if step == 0:
            pbar.set_postfix(arr_len=f'{len(arr):,}')

        if len(arr) == 1:
            pbar.close()
            return {"status": "converged", "step": step, "value": arr[0], "length": 1}

        # 循環偵測邏輯不變
        key_exact = tuple(arr)
        key_sorted = tuple(sorted(arr))
        # 為了公平比較，此處省略 seen_exact/seen_sorted，因為進階版只用了 sorted
        # if key_exact in seen_exact or key_sorted in seen_sorted: ...

        if len(set(arr)) == 1 and step > 0:
            pbar.close()
            return {"status": "constant_value_loop", "step": step, "value": arr[0], "length": len(arr)}
        if len(arr) > max_len:
            pbar.close()
            return {"status": "too_large", "step": step, "length": len(arr)}
        
        next_arr = []
        for a, b in combinations(arr, 2):
            next_arr.append(a ^ b)
        arr = next_arr

        # 2. 更新進度條後綴資訊
        pbar.set_postfix(arr_len=f'{len(arr):,}', refresh=True)

    pbar.close()
    return {"status": "max_steps_reached", "step": max_steps, "length": len(arr)}

def simulate_pytorch_with_progress(arr, max_steps=1000, max_len=10**6, device="cuda"):
    if device == "cuda" and torch.cuda.is_available():
        dev = torch.device("cuda")
    else:
        dev = torch.device("cpu")

    seen_exact = set()
    seen_sorted = set()

    pbar = tqdm(range(max_steps), desc="PyTorch Simulation ", unit=" step", ncols=100)
    for step in pbar:
        if step == 0:
            pbar.set_postfix(arr_len=f'{len(arr):,}')

        if len(arr) == 1:
            pbar.close()
            return {"status": "converged", "step": step, "value": arr[0], "length": 1}

        # 循環偵測邏輯不變
        key_exact = tuple(arr)
        key_sorted = tuple(sorted(arr))
        if key_exact in seen_exact or key_sorted in seen_sorted:
            pbar.close()
            return {"status": "loop_detected", "step": step, "value": arr, "length": len(arr)}
        seen_exact.add(key_exact)
        seen_sorted.add(key_sorted)
        
        if len(set(arr)) == 1 and step > 0:
             pbar.close()
             return {"status": "constant_value_loop", "step": step, "value": arr[0], "length": len(arr)}
        if len(arr) > max_len:
             pbar.close()
             return {"status": "too_large", "step": step, "length": len(arr)}

        n = len(arr)
        t = torch.tensor(arr, dtype=torch.long, device=dev)
        xor_matrix = t.view(-1, 1) ^ t
        indices = torch.triu_indices(n, n, offset=1)
        next_t = xor_matrix[indices[0], indices[1]]
        
        arr = next_t.cpu().tolist()
        
        pbar.set_postfix(arr_len=f'{len(arr):,}', refresh=True)
    
    pbar.close()
    return {"status": "max_steps_reached", "step": max_steps, "length": len(arr)}

def simulate_advanced_with_progress(arr, max_steps=1000, max_len=10**6, device="cuda"):
    if device == "cuda" and torch.cuda.is_available():
        dev = torch.device("cuda")
    else:
        dev = torch.device("cpu")

    seen_sorted_hashes = set()
    current_t = torch.tensor(arr, dtype=torch.long, device=dev)

    pbar = tqdm(range(max_steps), desc="Advanced PyTorch   ", unit=" step", ncols=100)
    for step in pbar:
        n = current_t.shape[0]
        if step == 0:
            pbar.set_postfix(arr_len=f'{n:,}')

        if n == 1:
            pbar.close()
            return {"status": "converged", "step": step, "value": current_t.item(), "length": 1}
        if n > max_len:
            pbar.close()
            return {"status": "too_large", "step": step, "length": n}
        if len(torch.unique(current_t)) == 1 and step > 0:
            pbar.close()
            return {"status": "constant_value_loop", "step": step, "value": current_t[0].item(), "length": n}

        sorted_t = torch.sort(current_t).values
        key_bytes = sorted_t.cpu().numpy().tobytes()
        hash_key = hashlib.md5(key_bytes).hexdigest()

        if hash_key in seen_sorted_hashes:
            pbar.close()
            return {"status": "loop_detected", "step": step, "value": current_t.cpu().tolist(), "length": n}
        seen_sorted_hashes.add(hash_key)

        xor_matrix = current_t.view(-1, 1) ^ current_t
        rows, cols = torch.triu_indices(n, n, 1, device=dev)
        current_t = xor_matrix[rows, cols]
        
        pbar.set_postfix(arr_len=f'{current_t.shape[0]:,}', refresh=True)

    pbar.close()
    return {"status": "max_steps_reached", "step": max_steps, "length": current_t.shape[0]}

if __name__ == '__main__':
    # 使用一個能執行多步的陣列來觀察進度條
    initial_arr = list(range(20))
    max_steps_test = 15
    
    print(f"=== 比較三種模擬方法的效能和進度條 ===")
    print(f"初始陣列長度: {len(initial_arr)}, 最大步數: {max_steps_test}\n")

    # --- 1. 測試原始版本 ---
    print("--- 1. 執行原始版本 ---")
    start_time_1 = time.time()
    result_1 = simulate_original_with_progress(initial_arr.copy(), max_steps=max_steps_test)
    end_time_1 = time.time()
    print(f"結果: {result_1}")
    print(f"耗時: {end_time_1 - start_time_1:.4f} 秒\n")
    
    # --- 2. 測試基本 PyTorch 版本 ---
    print("--- 2. 執行基本 PyTorch 版本 ---")
    start_time_2 = time.time()
    result_2 = simulate_pytorch_with_progress(initial_arr.copy(), max_steps=max_steps_test)
    end_time_2 = time.time()
    print(f"結果: {result_2}")
    print(f"耗時: {end_time_2 - start_time_2:.4f} 秒\n")

    # --- 3. 測試進階 PyTorch 版本 ---
    print("--- 3. 執行進階 PyTorch 版本 ---")
    start_time_3 = time.time()
    result_3 = simulate_advanced_with_progress(initial_arr.copy(), max_steps=max_steps_test)
    end_time_3 = time.time()
    print(f"結果: {result_3}")
    print(f"耗時: {end_time_3 - start_time_3:.4f} 秒\n")
