import torch
import hashlib
from itertools import combinations
import time
import gc
from collections import deque
import psutil
import os

# ==================== ORIGINAL VERSION ====================

def hash_array_original(arr):
    s = ','.join(map(str, arr))
    return hashlib.md5(s.encode()).hexdigest()

def simulate_original(arr, max_steps=1000, max_len=10**6):
    """Original pure Python implementation for comparison"""
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

# ==================== PYTORCH BASIC VERSION ====================

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

# ==================== PYTORCH ADVANCED VERSION ====================

def simulate_pytorch_advanced(arr, max_steps=1000, max_len=10**6, device="cuda"):
    """Advanced PyTorch version with optimized loop detection"""
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

# ==================== MEMORY OPTIMIZED VERSION ====================

def simulate_memory_optimized(arr, max_steps=1000, max_len=10**6, device="cuda", chunk_size=None):
    """
    Memory-optimized version with several improvements:
    1. Chunked processing for large arrays
    2. In-place operations where possible
    3. Explicit memory cleanup
    4. Reduced intermediate tensor creation
    5. Optimized loop detection with rolling hash window
    """
    if device == "cuda" and torch.cuda.is_available():
        dev = torch.device("cuda")
    else:
        dev = torch.device("cpu")

    # 用 deque 限制記憶體中的 hash 數量
    max_hash_history = 1000  # 只保留最近1000個hash
    seen_hashes = deque(maxlen=max_hash_history)
    seen_hashes_set = set()

    current_t = torch.tensor(arr, dtype=torch.long, device=dev)

    for step in range(max_steps):
        n = current_t.shape[0]
        
        # 早期終止條件
        if n == 1:
            return {"status": "converged", "step": step, "value": current_t.item(), "length": 1}
        if n > max_len:
            return {"status": "too_large", "step": step, "length": n}
        
        # 檢查所有元素是否相同（在GPU上操作）
        if step > 0:
            unique_count = torch.unique(current_t).shape[0]
            if unique_count == 1:
                return {"status": "constant_value_loop", "step": step, 
                       "value": current_t[0].item(), "length": n}

        # 優化的循環檢測 - 使用滾動窗口
        if step > 0:  # 跳過第一步
            # 在GPU上排序並直接計算hash
            sorted_t = torch.sort(current_t).values
            
            # 使用更高效的hash方法
            key_bytes = sorted_t.cpu().numpy().tobytes()
            hash_key = hashlib.blake2b(key_bytes, digest_size=16).digest()  # 更快的hash
            
            if hash_key in seen_hashes_set:
                return {
                    "status": "loop_detected",
                    "step": step,
                    "value": current_t.cpu().tolist(),
                    "length": n
                }
            
            # 管理hash歷史記錄的記憶體
            if len(seen_hashes) == max_hash_history:
                old_hash = seen_hashes[0]  # deque會自動移除
                seen_hashes_set.discard(old_hash)
            
            seen_hashes.append(hash_key)
            seen_hashes_set.add(hash_key)

        # 核心計算 - 根據陣列大小選擇策略
        if chunk_size is None:
            # 動態決定chunk大小
            if dev.type == 'cuda':
                available_memory = torch.cuda.get_device_properties(dev).total_memory
            else:
                available_memory = 8e9  # 假設8GB CPU RAM
            estimated_memory_needed = n * n * 8  # 8 bytes per int64
            
            if estimated_memory_needed > available_memory * 0.7:  # 使用70%記憶體上限
                chunk_size = max(100, int((available_memory * 0.7 / (n * 8)) ** 0.5))
            else:
                chunk_size = n

        if n <= chunk_size:
            # 小陣列：直接計算
            xor_matrix = current_t.view(-1, 1) ^ current_t
            rows, cols = torch.triu_indices(n, n, 1, device=dev)
            current_t = xor_matrix[rows, cols]
        else:
            # 大陣列：分塊處理
            current_t = _chunked_xor_computation(current_t, chunk_size, dev)

        # 定期清理記憶體
        if step % 10 == 0:
            if dev.type == 'cuda':
                torch.cuda.empty_cache()
            gc.collect()

    return {
        "status": "max_steps_reached",
        "step": max_steps,
        "length": current_t.shape[0]
    }

def _chunked_xor_computation(tensor, chunk_size, device):
    """
    分塊計算XOR組合，避免創建巨大的矩陣
    """
    n = tensor.shape[0]
    results = []
    
    for i in range(0, n, chunk_size):
        end_i = min(i + chunk_size, n)
        chunk_i = tensor[i:end_i]
        
        # 計算chunk內的組合
        if len(chunk_i) > 1:
            xor_matrix = chunk_i.view(-1, 1) ^ chunk_i
            rows, cols = torch.triu_indices(len(chunk_i), len(chunk_i), 1, device=device)
            results.append(xor_matrix[rows, cols])
        
        # 計算chunk與後續元素的組合
        for j in range(end_i, n, chunk_size):
            end_j = min(j + chunk_size, n)
            chunk_j = tensor[j:end_j]
            
            # 使用broadcasting計算所有組合
            xor_result = chunk_i.view(-1, 1) ^ chunk_j.view(1, -1)
            results.append(xor_result.flatten())
    
    return torch.cat(results) if results else torch.tensor([], dtype=torch.long, device=device)

# ==================== ULTRA MEMORY EFFICIENT VERSION ====================

def simulate_ultra_memory_efficient(arr, max_steps=1000, max_len=10**6, device="cuda"):
    """
    超級記憶體高效版本：
    1. 使用生成器避免儲存中間結果
    2. 流式處理大型陣列
    3. 最小化GPU記憶體使用
    """
    if device == "cuda" and torch.cuda.is_available():
        dev = torch.device("cuda")
    else:
        dev = torch.device("cpu")

    # 更激進的記憶體管理
    seen_hashes = set()
    max_seen = 500  # 更小的hash歷史

    current_list = arr.copy()  # 開始時保持在CPU

    for step in range(max_steps):
        n = len(current_list)
        
        if n == 1:
            return {"status": "converged", "step": step, "value": current_list[0], "length": 1}
        if n > max_len:
            return {"status": "too_large", "step": step, "length": n}
        
        # 簡化的循環檢測
        if step > 0 and len(set(current_list)) == 1:
            return {"status": "constant_value_loop", "step": step, 
                   "value": current_list[0], "length": n}

        # 輕量級循環檢測
        if step > 2:  # 減少檢測頻率
            sorted_key = tuple(sorted(current_list))
            hash_key = hash(sorted_key)  # 使用內建hash，更快但可能有碰撞
            
            if hash_key in seen_hashes:
                return {"status": "loop_detected", "step": step, 
                       "value": current_list, "length": n}
            
            seen_hashes.add(hash_key)
            if len(seen_hashes) > max_seen:
                seen_hashes.clear()  # 清空而不是漸進式移除

        # 根據大小選擇處理方式
        if n < 1000:
            # 小陣列：使用PyTorch
            current_t = torch.tensor(current_list, dtype=torch.long, device=dev)
            xor_matrix = current_t.view(-1, 1) ^ current_t
            rows, cols = torch.triu_indices(n, n, 1, device=dev)
            next_t = xor_matrix[rows, cols]
            current_list = next_t.cpu().tolist()
        else:
            # 大陣列：使用生成器和批次處理
            current_list = list(_memory_efficient_xor_generator(current_list, batch_size=1000, device=dev))

        # 頻繁清理
        if step % 5 == 0 and dev.type == 'cuda':
            torch.cuda.empty_cache()

    return {"status": "max_steps_reached", "step": max_steps, "length": len(current_list)}

def _memory_efficient_xor_generator(arr, batch_size=1000, device="cuda"):
    """
    生成器版本的XOR計算，節省記憶體
    """
    n = len(arr)
    
    for i in range(n):
        batch = []
        for j in range(i + 1, n):
            batch.append(arr[i] ^ arr[j])
            
            if len(batch) >= batch_size:
                # 批次處理
                yield from batch
                batch = []
        
        # 處理剩餘的batch
        if batch:
            yield from batch

# ==================== BENCHMARKING FUNCTIONS ====================

def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # MB

def benchmark_all_versions(initial_arr, max_steps=5):
    """
    Compare all versions for performance and memory usage
    """
    print(f"Benchmarking with array size: {len(initial_arr)}, max_steps: {max_steps}")
    print("=" * 80)
    
    versions = {
        'original': simulate_original,
        'pytorch_basic': simulate_pytorch,
        'pytorch_advanced': simulate_pytorch_advanced,
        'memory_optimized': simulate_memory_optimized,
        'ultra_memory_efficient': simulate_ultra_memory_efficient
    }
    
    results = {}
    
    for name, func in versions.items():
        # Clear memory before each test
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        mem_before = get_memory_usage()
        start_time = time.time()
        
        try:
            result = func(initial_arr.copy(), max_steps=max_steps)
            end_time = time.time()
            mem_after = get_memory_usage()
            
            results[name] = {
                'time': end_time - start_time,
                'memory_delta': mem_after - mem_before,
                'result': result,
                'success': True
            }
            
        except Exception as e:
            end_time = time.time()
            results[name] = {
                'time': end_time - start_time,
                'memory_delta': 0,
                'result': f"Error: {str(e)}",
                'success': False
            }
    
    # Print results
    for name, data in results.items():
        print(f"\n{name.upper().replace('_', ' ')}:")
        if data['success']:
            print(f"  Time: {data['time']:.6f} seconds")
            print(f"  Memory delta: {data['memory_delta']:.2f} MB")
            print(f"  Final status: {data['result']['status']}")
            print(f"  Final length: {data['result']['length']}")
            if data['result']['status'] == 'converged':
                print(f"  Final value: {data['result']['value']}")
        else:
            print(f"  FAILED: {data['result']}")
    
    return results

def recommend_function(array_size, available_gpu_memory_gb=None):
    """
    Recommend the best function based on array size and available memory
    """
    print(f"\nRecommendation for array size {array_size}:")
    
    if array_size < 100:
        print("→ Use simulate_original or simulate_pytorch_advanced")
        print("  Small arrays don't benefit much from optimization")
        
    elif array_size < 1000:
        print("→ Use simulate_pytorch_advanced")
        print("  Good balance of performance and simplicity")
        
    elif array_size < 5000:
        if available_gpu_memory_gb and available_gpu_memory_gb >= 4:
            print("→ Use simulate_memory_optimized")
            print("  Optimized for medium arrays with sufficient GPU memory")
        else:
            print("→ Use simulate_ultra_memory_efficient")
            print("  Memory-constrained environment")
            
    else:
        print("→ Use simulate_ultra_memory_efficient")
        print("  Large arrays require aggressive memory optimization")

# ==================== MAIN EXECUTION ====================

if __name__ == '__main__':
    # Test with different array sizes
    test_sizes = [8, 10, 12]
    
    for size in test_sizes:
        print(f"\n{'='*80}")
        print(f"TESTING ARRAY SIZE: {size}")
        print(f"{'='*80}")
        
        initial_arr = list(range(size))
        results = benchmark_all_versions(initial_arr, max_steps=3)
        
        # Show speedup comparisons
        if results['original']['success'] and results['pytorch_advanced']['success']:
            speedup_advanced = results['original']['time'] / results['pytorch_advanced']['time']
            print(f"\nPyTorch Advanced speedup: {speedup_advanced:.2f}x")
            
        if results['original']['success'] and results['memory_optimized']['success']:
            speedup_optimized = results['original']['time'] / results['memory_optimized']['time']
            print(f"Memory Optimized speedup: {speedup_optimized:.2f}x")
            
        if results['original']['success'] and results['ultra_memory_efficient']['success']:
            speedup_ultra = results['original']['time'] / results['ultra_memory_efficient']['time']
            print(f"Ultra Memory Efficient speedup: {speedup_ultra:.2f}x")
        
        # Get recommendation
        recommend_function(size)
        
        print("\n" + "-" * 80)
    
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print("Available functions:")
    print("1. simulate_original - Pure Python baseline")
    print("2. simulate_pytorch - Basic PyTorch acceleration")  
    print("3. simulate_pytorch_advanced - Advanced PyTorch with optimized loop detection")
    print("4. simulate_memory_optimized - Balanced memory optimization")
    print("5. simulate_ultra_memory_efficient - Extreme memory efficiency")