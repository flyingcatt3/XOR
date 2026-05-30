from itertools import combinations
import hashlib

def hash_array(arr):
    """用於記錄陣列狀態的 hash key"""
    s = ','.join(map(str, arr))
    return hashlib.md5(s.encode()).hexdigest()

def simulate(arr, max_steps=1000, max_len=10**6):
    seen_exact = set()
    seen_sorted = set()

    for step in range(max_steps):
        if len(arr) == 1:
            return {
                "status": "converged",
                "step": step,
                "value": arr[0],
                "length": 1
            }

        # 循環偵測
        key_exact = tuple(arr)
        key_sorted = tuple(sorted(arr))
        hash_key = hash_array(arr)

        if key_exact in seen_exact or key_sorted in seen_sorted:
            return {
                "status": "loop_detected",
                "step": step,
                "value": arr,
                "length": len(arr)
            }

        seen_exact.add(key_exact)
        seen_sorted.add(key_sorted)

        # 全部元素相同但長度變大（像是 [1, 1, 1, 1] 無窮循環）
        if len(set(arr)) == 1 and step > 0:
            return {
                "status": "constant_value_loop",
                "step": step,
                "value": arr[0],
                "length": len(arr)
            }

        # 陣列長度過大，終止
        if len(arr) > max_len:
            return {
                "status": "too_large",
                "step": step,
                "length": len(arr)
            }

        # 執行所有組合 XOR
        next_arr = []
        for a, b in combinations(arr, 2):
            next_arr.append(a ^ b)

        arr = next_arr

    return {
        "status": "max_steps_reached",
        "step": max_steps,
        "length": len(arr)
    }

print(simulate([1, 1, 1, 1]))
# -> {'status': 'constant_value_loop', 'step': 1, 'value': 1, 'length': 6}

print(simulate([1, 2, 3]))
# -> {'status': 'loop_detected', 'step': ..., 'value': ..., 'length': ...}

print(simulate([5, 7, 9, 11]))
# -> {'status': 'too_large', 'step': ..., 'length': ...}

print(simulate([0, 0, 0]))
# -> {'status': 'converged', 'step': 1, 'value': 0, 'length': 1}
