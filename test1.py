from itertools import combinations

def xor_all_pairs(arr):
    return [a ^ b for a, b in combinations(arr, 2)]

def simulate(arr, max_steps=1000, max_len=10000):
    seen = set()

    for step in range(max_steps):
        # 若所有值都一樣，可視為收斂
        if len(set(arr)) == 1:
            return {
                "status": "converged",
                "step": step,
                "value": arr[0],
                "length": len(arr)
            }

        if len(arr) > max_len:
            return {
                "status": "too_large",
                "step": step,
                "length": len(arr)
            }

        arr = xor_all_pairs(arr)
        key = tuple(sorted(arr))  # 如果順序重要，請改為 tuple(arr)

        if len(arr) <= 1:
            return {
                "status": "single_value",
                "step": step + 1,
                "value": arr[0] if arr else None,
                "length": len(arr)
            }

        if key in seen:
            return {
                "status": "loop_detected",
                "step": step + 1,
                "value": arr,
                "length": len(arr)
            }

        seen.add(key)

    return {
        "status": "max_step_exceeded",
        "step": max_steps,
        "length": len(arr)
    }

# 試試不同陣列
print(simulate([1, 2, 3]))
print(simulate([1, 1, 1, 1]))
print(simulate([5, 7, 9, 11]))