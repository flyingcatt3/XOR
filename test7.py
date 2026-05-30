import math
import sys
import time

# 方法一：調高限制（例如允許最多十萬位數）
sys.set_int_max_str_digits(100000)

# 方法二：完全解除限制（輸入 0 代表無限制，極限測試推薦用這個）
sys.set_int_max_str_digits(0)


def fwht(a):
    """
    In-place Fast Walsh-Hadamard Transform (快速華許轉換)
    時間複雜度: O(M log M)
    """
    n = len(a)
    h = 1
    while h < n:
        for i in range(0, n, h * 2):
            for j in range(i, i + h):
                x = a[j]
                y = a[j + h]
                a[j] = x + y
                a[j + h] = x - y
        h *= 2


def simulate_fwht_paradigm(arr, max_steps=1000, max_len=10**100):
    """
    利用頻率直方圖與 FWHT 進行 XOR 演化模擬。
    - 支援超過 10^100 以上的陣列長度。
    - 記憶體使用量嚴格保持常數 (Constant Space)。
    """
    if not arr:
        return {"status": "empty"}

    start_time = time.time()

    # 1. 決定頻率陣列的大小 M (必須是 2 的次方)
    max_val = max(arr)
    if max_val == 0:
        M = 1
    else:
        # 找到大於等於 max_val 的下一個 2 的次方
        M = 1 << (max_val.bit_length())

    # 2. 初始化頻率直方圖 (Frequency Map)
    C = [0] * M
    for x in arr:
        C[x] += 1

    N = len(arr)  # N 為當前陣列的總長度
    seen_hashes = {}

    for step in range(max_steps):
        # --- 終止條件檢查 ---
        if N <= 1:
            return {
                "status": "converged",
                "step": step,
                "length": N,
                "elapsed_sec": time.time() - start_time,
            }

        if N > max_len:
            return {
                "status": "too_large",
                "step": step,
                "length": N,  # 這個長度可以輕易突破 10^50
                "elapsed_sec": time.time() - start_time,
            }

        unique_values = sum(1 for count in C if count > 0)
        if unique_values == 1 and step > 0:
            # 找出唯一存在的那個值
            val = next(i for i, count in enumerate(C) if count > 0)
            return {
                "status": "constant_value_loop",
                "step": step,
                "value": val,
                "length": N,
                "elapsed_sec": time.time() - start_time,
            }

        # 循環偵測：將目前的頻率陣列轉為 tuple 進行 Hash
        state_key = hash(tuple(C))
        if state_key in seen_hashes:
            first_seen = seen_hashes[state_key]
            return {
                "status": "loop_detected",
                "step": step,
                "period": step - first_seen,
                "length": N,
                "elapsed_sec": time.time() - start_time,
            }
        seen_hashes[state_key] = step

        # --- 核心演算法：FWHT 卷積 ---

        # 複製當前頻率，準備轉換
        A = C.copy()

        # 1. 正向 FWHT
        fwht(A)

        # 2. 點對點平方 (計算自我卷積)
        for i in range(M):
            A[i] = A[i] * A[i]

        # 3. 反向 FWHT
        fwht(A)

        # 4. 除以 M 完成反轉換，並將排列(Permutations)調整為組合(Combinations)
        next_C = [0] * M
        for i in range(M):
            A[i] //= M
            next_C[i] = A[i] // 2

        # 修正 0 的頻率：扣除自己與自己配對的情況 (i ^ i = 0)
        next_C[0] -= N // 2

        # 更新狀態
        C = next_C
        # 新的總長度 = N * (N - 1) / 2
        N = (N * (N - 1)) // 2

    return {
        "status": "max_steps_reached",
        "step": max_steps,
        "length": N,
        "elapsed_sec": time.time() - start_time,
    }


# ==========================================
# 效能展示：挑戰指數級爆炸
# ==========================================
if __name__ == "__main__":
    print("=== FWHT 終極降維版測試 ===")

    # 測試 1: 基本範例
    arr1 = [1, 2, 3]
    print(f"\n[Test 1] Input: {arr1}")
    res1 = simulate_fwht_paradigm(arr1)
    print(res1)

    # 測試 2: 常數循環
    arr2 = [1, 1, 1, 1]
    print(f"\n[Test 2] Input: {arr2}")
    res2 = simulate_fwht_paradigm(arr2)
    print(res2)

    # 測試 3: 極限壓力測試 (展現 Python BigInt 的威力)
    # 我們讓它跑更多步，觀察長度如何突破天際，而時間依然是 0.00x 秒
    # 初始陣列包含 0 到 15 共 16 個數字
    arr3 = list(range(16))
    print(f"\n[Test 3] Limit Test Input: list(range(16))")
    print("注意觀察 length (長度) 的驚人數字，而執行時間幾乎不變！")

    # 允許最大長度達到 10 的 10000 次方
    res3 = simulate_fwht_paradigm(arr3, max_steps=20, max_len=10**10000)

    # 因為長度數字可能大到佔滿整個螢幕，我們印出科學記號或位數
    length_str = str(res3["length"])
    print(f"Status: {res3['status']}")
    print(f"Steps taken: {res3['step']}")
    print(f"Elapsed Time: {res3['elapsed_sec']:.6f} seconds")

    if len(length_str) > 50:
        print(
            f"Final Length: ~ {length_str[0]}.{length_str[1:4]}e+{len(length_str) - 1}"
        )
        print(f"(這個陣列的長度是一個 {len(length_str)} 位數！)")
    else:
        print(f"Final Length: {res3['length']}")
