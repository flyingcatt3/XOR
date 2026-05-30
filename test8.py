import math
import sys
import time

# 方法一：調高限制（例如允許最多十萬位數）
sys.set_int_max_str_digits(100000)

# 方法二：完全解除限制（輸入 0 代表無限制，極限測試推薦用這個）
sys.set_int_max_str_digits(0)


def format_time(seconds):
    if seconds >= 1:
        return f"{seconds:.3f} s"
    elif seconds >= 1e-3:
        return f"{seconds * 1000:.2f} ms"
    elif seconds >= 1e-6:
        return f"{seconds * 1_000_000:.2f} µs"
    else:
        return f"{seconds * 1_000_000_000:.2f} ns"


def fwht(a):
    """In-place Fast Walsh-Hadamard Transform (O(M log M))"""
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


def simulate_all_knowing(arr, max_steps=1000, max_len=10**100):
    """
    結合 FWHT 降維打擊與所有進階終止/觀測條件的終極版本 (test8)
    """
    start_time = time.time()

    # [終止條件 1] 空陣列防護 (Empty Array)
    if not arr:
        return {
            "status": "empty_array",
            "step": 0,
            "length": 0,
            "elapsed_sec": time.time() - start_time,
        }

    N = len(arr)

    # 長度宿命論：若 N=3，長度將永遠是 3，我們可以特別標記這個數學奇異點
    is_length_fixed_point = N == 3

    max_val = max(arr)
    initial_msb = max_val.bit_length()

    M = 1 if max_val == 0 else 1 << initial_msb
    C = [0] * M
    for x in arr:
        C[x] += 1

    seen_hashes = {}
    msb_collapse_history = []  # 記錄勢能崩潰的軌跡

    for step in range(max_steps):
        # 尋找當前陣列的最大值 (頻率大於 0 的最大 index)
        current_max = next((i for i in range(M - 1, -1, -1) if C[i] > 0), 0)
        current_msb = current_max.bit_length()

        # [觀測條件] MSB 勢能崩潰 (MSB Energy Collapse)
        # 如果最高有效位元下降了，代表系統的複雜度正在坍縮
        if step > 0 and current_msb < initial_msb:
            if (
                not msb_collapse_history
                or msb_collapse_history[-1]["msb"] != current_msb
            ):
                msb_collapse_history.append(
                    {
                        "step": step,
                        "msb_dropped_to": current_msb,
                        "max_val": current_max,
                    }
                )

        unique_values = sum(1 for count in C if count > 0)

        # [終止條件 2] 全零滅絕狀態 (Zero Extinction)
        if unique_values == 1 and C[0] == N:
            return {
                "status": "zero_extinction",
                "step": step,
                "length": N,
                "msb_history": msb_collapse_history,
                "note": "系統已失去所有能量，完全歸零",
                "elapsed_sec": time.time() - start_time,
            }

        # [終止條件 3] 常數循環 (Constant Value Loop)
        if unique_values == 1 and step > 0:
            val = next(i for i, count in enumerate(C) if count > 0)
            return {
                "status": "constant_value_loop",
                "step": step,
                "value": val,
                "length": N,
                "msb_history": msb_collapse_history,
                "elapsed_sec": time.time() - start_time,
            }

        # 基本終止條件
        if N <= 1:
            return {
                "status": "converged",
                "step": step,
                "length": N,
                "msb_history": msb_collapse_history,
                "elapsed_sec": time.time() - start_time,
            }

        if N > max_len:
            return {
                "status": "too_large",
                "step": step,
                "length": N,
                "msb_history": msb_collapse_history,
                "elapsed_sec": time.time() - start_time,
            }

        # 循環偵測
        state_key = hash(tuple(C))
        if state_key in seen_hashes:
            first_seen = seen_hashes[state_key]
            return {
                "status": "loop_detected",
                "step": step,
                "period": step - first_seen,
                "length": N,
                "is_length_fixed_point": is_length_fixed_point,
                "msb_history": msb_collapse_history,
                "elapsed_sec": time.time() - start_time,
            }
        seen_hashes[state_key] = step

        # --- FWHT 卷積運算 ---
        A = C.copy()
        fwht(A)
        for i in range(M):
            A[i] = A[i] * A[i]
        fwht(A)

        next_C = [0] * M
        for i in range(M):
            A[i] //= M
            next_C[i] = A[i] // 2

        next_C[0] -= N // 2

        C = next_C
        N = (N * (N - 1)) // 2

    return {
        "status": "max_steps_reached",
        "step": max_steps,
        "length": N,
        "msb_history": msb_collapse_history,
        "elapsed_sec": time.time() - start_time,
    }


if __name__ == "__main__":
    test_cases = [
        ("TC01: 空陣列", []),
        ("TC02: 單一元素", [0]),
        ("TC03: 大數值", [99999]),
        ("TC04: 基礎收斂", [0, 0]),
        ("TC05: 常數全 1", [1, 1, 1, 1]),
        ("TC06: 勢能崩潰", [255, 255, 255, 255]),
        ("TC07: 混合歸零", [7, 7, 7, 0, 0, 0]),
        ("TC08: 簡單震盪", [15, 15, 0]),
        ("TC09: 奇異點 123", [1, 2, 3]),
        ("TC10: 奇異點平移", [4, 8, 12]),
        ("TC11: 奇異點亂數", [10, 20, 30]),
        ("TC12: 2-bit 空間", [0, 1, 2, 3]),
        ("TC13: 3-bit 空間", [0, 1, 2, 3, 4, 5, 6, 7]),
        ("TC14: 2 的次方", [1, 2, 4, 8, 16]),
        ("TC15: 4-bit 空間", list(range(16))),
        ("TC16: 6-bit 空間", list(range(64))),
        ("TC17: 大數值基底", [1024, 2048, 4096, 8192, 16384]),
        ("TC18: 質數陣列", [3, 5, 7, 11, 13, 17, 19, 23]),
        ("TC19: 高頻率重覆", [1] * 100 + [2] * 100 + [3] * 100),
        ("TC20: 百連殺 (極限壓力測試)", list(range(100))),
    ]

    print("=== 執行 20 個綜合測試案例 ===")

    for name, arr in test_cases:
        # 設定最大步數 15 步，避免印出太多資訊
        res = simulate_all_knowing(arr, max_steps=15, max_len=10**1000)

        # 使用自訂的 format_time 將科學記號轉為直觀的時間字串
        formatted_time = format_time(res["elapsed_sec"])

        # 處理超級大數字的長度顯示
        length_str = str(res["length"])
        if len(length_str) > 15:
            length_str = f"~ {length_str[0]}.{length_str[1:3]}e+{len(length_str) - 1} (長度已達 {len(length_str)} 位數)"

        print(f"\n{name}")
        # 直接使用 formatted_time，不需要加 :.3f 或 ms
        print(
            f"狀態: {res['status']:<20} | 執行步數: {res['step']:<2} | 耗時: {formatted_time}"
        )
        print(f"最終長度: {length_str}")

        if "is_length_fixed_point" in res and res["is_length_fixed_point"]:
            print("=> ⚠️ 偵測到 N=3 數學奇異點！")

        if "msb_history" in res and res["msb_history"]:
            print(f"=> 📉 偵測到勢能崩潰次數: {len(res['msb_history'])}")
