import hashlib
import sys
import time
from collections import deque

# 方法一：調高限制（例如允許最多十萬位數）
sys.set_int_max_str_digits(100000)

# 方法二：完全解除限制（輸入 0 代表無限制，極限測試推薦用這個）
sys.set_int_max_str_digits(0)
# ==========================================
# 核心引擎：負責觀測、生命週期與終止條件
# ==========================================


class EvolutionEngine:
    def __init__(self, max_steps=1000, max_len=10**100, max_history=10000):
        self.max_steps = max_steps
        self.max_len = max_len
        self.max_history = max_history

    def _secure_hash(self, freq_map):
        """使用 blake2b 進行穩定、低碰撞的狀態 Hash"""
        # 將頻率陣列轉換為 bytes，過濾掉尾部的 0 以標準化狀態
        active_freqs = [str(x) for x in freq_map]
        byte_data = b",".join(x.encode() for x in active_freqs)
        return hashlib.blake2b(byte_data, digest_size=16).hexdigest()

    def run(self, initial_arr, strategy_fn):
        """
        執行模擬主迴圈
        :param initial_arr: 初始整數陣列
        :param strategy_fn: 負責計算下一步頻率直方圖的策略函數
        """
        start_time = time.time()

        if not initial_arr:
            return {
                "status": "empty_array",
                "step": 0,
                "length": 0,
                "elapsed_sec": time.time() - start_time,
            }

        # 初始化：將輸入陣列轉換為頻率直方圖
        max_val = max(initial_arr)
        initial_msb = max_val.bit_length()
        M = 1 if max_val == 0 else 1 << initial_msb

        freq_map = [0] * M
        for x in initial_arr:
            freq_map[x] += 1

        N = len(initial_arr)

        # 軌跡追蹤：使用 deque + set 實作滾動歷史，修復 test5 的 clear() 缺陷
        seen_hashes_set = set()
        seen_hashes_queue = deque()

        msb_collapse_history = []

        # 修復 test1 的盲區：在迴圈開始前，必須先將初始狀態加入觀測歷史
        initial_hash = self._secure_hash(freq_map)
        seen_hashes_set.add(initial_hash)
        seen_hashes_queue.append((initial_hash, 0))

        for step in range(1, self.max_steps + 1):
            current_max = next(
                (i for i in range(len(freq_map) - 1, -1, -1) if freq_map[i] > 0), 0
            )
            current_msb = current_max.bit_length()

            # 觀測：MSB 勢能崩潰
            if current_msb < initial_msb:
                if (
                    not msb_collapse_history
                    or msb_collapse_history[-1]["msb_dropped_to"] != current_msb
                ):
                    msb_collapse_history.append(
                        {"step": step, "msb_dropped_to": current_msb}
                    )

            unique_values = sum(1 for count in freq_map if count > 0)

            # 終止條件：全零滅絕
            if unique_values == 1 and freq_map[0] == N:
                return self._build_result(
                    "zero_extinction", step, N, start_time, msb_collapse_history
                )

            # 終止條件：常數循環 (例如 [1,1,1,1] -> 下一步必為全零)
            if unique_values == 1:
                return self._build_result(
                    "constant_value_loop", step, N, start_time, msb_collapse_history
                )

            # 終止條件：收斂與長度超載
            if N <= 1:
                return self._build_result(
                    "converged", step, N, start_time, msb_collapse_history
                )
            if N > self.max_len:
                return self._build_result(
                    "too_large", step, N, start_time, msb_collapse_history
                )

            # 執行策略：計算下一步 (這就是範式轉移的切入點)
            freq_map, N = strategy_fn(freq_map, N)

            # 循環偵測：修復 test7/8 依賴不穩定 hash() 的問題
            state_key = self._secure_hash(freq_map)
            if state_key in seen_hashes_set:
                # 找出第一次出現的步數計算週期
                first_seen = next(s for k, s in seen_hashes_queue if k == state_key)
                res = self._build_result(
                    "loop_detected", step, N, start_time, msb_collapse_history
                )
                res["period"] = step - first_seen
                res["is_length_fixed_point"] = N == 3  # 數學不動點標記
                return res

            # 更新歷史軌跡 (滾動淘汰)
            seen_hashes_set.add(state_key)
            seen_hashes_queue.append((state_key, step))
            if len(seen_hashes_queue) > self.max_history:
                old_hash, _ = seen_hashes_queue.popleft()
                seen_hashes_set.remove(old_hash)

        return self._build_result(
            "max_steps_reached", self.max_steps, N, start_time, msb_collapse_history
        )

    def _build_result(self, status, step, length, start_time, msb_history):
        return {
            "status": status,
            "step": step,
            "length": length,
            "msb_history": msb_history,
            "elapsed_sec": time.time() - start_time,
        }


# ==========================================
# 演算法策略 (Strategy Pattern)
# ==========================================


def fwht_strategy(freq_map, current_N):
    """
    最優解：利用 Fast Walsh-Hadamard Transform 進行 O(M log M) 的 XOR 卷積。
    """
    M = len(freq_map)
    A = freq_map.copy()

    # 1. 正向 FWHT
    h = 1
    while h < M:
        for i in range(0, M, h * 2):
            for j in range(i, i + h):
                x, y = A[j], A[j + h]
                A[j], A[j + h] = x + y, x - y
        h *= 2

    # 2. 點對點平方 (自我卷積)
    for i in range(M):
        A[i] *= A[i]

    # 3. 反向 FWHT
    h = 1
    while h < M:
        for i in range(0, M, h * 2):
            for j in range(i, i + h):
                x, y = A[j], A[j + h]
                A[j], A[j + h] = x + y, x - y
        h *= 2

    # 4. 正規化與組合數調整
    next_freq = [0] * M
    for i in range(M):
        next_freq[i] = (A[i] // M) // 2

    # 扣除自我配對 (i ^ i = 0)
    next_freq[0] -= current_N // 2
    next_N = (current_N * (current_N - 1)) // 2

    return next_freq, next_N


# ==========================================
# 執行介面與測試
# ==========================================


def format_time(seconds):
    if seconds >= 1:
        return f"{seconds:.3f} s"
    elif seconds >= 1e-3:
        return f"{seconds * 1000:.3f} ms"
    elif seconds >= 1e-6:
        return f"{seconds * 1_000_000:.2f} µs"
    else:
        return f"{seconds * 1_000_000_000:.2f} ns"


if __name__ == "__main__":
    # 建立演化引擎實體
    engine = EvolutionEngine(max_steps=15, max_history=5000)

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

    print("=== 執行 20 個綜合測試案例 (統一重構版) ===")

    for name, arr in test_cases:
        # 呼叫新版引擎，並注入 fwht_strategy 策略
        res = engine.run(arr, strategy_fn=fwht_strategy)

        formatted_time = format_time(res["elapsed_sec"])
        length_str = str(res["length"])

        if len(length_str) > 15:
            length_str = f"~ {length_str[0]}.{length_str[1:3]}e+{len(length_str) - 1} (長度已達 {len(length_str)} 位數)"

        print(f"\n{name}")
        print(
            f"狀態: {res['status']:<20} | 執行步數: {res['step']:<2} | 耗時: {formatted_time}"
        )
        print(f"最終長度: {length_str}")

        if "is_length_fixed_point" in res and res["is_length_fixed_point"]:
            print("=> ⚠️ 偵測到 N=3 數學奇異點！")

        if "msb_history" in res and res["msb_history"]:
            print(f"=> 📉 偵測到勢能崩潰次數: {len(res['msb_history'])}")
