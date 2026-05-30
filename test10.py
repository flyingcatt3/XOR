import hashlib
import sys
import time
from collections import deque
from typing import Dict

"""🛠️ `test10.py` 核心修復重點：

1. **修復 Off-by-One (語義對齊)：** 將初始狀態的檢查移至 Step 0，確保 `step` 的數值等同於「實際呼叫演化策略的次數」。
2. **$O(1)$ Loop 偵測：** 使用 `seen_hash_to_step: dict` 取代線性掃描，回歸 $O(1)$ 的查詢效能。
3. **實踐真正的尾部過濾：** 在 `_secure_hash` 中用 `while` 迴圈濾除尾端多餘的 0，既縮小 Hash 範圍又確保 BigInt 安全。
4. **$O(M)$ 掃描優化：** 將最大值的尋找交給 `fwht_strategy` 一併回傳，避免引擎層重複掃描。
5. **單一職責原則 (SRP)：** 將 $N_{next} = \frac{N(N-1)}{2}$ 的數學邏輯抽離策略函數，提升到 Engine 層處理。
"""


# 方法一：調高限制（例如允許最多十萬位數）
sys.set_int_max_str_digits(100000)

# 方法二：完全解除限制（輸入 0 代表無限制，極限測試推薦用這個）
sys.set_int_max_str_digits(0)
# ==========================================
# 核心引擎：負責觀測、生命週期與終止條件 (test10)
# ==========================================


class EvolutionEngine:
    def __init__(self, max_steps=1000, max_len=10**100, max_history=10000):
        self.max_steps = max_steps
        self.max_len = max_len
        self.max_history = max_history

    def _secure_hash(self, freq_map):
        """使用 blake2b 進行穩定、低碰撞的狀態 Hash，並支援 Python BigInt"""
        # 實踐註解承諾：過濾掉尾部的 0 以標準化狀態，並減少字串轉換開銷
        last_idx = len(freq_map) - 1
        while last_idx >= 0 and freq_map[last_idx] == 0:
            last_idx -= 1

        if last_idx < 0:
            return hashlib.blake2b(b"", digest_size=16).hexdigest()

        # 由於 freq_map 內的數字可能遠大於 64-bit (BigInt)，不可使用 struct.pack
        byte_data = b",".join(str(x).encode("ascii") for x in freq_map[: last_idx + 1])
        return hashlib.blake2b(byte_data, digest_size=16).hexdigest()

    def run(self, initial_arr, strategy_fn):
        """
        執行模擬主迴圈
        :param initial_arr: 初始整數陣列
        :param strategy_fn: 負責計算下一步頻率直方圖的策略函數
        """
        start_time = time.time()

        if not initial_arr:
            return self._build_result("empty_array", 0, 0, start_time, [])

        max_val = max(initial_arr)
        initial_msb = max_val.bit_length()
        M = 1 if max_val == 0 else 1 << initial_msb

        freq_map = [0] * M
        for x in initial_arr:
            freq_map[x] += 1

        N = len(initial_arr)
        msb_collapse_history = []

        # --- [修復 1] 初始狀態檢查 (Step 0) ---
        # 確保 step 的語義等同於「執行 strategy_fn 的次數」
        unique_values = sum(1 for count in freq_map if count > 0)

        if unique_values == 1 and freq_map[0] == N:
            return self._build_result(
                "zero_extinction", 0, N, start_time, msb_collapse_history
            )
        if unique_values == 1:
            return self._build_result(
                "constant_value_loop", 0, N, start_time, msb_collapse_history
            )
        if N <= 1:
            return self._build_result(
                "converged", 0, N, start_time, msb_collapse_history
            )
        if N > self.max_len:
            return self._build_result(
                "too_large", 0, N, start_time, msb_collapse_history
            )

        # --- [修復 2] 軌跡追蹤：O(1) 查詢 ---
        # 明確標註這是一個 key 為字串 (hash)、value 為整數 (step) 的字典
        seen_hash_to_step: Dict[str, int] = {}  # dict 負責 O(1) 查詢
        seen_hashes_queue = deque()  # deque 負責 FIFO 滾動淘汰

        initial_hash = self._secure_hash(freq_map)
        seen_hash_to_step[initial_hash] = 0
        seen_hashes_queue.append((initial_hash, 0))

        # 進入演化迴圈 (Step 1 ~ max_steps)
        for step in range(1, self.max_steps + 1):
            # --- 核心演化 ---
            # [修復 4 & 5] strategy_fn 只負責頻率計算並回傳最大值，N 的更新拉回引擎層
            freq_map, current_max = strategy_fn(freq_map, N)
            N = (N * (N - 1)) // 2

            # --- 循環偵測 O(1) ---
            state_key = self._secure_hash(freq_map)
            if state_key in seen_hash_to_step:
                first_seen = seen_hash_to_step[state_key]
                res = self._build_result(
                    "loop_detected", step, N, start_time, msb_collapse_history
                )
                res["period"] = step - first_seen
                res["is_length_fixed_point"] = N == 3
                return res

            # 歷史更新 (淘汰最舊的狀態)
            seen_hash_to_step[state_key] = step
            seen_hashes_queue.append((state_key, step))
            if len(seen_hashes_queue) > self.max_history:
                old_hash, _ = seen_hashes_queue.popleft()
                seen_hash_to_step.pop(old_hash, None)

            # --- 觀測：MSB 勢能崩潰 ---
            current_msb = current_max.bit_length()
            if current_msb < initial_msb:
                if (
                    not msb_collapse_history
                    or msb_collapse_history[-1]["msb_dropped_to"] != current_msb
                ):
                    msb_collapse_history.append(
                        {"step": step, "msb_dropped_to": current_msb}
                    )

            # --- 演化後狀態檢查 ---
            unique_values = sum(1 for count in freq_map if count > 0)

            if unique_values == 1 and freq_map[0] == N:
                return self._build_result(
                    "zero_extinction", step, N, start_time, msb_collapse_history
                )
            if unique_values == 1:
                return self._build_result(
                    "constant_value_loop", step, N, start_time, msb_collapse_history
                )
            if N <= 1:
                return self._build_result(
                    "converged", step, N, start_time, msb_collapse_history
                )
            if N > self.max_len:
                return self._build_result(
                    "too_large", step, N, start_time, msb_collapse_history
                )

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
    Fast Walsh-Hadamard Transform 策略。
    :return: (新的頻率直方圖, 出現的最大數字 index)
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

    # 4. 正規化與扣除自我配對
    next_freq = [0] * M
    current_max = 0
    for i in range(M):
        next_freq[i] = (A[i] // M) // 2

    next_freq[0] -= current_N // 2

    # 順便找出最大值，避免外部重複 O(M) 掃描
    for i in range(M - 1, -1, -1):
        if next_freq[i] > 0:
            current_max = i
            break

    return next_freq, current_max


# ==========================================
# 執行介面與 20 個綜合測試案例
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

    print("=== 執行 20 個綜合測試案例 (test10 循序執行版) ===")

    # 紀錄總開始時間
    total_start_time = time.time()

    for name, arr in test_cases:
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

    # 計算總耗時並印出結尾標語
    total_elapsed = time.time() - total_start_time
    print("\n" + "=" * 50)
    print(f"✅ 所有測試執行完畢！總耗時: {format_time(total_elapsed)}")
    print("=" * 50)
