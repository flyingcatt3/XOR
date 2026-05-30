from __future__ import annotations
import hashlib
import math
import time
from collections import deque
from typing import Dict, Optional, Tuple, Any

try:
    import torch
except Exception:  # pragma: no cover
    torch = None


def _select_device(preferred: str = "auto") -> Optional[torch.device]:
    """
    Decide which torch device to use.
    Returns None if torch is unavailable.
    """
    if torch is None:
        return None
    if preferred == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if preferred in ("cpu", "auto"):
        if preferred == "auto" and torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device("cpu")


def _available_memory_bytes(device: Optional[torch.device]) -> float:
    """
    Rough estimate of available memory.
    For CUDA, use free memory. For CPU, assume a conservative cap if psutil is unavailable.
    """
    try:
        import psutil  # local import to avoid hard dependency in environments without it
    except Exception:
        psutil = None

    if device is not None and device.type == "cuda":
        try:
            free_bytes, total_bytes = torch.cuda.mem_get_info(device)
            # keep a safety margin; caller can still apply its own ratio
            return float(free_bytes)
        except Exception:
            props = torch.cuda.get_device_properties(device)
            return float(getattr(props, "total_memory", 8 * 1024**3)) * 0.5

    # CPU path
    if psutil is not None:
        return float(psutil.virtual_memory().available)
    # Fallback: assume 8GB
    return 8.0 * 1024**3


def _hash_numpy_bytes(arr) -> str:
    m = hashlib.blake2b(digest_size=16)
    m.update(arr)
    return m.hexdigest()


def _hash_tensor_exact(t: torch.Tensor) -> str:
    return _hash_numpy_bytes(t.detach().contiguous().cpu().numpy().tobytes())


def _hash_tensor_sorted(t: torch.Tensor) -> str:
    sorted_t = torch.sort(t).values
    return _hash_tensor_exact(sorted_t)

# ---- New: safe hashing for Python list paths ----

def _hash_list_exact(values: list[int]) -> str:
    data = (",".join(map(str, values))).encode()
    return hashlib.blake2b(data, digest_size=16).hexdigest()


def _hash_list_sorted(values: list[int]) -> str:
    data = (",".join(map(str, sorted(values)))).encode()
    return hashlib.blake2b(data, digest_size=16).hexdigest()


def _estimate_next_length(n: int) -> int:
    return n * (n - 1) // 2


def _should_use_broadcast(n: int, device: Optional[torch.device]) -> bool:
    """
    Heuristic: broadcast XOR matrix if small enough (n^2 fits comfortably).
    """
    if device is None:
        return n <= 256  # pure python fallback will be used anyway
    # Estimate memory ~ n^2 elements * 8 bytes
    bytes_needed = n * n * 8
    avail = _available_memory_bytes(device)
    return bytes_needed <= 0.35 * avail  # 35% of available


def _xor_pairs_broadcast(current_t: torch.Tensor, device: torch.device) -> torch.Tensor:
    n = current_t.shape[0]
    xor_matrix = current_t.view(-1, 1) ^ current_t
    rows, cols = torch.triu_indices(n, n, 1, device=device)
    return xor_matrix[rows, cols]


def _xor_pairs_chunked(tensor: torch.Tensor, chunk_size: int, device: torch.device) -> torch.Tensor:
    """
    Compute all pairwise XORs by chunks to avoid constructing the full n x n matrix.
    Returns a flat tensor of length n*(n-1)/2 on the same device.
    """
    n = tensor.shape[0]
    results = []

    for i in range(0, n, chunk_size):
        end_i = min(i + chunk_size, n)
        chunk_i = tensor[i:end_i]

        if chunk_i.shape[0] > 1:
            xor_matrix = chunk_i.view(-1, 1) ^ chunk_i
            rows, cols = torch.triu_indices(chunk_i.shape[0], chunk_i.shape[0], 1, device=device)
            results.append(xor_matrix[rows, cols])

        for j in range(end_i, n, chunk_size):
            end_j = min(j + chunk_size, n)
            chunk_j = tensor[j:end_j]
            xor_result = chunk_i.view(-1, 1) ^ chunk_j.view(1, -1)
            results.append(xor_result.flatten())

    if not results:
        return torch.empty(0, dtype=torch.long, device=device)
    return torch.cat(results)


def _xor_pairs_python(arr: list[int]) -> list[int]:
    out = []
    n = len(arr)
    for i in range(n):
        ai = arr[i]
        for j in range(i + 1, n):
            out.append(ai ^ arr[j])
    return out


def simulate_xor_optimized(
    arr: list[int],
    *,
    max_steps: int = 1000,
    max_len: int = 10**10,
    device: str = "auto",
    max_memory_ratio: float = 0.7,
    stability_patience: int = 3,
) -> Dict[str, Any]:
    """
    以效能與記憶體為優先，模擬兩兩 XOR 的序列演化。

    停止條件與輸出資訊：
    - 收斂：長度為 1，輸出固定值
    - 常數循環：所有值相同（且步數 > 0）
    - 循環：
        - exact（完全相同序列）
        - multiset（排序後相同，多重集合重複）
      皆輸出週期與第一次出現步數
    - 長度過大：長度超過 max_len（預設 1e10）
    - 穩定規律：連續 stability_patience 步 unique_count、min、max 皆未變
    - 最多執行 max_steps 步

    參數：
    - device: 'auto' | 'cpu' | 'cuda'
    - max_memory_ratio: 單步分配上限比例（對可用記憶體）
    - stability_patience: 判定穩定的容忍步數
    """
    start_time = time.time()

    if any(x < 0 for x in arr):
        raise ValueError("所有元素必須為非負整數")

    dev = _select_device(device)

    seen_exact: Dict[str, int] = {}
    seen_sorted: Dict[str, int] = {}

    recent_unique = deque(maxlen=stability_patience)
    recent_min = deque(maxlen=stability_patience)
    recent_max = deque(maxlen=stability_patience)

    if dev is None:  # pure python fallback
        current_list = list(map(int, arr))
        for step in range(max_steps):
            n = len(current_list)
            if n == 1:
                return {
                    "status": "converged",
                    "step": step,
                    "value": current_list[0],
                    "length": 1,
                    "elapsed_sec": time.time() - start_time,
                }
            if n > max_len:
                return {
                    "status": "too_large",
                    "step": step,
                    "length": n,
                    "elapsed_sec": time.time() - start_time,
                }

            # Loop/regularity detection
            if step > 0:
                uniq = len(set(current_list))
                mn, mx = min(current_list), max(current_list)
                if uniq == 1:
                    return {
                        "status": "constant_value_loop",
                        "step": step,
                        "value": current_list[0],
                        "length": n,
                        "elapsed_sec": time.time() - start_time,
                    }
                recent_unique.append(uniq)
                recent_min.append(mn)
                recent_max.append(mx)
                if len(recent_unique) == stability_patience and len(set(recent_unique)) == 1 and len(set(recent_min)) == 1 and len(set(recent_max)) == 1:
                    return {
                        "status": "stable_pattern_detected",
                        "step": step,
                        "length": n,
                        "unique_count": uniq,
                        "min": mn,
                        "max": mx,
                        "elapsed_sec": time.time() - start_time,
                    }

            # Safe hashing for Python list
            exact_key = _hash_list_exact(current_list)
            sorted_key = _hash_list_sorted(current_list)
            if exact_key in seen_exact:
                first = seen_exact[exact_key]
                return {
                    "status": "loop_detected_exact",
                    "step": step,
                    "length": n,
                    "period": step - first,
                    "first_seen_step": first,
                    "elapsed_sec": time.time() - start_time,
                }
            if sorted_key in seen_sorted:
                first = seen_sorted[sorted_key]
                return {
                    "status": "loop_detected_multiset",
                    "step": step,
                    "length": n,
                    "period": step - first,
                    "first_seen_step": first,
                    "elapsed_sec": time.time() - start_time,
                }
            seen_exact[exact_key] = step
            seen_sorted[sorted_key] = step

            next_len = _estimate_next_length(n)
            if next_len > max_len:
                return {
                    "status": "too_large",
                    "step": step + 1,
                    "predicted_next_length": next_len,
                    "elapsed_sec": time.time() - start_time,
                }

            current_list = _xor_pairs_python(current_list)

        return {
            "status": "max_steps_reached",
            "step": max_steps,
            "length": len(current_list),
            "elapsed_sec": time.time() - start_time,
        }

    # torch path
    current_t = torch.tensor(arr, dtype=torch.long, device=dev)

    for step in range(max_steps):
        n = current_t.shape[0]

        if n == 1:
            return {
                "status": "converged",
                "step": step,
                "value": current_t.item(),
                "length": 1,
                "elapsed_sec": time.time() - start_time,
            }
        if n > max_len:
            return {
                "status": "too_large",
                "step": step,
                "length": n,
                "elapsed_sec": time.time() - start_time,
            }

        if step > 0:
            unique_count = torch.unique(current_t).shape[0]
            min_val = int(torch.min(current_t).item())
            max_val = int(torch.max(current_t).item())

            if unique_count == 1:
                return {
                    "status": "constant_value_loop",
                    "step": step,
                    "value": int(current_t[0].item()),
                    "length": n,
                    "elapsed_sec": time.time() - start_time,
                }
            recent_unique.append(int(unique_count))
            recent_min.append(min_val)
            recent_max.append(max_val)
            if len(recent_unique) == stability_patience and len(set(recent_unique)) == 1 and len(set(recent_min)) == 1 and len(set(recent_max)) == 1:
                return {
                    "status": "stable_pattern_detected",
                    "step": step,
                    "length": n,
                    "unique_count": int(unique_count),
                    "min": min_val,
                    "max": max_val,
                    "elapsed_sec": time.time() - start_time,
                }

        exact_key = _hash_tensor_exact(current_t)
        sorted_key = _hash_tensor_sorted(current_t)
        if exact_key in seen_exact:
            first = seen_exact[exact_key]
            return {
                "status": "loop_detected_exact",
                "step": step,
                "length": n,
                "period": step - first,
                "first_seen_step": first,
                "elapsed_sec": time.time() - start_time,
            }
        if sorted_key in seen_sorted:
            first = seen_sorted[sorted_key]
            return {
                "status": "loop_detected_multiset",
                "step": step,
                "length": n,
                "period": step - first,
                "first_seen_step": first,
                "elapsed_sec": time.time() - start_time,
            }
        seen_exact[exact_key] = step
        seen_sorted[sorted_key] = step

        next_len = _estimate_next_length(n)
        if next_len > max_len:
            return {
                "status": "too_large",
                "step": step + 1,
                "predicted_next_length": int(next_len),
                "elapsed_sec": time.time() - start_time,
            }

        avail_bytes = _available_memory_bytes(dev)
        bytes_to_store_next = next_len * 8.0
        if bytes_to_store_next > max_memory_ratio * avail_bytes:
            return {
                "status": "memory_limit_exceeded_next_step",
                "step": step + 1,
                "predicted_next_length": int(next_len),
                "predicted_next_bytes": int(bytes_to_store_next),
                "available_bytes": int(avail_bytes),
                "max_memory_ratio": max_memory_ratio,
                "elapsed_sec": time.time() - start_time,
            }

        if _should_use_broadcast(n, dev):
            next_t = _xor_pairs_broadcast(current_t, dev)
        else:
            # Choose chunk size based on available memory
            if dev.type == "cuda":
                # leave headroom for intermediate tensors; derive chunk size conservatively
                target_bytes = max(64 * 1024**2, int(0.25 * avail_bytes))
            else:
                target_bytes = max(64 * 1024**2, int(0.25 * avail_bytes))
            # per-block memory approximately ~ 2 * c^2 * 8 for intra-block + c^2 * 8 for inter-block buffers
            # Simplify to c so that c^2 * 8 ~ target_bytes => c ~ sqrt(target_bytes / 8)
            chunk_size = max(128, int(math.sqrt(target_bytes / 8)))
            chunk_size = min(chunk_size, n)
            next_t = _xor_pairs_chunked(current_t, chunk_size, dev)

        current_t = next_t

    return {
        "status": "max_steps_reached",
        "step": max_steps,
        "length": int(current_t.shape[0]),
        "elapsed_sec": time.time() - start_time,
    }


def _demo():  # pragma: no cover
    samples = [
        list(range(6)),
        [1, 1, 1, 1],
        [0, 2, 4, 8, 16],
    ]
    for arr in samples:
        print("=" * 60)
        print(f"Input size={len(arr)}: {arr}")
        res = simulate_xor_optimized(arr, max_steps=10)
        print(res)


if __name__ == "__main__":  # pragma: no cover
    _demo() 