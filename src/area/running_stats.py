"""
Online median tracker using two heaps (max-heap for lower half, min-heap for upper half).
Used by DG-LT to compute running median of Δ_txt and Δ_vis signals without storing the full history.
"""
import heapq


class OnlineMedian:
    """Streaming median via balanced max-heap / min-heap pair.

    push() is O(log n); get() is O(1).
    Works correctly for n=0 (returns 0.0 before any data).
    """

    def __init__(self):
        self._lo = []   # max-heap: stored as negatives
        self._hi = []   # min-heap

    def push(self, val: float) -> None:
        heapq.heappush(self._lo, -val)
        # Balance: ensure lo <= hi elementwise at the boundary
        if self._hi and (-self._lo[0]) > self._hi[0]:
            heapq.heappush(self._hi, -heapq.heappop(self._lo))
        # Sizes: |lo| == |hi| or |lo| == |hi| + 1
        if len(self._lo) > len(self._hi) + 1:
            heapq.heappush(self._hi, -heapq.heappop(self._lo))
        elif len(self._hi) > len(self._lo):
            heapq.heappush(self._lo, -heapq.heappop(self._hi))

    def get(self) -> float:
        """Return current median; 0.0 if no data yet."""
        n_lo, n_hi = len(self._lo), len(self._hi)
        if n_lo == 0:
            return 0.0
        if n_lo > n_hi:
            return -self._lo[0]
        return (-self._lo[0] + self._hi[0]) / 2.0

    def __len__(self) -> int:
        return len(self._lo) + len(self._hi)
