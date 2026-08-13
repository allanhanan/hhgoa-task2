import time
from typing import List, Dict, Any
import numpy as np

class MetricsTracker:
    """
    In-memory metrics accumulator for RAG pipeline.
    Aggregates request volumes, cache outcomes, grounding rates, and percentile latencies.
    """
    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.latencies: List[float] = []
        self.total_requests = 0
        self.grounding_passes = 0
        self.cache_hits = 0

    def log_request(self, latency_ms: float, is_grounded: bool, cache_hit: bool):
        self.total_requests += 1
        
        if cache_hit:
            self.cache_hits += 1
        else:
            self.latencies.append(latency_ms)
            # Maintain sliding window size
            if len(self.latencies) > self.window_size:
                self.latencies.pop(0)
                
        if is_grounded:
            self.grounding_passes += 1

    def get_report(self) -> Dict[str, Any]:
        """
        Computes percentile metrics and returns summary.
        """
        p50 = 0.0
        p95 = 0.0
        p99 = 0.0
        
        if self.latencies:
            arr = np.array(self.latencies)
            p50 = float(np.percentile(arr, 50))
            p95 = float(np.percentile(arr, 95))
            p99 = float(np.percentile(arr, 99))
            
        grounding_rate = (self.grounding_passes / self.total_requests) if self.total_requests > 0 else 0.0
        cache_hit_rate = (self.cache_hits / self.total_requests) if self.total_requests > 0 else 0.0
        
        return {
            "total_requests": self.total_requests,
            "cache_hits": self.cache_hits,
            "cache_hit_rate": cache_hit_rate,
            "grounding_passes": self.grounding_passes,
            "grounding_rate": grounding_rate,
            "latency_p50_ms": p50,
            "latency_p95_ms": p95,
            "latency_p99_ms": p99,
            "active_window_size": len(self.latencies)
        }
