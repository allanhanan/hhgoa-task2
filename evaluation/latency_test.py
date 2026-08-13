import os
import sys
import time
import asyncio
import argparse
import numpy as np
import httpx

BASE_URL = "http://127.0.0.1:8000"

TEST_QUERIES = [
    ("hi", "भारत की राजधानी क्या है?"),
    ("hi", "ईमानदारी या सच्चाई की परिभाषा"),
    ("hi", "कॉर्पोरेशन क्या है?"),
    ("ta", "இந்தியாவின் தலைநகரம் எது?"),
    ("te", "భారతదేశ రాజధాని ఏది?"),
    ("kn", "ಭಾರತದ ರಾಜಧಾನಿ ಯಾವುದು?"),
    ("ml", "ഇന്ത്യയുടെ തലസ്ഥാനം ഏതാണ്?"),
    ("mr", "भारताची राजधानी कोणती आहे?"),
    ("gu", "ભારતની રાજધાની કઈ છે?"),
    ("bn", "ভারতের রাজধানী কি?"),
    ("en", "what is the capital of India?"),
    ("en", "what is the financial capital of India?")
]

async def run_latency_benchmark(num_requests: int = 50, concurrency: int = 2):
    print("=" * 65)
    print("      VOICE-ENABLED RAG PIPELINE LATENCY ANALYTICS (P50/P70/P100)      ")
    print("=" * 65)

    async with httpx.AsyncClient(timeout=30.0) as client:
        # 1. Health check
        try:
            r = await client.get(f"{BASE_URL}/api/v1/health")
            if r.status_code != 200:
                print("Error: Backend is not healthy. Start uvicorn backend.app:app first.")
                return
            print(f"Server connected: {r.json()['status']} ({r.json()['environment']})")
        except Exception as e:
            print(f"Error connecting to backend: {e}")
            return

        retrieval_latencies = []
        total_latencies = []
        embedding_latencies = []
        dense_latencies = []
        sparse_latencies = []

        semaphore = asyncio.Semaphore(concurrency)

        async def send_query(idx: int):
            lang, q = TEST_QUERIES[idx % len(TEST_QUERIES)]
            async with semaphore:
                t0 = time.time()
                try:
                    res = await client.post(
                        f"{BASE_URL}/api/v1/text/query",
                        json={"query": q, "language": lang}
                    )
                    elapsed = (time.time() - t0) * 1000
                    if res.status_code == 200:
                        data = res.json()
                        lat = data.get("latency", {})
                        retrieval_latencies.append(lat.get("retrieval_total_ms", 0.0))
                        total_latencies.append(lat.get("total_ms", elapsed))
                        embedding_latencies.append(lat.get("embedding_ms", 0.0))
                        dense_latencies.append(lat.get("dense_ms", 0.0))
                        sparse_latencies.append(lat.get("sparse_ms", 0.0))
                        return True
                except Exception as exc:
                    print(f"Query request failed: {exc}")
                    return False

        print(f"Executing {num_requests} requests across {len(TEST_QUERIES)} multilingual test queries...")
        tasks = [send_query(i) for i in range(num_requests)]
        results = await asyncio.gather(*tasks)

        success_count = sum(1 for r in results if r)

        print("\n" + "=" * 65)
        print("                  LATENCY ANALYTICS REPORT                     ")
        print("=" * 65)
        print(f"Total Requests Processed : {success_count} / {num_requests}")
        print("-" * 65)

        def print_percentiles(name: str, values: list):
            if not values:
                return
            arr = np.array(values)
            p50 = np.percentile(arr, 50)
            p70 = np.percentile(arr, 70)
            p95 = np.percentile(arr, 95)
            p100 = np.max(arr)
            print(f"{name:<28} | P50: {p50:>6.2f}ms | P70: {p70:>6.2f}ms | P100: {p100:>6.2f}ms")

        print_percentiles("Retrieval Target (<200ms)", retrieval_latencies)
        print_percentiles("Query Embedding Stage", embedding_latencies)
        print_percentiles("Dense Search (Parallel)", dense_latencies)
        print_percentiles("Sparse Search (Parallel)", sparse_latencies)
        print_percentiles("End-to-End Total Latency", total_latencies)
        print("=" * 65)

        # Retrieve cumulative server report
        try:
            m_res = await client.get(f"{BASE_URL}/api/v1/metrics")
            if m_res.status_code == 200:
                metrics = m_res.json()
                print("\nServer Cumulative Metrics Summary:")
                print(f"  - Grounding Pass Rate : {metrics.get('grounding_rate', 0)*100:.1f}%")
                print(f"  - Cache Hit Rate      : {metrics.get('cache_hit_rate', 0)*100:.1f}%")
        except Exception:
            pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Latency Analytics Benchmark Suite")
    parser.add_argument("--requests", type=int, default=50, help="Number of test queries to run.")
    parser.add_argument("--concurrency", type=int, default=2, help="Concurrency limit.")
    args = parser.parse_args()

    asyncio.run(run_latency_benchmark(num_requests=args.requests, concurrency=args.concurrency))
