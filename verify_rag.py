import requests
import json
import time
import sys

# Reconfigure stdout for UTF-8 support on Windows CP1252 terminal
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')


BASE_URL = "http://127.0.0.1:8000"

def test_health():
    print("=== Testing Health Endpoint ===")
    try:
        r = requests.get(f"{BASE_URL}/api/v1/health")
        print(f"Status: {r.status_code}")
        print(json.dumps(r.json(), indent=2, ensure_ascii=False))
        return r.status_code == 200
    except Exception as e:
        print(f"Health check failed: {e}")
        return False

def test_text_query(query: str, lang: str):
    print(f"\n=== Testing Text Query ({lang.upper()}): '{query}' ===")
    payload = {
        "query": query,
        "language": lang
    }
    t0 = time.time()
    try:
        r = requests.post(f"{BASE_URL}/api/v1/text/query", json=payload)
        elapsed = (time.time() - t0) * 1000
        print(f"Status: {r.status_code} in {elapsed:.2f}ms")
        if r.status_code == 200:
            data = r.json()
            print(f"Answer: {data.get('answer')}")
            print(f"Grounded: {data.get('grounded')} (Confidence: {data.get('confidence')*100:.1f}%)")
            print(f"Source count: {len(data.get('sources', []))}")
            if 'grounding_details' in data and data['grounding_details'] is not None:
                print("Grounding signals:")
                relevance = data['grounding_details'].get('retrieval_max_relevance')
                if relevance is not None:
                    print(f"  - Max relevance: {relevance:.3f}")
                semantic = data['grounding_details'].get('semantic_similarity')
                if semantic is not None:
                    print(f"  - Semantic alignment: {semantic:.3f}")
                coverage = data['grounding_details'].get('word_intersection')
                if coverage is not None:
                    print(f"  - Citation coverage: {coverage:.3f}")
                print(f"  - LLM judge verdict: {data['grounding_details'].get('llm_judge_verdict')}")

            print("Pipeline Tracing latency:")
            for k, v in data.get('latency', {}).items():
                print(f"  - {k}: {v:.1f} ms")
        else:
            print(r.text)
    except Exception as e:
        print(f"Query failed: {e}")

def test_metrics():
    print("\n=== Testing Cumulative Metrics ===")
    try:
        r = requests.get(f"{BASE_URL}/api/v1/metrics")
        print(f"Status: {r.status_code}")
        print(json.dumps(r.json(), indent=2, ensure_ascii=False))
    except Exception as e:
        print(f"Metrics fetch failed: {e}")

if __name__ == "__main__":
    print("Starting End-to-End RAG Verification against FastAPI Server...")
    if test_health():
        # Test Hindi query - Corporation
        test_text_query("कॉर्पोरेशन क्या है?", "hi")
        # Test Hindi query - Honesty/Truth
        test_text_query("ईमानदारी या सच्चाई की परिभाषा", "hi")
        # Test out of scope
        test_text_query("yolov11 accurate training parameters?", "en")
        # Fetch metrics
        test_metrics()
    else:
        print("Backend server is not reachable. Please start uvicorn first.")

