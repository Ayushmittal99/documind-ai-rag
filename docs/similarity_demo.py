"""Cosine similarity vs Euclidean distance on tiny 2-D vectors.  Run: python docs/similarity_demo.py"""
import math

def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    return dot / (math.hypot(*a) * math.hypot(*b))

def euclid(a, b):
    return math.dist(a, b)

query = (1.0, 1.0)
docs = {
    "A (same direction, long)": (10.0, 10.0),
    "B (different direction, close)": (1.5, 0.2),
}
print(f"query = {query}\n")
for name, v in docs.items():
    print(f"{name:32s} cosine={cosine(query, v):.3f}  euclidean={euclid(query, v):.3f}")
print("""
Cosine ranks A first (angle 0 degrees) - it ignores length.
Euclidean ranks B first (it is physically closer) - it is affected by length.
If every vector is normalised to length 1, both give the SAME ranking
(because |a-b|^2 = 2 - 2*cos(a,b)). Gemini embeddings are not guaranteed to be
unit length at every output size, so we use Chroma's cosine space explicitly.""")
