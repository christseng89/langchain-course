"""
Semantic Cache
Shared response cache used by cost_optimization.py and monitoring.py
"""

import hashlib
from typing import Optional


class SemanticCache:
  """Cache responses with semantic similarity matching."""

  def __init__(self, similarity_threshold: float = 0.9):
    self.cache = {}
    self.threshold = similarity_threshold

  def _hash_query(self, query: str) -> str:
    """Create hash of normalized query."""
    normalized = query.lower().strip()
    return hashlib.md5(normalized.encode()).hexdigest()

  def get(self, query: str) -> Optional[str]:
    """Get cached response if similar query exists."""
    query_hash = self._hash_query(query)

    # Exact match
    if query_hash in self.cache:
      return self.cache[query_hash]["response"]

    # Could add embedding-based similarity here
    # For demo, just use exact match

    return None

  def set(self, query: str, response: str):
    """Cache a response."""
    query_hash = self._hash_query(query)
    self.cache[query_hash] = {"query": query, "response": response}

  def stats(self) -> dict:
    return {"cached_queries": len(self.cache)}
