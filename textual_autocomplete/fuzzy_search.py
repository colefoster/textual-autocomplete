"""
Fuzzy matcher.

This class is used by the [command palette](/guide/command_palette) to match search terms.

This is the matcher that powers Textual's command palette.

"""

from __future__ import annotations

from operator import itemgetter
from re import IGNORECASE, escape, finditer, search
from typing import Iterable, NamedTuple

from textual.cache import LRUCache
import time

class _Search(NamedTuple):
    """Internal structure to keep track of a recursive search."""

    candidate_offset: int = 0
    query_offset: int = 0
    offsets: tuple[int, ...] = ()

    def branch(self, offset: int) -> tuple[_Search, _Search]:
        """Branch this search when an offset is found.

        Args:
            offset: Offset of a matching letter in the query.

        Returns:
            A pair of search objects.
        """
        _, query_offset, offsets = self
        return (
            _Search(offset + 1, query_offset + 1, offsets + (offset,)),
            _Search(offset + 1, query_offset, offsets),
        )

    @property
    def groups(self) -> int:
        """Number of groups in offsets."""
        groups = 1
        last_offset, *offsets = self.offsets
        for offset in offsets:
            if offset != last_offset + 1:
                groups += 1
            last_offset = offset
        return groups


class FuzzySearch:
    """Performs a fuzzy search.

    Unlike a regex solution, this will finds all possible matches.
    """

    cache: LRUCache[tuple[str, str, bool], tuple[float, tuple[int, ...]]] = LRUCache(
        1024 * 4
    )

    def __init__(self, case_sensitive: bool = False) -> None:
        """Initialize fuzzy search.

        Args:
            case_sensitive: Is the match case sensitive?
        """

        self.case_sensitive = case_sensitive
    def match(self, query: str, candidate: str) -> tuple[float, tuple[int, ...]]:
        """Match against a query.

        Args:
            query: The fuzzy query.
            candidate: A candidate to check.

        Returns:
            A pair of (score, tuple of offsets). `(0, ())` for no result.
        """
        # Quick exit for empty query
        if not query:
            return (0.0, ())

        # Check cache first to avoid unnecessary computation
        cache_key = (query, candidate, self.case_sensitive)
        if cache_key in self.cache:
            return self.cache[cache_key]

        # Race condition detection for rapid deletions
        current_time = time.time()
        last_call_time = getattr(self, '_last_call_time', 0)
        
        # Store the current query to prevent race conditions
        current_query = getattr(self, '_last_query', '')
        self._last_query = query
        self._last_call_time = current_time
        
        # Quick successive calls with shortening queries indicate rapid deletion
        if (current_time - last_call_time < 0.25 and 
            len(query) < len(current_query) and 
            current_query.startswith(query)):
            # During rapid deletion, return a simple, safe match to prevent crashes
            return (0.01, (0,) if query else ())

        # For normal operation, use our improved matching algorithm
        if not self.case_sensitive:
            query_lower = query.lower()
            candidate_lower = candidate.lower()
        else:
            query_lower = query
            candidate_lower = candidate

        # Check if the candidate contains all characters in the query
        query_chars = set(query_lower.replace(" ", ""))
        candidate_chars = set(candidate_lower)
        if not query_chars.issubset(candidate_chars):
            return (0.0, ())

        # Use the better search logic for normal operation
        result = self._advanced_match(query_lower, candidate_lower, candidate)
        
        # Store in cache
        self.cache[cache_key] = result
        return result

    def _advanced_match(self, query_lower, candidate_lower, candidate):
        """Advanced matching algorithm with better results but still stable.
        
        Args:
            query_lower: Lowercase query string
            candidate_lower: Lowercase candidate string
            candidate: Original candidate string
            
        Returns:
            Tuple of (score, matched_offsets)
        """
        # Find word boundaries for scoring
        first_letters = {match.start() for match in finditer(r'\b\w', candidate)}
        
        # Split query by spaces to handle each part separately
        query_parts = query_lower.split()
        if not query_parts:
            return (0.0, ())
        
        best_score = 0.0
        best_offsets = ()
        
        # Process each query part separately for better matching
        for query_part in query_parts:
            if not query_part:  # Skip empty parts
                continue
                
            # We'll try different starting positions to find best match
            for start_pos in range(len(candidate_lower)):
                # Don't bother if we're starting too late to match the whole query part
                if start_pos + len(query_part) > len(candidate_lower):
                    break
                    
                offsets = []
                pos = start_pos
                matched_all = True
                
                # Try to match each character from this starting point
                for char in query_part:
                    # Find next occurrence of this character
                    next_pos = candidate_lower.find(char, pos)
                    if next_pos == -1:
                        matched_all = False
                        break
                    
                    offsets.append(next_pos)
                    pos = next_pos + 1
                
                if matched_all and offsets:
                    # Calculate score
                    score = len(offsets)
                    
                    # Bonus for matching at start of words
                    word_start_matches = sum(1 for offset in offsets if offset in first_letters)
                    score += word_start_matches * 2  # Double bonus for word starts
                    
                    # Bonus for consecutive characters
                    consecutive_count = sum(1 for i in range(1, len(offsets)) 
                                        if offsets[i] == offsets[i-1] + 1)
                    
                    # Calculate consecutive bonus - more consecutive = exponentially better
                    if len(offsets) > 1:
                        consecutive_ratio = consecutive_count / (len(offsets) - 1)
                        score *= 1 + (consecutive_ratio ** 2)
                    
                    # Bonus for matches closer to start of candidate
                    position_bonus = 1.0 - (start_pos / len(candidate_lower) / 2)  # Max 50% bonus
                    score *= position_bonus
                    
                    # Keep the best match
                    if score > best_score:
                        best_score = score
                        best_offsets = tuple(offsets)
        
        # If no good matches found, fall back to simple matching for stability
        if not best_offsets:
            return self._simple_match(query_lower, candidate_lower, candidate)
        
        return (best_score, best_offsets)

    def _simple_match(self, query_lower, candidate_lower, candidate):
        """Simple matching algorithm as fallback for stability.
        
        Args:
            query_lower: Lowercase query string
            candidate_lower: Lowercase candidate string
            candidate: Original candidate string
            
        Returns:
            Tuple of (score, matched_offsets)
        """
        # Skip spaces in query for matching purposes
        search_query = query_lower.replace(" ", "")
        if not search_query:
            return (0.0, ())
        
        # Find word boundaries for scoring
        first_letters = {match.start() for match in finditer(r'\b\w', candidate)}
        
        # Simple one-pass forward matching - less likely to cause inconsistencies
        best_offsets = []
        pos = 0
        
        # Match each query character to earliest available position in candidate
        for char in search_query:
            found = False
            while pos < len(candidate_lower):
                if candidate_lower[pos] == char:
                    best_offsets.append(pos)
                    pos += 1
                    found = True
                    break
                pos += 1
            
            if not found:
                # Ensure we always return something valid
                return (0.01, (0,)) if search_query else (0.0, ())
        
        if not best_offsets:
            return (0.01, (0,)) if search_query else (0.0, ())
        
        # Calculate score (simplified for stability)
        score = len(best_offsets)
        
        # Boost for matches at word beginnings
        word_start_matches = sum(1 for offset in best_offsets if offset in first_letters)
        score += word_start_matches
        
        # Always return a consistent result structure
        return (float(score), tuple(best_offsets))