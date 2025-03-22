"""
Fuzzy matcher.

This class is used by the [command palette](/guide/command_palette) to match search terms.

This is the matcher that powers Textual's command palette.

Thanks to Will McGugan for the implementation.
"""

from __future__ import annotations

from operator import itemgetter
from re import IGNORECASE, escape, finditer, search
from typing import Iterable, NamedTuple

from textual.cache import LRUCache


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
        # Handle empty query case gracefully
        if not query:
            return (0.0, ())
            
        cache_key = (query, candidate, self.case_sensitive)
        if cache_key in self.cache:
            return self.cache[cache_key]

        # Use a simple matching algorithm that can never fail on deletion
        if not self.case_sensitive:
            query = query.lower()
            candidate = candidate.lower()
        
        # Find all potential matches and their offsets
        matched_offsets = []
        best_score = 0.0
        
        # The simplest, most deletion-safe approach:
        # Find the longest common subsequence (not necessarily contiguous)
        # This guarantees stability when deleting any character
        
        candidate_chars = list(candidate)
        candidate_indices = list(range(len(candidate)))
        
        # First check: quick check if all characters in query exist in candidate
        # This is an optimization to avoid unnecessary work
        query_chars = set(query)
        candidate_char_set = set(candidate)
        if not query_chars.issubset(candidate_char_set):
            return (0.0, ())
        
        # Greedy approach to find offsets - match earliest occurrence of each character
        remaining_indices = list(enumerate(candidate_chars))
        result_offsets = []
        
        for q_char in query:
            found = False
            for idx, (pos, c_char) in enumerate(remaining_indices):
                if q_char == c_char:
                    result_offsets.append(pos)
                    remaining_indices = remaining_indices[idx+1:]
                    found = True
                    break
            
            if not found:
                # Should not happen due to our subset check above, but just in case
                return (0.0, ())
        
        # If we got this far, we have a valid match
        if result_offsets:
            # Calculate score based on characteristics of the match
            score = self._calculate_score(result_offsets, candidate)
            result = (score, tuple(result_offsets))
        else:
            result = (0.0, ())
        
        self.cache[cache_key] = result
        return result

    def _calculate_score(self, offsets, candidate):
        """Calculate a score for the match based on match characteristics.
        
        Args:
            offsets: List of offset positions in the candidate
            candidate: The candidate string
            
        Returns:
            A float score - higher is better
        """
        if not offsets:
            return 0.0
        
        # Base score is the number of matched characters
        score = len(offsets)
        
        # Bonus for matches at the beginning of words
        first_letters = {match.start() for match in finditer(r'\b\w', candidate)}
        first_letter_bonus = sum(1 for offset in offsets if offset in first_letters)
        score += first_letter_bonus
        
        # Bonus for consecutive matches
        consecutive_count = sum(1 for i in range(1, len(offsets)) if offsets[i] == offsets[i-1] + 1)
        if consecutive_count > 0:
            consecutive_ratio = consecutive_count / (len(offsets) - 1)
            score *= 1 + consecutive_ratio
        
        # Distance penalty - matched characters should be close together
        total_distance = sum(offsets[i] - offsets[i-1] for i in range(1, len(offsets)))
        avg_distance = total_distance / (len(offsets) - 1) if len(offsets) > 1 else 0
        
        # Normalize by candidate length to avoid bias for long strings
        normalized_distance = avg_distance / len(candidate) if candidate else 0
        distance_penalty = 1 / (1 + normalized_distance)
        
        score *= distance_penalty
        
        return score