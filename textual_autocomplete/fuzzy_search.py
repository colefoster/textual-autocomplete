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
        # Quick check if any match is possible
        if not query:
            return (0.0, ())
        
        if not self.case_sensitive:
            query = query.lower()
            candidate_lower = candidate.lower()
        else:
            candidate_lower = candidate
        
        # Quick check: does the candidate contain all unique characters from the query?
        query_chars = set(query.replace(" ", ""))
        candidate_chars = set(candidate_lower.replace(" ", ""))
        
        if not query_chars.issubset(candidate_chars):
            # Early out - candidate doesn't contain all needed characters
            return (0.0, ())

        cache_key = (query, candidate, self.case_sensitive)
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # Process the matches
        result = self._find_best_match(query, candidate, candidate_lower)
        self.cache[cache_key] = result
        return result

    def _find_best_match(self, query: str, candidate: str, candidate_lower: str) -> tuple[float, tuple[int, ...]]:
        """Find the best match for the query in the candidate.
        
        Args:
            query: The query string
            candidate: The original candidate string
            candidate_lower: Lowercase version of candidate (if case_sensitive is False)
            
        Returns:
            Tuple of (score, offsets)
        """
        if not self.case_sensitive:
            query_lower = query.lower()
        else:
            query_lower = query
        
        # Define pattern for first letters in words (for scoring)
        first_letters = {match.start() for match in finditer(r'\b\w', candidate)}
        
        best_score = 0.0
        best_offsets = ()
        
        # Split query by spaces to handle each word separately
        query_parts = query_lower.split()
        if not query_parts:
            return (0.0, ())
        
        # For each part of the query, find the best sequence of matching characters
        for query_part in query_parts:
            if not query_part:  # Skip empty parts (e.g., consecutive spaces)
                continue
                
            # Try different starting positions in the candidate
            for start_pos in range(len(candidate_lower)):
                offsets = []
                matched_all = True
                pos = start_pos
                
                # Try to match each character in the query part
                for char in query_part:
                    # Find the next occurrence of the character
                    next_pos = candidate_lower.find(char, pos)
                    if next_pos == -1:
                        matched_all = False
                        break
                        
                    offsets.append(next_pos)
                    pos = next_pos + 1
                    
                if matched_all and offsets:
                    # Calculate score - more consecutive characters = better score
                    # Also bonus for matching at word boundaries and first letters
                    score = len(offsets)
                    
                    # Bonus for matching at start of words
                    word_start_matches = sum(1 for offset in offsets if offset in first_letters)
                    score += word_start_matches
                    
                    # Bonus for consecutive characters
                    consecutive_count = 0
                    for i in range(1, len(offsets)):
                        if offsets[i] == offsets[i-1] + 1:
                            consecutive_count += 1
                    
                    # Calculate consecutive bonus - more consecutive = exponentially better
                    consecutive_bonus = 1 + (consecutive_count / len(offsets))**2
                    score *= consecutive_bonus
                    
                    if score > best_score:
                        best_score = score
                        best_offsets = tuple(offsets)
        
        return (best_score, best_offsets)
