"""Data classes for validation reports."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List


@dataclass
class DimensionResult:
    """Result of a single validation dimension evaluation.

    Attributes
    ----------
    dimension_name : str
        Name of the dimension evaluated.
    metrics : Dict[str, float]
        Dictionary of computed metric values.
    score : float
        Score between 0 and 100.
    passed : bool
        Whether this dimension meets its threshold.
    details : str
        Human-readable explanation of the result.
    """

    dimension_name: str
    metrics: Dict[str, float]
    score: float
    passed: bool
    details: str


@dataclass
class ValidationReport:
    """Complete validation report across all dimensions.

    Attributes
    ----------
    overall_score : float
        Weighted average score between 0 and 100.
    overall_passed : bool
        Whether the overall score meets the minimum threshold.
    dimension_results : List[DimensionResult]
        Individual results for each dimension.
    summary : str
        Human-readable summary of the validation.
    timestamp : datetime
        When the validation was performed.
    """

    overall_score: float
    overall_passed: bool
    dimension_results: List[DimensionResult] = field(default_factory=list)
    summary: str = ""
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict:
        """Convert the report to a dictionary representation.

        Returns
        -------
        Dict
            Dictionary containing all report data.
        """
        return {
            "overall_score": self.overall_score,
            "overall_passed": self.overall_passed,
            "summary": self.summary,
            "timestamp": self.timestamp.isoformat(),
            "dimension_results": [
                {
                    "dimension_name": dr.dimension_name,
                    "metrics": dr.metrics,
                    "score": dr.score,
                    "passed": dr.passed,
                    "details": dr.details,
                }
                for dr in self.dimension_results
            ],
        }

    def __str__(self) -> str:
        """Format a readable validation report."""
        lines = []
        lines.append("=" * 60)
        lines.append("ALPHA VALIDATION REPORT")
        lines.append("=" * 60)
        lines.append(f"Timestamp: {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"Overall Score: {self.overall_score:.1f} / 100")
        lines.append(f"Overall Passed: {'YES' if self.overall_passed else 'NO'}")
        lines.append("-" * 60)
        lines.append("DIMENSION RESULTS:")
        lines.append("-" * 60)
        for dr in self.dimension_results:
            status = "PASS" if dr.passed else "FAIL"
            lines.append(f"  [{status}] {dr.dimension_name}: {dr.score:.1f}/100")
            lines.append(f"         {dr.details}")
        lines.append("-" * 60)
        lines.append(f"Summary: {self.summary}")
        lines.append("=" * 60)
        return "\n".join(lines)
