"""Data contamination prevention for benchmark integrity.

This module provides tools to detect and prevent data contamination in
evaluation datasets. Implements ABC checks R.3 and R.4.
"""

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

from evaris.types import TestCase


class ContaminationConfig(BaseModel):
    """Configuration for contamination detection."""

    check_public_availability: bool = Field(
        default=True, description="Check if data is publicly available"
    )
    track_release_dates: bool = Field(default=True, description="Track benchmark release dates")
    require_version_control: bool = Field(
        default=True, description="Require version control for datasets"
    )
    allow_web_search: bool = Field(default=False, description="Allow agents to use web search")
    fingerprint_method: Literal["hash", "embedding"] = Field(
        default="hash", description="Method for fingerprinting test cases"
    )


class DatasetFingerprint(BaseModel):
    """Fingerprint for a dataset version."""

    version: str = Field(description="Dataset version identifier")
    release_date: str = Field(description="Release date (ISO format)")
    num_test_cases: int = Field(description="Number of test cases")
    content_hash: str = Field(description="Hash of dataset content")
    is_public: bool = Field(description="Whether dataset is publicly available")
    public_url: Optional[str] = Field(default=None, description="URL if publicly available")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class ContaminationCheckResult(BaseModel):
    """Result of contamination check."""

    is_contaminated: bool = Field(description="Whether contamination detected")
    contamination_type: Optional[str] = Field(
        default=None, description="Type of contamination detected"
    )
    evidence: list[str] = Field(default_factory=list, description="Evidence of contamination")
    recommendations: list[str] = Field(
        default_factory=list, description="Recommendations to address contamination"
    )
    severity: Literal["none", "low", "medium", "high"] = Field(
        default="none", description="Severity of contamination"
    )


class ContaminationDetector:
    """Detector for data contamination in benchmarks.

    Provides tools to detect potential data contamination, track dataset
    versions, and ensure benchmark integrity.

    ABC Compliance:
    - R.3: Reports whether benchmark is publicly available and release date
    - R.4: Addresses potential data contamination in evaluation

    Example:
        >>> from evaris.contamination import ContaminationDetector, ContaminationConfig
        >>> from evaris.types import TestCase
        >>>
        >>> config = ContaminationConfig(check_public_availability=True)
        >>> detector = ContaminationDetector(config)
        >>>
        >>> # Create dataset fingerprint
        >>> test_cases = [TestCase(input="q1", expected="a1")]
        >>> fingerprint = detector.create_fingerprint(
        ...     test_cases, version="1.0", release_date="2025-01-01"
        ... )
        >>> print(fingerprint.content_hash)
    """

    def __init__(self, config: Optional[ContaminationConfig] = None):
        """Initialize contamination detector.

        Args:
            config: Configuration for contamination detection. If None, uses defaults.
        """
        self.config = config or ContaminationConfig()
        self.fingerprints: dict[str, DatasetFingerprint] = {}

    def _compute_content_hash(self, test_cases: list[TestCase]) -> str:
        """Compute hash of test cases content.

        ABC R.4: Creates fingerprint for contamination detection.

        Args:
            test_cases: Test cases to hash

        Returns:
            SHA-256 hash of content
        """
        # Create deterministic representation
        content_parts = []
        for tc in sorted(test_cases, key=lambda x: str(x.input)):
            part = {
                "input": str(tc.input),
                "expected": str(tc.expected) if tc.expected is not None else None,
            }
            content_parts.append(part)

        content_str = json.dumps(content_parts, sort_keys=True)
        return hashlib.sha256(content_str.encode()).hexdigest()

    def create_fingerprint(
        self,
        test_cases: list[TestCase],
        version: str,
        release_date: str,
        is_public: bool = False,
        public_url: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> DatasetFingerprint:
        """Create fingerprint for a dataset version.

        ABC Compliance:
        - R.3: Records release date and public availability
        - R.4: Creates fingerprint for contamination tracking

        Args:
            test_cases: Test cases in dataset
            version: Version identifier
            release_date: Release date (ISO format)
            is_public: Whether dataset is publicly available
            public_url: URL if publicly available
            metadata: Additional metadata

        Returns:
            DatasetFingerprint for the dataset
        """
        content_hash = self._compute_content_hash(test_cases)

        fingerprint = DatasetFingerprint(
            version=version,
            release_date=release_date,
            num_test_cases=len(test_cases),
            content_hash=content_hash,
            is_public=is_public,
            public_url=public_url,
            metadata=metadata or {},
        )

        # Store fingerprint
        self.fingerprints[version] = fingerprint

        return fingerprint

    def register_dataset(
        self,
        test_cases: list[TestCase],
        version: str,
        release_date: Optional[str] = None,
        is_public: bool = False,
        public_url: Optional[str] = None,
    ) -> DatasetFingerprint:
        """Register a dataset version for contamination tracking.

        ABC R.3: Records dataset release information.

        Args:
            test_cases: Test cases in dataset
            version: Version identifier
            release_date: Release date (ISO format), defaults to now
            is_public: Whether dataset is publicly available
            public_url: URL if publicly available

        Returns:
            DatasetFingerprint for the dataset
        """
        if release_date is None:
            release_date = datetime.now().isoformat()

        return self.create_fingerprint(test_cases, version, release_date, is_public, public_url)

    def check_duplicate_content(self, test_cases: list[TestCase]) -> ContaminationCheckResult:
        """Check if test cases duplicate existing datasets.

        ABC R.4: Detects potential contamination from duplicate content.

        Args:
            test_cases: Test cases to check

        Returns:
            ContaminationCheckResult with check details
        """
        content_hash = self._compute_content_hash(test_cases)

        # Check against registered fingerprints
        evidence: list[str] = []
        recommendations: list[str] = []

        for version, fingerprint in self.fingerprints.items():
            if fingerprint.content_hash == content_hash:
                evidence.append(
                    f"Content matches dataset version '{version}' "
                    f"(released {fingerprint.release_date})"
                )
                if fingerprint.is_public:
                    evidence.append(
                        f"Version '{version}' is publicly available at {fingerprint.public_url}"
                    )
                    recommendations.append("Use hold-out test set or create new private dataset")

        if evidence:
            return ContaminationCheckResult(
                is_contaminated=True,
                contamination_type="duplicate_content",
                evidence=evidence,
                recommendations=recommendations,
                severity="high",
            )

        return ContaminationCheckResult(
            is_contaminated=False,
            severity="none",
        )

    def check_public_availability(self, version: str) -> ContaminationCheckResult:
        """Check if a dataset version is publicly available.

        ABC R.3: Reports public availability status.

        Args:
            version: Dataset version to check

        Returns:
            ContaminationCheckResult with availability info
        """
        if version not in self.fingerprints:
            return ContaminationCheckResult(
                is_contaminated=False,
                contamination_type="unknown_version",
                evidence=[f"Version '{version}' not registered"],
                severity="none",
            )

        fingerprint = self.fingerprints[version]

        if fingerprint.is_public:
            return ContaminationCheckResult(
                is_contaminated=True,
                contamination_type="public_dataset",
                evidence=[
                    f"Dataset is publicly available at {fingerprint.public_url}",
                    f"Released on {fingerprint.release_date}",
                ],
                recommendations=[
                    "Agents may have been trained on this data",
                    "Consider using private hold-out set for final evaluation",
                    "Report public availability in benchmark documentation",
                ],
                severity="medium",
            )

        return ContaminationCheckResult(
            is_contaminated=False,
            evidence=[f"Dataset version '{version}' is not publicly available"],
            severity="none",
        )

    def check_temporal_contamination(
        self, version: str, agent_training_cutoff: str
    ) -> ContaminationCheckResult:
        """Check for temporal contamination.

        ABC R.4: Detects contamination based on release dates.

        Args:
            version: Dataset version
            agent_training_cutoff: Agent's training data cutoff date (ISO format)

        Returns:
            ContaminationCheckResult with temporal analysis
        """
        if version not in self.fingerprints:
            return ContaminationCheckResult(
                is_contaminated=False,
                contamination_type="unknown_version",
                evidence=[f"Version '{version}' not registered"],
                severity="none",
            )

        fingerprint = self.fingerprints[version]

        try:
            release_date = datetime.fromisoformat(fingerprint.release_date)
            cutoff_date = datetime.fromisoformat(agent_training_cutoff)

            if release_date <= cutoff_date:
                return ContaminationCheckResult(
                    is_contaminated=True,
                    contamination_type="temporal_contamination",
                    evidence=[
                        f"Dataset released on {fingerprint.release_date}",
                        f"Agent training cutoff: {agent_training_cutoff}",
                        "Dataset may have been in training data",
                    ],
                    recommendations=[
                        "Use newer benchmark data released after training cutoff",
                        "Report potential contamination in results",
                    ],
                    severity="high" if fingerprint.is_public else "medium",
                )

            return ContaminationCheckResult(
                is_contaminated=False,
                evidence=[
                    f"Dataset released on {fingerprint.release_date} "
                    f"(after training cutoff {agent_training_cutoff})"
                ],
                severity="none",
            )

        except ValueError:
            return ContaminationCheckResult(
                is_contaminated=False,
                contamination_type="invalid_dates",
                evidence=["Could not parse dates for temporal analysis"],
                severity="none",
            )

    def comprehensive_check(
        self,
        test_cases: list[TestCase],
        version: str,
        agent_training_cutoff: Optional[str] = None,
    ) -> list[ContaminationCheckResult]:
        """Run comprehensive contamination checks.

        ABC Compliance:
        - R.3: Checks public availability
        - R.4: Detects multiple contamination types

        Args:
            test_cases: Test cases to check
            version: Dataset version
            agent_training_cutoff: Agent's training cutoff date (optional)

        Returns:
            List of contamination check results
        """
        results: list[ContaminationCheckResult] = []

        # Check duplicate content
        results.append(self.check_duplicate_content(test_cases))

        # Check public availability
        if self.config.check_public_availability:
            results.append(self.check_public_availability(version))

        # Check temporal contamination
        if agent_training_cutoff and self.config.track_release_dates:
            results.append(self.check_temporal_contamination(version, agent_training_cutoff))

        return results

    def generate_contamination_report(self, check_results: list[ContaminationCheckResult]) -> str:
        """Generate human-readable contamination report.

        ABC R.3/R.4: Provides transparent contamination reporting.

        Args:
            check_results: List of contamination check results

        Returns:
            Formatted report string
        """
        lines = ["Data Contamination Report", "=" * 50, ""]

        # Count by severity
        severity_counts = {
            "none": 0,
            "low": 0,
            "medium": 0,
            "high": 0,
        }

        for result in check_results:
            severity_counts[result.severity] += 1

        # Overall status
        has_contamination = any(r.is_contaminated for r in check_results)
        if has_contamination:
            lines.append("WARNING: CONTAMINATION DETECTED")
        else:
            lines.append("No contamination detected")

        lines.append("")

        # Summary
        lines.append("Summary:")
        lines.append(f"  High severity: {severity_counts['high']}")
        lines.append(f"  Medium severity: {severity_counts['medium']}")
        lines.append(f"  Low severity: {severity_counts['low']}")
        lines.append("")

        # Details
        for i, result in enumerate(check_results, 1):
            if result.is_contaminated:
                lines.append(f"Issue {i}: {result.contamination_type} ({result.severity})")
                lines.append("  Evidence:")
                for evidence in result.evidence:
                    lines.append(f"    - {evidence}")
                if result.recommendations:
                    lines.append("  Recommendations:")
                    for rec in result.recommendations:
                        lines.append(f"    - {rec}")
                lines.append("")

        return "\n".join(lines)

    def save_fingerprint(self, version: str, file_path: Path) -> None:
        """Save dataset fingerprint to file.

        ABC R.4: Enables version control for contamination tracking.

        Args:
            version: Dataset version
            file_path: Path to save fingerprint

        Raises:
            ValueError: If version not registered
        """
        if version not in self.fingerprints:
            raise ValueError(f"Version '{version}' not registered")

        fingerprint = self.fingerprints[version]
        file_path.write_text(fingerprint.model_dump_json(indent=2))

    def load_fingerprint(self, file_path: Path) -> DatasetFingerprint:
        """Load dataset fingerprint from file.

        ABC R.4: Supports version control for contamination tracking.

        Args:
            file_path: Path to fingerprint file

        Returns:
            Loaded DatasetFingerprint
        """
        content = file_path.read_text()
        fingerprint = DatasetFingerprint.model_validate_json(content)
        self.fingerprints[fingerprint.version] = fingerprint
        return fingerprint
