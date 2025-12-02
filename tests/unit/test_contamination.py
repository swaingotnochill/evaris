"""Tests for contamination module."""

from evaris.contamination import (
    ContaminationCheckResult,
    ContaminationConfig,
    ContaminationDetector,
    DatasetFingerprint,
)
from evaris.types import TestCase


class TestContaminationConfig:
    """Tests for ContaminationConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = ContaminationConfig()

        assert config.check_public_availability is True
        assert config.track_release_dates is True
        assert config.require_version_control is True
        assert config.allow_web_search is False
        assert config.fingerprint_method == "hash"

    def test_custom_config(self):
        """Test custom configuration."""
        config = ContaminationConfig(
            check_public_availability=False,
            allow_web_search=True,
            fingerprint_method="embedding",
        )

        assert config.check_public_availability is False
        assert config.allow_web_search is True
        assert config.fingerprint_method == "embedding"


class TestDatasetFingerprint:
    """Tests for DatasetFingerprint."""

    def test_fingerprint_creation(self):
        """Test creating dataset fingerprint."""
        fp = DatasetFingerprint(
            version="1.0",
            release_date="2025-01-01",
            num_test_cases=10,
            content_hash="abc123",
            is_public=True,
            public_url="https://example.com/data",
        )

        assert fp.version == "1.0"
        assert fp.release_date == "2025-01-01"
        assert fp.num_test_cases == 10
        assert fp.content_hash == "abc123"
        assert fp.is_public is True
        assert fp.public_url == "https://example.com/data"


class TestContaminationCheckResult:
    """Tests for ContaminationCheckResult."""

    def test_no_contamination(self):
        """Test result with no contamination."""
        result = ContaminationCheckResult(is_contaminated=False, severity="none")

        assert result.is_contaminated is False
        assert result.severity == "none"
        assert result.contamination_type is None

    def test_with_contamination(self):
        """Test result with contamination."""
        result = ContaminationCheckResult(
            is_contaminated=True,
            contamination_type="public_dataset",
            evidence=["Found in public dataset"],
            recommendations=["Use private dataset"],
            severity="high",
        )

        assert result.is_contaminated is True
        assert result.contamination_type == "public_dataset"
        assert len(result.evidence) == 1
        assert len(result.recommendations) == 1
        assert result.severity == "high"


class TestContaminationDetector:
    """Tests for ContaminationDetector."""

    def test_detector_default_config(self):
        """Test detector with default config."""
        detector = ContaminationDetector()

        assert detector.config.check_public_availability is True
        assert detector.fingerprints == {}

    def test_detector_custom_config(self):
        """Test detector with custom config."""
        config = ContaminationConfig(check_public_availability=False)
        detector = ContaminationDetector(config)

        assert detector.config.check_public_availability is False

    def test_compute_content_hash(self):
        """Test computing content hash."""
        detector = ContaminationDetector()
        test_cases = [
            TestCase(input="q1", expected="a1", actual_output="a1"),
            TestCase(input="q2", expected="a2", actual_output="a2"),
        ]

        hash1 = detector._compute_content_hash(test_cases)

        assert isinstance(hash1, str)
        assert len(hash1) == 64  # SHA-256 hex digest

    def test_compute_content_hash_deterministic(self):
        """Test hash is deterministic (same input → same hash)."""
        detector = ContaminationDetector()
        test_cases = [
            TestCase(input="q1", expected="a1", actual_output="a1"),
            TestCase(input="q2", expected="a2", actual_output="a2"),
        ]

        hash1 = detector._compute_content_hash(test_cases)
        hash2 = detector._compute_content_hash(test_cases)

        assert hash1 == hash2

    def test_compute_content_hash_different_data(self):
        """Test different data produces different hash."""
        detector = ContaminationDetector()
        test_cases1 = [TestCase(input="q1", expected="a1", actual_output="a1")]
        test_cases2 = [TestCase(input="q2", expected="a2", actual_output="a2")]

        hash1 = detector._compute_content_hash(test_cases1)
        hash2 = detector._compute_content_hash(test_cases2)

        assert hash1 != hash2

    def test_compute_content_hash_order_independent(self):
        """Test hash is order-independent."""
        detector = ContaminationDetector()
        test_cases1 = [
            TestCase(input="q1", expected="a1", actual_output="a1"),
            TestCase(input="q2", expected="a2", actual_output="a2"),
        ]
        test_cases2 = [
            TestCase(input="q2", expected="a2", actual_output="a2"),
            TestCase(input="q1", expected="a1", actual_output="a1"),
        ]

        hash1 = detector._compute_content_hash(test_cases1)
        hash2 = detector._compute_content_hash(test_cases2)

        # Should be same (sorted before hashing)
        assert hash1 == hash2

    def test_create_fingerprint(self):
        """Test creating dataset fingerprint."""
        detector = ContaminationDetector()
        test_cases = [
            TestCase(input="q1", expected="a1", actual_output="a1"),
            TestCase(input="q2", expected="a2", actual_output="a2"),
        ]

        fingerprint = detector.create_fingerprint(
            test_cases=test_cases,
            version="1.0",
            release_date="2025-01-01",
            is_public=True,
            public_url="https://example.com/data",
        )

        assert isinstance(fingerprint, DatasetFingerprint)
        assert fingerprint.version == "1.0"
        assert fingerprint.release_date == "2025-01-01"
        assert fingerprint.num_test_cases == 2
        assert fingerprint.is_public is True
        assert fingerprint.public_url == "https://example.com/data"
        assert len(fingerprint.content_hash) == 64

    def test_create_fingerprint_defaults(self):
        """Test fingerprint with default values."""
        detector = ContaminationDetector()
        test_cases = [TestCase(input="q1", expected="a1", actual_output="a1")]

        fingerprint = detector.create_fingerprint(
            test_cases=test_cases, version="1.0", release_date="2025-01-01"
        )

        assert fingerprint.version == "1.0"
        assert fingerprint.is_public is False
        assert fingerprint.public_url is None

    def test_register_dataset(self):
        """Test registering dataset."""
        detector = ContaminationDetector()
        test_cases = [TestCase(input="q1", expected="a1", actual_output="a1")]

        fingerprint = detector.register_dataset(
            test_cases, version="1.0", release_date="2025-01-01"
        )

        assert isinstance(fingerprint, DatasetFingerprint)
        assert "1.0" in detector.fingerprints
        assert detector.fingerprints["1.0"] == fingerprint

    def test_check_version_exists(self):
        """Test checking if version exists."""
        detector = ContaminationDetector()
        test_cases = [TestCase(input="q1", expected="a1", actual_output="a1")]

        detector.register_dataset(test_cases, version="1.0", release_date="2025-01-01")

        assert "1.0" in detector.fingerprints
        assert "2.0" not in detector.fingerprints

    def test_check_duplicate_content(self):
        """Test checking for duplicate content."""
        detector = ContaminationDetector()

        # Register first dataset
        test_cases_v1 = [TestCase(input="q1", expected="a1", actual_output="a1")]
        detector.register_dataset(test_cases_v1, version="1.0", release_date="2025-01-01")

        # Check for duplicates with same content
        test_cases_v2 = [TestCase(input="q1", expected="a1", actual_output="a1")]
        result = detector.check_duplicate_content(test_cases_v2)

        assert isinstance(result, ContaminationCheckResult)
        # Same content should be detected
        assert result.is_contaminated is True

    def test_check_duplicate_content_no_duplicates(self):
        """Test no duplicates detected."""
        detector = ContaminationDetector()

        # Register first dataset
        test_cases_v1 = [TestCase(input="q1", expected="a1", actual_output="a1")]
        detector.register_dataset(test_cases_v1, version="1.0", release_date="2025-01-01")

        # Check with different content
        test_cases_v2 = [TestCase(input="q2", expected="a2", actual_output="a2")]
        result = detector.check_duplicate_content(test_cases_v2)

        assert result.is_contaminated is False

    def test_check_public_availability_not_public(self):
        """Test checking public availability - private dataset."""
        detector = ContaminationDetector()
        test_cases = [TestCase(input="q1", expected="a1", actual_output="a1")]

        detector.register_dataset(
            test_cases,
            version="1.0",
            release_date="2025-01-01",
            is_public=False,
        )

        result = detector.check_public_availability("1.0")

        assert isinstance(result, ContaminationCheckResult)
        assert result.is_contaminated is False

    def test_check_public_availability_is_public(self):
        """Test checking public availability - public dataset."""
        detector = ContaminationDetector()
        test_cases = [TestCase(input="q1", expected="a1", actual_output="a1")]

        detector.register_dataset(
            test_cases,
            version="1.0",
            release_date="2025-01-01",
            is_public=True,
            public_url="https://example.com/data",
        )

        result = detector.check_public_availability("1.0")

        # Public dataset is flagged as potential contamination
        assert result.is_contaminated is True
        assert len(result.evidence) > 0

    def test_check_temporal_contamination_no_issue(self):
        """Test temporal contamination check - no issue."""
        detector = ContaminationDetector()
        test_cases = [TestCase(input="q1", expected="a1", actual_output="a1")]

        # Dataset released after agent training cutoff
        detector.register_dataset(test_cases, version="1.0", release_date="2025-06-01")

        result = detector.check_temporal_contamination(
            version="1.0", agent_training_cutoff="2025-01-01"
        )

        # No contamination if dataset is newer than cutoff
        assert result.is_contaminated is False

    def test_check_temporal_contamination_detected(self):
        """Test temporal contamination detected."""
        detector = ContaminationDetector()
        test_cases = [TestCase(input="q1", expected="a1", actual_output="a1")]

        # Dataset released before agent training cutoff
        detector.register_dataset(test_cases, version="1.0", release_date="2024-01-01")

        result = detector.check_temporal_contamination(
            version="1.0", agent_training_cutoff="2025-01-01"
        )

        # Contamination if dataset is older than cutoff
        assert result.is_contaminated is True
        assert result.severity in ["medium", "high"]

    def test_generate_contamination_report(self):
        """Test generating contamination report."""
        detector = ContaminationDetector()
        test_cases = [TestCase(input="q1", expected="a1", actual_output="a1")]

        detector.register_dataset(
            test_cases,
            version="1.0",
            release_date="2025-01-01",
            is_public=True,
        )

        # Run multiple checks
        result1 = detector.check_public_availability("1.0")
        result2 = detector.check_duplicate_content(test_cases)

        # generate_contamination_report takes list of check results
        report = detector.generate_contamination_report([result1, result2])

        assert isinstance(report, str)
        assert len(report) > 50


class TestABCCompliance:
    """Tests for ABC compliance (R.3, R.4)."""

    def test_abc_r_3_public_availability_tracking(self):
        """Test ABC R.3: Reports public availability and release date."""
        detector = ContaminationDetector()
        test_cases = [TestCase(input="q1", expected="a1", actual_output="a1")]

        fingerprint = detector.create_fingerprint(
            test_cases=test_cases,
            version="1.0",
            release_date="2025-01-01",
            is_public=True,
            public_url="https://example.com/benchmark",
        )

        # Should track public availability and release date
        assert fingerprint.is_public is True
        assert fingerprint.release_date == "2025-01-01"
        assert fingerprint.public_url is not None

    def test_abc_r_4_contamination_detection(self):
        """Test ABC R.4: Detects and addresses contamination."""
        detector = ContaminationDetector()
        test_cases = [TestCase(input="q1", expected="a1", actual_output="a1")]

        # Register public dataset
        detector.register_dataset(
            test_cases,
            version="1.0",
            release_date="2024-01-01",
            is_public=True,
        )

        # Check for contamination
        result = detector.check_temporal_contamination(
            version="1.0", agent_training_cutoff="2025-01-01"
        )

        # Should detect potential contamination
        assert isinstance(result, ContaminationCheckResult)
        assert result.is_contaminated is True
        # Should provide recommendations
        assert len(result.recommendations) > 0
