"""Integration tests for evaluate() with ABC compliance checking."""

import warnings

import pytest

from evaris import evaluate
from evaris.compliance import ABCComplianceConfig, ABCViolationError


def simple_agent(input_str: str) -> str:
    """Simple test agent."""
    return f"Response: {input_str}"


class TestEvaluateWithCompliance:
    """Integration tests for evaluate() with compliance_config."""

    def test_evaluate_without_compliance_config(self) -> None:
        """Test evaluate without compliance config (backward compatibility)."""
        result = evaluate(
            name="test",
            task=simple_agent,
            data=[{"input": "test", "expected": "Response: test"}],
            metrics=["exact_match"],
        )

        assert result.passed == 1
        # Should work fine without compliance config

    def test_evaluate_with_compliance_non_strict(self) -> None:
        """Test evaluate with non-strict compliance checking."""
        config = ABCComplianceConfig(enabled=True, strict_mode=False)

        # Should generate warnings but not block
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            result = evaluate(
                name="test",
                task=simple_agent,
                data=[
                    {"input": f"test{i}", "expected": f"Response: test{i}"} for i in range(50)
                ],  # Good sample size
                metrics=["exact_match"],
                compliance_config=config,
            )

            # Should complete successfully
            assert result.total == 50

            # Should have warnings about missing baselines/statistics
            assert len(w) > 0
            assert any("ABC Compliance" in str(warning.message) for warning in w)

    def test_evaluate_with_compliance_strict_small_sample(self) -> None:
        """Test evaluate with strict mode and small sample (should block)."""
        config = ABCComplianceConfig(enabled=True, strict_mode=True)

        with pytest.raises(ABCViolationError) as exc_info:
            evaluate(
                name="test",
                task=simple_agent,
                data=[{"input": "test1", "expected": "Response: test1"}] * 5,  # Only 5 samples
                metrics=["exact_match"],
                compliance_config=config,
            )

        # Should block due to small sample size
        assert "sample size" in str(exc_info.value).lower()

    def test_evaluate_with_compliance_strict_good_sample(self) -> None:
        """Test evaluate with strict mode and good sample size."""
        config = ABCComplianceConfig(enabled=True, strict_mode=True)

        # With good sample size, should only have warnings, not critical errors
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")

            result = evaluate(
                name="test",
                task=simple_agent,
                data=[
                    {"input": f"test{i}", "expected": f"Response: test{i}"} for i in range(50)
                ],  # Good sample size
                metrics=["exact_match"],
                compliance_config=config,
            )

            # Should complete successfully
            assert result.total == 50

    def test_evaluate_with_compliance_disabled(self) -> None:
        """Test evaluate with compliance disabled."""
        config = ABCComplianceConfig(enabled=False)

        result = evaluate(
            name="test",
            task=simple_agent,
            data=[{"input": "test", "expected": "Response: test"}] * 5,
            metrics=["exact_match"],
            compliance_config=config,
        )

        # Should work without any checks
        assert result.total == 5

    def test_evaluate_compliance_warning_content(self) -> None:
        """Test that compliance warnings contain helpful information."""
        config = ABCComplianceConfig(enabled=True, strict_mode=False)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            evaluate(
                name="test",
                task=simple_agent,
                data=[{"input": "test", "expected": "Response: test"}] * 20,
                metrics=["exact_match"],
                compliance_config=config,
            )

            # Check warning content
            warning_messages = [str(warning.message) for warning in w]
            combined_message = " ".join(warning_messages)

            # Should mention ABC checks
            assert (
                "ABC" in combined_message
                or "baseline" in combined_message
                or "statistical" in combined_message
            )

    def test_evaluate_compliance_custom_thresholds(self) -> None:
        """Test compliance with custom minimum sample size - warns but doesn't block."""
        config = ABCComplianceConfig(
            enabled=True,
            strict_mode=False,  # Sample size is warning not critical
            min_sample_size=100,  # Higher threshold
        )

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            result = evaluate(
                name="test",
                task=simple_agent,
                data=[
                    {"input": f"test{i}", "expected": f"Response: test{i}"} for i in range(50)
                ],  # Below custom threshold
                metrics=["exact_match"],
                compliance_config=config,
            )

            # Should complete with warnings
            assert result.total == 50
            # Should warn about sample size
            assert any("sample size" in str(warning.message).lower() for warning in w)

    def test_evaluate_compliance_checks_can_be_disabled(self) -> None:
        """Test that individual compliance checks can be disabled."""
        config = ABCComplianceConfig(
            enabled=True,
            strict_mode=False,
            check_baselines=False,  # Disable baseline check
            check_statistics=False,  # Disable statistics check
            check_sample_size=False,  # Disable sample size check
        )

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            result = evaluate(
                name="test",
                task=simple_agent,
                data=[{"input": "test", "expected": "Response: test"}] * 5,
                metrics=["exact_match"],
                compliance_config=config,
            )

            # Should complete without warnings about disabled checks
            assert result.total == 5

            # Should not warn about baselines, statistics, or sample size
            warning_messages = [str(warning.message).lower() for warning in w]
            combined = " ".join(warning_messages)

            # These specific warnings should not appear
            assert "baseline" not in combined or "R.12" not in combined
            assert "statistical" not in combined or "R.10" not in combined


class TestComplianceIntegrationRealWorld:
    """Real-world integration scenarios for compliance."""

    def test_minimal_evaluation_gets_warnings(self) -> None:
        """Test that minimal evaluation triggers appropriate warnings."""
        config = ABCComplianceConfig(enabled=True, strict_mode=False)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            evaluate(
                name="minimal",
                task=simple_agent,
                data=[{"input": "test", "expected": "Response: test"}] * 15,
                metrics=["exact_match"],
                compliance_config=config,
            )

            # Should have multiple warnings
            assert len(w) > 0

    def test_compliance_does_not_affect_results(self) -> None:
        """Test that compliance checking doesn't change evaluation results."""
        # Run without compliance
        result1 = evaluate(
            name="test1",
            task=simple_agent,
            data=[{"input": "test", "expected": "Response: test"}] * 30,
            metrics=["exact_match"],
        )

        # Run with non-strict compliance
        config = ABCComplianceConfig(enabled=True, strict_mode=False)
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")

            result2 = evaluate(
                name="test2",
                task=simple_agent,
                data=[{"input": "test", "expected": "Response: test"}] * 30,
                metrics=["exact_match"],
                compliance_config=config,
            )

        # Results should be identical
        assert result1.total == result2.total
        assert result1.passed == result2.passed
        assert result1.accuracy == result2.accuracy
