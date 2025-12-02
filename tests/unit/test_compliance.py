"""Tests for ABC compliance checking module."""

import pytest

from evaris.compliance import (
    ABCComplianceChecker,
    ABCComplianceConfig,
    ABCComplianceReport,
    ABCViolationError,
    ABCWarning,
    check_compliance,
)


class TestABCComplianceConfig:
    """Tests for ABCComplianceConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = ABCComplianceConfig()

        assert config.enabled is True
        assert config.strict_mode is False
        assert config.check_baselines is True
        assert config.check_statistics is True
        assert config.check_sample_size is True
        assert config.check_validation is True
        assert config.check_contamination is True
        assert config.min_sample_size == 30

    def test_custom_config(self):
        """Test custom configuration values."""
        config = ABCComplianceConfig(
            enabled=False,
            strict_mode=True,
            check_baselines=False,
            min_sample_size=50,
        )

        assert config.enabled is False
        assert config.strict_mode is True
        assert config.check_baselines is False
        assert config.min_sample_size == 50


class TestABCWarning:
    """Tests for ABCWarning."""

    def test_warning_creation(self):
        """Test creating an ABC warning."""
        warning = ABCWarning(
            code="R.10",
            severity="warning",
            message="Test warning",
            fix="Fix suggestion",
            impact="Test impact",
            tier=1,
        )

        assert warning.code == "R.10"
        assert warning.severity == "warning"
        assert warning.message == "Test warning"
        assert warning.fix == "Fix suggestion"
        assert warning.impact == "Test impact"
        assert warning.tier == 1

    def test_warning_optional_fields(self):
        """Test warning with optional fields as None."""
        warning = ABCWarning(code="T.3", severity="info", message="Info message", tier=1)

        assert warning.fix is None
        assert warning.impact is None


class TestABCComplianceReport:
    """Tests for ABCComplianceReport."""

    def test_empty_report(self):
        """Test empty compliance report."""
        report = ABCComplianceReport()

        assert report.warnings == []
        assert report.num_critical == 0
        assert report.num_warnings == 0
        assert report.num_info == 0
        assert report.is_compliant is True
        assert not report.has_critical()
        assert not report.has_warnings()

    def test_report_with_warnings(self):
        """Test report with warnings."""
        warnings = [
            ABCWarning(code="R.10", severity="critical", message="Critical issue", tier=1),
            ABCWarning(code="R.12", severity="warning", message="Warning issue", tier=1),
            ABCWarning(code="T.9", severity="info", message="Info message", tier=1),
        ]

        report = ABCComplianceReport(
            warnings=warnings,
            num_critical=1,
            num_warnings=1,
            num_info=1,
            is_compliant=False,
        )

        assert len(report.warnings) == 3
        assert report.num_critical == 1
        assert report.num_warnings == 1
        assert report.num_info == 1
        assert report.is_compliant is False
        assert report.has_critical()
        assert report.has_warnings()

    def test_format_warnings_empty(self):
        """Test formatting empty report."""
        report = ABCComplianceReport()
        output = report.format_warnings()

        assert "All checks passed" in output

    def test_format_warnings_with_issues(self):
        """Test formatting report with issues."""
        warnings = [
            ABCWarning(
                code="R.10",
                severity="critical",
                message="Critical issue",
                fix="Fix it",
                impact="Bad impact",
                tier=1,
            ),
            ABCWarning(
                code="R.12", severity="warning", message="Warning issue", fix="Fix warning", tier=1
            ),
            ABCWarning(code="T.9", severity="info", message="Info message", tier=1),
        ]

        report = ABCComplianceReport(
            warnings=warnings,
            num_critical=1,
            num_warnings=1,
            num_info=1,
            is_compliant=False,
        )

        output = report.format_warnings()

        assert "CRITICAL ISSUES" in output
        assert "[R.10]" in output
        assert "Critical issue" in output
        assert "Fix it" in output
        assert "Bad impact" in output
        assert "WARNINGS" in output
        assert "[R.12]" in output
        assert "RECOMMENDATIONS" in output
        assert "[T.9]" in output


class TestABCComplianceChecker:
    """Tests for ABCComplianceChecker."""

    def test_checker_default_config(self):
        """Test checker with default config."""
        checker = ABCComplianceChecker()

        assert checker.config.enabled is True
        assert checker.config.strict_mode is False

    def test_checker_custom_config(self):
        """Test checker with custom config."""
        config = ABCComplianceConfig(strict_mode=True, min_sample_size=50)
        checker = ABCComplianceChecker(config)

        assert checker.config.strict_mode is True
        assert checker.config.min_sample_size == 50

    def test_check_evaluation_config_all_good(self):
        """Test checking evaluation with all features enabled."""
        checker = ABCComplianceChecker()

        report = checker.check_evaluation_config(
            includes_baselines=True,
            includes_statistics=True,
            includes_validation=True,
            includes_contamination_check=True,
            sample_size=50,
            uses_oracle_validation=True,
        )

        # Should only have Tier 2 info messages (agent isolation, state reset)
        assert report.is_compliant is True
        assert report.num_critical == 0
        assert report.num_warnings == 0
        assert report.num_info == 2  # Tier 2 recommendations

    def test_check_evaluation_config_missing_baselines(self):
        """Test checking evaluation without baselines."""
        checker = ABCComplianceChecker()

        report = checker.check_evaluation_config(
            includes_baselines=False,
            includes_statistics=True,
            sample_size=50,
        )

        assert report.num_warnings >= 1
        assert any(w.code == "R.12/R.13" for w in report.warnings)
        assert any("baseline" in w.message.lower() for w in report.warnings)

    def test_check_evaluation_config_missing_statistics(self):
        """Test checking evaluation without statistics."""
        checker = ABCComplianceChecker()

        report = checker.check_evaluation_config(
            includes_baselines=True,
            includes_statistics=False,
            sample_size=50,
        )

        assert report.num_warnings >= 1
        assert any(w.code == "R.10" for w in report.warnings)
        assert any("statistical" in w.message.lower() for w in report.warnings)

    def test_check_evaluation_config_small_sample_size(self):
        """Test checking evaluation with small sample size."""
        checker = ABCComplianceChecker()

        report = checker.check_evaluation_config(
            includes_baselines=True,
            includes_statistics=True,
            sample_size=10,  # Below minimum of 30
        )

        assert report.num_warnings >= 1 or report.num_critical >= 1
        assert any(
            "sample size" in w.message.lower() or "Sample size" in w.message
            for w in report.warnings
        )

    def test_check_evaluation_config_very_small_sample_critical(self):
        """Test that very small sample size is critical."""
        checker = ABCComplianceChecker()

        report = checker.check_evaluation_config(
            includes_baselines=True,
            includes_statistics=True,
            sample_size=5,  # Very small
        )

        assert report.num_critical >= 1
        assert not report.is_compliant

    def test_check_evaluation_config_missing_validation(self):
        """Test checking evaluation without validation."""
        checker = ABCComplianceChecker()

        report = checker.check_evaluation_config(
            includes_baselines=True,
            includes_statistics=True,
            includes_validation=False,
            sample_size=50,
        )

        assert any(w.code == "T.3" for w in report.warnings)

    def test_check_evaluation_config_missing_contamination(self):
        """Test checking evaluation without contamination check."""
        checker = ABCComplianceChecker()

        report = checker.check_evaluation_config(
            includes_baselines=True,
            includes_statistics=True,
            includes_contamination_check=False,
            sample_size=50,
        )

        assert any(w.code == "R.3/R.4" for w in report.warnings)

    def test_check_evaluation_config_tier_2_isolation(self):
        """Test Tier 2 check for agent isolation."""
        checker = ABCComplianceChecker()

        report = checker.check_evaluation_config(
            includes_baselines=True,
            includes_statistics=True,
            sample_size=50,
            agent_isolated=False,  # Not isolated
        )

        # Should have Tier 2 info warning about isolation
        tier_2_warnings = [w for w in report.warnings if w.tier == 2]
        assert len(tier_2_warnings) >= 1
        assert any(w.code == "II.5" for w in tier_2_warnings)

    def test_check_evaluation_config_tier_2_state_reset(self):
        """Test Tier 2 check for state reset."""
        checker = ABCComplianceChecker()

        report = checker.check_evaluation_config(
            includes_baselines=True,
            includes_statistics=True,
            sample_size=50,
            state_reset_enabled=False,  # No state reset
        )

        # Should have Tier 2 info warning about state reset
        tier_2_warnings = [w for w in report.warnings if w.tier == 2]
        assert len(tier_2_warnings) >= 1
        assert any(w.code == "II.4" for w in tier_2_warnings)

    def test_check_evaluation_config_disabled_checks(self):
        """Test that disabled checks don't generate warnings."""
        config = ABCComplianceConfig(
            check_baselines=False,
            check_statistics=False,
            check_sample_size=False,
        )
        checker = ABCComplianceChecker(config)

        report = checker.check_evaluation_config(
            includes_baselines=False,
            includes_statistics=False,
            sample_size=5,
        )

        # Should only have Tier 2 info messages, no Tier 1 warnings
        assert not any(w.code in ["R.12/R.13", "R.10"] for w in report.warnings)


class TestABCViolationError:
    """Tests for ABCViolationError."""

    def test_violation_error_creation(self):
        """Test creating violation error."""
        warnings = [
            ABCWarning(
                code="R.10",
                severity="critical",
                message="Critical issue",
                fix="Fix it",
                tier=1,
            )
        ]
        report = ABCComplianceReport(warnings=warnings, num_critical=1, is_compliant=False)

        error = ABCViolationError(report)

        assert error.report == report
        assert "ABC compliance violations detected" in str(error)
        assert "Critical issue" in str(error)

    def test_violation_error_has_report(self):
        """Test that error contains the report."""
        report = ABCComplianceReport(
            warnings=[ABCWarning(code="T.3", severity="critical", message="Test", tier=1)],
            num_critical=1,
            is_compliant=False,
        )

        error = ABCViolationError(report)
        assert error.report is report


class TestCheckComplianceFunction:
    """Tests for check_compliance convenience function."""

    def test_check_compliance_non_strict(self):
        """Test check_compliance in non-strict mode."""
        report = check_compliance(
            includes_baselines=False,
            includes_statistics=False,
            sample_size=20,
            strict_mode=False,
        )

        assert isinstance(report, ABCComplianceReport)
        assert report.has_warnings()
        # Should not raise exception in non-strict mode

    def test_check_compliance_strict_with_critical(self):
        """Test check_compliance in strict mode with critical issues."""
        with pytest.raises(ABCViolationError) as exc_info:
            check_compliance(
                includes_baselines=False,
                includes_statistics=False,
                sample_size=5,  # Critical: too small
                strict_mode=True,
            )

        assert isinstance(exc_info.value, ABCViolationError)
        assert exc_info.value.report.has_critical()

    def test_check_compliance_strict_no_critical(self):
        """Test check_compliance in strict mode without critical issues."""
        # Should not raise if no critical issues
        report = check_compliance(
            includes_baselines=False,  # Warning, not critical
            includes_statistics=True,
            sample_size=50,  # Good sample size
            strict_mode=True,
        )

        assert isinstance(report, ABCComplianceReport)
        assert not report.has_critical()


class TestComplianceIntegration:
    """Integration tests for compliance checking."""

    def test_full_compliant_evaluation(self):
        """Test fully compliant evaluation configuration."""
        checker = ABCComplianceChecker()

        report = checker.check_evaluation_config(
            includes_baselines=True,
            includes_statistics=True,
            includes_validation=True,
            includes_contamination_check=True,
            sample_size=100,
            uses_oracle_validation=True,
            agent_isolated=True,
            state_reset_enabled=True,
        )

        # Fully compliant - no warnings at all
        assert report.is_compliant is True
        assert report.num_critical == 0
        assert report.num_warnings == 0
        # Only info messages should be present
        assert report.num_info == 0 or all(w.severity == "info" for w in report.warnings)

    def test_minimal_evaluation_warnings(self):
        """Test minimal evaluation generates appropriate warnings."""
        checker = ABCComplianceChecker()

        report = checker.check_evaluation_config(
            includes_baselines=False,
            includes_statistics=False,
            includes_validation=False,
            includes_contamination_check=False,
            sample_size=10,
        )

        # Should have multiple warnings
        assert report.has_warnings()
        assert report.num_warnings >= 3  # At least baselines, statistics, validation

    def test_custom_min_sample_size(self):
        """Test custom minimum sample size threshold."""
        config = ABCComplianceConfig(min_sample_size=100)
        checker = ABCComplianceChecker(config)

        report = checker.check_evaluation_config(
            includes_baselines=True,
            includes_statistics=True,
            sample_size=50,  # Below custom threshold
        )

        assert any("sample size" in w.message.lower() for w in report.warnings)
        assert any("100" in w.message for w in report.warnings)
