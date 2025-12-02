"""Tests for state match metric."""

import pytest

from evaris.metrics.state_match import StateMatchConfig, StateMatchMetric
from evaris.types import TestCase


class TestStateMatchConfig:
    """Tests for StateMatchConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = StateMatchConfig()

        assert config.comparison_mode == "exact"
        assert config.ignore_keys == []
        assert config.normalize_types is True
        assert config.tolerance == 1e-6
        assert config.check_side_effects is True
        assert config.allowed_side_effects == []
        assert config.custom_comparator is None

    def test_custom_config(self):
        """Test custom configuration values."""

        def custom_comp(a, b):
            return (True, "Match")

        config = StateMatchConfig(
            comparison_mode="subset",
            ignore_keys=["timestamp", "id"],
            normalize_types=False,
            tolerance=0.01,
            check_side_effects=False,
            allowed_side_effects=["log", "cache"],
            custom_comparator=custom_comp,
        )

        assert config.comparison_mode == "subset"
        assert config.ignore_keys == ["timestamp", "id"]
        assert config.normalize_types is False
        assert config.tolerance == 0.01
        assert config.check_side_effects is False
        assert config.allowed_side_effects == ["log", "cache"]
        assert config.custom_comparator == custom_comp


class TestStateMatchMetric:
    """Tests for StateMatchMetric."""

    @pytest.fixture
    def metric(self):
        """Create metric instance for testing."""
        config = StateMatchConfig()
        return StateMatchMetric(config)

    def test_metric_initialization(self, metric):
        """Test metric initializes correctly."""
        assert metric.config is not None
        assert isinstance(metric.config, StateMatchConfig)

    def test_metric_initialization_default_config(self):
        """Test metric with default config."""
        metric = StateMatchMetric()
        assert metric.config.comparison_mode == "exact"

    def test_normalize_value_int_to_float(self, metric):
        """Test normalization converts int to float."""
        result = metric._normalize_value(42)

        assert result == 42.0
        assert isinstance(result, float)

    def test_normalize_value_boolean_preserved(self, metric):
        """Test normalization preserves booleans."""
        result = metric._normalize_value(True)

        assert result is True
        assert isinstance(result, bool)

    def test_normalize_value_nested_dict(self, metric):
        """Test normalization of nested dictionary."""
        data = {"a": 1, "b": {"c": 2, "d": 3}, "e": [4, 5, 6]}

        result = metric._normalize_value(data)

        assert result["a"] == 1.0
        assert result["b"]["c"] == 2.0
        assert result["b"]["d"] == 3.0
        assert result["e"] == [4.0, 5.0, 6.0]

    def test_normalize_value_list(self, metric):
        """Test normalization of list."""
        data = [1, 2, 3]

        result = metric._normalize_value(data)

        assert result == [1.0, 2.0, 3.0]

    def test_normalize_value_string_unchanged(self, metric):
        """Test normalization keeps strings unchanged."""
        result = metric._normalize_value("test")

        assert result == "test"

    def test_normalize_value_none_unchanged(self, metric):
        """Test normalization keeps None unchanged."""
        result = metric._normalize_value(None)

        assert result is None

    def test_normalize_value_disabled(self):
        """Test normalization when disabled."""
        config = StateMatchConfig(normalize_types=False)
        metric = StateMatchMetric(config)

        result = metric._normalize_value(42)

        assert result == 42
        assert isinstance(result, int)

    def test_compare_numeric_within_tolerance(self, metric):
        """Test numeric comparison within tolerance."""
        result = metric._compare_numeric(1.0, 1.0000001)

        assert result is True

    def test_compare_numeric_outside_tolerance(self, metric):
        """Test numeric comparison outside tolerance."""
        result = metric._compare_numeric(1.0, 1.001)

        assert result is False

    def test_compare_numeric_custom_tolerance(self):
        """Test numeric comparison with custom tolerance."""
        config = StateMatchConfig(tolerance=0.1)
        metric = StateMatchMetric(config)

        result = metric._compare_numeric(1.0, 1.05)

        assert result is True

    def test_compare_values_exact_match(self, metric):
        """Test comparing identical values."""
        match, diffs = metric._compare_values(42, 42)

        assert match is True
        assert len(diffs) == 0

    def test_compare_values_type_mismatch(self, metric):
        """Test comparing values with different types."""
        match, diffs = metric._compare_values("42", 42)

        assert match is False
        assert len(diffs) == 1
        assert "Type mismatch" in diffs[0]

    def test_compare_values_dict_match(self, metric):
        """Test comparing identical dictionaries."""
        expected = {"a": 1, "b": 2}
        actual = {"a": 1, "b": 2}

        match, diffs = metric._compare_values(expected, actual)

        assert match is True
        assert len(diffs) == 0

    def test_compare_values_dict_missing_key(self, metric):
        """Test comparing dictionaries with missing key."""
        expected = {"a": 1, "b": 2}
        actual = {"a": 1}

        match, diffs = metric._compare_values(expected, actual)

        assert match is False
        assert len(diffs) == 1
        assert "Missing key" in diffs[0]

    def test_compare_values_dict_extra_key_exact_mode(self, metric):
        """Test comparing dictionaries with extra key in exact mode."""
        expected = {"a": 1}
        actual = {"a": 1, "b": 2}

        match, diffs = metric._compare_values(expected, actual)

        assert match is False
        assert len(diffs) == 1
        assert "Unexpected key" in diffs[0]

    def test_compare_values_dict_nested_mismatch(self, metric):
        """Test comparing nested dictionaries with mismatch."""
        expected = {"a": {"b": 1}}
        actual = {"a": {"b": 2}}

        match, diffs = metric._compare_values(expected, actual)

        assert match is False
        assert len(diffs) == 1
        assert "a.b" in diffs[0]

    def test_compare_values_list_match(self, metric):
        """Test comparing identical lists."""
        expected = [1, 2, 3]
        actual = [1, 2, 3]

        match, diffs = metric._compare_values(expected, actual)

        assert match is True
        assert len(diffs) == 0

    def test_compare_values_list_length_mismatch(self, metric):
        """Test comparing lists with different lengths."""
        expected = [1, 2, 3]
        actual = [1, 2]

        match, diffs = metric._compare_values(expected, actual)

        assert match is False
        assert len(diffs) == 1
        assert "length mismatch" in diffs[0].lower()

    def test_compare_values_list_value_mismatch(self, metric):
        """Test comparing lists with different values."""
        expected = [1, 2, 3]
        actual = [1, 2, 4]

        match, diffs = metric._compare_values(expected, actual)

        assert match is False
        assert len(diffs) == 1
        assert "[2]" in diffs[0]

    def test_compare_values_numeric_with_tolerance(self, metric):
        """Test comparing numeric values with tolerance."""
        match, diffs = metric._compare_values(1.0, 1.0000001)

        assert match is True
        assert len(diffs) == 0

    def test_compare_values_numeric_outside_tolerance(self, metric):
        """Test comparing numeric values outside tolerance."""
        match, diffs = metric._compare_values(1.0, 1.001)

        assert match is False
        assert len(diffs) == 1
        assert "Numeric mismatch" in diffs[0]

    def test_compare_values_string_mismatch(self, metric):
        """Test comparing different strings."""
        match, diffs = metric._compare_values("hello", "world")

        assert match is False
        assert len(diffs) == 1
        assert "Value mismatch" in diffs[0]

    def test_compare_values_ignore_keys(self):
        """Test comparing with ignored keys."""
        config = StateMatchConfig(ignore_keys=["timestamp"])
        metric = StateMatchMetric(config)

        expected = {"value": 1, "timestamp": "2025-01-01"}
        actual = {"value": 1, "timestamp": "2025-01-02"}

        match, diffs = metric._compare_values(expected, actual)

        assert match is True  # timestamp ignored
        assert len(diffs) == 0

    def test_check_side_effects_none(self, metric):
        """Test side effects check with no side effects."""
        expected = {"a": 1, "b": 2}
        actual = {"a": 1, "b": 2}

        no_violations, side_effects = metric._check_side_effects(expected, actual)

        assert no_violations is True
        assert len(side_effects) == 0

    def test_check_side_effects_detected(self, metric):
        """Test side effects check detects unexpected keys."""
        expected = {"a": 1}
        actual = {"a": 1, "unexpected_key": "value"}

        no_violations, side_effects = metric._check_side_effects(expected, actual)

        assert no_violations is False
        assert len(side_effects) == 1
        assert "unexpected_key" in side_effects[0]

    def test_check_side_effects_allowed(self):
        """Test side effects check with allowed side effects."""
        config = StateMatchConfig(allowed_side_effects=["log"])
        metric = StateMatchMetric(config)

        expected = {"a": 1}
        actual = {"a": 1, "log": "Some log message"}

        no_violations, side_effects = metric._check_side_effects(expected, actual)

        assert no_violations is True
        assert len(side_effects) == 0

    def test_check_side_effects_disabled(self):
        """Test side effects check when disabled."""
        config = StateMatchConfig(check_side_effects=False)
        metric = StateMatchMetric(config)

        expected = {"a": 1}
        actual = {"a": 1, "unexpected_key": "value"}

        no_violations, side_effects = metric._check_side_effects(expected, actual)

        assert no_violations is True
        assert len(side_effects) == 0

    def test_check_side_effects_ignore_keys(self):
        """Test side effects check with ignore_keys."""
        config = StateMatchConfig(ignore_keys=["metadata"])
        metric = StateMatchMetric(config)

        expected = {"a": 1}
        actual = {"a": 1, "metadata": {"created": "2025-01-01"}}

        no_violations, side_effects = metric._check_side_effects(expected, actual)

        assert no_violations is True  # metadata is ignored

    def test_compare_states_exact_mode(self, metric):
        """Test state comparison in exact mode."""
        expected = {"a": 1, "b": 2}
        actual = {"a": 1, "b": 2}

        match, score, diffs = metric._compare_states(expected, actual)

        assert match is True
        assert score == 1.0
        assert len(diffs) == 0

    def test_compare_states_subset_mode(self):
        """Test state comparison in subset mode."""
        config = StateMatchConfig(comparison_mode="subset", check_side_effects=False)
        metric = StateMatchMetric(config)

        expected = {"a": 1}
        actual = {"a": 1, "b": 2}  # Extra key is okay in subset mode

        match, score, diffs = metric._compare_states(expected, actual)

        assert match is True
        assert score == 1.0

    def test_compare_states_subset_mode_missing_key(self):
        """Test state comparison in subset mode with missing key."""
        config = StateMatchConfig(comparison_mode="subset")
        metric = StateMatchMetric(config)

        expected = {"a": 1, "b": 2}
        actual = {"a": 1}

        match, score, diffs = metric._compare_states(expected, actual)

        assert match is False
        assert "Missing expected key" in diffs[0]

    def test_compare_states_custom_comparator(self):
        """Test state comparison with custom comparator."""

        def custom_comp(expected, actual):
            return (True, "Custom match")

        config = StateMatchConfig(custom_comparator=custom_comp)
        metric = StateMatchMetric(config)

        match, score, diffs = metric._compare_states({"a": 1}, {"b": 2})

        assert match is True
        assert score == 1.0
        assert len(diffs) == 0

    def test_compare_states_custom_comparator_failure(self):
        """Test state comparison with custom comparator failure."""

        def custom_comp(expected, actual):
            return (False, "Custom failure reason")

        config = StateMatchConfig(custom_comparator=custom_comp)
        metric = StateMatchMetric(config)

        match, score, diffs = metric._compare_states({"a": 1}, {"b": 2})

        assert match is False
        assert score == 0.0
        assert len(diffs) == 1
        assert "Custom failure reason" in diffs[0]

    def test_compare_states_with_side_effects(self, metric):
        """Test state comparison detects side effects."""
        expected = {"a": 1}
        actual = {"a": 1, "unexpected": "value"}

        match, score, diffs = metric._compare_states(expected, actual)

        assert match is False
        assert score == 0.0  # Should be 0.0 since side effects make match False
        assert any("side effect" in d.lower() for d in diffs)

    def test_compare_states_non_dict_values(self, metric):
        """Test state comparison with non-dict values."""
        match, score, diffs = metric._compare_states(42, 42)

        assert match is True
        assert score == 1.0

    def test_compare_states_partial_credit(self, metric):
        """Test state comparison with partial credit."""
        expected = {"a": 1, "b": 2, "c": 3}
        actual = {"a": 1, "b": 2, "c": 999}  # Only c is wrong

        match, score, diffs = metric._compare_states(expected, actual)

        assert match is False
        assert 0.0 < score < 1.0  # Partial credit
        assert score == 2 / 3  # 2 out of 3 keys match

    def test_score_success(self, metric):
        """Test scoring with successful state match."""
        actual = {"files": ["/tmp/a.txt"], "moved": True}
        tc = TestCase(
            input="Move file",
            expected={"state": {"files": ["/tmp/a.txt"], "moved": True}},
            actual_output=actual,
        )

        result = metric.score(tc, actual)

        assert result.name == "state_match"
        assert result.score == 1.0
        assert result.passed is True
        assert result.metadata["match"] is True
        assert result.metadata["num_differences"] == 0

    def test_score_failure(self, metric):
        """Test scoring with state mismatch."""
        actual = {"value": 2}
        tc = TestCase(input="test", expected={"state": {"value": 1}}, actual_output=actual)

        result = metric.score(tc, actual)

        assert result.score < 1.0
        assert result.passed is False
        assert result.metadata["num_differences"] > 0

    def test_score_partial_match(self, metric):
        """Test scoring with partial state match."""
        actual = {"a": 1, "b": 2, "c": 999}
        tc = TestCase(
            input="test", expected={"state": {"a": 1, "b": 2, "c": 3}}, actual_output=actual
        )

        result = metric.score(tc, actual)

        assert 0.0 < result.score < 1.0
        assert result.passed is False

    def test_score_expected_without_state_key(self, metric):
        """Test scoring when expected doesn't have 'state' key."""
        actual = {"value": 1}
        tc = TestCase(input="test", expected={"value": 1}, actual_output=actual)

        result = metric.score(tc, actual)

        assert result.score == 1.0
        assert result.passed is True

    def test_score_no_expected_raises(self, metric):
        """Test score raises ValueError when expected is None."""
        tc = TestCase(input="test", expected=None, actual_output={"value": 1})

        with pytest.raises(ValueError, match="expected"):
            metric.score(tc, {"value": 1})

    def test_score_handles_exceptions(self, metric):
        """Test score handles exceptions gracefully."""
        actual = object()
        tc = TestCase(input="test", expected={"state": {"value": 1}}, actual_output=actual)

        # Pass invalid type - will cause type mismatch but not exception
        result = metric.score(tc, actual)

        assert result.score == 0.0
        assert result.passed is False
        # Type mismatch is reported in differences, not as error
        assert result.metadata["num_differences"] > 0

    def test_score_metadata_includes_config(self, metric):
        """Test score metadata includes configuration."""
        actual = {"value": 1}
        tc = TestCase(input="test", expected={"state": {"value": 1}}, actual_output=actual)

        result = metric.score(tc, actual)

        assert "comparison_mode" in result.metadata
        assert "check_side_effects" in result.metadata
        assert result.metadata["comparison_mode"] == "exact"


class TestABCCompliance:
    """Tests for ABC compliance (O.g.1, O.g.2, O.g.3)."""

    def test_abc_o_g_1_verifies_state_changes(self):
        """Test ABC O.g.1: Verifies state changes match expected outcomes."""
        metric = StateMatchMetric()

        actual = {"counter": 5}
        tc = TestCase(
            input="Increment counter", expected={"state": {"counter": 5}}, actual_output=actual
        )

        result = metric.score(tc, actual)

        # Should verify state matches expected
        assert result.passed is True
        assert result.metadata["match"] is True

    def test_abc_o_g_2_compares_final_states(self):
        """Test ABC O.g.2: Compares final states with goal states."""
        metric = StateMatchMetric()

        # Final state
        actual = {"position": [10, 10], "status": "complete"}
        # Goal state
        tc = TestCase(
            input="Reach goal state",
            expected={"state": {"position": [10, 10], "status": "complete"}},
            actual_output=actual,
        )

        result = metric.score(tc, actual)

        # Should compare final state with goal
        assert result.passed is True
        assert result.score == 1.0

    def test_abc_o_g_3_detects_unintended_side_effects(self):
        """Test ABC O.g.3: Detects unintended side effects."""
        config = StateMatchConfig(check_side_effects=True)
        metric = StateMatchMetric(config)

        # Actual has unintended side effect
        actual = {"field": "updated", "unintended_modification": "oops"}
        tc = TestCase(
            input="Update field", expected={"state": {"field": "updated"}}, actual_output=actual
        )

        result = metric.score(tc, actual)

        # Should detect side effect
        assert result.passed is False
        assert any("side effect" in d.lower() for d in result.metadata["differences"])

    def test_abc_o_g_3_no_false_positives_for_allowed_effects(self):
        """Test ABC O.g.3: No false positives for allowed side effects."""
        config = StateMatchConfig(
            comparison_mode="subset",  # Use subset to allow extra keys
            check_side_effects=True,
            allowed_side_effects=["log", "metadata"],
        )
        metric = StateMatchMetric(config)

        # Actual has allowed side effects
        actual = {"field": "updated", "log": "Operation logged", "metadata": {"timestamp": "2025"}}
        tc = TestCase(
            input="Update field", expected={"state": {"field": "updated"}}, actual_output=actual
        )

        result = metric.score(tc, actual)

        # Should not flag allowed side effects
        assert result.passed is True
