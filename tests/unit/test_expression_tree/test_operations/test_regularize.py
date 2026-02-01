#
# Tests for the RegularizeSqrtAndPower class
#

import pybamm


class TestRegularizeSqrtAndPower:
    def test_basic_sqrt_replacement(self):
        """Test that Sqrt nodes are replaced with regularized expressions."""
        c_e = pybamm.Variable("c_e")
        c_s = pybamm.Variable("c_s")

        inputs = {"c_e": c_e, "c_s": c_s}
        regularizer = pybamm.RegularizeSqrtAndPower(
            {
                c_e: pybamm.Scalar(1000.0),
                c_s: pybamm.Scalar(50000.0),
            },
            inputs=inputs,
        )

        expr = pybamm.sqrt(c_e) + pybamm.sqrt(c_s)
        result = regularizer(expr, inputs=inputs)

        # Check that result has no Sqrt nodes
        has_sqrt = any(isinstance(n, pybamm.Sqrt) for n in result.pre_order())
        assert not has_sqrt

    def test_basic_power_replacement(self):
        """Test that Power nodes are replaced with regularized expressions."""
        c_e = pybamm.Variable("c_e")

        inputs = {"c_e": c_e}
        regularizer = pybamm.RegularizeSqrtAndPower(
            {c_e: pybamm.Scalar(1000.0)},
            inputs=inputs,
        )

        expr = c_e**0.5
        result = regularizer(expr, inputs=inputs)

        # The original c_e**0.5 Power node should be replaced
        # (result will have different Power nodes from reg_power formula)
        assert result != expr

    def test_scale_default_to_one(self):
        """Test that unrecognized expressions get scale=1."""
        c_e = pybamm.Variable("c_e")
        c_s = pybamm.Variable("c_s")

        # Only c_e has a scale, c_s should get scale=1
        inputs = {"c_e": c_e, "c_s": c_s}
        regularizer = pybamm.RegularizeSqrtAndPower(
            {c_e: pybamm.Scalar(1000.0)},
            inputs=inputs,
        )

        expr = pybamm.sqrt(c_s)
        result = regularizer(expr, inputs=inputs)

        # Should still be replaced (no Sqrt)
        has_sqrt = any(isinstance(n, pybamm.Sqrt) for n in result.pre_order())
        assert not has_sqrt

    def test_exact_match_only(self):
        """Test that only exact matches are used for scales."""
        c_s = pybamm.Variable("c_s")
        c_s_max = pybamm.Parameter("c_s_max")

        # c_s has scale, but c_s / c_s_max should not inherit it
        inputs = {"c_s": c_s, "c_s_max": c_s_max}
        regularizer = pybamm.RegularizeSqrtAndPower(
            {c_s: c_s_max},
            inputs=inputs,
        )

        # sqrt(c_s) should be replaced
        expr1 = pybamm.sqrt(c_s)
        result1 = regularizer(expr1, inputs=inputs)
        has_sqrt = any(isinstance(n, pybamm.Sqrt) for n in result1.pre_order())
        assert not has_sqrt

        # sqrt(c_s / c_s_max) should also be replaced
        expr2 = pybamm.sqrt(c_s / c_s_max)
        result2 = regularizer(expr2, inputs=inputs)
        has_sqrt = any(isinstance(n, pybamm.Sqrt) for n in result2.pre_order())
        assert not has_sqrt

    def test_explicit_pattern_matching(self):
        """Test that explicit patterns like c_max - c_s can be added."""
        c_s = pybamm.Variable("c_s")
        c_s_max = pybamm.Parameter("c_s_max")

        inputs = {"c_s": c_s, "c_s_max": c_s_max}
        regularizer = pybamm.RegularizeSqrtAndPower(
            {
                c_s: c_s_max,
                c_s_max - c_s: c_s_max,  # explicit pattern
            },
            inputs=inputs,
        )

        expr = pybamm.sqrt(c_s_max - c_s)
        result = regularizer(expr, inputs=inputs)

        has_sqrt = any(isinstance(n, pybamm.Sqrt) for n in result.pre_order())
        assert not has_sqrt

    def test_custom_delta(self):
        """Test that custom delta from global settings is used."""
        c_e = pybamm.Variable("c_e")

        inputs = {"c_e": c_e}
        regularizer = pybamm.RegularizeSqrtAndPower(
            {c_e: pybamm.Scalar(1000.0)},
            inputs=inputs,
        )

        # Set custom delta via global settings
        old_delta = pybamm.settings.tolerances["reg_power"]
        try:
            pybamm.settings.tolerances["reg_power"] = 0.01

            expr = pybamm.sqrt(c_e)
            result = regularizer(expr, inputs=inputs)

            has_sqrt = any(isinstance(n, pybamm.Sqrt) for n in result.pre_order())
            assert not has_sqrt
        finally:
            pybamm.settings.tolerances["reg_power"] = old_delta

    def test_nested_expression(self):
        """Test that nested expressions are handled correctly."""
        c_e = pybamm.Variable("c_e")
        c_s = pybamm.Variable("c_s")

        inputs = {"c_e": c_e, "c_s": c_s}
        regularizer = pybamm.RegularizeSqrtAndPower(
            {
                c_e: pybamm.Scalar(1000.0),
                c_s: pybamm.Scalar(50000.0),
            },
            inputs=inputs,
        )

        # Nested expression with multiple sqrt
        expr = pybamm.sqrt(c_e) * pybamm.sqrt(c_s)

        result = regularizer(expr, inputs=inputs)

        # Should have no Sqrt nodes
        has_sqrt = any(isinstance(n, pybamm.Sqrt) for n in result.pre_order())
        assert not has_sqrt

    def test_processed_inputs(self):
        """Test that the regularizer works with processed (different) inputs."""
        # Original symbols (before processing)
        c_e_orig = pybamm.Variable("c_e")
        c_s_orig = pybamm.Variable("c_s")
        c_s_max_orig = pybamm.Parameter("c_s_max")

        original_inputs = {
            "c_e": c_e_orig,
            "c_s": c_s_orig,
            "c_s_max": c_s_max_orig,
        }

        regularizer = pybamm.RegularizeSqrtAndPower(
            {
                c_e_orig: pybamm.Scalar(1000.0),
                c_s_orig: c_s_max_orig,
                c_s_max_orig - c_s_orig: c_s_max_orig,
            },
            inputs=original_inputs,
        )

        # Simulated "processed" symbols (what ParameterSubstitutor would produce)
        c_e_proc = pybamm.StateVector(slice(0, 10), name="c_e")
        c_s_proc = pybamm.StateVector(slice(10, 20), name="c_s")
        c_s_max_proc = pybamm.Scalar(51765.0, name="c_s_max")

        processed_inputs = {
            "c_e": c_e_proc,
            "c_s": c_s_proc,
            "c_s_max": c_s_max_proc,
        }

        # Expression built from processed symbols (as returned by user's function)
        expr = pybamm.sqrt(c_e_proc) * pybamm.sqrt(c_s_proc) * pybamm.sqrt(c_s_max_proc - c_s_proc)
        result = regularizer(expr, inputs=processed_inputs)

        # Check that all sqrts were replaced
        has_sqrt = any(isinstance(n, pybamm.Sqrt) for n in result.pre_order())
        assert not has_sqrt
