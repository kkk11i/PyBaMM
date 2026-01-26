"""
Class to regularize sqrt and power operations in a PyBaMM expression tree.
"""

from __future__ import annotations

import pybamm


class RegularizeSqrtAndPower:
    """
    Callable that replaces Sqrt and Power nodes with reg_sqrt/reg_pow.

    Parameters
    ----------
    scales : dict[pybamm.Symbol, str | pybamm.Symbol]
        Mapping from original symbols to either:
        - Input name (str) where the scale value is stored
        - Direct scale value (pybamm.Symbol)
    inputs : dict[str, pybamm.Symbol]
        The inputs dict from FunctionParameter.
    delta : float, optional
        Regularization width.
    """

    __slots__ = ["scales", "symbol_to_name", "delta"]

    def __init__(
        self,
        scales: dict[pybamm.Symbol, str | pybamm.Symbol],
        inputs: dict[str, pybamm.Symbol],
        delta: float | None = None,
    ):
        self.scales = scales
        self.symbol_to_name = {symbol: name for name, symbol in inputs.items()}
        self.delta = delta

    def __call__(
        self,
        symbol: pybamm.Symbol,
        inputs: dict[str, pybamm.Symbol] | None = None,
        **kwargs,
    ) -> pybamm.Symbol:
        """Apply regularization to an expression tree."""
        if inputs is None:
            inputs = {}

        # Build resolved scales: processed_symbol -> scale
        resolved_scales = {}

        for original_symbol, scale_ref in self.scales.items():
            # Resolve the scale value
            if isinstance(scale_ref, str):
                # It's an input name - look up the processed value
                scale = inputs.get(scale_ref, pybamm.Scalar(1))
            else:
                # It's a direct Symbol (e.g., a Parameter) - use as-is
                # It will be processed later by process_symbol
                scale = scale_ref

            # Get the processed symbol
            if original_symbol in self.symbol_to_name:
                name = self.symbol_to_name[original_symbol]
                if name in inputs:
                    processed_symbol = inputs[name]
                    resolved_scales[processed_symbol] = scale
            else:
                # Complex expression - rebuild with processed symbols
                processed_expr = self._rebuild_expr(original_symbol, inputs)
                if processed_expr is not None:
                    resolved_scales[processed_expr] = scale

        return self._process(symbol, resolved_scales)

    def _rebuild_expr(self, expr, inputs):
        """Rebuild an expression using processed symbols."""
        if not expr.children:
            if expr in self.symbol_to_name:
                name = self.symbol_to_name[expr]
                return inputs.get(name)
            # Keep Scalars and other constants as-is
            if isinstance(expr, pybamm.Scalar):
                return expr
            return None

        new_children = []
        for child in expr.children:
            rebuilt = self._rebuild_expr(child, inputs)
            if rebuilt is None:
                return None
            new_children.append(rebuilt)

        return expr.create_copy(new_children=new_children)

    def _process(self, sym, resolved_scales):
        """Recursively replace Sqrt/Power with reg_sqrt/reg_pow."""
        if not sym.children:
            return sym

        new_children = [self._process(child, resolved_scales) for child in sym.children]

        if isinstance(sym, pybamm.Sqrt):
            child = new_children[0]
            scale = self._get_scale(child, resolved_scales)
            return pybamm.reg_pow(child, a=0.5, delta=self.delta, scale=scale)

        if isinstance(sym, pybamm.Power):
            base, exponent = new_children
            scale = self._get_scale(base, resolved_scales)
            return pybamm.reg_pow(base, exponent, delta=self.delta, scale=scale)

        if any(n is not o for n, o in zip(new_children, sym.children)):
            return sym.create_copy(new_children=new_children)
        return sym

    def _get_scale(self, expr, resolved_scales):
        """Get scale for an expression, defaulting to 1."""
        # First try exact match
        for var, scale in resolved_scales.items():
            if expr == var:
                return scale
        
        # For subtraction (c_max - c_s), try to match c_s
        if isinstance(expr, pybamm.Subtraction):
            right_child = expr.children[1]
            for var, scale in resolved_scales.items():
                if right_child == var:
                    return scale
        
        return pybamm.Scalar(1)
