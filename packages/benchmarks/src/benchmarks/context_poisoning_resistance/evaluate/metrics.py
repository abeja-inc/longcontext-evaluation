import ast
import re
from fractions import Fraction
from typing import Any

from ..._core.evaluate.metrics import BaseMetrics
from ...config import SubtaskConfig
from ..predict.data import ContextPoisoningResistanceOutput
from ..settings import ContextPoisoningResistanceSettings


class _SafeEvaluator(ast.NodeVisitor):
    def visit_Expression(self, node: ast.Expression) -> Fraction:
        return self.visit(node.body)

    def visit_BinOp(self, node: ast.BinOp) -> Fraction:
        left = self.visit(node.left)
        right = self.visit(node.right)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            if right == 0:
                raise ZeroDivisionError("division by zero")
            return left / right
        raise ValueError(f"Unsupported operator: {type(node.op).__name__}")

    def visit_UnaryOp(self, node: ast.UnaryOp) -> Fraction:
        operand = self.visit(node.operand)
        if isinstance(node.op, ast.UAdd):
            return operand
        if isinstance(node.op, ast.USub):
            return -operand
        raise ValueError(f"Unsupported unary operator: {type(node.op).__name__}")

    def visit_Constant(self, node: ast.Constant) -> Fraction:
        if not isinstance(node.value, int):
            raise ValueError("Only integers are supported")
        return Fraction(node.value)

    def generic_visit(self, node: ast.AST) -> Fraction:
        raise ValueError(f"Unsupported syntax: {type(node).__name__}")


def normalize_equation(text: str) -> str:
    normalized = text.strip()
    if "</think>" in normalized:
        normalized = normalized.split("</think>")[-1].strip()
    normalized = normalized.replace("＋", "+")
    normalized = normalized.replace("−", "-")
    normalized = normalized.replace("ー", "-")
    normalized = normalized.replace("×", "*")
    normalized = normalized.replace("x", "*")
    normalized = normalized.replace("÷", "/")
    normalized = re.sub(r"\s+", "", normalized)
    return normalized


def extract_used_numbers(expr: str) -> list[str]:
    return re.findall(r"\d+", expr)


def is_valid_equation_text(text: str, target_numbers: list[str]) -> bool:
    normalized = normalize_equation(text)
    if not normalized or "=" not in normalized:
        return False
    if re.search(r"[^0-9+\-*/=()]", normalized):
        return False

    left, right = normalized.split("=", 1)
    if right != "10":
        return False

    used_numbers = extract_used_numbers(left)
    if sorted(used_numbers) != sorted(target_numbers):
        return False

    try:
        value = _SafeEvaluator().visit(ast.parse(left, mode="eval"))
    except (SyntaxError, ValueError, ZeroDivisionError):
        return False
    return value == Fraction(10)


def is_valid_equation_output(output: ContextPoisoningResistanceOutput) -> bool:
    return is_valid_equation_text(output.output, output.target_numbers)


class ContextPoisoningResistanceMetrics(
    BaseMetrics[
        ContextPoisoningResistanceSettings,
        ContextPoisoningResistanceOutput,
    ]
):
    _metric_registry = {
        "equation_validity_accuracy": "eval_equation_validity_accuracy",
    }

    def eval_equation_validity_accuracy(
        self,
        output: ContextPoisoningResistanceOutput,
        config: SubtaskConfig,
        settings: ContextPoisoningResistanceSettings,
        default_error_message: str,
        **kwargs: Any,
    ) -> float:
        if not output.output or output.output == default_error_message:
            return 0.0
        return 1.0 if is_valid_equation_output(output) else 0.0
