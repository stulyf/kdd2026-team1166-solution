from __future__ import annotations

import ast
import operator
import re
from typing import Any, Callable


BINARY_OPS: dict[type[ast.operator], Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

UNARY_OPS: dict[type[ast.unaryop], Callable[[Any], Any]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def evaluate_expression(expression: str) -> float | int:
    tree = ast.parse(expression, mode="eval")
    return _eval_node(tree.body)


def _eval_node(node: ast.AST) -> float | int:
    if isinstance(node, ast.Constant):
        return node.value
    elif isinstance(node, ast.BinOp) and type(node.op) in BINARY_OPS:
        left_val = _eval_node(node.left)
        right_val = _eval_node(node.right)
        result = BINARY_OPS[type(node.op)](left_val, right_val)
        if isinstance(result, float) and result.is_integer():
            return int(result)
        return result
    elif isinstance(node, ast.UnaryOp) and type(node.op) in UNARY_OPS:
        operand_val = _eval_node(node.operand)
        return UNARY_OPS[type(node.op)](operand_val)
    else:
        raise ValueError(f"Unsupported expression: {ast.dump(node)}")


def calculate(expression: str) -> dict[str, Any]:
    cleaned = re.sub(r"\s+", "", expression)
    try:
        result = evaluate_expression(cleaned)
        return {"status": "success", "expression": expression, "result": result}
    except Exception as e:
        return {"status": "error", "expression": expression, "error": str(e)}
