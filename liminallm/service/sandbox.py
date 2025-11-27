from __future__ import annotations

import ast
import operator
from collections.abc import Mapping, Sequence
from typing import Any


_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.FloorDiv: operator.floordiv,
}

_CMP_OPS = {
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
    ast.In: lambda a, b: a in b,
    ast.NotIn: lambda a, b: a not in b,
    ast.Is: operator.is_,
    ast.IsNot: operator.is_not,
}


def _eval_node(node: ast.AST, names: Mapping[str, Any]) -> Any:
    if isinstance(node, ast.Expression):
        return _eval_node(node.body, names)

    if isinstance(node, ast.Constant):
        return node.value

    if isinstance(node, ast.Name):
        if node.id in names:
            return names[node.id]
        raise ValueError(f"unknown name {node.id}")

    if isinstance(node, ast.BoolOp):
        if isinstance(node.op, ast.And):
            result = True
            for value in node.values:
                result = bool(_eval_node(value, names))
                if not result:
                    break
            return result
        if isinstance(node.op, ast.Or):
            result = False
            for value in node.values:
                result = bool(_eval_node(value, names))
                if result:
                    break
            return result
        raise ValueError("unsupported boolean operator")

    if isinstance(node, ast.UnaryOp):
        if isinstance(node.op, ast.Not):
            return not bool(_eval_node(node.operand, names))
        if isinstance(node.op, ast.USub):
            return -_eval_node(node.operand, names)
        if isinstance(node.op, ast.UAdd):
            return +_eval_node(node.operand, names)
        raise ValueError("unsupported unary operator")

    if isinstance(node, ast.BinOp):
        op = _BIN_OPS.get(type(node.op))
        if op is None:
            raise ValueError("unsupported binary operator")
        return op(_eval_node(node.left, names), _eval_node(node.right, names))

    if isinstance(node, ast.Compare):
        left = _eval_node(node.left, names)
        for op_node, comparator in zip(node.ops, node.comparators):
            op = _CMP_OPS.get(type(op_node))
            if op is None:
                raise ValueError("unsupported comparator")
            right = _eval_node(comparator, names)
            if not op(left, right):
                return False
            left = right
        return True

    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ValueError("callable references must be simple names")
        if node.func.id not in names:
            raise ValueError(f"unknown callable {node.func.id}")
        func = names[node.func.id]
        if not callable(func):
            raise ValueError("call target is not callable")
        args = [_eval_node(arg, names) for arg in node.args]
        kwargs = {kw.arg: _eval_node(kw.value, names) for kw in node.keywords if kw.arg is not None}
        return func(*args, **kwargs)

    if isinstance(node, ast.Subscript):
        target = _eval_node(node.value, names)
        index = _eval_node(node.slice, names)
        if not isinstance(target, (Mapping, Sequence, str, bytes)):
            raise ValueError("subscript targets must be sequences or mappings")
        try:
            return target[index]
        except Exception as exc:
            raise ValueError(f"invalid subscript access: {exc}")

    if isinstance(node, ast.Tuple):
        return tuple(_eval_node(elt, names) for elt in node.elts)

    if isinstance(node, ast.List):
        return [_eval_node(elt, names) for elt in node.elts]

    if isinstance(node, ast.Dict):
        return {_eval_node(k, names): _eval_node(v, names) for k, v in zip(node.keys, node.values)}

    raise ValueError(f"unsupported expression node: {type(node).__name__}")


def safe_eval_expr(expr: str, names: Mapping[str, Any]) -> Any:
    """Evaluate an expression with a constrained AST allowlist.

    Only supports boolean operators, comparisons, indexing, numeric ops, and calling
    provided callables from the names mapping. Attribute access, comprehensions, and
    other dynamic constructs are rejected to prevent sandbox escapes.
    """

    try:
        parsed = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        raise ValueError("invalid expression") from exc

    for node in ast.walk(parsed):
        if isinstance(
            node,
            (
                ast.Attribute,
                ast.Lambda,
                ast.ListComp,
                ast.SetComp,
                ast.DictComp,
                ast.GeneratorExp,
                ast.Await,
                ast.Yield,
                ast.YieldFrom,
                ast.ClassDef,
                ast.FunctionDef,
            ),
        ):
            raise ValueError("disallowed syntax in expression")

    return _eval_node(parsed, names)
