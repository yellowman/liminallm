from liminallm.service.sandbox import safe_eval_expr


def test_safe_eval_expr_allows_basic_operations():
    names = {"x": 2, "y": 3, "add": lambda a, b: a + b}
    expr = "(x + y) == add(5, 0) and not (y < x)"

    result = safe_eval_expr(expr, names)

    assert result is True


def test_safe_eval_expr_blocks_disallowed_syntax():
    names = {"x": 1}

    for expr in ["(lambda z: z)(1)", "__import__('os').system('echo hi')"]:
        try:
            safe_eval_expr(expr, names)
        except ValueError:
            continue
        raise AssertionError("unsafe expression was not rejected")
