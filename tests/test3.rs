#![cfg(test)]

//! Comprehensive test suite for Arena allocator and automatic reference counting
//!
//! This test suite thoroughly validates:
//! - Basic allocation and deallocation
//! - Reference counting increment/decrement on clone/drop
//! - Recursive deallocation for composite types (Cons, Lambda, Symbol)
//! - Memory reuse after values are freed
//! - Arena exhaustion handling
//! - Complex nested structures
//! - Environment operations
//! - Circular reference prevention
//! - Integration with the evaluator
//! - Edge cases (NULL refs, boundary conditions, overflow protection)

use ruthe::*;

// ============================================================================
// Set! Tests with Result-based Error Format
// ============================================================================

#[test]
fn test_set_basic_mutation() -> Result<(), ErrorCode> {
    let arena = Arena::<10000>::new();
    let env = env_new(&arena)?;

    eval_string(&arena, "(define x 10)", &env)?;
    eval_string(&arena, "(set! x 20)", &env)?;

    let result = eval_string(&arena, "x", &env)?;
    if let Some(Value::Number(n)) = arena.get(result.inner) {
        if n != 20 {
            return Err(ErrorCode::InvalidExpression);
        }
    } else {
        return Err(ErrorCode::TypeError);
    }

    Ok(())
}

#[test]
fn test_set_returns_new_value() -> Result<(), ErrorCode> {
    let arena = Arena::<10000>::new();
    let env = env_new(&arena)?;

    eval_string(&arena, "(define x 10)", &env)?;
    let result = eval_string(&arena, "(set! x 42)", &env)?;

    if let Some(Value::Number(n)) = arena.get(result.inner) {
        if n != 42 {
            return Err(ErrorCode::InvalidExpression);
        }
    } else {
        return Err(ErrorCode::TypeError);
    }

    Ok(())
}

#[test]
fn test_set_unbound_variable_fails() -> Result<(), ErrorCode> {
    let arena = Arena::<10000>::new();
    let env = env_new(&arena)?;

    let result = eval_string(&arena, "(set! nonexistent 123)", &env);
    if result.is_ok() {
        return Err(ErrorCode::InvalidExpression);
    }

    Ok(())
}

#[test]
fn test_set_with_expression() -> Result<(), ErrorCode> {
    let arena = Arena::<10000>::new();
    let env = env_new(&arena)?;

    eval_string(&arena, "(define x 10)", &env)?;
    eval_string(&arena, "(define y 5)", &env)?;
    eval_string(&arena, "(set! x (+ y 15))", &env)?;

    let result = eval_string(&arena, "x", &env)?;
    if let Some(Value::Number(n)) = arena.get(result.inner) {
        if n != 20 {
            return Err(ErrorCode::InvalidExpression);
        }
    } else {
        return Err(ErrorCode::TypeError);
    }

    Ok(())
}

#[test]
fn test_set_multiple_mutations() -> Result<(), ErrorCode> {
    let arena = Arena::<10000>::new();
    let env = env_new(&arena)?;

    eval_string(&arena, "(define counter 0)", &env)?;
    eval_string(&arena, "(set! counter (+ counter 1))", &env)?;

    let result = eval_string(&arena, "counter", &env)?;
    if let Some(Value::Number(n)) = arena.get(result.inner) {
        if n != 1 {
            return Err(ErrorCode::InvalidExpression);
        }
    }

    eval_string(&arena, "(set! counter (+ counter 1))", &env)?;
    let result = eval_string(&arena, "counter", &env)?;
    if let Some(Value::Number(n)) = arena.get(result.inner) {
        if n != 2 {
            return Err(ErrorCode::InvalidExpression);
        }
    }

    eval_string(&arena, "(set! counter (+ counter 1))", &env)?;
    let result = eval_string(&arena, "counter", &env)?;
    if let Some(Value::Number(n)) = arena.get(result.inner) {
        if n != 3 {
            return Err(ErrorCode::InvalidExpression);
        }
    } else {
        return Err(ErrorCode::TypeError);
    }

    Ok(())
}

#[test]
fn test_set_in_lambda_closure() -> Result<(), ErrorCode> {
    let arena = Arena::<10000>::new();
    let env = env_new(&arena)?;

    let code = r#"
        (define make-counter
          (lambda ()
            (define count 0)
            (lambda ()
              (set! count (+ count 1))
              count)))
    "#;
    eval_string(&arena, code, &env)?;
    eval_string(&arena, "(define c1 (make-counter))", &env)?;

    let result = eval_string(&arena, "(c1)", &env)?;
    if let Some(Value::Number(n)) = arena.get(result.inner) {
        if n != 1 {
            return Err(ErrorCode::InvalidExpression);
        }
    }

    let result = eval_string(&arena, "(c1)", &env)?;
    if let Some(Value::Number(n)) = arena.get(result.inner) {
        if n != 2 {
            return Err(ErrorCode::InvalidExpression);
        }
    }

    let result = eval_string(&arena, "(c1)", &env)?;
    if let Some(Value::Number(n)) = arena.get(result.inner) {
        if n != 3 {
            return Err(ErrorCode::InvalidExpression);
        }
    } else {
        return Err(ErrorCode::TypeError);
    }

    Ok(())
}

#[test]
fn test_set_independent_closures() -> Result<(), ErrorCode> {
    let arena = Arena::<10000>::new();
    let env = env_new(&arena)?;

    let code = r#"
        (define make-counter
          (lambda ()
            (define count 0)
            (lambda ()
              (set! count (+ count 1))
              count)))
    "#;
    eval_string(&arena, code, &env)?;
    eval_string(&arena, "(define c1 (make-counter))", &env)?;
    eval_string(&arena, "(define c2 (make-counter))", &env)?;

    let result = eval_string(&arena, "(c1)", &env)?;
    if let Some(Value::Number(n)) = arena.get(result.inner) {
        if n != 1 {
            return Err(ErrorCode::InvalidExpression);
        }
    }

    let result = eval_string(&arena, "(c1)", &env)?;
    if let Some(Value::Number(n)) = arena.get(result.inner) {
        if n != 2 {
            return Err(ErrorCode::InvalidExpression);
        }
    }

    let result = eval_string(&arena, "(c2)", &env)?;
    if let Some(Value::Number(n)) = arena.get(result.inner) {
        if n != 1 {
            return Err(ErrorCode::InvalidExpression);
        }
    }

    let result = eval_string(&arena, "(c1)", &env)?;
    if let Some(Value::Number(n)) = arena.get(result.inner) {
        if n != 3 {
            return Err(ErrorCode::InvalidExpression);
        }
    }

    let result = eval_string(&arena, "(c2)", &env)?;
    if let Some(Value::Number(n)) = arena.get(result.inner) {
        if n != 2 {
            return Err(ErrorCode::InvalidExpression);
        }
    } else {
        return Err(ErrorCode::TypeError);
    }

    Ok(())
}

#[test]
fn test_set_in_nested_scope() -> Result<(), ErrorCode> {
    let arena = Arena::<10000>::new();
    let env = env_new(&arena)?;

    eval_string(&arena, "(define x 10)", &env)?;

    let code = r#"
        (define mutate-x
          (lambda (val)
            (set! x val)))
    "#;
    eval_string(&arena, code, &env)?;
    eval_string(&arena, "(mutate-x 42)", &env)?;

    let result = eval_string(&arena, "x", &env)?;
    if let Some(Value::Number(n)) = arena.get(result.inner) {
        if n != 42 {
            return Err(ErrorCode::InvalidExpression);
        }
    } else {
        return Err(ErrorCode::TypeError);
    }

    Ok(())
}

#[test]
fn test_set_shadowed_variable() -> Result<(), ErrorCode> {
    let arena = Arena::<10000>::new();
    let env = env_new(&arena)?;

    eval_string(&arena, "(define x 10)", &env)?;

    let code = r#"
        (define test
          (lambda ()
            (define x 20)
            (set! x 30)
            x))
    "#;
    eval_string(&arena, code, &env)?;

    let result = eval_string(&arena, "(test)", &env)?;
    if let Some(Value::Number(n)) = arena.get(result.inner) {
        if n != 30 {
            return Err(ErrorCode::InvalidExpression);
        }
    }

    let result = eval_string(&arena, "x", &env)?;
    if let Some(Value::Number(n)) = arena.get(result.inner) {
        if n != 10 {
            return Err(ErrorCode::InvalidExpression);
        }
    } else {
        return Err(ErrorCode::TypeError);
    }

    Ok(())
}

#[test]
fn test_set_with_recursion() -> Result<(), ErrorCode> {
    let arena = Arena::<10000>::new();
    let env = env_new(&arena)?;

    let code = r#"
        (define factorial-iter
          (lambda (n)
            (define result 1)
            (define counter n)
            (define iter
              (lambda ()
                (if (= counter 0)
                    result
                    (begin
                      (set! result (* result counter))
                      (set! counter (- counter 1))
                      (iter)))))
            (iter)))
    "#;
    eval_string(&arena, code, &env)?;

    let result = eval_string(&arena, "(factorial-iter 5)", &env)?;
    if let Some(Value::Number(n)) = arena.get(result.inner) {
        if n != 120 {
            return Err(ErrorCode::InvalidExpression);
        }
    } else {
        return Err(ErrorCode::TypeError);
    }

    Ok(())
}

#[test]
fn test_set_boolean_value() -> Result<(), ErrorCode> {
    let arena = Arena::<10000>::new();
    let env = env_new(&arena)?;

    eval_string(&arena, "(define flag #t)", &env)?;
    let result = eval_string(&arena, "flag", &env)?;
    if let Some(Value::Bool(b)) = arena.get(result.inner) {
        if !b {
            return Err(ErrorCode::InvalidExpression);
        }
    }

    eval_string(&arena, "(set! flag #f)", &env)?;
    let result = eval_string(&arena, "flag", &env)?;
    if let Some(Value::Bool(b)) = arena.get(result.inner) {
        if b {
            return Err(ErrorCode::InvalidExpression);
        }
    } else {
        return Err(ErrorCode::TypeError);
    }

    Ok(())
}

#[test]
fn test_set_list_value() -> Result<(), ErrorCode> {
    let arena = Arena::<10000>::new();
    let env = env_new(&arena)?;

    eval_string(&arena, "(define mylist (list 1 2 3))", &env)?;
    eval_string(&arena, "(set! mylist (list 4 5 6))", &env)?;

    let result = eval_string(&arena, "(car mylist)", &env)?;
    if let Some(Value::Number(n)) = arena.get(result.inner) {
        if n != 4 {
            return Err(ErrorCode::InvalidExpression);
        }
    } else {
        return Err(ErrorCode::TypeError);
    }

    Ok(())
}

#[test]
fn test_set_wrong_argument_count() -> Result<(), ErrorCode> {
    let arena = Arena::<10000>::new();
    let env = env_new(&arena)?;

    eval_string(&arena, "(define x 10)", &env)?;

    // Too few arguments
    if eval_string(&arena, "(set! x)", &env).is_ok() {
        return Err(ErrorCode::InvalidExpression);
    }

    // Too many arguments
    if eval_string(&arena, "(set! x 20 30)", &env).is_ok() {
        return Err(ErrorCode::InvalidExpression);
    }

    Ok(())
}

#[test]
fn test_set_non_symbol_first_arg() -> Result<(), ErrorCode> {
    let arena = Arena::<10000>::new();
    let env = env_new(&arena)?;

    if eval_string(&arena, "(set! 123 456)", &env).is_ok() {
        return Err(ErrorCode::InvalidExpression);
    }

    if eval_string(&arena, "(set! (+ 1 2) 10)", &env).is_ok() {
        return Err(ErrorCode::InvalidExpression);
    }

    Ok(())
}

#[test]
fn test_set_preserves_other_bindings() -> Result<(), ErrorCode> {
    let arena = Arena::<10000>::new();
    let env = env_new(&arena)?;

    eval_string(&arena, "(define x 10)", &env)?;
    eval_string(&arena, "(define y 20)", &env)?;
    eval_string(&arena, "(define z 30)", &env)?;
    eval_string(&arena, "(set! y 99)", &env)?;

    let result = eval_string(&arena, "x", &env)?;
    if let Some(Value::Number(n)) = arena.get(result.inner) {
        if n != 10 {
            return Err(ErrorCode::InvalidExpression);
        }
    }

    let result = eval_string(&arena, "y", &env)?;
    if let Some(Value::Number(n)) = arena.get(result.inner) {
        if n != 99 {
            return Err(ErrorCode::InvalidExpression);
        }
    }

    let result = eval_string(&arena, "z", &env)?;
    if let Some(Value::Number(n)) = arena.get(result.inner) {
        if n != 30 {
            return Err(ErrorCode::InvalidExpression);
        }
    } else {
        return Err(ErrorCode::TypeError);
    }

    Ok(())
}

#[test]
fn test_set_bank_account_example() -> Result<(), ErrorCode> {
    let arena = Arena::<10000>::new();
    let env = env_new(&arena)?;

    let code = r#"
        (define make-account
          (lambda (balance)
            (lambda (amount)
              (set! balance (+ balance amount))
              balance)))
    "#;
    eval_string(&arena, code, &env)?;
    eval_string(&arena, "(define acc (make-account 100))", &env)?;

    let result = eval_string(&arena, "(acc 50)", &env)?;
    if let Some(Value::Number(n)) = arena.get(result.inner) {
        if n != 150 {
            return Err(ErrorCode::InvalidExpression);
        }
    }

    let result = eval_string(&arena, "(acc -25)", &env)?;
    if let Some(Value::Number(n)) = arena.get(result.inner) {
        if n != 125 {
            return Err(ErrorCode::InvalidExpression);
        }
    }

    let result = eval_string(&arena, "(acc 10)", &env)?;
    if let Some(Value::Number(n)) = arena.get(result.inner) {
        if n != 135 {
            return Err(ErrorCode::InvalidExpression);
        }
    } else {
        return Err(ErrorCode::TypeError);
    }

    Ok(())
}

#[test]
fn test_set_parent_scope() -> Result<(), ErrorCode> {
    let arena = Arena::<DEFAULT_ARENA_SIZE>::new();
    let env = env_new(&arena)?;

    eval_string(&arena, "(define x 100)", &env)?;
    eval_string(
        &arena,
        "(define mutate-x (lambda (new-val) (set! x new-val)))",
        &env,
    )?;
    eval_string(&arena, "(mutate-x 200)", &env)?;

    let result = eval_string(&arena, "x", &env)?;
    if let Some(Value::Number(n)) = arena.get(result.inner) {
        if n != 200 {
            return Err(ErrorCode::InvalidExpression);
        }
    } else {
        return Err(ErrorCode::TypeError);
    }

    Ok(())
}

#[test]
fn test_set_vs_define_shadowing() -> Result<(), ErrorCode> {
    let arena = Arena::<DEFAULT_ARENA_SIZE>::new();
    let env = env_new(&arena)?;

    eval_string(&arena, "(define x 10)", &env)?;

    let code = r#"
        (define test
          (lambda ()
            (define x 20)
            (set! x 30)
            x))
    "#;
    eval_string(&arena, code, &env)?;

    let result = eval_string(&arena, "(test)", &env)?;
    if let Some(Value::Number(n)) = arena.get(result.inner) {
        if n != 30 {
            return Err(ErrorCode::InvalidExpression);
        }
    }

    let result = eval_string(&arena, "x", &env)?;
    if let Some(Value::Number(n)) = arena.get(result.inner) {
        if n != 10 {
            return Err(ErrorCode::InvalidExpression);
        }
    } else {
        return Err(ErrorCode::TypeError);
    }

    Ok(())
}

#[test]
fn test_set_finds_first_binding_in_chain() -> Result<(), ErrorCode> {
    let arena = Arena::<DEFAULT_ARENA_SIZE>::new();
    let env = env_new(&arena)?;

    eval_string(&arena, "(define x 1)", &env)?;
    eval_string(
        &arena,
        "(define outer (lambda () (define x 2) (define inner (lambda () (set! x 99))) (inner) x))",
        &env,
    )?;

    let result = eval_string(&arena, "(outer)", &env)?;
    if let Some(Value::Number(n)) = arena.get(result.inner) {
        if n != 99 {
            return Err(ErrorCode::InvalidExpression);
        }
    }

    let result = eval_string(&arena, "x", &env)?;
    if let Some(Value::Number(n)) = arena.get(result.inner) {
        if n != 1 {
            return Err(ErrorCode::InvalidExpression);
        }
    } else {
        return Err(ErrorCode::TypeError);
    }

    Ok(())
}

#[test]
fn test_set_with_boolean() -> Result<(), ErrorCode> {
    let arena = Arena::<DEFAULT_ARENA_SIZE>::new();
    let env = env_new(&arena)?;

    eval_string(&arena, "(define flag #t)", &env)?;
    eval_string(&arena, "(set! flag #f)", &env)?;

    let result = eval_string(&arena, "flag", &env)?;
    if let Some(Value::Bool(b)) = arena.get(result.inner) {
        if b {
            return Err(ErrorCode::InvalidExpression);
        }
    } else {
        return Err(ErrorCode::TypeError);
    }

    Ok(())
}

#[test]
fn test_set_with_list() -> Result<(), ErrorCode> {
    let arena = Arena::<DEFAULT_ARENA_SIZE>::new();
    let env = env_new(&arena)?;

    eval_string(&arena, "(define lst (list 1 2 3))", &env)?;
    eval_string(&arena, "(set! lst (list 4 5 6))", &env)?;

    let result = eval_string(&arena, "(car lst)", &env)?;
    if let Some(Value::Number(n)) = arena.get(result.inner) {
        if n != 4 {
            return Err(ErrorCode::InvalidExpression);
        }
    } else {
        return Err(ErrorCode::TypeError);
    }

    Ok(())
}

#[test]
fn test_set_accumulator_pattern() -> Result<(), ErrorCode> {
    let arena = Arena::<DEFAULT_ARENA_SIZE>::new();
    let env = env_new(&arena)?;

    eval_string(&arena, "(define sum 0)", &env)?;
    eval_string(
        &arena,
        "(define add-to-sum (lambda (n) (set! sum (+ sum n)) sum))",
        &env,
    )?;

    let r1 = eval_string(&arena, "(add-to-sum 5)", &env)?;
    if let Some(Value::Number(n)) = arena.get(r1.inner) {
        if n != 5 {
            return Err(ErrorCode::InvalidExpression);
        }
    }

    let r2 = eval_string(&arena, "(add-to-sum 10)", &env)?;
    if let Some(Value::Number(n)) = arena.get(r2.inner) {
        if n != 15 {
            return Err(ErrorCode::InvalidExpression);
        }
    }

    let r3 = eval_string(&arena, "(add-to-sum 7)", &env)?;
    if let Some(Value::Number(n)) = arena.get(r3.inner) {
        if n != 22 {
            return Err(ErrorCode::InvalidExpression);
        }
    } else {
        return Err(ErrorCode::TypeError);
    }

    Ok(())
}

#[test]
fn test_set_factorial_with_mutation() -> Result<(), ErrorCode> {
    let arena = Arena::<DEFAULT_ARENA_SIZE>::new();
    let env = env_new(&arena)?;

    eval_string(&arena, "(define result 1)", &env)?;
    eval_string(&arena, "(define n 5)", &env)?;

    eval_string(&arena, "(set! result (* result n))", &env)?;
    eval_string(&arena, "(set! n (- n 1))", &env)?;
    eval_string(&arena, "(set! result (* result n))", &env)?;
    eval_string(&arena, "(set! n (- n 1))", &env)?;
    eval_string(&arena, "(set! result (* result n))", &env)?;
    eval_string(&arena, "(set! n (- n 1))", &env)?;
    eval_string(&arena, "(set! result (* result n))", &env)?;

    let result = eval_string(&arena, "result", &env)?;
    if let Some(Value::Number(n)) = arena.get(result.inner) {
        if n != 120 {
            return Err(ErrorCode::InvalidExpression);
        }
    } else {
        return Err(ErrorCode::TypeError);
    }

    Ok(())
}

#[test]
fn test_set_swap_pattern() -> Result<(), ErrorCode> {
    let arena = Arena::<DEFAULT_ARENA_SIZE>::new();
    let env = env_new(&arena)?;

    eval_string(&arena, "(define a 10)", &env)?;
    eval_string(&arena, "(define b 20)", &env)?;
    eval_string(&arena, "(define temp a)", &env)?;
    eval_string(&arena, "(set! a b)", &env)?;
    eval_string(&arena, "(set! b temp)", &env)?;

    let result_a = eval_string(&arena, "a", &env)?;
    if let Some(Value::Number(n)) = arena.get(result_a.inner) {
        if n != 20 {
            return Err(ErrorCode::InvalidExpression);
        }
    }

    let result_b = eval_string(&arena, "b", &env)?;
    if let Some(Value::Number(n)) = arena.get(result_b.inner) {
        if n != 10 {
            return Err(ErrorCode::InvalidExpression);
        }
    } else {
        return Err(ErrorCode::TypeError);
    }

    Ok(())
}

#[test]
fn test_set_in_recursive_function() -> Result<(), ErrorCode> {
    let arena = Arena::<DEFAULT_ARENA_SIZE>::new();
    let env = env_new(&arena)?;

    eval_string(&arena, "(define call-count 0)", &env)?;
    eval_string(
        &arena,
        "(define counting-factorial (lambda (n) (set! call-count (+ call-count 1)) (if (= n 0) 1 (* n (counting-factorial (- n 1))))))",
        &env,
    )?;

    eval_string(&arena, "(counting-factorial 5)", &env)?;

    let result = eval_string(&arena, "call-count", &env)?;
    if let Some(Value::Number(n)) = arena.get(result.inner) {
        if n != 6 {
            return Err(ErrorCode::InvalidExpression);
        }
    } else {
        return Err(ErrorCode::TypeError);
    }

    Ok(())
}

#[test]
fn test_set_wrong_arg_count() -> Result<(), ErrorCode> {
    let arena = Arena::<DEFAULT_ARENA_SIZE>::new();
    let env = env_new(&arena)?;

    eval_string(&arena, "(define x 1)", &env)?;

    if eval_string(&arena, "(set! x)", &env).is_ok() {
        return Err(ErrorCode::InvalidExpression);
    }

    if eval_string(&arena, "(set! x 1 2)", &env).is_ok() {
        return Err(ErrorCode::InvalidExpression);
    }

    Ok(())
}

#[test]
fn test_set_requires_symbol() -> Result<(), ErrorCode> {
    let arena = Arena::<DEFAULT_ARENA_SIZE>::new();
    let env = env_new(&arena)?;

    if eval_string(&arena, "(set! 123 456)", &env).is_ok() {
        return Err(ErrorCode::InvalidExpression);
    }

    if eval_string(&arena, "(set! (+ 1 2) 10)", &env).is_ok() {
        return Err(ErrorCode::InvalidExpression);
    }

    Ok(())
}

#[test]
fn test_set_with_lambda_value() -> Result<(), ErrorCode> {
    let arena = Arena::<DEFAULT_ARENA_SIZE>::new();
    let env = env_new(&arena)?;

    eval_string(&arena, "(define func (lambda (x) x))", &env)?;
    eval_string(&arena, "(set! func (lambda (x) (* x 2)))", &env)?;

    let result = eval_string(&arena, "(func 5)", &env)?;
    if let Some(Value::Number(n)) = arena.get(result.inner) {
        if n != 10 {
            return Err(ErrorCode::InvalidExpression);
        }
    } else {
        return Err(ErrorCode::TypeError);
    }

    Ok(())
}

#[test]
fn test_set_multiple_variables_in_sequence() -> Result<(), ErrorCode> {
    let arena = Arena::<DEFAULT_ARENA_SIZE>::new();
    let env = env_new(&arena)?;

    eval_string(&arena, "(define a 1)", &env)?;
    eval_string(&arena, "(define b 2)", &env)?;
    eval_string(&arena, "(define c 3)", &env)?;

    eval_string(&arena, "(set! a 10)", &env)?;
    eval_string(&arena, "(set! b 20)", &env)?;
    eval_string(&arena, "(set! c 30)", &env)?;

    let result = eval_string(&arena, "(+ a b c)", &env)?;
    if let Some(Value::Number(n)) = arena.get(result.inner) {
        if n != 60 {
            return Err(ErrorCode::InvalidExpression);
        }
    } else {
        return Err(ErrorCode::TypeError);
    }

    Ok(())
}

#[test]
fn test_set_state_machine() -> Result<(), ErrorCode> {
    let arena = Arena::<DEFAULT_ARENA_SIZE>::new();
    let env = env_new(&arena)?;

    eval_string(&arena, "(define state 0)", &env)?;
    eval_string(
        &arena,
        "(define toggle (lambda () (if (= state 0) (set! state 1) (set! state 0)) state))",
        &env,
    )?;

    let r1 = eval_string(&arena, "(toggle)", &env)?;
    if let Some(Value::Number(n)) = arena.get(r1.inner) {
        if n != 1 {
            return Err(ErrorCode::InvalidExpression);
        }
    }

    let r2 = eval_string(&arena, "(toggle)", &env)?;
    if let Some(Value::Number(n)) = arena.get(r2.inner) {
        if n != 0 {
            return Err(ErrorCode::InvalidExpression);
        }
    }

    let r3 = eval_string(&arena, "(toggle)", &env)?;
    if let Some(Value::Number(n)) = arena.get(r3.inner) {
        if n != 1 {
            return Err(ErrorCode::InvalidExpression);
        }
    } else {
        return Err(ErrorCode::TypeError);
    }

    Ok(())
}

// ============================================================================
// Note: The old test_errors_exhaust_arena has been removed because it's no
// longer relevant. With the new ErrorCode enum system:
// - Error VALUES don't allocate in the arena (they're just enum variants)
// - However, evaluating expressions still allocates temporary values
//   (tokens, parsed structures) regardless of whether they succeed or error
// - So the arena can still be exhausted, but not specifically by "errors"
// ============================================================================

// ============================================================================
// Basic Allocation and Deallocation Tests (These use assertions, not Result)
// ============================================================================

#[test]
fn test_basic_allocation() {
    let arena = Arena::<100>::new();
    let val = arena.number(42).unwrap();

    assert!(!val.is_null());
    assert_eq!(arena.refcounts[val.raw().0 as usize].get(), 1);

    if let Some(Value::Number(n)) = val.get() {
        assert_eq!(n, 42);
    } else {
        panic!("Expected Number value");
    }
}

#[test]
fn test_allocation_and_drop() {
    let arena = Arena::<100>::new();

    {
        let val = arena.number(42).unwrap();
        assert_eq!(arena.refcounts[val.raw().0 as usize].get(), 1);
    }

    assert!(matches!(arena.values[0].get(), Value::Free));
    assert_eq!(arena.refcounts[0].get(), 0);
}

#[test]
fn test_memory_reuse_after_drop() {
    let arena = Arena::<100>::new();

    let idx = {
        let val = arena.number(42).unwrap();
        val.raw().0 as usize
    };

    assert!(matches!(arena.values[idx].get(), Value::Free));
    assert_eq!(arena.refcounts[idx].get(), 0);

    let val2 = arena.number(99).unwrap();
    assert!(!val2.is_null());

    if let Some(Value::Number(n)) = val2.get() {
        assert_eq!(n, 99);
    }
}

#[test]
fn test_arena_exhaustion() {
    let arena = Arena::<10>::new();
    let mut refs = Vec::new();

    for i in 0..10 {
        let val = arena.number(i as i64).unwrap();
        assert!(!val.is_null());
        refs.push(val);
    }

    let val = arena.number(999);
    assert!(val.is_err());
}

#[test]
fn test_arena_exhaustion_with_reuse() {
    let arena = Arena::<10>::new();

    let mut refs = Vec::new();
    for i in 0..10 {
        refs.push(arena.number(i as i64).unwrap());
    }

    let val = arena.number(999);
    assert!(val.is_err());

    refs.truncate(5);

    let val = arena.number(999).unwrap();
    assert!(!val.is_null());
}

// ============================================================================
// Reference Counting Tests
// ============================================================================

#[test]
fn test_refcount_increment_on_clone() {
    let arena = Arena::<100>::new();
    let val = arena.number(42).unwrap();
    let idx = val.raw().0 as usize;

    assert_eq!(arena.refcounts[idx].get(), 1);

    let val2 = val.clone();
    assert_eq!(arena.refcounts[idx].get(), 2);

    let val3 = val.clone();
    assert_eq!(arena.refcounts[idx].get(), 3);

    drop(val2);
    assert_eq!(arena.refcounts[idx].get(), 2);

    drop(val3);
    assert_eq!(arena.refcounts[idx].get(), 1);

    drop(val);
    assert_eq!(arena.refcounts[idx].get(), 0);
    assert!(matches!(arena.values[idx].get(), Value::Free));
}

#[test]
fn test_refcount_with_multiple_scopes() {
    let arena = Arena::<100>::new();
    let val = arena.number(42).unwrap();
    let idx = val.raw().0 as usize;

    {
        let val2 = val.clone();
        assert_eq!(arena.refcounts[idx].get(), 2);
        {
            let val3 = val2.clone();
            assert_eq!(arena.refcounts[idx].get(), 3);
        }
        assert_eq!(arena.refcounts[idx].get(), 2);
    }
    assert_eq!(arena.refcounts[idx].get(), 1);
}

#[test]
fn test_refcount_overflow_protection() {
    let arena = Arena::<100>::new();
    let val = arena.number(42).unwrap();
    let idx = val.raw().0 as usize;

    arena.refcounts[idx].set(u32::MAX - 1);

    arena.incref(val.raw());
    assert_eq!(arena.refcounts[idx].get(), u32::MAX);

    arena.incref(val.raw());
    assert_eq!(arena.refcounts[idx].get(), u32::MAX);
}

// ============================================================================
// Cons Cell Tests (Recursive Decref)
// ============================================================================

#[test]
fn test_cons_refcounting() {
    let arena = Arena::<100>::new();

    let car = arena.number(1).unwrap();
    let cdr = arena.number(2).unwrap();

    let car_idx = car.raw().0 as usize;
    let cdr_idx = cdr.raw().0 as usize;

    assert_eq!(arena.refcounts[car_idx].get(), 1);
    assert_eq!(arena.refcounts[cdr_idx].get(), 1);

    let cons = arena.cons(&car, &cdr).unwrap();

    assert_eq!(arena.refcounts[car_idx].get(), 2);
    assert_eq!(arena.refcounts[cdr_idx].get(), 2);

    drop(car);
    drop(cdr);

    assert_eq!(arena.refcounts[car_idx].get(), 1);
    assert_eq!(arena.refcounts[cdr_idx].get(), 1);

    drop(cons);

    assert_eq!(arena.refcounts[car_idx].get(), 0);
    assert_eq!(arena.refcounts[cdr_idx].get(), 0);
    assert!(matches!(arena.values[car_idx].get(), Value::Free));
    assert!(matches!(arena.values[cdr_idx].get(), Value::Free));
}

#[test]
fn test_nested_cons_refcounting() {
    let arena = Arena::<100>::new();

    let a = arena.number(1).unwrap();
    let b = arena.number(2).unwrap();
    let c = arena.number(3).unwrap();

    let a_idx = a.raw().0 as usize;
    let b_idx = b.raw().0 as usize;
    let c_idx = c.raw().0 as usize;

    let inner = arena.cons(&b, &c).unwrap();
    let outer = arena.cons(&a, &inner).unwrap();

    drop(a);
    drop(b);
    drop(c);
    drop(inner);

    assert_eq!(arena.refcounts[a_idx].get(), 1);
    assert_eq!(arena.refcounts[b_idx].get(), 1);
    assert_eq!(arena.refcounts[c_idx].get(), 1);

    drop(outer);

    assert_eq!(arena.refcounts[a_idx].get(), 0);
    assert_eq!(arena.refcounts[b_idx].get(), 0);
    assert_eq!(arena.refcounts[c_idx].get(), 0);
}

#[test]
fn test_list_refcounting() {
    let arena = Arena::<100>::new();

    let mut nums = Vec::new();
    let mut indices = Vec::new();
    let mut list = arena.nil().unwrap();

    for i in 0..5 {
        let num = arena.number(i).unwrap();
        indices.push(num.raw().0 as usize);
        list = arena.cons(&num, &list).unwrap();
        nums.push(num);
    }

    for idx in &indices {
        assert_eq!(arena.refcounts[*idx].get(), 2);
    }

    drop(list);

    for idx in &indices {
        assert_eq!(arena.refcounts[*idx].get(), 1);
    }

    drop(nums);

    for idx in &indices {
        assert_eq!(arena.refcounts[*idx].get(), 0);
    }
}

// ============================================================================
// Lambda and Environment Tests
// ============================================================================

#[test]
fn test_lambda_refcounting() {
    let arena = Arena::<100>::new();

    let params = arena.nil().unwrap();
    let body = arena.number(42).unwrap();
    let env = arena.nil().unwrap();

    let params_idx = params.raw().0 as usize;
    let body_idx = body.raw().0 as usize;
    let env_idx = env.raw().0 as usize;

    let lambda = arena.lambda(&params, &body, &env).unwrap();

    assert_eq!(arena.refcounts[params_idx].get(), 2);
    assert_eq!(arena.refcounts[body_idx].get(), 2);
    assert_eq!(arena.refcounts[env_idx].get(), 2);

    drop(params);
    drop(body);
    drop(env);

    assert_eq!(arena.refcounts[params_idx].get(), 1);
    assert_eq!(arena.refcounts[body_idx].get(), 1);
    assert_eq!(arena.refcounts[env_idx].get(), 1);

    drop(lambda);

    assert_eq!(arena.refcounts[params_idx].get(), 0);
    assert_eq!(arena.refcounts[body_idx].get(), 0);
    assert_eq!(arena.refcounts[env_idx].get(), 0);
}

#[test]
fn test_symbol_refcounting() {
    let arena = Arena::<100>::new();

    let string = arena.str_to_list("test").unwrap();
    let string_idx = string.raw().0 as usize;

    let sym = arena.symbol(&string).unwrap();

    assert_eq!(arena.refcounts[string_idx].get(), 2);

    drop(string);
    assert_eq!(arena.refcounts[string_idx].get(), 1);

    drop(sym);

    assert_eq!(arena.refcounts[string_idx].get(), 0);
}

// ============================================================================
// Set Operations Tests
// ============================================================================

#[test]
fn test_set_cons_refcounting() {
    let arena = Arena::<100>::new();

    let car1 = arena.number(1).unwrap();
    let cdr1 = arena.number(2).unwrap();
    let car1_idx = car1.raw().0 as usize;
    let cdr1_idx = cdr1.raw().0 as usize;

    let cons = arena.cons(&car1, &cdr1).unwrap();

    assert_eq!(arena.refcounts[car1_idx].get(), 2);
    assert_eq!(arena.refcounts[cdr1_idx].get(), 2);

    let car2 = arena.number(3).unwrap();
    let cdr2 = arena.number(4).unwrap();
    let car2_idx = car2.raw().0 as usize;
    let cdr2_idx = cdr2.raw().0 as usize;

    arena.set_cons(&cons, &car2, &cdr2);

    assert_eq!(arena.refcounts[car1_idx].get(), 1);
    assert_eq!(arena.refcounts[cdr1_idx].get(), 1);

    assert_eq!(arena.refcounts[car2_idx].get(), 2);
    assert_eq!(arena.refcounts[cdr2_idx].get(), 2);

    drop(car1);
    drop(cdr1);

    assert_eq!(arena.refcounts[car1_idx].get(), 0);
    assert_eq!(arena.refcounts[cdr1_idx].get(), 0);

    drop(cons);
    drop(car2);
    drop(cdr2);

    assert_eq!(arena.refcounts[car2_idx].get(), 0);
    assert_eq!(arena.refcounts[cdr2_idx].get(), 0);
}

// ============================================================================
// Complex Structure Tests
// ============================================================================

#[test]
fn test_deep_list_refcounting() {
    let arena = Arena::<1000>::new();

    let mut list = arena.nil().unwrap();
    let mut indices = Vec::new();

    for i in 0..100 {
        let num = arena.number(i).unwrap();
        indices.push(num.raw().0 as usize);
        list = arena.cons(&num, &list).unwrap();
    }

    for idx in &indices {
        assert!(arena.refcounts[*idx].get() >= 1);
    }

    drop(list);
}

#[test]
fn test_shared_structure_refcounting() {
    let arena = Arena::<100>::new();

    let shared = arena.number(42).unwrap();
    let shared_idx = shared.raw().0 as usize;

    let cons1 = arena.cons(&shared, &arena.nil().unwrap()).unwrap();
    let cons2 = arena.cons(&shared, &arena.nil().unwrap()).unwrap();
    let cons3 = arena.cons(&shared, &arena.nil().unwrap()).unwrap();

    assert_eq!(arena.refcounts[shared_idx].get(), 4);

    drop(cons1);
    assert_eq!(arena.refcounts[shared_idx].get(), 3);

    drop(cons2);
    assert_eq!(arena.refcounts[shared_idx].get(), 2);

    drop(cons3);
    assert_eq!(arena.refcounts[shared_idx].get(), 1);

    drop(shared);
    assert_eq!(arena.refcounts[shared_idx].get(), 0);
}

// ============================================================================
// Environment Structure Tests
// ============================================================================

#[test]
fn test_environment_refcounting() -> Result<(), ErrorCode> {
    let arena = Arena::<500>::new();
    let env = env_new(&arena)?;

    let name = arena.str_to_list("x")?;
    let value = arena.number(42)?;
    let value_idx = value.raw().0 as usize;

    env_set(&arena, &env, &name, &value)?;

    assert_eq!(arena.refcounts[value_idx].get(), 2);

    drop(value);
    assert_eq!(arena.refcounts[value_idx].get(), 1);

    if let Some(retrieved) = env_get(&arena, &env, &name) {
        assert_eq!(arena.refcounts[value_idx].get(), 2);
        drop(retrieved);
    }

    assert_eq!(arena.refcounts[value_idx].get(), 1);

    Ok(())
}

#[test]
fn test_environment_update_refcounting() -> Result<(), ErrorCode> {
    let arena = Arena::<500>::new();
    let env = env_new(&arena)?;
    let name = arena.str_to_list("x")?;

    let value1 = arena.number(42)?;
    let value1_idx = value1.raw().0 as usize;
    env_set(&arena, &env, &name, &value1)?;
    assert_eq!(
        arena.refcounts[value1_idx].get(),
        2,
        "value1 should have refcount 2 after first set"
    );

    let value2 = arena.number(99)?;
    let value2_idx = value2.raw().0 as usize;
    env_set(&arena, &env, &name, &value2)?;

    assert_eq!(
        arena.refcounts[value1_idx].get(),
        1,
        "value1 should have refcount 1 after being replaced"
    );
    assert_eq!(
        arena.refcounts[value2_idx].get(),
        2,
        "value2 should have refcount 2 after being set"
    );

    Ok(())
}

// ============================================================================
// Integration Tests with Eval
// ============================================================================

#[test]
fn test_eval_refcounting() -> Result<(), ErrorCode> {
    let arena = Arena::<1000>::new();
    let env = env_new(&arena)?;

    let result = eval_string(&arena, "(+ 1 2 3)", &env)?;
    if let Some(Value::Number(n)) = arena.get(result.inner) {
        if n != 6 {
            return Err(ErrorCode::InvalidExpression);
        }
    } else {
        return Err(ErrorCode::TypeError);
    }

    for _ in 0..10 {
        let _ = eval_string(&arena, "(* 2 3)", &env)?;
    }

    let val = arena.number(999)?;
    if val.is_null() {
        return Err(ErrorCode::ArenaExhausted);
    }

    Ok(())
}

#[test]
fn test_recursive_eval_refcounting() -> Result<(), ErrorCode> {
    let arena = Arena::<5000>::new();
    let env = env_new(&arena)?;

    let factorial_def = r#"
        (define factorial
          (lambda (n)
            (if (= n 0)
                1
                (* n (factorial (- n 1))))))
    "#;

    let _ = eval_string(&arena, factorial_def, &env)?;
    let result = eval_string(&arena, "(factorial 10)", &env)?;

    if let Some(Value::Number(n)) = arena.get(result.inner) {
        if n != 3628800 {
            return Err(ErrorCode::InvalidExpression);
        }
    } else {
        return Err(ErrorCode::TypeError);
    }

    Ok(())
}

#[test]
fn test_tail_call_refcounting() -> Result<(), ErrorCode> {
    let arena = Arena::<5000>::new();
    let env = env_new(&arena)?;

    let sum_def = r#"
        (define sum
          (lambda (n acc)
            (if (= n 0)
                acc
                (sum (- n 1) (+ acc n)))))
    "#;

    let _ = eval_string(&arena, sum_def, &env)?;
    let result = eval_string(&arena, "(sum 1000 0)", &env)?;

    if let Some(Value::Number(n)) = arena.get(result.inner) {
        if n != 500500 {
            return Err(ErrorCode::InvalidExpression);
        }
    } else {
        return Err(ErrorCode::TypeError);
    }

    Ok(())
}
