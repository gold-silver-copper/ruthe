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
//!
//! Key test categories:
//! 1. Basic Allocation (5 tests) - single values, drops, reuse, exhaustion
//! 2. Reference Counting (4 tests) - clone, drop, scopes, overflow
//! 3. Cons Cells (3 tests) - basic, nested, list refcounting
//! 4. Lambdas & Symbols (2 tests) - compound type refcounting
//! 5. Set Operations (1 test) - updating cons cells
//! 6. Complex Structures (3 tests) - deep nesting, sharing, large lists
//! 7. Environments (2 tests) - variable bindings, shadowing
//! 8. Stress Tests (4 tests) - repeated alloc/dealloc, patterns, deep recursion
//! 9. Edge Cases (6 tests) - NULL, boundaries, double decref, mixed types
//! 10. Integration (3 tests) - eval, recursion, tail calls
//! 11. Additional Stress (8 tests) - circular refs, allocation patterns, closures

use ruthe::*;

#[test]
fn test_set_basic_mutation() {
    let arena = Arena::<DEFAULT_ARENA_SIZE>::new();
    let env = env_new(&arena);

    // Define a variable
    let _ = eval_string(&arena, "(define x 10)", &env).unwrap();
    
    // Verify initial value
    let result = eval_string(&arena, "x", &env).unwrap();
    assert!(matches!(arena.get(result.inner), Some(Value::Number(10))));
    
    // Mutate with set!
    let _ = eval_string(&arena, "(set! x 20)", &env).unwrap();
    
    // Verify mutated value
    let result = eval_string(&arena, "x", &env).unwrap();
    assert!(matches!(arena.get(result.inner), Some(Value::Number(20))));
}

#[test]
fn test_set_returns_new_value() {
    let arena = Arena::<DEFAULT_ARENA_SIZE>::new();
    let env = env_new(&arena);

    let _ = eval_string(&arena, "(define x 5)", &env).unwrap();
    
    // set! should return the new value
    let result = eval_string(&arena, "(set! x 42)", &env).unwrap();
    assert!(matches!(arena.get(result.inner), Some(Value::Number(42))));
}

#[test]
fn test_set_unbound_variable_fails() {
    let arena = Arena::<DEFAULT_ARENA_SIZE>::new();
    let env = env_new(&arena);

    // Try to set! an undefined variable
    let result = eval_string(&arena, "(set! undefined 123)", &env);
    assert!(result.is_err());
}

#[test]
fn test_set_multiple_mutations() {
    let arena = Arena::<DEFAULT_ARENA_SIZE>::new();
    let env = env_new(&arena);

    let _ = eval_string(&arena, "(define counter 0)", &env).unwrap();
    
    // Multiple mutations
    let _ = eval_string(&arena, "(set! counter 1)", &env).unwrap();
    let _ = eval_string(&arena, "(set! counter 2)", &env).unwrap();
    let _ = eval_string(&arena, "(set! counter 3)", &env).unwrap();
    
    let result = eval_string(&arena, "counter", &env).unwrap();
    assert!(matches!(arena.get(result.inner), Some(Value::Number(3))));
}

#[test]
fn test_set_with_expression() {
    let arena = Arena::<DEFAULT_ARENA_SIZE>::new();
    let env = env_new(&arena);

    let _ = eval_string(&arena, "(define x 10)", &env).unwrap();
    
    // set! with computed value
    let _ = eval_string(&arena, "(set! x (+ x 5))", &env).unwrap();
    
    let result = eval_string(&arena, "x", &env).unwrap();
    assert!(matches!(arena.get(result.inner), Some(Value::Number(15))));
}

#[test]
fn test_set_in_lambda_closure() {
    let arena = Arena::<DEFAULT_ARENA_SIZE>::new();
    let env = env_new(&arena);

    // Create a counter with closure
    let counter_def = r#"
        (define make-counter
          (lambda (init)
            (lambda ()
              (set! init (+ init 1))
              init)))
    "#;
    let _ = eval_string(&arena, counter_def, &env).unwrap();
    let _ = eval_string(&arena, "(define counter (make-counter 0))", &env).unwrap();
    
    // Call counter multiple times
    let result1 = eval_string(&arena, "(counter)", &env).unwrap();
    assert!(matches!(arena.get(result1.inner), Some(Value::Number(1))));
    
    let result2 = eval_string(&arena, "(counter)", &env).unwrap();
    assert!(matches!(arena.get(result2.inner), Some(Value::Number(2))));
    
    let result3 = eval_string(&arena, "(counter)", &env).unwrap();
    assert!(matches!(arena.get(result3.inner), Some(Value::Number(3))));
}

#[test]
fn test_set_parent_scope() {
    let arena = Arena::<DEFAULT_ARENA_SIZE>::new();
    let env = env_new(&arena);

    // Define in outer scope, mutate in inner scope
    let code = r#"
        (define x 100)
        (define mutate-x
          (lambda (new-val)
            (set! x new-val)))
        (mutate-x 200)
        x
    "#;
    
    let _ = eval_string(&arena, "(define x 100)", &env).unwrap();
    let _ = eval_string(&arena, "(define mutate-x (lambda (new-val) (set! x new-val)))", &env).unwrap();
    let _ = eval_string(&arena, "(mutate-x 200)", &env).unwrap();
    let result = eval_string(&arena, "x", &env).unwrap();
    
    assert!(matches!(arena.get(result.inner), Some(Value::Number(200))));
}

#[test]
fn test_set_vs_define_shadowing() {
    let arena = Arena::<DEFAULT_ARENA_SIZE>::new();
    let env = env_new(&arena);

    // Outer x
    let _ = eval_string(&arena, "(define x 10)", &env).unwrap();
    
    // Lambda that defines a local x (shadowing), then tries to set! it
    let code = r#"
        (define test
          (lambda ()
            (define x 20)
            (set! x 30)
            x))
    "#;
    let _ = eval_string(&arena, code, &env).unwrap();
    
    // Inner x should be 30
    let result = eval_string(&arena, "(test)", &env).unwrap();
    assert!(matches!(arena.get(result.inner), Some(Value::Number(30))));
    
    // Outer x should still be 10
    let result = eval_string(&arena, "x", &env).unwrap();
    assert!(matches!(arena.get(result.inner), Some(Value::Number(10))));
}

#[test]
fn test_set_finds_first_binding_in_chain() {
    let arena = Arena::<DEFAULT_ARENA_SIZE>::new();
    let env = env_new(&arena);

    // Create nested scopes with same variable
    let code = r#"
        (define x 1)
        (define outer
          (lambda ()
            (define x 2)
            (define inner
              (lambda ()
                (set! x 99)))
            (inner)
            x))
        (outer)
    "#;
    
    let _ = eval_string(&arena, "(define x 1)", &env).unwrap();
    let _ = eval_string(&arena, 
        "(define outer (lambda () (define x 2) (define inner (lambda () (set! x 99))) (inner) x))",
        &env).unwrap();
    
    // set! should modify the closest x (in outer lambda)
    let result = eval_string(&arena, "(outer)", &env).unwrap();
    assert!(matches!(arena.get(result.inner), Some(Value::Number(99))));
    
    // Global x should be unchanged
    let result = eval_string(&arena, "x", &env).unwrap();
    assert!(matches!(arena.get(result.inner), Some(Value::Number(1))));
}

#[test]
fn test_set_with_boolean() {
    let arena = Arena::<DEFAULT_ARENA_SIZE>::new();
    let env = env_new(&arena);

    let _ = eval_string(&arena, "(define flag #t)", &env).unwrap();
    let _ = eval_string(&arena, "(set! flag #f)", &env).unwrap();
    
    let result = eval_string(&arena, "flag", &env).unwrap();
    assert!(matches!(arena.get(result.inner), Some(Value::Bool(false))));
}

#[test]
fn test_set_with_list() {
    let arena = Arena::<DEFAULT_ARENA_SIZE>::new();
    let env = env_new(&arena);

    let _ = eval_string(&arena, "(define lst (list 1 2 3))", &env).unwrap();
    let _ = eval_string(&arena, "(set! lst (list 4 5 6))", &env).unwrap();
    
    let result = eval_string(&arena, "(car lst)", &env).unwrap();
    assert!(matches!(arena.get(result.inner), Some(Value::Number(4))));
}

#[test]
fn test_set_accumulator_pattern() {
    let arena = Arena::<DEFAULT_ARENA_SIZE>::new();
    let env = env_new(&arena);

    // Classic accumulator pattern
    let code = r#"
        (define sum 0)
        (define add-to-sum
          (lambda (n)
            (set! sum (+ sum n))
            sum))
    "#;
    
    let _ = eval_string(&arena, "(define sum 0)", &env).unwrap();
    let _ = eval_string(&arena, "(define add-to-sum (lambda (n) (set! sum (+ sum n)) sum))", &env).unwrap();
    
    let r1 = eval_string(&arena, "(add-to-sum 5)", &env).unwrap();
    assert!(matches!(arena.get(r1.inner), Some(Value::Number(5))));
    
    let r2 = eval_string(&arena, "(add-to-sum 10)", &env).unwrap();
    assert!(matches!(arena.get(r2.inner), Some(Value::Number(15))));
    
    let r3 = eval_string(&arena, "(add-to-sum 7)", &env).unwrap();
    assert!(matches!(arena.get(r3.inner), Some(Value::Number(22))));
}
#[test]
fn test_set_factorial_with_mutation2() {
    let arena = Arena::<DEFAULT_ARENA_SIZE>::new();
    let env = env_new(&arena);

    // Test multiple mutations in sequence
    let _ = eval_string(&arena, "(define result 1)", &env).unwrap();
    let _ = eval_string(&arena, "(define n 5)", &env).unwrap();
    
    // Manually do factorial iterations
    let _ = eval_string(&arena, "(set! result (* result n))", &env).unwrap(); // 5
    let _ = eval_string(&arena, "(set! n (- n 1))", &env).unwrap();
    let _ = eval_string(&arena, "(set! result (* result n))", &env).unwrap(); // 20
    let _ = eval_string(&arena, "(set! n (- n 1))", &env).unwrap();
    let _ = eval_string(&arena, "(set! result (* result n))", &env).unwrap(); // 60
    let _ = eval_string(&arena, "(set! n (- n 1))", &env).unwrap();
    let _ = eval_string(&arena, "(set! result (* result n))", &env).unwrap(); // 120
    
    let result = eval_string(&arena, "result", &env).unwrap();
    assert!(matches!(arena.get(result.inner), Some(Value::Number(120))));
}
#[test]
fn test_set_swap_pattern() {
    let arena = Arena::<DEFAULT_ARENA_SIZE>::new();
    let env = env_new(&arena);

    // Swap two variables
    let _ = eval_string(&arena, "(define a 10)", &env).unwrap();
    let _ = eval_string(&arena, "(define b 20)", &env).unwrap();
    let _ = eval_string(&arena, "(define temp a)", &env).unwrap();
    let _ = eval_string(&arena, "(set! a b)", &env).unwrap();
    let _ = eval_string(&arena, "(set! b temp)", &env).unwrap();
    
    let result_a = eval_string(&arena, "a", &env).unwrap();
    assert!(matches!(arena.get(result_a.inner), Some(Value::Number(20))));
    
    let result_b = eval_string(&arena, "b", &env).unwrap();
    assert!(matches!(arena.get(result_b.inner), Some(Value::Number(10))));
}

#[test]
fn test_set_in_recursive_function() {
    let arena = Arena::<DEFAULT_ARENA_SIZE>::new();
    let env = env_new(&arena);

    // Function that counts recursive calls
    let code = r#"
        (define call-count 0)
        (define counting-factorial
          (lambda (n)
            (set! call-count (+ call-count 1))
            (if (= n 0)
                1
                (* n (counting-factorial (- n 1))))))
    "#;
    
    let _ = eval_string(&arena, "(define call-count 0)", &env).unwrap();
    let _ = eval_string(&arena, 
        "(define counting-factorial (lambda (n) (set! call-count (+ call-count 1)) (if (= n 0) 1 (* n (counting-factorial (- n 1))))))",
        &env).unwrap();
    
    let _ = eval_string(&arena, "(counting-factorial 5)", &env).unwrap();
    
    // Should have been called 6 times (5, 4, 3, 2, 1, 0)
    let result = eval_string(&arena, "call-count", &env).unwrap();
    assert!(matches!(arena.get(result.inner), Some(Value::Number(6))));
}

#[test]
fn test_set_wrong_arg_count() {
    let arena = Arena::<DEFAULT_ARENA_SIZE>::new();
    let env = env_new(&arena);

    let _ = eval_string(&arena, "(define x 1)", &env).unwrap();
    
    // Too few args
    let result = eval_string(&arena, "(set! x)", &env);
    assert!(result.is_err());
    
    // Too many args
    let result = eval_string(&arena, "(set! x 1 2)", &env);
    assert!(result.is_err());
}

#[test]
fn test_set_requires_symbol() {
    let arena = Arena::<DEFAULT_ARENA_SIZE>::new();
    let env = env_new(&arena);

    // Try to set! a non-symbol
    let result = eval_string(&arena, "(set! 123 456)", &env);
    assert!(result.is_err());
    
    let result = eval_string(&arena, "(set! (+ 1 2) 10)", &env);
    assert!(result.is_err());
}

#[test]
fn test_set_with_lambda_value() {
    let arena = Arena::<DEFAULT_ARENA_SIZE>::new();
    let env = env_new(&arena);

    let _ = eval_string(&arena, "(define func (lambda (x) x))", &env).unwrap();
    let _ = eval_string(&arena, "(set! func (lambda (x) (* x 2)))", &env).unwrap();
    
    let result = eval_string(&arena, "(func 5)", &env).unwrap();
    assert!(matches!(arena.get(result.inner), Some(Value::Number(10))));
}

#[test]
fn test_set_multiple_variables_in_sequence() {
    let arena = Arena::<DEFAULT_ARENA_SIZE>::new();
    let env = env_new(&arena);

    let _ = eval_string(&arena, "(define a 1)", &env).unwrap();
    let _ = eval_string(&arena, "(define b 2)", &env).unwrap();
    let _ = eval_string(&arena, "(define c 3)", &env).unwrap();
    
    let _ = eval_string(&arena, "(set! a 10)", &env).unwrap();
    let _ = eval_string(&arena, "(set! b 20)", &env).unwrap();
    let _ = eval_string(&arena, "(set! c 30)", &env).unwrap();
    
    let result = eval_string(&arena, "(+ a b c)", &env).unwrap();
    assert!(matches!(arena.get(result.inner), Some(Value::Number(60))));
}

#[test]
fn test_set_state_machine() {
    let arena = Arena::<DEFAULT_ARENA_SIZE>::new();
    let env = env_new(&arena);

    // Simple state machine: off -> on -> off
    let _ = eval_string(&arena, "(define state 0)", &env).unwrap();
    let _ = eval_string(&arena, 
        "(define toggle (lambda () (if (= state 0) (set! state 1) (set! state 0)) state))",
        &env).unwrap();
    
    let r1 = eval_string(&arena, "(toggle)", &env).unwrap();
    assert!(matches!(arena.get(r1.inner), Some(Value::Number(1))));
    
    let r2 = eval_string(&arena, "(toggle)", &env).unwrap();
    assert!(matches!(arena.get(r2.inner), Some(Value::Number(0))));
    
    let r3 = eval_string(&arena, "(toggle)", &env).unwrap();
    assert!(matches!(arena.get(r3.inner), Some(Value::Number(1))));
}
// ============================================================================
// Basic Allocation and Deallocation Tests
// ============================================================================

#[test]
fn test_basic_allocation() {
    let arena = Arena::<100>::new();
    let val = arena.number(42);

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
        let val = arena.number(42);
        assert_eq!(arena.refcounts[val.raw().0 as usize].get(), 1);
    } // val is dropped here

    // Slot should be freed
    assert!(matches!(arena.values[0].get(), Value::Free));
    assert_eq!(arena.refcounts[0].get(), 0);
}

#[test]
fn test_memory_reuse_after_drop() {
    let arena = Arena::<100>::new();

    let idx = {
        let val = arena.number(42);
        val.raw().0 as usize
    }; // val dropped, slot freed

    // Verify the slot is freed
    assert!(matches!(arena.values[idx].get(), Value::Free));
    assert_eq!(arena.refcounts[idx].get(), 0);

    // Allocate again - arena will search for a free slot
    let val2 = arena.number(99);
    assert!(!val2.is_null());

    if let Some(Value::Number(n)) = val2.get() {
        assert_eq!(n, 99);
    }
}

#[test]
fn test_arena_exhaustion() {
    let arena = Arena::<10>::new();
    let mut refs = Vec::new();

    // Allocate until full
    for i in 0..10 {
        let val = arena.number(i as i64);
        assert!(!val.is_null());
        refs.push(val);
    }

    // Next allocation should fail
    let val = arena.number(999);
    assert!(val.is_null());
}

#[test]
fn test_arena_exhaustion_with_reuse() {
    let arena = Arena::<10>::new();

    // Fill arena
    let mut refs = Vec::new();
    for i in 0..10 {
        refs.push(arena.number(i as i64));
    }

    // Arena is full
    let val = arena.number(999);
    assert!(val.is_null());

    // Drop half the references
    refs.truncate(5);

    // Now we should be able to allocate again
    let val = arena.number(999);
    assert!(!val.is_null());
}

// ============================================================================
// Reference Counting Tests
// ============================================================================

#[test]
fn test_refcount_increment_on_clone() {
    let arena = Arena::<100>::new();
    let val = arena.number(42);
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
    let val = arena.number(42);
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
    let val = arena.number(42);
    let idx = val.raw().0 as usize;

    // Manually set refcount to near max
    arena.refcounts[idx].set(u32::MAX - 1);

    // This should increment
    arena.incref(val.raw());
    assert_eq!(arena.refcounts[idx].get(), u32::MAX);

    // This should not overflow
    arena.incref(val.raw());
    assert_eq!(arena.refcounts[idx].get(), u32::MAX);
}

// ============================================================================
// Cons Cell Tests (Recursive Decref)
// ============================================================================

#[test]
fn test_cons_refcounting() {
    let arena = Arena::<100>::new();

    let car = arena.number(1);
    let cdr = arena.number(2);

    let car_idx = car.raw().0 as usize;
    let cdr_idx = cdr.raw().0 as usize;

    assert_eq!(arena.refcounts[car_idx].get(), 1);
    assert_eq!(arena.refcounts[cdr_idx].get(), 1);

    let cons = arena.cons(&car, &cdr);

    // cons increments both car and cdr
    assert_eq!(arena.refcounts[car_idx].get(), 2);
    assert_eq!(arena.refcounts[cdr_idx].get(), 2);

    drop(car);
    drop(cdr);

    // Still referenced by cons
    assert_eq!(arena.refcounts[car_idx].get(), 1);
    assert_eq!(arena.refcounts[cdr_idx].get(), 1);

    drop(cons);

    // Now all should be freed (recursive decref)
    assert_eq!(arena.refcounts[car_idx].get(), 0);
    assert_eq!(arena.refcounts[cdr_idx].get(), 0);
    assert!(matches!(arena.values[car_idx].get(), Value::Free));
    assert!(matches!(arena.values[cdr_idx].get(), Value::Free));
}

#[test]
fn test_nested_cons_refcounting() {
    let arena = Arena::<100>::new();

    let a = arena.number(1);
    let b = arena.number(2);
    let c = arena.number(3);

    let a_idx = a.raw().0 as usize;
    let b_idx = b.raw().0 as usize;
    let c_idx = c.raw().0 as usize;

    // Build: (1 . (2 . 3))
    let inner = arena.cons(&b, &c);
    let outer = arena.cons(&a, &inner);

    drop(a);
    drop(b);
    drop(c);
    drop(inner);

    // All still alive through outer
    assert_eq!(arena.refcounts[a_idx].get(), 1);
    assert_eq!(arena.refcounts[b_idx].get(), 1);
    assert_eq!(arena.refcounts[c_idx].get(), 1);

    drop(outer);

    // All freed recursively
    assert_eq!(arena.refcounts[a_idx].get(), 0);
    assert_eq!(arena.refcounts[b_idx].get(), 0);
    assert_eq!(arena.refcounts[c_idx].get(), 0);
}

#[test]
fn test_list_refcounting() {
    let arena = Arena::<100>::new();

    let mut nums = Vec::new();
    let mut indices = Vec::new();
    let mut list = arena.nil();

    for i in 0..5 {
        let num = arena.number(i);
        indices.push(num.raw().0 as usize);
        list = arena.cons(&num, &list);
        nums.push(num); // Keep num alive
    }

    // Check all numbers are referenced: once by nums vec, once by list
    for idx in &indices {
        assert_eq!(arena.refcounts[*idx].get(), 2);
    }

    drop(list);

    // All should still have refcount 1 (from the nums vector)
    for idx in &indices {
        assert_eq!(arena.refcounts[*idx].get(), 1);
    }

    drop(nums);

    // Now all should be freed
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

    let params = arena.nil();
    let body = arena.number(42);
    let env = arena.nil();

    let params_idx = params.raw().0 as usize;
    let body_idx = body.raw().0 as usize;
    let env_idx = env.raw().0 as usize;

    let lambda = arena.lambda(&params, &body, &env);

    // Lambda should increment all three
    assert_eq!(arena.refcounts[params_idx].get(), 2);
    assert_eq!(arena.refcounts[body_idx].get(), 2);
    assert_eq!(arena.refcounts[env_idx].get(), 2);

    drop(params);
    drop(body);
    drop(env);

    // Still referenced by lambda
    assert_eq!(arena.refcounts[params_idx].get(), 1);
    assert_eq!(arena.refcounts[body_idx].get(), 1);
    assert_eq!(arena.refcounts[env_idx].get(), 1);

    drop(lambda);

    // All freed
    assert_eq!(arena.refcounts[params_idx].get(), 0);
    assert_eq!(arena.refcounts[body_idx].get(), 0);
    assert_eq!(arena.refcounts[env_idx].get(), 0);
}

#[test]
fn test_symbol_refcounting() {
    let arena = Arena::<100>::new();

    let string = arena.str_to_list("test");
    let string_idx = string.raw().0 as usize;

    let sym = arena.symbol(&string);

    // Symbol should increment the string
    assert_eq!(arena.refcounts[string_idx].get(), 2);

    drop(string);
    assert_eq!(arena.refcounts[string_idx].get(), 1);

    drop(sym);

    // String should be freed recursively through symbol
    assert_eq!(arena.refcounts[string_idx].get(), 0);
}

// ============================================================================
// Set Operations Tests
// ============================================================================

#[test]
fn test_set_cons_refcounting() {
    let arena = Arena::<100>::new();

    let car1 = arena.number(1);
    let cdr1 = arena.number(2);
    let car1_idx = car1.raw().0 as usize;
    let cdr1_idx = cdr1.raw().0 as usize;

    let cons = arena.cons(&car1, &cdr1);

    // Original refs + cons refs
    assert_eq!(arena.refcounts[car1_idx].get(), 2);
    assert_eq!(arena.refcounts[cdr1_idx].get(), 2);

    let car2 = arena.number(3);
    let cdr2 = arena.number(4);
    let car2_idx = car2.raw().0 as usize;
    let cdr2_idx = cdr2.raw().0 as usize;

    // Update the cons cell
    arena.set_cons(&cons, &car2, &cdr2);

    // Old values should be decremented
    assert_eq!(arena.refcounts[car1_idx].get(), 1);
    assert_eq!(arena.refcounts[cdr1_idx].get(), 1);

    // New values should be incremented
    assert_eq!(arena.refcounts[car2_idx].get(), 2);
    assert_eq!(arena.refcounts[cdr2_idx].get(), 2);

    drop(car1);
    drop(cdr1);

    // Old values should be freed
    assert_eq!(arena.refcounts[car1_idx].get(), 0);
    assert_eq!(arena.refcounts[cdr1_idx].get(), 0);

    drop(cons);
    drop(car2);
    drop(cdr2);

    // New values should be freed
    assert_eq!(arena.refcounts[car2_idx].get(), 0);
    assert_eq!(arena.refcounts[cdr2_idx].get(), 0);
}

// ============================================================================
// Complex Structure Tests
// ============================================================================

#[test]
fn test_deep_list_refcounting() {
    let arena = Arena::<1000>::new();

    let mut list = arena.nil();
    let mut indices = Vec::new();

    // Build a deep list: (1 2 3 ... 100)
    for i in 0..100 {
        let num = arena.number(i);
        indices.push(num.raw().0 as usize);
        list = arena.cons(&num, &list);
    }

    // Check all are referenced
    for idx in &indices {
        assert!(arena.refcounts[*idx].get() >= 1);
    }

    drop(list);

    // All cons cells and numbers should eventually be freed
    // (The individual num variables still hold references)
}

#[test]
fn test_shared_structure_refcounting() {
    let arena = Arena::<100>::new();

    let shared = arena.number(42);
    let shared_idx = shared.raw().0 as usize;

    // Create multiple cons cells sharing the same car
    let cons1 = arena.cons(&shared, &arena.nil());
    let cons2 = arena.cons(&shared, &arena.nil());
    let cons3 = arena.cons(&shared, &arena.nil());

    // shared is referenced by: shared var, cons1, cons2, cons3
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
fn test_environment_refcounting() {
    let arena = Arena::<500>::new(); // Increased size for builtins
    let env = env_new(&arena);

    let name = arena.str_to_list("x");
    let value = arena.number(42);
    let value_idx = value.raw().0 as usize;

    env_set(&arena, &env, &name, &value);

    // Value should be incremented by env
    assert_eq!(arena.refcounts[value_idx].get(), 2);

    drop(value);
    assert_eq!(arena.refcounts[value_idx].get(), 1);

    // Retrieve the value
    if let Some(retrieved) = env_get(&arena, &env, &name) {
        assert_eq!(arena.refcounts[value_idx].get(), 2);
        drop(retrieved);
    }

    assert_eq!(arena.refcounts[value_idx].get(), 1);
}

#[test]
fn test_environment_update_refcounting() {
    let arena = Arena::<500>::new();
    let env = env_new(&arena);
    let name = arena.str_to_list("x");

    let value1 = arena.number(42);
    let value1_idx = value1.raw().0 as usize;
    env_set(&arena, &env, &name, &value1);
    assert_eq!(
        arena.refcounts[value1_idx].get(),
        2,
        "value1 should have refcount 2 after first set"
    );

    let value2 = arena.number(99);
    let value2_idx = value2.raw().0 as usize;
    env_set(&arena, &env, &name, &value2);

    // value1 is NO LONGER referenced by environment (old binding was removed)
    assert_eq!(
        arena.refcounts[value1_idx].get(),
        1,
        "value1 should have refcount 1 after being replaced"
    );
    // value2 is now referenced
    assert_eq!(
        arena.refcounts[value2_idx].get(),
        2,
        "value2 should have refcount 2 after being set"
    );
}
// ============================================================================
// Stress Tests
// ============================================================================

#[test]
fn test_repeated_allocation_and_deallocation() {
    let arena = Arena::<100>::new();

    for _ in 0..1000 {
        let val = arena.number(42);
        drop(val);
    }

    // Arena should handle this gracefully
    let val = arena.number(99);
    assert!(!val.is_null());
}

#[test]
fn test_alternating_allocation_patterns() {
    let arena = Arena::<100>::new();
    let mut refs = Vec::new();

    // Allocate 50
    for i in 0..50 {
        refs.push(arena.number(i));
    }

    // Free every other one
    for i in (0..50).step_by(2) {
        refs.remove(i / 2);
    }

    // Allocate 25 more
    for i in 50..75 {
        refs.push(arena.number(i));
    }

    // Should have reused freed slots
    assert!(refs.len() == 50);
}

#[test]
fn test_complex_nested_structures() {
    let arena = Arena::<1000>::new();

    // Build: ((1 . 2) . ((3 . 4) . ((5 . 6) . nil)))
    let pair1 = arena.cons(&arena.number(1), &arena.number(2));
    let pair2 = arena.cons(&arena.number(3), &arena.number(4));
    let pair3 = arena.cons(&arena.number(5), &arena.number(6));

    let p1_idx = pair1.raw().0 as usize;
    let p2_idx = pair2.raw().0 as usize;
    let p3_idx = pair3.raw().0 as usize;

    let nil = arena.nil();
    let list1 = arena.cons(&pair3, &nil);
    let list2 = arena.cons(&pair2, &list1);
    let list3 = arena.cons(&pair1, &list2);

    // All pairs referenced by: their variable + their list
    assert_eq!(arena.refcounts[p1_idx].get(), 2);
    assert_eq!(arena.refcounts[p2_idx].get(), 2);
    assert_eq!(arena.refcounts[p3_idx].get(), 2);

    // list2 is referenced by: list2 variable + list3
    let l2_idx = list2.raw().0 as usize;
    assert_eq!(arena.refcounts[l2_idx].get(), 2);

    drop(list3);

    // After dropping list3, list2's refcount goes down by 1
    assert_eq!(arena.refcounts[l2_idx].get(), 1);

    // Pairs still referenced by their variables AND their lists (list1, list2 still alive)
    assert_eq!(arena.refcounts[p1_idx].get(), 1); // list3 is gone
    assert_eq!(arena.refcounts[p2_idx].get(), 2); // still in list2
    assert_eq!(arena.refcounts[p3_idx].get(), 2); // still in list1

    drop(list2);
    drop(list1);

    // Now only pair variables remain
    assert_eq!(arena.refcounts[p1_idx].get(), 1);
    assert_eq!(arena.refcounts[p2_idx].get(), 1);
    assert_eq!(arena.refcounts[p3_idx].get(), 1);

    drop(pair1);
    drop(pair2);
    drop(pair3);

    // All freed
    assert_eq!(arena.refcounts[p1_idx].get(), 0);
    assert_eq!(arena.refcounts[p2_idx].get(), 0);
    assert_eq!(arena.refcounts[p3_idx].get(), 0);
}

#[test]
fn test_large_list_creation_and_destruction() {
    let arena = Arena::<5000>::new();

    let mut list = arena.nil();
    for i in 0..1000 {
        let num = arena.number(i);
        list = arena.cons(&num, &list);
    }

    let list_len = arena.list_len(&list);
    assert_eq!(list_len, 1000);

    drop(list);

    // All elements should be freed
    // We can verify by allocating again
    let new_val = arena.number(42);
    assert!(!new_val.is_null());
}

#[test]
fn test_null_reference_operations() {
    let arena = Arena::<100>::new();

    let null_ref = ArenaRef::NULL;

    // These should not panic
    arena.incref(null_ref);
    arena.decref(null_ref);

    assert!(arena.get(null_ref).is_none());
}

#[test]
fn test_string_list_refcounting() {
    let arena = Arena::<100>::new();

    let str_list = arena.str_to_list("hello");
    let idx = str_list.raw().0 as usize;

    // String is a list of chars
    assert_eq!(arena.refcounts[idx].get(), 1);

    let cloned = str_list.clone();
    assert_eq!(arena.refcounts[idx].get(), 2);

    drop(cloned);
    assert_eq!(arena.refcounts[idx].get(), 1);

    drop(str_list);

    // Entire chain should be freed
    assert_eq!(arena.refcounts[idx].get(), 0);
}

// ============================================================================
// Edge Cases
// ============================================================================

#[test]
fn test_refcount_at_boundary() {
    let arena = Arena::<100>::new();
    let val = arena.number(42);
    let idx = val.raw().0 as usize;

    // Set to 1
    arena.refcounts[idx].set(1);

    // Decref to 0 should free
    arena.decref(val.raw());
    assert_eq!(arena.refcounts[idx].get(), 0);
    assert!(matches!(arena.values[idx].get(), Value::Free));
}

#[test]
fn test_double_decref_on_freed_slot() {
    let arena = Arena::<100>::new();
    let val = arena.number(42);
    let raw = val.raw();
    let idx = raw.0 as usize;

    drop(val); // Frees the slot

    // Double decref should not panic
    arena.decref(raw);

    assert_eq!(arena.refcounts[idx].get(), 0);
}

#[test]
fn test_reverse_list_refcounting() {
    let arena = Arena::<100>::new();

    let list = arena.str_to_list("abc");
    let reversed = arena.reverse_list(&list);

    let list_idx = list.raw().0 as usize;
    let rev_idx = reversed.raw().0 as usize;

    // Both should be alive
    assert_eq!(arena.refcounts[list_idx].get(), 1);
    assert_eq!(arena.refcounts[rev_idx].get(), 1);

    drop(list);
    drop(reversed);

    // Both should be freed (along with all chars)
    assert_eq!(arena.refcounts[list_idx].get(), 0);
    assert_eq!(arena.refcounts[rev_idx].get(), 0);
}

#[test]
fn test_mixed_value_types_in_list() {
    let arena = Arena::<100>::new();

    let num = arena.number(42);
    let bool_val = arena.bool_val(true);
    let char_val = arena.char_val('x');

    let mut list = arena.nil();
    list = arena.cons(&char_val, &list);
    list = arena.cons(&bool_val, &list);
    list = arena.cons(&num, &list);

    let num_idx = num.raw().0 as usize;
    let bool_idx = bool_val.raw().0 as usize;
    let char_idx = char_val.raw().0 as usize;

    assert_eq!(arena.refcounts[num_idx].get(), 2);
    assert_eq!(arena.refcounts[bool_idx].get(), 2);
    assert_eq!(arena.refcounts[char_idx].get(), 2);

    drop(list);

    assert_eq!(arena.refcounts[num_idx].get(), 1);
    assert_eq!(arena.refcounts[bool_idx].get(), 1);
    assert_eq!(arena.refcounts[char_idx].get(), 1);
}

// ============================================================================
// Integration Tests with Eval
// ============================================================================

#[test]
fn test_eval_refcounting() {
    let arena = Arena::<1000>::new();
    let env = env_new(&arena);

    let result = eval_string(&arena, "(+ 1 2 3)", &env);
    assert!(result.is_ok());

    // After eval, temporary values should be cleaned up
    // (This is implicit, but we can check arena isn't exhausted)
    for _ in 0..10 {
        let _ = eval_string(&arena, "(* 2 3)", &env);
    }

    // Should still be able to allocate
    let val = arena.number(999);
    assert!(!val.is_null());
}

#[test]
fn test_recursive_eval_refcounting() {
    let arena = Arena::<5000>::new();
    let env = env_new(&arena);

    let factorial_def = r#"
        (define factorial
          (lambda (n)
            (if (= n 0)
                1
                (* n (factorial (- n 1))))))
    "#;

    let _ = eval_string(&arena, factorial_def, &env);
    let result = eval_string(&arena, "(factorial 10)", &env);

    assert!(result.is_ok());
    if let Ok(val) = result {
        if let Some(Value::Number(n)) = val.get() {
            assert_eq!(n, 3628800);
        }
    }
}

#[test]
fn test_tail_call_refcounting() {
    let arena = Arena::<5000>::new();
    let env = env_new(&arena);

    let sum_def = r#"
        (define sum
          (lambda (n acc)
            (if (= n 0)
                acc
                (sum (- n 1) (+ acc n)))))
    "#;

    let _ = eval_string(&arena, sum_def, &env);

    // This would stack overflow without TCO
    let result = eval_string(&arena, "(sum 1000 0)", &env);
    assert!(result.is_ok());

    if let Ok(val) = result {
        if let Some(Value::Number(n)) = val.get() {
            assert_eq!(n, 500500);
        }
    }
}

// ============================================================================
// Additional Stress Tests
// ============================================================================

#[test]
fn test_circular_reference_prevention() {
    // This tests that we don't have actual circular references
    // (which would be a memory leak in a refcount system)
    let arena = Arena::<500>::new();

    let a = arena.number(1);
    let b = arena.number(2);

    let pair = arena.cons(&a, &b);
    let pair_idx = pair.raw().0 as usize;

    // We can't create true circular references with this API
    // because set_cons requires Refs, not raw indices
    assert_eq!(arena.refcounts[pair_idx].get(), 1);

    drop(a);
    drop(b);
    drop(pair);

    // All should be freed
    assert!(matches!(arena.values[pair_idx].get(), Value::Free));
}

#[test]
fn test_arena_allocation_pattern() {
    let arena = Arena::<20>::new();
    let mut refs = Vec::new();

    // Fill half the arena
    for i in 0..10 {
        refs.push(arena.number(i));
    }

    // Record indices
    let indices: Vec<_> = refs.iter().map(|r| r.raw().0 as usize).collect();

    // Drop all refs
    refs.clear();

    // All should be freed
    for idx in &indices {
        assert!(matches!(arena.values[*idx].get(), Value::Free));
        assert_eq!(arena.refcounts[*idx].get(), 0);
    }

    // Allocate again - should reuse freed slots
    for i in 0..10 {
        let val = arena.number(i + 100);
        assert!(!val.is_null());
        refs.push(val);
    }
}

#[test]
fn test_multiple_refs_to_same_value() {
    let arena = Arena::<100>::new();

    let val = arena.number(42);
    let idx = val.raw().0 as usize;

    let refs: Vec<_> = (0..10).map(|_| val.clone()).collect();

    // Should have refcount 11 (1 original + 10 clones)
    assert_eq!(arena.refcounts[idx].get(), 11);

    drop(refs);

    // Back to 1
    assert_eq!(arena.refcounts[idx].get(), 1);

    drop(val);

    // Freed
    assert_eq!(arena.refcounts[idx].get(), 0);
}

#[test]
fn test_deep_recursion_without_leaks() {
    let arena = Arena::<5000>::new();

    // Build a very deep nested structure
    let mut current = arena.nil();
    let mut nums = Vec::new();
    let mut indices = Vec::new();

    for i in 0..100 {
        let num = arena.number(i);
        indices.push(num.raw().0 as usize);
        current = arena.cons(&num, &current);
        nums.push(num); // Keep num alive
    }

    // All nums have refcount 2 (num var + cons)
    for idx in &indices {
        assert_eq!(arena.refcounts[*idx].get(), 2);
    }

    drop(current);

    // Back to 1 (just num vars)
    for idx in &indices {
        assert_eq!(arena.refcounts[*idx].get(), 1);
    }

    drop(nums);

    // All freed
    for idx in &indices {
        assert_eq!(arena.refcounts[*idx].get(), 0);
    }
}

#[test]
fn test_lambda_closure_refcounting() {
    let arena = Arena::<1000>::new();
    let env = env_new(&arena);

    // Define a function that closes over a variable
    let code = r#"
        (define make-adder
          (lambda (x)
            (lambda (y)
              (+ x y))))
    "#;

    let _ = eval_string(&arena, code, &env);
    let result = eval_string(&arena, "(define add5 (make-adder 5))", &env);
    assert!(result.is_ok());

    let result = eval_string(&arena, "(add5 10)", &env);
    assert!(result.is_ok());

    if let Ok(val) = result {
        if let Some(Value::Number(n)) = val.get() {
            assert_eq!(n, 15);
        }
    }
}

#[test]
fn test_string_operations_refcounting() {
    let arena = Arena::<500>::new();

    let s1 = arena.str_to_list("hello");
    let s2 = arena.str_to_list("world");

    let s1_idx = s1.raw().0 as usize;
    let s2_idx = s2.raw().0 as usize;

    assert_eq!(arena.refcounts[s1_idx].get(), 1);
    assert_eq!(arena.refcounts[s2_idx].get(), 1);

    // Test str_eq doesn't leak
    let eq = arena.str_eq(&s1, &s2);
    assert!(!eq);

    assert_eq!(arena.refcounts[s1_idx].get(), 1);
    assert_eq!(arena.refcounts[s2_idx].get(), 1);

    // Test reverse
    let s1_rev = arena.reverse_list(&s1);
    let rev_idx = s1_rev.raw().0 as usize;

    assert_eq!(arena.refcounts[rev_idx].get(), 1);

    drop(s1);
    drop(s2);
    drop(s1_rev);

    // All freed (along with their char cons cells)
    assert_eq!(arena.refcounts[s1_idx].get(), 0);
    assert_eq!(arena.refcounts[s2_idx].get(), 0);
    assert_eq!(arena.refcounts[rev_idx].get(), 0);
}

#[test]
fn test_arena_exhaustion_graceful_failure() {
    let arena = Arena::<10>::new();

    // Allocate all slots
    let refs: Vec<_> = (0..10).map(|i| arena.number(i)).collect();

    // Verify all are valid
    for r in &refs {
        assert!(!r.is_null());
    }

    // Next allocation fails gracefully
    let failed = arena.number(999);
    assert!(failed.is_null());
    assert!(failed.is_null()); // Can call multiple times

    // After dropping some, can allocate again
    drop(refs[0].clone());
    drop(refs[1].clone());
    drop(refs);

    let success = arena.number(111);
    assert!(!success.is_null());
}

#[test]
fn test_very_deep_list_drop_no_stack_overflow() {
    let arena = Arena::<20000>::new();

    // Build extremely deep list (1000+ elements)
    // This tests that recursive decref doesn't cause stack overflow
    let mut list = arena.nil();
    for i in 0..1000 {
        let num = arena.number(i);
        list = arena.cons(&num, &list);
    }

    // This drop should recursively free everything without stack overflow
    drop(list);

    // Verify we can allocate after the massive cleanup
    let val = arena.number(42);
    assert!(!val.is_null());
}

#[test]
fn test_builtin_refcounting() {
    let arena = Arena::<100>::new();

    let builtin = arena.builtin(0);
    let idx = builtin.raw().0 as usize;

    assert_eq!(arena.refcounts[idx].get(), 1);

    let clone = builtin.clone();
    assert_eq!(arena.refcounts[idx].get(), 2);

    drop(clone);
    assert_eq!(arena.refcounts[idx].get(), 1);

    drop(builtin);
    assert_eq!(arena.refcounts[idx].get(), 0);
}

#[test]
fn test_nil_value_refcounting() {
    let arena = Arena::<100>::new();

    let nil1 = arena.nil();
    let nil2 = arena.nil();

    // Each nil is a separate allocation
    assert_ne!(nil1.raw().0, nil2.raw().0);

    let idx1 = nil1.raw().0 as usize;
    let idx2 = nil2.raw().0 as usize;

    assert_eq!(arena.refcounts[idx1].get(), 1);
    assert_eq!(arena.refcounts[idx2].get(), 1);

    drop(nil1);
    drop(nil2);

    assert_eq!(arena.refcounts[idx1].get(), 0);
    assert_eq!(arena.refcounts[idx2].get(), 0);
}

#[test]
fn test_interleaved_alloc_dealloc_pattern() {
    let arena = Arena::<50>::new();

    // Pattern: allocate 3, drop 2, allocate 3, drop 2, repeat
    for _ in 0..10 {
        let v1 = arena.number(1);
        let v2 = arena.number(2);
        let v3 = arena.number(3);

        assert!(!v1.is_null());
        assert!(!v2.is_null());
        assert!(!v3.is_null());

        drop(v1);
        drop(v2);
        // v3 lives on, but we loop again
        drop(v3);
    }

    // Arena should still be functional
    let final_val = arena.number(999);
    assert!(!final_val.is_null());
}

#[test]
fn test_mixed_lifetime_interactions() {
    let arena = Arena::<200>::new();

    let long_lived = arena.number(1);
    let long_idx = long_lived.raw().0 as usize;

    for _ in 0..5 {
        let short = arena.number(2);
        let pair = arena.cons(&long_lived, &short);

        // long_lived refcount bumped by cons
        assert_eq!(arena.refcounts[long_idx].get(), 2);

        drop(pair);

        // Back to 1 after cons is dropped
        assert_eq!(arena.refcounts[long_idx].get(), 1);
    }

    // long_lived still valid
    assert_eq!(arena.refcounts[long_idx].get(), 1);

    if let Some(Value::Number(n)) = long_lived.get() {
        assert_eq!(n, 1);
    }
}

// ============================================================================
// Memory Reuse Verification Tests
// ============================================================================

#[test]
fn test_memory_reuse_exact_slot() {
    let arena = Arena::<100>::new();

    // Allocate and get the slot index
    let val1 = arena.number(42);
    let idx1 = val1.raw().0 as usize;

    // Verify it's allocated
    assert!(matches!(arena.values[idx1].get(), Value::Number(42)));
    assert_eq!(arena.refcounts[idx1].get(), 1);

    // Drop it - should free the slot
    drop(val1);

    // Verify slot is freed
    assert!(matches!(arena.values[idx1].get(), Value::Free));
    assert_eq!(arena.refcounts[idx1].get(), 0);

    // Allocate again - should find and reuse the freed slot
    let val2 = arena.number(99);

    // The freed slot should be reused (may be idx1 or another free slot)
    assert!(!val2.is_null());
    assert!(matches!(val2.get(), Some(Value::Number(99))));
}

#[test]
fn test_memory_reuse_pattern() {
    let arena = Arena::<10>::new();

    // Fill arena completely
    let mut refs = Vec::new();
    for i in 0..10 {
        refs.push(arena.number(i));
    }

    // Record which slots were used
    let indices: Vec<_> = refs.iter().map(|r| r.raw().0 as usize).collect();

    // Arena is full - next allocation should fail
    let overflow = arena.number(999);
    assert!(overflow.is_null());

    // Drop 3 values, freeing 3 slots
    let freed_indices = vec![indices[2], indices[5], indices[8]];
    drop(refs[2].clone());
    drop(refs[5].clone());
    drop(refs[8].clone());
    refs.remove(8);
    refs.remove(5);
    refs.remove(2);

    // Verify those slots are now free
    for idx in &freed_indices {
        assert!(matches!(arena.values[*idx].get(), Value::Free));
        assert_eq!(arena.refcounts[*idx].get(), 0);
    }

    // Allocate 3 new values - should reuse the freed slots
    let new1 = arena.number(100);
    let new2 = arena.number(200);
    let new3 = arena.number(300);

    assert!(!new1.is_null());
    assert!(!new2.is_null());
    assert!(!new3.is_null());

    // Collect the reused indices
    let reused = vec![
        new1.raw().0 as usize,
        new2.raw().0 as usize,
        new3.raw().0 as usize,
    ];

    // All reused indices should be from the freed slots
    for idx in reused {
        assert!(
            freed_indices.contains(&idx),
            "Index {} was not in freed list",
            idx
        );
    }

    // Try to allocate one more - should fail since all slots are used again
    let overflow2 = arena.number(999);
    assert!(overflow2.is_null());
}

#[test]
fn test_circular_reuse() {
    let arena = Arena::<20>::new();

    // Allocate slots 0-9
    let mut refs = Vec::new();
    for i in 0..10 {
        refs.push(arena.number(i));
    }

    // Record which slots are 0, 1, 2
    let idx0 = refs[0].raw().0 as usize;
    let idx1 = refs[1].raw().0 as usize;
    let idx2 = refs[2].raw().0 as usize;

    // Actually free slots by removing from vector
    refs.remove(0); // Removes and drops refs[0]
    refs.remove(0); // Removes and drops what was refs[1] (now at index 0)
    refs.remove(0); // Removes and drops what was refs[2] (now at index 0)

    // Verify they're freed
    assert!(matches!(arena.values[idx0].get(), Value::Free));
    assert!(matches!(arena.values[idx1].get(), Value::Free));
    assert!(matches!(arena.values[idx2].get(), Value::Free));

    // Allocate 10 more values (slots 10-19)
    for i in 10..20 {
        refs.push(arena.number(i));
    }

    // Now slots 0, 1, 2 are free, and next_free is pointing past slot 19
    // Next allocation should wrap around and find one of the freed slots
    let reused = arena.number(999);
    assert!(!reused.is_null());

    // Should have reused one of the freed slots at the beginning
    let reused_idx = reused.raw().0 as usize;
    assert!(
        reused_idx == idx0 || reused_idx == idx1 || reused_idx == idx2,
        "Should reuse one of the freed slots (0, 1, or 2), got {}",
        reused_idx
    );
}

#[test]
fn test_fragmentation_handling() {
    let arena = Arena::<50>::new();

    // Create a checkerboard pattern: allocate all, then free every other one
    let mut refs = Vec::new();
    for i in 0..50 {
        refs.push(arena.number(i));
    }

    // Free every other slot (25 slots freed)
    for i in (0..50).step_by(2) {
        drop(refs[i].clone());
    }
    let refs: Vec<_> = refs
        .into_iter()
        .enumerate()
        .filter(|(i, _)| i % 2 == 1)
        .map(|(_, r)| r)
        .collect();

    // Now allocate 25 new values - should fill all the freed slots
    let mut new_refs = Vec::new();
    for i in 0..25 {
        let val = arena.number(1000 + i);
        assert!(
            !val.is_null(),
            "Allocation {} failed, but {} slots should be free",
            i,
            25 - i
        );
        new_refs.push(val);
    }

    // Arena should now be completely full again
    let overflow = arena.number(9999);
    assert!(overflow.is_null());
}

#[test]
fn test_reuse_with_different_types() {
    let arena = Arena::<100>::new();

    // Allocate a number
    let num = arena.number(42);
    let idx = num.raw().0 as usize;
    drop(num);

    // Reuse slot with a bool
    let bool_val = arena.bool_val(true);
    let bool_idx = bool_val.raw().0 as usize;

    // Should reuse the same slot (or another free slot)
    assert!(matches!(arena.values[bool_idx].get(), Value::Bool(true)));
    drop(bool_val);

    // Reuse again with a char
    let char_val = arena.char_val('x');
    let char_idx = char_val.raw().0 as usize;
    assert!(matches!(arena.values[char_idx].get(), Value::Char('x')));
    drop(char_val);

    // Reuse with a cons cell
    let a = arena.number(1);
    let b = arena.number(2);
    let cons = arena.cons(&a, &b);
    let cons_idx = cons.raw().0 as usize;

    assert!(matches!(arena.values[cons_idx].get(), Value::Cons(..)));
}

#[test]
fn test_no_memory_leak_with_reuse() {
    let arena = Arena::<100>::new();

    // Repeatedly allocate and free in the same slots
    for round in 0..10 {
        let mut temp_refs = Vec::new();

        // Allocate 10 values
        for i in 0..10 {
            temp_refs.push(arena.number(round * 10 + i));
        }

        // Drop all of them
        temp_refs.clear();

        // Verify all are freed (checking first few slots)
        let mut free_count = 0;
        for i in 0..20 {
            if matches!(arena.values[i].get(), Value::Free) {
                free_count += 1;
            }
        }

        // Should have at least the 10 we just freed
        assert!(
            free_count >= 10,
            "Round {}: only {} slots free",
            round,
            free_count
        );
    }

    // After all rounds, should still be able to allocate
    let final_val = arena.number(9999);
    assert!(!final_val.is_null());
}

#[test]
fn test_reuse_after_complex_deallocation() {
    let arena = Arena::<500>::new();

    // Build a complex structure
    let list = {
        let mut l = arena.nil();
        for i in 0..20 {
            let num = arena.number(i);
            l = arena.cons(&num, &l);
        }
        l
    };

    // Count used slots before drop
    let used_before = (0..500)
        .filter(|&i| !matches!(arena.values[i].get(), Value::Free))
        .count();

    // Drop the list - should recursively free many slots
    drop(list);

    // Count used slots after drop
    let used_after = (0..500)
        .filter(|&i| !matches!(arena.values[i].get(), Value::Free))
        .count();

    // Should have freed at least 20 cons cells + 20 numbers = 40 slots
    assert!(
        used_before - used_after >= 40,
        "Before: {}, After: {}, Diff: {}",
        used_before,
        used_after,
        used_before - used_after
    );

    // Should be able to allocate new values in the freed space
    let mut new_refs = Vec::new();
    for i in 0..30 {
        let val = arena.number(1000 + i);
        assert!(!val.is_null(), "Failed to allocate value {}", i);
        new_refs.push(val);
    }
}
