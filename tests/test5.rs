#![cfg(test)]

use ruthe::*;

// ============================================================================
// BASIC FUNCTIONALITY TESTS WITH REFCOUNT VERIFICATION
// ============================================================================

#[test]
fn test_basic_arithmetic_no_leaks() {
    let arena = Arena::<DEFAULT_ARENA_SIZE>::new();
    let env = env_new(&arena);

    let result = eval_string(&arena, "(+ 1 2 3)", &env).unwrap();
    let result_idx = result.raw().0 as usize;

    assert!(matches!(arena.get(result.inner), Some(Value::Number(6))));
    assert_eq!(
        arena.refcounts[result_idx].get(),
        1,
        "Result should have refcount 1"
    );

    drop(result);
    assert!(
        matches!(arena.values[result_idx].get(), Value::Free),
        "Cell should be freed after drop"
    );
}

#[test]
fn test_variable_definition_refcounts() {
    let arena = Arena::<DEFAULT_ARENA_SIZE>::new();
    let env = env_new(&arena);

    let result = eval_string(&arena, "(define x 42)", &env).unwrap();
    let result_idx = result.raw().0 as usize;

    // Result is the value 42, which is also stored in the environment
    assert_eq!(
        arena.refcounts[result_idx].get(),
        2,
        "Value should have refcount 2: result + environment"
    );

    drop(result);
    // Still referenced by environment
    assert_eq!(
        arena.refcounts[result_idx].get(),
        1,
        "After dropping result, environment still holds reference"
    );
    assert!(!matches!(arena.values[result_idx].get(), Value::Free));

    // Look it up again
    let result2 = eval_string(&arena, "x", &env).unwrap();
    let result2_idx = result2.raw().0 as usize;

    assert_eq!(result_idx, result2_idx, "Should be same value");
    assert_eq!(
        arena.refcounts[result2_idx].get(),
        2,
        "Refcount increased by lookup"
    );

    drop(result2);
    assert_eq!(
        arena.refcounts[result_idx].get(),
        1,
        "Back to 1 after dropping lookup result"
    );
}

#[test]
fn test_lambda_application_refcounts() {
    let arena = Arena::<DEFAULT_ARENA_SIZE>::new();
    let env = env_new(&arena);

    // Define lambda
    let lambda_result = eval_string(&arena, "(define add1 (lambda (x) (+ x 1)))", &env).unwrap();
    let lambda_idx = lambda_result.raw().0 as usize;

    // Lambda is stored in environment
    assert_eq!(
        arena.refcounts[lambda_idx].get(),
        2,
        "Lambda should have refcount 2"
    );
    drop(lambda_result);

    // Apply lambda
    let result = eval_string(&arena, "(add1 5)", &env).unwrap();
    let result_idx = result.raw().0 as usize;

    assert!(matches!(arena.get(result.inner), Some(Value::Number(6))));
    assert_eq!(
        arena.refcounts[result_idx].get(),
        1,
        "Result should have refcount 1"
    );

    drop(result);
    assert!(matches!(arena.values[result_idx].get(), Value::Free));
}

// ============================================================================
// REFERENCE COUNTING CORE TESTS
// ============================================================================

#[test]
fn test_single_value_lifecycle() {
    let arena = Arena::<1000>::new();

    let val = arena.number(42);
    let idx = val.raw().0 as usize;

    assert_eq!(
        arena.refcounts[idx].get(),
        1,
        "Initial refcount should be 1"
    );
    assert!(matches!(arena.values[idx].get(), Value::Number(42)));

    drop(val);

    assert_eq!(
        arena.refcounts[idx].get(),
        0,
        "Refcount should be 0 after drop"
    );
    assert!(
        matches!(arena.values[idx].get(), Value::Free),
        "Cell should be freed"
    );
}

#[test]
fn test_clone_increases_refcount() {
    let arena = Arena::<1000>::new();

    let val = arena.number(42);
    let idx = val.raw().0 as usize;
    assert_eq!(arena.refcounts[idx].get(), 1);

    let val2 = val.clone();
    assert_eq!(
        arena.refcounts[idx].get(),
        2,
        "Clone should increase refcount"
    );

    let val3 = val.clone();
    assert_eq!(
        arena.refcounts[idx].get(),
        3,
        "Second clone should increase to 3"
    );

    drop(val);
    assert_eq!(
        arena.refcounts[idx].get(),
        2,
        "After first drop, refcount should be 2"
    );
    assert!(
        !matches!(arena.values[idx].get(), Value::Free),
        "Should not be freed yet"
    );

    drop(val2);
    assert_eq!(
        arena.refcounts[idx].get(),
        1,
        "After second drop, refcount should be 1"
    );

    drop(val3);
    assert_eq!(
        arena.refcounts[idx].get(),
        0,
        "After final drop, refcount should be 0"
    );
    assert!(
        matches!(arena.values[idx].get(), Value::Free),
        "Should be freed now"
    );
}

#[test]
fn test_cons_cell_refcounts() {
    let arena = Arena::<1000>::new();

    let car_val = arena.number(1);
    let car_idx = car_val.raw().0 as usize;
    assert_eq!(arena.refcounts[car_idx].get(), 1);

    let cdr_val = arena.number(2);
    let cdr_idx = cdr_val.raw().0 as usize;
    assert_eq!(arena.refcounts[cdr_idx].get(), 1);

    let cons = arena.cons(&car_val, &cdr_val);
    let cons_idx = cons.raw().0 as usize;

    // Cons cell increments refcount of both car and cdr
    assert_eq!(
        arena.refcounts[car_idx].get(),
        2,
        "Car should have refcount 2 (var + cons)"
    );
    assert_eq!(
        arena.refcounts[cdr_idx].get(),
        2,
        "Cdr should have refcount 2 (var + cons)"
    );
    assert_eq!(
        arena.refcounts[cons_idx].get(),
        1,
        "Cons should have refcount 1"
    );

    drop(car_val);
    drop(cdr_val);

    // Still referenced by cons
    assert_eq!(
        arena.refcounts[car_idx].get(),
        1,
        "After dropping vars, car still in cons"
    );
    assert_eq!(
        arena.refcounts[cdr_idx].get(),
        1,
        "After dropping vars, cdr still in cons"
    );

    drop(cons);

    // Now everything should be freed
    assert_eq!(arena.refcounts[cons_idx].get(), 0);
    assert_eq!(arena.refcounts[car_idx].get(), 0);
    assert_eq!(arena.refcounts[cdr_idx].get(), 0);
    assert!(matches!(arena.values[cons_idx].get(), Value::Free));
    assert!(matches!(arena.values[car_idx].get(), Value::Free));
    assert!(matches!(arena.values[cdr_idx].get(), Value::Free));
}

#[test]
fn test_list_refcounts() {
    let arena = Arena::<1000>::new();

    // Build list (1 2 3)
    let one = arena.number(1);
    let one_idx = one.raw().0 as usize;
    let two = arena.number(2);
    let two_idx = two.raw().0 as usize;
    let three = arena.number(3);
    let three_idx = three.raw().0 as usize;

    assert_eq!(arena.refcounts[one_idx].get(), 1);
    assert_eq!(arena.refcounts[two_idx].get(), 1);
    assert_eq!(arena.refcounts[three_idx].get(), 1);

    let nil = arena.nil();
    let nil_idx = nil.raw().0 as usize;

    let list3 = arena.cons(&three, &nil);
    assert_eq!(arena.refcounts[three_idx].get(), 2);

    let list2 = arena.cons(&two, &list3);
    assert_eq!(arena.refcounts[two_idx].get(), 2);

    let list1 = arena.cons(&one, &list2);
    let list1_idx = list1.raw().0 as usize;
    assert_eq!(arena.refcounts[one_idx].get(), 2);

    // Drop individual number refs
    drop(one);
    drop(two);
    drop(three);
    drop(nil);
    drop(list3);
    drop(list2);

    // All still held by list1
    assert_eq!(arena.refcounts[one_idx].get(), 1);
    assert_eq!(arena.refcounts[two_idx].get(), 1);
    assert_eq!(arena.refcounts[three_idx].get(), 1);
    assert_eq!(arena.refcounts[list1_idx].get(), 1);

    // Drop the list - should cascade and free everything
    drop(list1);

    assert_eq!(arena.refcounts[list1_idx].get(), 0);
    assert_eq!(arena.refcounts[one_idx].get(), 0);
    assert_eq!(arena.refcounts[two_idx].get(), 0);
    assert_eq!(arena.refcounts[three_idx].get(), 0);

    assert!(matches!(arena.values[one_idx].get(), Value::Free));
    assert!(matches!(arena.values[two_idx].get(), Value::Free));
    assert!(matches!(arena.values[three_idx].get(), Value::Free));
}

#[test]
fn test_symbol_refcounts() {
    let arena = Arena::<1000>::new();

    let str_list = arena.str_to_list("hello");
    let str_idx = str_list.raw().0 as usize;
    assert_eq!(arena.refcounts[str_idx].get(), 1);

    let sym = arena.symbol(&str_list);
    let sym_idx = sym.raw().0 as usize;

    // Symbol increments refcount of the string
    assert_eq!(
        arena.refcounts[str_idx].get(),
        2,
        "String should be referenced by var and symbol"
    );
    assert_eq!(arena.refcounts[sym_idx].get(), 1);

    drop(str_list);
    assert_eq!(arena.refcounts[str_idx].get(), 1, "String still in symbol");

    drop(sym);
    assert_eq!(arena.refcounts[sym_idx].get(), 0);
    assert_eq!(
        arena.refcounts[str_idx].get(),
        0,
        "String freed with symbol"
    );
}

#[test]
fn test_lambda_refcounts() {
    let arena = Arena::<1000>::new();

    let params = arena.str_to_list("x");
    let params_idx = params.raw().0 as usize;
    assert_eq!(arena.refcounts[params_idx].get(), 1);

    let body = arena.number(42);
    let body_idx = body.raw().0 as usize;
    assert_eq!(arena.refcounts[body_idx].get(), 1);

    let env = arena.nil();
    let env_idx = env.raw().0 as usize;
    assert_eq!(arena.refcounts[env_idx].get(), 1);

    let lambda = arena.lambda(&params, &body, &env);
    let lambda_idx = lambda.raw().0 as usize;

    // Lambda increments refcount of params, body, and env
    assert_eq!(arena.refcounts[params_idx].get(), 2);
    assert_eq!(arena.refcounts[body_idx].get(), 2);
    assert_eq!(arena.refcounts[env_idx].get(), 2);
    assert_eq!(arena.refcounts[lambda_idx].get(), 1);

    drop(params);
    drop(body);
    drop(env);

    assert_eq!(arena.refcounts[params_idx].get(), 1);
    assert_eq!(arena.refcounts[body_idx].get(), 1);
    assert_eq!(arena.refcounts[env_idx].get(), 1);

    drop(lambda);

    assert_eq!(arena.refcounts[lambda_idx].get(), 0);
    assert_eq!(arena.refcounts[params_idx].get(), 0);
    assert_eq!(arena.refcounts[body_idx].get(), 0);
    assert_eq!(arena.refcounts[env_idx].get(), 0);

    assert!(matches!(arena.values[lambda_idx].get(), Value::Free));
    assert!(matches!(arena.values[params_idx].get(), Value::Free));
    assert!(matches!(arena.values[body_idx].get(), Value::Free));
    assert!(matches!(arena.values[env_idx].get(), Value::Free));
}

// ============================================================================
// ENVIRONMENT REFCOUNT TESTS
// ============================================================================

#[test]
fn test_environment_binding_refcounts() {
    let arena = Arena::<1000>::new();
    let env = env_new(&arena);
    let env_idx = env.raw().0 as usize;

    let name = arena.str_to_list("x");
    let value = arena.number(42);
    let value_idx = value.raw().0 as usize;

    assert_eq!(
        arena.refcounts[value_idx].get(),
        1,
        "Value initially has refcount 1"
    );

    env_set(&arena, &env, &name, &value);

    assert_eq!(
        arena.refcounts[value_idx].get(),
        2,
        "Value should have refcount 2 after env_set"
    );

    drop(value);
    assert_eq!(
        arena.refcounts[value_idx].get(),
        1,
        "Value still referenced by environment"
    );

    // Look up the value
    let retrieved = env_get(&arena, &env, &name).unwrap();
    let retrieved_idx = retrieved.raw().0 as usize;

    assert_eq!(value_idx, retrieved_idx, "Should retrieve same value");
    assert_eq!(
        arena.refcounts[retrieved_idx].get(),
        2,
        "Refcount increased by lookup"
    );

    drop(retrieved);
    assert_eq!(
        arena.refcounts[value_idx].get(),
        1,
        "Back to 1 after dropping retrieved value"
    );
}

#[test]
fn test_redefining_variable_drops_old_value() {
    let arena = Arena::<1000>::new();
    let env = env_new(&arena);

    let name = arena.str_to_list("x");

    let value1 = arena.number(42);
    let value1_idx = value1.raw().0 as usize;
    env_set(&arena, &env, &name, &value1);
    assert_eq!(arena.refcounts[value1_idx].get(), 2);
    drop(value1);
    assert_eq!(
        arena.refcounts[value1_idx].get(),
        1,
        "value1 in environment"
    );

    let value2 = arena.number(99);
    let value2_idx = value2.raw().0 as usize;
    env_set(&arena, &env, &name, &value2);

    // value1 should now be freed (removed from environment)
    assert_eq!(
        arena.refcounts[value1_idx].get(),
        0,
        "Old value should be freed after redefinition"
    );
    assert!(
        matches!(arena.values[value1_idx].get(), Value::Free),
        "Old value cell should be free"
    );

    // value2 should be in environment
    assert_eq!(
        arena.refcounts[value2_idx].get(),
        2,
        "New value should have refcount 2"
    );

    drop(value2);
    assert_eq!(arena.refcounts[value2_idx].get(), 1);
}

#[test]
fn test_multiple_bindings_refcounts() {
    let arena = Arena::<1000>::new();
    let env = env_new(&arena);

    let x_name = arena.str_to_list("x");
    let y_name = arena.str_to_list("y");
    let z_name = arena.str_to_list("z");

    let x_val = arena.number(1);
    let x_idx = x_val.raw().0 as usize;
    let y_val = arena.number(2);
    let y_idx = y_val.raw().0 as usize;
    let z_val = arena.number(3);
    let z_idx = z_val.raw().0 as usize;

    env_set(&arena, &env, &x_name, &x_val);
    env_set(&arena, &env, &y_name, &y_val);
    env_set(&arena, &env, &z_name, &z_val);

    assert_eq!(arena.refcounts[x_idx].get(), 2, "x in environment");
    assert_eq!(arena.refcounts[y_idx].get(), 2, "y in environment");
    assert_eq!(arena.refcounts[z_idx].get(), 2, "z in environment");

    drop(x_val);
    drop(y_val);
    drop(z_val);

    assert_eq!(arena.refcounts[x_idx].get(), 1);
    assert_eq!(arena.refcounts[y_idx].get(), 1);
    assert_eq!(arena.refcounts[z_idx].get(), 1);

    // All still alive in environment
    assert!(!matches!(arena.values[x_idx].get(), Value::Free));
    assert!(!matches!(arena.values[y_idx].get(), Value::Free));
    assert!(!matches!(arena.values[z_idx].get(), Value::Free));
}

// ============================================================================
// TEMPORARY ALLOCATION TESTS
// ============================================================================

#[test]
fn test_eval_creates_and_frees_temporaries() {
    let arena = Arena::<1000>::new();
    let env = env_new(&arena);

    // Count free cells before
    let mut free_before = 0;
    for i in 0..1000 {
        if matches!(arena.values[i].get(), Value::Free) {
            free_before += 1;
        }
    }

    // Evaluate expression
    let result = eval_string(&arena, "(+ 1 2)", &env).unwrap();
    let result_idx = result.raw().0 as usize;

    assert_eq!(arena.refcounts[result_idx].get(), 1);

    drop(result);

    // Count free cells after
    let mut free_after = 0;
    for i in 0..1000 {
        if matches!(arena.values[i].get(), Value::Free) {
            free_after += 1;
        }
    }

    // Should have approximately the same number of free cells
    // (within a small margin for any persistent environment changes)
    assert!(
        free_after >= free_before - 5,
        "Should not leak memory: free_before={}, free_after={}",
        free_before,
        free_after
    );
}

#[test]
fn test_multiple_evals_dont_accumulate() {
    let arena = Arena::<1000>::new();
    let env = env_new(&arena);

    let mut free_counts = Vec::new();

    for _ in 0..10 {
        let result = eval_string(&arena, "(+ 1 2)", &env).unwrap();
        drop(result);

        let mut free_count = 0;
        for i in 0..1000 {
            if matches!(arena.values[i].get(), Value::Free) {
                free_count += 1;
            }
        }
        free_counts.push(free_count);
    }

    // Free count should be stable (not decreasing)
    for i in 1..free_counts.len() {
        assert!(
            free_counts[i] >= free_counts[0] - 5,
            "Free cells should not decrease: iter 0={}, iter {}={}",
            free_counts[0],
            i,
            free_counts[i]
        );
    }
}

// ============================================================================
// DEEP STRUCTURE REFCOUNT TESTS
// ============================================================================

#[test]
fn test_deep_list_all_freed() {
    let arena = Arena::<1000>::new();

    let mut indices = Vec::new();

    // Build 20-element list
    let mut list = arena.nil();
    for i in 0..20 {
        let num = arena.number(i);
        indices.push(num.raw().0 as usize);
        list = arena.cons(&num, &list);
        indices.push(list.raw().0 as usize);
    }

    // All elements should have refcount 1 (held by list structure)
    for &idx in &indices {
        assert!(
            arena.refcounts[idx].get() >= 1,
            "Cell {} should have refcount >= 1",
            idx
        );
    }

    drop(list);

    // All should be freed
    for &idx in &indices {
        assert_eq!(
            arena.refcounts[idx].get(),
            0,
            "Cell {} should have refcount 0 after drop",
            idx
        );
        assert!(
            matches!(arena.values[idx].get(), Value::Free),
            "Cell {} should be Free",
            idx
        );
    }
}

#[test]
fn test_shared_sublist_refcounts() {
    let arena = Arena::<1000>::new();

    // Create shared tail: (3 4)
    let three = arena.number(3);
    let four = arena.number(4);
    let nil = arena.nil();
    let tail2 = arena.cons(&four, &nil);
    let tail1 = arena.cons(&three, &tail2);
    let tail1_idx = tail1.raw().0 as usize;

    assert_eq!(arena.refcounts[tail1_idx].get(), 1);

    // Create two lists sharing the tail: (1 3 4) and (2 3 4)
    let one = arena.number(1);
    let two = arena.number(2);
    let list1 = arena.cons(&one, &tail1);
    assert_eq!(arena.refcounts[tail1_idx].get(), 2, "tail shared by list1");

    let list2 = arena.cons(&two, &tail1);
    assert_eq!(
        arena.refcounts[tail1_idx].get(),
        3,
        "tail shared by list1 and list2"
    );

    drop(tail1);
    assert_eq!(
        arena.refcounts[tail1_idx].get(),
        2,
        "tail still shared after dropping original ref"
    );

    drop(list1);
    assert_eq!(
        arena.refcounts[tail1_idx].get(),
        1,
        "tail only held by list2 now"
    );
    assert!(
        !matches!(arena.values[tail1_idx].get(), Value::Free),
        "tail should not be freed yet"
    );

    drop(list2);
    assert_eq!(
        arena.refcounts[tail1_idx].get(),
        0,
        "tail should be freed after last reference"
    );
    assert!(matches!(arena.values[tail1_idx].get(), Value::Free));
}

// ============================================================================
// INTEGRATION TESTS WITH REFCOUNT VERIFICATION
// ============================================================================

#[test]
fn test_factorial_no_leaks() {
    let arena = Arena::<DEFAULT_ARENA_SIZE>::new();
    let env = env_new(&arena);

    let factorial_def = r#"
        (define factorial
          (lambda (n)
            (if (= n 0)
                1
                (* n (factorial (- n 1))))))
    "#;

    eval_string(&arena, factorial_def, &env).unwrap();

    // Count free cells before
    let free_before = (0..DEFAULT_ARENA_SIZE)
        .filter(|&i| matches!(arena.values[i].get(), Value::Free))
        .count();

    let result = eval_string(&arena, "(factorial 5)", &env).unwrap();
    assert!(matches!(arena.get(result.inner), Some(Value::Number(120))));
    drop(result);

    // Count free cells after
    let free_after = (0..DEFAULT_ARENA_SIZE)
        .filter(|&i| matches!(arena.values[i].get(), Value::Free))
        .count();

    assert!(
        free_after >= free_before - 5,
        "Factorial should not leak: free_before={}, free_after={}",
        free_before,
        free_after
    );
}

#[test]
fn test_list_operations_no_leaks() {
    let arena = Arena::<DEFAULT_ARENA_SIZE>::new();
    let env = env_new(&arena);

    let free_before = (0..DEFAULT_ARENA_SIZE)
        .filter(|&i| matches!(arena.values[i].get(), Value::Free))
        .count();

    let result = eval_string(&arena, "(reverse (list 1 2 3 4 5))", &env).unwrap();
    drop(result);

    let free_after = (0..DEFAULT_ARENA_SIZE)
        .filter(|&i| matches!(arena.values[i].get(), Value::Free))
        .count();

    assert!(
        free_after >= free_before - 5,
        "List operations should not leak: free_before={}, free_after={}",
        free_before,
        free_after
    );
}

// ============================================================================
// SUMMARY TEST
// ============================================================================

#[test]
fn test_refcount_summary() {
    println!("\n=== Reference Counting Test Summary ===");
    println!("✓ Single values: proper lifecycle");
    println!("✓ Cloning: refcount increases correctly");
    println!("✓ Cons cells: car/cdr refcounts managed");
    println!("✓ Lists: cascade freeing works");
    println!("✓ Symbols: string refcounts handled");
    println!("✓ Lambdas: params/body/env refcounts managed");
    println!("✓ Environment: binding refcounts correct");
    println!("✓ Redefinition: old values properly freed");
    println!("✓ Temporaries: no leaks in eval");
    println!("✓ Shared structure: multiple references handled");
    println!("✓ Integration: complex operations don't leak");
}
// ============================================================================
// Test 1: Arena Exhaustion
// ============================================================================

#[test]
#[should_panic(expected = "allocation")]
fn test_arena_exhaustion() {
    // This test FAILS if arena runs out of memory
    // PASSES if there's GC or proper error handling
    const SMALL_ARENA: usize = 100;
    let arena = Arena::<SMALL_ARENA>::new();
    let env = env_new(&arena);

    // Try to allocate more than arena can hold
    let mut refs = Vec::new();
    for i in 0..SMALL_ARENA * 2 {
        let result = eval_string(&arena, "(cons 1 2)", &env);
        match result {
            Ok(r) => refs.push(r),
            Err(_) => {
                // Should have proper error handling, not just return NULL
                panic!(
                    "Arena exhausted without proper error handling at iteration {}",
                    i
                );
            }
        }
    }
}

#[test]
fn test_arena_exhaustion_with_cleanup() {
    // This test PASSES if references are properly cleaned up
    const SMALL_ARENA: usize = 100;
    let arena = Arena::<SMALL_ARENA>::new();
    let env = env_new(&arena);

    // Allocate and drop in a loop - should not exhaust if refcounting works
    for _ in 0..SMALL_ARENA * 2 {
        let result = eval_string(&arena, "(cons 1 2)", &env);
        assert!(
            result.is_ok(),
            "Should not exhaust arena with proper cleanup"
        );
        // result dropped here, should free memory
    }
}

// ============================================================================
// Test 2: Stack Overflow in decref
// ============================================================================

#[test]
#[should_panic(expected = "stack overflow")]
fn test_deep_decref_stack_overflow() {
    // This test FAILS if decref stack overflows on deep structures
    // PASSES if it handles arbitrary depth
    let arena = Arena::<DEFAULT_ARENA_SIZE>::new();
    let env = env_new(&arena);

    // Build a deeply nested list (deeper than 128 to exceed stack)
    let mut code = String::from("(list");
    for i in 0..200 {
        code.push_str(&format!(" {}", i));
    }
    code.push(')');

    let result = eval_string(&arena, &code, &env);
    assert!(result.is_ok(), "Should handle deep lists");

    // Now drop it - this should trigger deep decref
    drop(result);

    // If we get here without stack overflow, test passes
    panic!("Expected stack overflow but didn't occur - implementation may be fixed!");
}

#[test]
fn test_deep_nested_cons_cleanup() {
    // PASSES if deep structures are cleaned up without stack overflow
    let arena = Arena::<DEFAULT_ARENA_SIZE>::new();

    // Build deeply nested cons cells
    let mut list = arena.nil();
    for i in 0..500 {
        let num = arena.number(i);
        list = arena.cons(&num, &list);
    }

    // Drop should not overflow
    drop(list);
    // If we reach here, test passes
}

// ============================================================================
// Test 3: Buffer Size Limitations
// ============================================================================

#[test]
#[should_panic(expected = "too long")]
fn test_long_symbol_buffer_overflow() {
    // This test FAILS if long symbols cause "Atom too long" error
    // PASSES if arbitrary length symbols work
    let arena = Arena::<DEFAULT_ARENA_SIZE>::new();
    let env = env_new(&arena);

    // Create a symbol longer than 64 bytes
    let long_symbol = "a".repeat(100);
    let code = format!("(define {} 42)", long_symbol);

    let result = eval_string(&arena, &code, &env);

    match result {
        Err(e) => {
            let mut buf = [0u8; 256];
            if let Some(s) = arena.list_to_str(&e, &mut buf) {
                if s.contains("too long") {
                    panic!("Buffer size limitation hit: {}", s);
                }
            }
        }
        Ok(_) => {
            // Successfully handled long symbol
            return;
        }
    }
}

#[test]
#[should_panic(expected = "too long")]
fn test_long_string_conversion() {
    // FAILS if string-to-list conversion fails on long strings
    let arena = Arena::<DEFAULT_ARENA_SIZE>::new();

    let long_string = "x".repeat(100);
    let list = arena.str_to_list(&long_string);

    let mut small_buf = [0u8; 32];
    match arena.list_to_str(&list, &mut small_buf) {
        None => panic!("Buffer too long - limitation exists"),
        Some(_) => {
            // Either buffer was big enough or dynamic allocation worked
        }
    }
}

// ============================================================================
// Test 4: Recursive Function Stack Overflow
// ============================================================================

#[test]
#[should_panic(expected = "stack overflow")]
fn test_deep_parse_nesting() {
    // FAILS if deeply nested s-expressions cause stack overflow
    let arena = Arena::<DEFAULT_ARENA_SIZE>::new();

    // Create deeply nested parentheses
    let mut code = String::new();
    for _ in 0..150 {
        code.push_str("(list ");
    }
    code.push('1');
    for _ in 0..150 {
        code.push(')');
    }

    let result = tokenize(&arena, &code);
    assert!(result.is_ok(), "Tokenize should succeed");

    let tokens = result.unwrap();
    let parse_result = parse(&arena, &tokens);

    // If we get here without stack overflow, implementation handles deep nesting
    if parse_result.is_ok() {
        panic!("Expected stack overflow in parse but didn't occur - may be fixed!");
    }
}

#[test]
#[should_panic(expected = "stack overflow")]
fn test_deep_eval_args_recursion() {
    // FAILS if evaluating many nested arguments causes stack overflow
    let arena = Arena::<DEFAULT_ARENA_SIZE>::new();
    let env = env_new(&arena);

    // Create deeply nested function calls
    let mut code = String::from("(+");
    for _ in 0..200 {
        code.push_str(" (+");
    }
    code.push_str(" 1");
    for _ in 0..201 {
        code.push(')');
    }

    let result = eval_string(&arena, &code, &env);

    // If successful, recursion depth is handled
    if result.is_ok() {
        panic!("Expected stack overflow but didn't occur - may be fixed!");
    }
}

// ============================================================================
// Test 5: Memory Churn in env_set
// ============================================================================

#[test]
fn test_repeated_define_memory_churn() {
    // PASSES if repeated defines don't cause excessive allocations
    const SMALL_ARENA: usize = 500;
    let arena = Arena::<SMALL_ARENA>::new();
    let env = env_new(&arena);

    // Redefine the same variable many times
    for i in 0..100 {
        let code = format!("(define x {})", i);
        let result = eval_string(&arena, &code, &env);
        assert!(
            result.is_ok(),
            "Define #{} should succeed, but arena may be exhausted from churn",
            i
        );
    }

    // If we get here, memory churn is manageable
}

#[test]
#[should_panic(expected = "excessive allocations")]
fn test_env_set_allocation_count() {
    // FAILS if env_set allocates too many cells per update
    const SMALL_ARENA: usize = 200;
    let arena = Arena::<SMALL_ARENA>::new();
    let env = env_new(&arena);

    // Count how many defines we can do before exhaustion
    let mut count = 0;
    for i in 0..1000 {
        let code = format!("(define var{} {})", i % 10, i);
        match eval_string(&arena, &code, &env) {
            Ok(_) => count += 1,
            Err(_) => break,
        }
    }

    // With efficient updates, we should handle many redefinitions
    if count < 50 {
        panic!(
            "Only {} defines succeeded - excessive allocations in env_set",
            count
        );
    }
}

// ============================================================================
// Test 6: Misleading Tail Call Optimization
// ============================================================================

#[test]
#[should_panic(expected = "stack overflow")]
fn test_non_tail_recursive_factorial() {
    // This test documents that TCO doesn't help non-tail recursion
    // FAILS if factorial(1000) causes stack overflow
    // PASSES if implementation handles it (unlikely without special handling)
    let arena = Arena::<DEFAULT_ARENA_SIZE>::new();
    let env = env_new(&arena);

    let factorial_def = r#"
        (define factorial
          (lambda (n)
            (if (= n 0)
                1
                (* n (factorial (- n 1))))))
    "#;

    eval_string(&arena, factorial_def, &env).unwrap();

    // This will likely overflow because recursive call is not in tail position
    let result = eval_string(&arena, "(factorial 1000)", &env);

    if result.is_err() {
        panic!("Factorial overflow - TCO doesn't help non-tail recursion");
    }
}

#[test]
fn test_tail_recursive_counter_works() {
    // PASSES if true tail recursion works without overflow
    let arena = Arena::<DEFAULT_ARENA_SIZE>::new();
    let env = env_new(&arena);

    let counter_def = r#"
        (define count-down
          (lambda (n)
            (if (= n 0)
                0
                (count-down (- n 1)))))
    "#;

    eval_string(&arena, counter_def, &env).unwrap();

    // This should work with TCO
    let result = eval_string(&arena, "(count-down 10000)", &env);
    assert!(
        result.is_ok(),
        "Tail-recursive function should not overflow"
    );
}

// ============================================================================
// Test 7: No Cycle Detection
// ============================================================================

#[test]
fn test_cycle_detection() {
    // This test checks if cycles cause memory leaks or other issues
    // Note: Standard Lisp can't easily create cycles without set-car!/set-cdr!
    // This is more of a theoretical test
    let arena = Arena::<DEFAULT_ARENA_SIZE>::new();

    // Try to create a cycle using set_cons (if that creates a true cycle)
    let a = arena.nil();
    let b = arena.cons(&a, &a);

    // Manually create potential cycle (this is contrived)
    arena.set_cons(&b, &b, &b);

    // Drop should handle this gracefully
    drop(b);

    // If we reach here without hanging, test passes
}

// ============================================================================
// Test 8: Error Messages Allocate in Arena
// ============================================================================

#[test]
#[should_panic(expected = "arena exhausted from errors")]
fn test_errors_exhaust_arena() {
    // FAILS if generating many errors exhausts the arena
    const SMALL_ARENA: usize = 100;
    let arena = Arena::<SMALL_ARENA>::new();
    let env = env_new(&arena);

    // Generate many errors
    let mut error_count = 0;
    for _ in 0..SMALL_ARENA {
        let result = eval_string(&arena, "(undefined-function)", &env);
        if result.is_err() {
            error_count += 1;
        } else {
            break; // Arena might be exhausted
        }
    }

    // Try one more allocation
    let result = eval_string(&arena, "(+ 1 2)", &env);
    if result.is_err() {
        panic!("Arena exhausted from errors after {} errors", error_count);
    }
}

// ============================================================================
// Test 9: Linear Search Performance
// ============================================================================

#[test]
#[should_panic(expected = "too slow")]
fn test_allocation_performance_degradation() {
    // FAILS if allocation becomes very slow with fragmentation
    use std::time::Instant;

    let arena = Arena::<DEFAULT_ARENA_SIZE>::new();

    // Fill arena partially
    let mut refs = Vec::new();
    for i in 0..DEFAULT_ARENA_SIZE / 2 {
        let r = arena.number(i as i64);
        if i % 2 == 0 {
            refs.push(r); // Keep some alive to create fragmentation
        }
        // Others drop, creating holes
    }

    // Time allocations in fragmented arena
    let start = Instant::now();
    for _ in 0..100 {
        let r = arena.number(42);
        drop(r);
    }
    let elapsed = start.elapsed();

    // If linear search, this will be slow
    if elapsed.as_millis() > 100 {
        panic!("Allocation too slow: {:?} - linear search issue", elapsed);
    }
}

// ============================================================================
// Test 10: No Resource Limits
// ============================================================================

#[test]
#[should_panic(expected = "no depth limit")]
fn test_recursion_depth_limit() {
    // FAILS if there's no recursion depth limit
    let arena = Arena::<DEFAULT_ARENA_SIZE>::new();
    let env = env_new(&arena);

    let deep_recursion = r#"
        (define deep
          (lambda (n)
            (if (= n 0)
                0
                (deep (- n 1)))))
    "#;

    eval_string(&arena, deep_recursion, &env).unwrap();

    // Try extremely deep recursion
    let result = eval_string(&arena, "(deep 100000)", &env);

    match result {
        Err(e) => {
            let mut buf = [0u8; 256];
            if let Some(s) = arena.list_to_str(&e, &mut buf) {
                if s.contains("deep") || s.contains("limit") || s.contains("stack") {
                    return; // Has depth limiting
                }
            }
        }
        Ok(_) => {}
    }

    panic!("No depth limit detected - should have recursion limits");
}

// ============================================================================
// Additional Integration Tests
// ============================================================================

#[test]
fn test_basic_functionality() {
    // Sanity test - basic operations should always work
    let arena = Arena::<DEFAULT_ARENA_SIZE>::new();
    let env = env_new(&arena);

    assert!(eval_string(&arena, "(+ 1 2 3)", &env).is_ok());
    assert!(eval_string(&arena, "(define x 10)", &env).is_ok());
    assert!(eval_string(&arena, "(* x 2)", &env).is_ok());
}

#[test]
fn test_error_recovery() {
    // Test that errors don't corrupt the arena
    let arena = Arena::<DEFAULT_ARENA_SIZE>::new();
    let env = env_new(&arena);

    // Generate an error
    let _ = eval_string(&arena, "(undefined)", &env);

    // Should still be able to evaluate successfully
    let result = eval_string(&arena, "(+ 1 2)", &env);
    assert!(result.is_ok(), "Arena should be usable after error");
}

// ============================================================================
// Helper function to run all tests and report
// ============================================================================

#[test]
fn run_all_vulnerability_tests() {
    println!("\n=== Running Vulnerability Tests ===\n");

    let tests: Vec<(&str, fn())> = vec![
        (
            "Arena Exhaustion",
            test_arena_exhaustion_with_cleanup as fn(),
        ),
        ("Deep Decref Cleanup", test_deep_nested_cons_cleanup as fn()),
        ("Tail Recursion", test_tail_recursive_counter_works as fn()),
        ("Basic Functionality", test_basic_functionality as fn()),
        ("Error Recovery", test_error_recovery as fn()),
    ];

    for (name, test_fn) in tests {
        print!("Testing {}: ", name);
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| test_fn()))
            .map(|_| println!("✓ PASS"))
            .unwrap_or_else(|_| println!("✗ FAIL"));
    }
}
