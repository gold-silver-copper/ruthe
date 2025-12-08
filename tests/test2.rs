#![cfg(test)]
extern crate alloc;
use alloc::rc::Rc;
use ruthe::*;

// ============================================================================
// Stress Test Helpers
// ============================================================================

fn get_env_refcount(env: &EnvRef) -> usize {
    Rc::strong_count(&env.0)
}

// ============================================================================
// Extreme Recursion Tests - Testing Stack Safety & Memory
// ============================================================================

#[test]
fn test_extreme_tail_recursion() {
    // Test that tail call optimization truly works without stack overflow
    let program = r#"
        (define countdown
          (lambda (n acc)
            (if (= n 0)
                acc
                (countdown (- n 1) (+ acc 1)))))
        (countdown 10000 0)
    "#;

    let env = EnvRef::new();
    let initial_count = get_env_refcount(&env);

    let result = eval_str_multiple(program, &env).map(|s| {
        let mut buf = [0u8; 256];
        s.to_display_str(&mut buf).unwrap().to_string()
    });

    assert_eq!(result, Ok("10000".to_string()));

    let final_count = get_env_refcount(&env);
    assert!(
        final_count <= initial_count + 1,
        "Extreme recursion leaked: {} -> {}",
        initial_count,
        final_count
    );
}

#[test]
fn test_very_deep_tail_recursion() {
    // Push tail call optimization to extreme limits
    let program = r#"
        (define sum-to
          (lambda (n acc)
            (if (= n 0)
                acc
                (sum-to (- n 1) (+ acc n)))))
        (sum-to 50000 0)
    "#;

    let env = EnvRef::new();
    let result = eval_str_multiple(program, &env);

    // Should complete without stack overflow
    assert!(result.is_ok(), "Deep tail recursion failed: {:?}", result);
}

#[test]
fn test_mutual_recursion_deep() {
    // Test deep mutual recursion doesn't leak
    let program = r#"
        (define is-even
          (lambda (n)
            (if (= n 0)
                #t
                (is-odd (- n 1)))))
        (define is-odd
          (lambda (n)
            (if (= n 0)
                #f
                (is-even (- n 1)))))
        (is-even 5000)
    "#;

    let env = EnvRef::new();
    let initial_count = get_env_refcount(&env);

    let result = eval_str_multiple(program, &env);
    assert!(result.is_ok());

    let final_count = get_env_refcount(&env);
    assert!(
        final_count <= initial_count + 2,
        "Mutual recursion leaked: {} -> {}",
        initial_count,
        final_count
    );
}

// ============================================================================
// Environment Chain Stress Tests
// ============================================================================

#[test]
fn test_deeply_nested_scopes() {
    // Create deeply nested lambda scopes - each creates a new environment
    let program = r#"
        (define nest
          (lambda (n)
            (if (= n 0)
                42
                ((lambda (x) (nest (- n 1))) n))))
        (nest 100)
    "#;

    let env = EnvRef::new();
    let initial_count = get_env_refcount(&env);

    let result = eval_str_multiple(program, &env);
    assert_eq!(
        result.map(|s| {
            let mut buf = [0u8; 256];
            s.to_display_str(&mut buf).unwrap().to_string()
        }),
        Ok("42".to_string())
    );

    let final_count = get_env_refcount(&env);
    assert!(
        final_count <= initial_count + 1,
        "Nested scopes leaked: {} -> {}",
        initial_count,
        final_count
    );
}

#[test]
fn test_multiple_closures_same_env() {
    // Multiple closures capturing the same environment
    let program = r#"
        (define x 100)
        (define f1 (lambda () x))
        (define f2 (lambda () (+ x 1)))
        (define f3 (lambda () (* x 2)))
        (+ (f1) (f2) (f3))
    "#;

    let env = EnvRef::new();
    let initial_count = get_env_refcount(&env);

    let result = eval_str_multiple(program, &env);
    assert_eq!(
        result.map(|s| {
            let mut buf = [0u8; 256];
            s.to_display_str(&mut buf).unwrap().to_string()
        }),
        Ok("401".to_string())
    );

    let final_count = get_env_refcount(&env);
    // 3 lambdas all reference the same env
    assert!(
        final_count <= initial_count + 3,
        "Multiple closures leaked: {} -> {}",
        initial_count,
        final_count
    );
}

#[test]
fn test_chain_of_environments() {
    // Test that parent environment chains don't leak
    let program = r#"
        (define make-adder
          (lambda (x)
            (lambda (y)
              (lambda (z)
                (+ x (+ y z))))))
        (define add5 (make-adder 5))
        (define add5and10 (add5 10))
        (add5and10 20)
    "#;

    let env = EnvRef::new();
    let initial_count = get_env_refcount(&env);

    let result = eval_str_multiple(program, &env);
    assert_eq!(
        result.map(|s| {
            let mut buf = [0u8; 256];
            s.to_display_str(&mut buf).unwrap().to_string()
        }),
        Ok("35".to_string())
    );

    let final_count = get_env_refcount(&env);
    assert!(
        final_count <= initial_count + 3,
        "Environment chain leaked: {} -> {}",
        initial_count,
        final_count
    );
}

// ============================================================================
// Massive Data Structure Tests
// ============================================================================

#[test]
fn test_huge_list_creation_and_destruction() {
    // Create a very large list and ensure it's cleaned up
    let program = r#"
        (define build-big-list
          (lambda (n acc)
            (if (= n 0)
                acc
                (build-big-list (- n 1) (cons n acc)))))
        (length (build-big-list 1000 nil))
    "#;

    let env = EnvRef::new();
    let initial_count = get_env_refcount(&env);

    let result = eval_str_multiple(program, &env);
    assert_eq!(
        result.map(|s| {
            let mut buf = [0u8; 256];
            s.to_display_str(&mut buf).unwrap().to_string()
        }),
        Ok("1000".to_string())
    );

    let final_count = get_env_refcount(&env);
    assert!(
        final_count <= initial_count + 1,
        "Huge list leaked: {} -> {}",
        initial_count,
        final_count
    );
}

#[test]
fn test_repeated_large_allocations() {
    // Repeatedly create and destroy large structures
    let env = EnvRef::new();
    eval_str(
        "(define make-list (lambda (n) (if (= n 0) nil (cons n (make-list (- n 1))))))",
        &env,
    )
    .unwrap();

    let count_after_define = get_env_refcount(&env);

    // Create large lists multiple times
    for _ in 0..10 {
        eval_str("(length (make-list 500))", &env).unwrap();
    }

    let final_count = get_env_refcount(&env);

    // Count should not grow with each iteration
    assert_eq!(
        count_after_define, final_count,
        "Repeated allocations leaked: {} -> {}",
        count_after_define, final_count
    );
}

#[test]
fn test_append_large_lists() {
    // Appending large lists shouldn't leak
    let program = r#"
        (define range
          (lambda (n acc)
            (if (= n 0)
                acc
                (range (- n 1) (cons n acc)))))
        (define list1 (range 500 nil))
        (define list2 (range 500 nil))
        (length (append list1 list2))
    "#;

    let env = EnvRef::new();
    let initial_count = get_env_refcount(&env);

    let result = eval_str_multiple(program, &env);
    assert_eq!(
        result.map(|s| {
            let mut buf = [0u8; 256];
            s.to_display_str(&mut buf).unwrap().to_string()
        }),
        Ok("1000".to_string())
    );

    let final_count = get_env_refcount(&env);
    assert!(
        final_count <= initial_count + 3,
        "Append large lists leaked: {} -> {}",
        initial_count,
        final_count
    );
}

// ============================================================================
// Repeated Operations Tests
// ============================================================================

#[test]
fn test_repeated_lambda_creation() {
    // Creating lambdas repeatedly shouldn't accumulate references
    let env = EnvRef::new();

    eval_str("(define make-fn (lambda (n) (lambda (x) (+ x n))))", &env).unwrap();
    let initial_count = get_env_refcount(&env);

    for i in 0..100 {
        let program = format!("((make-fn {}) 10)", i);
        eval_str(&program, &env).unwrap();
    }

    let final_count = get_env_refcount(&env);

    // Should be stable, not growing
    assert_eq!(
        initial_count, final_count,
        "Repeated lambda creation leaked: {} -> {}",
        initial_count, final_count
    );
}

#[test]
fn test_repeated_recursive_calls() {
    // Calling recursive functions repeatedly shouldn't leak
    let env = EnvRef::new();

    eval_str(
        "(define fib (lambda (n) (if (< n 2) n (+ (fib (- n 1)) (fib (- n 2))))))",
        &env,
    )
    .unwrap();
    let initial_count = get_env_refcount(&env);

    // Call multiple times
    for _ in 0..10 {
        eval_str("(fib 10)", &env).unwrap();
    }

    let final_count = get_env_refcount(&env);
    assert_eq!(
        initial_count, final_count,
        "Repeated recursive calls leaked: {} -> {}",
        initial_count, final_count
    );
}

#[test]
fn test_repeated_definitions() {
    // Redefining variables shouldn't leak (old values should be dropped)
    let env = EnvRef::new();
    let initial_count = get_env_refcount(&env);

    for i in 0..100 {
        let program = format!("(define x {})", i);
        eval_str(&program, &env).unwrap();
    }

    let final_count = get_env_refcount(&env);

    // Note: Each define adds a new binding to the front of the list
    // This is expected behavior - we're checking it doesn't grow unboundedly
    // The growth should be linear with definitions, not exponential
    let growth = final_count - initial_count;
    assert!(
        growth < 5,
        "Repeated definitions grew too much: {} -> {} (growth: {})",
        initial_count,
        final_count,
        growth
    );
}

// ============================================================================
// Error Path Memory Tests
// ============================================================================

#[test]
fn test_errors_dont_leak_during_evaluation() {
    let env = EnvRef::new();
    let initial_count = get_env_refcount(&env);

    // Try many operations that will fail
    for _ in 0..50 {
        let _ = eval_str("(/ 10 0)", &env); // Division by zero
        let _ = eval_str("undefined-var-xyz", &env); // Unbound variable
        let _ = eval_str("(+ 1 #t)", &env); // Type error
        let _ = eval_str("(car 5)", &env); // Type error
        let _ = eval_str("((lambda (x) x) 1 2 3)", &env); // Arity error
    }

    let final_count = get_env_refcount(&env);
    assert_eq!(
        initial_count, final_count,
        "Error paths leaked: {} -> {}",
        initial_count, final_count
    );
}

#[test]
fn test_errors_in_deeply_nested_calls() {
    let program = r#"
        (define deep-error
          (lambda (n)
            (if (= n 0)
                (/ 1 0)
                (deep-error (- n 1)))))
    "#;

    let env = EnvRef::new();
    eval_str_multiple(program, &env).unwrap();

    let initial_count = get_env_refcount(&env);

    // Call it and let it error deep in the stack
    for _ in 0..20 {
        let _ = eval_str("(deep-error 100)", &env);
    }

    let final_count = get_env_refcount(&env);
    assert_eq!(
        initial_count, final_count,
        "Deep error calls leaked: {} -> {}",
        initial_count, final_count
    );
}

#[test]
fn test_partial_evaluation_errors() {
    // Errors during argument evaluation shouldn't leak
    let env = EnvRef::new();
    let initial_count = get_env_refcount(&env);

    for _ in 0..50 {
        let _ = eval_str("(+ 1 (/ 10 0) 3)", &env); // Error in middle of args
        let _ = eval_str("(list 1 undefined-var 3)", &env); // Error in list
    }

    let final_count = get_env_refcount(&env);
    assert_eq!(
        initial_count, final_count,
        "Partial evaluation errors leaked: {} -> {}",
        initial_count, final_count
    );
}

// ============================================================================
// ValRef Reference Counting Tests
// ============================================================================

#[test]
fn test_valref_shared_substructure() {
    // Test that shared substructures are reference counted correctly
    let env = EnvRef::new();

    let program = r#"
        (define shared-list (list 1 2 3 4 5))
        (define list1 (cons 0 shared-list))
        (define list2 (cons 10 shared-list))
        (+ (length list1) (length list2))
    "#;

    let initial_count = get_env_refcount(&env);
    let result = eval_str_multiple(program, &env);
    assert_eq!(
        result.map(|s| {
            let mut buf = [0u8; 256];
            s.to_display_str(&mut buf).unwrap().to_string()
        }),
        Ok("12".to_string())
    );

    let final_count = get_env_refcount(&env);
    assert!(
        final_count <= initial_count + 1,
        "Shared substructure leaked: {} -> {}",
        initial_count,
        final_count
    );
}

#[test]
fn test_valref_clone_correctness() {
    // Verify that ValRef cloning works correctly
    let val1 = ValRef::number(42);
    let count1 = Rc::strong_count(&val1.0);
    assert_eq!(count1, 1);

    let val2 = val1.clone();
    let count2 = Rc::strong_count(&val1.0);
    assert_eq!(count2, 2);

    let val3 = val1.clone();
    let count3 = Rc::strong_count(&val1.0);
    assert_eq!(count3, 3);

    drop(val2);
    let count4 = Rc::strong_count(&val1.0);
    assert_eq!(count4, 2);

    drop(val3);
    let count5 = Rc::strong_count(&val1.0);
    assert_eq!(count5, 1);
}

#[test]
fn test_cons_cell_refcount() {
    // Test reference counting in cons cells
    let head = ValRef::number(1);
    let head_count = Rc::strong_count(&head.0);
    assert_eq!(head_count, 1);

    let tail = ValRef::number(2);
    let cons = ValRef::cons(head.clone(), tail.clone());

    // head and tail should have 2 refs each now
    let head_count = Rc::strong_count(&head.0);
    assert_eq!(head_count, 2);

    let tail_count = Rc::strong_count(&tail.0);
    assert_eq!(tail_count, 2);

    drop(cons);

    // Should be back to 1
    let head_count = Rc::strong_count(&head.0);
    assert_eq!(head_count, 1);

    let tail_count = Rc::strong_count(&tail.0);
    assert_eq!(tail_count, 1);
}

// ============================================================================
// Environment Parent Chain Tests
// ============================================================================

#[test]
fn test_environment_parent_chain_cleanup() {
    // Create a chain of environments and verify cleanup
    let env1 = EnvRef::new();
    let count1 = get_env_refcount(&env1);

    let env2 = EnvRef::with_parent(env1.clone());
    let count2 = get_env_refcount(&env1);
    assert_eq!(count2, count1 + 1, "Parent ref not added");

    let env3 = EnvRef::with_parent(env2.clone());
    let count3 = get_env_refcount(&env2);
    assert!(count3 > 1, "Child->parent ref not working");

    drop(env3);
    let count4 = get_env_refcount(&env2);
    // Should decrease

    drop(env2);
    let count5 = get_env_refcount(&env1);
    assert_eq!(
        count5, count1,
        "Environment chain didn't clean up: {} -> {}",
        count1, count5
    );
}

#[test]
fn test_lambda_environment_capture() {
    // Test that lambdas properly capture and release environments
    let env = EnvRef::new();
    let initial_count = get_env_refcount(&env);

    let program = "(lambda (x) (+ x 1))";
    let lambda_result = parse(program).unwrap();
    let evaluated = eval(lambda_result, &env).unwrap();

    // Lambda should have captured the env
    let after_eval_count = get_env_refcount(&env);
    assert!(
        after_eval_count > initial_count,
        "Lambda didn't capture env"
    );

    drop(evaluated);

    // Should be back to initial
    let final_count = get_env_refcount(&env);
    assert_eq!(
        final_count, initial_count,
        "Lambda didn't release env: {} -> {}",
        initial_count, final_count
    );
}

// ============================================================================
// Stress Test: Combined Operations
// ============================================================================

#[test]
fn test_kitchen_sink_stress() {
    // Combine many operations in one test
    let program = r#"
        (define fact (lambda (n acc) 
          (if (= n 0) acc (fact (- n 1) (* n acc)))))
        
        (define fib (lambda (n)
          (if (< n 2) n (+ (fib (- n 1)) (fib (- n 2))))))
        
        (define map (lambda (f lst)
          (if (null? lst)
              nil
              (cons (f (car lst)) (map f (cdr lst))))))
        
        (define range (lambda (n acc)
          (if (= n 0) acc (range (- n 1) (cons n acc)))))
        
        (define double (lambda (x) (* x 2)))
        
        (define numbers (range 20 nil))
        (define doubled (map double numbers))
        (+ (fact 10 1) (fib 10) (length doubled))
    "#;

    let env = EnvRef::new();
    let initial_count = get_env_refcount(&env);

    let result = eval_str_multiple(program, &env);
    assert!(result.is_ok(), "Kitchen sink failed: {:?}", result);

    let final_count = get_env_refcount(&env);
    let growth = final_count - initial_count;

    // Should have stable growth (one per defined function)
    assert!(
        growth <= 6,
        "Kitchen sink leaked too much: {} -> {} (growth: {})",
        initial_count,
        final_count,
        growth
    );
}

#[test]
fn test_stress_many_small_operations() {
    // Many small operations shouldn't accumulate memory
    let env = EnvRef::new();
    let initial_count = get_env_refcount(&env);

    for i in 0..1000 {
        let _ = eval_str(&format!("(+ {} {})", i, i + 1), &env);
        let _ = eval_str(&format!("(* {} {})", i, 2), &env);
        let _ = eval_str(&format!("(< {} {})", i, 500), &env);
    }

    let final_count = get_env_refcount(&env);
    assert_eq!(
        initial_count, final_count,
        "Many small operations leaked: {} -> {}",
        initial_count, final_count
    );
}

#[test]
fn test_stress_alternating_success_and_error() {
    // Alternating between success and errors
    let env = EnvRef::new();
    eval_str(
        "(define safe-div (lambda (a b) (if (= b 0) 0 (/ a b))))",
        &env,
    )
    .unwrap();

    let initial_count = get_env_refcount(&env);

    for i in 0..100 {
        if i % 2 == 0 {
            eval_str(&format!("(safe-div {} 2)", i), &env).unwrap();
        } else {
            let _ = eval_str("undefined-variable", &env);
        }
    }

    let final_count = get_env_refcount(&env);
    assert_eq!(
        initial_count, final_count,
        "Alternating operations leaked: {} -> {}",
        initial_count, final_count
    );
}

// ============================================================================
// Edge Case Tests
// ============================================================================

#[test]
fn test_empty_environment_operations() {
    // Operations with empty/minimal state
    let env = EnvRef::new();
    let initial_count = get_env_refcount(&env);

    for _ in 0..100 {
        let _ = eval_str("42", &env);
        let _ = eval_str("#t", &env);
        let _ = eval_str("nil", &env);
    }

    let final_count = get_env_refcount(&env);
    assert_eq!(
        initial_count, final_count,
        "Empty operations leaked: {} -> {}",
        initial_count, final_count
    );
}

#[test]
fn test_immediate_drop_stress() {
    // Create and immediately drop many values
    for _ in 0..1000 {
        let env = EnvRef::new();
        let _ = eval_str("(+ 1 2)", &env);
        drop(env);
    }
    // If we get here without crashing, memory management is working
}

#[test]
fn test_nested_quote_structures() {
    // Quoted structures shouldn't leak
    let env = EnvRef::new();
    let initial_count = get_env_refcount(&env);

    for _ in 0..100 {
        let _ = eval_str("'(1 2 3 4 5)", &env);
        let _ = eval_str("'((1 2) (3 4) (5 6))", &env);
    }

    let final_count = get_env_refcount(&env);
    assert_eq!(
        initial_count, final_count,
        "Quoted structures leaked: {} -> {}",
        initial_count, final_count
    );
}

// ============================================================================
// Final Boss: Maximum Stress Test
// ============================================================================

#[test]
fn test_maximum_stress() {
    // The ultimate stress test - everything at once
    let program = r#"
        (define factorial (lambda (n acc)
          (if (= n 0) acc (factorial (- n 1) (* n acc)))))
        
        (define sum-range (lambda (n acc)
          (if (= n 0) acc (sum-range (- n 1) (+ acc n)))))
        
        (define build-list (lambda (n acc)
          (if (= n 0) acc (build-list (- n 1) (cons n acc)))))
        
        (define process-list (lambda (lst acc)
          (if (null? lst)
              acc
              (process-list (cdr lst) (+ acc (car lst))))))
    "#;

    let env = EnvRef::new();
    eval_str_multiple(program, &env).unwrap();
    let initial_count = get_env_refcount(&env);

    // Run many intensive operations
    for _ in 0..50 {
        let _ = eval_str("(factorial 20 1)", &env);
        let _ = eval_str("(sum-range 100 0)", &env);
        let _ = eval_str("(process-list (build-list 50 nil) 0)", &env);
    }

    let final_count = get_env_refcount(&env);
    assert_eq!(
        initial_count, final_count,
        "Maximum stress test leaked: {} -> {}",
        initial_count, final_count
    );
}
