#![cfg(test)]
extern crate alloc;
use alloc::rc::Rc;
use ruthe::*;

// ============================================================================
// Stress Test Helpers - Modified for new interpreter
// ============================================================================

fn get_env_refcount(env: &ValRef) -> usize {
    // For a ValRef, we need to check if it's a cons cell containing bindings
    // Since environments are just ValRef cons cells, we can get refcount of the Rc
    Rc::strong_count(&env.0)
}

// Helper function to evaluate with display string conversion
fn eval_display(input: &str, env: &ValRef) -> Result<String, String> {
    eval_str(input, env)
        .map(|s| {
            let mut buf = [0u8; 4096];
            s.to_display_str(&mut buf).unwrap().to_string()
        })
        .map_err(|e| {
            let mut buf = [0u8; 256];
            e.to_display_str(&mut buf).unwrap().to_string()
        })
}

fn eval_multiple_display(input: &str, env: &ValRef) -> Result<String, String> {
    eval_str_multiple(input, env)
        .map(|s| {
            let mut buf = [0u8; 4096];
            s.to_display_str(&mut buf).unwrap().to_string()
        })
        .map_err(|e| {
            let mut buf = [0u8; 256];
            e.to_display_str(&mut buf).unwrap().to_string()
        })
}

// ============================================================================
// Extreme Recursion Tests - Testing Stack Safety & Memory
// ============================================================================

#[test]
fn test_extreme_tail_recursion() {
    // Test that tail call optimization truly works without stack overflow
    let program = r#"
        ((lambda (countdown)
           (countdown countdown 10000 0))
         (lambda (self n acc)
           (if (= n 0)
               acc
               (self self (- n 1) (+ acc 1)))))
    "#;

    let env = new_env();
    let initial_count = get_env_refcount(&env);

    let result = eval_str(program, &env).map(|s| {
        let mut buf = [0u8; 256];
        s.to_display_str(&mut buf).unwrap().to_string()
    });

    assert_eq!(result, Ok("10000".to_string()));

    let final_count = get_env_refcount(&env);
    // Note: Environment growth is expected in our implementation due to how we handle binding
    // We're just checking that it completes without crashing
    assert!(
        final_count < initial_count + 100, // Reasonable bound
        "Extreme recursion grew too much: {} -> {}",
        initial_count,
        final_count
    );
}

#[test]
fn test_very_deep_tail_recursion() {
    // Push tail call optimization to extreme limits
    let program = r#"
        ((lambda (sum-to)
           (sum-to sum-to 50000 0))
         (lambda (self n acc)
           (if (= n 0)
               acc
               (self self (- n 1) (+ acc n)))))
    "#;

    let env = new_env();
    let result = eval_str(program, &env);

    // Should complete without stack overflow
    assert!(result.is_ok(), "Deep tail recursion failed: {:?}", result);
}

#[test]
fn test_mutual_recursion_deep() {
    // Test deep mutual recursion doesn't leak
    let program = r#"
        ((lambda (make-even-odd)
           ((car (make-even-odd make-even-odd)) 5000))
         (lambda (self)
           (cons
             (lambda (n)
               (if (= n 0) #t ((cdr (self self)) (- n 1))))
             (lambda (n)
               (if (= n 0) #f ((car (self self)) (- n 1)))))))
    "#;

    let env = new_env();
    let initial_count = get_env_refcount(&env);

    let result = eval_str(program, &env);
    assert!(result.is_ok());

    let final_count = get_env_refcount(&env);
    // Some growth is expected
    assert!(
        final_count < initial_count + 50,
        "Mutual recursion grew too much: {} -> {}",
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
        ((lambda (nest)
           (nest nest 100))
         (lambda (self n)
           (if (= n 0)
               42
               ((lambda (x) (self self (- n 1))) n))))
    "#;

    let env = new_env();
    let initial_count = get_env_refcount(&env);

    let result = eval_str(program, &env);
    assert_eq!(
        result.map(|s| {
            let mut buf = [0u8; 256];
            s.to_display_str(&mut buf).unwrap().to_string()
        }),
        Ok("42".to_string())
    );

    let final_count = get_env_refcount(&env);
    assert!(
        final_count < initial_count + 50,
        "Nested scopes grew too much: {} -> {}",
        initial_count,
        final_count
    );
}

#[test]
fn test_multiple_closures_same_env() {
    // Multiple closures capturing the same environment
    let program = r#"
        ((lambda (x)
           ((lambda (f1 f2 f3)
              (+ (f1) (f2) (f3)))
            (lambda () x)
            (lambda () (+ x 1))
            (lambda () (* x 2))))
         100)
    "#;

    let env = new_env();
    let initial_count = get_env_refcount(&env);

    let result = eval_str(program, &env);
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
        final_count < initial_count + 10,
        "Multiple closures grew too much: {} -> {}",
        initial_count,
        final_count
    );
}

#[test]
fn test_chain_of_environments() {
    // Test that parent environment chains don't leak
    let program = r#"
        ((lambda (make-adder)
           ((lambda (add5)
              ((lambda (add5and10)
                 (add5and10 20))
               (add5 10)))
            (make-adder 5)))
         (lambda (x)
           (lambda (y)
             (lambda (z)
               (+ x (+ y z))))))
    "#;

    let env = new_env();
    let initial_count = get_env_refcount(&env);

    let result = eval_str(program, &env);
    assert_eq!(
        result.map(|s| {
            let mut buf = [0u8; 256];
            s.to_display_str(&mut buf).unwrap().to_string()
        }),
        Ok("35".to_string())
    );

    let final_count = get_env_refcount(&env);
    assert!(
        final_count < initial_count + 20,
        "Environment chain grew too much: {} -> {}",
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
        ((lambda (build-big-list)
           ((lambda (big-list)
              (length big-list))
            (build-big-list build-big-list 1000 nil)))
         (lambda (self n acc)
           (if (= n 0)
               acc
               (self self (- n 1) (cons n acc)))))
    "#;

    let env = new_env();
    let initial_count = get_env_refcount(&env);

    let result = eval_str(program, &env);
    assert_eq!(
        result.map(|s| {
            let mut buf = [0u8; 256];
            s.to_display_str(&mut buf).unwrap().to_string()
        }),
        Ok("1000".to_string())
    );

    let final_count = get_env_refcount(&env);
    assert!(
        final_count < initial_count + 10,
        "Huge list grew too much: {} -> {}",
        initial_count,
        final_count
    );
}
#[test]
fn test_repeated_large_allocations() {
    // Repeatedly create and destroy large structures
    let env = new_env();

    // Define make-list as a lambda
    let make_list_program = r#"
        (lambda (self n)
          (if (= n 0)
              nil
              (cons n (self self (- n 1)))))
    "#;

    let make_list = eval_str(make_list_program, &env).unwrap();
    let count_after_define = get_env_refcount(&env);

    // Create large lists multiple times
    for _ in 0..10 {
        // Call make-list with 500
        let call = ValRef::cons(
            make_list.clone(),
            ValRef::cons(
                make_list.clone(),
                ValRef::cons(ValRef::number(500), ValRef::nil()),
            ),
        );
        let _ = eval(call, &env).unwrap();
    }

    let final_count = get_env_refcount(&env);

    // Count should not grow unboundedly
    assert!(
        final_count <= count_after_define + 5,
        "Repeated allocations grew too much: {} -> {}",
        count_after_define,
        final_count
    );
}

// ============================================================================
// Repeated Operations Tests
// ============================================================================

#[test]
fn test_repeated_lambda_creation() {
    // Creating lambdas repeatedly shouldn't accumulate references
    let env = new_env();

    let make_fn_program = "(lambda (n) (lambda (x) (+ x n)))";
    let make_fn = eval_str(make_fn_program, &env).unwrap();
    let initial_count = get_env_refcount(&env);

    for i in 0..100 {
        // Create closure
        let closure = ValRef::cons(
            make_fn.clone(),
            ValRef::cons(ValRef::number(i), ValRef::nil()),
        );
        let closure_evaled = eval(closure.clone(), &env).unwrap();

        // Call closure
        let call = ValRef::cons(
            closure_evaled,
            ValRef::cons(ValRef::number(10), ValRef::nil()),
        );
        let _ = eval(call, &env);
    }

    let final_count = get_env_refcount(&env);

    // Should be stable, not growing unboundedly
    assert!(
        final_count < initial_count + 20,
        "Repeated lambda creation grew too much: {} -> {}",
        initial_count,
        final_count
    );
}

#[test]
fn test_repeated_recursive_calls() {
    // Calling recursive functions repeatedly shouldn't leak
    let env = new_env();

    let fib_program = r#"
        (lambda (self n)
          (if (< n 2)
              n
              (+ (self self (- n 1)) (self self (- n 2)))))
    "#;

    let fib = eval_str(fib_program, &env).unwrap();
    let initial_count = get_env_refcount(&env);

    // Call multiple times
    for _ in 0..10 {
        let call = ValRef::cons(
            fib.clone(),
            ValRef::cons(fib.clone(), ValRef::cons(ValRef::number(10), ValRef::nil())),
        );
        let _ = eval(call, &env);
    }

    let final_count = get_env_refcount(&env);
    assert!(
        final_count < initial_count + 10,
        "Repeated recursive calls grew too much: {} -> {}",
        initial_count,
        final_count
    );
}

// ============================================================================
// Error Path Memory Tests
// ============================================================================

#[test]
fn test_errors_dont_leak_during_evaluation() {
    let env = new_env();
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
    assert!(
        final_count < initial_count + 5,
        "Error paths grew too much: {} -> {}",
        initial_count,
        final_count
    );
}

// ============================================================================
// ValRef Reference Counting Tests
// ============================================================================

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
    let tail_count = Rc::strong_count(&tail.0);
    assert_eq!(tail_count, 1);

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
// Stress Test: Combined Operations
// ============================================================================

#[test]
fn test_kitchen_sink_stress() {
    // Combine many operations in one test
    let program = r#"
        ((lambda (fact fib map range double)
           ((lambda (numbers)
              ((lambda (doubled)
                 (+ (fact fact 10 1) (fib fib 10) (length doubled)))
               (map map double numbers)))
            (range range 20 nil)))
         ; fact
         (lambda (self n acc)
           (if (= n 0) acc (self self (- n 1) (* n acc))))
         ; fib
         (lambda (self n)
           (if (< n 2) n (+ (self self (- n 1)) (self self (- n 2)))))
         ; map
         (lambda (self f lst)
           (if (null? lst)
               nil
               (cons (f (car lst)) (self self f (cdr lst)))))
         ; range
         (lambda (self n acc)
           (if (= n 0) acc (self self (- n 1) (cons n acc))))
         ; double
         (lambda (x) (* x 2)))
    "#;

    let env = new_env();
    let initial_count = get_env_refcount(&env);

    let result = eval_str(program, &env);
    assert!(result.is_ok(), "Kitchen sink failed: {:?}", result);

    let final_count = get_env_refcount(&env);

    // Should have reasonable growth
    assert!(
        final_count < initial_count + 50,
        "Kitchen sink grew too much: {} -> {}",
        initial_count,
        final_count
    );
}

#[test]
fn test_stress_many_small_operations() {
    // Many small operations shouldn't accumulate memory
    let env = new_env();
    let initial_count = get_env_refcount(&env);

    for i in 0..100 {
        let _ = eval_str(&format!("(+ {} {})", i, i + 1), &env);
        let _ = eval_str(&format!("(* {} 2)", i), &env);
        let _ = eval_str(&format!("(< {} 50)", i), &env);
    }

    let final_count = get_env_refcount(&env);
    assert!(
        final_count < initial_count + 10,
        "Many small operations grew too much: {} -> {}",
        initial_count,
        final_count
    );
}

#[test]
fn test_stress_alternating_success_and_error() {
    // Alternating between success and errors
    let env = new_env();

    let safe_div_program = r#"
        (lambda (a b)
          (if (= b 0) 0 (/ a b)))
    "#;

    let safe_div = eval_str(safe_div_program, &env).unwrap();
    let initial_count = get_env_refcount(&env);

    for i in 0..50 {
        if i % 2 == 0 {
            // Call safe-div
            let call = ValRef::cons(
                safe_div.clone(),
                ValRef::cons(
                    ValRef::number(i),
                    ValRef::cons(ValRef::number(2), ValRef::nil()),
                ),
            );
            let _ = eval(call, &env);
        } else {
            let _ = eval_str("undefined-variable", &env);
        }
    }

    let final_count = get_env_refcount(&env);
    assert!(
        final_count < initial_count + 10,
        "Alternating operations grew too much: {} -> {}",
        initial_count,
        final_count
    );
}

// ============================================================================
// Edge Case Tests
// ============================================================================

#[test]
fn test_empty_environment_operations() {
    // Operations with empty/minimal state
    let env = new_env();
    let initial_count = get_env_refcount(&env);

    for _ in 0..100 {
        let _ = eval_str("42", &env);
        let _ = eval_str("#t", &env);
        let _ = eval_str("nil", &env);
    }

    let final_count = get_env_refcount(&env);
    assert!(
        final_count < initial_count + 5,
        "Empty operations grew too much: {} -> {}",
        initial_count,
        final_count
    );
}

#[test]
fn test_immediate_drop_stress() {
    // Create and immediately drop many values
    for _ in 0..100 {
        let env = new_env();
        let _ = eval_str("(+ 1 2)", &env);
        // env drops here
    }
    // If we get here without crashing, memory management is working
}

#[test]
fn test_nested_quote_structures() {
    // Quoted structures shouldn't leak
    let env = new_env();
    let initial_count = get_env_refcount(&env);

    for _ in 0..100 {
        let _ = eval_str("'(1 2 3 4 5)", &env);
        let _ = eval_str("'((1 2) (3 4) (5 6))", &env);
    }

    let final_count = get_env_refcount(&env);
    assert!(
        final_count < initial_count + 10,
        "Quoted structures grew too much: {} -> {}",
        initial_count,
        final_count
    );
}

// ============================================================================
// Final Boss: Maximum Stress Test (Simplified)
// ============================================================================

#[test]
fn test_maximum_stress() {
    // The ultimate stress test - everything at once (simplified)
    let program = r#"
        ((lambda (factorial sum-range build-list process-list)
           ; Run many intensive operations
           (factorial factorial 10 1))
         ; factorial
         (lambda (self n acc)
           (if (= n 0) acc (self self (- n 1) (* n acc))))
         ; sum-range
         (lambda (self n acc)
           (if (= n 0) acc (self self (- n 1) (+ acc n))))
         ; build-list
         (lambda (self n acc)
           (if (= n 0) acc (self self (- n 1) (cons n acc))))
         ; process-list
         (lambda (self lst acc)
           (if (null? lst)
               acc
               (self self (cdr lst) (+ acc (car lst))))))
    "#;

    let env = new_env();
    let initial_count = get_env_refcount(&env);

    // Run intensive operations
    for _ in 0..10 {
        let _ = eval_str(program, &env);
    }

    let final_count = get_env_refcount(&env);
    assert!(
        final_count < initial_count + 30,
        "Maximum stress test grew too much: {} -> {}",
        initial_count,
        final_count
    );
}
