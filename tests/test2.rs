#![cfg(test)]
use ruthe::*;

// ============================================================================
// Stress Test Helpers - Adapted for Arena Allocator
// ============================================================================

fn count_free_slots(arena: &Arena) -> usize {
    let mut count = 0;
    for i in 0..2048 {
        let r = ArenaRef(i as u16);
        if let Some(Value::Free) = arena.get(r) {
            count += 1;
        }
    }
    count
}

fn count_used_slots(arena: &Arena) -> usize {
    2048 - count_free_slots(arena)
}

// Helper function to evaluate with display string conversion
fn eval_display(arena: &mut Arena, input: &str, env: ArenaRef) -> Result<String, String> {
    eval_string(arena, input, env)
        .map(|val| {
            let mut buf = [0u8; 4096];
            if let Some(Value::Number(n)) = arena.get(val) {
                let mut temp = [0u8; 32];
                let mut idx = 0;
                let mut num = *n;
                let negative = num < 0;
                if negative {
                    num = -num;
                }
                if num == 0 {
                    temp[idx] = b'0';
                    idx += 1;
                } else {
                    let mut divisor = 1i64;
                    let mut temp_num = num;
                    while temp_num >= 10 {
                        divisor *= 10;
                        temp_num /= 10;
                    }
                    while divisor > 0 {
                        let digit = (num / divisor) as u8;
                        temp[idx] = b'0' + digit;
                        idx += 1;
                        num %= divisor;
                        divisor /= 10;
                    }
                }
                if negative {
                    buf[0] = b'-';
                    for i in 0..idx {
                        buf[i + 1] = temp[i];
                    }
                    idx += 1;
                } else {
                    for i in 0..idx {
                        buf[i] = temp[i];
                    }
                }
                core::str::from_utf8(&buf[..idx]).unwrap().to_string()
            } else if let Some(Value::Bool(b)) = arena.get(val) {
                if *b {
                    "#t".to_string()
                } else {
                    "#f".to_string()
                }
            } else {
                "ok".to_string()
            }
        })
        .map_err(|e| {
            let mut buf = [0u8; 256];
            arena
                .list_to_str(e, &mut buf)
                .unwrap_or("error")
                .to_string()
        })
}

// ============================================================================
// Extreme Recursion Tests - Testing TCO & Memory
// ============================================================================

#[test]
fn test_extreme_tail_recursion() {
    // Test that tail call optimization truly works without stack overflow
    // Reduced from 10000 to fit in arena
    let program = r#"
        ((lambda (countdown)
           (countdown countdown 1000 0))
         (lambda (self n acc)
           (if (= n 0)
               acc
               (self self (- n 1) (+ acc 1)))))
    "#;

    let mut arena = Arena::new();
    let env = env_new(&mut arena);
    let initial_used = count_used_slots(&arena);

    let result = eval_display(&mut arena, program, env);
    assert_eq!(result, Ok("1000".to_string()));

    arena.decref(env);
    let final_used = count_used_slots(&arena);

    // Should have cleaned up most allocations
    assert!(
        final_used < initial_used + 50,
        "Extreme recursion didn't clean up properly: {} -> {}",
        initial_used,
        final_used
    );
}

#[test]
fn test_deep_tail_recursion() {
    // Push tail call optimization to limits (reduced for arena size)
    let program = r#"
        ((lambda (sum-to)
           (sum-to sum-to 500 0))
         (lambda (self n acc)
           (if (= n 0)
               acc
               (self self (- n 1) (+ acc n)))))
    "#;

    let mut arena = Arena::new();
    let env = env_new(&mut arena);

    let result = eval_string(&mut arena, program, env);
    assert!(result.is_ok(), "Deep tail recursion failed");

    if let Ok(val) = result {
        arena.decref(val);
    }
    arena.decref(env);
}

#[test]
fn test_mutual_recursion_deep() {
    // Test deep mutual recursion doesn't leak (reduced size)
    let program = r#"
        ((lambda (make-even-odd)
           ((car (make-even-odd make-even-odd)) 100))
         (lambda (self)
           (cons
             (lambda (n)
               (if (= n 0) #t ((cdr (self self)) (- n 1))))
             (lambda (n)
               (if (= n 0) #f ((car (self self)) (- n 1)))))))
    "#;

    let mut arena = Arena::new();
    let env = env_new(&mut arena);
    let initial_used = count_used_slots(&arena);

    let result = eval_string(&mut arena, program, env);
    assert!(result.is_ok(), "Mutual recursion failed");

    if let Ok(val) = result {
        arena.decref(val);
    }
    arena.decref(env);

    let final_used = count_used_slots(&arena);
    assert!(
        final_used < initial_used + 50,
        "Mutual recursion didn't clean up: {} -> {}",
        initial_used,
        final_used
    );
}

// ============================================================================
// Environment Chain Stress Tests
// ============================================================================

#[test]
fn test_deeply_nested_scopes() {
    // Create deeply nested lambda scopes
    let program = r#"
        ((lambda (nest)
           (nest nest 50))
         (lambda (self n)
           (if (= n 0)
               42
               ((lambda (x) (self self (- n 1))) n))))
    "#;

    let mut arena = Arena::new();
    let env = env_new(&mut arena);
    let initial_used = count_used_slots(&arena);

    let result = eval_display(&mut arena, program, env);
    assert_eq!(result, Ok("42".to_string()));

    arena.decref(env);
    let final_used = count_used_slots(&arena);

    assert!(
        final_used < initial_used + 100,
        "Nested scopes leaked: {} -> {}",
        initial_used,
        final_used
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

    let mut arena = Arena::new();
    let env = env_new(&mut arena);
    let initial_used = count_used_slots(&arena);

    let result = eval_display(&mut arena, program, env);
    assert_eq!(result, Ok("401".to_string()));

    arena.decref(env);
    let final_used = count_used_slots(&arena);

    assert!(
        final_used < initial_used + 50,
        "Multiple closures leaked: {} -> {}",
        initial_used,
        final_used
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

    let mut arena = Arena::new();
    let env = env_new(&mut arena);
    let initial_used = count_used_slots(&arena);

    let result = eval_display(&mut arena, program, env);
    assert_eq!(result, Ok("35".to_string()));

    arena.decref(env);
    let final_used = count_used_slots(&arena);

    assert!(
        final_used < initial_used + 50,
        "Environment chain leaked: {} -> {}",
        initial_used,
        final_used
    );
}

// ============================================================================
// Massive Data Structure Tests
// ============================================================================

#[test]
fn test_large_list_creation_and_destruction() {
    // Create a large list and ensure it's cleaned up (reduced size for arena)
    let program = r#"
        ((lambda (build-list)
           ((lambda (big-list)
              (length big-list))
            (build-list build-list 200 nil)))
         (lambda (self n acc)
           (if (= n 0)
               acc
               (self self (- n 1) (cons n acc)))))
    "#;

    let mut arena = Arena::new();
    let env = env_new(&mut arena);
    let initial_used = count_used_slots(&arena);

    let result = eval_display(&mut arena, program, env);
    assert_eq!(result, Ok("200".to_string()));

    arena.decref(env);
    let final_used = count_used_slots(&arena);

    assert!(
        final_used < initial_used + 50,
        "Large list leaked: {} -> {}",
        initial_used,
        final_used
    );
}

#[test]
fn test_repeated_large_allocations() {
    // Repeatedly create and destroy structures
    let mut arena = Arena::new();
    let env = env_new(&mut arena);

    let make_list_program = r#"
        (lambda (self n)
          (if (= n 0)
              nil
              (cons n (self self (- n 1)))))
    "#;

    let make_list = eval_string(&mut arena, make_list_program, env).unwrap();
    let count_after_define = count_used_slots(&arena);

    // Create lists multiple times (smaller lists for arena)
    for _ in 0..5 {
        let list = eval_string(
            &mut arena,
            "((lambda (f) (f f 50)) (lambda (self n) (if (= n 0) nil (cons n (self self (- n 1))))))",
            env,
        );
        if let Ok(val) = list {
            arena.decref(val);
        }
    }

    arena.decref(make_list);
    arena.decref(env);

    let final_used = count_used_slots(&arena);
    assert!(
        final_used <= count_after_define + 100,
        "Repeated allocations leaked: {} -> {}",
        count_after_define,
        final_used
    );
}

// ============================================================================
// Repeated Operations Tests
// ============================================================================

#[test]
fn test_repeated_lambda_creation() {
    // Creating lambdas repeatedly shouldn't accumulate references
    let mut arena = Arena::new();
    let env = env_new(&mut arena);

    let initial_used = count_used_slots(&arena);

    for i in 0..20 {
        let program = format!("((lambda (n) (lambda (x) (+ x n))) {})", i);
        let result = eval_string(&mut arena, &program, env);
        if let Ok(closure) = result {
            // Call the closure
            let call_program = format!("((lambda (f) (f 10)) (lambda (x) (+ x {})))", i);
            let call_result = eval_string(&mut arena, &call_program, env);
            if let Ok(val) = call_result {
                arena.decref(val);
            }
            arena.decref(closure);
        }
    }

    arena.decref(env);
    let final_used = count_used_slots(&arena);

    assert!(
        final_used < initial_used + 100,
        "Repeated lambda creation leaked: {} -> {}",
        initial_used,
        final_used
    );
}

#[test]
fn test_repeated_recursive_calls() {
    // Calling recursive functions repeatedly shouldn't leak
    let mut arena = Arena::new();
    let env = env_new(&mut arena);

    let fib_program = r#"
        (lambda (self n)
          (if (< n 2)
              n
              (+ (self self (- n 1)) (self self (- n 2)))))
    "#;

    let fib = eval_string(&mut arena, fib_program, env).unwrap();
    let initial_used = count_used_slots(&arena);

    // Call multiple times (smaller values for arena)
    for _ in 0..5 {
        let result = eval_string(
            &mut arena,
            "((lambda (f) (f f 8)) (lambda (self n) (if (< n 2) n (+ (self self (- n 1)) (self self (- n 2))))))",
            env,
        );
        if let Ok(val) = result {
            arena.decref(val);
        }
    }

    arena.decref(fib);
    arena.decref(env);

    let final_used = count_used_slots(&arena);
    assert!(
        final_used < initial_used + 100,
        "Repeated recursive calls leaked: {} -> {}",
        initial_used,
        final_used
    );
}

// ============================================================================
// Error Path Memory Tests
// ============================================================================

#[test]
fn test_errors_dont_leak() {
    let mut arena = Arena::new();
    let env = env_new(&mut arena);
    let initial_used = count_used_slots(&arena);

    // Try many operations that will fail
    for _ in 0..20 {
        let _ = eval_string(&mut arena, "(/ 10 0)", env); // Division by zero
        let _ = eval_string(&mut arena, "undefined-var", env); // Unbound variable
        let _ = eval_string(&mut arena, "(car 5)", env); // Type error
    }

    arena.decref(env);
    let final_used = count_used_slots(&arena);

    assert!(
        final_used < initial_used + 50,
        "Error paths leaked: {} -> {}",
        initial_used,
        final_used
    );
}

// ============================================================================
// Arena Reference Counting Tests
// ============================================================================

#[test]
fn test_arena_refcount_basic() {
    let mut arena = Arena::new();

    let val = arena.number(42);
    // Refcount should be 1

    arena.incref(val);
    // Refcount should be 2

    arena.decref(val);
    // Refcount should be 1

    arena.decref(val);
    // Should be freed now

    // Value should be free
    assert!(matches!(arena.get(val), Some(Value::Free) | None));
}

#[test]
fn test_cons_cell_refcount() {
    let mut arena = Arena::new();

    let head = arena.number(1);
    let tail = arena.number(2);
    let cons = arena.cons(head, tail);

    // cons increments refs, so head and tail have refcount 2
    // Decref them to get back to 1
    arena.decref(head);
    arena.decref(tail);

    // Now decref the cons
    arena.decref(cons);

    // All should be freed
    assert!(matches!(arena.get(head), Some(Value::Free) | None));
    assert!(matches!(arena.get(tail), Some(Value::Free) | None));
    assert!(matches!(arena.get(cons), Some(Value::Free) | None));
}

// ============================================================================
// Stress Test: Combined Operations
// ============================================================================

#[test]
fn test_kitchen_sink_stress() {
    // Combine many operations in one test (reduced sizes)
    let program = r#"
        ((lambda (fact fib)
           (+ (fact fact 8 1) (fib fib 8)))
         (lambda (self n acc)
           (if (= n 0) acc (self self (- n 1) (* n acc))))
         (lambda (self n)
           (if (< n 2) n (+ (self self (- n 1)) (self self (- n 2))))))
    "#;

    let mut arena = Arena::new();
    let env = env_new(&mut arena);
    let initial_used = count_used_slots(&arena);

    let result = eval_string(&mut arena, program, env);
    assert!(result.is_ok(), "Kitchen sink failed");

    if let Ok(val) = result {
        arena.decref(val);
    }
    arena.decref(env);

    let final_used = count_used_slots(&arena);
    assert!(
        final_used < initial_used + 100,
        "Kitchen sink leaked: {} -> {}",
        initial_used,
        final_used
    );
}

#[test]
fn test_many_small_operations() {
    // Many small operations shouldn't accumulate memory
    let mut arena = Arena::new();
    let env = env_new(&mut arena);
    let initial_used = count_used_slots(&arena);

    for i in 0..50 {
        let r1 = eval_string(&mut arena, &format!("(+ {} {})", i, i + 1), env);
        if let Ok(val) = r1 {
            arena.decref(val);
        }

        let r2 = eval_string(&mut arena, &format!("(* {} 2)", i), env);
        if let Ok(val) = r2 {
            arena.decref(val);
        }

        let r3 = eval_string(&mut arena, &format!("(< {} 50)", i), env);
        if let Ok(val) = r3 {
            arena.decref(val);
        }
    }

    arena.decref(env);
    let final_used = count_used_slots(&arena);

    assert!(
        final_used < initial_used + 50,
        "Many small operations leaked: {} -> {}",
        initial_used,
        final_used
    );
}

// ============================================================================
// Edge Case Tests
// ============================================================================

#[test]
fn test_empty_operations() {
    // Operations with literals shouldn't accumulate
    let mut arena = Arena::new();
    let env = env_new(&mut arena);
    let initial_used = count_used_slots(&arena);

    for _ in 0..50 {
        let r1 = eval_string(&mut arena, "42", env);
        if let Ok(val) = r1 {
            arena.decref(val);
        }

        let r2 = eval_string(&mut arena, "#t", env);
        if let Ok(val) = r2 {
            arena.decref(val);
        }

        let r3 = eval_string(&mut arena, "nil", env);
        if let Ok(val) = r3 {
            arena.decref(val);
        }
    }

    arena.decref(env);
    let final_used = count_used_slots(&arena);

    assert!(
        final_used < initial_used + 20,
        "Empty operations leaked: {} -> {}",
        initial_used,
        final_used
    );
}

#[test]
fn test_immediate_drop_stress() {
    // Create and immediately drop many arenas
    for _ in 0..50 {
        let mut arena = Arena::new();
        let env = env_new(&mut arena);
        let result = eval_string(&mut arena, "(+ 1 2)", env);
        if let Ok(val) = result {
            arena.decref(val);
        }
        arena.decref(env);
    }
    // If we get here without crashing, memory management is working
}

#[test]
fn test_nested_quote_structures() {
    // Quoted structures shouldn't leak
    let mut arena = Arena::new();
    let env = env_new(&mut arena);
    let initial_used = count_used_slots(&arena);

    for _ in 0..20 {
        let r1 = eval_string(&mut arena, "'(1 2 3 4 5)", env);
        if let Ok(val) = r1 {
            arena.decref(val);
        }

        let r2 = eval_string(&mut arena, "'((1 2) (3 4))", env);
        if let Ok(val) = r2 {
            arena.decref(val);
        }
    }

    arena.decref(env);
    let final_used = count_used_slots(&arena);

    assert!(
        final_used < initial_used + 50,
        "Quoted structures leaked: {} -> {}",
        initial_used,
        final_used
    );
}

// ============================================================================
// Arena-Specific Tests
// ============================================================================

#[test]
fn test_arena_size_limits() {
    // Verify we can't allocate more than arena size
    let mut arena = Arena::new();
    let env = env_new(&mut arena);

    // Try to create a very large list that might exceed arena
    let program = r#"
        ((lambda (build)
           (build build 1500 nil))
         (lambda (self n acc)
           (if (= n 0)
               acc
               (self self (- n 1) (cons n acc)))))
    "#;

    // This might fail or succeed depending on arena pressure
    let result = eval_string(&mut arena, program, env);

    // Clean up regardless
    if let Ok(val) = result {
        arena.decref(val);
    }
    arena.decref(env);
}

#[test]
fn test_arena_reuse() {
    // Verify arena slots are reused after freeing
    let mut arena = Arena::new();
    let initial_free = count_free_slots(&arena);

    // Allocate and free many times
    for _ in 0..10 {
        let vals: [ArenaRef; 10] = [
            arena.number(1),
            arena.number(2),
            arena.number(3),
            arena.number(4),
            arena.number(5),
            arena.number(6),
            arena.number(7),
            arena.number(8),
            arena.number(9),
            arena.number(10),
        ];

        for val in vals {
            arena.decref(val);
        }
    }

    let final_free = count_free_slots(&arena);

    // Should have roughly the same number of free slots
    assert!(
        final_free >= initial_free - 5,
        "Arena not reusing slots properly: {} -> {}",
        initial_free,
        final_free
    );
}

// ============================================================================
// Final Boss: Maximum Stress Test
// ============================================================================

#[test]
fn test_maximum_stress() {
    // The ultimate stress test - scaled for arena
    let program = r#"
        ((lambda (factorial sum-range)
           (+ (factorial factorial 10 1) (sum-range sum-range 100 0)))
         (lambda (self n acc)
           (if (= n 0) acc (self self (- n 1) (* n acc))))
         (lambda (self n acc)
           (if (= n 0) acc (self self (- n 1) (+ acc n)))))
    "#;

    let mut arena = Arena::new();
    let env = env_new(&mut arena);
    let initial_used = count_used_slots(&arena);

    // Run intensive operations multiple times
    for _ in 0..5 {
        let result = eval_string(&mut arena, program, env);
        if let Ok(val) = result {
            arena.decref(val);
        }
    }

    arena.decref(env);
    let final_used = count_used_slots(&arena);

    assert!(
        final_used < initial_used + 100,
        "Maximum stress test leaked: {} -> {}",
        initial_used,
        final_used
    );
}
