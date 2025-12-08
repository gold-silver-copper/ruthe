#[cfg(test)]
mod gc_tests {
    use ruthe::*;

    // ============================================================================
    // Basic GC Functionality Tests
    // ============================================================================

    #[test]
    fn test_gc_reclaims_unreachable_cells() {
        let arena: Arena<1024, 256> = Arena::new();
        let env = init_env(&arena).unwrap();

        // Allocate some temporary values
        for i in 0..50 {
            let expr = parse(&arena, &format!("(+ {} 1)", i)).unwrap();
            let _ = eval(&arena, expr, env).unwrap();
        }

        let used_before = arena.used();
        arena.collect(&[env]);
        let used_after = arena.used();

        // Should have freed most temporary allocations
        assert!(used_after < used_before);
        assert!(used_after < 300); // Env and builtins should be small
    }

    #[test]
    fn test_gc_preserves_environment() {
        let arena: Arena<1024, 256> = Arena::new();
        let env = init_env(&arena).unwrap();

        // Define variables
        let expr = parse(&arena, "(define x 42)").unwrap();
        let _ = eval(&arena, expr, env).unwrap();
        let expr = parse(&arena, "(define y 100)").unwrap();
        let _ = eval(&arena, expr, env).unwrap();

        arena.collect(&[env]);

        // Variables should still be accessible
        assert_eq!(eval_str(&arena, "x", env).unwrap(), 42);
        assert_eq!(eval_str(&arena, "y", env).unwrap(), 100);
    }

    #[test]
    fn test_gc_preserves_lambda_closures() {
        let arena: Arena<2048, 256> = Arena::new();
        let env = init_env(&arena).unwrap();

        let code = "(define make-counter (lambda (n) (lambda () n)))";
        let expr = parse(&arena, code).unwrap();
        let _ = eval(&arena, expr, env).unwrap();

        let expr = parse(&arena, "(define counter (make-counter 42))").unwrap();
        let _ = eval(&arena, expr, env).unwrap();

        arena.collect(&[env]);

        assert_eq!(eval_str(&arena, "(counter)", env).unwrap(), 42);
    }

    #[test]
    fn test_gc_with_deeply_nested_lists() {
        let arena: Arena<2048, 256> = Arena::new();
        let env = init_env(&arena).unwrap();

        // Create deeply nested list
        let code = "(define deep (cons 1 (cons 2 (cons 3 (cons 4 (cons 5 nil))))))";
        let expr = parse(&arena, code).unwrap();
        let _ = eval(&arena, expr, env).unwrap();

        arena.collect(&[env]);

        // Should still be able to access nested elements
        assert_eq!(eval_str(&arena, "(car deep)", env).unwrap(), 1);
        assert_eq!(eval_str(&arena, "(car (cdr deep))", env).unwrap(), 2);
        assert_eq!(eval_str(&arena, "(car (cdr (cdr deep)))", env).unwrap(), 3);
    }

    #[test]
    fn test_gc_multiple_roots() {
        let arena: Arena<2048, 256> = Arena::new();
        let env = init_env(&arena).unwrap();

        let expr1 = parse(&arena, "(cons 1 2)").unwrap();
        let val1 = eval(&arena, expr1, env).unwrap();

        let expr2 = parse(&arena, "(cons 3 4)").unwrap();
        let val2 = eval(&arena, expr2, env).unwrap();

        // Collect with multiple roots
        arena.collect(&[env, val1, val2]);

        // Both values should still be accessible
        assert!(arena.get(val1).is_ok());
        assert!(arena.get(val2).is_ok());
    }

    // ============================================================================
    // Auto-GC Tests
    // ============================================================================

    #[test]
    fn test_auto_gc_triggers_on_oom() {
        let arena: Arena<512, 256> = Arena::new(); // Small arena
        let env = init_env(&arena).unwrap();

        // Fill arena with temporary allocations
        for i in 0..100 {
            let code = format!("(+ {} {})", i, i + 1);
            let _ = eval_str(&arena, &code, env);
        }

        // Should still work due to auto-GC
        assert_eq!(eval_str(&arena, "(+ 1 2)", env).unwrap(), 3);
    }

    #[test]
    fn test_auto_gc_with_recursive_function() {
        let arena: Arena<2048, 256> = Arena::new();
        let env = init_env(&arena).unwrap();

        let code = "
            (define sum-to
              (lambda (n acc)
                (if (= n 0)
                    acc
                    (sum-to (- n 1) (+ acc n)))))
        ";
        let expr = parse(&arena, code).unwrap();
        let _ = eval(&arena, expr, env).unwrap();

        // This will create many temporary allocations
        // Auto-GC should keep memory under control
        assert_eq!(eval_str(&arena, "(sum-to 30 0)", env).unwrap(), 465);
    }

    #[test]
    fn test_push_pop_roots_maintains_gc_correctness() {
        let arena: Arena<2048, 256> = Arena::new();
        let env = init_env(&arena).unwrap();

        arena.push_root(env);

        let expr = parse(&arena, "(define x 42)").unwrap();
        arena.push_root(expr);
        let result = eval(&arena, expr, env).unwrap();
        arena.pop_root();

        arena.collect(&[env]);

        assert_eq!(eval_str(&arena, "x", env).unwrap(), 42);
        arena.pop_root();
    }

    // ============================================================================
    // Lambda and Closure GC Tests
    // ============================================================================

    #[test]
    fn test_gc_lambda_captures_environment() {
        let arena: Arena<2048, 256> = Arena::new();
        let env = init_env(&arena).unwrap();

        let code = "
            (define make-adder
              (lambda (x)
                (lambda (y) (+ x y))))
        ";
        let expr = parse(&arena, code).unwrap();
        let _ = eval(&arena, expr, env).unwrap();

        let expr = parse(&arena, "(define add10 (make-adder 10))").unwrap();
        let _ = eval(&arena, expr, env).unwrap();

        // Create garbage
        for i in 0..50 {
            let code = format!("(+ {} 1)", i);
            let _ = eval_str(&arena, &code, env);
        }

        arena.collect(&[env]);

        // Closure should still work
        assert_eq!(eval_str(&arena, "(add10 5)", env).unwrap(), 15);
    }

    #[test]
    fn test_gc_multiple_closures() {
        let arena: Arena<2048, 256> = Arena::new();
        let env = init_env(&arena).unwrap();

        let code = "
            (define make-multiplier
              (lambda (n)
                (lambda (x) (* x n))))
        ";
        let expr = parse(&arena, code).unwrap();
        let _ = eval(&arena, expr, env).unwrap();

        let expr = parse(&arena, "(define times2 (make-multiplier 2))").unwrap();
        let _ = eval(&arena, expr, env).unwrap();

        let expr = parse(&arena, "(define times3 (make-multiplier 3))").unwrap();
        let _ = eval(&arena, expr, env).unwrap();

        arena.collect(&[env]);

        assert_eq!(eval_str(&arena, "(times2 5)", env).unwrap(), 10);
        assert_eq!(eval_str(&arena, "(times3 5)", env).unwrap(), 15);
    }

    #[test]
    fn test_gc_nested_lambda_environments() {
        let arena: Arena<3072, 256> = Arena::new();
        let env = init_env(&arena).unwrap();

        let code = "
            (define make-counter
              (lambda (start)
                (lambda (inc)
                  (lambda ()
                    (+ start inc)))))
        ";
        let expr = parse(&arena, code).unwrap();
        let _ = eval(&arena, expr, env).unwrap();

        let expr = parse(&arena, "(define counter ((make-counter 10) 5))").unwrap();
        let _ = eval(&arena, expr, env).unwrap();

        arena.collect(&[env]);

        assert_eq!(eval_str(&arena, "(counter)", env).unwrap(), 15);
    }

    #[test]
    fn test_gc_lambda_with_list_capture() {
        let arena: Arena<2048, 256> = Arena::new();
        let env = init_env(&arena).unwrap();

        let code = "
            (define make-list-accessor
              (lambda (lst)
                (lambda () (car lst))))
        ";
        let expr = parse(&arena, code).unwrap();
        let _ = eval(&arena, expr, env).unwrap();

        let expr = parse(
            &arena,
            "(define get-first (make-list-accessor (cons 42 nil)))",
        )
        .unwrap();
        let _ = eval(&arena, expr, env).unwrap();

        arena.collect(&[env]);

        assert_eq!(eval_str(&arena, "(get-first)", env).unwrap(), 42);
    }

    // ============================================================================
    // Recursive Function GC Tests
    // ============================================================================

    #[test]
    fn test_gc_during_factorial() {
        let arena: Arena<2048, 256> = Arena::new();
        let env = init_env(&arena).unwrap();

        let code = "
            (define fact
              (lambda (n)
                (if (= n 0)
                    1
                    (* n (fact (- n 1))))))
        ";
        let expr = parse(&arena, code).unwrap();
        let _ = eval(&arena, expr, env).unwrap();

        // Force multiple GCs during execution
        for _ in 0..5 {
            assert_eq!(eval_str(&arena, "(fact 6)", env).unwrap(), 720);
        }

        arena.collect(&[env]);
        assert_eq!(eval_str(&arena, "(fact 5)", env).unwrap(), 120);
    }

    #[test]
    fn test_gc_tail_recursive_sum() {
        let arena: Arena<2048, 256> = Arena::new();
        let env = init_env(&arena).unwrap();

        let code = "
            (define sum-iter
              (lambda (n acc)
                (if (= n 0)
                    acc
                    (sum-iter (- n 1) (+ acc n)))))
        ";
        let expr = parse(&arena, code).unwrap();
        let _ = eval(&arena, expr, env).unwrap();

        // Should handle large recursion with GC
        assert_eq!(eval_str(&arena, "(sum-iter 100 0)", env).unwrap(), 5050);

        arena.collect(&[env]);
        assert_eq!(eval_str(&arena, "(sum-iter 50 0)", env).unwrap(), 1275);
    }

    #[test]
    fn test_gc_mutual_recursion() {
        let arena: Arena<3072, 256> = Arena::new();
        let env = init_env(&arena).unwrap();

        let code = "
            (define is-even
              (lambda (n)
                (if (= n 0)
                    #t
                    (is-odd (- n 1)))))
        ";
        let expr = parse(&arena, code).unwrap();
        let _ = eval(&arena, expr, env).unwrap();

        let code = "
            (define is-odd
              (lambda (n)
                (if (= n 0)
                    #f
                    (is-even (- n 1)))))
        ";
        let expr = parse(&arena, code).unwrap();
        let _ = eval(&arena, expr, env).unwrap();

        arena.collect(&[env]);

        assert_eq!(eval_str(&arena, "(if (is-even 4) 1 0)", env).unwrap(), 1);
        assert_eq!(eval_str(&arena, "(if (is-odd 4) 1 0)", env).unwrap(), 0);
    }

    // ============================================================================
    // List Structure GC Tests
    // ============================================================================

    #[test]
    fn test_gc_preserves_cons_chains() {
        let arena: Arena<2048, 256> = Arena::new();
        let env = init_env(&arena).unwrap();

        let code = "(define lst (cons 1 (cons 2 (cons 3 (cons 4 (cons 5 nil))))))";
        let expr = parse(&arena, code).unwrap();
        let _ = eval(&arena, expr, env).unwrap();

        // Create garbage
        for i in 0..100 {
            let _ = arena.alloc(LispValue::Number(i));
        }

        arena.collect(&[env]);

        // All list elements should be preserved
        assert_eq!(eval_str(&arena, "(car lst)", env).unwrap(), 1);
        assert_eq!(
            eval_str(&arena, "(car (cdr (cdr (cdr (cdr lst)))))", env).unwrap(),
            5
        );
    }

    #[test]
    fn test_gc_circular_reference_prevention() {
        let arena: Arena<1024, 256> = Arena::new();
        let env = init_env(&arena).unwrap();

        // Create a simple list
        let code = "(define x (cons 1 (cons 2 nil)))";
        let expr = parse(&arena, code).unwrap();
        let _ = eval(&arena, expr, env).unwrap();

        // GC should handle this without infinite loop
        arena.collect(&[env]);

        assert_eq!(eval_str(&arena, "(car x)", env).unwrap(), 1);
    }

    #[test]
    fn test_gc_shared_substructures() {
        let arena: Arena<2048, 256> = Arena::new();
        let env = init_env(&arena).unwrap();

        let code = "(define tail (cons 3 (cons 4 nil)))";
        let expr = parse(&arena, code).unwrap();
        let _ = eval(&arena, expr, env).unwrap();

        let code = "(define lst1 (cons 1 tail))";
        let expr = parse(&arena, code).unwrap();
        let _ = eval(&arena, expr, env).unwrap();

        let code = "(define lst2 (cons 2 tail))";
        let expr = parse(&arena, code).unwrap();
        let _ = eval(&arena, expr, env).unwrap();

        arena.collect(&[env]);

        // Both lists should share tail and work correctly
        assert_eq!(eval_str(&arena, "(car lst1)", env).unwrap(), 1);
        assert_eq!(eval_str(&arena, "(car lst2)", env).unwrap(), 2);
        assert_eq!(eval_str(&arena, "(car (cdr lst1))", env).unwrap(), 3);
        assert_eq!(eval_str(&arena, "(car (cdr lst2))", env).unwrap(), 3);
    }

    #[test]
    fn test_gc_deeply_nested_cons() {
        let arena: Arena<4096, 256> = Arena::new();
        let env = init_env(&arena).unwrap();

        // Build deeply nested structure
        let mut code = String::from("(define deep ");
        for _ in 0..50 {
            code.push_str("(cons 1 ");
        }
        code.push_str("nil");
        for _ in 0..50 {
            code.push(')');
        }
        code.push(')');

        let expr = parse(&arena, &code).unwrap();
        let _ = eval(&arena, expr, env).unwrap();

        arena.collect(&[env]);

        assert_eq!(eval_str(&arena, "(car deep)", env).unwrap(), 1);
    }

    // ============================================================================
    // Stress Tests
    // ============================================================================

    #[test]
    fn test_gc_repeated_allocation_deallocation() {
        let arena: Arena<1024, 256> = Arena::new();
        let env = init_env(&arena).unwrap();

        for round in 0..10 {
            // Allocate many temps
            for i in 0..50 {
                let code = format!("(+ {} {})", i, round);
                let _ = eval_str(&arena, &code, env);
            }

            // Force GC
            arena.collect(&[env]);

            // Should still work
            assert_eq!(eval_str(&arena, "(+ 1 2)", env).unwrap(), 3);
        }
    }

    #[test]
    fn test_gc_alternating_define_and_gc() {
        let arena: Arena<2048, 256> = Arena::new();
        let env = init_env(&arena).unwrap();

        for i in 0..20 {
            let code = format!("(define x{} {})", i, i * 10);
            let expr = parse(&arena, &code).unwrap();
            let _ = eval(&arena, expr, env).unwrap();

            if i % 3 == 0 {
                arena.collect(&[env]);
            }
        }

        // All definitions should be preserved
        assert_eq!(eval_str(&arena, "x0", env).unwrap(), 0);
        assert_eq!(eval_str(&arena, "x10", env).unwrap(), 100);
        assert_eq!(eval_str(&arena, "x19", env).unwrap(), 190);
    }

    #[test]
    fn test_gc_large_computation() {
        let arena: Arena<4096, 256> = Arena::new();
        let env = init_env(&arena).unwrap();

        let code = "
            (define compute
              (lambda (n)
                (if (= n 0)
                    0
                    (+ n (compute (- n 1))))))
        ";
        let expr = parse(&arena, code).unwrap();
        let _ = eval(&arena, expr, env).unwrap();

        // Should handle with auto-GC
        assert_eq!(eval_str(&arena, "(compute 100)", env).unwrap(), 5050);

        arena.collect(&[env]);

        // Should still work after explicit GC
        assert_eq!(eval_str(&arena, "(compute 50)", env).unwrap(), 1275);
    }

    #[test]
    fn test_gc_many_small_allocations() {
        let arena: Arena<2048, 256> = Arena::new();
        let env = init_env(&arena).unwrap();

        // Allocate many small numbers
        for i in 0..500 {
            let _ = arena.alloc(LispValue::Number(i));
        }

        let used_before = arena.used();
        arena.collect(&[env]);
        let used_after = arena.used();

        // Most should be collected
        assert!(used_after < used_before / 2);
    }

    #[test]
    fn test_gc_mixed_types() {
        let arena: Arena<2048, 256> = Arena::new();
        let env = init_env(&arena).unwrap();

        let expr = parse(&arena, "(define x 42)").unwrap();
        let _ = eval(&arena, expr, env).unwrap();

        let expr = parse(&arena, "(define y #t)").unwrap();
        let _ = eval(&arena, expr, env).unwrap();

        let expr = parse(&arena, "(define z (cons 1 2))").unwrap();
        let _ = eval(&arena, expr, env).unwrap();

        let expr = parse(&arena, "(define f (lambda (n) (* n 2)))").unwrap();
        let _ = eval(&arena, expr, env).unwrap();

        arena.collect(&[env]);

        // All types should be preserved
        assert_eq!(eval_str(&arena, "x", env).unwrap(), 42);
        assert_eq!(eval_str(&arena, "(if y 1 0)", env).unwrap(), 1);
        assert_eq!(eval_str(&arena, "(car z)", env).unwrap(), 1);
        assert_eq!(eval_str(&arena, "(f 10)", env).unwrap(), 20);
    }

    // ============================================================================
    // Edge Cases
    // ============================================================================

    #[test]
    fn test_gc_empty_environment() {
        let arena: Arena<1024, 256> = Arena::new();
        let env = env_new(&arena).unwrap();

        // Create and collect garbage
        for i in 0..100 {
            let _ = arena.alloc(LispValue::Number(i));
        }

        arena.collect(&[env]);

        // Should not crash
        assert!(arena.used() < 100);
    }

    #[test]
    fn test_gc_nil_preservation() {
        let arena: Arena<1024, 256> = Arena::new();
        let env = init_env(&arena).unwrap();

        let expr = parse(&arena, "(define x nil)").unwrap();
        let _ = eval(&arena, expr, env).unwrap();

        arena.collect(&[env]);

        // Nil should work after GC
        let expr = parse(&arena, "x").unwrap();
        let result = eval(&arena, expr, env).unwrap();
        assert!(matches!(arena.get(result), Ok(LispValue::Nil)));
    }

    #[test]
    fn test_gc_with_no_roots() {
        let arena: Arena<1024, 256> = Arena::new();

        // Allocate without roots
        for i in 0..100 {
            let _ = arena.alloc(LispValue::Number(i));
        }

        let used_before = arena.used();
        arena.collect(&[]);
        let used_after = arena.used();

        // Everything should be collected
        assert_eq!(used_after, 0);
        assert!(used_before > 0);
    }

    #[test]
    fn test_gc_single_root() {
        let arena: Arena<1024, 256> = Arena::new();

        let root = arena.alloc(LispValue::Number(42)).unwrap();

        // Create garbage
        for i in 0..100 {
            let _ = arena.alloc(LispValue::Number(i));
        }

        arena.collect(&[root]);

        // Root should be preserved
        assert!(matches!(arena.get(root), Ok(LispValue::Number(42))));
    }

    #[test]
    fn test_gc_preserves_builtin_functions() {
        let arena: Arena<1024, 256> = Arena::new();
        let env = init_env(&arena).unwrap();

        // Create garbage
        for i in 0..100 {
            let _ = arena.alloc(LispValue::Number(i));
        }

        arena.collect(&[env]);

        // All builtins should still work
        assert_eq!(eval_str(&arena, "(+ 1 2)", env).unwrap(), 3);
        assert_eq!(eval_str(&arena, "(- 5 3)", env).unwrap(), 2);
        assert_eq!(eval_str(&arena, "(* 4 5)", env).unwrap(), 20);
        assert_eq!(eval_str(&arena, "(car (cons 1 2))", env).unwrap(), 1);
    }

    // ============================================================================
    // Memory Exhaustion Tests
    // ============================================================================

    #[test]
    fn test_gc_prevents_oom_in_loop() {
        let arena: Arena<2048, 256> = Arena::new();
        let env = init_env(&arena).unwrap();

        let code = "
            (define loop
              (lambda (n)
                (if (= n 0)
                    0
                    (loop (- n 1)))))
        ";
        let expr = parse(&arena, code).unwrap();
        let _ = eval(&arena, expr, env).unwrap();

        // Should complete without OOM due to auto-GC and TCO
        assert_eq!(eval_str(&arena, "(loop 100)", env).unwrap(), 0);
    }

    #[test]
    fn test_gc_recovers_from_near_oom() {
        let arena: Arena<512, 256> = Arena::new();
        let env = init_env(&arena).unwrap();

        // Fill arena almost completely
        for i in 0..100 {
            let _ = arena.alloc(LispValue::Number(i));
        }

        // This should trigger GC and succeed
        let result = eval_str(&arena, "(+ 1 2)", env);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 3);
    }

    #[test]
    fn test_gc_handles_fragmentation() {
        let arena: Arena<2048, 256> = Arena::new();
        let env = init_env(&arena).unwrap();

        // Create many small objects
        for _ in 0..10 {
            for i in 0..50 {
                let _ = arena.alloc(LispValue::Number(i));
            }
            arena.collect(&[env]);
        }

        // Should still be able to allocate
        assert_eq!(eval_str(&arena, "(+ 1 2)", env).unwrap(), 3);
    }

    // ============================================================================
    // Complex Scenarios
    // ============================================================================

    #[test]
    fn test_gc_during_nested_function_calls() {
        let arena: Arena<2048, 256> = Arena::new();
        let env = init_env(&arena).unwrap();

        let expr = parse(&arena, "(define f (lambda (x) (* x 2)))").unwrap();
        let _ = eval(&arena, expr, env).unwrap();

        let expr = parse(&arena, "(define g (lambda (x) (+ x 1)))").unwrap();
        let _ = eval(&arena, expr, env).unwrap();

        let expr = parse(&arena, "(define h (lambda (x) (- x 1)))").unwrap();
        let _ = eval(&arena, expr, env).unwrap();

        // Create garbage between calls
        for i in 0..30 {
            let _ = arena.alloc(LispValue::Number(i));
        }

        arena.collect(&[env]);

        // Nested calls should work
        assert_eq!(eval_str(&arena, "(f (g (h 10)))", env).unwrap(), 18);
    }

    #[test]
    fn test_gc_preserves_multiple_lambda_definitions() {
        let arena: Arena<3072, 256> = Arena::new();
        let env = init_env(&arena).unwrap();

        // Define multiple lambdas
        for i in 0..10 {
            let code = format!("(define f{} (lambda (x) (+ x {})))", i, i * 10);
            let expr = parse(&arena, &code).unwrap();
            let _ = eval(&arena, expr, env).unwrap();
        }

        arena.collect(&[env]);

        // All should work
        assert_eq!(eval_str(&arena, "(f0 5)", env).unwrap(), 5);
        assert_eq!(eval_str(&arena, "(f5 5)", env).unwrap(), 55);
        assert_eq!(eval_str(&arena, "(f9 5)", env).unwrap(), 95);
    }

    #[test]
    fn test_gc_with_list_manipulation() {
        let arena: Arena<2048, 256> = Arena::new();
        let env = init_env(&arena).unwrap();

        let code = "
            (define build-list
              (lambda (n)
                (if (= n 0)
                    nil
                    (cons n (build-list (- n 1))))))
        ";
        let expr = parse(&arena, code).unwrap();
        let _ = eval(&arena, expr, env).unwrap();

        let expr = parse(&arena, "(define mylist (build-list 10))").unwrap();
        let _ = eval(&arena, expr, env).unwrap();

        arena.collect(&[env]);

        assert_eq!(eval_str(&arena, "(car mylist)", env).unwrap(), 10);
        assert_eq!(eval_str(&arena, "(car (cdr mylist))", env).unwrap(), 9);
    }

    #[test]
    fn test_gc_doesnt_collect_intermediate_results() {
        let arena: Arena<2048, 256> = Arena::new();
        let env = init_env(&arena).unwrap();

        // Complex expression that creates many intermediates
        let code = "(+ (* 2 3) (* 4 5) (* 6 7))";
        assert_eq!(eval_str(&arena, code, env).unwrap(), 68);

        arena.collect(&[env]);

        // Should still work
        assert_eq!(eval_str(&arena, "(+ 1 2)", env).unwrap(), 3);
    }

    #[test]
    fn test_gc_mark_phase_completeness() {
        let arena: Arena<2048, 256> = Arena::new();
        let env = init_env(&arena).unwrap();

        // Create interconnected structures
        let code = "
            (define a (cons 1 nil))
            (define b (cons 2 a))
            (define c (cons 3 b))
        ";

        for line in code.lines() {
            let trimmed = line.trim();
            if !trimmed.is_empty() {
                let expr = parse(&arena, trimmed).unwrap();
                let _ = eval(&arena, expr, env).unwrap();
            }
        }

        arena.collect(&[env]);

        // All should be reachable
        assert_eq!(eval_str(&arena, "(car c)", env).unwrap(), 3);
        assert_eq!(eval_str(&arena, "(car (cdr c))", env).unwrap(), 2);
        assert_eq!(eval_str(&arena, "(car (cdr (cdr c)))", env).unwrap(), 1);
    }

    #[test]
    fn test_gc_sweep_phase_reclaims_all_garbage() {
        let arena: Arena<1024, 256> = Arena::new();
        let env = init_env(&arena).unwrap();

        // Allocate and orphan many objects
        for i in 0..200 {
            let _ = arena.alloc(LispValue::Number(i));
        }

        let used_before = arena.used();
        arena.collect(&[env]);
        let used_after = arena.used();

        // Should have swept most garbage
        assert!(used_after < used_before / 3);
    }
}
