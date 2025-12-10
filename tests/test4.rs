#![cfg(test)]

//! Test suite for detecting and verifying issues in the Lisp interpreter
//!
//! This file contains tests that specifically check for the issues identified
//! in the deep analysis. Tests FAIL when bugs are present and PASS when fixed.

use ruthe::*;

// ============================================================================
// CRITICAL ISSUE #1: Stack Overflow in Recursive Decref - FIXED!
// ============================================================================

#[test]
fn test_deep_list_stack_overflow() {
    // This test demonstrates the stack overflow issue with very deep lists
    // FIXED: The new implementation uses iterative decref with an explicit stack

    let arena = Arena::<50000>::new();

    // Build an extremely deep list (10,000+ elements)
    let mut list = arena.nil().unwrap();
    for i in 0..10000 {
        let num = arena.number(i).unwrap();
        list = arena.cons(&num, &list).unwrap();
    }

    // Dropping this list will cause recursive decref
    // FIXED: With iterative decref, this completes successfully
    drop(list);

    // If we get here, the bug is fixed!
}

#[test]
fn test_deep_list_causes_many_stack_frames() {
    // This test shows that even moderately deep lists use significant stack
    let arena = Arena::<10000>::new();

    // Build a 500-element list
    let mut list = arena.nil().unwrap();
    for i in 0..500 {
        let num = arena.number(i).unwrap();
        list = arena.cons(&num, &list).unwrap();
    }

    // Count free slots before drop
    let free_before = (0..10000)
        .filter(|&i| matches!(arena.values[i].get(), Value::Free))
        .count();

    // Drop the list - should free all memory
    drop(list);

    // Verify memory was freed
    let free_after = (0..10000)
        .filter(|&i| matches!(arena.values[i].get(), Value::Free))
        .count();

    // Should have freed at least 500 cons cells + 500 numbers = 1000 allocations
    assert!(
        free_after >= free_before + 900,
        "Memory not properly freed: expected to free ~1000 slots, only freed {}",
        free_after - free_before
    );
}

#[test]
fn test_nested_structures_cause_deep_recursion() {
    // Nested lambdas and environments also cause deep recursion
    let arena = Arena::<5000>::new();

    // Build deeply nested lambda: (lambda () (lambda () (lambda () ...)))
    let mut body = arena.number(42).unwrap();
    for _ in 0..200 {
        let params = arena.nil().unwrap();
        let env = arena.nil().unwrap();
        body = arena.lambda(&params, &body, &env).unwrap();
    }

    // FIXED: Iterative decref handles this without stack issues
    drop(body);

    // If we get here without crashing, we're good
}

// ============================================================================
// CRITICAL ISSUE #2: Memory Leak in env_set - FIXED!
// ============================================================================

#[test]
fn test_env_set_does_not_leak_memory() {
    let arena = Arena::<5000>::new();
    let env = env_new(&arena).unwrap();

    // Count allocations before
    let used_before = (0..5000)
        .filter(|&i| !matches!(arena.values[i].get(), Value::Free))
        .count();

    // Define 100 variables
    for i in 0..100 {
        let name = arena.str_to_list(&format!("var{}", i)).unwrap();
        let value = arena.number(i).unwrap();
        env_set(&arena, &env, &name, &value).unwrap();
    }

    // Count allocations after
    let used_after = (0..5000)
        .filter(|&i| !matches!(arena.values[i].get(), Value::Free))
        .count();

    let allocated = used_after - used_before;

    // Expected allocations:
    // - 100 variables * ~6 chars * 2 (char + cons) = ~1200 for names
    // - 100 numbers = 100
    // - 100 bindings (cons of symbol and value) = 100
    // - Some overhead for symbols and env structure
    // Total: ~1500-2000 allocations expected

    println!("Allocated {} slots for 100 definitions", allocated);

    // FIXED: With remove_binding, memory usage should be reasonable
    assert!(
        allocated < 2000,
        "Memory leak detected: used {} allocations for 100 definitions (expected < 2000)",
        allocated
    );
}

#[test]
fn test_env_set_does_not_accumulate_old_bindings() {
    let arena = Arena::<2000>::new();
    let env = env_new(&arena).unwrap();

    let name = arena.str_to_list("x").unwrap();

    // Set the same variable 50 times
    for i in 0..50 {
        let value = arena.number(i).unwrap();
        env_set(&arena, &env, &name, &value).unwrap();
    }

    // Retrieve the value - should be 49 (last one)
    if let Some(val) = env_get(&arena, &env, &name) {
        if let Some(Value::Number(n)) = val.get() {
            assert_eq!(n, 49, "Should retrieve the most recent value");
        }
    }

    // Count how many numbers 0-49 exist in memory
    let mut count = 0;
    for i in 0..2000 {
        if let Value::Number(n) = arena.values[i].get() {
            if n < 50 {
                count += 1;
            }
        }
    }

    println!("Found {} numbers in memory after 50 redefinitions", count);

    // FIXED: Only 1 number (the current value) should remain
    assert!(
        count <= 1,
        "Memory leak: found {} old values in memory (expected 1)",
        count
    );
}

#[test]
fn test_repeated_defines_do_not_exhaust_memory() {
    let arena = Arena::<1000>::new();
    let env = env_new(&arena).unwrap();

    let name = arena.str_to_list("counter").unwrap();

    // Keep redefining the same variable
    let mut successful = 0;
    for i in 0..500 {
        let value = arena.number(i).unwrap();
        env_set(&arena, &env, &name, &value).unwrap();
        successful = i;

        // Check if arena is getting full
        let free_count = (0..1000)
            .filter(|&j| matches!(arena.values[j].get(), Value::Free))
            .count();

        if free_count < 10 {
            println!("Arena nearly exhausted after {} definitions", i);
            break;
        }
    }

    println!(
        "Successfully defined {} times before running low on memory",
        successful
    );

    // FIXED: We should handle all 500 definitions without issue
    assert!(
        successful >= 400,
        "Memory exhaustion detected: only completed {} definitions (expected >= 400)",
        successful
    );
}

// ============================================================================
// ISSUE #3: Inefficient O(N) Allocation - Still Present but Documented
// ============================================================================

#[test]
fn test_allocation_performance_remains_consistent() {
    let arena = Arena::<1000>::new();

    // Allocate 500 values
    let mut refs = Vec::new();
    for i in 0..500 {
        refs.push(arena.number(i).unwrap());
    }

    // Now arena is 50% full
    let next_free = arena.next_free.get();

    // Drop one in the middle to create a free slot
    let drop_idx = 250;
    drop(refs[drop_idx].clone());
    refs.remove(drop_idx);

    // Note: Next allocation must scan from next_free (~500) back to slot 250
    // This is a documented performance issue but not a critical bug

    // Allocate a new value - should succeed
    let new_val = arena.number(999).unwrap();
    assert!(!new_val.is_null(), "Should successfully allocate");

    println!(
        "Allocated with next_free at {} (scans ~{} slots without free list)",
        next_free,
        next_free - drop_idx
    );
}

#[test]
fn test_fragmented_arena_handles_allocation() {
    let arena = Arena::<1000>::new();

    // Create fragmentation: allocate all, then free every other slot
    let mut refs = Vec::new();
    for i in 0..1000 {
        refs.push(arena.number(i).unwrap());
    }

    // Free every other slot (500 slots freed)
    let refs: Vec<_> = refs
        .into_iter()
        .enumerate()
        .filter(|(i, _)| i % 2 == 0)
        .map(|(_, r)| r)
        .collect();

    // Now arena is fragmented: slot 0 used, 1 free, 2 used, 3 free, etc.

    // Allocate 500 new values - all should succeed
    for i in 0..500 {
        let val = arena.number(1000 + i).unwrap();
        assert!(!val.is_null(), "Should find free slot even when fragmented");
    }

    // TEST PASSES if all allocations succeed
    // Performance note: each allocation searches for free slots (documented issue)
}

// ============================================================================
// ISSUE #4: String Representation Overhead - Architectural Limitation
// ============================================================================

#[test]
fn test_string_allocation_overhead() {
    let arena = Arena::<1000>::new();

    let free_before = (0..1000)
        .filter(|&i| matches!(arena.values[i].get(), Value::Free))
        .count();

    // Create a simple 5-character string
    let string = arena.str_to_list("hello").unwrap();

    let free_after = (0..1000)
        .filter(|&i| matches!(arena.values[i].get(), Value::Free))
        .count();

    let used = free_before - free_after;

    println!("'hello' used {} allocations", used);

    // Current implementation: 5 chars + 5 cons cells = 10 allocations
    // This is a known limitation, not a bug to fix
    // With inline strings, this would be just 1 allocation

    // This test documents the overhead but doesn't fail
    assert!(
        used >= 10,
        "String uses {} allocations (expected >= 10 for linked-list representation)",
        used
    );
}

#[test]
fn test_many_strings_memory_usage() {
    let arena = Arena::<5000>::new();

    // Create 100 5-character strings
    let mut strings = Vec::new();
    for i in 0..100 {
        let s = arena.str_to_list(&format!("str{:02}", i)).unwrap();
        strings.push(s);
    }

    let free_count = (0..5000)
        .filter(|&i| matches!(arena.values[i].get(), Value::Free))
        .count();

    let used = 5000 - free_count;

    println!("100 short strings used {} allocations", used);

    // Each string "strXX" is 5 chars = ~10 allocations
    // 100 strings * 10 = ~1000 allocations
    // This is expected behavior, not a bug

    assert!(
        used <= 1500,
        "Used {} allocations for 100 strings (expected ~1000-1500)",
        used
    );
}

#[test]
fn test_symbol_comparison_works_correctly() {
    let arena = Arena::<500>::new();

    // Create two identical symbols
    let sym1 = arena.str_to_list("variable").unwrap();
    let sym2 = arena.str_to_list("variable").unwrap();

    // These are different allocations but should compare equal
    assert_ne!(sym1.raw().0, sym2.raw().0, "Should be different objects");
    assert!(arena.str_eq(&sym1, &sym2), "Should compare equal");

    // Create a different symbol
    let sym3 = arena.str_to_list("different").unwrap();
    assert!(!arena.str_eq(&sym1, &sym3), "Should compare not equal");

    // Note: This is O(n) comparison, documented issue but not a bug
}

// ============================================================================
// ISSUE #5: Environment Lookup Performance - O(n*m) but Correct
// ============================================================================

#[test]
fn test_environment_lookup_correctness() {
    let arena = Arena::<5000>::new();
    let env = env_new(&arena).unwrap();

    // Add 50 bindings
    for i in 0..50 {
        let name = arena.str_to_list(&format!("var{:02}", i)).unwrap();
        let value = arena.number(i).unwrap();
        env_set(&arena, &env, &name, &value).unwrap();
    }

    // Look up each variable and verify correct value
    for i in 0..50 {
        let name = arena.str_to_list(&format!("var{:02}", i)).unwrap();

        if let Some(val) = env_get(&arena, &env, &name) {
            if let Some(Value::Number(n)) = val.get() {
                assert_eq!(n, i, "Should retrieve correct value for var{:02}", i);
            } else {
                panic!("Expected number for var{:02}", i);
            }
        } else {
            panic!("Failed to find var{:02}", i);
        }
    }

    // Note: Lookup is O(n*m), documented performance issue but functionally correct
}

#[test]
fn test_nested_environments_lookup_correctness() {
    let arena = Arena::<5000>::new();
    let mut env = env_new(&arena).unwrap();

    // Create 5 nested environments
    for level in 0..5 {
        env = env_with_parent(&arena, &env).unwrap();

        // Add 10 bindings to each level
        for i in 0..10 {
            let name = arena.str_to_list(&format!("v{}_{}", level, i)).unwrap();
            let value = arena.number((level * 10 + i) as i64).unwrap();
            env_set(&arena, &env, &name, &value).unwrap();
        }
    }

    // Look up variables from different levels
    for level in 0..5 {
        for i in 0..10 {
            let name = arena.str_to_list(&format!("v{}_{}", level, i)).unwrap();

            if let Some(val) = env_get(&arena, &env, &name) {
                if let Some(Value::Number(n)) = val.get() {
                    assert_eq!(
                        n,
                        (level * 10 + i) as i64,
                        "Should find correct value for v{}_{}",
                        level,
                        i
                    );
                }
            } else {
                panic!("Failed to find v{}_{}", level, i);
            }
        }
    }

    // Note: O(d*n) lookup is a performance issue but functionally correct
}

// ============================================================================
// ISSUE #6: Helper Functions Create Temporary Refs - Still Present
// ============================================================================

#[test]
fn test_list_len_correctness() {
    let arena = Arena::<1000>::new();

    // Build a 100-element list
    let mut list = arena.nil().unwrap();
    for i in 0..100 {
        let num = arena.number(i).unwrap();
        list = arena.cons(&num, &list).unwrap();
    }

    // Get the length
    let len = arena.list_len(&list);
    assert_eq!(len, 100, "Should correctly count list length");

    // Note: This creates temporary Refs (performance issue) but works correctly
}

#[test]
fn test_str_eq_correctness() {
    let arena = Arena::<500>::new();

    let s1 = arena.str_to_list("hello").unwrap();
    let s2 = arena.str_to_list("hello").unwrap();
    let s3 = arena.str_to_list("world").unwrap();

    assert!(arena.str_eq(&s1, &s2), "Equal strings should compare equal");
    assert!(
        !arena.str_eq(&s1, &s3),
        "Different strings should compare not equal"
    );

    // Note: Creates temporary Refs (performance issue) but functionally correct
}

// ============================================================================
// Integration Tests: Real-World Scenarios
// ============================================================================

#[test]
fn test_recursive_function_handles_reasonable_depth() {
    let arena = Arena::<5000>::new();
    let env = env_new(&arena).unwrap();

    // Define a recursive counter
    let code = r#"
        (define count
          (lambda (n)
            (if (= n 0)
                0
                (count (- n 1)))))
    "#;

    let _ = eval_string(&arena, code, &env);

    // Call it with reasonable depths - should all succeed
    for depth in [10, 50, 100].iter() {
        let result = eval_string(&arena, &format!("(count {})", depth), &env);

        assert!(
            result.is_ok(),
            "Should handle depth {} without error",
            depth
        );

        // Check memory usage
        let free_count = (0..5000)
            .filter(|&i| matches!(arena.values[i].get(), Value::Free))
            .count();

        assert!(
            free_count > 500,
            "Should have reasonable free memory after depth {}",
            depth
        );
    }
}

#[test]
fn test_building_large_list_succeeds() {
    let arena = Arena::<10000>::new();
    let env = env_new(&arena).unwrap();

    // Build a list using recursion
    let code = r#"
        (define build-list
          (lambda (n acc)
            (if (= n 0)
                acc
                (build-list (- n 1) (cons n acc)))))
    "#;

    let result = eval_string(&arena, code, &env);
    assert!(result.is_ok(), "Should define build-list function");

    // Build a 500-element list
    let result = eval_string(&arena, "(build-list 500 nil)", &env);
    assert!(result.is_ok(), "Should build 500-element list");

    if let Ok(list) = result {
        let len = arena.list_len(&list);
        assert_eq!(len, 500, "List should have 500 elements");

        // Drop it - should not crash
        drop(list);
    }
}

#[test]
fn test_set_bang_mutation() {
    let arena = Arena::<2000>::new();
    let env = env_new(&arena).unwrap();

    // Test basic set!
    let _ = eval_string(&arena, "(define x 10)", &env);
    let _ = eval_string(&arena, "(set! x 20)", &env);
    let result = eval_string(&arena, "x", &env).unwrap();

    if let Some(Value::Number(n)) = result.get() {
        assert_eq!(n, 20, "set! should mutate existing binding");
    }

    // Test set! in nested scope
    let code = r#"
        (define make-counter
          (lambda ()
            (define count 0)
            (lambda ()
              (set! count (+ count 1))
              count)))
    "#;
    let _ = eval_string(&arena, code, &env);
    let _ = eval_string(&arena, "(define c (make-counter))", &env);

    let r1 = eval_string(&arena, "(c)", &env).unwrap();
    let r2 = eval_string(&arena, "(c)", &env).unwrap();
    let r3 = eval_string(&arena, "(c)", &env).unwrap();

    if let Some(Value::Number(n)) = r3.get() {
        assert_eq!(n, 3, "Counter should increment across calls");
    }
}

#[test]
fn test_begin_with_side_effects() {
    let arena = Arena::<2000>::new();
    let env = env_new(&arena).unwrap();

    let _ = eval_string(&arena, "(define x 0)", &env);

    let code = r#"
        (begin
          (set! x 1)
          (set! x 2)
          (set! x 3)
          x)
    "#;

    let result = eval_string(&arena, code, &env).unwrap();

    if let Some(Value::Number(n)) = result.get() {
        assert_eq!(n, 3, "begin should execute all expressions and return last");
    }

    // Verify x was mutated
    let x_val = eval_string(&arena, "x", &env).unwrap();
    if let Some(Value::Number(n)) = x_val.get() {
        assert_eq!(n, 3, "Side effects should persist");
    }
}

#[test]
fn test_lambda_with_multiple_body_expressions() {
    let arena = Arena::<2000>::new();
    let env = env_new(&arena).unwrap();

    let _ = eval_string(&arena, "(define x 0)", &env);

    // Lambda with multiple body expressions should work
    let code = r#"
        (define f
          (lambda ()
            (set! x 10)
            (set! x 20)
            x))
    "#;

    let _ = eval_string(&arena, code, &env);
    let result = eval_string(&arena, "(f)", &env).unwrap();

    if let Some(Value::Number(n)) = result.get() {
        assert_eq!(n, 20, "Multi-expression lambda should return last value");
    }
}

// ============================================================================
// Summary Test: Document All Issues
// ============================================================================

#[test]
fn test_document_known_issues() {
    println!("\n=== KNOWN ISSUES SUMMARY ===\n");

    println!("1. DEEP LIST RECURSION: ✅ FIXED");
    println!("   - Iterative decref with explicit stack");
    println!("   - No more stack overflow on deep lists\n");

    println!("2. ENV_SET MEMORY LEAK: ✅ FIXED");
    println!("   - remove_binding prevents accumulation");
    println!("   - Old bindings are properly freed\n");

    println!("3. INEFFICIENT ALLOCATION: ⚠️  STILL PRESENT");
    println!("   - O(N) linear search for free slots");
    println!("   - Worst case: scan entire arena");
    println!("   - Solution: Maintain O(1) free list\n");

    println!("4. STRING REPRESENTATION: ⚠️  ARCHITECTURAL");
    println!("   - Linked-list strings use many allocations");
    println!("   - 5-char string uses 10 allocations");
    println!("   - Not a bug, design limitation\n");

    println!("5. ENVIRONMENT LOOKUP: ⚠️  PERFORMANCE ISSUE");
    println!("   - O(n*m) where n=bindings, m=name length");
    println!("   - Nested envs multiply cost");
    println!("   - Functionally correct\n");

    println!("6. HELPER FUNCTION OVERHEAD: ⚠️  PERFORMANCE ISSUE");
    println!("   - list_len/str_eq create temporary Refs");
    println!("   - Each Ref does refcount manipulation");
    println!("   - Functionally correct\n");

    println!("Run individual tests to verify each issue.");
}
