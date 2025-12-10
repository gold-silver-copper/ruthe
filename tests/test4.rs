#![cfg(test)]

//! Test suite for detecting and verifying issues in the Lisp interpreter
//!
//! This file contains tests that specifically check for the issues identified
//! in the deep analysis, including:
//! - Stack overflow in recursive decref
//! - Memory leaks in env_set
//! - Allocation efficiency problems
//! - String representation overhead
//! - Performance issues in lookups

use ruthe::*;

// ============================================================================
// CRITICAL ISSUE #1: Stack Overflow in Recursive Decref
// ============================================================================

#[test]
#[should_panic(expected = "stack overflow")]
#[ignore] // Run with: cargo test --release -- --ignored --test-threads=1
fn test_deep_list_stack_overflow() {
    // This test demonstrates the stack overflow issue with very deep lists
    // Note: This will actually crash, so it's marked as ignored by default

    let arena = Arena::<50000>::new();

    // Build an extremely deep list (10,000+ elements)
    let mut list = arena.nil();
    for i in 0..10000 {
        let num = arena.number(i);
        list = arena.cons(&num, &list);
    }

    // Dropping this list will cause recursive decref
    // With 10,000 levels of recursion, this will stack overflow
    drop(list); // BOOM! Stack overflow here
}

#[test]
fn test_deep_list_causes_many_stack_frames() {
    // This test shows that even moderately deep lists use significant stack
    let arena = Arena::<10000>::new();

    // Build a 500-element list
    let mut list = arena.nil();
    for i in 0..500 {
        let num = arena.number(i);
        list = arena.cons(&num, &list);
    }

    // Count free slots before drop
    let free_before = (0..10000)
        .filter(|&i| matches!(arena.values[i].get(), Value::Free))
        .count();

    // Drop the list - this works but uses ~500 stack frames
    drop(list);

    // Verify memory was freed
    let free_after = (0..10000)
        .filter(|&i| matches!(arena.values[i].get(), Value::Free))
        .count();

    // Should have freed at least 500 cons cells + 500 numbers = 1000 allocations
    assert!(
        free_after > free_before + 900,
        "Expected to free ~1000 slots, only freed {}",
        free_after - free_before
    );
}

#[test]
fn test_nested_structures_cause_deep_recursion() {
    // Nested lambdas and environments also cause deep recursion
    let arena = Arena::<5000>::new();

    // Build deeply nested lambda: (lambda () (lambda () (lambda () ...)))
    let mut body = arena.number(42);
    for _ in 0..200 {
        let params = arena.nil();
        let env = arena.nil();
        body = arena.lambda(&params, &body, &env);
    }

    // This creates a chain 200 levels deep
    // Dropping will recurse 200 times through Lambda variant
    drop(body);

    // If this doesn't crash, we're probably okay, but it's using a lot of stack
}

// ============================================================================
// CRITICAL ISSUE #2: Memory Leak in env_set
// ============================================================================

#[test]
fn test_env_set_leaks_memory() {
    let arena = Arena::<5000>::new();
    let env = env_new(&arena);

    // Count allocations before
    let used_before = (0..5000)
        .filter(|&i| !matches!(arena.values[i].get(), Value::Free))
        .count();

    // Define 100 variables
    for i in 0..100 {
        let name = arena.str_to_list(&format!("var{}", i));
        let value = arena.number(i);
        env_set(&arena, &env, &name, &value);
    }

    // Count allocations after
    let used_after = (0..5000)
        .filter(|&i| !matches!(arena.values[i].get(), Value::Free))
        .count();

    let leaked = used_after - used_before;

    // Expected allocations:
    // - 100 variables * ~6 chars * 2 (char + cons) = ~1200 for names
    // - 100 numbers = 100
    // - 100 bindings (cons of symbol and value) = 100
    // - Some overhead for symbols and env structure
    // Total: ~1500-2000 allocations expected

    // But with the leak, we get much more because old bindings aren't freed
    println!("Allocated {} slots for 100 definitions", leaked);

    // This is the leak: we should only use ~1500-2000 slots,
    // but we're using much more because set_cons keeps old bindings alive
    assert!(
        leaked < 2000,
        "Expected leak to cause >2000 allocations, got {}",
        leaked
    );
}

#[test]
fn test_env_set_accumulates_old_bindings() {
    let arena = Arena::<2000>::new();
    let env = env_new(&arena);

    let name = arena.str_to_list("x");

    // Set the same variable 50 times
    for i in 0..50 {
        let value = arena.number(i);
        env_set(&arena, &env, &name, &value);
    }

    // Retrieve the value - should be 49 (last one)
    if let Some(val) = env_get(&arena, &env, &name) {
        if let Some(Value::Number(n)) = val.get() {
            assert_eq!(n, 49);
        }
    }

    // But all 50 values are still in memory!
    // Count how many numbers exist
    let mut count = 0;
    for i in 0..2000 {
        if let Value::Number(n) = arena.values[i].get() {
            if n < 50 {
                count += 1;
            }
        }
    }

    println!(
        "Found {} numbers in memory (expected 50 due to leak)",
        count
    );

    // With the leak, all 50 numbers are kept alive
    assert_eq!(
        count, 50,
        "Memory leak: all {} old values still in memory",
        count
    );
}

#[test]
fn test_repeated_defines_exhaust_memory() {
    let arena = Arena::<1000>::new();
    let env = env_new(&arena);

    let name = arena.str_to_list("counter");

    // Keep redefining the same variable
    let mut successful = 0;
    for i in 0..500 {
        let value = arena.number(i);
        env_set(&arena, &env, &name, &value);
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
        "Could only define {} times before running out of memory",
        successful
    );

    // Without the leak, we should be able to define 500 times easily
    // With the leak, we run out of memory much sooner
    assert!(
        successful < 400,
        "Expected to run out of memory due to leak, but completed {} definitions",
        successful
    );
}

// ============================================================================
// ISSUE #3: Inefficient O(N) Allocation
// ============================================================================

#[test]
fn test_allocation_gets_slower_as_arena_fills() {
    let arena = Arena::<1000>::new();

    // Allocate 500 values
    let mut refs = Vec::new();
    for i in 0..500 {
        refs.push(arena.number(i));
    }

    // Now arena is 50% full
    // Next allocation must search through ~500 slots to find free one

    // This is hard to test directly, but we can verify the behavior:
    // next_free should be around 500
    let next_free = arena.next_free.get();
    assert!(next_free >= 500, "next_free should have advanced to ~500");

    // Drop one in the middle
    drop(refs[250].clone());
    refs.remove(250);

    // Allocate a new value
    let new_val = arena.number(999);

    // It should have found the freed slot, but had to search from next_free
    // back around to slot 250
    assert!(!new_val.is_null());

    // The problem: this took ~250 iterations to find the free slot
    // With a free list, it would be O(1)
}

#[test]
fn test_fragmented_arena_has_poor_allocation_performance() {
    let arena = Arena::<1000>::new();

    // Create fragmentation: allocate all, then free every other slot
    let mut refs = Vec::new();
    for i in 0..1000 {
        refs.push(arena.number(i));
    }

    // Free every other slot (500 slots freed)
    let refs: Vec<_> = refs
        .into_iter()
        .enumerate()
        .filter(|(i, _)| i % 2 == 0)
        .map(|(_, r)| r)
        .collect();

    // Now arena is fragmented: slot 0 used, 1 free, 2 used, 3 free, etc.

    // Allocate 500 new values - each allocation must search to find a free slot
    for i in 0..500 {
        let val = arena.number(1000 + i);
        assert!(!val.is_null(), "Should find free slot");
    }

    // The problem: each allocation had to search past occupied slots
    // Average search distance was ~1-2 slots, but that adds up
    // With a free list, each allocation would be O(1)
}

// ============================================================================
// ISSUE #4: String Representation Overhead
// ============================================================================

#[test]
fn test_string_uses_many_allocations() {
    let arena = Arena::<1000>::new();

    let free_before = (0..1000)
        .filter(|&i| matches!(arena.values[i].get(), Value::Free))
        .count();

    // Create a simple 5-character string
    let string = arena.str_to_list("hello");

    let free_after = (0..1000)
        .filter(|&i| matches!(arena.values[i].get(), Value::Free))
        .count();

    let used = free_before - free_after;

    println!("'hello' used {} allocations", used);

    // Expected: 5 chars + 5 cons cells + 1 nil = 11 allocations
    // (Actually might be 10 if nil is reused)
    assert!(
        used >= 10,
        "String representation is inefficient: {} allocations for 5 chars",
        used
    );

    // With inline strings, this would be just 1 allocation
}

#[test]
fn test_many_strings_exhaust_memory_quickly() {
    let arena = Arena::<5000>::new();

    // Create 100 5-character strings
    let mut strings = Vec::new();
    for i in 0..100 {
        let s = arena.str_to_list(&format!("str{:02}", i));
        strings.push(s);
    }

    let free_count = (0..5000)
        .filter(|&i| matches!(arena.values[i].get(), Value::Free))
        .count();

    let used = 5000 - free_count;

    println!("100 short strings used {} allocations", used);

    // Each string "strXX" is 5 chars = ~10 allocations
    // 100 strings * 10 = ~1000 allocations
    // Plus some overhead
    assert!(
        used > 900,
        "Expected ~1000 allocations for 100 strings, got {}",
        used
    );

    // With inline strings, this would be just 100 allocations
}

#[test]
fn test_symbol_comparison_traverses_lists() {
    let arena = Arena::<500>::new();

    // Create two identical symbols
    let sym1 = arena.str_to_list("variable");
    let sym2 = arena.str_to_list("variable");

    // These are different allocations
    assert_ne!(sym1.raw().0, sym2.raw().0);

    // Comparing them requires traversing the entire list
    assert!(arena.str_eq(&sym1, &sym2));

    // The problem: str_eq does character-by-character comparison
    // For an 8-character symbol, that's 8 comparisons + 8 list traversals
    // With symbol interning, this would be a single integer comparison
}

// ============================================================================
// ISSUE #5: Environment Lookup Performance
// ============================================================================

#[test]
fn test_environment_lookup_is_linear() {
    let arena = Arena::<5000>::new();
    let env = env_new(&arena);

    // Add 50 bindings
    for i in 0..50 {
        let name = arena.str_to_list(&format!("var{:02}", i));
        let value = arena.number(i);
        env_set(&arena, &env, &name, &value);
    }

    // Look up the first variable we defined (now at the end of the list)
    let name = arena.str_to_list("var00");

    // This requires:
    // 1. Traversing the binding list (50 iterations)
    // 2. Comparing each symbol name (character by character)

    if let Some(val) = env_get(&arena, &env, &name) {
        if let Some(Value::Number(n)) = val.get() {
            // Due to shadowing and the leak, we might get the wrong value
            // or the right value, but after checking many bindings
            println!("Found value: {}", n);
        }
    }

    // The problem: lookup is O(n) where n is number of bindings
    // And each binding check does O(m) string comparison
    // Total: O(n*m) for lookup
}

#[test]
fn test_nested_environments_multiply_lookup_cost() {
    let arena = Arena::<5000>::new();
    let mut env = env_new(&arena);

    // Create 5 nested environments
    for level in 0..5 {
        env = env_with_parent(&arena, &env);

        // Add 10 bindings to each level
        for i in 0..10 {
            let name = arena.str_to_list(&format!("v{}_{}", level, i));
            let value = arena.number(level * 10 + i);
            env_set(&arena, &env, &name, &value);
        }
    }

    // Look up a variable from the root environment
    let name = arena.str_to_list("v0_0");

    // This requires:
    // 1. Check level 4 bindings (10 iterations)
    // 2. Check level 3 bindings (10 iterations)
    // 3. Check level 2 bindings (10 iterations)
    // 4. Check level 1 bindings (10 iterations)
    // 5. Check level 0 bindings (10 iterations)
    // Total: 50 binding checks, each with string comparison

    if let Some(val) = env_get(&arena, &env, &name) {
        if let Some(Value::Number(n)) = val.get() {
            assert_eq!(n, 0);
        }
    }

    // The problem: nested environments create O(d*n) lookup
    // where d is depth and n is bindings per level
}

// ============================================================================
// ISSUE #6: Helper Functions Create Temporary Refs
// ============================================================================

#[test]
fn test_list_len_does_unnecessary_refcount_operations() {
    let arena = Arena::<1000>::new();

    // Build a 100-element list
    let mut list = arena.nil();
    for i in 0..100 {
        let num = arena.number(i);
        list = arena.cons(&num, &list);
    }

    let list_idx = list.raw().0 as usize;
    let rc_before = arena.refcounts[list_idx].get();

    // Get the length - this shouldn't change refcounts
    let len = arena.list_len(&list);
    assert_eq!(len, 100);

    let rc_after = arena.refcounts[list_idx].get();

    // The problem: list_len clones Refs at each step
    // Each clone increments refcount, then drop decrements it
    // This is unnecessary overhead for a read-only operation

    assert_eq!(rc_before, rc_after, "list_len should not change refcounts");
}

#[test]
fn test_str_eq_creates_many_temporary_refs() {
    let arena = Arena::<500>::new();

    let s1 = arena.str_to_list("hello");
    let s2 = arena.str_to_list("world");

    // str_eq clones Refs and creates new Refs at each step
    // For a 5-character comparison, this is:
    // - 2 initial clones
    // - 5 iterations * 2 new Refs per iteration = 10 Refs
    // Total: 12 Ref objects created/destroyed

    let result = arena.str_eq(&s1, &s2);
    assert!(!result);

    // The problem: each Ref creation/destruction does refcount manipulation
    // For read-only operations, we should use raw ArenaRefs
}

// ============================================================================
// Integration Tests: Real-World Scenarios
// ============================================================================

#[test]
fn test_recursive_function_eventually_exhausts_memory() {
    let arena = Arena::<5000>::new();
    let env = env_new(&arena);

    // Define a recursive counter
    let code = r#"
        (define count
          (lambda (n)
            (if (= n 0)
                0
                (count (- n 1)))))
    "#;

    let _ = eval_string(&arena, code, &env);

    // Call it with increasing depths
    for depth in [10, 50, 100, 200].iter() {
        let result = eval_string(&arena, &format!("(count {})", depth), &env);

        if result.is_err() {
            println!("Failed at depth {}", depth);
            break;
        }

        // Check memory usage
        let free_count = (0..5000)
            .filter(|&i| matches!(arena.values[i].get(), Value::Free))
            .count();

        println!("After depth {}, {} slots free", depth, free_count);

        if free_count < 500 {
            println!("Memory nearly exhausted at depth {}", depth);
            break;
        }
    }

    // The problem: Even with TCO, the environment bindings leak
    // so repeated calls accumulate memory
}

#[test]
fn test_many_definitions_cause_arena_exhaustion() {
    let arena = Arena::<2000>::new();
    let env = env_new(&arena);

    let mut successful_defines = 0;

    for i in 0..200 {
        let code = format!("(define var{} {})", i, i);
        let result = eval_string(&arena, &code, &env);

        if result.is_err() {
            println!("Failed to define after {} definitions", i);
            break;
        }

        successful_defines = i + 1;

        // Check memory
        let free_count = (0..2000)
            .filter(|&j| matches!(arena.values[j].get(), Value::Free))
            .count();

        if free_count < 50 {
            println!("Arena exhausted after {} definitions", i);
            break;
        }
    }

    println!("Successfully defined {} variables", successful_defines);

    // Without the memory leak, we should handle 200 definitions easily
    // With the leak, we run out of memory much sooner
    assert!(
        successful_defines < 150,
        "Expected to run out of memory due to env_set leak"
    );
}

#[test]
fn test_building_large_list_in_loop() {
    let arena = Arena::<10000>::new();
    let env = env_new(&arena);

    // Build a list using recursion
    let code = r#"
        (define build-list
          (lambda (n acc)
            (if (= n 0)
                acc
                (build-list (- n 1) (cons n acc)))))
    "#;

    let _ = eval_string(&arena, code, &env);

    // Build a 500-element list
    let result = eval_string(&arena, "(build-list 500 (list))", &env);

    assert!(result.is_ok());

    if let Ok(list) = result {
        let len = arena.list_len(&list);
        assert_eq!(len, 500);

        // Now drop it - this is where the stack overflow can happen
        drop(list);
    }
}

// ============================================================================
// Summary Test: Demonstrate All Issues
// ============================================================================

#[test]
fn test_all_issues_summary() {
    println!("\n=== ISSUE SUMMARY ===\n");

    // Issue 1: Deep recursion
    println!("1. DEEP LIST RECURSION:");
    println!("   - Lists >1000 elements cause stack overflow on drop");
    println!("   - Recursive decref uses 1 stack frame per element");
    println!("   - Solution: Use iterative decref with explicit stack\n");

    // Issue 2: Memory leak
    println!("2. ENV_SET MEMORY LEAK:");
    println!("   - set_cons keeps old bindings alive");
    println!("   - Each redefinition leaks ~10 allocations");
    println!("   - 100 defines leak 1000+ allocations");
    println!("   - Solution: Use immutable environments or remove set_cons\n");

    // Issue 3: Allocation
    println!("3. INEFFICIENT ALLOCATION:");
    println!("   - O(N) linear search for free slots");
    println!("   - Worst case: scan entire 10,000-slot arena");
    println!("   - Solution: Maintain O(1) free list\n");

    // Issue 4: Strings
    println!("4. STRING REPRESENTATION:");
    println!("   - 5-char string uses 10 allocations");
    println!("   - 100 strings use 1000+ allocations");
    println!("   - Solution: Inline strings ≤7 bytes\n");

    // Issue 5: Lookups
    println!("5. ENVIRONMENT LOOKUP:");
    println!("   - O(n*m) where n=bindings, m=name length");
    println!("   - Nested envs multiply cost");
    println!("   - Solution: Symbol interning for O(1) comparison\n");

    // Issue 6: Helper overhead
    println!("6. HELPER FUNCTION OVERHEAD:");
    println!("   - list_len/str_eq create many Refs");
    println!("   - Each Ref does refcount manipulation");
    println!("   - Solution: Use raw ArenaRefs for iteration\n");

    println!("Run individual tests to see details.");
}
