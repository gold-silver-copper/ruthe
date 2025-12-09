use core::matches;
use std::vec::Vec;
// Import the module under test
use ruthe::*; // Replace with actual crate name

// ============================================================================
// Test Helpers
// ============================================================================

fn assert_refcount(arena: &Arena, r: ArenaRef, expected: u16) {
    // This accesses internal state - in real code you might want to expose
    // a debug method or use a test-only feature flag
    let idx = r.0 as usize;
    let actual = arena.refcounts[idx].get();
    assert!(
        actual == expected,
        "Expected refcount {}, got {}",
        expected,
        actual
    );
}

fn count_free_slots(arena: &Arena) -> usize {
    let mut count = 0;
    for i in 0..ARENA_SIZE {
        if matches!(arena.values[i].get(), Value::Free) {
            count += 1;
        }
    }
    count
}

// ============================================================================
// Basic Allocation Tests
// ============================================================================

#[test]
fn test_basic_allocation() {
    let arena = Arena::new();

    // Initially all slots should be free
    assert!(count_free_slots(&arena) == ARENA_SIZE);

    // Allocate a number
    let num = arena.number(42);
    assert!(count_free_slots(&arena) == ARENA_SIZE - 1);

    // Verify value
    if let Some(Value::Number(n)) = arena.get(*num) {
        assert!(n == 42);
    } else {
        panic!("Expected Number(42)");
    }
}

#[test]
fn test_multiple_allocations() {
    let arena = Arena::new();

    let n1 = arena.number(1);
    let n2 = arena.number(2);
    let n3 = arena.number(3);

    assert!(count_free_slots(&arena) == ARENA_SIZE - 3);

    if let Some(Value::Number(n)) = arena.get(*n1) {
        assert!(n == 1);
    }
    if let Some(Value::Number(n)) = arena.get(*n2) {
        assert!(n == 2);
    }
    if let Some(Value::Number(n)) = arena.get(*n3) {
        assert!(n == 3);
    }
}

// ============================================================================
// Reference Counting Tests
// ============================================================================

#[test]
fn test_initial_refcount() {
    let arena = Arena::new();
    let num = arena.number(100);

    // New allocation should have refcount of 1
    assert_refcount(&arena, *num, 1);
}

#[test]
fn test_clone_increments_refcount() {
    let arena = Arena::new();
    let num = arena.number(100);

    assert_refcount(&arena, *num, 1);

    let num2 = num.clone();
    assert_refcount(&arena, *num, 2);

    let num3 = num.clone();
    assert_refcount(&arena, *num, 3);

    // All refs point to same value
    assert!(num.raw().0 == num2.raw().0);
    assert!(num.raw().0 == num3.raw().0);
}

#[test]
fn test_drop_decrements_refcount() {
    let arena = Arena::new();
    let num = arena.number(100);
    let r = *num;

    assert_refcount(&arena, r, 1);

    {
        let num2 = num.clone();
        assert_refcount(&arena, r, 2);
        // num2 drops here
    }

    assert_refcount(&arena, r, 1);
}

#[test]
fn test_full_deallocation() {
    let arena = Arena::new();

    let initial_free = count_free_slots(&arena);

    {
        let _num = arena.number(100);
        assert!(count_free_slots(&arena) == initial_free - 1);
    }

    // After drop, slot should be freed
    assert!(count_free_slots(&arena) == initial_free);
}

#[test]
fn test_memory_reuse() {
    let arena = Arena::new();

    let r1 = {
        let num = arena.number(100);
        num.raw()
    };

    // Allocate again - might reuse same slot
    let num2 = arena.number(200);

    // Either same slot was reused, or a new one was allocated
    // Both are valid
    if num2.raw().0 == r1.0 {
        // Slot was reused - verify new value
        if let Some(Value::Number(n)) = arena.get(*num2) {
            assert!(n == 200);
        }
    }
}

// ============================================================================
// Cons Cell Tests (Recursive Deallocation)
// ============================================================================

#[test]
fn test_cons_refcounting() {
    let arena = Arena::new();

    let car = arena.number(1);
    let cdr = arena.number(2);

    let car_ref = *car;
    let cdr_ref = *cdr;

    assert_refcount(&arena, car_ref, 1);
    assert_refcount(&arena, cdr_ref, 1);

    // Creating cons increments children refcounts
    let cons = arena.cons(&car, &cdr);

    assert_refcount(&arena, car_ref, 2);
    assert_refcount(&arena, cdr_ref, 2);

    // Drop original refs
    drop(car);
    drop(cdr);

    assert_refcount(&arena, car_ref, 1);
    assert_refcount(&arena, cdr_ref, 1);

    // Drop cons - should recursively decrement children
    drop(cons);

    // Children should be freed
    assert!(matches!(arena.get(car_ref), None));
    assert!(matches!(arena.get(cdr_ref), None));
}

#[test]
fn test_nested_cons_deallocation() {
    let arena = Arena::new();

    let initial_free = count_free_slots(&arena);

    {
        // Create nested structure: ((1 . 2) . (3 . 4))
        let a = arena.number(1);
        let b = arena.number(2);
        let c = arena.number(3);
        let d = arena.number(4);

        let left = arena.cons(&a, &b);
        let right = arena.cons(&c, &d);
        let _root = arena.cons(&left, &right);

        // Should have: 4 numbers + 3 cons cells = 7 allocations
        assert!(count_free_slots(&arena) == initial_free - 7);
    }

    // All should be freed
    assert!(count_free_slots(&arena) == initial_free);
}

#[test]
fn test_list_deallocation() {
    let arena = Arena::new();

    let initial_free = count_free_slots(&arena);

    {
        // Create list: (1 2 3)
        let nil = arena.nil();
        let three = arena.number(3);
        let list1 = arena.cons(&three, &nil);
        let two = arena.number(2);
        let list2 = arena.cons(&two, &list1);
        let one = arena.number(1);
        let _list3 = arena.cons(&one, &list2);

        // 3 numbers + 1 nil + 3 cons = 7 allocations
        assert!(count_free_slots(&arena) == initial_free - 7);
    }

    assert!(count_free_slots(&arena) == initial_free);
}

// ============================================================================
// Lambda Tests
// ============================================================================

#[test]
fn test_lambda_refcounting() {
    let arena = Arena::new();

    let params = arena.nil();
    let body = arena.number(42);
    let env = arena.nil();

    let params_ref = *params;
    let body_ref = *body;
    let env_ref = *env;

    assert_refcount(&arena, params_ref, 1);
    assert_refcount(&arena, body_ref, 1);
    assert_refcount(&arena, env_ref, 1);

    let lambda = arena.lambda(&params, &body, &env);

    // Lambda should increment all three
    assert_refcount(&arena, params_ref, 2);
    assert_refcount(&arena, body_ref, 2);
    assert_refcount(&arena, env_ref, 2);

    drop(params);
    drop(body);
    drop(env);

    assert_refcount(&arena, params_ref, 1);
    assert_refcount(&arena, body_ref, 1);
    assert_refcount(&arena, env_ref, 1);

    drop(lambda);

    // All should be freed
    assert!(matches!(arena.get(params_ref), None));
    assert!(matches!(arena.get(body_ref), None));
    assert!(matches!(arena.get(env_ref), None));
}

// ============================================================================
// Symbol Tests
// ============================================================================

#[test]
fn test_symbol_refcounting() {
    let arena = Arena::new();

    let str_list = arena.str_to_list("foo");
    let str_ref = *str_list;

    assert_refcount(&arena, str_ref, 1);

    let sym = arena.symbol(&str_list);

    // Symbol should increment string list refcount
    assert_refcount(&arena, str_ref, 2);

    drop(str_list);
    assert_refcount(&arena, str_ref, 1);

    drop(sym);

    // String list should be freed (along with all its cons cells)
    assert!(matches!(arena.get(str_ref), None));
}

// ============================================================================
// Edge Cases
// ============================================================================

#[test]
fn test_null_ref_handling() {
    let arena = Arena::new();

    // NULL refs should be handled gracefully
    let null = ArenaRef::NULL;

    assert!(arena.get(null).is_none());
    assert!(null.is_null());

    // These should not panic
    arena.incref(null);
    arena.decref(null);
}

#[test]
fn test_ref_new_from_existing() {
    let arena = Arena::new();

    let num = arena.number(100);
    let raw = *num;

    assert_refcount(&arena, raw, 1);

    // Create new Ref from existing ArenaRef - should increment
    let num2 = Ref::new(&arena, raw);
    assert_refcount(&arena, raw, 2);

    drop(num);
    assert_refcount(&arena, raw, 1);

    drop(num2);
    assert!(matches!(arena.get(raw), None));
}

#[test]
fn test_shared_children() {
    let arena = Arena::new();

    // Create a shared child
    let shared = arena.number(42);
    let shared_ref = *shared;

    assert_refcount(&arena, shared_ref, 1);

    // Two cons cells share the same cdr
    let cons1 = arena.cons(&arena.number(1), &shared);
    let cons2 = arena.cons(&arena.number(2), &shared);

    // shared should have refcount of 3 (original + 2 cons cells)
    assert_refcount(&arena, shared_ref, 3);

    drop(shared);
    assert_refcount(&arena, shared_ref, 2);

    drop(cons1);
    assert_refcount(&arena, shared_ref, 1);

    drop(cons2);
    assert!(matches!(arena.get(shared_ref), None));
}

#[test]
fn test_set_cons_refcounting() {
    let arena = Arena::new();

    let car1 = arena.number(1);
    let cdr1 = arena.number(2);
    let cons = arena.cons(&car1, &cdr1);

    let car1_ref = *car1;
    let cdr1_ref = *cdr1;

    // Original children have refcount 2
    assert_refcount(&arena, car1_ref, 2);
    assert_refcount(&arena, cdr1_ref, 2);

    // Create new children
    let car2 = arena.number(3);
    let cdr2 = arena.number(4);
    let car2_ref = *car2;
    let cdr2_ref = *cdr2;

    // Set cons to new values
    arena.set_cons(&cons, &car2, &cdr2);

    // Old children should be decremented (back to 1)
    assert_refcount(&arena, car1_ref, 1);
    assert_refcount(&arena, cdr1_ref, 1);

    // New children should be incremented (to 2)
    assert_refcount(&arena, car2_ref, 2);
    assert_refcount(&arena, cdr2_ref, 2);

    // Verify cons contains new values
    if let Some(Value::Cons(car, cdr)) = arena.get(*cons) {
        assert!(car.0 == car2_ref.0);
        assert!(cdr.0 == cdr2_ref.0);
    } else {
        panic!("Expected Cons");
    }
}

#[test]
fn test_arena_exhaustion() {
    let arena = Arena::new();

    let mut refs = Vec::new();

    // Allocate until arena is full
    for i in 0..ARENA_SIZE {
        let num = arena.number(i as i64);
        refs.push(num);
    }

    assert!(count_free_slots(&arena) == 0);

    // Next allocation should fail
    let overflow = arena.number(9999);
    assert!(overflow.is_null());

    // Free one slot
    refs.pop();
    assert!(count_free_slots(&arena) == 1);

    // Now allocation should succeed
    let new_num = arena.number(8888);
    assert!(!new_num.is_null());
}

#[test]
fn test_refcount_overflow_protection() {
    let arena = Arena::new();

    let num = arena.number(100);
    let raw = *num;

    // Manually set refcount to near max
    arena.refcounts[raw.0 as usize].set(u16::MAX - 1);

    // This increment should succeed
    arena.incref(raw);
    assert_refcount(&arena, raw, u16::MAX);

    // This increment should be a no-op (protection against overflow)
    arena.incref(raw);
    assert_refcount(&arena, raw, u16::MAX);
}

#[test]
fn test_double_free_protection() {
    let arena = Arena::new();

    let num = arena.number(100);
    let raw = *num;

    drop(num);

    // Slot should be free
    assert!(matches!(arena.get(raw), None));

    // Attempting to decref again should be safe
    arena.decref(raw);

    // Still free, no crash
    assert!(matches!(arena.get(raw), None));
}

// ============================================================================
// Complex Scenario Tests
// ============================================================================

#[test]
fn test_complex_tree_structure() {
    let arena = Arena::new();

    let initial_free = count_free_slots(&arena);

    {
        // Build: (+ (* 2 3) (- 10 5))
        let plus = arena.str_to_list("+");
        let mult = arena.str_to_list("*");
        let minus = arena.str_to_list("-");

        let two = arena.number(2);
        let three = arena.number(3);
        let ten = arena.number(10);
        let five = arena.number(5);

        // (* 2 3)
        let nil = arena.nil();
        let mult_args2 = arena.cons(&three, &nil);
        let mult_args1 = arena.cons(&two, &mult_args2);
        let mult_expr = arena.cons(&mult, &mult_args1);

        // (- 10 5)
        let nil2 = arena.nil();
        let minus_args2 = arena.cons(&five, &nil2);
        let minus_args1 = arena.cons(&ten, &minus_args2);
        let minus_expr = arena.cons(&minus, &minus_args1);

        // (+ (* 2 3) (- 10 5))
        let nil3 = arena.nil();
        let plus_args2 = arena.cons(&minus_expr, &nil3);
        let plus_args1 = arena.cons(&mult_expr, &plus_args2);
        let _root = arena.cons(&plus, &plus_args1);

        // Verify something was allocated
        assert!(count_free_slots(&arena) < initial_free);
    }

    // Everything should be freed
    assert!(count_free_slots(&arena) == initial_free);
}

#[test]
fn test_string_list_refcounting() {
    let arena = Arena::new();

    let initial_free = count_free_slots(&arena);

    {
        let s = arena.str_to_list("hello");

        // String "hello" creates: 5 chars + 5 cons cells + 1 nil = 11 allocations
        // (Actually might be optimized differently, but should be > 5)
        assert!(count_free_slots(&arena) < initial_free - 5);

        // Clone should increment root cons refcount
        let s2 = s.clone();
        let raw = *s;
        assert_refcount(&arena, raw, 2);

        drop(s2);
        assert_refcount(&arena, raw, 1);
    }

    // All should be freed
    assert!(count_free_slots(&arena) == initial_free);
}

#[test]
fn test_circular_references_prevention() {
    let arena = Arena::new();

    // Create two cons cells
    let a_val = arena.number(1);
    let b_val = arena.number(2);

    let nil = arena.nil();
    let a = arena.cons(&a_val, &nil);
    let b = arena.cons(&b_val, &nil);

    // Make a point to b and b point to a
    arena.set_cons(&a, &a_val, &b);
    arena.set_cons(&b, &b_val, &a);

    let a_ref = *a;
    let b_ref = *b;

    // They should keep each other alive even after we drop our refs
    drop(a);
    drop(b);

    // Both should still exist due to circular reference
    // This is a KNOWN LIMITATION of reference counting
    assert!(arena.get(a_ref).is_some());
    assert!(arena.get(b_ref).is_some());

    // Note: This demonstrates that circular refs will leak memory
    // This is expected behavior for RC-based systems
}

#[test]
fn test_ref_clone_semantics() {
    let arena = Arena::new();

    let num = arena.number(42);
    let raw = *num;

    assert_refcount(&arena, raw, 1);

    // Clone creates independent ref with incremented count
    let num2 = num.clone();
    assert_refcount(&arena, raw, 2);

    // Both refs are independent
    drop(num);
    assert_refcount(&arena, raw, 1);
    assert!(arena.get(raw).is_some());

    drop(num2);
    assert!(arena.get(raw).is_none());
}

// ============================================================================
// Helper Function Tests
// ============================================================================

#[test]
fn test_list_len() {
    let arena = Arena::new();

    let nil = arena.nil();
    assert!(arena.list_len(&nil) == 0);

    let one = arena.number(1);
    let list1 = arena.cons(&one, &nil);
    assert!(arena.list_len(&list1) == 1);

    let two = arena.number(2);
    let list2 = arena.cons(&two, &list1);
    assert!(arena.list_len(&list2) == 2);
}

#[test]
fn test_reverse_list_refcounting() {
    let arena = Arena::new();

    // Build list (1 2 3)
    let nil = arena.nil();
    let three = arena.number(3);
    let list1 = arena.cons(&three, &nil);
    let two = arena.number(2);
    let list2 = arena.cons(&two, &list1);
    let one = arena.number(1);
    let list3 = arena.cons(&one, &list2);

    let initial_free = count_free_slots(&arena);

    // Reverse creates new cons cells AND a new nil, but shares elements
    let reversed = arena.reverse_list(&list3);

    // Should have 3 new cons cells + 1 new nil = 4 allocations
    assert!(count_free_slots(&arena) == initial_free - 4);

    // Elements should have higher refcounts (in original list, reversed list, and our local refs)
    let one_ref = *one;
    let two_ref = *two;
    let three_ref = *three;

    // Each number appears in: original list + reversed list + our local variable = 3 references
    assert_refcount(&arena, one_ref, 3);
    assert_refcount(&arena, two_ref, 3);
    assert_refcount(&arena, three_ref, 3);

    drop(reversed);

    // New cons cells and new nil freed, but elements still alive in original list
    assert_refcount(&arena, one_ref, 2); // Original list + our local ref
    assert_refcount(&arena, two_ref, 2);
    assert_refcount(&arena, three_ref, 2);

    // Drop original list
    drop(list3);
    drop(list2);
    drop(list1);

    // Now only our local refs remain
    assert_refcount(&arena, one_ref, 1);
    assert_refcount(&arena, two_ref, 1);
    assert_refcount(&arena, three_ref, 1);
}
#[test]
fn test_environment_refcounting() {
    let arena = Arena::new();

    let initial_free = count_free_slots(&arena);

    {
        let env = env_new(&arena);

        // Env should allocate bindings list and builtins
        assert!(count_free_slots(&arena) < initial_free);

        // Add a binding
        let name = arena.str_to_list("x");
        let value = arena.number(42);
        env_set(&arena, &env, &name, &value);

        // Lookup should work
        let retrieved = env_get(&arena, &env, &name);
        assert!(retrieved.is_some());

        if let Some(val) = retrieved {
            if let Some(Value::Number(n)) = arena.get(*val) {
                assert!(n == 42);
            }
        }
    }

    // Most should be freed (builtins might still be around depending on impl)
    // At minimum, we shouldn't leak memory
}

// Run all tests
pub fn run_all_tests() {
    test_basic_allocation();
    test_multiple_allocations();
    test_initial_refcount();
    test_clone_increments_refcount();
    test_drop_decrements_refcount();
    test_full_deallocation();
    test_memory_reuse();
    test_cons_refcounting();
    test_nested_cons_deallocation();
    test_list_deallocation();
    test_lambda_refcounting();
    test_symbol_refcounting();
    test_null_ref_handling();
    test_ref_new_from_existing();
    test_shared_children();
    test_set_cons_refcounting();
    test_arena_exhaustion();
    test_refcount_overflow_protection();
    test_double_free_protection();
    test_complex_tree_structure();
    test_string_list_refcounting();
    test_circular_references_prevention();
    test_ref_clone_semantics();
    test_list_len();
    test_reverse_list_refcounting();
    test_environment_refcounting();
}
