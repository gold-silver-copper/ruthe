use core::matches;
use ruthe::*;
use std::vec::Vec;

const ARENA_SIZE: usize = DEFAULT_ARENA_SIZE;

// ============================================================================
// Test Helpers
// ============================================================================

fn assert_refcount<const N: usize>(arena: &Arena<N>, r: ArenaRef, expected: u32) {
    let idx = r.0 as usize;
    let actual = arena.refcounts[idx].get();
    assert!(
        actual == expected,
        "Expected refcount {}, got {}",
        expected,
        actual
    );
}

fn count_free_slots<const N: usize>(arena: &Arena<N>) -> usize {
    let mut count = 0;
    for i in 0..N {
        if matches!(arena.values[i].get(), Value::Free) {
            count += 1;
        }
    }
    count
}

fn count_allocated_slots<const N: usize>(arena: &Arena<N>) -> usize {
    N - count_free_slots(arena)
}

// ============================================================================
// Basic Allocation Tests
// ============================================================================

#[test]
fn test_basic_allocation() {
    let arena = Arena::<ARENA_SIZE>::new();
    assert!(count_free_slots(&arena) == ARENA_SIZE);

    let num = arena.number(42);
    assert!(count_free_slots(&arena) == ARENA_SIZE - 1);

    if let Some(Value::Number(n)) = arena.get(num.raw()) {
        assert!(n == 42);
    } else {
        panic!("Expected Number(42)");
    }
}

#[test]
fn test_multiple_allocations() {
    let arena = Arena::<ARENA_SIZE>::new();
    let n1 = arena.number(1);
    let n2 = arena.number(2);
    let n3 = arena.number(3);

    assert!(count_free_slots(&arena) == ARENA_SIZE - 3);

    if let Some(Value::Number(n)) = arena.get(n1.raw()) {
        assert!(n == 1);
    }
    if let Some(Value::Number(n)) = arena.get(n2.raw()) {
        assert!(n == 2);
    }
    if let Some(Value::Number(n)) = arena.get(n3.raw()) {
        assert!(n == 3);
    }
}

// ============================================================================
// Reference Counting Tests
// ============================================================================

#[test]
fn test_initial_refcount() {
    let arena = Arena::<ARENA_SIZE>::new();
    let num = arena.number(100);
    assert_refcount(&arena, num.raw(), 1);
}

#[test]
fn test_clone_increments_refcount() {
    let arena = Arena::<ARENA_SIZE>::new();
    let num = arena.number(100);
    assert_refcount(&arena, num.raw(), 1);

    let num2 = num.clone();
    assert_refcount(&arena, num.raw(), 2);

    let num3 = num.clone();
    assert_refcount(&arena, num.raw(), 3);

    assert!(num.raw().0 == num2.raw().0);
    assert!(num.raw().0 == num3.raw().0);
}

#[test]
fn test_drop_decrements_refcount() {
    let arena = Arena::<ARENA_SIZE>::new();
    let num = arena.number(100);
    let r = num.raw();

    assert_refcount(&arena, r, 1);
    {
        let num2 = num.clone();
        assert_refcount(&arena, r, 2);
    }
    assert_refcount(&arena, r, 1);
}

#[test]
fn test_full_deallocation() {
    let arena = Arena::<ARENA_SIZE>::new();
    let initial_free = count_free_slots(&arena);

    {
        let _num = arena.number(100);
        assert!(count_free_slots(&arena) == initial_free - 1);
    }

    assert!(count_free_slots(&arena) == initial_free);
}

#[test]
fn test_memory_reuse() {
    let arena = Arena::<ARENA_SIZE>::new();
    let r1 = {
        let num = arena.number(100);
        num.raw()
    };

    let num2 = arena.number(200);
    if num2.raw().0 == r1.0 {
        if let Some(Value::Number(n)) = arena.get(num2.raw()) {
            assert!(n == 200);
        }
    }
}

// ============================================================================
// Cons Cell Tests
// ============================================================================

#[test]
fn test_cons_refcounting() {
    let arena = Arena::<ARENA_SIZE>::new();
    let car = arena.number(1);
    let cdr = arena.number(2);
    let car_ref = car.raw();
    let cdr_ref = cdr.raw();

    assert_refcount(&arena, car_ref, 1);
    assert_refcount(&arena, cdr_ref, 1);

    let cons = arena.cons(&car, &cdr);
    assert_refcount(&arena, car_ref, 2);
    assert_refcount(&arena, cdr_ref, 2);

    drop(car);
    drop(cdr);
    assert_refcount(&arena, car_ref, 1);
    assert_refcount(&arena, cdr_ref, 1);

    drop(cons);
    assert!(matches!(arena.get(car_ref), None));
    assert!(matches!(arena.get(cdr_ref), None));
}

#[test]
fn test_nested_cons_deallocation() {
    let arena = Arena::<ARENA_SIZE>::new();
    let initial_free = count_free_slots(&arena);

    {
        let a = arena.number(1);
        let b = arena.number(2);
        let c = arena.number(3);
        let d = arena.number(4);
        let left = arena.cons(&a, &b);
        let right = arena.cons(&c, &d);
        let _root = arena.cons(&left, &right);

        assert!(count_free_slots(&arena) == initial_free - 7);
    }

    assert!(count_free_slots(&arena) == initial_free);
}

#[test]
fn test_list_deallocation() {
    let arena = Arena::<ARENA_SIZE>::new();
    let initial_free = count_free_slots(&arena);

    {
        let nil = arena.nil();
        let three = arena.number(3);
        let list1 = arena.cons(&three, &nil);
        let two = arena.number(2);
        let list2 = arena.cons(&two, &list1);
        let one = arena.number(1);
        let _list3 = arena.cons(&one, &list2);

        assert!(count_free_slots(&arena) == initial_free - 7);
    }

    assert!(count_free_slots(&arena) == initial_free);
}

// ============================================================================
// Lambda Tests
// ============================================================================

#[test]
fn test_lambda_refcounting() {
    let arena = Arena::<ARENA_SIZE>::new();
    let params = arena.nil();
    let body = arena.number(42);
    let env = arena.nil();

    let params_ref = params.raw();
    let body_ref = body.raw();
    let env_ref = env.raw();

    assert_refcount(&arena, params_ref, 1);
    assert_refcount(&arena, body_ref, 1);
    assert_refcount(&arena, env_ref, 1);

    let lambda = arena.lambda(&params, &body, &env);

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

    assert!(matches!(arena.get(params_ref), None));
    assert!(matches!(arena.get(body_ref), None));
    assert!(matches!(arena.get(env_ref), None));
}

// ============================================================================
// Symbol Tests
// ============================================================================

#[test]
fn test_symbol_refcounting() {
    let arena = Arena::<ARENA_SIZE>::new();
    let str_list = arena.str_to_list("foo");
    let str_ref = str_list.raw();

    assert_refcount(&arena, str_ref, 1);

    let sym = arena.symbol(&str_list);
    assert_refcount(&arena, str_ref, 2);

    drop(str_list);
    assert_refcount(&arena, str_ref, 1);

    drop(sym);
    assert!(matches!(arena.get(str_ref), None));
}

// ============================================================================
// Edge Cases
// ============================================================================

#[test]
fn test_null_ref_handling() {
    let arena = Arena::<ARENA_SIZE>::new();
    let null = ArenaRef::NULL;

    assert!(arena.get(null).is_none());
    assert!(null.is_null());

    arena.incref(null);
    arena.decref(null);
}

#[test]
fn test_ref_new_from_existing() {
    let arena = Arena::<ARENA_SIZE>::new();
    let num = arena.number(100);
    let raw = num.raw();

    assert_refcount(&arena, raw, 1);

    let num2 = Ref::new(&arena, raw);
    assert_refcount(&arena, raw, 2);

    drop(num);
    assert_refcount(&arena, raw, 1);

    drop(num2);
    assert!(matches!(arena.get(raw), None));
}

#[test]
fn test_shared_children() {
    let arena = Arena::<ARENA_SIZE>::new();
    let shared = arena.number(42);
    let shared_ref = shared.raw();

    assert_refcount(&arena, shared_ref, 1);

    let cons1 = arena.cons(&arena.number(1), &shared);
    let cons2 = arena.cons(&arena.number(2), &shared);

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
    let arena = Arena::<ARENA_SIZE>::new();
    let car1 = arena.number(1);
    let cdr1 = arena.number(2);
    let cons = arena.cons(&car1, &cdr1);

    let car1_ref = car1.raw();
    let cdr1_ref = cdr1.raw();

    assert_refcount(&arena, car1_ref, 2);
    assert_refcount(&arena, cdr1_ref, 2);

    let car2 = arena.number(3);
    let cdr2 = arena.number(4);
    let car2_ref = car2.raw();
    let cdr2_ref = cdr2.raw();

    arena.set_cons(&cons, &car2, &cdr2);

    assert_refcount(&arena, car1_ref, 1);
    assert_refcount(&arena, cdr1_ref, 1);
    assert_refcount(&arena, car2_ref, 2);
    assert_refcount(&arena, cdr2_ref, 2);

    if let Some(Value::Cons(car, cdr)) = arena.get(cons.raw()) {
        assert!(car.0 == car2_ref.0);
        assert!(cdr.0 == cdr2_ref.0);
    } else {
        panic!("Expected Cons");
    }
}

#[test]
fn test_refcount_overflow_protection() {
    let arena = Arena::<ARENA_SIZE>::new();
    let num = arena.number(100);
    let raw = num.raw();

    arena.refcounts[raw.0 as usize].set(u32::MAX - 1);

    arena.incref(raw);
    assert_refcount(&arena, raw, u32::MAX);

    arena.incref(raw);
    assert_refcount(&arena, raw, u32::MAX);
}

#[test]
fn test_double_free_protection() {
    let arena = Arena::<ARENA_SIZE>::new();
    let num = arena.number(100);
    let raw = num.raw();

    drop(num);
    assert!(matches!(arena.get(raw), None));

    arena.decref(raw);
    assert!(matches!(arena.get(raw), None));
}

// ============================================================================
// Complex Scenario Tests
// ============================================================================

#[test]
fn test_complex_tree_structure() {
    let arena = Arena::<ARENA_SIZE>::new();
    let initial_free = count_free_slots(&arena);

    {
        let plus = arena.str_to_list("+");
        let mult = arena.str_to_list("*");
        let minus = arena.str_to_list("-");

        let two = arena.number(2);
        let three = arena.number(3);
        let ten = arena.number(10);
        let five = arena.number(5);

        let nil = arena.nil();
        let mult_args2 = arena.cons(&three, &nil);
        let mult_args1 = arena.cons(&two, &mult_args2);
        let mult_expr = arena.cons(&mult, &mult_args1);

        let nil2 = arena.nil();
        let minus_args2 = arena.cons(&five, &nil2);
        let minus_args1 = arena.cons(&ten, &minus_args2);
        let minus_expr = arena.cons(&minus, &minus_args1);

        let nil3 = arena.nil();
        let plus_args2 = arena.cons(&minus_expr, &nil3);
        let plus_args1 = arena.cons(&mult_expr, &plus_args2);
        let _root = arena.cons(&plus, &plus_args1);

        assert!(count_free_slots(&arena) < initial_free);
    }

    assert!(count_free_slots(&arena) == initial_free);
}

#[test]
fn test_string_list_refcounting() {
    let arena = Arena::<ARENA_SIZE>::new();
    let initial_free = count_free_slots(&arena);

    {
        let s = arena.str_to_list("hello");
        assert!(count_free_slots(&arena) < initial_free - 5);

        let s2 = s.clone();
        let raw = s.raw();
        assert_refcount(&arena, raw, 2);

        drop(s2);
        assert_refcount(&arena, raw, 1);
    }

    assert!(count_free_slots(&arena) == initial_free);
}

#[test]
fn test_circular_references_prevention() {
    let arena = Arena::<ARENA_SIZE>::new();
    let a_val = arena.number(1);
    let b_val = arena.number(2);

    let nil = arena.nil();
    let a = arena.cons(&a_val, &nil);
    let b = arena.cons(&b_val, &nil);

    arena.set_cons(&a, &a_val, &b);
    arena.set_cons(&b, &b_val, &a);

    let a_ref = a.raw();
    let b_ref = b.raw();

    drop(a);
    drop(b);

    assert!(arena.get(a_ref).is_some());
    assert!(arena.get(b_ref).is_some());
}

#[test]
fn test_ref_clone_semantics() {
    let arena = Arena::<ARENA_SIZE>::new();
    let num = arena.number(42);
    let raw = num.raw();

    assert_refcount(&arena, raw, 1);

    let num2 = num.clone();
    assert_refcount(&arena, raw, 2);

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
    let arena = Arena::<ARENA_SIZE>::new();
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
    let arena = Arena::<ARENA_SIZE>::new();
    let nil = arena.nil();
    let three = arena.number(3);
    let list1 = arena.cons(&three, &nil);
    let two = arena.number(2);
    let list2 = arena.cons(&two, &list1);
    let one = arena.number(1);
    let list3 = arena.cons(&one, &list2);

    let initial_free = count_free_slots(&arena);

    let reversed = arena.reverse_list(&list3);

    assert!(count_free_slots(&arena) == initial_free - 4);

    let one_ref = one.raw();
    let two_ref = two.raw();
    let three_ref = three.raw();

    assert_refcount(&arena, one_ref, 3);
    assert_refcount(&arena, two_ref, 3);
    assert_refcount(&arena, three_ref, 3);

    drop(reversed);

    assert_refcount(&arena, one_ref, 2);
    assert_refcount(&arena, two_ref, 2);
    assert_refcount(&arena, three_ref, 2);

    drop(list3);
    drop(list2);
    drop(list1);

    assert_refcount(&arena, one_ref, 1);
    assert_refcount(&arena, two_ref, 1);
    assert_refcount(&arena, three_ref, 1);
}

#[test]
fn test_environment_refcounting() {
    let arena = Arena::<ARENA_SIZE>::new();
    let initial_free = count_free_slots(&arena);

    {
        let env = env_new(&arena);
        assert!(count_free_slots(&arena) < initial_free);

        let name = arena.str_to_list("x");
        let value = arena.number(42);
        env_set(&arena, &env, &name, &value);

        let retrieved = env_get(&arena, &env, &name);
        assert!(retrieved.is_some());

        if let Some(val) = retrieved {
            if let Some(Value::Number(n)) = arena.get(val.raw()) {
                assert!(n == 42);
            }
        }
    }
}

// ============================================================================
// Interpreter Tests
// ============================================================================

#[test]
fn test_basic_arithmetic() {
    let arena = Arena::<ARENA_SIZE>::new();
    let env = env_new(&arena);

    let result = eval_string(&arena, "(+ 1 2 3)", &env).unwrap();
    if let Some(Value::Number(n)) = arena.get(result.raw()) {
        assert_eq!(n, 6);
    } else {
        panic!("Expected 6");
    }

    let result = eval_string(&arena, "(* 2 3 4)", &env).unwrap();
    if let Some(Value::Number(n)) = arena.get(result.raw()) {
        assert_eq!(n, 24);
    } else {
        panic!("Expected 24");
    }

    let result = eval_string(&arena, "(- 10 3 2)", &env).unwrap();
    if let Some(Value::Number(n)) = arena.get(result.raw()) {
        assert_eq!(n, 5);
    } else {
        panic!("Expected 5");
    }

    let result = eval_string(&arena, "(/ 100 5 2)", &env).unwrap();
    if let Some(Value::Number(n)) = arena.get(result.raw()) {
        assert_eq!(n, 10);
    } else {
        panic!("Expected 10");
    }
}

#[test]
fn test_comparisons() {
    let arena = Arena::<ARENA_SIZE>::new();
    let env = env_new(&arena);

    let result = eval_string(&arena, "(= 5 5)", &env).unwrap();
    assert!(matches!(arena.get(result.raw()), Some(Value::Bool(true))));

    let result = eval_string(&arena, "(= 5 6)", &env).unwrap();
    assert!(matches!(arena.get(result.raw()), Some(Value::Bool(false))));

    let result = eval_string(&arena, "(< 3 5)", &env).unwrap();
    assert!(matches!(arena.get(result.raw()), Some(Value::Bool(true))));

    let result = eval_string(&arena, "(> 10 5)", &env).unwrap();
    assert!(matches!(arena.get(result.raw()), Some(Value::Bool(true))));
}

#[test]
fn test_list_operations() {
    let arena = Arena::<ARENA_SIZE>::new();
    let env = env_new(&arena);

    let result = eval_string(&arena, "(car '(1 2 3))", &env).unwrap();
    if let Some(Value::Number(n)) = arena.get(result.raw()) {
        assert_eq!(n, 1);
    }

    let result = eval_string(&arena, "(length '(1 2 3 4 5))", &env).unwrap();
    if let Some(Value::Number(n)) = arena.get(result.raw()) {
        assert_eq!(n, 5);
    }

    let result = eval_string(&arena, "(null? nil)", &env).unwrap();
    assert!(matches!(arena.get(result.raw()), Some(Value::Bool(true))));

    let result = eval_string(&arena, "(null? '(1))", &env).unwrap();
    assert!(matches!(arena.get(result.raw()), Some(Value::Bool(false))));
}

#[test]
fn test_conditionals() {
    let arena = Arena::<ARENA_SIZE>::new();
    let env = env_new(&arena);

    let result = eval_string(&arena, "(if #t 1 2)", &env).unwrap();
    if let Some(Value::Number(n)) = arena.get(result.raw()) {
        assert_eq!(n, 1);
    }

    let result = eval_string(&arena, "(if #f 1 2)", &env).unwrap();
    if let Some(Value::Number(n)) = arena.get(result.raw()) {
        assert_eq!(n, 2);
    }

    let result = eval_string(&arena, "(if (< 3 5) 10 20)", &env).unwrap();
    if let Some(Value::Number(n)) = arena.get(result.raw()) {
        assert_eq!(n, 10);
    }
}

#[test]
fn test_define_and_lookup() {
    let arena = Arena::<ARENA_SIZE>::new();
    let env = env_new(&arena);

    eval_string(&arena, "(define x 42)", &env).unwrap();
    let result = eval_string(&arena, "x", &env).unwrap();
    if let Some(Value::Number(n)) = arena.get(result.raw()) {
        assert_eq!(n, 42);
    }

    eval_string(&arena, "(define y (* x 2))", &env).unwrap();
    let result = eval_string(&arena, "y", &env).unwrap();
    if let Some(Value::Number(n)) = arena.get(result.raw()) {
        assert_eq!(n, 84);
    }
}

#[test]
fn test_lambda_basic() {
    let arena = Arena::<ARENA_SIZE>::new();
    let env = env_new(&arena);

    let result = eval_string(&arena, "((lambda (x) (* x x)) 5)", &env).unwrap();
    if let Some(Value::Number(n)) = arena.get(result.raw()) {
        assert_eq!(n, 25);
    }

    let result = eval_string(&arena, "((lambda (x y) (+ x y)) 3 4)", &env).unwrap();
    if let Some(Value::Number(n)) = arena.get(result.raw()) {
        assert_eq!(n, 7);
    }
}

#[test]
fn test_closures() {
    let arena = Arena::<ARENA_SIZE>::new();
    let env = env_new(&arena);

    eval_string(
        &arena,
        "(define make-adder (lambda (n) (lambda (x) (+ x n))))",
        &env,
    )
    .unwrap();
    eval_string(&arena, "(define add5 (make-adder 5))", &env).unwrap();

    let result = eval_string(&arena, "(add5 10)", &env).unwrap();
    if let Some(Value::Number(n)) = arena.get(result.raw()) {
        assert_eq!(n, 15);
    }

    eval_string(&arena, "(define add10 (make-adder 10))", &env).unwrap();
    let result = eval_string(&arena, "(add10 5)", &env).unwrap();
    if let Some(Value::Number(n)) = arena.get(result.raw()) {
        assert_eq!(n, 15);
    }
}

// ============================================================================
// Tail Call Optimization Tests
// ============================================================================

#[test]
fn test_simple_tail_recursion() {
    let arena = Arena::<ARENA_SIZE>::new();
    let env = env_new(&arena);

    eval_string(
        &arena,
        r#"
        (define countdown
          (lambda (n acc)
            (if (= n 0)
                acc
                (countdown (- n 1) (+ acc 1)))))
    "#,
        &env,
    )
    .unwrap();

    let result = eval_string(&arena, "(countdown 100 0)", &env).unwrap();
    if let Some(Value::Number(n)) = arena.get(result.raw()) {
        assert_eq!(n, 100);
    }
}

#[test]
fn test_deep_tail_recursion() {
    let arena = Arena::<ARENA_SIZE>::new();
    let env = env_new(&arena);

    eval_string(
        &arena,
        r#"
        (define sum-tail
          (lambda (n acc)
            (if (= n 0)
                acc
                (sum-tail (- n 1) (+ acc n)))))
    "#,
        &env,
    )
    .unwrap();

    let result = eval_string(&arena, "(sum-tail 1000 0)", &env).unwrap();
    if let Some(Value::Number(n)) = arena.get(result.raw()) {
        assert_eq!(n, 500500); // sum from 1 to 1000
    }
}

#[test]
fn test_factorial_tail_recursive() {
    let arena = Arena::<ARENA_SIZE>::new();
    let env = env_new(&arena);

    eval_string(
        &arena,
        r#"
        (define factorial
          (lambda (n acc)
            (if (= n 0)
                acc
                (factorial (- n 1) (* n acc)))))
    "#,
        &env,
    )
    .unwrap();

    let result = eval_string(&arena, "(factorial 10 1)", &env).unwrap();
    if let Some(Value::Number(n)) = arena.get(result.raw()) {
        assert_eq!(n, 3628800);
    }
}

#[test]
fn test_tco_memory_efficiency() {
    let arena = Arena::<ARENA_SIZE>::new();
    let env = env_new(&arena);

    eval_string(
        &arena,
        r#"
        (define count-down
          (lambda (n)
            (if (= n 0)
                0
                (count-down (- n 1)))))
    "#,
        &env,
    )
    .unwrap();

    let before_allocated = count_allocated_slots(&arena);

    // This should not exhaust memory with proper TCO
    let result = eval_string(&arena, "(count-down 10000)", &env);
    assert!(result.is_ok());

    let after_allocated = count_allocated_slots(&arena);

    // Memory usage should be bounded (not O(n))
    // Allow some growth but not 10,000 allocations worth
    let growth = after_allocated - before_allocated;
    assert!(
        growth < 1000,
        "Memory grew too much: {} allocations",
        growth
    );
}

// ============================================================================
// Stress Tests
// ============================================================================

#[test]
fn test_very_deep_recursion() {
    let arena = Arena::<ARENA_SIZE>::new();
    let env = env_new(&arena);

    eval_string(
        &arena,
        r#"
        (define deep-sum
          (lambda (n acc)
            (if (= n 0)
                acc
                (deep-sum (- n 1) (+ acc 1)))))
    "#,
        &env,
    )
    .unwrap();

    // 50,000 iterations - should work with proper TCO
    let result = eval_string(&arena, "(deep-sum 50000 0)", &env).unwrap();
    if let Some(Value::Number(n)) = arena.get(result.raw()) {
        assert_eq!(n, 50000);
    }
}

#[test]
fn test_fibonacci_tree_recursion() {
    let arena = Arena::<ARENA_SIZE>::new();
    let env = env_new(&arena);

    eval_string(
        &arena,
        r#"
        (define fib
          (lambda (n)
            (if (< n 2)
                n
                (+ (fib (- n 1)) (fib (- n 2))))))
    "#,
        &env,
    )
    .unwrap();

    let result = eval_string(&arena, "(fib 15)", &env).unwrap();
    if let Some(Value::Number(n)) = arena.get(result.raw()) {
        assert_eq!(n, 610);
    }
}

#[test]
fn test_mutual_recursion() {
    let arena = Arena::<ARENA_SIZE>::new();
    let env = env_new(&arena);

    eval_string(
        &arena,
        r#"
        (define is-even
          (lambda (n)
            (if (= n 0)
                #t
                (is-odd (- n 1)))))
    "#,
        &env,
    )
    .unwrap();

    eval_string(
        &arena,
        r#"
        (define is-odd
          (lambda (n)
            (if (= n 0)
                #f
                (is-even (- n 1)))))
    "#,
        &env,
    )
    .unwrap();

    let result = eval_string(&arena, "(is-even 100)", &env).unwrap();
    assert!(matches!(arena.get(result.raw()), Some(Value::Bool(true))));

    let result = eval_string(&arena, "(is-odd 99)", &env).unwrap();
    assert!(matches!(arena.get(result.raw()), Some(Value::Bool(true))));

    // Deep mutual recursion
    let result = eval_string(&arena, "(is-even 1000)", &env).unwrap();
    assert!(matches!(arena.get(result.raw()), Some(Value::Bool(true))));
}

#[test]
fn test_ackermann_function() {
    let arena = Arena::<ARENA_SIZE>::new();
    let env = env_new(&arena);

    eval_string(
        &arena,
        r#"
        (define ackermann
          (lambda (m n)
            (if (= m 0)
                (+ n 1)
                (if (= n 0)
                    (ackermann (- m 1) 1)
                    (ackermann (- m 1) (ackermann m (- n 1)))))))
    "#,
        &env,
    )
    .unwrap();

    let result = eval_string(&arena, "(ackermann 2 2)", &env).unwrap();
    if let Some(Value::Number(n)) = arena.get(result.raw()) {
        assert_eq!(n, 7);
    }

    let result = eval_string(&arena, "(ackermann 3 2)", &env).unwrap();
    if let Some(Value::Number(n)) = arena.get(result.raw()) {
        assert_eq!(n, 29);
    }
}

#[test]
fn test_higher_order_functions() {
    let arena = Arena::<ARENA_SIZE>::new();
    let env = env_new(&arena);

    eval_string(
        &arena,
        r#"
        (define map
          (lambda (f lst)
            (if (null? lst)
                nil
                (cons (f (car lst)) (map f (cdr lst))))))
    "#,
        &env,
    )
    .unwrap();

    let result = eval_string(&arena, "(map (lambda (x) (* x x)) '(1 2 3 4))", &env).unwrap();

    // Verify list contains 1, 4, 9, 16
    let first = arena.list_nth(&result, 0).unwrap();
    if let Some(Value::Number(n)) = arena.get(first.raw()) {
        assert_eq!(n, 1);
    }

    let second = arena.list_nth(&result, 1).unwrap();
    if let Some(Value::Number(n)) = arena.get(second.raw()) {
        assert_eq!(n, 4);
    }

    let third = arena.list_nth(&result, 2).unwrap();
    if let Some(Value::Number(n)) = arena.get(third.raw()) {
        assert_eq!(n, 9);
    }

    let fourth = arena.list_nth(&result, 3).unwrap();
    if let Some(Value::Number(n)) = arena.get(fourth.raw()) {
        assert_eq!(n, 16);
    }
}

#[test]
fn test_filter_function() {
    let arena = Arena::<ARENA_SIZE>::new();
    let env = env_new(&arena);

    eval_string(
        &arena,
        r#"
        (define filter
          (lambda (pred lst)
            (if (null? lst)
                nil
                (if (pred (car lst))
                    (cons (car lst) (filter pred (cdr lst)))
                    (filter pred (cdr lst))))))
    "#,
        &env,
    )
    .unwrap();

    let result = eval_string(
        &arena,
        "(filter (lambda (x) (> x 0)) '(-2 -1 0 1 2 3))",
        &env,
    )
    .unwrap();

    assert_eq!(arena.list_len(&result), 3);

    let first = arena.list_nth(&result, 0).unwrap();
    if let Some(Value::Number(n)) = arena.get(first.raw()) {
        assert_eq!(n, 1);
    }

    let second = arena.list_nth(&result, 1).unwrap();
    if let Some(Value::Number(n)) = arena.get(second.raw()) {
        assert_eq!(n, 2);
    }
}

#[test]
fn test_compose_function() {
    let arena = Arena::<ARENA_SIZE>::new();
    let env = env_new(&arena);

    eval_string(
        &arena,
        r#"
        (define compose
          (lambda (f g)
            (lambda (x) (f (g x)))))
    "#,
        &env,
    )
    .unwrap();

    eval_string(&arena, "(define add1 (lambda (x) (+ x 1)))", &env).unwrap();
    eval_string(&arena, "(define square (lambda (x) (* x x)))", &env).unwrap();
    eval_string(&arena, "(define composed (compose add1 square))", &env).unwrap();

    let result = eval_string(&arena, "(composed 5)", &env).unwrap();
    if let Some(Value::Number(n)) = arena.get(result.raw()) {
        assert_eq!(n, 26); // square(5) + 1 = 25 + 1 = 26
    }
}

#[test]
fn test_list_building() {
    let arena = Arena::<ARENA_SIZE>::new();
    let env = env_new(&arena);

    eval_string(
        &arena,
        r#"
        (define build-list
          (lambda (n acc)
            (if (= n 0)
                acc
                (build-list (- n 1) (cons n acc)))))
    "#,
        &env,
    )
    .unwrap();

    let result = eval_string(&arena, "(build-list 100 nil)", &env).unwrap();
    assert_eq!(arena.list_len(&result), 100);

    let first = arena.list_nth(&result, 0).unwrap();
    if let Some(Value::Number(n)) = arena.get(first.raw()) {
        assert_eq!(n, 1);
    }
}

#[test]
fn test_nested_lambdas() {
    let arena = Arena::<ARENA_SIZE>::new();
    let env = env_new(&arena);

    let result = eval_string(&arena, "((lambda (x) ((lambda (y) (+ x y)) 10)) 5)", &env).unwrap();

    if let Some(Value::Number(n)) = arena.get(result.raw()) {
        assert_eq!(n, 15);
    }
}

#[test]
fn test_memory_stability() {
    let arena = Arena::<ARENA_SIZE>::new();
    let env = env_new(&arena);

    eval_string(
        &arena,
        r#"
        (define loop
          (lambda (n)
            (if (= n 0)
                #t
                (loop (- n 1)))))
    "#,
        &env,
    )
    .unwrap();

    let before = count_allocated_slots(&arena);

    // Run multiple iterations
    for _ in 0..10 {
        eval_string(&arena, "(loop 1000)", &env).unwrap();
    }

    let after = count_allocated_slots(&arena);
    let growth = after - before;

    // Should have minimal memory growth across iterations
    assert!(
        growth < 100,
        "Memory not stable across iterations: {} slots growth",
        growth
    );
}

#[test]
fn test_error_handling() {
    let arena = Arena::<ARENA_SIZE>::new();
    let env = env_new(&arena);

    // Division by zero
    let result = eval_string(&arena, "(/ 10 0)", &env);
    assert!(result.is_err());

    // Unbound symbol
    let result = eval_string(&arena, "undefined-var", &env);
    assert!(result.is_err());

    let result = eval_string(&arena, "(if #t)", &env);
    assert!(result.is_err());
}

#[test]
fn test_complex_expression() {
    let arena = Arena::<ARENA_SIZE>::new();
    let env = env_new(&arena);

    let result = eval_string(&arena, "(+ (* 2 3) (- 10 5) (/ 20 4))", &env).unwrap();

    if let Some(Value::Number(n)) = arena.get(result.raw()) {
        assert_eq!(n, 16); // 6 + 5 + 5
    }
}

// ============================================================================
// Additional Stress Tests
// ============================================================================

#[test]
fn test_massive_list_operations() {
    let arena = Arena::<ARENA_SIZE>::new();
    let env = env_new(&arena);

    eval_string(
        &arena,
        r#"
        (define range
          (lambda (n acc)
            (if (= n 0)
                acc
                (range (- n 1) (cons n acc)))))
    "#,
        &env,
    )
    .unwrap();

    // Create a list of 500 elements
    let result = eval_string(&arena, "(length (range 500 nil))", &env).unwrap();
    if let Some(Value::Number(n)) = arena.get(result.raw()) {
        assert_eq!(n, 500);
    }
}

#[test]
fn test_nested_closure_chains() {
    let arena = Arena::<ARENA_SIZE>::new();
    let env = env_new(&arena);

    eval_string(
        &arena,
        r#"
        (define make-chain
          (lambda (n)
            (if (= n 0)
                (lambda (x) x)
                (lambda (x) ((make-chain (- n 1)) (+ x 1))))))
    "#,
        &env,
    )
    .unwrap();

    // Create chain of 50 closures
    eval_string(&arena, "(define chain50 (make-chain 50))", &env).unwrap();
    let result = eval_string(&arena, "(chain50 0)", &env).unwrap();
    if let Some(Value::Number(n)) = arena.get(result.raw()) {
        assert_eq!(n, 50);
    }
}

#[test]
fn test_repeated_environment_modifications() {
    let arena = Arena::<ARENA_SIZE>::new();
    let env = env_new(&arena);

    // Define and redefine many variables
    for i in 0..100 {
        let def_expr = format!("(define var{} {})", i, i * 2);
        eval_string(&arena, &def_expr, &env).unwrap();
    }

    // Verify a few
    let result = eval_string(&arena, "var0", &env).unwrap();
    if let Some(Value::Number(n)) = arena.get(result.raw()) {
        assert_eq!(n, 0);
    }

    let result = eval_string(&arena, "var50", &env).unwrap();
    if let Some(Value::Number(n)) = arena.get(result.raw()) {
        assert_eq!(n, 100);
    }

    let result = eval_string(&arena, "var99", &env).unwrap();
    if let Some(Value::Number(n)) = arena.get(result.raw()) {
        assert_eq!(n, 198);
    }
}

#[test]
fn test_fold_operations() {
    let arena = Arena::<ARENA_SIZE>::new();
    let env = env_new(&arena);

    eval_string(
        &arena,
        r#"
        (define fold
          (lambda (f acc lst)
            (if (null? lst)
                acc
                (fold f (f acc (car lst)) (cdr lst)))))
    "#,
        &env,
    )
    .unwrap();

    // Sum using fold
    let result = eval_string(
        &arena,
        "(fold (lambda (a b) (+ a b)) 0 '(1 2 3 4 5 6 7 8 9 10))",
        &env,
    )
    .unwrap();
    if let Some(Value::Number(n)) = arena.get(result.raw()) {
        assert_eq!(n, 55);
    }

    // Product using fold
    let result = eval_string(&arena, "(fold (lambda (a b) (* a b)) 1 '(1 2 3 4 5))", &env).unwrap();
    if let Some(Value::Number(n)) = arena.get(result.raw()) {
        assert_eq!(n, 120);
    }
}

#[test]
fn test_deeply_nested_data_structures() {
    let arena = Arena::<ARENA_SIZE>::new();
    let env = env_new(&arena);

    eval_string(
        &arena,
        r#"
        (define make-nested
          (lambda (depth val)
            (if (= depth 0)
                val
                (cons (make-nested (- depth 1) val) nil))))
    "#,
        &env,
    )
    .unwrap();

    eval_string(
        &arena,
        r#"
        (define extract-nested
          (lambda (depth lst)
            (if (= depth 0)
                lst
                (extract-nested (- depth 1) (car lst)))))
    "#,
        &env,
    )
    .unwrap();

    // Create deeply nested structure (100 levels)
    eval_string(&arena, "(define nested (make-nested 100 42))", &env).unwrap();

    // Extract the value
    let result = eval_string(&arena, "(extract-nested 100 nested)", &env).unwrap();
    if let Some(Value::Number(n)) = arena.get(result.raw()) {
        assert_eq!(n, 42);
    }
}

#[test]
fn test_alternating_recursion_patterns() {
    let arena = Arena::<ARENA_SIZE>::new();
    let env = env_new(&arena);

    eval_string(
        &arena,
        r#"
        (define bounce
          (lambda (n mode)
            (if (= n 0)
                mode
                (if (= mode 0)
                    (bounce (- n 1) 1)
                    (bounce (- n 1) 0)))))
    "#,
        &env,
    )
    .unwrap();

    let result = eval_string(&arena, "(bounce 1000 0)", &env).unwrap();
    if let Some(Value::Number(n)) = arena.get(result.raw()) {
        assert_eq!(n, 0); // Even number of bounces
    }

    let result = eval_string(&arena, "(bounce 999 0)", &env).unwrap();
    if let Some(Value::Number(n)) = arena.get(result.raw()) {
        assert_eq!(n, 1); // Odd number of bounces
    }
}

#[test]
fn test_memory_pressure_with_large_computations() {
    let arena = Arena::<ARENA_SIZE>::new();
    let env = env_new(&arena);

    eval_string(
        &arena,
        r#"
        (define compute-sum
          (lambda (n)
            (if (= n 0)
                0
                (+ n (compute-sum (- n 1))))))
    "#,
        &env,
    )
    .unwrap();

    let initial_free = count_free_slots(&arena);

    // This creates temporary values but should free them
    let result = eval_string(&arena, "(compute-sum 100)", &env).unwrap();
    if let Some(Value::Number(n)) = arena.get(result.raw()) {
        assert_eq!(n, 5050);
    }

    drop(result);

    // Most memory should be freed
    let final_free = count_free_slots(&arena);
    let leaked = initial_free - final_free;

    // Allow some leakage for env bindings but not much
    assert!(leaked < 50, "Too much memory leaked: {} slots", leaked);
}

#[test]
fn test_currying_chains() {
    let arena = Arena::<ARENA_SIZE>::new();
    let env = env_new(&arena);

    eval_string(
        &arena,
        "(define curry3 (lambda (f) (lambda (a) (lambda (b) (lambda (c) (f a b c))))))",
        &env,
    )
    .unwrap();

    eval_string(&arena, "(define add3 (lambda (x y z) (+ x (+ y z))))", &env).unwrap();

    eval_string(&arena, "(define curried-add3 (curry3 add3))", &env).unwrap();

    let result = eval_string(&arena, "(((curried-add3 1) 2) 3)", &env).unwrap();
    if let Some(Value::Number(n)) = arena.get(result.raw()) {
        assert_eq!(n, 6);
    }
}

#[test]
fn test_y_combinator() {
    let arena = Arena::<ARENA_SIZE>::new();
    let env = env_new(&arena);

    // Y combinator for recursion without define
    eval_string(
        &arena,
        r#"
        (define Y
          (lambda (f)
            ((lambda (x) (f (lambda (y) ((x x) y))))
             (lambda (x) (f (lambda (y) ((x x) y)))))))
    "#,
        &env,
    )
    .unwrap();

    eval_string(
        &arena,
        r#"
        (define fac-gen
          (lambda (f)
            (lambda (n)
              (if (= n 0)
                  1
                  (* n (f (- n 1)))))))
    "#,
        &env,
    )
    .unwrap();

    eval_string(&arena, "(define factorial (Y fac-gen))", &env).unwrap();

    let result = eval_string(&arena, "(factorial 6)", &env).unwrap();
    if let Some(Value::Number(n)) = arena.get(result.raw()) {
        assert_eq!(n, 720);
    }
}
