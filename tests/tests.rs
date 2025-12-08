#![cfg(test)]

use ruthe::*;
extern crate alloc;
// ============================================================================
// Helper Functions
// ============================================================================

fn eval_test(input: &str) -> Result<String, String> {
    let env = EnvRef::new();
    eval_str(input, &env)
        .map(|s| {
            let mut buf = [0u8; 4096];
            s.to_display_str(&mut buf).unwrap().to_string() // Changed from to_display_str
        })
        .map_err(|e| {
            let mut buf = [0u8; 256];
            e.to_display_str(&mut buf).unwrap().to_string()
        })
}

fn eval_multiple_test(input: &str) -> Result<String, String> {
    let env = EnvRef::new();
    eval_str_multiple(input, &env)
        .map(|s| {
            let mut buf = [0u8; 4096];
            s.to_display_str(&mut buf).unwrap().to_string() // Changed from to_display_str
        })
        .map_err(|e| {
            let mut buf = [0u8; 256];
            e.to_display_str(&mut buf).unwrap().to_string()
        })
}
// ============================================================================
// Basic Value Tests
// ============================================================================

#[test]
fn test_numbers() {
    assert_eq!(eval_test("42"), Ok("42".to_string()));
    assert_eq!(eval_test("-17"), Ok("-17".to_string()));
    assert_eq!(eval_test("0"), Ok("0".to_string()));
}

#[test]
fn test_booleans() {
    assert_eq!(eval_test("#t"), Ok("#t".to_string()));
    assert_eq!(eval_test("#f"), Ok("#f".to_string()));
}

#[test]
fn test_nil() {
    assert_eq!(eval_test("nil"), Ok("nil".to_string()));
}

// ============================================================================
// Arithmetic Tests
// ============================================================================

#[test]
fn test_addition() {
    assert_eq!(eval_test("(+ 1 2)"), Ok("3".to_string()));
    assert_eq!(eval_test("(+ 1 2 3 4 5)"), Ok("15".to_string()));
    assert_eq!(eval_test("(+)"), Ok("0".to_string()));
    assert_eq!(eval_test("(+ -5 10)"), Ok("5".to_string()));
}

#[test]
fn test_subtraction() {
    assert_eq!(eval_test("(- 10 3)"), Ok("7".to_string()));
    assert_eq!(eval_test("(- 5)"), Ok("-5".to_string()));
    assert_eq!(eval_test("(- 20 5 3)"), Ok("12".to_string()));
    assert_eq!(eval_test("(- 0 5)"), Ok("-5".to_string()));
}

#[test]
fn test_multiplication() {
    assert_eq!(eval_test("(* 2 3)"), Ok("6".to_string()));
    assert_eq!(eval_test("(* 2 3 4)"), Ok("24".to_string()));
    assert_eq!(eval_test("(*)"), Ok("1".to_string()));
    assert_eq!(eval_test("(* -2 5)"), Ok("-10".to_string()));
}

#[test]
fn test_division() {
    assert_eq!(eval_test("(/ 10 2)"), Ok("5".to_string()));
    assert_eq!(eval_test("(/ 20 4 2)"), Ok("2".to_string()));
    assert_eq!(eval_test("(/ 100 10)"), Ok("10".to_string()));
    assert!(eval_test("(/ 10 0)").is_err());
}

#[test]
fn test_nested_arithmetic() {
    assert_eq!(eval_test("(+ (* 2 3) (- 10 5))"), Ok("11".to_string()));
    assert_eq!(eval_test("(* (+ 1 2) (+ 3 4))"), Ok("21".to_string()));
    assert_eq!(eval_test("(- (* 10 5) (/ 20 4))"), Ok("45".to_string()));
}

// ============================================================================
// Comparison Tests
// ============================================================================

#[test]
fn test_equality() {
    assert_eq!(eval_test("(= 5 5)"), Ok("#t".to_string()));
    assert_eq!(eval_test("(= 5 6)"), Ok("#f".to_string()));
    assert_eq!(eval_test("(= (+ 2 3) 5)"), Ok("#t".to_string()));
}

#[test]
fn test_less_than() {
    assert_eq!(eval_test("(< 3 5)"), Ok("#t".to_string()));
    assert_eq!(eval_test("(< 5 3)"), Ok("#f".to_string()));
    assert_eq!(eval_test("(< 5 5)"), Ok("#f".to_string()));
}

#[test]
fn test_greater_than() {
    assert_eq!(eval_test("(> 5 3)"), Ok("#t".to_string()));
    assert_eq!(eval_test("(> 3 5)"), Ok("#f".to_string()));
    assert_eq!(eval_test("(> 5 5)"), Ok("#f".to_string()));
}

// ============================================================================
// List Operations Tests
// ============================================================================

#[test]
fn test_list() {
    assert_eq!(eval_test("(list 1 2 3)"), Ok("(1 2 3)".to_string()));
    assert_eq!(eval_test("(list)"), Ok("nil".to_string()));
    assert_eq!(eval_test("(list 1)"), Ok("(1)".to_string()));
}

#[test]
fn test_cons() {
    assert_eq!(eval_test("(cons 1 (list 2 3))"), Ok("(1 2 3)".to_string()));
    assert_eq!(eval_test("(cons 1 nil)"), Ok("(1)".to_string()));
    assert_eq!(eval_test("(cons 1 2)"), Ok("(1 . 2)".to_string()));
}

#[test]
fn test_car() {
    assert_eq!(eval_test("(car (list 1 2 3))"), Ok("1".to_string()));
    assert_eq!(eval_test("(car (cons 5 10))"), Ok("5".to_string()));
    assert!(eval_test("(car nil)").is_err());
}

#[test]
fn test_cdr() {
    assert_eq!(eval_test("(cdr (list 1 2 3))"), Ok("(2 3)".to_string()));
    assert_eq!(eval_test("(cdr (list 1))"), Ok("nil".to_string()));
    assert_eq!(eval_test("(cdr (cons 5 10))"), Ok("10".to_string()));
    assert!(eval_test("(cdr nil)").is_err());
}

#[test]
fn test_null_predicate() {
    assert_eq!(eval_test("(null? nil)"), Ok("#t".to_string()));
    assert_eq!(eval_test("(null? (list))"), Ok("#t".to_string()));
    assert_eq!(eval_test("(null? (list 1))"), Ok("#f".to_string()));
    assert_eq!(eval_test("(null? 5)"), Ok("#f".to_string()));
}

#[test]
fn test_cons_predicate() {
    assert_eq!(eval_test("(cons? (list 1 2))"), Ok("#t".to_string()));
    assert_eq!(eval_test("(cons? (cons 1 2))"), Ok("#t".to_string()));
    assert_eq!(eval_test("(cons? nil)"), Ok("#f".to_string()));
    assert_eq!(eval_test("(cons? 5)"), Ok("#f".to_string()));
}

#[test]
fn test_length() {
    assert_eq!(eval_test("(length (list 1 2 3))"), Ok("3".to_string()));
    assert_eq!(eval_test("(length nil)"), Ok("0".to_string()));
    assert_eq!(eval_test("(length (list 1))"), Ok("1".to_string()));
}

#[test]
fn test_append() {
    assert_eq!(
        eval_test("(append (list 1 2) (list 3 4))"),
        Ok("(1 2 3 4)".to_string())
    );
    assert_eq!(eval_test("(append (list 1) nil)"), Ok("(1)".to_string()));
    assert_eq!(eval_test("(append nil (list 1))"), Ok("(1)".to_string()));
    assert_eq!(eval_test("(append nil nil)"), Ok("nil".to_string()));
}

#[test]
fn test_reverse() {
    assert_eq!(
        eval_test("(reverse (list 1 2 3))"),
        Ok("(3 2 1)".to_string())
    );
    assert_eq!(eval_test("(reverse nil)"), Ok("nil".to_string()));
    assert_eq!(eval_test("(reverse (list 1))"), Ok("(1)".to_string()));
}

// ============================================================================
// Quote Tests
// ============================================================================

#[test]
fn test_quote() {
    assert_eq!(eval_test("(quote x)"), Ok("x".to_string()));
    assert_eq!(eval_test("(quote (1 2 3))"), Ok("(1 2 3)".to_string()));
    assert_eq!(eval_test("'x"), Ok("x".to_string()));
    assert_eq!(eval_test("'(1 2 3)"), Ok("(1 2 3)".to_string()));
}

// ============================================================================
// Define Tests
// ============================================================================

#[test]
fn test_define() {
    let env = EnvRef::new();
    assert_eq!(
        eval_str("(define x 42)", &env).map(|s| {
            let mut buf = [0u8; 256];
            s.to_display_str(&mut buf).unwrap().to_string()
        }),
        Ok("42".to_string())
    );
    assert_eq!(
        eval_str("x", &env).map(|s| {
            let mut buf = [0u8; 256];
            s.to_display_str(&mut buf).unwrap().to_string()
        }),
        Ok("42".to_string())
    );
}

#[test]
fn test_define_expression() {
    let env = EnvRef::new();
    eval_str("(define y (+ 10 20))", &env).unwrap();
    assert_eq!(
        eval_str("y", &env).map(|s| {
            let mut buf = [0u8; 256];
            s.to_display_str(&mut buf).unwrap().to_string()
        }),
        Ok("30".to_string())
    );
}

// ============================================================================
// Lambda Tests
// ============================================================================

#[test]
fn test_lambda_basic() {
    assert_eq!(eval_test("((lambda (x) (+ x 1)) 5)"), Ok("6".to_string()));
    assert_eq!(
        eval_test("((lambda (x y) (+ x y)) 3 4)"),
        Ok("7".to_string())
    );
}

#[test]
fn test_lambda_no_args() {
    assert_eq!(eval_test("((lambda () 42))"), Ok("42".to_string()));
}

#[test]
fn test_lambda_closure() {
    let env = EnvRef::new();
    eval_str("(define x 10)", &env).unwrap();
    eval_str("(define f (lambda (y) (+ x y)))", &env).unwrap();
    assert_eq!(
        eval_str("(f 5)", &env).map(|s| {
            let mut buf = [0u8; 256];
            s.to_display_str(&mut buf).unwrap().to_string()
        }),
        Ok("15".to_string())
    );
}

#[test]
fn test_lambda_nested() {
    assert_eq!(
        eval_test("((lambda (x) ((lambda (y) (+ x y)) 3)) 5)"),
        Ok("8".to_string())
    );
}

#[test]
fn test_higher_order_function() {
    let env = EnvRef::new();
    eval_str(
        "(define make-adder (lambda (n) (lambda (x) (+ x n))))",
        &env,
    )
    .unwrap();
    eval_str("(define add5 (make-adder 5))", &env).unwrap();
    assert_eq!(
        eval_str("(add5 10)", &env).map(|s| {
            let mut buf = [0u8; 256];
            s.to_display_str(&mut buf).unwrap().to_string()
        }),
        Ok("15".to_string())
    );
}

// ============================================================================
// If Tests
// ============================================================================

#[test]
fn test_if_true() {
    assert_eq!(eval_test("(if #t 1 2)"), Ok("1".to_string()));
    assert_eq!(eval_test("(if (< 3 5) 10 20)"), Ok("10".to_string()));
}

#[test]
fn test_if_false() {
    assert_eq!(eval_test("(if #f 1 2)"), Ok("2".to_string()));
    assert_eq!(eval_test("(if (> 3 5) 10 20)"), Ok("20".to_string()));
}

#[test]
fn test_if_nil_is_false() {
    assert_eq!(eval_test("(if nil 1 2)"), Ok("2".to_string()));
}

#[test]
fn test_if_non_boolean_is_true() {
    assert_eq!(eval_test("(if 5 1 2)"), Ok("1".to_string()));
    assert_eq!(eval_test("(if (list 1) 1 2)"), Ok("1".to_string()));
}

#[test]
fn test_if_nested() {
    assert_eq!(eval_test("(if #t (if #f 1 2) 3)"), Ok("2".to_string()));
}

// ============================================================================
// Recursive Function Tests
// ============================================================================

#[test]
fn test_factorial() {
    let program = r#"
        (define fact
          (lambda (n)
            (if (= n 0)
                1
                (* n (fact (- n 1))))))
        (fact 5)
    "#;
    assert_eq!(eval_multiple_test(program), Ok("120".to_string()));
}

#[test]
fn test_factorial_larger() {
    let program = r#"
        (define fact
          (lambda (n)
            (if (= n 0)
                1
                (* n (fact (- n 1))))))
        (fact 10)
    "#;
    assert_eq!(eval_multiple_test(program), Ok("3628800".to_string()));
}

#[test]
fn test_fibonacci() {
    let program = r#"
        (define fib
          (lambda (n)
            (if (< n 2)
                n
                (+ (fib (- n 1)) (fib (- n 2))))))
        (fib 10)
    "#;
    assert_eq!(eval_multiple_test(program), Ok("55".to_string()));
}

#[test]
fn test_fibonacci_small() {
    let program = r#"
        (define fib
          (lambda (n)
            (if (< n 2)
                n
                (+ (fib (- n 1)) (fib (- n 2))))))
        (fib 6)
    "#;
    assert_eq!(eval_multiple_test(program), Ok("8".to_string()));
}

#[test]
fn test_countdown() {
    let program = r#"
        (define countdown
          (lambda (n)
            (if (= n 0)
                0
                (countdown (- n 1)))))
        (countdown 1000)
    "#;
    assert_eq!(eval_multiple_test(program), Ok("0".to_string()));
}

#[test]
fn test_countdown_large() {
    let program = r#"
        (define countdown
          (lambda (n)
            (if (= n 0)
                0
                (countdown (- n 1)))))
        (countdown 5000)
    "#;
    assert_eq!(eval_multiple_test(program), Ok("0".to_string()));
}

#[test]
fn test_ackermann() {
    let program = r#"
        (define ack
          (lambda (m n)
            (if (= m 0)
                (+ n 1)
                (if (= n 0)
                    (ack (- m 1) 1)
                    (ack (- m 1) (ack m (- n 1)))))))
        (ack 3 3)
    "#;
    assert_eq!(eval_multiple_test(program), Ok("61".to_string()));
}

#[test]
fn test_ackermann_small() {
    let program = r#"
        (define ack
          (lambda (m n)
            (if (= m 0)
                (+ n 1)
                (if (= n 0)
                    (ack (- m 1) 1)
                    (ack (- m 1) (ack m (- n 1)))))))
        (ack 2 2)
    "#;
    assert_eq!(eval_multiple_test(program), Ok("7".to_string()));
}

#[test]
fn test_sum_list() {
    let program = r#"
        (define sum
          (lambda (lst)
            (if (null? lst)
                0
                (+ (car lst) (sum (cdr lst))))))
        (sum (list 1 2 3 4 5))
    "#;
    assert_eq!(eval_multiple_test(program), Ok("15".to_string()));
}

#[test]
fn test_map_function() {
    let program = r#"
        (define map
          (lambda (f lst)
            (if (null? lst)
                nil
                (cons (f (car lst)) (map f (cdr lst))))))
        (define double (lambda (x) (* x 2)))
        (map double (list 1 2 3))
    "#;
    assert_eq!(eval_multiple_test(program), Ok("(2 4 6)".to_string()));
}

#[test]
fn test_filter_function() {
    let program = r#"
        (define filter
          (lambda (pred lst)
            (if (null? lst)
                nil
                (if (pred (car lst))
                    (cons (car lst) (filter pred (cdr lst)))
                    (filter pred (cdr lst))))))
        (define positive? (lambda (x) (> x 0)))
        (filter positive? (list -1 2 -3 4 5))
    "#;
    assert_eq!(eval_multiple_test(program), Ok("(2 4 5)".to_string()));
}

// ============================================================================
// Tail Call Optimization Tests
// ============================================================================

#[test]
fn test_tail_recursive_sum() {
    let program = r#"
        (define sum-tail
          (lambda (n acc)
            (if (= n 0)
                acc
                (sum-tail (- n 1) (+ acc n)))))
        (sum-tail 1000 0)
    "#;
    assert_eq!(eval_multiple_test(program), Ok("500500".to_string()));
}

#[test]
fn test_tail_recursive_factorial() {
    let program = r#"
        (define fact-tail
          (lambda (n acc)
            (if (= n 0)
                acc
                (fact-tail (- n 1) (* acc n)))))
        (fact-tail 10 1)
    "#;
    assert_eq!(eval_multiple_test(program), Ok("3628800".to_string()));
}

// ============================================================================
// Cyclic Reference Tests
// ============================================================================

#[test]
fn test_self_referential_list() {
    // Test that we can create and handle structures that reference themselves
    // Without proper handling, this could cause infinite loops
    let program = r#"
        (define make-cyclic
          (lambda (x)
            (cons x x)))
        (define cyc (make-cyclic 5))
        (car cyc)
    "#;
    assert_eq!(eval_multiple_test(program), Ok("5".to_string()));
}

#[test]
fn test_mutual_recursion() {
    // Tests functions that call each other
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
        (is-even 10)
    "#;
    assert_eq!(eval_multiple_test(program), Ok("#t".to_string()));
}

#[test]
fn test_mutual_recursion_odd() {
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
        (is-odd 7)
    "#;
    assert_eq!(eval_multiple_test(program), Ok("#t".to_string()));
}

// ============================================================================
// Memory Leak Detection Tests
// ============================================================================

#[test]
fn test_no_memory_leak_simple_recursion() {
    // Test that recursive functions don't leak memory beyond expected references
    use alloc::rc::Rc;

    let program = r#"
        (define countdown
          (lambda (n)
            (if (= n 0)
                0
                (countdown (- n 1)))))
        (countdown 100)
    "#;

    let env = EnvRef::new();
    let initial_count = Rc::strong_count(&env.0);
    let result = eval_str_multiple(program, &env);
    assert!(result.is_ok());

    let final_count = Rc::strong_count(&env.0);

    // The environment will have references from lambdas that capture it
    // What matters is that the count is stable (not growing unboundedly)
    // Expected: 1 (our ref) + 1 (countdown lambda captures env) = 2
    assert!(
        final_count <= initial_count + 1,
        "Environment references growing: initial={}, final={}",
        initial_count,
        final_count
    );
}

#[test]
fn test_no_memory_leak_lambda_closure() {
    // Test that lambda closures don't create unbounded reference growth
    use alloc::rc::Rc;

    let program = r#"
        (define make-counter
          (lambda (n)
            (lambda () n)))
        (define counter (make-counter 10))
        (counter)
    "#;

    let env = EnvRef::new();
    let initial_count = Rc::strong_count(&env.0);
    let result = eval_str_multiple(program, &env);
    assert!(result.is_ok());

    let final_count = Rc::strong_count(&env.0);

    // Expected: initial + 2 lambdas that capture the env
    assert!(
        final_count <= initial_count + 2,
        "Lambda closures growing unbounded: initial={}, final={}",
        initial_count,
        final_count
    );
}

#[test]
fn test_no_memory_leak_nested_lambdas() {
    // Test deeply nested lambda creation and evaluation
    use alloc::rc::Rc;

    let program = r#"
        (define nest
          (lambda (n)
            (if (= n 0)
                (lambda (x) x)
                (lambda (x) ((nest (- n 1)) x)))))
        ((nest 50) 42)
    "#;

    let env = EnvRef::new();
    let initial_count = Rc::strong_count(&env.0);
    let result = eval_str_multiple(program, &env);
    assert!(result.is_ok());

    let final_count = Rc::strong_count(&env.0);

    // The 'nest' lambda captures env, so we expect a small stable increase
    assert!(
        final_count <= initial_count + 1,
        "Nested lambdas growing unbounded: initial={}, final={}",
        initial_count,
        final_count
    );
}

#[test]
fn test_no_memory_leak_list_operations() {
    // Test that list building and destruction doesn't leak
    use alloc::rc::Rc;

    let program = r#"
        (define build-list
          (lambda (n)
            (if (= n 0)
                nil
                (cons n (build-list (- n 1))))))
        (define my-list (build-list 100))
        (length my-list)
    "#;

    let env = EnvRef::new();
    let initial_count = Rc::strong_count(&env.0);
    let result = eval_str_multiple(program, &env);
    assert!(result.is_ok());

    let final_count = Rc::strong_count(&env.0);

    // build-list lambda captures env
    assert!(
        final_count <= initial_count + 1,
        "List operations growing unbounded: initial={}, final={}",
        initial_count,
        final_count
    );
}

#[test]
fn test_no_memory_leak_multiple_evaluations() {
    // Test that multiple evaluations in the same environment don't accumulate references
    use alloc::rc::Rc;

    let env = EnvRef::new();
    let initial_count = Rc::strong_count(&env.0);

    // Run multiple evaluations
    for i in 0..10 {
        let program = format!("(+ {} {})", i, i + 1);
        let result = eval_str(&program, &env);
        assert!(result.is_ok());
    }

    let final_count = Rc::strong_count(&env.0);
    // These simple evaluations shouldn't increase the count at all
    assert_eq!(
        final_count, initial_count,
        "Multiple evaluations leaked references: initial={}, final={}",
        initial_count, final_count
    );
}

#[test]
fn test_no_unbounded_growth() {
    // Critical test: ensure repeated operations don't grow memory unboundedly
    use alloc::rc::Rc;

    let env = EnvRef::new();

    // Define a recursive function
    eval_str(
        "(define countdown (lambda (n) (if (= n 0) 0 (countdown (- n 1)))))",
        &env,
    )
    .unwrap();
    let count_after_define = Rc::strong_count(&env.0);

    // Run it multiple times - the count should stabilize
    eval_str("(countdown 100)", &env).unwrap();
    let count_after_first = Rc::strong_count(&env.0);

    eval_str("(countdown 100)", &env).unwrap();
    let count_after_second = Rc::strong_count(&env.0);

    eval_str("(countdown 100)", &env).unwrap();
    let count_after_third = Rc::strong_count(&env.0);

    // The count should stabilize after first eval (not grow with each call)
    assert_eq!(
        count_after_first, count_after_second,
        "Reference count growing between calls: {} vs {}",
        count_after_first, count_after_second
    );
    assert_eq!(
        count_after_second, count_after_third,
        "Reference count growing between calls: {} vs {}",
        count_after_second, count_after_third
    );
}

#[test]
fn test_temporary_values_cleaned_up() {
    // Test that temporary values created during evaluation are cleaned up
    use alloc::rc::Rc;

    let env = EnvRef::new();
    let initial_count = Rc::strong_count(&env.0);

    // Create and immediately discard large temporary structures
    let program = r#"
        (length (append 
                  (list 1 2 3 4 5)
                  (list 6 7 8 9 10)
                  (list 11 12 13 14 15)))
    "#;

    let result = eval_str(program, &env);
    assert_eq!(
        result.map(|s| {
            let mut buf = [0u8; 256];
            s.to_display_str(&mut buf).unwrap().to_string()
        }),
        Ok("15".to_string())
    );

    let final_count = Rc::strong_count(&env.0);

    // No persistent references should remain from temporary values
    assert_eq!(
        final_count, initial_count,
        "Temporary values not cleaned up: initial={}, final={}",
        initial_count, final_count
    );
}

#[test]
fn test_valref_cleanup() {
    // Test that ValRef properly cleans up when dropped
    use alloc::rc::Rc;

    {
        let val = ValRef::number(42);
        let rc_count = Rc::strong_count(&val.0);
        assert_eq!(rc_count, 1);

        // Clone it a few times
        let val2 = val.clone();
        let val3 = val.clone();
        let rc_count = Rc::strong_count(&val.0);
        assert_eq!(rc_count, 3);

        drop(val2);
        let rc_count = Rc::strong_count(&val.0);
        assert_eq!(rc_count, 2);

        drop(val3);
        let rc_count = Rc::strong_count(&val.0);
        assert_eq!(rc_count, 1);
    }
    // val is dropped here, all references should be gone
}

#[test]
fn test_environment_bindings_cleanup() {
    // Test that environment bindings are properly cleaned up
    use alloc::rc::Rc;

    let env = EnvRef::new();

    // Create some bindings
    eval_str("(define x 10)", &env).unwrap();
    eval_str("(define y 20)", &env).unwrap();
    eval_str("(define z (+ x y))", &env).unwrap();

    let strong_count = Rc::strong_count(&env.0);
    assert_eq!(
        strong_count, 1,
        "Environment bindings leaked references: {}",
        strong_count
    );
}

#[test]
fn test_no_leak_with_errors() {
    // Test that errors during evaluation don't leak memory
    use alloc::rc::Rc;

    let env = EnvRef::new();

    // Try some operations that will fail
    let _ = eval_str("(/ 10 0)", &env); // Division by zero
    let _ = eval_str("undefined-var", &env); // Unbound symbol
    let _ = eval_str("(+ 1 #t)", &env); // Type error

    let strong_count = Rc::strong_count(&env.0);
    assert_eq!(
        strong_count, 1,
        "Error handling leaked references: {}",
        strong_count
    );
}

// ============================================================================
// Error Handling Tests
// ============================================================================

#[test]
fn test_unbound_symbol() {
    assert!(eval_test("undefined-var").is_err());
}

#[test]
fn test_invalid_function_call() {
    assert!(eval_test("(5 10)").is_err());
}

#[test]
fn test_arity_mismatch() {
    assert!(eval_test("((lambda (x) x) 1 2)").is_err());
}

#[test]
fn test_type_error_arithmetic() {
    assert!(eval_test("(+ 1 #t)").is_err());
}

#[test]
fn test_division_by_zero() {
    assert!(eval_test("(/ 10 0)").is_err());
}

// ============================================================================
// Complex Expression Tests
// ============================================================================

#[test]
fn test_nested_list_operations() {
    assert_eq!(eval_test("(car (cdr (list 1 2 3)))"), Ok("2".to_string()));
    assert_eq!(
        eval_test("(length (append (list 1 2) (list 3 4)))"),
        Ok("4".to_string())
    );
}

#[test]
fn test_complex_lambda_expression() {
    let program = r#"
        ((lambda (x y z)
           (+ (* x y) z))
         2 3 4)
    "#;
    assert_eq!(eval_test(program), Ok("10".to_string()));
}

#[test]
fn test_multiple_definitions() {
    let program = r#"
        (define a 10)
        (define b 20)
        (define c (+ a b))
        c
    "#;
    assert_eq!(eval_multiple_test(program), Ok("30".to_string()));
}

// ============================================================================
// Edge Cases
// ============================================================================

#[test]
fn test_empty_list() {
    assert_eq!(eval_test("(list)"), Ok("nil".to_string()));
}

#[test]
fn test_single_element_list() {
    assert_eq!(eval_test("(list 42)"), Ok("(42)".to_string()));
}

#[test]
fn test_nested_empty_lists() {
    assert_eq!(
        eval_test("(list (list) (list))"),
        Ok("(nil nil)".to_string())
    );
}

#[test]
fn test_zero_arithmetic() {
    assert_eq!(eval_test("(+ 0)"), Ok("0".to_string()));
    assert_eq!(eval_test("(* 0)"), Ok("0".to_string()));
    assert_eq!(eval_test("(- 0)"), Ok("0".to_string()));
}

// ============================================================================
// Parser Tests
// ============================================================================

#[test]
fn test_parse_comments() {
    let program = r#"
        ; This is a comment
        (+ 1 2) ; inline comment
        ; another comment
    "#;
    assert_eq!(eval_test(program), Ok("3".to_string()));
}

#[test]
fn test_parse_whitespace() {
    assert_eq!(eval_test("   (  +   1    2   )   "), Ok("3".to_string()));
}

#[test]
fn test_parse_multiline() {
    let program = r#"
        (+
         1
         2
         3)
    "#;
    assert_eq!(eval_test(program), Ok("6".to_string()));
}
