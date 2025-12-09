#![cfg(test)]
extern crate alloc;

// Import all the types and functions from your interpreter
use ruthe::*;

// ============================================================================
// Helper functions for testing
// ============================================================================

fn eval_test(program: &str) -> Result<ValRef, ValRef> {
    let env = new_env();
    eval_str(program, &env)
}

fn assert_number_eq(program: &str, expected: i64) {
    let result = eval_test(program);
    assert!(result.is_ok(), "Evaluation failed: {:?}", result);
    let val = result.unwrap();
    assert_eq!(val.as_number(), Some(expected), "Program: {}", program);
}

fn assert_bool_eq(program: &str, expected: bool) {
    let result = eval_test(program);
    assert!(result.is_ok(), "Evaluation failed: {:?}", result);
    let val = result.unwrap();
    assert_eq!(val.as_bool(), Some(expected), "Program: {}", program);
}

fn assert_err(program: &str) {
    let result = eval_test(program);
    assert!(result.is_err(), "Evaluation should fail: {:?}", result);
}

// ============================================================================
// Basic value tests
// ============================================================================

#[test]
fn test_number_literals() {
    // Test basic numbers
    assert_number_eq("42", 42);
    assert_number_eq("-123", -123);
    assert_number_eq("0", 0);

    // Test that numbers are self-evaluating
    let env = new_env();
    let result = eval_str("42", &env).unwrap();
    assert_eq!(result.as_number(), Some(42));
}

#[test]
fn test_boolean_literals() {
    // Test boolean literals
    let env = new_env();

    let result = eval_str("#t", &env).unwrap();
    assert_eq!(result.as_bool(), Some(true));

    let result = eval_str("#f", &env).unwrap();
    assert_eq!(result.as_bool(), Some(false));

    // Test that booleans are self-evaluating
    assert_bool_eq("#t", true);
    assert_bool_eq("#f", false);
}

#[test]
fn test_nil() {
    // Test nil
    let env = new_env();
    let result = eval_str("nil", &env).unwrap();
    assert!(result.is_nil());

    // Test that nil is self-evaluating
    let result = eval_test("nil").unwrap();
    assert!(result.is_nil());
}

// ============================================================================
// Arithmetic tests
// ============================================================================

#[test]
fn test_addition() {
    assert_number_eq("(+ 1 2)", 3);
    assert_number_eq("(+ 1 2 3 4)", 10);
    assert_number_eq("(+ 5)", 5);
    assert_number_eq("(+ -5 10)", 5);
    assert_number_eq("(+ 1000000 2000000)", 3000000);
}

#[test]
fn test_subtraction() {
    assert_number_eq("(- 10 4)", 6);
    assert_number_eq("(- 10 4 2)", 4);
    assert_number_eq("(- 5)", -5);
    assert_number_eq("(- 0 10)", -10);
}

#[test]
fn test_multiplication() {
    assert_number_eq("(* 2 3)", 6);
    assert_number_eq("(* 2 3 4)", 24);
    assert_number_eq("(* 5)", 5);
    assert_number_eq("(* -2 3)", -6);
}

#[test]
fn test_division() {
    assert_number_eq("(/ 10 2)", 5);
    assert_number_eq("(/ 20 5 2)", 2);
    assert_number_eq("(/ 3 2)", 1); // Integer division
    assert_number_eq("(/ -10 2)", -5);
}

#[test]
fn test_arithmetic_errors() {
    // Division by zero
    assert_err("(/ 1 0)");

    // Wrong argument types
    assert_err("(+ 1 #t)");
    assert_err("(- #f 5)");
    assert_err("(* 2 'x)");
    assert_err("(/ 'a 2)");

    // Insufficient arguments (for subtraction)
    assert_err("(-)");
}

// ============================================================================
// Comparison tests
// ============================================================================

#[test]
fn test_numeric_equality() {
    assert_bool_eq("(= 5 5)", true);
    assert_bool_eq("(= 5 6)", false);
    assert_bool_eq("(= 0 0)", true);
    assert_bool_eq("(= -5 -5)", true);
}

#[test]
fn test_less_than() {
    assert_bool_eq("(< 1 2)", true);
    assert_bool_eq("(< 2 1)", false);
    assert_bool_eq("(< 5 5)", false);
    assert_bool_eq("(< -10 -5)", true);
}

#[test]
fn test_greater_than() {
    assert_bool_eq("(> 3 2)", true);
    assert_bool_eq("(> 2 3)", false);
    assert_bool_eq("(> 5 5)", false);
    assert_bool_eq("(> -5 -10)", true);
}

#[test]
fn test_comparison_errors() {
    // Wrong number of arguments
    assert_err("(= 1)");
    assert_err("(< 1 2 3)");
    assert_err("(> 1)");

    // Wrong argument types
    assert_err("(= 1 #t)");
    assert_err("(< 'x 5)");
    assert_err("(> 2 'y)");
}

// ============================================================================
// List and pair tests
// ============================================================================

#[test]
fn test_cons() {
    // Test creating pairs
    let env = new_env();
    let result = eval_str("(cons 1 2)", &env).unwrap();
    assert!(result.as_cons().is_some());

    // Display the pair
    let mut buf = [0u8; 32];
    let display = result.to_display_str(&mut buf).unwrap();
    assert_eq!(display, "(1 . 2)");
}

#[test]
fn test_car_and_cdr() {
    assert_number_eq("(car (cons 1 2))", 1);
    assert_number_eq("(cdr (cons 1 2))", 2);

    // Nested pairs
    let env = new_env();
    let result = eval_str("(car (car (cons (cons 3 4) 5)))", &env).unwrap();
    assert_eq!(result.as_number(), Some(3));
}

#[test]
fn test_list_function() {
    // Test list creation
    let env = new_env();
    let result = eval_str("(list 1 2 3)", &env).unwrap();

    // Check it's a proper list
    let mut buf = [0u8; 32];
    let display = result.to_display_str(&mut buf).unwrap();
    assert_eq!(display, "(1 2 3)");

    // Empty list
    let result = eval_str("(list)", &env).unwrap();
    assert!(result.is_nil());
}

#[test]
fn test_null_predicate() {
    assert_bool_eq("(null? nil)", true);
    assert_bool_eq("(null? (list 1 2 3))", false);
    assert_bool_eq("(null? (cons 1 2))", false);
}

#[test]
fn test_cons_predicate() {
    assert_bool_eq("(cons? (cons 1 2))", true);
    assert_bool_eq("(cons? nil)", false);
    assert_bool_eq("(cons? 42)", false);
}

#[test]
fn test_list_functions() {
    // length
    assert_number_eq("(length (list 1 2 3 4 5))", 5);
    assert_number_eq("(length nil)", 0);

    // append
    let env = new_env();
    let result = eval_str("(append (list 1 2) (list 3 4))", &env).unwrap();
    let mut buf = [0u8; 32];
    let display = result.to_display_str(&mut buf).unwrap();
    assert_eq!(display, "(1 2 3 4)");

    // reverse
    let result = eval_str("(reverse (list 1 2 3))", &env).unwrap();
    let mut buf = [0u8; 32];
    let display = result.to_display_str(&mut buf).unwrap();
    assert_eq!(display, "(3 2 1)");
}

// ============================================================================
// Symbol and quote tests
// ============================================================================

#[test]
fn test_quote() {
    // Quote prevents evaluation
    let env = new_env();

    // Quoted symbol
    let result = eval_str("'x", &env).unwrap();
    assert!(result.as_symbol().is_some());

    // Quoted list
    let result = eval_str("'(1 2 3)", &env).unwrap();
    let mut buf = [0u8; 32];
    let display = result.to_display_str(&mut buf).unwrap();
    assert_eq!(display, "(1 2 3)");

    // Nested quotes
    let result = eval_str("'(a (b c) d)", &env).unwrap();
    let mut buf = [0u8; 32];
    let display = result.to_display_str(&mut buf).unwrap();
    assert_eq!(display, "(a (b c) d)");
}

// ============================================================================
// Define tests
// ============================================================================

#[test]
fn test_define() {
    // Simple definition
    let program = r#"
        (define x 42)
        x
    "#;
    assert_number_eq(program, 42);

    // Define with computation
    let program = r#"
        (define y (+ 10 20))
        y
    "#;
    assert_number_eq(program, 30);

    // Redefinition
    let program = r#"
        (define z 1)
        (define z 2)
        z
    "#;
    assert_number_eq(program, 2);

    // Using defined values
    let program = r#"
        (define a 5)
        (define b 10)
        (+ a b)
    "#;
    assert_number_eq(program, 15);
}

#[test]
fn test_define_errors() {
    // Wrong number of arguments
    assert_err("(define)");
    assert_err("(define x)");
    assert_err("(define x 1 2)");

    // First argument must be a symbol
    assert_err("(define 42 100)");
    assert_err("(define (+ 1 2) 100)");
}

// ============================================================================
// Lambda tests
// ============================================================================

#[test]
fn test_lambda() {
    // Simple lambda application
    assert_number_eq("((lambda (x) (+ x 1)) 5)", 6);

    // Multiple parameters
    assert_number_eq("((lambda (x y) (+ x y)) 3 4)", 7);

    // Lambda with no parameters
    assert_number_eq("((lambda () 42))", 42);
}

#[test]
fn test_define_lambda() {
    // Define a function
    let program = r#"
        (define add1 (lambda (x) (+ x 1)))
        (add1 10)
    "#;
    assert_number_eq(program, 11);

    // Multiple functions
    let program = r#"
        (define square (lambda (x) (* x x)))
        (define add (lambda (x y) (+ x y)))
        (square (add 3 4))
    "#;
    assert_number_eq(program, 49);
}

#[test]
fn test_nested_lambda() {
    // Nested lambda application
    assert_number_eq("((lambda (x) ((lambda (y) (+ x y)) 3)) 4)", 7);

    // Lambda returning lambda
    let program = r#"
        (define make-adder (lambda (n) (lambda (x) (+ x n))))
        (define add5 (make-adder 5))
        (add5 10)
    "#;
    assert_number_eq(program, 15);
}

#[test]
fn test_lambda_errors() {
    // Wrong number of arguments
    assert_err("(lambda)");
    assert_err("(lambda (x))");

    // Parameter must be a symbol
    assert_err("(lambda (1) 2)");
    assert_err("(lambda ((+ 1 2)) 3)");

    // Wrong number of arguments in application
    assert_err("((lambda (x y) (+ x y)) 1)");
    assert_err("((lambda (x) x) 1 2)");
}

// ============================================================================
// If tests
// ============================================================================

#[test]
fn test_if() {
    // Basic if
    assert_number_eq("(if #t 1 2)", 1);
    assert_number_eq("(if #f 1 2)", 2);

    // Truthiness
    assert_number_eq("(if 0 10 20)", 10); // Non-zero is truthy
    assert_number_eq("(if nil 10 20)", 20); // nil is falsey

    // Using comparisons
    assert_number_eq("(if (< 1 2) 100 200)", 100);
    assert_number_eq("(if (> 1 2) 100 200)", 200);
}

#[test]
fn test_if_errors() {
    // Wrong number of arguments
    assert_err("(if)");
    assert_err("(if #t)");
    assert_err("(if #t 1 2 3)");
}

// ============================================================================
// Recursion tests
// ============================================================================

#[test]
fn test_recursive_factorial() {
    let program = r#"
        (define factorial 
          (lambda (n)
            (if (= n 0)
                1
                (* n (factorial (- n 1))))))
        (factorial 5)
    "#;
    assert_number_eq(program, 120);
}

#[test]
fn test_tail_recursive_factorial() {
    let program = r#"
        (define factorial-tail
          (lambda (n acc)
            (if (= n 0)
                acc
                (factorial-tail (- n 1) (* n acc)))))
        (factorial-tail 5 1)
    "#;
    assert_number_eq(program, 120);
}

#[test]
fn test_mutual_recursion() {
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
        (is-even 4)
    "#;
    assert_bool_eq(program, true);
}

// ============================================================================
// Higher-order function tests
// ============================================================================

#[test]
fn test_map_function() {
    let program = r#"
        (define map
          (lambda (f lst)
            (if (null? lst)
                nil
                (cons (f (car lst))
                      (map f (cdr lst))))))
        (map (lambda (x) (+ x 1)) (list 1 2 3))
    "#;

    let env = new_env();
    let result = eval_str_multiple(program, &env).unwrap();
    let mut buf = [0u8; 32];
    let display = result.to_display_str(&mut buf).unwrap();
    assert_eq!(display, "(2 3 4)");
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
        (filter (lambda (x) (> x 2)) (list 1 2 3 4 5))
    "#;

    let env = new_env();
    let result = eval_str_multiple(program, &env).unwrap();
    let mut buf = [0u8; 32];
    let display = result.to_display_str(&mut buf).unwrap();
    assert_eq!(display, "(3 4 5)");
}

// ============================================================================
// Complex expression tests
// ============================================================================

#[test]
fn test_complex_nesting() {
    assert_number_eq("(+ 1 (* 2 3) (- 10 4))", 13);
    assert_number_eq("(* (+ 1 2) (- 10 3))", 21);
}

#[test]
fn test_function_composition() {
    let program = r#"
        (define compose
          (lambda (f g)
            (lambda (x)
              (f (g x)))))
        (define add2 (lambda (x) (+ x 2)))
        (define mul3 (lambda (x) (* x 3)))
        ((compose add2 mul3) 5)
    "#;
    assert_number_eq(program, 17); // (5 * 3) + 2 = 17
}

#[test]
fn test_y_combinator() {
    // Y-combinator for factorial
    let program = r#"
        (define Y
          (lambda (f)
            ((lambda (x) (f (lambda (y) ((x x) y))))
             (lambda (x) (f (lambda (y) ((x x) y)))))))
        (define factorial
          (Y (lambda (fact)
               (lambda (n)
                 (if (= n 0)
                     1
                     (* n (fact (- n 1))))))))
        (factorial 5)
    "#;
    assert_number_eq(program, 120);
}

// ============================================================================
// Environment tests
// ============================================================================

#[test]
fn test_environment_isolation() {
    // Test that environments are isolated
    let env1 = new_env();
    let env2 = new_env();

    // Define x in env1
    eval_str("(define x 42)", &env1).unwrap();

    // Should find x in env1
    let result = eval_str("x", &env1).unwrap();
    assert_eq!(result.as_number(), Some(42));

    // Should NOT find x in env2
    let result = eval_str("x", &env2);
    assert!(result.is_err());

    // Define different x in env2
    eval_str("(define x 100)", &env2).unwrap();

    let result = eval_str("x", &env2).unwrap();
    assert_eq!(result.as_number(), Some(100));

    // env1 should still have its own x
    let result = eval_str("x", &env1).unwrap();
    assert_eq!(result.as_number(), Some(42));
}

#[test]
fn test_closures() {
    // Test that closures capture their environment
    let program = r#"
        (define make-counter
          (lambda ()
            (define count 0)
            (lambda ()
              (define count (+ count 1))
              count)))
        (define counter1 (make-counter))
        (define counter2 (make-counter))
        (list (counter1) (counter1) (counter2) (counter1) (counter2))
    "#;

    let env = new_env();
    let result = eval_str_multiple(program, &env).unwrap();
    let mut buf = [0u8; 32];
    let display = result.to_display_str(&mut buf).unwrap();
    assert_eq!(display, "(1 2 1 3 2)");
}

// ============================================================================
// Edge case tests
// ============================================================================

#[test]
fn test_empty_program() {
    assert_err("");
    assert_err("   ");
    assert_err("; comment only");
}

#[test]
fn test_comments() {
    assert_number_eq("; comment\n42", 42);
    assert_number_eq("(+ 1 2) ; comment", 3);
    assert_number_eq("(+ 1 ; comment\n 2)", 3);
}

#[test]
fn test_whitespace() {
    assert_number_eq("   42   ", 42);
    assert_number_eq("( +   1   2   ) ", 3);
    assert_number_eq("(+\n1\n2\n3)", 6);
}

#[test]
fn test_nested_parentheses() {
    assert_number_eq(
        "((((lambda (x) (lambda (y) (lambda (z) (+ x (+ y z))))) 1) 2) 3)",
        6,
    );
}

#[test]
fn test_unbound_symbol() {
    assert_err("undefined-symbol");
}

// ============================================================================
// Tail call optimization tests
// ============================================================================

#[test]
fn test_tail_recursion_depth() {
    // This should work with TCO (no stack overflow)
    let program = r#"
        (define count-down
          (lambda (n)
            (if (= n 0)
                0
                (count-down (- n 1)))))
        (count-down 10000)
    "#;
    assert_number_eq(program, 0);
}

#[test]
fn test_mutual_tail_recursion() {
    // Mutual tail recursion
    let program = r#"
        (define is-even-tail
          (lambda (n)
            (if (= n 0)
                #t
                (is-odd-tail (- n 1)))))
        (define is-odd-tail
          (lambda (n)
            (if (= n 0)
                #f
                (is-even-tail (- n 1)))))
        (is-even-tail 10000)
    "#;
    assert_bool_eq(program, true);
}

// ============================================================================
// String representation tests
// ============================================================================

#[test]
fn test_display_representations() {
    let env = new_env();

    // Numbers
    let result = eval_str("42", &env).unwrap();
    let mut buf = [0u8; 32];
    let display = result.to_display_str(&mut buf).unwrap();
    assert_eq!(display, "42");

    // Booleans
    let result = eval_str("#t", &env).unwrap();
    let mut buf = [0u8; 32];
    let display = result.to_display_str(&mut buf).unwrap();
    assert_eq!(display, "#t");

    // Nil
    let result = eval_str("nil", &env).unwrap();
    let mut buf = [0u8; 32];
    let display = result.to_display_str(&mut buf).unwrap();
    assert_eq!(display, "nil");

    // Lists
    let result = eval_str("(list 1 2 3)", &env).unwrap();
    let mut buf = [0u8; 32];
    let display = result.to_display_str(&mut buf).unwrap();
    assert_eq!(display, "(1 2 3)");

    // Pairs
    let result = eval_str("(cons 1 2)", &env).unwrap();
    let mut buf = [0u8; 32];
    let display = result.to_display_str(&mut buf).unwrap();
    assert_eq!(display, "(1 . 2)");

    // Improper lists
    let result = eval_str("(cons 1 (cons 2 3))", &env).unwrap();
    let mut buf = [0u8; 32];
    let display = result.to_display_str(&mut buf).unwrap();
    assert_eq!(display, "(1 2 . 3)");
}

// ============================================================================
// Multiple expression tests
// ============================================================================

#[test]
fn test_multiple_expressions() {
    let program = r#"
        (define a 1)
        (define b 2)
        (define c (+ a b))
        (* c 10)
    "#;
    assert_number_eq(program, 30);

    // Test that intermediate results are available
    let program = r#"
        (define x 5)
        (define y (* x 2))
        (define z (+ x y))
        z
    "#;
    assert_number_eq(program, 15);
}

// ============================================================================
// Performance/stress tests
// ============================================================================

#[test]
fn test_large_list_operations() {
    // Create and process a large list
    let program = r#"
        (define make-list
          (lambda (n acc)
            (if (= n 0)
                acc
                (make-list (- n 1) (cons n acc)))))
        (define lst (make-list 1000 nil))
        (length lst)
    "#;
    assert_number_eq(program, 1000);
}

#[test]
fn test_deep_recursion() {
    // Test deep but not infinite recursion
    let program = r#"
        (define deep-nest
          (lambda (n)
            (if (= n 0)
                42
                (deep-nest (- n 1)))))
        (deep-nest 1000)
    "#;
    assert_number_eq(program, 42);
}

// ============================================================================
// Integration tests
// ============================================================================

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
    assert_number_eq(program, 55);
}

#[test]
fn test_factorial_variations() {
    // Regular factorial
    let program1 = r#"
        (define fact
          (lambda (n)
            (if (= n 0)
                1
                (* n (fact (- n 1))))))
        (fact 6)
    "#;
    assert_number_eq(program1, 720);

    // Tail-recursive factorial
    let program2 = r#"
        (define fact-tail
          (lambda (n acc)
            (if (= n 0)
                acc
                (fact-tail (- n 1) (* n acc)))))
        (fact-tail 6 1)
    "#;
    assert_number_eq(program2, 720);
}

#[test]
fn test_list_processing() {
    let program = r#"
        (define sum-list
          (lambda (lst)
            (if (null? lst)
                0
                (+ (car lst) (sum-list (cdr lst))))))
        (sum-list (list 1 2 3 4 5 6 7 8 9 10))
    "#;
    assert_number_eq(program, 55);
}

// ============================================================================
// Memory/GC tests (lightweight)
// ============================================================================

#[test]
fn test_memory_reuse() {
    // Test that we can create and discard many values
    // Note: 'begin' isn't implemented, so let's adjust
    let program = r#"
        (define loop
          (lambda (n)
            (if (= n 0)
                0
                ((lambda (dummy) (loop (- n 1))) (cons n nil)))))
        (loop 100)
    "#;
    assert_number_eq(program, 0);
}
