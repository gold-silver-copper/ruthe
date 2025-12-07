// tests/tests.rs

use ruthe::Interpreter;

fn setup() -> (Interpreter<4096>, ruthe::EnvRef) {
    let interp = Interpreter::new();
    let env = interp.create_global_env().expect("Failed to create env");
    (interp, env)
}

fn eval_to_string(
    interp: &Interpreter<4096>,
    env: ruthe::EnvRef,
    expr: &str,
) -> Result<String, String> {
    match interp.eval_str(expr, env) {
        Ok(buf) => {
            let len = buf.iter().position(|&b| b == 0).unwrap_or(4096);
            Ok(String::from_utf8_lossy(&buf[..len]).to_string())
        }
        Err(e) => {
            let mut buf = [0u8; 128];
            let err_str = interp.strings.get(e, &mut buf).unwrap_or("unknown error");
            Err(err_str.to_string())
        }
    }
}

// ============================================================================
// Basic Arithmetic Tests
// ============================================================================

#[test]
fn test_addition() {
    let (interp, env) = setup();
    assert_eq!(eval_to_string(&interp, env, "(+ 1 2 3)").unwrap(), "6");
    assert_eq!(eval_to_string(&interp, env, "(+ 0)").unwrap(), "0");
    assert_eq!(eval_to_string(&interp, env, "(+)").unwrap(), "0");
    assert_eq!(eval_to_string(&interp, env, "(+ -5 10)").unwrap(), "5");
}

#[test]
fn test_subtraction() {
    let (interp, env) = setup();
    assert_eq!(eval_to_string(&interp, env, "(- 10 3)").unwrap(), "7");
    assert_eq!(eval_to_string(&interp, env, "(- 5)").unwrap(), "-5");
    assert_eq!(eval_to_string(&interp, env, "(- 10 3 2)").unwrap(), "5");
    assert_eq!(eval_to_string(&interp, env, "(- 0 5)").unwrap(), "-5");
}

#[test]
fn test_multiplication() {
    let (interp, env) = setup();
    assert_eq!(eval_to_string(&interp, env, "(* 2 3 4)").unwrap(), "24");
    assert_eq!(eval_to_string(&interp, env, "(* 5)").unwrap(), "5");
    assert_eq!(eval_to_string(&interp, env, "(*)").unwrap(), "1");
    assert_eq!(eval_to_string(&interp, env, "(* -2 3)").unwrap(), "-6");
}

#[test]
fn test_division() {
    let (interp, env) = setup();
    assert_eq!(eval_to_string(&interp, env, "(/ 10 2)").unwrap(), "5");
    assert_eq!(eval_to_string(&interp, env, "(/ 20 2 2)").unwrap(), "5");
    assert_eq!(eval_to_string(&interp, env, "(/ 7 2)").unwrap(), "3");
    assert!(eval_to_string(&interp, env, "(/ 10 0)").is_err());
}

#[test]
fn test_nested_arithmetic() {
    let (interp, env) = setup();
    assert_eq!(
        eval_to_string(&interp, env, "(+ (* 2 3) (/ 10 2))").unwrap(),
        "11"
    );
    assert_eq!(
        eval_to_string(&interp, env, "(* (+ 1 2) (- 5 2))").unwrap(),
        "9"
    );
    assert_eq!(
        eval_to_string(&interp, env, "(/ (+ 10 20) (- 10 5))").unwrap(),
        "6"
    );
}

// ============================================================================
// Comparison Tests
// ============================================================================

#[test]
fn test_equality() {
    let (interp, env) = setup();
    assert_eq!(eval_to_string(&interp, env, "(= 5 5)").unwrap(), "#t");
    assert_eq!(eval_to_string(&interp, env, "(= 5 6)").unwrap(), "#f");
    assert_eq!(eval_to_string(&interp, env, "(= 0 0)").unwrap(), "#t");
    assert_eq!(eval_to_string(&interp, env, "(= -5 -5)").unwrap(), "#t");
}

#[test]
fn test_less_than() {
    let (interp, env) = setup();
    assert_eq!(eval_to_string(&interp, env, "(< 3 5)").unwrap(), "#t");
    assert_eq!(eval_to_string(&interp, env, "(< 5 3)").unwrap(), "#f");
    assert_eq!(eval_to_string(&interp, env, "(< 5 5)").unwrap(), "#f");
    assert_eq!(eval_to_string(&interp, env, "(< -10 0)").unwrap(), "#t");
}

#[test]
fn test_greater_than() {
    let (interp, env) = setup();
    assert_eq!(eval_to_string(&interp, env, "(> 5 3)").unwrap(), "#t");
    assert_eq!(eval_to_string(&interp, env, "(> 3 5)").unwrap(), "#f");
    assert_eq!(eval_to_string(&interp, env, "(> 5 5)").unwrap(), "#f");
    assert_eq!(eval_to_string(&interp, env, "(> 0 -10)").unwrap(), "#t");
}

// ============================================================================
// Boolean Tests
// ============================================================================

#[test]
fn test_booleans() {
    let (interp, env) = setup();
    assert_eq!(eval_to_string(&interp, env, "#t").unwrap(), "#t");
    assert_eq!(eval_to_string(&interp, env, "#f").unwrap(), "#f");
}

// ============================================================================
// List Operations Tests
// ============================================================================

#[test]
fn test_list_creation() {
    let (interp, env) = setup();
    assert_eq!(
        eval_to_string(&interp, env, "(list 1 2 3)").unwrap(),
        "(1 2 3)"
    );
    assert_eq!(eval_to_string(&interp, env, "(list)").unwrap(), "nil");
    assert_eq!(eval_to_string(&interp, env, "(list 1)").unwrap(), "(1)");
}

#[test]
fn test_cons() {
    let (interp, env) = setup();
    assert_eq!(eval_to_string(&interp, env, "(cons 1 nil)").unwrap(), "(1)");
    assert_eq!(
        eval_to_string(&interp, env, "(cons 1 (cons 2 nil))").unwrap(),
        "(1 2)"
    );
    assert_eq!(
        eval_to_string(&interp, env, "(cons 1 2)").unwrap(),
        "(1 . 2)"
    );
}

#[test]
fn test_car() {
    let (interp, env) = setup();
    assert_eq!(
        eval_to_string(&interp, env, "(car (list 1 2 3))").unwrap(),
        "1"
    );
    assert_eq!(eval_to_string(&interp, env, "(car '(5 6 7))").unwrap(), "5");
    assert!(eval_to_string(&interp, env, "(car nil)").is_err());
}

#[test]
fn test_cdr() {
    let (interp, env) = setup();
    assert_eq!(
        eval_to_string(&interp, env, "(cdr (list 1 2 3))").unwrap(),
        "(2 3)"
    );
    assert_eq!(
        eval_to_string(&interp, env, "(cdr '(5 6 7))").unwrap(),
        "(6 7)"
    );
    assert_eq!(
        eval_to_string(&interp, env, "(cdr (list 1))").unwrap(),
        "nil"
    );
}

#[test]
fn test_null_predicate() {
    let (interp, env) = setup();
    assert_eq!(eval_to_string(&interp, env, "(null? nil)").unwrap(), "#t");
    assert_eq!(
        eval_to_string(&interp, env, "(null? (list))").unwrap(),
        "#t"
    );
    assert_eq!(
        eval_to_string(&interp, env, "(null? (list 1))").unwrap(),
        "#f"
    );
    assert_eq!(eval_to_string(&interp, env, "(null? 5)").unwrap(), "#f");
}

#[test]
fn test_cons_predicate() {
    let (interp, env) = setup();
    assert_eq!(
        eval_to_string(&interp, env, "(cons? (list 1 2))").unwrap(),
        "#t"
    );
    assert_eq!(
        eval_to_string(&interp, env, "(cons? (cons 1 2))").unwrap(),
        "#t"
    );
    assert_eq!(eval_to_string(&interp, env, "(cons? nil)").unwrap(), "#f");
    assert_eq!(eval_to_string(&interp, env, "(cons? 5)").unwrap(), "#f");
}

#[test]
fn test_length() {
    let (interp, env) = setup();
    assert_eq!(
        eval_to_string(&interp, env, "(length (list 1 2 3))").unwrap(),
        "3"
    );
    assert_eq!(
        eval_to_string(&interp, env, "(length (list))").unwrap(),
        "0"
    );
    assert_eq!(
        eval_to_string(&interp, env, "(length '(a b c d e))").unwrap(),
        "5"
    );
}

#[test]
fn test_reverse() {
    let (interp, env) = setup();
    assert_eq!(
        eval_to_string(&interp, env, "(reverse (list 1 2 3))").unwrap(),
        "(3 2 1)"
    );
    assert_eq!(
        eval_to_string(&interp, env, "(reverse (list))").unwrap(),
        "nil"
    );
    assert_eq!(
        eval_to_string(&interp, env, "(reverse (list 1))").unwrap(),
        "(1)"
    );
}

#[test]
fn test_append() {
    let (interp, env) = setup();
    assert_eq!(
        eval_to_string(&interp, env, "(append (list 1 2) (list 3 4))").unwrap(),
        "(1 2 3 4)"
    );
    assert_eq!(
        eval_to_string(&interp, env, "(append (list 1) (list 2) (list 3))").unwrap(),
        "(1 2 3)"
    );
    assert_eq!(
        eval_to_string(&interp, env, "(append (list) (list 1))").unwrap(),
        "(1)"
    );
}

// ============================================================================
// Quote Tests
// ============================================================================

#[test]
fn test_quote() {
    let (interp, env) = setup();
    assert_eq!(eval_to_string(&interp, env, "'5").unwrap(), "5");
    assert_eq!(eval_to_string(&interp, env, "'(1 2 3)").unwrap(), "(1 2 3)");
    assert_eq!(eval_to_string(&interp, env, "(quote x)").unwrap(), "x");
    assert_eq!(eval_to_string(&interp, env, "'(+ 1 2)").unwrap(), "(+ 1 2)");
}

// ============================================================================
// If Expression Tests
// ============================================================================

#[test]
fn test_if_true() {
    let (interp, env) = setup();
    assert_eq!(eval_to_string(&interp, env, "(if #t 1 2)").unwrap(), "1");
    assert_eq!(
        eval_to_string(&interp, env, "(if (= 5 5) 10 20)").unwrap(),
        "10"
    );
}

#[test]
fn test_if_false() {
    let (interp, env) = setup();
    assert_eq!(eval_to_string(&interp, env, "(if #f 1 2)").unwrap(), "2");
    assert_eq!(
        eval_to_string(&interp, env, "(if (= 5 6) 10 20)").unwrap(),
        "20"
    );
}

#[test]
fn test_if_nil_is_false() {
    let (interp, env) = setup();
    assert_eq!(eval_to_string(&interp, env, "(if nil 1 2)").unwrap(), "2");
}

#[test]
fn test_if_nonzero_is_true() {
    let (interp, env) = setup();
    assert_eq!(eval_to_string(&interp, env, "(if 5 1 2)").unwrap(), "1");
    assert_eq!(eval_to_string(&interp, env, "(if 0 1 2)").unwrap(), "1");
}

#[test]
fn test_nested_if() {
    let (interp, env) = setup();
    let expr = "(if (> 5 3) (if (< 2 4) 1 2) 3)";
    assert_eq!(eval_to_string(&interp, env, expr).unwrap(), "1");
}

// ============================================================================
// Define Tests
// ============================================================================

#[test]
fn test_define_number() {
    let (interp, env) = setup();
    assert_eq!(eval_to_string(&interp, env, "(define x 42)").unwrap(), "42");
    assert_eq!(eval_to_string(&interp, env, "x").unwrap(), "42");
}

#[test]
fn test_define_expression() {
    let (interp, env) = setup();
    eval_to_string(&interp, env, "(define y (+ 1 2 3))").unwrap();
    assert_eq!(eval_to_string(&interp, env, "y").unwrap(), "6");
}

#[test]
fn test_define_list() {
    let (interp, env) = setup();
    eval_to_string(&interp, env, "(define lst (list 1 2 3))").unwrap();
    assert_eq!(eval_to_string(&interp, env, "lst").unwrap(), "(1 2 3)");
}

// ============================================================================
// Lambda Tests
// ============================================================================

#[test]
fn test_lambda_creation() {
    let (interp, env) = setup();
    assert_eq!(
        eval_to_string(&interp, env, "(lambda (x) x)").unwrap(),
        "<lambda>"
    );
}

#[test]
fn test_lambda_application() {
    let (interp, env) = setup();
    assert_eq!(
        eval_to_string(&interp, env, "((lambda (x) x) 5)").unwrap(),
        "5"
    );
    assert_eq!(
        eval_to_string(&interp, env, "((lambda (x) (* x 2)) 5)").unwrap(),
        "10"
    );
}

#[test]
fn test_lambda_multiple_params() {
    let (interp, env) = setup();
    assert_eq!(
        eval_to_string(&interp, env, "((lambda (x y) (+ x y)) 3 4)").unwrap(),
        "7"
    );
    assert_eq!(
        eval_to_string(&interp, env, "((lambda (a b c) (* a b c)) 2 3 4)").unwrap(),
        "24"
    );
}

#[test]
fn test_lambda_with_define() {
    let (interp, env) = setup();
    eval_to_string(&interp, env, "(define square (lambda (x) (* x x)))").unwrap();
    assert_eq!(eval_to_string(&interp, env, "(square 5)").unwrap(), "25");
    assert_eq!(eval_to_string(&interp, env, "(square 10)").unwrap(), "100");
}

#[test]
fn test_nested_lambda() {
    let (interp, env) = setup();
    let expr = "(define make-adder (lambda (n) (lambda (x) (+ x n))))";
    eval_to_string(&interp, env, expr).unwrap();
    eval_to_string(&interp, env, "(define add5 (make-adder 5))").unwrap();
    assert_eq!(eval_to_string(&interp, env, "(add5 10)").unwrap(), "15");
    assert_eq!(eval_to_string(&interp, env, "(add5 3)").unwrap(), "8");
}

#[test]
fn test_closure() {
    let (interp, env) = setup();
    eval_to_string(&interp, env, "(define x 10)").unwrap();
    eval_to_string(&interp, env, "(define f (lambda (y) (+ x y)))").unwrap();
    assert_eq!(eval_to_string(&interp, env, "(f 5)").unwrap(), "15");
}

// ============================================================================
// Recursion Tests
// ============================================================================

#[test]
fn test_factorial() {
    let (interp, env) = setup();
    let fact_def = r#"
        (define fact
            (lambda (n)
                (if (= n 0)
                    1
                    (* n (fact (- n 1)))
                )
            )
        )
    "#;
    eval_to_string(&interp, env, fact_def).unwrap();
    assert_eq!(eval_to_string(&interp, env, "(fact 0)").unwrap(), "1");
    assert_eq!(eval_to_string(&interp, env, "(fact 1)").unwrap(), "1");
    assert_eq!(eval_to_string(&interp, env, "(fact 5)").unwrap(), "120");
    assert_eq!(
        eval_to_string(&interp, env, "(fact 10)").unwrap(),
        "3628800"
    );
}

#[test]
fn test_fibonacci() {
    let (interp, env) = setup();
    let fib_def = r#"
        (define fib
            (lambda (n)
                (if (= n 0)
                    0
                    (if (= n 1)
                        1
                        (+ (fib (- n 1)) (fib (- n 2)))
                    )
                )
            )
        )
    "#;
    eval_to_string(&interp, env, fib_def).unwrap();
    assert_eq!(eval_to_string(&interp, env, "(fib 0)").unwrap(), "0");
    assert_eq!(eval_to_string(&interp, env, "(fib 1)").unwrap(), "1");
    assert_eq!(eval_to_string(&interp, env, "(fib 5)").unwrap(), "5");
    assert_eq!(eval_to_string(&interp, env, "(fib 10)").unwrap(), "55");
}

#[test]
fn test_tail_recursion() {
    let (interp, env) = setup();
    let countdown_def = r#"
        (define countdown
            (lambda (n)
                (if (= n 0)
                    0
                    (countdown (- n 1))
                )
            )
        )
    "#;
    eval_to_string(&interp, env, countdown_def).unwrap();
    assert_eq!(
        eval_to_string(&interp, env, "(countdown 100)").unwrap(),
        "0"
    );
    assert_eq!(
        eval_to_string(&interp, env, "(countdown 1000)").unwrap(),
        "0"
    );
}

// ============================================================================
// Complex Programs
// ============================================================================

#[test]
fn test_list_sum() {
    let (interp, env) = setup();
    let sum_def = r#"
        (define sum
            (lambda (lst)
                (if (null? lst)
                    0
                    (+ (car lst) (sum (cdr lst)))
                )
            )
        )
    "#;
    eval_to_string(&interp, env, sum_def).unwrap();
    assert_eq!(
        eval_to_string(&interp, env, "(sum (list 1 2 3 4 5))").unwrap(),
        "15"
    );
    assert_eq!(eval_to_string(&interp, env, "(sum (list))").unwrap(), "0");
}

#[test]
fn test_map() {
    let (interp, env) = setup();
    let map_def = r#"
        (define map
            (lambda (f lst)
                (if (null? lst)
                    nil
                    (cons (f (car lst)) (map f (cdr lst)))
                )
            )
        )
    "#;
    eval_to_string(&interp, env, map_def).unwrap();
    eval_to_string(&interp, env, "(define double (lambda (x) (* x 2)))").unwrap();
    assert_eq!(
        eval_to_string(&interp, env, "(map double (list 1 2 3))").unwrap(),
        "(2 4 6)"
    );
}

#[test]
fn test_filter() {
    let (interp, env) = setup();
    let filter_def = r#"
        (define filter
            (lambda (pred lst)
                (if (null? lst)
                    nil
                    (if (pred (car lst))
                        (cons (car lst) (filter pred (cdr lst)))
                        (filter pred (cdr lst))
                    )
                )
            )
        )
    "#;
    eval_to_string(&interp, env, filter_def).unwrap();
    eval_to_string(&interp, env, "(define positive? (lambda (x) (> x 0)))").unwrap();
    assert_eq!(
        eval_to_string(&interp, env, "(filter positive? (list -1 2 -3 4 5))").unwrap(),
        "(2 4 5)"
    );
}

// ============================================================================
// Error Cases
// ============================================================================

#[test]
fn test_unbound_symbol() {
    let (interp, env) = setup();
    assert!(eval_to_string(&interp, env, "undefined-var").is_err());
}

#[test]
fn test_invalid_function_call() {
    let (interp, env) = setup();
    assert!(eval_to_string(&interp, env, "(5 10)").is_err());
}

#[test]
fn test_wrong_argument_count() {
    let (interp, env) = setup();
    eval_to_string(&interp, env, "(define f (lambda (x y) (+ x y)))").unwrap();
    assert!(eval_to_string(&interp, env, "(f 1)").is_err());
    assert!(eval_to_string(&interp, env, "(f 1 2 3)").is_err());
}

#[test]
fn test_type_errors() {
    let (interp, env) = setup();
    assert!(eval_to_string(&interp, env, "(+ 1 #t)").is_err());
    assert!(eval_to_string(&interp, env, "(car 5)").is_err());
}

// ============================================================================
// Edge Cases
// ============================================================================

#[test]
fn test_empty_list() {
    let (interp, env) = setup();
    assert_eq!(eval_to_string(&interp, env, "nil").unwrap(), "nil");
    assert_eq!(eval_to_string(&interp, env, "(list)").unwrap(), "nil");
}

#[test]
fn test_nested_lists() {
    let (interp, env) = setup();
    assert_eq!(
        eval_to_string(&interp, env, "(list (list 1 2) (list 3 4))").unwrap(),
        "((1 2) (3 4))"
    );
}

#[test]
fn test_zero_operations() {
    let (interp, env) = setup();
    assert_eq!(eval_to_string(&interp, env, "(+ 0 0)").unwrap(), "0");
    assert_eq!(eval_to_string(&interp, env, "(* 5 0)").unwrap(), "0");
    assert_eq!(eval_to_string(&interp, env, "(- 0 0)").unwrap(), "0");
}

#[test]
fn test_negative_numbers() {
    let (interp, env) = setup();
    assert_eq!(eval_to_string(&interp, env, "-5").unwrap(), "-5");
    assert_eq!(eval_to_string(&interp, env, "(+ -5 10)").unwrap(), "5");
    assert_eq!(eval_to_string(&interp, env, "(* -2 -3)").unwrap(), "6");
}

// ============================================================================
// Symbol Tests
// ============================================================================

#[test]
fn test_symbol_names() {
    let (interp, env) = setup();
    eval_to_string(&interp, env, "(define foo 123)").unwrap();
    eval_to_string(&interp, env, "(define bar-baz 456)").unwrap();
    assert_eq!(eval_to_string(&interp, env, "foo").unwrap(), "123");
    assert_eq!(eval_to_string(&interp, env, "bar-baz").unwrap(), "456");
}

// ============================================================================
// Shadowing Tests
// ============================================================================

#[test]
fn test_lambda_parameter_shadowing() {
    let (interp, env) = setup();
    eval_to_string(&interp, env, "(define x 100)").unwrap();
    eval_to_string(&interp, env, "(define f (lambda (x) (* x 2)))").unwrap();
    assert_eq!(eval_to_string(&interp, env, "(f 5)").unwrap(), "10");
    assert_eq!(eval_to_string(&interp, env, "x").unwrap(), "100");
}

// ============================================================================
// Comments Test
// ============================================================================

#[test]
fn test_comments() {
    let (interp, env) = setup();
    let expr = r#"
        ; This is a comment
        (+ 1 2) ; Another comment
        ; (+ 3 4) This should be ignored
    "#;
    assert_eq!(eval_to_string(&interp, env, expr).unwrap(), "3");
}
