#![cfg(test)]

use ruthe::*;
extern crate alloc;

// ============================================================================
// Helper Functions
// ============================================================================

fn eval_test(input: &str) -> Result<String, String> {
    let env = new_env();
    eval_str(input, &env)
        .map(|s| {
            let mut buf = [0u8; 4096];
            s.to_display_str(&mut buf).unwrap().to_string()
        })
        .map_err(|e| {
            let mut buf = [0u8; 256];
            e.to_display_str(&mut buf).unwrap().to_string()
        })
}

fn eval_multiple_test(input: &str) -> Result<String, String> {
    let env = new_env();
    eval_str_multiple(input, &env)
        .map(|s| {
            let mut buf = [0u8; 4096];
            s.to_display_str(&mut buf).unwrap().to_string()
        })
        .map_err(|e| {
            let mut buf = [0u8; 256];
            e.to_display_str(&mut buf).unwrap().to_string()
        })
}

// Helper for tests that need to evaluate multiple expressions with shared environment
fn eval_with_env(input: &str, env: &ValRef) -> Result<String, String> {
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

// ============================================================================
// Basic Value Tests
// ============================================================================

#[test]
fn test_numbers() {
    assert_eq!(eval_test("42"), Ok("42".to_string()));
    assert_eq!(eval_test("-17"), Ok("-17".to_string()));
    assert_eq!(eval_test("0"), Ok("0".to_string()));
    assert_eq!(eval_test("999999"), Ok("999999".to_string()));
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
    assert_eq!(eval_test("(+ 0 0)"), Ok("0".to_string()));
}

#[test]
fn test_subtraction() {
    assert_eq!(eval_test("(- 10 3)"), Ok("7".to_string()));
    assert_eq!(eval_test("(- 5)"), Ok("-5".to_string()));
    assert_eq!(eval_test("(- 20 5 3)"), Ok("12".to_string()));
    assert_eq!(eval_test("(- 0 5)"), Ok("-5".to_string()));
    assert_eq!(eval_test("(- 100 100)"), Ok("0".to_string()));
}

#[test]
fn test_multiplication() {
    assert_eq!(eval_test("(* 2 3)"), Ok("6".to_string()));
    assert_eq!(eval_test("(* 2 3 4)"), Ok("24".to_string()));
    assert_eq!(eval_test("(*)"), Ok("1".to_string()));
    assert_eq!(eval_test("(* -2 5)"), Ok("-10".to_string()));
    assert_eq!(eval_test("(* 0 100)"), Ok("0".to_string()));
}

#[test]
fn test_division() {
    assert_eq!(eval_test("(/ 10 2)"), Ok("5".to_string()));
    assert_eq!(eval_test("(/ 20 4 2)"), Ok("2".to_string()));
    assert_eq!(eval_test("(/ 100 10)"), Ok("10".to_string()));
    assert_eq!(eval_test("(/ 7 2)"), Ok("3".to_string())); // Integer division
    assert!(eval_test("(/ 10 0)").is_err());
}

#[test]
fn test_nested_arithmetic() {
    assert_eq!(eval_test("(+ (* 2 3) (- 10 5))"), Ok("11".to_string()));
    assert_eq!(eval_test("(* (+ 1 2) (+ 3 4))"), Ok("21".to_string()));
    assert_eq!(eval_test("(- (* 10 5) (/ 20 4))"), Ok("45".to_string()));
    assert_eq!(
        eval_test("(+ (- 10 5) (* 2 3) (/ 20 4))"),
        Ok("16".to_string())
    );
}

// ============================================================================
// Comparison Tests
// ============================================================================

#[test]
fn test_equality() {
    assert_eq!(eval_test("(= 5 5)"), Ok("#t".to_string()));
    assert_eq!(eval_test("(= 5 6)"), Ok("#f".to_string()));
    assert_eq!(eval_test("(= (+ 2 3) 5)"), Ok("#t".to_string()));
    assert_eq!(eval_test("(= 0 0)"), Ok("#t".to_string()));
    assert_eq!(eval_test("(= -5 -5)"), Ok("#t".to_string()));
}

#[test]
fn test_less_than() {
    assert_eq!(eval_test("(< 3 5)"), Ok("#t".to_string()));
    assert_eq!(eval_test("(< 5 3)"), Ok("#f".to_string()));
    assert_eq!(eval_test("(< 5 5)"), Ok("#f".to_string()));
    assert_eq!(eval_test("(< -10 0)"), Ok("#t".to_string()));
    assert_eq!(eval_test("(< 0 1)"), Ok("#t".to_string()));
}

#[test]
fn test_greater_than() {
    assert_eq!(eval_test("(> 5 3)"), Ok("#t".to_string()));
    assert_eq!(eval_test("(> 3 5)"), Ok("#f".to_string()));
    assert_eq!(eval_test("(> 5 5)"), Ok("#f".to_string()));
    assert_eq!(eval_test("(> 0 -10)"), Ok("#t".to_string()));
    assert_eq!(eval_test("(> 1 0)"), Ok("#t".to_string()));
}

// ============================================================================
// List Operations Tests
// ============================================================================

#[test]
fn test_list() {
    assert_eq!(eval_test("(list 1 2 3)"), Ok("(1 2 3)".to_string()));
    assert_eq!(eval_test("(list)"), Ok("nil".to_string()));
    assert_eq!(eval_test("(list 1)"), Ok("(1)".to_string()));
    assert_eq!(eval_test("(list 1 2 3 4 5)"), Ok("(1 2 3 4 5)".to_string()));
}

#[test]
fn test_cons() {
    assert_eq!(eval_test("(cons 1 (list 2 3))"), Ok("(1 2 3)".to_string()));
    assert_eq!(eval_test("(cons 1 nil)"), Ok("(1)".to_string()));
    assert_eq!(eval_test("(cons 1 2)"), Ok("(1 . 2)".to_string()));
    assert_eq!(
        eval_test("(cons 0 (cons 1 (cons 2 nil)))"),
        Ok("(0 1 2)".to_string())
    );
}

#[test]
fn test_car() {
    assert_eq!(eval_test("(car (list 1 2 3))"), Ok("1".to_string()));
    assert_eq!(eval_test("(car (cons 5 10))"), Ok("5".to_string()));
    assert_eq!(eval_test("(car (list 42))"), Ok("42".to_string()));
    assert!(eval_test("(car nil)").is_err());
}

#[test]
fn test_cdr() {
    assert_eq!(eval_test("(cdr (list 1 2 3))"), Ok("(2 3)".to_string()));
    assert_eq!(eval_test("(cdr (list 1))"), Ok("nil".to_string()));
    assert_eq!(eval_test("(cdr (cons 5 10))"), Ok("10".to_string()));
    assert_eq!(eval_test("(cdr (list 1 2))"), Ok("(2)".to_string()));
    assert!(eval_test("(cdr nil)").is_err());
}

#[test]
fn test_null_predicate() {
    assert_eq!(eval_test("(null? nil)"), Ok("#t".to_string()));
    assert_eq!(eval_test("(null? (list))"), Ok("#t".to_string()));
    assert_eq!(eval_test("(null? (list 1))"), Ok("#f".to_string()));
    assert_eq!(eval_test("(null? 5)"), Ok("#f".to_string()));
    assert_eq!(eval_test("(null? #f)"), Ok("#f".to_string()));
}

#[test]
fn test_cons_predicate() {
    assert_eq!(eval_test("(cons? (list 1 2))"), Ok("#t".to_string()));
    assert_eq!(eval_test("(cons? (cons 1 2))"), Ok("#t".to_string()));
    assert_eq!(eval_test("(cons? nil)"), Ok("#f".to_string()));
    assert_eq!(eval_test("(cons? 5)"), Ok("#f".to_string()));
    assert_eq!(eval_test("(cons? #t)"), Ok("#f".to_string()));
}

#[test]
fn test_length() {
    assert_eq!(eval_test("(length (list 1 2 3))"), Ok("3".to_string()));
    assert_eq!(eval_test("(length nil)"), Ok("0".to_string()));
    assert_eq!(eval_test("(length (list 1))"), Ok("1".to_string()));
    assert_eq!(
        eval_test("(length (list 1 2 3 4 5 6 7 8 9 10))"),
        Ok("10".to_string())
    );
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
    assert_eq!(
        eval_test("(append (list 1) (list 2) (list 3))"),
        Ok("(1 2 3)".to_string())
    );
}

#[test]
fn test_reverse() {
    assert_eq!(
        eval_test("(reverse (list 1 2 3))"),
        Ok("(3 2 1)".to_string())
    );
    assert_eq!(eval_test("(reverse nil)"), Ok("nil".to_string()));
    assert_eq!(eval_test("(reverse (list 1))"), Ok("(1)".to_string()));
    assert_eq!(
        eval_test("(reverse (list 1 2 3 4 5))"),
        Ok("(5 4 3 2 1)".to_string())
    );
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
    assert_eq!(eval_test("'(+ 1 2)"), Ok("(+ 1 2)".to_string()));
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
    assert_eq!(
        eval_test("((lambda (a b c) (+ a (* b c))) 1 2 3)"),
        Ok("7".to_string())
    );
}

#[test]
fn test_lambda_no_args() {
    assert_eq!(eval_test("((lambda () 42))"), Ok("42".to_string()));
    assert_eq!(eval_test("((lambda () (+ 1 2)))"), Ok("3".to_string()));
}

#[test]
fn test_lambda_closure() {
    let program = r#"
        ((lambda (x)
           ((lambda (y) (+ x y)) 3))
         5)
    "#;
    assert_eq!(eval_test(program), Ok("8".to_string()));
}

#[test]
fn test_lambda_nested() {
    assert_eq!(
        eval_test("((lambda (x) ((lambda (y) (+ x y)) 3)) 5)"),
        Ok("8".to_string())
    );
    assert_eq!(
        eval_test("((lambda (x) ((lambda (x) (* x 2)) 10)) 5)"),
        Ok("20".to_string())
    );
}

#[test]
fn test_higher_order_function() {
    let program = r#"
        ((lambda (make-adder)
           ((make-adder 5) 10))
         (lambda (n) (lambda (x) (+ x n))))
    "#;
    assert_eq!(eval_multiple_test(program), Ok("15".to_string()));
}

// ============================================================================
// If Tests
// ============================================================================

#[test]
fn test_if_true() {
    assert_eq!(eval_test("(if #t 1 2)"), Ok("1".to_string()));
    assert_eq!(eval_test("(if (< 3 5) 10 20)"), Ok("10".to_string()));
    assert_eq!(eval_test("(if (= 5 5) 100 200)"), Ok("100".to_string()));
}

#[test]
fn test_if_false() {
    assert_eq!(eval_test("(if #f 1 2)"), Ok("2".to_string()));
    assert_eq!(eval_test("(if (> 3 5) 10 20)"), Ok("20".to_string()));
    assert_eq!(eval_test("(if (= 5 6) 100 200)"), Ok("200".to_string()));
}

#[test]
fn test_if_nil_is_false() {
    assert_eq!(eval_test("(if nil 1 2)"), Ok("2".to_string()));
}

#[test]
fn test_if_non_boolean_is_true() {
    assert_eq!(eval_test("(if 5 1 2)"), Ok("1".to_string()));
    assert_eq!(eval_test("(if (list 1) 1 2)"), Ok("1".to_string()));
    assert_eq!(eval_test("(if 0 1 2)"), Ok("1".to_string()));
}

#[test]
fn test_if_nested() {
    assert_eq!(eval_test("(if #t (if #f 1 2) 3)"), Ok("2".to_string()));
    assert_eq!(eval_test("(if #f 1 (if #t 2 3))"), Ok("2".to_string()));
}

// ============================================================================
// Recursive Function Tests
// ============================================================================

#[test]
fn test_factorial() {
    let program = r#"
        ((lambda (fact)
           (fact fact 5))
         (lambda (self n)
           (if (= n 0)
               1
               (* n (self self (- n 1))))))
    "#;
    assert_eq!(eval_test(program), Ok("120".to_string()));
}

#[test]
fn test_factorial_larger() {
    let program = r#"
        ((lambda (fact)
           (fact fact 10))
         (lambda (self n)
           (if (= n 0)
               1
               (* n (self self (- n 1))))))
    "#;
    assert_eq!(eval_test(program), Ok("3628800".to_string()));
}

#[test]
fn test_fibonacci() {
    let program = r#"
        ((lambda (fib)
           (fib fib 10))
         (lambda (self n)
           (if (< n 2)
               n
               (+ (self self (- n 1)) (self self (- n 2))))))
    "#;
    assert_eq!(eval_test(program), Ok("55".to_string()));
}

#[test]
fn test_fibonacci_small() {
    let program = r#"
        ((lambda (fib)
           (fib fib 6))
         (lambda (self n)
           (if (< n 2)
               n
               (+ (self self (- n 1)) (self self (- n 2))))))
    "#;
    assert_eq!(eval_test(program), Ok("8".to_string()));
}

#[test]
fn test_countdown() {
    let program = r#"
        ((lambda (countdown)
           (countdown countdown 1000))
         (lambda (self n)
           (if (= n 0)
               0
               (self self (- n 1)))))
    "#;
    assert_eq!(eval_test(program), Ok("0".to_string()));
}

#[test]
fn test_countdown_large() {
    let program = r#"
        ((lambda (countdown)
           (countdown countdown 5000))
         (lambda (self n)
           (if (= n 0)
               0
               (self self (- n 1)))))
    "#;
    assert_eq!(eval_test(program), Ok("0".to_string()));
}

#[test]
fn test_ackermann() {
    let program = r#"
        ((lambda (ack)
           (ack ack 3 3))
         (lambda (self m n)
           (if (= m 0)
               (+ n 1)
               (if (= n 0)
                   (self self (- m 1) 1)
                   (self self (- m 1) (self self m (- n 1)))))))
    "#;
    assert_eq!(eval_test(program), Ok("61".to_string()));
}

#[test]
fn test_ackermann_small() {
    let program = r#"
        ((lambda (ack)
           (ack ack 2 2))
         (lambda (self m n)
           (if (= m 0)
               (+ n 1)
               (if (= n 0)
                   (self self (- m 1) 1)
                   (self self (- m 1) (self self m (- n 1)))))))
    "#;
    assert_eq!(eval_test(program), Ok("7".to_string()));
}

#[test]
fn test_sum_list() {
    let program = r#"
        ((lambda (sum)
           (sum sum (list 1 2 3 4 5)))
         (lambda (self lst)
           (if (null? lst)
               0
               (+ (car lst) (self self (cdr lst))))))
    "#;
    assert_eq!(eval_test(program), Ok("15".to_string()));
}

#[test]
fn test_map_function() {
    let program = r#"
        ((lambda (map)
           (map map (lambda (x) (* x 2)) (list 1 2 3)))
         (lambda (self f lst)
           (if (null? lst)
               nil
               (cons (f (car lst)) (self self f (cdr lst))))))
    "#;
    assert_eq!(eval_test(program), Ok("(2 4 6)".to_string()));
}

#[test]
fn test_filter_function() {
    let program = r#"
        ((lambda (filter)
           (filter filter (lambda (x) (> x 0)) (list -1 2 -3 4 5)))
         (lambda (self pred lst)
           (if (null? lst)
               nil
               (if (pred (car lst))
                   (cons (car lst) (self self pred (cdr lst)))
                   (self self pred (cdr lst))))))
    "#;
    assert_eq!(eval_test(program), Ok("(2 4 5)".to_string()));
}

// ============================================================================
// Tail Call Optimization Tests
// ============================================================================

#[test]
fn test_tail_recursive_sum() {
    let program = r#"
        ((lambda (sum-tail)
           (sum-tail sum-tail 1000 0))
         (lambda (self n acc)
           (if (= n 0)
               acc
               (self self (- n 1) (+ acc n)))))
    "#;
    assert_eq!(eval_test(program), Ok("500500".to_string()));
}

#[test]
fn test_tail_recursive_factorial() {
    let program = r#"
        ((lambda (fact-tail)
           (fact-tail fact-tail 10 1))
         (lambda (self n acc)
           (if (= n 0)
               acc
               (self self (- n 1) (* acc n)))))
    "#;
    assert_eq!(eval_test(program), Ok("3628800".to_string()));
}

// ============================================================================
// Mutual Recursion Tests
// ============================================================================

#[test]
fn test_mutual_recursion() {
    let program = r#"
        ((lambda (make-even-odd)
           ((car (make-even-odd)) 10))
         (lambda ()
           (cons
             (lambda (n)
               (if (= n 0)
                   #t
                   ((cdr (make-even-odd)) (- n 1))))
             (lambda (n)
               (if (= n 0)
                   #f
                   ((car (make-even-odd)) (- n 1)))))))
    "#;
    // This won't work without define due to forward references
    // We'll test mutual recursion with a different pattern
    let simpler = r#"
        ((lambda (is-even-odd)
           ((car is-even-odd) 10))
         ((lambda (make-pair)
            (make-pair make-pair))
          (lambda (self)
            (cons
              (lambda (n)
                (if (= n 0) #t ((cdr (self self)) (- n 1))))
              (lambda (n)
                (if (= n 0) #f ((car (self self)) (- n 1))))))))
    "#;
    assert_eq!(eval_test(simpler), Ok("#t".to_string()));
}

// ============================================================================
// Memory and Reference Tests
// ============================================================================

#[test]
fn test_self_referential_list() {
    let program = r#"
        ((lambda (cyc)
           (car cyc))
         (cons 5 5))
    "#;
    assert_eq!(eval_test(program), Ok("5".to_string()));
}

#[test]
fn test_shared_structure() {
    let program = r#"
        ((lambda (shared)
           (+ (car shared) (cdr shared)))
         ((lambda (x) (cons x x)) 5))
    "#;
    assert_eq!(eval_test(program), Ok("10".to_string()));
}

// ============================================================================
// Error Handling Tests
// ============================================================================

#[test]
fn test_unbound_symbol() {
    assert!(eval_test("undefined-var").is_err());
    assert!(eval_test("xyz").is_err());
}

#[test]
fn test_invalid_function_call() {
    assert!(eval_test("(5 10)").is_err());
    assert!(eval_test("(#t)").is_err());
}

#[test]
fn test_arity_mismatch() {
    assert!(eval_test("((lambda (x) x) 1 2)").is_err());
    assert!(eval_test("((lambda (x y) x))").is_err());
    assert!(eval_test("((lambda () 5) 1)").is_err());
}

#[test]
fn test_type_error_arithmetic() {
    assert!(eval_test("(+ 1 #t)").is_err());
    assert!(eval_test("(* #f 5)").is_err());
    assert!(eval_test("(- (list 1 2))").is_err());
}

#[test]
fn test_division_by_zero() {
    assert!(eval_test("(/ 10 0)").is_err());
    assert!(eval_test("(/ 5 (- 2 2))").is_err());
}

#[test]
fn test_car_cdr_errors() {
    assert!(eval_test("(car nil)").is_err());
    assert!(eval_test("(cdr nil)").is_err());
    assert!(eval_test("(car 5)").is_err());
    assert!(eval_test("(cdr #t)").is_err());
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
    assert_eq!(
        eval_test("(car (reverse (list 1 2 3)))"),
        Ok("3".to_string())
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
fn test_nested_conditionals() {
    let program = r#"
        (if (> 5 3)
            (if (< 2 4)
                (+ 10 20)
                (- 10 20))
            (* 10 20))
    "#;
    assert_eq!(eval_test(program), Ok("30".to_string()));
}

// ============================================================================
// Edge Cases
// ============================================================================

#[test]
fn test_empty_list() {
    assert_eq!(eval_test("(list)"), Ok("nil".to_string()));
    assert_eq!(eval_test("(length (list))"), Ok("0".to_string()));
}

#[test]
fn test_single_element_list() {
    assert_eq!(eval_test("(list 42)"), Ok("(42)".to_string()));
    assert_eq!(eval_test("(car (list 42))"), Ok("42".to_string()));
    assert_eq!(eval_test("(cdr (list 42))"), Ok("nil".to_string()));
}

#[test]
fn test_nested_empty_lists() {
    assert_eq!(
        eval_test("(list (list) (list))"),
        Ok("(nil nil)".to_string())
    );
    assert_eq!(eval_test("(length (list (list)))"), Ok("1".to_string()));
}

#[test]
fn test_zero_arithmetic() {
    assert_eq!(eval_test("(+ 0)"), Ok("0".to_string()));
    assert_eq!(eval_test("(* 0)"), Ok("0".to_string()));
    assert_eq!(eval_test("(- 0)"), Ok("0".to_string()));
    assert_eq!(eval_test("(+ 0 0 0)"), Ok("0".to_string()));
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
    assert_eq!(eval_test("(\n+\n1\n2\n)"), Ok("3".to_string()));
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

// ============================================================================
// Integration Tests
// ============================================================================

#[test]
fn test_list_utilities() {
    let program = r#"
        ((lambda (lst)
           (+ (length lst)
              (car (reverse lst))))
         (append (list 1 2) (list 3 4)))
    "#;
    assert_eq!(eval_test(program), Ok("8".to_string()));
}
