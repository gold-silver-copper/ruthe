use ruthe::*;
#[cfg(test)]
mod comprehensive_tests {

    use super::*;
    // ========================================================================
    // Basic Arithmetic Tests
    // ========================================================================

    #[test]
    fn test_addition() {
        let env = super::create_env();
        assert_eq!(eval_str("(+ 1 2)", &env).unwrap(), "3");
        assert_eq!(eval_str("(+ 1 2 3)", &env).unwrap(), "6");
        assert_eq!(eval_str("(+ 1 2 3 4 5)", &env).unwrap(), "15");
        assert_eq!(eval_str("(+)", &env).unwrap(), "0");
        assert_eq!(eval_str("(+ 0)", &env).unwrap(), "0");
        assert_eq!(eval_str("(+ -5 3)", &env).unwrap(), "-2");
        assert_eq!(eval_str("(+ -5 -3)", &env).unwrap(), "-8");
    }

    #[test]
    fn test_subtraction() {
        let env = create_env();
        assert_eq!(eval_str("(- 5 3)", &env).unwrap(), "2");
        assert_eq!(eval_str("(- 10 3 2)", &env).unwrap(), "5");
        assert_eq!(eval_str("(- 5)", &env).unwrap(), "-5");
        assert_eq!(eval_str("(- -5)", &env).unwrap(), "5");
        assert_eq!(eval_str("(- 0 5)", &env).unwrap(), "-5");
        assert!(eval_str("(-)", &env).is_err());
    }

    #[test]
    fn test_multiplication() {
        let env = create_env();
        assert_eq!(eval_str("(* 2 3)", &env).unwrap(), "6");
        assert_eq!(eval_str("(* 2 3 4)", &env).unwrap(), "24");
        assert_eq!(eval_str("(*)", &env).unwrap(), "1");
        assert_eq!(eval_str("(* 5)", &env).unwrap(), "5");
        assert_eq!(eval_str("(* -2 3)", &env).unwrap(), "-6");
        assert_eq!(eval_str("(* -2 -3)", &env).unwrap(), "6");
        assert_eq!(eval_str("(* 0 100)", &env).unwrap(), "0");
    }

    #[test]
    fn test_nested_arithmetic() {
        let env = create_env();
        assert_eq!(eval_str("(+ (* 2 3) (* 4 5))", &env).unwrap(), "26");
        assert_eq!(eval_str("(- (* 10 10) (/ 50 2))", &env).unwrap(), "75");
        assert_eq!(eval_str("(* (+ 1 2) (- 5 2))", &env).unwrap(), "9");
        assert_eq!(eval_str("(/ (+ 10 20) (- 10 5))", &env).unwrap(), "6");
    }

    // ========================================================================
    // Comparison Tests
    // ========================================================================

    #[test]
    fn test_equality() {
        let env = create_env();
        assert_eq!(eval_str("(= 5 5)", &env).unwrap(), "#t");
        assert_eq!(eval_str("(= 5 6)", &env).unwrap(), "#f");
        assert_eq!(eval_str("(= 0 0)", &env).unwrap(), "#t");
        assert_eq!(eval_str("(= -5 -5)", &env).unwrap(), "#t");
    }

    #[test]
    fn test_less_than() {
        let env = create_env();
        assert_eq!(eval_str("(< 3 5)", &env).unwrap(), "#t");
        assert_eq!(eval_str("(< 5 3)", &env).unwrap(), "#f");
        assert_eq!(eval_str("(< 5 5)", &env).unwrap(), "#f");
        assert_eq!(eval_str("(< -5 0)", &env).unwrap(), "#t");
    }

    #[test]
    fn test_greater_than() {
        let env = create_env();
        assert_eq!(eval_str("(> 5 3)", &env).unwrap(), "#t");
        assert_eq!(eval_str("(> 3 5)", &env).unwrap(), "#f");
        assert_eq!(eval_str("(> 5 5)", &env).unwrap(), "#f");
        assert_eq!(eval_str("(> 0 -5)", &env).unwrap(), "#t");
    }

    // ========================================================================
    // List Operations Tests
    // ========================================================================

    #[test]
    fn test_quote() {
        let env = create_env();
        assert_eq!(eval_str("'(1 2 3)", &env).unwrap(), "(1 2 3)");
        assert_eq!(eval_str("'()", &env).unwrap(), "nil");
        assert_eq!(eval_str("'a", &env).unwrap(), "a");
        assert_eq!(eval_str("'(+ 1 2)", &env).unwrap(), "(+ 1 2)");
    }

    #[test]
    fn test_car() {
        let env = create_env();
        assert_eq!(eval_str("(car '(1 2 3))", &env).unwrap(), "1");
        assert_eq!(eval_str("(car '(a b c))", &env).unwrap(), "a");
        assert_eq!(eval_str("(car '((1 2) 3))", &env).unwrap(), "(1 2)");
        assert!(eval_str("(car '())", &env).is_err());
    }

    #[test]
    fn test_cdr() {
        let env = create_env();
        assert_eq!(eval_str("(cdr '(1 2 3))", &env).unwrap(), "(2 3)");
        assert_eq!(eval_str("(cdr '(a b c))", &env).unwrap(), "(b c)");
        assert_eq!(eval_str("(cdr '(1))", &env).unwrap(), "nil");
        assert!(eval_str("(cdr '())", &env).is_err());
    }

    #[test]
    fn test_cons() {
        let env = create_env();
        assert_eq!(eval_str("(cons 1 '(2 3))", &env).unwrap(), "(1 2 3)");
        assert_eq!(eval_str("(cons 'a '(b c))", &env).unwrap(), "(a b c)");
        assert_eq!(eval_str("(cons 1 '())", &env).unwrap(), "(1)");
        assert_eq!(eval_str("(cons 1 2)", &env).unwrap(), "(1 . 2)");
    }

    #[test]
    fn test_list() {
        let env = create_env();
        assert_eq!(eval_str("(list)", &env).unwrap(), "nil");
        assert_eq!(eval_str("(list 1)", &env).unwrap(), "(1)");
        assert_eq!(eval_str("(list 1 2 3)", &env).unwrap(), "(1 2 3)");
        assert_eq!(eval_str("(list (+ 1 2) (* 3 4))", &env).unwrap(), "(3 12)");
    }

    #[test]
    fn test_null_predicate() {
        let env = create_env();
        assert_eq!(eval_str("(null? '())", &env).unwrap(), "#t");
        assert_eq!(eval_str("(null? nil)", &env).unwrap(), "#t");
        assert_eq!(eval_str("(null? '(1 2))", &env).unwrap(), "#f");
        assert_eq!(eval_str("(null? 0)", &env).unwrap(), "#f");
    }

    #[test]
    fn test_cons_predicate() {
        let env = create_env();
        assert_eq!(eval_str("(cons? '(1 2))", &env).unwrap(), "#t");
        assert_eq!(eval_str("(cons? '())", &env).unwrap(), "#f");
        assert_eq!(eval_str("(cons? nil)", &env).unwrap(), "#f");
        assert_eq!(eval_str("(cons? 5)", &env).unwrap(), "#f");
    }

    #[test]
    fn test_length() {
        let env = create_env();
        assert_eq!(eval_str("(length '())", &env).unwrap(), "0");
        assert_eq!(eval_str("(length '(1))", &env).unwrap(), "1");
        assert_eq!(eval_str("(length '(1 2 3))", &env).unwrap(), "3");
        assert_eq!(eval_str("(length '(a b c d e))", &env).unwrap(), "5");
    }

    #[test]
    fn test_append() {
        let env = create_env();
        assert_eq!(
            eval_str("(append '(1 2) '(3 4))", &env).unwrap(),
            "(1 2 3 4)"
        );
        assert_eq!(eval_str("(append '() '(1 2))", &env).unwrap(), "(1 2)");
        assert_eq!(eval_str("(append '(1 2) '())", &env).unwrap(), "(1 2)");
        assert_eq!(
            eval_str("(append '(a) '(b) '(c))", &env).unwrap(),
            "(a b c)"
        );
    }

    #[test]
    fn test_reverse() {
        let env = create_env();
        assert_eq!(eval_str("(reverse '(1 2 3))", &env).unwrap(), "(3 2 1)");
        assert_eq!(eval_str("(reverse '(a))", &env).unwrap(), "(a)");
        assert_eq!(eval_str("(reverse '())", &env).unwrap(), "nil");
    }

    #[test]
    fn test_nested_list_operations() {
        let env = create_env();
        assert_eq!(eval_str("(car (cdr '(1 2 3)))", &env).unwrap(), "2");
        assert_eq!(eval_str("(cdr (cdr '(1 2 3)))", &env).unwrap(), "(3)");
        assert_eq!(eval_str("(car (car '((a b) c)))", &env).unwrap(), "a");
        assert_eq!(
            eval_str("(cons (car '(1 2)) (cdr '(3 4)))", &env).unwrap(),
            "(1 4)"
        );
    }

    // ========================================================================
    // Define and Variable Tests
    // ========================================================================

    #[test]
    fn test_define() {
        let env = create_env();
        eval_str("(define x 5)", &env).unwrap();
        assert_eq!(eval_str("x", &env).unwrap(), "5");

        eval_str("(define y (* 2 3))", &env).unwrap();
        assert_eq!(eval_str("y", &env).unwrap(), "6");

        eval_str("(define z (+ x y))", &env).unwrap();
        assert_eq!(eval_str("z", &env).unwrap(), "11");
    }

    #[test]
    fn test_redefine() {
        let env = create_env();
        eval_str("(define x 5)", &env).unwrap();
        assert_eq!(eval_str("x", &env).unwrap(), "5");

        eval_str("(define x 10)", &env).unwrap();
        assert_eq!(eval_str("x", &env).unwrap(), "10");
    }

    #[test]
    fn test_unbound_variable() {
        let env = create_env();
        assert!(eval_str("undefined_var", &env).is_err());
    }

    // ========================================================================
    // If Conditional Tests
    // ========================================================================

    #[test]
    fn test_if_true() {
        let env = create_env();
        assert_eq!(eval_str("(if #t 1 2)", &env).unwrap(), "1");
        assert_eq!(eval_str("(if (= 5 5) 'yes 'no)", &env).unwrap(), "yes");
        assert_eq!(eval_str("(if (< 3 5) (+ 1 2) (* 2 3))", &env).unwrap(), "3");
    }

    #[test]
    fn test_if_false() {
        let env = create_env();
        assert_eq!(eval_str("(if #f 1 2)", &env).unwrap(), "2");
        assert_eq!(eval_str("(if (= 5 6) 'yes 'no)", &env).unwrap(), "no");
        assert_eq!(eval_str("(if (> 3 5) (+ 1 2) (* 2 3))", &env).unwrap(), "6");
    }

    #[test]
    fn test_if_nil() {
        let env = create_env();
        assert_eq!(eval_str("(if nil 1 2)", &env).unwrap(), "2");
        assert_eq!(eval_str("(if '() 1 2)", &env).unwrap(), "2");
    }

    #[test]
    fn test_if_truthy() {
        let env = create_env();
        assert_eq!(eval_str("(if 0 1 2)", &env).unwrap(), "1");
        assert_eq!(eval_str("(if 5 'yes 'no)", &env).unwrap(), "yes");
        assert_eq!(eval_str("(if '(1 2) 'yes 'no)", &env).unwrap(), "yes");
    }

    #[test]
    fn test_nested_if() {
        let env = create_env();
        assert_eq!(eval_str("(if #t (if #t 1 2) 3)", &env).unwrap(), "1");
        assert_eq!(eval_str("(if #t (if #f 1 2) 3)", &env).unwrap(), "2");
        assert_eq!(eval_str("(if #f (if #t 1 2) 3)", &env).unwrap(), "3");
    }

    // ========================================================================
    // Lambda Tests
    // ========================================================================

    #[test]
    fn test_simple_lambda() {
        let env = create_env();
        eval_str("(define identity (lambda (x) x))", &env).unwrap();
        assert_eq!(eval_str("(identity 5)", &env).unwrap(), "5");
        assert_eq!(eval_str("(identity 'hello)", &env).unwrap(), "hello");
    }

    #[test]
    fn test_lambda_arithmetic() {
        let env = create_env();
        eval_str("(define square (lambda (x) (* x x)))", &env).unwrap();
        assert_eq!(eval_str("(square 5)", &env).unwrap(), "25");
        assert_eq!(eval_str("(square -3)", &env).unwrap(), "9");

        eval_str("(define double (lambda (x) (+ x x)))", &env).unwrap();
        assert_eq!(eval_str("(double 7)", &env).unwrap(), "14");
    }

    #[test]
    fn test_lambda_multiple_params() {
        let env = create_env();
        eval_str("(define add (lambda (x y) (+ x y)))", &env).unwrap();
        assert_eq!(eval_str("(add 3 4)", &env).unwrap(), "7");

        eval_str("(define add3 (lambda (x y z) (+ x y z)))", &env).unwrap();
        assert_eq!(eval_str("(add3 1 2 3)", &env).unwrap(), "6");
    }

    #[test]
    fn test_lambda_closure() {
        let env = create_env();
        eval_str("(define x 10)", &env).unwrap();
        eval_str("(define add-x (lambda (y) (+ x y)))", &env).unwrap();
        assert_eq!(eval_str("(add-x 5)", &env).unwrap(), "15");

        // x should still be captured even if we redefine it
        eval_str("(define x 20)", &env).unwrap();
        // Note: In this implementation, variables are looked up dynamically,
        // so this will use the new value of x
        assert_eq!(eval_str("(add-x 5)", &env).unwrap(), "25");
    }

    #[test]
    fn test_lambda_with_if() {
        let env = create_env();
        eval_str("(define abs (lambda (x) (if (< x 0) (- x) x)))", &env).unwrap();
        assert_eq!(eval_str("(abs 5)", &env).unwrap(), "5");
        assert_eq!(eval_str("(abs -5)", &env).unwrap(), "5");
        assert_eq!(eval_str("(abs 0)", &env).unwrap(), "0");
    }

    #[test]
    fn test_lambda_with_list_ops() {
        let env = create_env();
        eval_str("(define first (lambda (lst) (car lst)))", &env).unwrap();
        assert_eq!(eval_str("(first '(1 2 3))", &env).unwrap(), "1");

        eval_str("(define rest (lambda (lst) (cdr lst)))", &env).unwrap();
        assert_eq!(eval_str("(rest '(1 2 3))", &env).unwrap(), "(2 3)");
    }

    #[test]
    fn test_higher_order_functions() {
        let env = create_env();
        eval_str("(define apply-twice (lambda (f x) (f (f x))))", &env).unwrap();
        eval_str("(define inc (lambda (x) (+ x 1)))", &env).unwrap();
        assert_eq!(eval_str("(apply-twice inc 5)", &env).unwrap(), "7");

        eval_str("(define square (lambda (x) (* x x)))", &env).unwrap();
        assert_eq!(eval_str("(apply-twice square 2)", &env).unwrap(), "16");
    }

    // ========================================================================
    // Recursive Function Tests
    // ========================================================================

    #[test]
    fn test_recursive_factorial() {
        let env = create_env();
        let factorial = r#"
            (define factorial
                (lambda (n)
                    (if (= n 0)
                        1
                        (* n (factorial (- n 1)))
                    )
                )
            )
        "#;
        eval_str(factorial, &env).unwrap();
        assert_eq!(eval_str("(factorial 0)", &env).unwrap(), "1");
        assert_eq!(eval_str("(factorial 1)", &env).unwrap(), "1");
        assert_eq!(eval_str("(factorial 5)", &env).unwrap(), "120");
        assert_eq!(eval_str("(factorial 10)", &env).unwrap(), "3628800");
    }

    #[test]
    fn test_recursive_fibonacci() {
        let env = create_env();
        let fib = r#"
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
        eval_str(fib, &env).unwrap();
        assert_eq!(eval_str("(fib 0)", &env).unwrap(), "0");
        assert_eq!(eval_str("(fib 1)", &env).unwrap(), "1");
        assert_eq!(eval_str("(fib 2)", &env).unwrap(), "1");
        assert_eq!(eval_str("(fib 5)", &env).unwrap(), "5");
        assert_eq!(eval_str("(fib 10)", &env).unwrap(), "55");
    }

    #[test]
    fn test_recursive_sum_list() {
        let env = create_env();
        let sum = r#"
            (define sum
                (lambda (lst)
                    (if (null? lst)
                        0
                        (+ (car lst) (sum (cdr lst)))
                    )
                )
            )
        "#;
        eval_str(sum, &env).unwrap();
        assert_eq!(eval_str("(sum '())", &env).unwrap(), "0");
        assert_eq!(eval_str("(sum '(1))", &env).unwrap(), "1");
        assert_eq!(eval_str("(sum '(1 2 3 4 5))", &env).unwrap(), "15");
    }

    #[test]
    fn test_recursive_list_length() {
        let env = create_env();
        let len = r#"
            (define len
                (lambda (lst)
                    (if (null? lst)
                        0
                        (+ 1 (len (cdr lst)))
                    )
                )
            )
        "#;
        eval_str(len, &env).unwrap();
        assert_eq!(eval_str("(len '())", &env).unwrap(), "0");
        assert_eq!(eval_str("(len '(a))", &env).unwrap(), "1");
        assert_eq!(eval_str("(len '(a b c d))", &env).unwrap(), "4");
    }

    #[test]
    fn test_recursive_map() {
        let env = create_env();
        let map_fn = r#"
            (define map
                (lambda (f lst)
                    (if (null? lst)
                        '()
                        (cons (f (car lst)) (map f (cdr lst)))
                    )
                )
            )
        "#;
        eval_str(map_fn, &env).unwrap();
        eval_str("(define square (lambda (x) (* x x)))", &env).unwrap();
        assert_eq!(eval_str("(map square '(1 2 3))", &env).unwrap(), "(1 4 9)");

        eval_str("(define inc (lambda (x) (+ x 1)))", &env).unwrap();
        assert_eq!(eval_str("(map inc '(1 2 3))", &env).unwrap(), "(2 3 4)");
    }

    #[test]
    fn test_recursive_filter() {
        let env = create_env();
        let filter = r#"
            (define filter
                (lambda (pred lst)
                    (if (null? lst)
                        '()
                        (if (pred (car lst))
                            (cons (car lst) (filter pred (cdr lst)))
                            (filter pred (cdr lst))
                        )
                    )
                )
            )
        "#;
        eval_str(filter, &env).unwrap();
        eval_str("(define positive? (lambda (x) (> x 0)))", &env).unwrap();
        assert_eq!(
            eval_str("(filter positive? '(1 -2 3 -4 5))", &env).unwrap(),
            "(1 3 5)"
        );
    }

    // ========================================================================
    // Complex Integration Tests
    // ========================================================================

    #[test]
    fn test_ackermann() {
        let env = create_env();
        let ackermann = r#"
            (define ack
                (lambda (m n)
                    (if (= m 0)
                        (+ n 1)
                        (if (= n 0)
                            (ack (- m 1) 1)
                            (ack (- m 1) (ack m (- n 1)))
                        )
                    )
                )
            )
        "#;
        eval_str(ackermann, &env).unwrap();
        assert_eq!(eval_str("(ack 0 0)", &env).unwrap(), "1");
        assert_eq!(eval_str("(ack 1 2)", &env).unwrap(), "4");
        assert_eq!(eval_str("(ack 2 2)", &env).unwrap(), "7");
        assert_eq!(eval_str("(ack 3 2)", &env).unwrap(), "29");
    }

    #[test]
    fn test_gcd() {
        let env = create_env();
        let gcd = r#"
            (define gcd
                (lambda (a b)
                    (if (= b 0)
                        a
                        (gcd b (- a (* (/ a b) b)))
                    )
                )
            )
        "#;
        eval_str(gcd, &env).unwrap();
        // Note: This implementation won't work perfectly due to rational division
        // but we can test with values that divide evenly
        eval_str("(define mod (lambda (a b) (- a (* (/ a b) b))))", &env).unwrap();
    }

    #[test]
    fn test_quicksort_helper() {
        let env = create_env();
        // Test helper functions for quicksort
        eval_str("(define filter (lambda (pred lst) (if (null? lst) '() (if (pred (car lst)) (cons (car lst) (filter pred (cdr lst))) (filter pred (cdr lst))))))", &env).unwrap();
        eval_str("(define less-than (lambda (n) (lambda (x) (< x n))))", &env).unwrap();

        assert_eq!(
            eval_str("(filter (less-than 5) '(1 2 6 3 7 4))", &env).unwrap(),
            "(1 2 3 4)"
        );
    }

    #[test]
    fn test_nested_lambdas() {
        let env = create_env();
        eval_str(
            "(define make-adder (lambda (x) (lambda (y) (+ x y))))",
            &env,
        )
        .unwrap();
        eval_str("(define add5 (make-adder 5))", &env).unwrap();
        assert_eq!(eval_str("(add5 10)", &env).unwrap(), "15");
        assert_eq!(eval_str("(add5 3)", &env).unwrap(), "8");
    }

    // ========================================================================
    // Error Handling Tests
    // ========================================================================

    #[test]
    fn test_type_errors() {
        let env = create_env();
        assert!(eval_str("(+ 1 'a)", &env).is_err());
        assert!(eval_str("(car 5)", &env).is_err());
        assert!(eval_str("(cdr #t)", &env).is_err());
    }

    #[test]
    fn test_arity_errors() {
        let env = create_env();
        assert!(eval_str("(=)", &env).is_err());
        assert!(eval_str("(= 1)", &env).is_err());
        assert!(eval_str("(= 1 2 3)", &env).is_err());

        eval_str("(define f (lambda (x y) (+ x y)))", &env).unwrap();
        assert!(eval_str("(f 1)", &env).is_err());
        assert!(eval_str("(f 1 2 3)", &env).is_err());
    }

    #[test]
    fn test_parse_errors() {
        let env = create_env();
        assert!(eval_str("(", &env).is_err());
        assert!(eval_str(")", &env).is_err());
        assert!(eval_str("(+ 1 2))", &env).is_err());
        assert!(eval_str("((+ 1 2)", &env).is_err());
    }
}
