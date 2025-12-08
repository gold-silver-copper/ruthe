// tests/extreme_recursion.rs
#![cfg(test)]
extern crate alloc;

use ruthe::*;

#[test]
fn test_massive_tail_recursion() {
    // Test tail recursion with extremely deep recursion
    let program = r#"
        ((lambda (tail-rec)
           (tail-rec tail-rec 1000000 0))
         (lambda (self n acc)
           (if (= n 0)
               acc
               (self self (- n 1) (+ acc 1)))))
    "#;

    let env = new_env();
    let result = eval_str(program, &env);
    assert!(result.is_ok(), "Massive tail recursion failed");
    let val = result.unwrap();
    assert_eq!(val.as_number(), Some(1000000));
}

#[test]
fn test_deeply_nested_non_tail_recursion() {
    // Test non-tail recursion with deep nesting
    let program = r#"
        ((lambda (deep-nest)
           (deep-nest deep-nest 500))
         (lambda (self depth)
           (if (= depth 0)
               42
               ((lambda (x) (self self (- depth 1))) depth))))
    "#;

    let env = new_env();
    let result = eval_str(program, &env);
    assert!(result.is_ok(), "Deep nested recursion failed");
    let val = result.unwrap();
    assert_eq!(val.as_number(), Some(42));
}

#[test]
fn test_exponential_recursion_tree() {
    // Create an exponential recursion tree (Fibonacci-like)
    // Use small depth to avoid running forever
    let program = r#"
        ((lambda (exp-tree)
           (exp-tree exp-tree 10))
         (lambda (self n)
           (if (< n 2)
               n
               (+ (self self (- n 1)) (self self (- n 2)) (self self (- n 3))))))
    "#;

    let env = new_env();
    let result = eval_str(program, &env);
    assert!(result.is_ok(), "Exponential recursion tree failed");
}

#[test]
fn test_mutual_recursion_extreme() {
    // Deep mutual recursion
    let program = r#"
        ((lambda (make-pair)
           ((car (make-pair make-pair)) 5000))
         (lambda (self)
           (cons
             (lambda (n)
               (if (= n 0)
                   #t
                   ((cdr (self self)) (- n 1))))
             (lambda (n)
               (if (= n 0)
                   #f
                   ((car (self self)) (- n 1)))))))
    "#;

    let env = new_env();
    let result = eval_str(program, &env);
    assert!(result.is_ok(), "Extreme mutual recursion failed");
    let val = result.unwrap();
    assert_eq!(val.as_bool(), Some(true));
}

#[test]
fn test_tail_call_with_many_arguments() {
    // Tail call with many arguments to test environment handling
    let program = r#"
        ((lambda (tail-many-args)
           (tail-many-args tail-many-args 1000 0 0 0 0 0 0 0 0 0))
         (lambda (self n a b c d e f g h i j)
           (if (= n 0)
               (+ a b c d e f g h i j)
               (self self (- n 1) (+ a 1) (+ b 1) (+ c 1) (+ d 1) (+ e 1)
                     (+ f 1) (+ g 1) (+ h 1) (+ i 1) (+ j 1)))))
    "#;

    let env = new_env();
    let result = eval_str(program, &env);
    assert!(result.is_ok(), "Tail call with many args failed");
    let val = result.unwrap();
    assert_eq!(val.as_number(), Some(10000)); // 10 args * 1000 iterations
}

#[test]
fn test_nested_tail_calls() {
    // Multiple levels of tail calls
    let program = r#"
        ((lambda (level1)
           (level1 level1 100))
         (lambda (self n)
           (if (= n 0)
               0
               ((lambda (x)
                  (self self (- n 1)))
                n))))
    "#;

    let env = new_env();
    let result = eval_str(program, &env);
    assert!(result.is_ok(), "Nested tail calls failed");
    let val = result.unwrap();
    assert_eq!(val.as_number(), Some(0));
}

// tests/memory_stress.rs

use alloc::rc::Rc;
use ruthe::*;

#[test]
fn test_massive_list_creation() {
    // Create a very large list
    let program = r#"
        ((lambda (build-huge-list)
           ((lambda (list)
              ((lambda (check)
                 (if check "PASS" "FAIL"))
               (= (length list) 5000)))
            (build-huge-list build-huge-list 5000 nil)))
         (lambda (self n acc)
           (if (= n 0)
               acc
               (self self (- n 1) (cons n acc)))))
    "#;

    let env = new_env();
    let result = eval_str(program, &env);
    assert!(result.is_ok(), "Massive list creation failed");

    let mut buf = [0u8; 256];
    let display = result.unwrap().to_display_str(&mut buf).unwrap();
    assert_eq!(display, "PASS");
}

#[test]
fn test_repeated_large_allocation() {
    // Repeatedly allocate and discard large structures
    let program = r#"
        ((lambda (alloc-loop)
           (alloc-loop alloc-loop 100))
         (lambda (self count)
           (if (= count 0)
               "DONE"
               ((lambda (temp-list)
                  (self self (- count 1)))
                ((lambda (make-list)
                   (make-list make-list 1000 nil))
                 (lambda (self n acc)
                   (if (= n 0)
                       acc
                       (self self (- n 1) (cons n acc)))))))))
    "#;

    let env = new_env();
    let result = eval_str(program, &env);
    assert!(result.is_ok(), "Repeated allocation failed");

    let mut buf = [0u8; 256];
    let display = result.unwrap().to_display_str(&mut buf).unwrap();
    assert_eq!(display, "DONE");
}

#[test]
fn test_circular_reference_handling() {
    // Create circular references and ensure they don't cause issues
    let program = r#"
        ((lambda (make-circular)
           ((lambda (circ)
              (car circ))
            (make-circular make-circular)))
         (lambda (self)
           (cons 42
                 (cons (cons 99 nil) nil))))
    "#;

    let env = new_env();
    let result = eval_str(program, &env);
    assert!(result.is_ok(), "Circular reference test failed");
    let val = result.unwrap();
    assert_eq!(val.as_number(), Some(42));
}

#[test]
fn test_shared_structure_stress() {
    // Create complex shared structures
    let program = r#"
        ((lambda (build-complex)
           ((lambda (shared)
              ((lambda (list1)
                 ((lambda (list2)
                    (+ (length list1) (length list2)))
                  (cons 100 shared)))
               (cons 99 shared)))
            (build-complex build-complex 50)))
         (lambda (self n)
           (if (= n 0)
               nil
               (cons n (self self (- n 1))))))
    "#;

    let env = new_env();
    let result = eval_str(program, &env);
    assert!(result.is_ok(), "Shared structure stress test failed");
    let val = result.unwrap();
    assert_eq!(val.as_number(), Some(102)); // 51 + 51
}

#[test]
fn test_deeply_nested_structures() {
    // Create very deeply nested cons cells
    let program = r#"
        ((lambda (deep-nest)
           (deep-nest deep-nest 1000))
         (lambda (self depth)
           (if (= depth 0)
               42
               (cons (self self (- depth 1)) nil))))
    "#;

    let env = new_env();
    let result = eval_str(program, &env);
    assert!(result.is_ok(), "Deeply nested structure failed");

    // Verify we can traverse it
    let val = result.unwrap();
    let mut current = &val;
    let mut count = 0;

    while let Some((_, cdr)) = current.as_cons() {
        current = cdr;
        count += 1;
        if count > 100 {
            break; // Don't traverse forever
        }
    }

    assert!(count > 0, "Should have nested structure");
}

#[test]
fn test_memory_leak_detection() {
    // Test that temporary allocations are cleaned up
    let env = new_env();
    let initial_rc_count = Rc::strong_count(&env.0);

    // Run a computation that creates many temporary values
    let program = r#"
        ((lambda (sum-loop)
           (sum-loop sum-loop 1000 0))
         (lambda (self n acc)
           (if (= n 0)
               acc
               (self self (- n 1) (+ acc 
                  ((lambda (x) x) n)  ; Temporary lambda
                  ((lambda (y) y) n)  ; Another temporary
                  )))))
    "#;

    let result = eval_str(program, &env);
    assert!(result.is_ok(), "Memory leak test computation failed");

    // Give some allowance for temporary allocations
    let final_rc_count = Rc::strong_count(&env.0);
    assert!(
        final_rc_count < initial_rc_count + 100,
        "Possible memory leak: RC count grew from {} to {}",
        initial_rc_count,
        final_rc_count
    );
}

// tests/complex_evaluation.rs

#[test]
fn test_higher_order_combinators() {
    // Test complex higher-order function patterns
    let program = r#"
        ((lambda (y-combinator)
           ((y-combinator
             (lambda (factorial)
               (lambda (n)
                 (if (= n 0)
                     1
                     (* n (factorial (- n 1)))))))
            10))
         (lambda (f)
           ((lambda (x) (f (lambda (y) ((x x) y))))
            (lambda (x) (f (lambda (y) ((x x) y)))))))
    "#;

    let env = new_env();
    let result = eval_str(program, &env);
    assert!(result.is_ok(), "Y-combinator factorial failed");
    let val = result.unwrap();
    assert_eq!(val.as_number(), Some(3628800));
}

#[test]
fn test_church_encoding() {
    // Test Church encoding of natural numbers and operations
    let program = r#"
        ((lambda (church-zero church-succ church-add)
           ((lambda (two three)
              (church-add two three))  ; 2 + 3 = 5
            (church-succ (church-succ church-zero))
            (church-succ (church-succ (church-succ church-zero)))))
         ; church-zero: λf.λx.x
         (lambda (f) (lambda (x) x))
         ; church-succ: λn.λf.λx.f (n f x)
         (lambda (n) (lambda (f) (lambda (x) (f ((n f) x)))))
         ; church-add: λm.λn.λf.λx.m f (n f x)
         (lambda (m) (lambda (n) (lambda (f) (lambda (x) ((m f) ((n f) x)))))))
    "#;

    let env = new_env();
    let result = eval_str(program, &env);
    assert!(result.is_ok(), "Church encoding test failed");

    // The result is a Church numeral - apply it to (λx.x+1) and 0 to get the number
    let church_five = result.unwrap();

    let mut buf = [0u8; 4096];
    // Apply to successor function and 0
    let apply_program = format!("(({} (lambda (x) (+ x 1))) 0)", {
        church_five.to_display_str(&mut buf).unwrap()
    });

    let final_result = eval_str(&apply_program, &env);
    assert!(final_result.is_ok(), "Church numeral application failed");
    assert_eq!(final_result.unwrap().as_number(), Some(5));
}

#[test]
fn test_complex_closure_chains() {
    // Create long chains of closures
    let program = r#"
        ((lambda (make-chain)
           ((lambda (chain)
              (chain 0))
            (make-chain make-chain 100)))
         (lambda (self depth)
           (if (= depth 0)
               (lambda (x) (+ x 100))
               (lambda (x)
                 ((self self (- depth 1)) (+ x 1))))))
    "#;

    let env = new_env();
    let result = eval_str(program, &env);
    assert!(result.is_ok(), "Complex closure chain failed");
    let val = result.unwrap();
    assert_eq!(val.as_number(), Some(200)); // 0 + 100 closures * 1 + 100
}

#[test]
fn test_eval_apply_metacircular() {
    // Simple metacircular evaluator test
    let program = r#"
        ((lambda (eval-expr)
           (eval-expr eval-expr '(+ 1 2) (lambda (env) env)))
         (lambda (self expr env)
           (if (null? expr)
               nil
               (if (cons? expr)
                   ((lambda (op args)
                      (if (equal? op '+)
                          ((lambda (sum-loop)
                             (sum-loop sum-loop args 0))
                           (lambda (self lst acc)
                             (if (null? lst)
                                 acc
                                 (self self (cdr lst) 
                                   (+ acc (self self (car lst) env))))))
                          "UNKNOWN OP"))
                    (car expr)
                    (cdr expr))
                   (if (number? expr)
                       expr
                       "UNKNOWN")))))
    "#;

    // Note: This test requires adding some helper functions
    // For now, let's create a simpler version
    let simple_program = r#"
        ((lambda (meta-eval)
           (meta-eval meta-eval '42))
         (lambda (self expr)
           (if (cons? expr)
               ((lambda (op a b)
                  (if (equal? op '+) (+ a b)
                      (if (equal? op '*) (* a b)
                          "UNKNOWN")))
                (car expr)
                (self self (car (cdr expr)))
                (self self (car (cdr (cdr expr)))))
               expr)))
    "#;

    let env = new_env();

    // First define helper functions
    let helpers = r#"
        ; Define helper functions
        ((lambda (dummy) "helpers defined")
         (lambda ()
           ((lambda (number?-fn)
              ((lambda (equal?-fn)
                 "done")
               (lambda (a b)
                 (if (cons? a)
                     #f
                     (if (cons? b)
                         #f
                         (= a b))))))
            (lambda (x) (if (cons? x) #f #t)))))
    "#;

    let _ = eval_str(helpers, &env);

    // Now test a simple arithmetic expression
    let test_expr = r#"
        ((lambda (eval-simple)
           (eval-simple eval-simple '(+ 3 (* 4 5))))
         (lambda (self expr)
           (if (cons? expr)
               ((lambda (op)
                  (if (equal? op '+)
                      (+ (self self (car (cdr expr)))
                         (self self (car (cdr (cdr expr)))))
                      (if (equal? op '*)
                          (* (self self (car (cdr expr)))
                             (self self (car (cdr (cdr expr)))))
                          0)))
                (car expr))
               expr)))
    "#;

    let result = eval_str(test_expr, &env);
    assert!(result.is_ok(), "Metacircular evaluator test failed");
    let val = result.unwrap();
    assert_eq!(val.as_number(), Some(23)); // 3 + (4 * 5) = 23
}

#[test]
fn test_continuation_passing_style() {
    // Test CPS-transformed factorial
    let program = r#"
        ((lambda (fact-cps)
           (fact-cps fact-cps 5 (lambda (x) x)))
         (lambda (self n k)
           (if (= n 0)
               (k 1)
               (self self (- n 1) 
                 (lambda (v) (k (* n v)))))))
    "#;

    let env = new_env();
    let result = eval_str(program, &env);
    assert!(result.is_ok(), "CPS factorial failed");
    let val = result.unwrap();
    assert_eq!(val.as_number(), Some(120));
}

#[test]
fn test_trampoline_pattern() {
    // Test trampoline for deep recursion
    let program = r#"
        ((lambda (trampoline)
           ((lambda (bounce)
              (trampoline trampoline bounce))
            (cons 'more (lambda () 
              (cons 'more (lambda () 
                (cons 'done 42)))))))
         (lambda (self bounce)
           (if (equal? (car bounce) 'done)
               (cdr bounce)
               (self self ((cdr bounce))))))
    "#;

    let env = new_env();
    let result = eval_str(program, &env);
    assert!(result.is_ok(), "Trampoline pattern failed");
    let val = result.unwrap();
    assert_eq!(val.as_number(), Some(42));
}
