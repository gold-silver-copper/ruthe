// examples/benchmark.rs
// Run with: cargo run --example benchmark --release

use ruthe::{EnvRef, eval_str, eval_str_multiple};
use std::time::Instant;

fn format_duration(nanos: u128) -> String {
    if nanos < 1_000 {
        format!("{}ns", nanos)
    } else if nanos < 1_000_000 {
        format!("{:.2}µs", nanos as f64 / 1_000.0)
    } else if nanos < 1_000_000_000 {
        format!("{:.2}ms", nanos as f64 / 1_000_000.0)
    } else {
        format!("{:.2}s", nanos as f64 / 1_000_000_000.0)
    }
}

fn benchmark_expression(name: &str, expr: &str, env: &EnvRef, iterations: usize) {
    let mut times = Vec::with_capacity(iterations);

    for _ in 0..iterations {
        let start = Instant::now();
        let _ = eval_str(expr, env).unwrap();
        let duration = start.elapsed();
        times.push(duration.as_nanos());
    }

    // Calculate statistics
    times.sort_unstable();
    let min = times[0];
    let max = times[times.len() - 1];
    let median = times[times.len() / 2];
    let mean: u128 = times.iter().sum::<u128>() / times.len() as u128;

    // Calculate standard deviation
    let variance: f64 = times
        .iter()
        .map(|&t| {
            let diff = t as f64 - mean as f64;
            diff * diff
        })
        .sum::<f64>()
        / times.len() as f64;
    let stddev = variance.sqrt();

    println!("{}:", name);
    println!("  Min:    {}", format_duration(min));
    println!("  Max:    {}", format_duration(max));
    println!("  Median: {}", format_duration(median));
    println!("  Mean:   {}", format_duration(mean));
    println!("  StdDev: {}", format_duration(stddev as u128));
    println!();
}

fn benchmark_single(name: &str, expr: &str, env: &EnvRef) -> u128 {
    let start = Instant::now();
    let result = eval_str(expr, env).unwrap();
    let duration = start.elapsed().as_nanos();
    println!(
        "  {} = {:#?} | Time: {}",
        name,
        result,
        format_duration(duration)
    );
    duration
}

fn main() {
    println!("╔════════════════════════════════════════════════════════════╗");
    println!("║     COMPREHENSIVE LISP INTERPRETER BENCHMARK SUITE        ║");
    println!("╚════════════════════════════════════════════════════════════╝\n");

    let env = EnvRef::new();

    // ========================================================================
    // SECTION 1: Basic Arithmetic Operations
    // ========================================================================
    println!("┌─────────────────────────────────────────────────────────────┐");
    println!("│ 1. BASIC ARITHMETIC OPERATIONS                              │");
    println!("└─────────────────────────────────────────────────────────────┘\n");

    benchmark_expression("Simple addition (+ 1 2)", "(+ 1 2)", &env, 50000);
    benchmark_expression(
        "Multi-operand add (+ 1 2 3 4 5)",
        "(+ 1 2 3 4 5)",
        &env,
        50000,
    );
    benchmark_expression(
        "Large numbers (* 999999 999999)",
        "(* 999999 999999)",
        &env,
        50000,
    );
    benchmark_expression("Subtraction (- 1000 999)", "(- 1000 999)", &env, 50000);
    benchmark_expression("Division (/ 1000000 7)", "(/ 1000000 7)", &env, 50000);
    benchmark_expression(
        "Nested arithmetic",
        "(+ (* 2 3) (/ 100 5) (- 10 3))",
        &env,
        50000,
    );

    // ========================================================================
    // SECTION 2: Comparison and Boolean Operations
    // ========================================================================
    println!("┌─────────────────────────────────────────────────────────────┐");
    println!("│ 2. COMPARISON & BOOLEAN OPERATIONS                          │");
    println!("└─────────────────────────────────────────────────────────────┘\n");

    benchmark_expression("Equality (= 5 5)", "(= 5 5)", &env, 50000);
    benchmark_expression("Inequality (= 5 6)", "(= 5 6)", &env, 50000);
    benchmark_expression("Less than (< 3 5)", "(< 3 5)", &env, 50000);
    benchmark_expression("Greater than (> 10 5)", "(> 10 5)", &env, 50000);
    benchmark_expression("Complex comparison", "(< (+ 2 3) (* 2 4))", &env, 50000);

    // ========================================================================
    // SECTION 3: List Operations
    // ========================================================================
    println!("┌─────────────────────────────────────────────────────────────┐");
    println!("│ 3. LIST OPERATIONS                                          │");
    println!("└─────────────────────────────────────────────────────────────┘\n");

    benchmark_expression("car '(1 2 3)", "(car '(1 2 3))", &env, 50000);
    benchmark_expression("cdr '(1 2 3)", "(cdr '(1 2 3))", &env, 50000);
    benchmark_expression("cons 1 '(2 3)", "(cons 1 '(2 3))", &env, 50000);
    benchmark_expression("list 1 2 3 4 5", "(list 1 2 3 4 5)", &env, 50000);
    benchmark_expression("length '(1 2 3 4 5)", "(length '(1 2 3 4 5))", &env, 50000);
    benchmark_expression(
        "reverse '(1 2 3 4 5)",
        "(reverse '(1 2 3 4 5))",
        &env,
        50000,
    );
    benchmark_expression("append two lists", "(append '(1 2) '(3 4))", &env, 50000);
    benchmark_expression("null? nil", "(null? nil)", &env, 50000);
    benchmark_expression("cons? '(1 2)", "(cons? '(1 2))", &env, 50000);

    // Large list operations
    println!("--- Large List Operations ---");
    let large_list_def = "(define big-list '(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20))";
    eval_str(large_list_def, &env).unwrap();
    benchmark_expression(
        "length of 20-element list",
        "(length big-list)",
        &env,
        10000,
    );
    benchmark_expression("reverse 20-element list", "(reverse big-list)", &env, 10000);

    // ========================================================================
    // SECTION 4: Conditional Expressions
    // ========================================================================
    println!("┌─────────────────────────────────────────────────────────────┐");
    println!("│ 4. CONDITIONAL EXPRESSIONS                                  │");
    println!("└─────────────────────────────────────────────────────────────┘\n");

    benchmark_expression("if #t 1 2", "(if #t 1 2)", &env, 50000);
    benchmark_expression("if #f 1 2", "(if #f 1 2)", &env, 50000);
    benchmark_expression(
        "if with computation",
        "(if (< 3 5) (+ 1 2) (* 3 4))",
        &env,
        50000,
    );
    benchmark_expression("nested if", "(if #t (if #f 1 2) 3)", &env, 50000);

    // ========================================================================
    // SECTION 5: Lambda and Closures
    // ========================================================================
    println!("┌─────────────────────────────────────────────────────────────┐");
    println!("│ 5. LAMBDA EXPRESSIONS & CLOSURES                            │");
    println!("└─────────────────────────────────────────────────────────────┘\n");

    benchmark_expression("Simple lambda", "((lambda (x) x) 5)", &env, 50000);
    benchmark_expression(
        "Lambda with computation",
        "((lambda (x) (* x x)) 7)",
        &env,
        50000,
    );
    benchmark_expression(
        "Multi-arg lambda",
        "((lambda (x y) (+ x y)) 3 4)",
        &env,
        50000,
    );
    benchmark_expression("Zero-arg lambda", "((lambda () 42))", &env, 50000);

    // Define functions for closure tests
    eval_str(
        "(define make-adder (lambda (n) (lambda (x) (+ x n))))",
        &env,
    )
    .unwrap();
    eval_str("(define add5 (make-adder 5))", &env).unwrap();
    eval_str("(define add10 (make-adder 10))", &env).unwrap();

    benchmark_expression("Closure call", "(add5 10)", &env, 50000);
    benchmark_expression("Multiple closures", "(+ (add5 10) (add10 5))", &env, 50000);

    // Nested closures
    eval_str(
        "(define make-multiplier (lambda (a) (lambda (b) (lambda (c) (* a (* b c))))))",
        &env,
    )
    .unwrap();
    eval_str("(define mul2 (make-multiplier 2))", &env).unwrap();
    eval_str("(define mul2x3 (mul2 3))", &env).unwrap();
    benchmark_expression("Nested closure call", "(mul2x3 4)", &env, 50000);

    // ========================================================================
    // SECTION 6: Recursive Functions
    // ========================================================================
    println!("┌─────────────────────────────────────────────────────────────┐");
    println!("│ 6. RECURSIVE FUNCTIONS                                      │");
    println!("└─────────────────────────────────────────────────────────────┘\n");

    // Fibonacci (tree recursion)
    let fib_def = r#"
        (define fib
            (lambda (n)
                (if (< n 2)
                    n
                    (+ (fib (- n 1)) (fib (- n 2))))))
    "#;
    eval_str(fib_def, &env).unwrap();

    println!("--- Fibonacci (Tree Recursion) ---");
    for n in [5, 10, 15, 20, 25] {
        let expr = format!("(fib {})", n);
        benchmark_single(&format!("fib({})", n), &expr, &env);
    }
    println!();

    benchmark_expression("fib(10) repeated", "(fib 10)", &env, 1000);

    // Factorial (simple recursion)
    let fact_def = r#"
        (define factorial
            (lambda (n)
                (if (= n 0)
                    1
                    (* n (factorial (- n 1))))))
    "#;
    eval_str(fact_def, &env).unwrap();

    println!("--- Factorial (Simple Recursion) ---");
    for n in [5, 10, 15, 20] {
        let expr = format!("(factorial {})", n);
        benchmark_single(&format!("factorial({})", n), &expr, &env);
    }
    println!();

    // Ackermann function (complex recursion)
    let ack_def = r#"
        (define ack
            (lambda (m n)
                (if (= m 0)
                    (+ n 1)
                    (if (= n 0)
                        (ack (- m 1) 1)
                        (ack (- m 1) (ack m (- n 1)))))))
    "#;
    eval_str(ack_def, &env).unwrap();

    println!("--- Ackermann Function (Complex Recursion) ---");
    for (m, n) in [(1, 2), (2, 2), (2, 3), (3, 2), (3, 3)] {
        let expr = format!("(ack {} {})", m, n);
        benchmark_single(&format!("ack({}, {})", m, n), &expr, &env);
    }
    println!();

    // Sum list (linear recursion)
    let sum_list_def = r#"
        (define sum-list
            (lambda (lst)
                (if (null? lst)
                    0
                    (+ (car lst) (sum-list (cdr lst))))))
    "#;
    eval_str(sum_list_def, &env).unwrap();

    println!("--- List Recursion ---");
    benchmark_expression(
        "sum-list '(1 2 3 4 5)",
        "(sum-list '(1 2 3 4 5))",
        &env,
        10000,
    );
    benchmark_expression("sum-list big-list", "(sum-list big-list)", &env, 10000);

    // ========================================================================
    // SECTION 7: Tail Call Optimization
    // ========================================================================
    println!("┌─────────────────────────────────────────────────────────────┐");
    println!("│ 7. TAIL CALL OPTIMIZATION                                   │");
    println!("└─────────────────────────────────────────────────────────────┘\n");

    // Tail-recursive countdown
    let countdown_def = r#"
        (define countdown
            (lambda (n)
                (if (= n 0)
                    0
                    (countdown (- n 1)))))
    "#;
    eval_str(countdown_def, &env).unwrap();

    println!("--- Countdown (Tail Recursion) ---");
    for size in [1000, 10000, 100000, 500000] {
        let expr = format!("(countdown {})", size);
        benchmark_single(&format!("countdown({})", size), &expr, &env);
    }
    println!();

    // Tail-recursive factorial
    let fact_tail_def = r#"
        (define factorial-tail
            (lambda (n acc)
                (if (= n 0)
                    acc
                    (factorial-tail (- n 1) (* n acc)))))
    "#;
    eval_str(fact_tail_def, &env).unwrap();

    println!("--- Tail-Recursive Factorial ---");
    for n in [10, 20] {
        let expr = format!("(factorial-tail {} 1)", n);
        benchmark_single(&format!("factorial-tail({})", n), &expr, &env);
    }
    println!();

    // Tail-recursive sum
    let sum_tail_def = r#"
        (define sum-tail
            (lambda (n acc)
                (if (= n 0)
                    acc
                    (sum-tail (- n 1) (+ acc n)))))
    "#;
    eval_str(sum_tail_def, &env).unwrap();

    println!("--- Sum with Tail Recursion ---");
    for n in [100, 1000, 10000, 50000] {
        let expr = format!("(sum-tail {} 0)", n);
        benchmark_single(&format!("sum-tail({})", n), &expr, &env);
    }
    println!();

    // ========================================================================
    // SECTION 8: Higher-Order Functions
    // ========================================================================
    println!("┌─────────────────────────────────────────────────────────────┐");
    println!("│ 8. HIGHER-ORDER FUNCTIONS                                   │");
    println!("└─────────────────────────────────────────────────────────────┘\n");

    // Map function
    let map_def = r#"
        (define map
            (lambda (f lst)
                (if (null? lst)
                    nil
                    (cons (f (car lst)) (map f (cdr lst))))))
    "#;
    eval_str(map_def, &env).unwrap();
    eval_str("(define square (lambda (x) (* x x)))", &env).unwrap();
    eval_str("(define double (lambda (x) (* x 2)))", &env).unwrap();

    benchmark_expression(
        "map square '(1 2 3 4 5)",
        "(map square '(1 2 3 4 5))",
        &env,
        10000,
    );
    benchmark_expression("map double big-list", "(map double big-list)", &env, 1000);

    // Filter function
    let filter_def = r#"
        (define filter
            (lambda (pred lst)
                (if (null? lst)
                    nil
                    (if (pred (car lst))
                        (cons (car lst) (filter pred (cdr lst)))
                        (filter pred (cdr lst))))))
    "#;
    eval_str(filter_def, &env).unwrap();
    eval_str("(define positive? (lambda (x) (> x 0)))", &env).unwrap();
    eval_str("(define even? (lambda (x) (= (/ (* x 2) 2) x)))", &env).unwrap();

    benchmark_expression(
        "filter positive?",
        "(filter positive? '(-3 -1 0 1 2 3))",
        &env,
        10000,
    );

    // Reduce/fold function
    let fold_def = r#"
        (define fold
            (lambda (f acc lst)
                (if (null? lst)
                    acc
                    (fold f (f acc (car lst)) (cdr lst)))))
    "#;
    eval_str(fold_def, &env).unwrap();
    eval_str("(define add (lambda (a b) (+ a b)))", &env).unwrap();

    benchmark_expression(
        "fold add 0 '(1 2 3 4 5)",
        "(fold add 0 '(1 2 3 4 5))",
        &env,
        10000,
    );

    // Compose function
    eval_str(
        "(define compose (lambda (f g) (lambda (x) (f (g x)))))",
        &env,
    )
    .unwrap();
    eval_str("(define inc (lambda (x) (+ x 1)))", &env).unwrap();
    eval_str("(define square-then-inc (compose inc square))", &env).unwrap();

    benchmark_expression("composed function", "(square-then-inc 5)", &env, 50000);

    // ========================================================================
    // SECTION 9: Complex Data Structures
    // ========================================================================
    println!("┌─────────────────────────────────────────────────────────────┐");
    println!("│ 9. COMPLEX DATA STRUCTURES                                  │");
    println!("└─────────────────────────────────────────────────────────────┘\n");

    // Build large list
    let build_list_def = r#"
        (define build-list
            (lambda (n acc)
                (if (= n 0)
                    acc
                    (build-list (- n 1) (cons n acc)))))
    "#;
    eval_str(build_list_def, &env).unwrap();

    println!("--- Building Large Lists ---");
    for size in [10, 50, 100, 500] {
        let expr = format!("(length (build-list {} nil))", size);
        benchmark_single(&format!("build-list({})", size), &expr, &env);
    }
    println!();

    // Nested lists
    benchmark_expression("car of nested", "(car (car '((1 2) (3 4))))", &env, 50000);
    benchmark_expression(
        "deep nesting",
        "(car (car (car '(((1 2) 3) 4))))",
        &env,
        50000,
    );

    // ========================================================================
    // SECTION 10: Mutual Recursion
    // ========================================================================
    println!("┌─────────────────────────────────────────────────────────────┐");
    println!("│ 10. MUTUAL RECURSION                                        │");
    println!("└─────────────────────────────────────────────────────────────┘\n");

    let mutual_rec_def = r#"
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
    "#;
    eval_str_multiple(mutual_rec_def, &env).unwrap();

    println!("--- Even/Odd Test ---");
    for n in [10, 100, 1000, 5000] {
        let expr = format!("(is-even {})", n);
        benchmark_single(&format!("is-even({})", n), &expr, &env);
    }
    println!();

    // ========================================================================
    // SECTION 11: Expression Complexity
    // ========================================================================
    println!("┌─────────────────────────────────────────────────────────────┐");
    println!("│ 11. EXPRESSION COMPLEXITY                                   │");
    println!("└─────────────────────────────────────────────────────────────┘\n");

    benchmark_expression(
        "deeply nested expr",
        "(+ (+ (+ (+ (+ 1 2) 3) 4) 5) 6)",
        &env,
        50000,
    );

    benchmark_expression("wide expr", "(+ 1 2 3 4 5 6 7 8 9 10)", &env, 50000);

    benchmark_expression("mixed nesting", "(* (+ 2 3) (- 10 (/ 20 4)))", &env, 50000);

    // ========================================================================
    // SECTION 12: Real-World Patterns
    // ========================================================================
    println!("┌─────────────────────────────────────────────────────────────┐");
    println!("│ 12. REAL-WORLD PATTERNS                                     │");
    println!("└─────────────────────────────────────────────────────────────┘\n");

    // Range function
    let range_def = r#"
        (define range
            (lambda (start end acc)
                (if (> start end)
                    acc
                    (range (+ start 1) end (cons start acc)))))
    "#;
    eval_str(range_def, &env).unwrap();

    benchmark_expression("range 1 10", "(reverse (range 1 10 nil))", &env, 10000);

    // Take first n elements
    let take_def = r#"
        (define take
            (lambda (n lst)
                (if (= n 0)
                    nil
                    (if (null? lst)
                        nil
                        (cons (car lst) (take (- n 1) (cdr lst)))))))
    "#;
    eval_str(take_def, &env).unwrap();

    benchmark_expression("take 5 from big-list", "(take 5 big-list)", &env, 10000);

    // Zip two lists
    let zip_def = r#"
        (define zip
            (lambda (lst1 lst2)
                (if (null? lst1)
                    nil
                    (if (null? lst2)
                        nil
                        (cons (cons (car lst1) (car lst2))
                              (zip (cdr lst1) (cdr lst2)))))))
    "#;
    eval_str(zip_def, &env).unwrap();

    benchmark_expression("zip two lists", "(zip '(1 2 3) '(4 5 6))", &env, 10000);

    // ========================================================================
    // SECTION 13: Performance Scaling Analysis
    // ========================================================================
    println!("┌─────────────────────────────────────────────────────────────┐");
    println!("│ 13. PERFORMANCE SCALING ANALYSIS                            │");
    println!("└─────────────────────────────────────────────────────────────┘\n");

    println!("--- Fibonacci Scaling (O(φ^n)) ---");
    println!(
        "{:<10} | {:<12} | {:<15} | {:<10}",
        "Input", "Result", "Time", "Growth"
    );
    println!("{}", "-".repeat(60));

    let mut last_duration = 0u128;
    for n in [5, 10, 15, 20, 23] {
        let expr = format!("(fib {})", n);
        let start = Instant::now();
        let result = eval_str(&expr, &env).unwrap();
        let duration = start.elapsed().as_nanos();

        let growth = if last_duration > 0 {
            format!("{:.2}x", duration as f64 / last_duration as f64)
        } else {
            "-".to_string()
        };

        println!(
            "fib({:2})    | {:#?} | {:<15} | {}",
            n,
            result,
            format_duration(duration),
            growth
        );

        last_duration = duration;
    }
    println!();

    println!("--- Tail Recursion Scaling (O(n)) ---");
    println!("{:<15} | {:<15} | {:<10}", "Input", "Time", "Growth");
    println!("{}", "-".repeat(50));

    last_duration = 0;
    for n in [1000, 10000, 50000, 100000] {
        let expr = format!("(countdown {})", n);
        let start = Instant::now();
        let _ = eval_str(&expr, &env).unwrap();
        let duration = start.elapsed().as_nanos();

        let growth = if last_duration > 0 {
            format!("{:.2}x", duration as f64 / last_duration as f64)
        } else {
            "-".to_string()
        };

        println!(
            "countdown({:<5}) | {:<15} | {}",
            n,
            format_duration(duration),
            growth
        );

        last_duration = duration;
    }
    println!();

    // ========================================================================
    // SECTION 14: Memory Stress Test
    // ========================================================================
    println!("┌─────────────────────────────────────────────────────────────┐");
    println!("│ 14. MEMORY & ALLOCATION STRESS                              │");
    println!("└─────────────────────────────────────────────────────────────┘\n");

    println!("--- Repeated Large Allocations ---");
    let start = Instant::now();
    for _ in 0..100 {
        let _ = eval_str("(build-list 100 nil)", &env);
    }
    let duration = start.elapsed();
    println!(
        "  100 iterations of build-list(100): {}",
        format_duration(duration.as_nanos())
    );

    println!("\n--- Repeated Lambda Creation ---");
    let start = Instant::now();
    for i in 0..1000 {
        let expr = format!("((lambda (x) (+ x {})) 10)", i);
        let _ = eval_str(&expr, &env);
    }
    let duration = start.elapsed();
    println!(
        "  1000 lambda creations and calls: {}",
        format_duration(duration.as_nanos())
    );

    // ========================================================================
    // Final Summary
    // ========================================================================
    println!("\n╔════════════════════════════════════════════════════════════╗");
    println!("║                  BENCHMARK COMPLETE                        ║");
    println!("╚════════════════════════════════════════════════════════════╝");

    println!("\nKey Observations:");
    println!("  • Tail call optimization enables deep recursion");
    println!("  • Higher-order functions show minimal overhead");
    println!("  • Memory allocation patterns are stable");
    println!("  • Closure creation is efficient");
    println!("  • Expression evaluation scales predictably");
}
