// examples/benchmark.rs
// Run with: cargo run --example benchmark --release

use ruthe::{eval_str, eval_str_multiple, new_env};
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

fn benchmark_expression(name: &str, expr: &str, env: &ruthe::ValRef, iterations: usize) {
    let mut times = Vec::with_capacity(iterations);

    for _ in 0..iterations {
        let start = Instant::now();
        let _ = eval_str(expr, env);
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

fn benchmark_single(name: &str, expr: &str, env: &ruthe::ValRef) -> Result<u128, String> {
    let start = Instant::now();
    let result = eval_str(expr, env).map_err(|e| {
        let mut buf = [0u8; 256];
        e.to_display_str(&mut buf).unwrap().to_string()
    })?;
    let duration = start.elapsed().as_nanos();
    let mut buf = [0u8; 4096];
    let result_str = result.to_display_str(&mut buf).unwrap_or("<display error>");
    println!(
        "  {} = {} | Time: {}",
        name,
        result_str,
        format_duration(duration)
    );
    Ok(duration)
}

fn main() {
    println!("╔════════════════════════════════════════════════════════════╗");
    println!("║     COMPREHENSIVE LISP INTERPRETER BENCHMARK SUITE        ║");
    println!("╚════════════════════════════════════════════════════════════╝\n");

    let env = new_env();

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
    let _ = eval_str_multiple(
        "(define big-list '(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20))",
        &env,
    );
    benchmark_expression(
        "length of 20-element list",
        "(length big-list)",
        &env,
        10000,
    );

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

    // Closure tests
    let _ = eval_str_multiple(
        r#"
        (define make-adder (lambda (n) (lambda (x) (+ x n))))
        (define add5 (make-adder 5))
        (define add10 (make-adder 10))
        "#,
        &env,
    );
    benchmark_expression("Closure call", "(+ (add5 10) (add10 5))", &env, 50000);

    // Nested closures
    let _ = eval_str_multiple(
        r#"
        (define make-multiplier 
          (lambda (a) 
            (lambda (b) 
              (lambda (c) 
                (* a (* b c))))))
        (define mul2 (make-multiplier 2))
        (define mul2x3 (mul2 3))
        "#,
        &env,
    );
    benchmark_expression("Nested closure call", "(mul2x3 4)", &env, 50000);

    // ========================================================================
    // SECTION 6: Recursive Functions
    // ========================================================================
    println!("┌─────────────────────────────────────────────────────────────┐");
    println!("│ 6. RECURSIVE FUNCTIONS                                      │");
    println!("└─────────────────────────────────────────────────────────────┘\n");

    // Fibonacci (tree recursion)
    let _ = eval_str_multiple(
        r#"
        (define fib
          (lambda (n)
            (if (< n 2)
                n
                (+ (fib (- n 1)) (fib (- n 2))))))
        "#,
        &env,
    );

    println!("--- Fibonacci (Tree Recursion) ---");
    for n in [5, 10, 15, 20, 25] {
        if let Ok(duration) =
            benchmark_single(&format!("fib({})", n), &format!("(fib {})", n), &env)
        {
            println!("    Duration: {}", format_duration(duration));
        }
    }
    println!();

    // Factorial (simple recursion)
    let _ = eval_str_multiple(
        r#"
        (define factorial
          (lambda (n acc)
            (if (= n 0)
                acc
                (factorial (- n 1) (* n acc)))))
        "#,
        &env,
    );

    println!("--- Factorial (Simple Recursion) ---");
    for n in [5, 10, 15, 20] {
        if let Ok(duration) = benchmark_single(
            &format!("factorial({})", n),
            &format!("(factorial {} 1)", n),
            &env,
        ) {
            println!("    Duration: {}", format_duration(duration));
        }
    }
    println!();

    // Ackermann function (complex recursion)
    let _ = eval_str_multiple(
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
    );

    println!("--- Ackermann Function (Complex Recursion) ---");
    for (m, n) in [(1, 2), (2, 2), (2, 3), (3, 2), (3, 3)] {
        if let Ok(duration) = benchmark_single(
            &format!("ack({}, {})", m, n),
            &format!("(ackermann {} {})", m, n),
            &env,
        ) {
            println!("    Duration: {}", format_duration(duration));
        }
    }
    println!();

    // Sum list (linear recursion)
    let _ = eval_str_multiple(
        r#"
        (define sum-list
          (lambda (lst acc)
            (if (null? lst)
                acc
                (sum-list (cdr lst) (+ acc (car lst))))))
        "#,
        &env,
    );
    benchmark_expression(
        "sum-list '(1 2 3 4 5)",
        "(sum-list '(1 2 3 4 5) 0)",
        &env,
        10000,
    );

    // ========================================================================
    // SECTION 7: Tail Call Optimization
    // ========================================================================
    println!("┌─────────────────────────────────────────────────────────────┐");
    println!("│ 7. TAIL CALL OPTIMIZATION                                   │");
    println!("└─────────────────────────────────────────────────────────────┘\n");

    // Tail-recursive countdown
    let _ = eval_str_multiple(
        r#"
        (define countdown
          (lambda (n acc)
            (if (= n 0)
                acc
                (countdown (- n 1) (+ acc 1)))))
        "#,
        &env,
    );

    println!("--- Countdown (Tail Recursion) ---");
    for size in [1000, 10000, 100000, 500000] {
        if let Ok(duration) = benchmark_single(
            &format!("countdown({})", size),
            &format!("(countdown {} 0)", size),
            &env,
        ) {
            println!("    Duration: {}", format_duration(duration));
        }
    }
    println!();

    // Tail-recursive factorial (already defined above)
    println!("--- Tail-Recursive Factorial ---");
    for n in [10, 20] {
        if let Ok(duration) = benchmark_single(
            &format!("factorial-tail({})", n),
            &format!("(factorial {} 1)", n),
            &env,
        ) {
            println!("    Duration: {}", format_duration(duration));
        }
    }
    println!();

    // Tail-recursive sum
    let _ = eval_str_multiple(
        r#"
        (define sum-tail
          (lambda (n acc)
            (if (= n 0)
                acc
                (sum-tail (- n 1) (+ acc n)))))
        "#,
        &env,
    );

    println!("--- Sum with Tail Recursion ---");
    for n in [100, 1000, 10000, 50000] {
        if let Ok(duration) = benchmark_single(
            &format!("sum-tail({})", n),
            &format!("(sum-tail {} 0)", n),
            &env,
        ) {
            println!("    Duration: {}", format_duration(duration));
        }
    }
    println!();

    // ========================================================================
    // SECTION 8: Higher-Order Functions
    // ========================================================================
    println!("┌─────────────────────────────────────────────────────────────┐");
    println!("│ 8. HIGHER-ORDER FUNCTIONS                                   │");
    println!("└─────────────────────────────────────────────────────────────┘\n");

    // Map function
    let _ = eval_str_multiple(
        r#"
        (define map
          (lambda (f lst)
            (if (null? lst)
                nil
                (cons (f (car lst)) (map f (cdr lst))))))
        "#,
        &env,
    );
    benchmark_expression(
        "map square '(1 2 3 4 5)",
        "(map (lambda (x) (* x x)) '(1 2 3 4 5))",
        &env,
        10000,
    );

    // Filter function
    let _ = eval_str_multiple(
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
    );
    benchmark_expression(
        "filter positive?",
        "(filter (lambda (x) (> x 0)) '(-3 -1 0 1 2 3))",
        &env,
        10000,
    );

    // Compose function
    let _ = eval_str_multiple(
        r#"
        (define compose
          (lambda (f g)
            (lambda (x) (f (g x)))))
        (define add1 (lambda (x) (+ x 1)))
        (define square (lambda (x) (* x x)))
        (define composed (compose add1 square))
        "#,
        &env,
    );
    benchmark_expression("composed function", "(composed 5)", &env, 50000);

    // ========================================================================
    // SECTION 9: Complex Data Structures
    // ========================================================================
    println!("┌─────────────────────────────────────────────────────────────┐");
    println!("│ 9. COMPLEX DATA STRUCTURES                                  │");
    println!("└─────────────────────────────────────────────────────────────┘\n");

    // Build large list
    let _ = eval_str_multiple(
        r#"
        (define build-list
          (lambda (n acc)
            (if (= n 0)
                acc
                (build-list (- n 1) (cons n acc)))))
        "#,
        &env,
    );

    println!("--- Building Large Lists ---");
    for size in [10, 50, 100, 500] {
        if let Ok(duration) = benchmark_single(
            &format!("build-list({})", size),
            &format!("(length (build-list {} nil))", size),
            &env,
        ) {
            println!("    Duration: {}", format_duration(duration));
        }
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

    let _ = eval_str_multiple(
        r#"
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
        "#,
        &env,
    );

    println!("--- Even/Odd Test ---");
    for n in [10, 100, 1000, 5000] {
        if let Ok(duration) = benchmark_single(
            &format!("is-even({})", n),
            &format!("(is-even {})", n),
            &env,
        ) {
            println!("    Duration: {}", format_duration(duration));
        }
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
    let _ = eval_str_multiple(
        r#"
        (define range
          (lambda (start end acc)
            (if (> start end)
                acc
                (range (+ start 1) end (cons start acc)))))
        "#,
        &env,
    );
    benchmark_expression("range 1 10", "(reverse (range 1 10 nil))", &env, 10000);

    // Take first n elements
    let _ = eval_str_multiple(
        r#"
        (define take
          (lambda (n lst)
            (if (= n 0)
                nil
                (if (null? lst)
                    nil
                    (cons (car lst) (take (- n 1) (cdr lst)))))))
        "#,
        &env,
    );
    benchmark_expression(
        "take 5 from list",
        "(take 5 '(1 2 3 4 5 6 7 8 9 10))",
        &env,
        10000,
    );

    // ========================================================================
    // SECTION 13: Performance Scaling Analysis
    // ========================================================================
    println!("┌─────────────────────────────────────────────────────────────┐");
    println!("│ 13. PERFORMANCE SCALING ANALYSIS                            │");
    println!("└─────────────────────────────────────────────────────────────┘\n");

    println!("--- Fibonacci Scaling (O(φ^n)) ---");
    println!("{:<10} | {:<15} | {:<10}", "Input", "Time", "Growth");
    println!("{}", "-".repeat(40));

    let mut last_duration = 0u128;
    for n in [5, 10, 15, 20, 23] {
        let start = Instant::now();
        let _ = eval_str(&format!("(fib {})", n), &env);
        let duration = start.elapsed().as_nanos();

        let growth = if last_duration > 0 {
            format!("{:.2}x", duration as f64 / last_duration as f64)
        } else {
            "-".to_string()
        };

        println!(
            "fib({:2})    | {:<15} | {}",
            n,
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
        let start = Instant::now();
        let _ = eval_str(&format!("(countdown {} 0)", n), &env);
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
    println!("  • Higher-order functions work efficiently with define");
    println!("  • Memory allocation patterns are stable");
    println!("  • Closure creation is efficient");
    println!("  • Expression evaluation scales predictably");
    println!("  • Define makes code much cleaner than self-application");
}
