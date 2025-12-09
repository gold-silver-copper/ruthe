// examples/bench.rs
// Run with: cargo run --example bench --release

#![no_std]

extern crate std;
use std::format;
use std::println;
use std::string::String;
use std::time::Instant;
use std::vec::Vec;

use ruthe::{Arena, Ref, Value, env_new, eval_string};
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

fn format_value(arena: &Arena, val: &Ref) -> String {
    match arena.get(**val) {
        Some(Value::Number(n)) => format!("{}", n),
        Some(Value::Bool(true)) => String::from("#t"),
        Some(Value::Bool(false)) => String::from("#f"),
        Some(Value::Nil) => String::from("nil"),
        Some(Value::Cons(_car, _cdr)) => {
            let mut result = String::from("(");
            let mut current = val.clone();
            let mut first = true;

            loop {
                match arena.get(*current) {
                    Some(Value::Cons(car, cdr)) => {
                        if !first {
                            result.push(' ');
                        }
                        first = false;
                        let car_ref = Ref::new(arena, car);
                        result.push_str(&format_value(arena, &car_ref));

                        match arena.get(cdr) {
                            Some(Value::Nil) => break,
                            Some(Value::Cons(..)) => current = Ref::new(arena, cdr),
                            _ => {
                                result.push_str(" . ");
                                let cdr_ref = Ref::new(arena, cdr);
                                result.push_str(&format_value(arena, &cdr_ref));
                                break;
                            }
                        }
                    }
                    _ => break,
                }
            }

            result.push(')');
            result
        }
        Some(Value::Symbol(s)) => {
            let s_ref = Ref::new(arena, s);
            let mut buf = [0u8; 256];
            if let Some(str) = arena.list_to_str(&s_ref, &mut buf) {
                String::from(str)
            } else {
                String::from("symbol")
            }
        }
        Some(Value::Lambda(..)) => String::from("#<lambda>"),
        Some(Value::Builtin(_)) => String::from("#<builtin>"),
        Some(Value::Char(c)) => format!("{}", c),
        Some(Value::Free) => String::from("#<free>"),
        None => String::from("#<null>"),
    }
}

fn benchmark_expression(name: &str, expr: &str, iterations: usize) {
    let mut times = Vec::with_capacity(iterations);

    for _ in 0..iterations {
        let arena = Arena::new();
        let env = env_new(&arena);

        let start = Instant::now();
        let _result = eval_string(&arena, expr, &env);
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

fn benchmark_single(arena: &Arena, name: &str, expr: &str, env: &Ref) -> Result<u128, String> {
    let start = Instant::now();
    let result = eval_string(arena, expr, env).map_err(|e| {
        let mut buf = [0u8; 256];
        if let Some(s) = arena.list_to_str(&e, &mut buf) {
            String::from(s)
        } else {
            String::from("Error")
        }
    })?;
    let duration = start.elapsed().as_nanos();
    let result_str = format_value(arena, &result);
    println!(
        "  {} = {} ({})",
        name,
        result_str,
        format_duration(duration)
    );
    Ok(duration)
}

fn main() {
    println!("╔════════════════════════════════════════════════════════════╗");
    println!("║     COMPREHENSIVE LISP INTERPRETER BENCHMARK SUITE        ║");
    println!("║         (Arena-Based with Automatic RefCounting)           ║");
    println!("╚════════════════════════════════════════════════════════════╝\n");

    // ========================================================================
    // SECTION 1: Basic Arithmetic Operations
    // ========================================================================
    println!("┌─────────────────────────────────────────────────────────────┐");
    println!("│ 1. BASIC ARITHMETIC OPERATIONS                              │");
    println!("└─────────────────────────────────────────────────────────────┘\n");

    benchmark_expression("Simple addition (+ 1 2)", "(+ 1 2)", 50000);
    benchmark_expression("Multi-operand add (+ 1 2 3 4 5)", "(+ 1 2 3 4 5)", 50000);
    benchmark_expression(
        "Large numbers (* 999999 999999)",
        "(* 999999 999999)",
        50000,
    );
    benchmark_expression("Subtraction (- 1000 999)", "(- 1000 999)", 50000);
    benchmark_expression("Division (/ 1000000 7)", "(/ 1000000 7)", 50000);
    benchmark_expression("Nested arithmetic", "(+ (* 2 3) (/ 100 5) (- 10 3))", 50000);

    // ========================================================================
    // SECTION 2: Comparison and Boolean Operations
    // ========================================================================
    println!("┌─────────────────────────────────────────────────────────────┐");
    println!("│ 2. COMPARISON & BOOLEAN OPERATIONS                          │");
    println!("└─────────────────────────────────────────────────────────────┘\n");

    benchmark_expression("Equality (= 5 5)", "(= 5 5)", 50000);
    benchmark_expression("Inequality (= 5 6)", "(= 5 6)", 50000);
    benchmark_expression("Less than (< 3 5)", "(< 3 5)", 50000);
    benchmark_expression("Greater than (> 10 5)", "(> 10 5)", 50000);
    benchmark_expression("Complex comparison", "(< (+ 2 3) (* 2 4))", 50000);

    // ========================================================================
    // SECTION 3: List Operations
    // ========================================================================
    println!("┌─────────────────────────────────────────────────────────────┐");
    println!("│ 3. LIST OPERATIONS                                          │");
    println!("└─────────────────────────────────────────────────────────────┘\n");

    benchmark_expression("car '(1 2 3)", "(car '(1 2 3))", 50000);
    benchmark_expression("cdr '(1 2 3)", "(cdr '(1 2 3))", 50000);
    benchmark_expression("cons 1 '(2 3)", "(cons 1 '(2 3))", 50000);
    benchmark_expression("list 1 2 3 4 5", "(list 1 2 3 4 5)", 50000);
    benchmark_expression("length '(1 2 3 4 5)", "(length '(1 2 3 4 5))", 50000);
    benchmark_expression("reverse '(1 2 3 4 5)", "(reverse '(1 2 3 4 5))", 50000);
    benchmark_expression("append two lists", "(append '(1 2) '(3 4))", 50000);
    benchmark_expression("null? nil", "(null? nil)", 50000);

    // ========================================================================
    // SECTION 4: Conditional Expressions
    // ========================================================================
    println!("┌─────────────────────────────────────────────────────────────┐");
    println!("│ 4. CONDITIONAL EXPRESSIONS                                  │");
    println!("└─────────────────────────────────────────────────────────────┘\n");

    benchmark_expression("if #t 1 2", "(if #t 1 2)", 50000);
    benchmark_expression("if #f 1 2", "(if #f 1 2)", 50000);
    benchmark_expression("if with computation", "(if (< 3 5) (+ 1 2) (* 3 4))", 50000);
    benchmark_expression("nested if", "(if #t (if #f 1 2) 3)", 50000);

    // ========================================================================
    // SECTION 5: Lambda and Closures
    // ========================================================================
    println!("┌─────────────────────────────────────────────────────────────┐");
    println!("│ 5. LAMBDA EXPRESSIONS & CLOSURES                            │");
    println!("└─────────────────────────────────────────────────────────────┘\n");

    benchmark_expression("Simple lambda", "((lambda (x) x) 5)", 50000);
    benchmark_expression("Lambda with computation", "((lambda (x) (* x x)) 7)", 50000);
    benchmark_expression("Multi-arg lambda", "((lambda (x y) (+ x y)) 3 4)", 50000);
    benchmark_expression("Zero-arg lambda", "((lambda () 42))", 50000);

    // Closure tests - need shared arena
    {
        let arena = Arena::new();
        let env = env_new(&arena);

        let _ = eval_string(
            &arena,
            "(define make-adder (lambda (n) (lambda (x) (+ x n))))",
            &env,
        );
        let _ = eval_string(&arena, "(define add5 (make-adder 5))", &env);
        let _ = eval_string(&arena, "(define add10 (make-adder 10))", &env);

        let mut times = Vec::with_capacity(50000);
        for _ in 0..50000 {
            let start = Instant::now();
            let _result = eval_string(&arena, "(+ (add5 10) (add10 5))", &env);
            let duration = start.elapsed();
            times.push(duration.as_nanos());
        }

        times.sort_unstable();
        let median = times[times.len() / 2];
        let mean: u128 = times.iter().sum::<u128>() / times.len() as u128;

        println!("Closure call:");
        println!("  Median: {}", format_duration(median));
        println!("  Mean:   {}", format_duration(mean));
        println!();
    }

    // ========================================================================
    // SECTION 6: Recursive Functions
    // ========================================================================
    println!("┌─────────────────────────────────────────────────────────────┐");
    println!("│ 6. RECURSIVE FUNCTIONS                                      │");
    println!("└─────────────────────────────────────────────────────────────┘\n");

    // Use shared arena for define-based tests
    let arena = Arena::new();
    let env = env_new(&arena);

    // Fibonacci (tree recursion)
    let fib_result = eval_string(
        &arena,
        r#"
        (define fib
          (lambda (n)
            (if (< n 2)
                n
                (+ (fib (- n 1)) (fib (- n 2))))))
        "#,
        &env,
    );

    if let Err(e) = fib_result {
        println!("ERROR defining fib: {}", format_value(&arena, &e));
    } else {
        println!("--- Fibonacci (Tree Recursion) ---");
        for n in [5, 10, 15, 20] {
            match benchmark_single(
                &arena,
                &format!("fib({})", n),
                &format!("(fib {})", n),
                &env,
            ) {
                Ok(_) => {}
                Err(e) => println!("  fib({}) ERROR: {}", n, e),
            }
        }
        println!();
    }

    // Factorial (tail recursion)
    let fact_result = eval_string(
        &arena,
        r#"
        (define factorial
          (lambda (n acc)
            (if (= n 0)
                acc
                (factorial (- n 1) (* n acc)))))
        "#,
        &env,
    );

    if let Err(e) = fact_result {
        println!("ERROR defining factorial: {}", format_value(&arena, &e));
    } else {
        println!("--- Factorial (Tail Recursion) ---");
        for n in [5, 10, 15, 20] {
            match benchmark_single(
                &arena,
                &format!("factorial({})", n),
                &format!("(factorial {} 1)", n),
                &env,
            ) {
                Ok(_) => {}
                Err(e) => println!("  factorial({}) ERROR: {}", n, e),
            }
        }
        println!();
    }

    // Ackermann function (complex recursion)
    let ack_result = eval_string(
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
    );

    if let Err(e) = ack_result {
        println!("ERROR defining ackermann: {}", format_value(&arena, &e));
    } else {
        println!("--- Ackermann Function (Complex Recursion) ---");
        for (m, n) in [(1, 2), (2, 2), (2, 3), (3, 2)] {
            match benchmark_single(
                &arena,
                &format!("ack({}, {})", m, n),
                &format!("(ackermann {} {})", m, n),
                &env,
            ) {
                Ok(_) => {}
                Err(e) => println!("  ack({}, {}) ERROR: {}", m, n, e),
            }
        }
        println!();
    }

    // Sum list (linear recursion)
    let sum_list_result = eval_string(
        &arena,
        r#"
        (define sum-list
          (lambda (lst acc)
            (if (null? lst)
                acc
                (sum-list (cdr lst) (+ acc (car lst))))))
        "#,
        &env,
    );

    if let Err(e) = sum_list_result {
        println!("ERROR defining sum-list: {}", format_value(&arena, &e));
    } else {
        let mut times = Vec::with_capacity(10000);
        for _ in 0..10000 {
            let start = Instant::now();
            let result = eval_string(&arena, "(sum-list '(1 2 3 4 5) 0)", &env);
            let duration = start.elapsed();
            if let Err(e) = result {
                println!("ERROR in sum-list: {}", format_value(&arena, &e));
                break;
            }
            times.push(duration.as_nanos());
        }

        if !times.is_empty() {
            times.sort_unstable();
            let median = times[times.len() / 2];
            let mean: u128 = times.iter().sum::<u128>() / times.len() as u128;
            println!("sum-list '(1 2 3 4 5):");
            println!("  Median: {}", format_duration(median));
            println!("  Mean:   {}", format_duration(mean));
            println!();
        }
    }

    // ========================================================================
    // SECTION 7: Tail Call Optimization
    // ========================================================================
    println!("┌─────────────────────────────────────────────────────────────┐");
    println!("│ 7. TAIL CALL OPTIMIZATION                                   │");
    println!("└─────────────────────────────────────────────────────────────┘\n");

    // Tail-recursive countdown
    let countdown_result = eval_string(
        &arena,
        r#"
        (define countdown
          (lambda (n acc)
            (if (= n 0)
                acc
                (countdown (- n 1) (+ acc 1)))))
        "#,
        &env,
    );

    if let Err(e) = countdown_result {
        println!("ERROR defining countdown: {}", format_value(&arena, &e));
    } else {
        println!("--- Countdown (Tail Recursion) ---");
        for size in [1000, 10000, 100000] {
            match benchmark_single(
                &arena,
                &format!("countdown({})", size),
                &format!("(countdown {} 0)", size),
                &env,
            ) {
                Ok(_) => {}
                Err(e) => println!("  countdown({}) ERROR: {}", size, e),
            }
        }
        println!();
    }

    // Tail-recursive sum
    let sum_tail_result = eval_string(
        &arena,
        r#"
        (define sum-tail
          (lambda (n acc)
            (if (= n 0)
                acc
                (sum-tail (- n 1) (+ acc n)))))
        "#,
        &env,
    );

    if let Err(e) = sum_tail_result {
        println!("ERROR defining sum-tail: {}", format_value(&arena, &e));
    } else {
        println!("--- Sum with Tail Recursion ---");
        for n in [100, 1000, 10000] {
            match benchmark_single(
                &arena,
                &format!("sum-tail({})", n),
                &format!("(sum-tail {} 0)", n),
                &env,
            ) {
                Ok(_) => {}
                Err(e) => println!("  sum-tail({}) ERROR: {}", n, e),
            }
        }
        println!();
    }

    // ========================================================================
    // SECTION 8: Higher-Order Functions
    // ========================================================================
    println!("┌─────────────────────────────────────────────────────────────┐");
    println!("│ 8. HIGHER-ORDER FUNCTIONS                                   │");
    println!("└─────────────────────────────────────────────────────────────┘\n");

    // Map function
    let map_result = eval_string(
        &arena,
        r#"
        (define map
          (lambda (f lst)
            (if (null? lst)
                nil
                (cons (f (car lst)) (map f (cdr lst))))))
        "#,
        &env,
    );

    if let Err(e) = map_result {
        println!("ERROR defining map: {}", format_value(&arena, &e));
    } else {
        let mut times = Vec::with_capacity(10000);
        for _ in 0..10000 {
            let start = Instant::now();
            let result = eval_string(&arena, "(map (lambda (x) (* x x)) '(1 2 3 4 5))", &env);
            let duration = start.elapsed();
            if let Err(e) = result {
                println!("ERROR in map: {}", format_value(&arena, &e));
                break;
            }
            times.push(duration.as_nanos());
        }

        if !times.is_empty() {
            times.sort_unstable();
            let median = times[times.len() / 2];
            let mean: u128 = times.iter().sum::<u128>() / times.len() as u128;
            println!("map square '(1 2 3 4 5):");
            println!("  Median: {}", format_duration(median));
            println!("  Mean:   {}", format_duration(mean));
            println!();
        }
    }

    // Filter function
    let filter_result = eval_string(
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
    );

    if let Err(e) = filter_result {
        println!("ERROR defining filter: {}", format_value(&arena, &e));
    } else {
        let mut times = Vec::with_capacity(10000);
        for _ in 0..10000 {
            let start = Instant::now();
            let result = eval_string(
                &arena,
                "(filter (lambda (x) (> x 0)) '(-3 -1 0 1 2 3))",
                &env,
            );
            let duration = start.elapsed();
            if let Err(e) = result {
                println!("ERROR in filter: {}", format_value(&arena, &e));
                break;
            }
            times.push(duration.as_nanos());
        }

        if !times.is_empty() {
            times.sort_unstable();
            let median = times[times.len() / 2];
            let mean: u128 = times.iter().sum::<u128>() / times.len() as u128;
            println!("filter positive?:");
            println!("  Median: {}", format_duration(median));
            println!("  Mean:   {}", format_duration(mean));
            println!();
        }
    }

    // Compose function
    let compose_result = eval_string(
        &arena,
        r#"
        (define compose
          (lambda (f g)
            (lambda (x) (f (g x)))))
        "#,
        &env,
    );

    if let Err(e) = compose_result {
        println!("ERROR defining compose: {}", format_value(&arena, &e));
    } else {
        let _ = eval_string(&arena, "(define add1 (lambda (x) (+ x 1)))", &env);
        let _ = eval_string(&arena, "(define square (lambda (x) (* x x)))", &env);
        let _ = eval_string(&arena, "(define composed (compose add1 square))", &env);

        let mut times = Vec::with_capacity(50000);
        for _ in 0..50000 {
            let start = Instant::now();
            let _result = eval_string(&arena, "(composed 5)", &env);
            let duration = start.elapsed();
            times.push(duration.as_nanos());
        }

        times.sort_unstable();
        let median = times[times.len() / 2];
        let mean: u128 = times.iter().sum::<u128>() / times.len() as u128;
        println!("composed function:");
        println!("  Median: {}", format_duration(median));
        println!("  Mean:   {}", format_duration(mean));
        println!();
    }

    // ========================================================================
    // SECTION 9: Complex Data Structures
    // ========================================================================
    println!("┌─────────────────────────────────────────────────────────────┐");
    println!("│ 9. COMPLEX DATA STRUCTURES                                  │");
    println!("└─────────────────────────────────────────────────────────────┘\n");

    // Build large list
    let build_list_result = eval_string(
        &arena,
        r#"
        (define build-list
          (lambda (n acc)
            (if (= n 0)
                acc
                (build-list (- n 1) (cons n acc)))))
        "#,
        &env,
    );

    if let Err(e) = build_list_result {
        println!("ERROR defining build-list: {}", format_value(&arena, &e));
    } else {
        println!("--- Building Large Lists ---");
        for size in [10, 50, 100, 500] {
            match benchmark_single(
                &arena,
                &format!("build-list({})", size),
                &format!("(length (build-list {} nil))", size),
                &env,
            ) {
                Ok(_) => {}
                Err(e) => println!("  build-list({}) ERROR: {}", size, e),
            }
        }
        println!();
    }

    // ========================================================================
    // SECTION 10: Mutual Recursion
    // ========================================================================
    println!("┌─────────────────────────────────────────────────────────────┐");
    println!("│ 10. MUTUAL RECURSION                                        │");
    println!("└─────────────────────────────────────────────────────────────┘\n");

    let _ = eval_string(
        &arena,
        r#"
        (define is-even
          (lambda (n)
            (if (= n 0)
                #t
                (is-odd (- n 1)))))
        "#,
        &env,
    );

    let _ = eval_string(
        &arena,
        r#"
        (define is-odd
          (lambda (n)
            (if (= n 0)
                #f
                (is-even (- n 1)))))
        "#,
        &env,
    );

    println!("--- Even/Odd Test ---");
    for n in [10, 100, 1000] {
        match benchmark_single(
            &arena,
            &format!("is-even({})", n),
            &format!("(is-even {})", n),
            &env,
        ) {
            Ok(_) => {}
            Err(e) => println!("  is-even({}) ERROR: {}", n, e),
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
        50000,
    );

    benchmark_expression("wide expr", "(+ 1 2 3 4 5 6 7 8 9 10)", 50000);

    benchmark_expression("mixed nesting", "(* (+ 2 3) (- 10 (/ 20 4)))", 50000);

    // ========================================================================
    // SECTION 12: Performance Scaling Analysis
    // ========================================================================
    println!("┌─────────────────────────────────────────────────────────────┐");
    println!("│ 12. PERFORMANCE SCALING ANALYSIS                            │");
    println!("└─────────────────────────────────────────────────────────────┘\n");

    let arena2 = Arena::new();
    let env2 = env_new(&arena2);

    let _ = eval_string(
        &arena2,
        r#"
        (define fib
          (lambda (n)
            (if (< n 2)
                n
                (+ (fib (- n 1)) (fib (- n 2))))))
        "#,
        &env2,
    );

    println!("--- Fibonacci Scaling (O(φ^n)) ---");
    println!("{:<10} | {:<15} | {:<10}", "Input", "Time", "Growth");
    println!("{}", "-".repeat(40));

    let mut last_duration = 0u128;
    for n in [5, 10, 15, 20, 30] {
        let start = Instant::now();
        let result = eval_string(&arena2, &format!("(fib {})", n), &env2);
        let duration = start.elapsed().as_nanos();
        if let Ok(val) = result {
            let result_str = format_value(&arena2, &val);

            let growth = if last_duration > 0 {
                format!("{:.2}x", duration as f64 / last_duration as f64)
            } else {
                String::from("-")
            };

            println!(
                "fib({:2})    | {:<15} | {} (result: {})",
                n,
                format_duration(duration),
                growth,
                result_str
            );

            last_duration = duration;
        } else if let Err(e) = result {
            println!("fib({}) ERROR: {}", n, format_value(&arena2, &e));
        }
    }
    println!();

    let _ = eval_string(
        &arena2,
        r#"
        (define countdown
          (lambda (n acc)
            (if (= n 0)
                acc
                (countdown (- n 1) (+ acc 1)))))
        "#,
        &env2,
    );

    println!("--- Tail Recursion Scaling (O(n)) ---");
    println!("{:<15} | {:<15} | {:<10}", "Input", "Time", "Growth");
    println!("{}", "-".repeat(50));

    last_duration = 0;
    for n in [1000, 10000, 50000, 100000] {
        let start = Instant::now();
        let result = eval_string(&arena2, &format!("(countdown {} 0)", n), &env2);
        let duration = start.elapsed().as_nanos();
        if let Ok(_val) = result {
            let growth = if last_duration > 0 {
                format!("{:.2}x", duration as f64 / last_duration as f64)
            } else {
                String::from("-")
            };

            println!(
                "countdown({:<5}) | {:<15} | {}",
                n,
                format_duration(duration),
                growth
            );

            last_duration = duration;
        } else if let Err(e) = result {
            println!("countdown({}) ERROR: {}", n, format_value(&arena2, &e));
        }
    }
    println!();

    // ========================================================================
    // Final Summary
    // ========================================================================
    println!("\n╔════════════════════════════════════════════════════════════╗");
    println!("║                  BENCHMARK COMPLETE                        ║");
    println!("╚════════════════════════════════════════════════════════════╝");

    println!("\nKey Observations:");
    println!("  • Arena allocation enables efficient memory management");
    println!("  • Tail call optimization enables deep recursion");
    println!("  • Higher-order functions work efficiently with define");
    println!("  • AUTOMATIC reference counting - no manual incref/decref!");
    println!("  • RAII-based memory management prevents leaks");
    println!("  • Closure creation is efficient");
    println!("  • Expression evaluation scales predictably");
    println!("  • Define makes code much cleaner than self-application");
}
