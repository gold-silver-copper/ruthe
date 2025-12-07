// examples/benchmark.rs
// Run with: cargo run --example benchmark --release

use ruthe::{EnvRef, eval_str};
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

    println!("{}:", name);
    println!("  Min:    {}", format_duration(min));
    println!("  Max:    {}", format_duration(max));
    println!("  Median: {}", format_duration(median));
    println!("  Mean:   {}", format_duration(mean));
    println!();
}

fn main() {
    println!("=== Lisp Interpreter Benchmarks ===\n");

    // Create environment using new API
    let env = EnvRef::new();

    // Define fibonacci function
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

    println!("Defining fibonacci function...");
    let start = Instant::now();
    eval_str(fib_def, &env).unwrap();
    let duration = start.elapsed();
    println!(
        "Definition took: {}\n",
        format_duration(duration.as_nanos())
    );

    // Benchmark simple arithmetic
    println!("--- Simple Arithmetic ---");
    benchmark_expression("Addition (+ 1 2 3)", "(+ 1 2 3)", &env, 10000);
    benchmark_expression("Multiplication (* 2 3 4)", "(* 2 3 4)", &env, 10000);
    benchmark_expression("Division (/ 22 7)", "(/ 22 7)", &env, 10000);

    // Benchmark fibonacci with single run
    println!("--- Fibonacci Calculations (Single Run) ---");
    for n in [5, 10, 15, 20, 25, 30] {
        let expr = format!("(fib {})", n);
        let start = Instant::now();
        let result = eval_str(&expr, &env).unwrap();
        let duration = start.elapsed();

        println!(
            "fib({:2}) = {:>8} | Time: {}",
            n,
            result,
            format_duration(duration.as_nanos())
        );
    }
    println!();

    // Benchmark smaller fibonacci values with multiple iterations
    println!("--- Fibonacci Benchmarks (Multiple Iterations) ---");
    benchmark_expression("fib(5)", "(fib 5)", &env, 1000);
    benchmark_expression("fib(10)", "(fib 10)", &env, 1000);
    benchmark_expression("fib(15)", "(fib 15)", &env, 100);
    benchmark_expression("fib(20)", "(fib 20)", &env, 10);

    // Define factorial function
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

    eval_str(fact_def, &env).unwrap();

    println!("--- Factorial Calculations ---");
    for n in [5, 10, 15, 20] {
        let expr = format!("(fact {})", n);
        let start = Instant::now();
        let result = eval_str(&expr, &env).unwrap();
        let duration = start.elapsed();

        println!(
            "fact({:2}) = {:>20} | Time: {}",
            n,
            result,
            format_duration(duration.as_nanos())
        );
    }
    println!();

    // Benchmark list operations
    println!("--- List Operations ---");
    benchmark_expression("car '(1 2 3)", "(car '(1 2 3))", &env, 10000);
    benchmark_expression("cdr '(1 2 3)", "(cdr '(1 2 3))", &env, 10000);
    benchmark_expression("length '(1 2 3 4 5)", "(length '(1 2 3 4 5))", &env, 10000);
    benchmark_expression(
        "reverse '(1 2 3 4 5)",
        "(reverse '(1 2 3 4 5))",
        &env,
        10000,
    );

    // Lambda creation and application benchmark
    println!("--- Lambda Performance ---");
    let square_def = "(define square (lambda (x) (* x x)))";
    eval_str(square_def, &env).unwrap();
    benchmark_expression("square 5", "(square 5)", &env, 10000);

    // Nested lambda
    let nested_def = r#"
        (define make-adder 
            (lambda (n) 
                (lambda (x) (+ x n))
            )
        )
    "#;
    eval_str(nested_def, &env).unwrap();
    eval_str("(define add5 (make-adder 5))", &env).unwrap();
    benchmark_expression("add5 10", "(add5 10)", &env, 10000);

    // Complex expression
    println!("--- Complex Expression ---");
    let complex_expr = "(+ (* 2 3) (/ 10 2) (- 8 3))";
    benchmark_expression("Complex arithmetic", complex_expr, &env, 10000);

    // Tail call optimization test
    println!("--- Tail Call Optimization Test ---");
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
    eval_str(countdown_def, &env).unwrap();

    println!("Testing TCO with deep recursion (should not overflow):");
    let test_sizes = [1000, 10000, 100000, 1000000];
    for size in test_sizes {
        let expr = format!("(countdown {})", size);
        let start = Instant::now();
        match eval_str(&expr, &env) {
            Ok(result) => {
                let duration = start.elapsed();
                println!(
                    "  countdown({:>7}) = {} | Time: {}",
                    size,
                    result,
                    format_duration(duration.as_nanos())
                );
            }
            Err(e) => {
                println!("  countdown({:>7}) FAILED: {}", size, e);
                break;
            }
        }
    }
    println!();

    // Performance scaling analysis
    println!("--- Performance Scaling (fib) ---");
    println!(
        "{:<10} | {:<12} | {:<15} | {:<10}",
        "Input", "Result", "Time", "Growth"
    );
    println!("{}", "-".repeat(60));

    let mut last_duration = 0u128;
    for n in [5, 10, 15, 20, 35] {
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
            "fib({:2})    | {:<12} | {:<15} | {}",
            n,
            result,
            format_duration(duration),
            growth
        );

        last_duration = duration;
    }

    println!("\n=== Benchmark Complete ===");
}
