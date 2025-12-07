// examples/benchmark.rs
// Run with: cargo run --example benchmark --release

use ruthe::Interpreter;
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

fn benchmark_expression(name: &str, expr: &str, interp: &Interpreter<8192>, iterations: usize) {
    let mut times = Vec::with_capacity(iterations);

    // Get the global env (assuming it's stored at EnvRef(0))
    let env = ruthe::EnvRef(0);

    for _ in 0..iterations {
        let start = Instant::now();
        let _ = interp.eval_str(expr, env);
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

fn bytes_to_string(buf: &[u8; 4096]) -> String {
    // Find null terminator or end of meaningful data
    let len = buf.iter().position(|&b| b == 0).unwrap_or(4096);
    String::from_utf8_lossy(&buf[..len]).to_string()
}

fn main() {
    println!("=== Arena-Based Lisp Interpreter Benchmarks ===\n");

    // Create interpreter with 8K arena
    let interp: Interpreter<8192> = Interpreter::new();

    // Create global environment
    let env = interp
        .create_global_env()
        .expect("Failed to create global env");

    println!("Arena capacity: {} cells", 8192);
    println!("Initial arena usage: {} cells\n", interp.arena.used());

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
    match interp.eval_str(fib_def, env) {
        Ok(_) => {
            let duration = start.elapsed();
            println!("Definition took: {}", format_duration(duration.as_nanos()));
            println!("Arena usage after fib def: {} cells\n", interp.arena.used());
        }
        Err(e) => {
            let mut buf = [0u8; 64];
            let err_str = interp.strings.get(e, &mut buf).unwrap_or("error");
            println!("Error defining fib: {}\n", err_str);
            return;
        }
    }

    // Benchmark simple arithmetic
    println!("--- Simple Arithmetic ---");
    benchmark_expression("Addition (+ 1 2 3)", "(+ 1 2 3)", &interp, 10000);
    benchmark_expression("Multiplication (* 2 3 4)", "(* 2 3 4)", &interp, 10000);
    benchmark_expression("Division (/ 22 7)", "(/ 22 7)", &interp, 10000);

    // Benchmark fibonacci with single run
    println!("--- Fibonacci Calculations (Single Run) ---");
    for n in [5, 10, 15, 20] {
        let expr = format!("(fib {})", n);
        let start = Instant::now();
        match interp.eval_str(&expr, env) {
            Ok(result) => {
                let duration = start.elapsed();
                let result_str = bytes_to_string(&result);
                println!(
                    "fib({:2}) = {:>10} | Time: {} | Arena: {} cells",
                    n,
                    result_str,
                    format_duration(duration.as_nanos()),
                    interp.arena.used()
                );
            }
            Err(e) => {
                let mut buf = [0u8; 64];
                let err_str = interp.strings.get(e, &mut buf).unwrap_or("error");
                println!("fib({:2}) FAILED: {}", n, err_str);
            }
        }
    }
    println!();

    // Benchmark smaller fibonacci values with multiple iterations
    println!("--- Fibonacci Benchmarks (Multiple Iterations) ---");
    benchmark_expression("fib(5)", "(fib 5)", &interp, 1000);
    benchmark_expression("fib(10)", "(fib 10)", &interp, 1000);
    benchmark_expression("fib(15)", "(fib 15)", &interp, 100);
    benchmark_expression("fib(20)", "(fib 20)", &interp, 10);

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

    match interp.eval_str(fact_def, env) {
        Ok(_) => println!("Factorial function defined"),
        Err(e) => {
            let mut buf = [0u8; 64];
            let err_str = interp.strings.get(e, &mut buf).unwrap_or("error");
            println!("Error defining fact: {}", err_str);
        }
    }

    println!("--- Factorial Calculations ---");
    for n in [5, 10, 15, 20] {
        let expr = format!("(fact {})", n);
        let start = Instant::now();
        match interp.eval_str(&expr, env) {
            Ok(result) => {
                let duration = start.elapsed();
                let result_str = bytes_to_string(&result);
                println!(
                    "fact({:2}) = {:>20} | Time: {}",
                    n,
                    result_str,
                    format_duration(duration.as_nanos())
                );
            }
            Err(e) => {
                let mut buf = [0u8; 64];
                let err_str = interp.strings.get(e, &mut buf).unwrap_or("error");
                println!("fact({:2}) FAILED: {}", n, err_str);
            }
        }
    }
    println!();

    // Benchmark list operations
    println!("--- List Operations ---");
    benchmark_expression("car '(1 2 3)", "(car '(1 2 3))", &interp, 10000);
    benchmark_expression("cdr '(1 2 3)", "(cdr '(1 2 3))", &interp, 10000);
    benchmark_expression(
        "length '(1 2 3 4 5)",
        "(length '(1 2 3 4 5))",
        &interp,
        10000,
    );
    benchmark_expression(
        "reverse '(1 2 3 4 5)",
        "(reverse '(1 2 3 4 5))",
        &interp,
        10000,
    );

    // Lambda creation and application benchmark
    println!("--- Lambda Performance ---");
    let square_def = "(define square (lambda (x) (* x x)))";
    interp.eval_str(square_def, env).ok();
    benchmark_expression("square 5", "(square 5)", &interp, 10000);

    // Nested lambda
    let nested_def = r#"
        (define make-adder 
            (lambda (n) 
                (lambda (x) (+ x n))
            )
        )
    "#;
    interp.eval_str(nested_def, env).ok();
    interp.eval_str("(define add5 (make-adder 5))", env).ok();
    benchmark_expression("add5 10", "(add5 10)", &interp, 10000);

    // Complex expression
    println!("--- Complex Expression ---");
    let complex_expr = "(+ (* 2 3) (/ 10 2) (- 8 3))";
    benchmark_expression("Complex arithmetic", complex_expr, &interp, 10000);

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
    interp.eval_str(countdown_def, env).ok();

    println!("Testing TCO with deep recursion (should not overflow):");
    let test_sizes = [1000, 10000];
    for size in test_sizes {
        let expr = format!("(countdown {})", size);
        let start = Instant::now();
        match interp.eval_str(&expr, env) {
            Ok(result) => {
                let duration = start.elapsed();
                let result_str = bytes_to_string(&result);
                println!(
                    "  countdown({:>7}) = {:>10} | Time: {} | Arena: {} cells",
                    size,
                    result_str,
                    format_duration(duration.as_nanos()),
                    interp.arena.used()
                );
            }
            Err(e) => {
                let mut buf = [0u8; 64];
                let err_str = interp.strings.get(e, &mut buf).unwrap_or("error");
                println!("  countdown({:>7}) FAILED: {}", size, err_str);
                break;
            }
        }
    }
    println!();

    // Performance scaling analysis
    println!("--- Performance Scaling (fib) ---");
    println!(
        "{:<10} | {:<12} | {:<15} | {:<10} | {:<12}",
        "Input", "Result", "Time", "Growth", "Arena Usage"
    );
    println!("{}", "-".repeat(75));

    let mut last_duration = 0u128;
    for n in [5, 10, 15, 20] {
        let expr = format!("(fib {})", n);
        let start = Instant::now();
        match interp.eval_str(&expr, env) {
            Ok(result) => {
                let duration = start.elapsed().as_nanos();
                let result_str = bytes_to_string(&result);

                let growth = if last_duration > 0 {
                    format!("{:.2}x", duration as f64 / last_duration as f64)
                } else {
                    "-".to_string()
                };

                println!(
                    "fib({:2})    | {:>12} | {:<15} | {:<10} | {} cells",
                    n,
                    result_str,
                    format_duration(duration),
                    growth,
                    interp.arena.used()
                );

                last_duration = duration;
            }
            Err(e) => {
                let mut buf = [0u8; 64];
                let err_str = interp.strings.get(e, &mut buf).unwrap_or("error");
                println!("fib({:2}) FAILED: {}", n, err_str);
                break;
            }
        }
    }

    println!("\n--- Memory Usage Summary ---");
    println!("Arena capacity: {} cells", interp.arena.capacity());
    println!("Arena used: {} cells", interp.arena.used());
    println!("Arena available: {} cells", interp.arena.available());
    println!(
        "Usage: {:.2}%",
        (interp.arena.used() as f64 / interp.arena.capacity() as f64) * 100.0
    );

    println!("\n=== Benchmark Complete ===");
}
