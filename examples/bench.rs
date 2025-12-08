#![cfg_attr(not(test), no_std)]

extern crate std;

use ruthe::{Arena, eval, init_env, parse};
use std::format;
use std::println;
use std::string::String;
use std::time::Instant;

// ===========================================================
// Error Handling Helpers
// ===========================================================

fn extract_error_message<const N: usize, const MAX_ROOTS: usize>(
    arena: &Arena<N, MAX_ROOTS>,
    error_idx: usize,
) -> String {
    let mut buffer = [0u8; 256];

    // Try to read the error as a list of characters
    let mut current = error_idx;
    let mut pos = 0;

    loop {
        match arena.get(current) {
            Ok(ruthe::LispValue::Cons(car, cdr)) => {
                if let Ok(ruthe::LispValue::Char(ch)) = arena.get(car) {
                    let mut char_buf = [0u8; 4];
                    let s = ch.encode_utf8(&mut char_buf);
                    for &b in s.as_bytes() {
                        if pos >= buffer.len() {
                            break;
                        }
                        buffer[pos] = b;
                        pos += 1;
                    }
                    current = cdr;
                } else {
                    break;
                }
            }
            Ok(ruthe::LispValue::Nil) => break,
            _ => break,
        }
    }

    if pos > 0 {
        String::from_utf8_lossy(&buffer[..pos]).into()
    } else {
        format!("Unknown error (index: {})", error_idx)
    }
}

// ===========================================================
// Timing Helpers
// ===========================================================

fn bench<F: FnOnce()>(label: &str, f: F) {
    let start = Instant::now();
    f();
    let dur = start.elapsed();
    println!("{:<40}  {:>12?}", label, dur);
}

fn bench_many<F: FnMut()>(label: &str, iters: usize, mut f: F) {
    let start = Instant::now();
    for _ in 0..iters {
        f();
    }
    let dur = start.elapsed();
    let avg = dur / iters as u32;
    println!(
        "{:<40}  {:>12?}  (avg: {:>8?}, {} iters)",
        label, dur, avg, iters
    );
}

// ===========================================================
// Lisp Execution Helpers
// ===========================================================

fn run_lisp<const N: usize, const MAX_ROOTS: usize>(
    arena: &Arena<N, MAX_ROOTS>,
    env: usize,
    src: &str,
) -> usize {
    let expr = parse(arena, src)
        .map_err(|e| {
            let msg = extract_error_message(arena, e);
            panic!("Parse failed for '{}': {}", src, msg);
        })
        .unwrap();

    eval(arena, expr, env)
        .map_err(|e| {
            let msg = extract_error_message(arena, e);
            panic!("Eval failed for '{}': {}", src, msg);
        })
        .unwrap()
}

fn run_lisp_expect_number<const N: usize, const MAX_ROOTS: usize>(
    arena: &Arena<N, MAX_ROOTS>,
    env: usize,
    src: &str,
) -> i64 {
    let result = run_lisp(arena, env, src);
    match arena.get(result) {
        Ok(ruthe::LispValue::Number(n)) => n,
        Ok(other) => panic!("Expected number result, got: {:?}", other),
        Err(e) => panic!("Failed to get result: {:?}", e),
    }
}

fn run_lisp_multi<const N: usize, const MAX_ROOTS: usize>(
    arena: &Arena<N, MAX_ROOTS>,
    env: usize,
    src: &str,
) {
    let mut current_expr = String::new();
    let mut paren_depth = 0;

    for line in src.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with(';') {
            continue;
        }

        current_expr.push_str(trimmed);
        current_expr.push(' ');

        // Count parentheses
        for ch in trimmed.chars() {
            match ch {
                '(' => paren_depth += 1,
                ')' => paren_depth -= 1,
                _ => {}
            }
        }

        // When balanced, we have a complete expression
        if paren_depth == 0 && !current_expr.trim().is_empty() {
            run_lisp(arena, env, current_expr.trim());
            current_expr.clear();
        }
    }
}

// ===========================================================
// Lisp Test Programs
// ===========================================================

const FIB_RECURSIVE: &str = r#"
(define fib
  (lambda (n)
    (if (< n 2)
        n
        (+ (fib (- n 1)) (fib (- n 2))))))
"#;

const FIB_ITERATIVE: &str = r#"
(define fib-iter
  (lambda (a b count)
    (if (= count 0)
        a
        (fib-iter b (+ a b) (- count 1)))))

(define fib
  (lambda (n)
    (fib-iter 0 1 n)))
"#;

const FACTORIAL_RECURSIVE: &str = r#"
(define fact
  (lambda (n)
    (if (= n 0)
        1
        (* n (fact (- n 1))))))
"#;

const FACTORIAL_ITERATIVE: &str = r#"
(define fact-iter
  (lambda (acc n)
    (if (= n 0)
        acc
        (fact-iter (* acc n) (- n 1)))))

(define fact
  (lambda (n)
    (fact-iter 1 n)))
"#;

const COUNTDOWN_TCO: &str = r#"
(define countdown
  (lambda (n)
    (if (= n 0)
        0
        (countdown (- n 1)))))
"#;

const COUNTUP_TCO: &str = r#"
(define countup
  (lambda (n target)
    (if (= n target)
        n
        (countup (+ n 1) target))))
"#;

const BUILD_LIST: &str = r#"
(define build-list
  (lambda (n)
    (if (= n 0)
        nil
        (cons n (build-list (- n 1))))))
"#;

const LIST_LENGTH: &str = r#"
(define length
  (lambda (lst)
    (if (= lst nil)
        0
        (+ 1 (length (cdr lst))))))
"#;

const LIST_SUM: &str = r#"
(define sum
  (lambda (lst)
    (if (= lst nil)
        0
        (+ (car lst) (sum (cdr lst))))))
"#;

const LIST_MAP_DOUBLE: &str = r#"
(define map-double
  (lambda (lst)
    (if (= lst nil)
        nil
        (cons (* 2 (car lst)) (map-double (cdr lst))))))
"#;

const ACKERMANN: &str = r#"
(define ack
  (lambda (m n)
    (if (= m 0)
        (+ n 1)
        (if (= n 0)
            (ack (- m 1) 1)
            (ack (- m 1) (ack m (- n 1)))))))
"#;

const NESTED_ARITHMETIC: &str = r#"
(define compute
  (lambda (x)
    (+ (* (- x 5) (+ x 10))
       (- (* x 2) (+ x 1)))))
"#;

// ===========================================================
// Benchmark Runners
// ===========================================================

fn bench_parser() {
    println!("\n╔═══════════════════════════════════════════════════════════════╗");
    println!("║                      PARSER BENCHMARKS                        ║");
    println!("╚═══════════════════════════════════════════════════════════════╝\n");

    const N: usize = 8192;
    const MAX_ROOTS: usize = 256;
    let arena = Arena::<N, MAX_ROOTS>::new();

    bench_many("Parse simple number", 10000, || {
        let _ = parse(&arena, "42");
    });

    bench_many("Parse simple expression", 5000, || {
        let _ = parse(&arena, "(+ 1 2)");
    });

    bench_many("Parse nested expression", 2000, || {
        let _ = parse(&arena, "(+ (* 2 3) (- 10 5))");
    });

    bench_many("Parse lambda definition", 1000, || {
        let _ = parse(&arena, "(lambda (x y) (+ x y))");
    });

    bench_many("Parse fibonacci function", 500, || {
        let _ = parse(&arena, FIB_RECURSIVE);
    });

    bench_many("Parse complex lambda", 200, || {
        let _ = parse(
            &arena,
            "(define fib-iter (lambda (a b count) (if (= count 0) a (fib-iter b (+ a b) (- count 1)))))",
        );
    });
}

fn bench_arithmetic() {
    println!("\n╔═══════════════════════════════════════════════════════════════╗");
    println!("║                    ARITHMETIC BENCHMARKS                      ║");
    println!("╚═══════════════════════════════════════════════════════════════╝\n");

    const N: usize = 16384;
    const MAX_ROOTS: usize = 256;
    let arena = Arena::<N, MAX_ROOTS>::new();
    let env = init_env(&arena).expect("init env failed");

    bench_many("Eval simple addition", 1000, || {
        run_lisp(&arena, env, "(+ 1 2)");
        if arena.used() > N / 2 {
            arena.collect(&[env]);
        }
    });

    bench_many("Eval nested arithmetic", 500, || {
        run_lisp(&arena, env, "(+ (* 2 3) (- 10 5))");
        if arena.used() > N / 2 {
            arena.collect(&[env]);
        }
    });

    bench_many("Eval complex expression", 300, || {
        run_lisp(&arena, env, "(* (+ 1 2) (- (+ 4 5) (* 2 3)))");
        if arena.used() > N / 2 {
            arena.collect(&[env]);
        }
    });

    run_lisp_multi(&arena, env, NESTED_ARITHMETIC);
    bench_many("Eval arithmetic function call", 500, || {
        run_lisp(&arena, env, "(compute 20)");
        if arena.used() > N / 2 {
            arena.collect(&[env]);
        }
    });
}

fn bench_fibonacci() {
    println!("\n╔═══════════════════════════════════════════════════════════════╗");
    println!("║                    FIBONACCI BENCHMARKS                       ║");
    println!("╚═══════════════════════════════════════════════════════════════╝\n");

    const N: usize = 62700;
    const MAX_ROOTS: usize = 10000;
    let arena = Arena::<N, MAX_ROOTS>::new();
    let env = init_env(&arena).expect("init env failed");

    // Recursive Fibonacci
    println!("--- Recursive Fibonacci ---");
    run_lisp_multi(&arena, env, FIB_RECURSIVE);

    bench("fib(10) recursive", || {
        let result = run_lisp_expect_number(&arena, env, "(fib 10)");
        assert_eq!(result, 55);
    });

    bench("fib(15) recursive", || {
        let result = run_lisp_expect_number(&arena, env, "(fib 15)");
        assert_eq!(result, 610);
    });

    // Iterative Fibonacci
    println!("\n--- Iterative Fibonacci (TCO) ---");
    let arena2 = Arena::<N, MAX_ROOTS>::new();
    let env2 = init_env(&arena2).expect("init env failed");
    run_lisp_multi(&arena2, env2, FIB_ITERATIVE);

    bench("fib(10) iterative", || {
        let result = run_lisp_expect_number(&arena2, env2, "(fib 10)");
        // assert_eq!(result, -980107325); // Overflow wrapping
    });

    bench("fib(20) iterative", || {
        run_lisp(&arena2, env2, "(fib 20)");
    });

    bench("fib(30) iterative", || {
        run_lisp(&arena2, env2, "(fib 30)");
    });
    bench("fib(40) iterative", || {
        run_lisp(&arena2, env2, "(fib 40)");
    });
}

fn bench_factorial() {
    println!("\n╔═══════════════════════════════════════════════════════════════╗");
    println!("║                    FACTORIAL BENCHMARKS                       ║");
    println!("╚═══════════════════════════════════════════════════════════════╝\n");

    const N: usize = 8192;
    const MAX_ROOTS: usize = 256;
    let arena = Arena::<N, MAX_ROOTS>::new();
    let env = init_env(&arena).expect("init env failed");

    // Recursive Factorial
    println!("--- Recursive Factorial ---");
    run_lisp_multi(&arena, env, FACTORIAL_RECURSIVE);

    bench("fact(5) recursive", || {
        let result = run_lisp_expect_number(&arena, env, "(fact 5)");
        assert_eq!(result, 120);
    });

    bench("fact(10) recursive", || {
        let result = run_lisp_expect_number(&arena, env, "(fact 10)");
        assert_eq!(result, 3628800);
    });

    bench("fact(12) recursive", || {
        run_lisp(&arena, env, "(fact 12)");
    });

    // Iterative Factorial
    println!("\n--- Iterative Factorial (TCO) ---");
    let arena2 = Arena::<N, MAX_ROOTS>::new();
    let env2 = init_env(&arena2).expect("init env failed");
    run_lisp_multi(&arena2, env2, FACTORIAL_ITERATIVE);

    /*



        bench("fact(100) iterative", || {
            run_lisp(&arena2, env2, "(fact 100)");
        });

        bench("fact(500) iterative", || {
            run_lisp(&arena2, env2, "(fact 500)");
        });

        bench("fact(1000) iterative", || {
            run_lisp(&arena2, env2, "(fact 1000)");
        });

    */
}

fn bench_tail_call_optimization() {
    println!("\n╔═══════════════════════════════════════════════════════════════╗");
    println!("║              TAIL CALL OPTIMIZATION BENCHMARKS                ║");
    println!("╚═══════════════════════════════════════════════════════════════╝\n");

    const N: usize = 72768;
    const MAX_ROOTS: usize = 2560;
    let arena = Arena::<N, MAX_ROOTS>::new();
    let env = init_env(&arena).expect("init env failed");

    // Countdown test
    run_lisp_multi(&arena, env, COUNTDOWN_TCO);

    bench("countdown(1000)", || {
        let result = run_lisp_expect_number(&arena, env, "(countdown 1000)");
        assert_eq!(result, 0);
    });

    arena.collect(&[env]);

    bench("countdown(5000)", || {
        let result = run_lisp_expect_number(&arena, env, "(countdown 5000)");
        assert_eq!(result, 0);
    });

    arena.collect(&[env]);

    bench("countdown(10000)", || {
        let result = run_lisp_expect_number(&arena, env, "(countdown 10000)");
        assert_eq!(result, 0);
    });

    // Countup test
    run_lisp_multi(&arena, env, COUNTUP_TCO);

    arena.collect(&[env]);

    bench("countup(0, 5000)", || {
        let result = run_lisp_expect_number(&arena, env, "(countup 0 5000)");
        assert_eq!(result, 5000);
    });

    arena.collect(&[env]);

    bench("countup(0, 10000)", || {
        let result = run_lisp_expect_number(&arena, env, "(countup 0 10000)");
        assert_eq!(result, 10000);
    });
}

fn bench_list_operations() {
    println!("\n╔═══════════════════════════════════════════════════════════════╗");
    println!("║                  LIST OPERATION BENCHMARKS                    ║");
    println!("╚═══════════════════════════════════════════════════════════════╝\n");

    const N: usize = 65536;
    const MAX_ROOTS: usize = 256;
    let arena = Arena::<N, MAX_ROOTS>::new();
    let env = init_env(&arena).expect("init env failed");

    // Build list
    run_lisp_multi(&arena, env, BUILD_LIST);

    bench("build-list(50)", || {
        run_lisp(&arena, env, "(build-list 50)");
    });

    arena.collect(&[env]);

    bench("build-list(100)", || {
        run_lisp(&arena, env, "(build-list 100)");
    });

    arena.collect(&[env]);

    bench("build-list(200)", || {
        run_lisp(&arena, env, "(build-list 200)");
    });

    arena.collect(&[env]);

    // List length
    run_lisp_multi(&arena, env, LIST_LENGTH);

    bench("length of 50-element list", || {
        run_lisp(&arena, env, "(length (build-list 50))");
    });

    arena.collect(&[env]);

    // List sum
    run_lisp_multi(&arena, env, LIST_SUM);

    bench("sum of 50-element list", || {
        let result = run_lisp_expect_number(&arena, env, "(sum (build-list 50))");
        assert_eq!(result, 1275);
    });

    arena.collect(&[env]);

    bench("sum of 100-element list", || {
        let result = run_lisp_expect_number(&arena, env, "(sum (build-list 100))");
        assert_eq!(result, 5050);
    });

    arena.collect(&[env]);

    // Map double
    run_lisp_multi(&arena, env, LIST_MAP_DOUBLE);

    bench("map-double on 30-element list", || {
        run_lisp(&arena, env, "(map-double (build-list 30))");
    });

    arena.collect(&[env]);

    bench("map-double on 50-element list", || {
        run_lisp(&arena, env, "(map-double (build-list 50))");
    });
}

fn bench_garbage_collection() {
    println!("\n╔═══════════════════════════════════════════════════════════════╗");
    println!("║               GARBAGE COLLECTION BENCHMARKS                   ║");
    println!("╚═══════════════════════════════════════════════════════════════╝\n");

    const N: usize = 32768;
    const MAX_ROOTS: usize = 256;
    let arena = Arena::<N, MAX_ROOTS>::new();
    let env = init_env(&arena).expect("init env failed");

    run_lisp_multi(&arena, env, BUILD_LIST);

    println!("Initial arena usage: {} cells", arena.used());

    // Create some garbage
    for i in 1..=10 {
        run_lisp(&arena, env, "(build-list 50)");
        if i % 5 == 0 {
            println!("After {} iterations: {} cells used", i, arena.used());
        }
    }

    bench("GC with 1 root", || {
        arena.collect(&[env]);
    });

    println!("After GC: {} cells used", arena.used());

    // Build up memory again
    let roots: std::vec::Vec<usize> = (0..5)
        .map(|_| run_lisp(&arena, env, "(build-list 30)"))
        .collect();

    println!("Before GC with multiple roots: {} cells used", arena.used());

    bench("GC with 6 roots", || {
        let mut all_roots = std::vec![env];
        all_roots.extend_from_slice(&roots);
        arena.collect(&all_roots);
    });

    println!("After GC with roots: {} cells used", arena.used());

    // Stress test: repeated allocations and GC
    bench_many("Repeated alloc + GC cycles", 20, || {
        run_lisp(&arena, env, "(build-list 50)");
        arena.collect(&[env]);
    });

    println!("Final arena usage: {} cells used", arena.used());

    // Memory pressure test
    println!("\n--- Memory Pressure Test ---");
    let arena2 = Arena::<16384, MAX_ROOTS>::new();
    let env2 = init_env(&arena2).expect("init env failed");
    run_lisp_multi(&arena2, env2, BUILD_LIST);

    bench("Build list with small arena + GC", || {
        run_lisp(&arena2, env2, "(build-list 100)");
        arena2.collect(&[env2]);
    });
}

fn bench_ackermann() {
    println!("\n╔═══════════════════════════════════════════════════════════════╗");
    println!("║                   ACKERMANN BENCHMARKS                        ║");
    println!("╚═══════════════════════════════════════════════════════════════╝\n");

    const N: usize = 16384;
    const MAX_ROOTS: usize = 256;
    let arena = Arena::<N, MAX_ROOTS>::new();
    let env = init_env(&arena).expect("init env failed");

    run_lisp_multi(&arena, env, ACKERMANN);

    bench("ack(1, 5)", || {
        let result = run_lisp_expect_number(&arena, env, "(ack 1 5)");
        assert_eq!(result, 7);
    });

    bench("ack(2, 5)", || {
        let result = run_lisp_expect_number(&arena, env, "(ack 2 5)");
        assert_eq!(result, 13);
    });

    bench("ack(3, 4)", || {
        let result = run_lisp_expect_number(&arena, env, "(ack 3 4)");
        assert_eq!(result, 125);
    });

    bench("ack(3, 5)", || {
        let result = run_lisp_expect_number(&arena, env, "(ack 3 5)");
        assert_eq!(result, 253);
    });
}

fn bench_lambda_and_closures() {
    println!("\n╔═══════════════════════════════════════════════════════════════╗");
    println!("║              LAMBDA AND CLOSURE BENCHMARKS                    ║");
    println!("╚═══════════════════════════════════════════════════════════════╝\n");

    const N: usize = 16384;
    const MAX_ROOTS: usize = 256;
    let arena = Arena::<N, MAX_ROOTS>::new();
    let env = init_env(&arena).expect("init env failed");

    bench_many("Create simple lambda", 500, || {
        run_lisp(&arena, env, "(lambda (x) (* x 2))");
        if arena.used() > N / 2 {
            arena.collect(&[env]);
        }
    });

    bench_many("Define and call lambda", 200, || {
        run_lisp(&arena, env, "(define double (lambda (x) (* x 2)))");
        run_lisp(&arena, env, "(double 21)");
        if arena.used() > N / 2 {
            arena.collect(&[env]);
        }
    });

    // Higher-order function
    let apply_twice = r#"
(define apply-twice
  (lambda (f x)
    (f (f x))))
"#;

    run_lisp_multi(&arena, env, apply_twice);
    run_lisp(&arena, env, "(define inc (lambda (x) (+ x 1)))");

    bench_many("Higher-order function call", 200, || {
        let result = run_lisp_expect_number(&arena, env, "(apply-twice inc 5)");
        assert_eq!(result, 7);
        if arena.used() > N / 2 {
            arena.collect(&[env]);
        }
    });

    arena.collect(&[env]);

    // Nested lambdas
    let make_adder = r#"
(define make-adder
  (lambda (x)
    (lambda (y) (+ x y))))
"#;

    run_lisp_multi(&arena, env, make_adder);

    bench_many("Closure creation and use", 150, || {
        run_lisp(&arena, env, "(define add5 (make-adder 5))");
        let result = run_lisp_expect_number(&arena, env, "(add5 10)");
        assert_eq!(result, 15);
        if arena.used() > N / 2 {
            arena.collect(&[env]);
        }
    });
}

fn bench_conditional_expressions() {
    println!("\n╔═══════════════════════════════════════════════════════════════╗");
    println!("║            CONDITIONAL EXPRESSION BENCHMARKS                  ║");
    println!("╚═══════════════════════════════════════════════════════════════╝\n");

    const N: usize = 16384;
    const MAX_ROOTS: usize = 256;
    let arena = Arena::<N, MAX_ROOTS>::new();
    let env = init_env(&arena).expect("init env failed");

    bench_many("Simple if-true", 1000, || {
        run_lisp(&arena, env, "(if #t 1 2)");
        if arena.used() > N / 2 {
            arena.collect(&[env]);
        }
    });

    bench_many("Simple if-false", 1000, || {
        run_lisp(&arena, env, "(if #f 1 2)");
        if arena.used() > N / 2 {
            arena.collect(&[env]);
        }
    });

    bench_many("Nested if expressions", 500, || {
        run_lisp(&arena, env, "(if (< 3 5) (if #t 10 20) 30)");
        if arena.used() > N / 2 {
            arena.collect(&[env]);
        }
    });

    let max_fn = r#"
(define max
  (lambda (a b)
    (if (< a b) b a)))
"#;

    run_lisp_multi(&arena, env, max_fn);

    bench_many("Conditional function call", 500, || {
        let result = run_lisp_expect_number(&arena, env, "(max 10 20)");
        assert_eq!(result, 20);
        if arena.used() > N / 2 {
            arena.collect(&[env]);
        }
    });
}

fn bench_mixed_workload() {
    println!("\n╔═══════════════════════════════════════════════════════════════╗");
    println!("║                  MIXED WORKLOAD BENCHMARKS                    ║");
    println!("╚═══════════════════════════════════════════════════════════════╝\n");

    const N: usize = 32768;
    const MAX_ROOTS: usize = 256;
    let arena = Arena::<N, MAX_ROOTS>::new();
    let env = init_env(&arena).expect("init env failed");

    // Complex program combining multiple features
    let complex_program = r#"
(define range-sum
  (lambda (start end)
    (if (< end start)
        0
        (+ start (range-sum (+ start 1) end)))))

(define is-even
  (lambda (n)
    (= 0 (- n (* 2 (- n (- n n)))))))

(define process
  (lambda (n)
    (if (is-even n)
        (* n 2)
        (+ n 1))))
"#;

    run_lisp_multi(&arena, env, complex_program);

    bench("range-sum(1, 100)", || {
        let result = run_lisp_expect_number(&arena, env, "(range-sum 1 100)");
        assert_eq!(result, 5050);
    });

    bench("process even number", || {
        let result = run_lisp_expect_number(&arena, env, "(process 10)");
        assert_eq!(result, 20);
    });

    bench("Mixed operations with GC", || {
        run_lisp(&arena, env, "(range-sum 1 50)");
        run_lisp(&arena, env, "(process 25)");
        arena.collect(&[env]);
    });
}

// ===========================================================
// Main Benchmark Runner
// ===========================================================

pub fn run_all_benchmarks() {
    println!("\n");
    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║                                                               ║");
    println!("║          LISP INTERPRETER COMPREHENSIVE BENCHMARK SUITE       ║");
    println!("║                                                               ║");
    println!("╚═══════════════════════════════════════════════════════════════╝");

    bench_parser();
    bench_arithmetic();
    bench_fibonacci();
    bench_factorial();
    bench_tail_call_optimization();
    bench_list_operations();
    bench_garbage_collection();
    bench_ackermann();
    bench_lambda_and_closures();
    bench_conditional_expressions();
    bench_mixed_workload();

    println!("\n");
    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║                    BENCHMARKS COMPLETE                        ║");
    println!("╚═══════════════════════════════════════════════════════════════╝");
    println!("\n");
}

fn main() {
    run_all_benchmarks();
}
