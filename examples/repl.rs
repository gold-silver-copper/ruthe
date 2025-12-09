/*


//! Minimal REPL example for the Lisp interpreter
//!
//! Run with: cargo run --example repl --features=std

use ruthe::*;
use std::io::{self, Write};

fn main() -> io::Result<()> {
    println!("Simple Lisp REPL");
    println!("Type 'quit' to exit, 'env' to show environment\n");

    let env = new_env();
    let mut buffer = String::new();

    loop {
        print!("> ");
        io::stdout().flush()?;

        buffer.clear();
        io::stdin().read_line(&mut buffer)?;
        let input = buffer.trim();

        match input {
            "quit" | "exit" => break,
            "env" => {
                println!("(Environment display not implemented)");
                continue;
            }
            "" => continue,
            _ => {}
        }

        match eval_str(input, &env) {
            Ok(result) => {
                let mut buf = [0u8; 1024];
                match result.to_display_str(&mut buf) {
                    Ok(display) => println!("{}", display),
                    Err(_) => println!("<could not display result>"),
                }
            }
            Err(err) => {
                let mut buf = [0u8; 256];
                match err.to_display_str(&mut buf) {
                    Ok(err_msg) => println!("Error: {}", err_msg),
                    Err(_) => println!("Error: <unknown>"),
                }
            }
        }
    }

    println!("Goodbye!");
    Ok(())
}

*/
fn main() {}
