use std::collections::{HashMap, LinkedList};
use std::fmt;
use std::io::{self, Write};
use std::rc::Rc;

// ============================================================================
// Value Trait - Core abstraction for all Lisp values
// ============================================================================

pub trait Value: fmt::Debug {
    fn type_name(&self) -> &'static str;
    fn as_bool(&self) -> Option<bool> {
        None
    }
    fn as_number(&self) -> Option<f64> {
        None
    }
    fn as_number_exact(&self) -> Option<&Number> {
        None
    }
    fn as_symbol(&self) -> Option<&str> {
        None
    }
    fn as_cons(&self) -> Option<ConsRef> {
        None
    }
    fn is_nil(&self) -> bool {
        false
    }

    // For display purposes
    fn to_string(&self) -> String {
        format!("{:?}", self)
    }
}

pub type ValRef = Rc<dyn Value>;

// ============================================================================
// Cons Cell - Proper Lisp cons using LinkedList
// ============================================================================

#[derive(Debug, Clone)]
pub struct Cons {
    // Using LinkedList internally for true linked structure
    list: LinkedList<ValRef>,
}

pub type ConsRef = Rc<Cons>;

impl Cons {
    pub fn new() -> Self {
        Self {
            list: LinkedList::new(),
        }
    }

    pub fn from_vec(items: Vec<ValRef>) -> Self {
        let mut list = LinkedList::new();
        for item in items {
            list.push_back(item);
        }
        Self { list }
    }

    pub fn from_list(list: LinkedList<ValRef>) -> Self {
        Self { list }
    }

    pub fn is_empty(&self) -> bool {
        self.list.is_empty()
    }

    pub fn len(&self) -> usize {
        self.list.len()
    }

    pub fn car(&self) -> Option<ValRef> {
        self.list.front().map(|v| Rc::clone(v))
    }

    pub fn cdr(&self) -> ConsRef {
        let mut new_list = self.list.clone();
        new_list.pop_front();
        Rc::new(Cons::from_list(new_list))
    }

    pub fn cons(val: ValRef, rest: ConsRef) -> ConsRef {
        let mut new_list = LinkedList::new();
        new_list.push_back(val);
        new_list.append(&mut rest.list.clone());
        Rc::new(Cons::from_list(new_list))
    }

    pub fn iter(&self) -> impl Iterator<Item = &ValRef> {
        self.list.iter()
    }

    pub fn to_vec(&self) -> Vec<ValRef> {
        self.list.iter().map(|v| Rc::clone(v)).collect()
    }

    // Append a value to the end (useful for building lists)
    pub fn append(&self, val: ValRef) -> ConsRef {
        let mut new_list = self.list.clone();
        new_list.push_back(val);
        Rc::new(Cons::from_list(new_list))
    }
}

impl Value for Cons {
    fn type_name(&self) -> &'static str {
        "cons"
    }
    fn as_cons(&self) -> Option<ConsRef> {
        Some(Rc::new(self.clone()))
    }
    fn to_string(&self) -> String {
        if self.is_empty() {
            return "()".to_string();
        }
        let items: Vec<String> = self.iter().map(|v| v.to_string()).collect();
        format!("({})", items.join(" "))
    }
}

// Implement Value for Rc<Cons> to enable direct usage as ValRef
impl Value for Rc<Cons> {
    fn type_name(&self) -> &'static str {
        "cons"
    }
    fn as_cons(&self) -> Option<ConsRef> {
        Some(Rc::clone(self))
    }
    fn to_string(&self) -> String {
        if self.is_empty() {
            return "()".to_string();
        }
        let items: Vec<String> = self.iter().map(|v| v.to_string()).collect();
        format!("({})", items.join(" "))
    }
}

// ============================================================================
// Number Type - Exact arithmetic with integers and rationals
// ============================================================================

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Number {
    Integer(i64),
    Rational(i64, i64), // numerator, denominator (always in reduced form)
}

impl Number {
    pub fn integer(n: i64) -> Self {
        Number::Integer(n)
    }

    pub fn rational(num: i64, den: i64) -> Self {
        if den == 0 {
            panic!("Division by zero in rational");
        }
        if num == 0 {
            return Number::Integer(0);
        }

        let gcd = Self::gcd(num.abs(), den.abs());
        let num = num / gcd;
        let den = den / gcd;

        // Keep denominator positive
        let (num, den) = if den < 0 { (-num, -den) } else { (num, den) };

        if den == 1 {
            Number::Integer(num)
        } else {
            Number::Rational(num, den)
        }
    }

    fn gcd(mut a: i64, mut b: i64) -> i64 {
        while b != 0 {
            let t = b;
            b = a % b;
            a = t;
        }
        a
    }

    pub fn to_f64(&self) -> f64 {
        match self {
            Number::Integer(n) => *n as f64,
            Number::Rational(num, den) => (*num as f64) / (*den as f64),
        }
    }

    pub fn add(&self, other: &Number) -> Number {
        match (self, other) {
            (Number::Integer(a), Number::Integer(b)) => Number::Integer(a + b),
            (Number::Integer(a), Number::Rational(num, den))
            | (Number::Rational(num, den), Number::Integer(a)) => {
                Number::rational(a * den + num, *den)
            }
            (Number::Rational(n1, d1), Number::Rational(n2, d2)) => {
                Number::rational(n1 * d2 + n2 * d1, d1 * d2)
            }
        }
    }

    pub fn sub(&self, other: &Number) -> Number {
        match (self, other) {
            (Number::Integer(a), Number::Integer(b)) => Number::Integer(a - b),
            (Number::Integer(a), Number::Rational(num, den)) => {
                Number::rational(a * den - num, *den)
            }
            (Number::Rational(num, den), Number::Integer(a)) => {
                Number::rational(num - a * den, *den)
            }
            (Number::Rational(n1, d1), Number::Rational(n2, d2)) => {
                Number::rational(n1 * d2 - n2 * d1, d1 * d2)
            }
        }
    }

    pub fn mul(&self, other: &Number) -> Number {
        match (self, other) {
            (Number::Integer(a), Number::Integer(b)) => Number::Integer(a * b),
            (Number::Integer(a), Number::Rational(num, den))
            | (Number::Rational(num, den), Number::Integer(a)) => Number::rational(a * num, *den),
            (Number::Rational(n1, d1), Number::Rational(n2, d2)) => {
                Number::rational(n1 * n2, d1 * d2)
            }
        }
    }

    pub fn div(&self, other: &Number) -> Result<Number, String> {
        match (self, other) {
            (_, Number::Integer(0)) => Err("Division by zero".to_string()),
            (Number::Integer(a), Number::Integer(b)) => {
                if a % b == 0 {
                    Ok(Number::Integer(a / b))
                } else {
                    Ok(Number::rational(*a, *b))
                }
            }
            (Number::Integer(a), Number::Rational(num, den)) => Ok(Number::rational(a * den, *num)),
            (Number::Rational(num, den), Number::Integer(b)) => Ok(Number::rational(*num, den * b)),
            (Number::Rational(n1, d1), Number::Rational(n2, d2)) => {
                Ok(Number::rational(n1 * d2, d1 * n2))
            }
        }
    }

    pub fn neg(&self) -> Number {
        match self {
            Number::Integer(n) => Number::Integer(-n),
            Number::Rational(num, den) => Number::Rational(-num, *den),
        }
    }

    pub fn cmp(&self, other: &Number) -> std::cmp::Ordering {
        match (self, other) {
            (Number::Integer(a), Number::Integer(b)) => a.cmp(b),
            (Number::Integer(a), Number::Rational(num, den)) => (a * den).cmp(num),
            (Number::Rational(num, den), Number::Integer(b)) => num.cmp(&(b * den)),
            (Number::Rational(n1, d1), Number::Rational(n2, d2)) => (n1 * d2).cmp(&(n2 * d1)),
        }
    }
}

impl Value for Number {
    fn type_name(&self) -> &'static str {
        "number"
    }
    fn as_number(&self) -> Option<f64> {
        Some(self.to_f64())
    }
    fn as_number_exact(&self) -> Option<&Number> {
        Some(self)
    }
    fn to_string(&self) -> String {
        match self {
            Number::Integer(n) => format!("{}", n),
            Number::Rational(num, den) => format!("{}/{}", num, den),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Symbol(pub String);

impl Value for Symbol {
    fn type_name(&self) -> &'static str {
        "symbol"
    }
    fn as_symbol(&self) -> Option<&str> {
        Some(&self.0)
    }
    fn to_string(&self) -> String {
        self.0.clone()
    }
}

#[derive(Debug, Clone)]
pub struct Bool(pub bool);

impl Value for Bool {
    fn type_name(&self) -> &'static str {
        "bool"
    }
    fn as_bool(&self) -> Option<bool> {
        Some(self.0)
    }
    fn to_string(&self) -> String {
        if self.0 {
            "#t".to_string()
        } else {
            "#f".to_string()
        }
    }
}

#[derive(Debug)]
pub struct Nil;

impl Value for Nil {
    fn type_name(&self) -> &'static str {
        "nil"
    }
    fn is_nil(&self) -> bool {
        true
    }
    fn to_string(&self) -> String {
        "nil".to_string()
    }
}

// ============================================================================
// Constructors
// ============================================================================

pub fn val_number(n: i64) -> ValRef {
    Rc::new(Number::integer(n))
}
pub fn val_rational(num: i64, den: i64) -> ValRef {
    Rc::new(Number::rational(num, den))
}
pub fn val_number_from_num(n: Number) -> ValRef {
    Rc::new(n)
}
pub fn val_symbol(s: &str) -> ValRef {
    Rc::new(Symbol(s.to_string()))
}
pub fn val_bool(b: bool) -> ValRef {
    Rc::new(Bool(b))
}
pub fn val_cons(items: Vec<ValRef>) -> ValRef {
    Rc::new(Cons::from_vec(items))
}
pub fn val_cons_from_list(list: LinkedList<ValRef>) -> ValRef {
    Rc::new(Cons::from_list(list))
}
pub fn val_nil() -> ValRef {
    Rc::new(Nil)
}
pub fn empty_cons() -> ConsRef {
    Rc::new(Cons::new())
}

// ============================================================================
// Environment - Variable bindings
// ============================================================================

pub struct Env {
    vars: HashMap<String, ValRef>,
}

impl Env {
    pub fn new() -> Self {
        let mut env = Self {
            vars: HashMap::new(),
        };
        env.register_builtins();
        env
    }

    pub fn set(&mut self, name: String, v: ValRef) {
        self.vars.insert(name, v);
    }

    pub fn get(&self, name: &str) -> Option<&ValRef> {
        self.vars.get(name)
    }

    fn register_builtins(&mut self) {
        // Register nil as a global constant
        self.set("nil".to_string(), val_nil());
    }
}

// ============================================================================
// Tokenizer
// ============================================================================

#[derive(Debug, Clone, PartialEq)]
enum Token {
    LParen,
    RParen,
    Symbol(String),
    Number(i64),
    Rational(i64, i64),
    Bool(bool),
    Quote,
}

fn tokenize(input: &str) -> Result<Vec<Token>, String> {
    let mut tokens = Vec::new();
    let mut chars = input.chars().peekable();

    while let Some(&ch) = chars.peek() {
        match ch {
            ' ' | '\t' | '\n' | '\r' => {
                chars.next();
            }
            '(' => {
                tokens.push(Token::LParen);
                chars.next();
            }
            ')' => {
                tokens.push(Token::RParen);
                chars.next();
            }
            '\'' => {
                tokens.push(Token::Quote);
                chars.next();
            }
            ';' => {
                // Comment - skip to end of line
                while let Some(&c) = chars.peek() {
                    chars.next();
                    if c == '\n' {
                        break;
                    }
                }
            }
            '#' => {
                chars.next();
                match chars.peek() {
                    Some(&'t') => {
                        tokens.push(Token::Bool(true));
                        chars.next();
                    }
                    Some(&'f') => {
                        tokens.push(Token::Bool(false));
                        chars.next();
                    }
                    _ => return Err("Invalid boolean literal".to_string()),
                }
            }
            _ => {
                let mut atom = String::new();
                while let Some(&c) = chars.peek() {
                    if c.is_whitespace() || c == '(' || c == ')' || c == '\'' {
                        break;
                    }
                    atom.push(c);
                    chars.next();
                }

                // Try to parse as rational (e.g., 3/4)
                if atom.contains('/') {
                    let parts: Vec<&str> = atom.split('/').collect();
                    if parts.len() == 2 {
                        if let (Ok(num), Ok(den)) =
                            (parts[0].parse::<i64>(), parts[1].parse::<i64>())
                        {
                            tokens.push(Token::Rational(num, den));
                            continue;
                        }
                    }
                }

                // Try to parse as integer
                if let Ok(num) = atom.parse::<i64>() {
                    tokens.push(Token::Number(num));
                } else {
                    tokens.push(Token::Symbol(atom));
                }
            }
        }
    }

    Ok(tokens)
}

// ============================================================================
// Parser
// ============================================================================

fn parse_tokens(tokens: &[Token]) -> Result<(ValRef, usize), String> {
    if tokens.is_empty() {
        return Err("Unexpected end of input".to_string());
    }

    match &tokens[0] {
        Token::Number(n) => Ok((val_number(*n), 1)),
        Token::Rational(num, den) => Ok((val_rational(*num, *den), 1)),
        Token::Bool(b) => Ok((val_bool(*b), 1)),
        Token::Symbol(s) => Ok((val_symbol(s), 1)),
        Token::Quote => {
            if tokens.len() < 2 {
                return Err("Quote requires an expression".to_string());
            }
            let (val, consumed) = parse_tokens(&tokens[1..])?;
            let quoted = val_cons(vec![val_symbol("quote"), val]);
            Ok((quoted, consumed + 1))
        }
        Token::LParen => {
            let mut items = Vec::new();
            let mut pos = 1;

            while pos < tokens.len() {
                if tokens[pos] == Token::RParen {
                    return Ok((val_cons(items), pos + 1));
                }
                let (val, consumed) = parse_tokens(&tokens[pos..])?;
                items.push(val);
                pos += consumed;
            }

            Err("Unmatched '('".to_string())
        }
        Token::RParen => Err("Unexpected ')'".to_string()),
    }
}

pub fn parse(input: &str) -> Result<ValRef, String> {
    let tokens = tokenize(input)?;
    if tokens.is_empty() {
        return Err("Empty input".to_string());
    }
    let (val, consumed) = parse_tokens(&tokens)?;
    if consumed < tokens.len() {
        return Err("Unexpected tokens after expression".to_string());
    }
    Ok(val)
}

// ============================================================================
// Evaluator
// ============================================================================

pub fn eval(expr: ValRef, env: &mut Env) -> Result<ValRef, String> {
    // Self-evaluating values
    if expr.as_number().is_some() || expr.as_bool().is_some() || expr.is_nil() {
        return Ok(expr);
    }

    // Symbol lookup
    if let Some(sym) = expr.as_symbol() {
        // Special case for nil
        if sym == "nil" {
            return Ok(val_nil());
        }
        return env
            .get(sym)
            .map(|v| Rc::clone(v))
            .ok_or_else(|| format!("Unbound symbol: {}", sym));
    }

    // Cons cell evaluation (function application)
    if let Some(cons) = expr.as_cons() {
        if cons.is_empty() {
            return Ok(val_nil());
        }

        let first = cons.car().ok_or("Empty cons in eval")?;

        // Special forms
        if let Some(sym) = first.as_symbol() {
            match sym {
                "define" => {
                    if cons.len() != 3 {
                        return Err("define requires 2 arguments".to_string());
                    }
                    let items = cons.to_vec();
                    let name = items[1]
                        .as_symbol()
                        .ok_or("define requires symbol as first arg")?;
                    let val = eval(Rc::clone(&items[2]), env)?;
                    env.set(name.to_string(), Rc::clone(&val));
                    return Ok(val);
                }
                "if" => {
                    if cons.len() != 4 {
                        return Err("if requires 3 arguments".to_string());
                    }
                    let items = cons.to_vec();
                    let cond = eval(Rc::clone(&items[1]), env)?;
                    let is_true = match cond.as_bool() {
                        Some(b) => b,
                        None => !cond.is_nil(), // nil is false, everything else is true
                    };
                    return eval(Rc::clone(&items[if is_true { 2 } else { 3 }]), env);
                }
                "quote" => {
                    if cons.len() != 2 {
                        return Err("quote requires 1 argument".to_string());
                    }
                    let items = cons.to_vec();
                    return Ok(Rc::clone(&items[1]));
                }
                "lambda" => {
                    // For future implementation
                    return Err("lambda not yet implemented".to_string());
                }
                _ => {}
            }
        }

        // Function application
        let func_name = first.as_symbol().ok_or("First element must be a symbol")?;

        // Evaluate all arguments
        let rest = cons.cdr();
        let args: Result<Vec<ValRef>, String> =
            rest.iter().map(|arg| eval(Rc::clone(arg), env)).collect();
        let args = args?;

        return apply_builtin(func_name, &args);
    }

    Err(format!("Cannot evaluate: {:?}", expr))
}

// ============================================================================
// Built-in Functions
// ============================================================================

fn apply_builtin(name: &str, args: &[ValRef]) -> Result<ValRef, String> {
    match name {
        "+" => {
            let mut result = Number::integer(0);
            for arg in args {
                let num = arg.as_number_exact().ok_or("+ requires numbers")?;
                result = result.add(num);
            }
            Ok(val_number_from_num(result))
        }
        "-" => {
            if args.is_empty() {
                return Err("- requires at least 1 argument".to_string());
            }
            let first = args[0].as_number_exact().ok_or("- requires numbers")?;
            if args.len() == 1 {
                return Ok(val_number_from_num(first.neg()));
            }
            let mut result = first.clone();
            for arg in &args[1..] {
                let num = arg.as_number_exact().ok_or("- requires numbers")?;
                result = result.sub(num);
            }
            Ok(val_number_from_num(result))
        }
        "*" => {
            let mut result = Number::integer(1);
            for arg in args {
                let num = arg.as_number_exact().ok_or("* requires numbers")?;
                result = result.mul(num);
            }
            Ok(val_number_from_num(result))
        }
        "/" => {
            if args.len() < 2 {
                return Err("/ requires at least 2 arguments".to_string());
            }
            let first = args[0].as_number_exact().ok_or("/ requires numbers")?;
            let mut result = first.clone();
            for arg in &args[1..] {
                let num = arg.as_number_exact().ok_or("/ requires numbers")?;
                result = result.div(num)?;
            }
            Ok(val_number_from_num(result))
        }
        "=" => {
            if args.len() != 2 {
                return Err("= requires 2 arguments".to_string());
            }
            let a = args[0].as_number_exact().ok_or("= requires numbers")?;
            let b = args[1].as_number_exact().ok_or("= requires numbers")?;
            Ok(val_bool(a.cmp(b) == std::cmp::Ordering::Equal))
        }
        "<" => {
            if args.len() != 2 {
                return Err("< requires 2 arguments".to_string());
            }
            let a = args[0].as_number_exact().ok_or("< requires numbers")?;
            let b = args[1].as_number_exact().ok_or("< requires numbers")?;
            Ok(val_bool(a.cmp(b) == std::cmp::Ordering::Less))
        }
        ">" => {
            if args.len() != 2 {
                return Err("> requires 2 arguments".to_string());
            }
            let a = args[0].as_number_exact().ok_or("> requires numbers")?;
            let b = args[1].as_number_exact().ok_or("> requires numbers")?;
            Ok(val_bool(a.cmp(b) == std::cmp::Ordering::Greater))
        }
        "list" => Ok(val_cons(args.to_vec())),
        "car" => {
            if args.len() != 1 {
                return Err("car requires 1 argument".to_string());
            }
            let cons = args[0].as_cons().ok_or("car requires a cons/list")?;
            cons.car().ok_or("car of empty list".to_string())
        }
        "cdr" => {
            if args.len() != 1 {
                return Err("cdr requires 1 argument".to_string());
            }
            let cons = args[0].as_cons().ok_or("cdr requires a cons/list")?;
            let cdr_cons = cons.cdr();
            Ok(cdr_cons as ValRef)
        }
        "cons" => {
            if args.len() != 2 {
                return Err("cons requires 2 arguments".to_string());
            }
            if let Some(rest_cons) = args[1].as_cons() {
                let new_cons = Cons::cons(Rc::clone(&args[0]), rest_cons);
                Ok(new_cons as ValRef)
            } else if args[1].is_nil() {
                // cons with nil creates a single-element list
                Ok(val_cons(vec![Rc::clone(&args[0])]))
            } else {
                // Improper list (dotted pair) - for now just create a 2-element list
                Ok(val_cons(vec![Rc::clone(&args[0]), Rc::clone(&args[1])]))
            }
        }
        "null?" | "nil?" => {
            if args.len() != 1 {
                return Err("null? requires 1 argument".to_string());
            }
            let is_nil = if let Some(cons) = args[0].as_cons() {
                cons.is_empty()
            } else {
                args[0].is_nil()
            };
            Ok(val_bool(is_nil))
        }
        "cons?" | "pair?" => {
            if args.len() != 1 {
                return Err("cons? requires 1 argument".to_string());
            }
            Ok(val_bool(args[0].as_cons().is_some()))
        }
        "length" => {
            if args.len() != 1 {
                return Err("length requires 1 argument".to_string());
            }
            let cons = args[0].as_cons().ok_or("length requires a list")?;
            Ok(val_number(cons.len() as i64))
        }
        "append" => {
            let mut result = LinkedList::new();
            for arg in args {
                if let Some(cons) = arg.as_cons() {
                    for item in cons.iter() {
                        result.push_back(Rc::clone(item));
                    }
                } else {
                    return Err("append requires lists".to_string());
                }
            }
            Ok(val_cons_from_list(result))
        }
        "reverse" => {
            if args.len() != 1 {
                return Err("reverse requires 1 argument".to_string());
            }
            let cons = args[0].as_cons().ok_or("reverse requires a list")?;
            let mut vec: Vec<ValRef> = cons.to_vec();
            vec.reverse();
            Ok(val_cons(vec))
        }
        _ => Err(format!("Unknown function: {}", name)),
    }
}

// ============================================================================
// REPL
// ============================================================================

fn repl() {
    let mut env = Env::new();
    println!("Tiny Lisp REPL with LinkedList Cons Cells");
    println!("Type expressions or 'exit' to quit");
    println!();
    println!("Examples:");
    println!("  (+ 1 2 3)              => 6");
    println!("  (/ 22 7)               => 22/7 (exact rational)");
    println!("  (* 1/2 2/3)            => 1/3");
    println!("  (cons 1 (cons 2 nil))  => (1 2)");
    println!("  '(1 2 3)               => (1 2 3)");
    println!("  (car '(a b c))         => a");
    println!("  (cdr '(a b c))         => (b c)");
    println!("  (define x 42)          => 42");
    println!("  (length '(1 2 3 4))    => 4");
    println!("  (append '(1 2) '(3 4)) => (1 2 3 4)");
    println!();

    loop {
        print!("lisp> ");
        io::stdout().flush().unwrap();

        let mut input = String::new();
        io::stdin().read_line(&mut input).unwrap();
        let input = input.trim();

        if input.is_empty() {
            continue;
        }

        if input == "exit" || input == "quit" {
            println!("Goodbye!");
            break;
        }

        match parse(input) {
            Ok(expr) => match eval(expr, &mut env) {
                Ok(result) => println!("{}", result.to_string()),
                Err(e) => println!("Error: {}", e),
            },
            Err(e) => println!("Parse error: {}", e),
        }
    }
}

// ============================================================================
// Main
// ============================================================================

fn main() {
    repl();
}
// tests.rs - Comprehensive test suite for the Lisp interpreter
// Place this in your src/ directory or tests/ directory

#[cfg(test)]
mod lisp_tests {

    use super::*;
    // Helper function to evaluate a string expression
    fn eval_str(input: &str) -> Result<ValRef, String> {
        let mut env = Env::new();
        let expr = parse(input)?;
        eval(expr, &mut env)
    }

    // Helper to evaluate with a given environment
    fn eval_str_with_env(input: &str, env: &mut Env) -> Result<ValRef, String> {
        let expr = parse(input)?;
        eval(expr, env)
    }

    // Helper to check if a value is an integer with specific value
    fn assert_int(val: &ValRef, expected: i64) {
        let num = val.as_number_exact().expect("Expected number");
        assert_eq!(
            num,
            &Number::Integer(expected),
            "Expected integer {}",
            expected
        );
    }

    // Helper to check if a value is a rational
    fn assert_rational(val: &ValRef, num: i64, den: i64) {
        let n = val.as_number_exact().expect("Expected number");
        assert_eq!(n, &Number::Rational(num, den), "Expected {}/{}", num, den);
    }

    // ========================================================================
    // Number Type Tests
    // ========================================================================

    #[test]
    fn test_number_integer_creation() {
        let n = Number::integer(42);
        assert_eq!(n, Number::Integer(42));
    }

    #[test]
    fn test_number_rational_creation() {
        let n = Number::rational(3, 4);
        assert_eq!(n, Number::Rational(3, 4));
    }

    #[test]
    fn test_rational_reduction() {
        let n = Number::rational(6, 8);
        assert_eq!(n, Number::Rational(3, 4));
    }

    #[test]
    fn test_rational_to_integer() {
        let n = Number::rational(10, 5);
        assert_eq!(n, Number::Integer(2));
    }

    #[test]
    fn test_rational_negative_denominator() {
        let n = Number::rational(3, -4);
        assert_eq!(n, Number::Rational(-3, 4));
    }

    #[test]
    fn test_rational_zero_numerator() {
        let n = Number::rational(0, 5);
        assert_eq!(n, Number::Integer(0));
    }

    #[test]
    fn test_number_gcd() {
        assert_eq!(Number::gcd(48, 18), 6);
        assert_eq!(Number::gcd(7, 13), 1);
        assert_eq!(Number::gcd(100, 50), 50);
    }

    // ========================================================================
    // Arithmetic Tests
    // ========================================================================

    #[test]
    fn test_integer_addition() {
        let result = eval_str("(+ 1 2 3)").unwrap();
        assert_int(&result, 6);
    }

    #[test]
    fn test_integer_subtraction() {
        let result = eval_str("(- 10 3 2)").unwrap();
        assert_int(&result, 5);
    }

    #[test]
    fn test_integer_multiplication() {
        let result = eval_str("(* 2 3 4)").unwrap();
        assert_int(&result, 24);
    }

    #[test]
    fn test_integer_division_exact() {
        let result = eval_str("(/ 12 3)").unwrap();
        assert_int(&result, 4);
    }

    #[test]
    fn test_integer_division_rational() {
        let result = eval_str("(/ 1 3)").unwrap();
        assert_rational(&result, 1, 3);
    }

    #[test]
    fn test_rational_addition() {
        let result = eval_str("(+ 1/2 1/3)").unwrap();
        assert_rational(&result, 5, 6);
    }

    #[test]
    fn test_rational_subtraction() {
        let result = eval_str("(- 3/4 1/4)").unwrap();
        assert_rational(&result, 1, 2);
    }

    #[test]
    fn test_rational_multiplication() {
        let result = eval_str("(* 2/3 3/4)").unwrap();
        assert_rational(&result, 1, 2);
    }

    #[test]
    fn test_rational_division() {
        let result = eval_str("(/ 1/2 1/3)").unwrap();
        assert_rational(&result, 3, 2);
    }

    #[test]
    fn test_mixed_arithmetic() {
        let result = eval_str("(+ 1 1/2)").unwrap();
        assert_rational(&result, 3, 2);
    }

    #[test]
    fn test_nested_arithmetic() {
        let result = eval_str("(+ (* 2 3) (/ 10 2))").unwrap();
        assert_int(&result, 11);
    }

    #[test]
    fn test_negation() {
        let result = eval_str("(- 5)").unwrap();
        assert_int(&result, -5);
    }

    #[test]
    fn test_negative_rational() {
        let result = eval_str("(- 1/3)").unwrap();
        assert_rational(&result, -1, 3);
    }

    #[test]
    fn test_complex_arithmetic() {
        let result = eval_str("(/ (+ 1 2 3) (* 2 3))").unwrap();
        assert_int(&result, 1);
    }

    #[test]
    fn test_addition_empty() {
        let result = eval_str("(+)").unwrap();
        assert_int(&result, 0);
    }

    #[test]
    fn test_multiplication_empty() {
        let result = eval_str("(*)").unwrap();
        assert_int(&result, 1);
    }

    #[test]
    fn test_division_by_zero() {
        let result = eval_str("(/ 1 0)");
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("zero"));
    }

    // ========================================================================
    // Comparison Tests
    // ========================================================================

    #[test]
    fn test_equality_integers() {
        let result = eval_str("(= 5 5)").unwrap();
        assert_eq!(result.as_bool(), Some(true));
    }

    #[test]
    fn test_equality_integers_false() {
        let result = eval_str("(= 5 6)").unwrap();
        assert_eq!(result.as_bool(), Some(false));
    }

    #[test]
    fn test_equality_rationals() {
        let result = eval_str("(= 1/2 2/4)").unwrap();
        assert_eq!(result.as_bool(), Some(true));
    }

    #[test]
    fn test_less_than() {
        let result = eval_str("(< 3 5)").unwrap();
        assert_eq!(result.as_bool(), Some(true));
    }

    #[test]
    fn test_less_than_false() {
        let result = eval_str("(< 5 3)").unwrap();
        assert_eq!(result.as_bool(), Some(false));
    }

    #[test]
    fn test_greater_than() {
        let result = eval_str("(> 5 3)").unwrap();
        assert_eq!(result.as_bool(), Some(true));
    }

    #[test]
    fn test_comparison_mixed() {
        let result = eval_str("(< 1/2 1)").unwrap();
        assert_eq!(result.as_bool(), Some(true));
    }

    #[test]
    fn test_comparison_rationals() {
        let result = eval_str("(> 2/3 1/2)").unwrap();
        assert_eq!(result.as_bool(), Some(true));
    }

    // ========================================================================
    // List Operations Tests
    // ========================================================================

    #[test]
    fn test_list_creation() {
        let result = eval_str("(list 1 2 3)").unwrap();
        let cons = result.as_cons().expect("Expected cons");
        assert_eq!(cons.len(), 3);
    }

    #[test]
    fn test_empty_list() {
        let result = eval_str("(list)").unwrap();
        let cons = result.as_cons().expect("Expected cons");
        assert_eq!(cons.len(), 0);
    }

    #[test]
    fn test_car() {
        let result = eval_str("(car (list 1 2 3))").unwrap();
        assert_int(&result, 1);
    }

    #[test]
    fn test_cdr() {
        let result = eval_str("(cdr (list 1 2 3))").unwrap();
        let cons = result.as_cons().expect("Expected cons");
        assert_eq!(cons.len(), 2);
    }

    #[test]
    fn test_car_of_cdr() {
        let result = eval_str("(car (cdr (list 1 2 3)))").unwrap();
        assert_int(&result, 2);
    }

    #[test]
    fn test_cons_operation() {
        let result = eval_str("(cons 1 (cons 2 nil))").unwrap();
        let cons = result.as_cons().expect("Expected cons");
        assert_eq!(cons.len(), 2);
    }

    #[test]
    fn test_cons_with_nil() {
        let result = eval_str("(cons 42 nil)").unwrap();
        let cons = result.as_cons().expect("Expected cons");
        assert_eq!(cons.len(), 1);
    }

    #[test]
    fn test_cons_with_list() {
        let result = eval_str("(cons 0 (list 1 2 3))").unwrap();
        let cons = result.as_cons().expect("Expected cons");
        assert_eq!(cons.len(), 4);
    }

    #[test]
    fn test_length() {
        let result = eval_str("(length (list 1 2 3 4 5))").unwrap();
        assert_int(&result, 5);
    }

    #[test]
    fn test_length_empty() {
        let result = eval_str("(length (list))").unwrap();
        assert_int(&result, 0);
    }

    #[test]
    fn test_append_two_lists() {
        let result = eval_str("(append (list 1 2) (list 3 4))").unwrap();
        let cons = result.as_cons().expect("Expected cons");
        assert_eq!(cons.len(), 4);
    }

    #[test]
    fn test_append_multiple_lists() {
        let result = eval_str("(append (list 1) (list 2) (list 3))").unwrap();
        let cons = result.as_cons().expect("Expected cons");
        assert_eq!(cons.len(), 3);
    }

    #[test]
    fn test_append_empty_lists() {
        let result = eval_str("(append (list) (list 1 2))").unwrap();
        let cons = result.as_cons().expect("Expected cons");
        assert_eq!(cons.len(), 2);
    }

    #[test]
    fn test_reverse() {
        let mut env = Env::new();
        eval_str_with_env("(define lst (list 1 2 3))", &mut env).unwrap();
        let result = eval_str_with_env("(car (reverse lst))", &mut env).unwrap();
        assert_int(&result, 3);
    }

    #[test]
    fn test_reverse_empty() {
        let result = eval_str("(reverse (list))").unwrap();
        let cons = result.as_cons().expect("Expected cons");
        assert_eq!(cons.len(), 0);
    }

    #[test]
    fn test_null_predicate_true() {
        let result = eval_str("(null? (list))").unwrap();
        assert_eq!(result.as_bool(), Some(true));
    }

    #[test]
    fn test_null_predicate_false() {
        let result = eval_str("(null? (list 1))").unwrap();
        assert_eq!(result.as_bool(), Some(false));
    }

    #[test]
    fn test_null_predicate_nil() {
        let result = eval_str("(null? nil)").unwrap();
        assert_eq!(result.as_bool(), Some(true));
    }

    #[test]
    fn test_cons_predicate_true() {
        let result = eval_str("(cons? (list 1 2 3))").unwrap();
        assert_eq!(result.as_bool(), Some(true));
    }

    #[test]
    fn test_cons_predicate_false() {
        let result = eval_str("(cons? 42)").unwrap();
        assert_eq!(result.as_bool(), Some(false));
    }

    // ========================================================================
    // Quote Tests
    // ========================================================================

    #[test]
    fn test_quote_number() {
        let result = eval_str("(quote 42)").unwrap();
        assert_int(&result, 42);
    }

    #[test]
    fn test_quote_symbol() {
        let result = eval_str("(quote foo)").unwrap();
        assert_eq!(result.as_symbol(), Some("foo"));
    }

    #[test]
    fn test_quote_list() {
        let result = eval_str("(quote (1 2 3))").unwrap();
        let cons = result.as_cons().expect("Expected cons");
        assert_eq!(cons.len(), 3);
    }

    #[test]
    fn test_quote_shorthand() {
        let result = eval_str("'(1 2 3)").unwrap();
        let cons = result.as_cons().expect("Expected cons");
        assert_eq!(cons.len(), 3);
    }

    #[test]
    fn test_quote_nested() {
        let result = eval_str("'(a (b c) d)").unwrap();
        let cons = result.as_cons().expect("Expected cons");
        assert_eq!(cons.len(), 3);
    }

    #[test]
    fn test_quoted_symbol_not_evaluated() {
        let result = eval_str("'x");
        assert!(result.is_ok());
        assert_eq!(result.unwrap().as_symbol(), Some("x"));
    }

    // ========================================================================
    // Define Tests
    // ========================================================================

    #[test]
    fn test_define_number() {
        let mut env = Env::new();
        eval_str_with_env("(define x 42)", &mut env).unwrap();
        let result = eval_str_with_env("x", &mut env).unwrap();
        assert_int(&result, 42);
    }

    #[test]
    fn test_define_expression() {
        let mut env = Env::new();
        eval_str_with_env("(define x (+ 1 2 3))", &mut env).unwrap();
        let result = eval_str_with_env("x", &mut env).unwrap();
        assert_int(&result, 6);
    }

    #[test]
    fn test_define_list() {
        let mut env = Env::new();
        eval_str_with_env("(define lst '(1 2 3))", &mut env).unwrap();
        let result = eval_str_with_env("(length lst)", &mut env).unwrap();
        assert_int(&result, 3);
    }

    #[test]
    fn test_define_redefine() {
        let mut env = Env::new();
        eval_str_with_env("(define x 10)", &mut env).unwrap();
        eval_str_with_env("(define x 20)", &mut env).unwrap();
        let result = eval_str_with_env("x", &mut env).unwrap();
        assert_int(&result, 20);
    }

    #[test]
    fn test_define_use_in_expression() {
        let mut env = Env::new();
        eval_str_with_env("(define a 5)", &mut env).unwrap();
        eval_str_with_env("(define b 3)", &mut env).unwrap();
        let result = eval_str_with_env("(+ a b)", &mut env).unwrap();
        assert_int(&result, 8);
    }

    // ========================================================================
    // If Tests
    // ========================================================================

    #[test]
    fn test_if_true_branch() {
        let result = eval_str("(if #t 1 2)").unwrap();
        assert_int(&result, 1);
    }

    #[test]
    fn test_if_false_branch() {
        let result = eval_str("(if #f 1 2)").unwrap();
        assert_int(&result, 2);
    }

    #[test]
    fn test_if_with_comparison() {
        let result = eval_str("(if (> 5 3) 'yes 'no)").unwrap();
        assert_eq!(result.as_symbol(), Some("yes"));
    }

    #[test]
    fn test_if_truthy_number() {
        let result = eval_str("(if 42 'yes 'no)").unwrap();
        assert_eq!(result.as_symbol(), Some("yes"));
    }

    #[test]
    fn test_if_nil_is_falsy() {
        let result = eval_str("(if nil 'yes 'no)").unwrap();
        assert_eq!(result.as_symbol(), Some("no"));
    }

    #[test]
    fn test_if_nested() {
        let result = eval_str("(if #t (if #f 1 2) 3)").unwrap();
        assert_int(&result, 2);
    }

    #[test]
    fn test_if_with_define() {
        let mut env = Env::new();
        eval_str_with_env("(define x 10)", &mut env).unwrap();
        let result = eval_str_with_env("(if (> x 5) 'big 'small)", &mut env).unwrap();
        assert_eq!(result.as_symbol(), Some("big"));
    }

    // ========================================================================
    // Complex Expression Tests
    // ========================================================================

    #[test]
    fn test_factorial_like_computation() {
        let result = eval_str("(* 1 2 3 4 5)").unwrap();
        assert_int(&result, 120);
    }

    #[test]
    fn test_nested_list_operations() {
        let result = eval_str("(car (cdr (cdr '(a b c d))))").unwrap();
        assert_eq!(result.as_symbol(), Some("c"));
    }

    #[test]
    fn test_complex_arithmetic_chain() {
        let result = eval_str("(+ (* 2 3) (- 10 5) (/ 20 4))").unwrap();
        assert_int(&result, 16);
    }

    #[test]
    fn test_list_construction_chain() {
        let result = eval_str("(cons 1 (cons 2 (cons 3 (cons 4 nil))))").unwrap();
        let cons = result.as_cons().expect("Expected cons");
        assert_eq!(cons.len(), 4);
    }

    #[test]
    fn test_mixed_data_types_in_list() {
        let mut env = Env::new();
        eval_str_with_env("(define mixed '(1 foo 3/4 #t))", &mut env).unwrap();
        let result = eval_str_with_env("(length mixed)", &mut env).unwrap();
        assert_int(&result, 4);
    }

    // ========================================================================
    // Error Handling Tests
    // ========================================================================

    #[test]
    fn test_undefined_symbol() {
        let result = eval_str("undefined-var");
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Unbound"));
    }

    #[test]
    fn test_car_on_non_list() {
        let result = eval_str("(car 42)");
        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_define() {
        let result = eval_str("(define)");
        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_if() {
        let result = eval_str("(if #t)");
        assert!(result.is_err());
    }

    #[test]
    fn test_wrong_arg_count_plus() {
        // This should work (+ with no args is 0)
        let result = eval_str("(+)");
        assert!(result.is_ok());
    }

    #[test]
    fn test_wrong_arg_count_car() {
        let result = eval_str("(car)");
        assert!(result.is_err());
    }

    #[test]
    fn test_unknown_function() {
        let result = eval_str("(unknown-func 1 2 3)");
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Unknown"));
    }

    // ========================================================================
    // Parser Tests
    // ========================================================================

    #[test]
    fn test_parse_integer() {
        let result = parse("42");
        assert!(result.is_ok());
    }

    #[test]
    fn test_parse_negative_integer() {
        let result = parse("-42");
        assert!(result.is_ok());
    }

    #[test]
    fn test_parse_rational() {
        let result = parse("3/4");
        assert!(result.is_ok());
    }

    #[test]
    fn test_parse_symbol() {
        let result = parse("foo");
        assert!(result.is_ok());
    }

    #[test]
    fn test_parse_bool_true() {
        let result = parse("#t");
        assert!(result.is_ok());
    }

    #[test]
    fn test_parse_bool_false() {
        let result = parse("#f");
        assert!(result.is_ok());
    }

    #[test]
    fn test_parse_empty_list() {
        let result = parse("()");
        assert!(result.is_ok());
    }

    #[test]
    fn test_parse_nested_list() {
        let result = parse("(a (b c) d)");
        assert!(result.is_ok());
    }

    #[test]
    fn test_parse_unmatched_paren() {
        let result = parse("(a b c");
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_extra_paren() {
        let result = parse("(a b c))");
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_with_comments() {
        let result = parse("(+ 1 2) ; this is a comment");
        assert!(result.is_ok());
    }

    #[test]
    fn test_parse_multiline() {
        let result = parse("(+ 1\n   2\n   3)");
        assert!(result.is_ok());
    }

    // ========================================================================
    // Tokenizer Tests
    // ========================================================================

    #[test]
    fn test_tokenize_simple() {
        let tokens = tokenize("(+ 1 2)").unwrap();
        assert_eq!(tokens.len(), 5);
    }

    #[test]
    fn test_tokenize_with_whitespace() {
        let tokens = tokenize("  ( +   1    2  )  ").unwrap();
        assert_eq!(tokens.len(), 5);
    }

    #[test]
    fn test_tokenize_rational() {
        let tokens = tokenize("3/4").unwrap();
        assert_eq!(tokens.len(), 1);
        match &tokens[0] {
            Token::Rational(n, d) => {
                assert_eq!(*n, 3);
                assert_eq!(*d, 4);
            }
            _ => panic!("Expected rational token"),
        }
    }

    #[test]
    fn test_tokenize_quote() {
        let tokens = tokenize("'foo").unwrap();
        assert_eq!(tokens.len(), 2);
    }

    #[test]
    fn test_tokenize_comment() {
        let tokens = tokenize("1 ; comment\n2").unwrap();
        assert_eq!(tokens.len(), 2);
    }

    // ========================================================================
    // Integration Tests
    // ========================================================================

    #[test]
    fn test_sum_of_list() {
        let mut env = Env::new();
        eval_str_with_env("(define nums '(1 2 3 4 5))", &mut env).unwrap();
        // Since we don't have reduce/fold yet, we'll compute manually
        let result = eval_str_with_env("(+ 1 2 3 4 5)", &mut env).unwrap();
        assert_int(&result, 15);
    }

    #[test]
    fn test_rational_precision() {
        // Test that 1/3 + 1/3 + 1/3 = 1 exactly
        let result = eval_str("(+ 1/3 1/3 1/3)").unwrap();
        assert_int(&result, 1);
    }

    #[test]
    fn test_build_list_incrementally() {
        let mut env = Env::new();
        eval_str_with_env("(define lst nil)", &mut env).unwrap();
        eval_str_with_env("(define lst (cons 3 lst))", &mut env).unwrap();
        eval_str_with_env("(define lst (cons 2 lst))", &mut env).unwrap();
        eval_str_with_env("(define lst (cons 1 lst))", &mut env).unwrap();
        let result = eval_str_with_env("(length lst)", &mut env).unwrap();
        assert_int(&result, 3);
    }

    #[test]
    fn test_conditional_computation() {
        let mut env = Env::new();
        eval_str_with_env("(define x 10)", &mut env).unwrap();
        let result = eval_str_with_env("(if (> x 5) (* x 2) (+ x 5))", &mut env).unwrap();
        assert_int(&result, 20);
    }

    #[test]
    fn test_nil_propagation() {
        let mut env = Env::new();
        eval_str_with_env("(define x nil)", &mut env).unwrap();
        let result = eval_str_with_env("(null? x)", &mut env).unwrap();
        assert_eq!(result.as_bool(), Some(true));
    }

    #[test]
    fn test_symbol_preservation_in_quote() {
        let result = eval_str("'(+ 1 2)").unwrap();
        let cons = result.as_cons().expect("Expected cons");
        let first = cons.car().unwrap();
        assert_eq!(first.as_symbol(), Some("+"));
    }

    #[test]
    fn test_nested_if_expressions() {
        let result = eval_str("(if (< 3 5) (if (> 10 8) 'yes 'no) 'outer-no)").unwrap();
        assert_eq!(result.as_symbol(), Some("yes"));
    }

    #[test]
    fn test_arithmetic_with_variables() {
        let mut env = Env::new();
        eval_str_with_env("(define a 10)", &mut env).unwrap();
        eval_str_with_env("(define b 20)", &mut env).unwrap();
        eval_str_with_env("(define c (+ a b))", &mut env).unwrap();
        let result = eval_str_with_env("(* c 2)", &mut env).unwrap();
        assert_int(&result, 60);
    }

    #[test]
    fn test_list_of_rationals() {
        let result = eval_str("'(1/2 1/3 1/4)").unwrap();
        let cons = result.as_cons().expect("Expected cons");
        assert_eq!(cons.len(), 3);
    }

    #[test]
    fn test_complex_boolean_logic() {
        let result = eval_str("(if (< 1 2) (if (> 5 3) #t #f) #f)").unwrap();
        assert_eq!(result.as_bool(), Some(true));
    }
}
