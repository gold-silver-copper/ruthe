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
// Concrete Value Types
// ============================================================================

#[derive(Debug, Clone)]
pub struct Number(pub f64);

impl Value for Number {
    fn type_name(&self) -> &'static str {
        "number"
    }
    fn as_number(&self) -> Option<f64> {
        Some(self.0)
    }
    fn to_string(&self) -> String {
        if self.0.fract() == 0.0 {
            format!("{}", self.0 as i64)
        } else {
            format!("{}", self.0)
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

pub fn val_number(n: f64) -> ValRef {
    Rc::new(Number(n))
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
    Number(f64),
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

                // Try to parse as number
                if let Ok(num) = atom.parse::<f64>() {
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
            let sum: f64 = args
                .iter()
                .map(|a| a.as_number().ok_or("+ requires numbers"))
                .collect::<Result<Vec<f64>, _>>()?
                .iter()
                .sum();
            Ok(val_number(sum))
        }
        "-" => {
            if args.is_empty() {
                return Err("- requires at least 1 argument".to_string());
            }
            let first = args[0].as_number().ok_or("- requires numbers")?;
            if args.len() == 1 {
                return Ok(val_number(-first));
            }
            let rest: f64 = args[1..]
                .iter()
                .map(|a| a.as_number().ok_or("- requires numbers"))
                .collect::<Result<Vec<f64>, _>>()?
                .iter()
                .sum();
            Ok(val_number(first - rest))
        }
        "*" => {
            let product: f64 = args
                .iter()
                .map(|a| a.as_number().ok_or("* requires numbers"))
                .collect::<Result<Vec<f64>, _>>()?
                .iter()
                .product();
            Ok(val_number(product))
        }
        "/" => {
            if args.len() < 2 {
                return Err("/ requires at least 2 arguments".to_string());
            }
            let first = args[0].as_number().ok_or("/ requires numbers")?;
            let rest: f64 = args[1..]
                .iter()
                .map(|a| a.as_number().ok_or("/ requires numbers"))
                .collect::<Result<Vec<f64>, _>>()?
                .iter()
                .product();
            if rest == 0.0 {
                return Err("Division by zero".to_string());
            }
            Ok(val_number(first / rest))
        }
        "=" => {
            if args.len() != 2 {
                return Err("= requires 2 arguments".to_string());
            }
            let a = args[0].as_number().ok_or("= requires numbers")?;
            let b = args[1].as_number().ok_or("= requires numbers")?;
            Ok(val_bool((a - b).abs() < f64::EPSILON))
        }
        "<" => {
            if args.len() != 2 {
                return Err("< requires 2 arguments".to_string());
            }
            let a = args[0].as_number().ok_or("< requires numbers")?;
            let b = args[1].as_number().ok_or("< requires numbers")?;
            Ok(val_bool(a < b))
        }
        ">" => {
            if args.len() != 2 {
                return Err("> requires 2 arguments".to_string());
            }
            let a = args[0].as_number().ok_or("> requires numbers")?;
            let b = args[1].as_number().ok_or("> requires numbers")?;
            Ok(val_bool(a > b))
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
            Ok(val_number(cons.len() as f64))
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cons_operations() {
        let mut env = Env::new();

        // Test cons construction
        let expr = parse("(cons 1 (cons 2 (cons 3 nil)))").unwrap();
        let result = eval(expr, &mut env).unwrap();
        assert!(result.as_cons().is_some());

        // Test car
        let expr = parse("(car (cons 1 (cons 2 nil)))").unwrap();
        let result = eval(expr, &mut env).unwrap();
        assert_eq!(result.as_number(), Some(1.0));

        // Test cdr
        let expr = parse("(cdr (cons 1 (cons 2 nil)))").unwrap();
        let result = eval(expr, &mut env).unwrap();
        let cons = result.as_cons().unwrap();
        assert_eq!(cons.car().unwrap().as_number(), Some(2.0));
    }

    #[test]
    fn test_list_length() {
        let mut env = Env::new();
        let expr = parse("(length '(1 2 3 4 5))").unwrap();
        let result = eval(expr, &mut env).unwrap();
        assert_eq!(result.as_number(), Some(5.0));
    }

    #[test]
    fn test_append() {
        let mut env = Env::new();
        let expr = parse("(length (append '(1 2) '(3 4) '(5)))").unwrap();
        let result = eval(expr, &mut env).unwrap();
        assert_eq!(result.as_number(), Some(5.0));
    }

    #[test]
    fn test_quote_syntax() {
        let mut env = Env::new();
        let expr = parse("'(1 2 3)").unwrap();
        let result = eval(expr, &mut env).unwrap();
        let cons = result.as_cons().unwrap();
        assert_eq!(cons.len(), 3);
    }
}
