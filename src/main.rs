use std::cell::RefCell;
use std::collections::{HashMap, LinkedList};
use std::rc::Rc;

// ============================================================================
// REPL
// ============================================================================

fn repl() {
    use std::io::{self, Write};
    let mut env = Env::new();
    println!("Tiny Lisp REPL with Lambda Support");
    println!("Type expressions or 'exit' to quit");
    println!();
    println!("Examples:");
    println!("  (+ 1 2 3)              => 6");
    println!("  (/ 22 7)               => 22/7 (exact rational)");
    println!("  (define square (lambda (x) (* x x)))");
    println!("  (square 5)             => 25");
    println!(
        "  (define fib (lambda (n) (if (= n 0) 0 (if (= n 1) 1 (+ (fib (- n 1)) (fib (- n 2)))))))"
    );
    println!("  (fib 10)               => 55");
    println!("  '(1 2 3)               => (1 2 3)");
    println!("  (car '(a b c))         => a");
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
// Optimized Value Type - Lambda is now a Rust closure
// ============================================================================

pub type BuiltinFn = fn(&[ValRef]) -> Result<ValRef, String>;
pub type ClosureFn = Rc<dyn Fn(&[ValRef]) -> Result<ValRef, String>>;

#[derive(Clone)]
pub enum Value {
    Number(Number),
    Symbol(String),
    Bool(bool),
    Cons(Rc<Cons>),
    Builtin(BuiltinFn),
    Closure(ClosureFn),
    Nil,
}

// Manual Debug implementation since closures don't implement Debug
impl std::fmt::Debug for Value {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Value::Number(n) => write!(f, "Number({:?})", n),
            Value::Symbol(s) => write!(f, "Symbol({:?})", s),
            Value::Bool(b) => write!(f, "Bool({:?})", b),
            Value::Cons(c) => write!(f, "Cons({:?})", c),
            Value::Builtin(_) => write!(f, "Builtin(<fn>)"),
            Value::Closure(_) => write!(f, "Closure(<fn>)"),
            Value::Nil => write!(f, "Nil"),
        }
    }
}

pub type ValRef = Rc<Value>;

impl Value {
    pub fn type_name(&self) -> &'static str {
        match self {
            Value::Number(_) => "number",
            Value::Symbol(_) => "symbol",
            Value::Bool(_) => "bool",
            Value::Cons(_) => "cons",
            Value::Builtin(_) => "builtin",
            Value::Closure(_) => "closure",
            Value::Nil => "nil",
        }
    }

    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Value::Bool(b) => Some(*b),
            _ => None,
        }
    }

    pub fn as_number(&self) -> Option<f64> {
        match self {
            Value::Number(n) => Some(n.to_f64()),
            _ => None,
        }
    }

    pub fn as_number_exact(&self) -> Option<&Number> {
        match self {
            Value::Number(n) => Some(n),
            _ => None,
        }
    }

    pub fn as_symbol(&self) -> Option<&str> {
        match self {
            Value::Symbol(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_cons(&self) -> Option<&Cons> {
        match self {
            Value::Cons(c) => Some(c),
            _ => None,
        }
    }

    pub fn as_builtin(&self) -> Option<BuiltinFn> {
        match self {
            Value::Builtin(f) => Some(*f),
            _ => None,
        }
    }

    pub fn as_closure(&self) -> Option<&ClosureFn> {
        match self {
            Value::Closure(f) => Some(f),
            _ => None,
        }
    }

    pub fn is_callable(&self) -> bool {
        matches!(self, Value::Builtin(_) | Value::Closure(_))
    }

    pub fn is_nil(&self) -> bool {
        matches!(self, Value::Nil)
    }

    pub fn to_string(&self) -> String {
        match self {
            Value::Number(n) => n.to_string(),
            Value::Symbol(s) => s.clone(),
            Value::Bool(b) => if *b { "#t" } else { "#f" }.to_string(),
            Value::Cons(c) => c.to_string(),
            Value::Builtin(_) => "<builtin>".to_string(),
            Value::Closure(_) => "<closure>".to_string(),
            Value::Nil => "nil".to_string(),
        }
    }
}

// ============================================================================
// Number Type - Exact arithmetic with integers and rationals
// ============================================================================

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Number {
    Integer(i64),
    Rational(i64, i64),
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

        let (num, den) = if den < 0 { (-num, -den) } else { (num, den) };

        if den == 1 {
            Number::Integer(num)
        } else {
            Number::Rational(num, den)
        }
    }

    pub fn gcd(mut a: i64, mut b: i64) -> i64 {
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

    pub fn to_string(&self) -> String {
        match self {
            Number::Integer(n) => format!("{}", n),
            Number::Rational(num, den) => format!("{}/{}", num, den),
        }
    }
}

// ============================================================================
// Cons Cell
// ============================================================================

#[derive(Debug, Clone)]
pub struct Cons {
    list: LinkedList<ValRef>,
}

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

    pub fn cdr(&self) -> Rc<Cons> {
        let mut new_list = self.list.clone();
        new_list.pop_front();
        Rc::new(Cons::from_list(new_list))
    }

    pub fn cons(val: ValRef, rest: &Cons) -> Rc<Cons> {
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

    pub fn to_string(&self) -> String {
        if self.is_empty() {
            return "()".to_string();
        }
        let items: Vec<String> = self.iter().map(|v| v.to_string()).collect();
        format!("({})", items.join(" "))
    }
}

// ============================================================================
// Constructors
// ============================================================================

pub fn val_number(n: i64) -> ValRef {
    Rc::new(Value::Number(Number::integer(n)))
}

pub fn val_rational(num: i64, den: i64) -> ValRef {
    Rc::new(Value::Number(Number::rational(num, den)))
}

pub fn val_number_from_num(n: Number) -> ValRef {
    Rc::new(Value::Number(n))
}

pub fn val_symbol(s: &str) -> ValRef {
    Rc::new(Value::Symbol(s.to_string()))
}

pub fn val_bool(b: bool) -> ValRef {
    Rc::new(Value::Bool(b))
}

pub fn val_cons(items: Vec<ValRef>) -> ValRef {
    Rc::new(Value::Cons(Rc::new(Cons::from_vec(items))))
}

pub fn val_cons_from_list(list: LinkedList<ValRef>) -> ValRef {
    Rc::new(Value::Cons(Rc::new(Cons::from_list(list))))
}

pub fn val_builtin(f: BuiltinFn) -> ValRef {
    Rc::new(Value::Builtin(f))
}

pub fn val_closure(f: ClosureFn) -> ValRef {
    Rc::new(Value::Closure(f))
}

pub fn val_nil() -> ValRef {
    Rc::new(Value::Nil)
}

// ============================================================================
// Environment - Now uses RefCell for interior mutability
// ============================================================================

pub type EnvRef = Rc<RefCell<Env>>;

#[derive(Debug, Clone)]
pub struct Env {
    vars: HashMap<String, ValRef>,
    parent: Option<EnvRef>,
}

impl Env {
    pub fn new() -> EnvRef {
        let mut env = Self {
            vars: HashMap::new(),
            parent: None,
        };
        env.register_builtins();
        Rc::new(RefCell::new(env))
    }

    pub fn with_parent(parent: EnvRef) -> EnvRef {
        let env = Self {
            vars: HashMap::new(),
            parent: Some(parent),
        };
        Rc::new(RefCell::new(env))
    }

    pub fn set(&mut self, name: String, v: ValRef) {
        self.vars.insert(name, v);
    }

    pub fn get(&self, name: &str) -> Option<ValRef> {
        if let Some(val) = self.vars.get(name) {
            Some(Rc::clone(val))
        } else if let Some(parent) = &self.parent {
            parent.borrow().get(name)
        } else {
            None
        }
    }

    fn register_builtins(&mut self) {
        self.set("nil".to_string(), val_nil());
        self.set("+".to_string(), val_builtin(builtin_add));
        self.set("-".to_string(), val_builtin(builtin_sub));
        self.set("*".to_string(), val_builtin(builtin_mul));
        self.set("/".to_string(), val_builtin(builtin_div));
        self.set("=".to_string(), val_builtin(builtin_eq));
        self.set("<".to_string(), val_builtin(builtin_lt));
        self.set(">".to_string(), val_builtin(builtin_gt));
        self.set("list".to_string(), val_builtin(builtin_list));
        self.set("car".to_string(), val_builtin(builtin_car));
        self.set("cdr".to_string(), val_builtin(builtin_cdr));
        self.set("cons".to_string(), val_builtin(builtin_cons));
        self.set("null?".to_string(), val_builtin(builtin_null));
        self.set("cons?".to_string(), val_builtin(builtin_cons_p));
        self.set("length".to_string(), val_builtin(builtin_length));
        self.set("append".to_string(), val_builtin(builtin_append));
        self.set("reverse".to_string(), val_builtin(builtin_reverse));
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
// Evaluator - Now works with EnvRef (Rc<RefCell<Env>>)
// ============================================================================

pub fn eval(expr: ValRef, env: &EnvRef) -> Result<ValRef, String> {
    match expr.as_ref() {
        Value::Number(_) | Value::Bool(_) | Value::Nil | Value::Builtin(_) | Value::Closure(_) => {
            Ok(Rc::clone(&expr))
        }
        Value::Symbol(s) => {
            if s == "nil" {
                return Ok(val_nil());
            }
            env.borrow()
                .get(s)
                .ok_or_else(|| format!("Unbound symbol: {}", s))
        }
        Value::Cons(cons) => {
            if cons.is_empty() {
                return Ok(val_nil());
            }

            let first = cons.car().ok_or("Empty cons in eval")?;

            // Special forms
            if let Value::Symbol(sym) = first.as_ref() {
                match sym.as_str() {
                    "define" => {
                        if cons.len() != 3 {
                            return Err("define requires 2 arguments".to_string());
                        }
                        let items = cons.to_vec();
                        let name = items[1]
                            .as_symbol()
                            .ok_or("define requires symbol as first arg")?;
                        let val = eval(Rc::clone(&items[2]), env)?;
                        env.borrow_mut().set(name.to_string(), Rc::clone(&val));
                        return Ok(val);
                    }
                    "lambda" => {
                        if cons.len() != 3 {
                            return Err("lambda requires 2 arguments (params body)".to_string());
                        }
                        let items = cons.to_vec();

                        // Extract parameter names
                        let params_cons =
                            items[1].as_cons().ok_or("lambda params must be a list")?;
                        let params: Result<Vec<String>, String> = params_cons
                            .iter()
                            .map(|p| {
                                p.as_symbol()
                                    .map(|s| s.to_string())
                                    .ok_or_else(|| "lambda params must be symbols".to_string())
                            })
                            .collect();
                        let params = params?;

                        let body = Rc::clone(&items[2]);

                        // Capture the environment by reference (Rc)
                        // This allows recursive functions to work!
                        let captured_env = Rc::clone(env);

                        // Create a Rust closure that captures the environment and body
                        let closure = move |args: &[ValRef]| -> Result<ValRef, String> {
                            if args.len() != params.len() {
                                return Err(format!(
                                    "Lambda expects {} arguments, got {}",
                                    params.len(),
                                    args.len()
                                ));
                            }

                            // Create new environment with captured env as parent
                            let lambda_env = Env::with_parent(Rc::clone(&captured_env));

                            // Bind parameters
                            for (param, arg) in params.iter().zip(args.iter()) {
                                lambda_env.borrow_mut().set(param.clone(), Rc::clone(arg));
                            }

                            // Evaluate body
                            eval(Rc::clone(&body), &lambda_env)
                        };

                        return Ok(val_closure(Rc::new(closure)));
                    }
                    "if" => {
                        if cons.len() != 4 {
                            return Err("if requires 3 arguments".to_string());
                        }
                        let items = cons.to_vec();
                        let cond = eval(Rc::clone(&items[1]), env)?;
                        let is_true = match cond.as_ref() {
                            Value::Bool(b) => *b,
                            Value::Nil => false,
                            _ => true,
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
                    _ => {}
                }
            }

            // Function application
            let func = eval(Rc::clone(&first), env)?;

            let rest = cons.cdr();
            let args: Result<Vec<ValRef>, String> =
                rest.iter().map(|arg| eval(Rc::clone(arg), env)).collect();
            let args = args?;

            // Call builtin or closure
            match func.as_ref() {
                Value::Builtin(f) => f(&args),
                Value::Closure(f) => f(&args),
                _ => Err(format!("Cannot call non-function: {}", func.to_string())),
            }
        }
    }
}

// ============================================================================
// Built-in Functions - Now regular Rust functions!
// ============================================================================

fn builtin_add(args: &[ValRef]) -> Result<ValRef, String> {
    let mut result = Number::integer(0);
    for arg in args {
        let num = arg.as_number_exact().ok_or("+ requires numbers")?;
        result = result.add(num);
    }
    Ok(val_number_from_num(result))
}

fn builtin_sub(args: &[ValRef]) -> Result<ValRef, String> {
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

fn builtin_mul(args: &[ValRef]) -> Result<ValRef, String> {
    let mut result = Number::integer(1);
    for arg in args {
        let num = arg.as_number_exact().ok_or("* requires numbers")?;
        result = result.mul(num);
    }
    Ok(val_number_from_num(result))
}

fn builtin_div(args: &[ValRef]) -> Result<ValRef, String> {
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

fn builtin_eq(args: &[ValRef]) -> Result<ValRef, String> {
    if args.len() != 2 {
        return Err("= requires 2 arguments".to_string());
    }
    let a = args[0].as_number_exact().ok_or("= requires numbers")?;
    let b = args[1].as_number_exact().ok_or("= requires numbers")?;
    Ok(val_bool(a.cmp(b) == std::cmp::Ordering::Equal))
}

fn builtin_lt(args: &[ValRef]) -> Result<ValRef, String> {
    if args.len() != 2 {
        return Err("< requires 2 arguments".to_string());
    }
    let a = args[0].as_number_exact().ok_or("< requires numbers")?;
    let b = args[1].as_number_exact().ok_or("< requires numbers")?;
    Ok(val_bool(a.cmp(b) == std::cmp::Ordering::Less))
}

fn builtin_gt(args: &[ValRef]) -> Result<ValRef, String> {
    if args.len() != 2 {
        return Err("> requires 2 arguments".to_string());
    }
    let a = args[0].as_number_exact().ok_or("> requires numbers")?;
    let b = args[1].as_number_exact().ok_or("> requires numbers")?;
    Ok(val_bool(a.cmp(b) == std::cmp::Ordering::Greater))
}

fn builtin_list(args: &[ValRef]) -> Result<ValRef, String> {
    Ok(val_cons(args.to_vec()))
}

fn builtin_car(args: &[ValRef]) -> Result<ValRef, String> {
    if args.len() != 1 {
        return Err("car requires 1 argument".to_string());
    }
    let cons = args[0].as_cons().ok_or("car requires a cons/list")?;
    cons.car().ok_or("car of empty list".to_string())
}

fn builtin_cdr(args: &[ValRef]) -> Result<ValRef, String> {
    if args.len() != 1 {
        return Err("cdr requires 1 argument".to_string());
    }
    let cons = args[0].as_cons().ok_or("cdr requires a cons/list")?;
    Ok(Rc::new(Value::Cons(cons.cdr())))
}

fn builtin_cons(args: &[ValRef]) -> Result<ValRef, String> {
    if args.len() != 2 {
        return Err("cons requires 2 arguments".to_string());
    }
    if let Some(rest_cons) = args[1].as_cons() {
        Ok(Rc::new(Value::Cons(Cons::cons(
            Rc::clone(&args[0]),
            rest_cons,
        ))))
    } else if args[1].is_nil() {
        Ok(val_cons(vec![Rc::clone(&args[0])]))
    } else {
        Ok(val_cons(vec![Rc::clone(&args[0]), Rc::clone(&args[1])]))
    }
}

fn builtin_null(args: &[ValRef]) -> Result<ValRef, String> {
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

fn builtin_cons_p(args: &[ValRef]) -> Result<ValRef, String> {
    if args.len() != 1 {
        return Err("cons? requires 1 argument".to_string());
    }
    Ok(val_bool(args[0].as_cons().is_some()))
}

fn builtin_length(args: &[ValRef]) -> Result<ValRef, String> {
    if args.len() != 1 {
        return Err("length requires 1 argument".to_string());
    }
    let cons = args[0].as_cons().ok_or("length requires a list")?;
    Ok(val_number(cons.len() as i64))
}

fn builtin_append(args: &[ValRef]) -> Result<ValRef, String> {
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

fn builtin_reverse(args: &[ValRef]) -> Result<ValRef, String> {
    if args.len() != 1 {
        return Err("reverse requires 1 argument".to_string());
    }
    let cons = args[0].as_cons().ok_or("reverse requires a list")?;
    let mut vec: Vec<ValRef> = cons.to_vec();
    vec.reverse();
    Ok(val_cons(vec))
}

use std::time::Instant;

fn main() {
    let env = Env::new();

    // Quick test
    let result = parse("(+ 1 2 3)").unwrap();
    let result = eval(result, &env).unwrap();
    println!("Result: {}", result.to_string());

    // Test lambda
    println!("\nTesting lambda:");
    let _ = eval(parse("(define square (lambda (x) (* x x)))").unwrap(), &env).unwrap();
    let result = eval(parse("(square 5)").unwrap(), &env).unwrap();
    println!("(square 5) = {}", result.to_string());

    // Test recursive fibonacci
    println!("\nTesting recursive fib:");
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
    let _ = eval(parse(fib_def).unwrap(), &env).unwrap();

    // Time the fibonacci execution
    let start = Instant::now();
    let result = eval(parse("(fib 10)").unwrap(), &env).unwrap();
    let duration = start.elapsed();

    println!("(fib 10) = {}", result.to_string());
    println!("Time taken: {:?}", duration);

    // You could also test with larger values to see the exponential time growth
    println!("\nTesting fib with larger value:");
    let start = Instant::now();
    let result = eval(parse("(fib 15)").unwrap(), &env).unwrap();
    let duration = start.elapsed();
    println!("(fib 15) = {}", result.to_string());
    println!("Time taken: {:?}", duration);

    // Even larger - this will be noticeably slower
    println!("\nTesting fib with even larger value:");
    let start = Instant::now();
    let result = eval(parse("(fib 30)").unwrap(), &env).unwrap();
    let duration = start.elapsed();
    println!("(fib 60000) = {}", result.to_string());
    println!("Time taken: {:?}", duration);
}
