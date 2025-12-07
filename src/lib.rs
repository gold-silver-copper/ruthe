#![no_std]

extern crate alloc;

use alloc::format;
use alloc::rc::Rc;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use core::cell::RefCell;
use core::cmp::Ordering;
use core::hash::BuildHasherDefault;
use hashbrown::HashMap;
use rustc_hash::FxHasher;

// Type alias for our hash map with FxHasher
type FxHashMap<K, V> = HashMap<K, V, BuildHasherDefault<FxHasher>>;

// ============================================================================
// Trampoline System for Proper TCO
// ============================================================================

/// Result of an evaluation that may need to continue
pub enum EvalResult {
    /// Final value - evaluation is complete
    Done(ValRef),
    /// Tail call - continue evaluating with new expr and env
    TailCall(ValRef, EnvRef),
}

// ============================================================================
// Optimized Value Type - Lambda now stores body and params for TCO
// ============================================================================

pub type BuiltinFn = fn(&[ValRef]) -> Result<ValRef, String>;

#[derive(Clone)]
pub enum Value {
    Number(i64),
    Symbol(String),
    Bool(bool),
    Cons(ValRef, ValRef), // Traditional cons: (car . cdr)
    Builtin(BuiltinFn),
    Lambda {
        params: Vec<String>,
        body: ValRef,
        env: EnvRef,
    },
    Nil,
}

// Manual Debug implementation
impl core::fmt::Debug for Value {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Value::Number(n) => write!(f, "Number({})", n),
            Value::Symbol(s) => write!(f, "Symbol({:?})", s),
            Value::Bool(b) => write!(f, "Bool({:?})", b),
            Value::Cons(car, cdr) => write!(f, "Cons({:?}, {:?})", car, cdr),
            Value::Builtin(_) => write!(f, "Builtin(<fn>)"),
            Value::Lambda { .. } => write!(f, "Lambda(<fn>)"),
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
            Value::Cons(_, _) => "cons",
            Value::Builtin(_) => "builtin",
            Value::Lambda { .. } => "lambda",
            Value::Nil => "nil",
        }
    }

    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Value::Bool(b) => Some(*b),
            _ => None,
        }
    }

    pub fn as_number(&self) -> Option<i64> {
        match self {
            Value::Number(n) => Some(*n),
            _ => None,
        }
    }

    pub fn as_symbol(&self) -> Option<&str> {
        match self {
            Value::Symbol(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_cons(&self) -> Option<(&ValRef, &ValRef)> {
        match self {
            Value::Cons(car, cdr) => Some((car, cdr)),
            _ => None,
        }
    }

    pub fn as_builtin(&self) -> Option<BuiltinFn> {
        match self {
            Value::Builtin(f) => Some(*f),
            _ => None,
        }
    }

    pub fn as_lambda(&self) -> Option<(&Vec<String>, &ValRef, &EnvRef)> {
        match self {
            Value::Lambda { params, body, env } => Some((params, body, env)),
            _ => None,
        }
    }

    pub fn is_callable(&self) -> bool {
        matches!(self, Value::Builtin(_) | Value::Lambda { .. })
    }

    pub fn is_nil(&self) -> bool {
        matches!(self, Value::Nil)
    }

    pub fn to_string(&self) -> String {
        match self {
            Value::Number(n) => format!("{}", n),
            Value::Symbol(s) => s.clone(),
            Value::Bool(b) => if *b { "#t" } else { "#f" }.to_string(),
            Value::Cons(_, _) => list_to_string(self),
            Value::Builtin(_) => "<builtin>".to_string(),
            Value::Lambda { .. } => "<lambda>".to_string(),
            Value::Nil => "nil".to_string(),
        }
    }
}

// Helper function to convert a cons list to string

fn list_to_string(val: &Value) -> String {
    let mut items = Vec::new();
    let mut current = val;

    loop {
        match current {
            Value::Cons(car, cdr) => {
                items.push(car.to_string());
                current = cdr.as_ref();
            }
            Value::Nil => break,
            _ => {
                // Improper list (dotted pair)
                items.push(".".to_string());
                items.push(current.to_string());
                break;
            }
        }
    }

    let mut result = String::from("(");
    for (i, item) in items.iter().enumerate() {
        if i > 0 {
            result.push(' ');
        }
        result.push_str(item);
    }
    result.push(')');
    result
}

// Helper function to get length of a list

fn list_len(val: &Value) -> usize {
    let mut count = 0;
    let mut current = val;

    loop {
        match current {
            Value::Cons(_, cdr) => {
                count += 1;
                current = cdr.as_ref();
            }
            Value::Nil => break,
            _ => break, // Improper list
        }
    }

    count
}

// Helper function to convert list to vector

fn list_to_vec(val: &Value) -> Vec<ValRef> {
    let mut items = Vec::new();
    let mut current = val;

    loop {
        match current {
            Value::Cons(car, cdr) => {
                items.push(Rc::clone(car));
                current = cdr.as_ref();
            }
            Value::Nil => break,
            _ => break,
        }
    }

    items
}

// ============================================================================
// Constructors
// ============================================================================

pub fn val_number(n: i64) -> ValRef {
    Rc::new(Value::Number(n))
}

pub fn val_symbol(s: &str) -> ValRef {
    Rc::new(Value::Symbol(s.to_string()))
}

pub fn val_bool(b: bool) -> ValRef {
    Rc::new(Value::Bool(b))
}

pub fn val_cons(car: ValRef, cdr: ValRef) -> ValRef {
    Rc::new(Value::Cons(car, cdr))
}

pub fn val_list(items: Vec<ValRef>) -> ValRef {
    items
        .into_iter()
        .rev()
        .fold(val_nil(), |acc, item| val_cons(item, acc))
}

pub fn val_builtin(f: BuiltinFn) -> ValRef {
    Rc::new(Value::Builtin(f))
}

pub fn val_lambda(params: Vec<String>, body: ValRef, env: EnvRef) -> ValRef {
    Rc::new(Value::Lambda { params, body, env })
}

pub fn val_nil() -> ValRef {
    Rc::new(Value::Nil)
}

// ============================================================================
// Environment - Now uses HashMap with FxHasher
// ============================================================================

pub type EnvRef = Rc<RefCell<Env>>;

#[derive(Debug, Clone)]
pub struct Env {
    bindings: FxHashMap<String, ValRef>,
    parent: Option<EnvRef>,
}

impl Env {
    pub fn new() -> EnvRef {
        let mut env = Self {
            bindings: FxHashMap::default(),
            parent: None,
        };
        env.register_builtins();
        Rc::new(RefCell::new(env))
    }

    pub fn with_parent(parent: EnvRef) -> EnvRef {
        let env = Self {
            bindings: FxHashMap::default(),
            parent: Some(parent),
        };
        Rc::new(RefCell::new(env))
    }

    pub fn set(&mut self, name: String, v: ValRef) {
        self.bindings.insert(name, v);
    }

    pub fn get(&self, name: &str) -> Option<ValRef> {
        // Check current environment
        if let Some(val) = self.bindings.get(name) {
            return Some(Rc::clone(val));
        }

        // Not found in current env, try parent
        if let Some(parent) = &self.parent {
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
        self.set("cons".to_string(), val_builtin(builtin_cons_fn));
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
    Bool(bool),
    Quote,
}

fn parse_i64(s: &str) -> Result<i64, ()> {
    let bytes = s.as_bytes();
    if bytes.is_empty() {
        return Err(());
    }

    let (negative, start) = if bytes[0] == b'-' {
        if bytes.len() == 1 {
            return Err(()); // Just a minus sign
        }
        (true, 1)
    } else if bytes[0] == b'+' {
        if bytes.len() == 1 {
            return Err(()); // Just a plus sign
        }
        (false, 1)
    } else {
        (false, 0)
    };

    if bytes[start..].is_empty() {
        return Err(());
    }

    let mut result: i64 = 0;
    for &b in &bytes[start..] {
        if !(b'0'..=b'9').contains(&b) {
            return Err(());
        }
        let digit = (b - b'0') as i64;
        result = result
            .checked_mul(10)
            .and_then(|r| r.checked_add(digit))
            .ok_or(())?;
    }

    if negative {
        result.checked_neg().ok_or(())
    } else {
        Ok(result)
    }
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

                if atom.is_empty() {
                    continue;
                }

                // Try to parse as integer
                if let Ok(num) = parse_i64(&atom) {
                    tokens.push(Token::Number(num));
                } else {
                    // It's a symbol
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
            let quoted = val_list(Vec::from([val_symbol("quote"), val]));
            Ok((quoted, consumed + 1))
        }
        Token::LParen => {
            let mut items = Vec::new();
            let mut pos = 1;

            while pos < tokens.len() {
                if tokens[pos] == Token::RParen {
                    return Ok((val_list(items), pos + 1));
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

pub fn parse_multiple(input: &str) -> Result<Vec<ValRef>, String> {
    let tokens = tokenize(input)?;
    if tokens.is_empty() {
        return Err("Empty input".to_string());
    }

    let mut expressions = Vec::new();
    let mut pos = 0;

    while pos < tokens.len() {
        let (val, consumed) = parse_tokens(&tokens[pos..])?;
        expressions.push(val);
        pos += consumed;
    }

    Ok(expressions)
}

// ============================================================================
// Evaluator - With Proper Tail Call Optimization via Trampoline
// ============================================================================

fn eval_step(expr: ValRef, env: &EnvRef) -> Result<EvalResult, String> {
    match expr.as_ref() {
        Value::Number(_)
        | Value::Bool(_)
        | Value::Nil
        | Value::Builtin(_)
        | Value::Lambda { .. } => Ok(EvalResult::Done(Rc::clone(&expr))),
        Value::Symbol(s) => {
            if s == "nil" {
                return Ok(EvalResult::Done(val_nil()));
            }
            env.borrow()
                .get(s)
                .map(EvalResult::Done)
                .ok_or_else(|| format!("Unbound symbol: {}", s))
        }
        Value::Cons(car, cdr) => {
            // Special forms
            if let Value::Symbol(sym) = car.as_ref() {
                match sym.as_str() {
                    "define" => {
                        let items = list_to_vec(expr.as_ref());
                        if items.len() != 3 {
                            return Err("define requires 2 arguments".to_string());
                        }
                        let name = items[1]
                            .as_symbol()
                            .ok_or("define requires symbol as first arg")?;
                        let val = eval(Rc::clone(&items[2]), env)?;
                        env.borrow_mut().set(name.to_string(), Rc::clone(&val));
                        return Ok(EvalResult::Done(val));
                    }
                    "lambda" => {
                        let items = list_to_vec(expr.as_ref());
                        if items.len() != 3 {
                            return Err("lambda requires 2 arguments (params body)".to_string());
                        }

                        // Extract parameter names
                        let params_list = list_to_vec(items[1].as_ref());
                        let params: Result<Vec<String>, String> = params_list
                            .iter()
                            .map(|p| {
                                p.as_symbol()
                                    .map(|s| s.to_string())
                                    .ok_or_else(|| "lambda params must be symbols".to_string())
                            })
                            .collect();
                        let params = params?;

                        let body = Rc::clone(&items[2]);
                        let captured_env = Rc::clone(env);

                        return Ok(EvalResult::Done(val_lambda(params, body, captured_env)));
                    }
                    "if" => {
                        let items = list_to_vec(expr.as_ref());
                        if items.len() != 4 {
                            return Err("if requires 3 arguments".to_string());
                        }
                        let cond = eval(Rc::clone(&items[1]), env)?;
                        let is_true = match cond.as_ref() {
                            Value::Bool(b) => *b,
                            Value::Nil => false,
                            _ => true,
                        };
                        // Tail call: return the branch to evaluate
                        return Ok(EvalResult::TailCall(
                            Rc::clone(&items[if is_true { 2 } else { 3 }]),
                            Rc::clone(env),
                        ));
                    }
                    "quote" => {
                        let items = list_to_vec(expr.as_ref());
                        if items.len() != 2 {
                            return Err("quote requires 1 argument".to_string());
                        }
                        return Ok(EvalResult::Done(Rc::clone(&items[1])));
                    }
                    _ => {}
                }
            }

            // Function application
            let func = eval(Rc::clone(car), env)?;

            // Evaluate arguments
            let mut args = Vec::new();
            let mut current = cdr.as_ref();
            loop {
                match current {
                    Value::Cons(arg_car, arg_cdr) => {
                        args.push(eval(Rc::clone(arg_car), env)?);
                        current = arg_cdr.as_ref();
                    }
                    Value::Nil => break,
                    _ => return Err("Malformed argument list".to_string()),
                }
            }

            // Call function
            match func.as_ref() {
                Value::Builtin(f) => Ok(EvalResult::Done(f(&args)?)),
                Value::Lambda {
                    params,
                    body,
                    env: lambda_env,
                } => {
                    if args.len() != params.len() {
                        return Err(format!(
                            "Lambda expects {} arguments, got {}",
                            params.len(),
                            args.len()
                        ));
                    }

                    // Create new environment with lambda's captured env as parent
                    let call_env = Env::with_parent(Rc::clone(lambda_env));

                    // Bind parameters
                    for (param, arg) in params.iter().zip(args.iter()) {
                        call_env.borrow_mut().set(param.clone(), Rc::clone(arg));
                    }

                    // Tail call: evaluate body in new environment
                    Ok(EvalResult::TailCall(Rc::clone(body), call_env))
                }
                _ => Err(format!("Cannot call non-function: {}", func.to_string())),
            }
        }
        Value::Nil => Ok(EvalResult::Done(val_nil())),
    }
}

pub fn eval(mut expr: ValRef, env: &EnvRef) -> Result<ValRef, String> {
    let mut current_env = Rc::clone(env);

    loop {
        match eval_step(expr, &current_env)? {
            EvalResult::Done(val) => return Ok(val),
            EvalResult::TailCall(new_expr, new_env) => {
                expr = new_expr;
                current_env = new_env;
                // Continue loop - this is the key to TCO!
            }
        }
    }
}

// ============================================================================
// Built-in Functions
// ============================================================================

fn builtin_add(args: &[ValRef]) -> Result<ValRef, String> {
    let mut result: i64 = 0;
    for arg in args {
        let num = arg.as_number().ok_or("+ requires numbers")?;
        result = result.checked_add(num).ok_or("Integer overflow")?;
    }
    Ok(val_number(result))
}

fn builtin_sub(args: &[ValRef]) -> Result<ValRef, String> {
    if args.is_empty() {
        return Err("- requires at least 1 argument".to_string());
    }
    let first = args[0].as_number().ok_or("- requires numbers")?;
    if args.len() == 1 {
        return Ok(val_number(first.checked_neg().ok_or("Integer overflow")?));
    }
    let mut result = first;
    for arg in &args[1..] {
        let num = arg.as_number().ok_or("- requires numbers")?;
        result = result.checked_sub(num).ok_or("Integer overflow")?;
    }
    Ok(val_number(result))
}

fn builtin_mul(args: &[ValRef]) -> Result<ValRef, String> {
    let mut result: i64 = 1;
    for arg in args {
        let num = arg.as_number().ok_or("* requires numbers")?;
        result = result.checked_mul(num).ok_or("Integer overflow")?;
    }
    Ok(val_number(result))
}

fn builtin_div(args: &[ValRef]) -> Result<ValRef, String> {
    if args.len() < 2 {
        return Err("/ requires at least 2 arguments".to_string());
    }
    let first = args[0].as_number().ok_or("/ requires numbers")?;
    let mut result = first;
    for arg in &args[1..] {
        let num = arg.as_number().ok_or("/ requires numbers")?;
        if num == 0 {
            return Err("Division by zero".to_string());
        }
        result = result.checked_div(num).ok_or("Integer overflow")?;
    }
    Ok(val_number(result))
}

fn builtin_eq(args: &[ValRef]) -> Result<ValRef, String> {
    if args.len() != 2 {
        return Err("= requires 2 arguments".to_string());
    }
    let a = args[0].as_number().ok_or("= requires numbers")?;
    let b = args[1].as_number().ok_or("= requires numbers")?;
    Ok(val_bool(a == b))
}

fn builtin_lt(args: &[ValRef]) -> Result<ValRef, String> {
    if args.len() != 2 {
        return Err("< requires 2 arguments".to_string());
    }
    let a = args[0].as_number().ok_or("< requires numbers")?;
    let b = args[1].as_number().ok_or("< requires numbers")?;
    Ok(val_bool(a < b))
}

fn builtin_gt(args: &[ValRef]) -> Result<ValRef, String> {
    if args.len() != 2 {
        return Err("> requires 2 arguments".to_string());
    }
    let a = args[0].as_number().ok_or("> requires numbers")?;
    let b = args[1].as_number().ok_or("> requires numbers")?;
    Ok(val_bool(a > b))
}

fn builtin_list(args: &[ValRef]) -> Result<ValRef, String> {
    Ok(val_list(args.to_vec()))
}

fn builtin_car(args: &[ValRef]) -> Result<ValRef, String> {
    if args.len() != 1 {
        return Err("car requires 1 argument".to_string());
    }
    let (car, _) = args[0].as_cons().ok_or("car requires a cons/list")?;
    Ok(Rc::clone(car))
}

fn builtin_cdr(args: &[ValRef]) -> Result<ValRef, String> {
    if args.len() != 1 {
        return Err("cdr requires 1 argument".to_string());
    }
    let (_, cdr) = args[0].as_cons().ok_or("cdr requires a cons/list")?;
    Ok(Rc::clone(cdr))
}

fn builtin_cons_fn(args: &[ValRef]) -> Result<ValRef, String> {
    if args.len() != 2 {
        return Err("cons requires 2 arguments".to_string());
    }
    Ok(val_cons(Rc::clone(&args[0]), Rc::clone(&args[1])))
}

fn builtin_null(args: &[ValRef]) -> Result<ValRef, String> {
    if args.len() != 1 {
        return Err("null? requires 1 argument".to_string());
    }
    Ok(val_bool(args[0].is_nil()))
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
    let len = list_len(args[0].as_ref());
    Ok(val_number(len as i64))
}

fn builtin_append(args: &[ValRef]) -> Result<ValRef, String> {
    let mut result = Vec::new();
    for arg in args {
        let items = list_to_vec(arg.as_ref());
        result.extend(items);
    }
    Ok(val_list(result))
}

fn builtin_reverse(args: &[ValRef]) -> Result<ValRef, String> {
    if args.len() != 1 {
        return Err("reverse requires 1 argument".to_string());
    }
    let mut vec = list_to_vec(args[0].as_ref());
    vec.reverse();
    Ok(val_list(vec))
}

// ============================================================================
// Public API for no_std environments
// ============================================================================

/// Create a new environment with all built-in functions registered
pub fn create_env() -> EnvRef {
    Env::new()
}

/// Parse and evaluate a Lisp expression string
pub fn eval_str(input: &str, env: &EnvRef) -> Result<String, String> {
    let expr = parse(input)?;
    let result = eval(expr, env)?;
    Ok(result.to_string())
}

/// Parse and evaluate multiple Lisp expressions, returning the last result
pub fn eval_str_multiple(input: &str, env: &EnvRef) -> Result<String, String> {
    let expressions = parse_multiple(input)?;
    if expressions.is_empty() {
        return Err("No expressions to evaluate".to_string());
    }

    let mut last_result = val_nil();
    for expr in expressions {
        last_result = eval(expr, env)?;
    }

    Ok(last_result.to_string())
}
