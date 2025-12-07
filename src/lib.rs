#![no_std]

extern crate alloc;

use alloc::format;
use alloc::rc::Rc;
use alloc::string::{String, ToString};
use core::cell::RefCell;
use core::ops::Deref;

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

pub type BuiltinFn = fn(&ValRef) -> Result<ValRef, String>;

#[derive(Clone)]
pub enum Value {
    Number(i64),
    Symbol(String),
    Bool(bool),
    Cons(ValRef, ValRef),
    Builtin(BuiltinFn),
    Lambda {
        params: ValRef, // Cons list of symbols
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

// ============================================================================
// ValRef - Newtype wrapper around Rc<Value>
// ============================================================================

#[derive(Clone, Debug)]
pub struct ValRef(Rc<Value>);

impl ValRef {
    pub fn new(value: Value) -> Self {
        ValRef(Rc::new(value))
    }

    pub fn number(n: i64) -> Self {
        Self::new(Value::Number(n))
    }

    pub fn symbol(s: &str) -> Self {
        Self::new(Value::Symbol(s.to_string()))
    }

    pub fn bool_val(b: bool) -> Self {
        Self::new(Value::Bool(b))
    }

    pub fn cons(car: ValRef, cdr: ValRef) -> Self {
        Self::new(Value::Cons(car, cdr))
    }

    pub fn builtin(f: BuiltinFn) -> Self {
        Self::new(Value::Builtin(f))
    }

    pub fn lambda(params: ValRef, body: ValRef, env: EnvRef) -> Self {
        Self::new(Value::Lambda { params, body, env })
    }

    pub fn nil() -> Self {
        Self::new(Value::Nil)
    }
}

impl Deref for ValRef {
    type Target = Value;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl AsRef<Value> for ValRef {
    fn as_ref(&self) -> &Value {
        &self.0
    }
}

impl From<Value> for ValRef {
    fn from(value: Value) -> Self {
        ValRef::new(value)
    }
}

impl From<Rc<Value>> for ValRef {
    fn from(rc: Rc<Value>) -> Self {
        ValRef(rc)
    }
}

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

    pub fn as_lambda(&self) -> Option<(&ValRef, &ValRef, &EnvRef)> {
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
            Value::Cons(_, _) => self.list_to_string(),
            Value::Builtin(_) => "<builtin>".to_string(),
            Value::Lambda { .. } => "<lambda>".to_string(),
            Value::Nil => "nil".to_string(),
        }
    }

    // Helper function to convert a cons list to string
    fn list_to_string(&self) -> String {
        let mut result = String::from("(");
        let mut current = self;
        let mut first = true;

        loop {
            match current {
                Value::Cons(car, cdr) => {
                    if !first {
                        result.push(' ');
                    }
                    first = false;
                    result.push_str(&car.to_string());
                    current = cdr.as_ref();
                }
                Value::Nil => break,
                _ => {
                    if !first {
                        result.push(' ');
                    }
                    result.push_str(". ");
                    result.push_str(&current.to_string());
                    break;
                }
            }
        }

        result.push(')');
        result
    }

    // Helper function to get length of a list
    fn list_len(&self) -> usize {
        let mut count = 0;
        let mut current = self;

        loop {
            match current {
                Value::Cons(_, cdr) => {
                    count += 1;
                    current = cdr.as_ref();
                }
                Value::Nil => break,
                _ => break,
            }
        }

        count
    }

    // Helper to get nth element of a list
    fn list_nth(&self, n: usize) -> Option<ValRef> {
        let mut current = self;
        let mut idx = 0;

        loop {
            match current {
                Value::Cons(car, cdr) => {
                    if idx == n {
                        return Some(car.clone());
                    }
                    idx += 1;
                    current = cdr.as_ref();
                }
                _ => return None,
            }
        }
    }
}

// ============================================================================
// Environment - Now uses Cons cells as a linked list
// ============================================================================

#[derive(Clone, Debug)]
pub struct EnvRef(Rc<RefCell<Env>>);

#[derive(Debug, Clone)]
pub struct Env {
    bindings: ValRef,
    parent: Option<EnvRef>,
}

impl EnvRef {
    pub fn new() -> Self {
        let mut env = Env {
            bindings: ValRef::nil(),
            parent: None,
        };
        env.register_builtins();
        EnvRef(Rc::new(RefCell::new(env)))
    }

    pub fn with_parent(parent: EnvRef) -> Self {
        let env = Env {
            bindings: ValRef::nil(),
            parent: Some(parent),
        };
        EnvRef(Rc::new(RefCell::new(env)))
    }

    pub fn borrow(&self) -> core::cell::Ref<Env> {
        self.0.borrow()
    }

    pub fn borrow_mut(&self) -> core::cell::RefMut<Env> {
        self.0.borrow_mut()
    }
}

impl Deref for EnvRef {
    type Target = Rc<RefCell<Env>>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl From<Rc<RefCell<Env>>> for EnvRef {
    fn from(rc: Rc<RefCell<Env>>) -> Self {
        EnvRef(rc)
    }
}

impl Env {
    pub fn set(&mut self, name: String, v: ValRef) {
        let binding = ValRef::cons(ValRef::symbol(&name), v);
        self.bindings = ValRef::cons(binding, self.bindings.clone());
    }

    pub fn get(&self, name: &str) -> Option<ValRef> {
        let mut current = self.bindings.as_ref();

        loop {
            match current {
                Value::Cons(binding, rest) => {
                    if let Value::Cons(key, value) = binding.as_ref() {
                        if let Value::Symbol(s) = key.as_ref() {
                            if s == name {
                                return Some(value.clone());
                            }
                        }
                    }
                    current = rest.as_ref();
                }
                Value::Nil => break,
                _ => break,
            }
        }

        if let Some(parent) = &self.parent {
            parent.borrow().get(name)
        } else {
            None
        }
    }

    fn register_builtins(&mut self) {
        self.set("nil".to_string(), ValRef::nil());
        self.set("+".to_string(), ValRef::builtin(builtin_add));
        self.set("-".to_string(), ValRef::builtin(builtin_sub));
        self.set("*".to_string(), ValRef::builtin(builtin_mul));
        self.set("/".to_string(), ValRef::builtin(builtin_div));
        self.set("=".to_string(), ValRef::builtin(builtin_eq));
        self.set("<".to_string(), ValRef::builtin(builtin_lt));
        self.set(">".to_string(), ValRef::builtin(builtin_gt));
        self.set("list".to_string(), ValRef::builtin(builtin_list));
        self.set("car".to_string(), ValRef::builtin(builtin_car));
        self.set("cdr".to_string(), ValRef::builtin(builtin_cdr));
        self.set("cons".to_string(), ValRef::builtin(builtin_cons_fn));
        self.set("null?".to_string(), ValRef::builtin(builtin_null));
        self.set("cons?".to_string(), ValRef::builtin(builtin_cons_p));
        self.set("length".to_string(), ValRef::builtin(builtin_length));
        self.set("append".to_string(), ValRef::builtin(builtin_append));
        self.set("reverse".to_string(), ValRef::builtin(builtin_reverse));
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
            return Err(());
        }
        (true, 1)
    } else if bytes[0] == b'+' {
        if bytes.len() == 1 {
            return Err(());
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

// Build a cons list from tokens
fn tokens_to_list(tokens: &[Token]) -> ValRef {
    let mut result = ValRef::nil();
    let mut i = tokens.len();
    while i > 0 {
        i -= 1;
        let token_val = match &tokens[i] {
            Token::LParen => ValRef::symbol("("),
            Token::RParen => ValRef::symbol(")"),
            Token::Symbol(s) => ValRef::symbol(s),
            Token::Number(n) => ValRef::number(*n),
            Token::Bool(b) => ValRef::bool_val(*b),
            Token::Quote => ValRef::symbol("'"),
        };
        result = ValRef::cons(token_val, result);
    }
    result
}

fn tokenize(input: &str) -> Result<ValRef, String> {
    let mut result = ValRef::nil();
    let mut chars = input.chars().peekable();

    while let Some(&ch) = chars.peek() {
        match ch {
            ' ' | '\t' | '\n' | '\r' => {
                chars.next();
            }
            '(' => {
                result = ValRef::cons(ValRef::symbol("("), result);
                chars.next();
            }
            ')' => {
                result = ValRef::cons(ValRef::symbol(")"), result);
                chars.next();
            }
            '\'' => {
                result = ValRef::cons(ValRef::symbol("'"), result);
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
                        result = ValRef::cons(ValRef::bool_val(true), result);
                        chars.next();
                    }
                    Some(&'f') => {
                        result = ValRef::cons(ValRef::bool_val(false), result);
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

                if let Ok(num) = parse_i64(&atom) {
                    result = ValRef::cons(ValRef::number(num), result);
                } else {
                    result = ValRef::cons(ValRef::symbol(&atom), result);
                }
            }
        }
    }

    // Reverse the list to get correct order
    Ok(reverse_list(result))
}

// Helper to reverse a cons list
fn reverse_list(list: ValRef) -> ValRef {
    let mut result = ValRef::nil();
    let mut current = list.as_ref();

    loop {
        match current {
            Value::Cons(car, cdr) => {
                result = ValRef::cons(car.clone(), result);
                current = cdr.as_ref();
            }
            Value::Nil => break,
            _ => break,
        }
    }

    result
}

// ============================================================================
// Parser
// ============================================================================

fn parse_tokens(tokens: ValRef) -> Result<(ValRef, ValRef), String> {
    match tokens.as_ref() {
        Value::Nil => Err("Unexpected end of input".to_string()),
        Value::Cons(first, rest) => match first.as_ref() {
            Value::Number(_) | Value::Bool(_) => Ok((first.clone(), rest.clone())),
            Value::Symbol(s) if s == "'" => {
                if let Value::Cons(next_expr, remaining) = rest.as_ref() {
                    let (val, consumed) =
                        parse_tokens(ValRef::cons(next_expr.clone(), remaining.clone()))?;
                    let quoted =
                        ValRef::cons(ValRef::symbol("quote"), ValRef::cons(val, ValRef::nil()));
                    Ok((quoted, consumed))
                } else {
                    Err("Quote requires an expression".to_string())
                }
            }
            Value::Symbol(s) if s == "(" => {
                let mut items = ValRef::nil();
                let mut pos = rest.clone();

                loop {
                    match pos.as_ref() {
                        Value::Nil => return Err("Unmatched '('".to_string()),
                        Value::Cons(token, rest_tokens) => {
                            if let Value::Symbol(s) = token.as_ref() {
                                if s == ")" {
                                    return Ok((reverse_list(items), rest_tokens.clone()));
                                }
                            }
                            let (val, consumed) = parse_tokens(pos)?;
                            items = ValRef::cons(val, items);
                            pos = consumed;
                        }
                        _ => return Err("Invalid token stream".to_string()),
                    }
                }
            }
            Value::Symbol(s) if s == ")" => Err("Unexpected ')'".to_string()),
            Value::Symbol(_) => Ok((first.clone(), rest.clone())),
            _ => Err("Unexpected token type".to_string()),
        },
        _ => Err("Invalid token stream".to_string()),
    }
}

pub fn parse(input: &str) -> Result<ValRef, String> {
    let tokens = tokenize(input)?;
    if tokens.is_nil() {
        return Err("Empty input".to_string());
    }
    let (val, remaining) = parse_tokens(tokens)?;
    if !remaining.is_nil() {
        return Err("Unexpected tokens after expression".to_string());
    }
    Ok(val)
}

pub fn parse_multiple(input: &str) -> Result<ValRef, String> {
    let tokens = tokenize(input)?;
    if tokens.is_nil() {
        return Err("Empty input".to_string());
    }

    let mut expressions = ValRef::nil();
    let mut pos = tokens;

    loop {
        if pos.is_nil() {
            break;
        }
        let (val, consumed) = parse_tokens(pos)?;
        expressions = ValRef::cons(val, expressions);
        pos = consumed;
    }

    Ok(reverse_list(expressions))
}

// ============================================================================
// Evaluator - With Proper Tail Call Optimization via Trampoline
// ============================================================================

fn eval_step(expr: ValRef, env: &EnvRef) -> Result<EvalResult, String> {
    match expr.as_ref() {
        Value::Number(_) | Value::Bool(_) | Value::Builtin(_) | Value::Lambda { .. } => {
            Ok(EvalResult::Done(expr.clone()))
        }
        Value::Symbol(s) => {
            if s == "nil" {
                return Ok(EvalResult::Done(ValRef::nil()));
            }
            env.borrow()
                .get(s)
                .map(EvalResult::Done)
                .ok_or_else(|| format!("Unbound symbol: {}", s))
        }
        Value::Cons(car, cdr) => {
            if let Value::Symbol(sym) = car.as_ref() {
                match sym.as_str() {
                    "define" => {
                        let len = expr.as_ref().list_len();
                        if len != 3 {
                            return Err("define requires 2 arguments".to_string());
                        }
                        let name_val = expr.as_ref().list_nth(1).ok_or("define missing name")?;
                        let name = name_val
                            .as_symbol()
                            .ok_or("define requires symbol as first arg")?;
                        let body_val = expr.as_ref().list_nth(2).ok_or("define missing body")?;
                        let val = eval(body_val, env)?;
                        env.borrow_mut().set(name.to_string(), val.clone());
                        return Ok(EvalResult::Done(val));
                    }
                    "lambda" => {
                        let len = expr.as_ref().list_len();
                        if len != 3 {
                            return Err("lambda requires 2 arguments (params body)".to_string());
                        }

                        let params_list =
                            expr.as_ref().list_nth(1).ok_or("lambda missing params")?;

                        // Validate all params are symbols
                        let mut current = params_list.as_ref();
                        loop {
                            match current {
                                Value::Cons(param, rest) => {
                                    if param.as_symbol().is_none() {
                                        return Err("lambda params must be symbols".to_string());
                                    }
                                    current = rest.as_ref();
                                }
                                Value::Nil => break,
                                _ => return Err("lambda params must be a list".to_string()),
                            }
                        }

                        let body = expr.as_ref().list_nth(2).ok_or("lambda missing body")?;
                        let captured_env = env.clone();

                        return Ok(EvalResult::Done(ValRef::lambda(
                            params_list,
                            body,
                            captured_env,
                        )));
                    }
                    "if" => {
                        let len = expr.as_ref().list_len();
                        if len != 4 {
                            return Err("if requires 3 arguments".to_string());
                        }
                        let cond_expr = expr.as_ref().list_nth(1).ok_or("if missing condition")?;
                        let cond = eval(cond_expr, env)?;
                        let is_true = match cond.as_ref() {
                            Value::Bool(b) => *b,
                            Value::Nil => false,
                            _ => true,
                        };
                        let branch_idx = if is_true { 2 } else { 3 };
                        let branch = expr
                            .as_ref()
                            .list_nth(branch_idx)
                            .ok_or("if missing branch")?;
                        return Ok(EvalResult::TailCall(branch, env.clone()));
                    }
                    "quote" => {
                        let len = expr.as_ref().list_len();
                        if len != 2 {
                            return Err("quote requires 1 argument".to_string());
                        }
                        let quoted = expr.as_ref().list_nth(1).ok_or("quote missing argument")?;
                        return Ok(EvalResult::Done(quoted));
                    }
                    _ => {}
                }
            }

            let func = eval(car.clone(), env)?;

            // Evaluate all arguments
            let mut args = ValRef::nil();
            let mut current = cdr.as_ref();
            loop {
                match current {
                    Value::Cons(arg_car, arg_cdr) => {
                        let evaled = eval(arg_car.clone(), env)?;
                        args = ValRef::cons(evaled, args);
                        current = arg_cdr.as_ref();
                    }
                    Value::Nil => break,
                    _ => return Err("Malformed argument list".to_string()),
                }
            }
            args = reverse_list(args);

            match func.as_ref() {
                Value::Builtin(f) => Ok(EvalResult::Done(f(&args)?)),
                Value::Lambda {
                    params,
                    body,
                    env: lambda_env,
                } => {
                    // Count parameters and arguments
                    let param_count = params.as_ref().list_len();
                    let arg_count = args.as_ref().list_len();

                    if arg_count != param_count {
                        return Err(format!(
                            "Lambda expects {} arguments, got {}",
                            param_count, arg_count
                        ));
                    }

                    let call_env = EnvRef::with_parent(lambda_env.clone());

                    // Bind parameters to arguments
                    let mut param_cur = params.as_ref();
                    let mut arg_cur = args.as_ref();

                    loop {
                        match (param_cur, arg_cur) {
                            (Value::Cons(p_car, p_cdr), Value::Cons(a_car, a_cdr)) => {
                                if let Value::Symbol(param_name) = p_car.as_ref() {
                                    call_env.borrow_mut().set(param_name.clone(), a_car.clone());
                                }
                                param_cur = p_cdr.as_ref();
                                arg_cur = a_cdr.as_ref();
                            }
                            (Value::Nil, Value::Nil) => break,
                            _ => {
                                return Err("Parameter/argument mismatch".to_string());
                            }
                        }
                    }

                    Ok(EvalResult::TailCall(body.clone(), call_env))
                }
                _ => Err(format!("Cannot call non-function: {}", func.to_string())),
            }
        }
        Value::Nil => Ok(EvalResult::Done(ValRef::nil())),
    }
}

pub fn eval(mut expr: ValRef, env: &EnvRef) -> Result<ValRef, String> {
    let mut current_env = env.clone();

    loop {
        match eval_step(expr, &current_env)? {
            EvalResult::Done(val) => return Ok(val),
            EvalResult::TailCall(new_expr, new_env) => {
                expr = new_expr;
                current_env = new_env;
            }
        }
    }
}

// ============================================================================
// Built-in Functions
// ============================================================================

fn builtin_add(args: &ValRef) -> Result<ValRef, String> {
    let mut result: i64 = 0;
    let mut current = args.as_ref();

    loop {
        match current {
            Value::Cons(car, cdr) => {
                let num = car.as_number().ok_or("+ requires numbers")?;
                result = result.checked_add(num).ok_or("Integer overflow")?;
                current = cdr.as_ref();
            }
            Value::Nil => break,
            _ => return Err("Invalid argument list".to_string()),
        }
    }

    Ok(ValRef::number(result))
}

fn builtin_sub(args: &ValRef) -> Result<ValRef, String> {
    let len = args.as_ref().list_len();
    if len == 0 {
        return Err("- requires at least 1 argument".to_string());
    }

    let first = args
        .as_ref()
        .list_nth(0)
        .ok_or("- missing first argument")?;
    let first_num = first.as_number().ok_or("- requires numbers")?;

    if len == 1 {
        return Ok(ValRef::number(
            first_num.checked_neg().ok_or("Integer overflow")?,
        ));
    }

    let mut result = first_num;
    let mut current = args.as_ref();
    if let Value::Cons(_, rest) = current {
        current = rest.as_ref();
    }

    loop {
        match current {
            Value::Cons(car, cdr) => {
                let num = car.as_number().ok_or("- requires numbers")?;
                result = result.checked_sub(num).ok_or("Integer overflow")?;
                current = cdr.as_ref();
            }
            Value::Nil => break,
            _ => return Err("Invalid argument list".to_string()),
        }
    }

    Ok(ValRef::number(result))
}

fn builtin_mul(args: &ValRef) -> Result<ValRef, String> {
    let mut result: i64 = 1;
    let mut current = args.as_ref();

    loop {
        match current {
            Value::Cons(car, cdr) => {
                let num = car.as_number().ok_or("* requires numbers")?;
                result = result.checked_mul(num).ok_or("Integer overflow")?;
                current = cdr.as_ref();
            }
            Value::Nil => break,
            _ => return Err("Invalid argument list".to_string()),
        }
    }

    Ok(ValRef::number(result))
}

fn builtin_div(args: &ValRef) -> Result<ValRef, String> {
    let len = args.as_ref().list_len();
    if len < 2 {
        return Err("/ requires at least 2 arguments".to_string());
    }

    let first = args
        .as_ref()
        .list_nth(0)
        .ok_or("/ missing first argument")?;
    let mut result = first.as_number().ok_or("/ requires numbers")?;

    let mut current = args.as_ref();
    if let Value::Cons(_, rest) = current {
        current = rest.as_ref();
    }

    loop {
        match current {
            Value::Cons(car, cdr) => {
                let num = car.as_number().ok_or("/ requires numbers")?;
                if num == 0 {
                    return Err("Division by zero".to_string());
                }
                result = result.checked_div(num).ok_or("Integer overflow")?;
                current = cdr.as_ref();
            }
            Value::Nil => break,
            _ => return Err("Invalid argument list".to_string()),
        }
    }

    Ok(ValRef::number(result))
}

fn builtin_eq(args: &ValRef) -> Result<ValRef, String> {
    if args.as_ref().list_len() != 2 {
        return Err("= requires 2 arguments".to_string());
    }
    let a = args.as_ref().list_nth(0).ok_or("= missing arg 1")?;
    let b = args.as_ref().list_nth(1).ok_or("= missing arg 2")?;
    let a_num = a.as_number().ok_or("= requires numbers")?;
    let b_num = b.as_number().ok_or("= requires numbers")?;
    Ok(ValRef::bool_val(a_num == b_num))
}

fn builtin_lt(args: &ValRef) -> Result<ValRef, String> {
    if args.as_ref().list_len() != 2 {
        return Err("< requires 2 arguments".to_string());
    }
    let a = args.as_ref().list_nth(0).ok_or("< missing arg 1")?;
    let b = args.as_ref().list_nth(1).ok_or("< missing arg 2")?;
    let a_num = a.as_number().ok_or("< requires numbers")?;
    let b_num = b.as_number().ok_or("< requires numbers")?;
    Ok(ValRef::bool_val(a_num < b_num))
}

fn builtin_gt(args: &ValRef) -> Result<ValRef, String> {
    if args.as_ref().list_len() != 2 {
        return Err("> requires 2 arguments".to_string());
    }
    let a = args.as_ref().list_nth(0).ok_or("> missing arg 1")?;
    let b = args.as_ref().list_nth(1).ok_or("> missing arg 2")?;
    let a_num = a.as_number().ok_or("> requires numbers")?;
    let b_num = b.as_number().ok_or("> requires numbers")?;
    Ok(ValRef::bool_val(a_num > b_num))
}

fn builtin_list(args: &ValRef) -> Result<ValRef, String> {
    Ok(args.clone())
}

fn builtin_car(args: &ValRef) -> Result<ValRef, String> {
    if args.as_ref().list_len() != 1 {
        return Err("car requires 1 argument".to_string());
    }
    let list = args.as_ref().list_nth(0).ok_or("car missing argument")?;
    let (car, _) = list.as_cons().ok_or("car requires a cons/list")?;
    Ok(car.clone())
}

fn builtin_cdr(args: &ValRef) -> Result<ValRef, String> {
    if args.as_ref().list_len() != 1 {
        return Err("cdr requires 1 argument".to_string());
    }
    let list = args.as_ref().list_nth(0).ok_or("cdr missing argument")?;
    let (_, cdr) = list.as_cons().ok_or("cdr requires a cons/list")?;
    Ok(cdr.clone())
}

fn builtin_cons_fn(args: &ValRef) -> Result<ValRef, String> {
    if args.as_ref().list_len() != 2 {
        return Err("cons requires 2 arguments".to_string());
    }
    let car = args.as_ref().list_nth(0).ok_or("cons missing arg 1")?;
    let cdr = args.as_ref().list_nth(1).ok_or("cons missing arg 2")?;
    Ok(ValRef::cons(car, cdr))
}

fn builtin_null(args: &ValRef) -> Result<ValRef, String> {
    if args.as_ref().list_len() != 1 {
        return Err("null? requires 1 argument".to_string());
    }
    let val = args.as_ref().list_nth(0).ok_or("null? missing argument")?;
    Ok(ValRef::bool_val(val.is_nil()))
}

fn builtin_cons_p(args: &ValRef) -> Result<ValRef, String> {
    if args.as_ref().list_len() != 1 {
        return Err("cons? requires 1 argument".to_string());
    }
    let val = args.as_ref().list_nth(0).ok_or("cons? missing argument")?;
    Ok(ValRef::bool_val(val.as_cons().is_some()))
}

fn builtin_length(args: &ValRef) -> Result<ValRef, String> {
    if args.as_ref().list_len() != 1 {
        return Err("length requires 1 argument".to_string());
    }
    let list = args.as_ref().list_nth(0).ok_or("length missing argument")?;
    let len = list.as_ref().list_len();
    Ok(ValRef::number(len as i64))
}

fn builtin_append(args: &ValRef) -> Result<ValRef, String> {
    let mut result = ValRef::nil();
    let mut current = args.as_ref();

    // Collect all items from all lists
    loop {
        match current {
            Value::Cons(list, rest) => {
                let mut list_cur = list.as_ref();
                loop {
                    match list_cur {
                        Value::Cons(item, item_rest) => {
                            result = ValRef::cons(item.clone(), result);
                            list_cur = item_rest.as_ref();
                        }
                        Value::Nil => break,
                        _ => break,
                    }
                }
                current = rest.as_ref();
            }
            Value::Nil => break,
            _ => return Err("Invalid argument list".to_string()),
        }
    }

    Ok(reverse_list(result))
}

fn builtin_reverse(args: &ValRef) -> Result<ValRef, String> {
    if args.as_ref().list_len() != 1 {
        return Err("reverse requires 1 argument".to_string());
    }
    let list = args
        .as_ref()
        .list_nth(0)
        .ok_or("reverse missing argument")?;
    Ok(reverse_list(list))
}

// ============================================================================
// Public API
// ============================================================================

/// Parse and evaluate a Lisp expression string
pub fn eval_str(input: &str, env: &EnvRef) -> Result<String, String> {
    let expr = parse(input)?;
    let result = eval(expr, env)?;
    Ok(result.to_string())
}

/// Parse and evaluate multiple Lisp expressions, returning the last result
pub fn eval_str_multiple(input: &str, env: &EnvRef) -> Result<String, String> {
    let expressions = parse_multiple(input)?;
    if expressions.is_nil() {
        return Err("No expressions to evaluate".to_string());
    }

    let mut last_result = ValRef::nil();
    let mut current = expressions.as_ref();

    loop {
        match current {
            Value::Cons(expr, rest) => {
                last_result = eval(expr.clone(), env)?;
                current = rest.as_ref();
            }
            Value::Nil => break,
            _ => return Err("Invalid expression list".to_string()),
        }
    }

    Ok(last_result.to_string())
}
