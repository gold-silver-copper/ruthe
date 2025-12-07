#![no_std]

extern crate alloc;

use alloc::rc::Rc;
use core::cell::RefCell;
use core::ops::Deref;

// ============================================================================
// String representation as cons list of chars
// ============================================================================

#[derive(Clone, Debug)]
pub struct StrRef(ValRef);

impl StrRef {
    pub fn new(s: &str) -> Self {
        let mut result = ValRef::nil();
        for ch in s.chars().rev() {
            result = ValRef::cons(ValRef::char_val(ch), result);
        }
        StrRef(result)
    }

    pub fn empty() -> Self {
        StrRef(ValRef::nil())
    }

    pub fn as_valref(&self) -> &ValRef {
        &self.0
    }

    pub fn to_valref(self) -> ValRef {
        self.0
    }

    // Convert to Rust str for display (uses stack buffer)
    pub fn to_str_buf<'a>(&self, buf: &'a mut [u8]) -> Result<&'a str, ()> {
        let mut idx = 0;
        let mut current = self.0.as_ref();

        loop {
            match current {
                Value::Char(ch) => {
                    let mut char_buf = [0u8; 4];
                    let char_str = ch.encode_utf8(&mut char_buf);
                    let char_bytes = char_str.as_bytes();

                    if idx + char_bytes.len() > buf.len() {
                        return Err(());
                    }

                    for &byte in char_bytes {
                        buf[idx] = byte;
                        idx += 1;
                    }
                }
                Value::Cons(car, cdr) => {
                    if let Value::Char(ch) = car.as_ref() {
                        let mut char_buf = [0u8; 4];
                        let char_str = ch.encode_utf8(&mut char_buf);
                        let char_bytes = char_str.as_bytes();

                        if idx + char_bytes.len() > buf.len() {
                            return Err(());
                        }

                        for &byte in char_bytes {
                            buf[idx] = byte;
                            idx += 1;
                        }
                        current = cdr.as_ref();
                    } else {
                        return Err(());
                    }
                }
                Value::Nil => break,
                _ => return Err(()),
            }
        }

        core::str::from_utf8(&buf[..idx]).map_err(|_| ())
    }

    pub fn eq(&self, other: &StrRef) -> bool {
        let mut cur1 = self.0.as_ref();
        let mut cur2 = other.0.as_ref();

        loop {
            match (cur1, cur2) {
                (Value::Cons(c1, r1), Value::Cons(c2, r2)) => {
                    if let (Value::Char(ch1), Value::Char(ch2)) = (c1.as_ref(), c2.as_ref()) {
                        if ch1 != ch2 {
                            return false;
                        }
                        cur1 = r1.as_ref();
                        cur2 = r2.as_ref();
                    } else {
                        return false;
                    }
                }
                (Value::Nil, Value::Nil) => return true,
                _ => return false,
            }
        }
    }

    pub fn from_static(s: &'static str) -> Self {
        Self::new(s)
    }
}

impl PartialEq for StrRef {
    fn eq(&self, other: &Self) -> bool {
        self.eq(other)
    }
}

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

pub type BuiltinFn = fn(&ValRef) -> Result<ValRef, StrRef>;

#[derive(Clone)]
pub enum Value {
    Number(i64),
    Symbol(StrRef),
    Bool(bool),
    Char(char),
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
            Value::Symbol(_) => write!(f, "Symbol(...)"),
            Value::Bool(b) => write!(f, "Bool({:?})", b),
            Value::Char(c) => write!(f, "Char({:?})", c),
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

    pub fn symbol(s: StrRef) -> Self {
        Self::new(Value::Symbol(s))
    }

    pub fn bool_val(b: bool) -> Self {
        Self::new(Value::Bool(b))
    }

    pub fn char_val(c: char) -> Self {
        Self::new(Value::Char(c))
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
            Value::Char(_) => "char",
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

    pub fn as_char(&self) -> Option<char> {
        match self {
            Value::Char(c) => Some(*c),
            _ => None,
        }
    }

    pub fn as_symbol(&self) -> Option<&StrRef> {
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

    pub fn to_display_str<'a>(&self, buf: &'a mut [u8]) -> Result<&'a str, ()> {
        match self {
            Value::Number(n) => {
                // Convert number to string manually
                let mut temp = [0u8; 32];
                let mut idx = 0;
                let mut num = *n;
                let negative = num < 0;

                if negative {
                    num = -num;
                }

                if num == 0 {
                    temp[idx] = b'0';
                    idx += 1;
                } else {
                    let mut divisor = 1i64;
                    let mut temp_num = num;
                    while temp_num >= 10 {
                        divisor *= 10;
                        temp_num /= 10;
                    }

                    while divisor > 0 {
                        let digit = (num / divisor) as u8;
                        temp[idx] = b'0' + digit;
                        idx += 1;
                        num %= divisor;
                        divisor /= 10;
                    }
                }

                if negative {
                    if idx + 1 > buf.len() {
                        return Err(());
                    }
                    buf[0] = b'-';
                    for i in 0..idx {
                        buf[i + 1] = temp[i];
                    }
                    idx += 1;
                } else {
                    if idx > buf.len() {
                        return Err(());
                    }
                    for i in 0..idx {
                        buf[i] = temp[i];
                    }
                }

                core::str::from_utf8(&buf[..idx]).map_err(|_| ())
            }
            Value::Symbol(s) => s.to_str_buf(buf),
            Value::Bool(b) => {
                let s = if *b { "#t" } else { "#f" };
                let bytes = s.as_bytes();
                if bytes.len() > buf.len() {
                    return Err(());
                }
                for (i, &b) in bytes.iter().enumerate() {
                    buf[i] = b;
                }
                core::str::from_utf8(&buf[..bytes.len()]).map_err(|_| ())
            }
            Value::Char(c) => {
                let mut char_buf = [0u8; 4];
                let s = c.encode_utf8(&mut char_buf);
                let bytes = s.as_bytes();
                if bytes.len() > buf.len() {
                    return Err(());
                }
                for (i, &b) in bytes.iter().enumerate() {
                    buf[i] = b;
                }
                core::str::from_utf8(&buf[..bytes.len()]).map_err(|_| ())
            }
            Value::Cons(_, _) => self.list_to_display_str(buf),
            Value::Builtin(_) => {
                let s = "<builtin>";
                let bytes = s.as_bytes();
                if bytes.len() > buf.len() {
                    return Err(());
                }
                for (i, &b) in bytes.iter().enumerate() {
                    buf[i] = b;
                }
                core::str::from_utf8(&buf[..bytes.len()]).map_err(|_| ())
            }
            Value::Lambda { .. } => {
                let s = "<lambda>";
                let bytes = s.as_bytes();
                if bytes.len() > buf.len() {
                    return Err(());
                }
                for (i, &b) in bytes.iter().enumerate() {
                    buf[i] = b;
                }
                core::str::from_utf8(&buf[..bytes.len()]).map_err(|_| ())
            }
            Value::Nil => {
                let s = "nil";
                let bytes = s.as_bytes();
                if bytes.len() > buf.len() {
                    return Err(());
                }
                for (i, &b) in bytes.iter().enumerate() {
                    buf[i] = b;
                }
                core::str::from_utf8(&buf[..bytes.len()]).map_err(|_| ())
            }
        }
    }

    // Helper function to convert a cons list to string
    fn list_to_display_str<'a>(&self, buf: &'a mut [u8]) -> Result<&'a str, ()> {
        let mut idx = 0;
        if idx >= buf.len() {
            return Err(());
        }
        buf[idx] = b'(';
        idx += 1;

        let mut current = self;
        let mut first = true;

        loop {
            match current {
                Value::Cons(car, cdr) => {
                    if !first {
                        if idx >= buf.len() {
                            return Err(());
                        }
                        buf[idx] = b' ';
                        idx += 1;
                    }
                    first = false;

                    let item_str = car.as_ref().to_display_str(&mut buf[idx..])?;
                    let item_len = item_str.len();
                    idx += item_len;

                    current = cdr.as_ref();
                }
                Value::Nil => break,
                _ => {
                    if !first {
                        if idx + 2 >= buf.len() {
                            return Err(());
                        }
                        buf[idx] = b' ';
                        idx += 1;
                        buf[idx] = b'.';
                        idx += 1;
                        buf[idx] = b' ';
                        idx += 1;
                    }

                    let item_str = current.to_display_str(&mut buf[idx..])?;
                    let item_len = item_str.len();
                    idx += item_len;
                    break;
                }
            }
        }

        if idx >= buf.len() {
            return Err(());
        }
        buf[idx] = b')';
        idx += 1;

        core::str::from_utf8(&buf[..idx]).map_err(|_| ())
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
    pub fn set(&mut self, name: StrRef, v: ValRef) {
        let binding = ValRef::cons(ValRef::symbol(name), v);
        self.bindings = ValRef::cons(binding, self.bindings.clone());
    }

    pub fn get(&self, name: &StrRef) -> Option<ValRef> {
        let mut current = self.bindings.as_ref();

        loop {
            match current {
                Value::Cons(binding, rest) => {
                    if let Value::Cons(key, value) = binding.as_ref() {
                        if let Value::Symbol(s) = key.as_ref() {
                            if s.eq(name) {
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
        self.set(StrRef::new("nil"), ValRef::nil());
        self.set(StrRef::new("+"), ValRef::builtin(builtin_add));
        self.set(StrRef::new("-"), ValRef::builtin(builtin_sub));
        self.set(StrRef::new("*"), ValRef::builtin(builtin_mul));
        self.set(StrRef::new("/"), ValRef::builtin(builtin_div));
        self.set(StrRef::new("="), ValRef::builtin(builtin_eq));
        self.set(StrRef::new("<"), ValRef::builtin(builtin_lt));
        self.set(StrRef::new(">"), ValRef::builtin(builtin_gt));
        self.set(StrRef::new("list"), ValRef::builtin(builtin_list));
        self.set(StrRef::new("car"), ValRef::builtin(builtin_car));
        self.set(StrRef::new("cdr"), ValRef::builtin(builtin_cdr));
        self.set(StrRef::new("cons"), ValRef::builtin(builtin_cons_fn));
        self.set(StrRef::new("null?"), ValRef::builtin(builtin_null));
        self.set(StrRef::new("cons?"), ValRef::builtin(builtin_cons_p));
        self.set(StrRef::new("length"), ValRef::builtin(builtin_length));
        self.set(StrRef::new("append"), ValRef::builtin(builtin_append));
        self.set(StrRef::new("reverse"), ValRef::builtin(builtin_reverse));
    }
}

// ============================================================================
// Tokenizer
// ============================================================================

#[derive(Debug, Clone)]
enum Token {
    LParen,
    RParen,
    Symbol(StrRef),
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

fn tokenize(input: &str) -> Result<ValRef, StrRef> {
    let mut result = ValRef::nil();
    let mut chars = input.chars().peekable();

    while let Some(&ch) = chars.peek() {
        match ch {
            ' ' | '\t' | '\n' | '\r' => {
                chars.next();
            }
            '(' => {
                result = ValRef::cons(ValRef::symbol(StrRef::new("(")), result);
                chars.next();
            }
            ')' => {
                result = ValRef::cons(ValRef::symbol(StrRef::new(")")), result);
                chars.next();
            }
            '\'' => {
                result = ValRef::cons(ValRef::symbol(StrRef::new("'")), result);
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
                    _ => return Err(StrRef::new("Invalid boolean literal")),
                }
            }
            _ => {
                let mut atom_chars = ValRef::nil();
                while let Some(&c) = chars.peek() {
                    if c.is_whitespace() || c == '(' || c == ')' || c == '\'' {
                        break;
                    }
                    atom_chars = ValRef::cons(ValRef::char_val(c), atom_chars);
                    chars.next();
                }

                if atom_chars.is_nil() {
                    continue;
                }

                atom_chars = reverse_list(atom_chars);

                // Convert to string for parsing
                let mut buf = [0u8; 128];
                let mut idx = 0;
                let mut cur = atom_chars.as_ref();
                loop {
                    match cur {
                        Value::Cons(car, cdr) => {
                            if let Value::Char(ch) = car.as_ref() {
                                let mut char_buf = [0u8; 4];
                                let s = ch.encode_utf8(&mut char_buf);
                                for &b in s.as_bytes() {
                                    if idx >= buf.len() {
                                        return Err(StrRef::new("Atom too long"));
                                    }
                                    buf[idx] = b;
                                    idx += 1;
                                }
                            }
                            cur = cdr.as_ref();
                        }
                        Value::Nil => break,
                        _ => break,
                    }
                }

                let atom_str =
                    core::str::from_utf8(&buf[..idx]).map_err(|_| StrRef::new("Invalid UTF-8"))?;

                if let Ok(num) = parse_i64(atom_str) {
                    result = ValRef::cons(ValRef::number(num), result);
                } else {
                    result = ValRef::cons(ValRef::symbol(StrRef::new(atom_str)), result);
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

fn parse_tokens(tokens: ValRef) -> Result<(ValRef, ValRef), StrRef> {
    match tokens.as_ref() {
        Value::Nil => Err(StrRef::new("Unexpected end of input")),
        Value::Cons(first, rest) => match first.as_ref() {
            Value::Number(_) | Value::Bool(_) => Ok((first.clone(), rest.clone())),
            Value::Symbol(s) => {
                let mut buf = [0u8; 32];
                let s_str = s
                    .to_str_buf(&mut buf)
                    .map_err(|_| StrRef::new("Symbol too long"))?;

                if s_str == "'" {
                    if let Value::Cons(next_expr, remaining) = rest.as_ref() {
                        let (val, consumed) =
                            parse_tokens(ValRef::cons(next_expr.clone(), remaining.clone()))?;
                        let quoted = ValRef::cons(
                            ValRef::symbol(StrRef::new("quote")),
                            ValRef::cons(val, ValRef::nil()),
                        );
                        Ok((quoted, consumed))
                    } else {
                        Err(StrRef::new("Quote requires an expression"))
                    }
                } else if s_str == "(" {
                    let mut items = ValRef::nil();
                    let mut pos = rest.clone();

                    loop {
                        match pos.as_ref() {
                            Value::Nil => return Err(StrRef::new("Unmatched '('")),
                            Value::Cons(token, rest_tokens) => {
                                if let Value::Symbol(tok_s) = token.as_ref() {
                                    let mut tok_buf = [0u8; 32];
                                    let tok_str = tok_s
                                        .to_str_buf(&mut tok_buf)
                                        .map_err(|_| StrRef::new("Symbol too long"))?;
                                    if tok_str == ")" {
                                        return Ok((reverse_list(items), rest_tokens.clone()));
                                    }
                                }
                                let (val, consumed) = parse_tokens(pos)?;
                                items = ValRef::cons(val, items);
                                pos = consumed;
                            }
                            _ => return Err(StrRef::new("Invalid token stream")),
                        }
                    }
                } else if s_str == ")" {
                    Err(StrRef::new("Unexpected ')'"))
                } else {
                    Ok((first.clone(), rest.clone()))
                }
            }
            _ => Err(StrRef::new("Unexpected token type")),
        },
        _ => Err(StrRef::new("Invalid token stream")),
    }
}

pub fn parse(input: &str) -> Result<ValRef, StrRef> {
    let tokens = tokenize(input)?;
    if tokens.is_nil() {
        return Err(StrRef::new("Empty input"));
    }
    let (val, remaining) = parse_tokens(tokens)?;
    if !remaining.is_nil() {
        return Err(StrRef::new("Unexpected tokens after expression"));
    }
    Ok(val)
}

pub fn parse_multiple(input: &str) -> Result<ValRef, StrRef> {
    let tokens = tokenize(input)?;
    if tokens.is_nil() {
        return Err(StrRef::new("Empty input"));
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

fn eval_step(expr: ValRef, env: &EnvRef) -> Result<EvalResult, StrRef> {
    match expr.as_ref() {
        Value::Number(_)
        | Value::Bool(_)
        | Value::Char(_)
        | Value::Builtin(_)
        | Value::Lambda { .. } => Ok(EvalResult::Done(expr.clone())),
        Value::Symbol(s) => {
            let mut buf = [0u8; 32];
            let s_str = s
                .to_str_buf(&mut buf)
                .map_err(|_| StrRef::new("Symbol too long"))?;

            if s_str == "nil" {
                return Ok(EvalResult::Done(ValRef::nil()));
            }
            env.borrow()
                .get(s)
                .map(EvalResult::Done)
                .ok_or_else(|| StrRef::new("Unbound symbol"))
        }
        Value::Cons(car, cdr) => {
            if let Value::Symbol(sym) = car.as_ref() {
                let mut buf = [0u8; 32];
                let sym_str = sym
                    .to_str_buf(&mut buf)
                    .map_err(|_| StrRef::new("Symbol too long"))?;

                match sym_str {
                    "define" => {
                        let len = expr.as_ref().list_len();
                        if len != 3 {
                            return Err(StrRef::new("define requires 2 arguments"));
                        }
                        let name_val = expr
                            .as_ref()
                            .list_nth(1)
                            .ok_or(StrRef::new("define missing name"))?;
                        let name = name_val
                            .as_symbol()
                            .ok_or(StrRef::new("define requires symbol as first arg"))?;
                        let body_val = expr
                            .as_ref()
                            .list_nth(2)
                            .ok_or(StrRef::new("define missing body"))?;
                        let val = eval(body_val, env)?;
                        env.borrow_mut().set(name.clone(), val.clone());
                        return Ok(EvalResult::Done(val));
                    }
                    "lambda" => {
                        let len = expr.as_ref().list_len();
                        if len != 3 {
                            return Err(StrRef::new("lambda requires 2 arguments (params body)"));
                        }

                        let params_list = expr
                            .as_ref()
                            .list_nth(1)
                            .ok_or(StrRef::new("lambda missing params"))?;

                        // Validate all params are symbols
                        let mut current = params_list.as_ref();
                        loop {
                            match current {
                                Value::Cons(param, rest) => {
                                    if param.as_symbol().is_none() {
                                        return Err(StrRef::new("lambda params must be symbols"));
                                    }
                                    current = rest.as_ref();
                                }
                                Value::Nil => break,
                                _ => return Err(StrRef::new("lambda params must be a list")),
                            }
                        }

                        let body = expr
                            .as_ref()
                            .list_nth(2)
                            .ok_or(StrRef::new("lambda missing body"))?;
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
                            return Err(StrRef::new("if requires 3 arguments"));
                        }
                        let cond_expr = expr
                            .as_ref()
                            .list_nth(1)
                            .ok_or(StrRef::new("if missing condition"))?;
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
                            .ok_or(StrRef::new("if missing branch"))?;
                        return Ok(EvalResult::TailCall(branch, env.clone()));
                    }
                    "quote" => {
                        let len = expr.as_ref().list_len();
                        if len != 2 {
                            return Err(StrRef::new("quote requires 1 argument"));
                        }
                        let quoted = expr
                            .as_ref()
                            .list_nth(1)
                            .ok_or(StrRef::new("quote missing argument"))?;
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
                    _ => return Err(StrRef::new("Malformed argument list")),
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
                        return Err(StrRef::new("Lambda argument count mismatch"));
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
                                return Err(StrRef::new("Parameter/argument mismatch"));
                            }
                        }
                    }

                    Ok(EvalResult::TailCall(body.clone(), call_env))
                }
                _ => Err(StrRef::new("Cannot call non-function")),
            }
        }
        Value::Nil => Ok(EvalResult::Done(ValRef::nil())),
    }
}

pub fn eval(mut expr: ValRef, env: &EnvRef) -> Result<ValRef, StrRef> {
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

fn builtin_add(args: &ValRef) -> Result<ValRef, StrRef> {
    let mut result: i64 = 0;
    let mut current = args.as_ref();

    loop {
        match current {
            Value::Cons(car, cdr) => {
                let num = car.as_number().ok_or(StrRef::new("+ requires numbers"))?;
                result = result
                    .checked_add(num)
                    .ok_or(StrRef::new("Integer overflow"))?;
                current = cdr.as_ref();
            }
            Value::Nil => break,
            _ => return Err(StrRef::new("Invalid argument list")),
        }
    }

    Ok(ValRef::number(result))
}

fn builtin_sub(args: &ValRef) -> Result<ValRef, StrRef> {
    let len = args.as_ref().list_len();
    if len == 0 {
        return Err(StrRef::new("- requires at least 1 argument"));
    }

    let first = args
        .as_ref()
        .list_nth(0)
        .ok_or(StrRef::new("- missing first argument"))?;
    let first_num = first.as_number().ok_or(StrRef::new("- requires numbers"))?;

    if len == 1 {
        return Ok(ValRef::number(
            first_num
                .checked_neg()
                .ok_or(StrRef::new("Integer overflow"))?,
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
                let num = car.as_number().ok_or(StrRef::new("- requires numbers"))?;
                result = result
                    .checked_sub(num)
                    .ok_or(StrRef::new("Integer overflow"))?;
                current = cdr.as_ref();
            }
            Value::Nil => break,
            _ => return Err(StrRef::new("Invalid argument list")),
        }
    }

    Ok(ValRef::number(result))
}

fn builtin_mul(args: &ValRef) -> Result<ValRef, StrRef> {
    let mut result: i64 = 1;
    let mut current = args.as_ref();

    loop {
        match current {
            Value::Cons(car, cdr) => {
                let num = car.as_number().ok_or(StrRef::new("* requires numbers"))?;
                result = result
                    .checked_mul(num)
                    .ok_or(StrRef::new("Integer overflow"))?;
                current = cdr.as_ref();
            }
            Value::Nil => break,
            _ => return Err(StrRef::new("Invalid argument list")),
        }
    }

    Ok(ValRef::number(result))
}

fn builtin_div(args: &ValRef) -> Result<ValRef, StrRef> {
    let len = args.as_ref().list_len();
    if len < 2 {
        return Err(StrRef::new("/ requires at least 2 arguments"));
    }

    let first = args
        .as_ref()
        .list_nth(0)
        .ok_or(StrRef::new("/ missing first argument"))?;
    let mut result = first.as_number().ok_or(StrRef::new("/ requires numbers"))?;

    let mut current = args.as_ref();
    if let Value::Cons(_, rest) = current {
        current = rest.as_ref();
    }

    loop {
        match current {
            Value::Cons(car, cdr) => {
                let num = car.as_number().ok_or(StrRef::new("/ requires numbers"))?;
                if num == 0 {
                    return Err(StrRef::new("Division by zero"));
                }
                result = result
                    .checked_div(num)
                    .ok_or(StrRef::new("Integer overflow"))?;
                current = cdr.as_ref();
            }
            Value::Nil => break,
            _ => return Err(StrRef::new("Invalid argument list")),
        }
    }

    Ok(ValRef::number(result))
}

fn builtin_eq(args: &ValRef) -> Result<ValRef, StrRef> {
    if args.as_ref().list_len() != 2 {
        return Err(StrRef::new("= requires 2 arguments"));
    }
    let a = args
        .as_ref()
        .list_nth(0)
        .ok_or(StrRef::new("= missing arg 1"))?;
    let b = args
        .as_ref()
        .list_nth(1)
        .ok_or(StrRef::new("= missing arg 2"))?;
    let a_num = a.as_number().ok_or(StrRef::new("= requires numbers"))?;
    let b_num = b.as_number().ok_or(StrRef::new("= requires numbers"))?;
    Ok(ValRef::bool_val(a_num == b_num))
}

fn builtin_lt(args: &ValRef) -> Result<ValRef, StrRef> {
    if args.as_ref().list_len() != 2 {
        return Err(StrRef::new("< requires 2 arguments"));
    }
    let a = args
        .as_ref()
        .list_nth(0)
        .ok_or(StrRef::new("< missing arg 1"))?;
    let b = args
        .as_ref()
        .list_nth(1)
        .ok_or(StrRef::new("< missing arg 2"))?;
    let a_num = a.as_number().ok_or(StrRef::new("< requires numbers"))?;
    let b_num = b.as_number().ok_or(StrRef::new("< requires numbers"))?;
    Ok(ValRef::bool_val(a_num < b_num))
}

fn builtin_gt(args: &ValRef) -> Result<ValRef, StrRef> {
    if args.as_ref().list_len() != 2 {
        return Err(StrRef::new("> requires 2 arguments"));
    }
    let a = args
        .as_ref()
        .list_nth(0)
        .ok_or(StrRef::new("> missing arg 1"))?;
    let b = args
        .as_ref()
        .list_nth(1)
        .ok_or(StrRef::new("> missing arg 2"))?;
    let a_num = a.as_number().ok_or(StrRef::new("> requires numbers"))?;
    let b_num = b.as_number().ok_or(StrRef::new("> requires numbers"))?;
    Ok(ValRef::bool_val(a_num > b_num))
}

fn builtin_list(args: &ValRef) -> Result<ValRef, StrRef> {
    Ok(args.clone())
}

fn builtin_car(args: &ValRef) -> Result<ValRef, StrRef> {
    if args.as_ref().list_len() != 1 {
        return Err(StrRef::new("car requires 1 argument"));
    }
    let list = args
        .as_ref()
        .list_nth(0)
        .ok_or(StrRef::new("car missing argument"))?;
    let (car, _) = list
        .as_cons()
        .ok_or(StrRef::new("car requires a cons/list"))?;
    Ok(car.clone())
}

fn builtin_cdr(args: &ValRef) -> Result<ValRef, StrRef> {
    if args.as_ref().list_len() != 1 {
        return Err(StrRef::new("cdr requires 1 argument"));
    }
    let list = args
        .as_ref()
        .list_nth(0)
        .ok_or(StrRef::new("cdr missing argument"))?;
    let (_, cdr) = list
        .as_cons()
        .ok_or(StrRef::new("cdr requires a cons/list"))?;
    Ok(cdr.clone())
}

fn builtin_cons_fn(args: &ValRef) -> Result<ValRef, StrRef> {
    if args.as_ref().list_len() != 2 {
        return Err(StrRef::new("cons requires 2 arguments"));
    }
    let car = args
        .as_ref()
        .list_nth(0)
        .ok_or(StrRef::new("cons missing arg 1"))?;
    let cdr = args
        .as_ref()
        .list_nth(1)
        .ok_or(StrRef::new("cons missing arg 2"))?;
    Ok(ValRef::cons(car, cdr))
}

fn builtin_null(args: &ValRef) -> Result<ValRef, StrRef> {
    if args.as_ref().list_len() != 1 {
        return Err(StrRef::new("null? requires 1 argument"));
    }
    let val = args
        .as_ref()
        .list_nth(0)
        .ok_or(StrRef::new("null? missing argument"))?;
    Ok(ValRef::bool_val(val.is_nil()))
}

fn builtin_cons_p(args: &ValRef) -> Result<ValRef, StrRef> {
    if args.as_ref().list_len() != 1 {
        return Err(StrRef::new("cons? requires 1 argument"));
    }
    let val = args
        .as_ref()
        .list_nth(0)
        .ok_or(StrRef::new("cons? missing argument"))?;
    Ok(ValRef::bool_val(val.as_cons().is_some()))
}

fn builtin_length(args: &ValRef) -> Result<ValRef, StrRef> {
    if args.as_ref().list_len() != 1 {
        return Err(StrRef::new("length requires 1 argument"));
    }
    let list = args
        .as_ref()
        .list_nth(0)
        .ok_or(StrRef::new("length missing argument"))?;
    let len = list.as_ref().list_len();
    Ok(ValRef::number(len as i64))
}

fn builtin_append(args: &ValRef) -> Result<ValRef, StrRef> {
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
            _ => return Err(StrRef::new("Invalid argument list")),
        }
    }

    Ok(reverse_list(result))
}

fn builtin_reverse(args: &ValRef) -> Result<ValRef, StrRef> {
    if args.as_ref().list_len() != 1 {
        return Err(StrRef::new("reverse requires 1 argument"));
    }
    let list = args
        .as_ref()
        .list_nth(0)
        .ok_or(StrRef::new("reverse missing argument"))?;
    Ok(reverse_list(list))
}

// ============================================================================
// Public API
// ============================================================================

/// Parse and evaluate a Lisp expression string
pub fn eval_str(input: &str, env: &EnvRef) -> Result<StrRef, StrRef> {
    let expr = parse(input)?;
    let result = eval(expr, env)?;

    // Convert result to StrRef for return
    let mut buf = [0u8; 4096];
    let s = result
        .as_ref()
        .to_display_str(&mut buf)
        .map_err(|_| StrRef::new("Result too large to display"))?;
    Ok(StrRef::new(s))
}

/// Parse and evaluate multiple Lisp expressions, returning the last result
pub fn eval_str_multiple(input: &str, env: &EnvRef) -> Result<StrRef, StrRef> {
    let expressions = parse_multiple(input)?;
    if expressions.is_nil() {
        return Err(StrRef::new("No expressions to evaluate"));
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
            _ => return Err(StrRef::new("Invalid expression list")),
        }
    }

    // Convert result to StrRef for return
    let mut buf = [0u8; 4096];
    let s = last_result
        .as_ref()
        .to_display_str(&mut buf)
        .map_err(|_| StrRef::new("Result too large to display"))?;
    Ok(StrRef::new(s))
}
