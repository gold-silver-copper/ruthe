#![no_std]

extern crate alloc;
use alloc::rc::Rc;
use core::cell::RefCell;
use core::ops::Deref;

// ============================================================================
// Optimized Value Type - Everything is cons cells!
// ============================================================================

pub type BuiltinFn = fn(&ValRef) -> Result<ValRef, ValRef>;

#[derive(Clone)]
pub enum Value {
    Number(i64),
    Symbol(ValRef), // String as cons list of chars
    Bool(bool),
    Char(char),
    Cons(Rc<RefCell<(ValRef, ValRef)>>), // Unified cons cell - always mutable
    Builtin(BuiltinFn),
    Lambda {
        params: ValRef,
        body: ValRef,
        env: ValRef, // Environment is just a ValRef (cons list)
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
            Value::Cons(_) => write!(f, "Cons(...)"),
            Value::Builtin(_) => write!(f, "Builtin(<fn>)"),
            Value::Lambda { .. } => write!(f, "Lambda(<fn>)"),
            Value::Nil => write!(f, "Nil"),
        }
    }
}

// ============================================================================
// Trampoline System for Proper TCO
// ============================================================================

pub enum EvalResult {
    Done(ValRef),
    TailCall(ValRef, ValRef), // expr, env (both ValRef)
}

// ============================================================================
// ValRef - Newtype wrapper around Rc<Value>
// ============================================================================

#[derive(Clone, Debug)]
pub struct ValRef(pub Rc<Value>);

impl ValRef {
    pub fn new(value: Value) -> Self {
        ValRef(Rc::new(value))
    }

    pub fn to_display_str<'a>(&self, buf: &'a mut [u8]) -> Result<&'a str, ()> {
        self.as_ref().to_display_str(buf)
    }

    pub fn number(n: i64) -> Self {
        Self::new(Value::Number(n))
    }

    pub fn new_str(s: &str) -> Self {
        let mut result = ValRef::nil();
        for ch in s.chars().rev() {
            result = ValRef::cons(ValRef::char_val(ch), result);
        }
        result
    }

    pub fn symbol(s: ValRef) -> Self {
        Self::new(Value::Symbol(s))
    }

    pub fn bool_val(b: bool) -> Self {
        Self::new(Value::Bool(b))
    }

    pub fn char_val(c: char) -> Self {
        Self::new(Value::Char(c))
    }

    pub fn cons(car: ValRef, cdr: ValRef) -> Self {
        Self::new(Value::Cons(Rc::new(RefCell::new((car, cdr)))))
    }

    pub fn builtin(f: BuiltinFn) -> Self {
        Self::new(Value::Builtin(f))
    }

    pub fn lambda(params: ValRef, body: ValRef, env: ValRef) -> Self {
        Self::new(Value::Lambda { params, body, env })
    }

    pub fn nil() -> Self {
        Self::new(Value::Nil)
    }

    pub fn to_str_buf<'a>(&self, buf: &'a mut [u8]) -> Result<&'a str, ()> {
        let mut idx = 0;
        let mut current = self.clone();

        loop {
            match current.as_ref() {
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
                    break;
                }
                Value::Cons(cell) => {
                    let (car, cdr) = cell.borrow().clone();
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
                        current = cdr;
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

    pub fn str_eq(&self, other: &ValRef) -> bool {
        let mut cur1 = self.clone();
        let mut cur2 = other.clone();

        loop {
            match (cur1.as_ref(), cur2.as_ref()) {
                (Value::Cons(cell1), Value::Cons(cell2)) => {
                    let (c1, r1) = cell1.borrow().clone();
                    let (c2, r2) = cell2.borrow().clone();
                    if let (Value::Char(ch1), Value::Char(ch2)) = (c1.as_ref(), c2.as_ref()) {
                        if ch1 != ch2 {
                            return false;
                        }
                        cur1 = r1;
                        cur2 = r2;
                    } else {
                        return false;
                    }
                }
                (Value::Nil, Value::Nil) => return true,
                (Value::Char(c1), Value::Char(c2)) => return c1 == c2,
                _ => return false,
            }
        }
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
            Value::Cons(_) => "cons",
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

    pub fn as_symbol(&self) -> Option<&ValRef> {
        match self {
            Value::Symbol(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_cons(&self) -> Option<&Rc<RefCell<(ValRef, ValRef)>>> {
        match self {
            Value::Cons(cell) => Some(cell),
            _ => None,
        }
    }

    pub fn as_builtin(&self) -> Option<BuiltinFn> {
        match self {
            Value::Builtin(f) => Some(*f),
            _ => None,
        }
    }

    pub fn as_lambda(&self) -> Option<(&ValRef, &ValRef, &ValRef)> {
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
            Value::Cons(_) => self.list_to_display_str(buf),
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

    fn list_to_display_str<'a>(&self, buf: &'a mut [u8]) -> Result<&'a str, ()> {
        let mut idx = 0;
        if idx >= buf.len() {
            return Err(());
        }
        buf[idx] = b'(';
        idx += 1;

        let mut current = ValRef::new(self.clone());
        let mut first = true;

        loop {
            match current.as_ref() {
                Value::Cons(cell) => {
                    let (car, cdr) = cell.borrow().clone();
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

                    current = cdr;
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

                    let item_str = current.as_ref().to_display_str(&mut buf[idx..])?;
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

    fn list_len(&self) -> usize {
        let mut count = 0;
        let mut current = ValRef::new(self.clone());

        loop {
            match current.as_ref() {
                Value::Cons(cell) => {
                    count += 1;
                    let (_, cdr) = cell.borrow().clone();
                    current = cdr;
                }
                Value::Nil => break,
                _ => break,
            }
        }

        count
    }

    fn list_nth(&self, n: usize) -> Option<ValRef> {
        let mut current = ValRef::new(self.clone());
        let mut idx = 0;

        loop {
            match current.as_ref() {
                Value::Cons(cell) => {
                    let (car, cdr) = cell.borrow().clone();
                    if idx == n {
                        return Some(car);
                    }
                    idx += 1;
                    current = cdr;
                }
                _ => return None,
            }
        }
    }
}

// ============================================================================
// Environment Operations - Mutable cons list manipulation
// ============================================================================

/// Environment structure:
/// Cons((bindings . parent_env))
/// where bindings is a cons list of (symbol . value) pairs

fn env_new() -> ValRef {
    let env = ValRef::cons(ValRef::nil(), ValRef::nil());
    register_builtins(&env);
    env
}

fn env_with_parent(parent: ValRef) -> ValRef {
    ValRef::cons(ValRef::nil(), parent)
}

fn env_set(env: &ValRef, name: ValRef, value: ValRef) {
    // env is Cons(bindings, parent)
    if let Some(cell) = env.as_cons() {
        let (bindings, parent) = cell.borrow().clone();
        let new_binding = ValRef::cons(ValRef::symbol(name), value);
        let new_bindings = ValRef::cons(new_binding, bindings);
        *cell.borrow_mut() = (new_bindings, parent);
    }
}

fn env_get(env: &ValRef, name: &ValRef) -> Option<ValRef> {
    match env.as_ref() {
        Value::Cons(cell) => {
            let (bindings, parent) = cell.borrow().clone();

            // Search in current bindings
            let mut current = bindings;
            loop {
                match current.as_ref() {
                    Value::Cons(binding_cell) => {
                        let (binding, rest) = binding_cell.borrow().clone();
                        if let Value::Cons(key_value_cell) = binding.as_ref() {
                            let (key, value) = key_value_cell.borrow().clone();
                            if let Value::Symbol(s) = key.as_ref() {
                                if s.str_eq(name) {
                                    return Some(value);
                                }
                            }
                        }
                        current = rest;
                    }
                    Value::Nil => break,
                    _ => break,
                }
            }

            // Search in parent
            if !parent.is_nil() {
                env_get(&parent, name)
            } else {
                None
            }
        }
        _ => None,
    }
}

fn register_builtins(env: &ValRef) {
    env_set(env, ValRef::new_str("nil"), ValRef::nil());
    env_set(env, ValRef::new_str("+"), ValRef::builtin(builtin_add));
    env_set(env, ValRef::new_str("-"), ValRef::builtin(builtin_sub));
    env_set(env, ValRef::new_str("*"), ValRef::builtin(builtin_mul));
    env_set(env, ValRef::new_str("/"), ValRef::builtin(builtin_div));
    env_set(env, ValRef::new_str("="), ValRef::builtin(builtin_eq));
    env_set(env, ValRef::new_str("<"), ValRef::builtin(builtin_lt));
    env_set(env, ValRef::new_str(">"), ValRef::builtin(builtin_gt));
    env_set(env, ValRef::new_str("list"), ValRef::builtin(builtin_list));
    env_set(env, ValRef::new_str("car"), ValRef::builtin(builtin_car));
    env_set(env, ValRef::new_str("cdr"), ValRef::builtin(builtin_cdr));
    env_set(
        env,
        ValRef::new_str("cons"),
        ValRef::builtin(builtin_cons_fn),
    );
    env_set(env, ValRef::new_str("null?"), ValRef::builtin(builtin_null));
    env_set(
        env,
        ValRef::new_str("cons?"),
        ValRef::builtin(builtin_cons_p),
    );
    env_set(
        env,
        ValRef::new_str("length"),
        ValRef::builtin(builtin_length),
    );
    env_set(
        env,
        ValRef::new_str("append"),
        ValRef::builtin(builtin_append),
    );
    env_set(
        env,
        ValRef::new_str("reverse"),
        ValRef::builtin(builtin_reverse),
    );
}

// ============================================================================
// Tokenizer
// ============================================================================

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

fn tokenize(input: &str) -> Result<ValRef, ValRef> {
    let mut result = ValRef::nil();
    let mut chars = input.chars().peekable();

    while let Some(&ch) = chars.peek() {
        match ch {
            ' ' | '\t' | '\n' | '\r' => {
                chars.next();
            }
            '(' => {
                result = ValRef::cons(ValRef::symbol(ValRef::new_str("(")), result);
                chars.next();
            }
            ')' => {
                result = ValRef::cons(ValRef::symbol(ValRef::new_str(")")), result);
                chars.next();
            }
            '\'' => {
                result = ValRef::cons(ValRef::symbol(ValRef::new_str("'")), result);
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
                    _ => return Err(ValRef::new_str("Invalid boolean literal")),
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

                let mut buf = [0u8; 128];
                let mut idx = 0;
                let mut cur = atom_chars;
                loop {
                    match cur.as_ref() {
                        Value::Cons(cell) => {
                            let (car, cdr) = cell.borrow().clone();
                            if let Value::Char(ch) = car.as_ref() {
                                let mut char_buf = [0u8; 4];
                                let s = ch.encode_utf8(&mut char_buf);
                                for &b in s.as_bytes() {
                                    if idx >= buf.len() {
                                        return Err(ValRef::new_str("Atom too long"));
                                    }
                                    buf[idx] = b;
                                    idx += 1;
                                }
                            }
                            cur = cdr;
                        }
                        Value::Nil => break,
                        _ => break,
                    }
                }

                let atom_str = core::str::from_utf8(&buf[..idx])
                    .map_err(|_| ValRef::new_str("Invalid UTF-8"))?;

                if let Ok(num) = parse_i64(atom_str) {
                    result = ValRef::cons(ValRef::number(num), result);
                } else {
                    result = ValRef::cons(ValRef::symbol(ValRef::new_str(atom_str)), result);
                }
            }
        }
    }

    Ok(reverse_list(result))
}

fn reverse_list(list: ValRef) -> ValRef {
    let mut result = ValRef::nil();
    let mut current = list;

    loop {
        match current.as_ref() {
            Value::Cons(cell) => {
                let (car, cdr) = cell.borrow().clone();
                result = ValRef::cons(car, result);
                current = cdr;
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

fn parse_tokens(tokens: ValRef) -> Result<(ValRef, ValRef), ValRef> {
    match tokens.as_ref() {
        Value::Nil => Err(ValRef::new_str("Unexpected end of input")),
        Value::Cons(cell) => {
            let (first, rest) = cell.borrow().clone();
            match first.as_ref() {
                Value::Number(_) | Value::Bool(_) => Ok((first, rest)),
                Value::Symbol(s) => {
                    let mut buf = [0u8; 32];
                    let s_str = s
                        .to_str_buf(&mut buf)
                        .map_err(|_| ValRef::new_str("Symbol too long"))?;

                    if s_str == "'" {
                        if let Value::Cons(next_cell) = rest.as_ref() {
                            let (next_expr, remaining) = next_cell.borrow().clone();
                            let (val, consumed) = parse_tokens(ValRef::cons(next_expr, remaining))?;
                            let quoted = ValRef::cons(
                                ValRef::symbol(ValRef::new_str("quote")),
                                ValRef::cons(val, ValRef::nil()),
                            );
                            Ok((quoted, consumed))
                        } else {
                            Err(ValRef::new_str("Quote requires an expression"))
                        }
                    } else if s_str == "(" {
                        let mut items = ValRef::nil();
                        let mut pos = rest;

                        loop {
                            match pos.as_ref() {
                                Value::Nil => return Err(ValRef::new_str("Unmatched '('")),
                                Value::Cons(token_cell) => {
                                    let (token, rest_tokens) = token_cell.borrow().clone();
                                    if let Value::Symbol(tok_s) = token.as_ref() {
                                        let mut tok_buf = [0u8; 32];
                                        let tok_str = tok_s
                                            .to_str_buf(&mut tok_buf)
                                            .map_err(|_| ValRef::new_str("Symbol too long"))?;
                                        if tok_str == ")" {
                                            return Ok((reverse_list(items), rest_tokens));
                                        }
                                    }
                                    let (val, consumed) = parse_tokens(pos)?;
                                    items = ValRef::cons(val, items);
                                    pos = consumed;
                                }
                                _ => return Err(ValRef::new_str("Invalid token stream")),
                            }
                        }
                    } else if s_str == ")" {
                        Err(ValRef::new_str("Unexpected ')'"))
                    } else {
                        Ok((first, rest))
                    }
                }
                _ => Err(ValRef::new_str("Unexpected token type")),
            }
        }
        _ => Err(ValRef::new_str("Invalid token stream")),
    }
}

pub fn parse(input: &str) -> Result<ValRef, ValRef> {
    let tokens = tokenize(input)?;
    if tokens.is_nil() {
        return Err(ValRef::new_str("Empty input"));
    }
    let (val, remaining) = parse_tokens(tokens)?;
    if !remaining.is_nil() {
        return Err(ValRef::new_str("Unexpected tokens after expression"));
    }
    Ok(val)
}

pub fn parse_multiple(input: &str) -> Result<ValRef, ValRef> {
    let tokens = tokenize(input)?;
    if tokens.is_nil() {
        return Err(ValRef::new_str("Empty input"));
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

fn eval_step(expr: ValRef, env: &ValRef) -> Result<EvalResult, ValRef> {
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
                .map_err(|_| ValRef::new_str("Symbol too long"))?;

            if s_str == "nil" {
                return Ok(EvalResult::Done(ValRef::nil()));
            }
            env_get(env, s)
                .map(EvalResult::Done)
                .ok_or_else(|| ValRef::new_str("Unbound symbol"))
        }
        Value::Cons(cell) => {
            let (car, cdr) = cell.borrow().clone();
            if let Value::Symbol(sym) = car.as_ref() {
                let mut buf = [0u8; 32];
                let sym_str = sym
                    .to_str_buf(&mut buf)
                    .map_err(|_| ValRef::new_str("Symbol too long"))?;

                match sym_str {
                    "define" => {
                        let len = expr.as_ref().list_len();
                        if len != 3 {
                            return Err(ValRef::new_str("define requires 2 arguments"));
                        }
                        let name_val = expr
                            .as_ref()
                            .list_nth(1)
                            .ok_or(ValRef::new_str("define missing name"))?;
                        let name = name_val
                            .as_symbol()
                            .ok_or(ValRef::new_str("define requires symbol as first arg"))?
                            .clone();
                        let body_val = expr
                            .as_ref()
                            .list_nth(2)
                            .ok_or(ValRef::new_str("define missing body"))?;
                        let val = eval(body_val, env)?;
                        env_set(env, name, val.clone());
                        return Ok(EvalResult::Done(val));
                    }
                    "lambda" => {
                        let len = expr.as_ref().list_len();
                        if len != 3 {
                            return Err(ValRef::new_str(
                                "lambda requires 2 arguments (params body)",
                            ));
                        }

                        let params_list = expr
                            .as_ref()
                            .list_nth(1)
                            .ok_or(ValRef::new_str("lambda missing params"))?;

                        let mut current = params_list.clone();
                        loop {
                            match current.as_ref() {
                                Value::Cons(param_cell) => {
                                    let (param, rest) = param_cell.borrow().clone();
                                    if param.as_symbol().is_none() {
                                        return Err(ValRef::new_str(
                                            "lambda params must be symbols",
                                        ));
                                    }
                                    current = rest;
                                }
                                Value::Nil => break,
                                _ => return Err(ValRef::new_str("lambda params must be a list")),
                            }
                        }

                        let body = expr
                            .as_ref()
                            .list_nth(2)
                            .ok_or(ValRef::new_str("lambda missing body"))?;

                        return Ok(EvalResult::Done(ValRef::lambda(
                            params_list,
                            body,
                            env.clone(),
                        )));
                    }
                    "if" => {
                        let len = expr.as_ref().list_len();
                        if len != 4 {
                            return Err(ValRef::new_str("if requires 3 arguments"));
                        }
                        let cond_expr = expr
                            .as_ref()
                            .list_nth(1)
                            .ok_or(ValRef::new_str("if missing condition"))?;
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
                            .ok_or(ValRef::new_str("if missing branch"))?;
                        return Ok(EvalResult::TailCall(branch, env.clone()));
                    }
                    "quote" => {
                        let len = expr.as_ref().list_len();
                        if len != 2 {
                            return Err(ValRef::new_str("quote requires 1 argument"));
                        }
                        let quoted = expr
                            .as_ref()
                            .list_nth(1)
                            .ok_or(ValRef::new_str("quote missing argument"))?;
                        return Ok(EvalResult::Done(quoted));
                    }
                    _ => {}
                }
            }

            let func = eval(car, env)?;

            let mut args = ValRef::nil();
            let mut current = cdr;
            loop {
                match current.as_ref() {
                    Value::Cons(arg_cell) => {
                        let (arg_car, arg_cdr) = arg_cell.borrow().clone();
                        let evaled = eval(arg_car, env)?;
                        args = ValRef::cons(evaled, args);
                        current = arg_cdr;
                    }
                    Value::Nil => break,
                    _ => return Err(ValRef::new_str("Malformed argument list")),
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
                    let param_count = params.as_ref().list_len();
                    let arg_count = args.as_ref().list_len();

                    if arg_count != param_count {
                        return Err(ValRef::new_str("Lambda argument count mismatch"));
                    }

                    let call_env = env_with_parent(lambda_env.clone());

                    let mut param_cur = params.clone();
                    let mut arg_cur = args.clone();

                    loop {
                        match (param_cur.as_ref(), arg_cur.as_ref()) {
                            (Value::Cons(p_cell), Value::Cons(a_cell)) => {
                                let (p_car, p_cdr) = p_cell.borrow().clone();
                                let (a_car, a_cdr) = a_cell.borrow().clone();
                                if let Value::Symbol(param_name) = p_car.as_ref() {
                                    env_set(&call_env, param_name.clone(), a_car);
                                }
                                param_cur = p_cdr;
                                arg_cur = a_cdr;
                            }
                            (Value::Nil, Value::Nil) => break,
                            _ => {
                                return Err(ValRef::new_str("Parameter/argument mismatch"));
                            }
                        }
                    }

                    Ok(EvalResult::TailCall(body.clone(), call_env))
                }
                _ => Err(ValRef::new_str("Cannot call non-function")),
            }
        }
        Value::Nil => Ok(EvalResult::Done(ValRef::nil())),
    }
}

pub fn eval(mut expr: ValRef, env: &ValRef) -> Result<ValRef, ValRef> {
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

fn builtin_add(args: &ValRef) -> Result<ValRef, ValRef> {
    let mut result: i64 = 0;
    let mut current = args.clone();

    loop {
        match current.as_ref() {
            Value::Cons(cell) => {
                let (car, cdr) = cell.borrow().clone();
                let num = car
                    .as_number()
                    .ok_or(ValRef::new_str("+ requires numbers"))?;
                result = result
                    .checked_add(num)
                    .ok_or(ValRef::new_str("Integer overflow"))?;
                current = cdr;
            }
            Value::Nil => break,
            _ => return Err(ValRef::new_str("Invalid argument list")),
        }
    }

    Ok(ValRef::number(result))
}

fn builtin_sub(args: &ValRef) -> Result<ValRef, ValRef> {
    let len = args.as_ref().list_len();
    if len == 0 {
        return Err(ValRef::new_str("- requires at least 1 argument"));
    }

    let first = args
        .as_ref()
        .list_nth(0)
        .ok_or(ValRef::new_str("- missing first argument"))?;
    let first_num = first
        .as_number()
        .ok_or(ValRef::new_str("- requires numbers"))?;

    if len == 1 {
        return Ok(ValRef::number(
            first_num
                .checked_neg()
                .ok_or(ValRef::new_str("Integer overflow"))?,
        ));
    }

    let mut result = first_num;
    let mut current = args.clone();
    if let Value::Cons(cell) = current.as_ref() {
        let (_, rest) = cell.borrow().clone();
        current = rest;
    }

    loop {
        match current.as_ref() {
            Value::Cons(cell) => {
                let (car, cdr) = cell.borrow().clone();
                let num = car
                    .as_number()
                    .ok_or(ValRef::new_str("- requires numbers"))?;
                result = result
                    .checked_sub(num)
                    .ok_or(ValRef::new_str("Integer overflow"))?;
                current = cdr;
            }
            Value::Nil => break,
            _ => return Err(ValRef::new_str("Invalid argument list")),
        }
    }

    Ok(ValRef::number(result))
}

fn builtin_mul(args: &ValRef) -> Result<ValRef, ValRef> {
    let mut result: i64 = 1;
    let mut current = args.clone();

    loop {
        match current.as_ref() {
            Value::Cons(cell) => {
                let (car, cdr) = cell.borrow().clone();
                let num = car
                    .as_number()
                    .ok_or(ValRef::new_str("* requires numbers"))?;
                result = result
                    .checked_mul(num)
                    .ok_or(ValRef::new_str("Integer overflow"))?;
                current = cdr;
            }
            Value::Nil => break,
            _ => return Err(ValRef::new_str("Invalid argument list")),
        }
    }

    Ok(ValRef::number(result))
}

fn builtin_div(args: &ValRef) -> Result<ValRef, ValRef> {
    let len = args.as_ref().list_len();
    if len < 2 {
        return Err(ValRef::new_str("/ requires at least 2 arguments"));
    }

    let first = args
        .as_ref()
        .list_nth(0)
        .ok_or(ValRef::new_str("/ missing first argument"))?;
    let mut result = first
        .as_number()
        .ok_or(ValRef::new_str("/ requires numbers"))?;

    let mut current = args.clone();
    if let Value::Cons(cell) = current.as_ref() {
        let (_, rest) = cell.borrow().clone();
        current = rest;
    }

    loop {
        match current.as_ref() {
            Value::Cons(cell) => {
                let (car, cdr) = cell.borrow().clone();
                let num = car
                    .as_number()
                    .ok_or(ValRef::new_str("/ requires numbers"))?;
                if num == 0 {
                    return Err(ValRef::new_str("Division by zero"));
                }
                result = result
                    .checked_div(num)
                    .ok_or(ValRef::new_str("Integer overflow"))?;
                current = cdr;
            }
            Value::Nil => break,
            _ => return Err(ValRef::new_str("Invalid argument list")),
        }
    }

    Ok(ValRef::number(result))
}

fn builtin_eq(args: &ValRef) -> Result<ValRef, ValRef> {
    if args.as_ref().list_len() != 2 {
        return Err(ValRef::new_str("= requires 2 arguments"));
    }
    let a = args
        .as_ref()
        .list_nth(0)
        .ok_or(ValRef::new_str("= missing arg 1"))?;
    let b = args
        .as_ref()
        .list_nth(1)
        .ok_or(ValRef::new_str("= missing arg 2"))?;
    let a_num = a.as_number().ok_or(ValRef::new_str("= requires numbers"))?;
    let b_num = b.as_number().ok_or(ValRef::new_str("= requires numbers"))?;
    Ok(ValRef::bool_val(a_num == b_num))
}

fn builtin_lt(args: &ValRef) -> Result<ValRef, ValRef> {
    if args.as_ref().list_len() != 2 {
        return Err(ValRef::new_str("< requires 2 arguments"));
    }
    let a = args
        .as_ref()
        .list_nth(0)
        .ok_or(ValRef::new_str("< missing arg 1"))?;
    let b = args
        .as_ref()
        .list_nth(1)
        .ok_or(ValRef::new_str("< missing arg 2"))?;
    let a_num = a.as_number().ok_or(ValRef::new_str("< requires numbers"))?;
    let b_num = b.as_number().ok_or(ValRef::new_str("< requires numbers"))?;
    Ok(ValRef::bool_val(a_num < b_num))
}

fn builtin_gt(args: &ValRef) -> Result<ValRef, ValRef> {
    if args.as_ref().list_len() != 2 {
        return Err(ValRef::new_str("> requires 2 arguments"));
    }
    let a = args
        .as_ref()
        .list_nth(0)
        .ok_or(ValRef::new_str("> missing arg 1"))?;
    let b = args
        .as_ref()
        .list_nth(1)
        .ok_or(ValRef::new_str("> missing arg 2"))?;
    let a_num = a.as_number().ok_or(ValRef::new_str("> requires numbers"))?;
    let b_num = b.as_number().ok_or(ValRef::new_str("> requires numbers"))?;
    Ok(ValRef::bool_val(a_num > b_num))
}

fn builtin_list(args: &ValRef) -> Result<ValRef, ValRef> {
    Ok(args.clone())
}

fn builtin_car(args: &ValRef) -> Result<ValRef, ValRef> {
    if args.as_ref().list_len() != 1 {
        return Err(ValRef::new_str("car requires 1 argument"));
    }
    let list = args
        .as_ref()
        .list_nth(0)
        .ok_or(ValRef::new_str("car missing argument"))?;
    let cell = list
        .as_cons()
        .ok_or(ValRef::new_str("car requires a cons/list"))?;
    let (car, _) = cell.borrow().clone();
    Ok(car)
}

fn builtin_cdr(args: &ValRef) -> Result<ValRef, ValRef> {
    if args.as_ref().list_len() != 1 {
        return Err(ValRef::new_str("cdr requires 1 argument"));
    }
    let list = args
        .as_ref()
        .list_nth(0)
        .ok_or(ValRef::new_str("cdr missing argument"))?;
    let cell = list
        .as_cons()
        .ok_or(ValRef::new_str("cdr requires a cons/list"))?;
    let (_, cdr) = cell.borrow().clone();
    Ok(cdr)
}

fn builtin_cons_fn(args: &ValRef) -> Result<ValRef, ValRef> {
    if args.as_ref().list_len() != 2 {
        return Err(ValRef::new_str("cons requires 2 arguments"));
    }
    let car = args
        .as_ref()
        .list_nth(0)
        .ok_or(ValRef::new_str("cons missing arg 1"))?;
    let cdr = args
        .as_ref()
        .list_nth(1)
        .ok_or(ValRef::new_str("cons missing arg 2"))?;
    Ok(ValRef::cons(car, cdr))
}

fn builtin_null(args: &ValRef) -> Result<ValRef, ValRef> {
    if args.as_ref().list_len() != 1 {
        return Err(ValRef::new_str("null? requires 1 argument"));
    }
    let val = args
        .as_ref()
        .list_nth(0)
        .ok_or(ValRef::new_str("null? missing argument"))?;
    Ok(ValRef::bool_val(val.is_nil()))
}

fn builtin_cons_p(args: &ValRef) -> Result<ValRef, ValRef> {
    if args.as_ref().list_len() != 1 {
        return Err(ValRef::new_str("cons? requires 1 argument"));
    }
    let val = args
        .as_ref()
        .list_nth(0)
        .ok_or(ValRef::new_str("cons? missing argument"))?;
    Ok(ValRef::bool_val(val.as_cons().is_some()))
}

fn builtin_length(args: &ValRef) -> Result<ValRef, ValRef> {
    if args.as_ref().list_len() != 1 {
        return Err(ValRef::new_str("length requires 1 argument"));
    }
    let list = args
        .as_ref()
        .list_nth(0)
        .ok_or(ValRef::new_str("length missing argument"))?;
    let len = list.as_ref().list_len();
    Ok(ValRef::number(len as i64))
}

fn builtin_append(args: &ValRef) -> Result<ValRef, ValRef> {
    let mut result = ValRef::nil();
    let mut current = args.clone();

    loop {
        match current.as_ref() {
            Value::Cons(cell) => {
                let (list, rest) = cell.borrow().clone();
                let mut list_cur = list;
                loop {
                    match list_cur.as_ref() {
                        Value::Cons(item_cell) => {
                            let (item, item_rest) = item_cell.borrow().clone();
                            result = ValRef::cons(item, result);
                            list_cur = item_rest;
                        }
                        Value::Nil => break,
                        _ => break,
                    }
                }
                current = rest;
            }
            Value::Nil => break,
            _ => return Err(ValRef::new_str("Invalid argument list")),
        }
    }

    Ok(reverse_list(result))
}

fn builtin_reverse(args: &ValRef) -> Result<ValRef, ValRef> {
    if args.as_ref().list_len() != 1 {
        return Err(ValRef::new_str("reverse requires 1 argument"));
    }
    let list = args
        .as_ref()
        .list_nth(0)
        .ok_or(ValRef::new_str("reverse missing argument"))?;
    Ok(reverse_list(list))
}

// ============================================================================
// Public API
// ============================================================================

pub fn eval_str(input: &str, env: &ValRef) -> Result<ValRef, ValRef> {
    let expr = parse(input)?;
    let result = eval(expr, env)?;
    Ok(result)
}

pub fn eval_str_multiple(input: &str, env: &ValRef) -> Result<ValRef, ValRef> {
    let expressions = parse_multiple(input)?;
    if expressions.is_nil() {
        return Err(ValRef::new_str("No expressions to evaluate"));
    }

    let mut last_result = ValRef::nil();
    let mut current = expressions;

    loop {
        match current.as_ref() {
            Value::Cons(cell) => {
                let (expr, rest) = cell.borrow().clone();
                last_result = eval(expr, env)?;
                current = rest;
            }
            Value::Nil => break,
            _ => return Err(ValRef::new_str("Invalid expression list")),
        }
    }

    Ok(last_result)
}

impl PartialEq for ValRef {
    fn eq(&self, other: &Self) -> bool {
        match (self.as_ref(), other.as_ref()) {
            (Value::Number(a), Value::Number(b)) => a == b,
            (Value::Bool(a), Value::Bool(b)) => a == b,
            (Value::Char(a), Value::Char(b)) => a == b,
            (Value::Symbol(a), Value::Symbol(b)) => a.str_eq(b),
            (Value::Nil, Value::Nil) => true,
            (Value::Cons(a_cell), Value::Cons(b_cell)) => {
                let (a_car, a_cdr) = a_cell.borrow().clone();
                let (b_car, b_cdr) = b_cell.borrow().clone();
                a_car == b_car && a_cdr == b_cdr
            }
            (Value::Builtin(a), Value::Builtin(b)) => core::ptr::eq(a as *const _, b as *const _),
            (
                Value::Lambda {
                    params: p1,
                    body: b1,
                    env: _,
                },
                Value::Lambda {
                    params: p2,
                    body: b2,
                    env: _,
                },
            ) => p1 == p2 && b1 == b2,
            _ => false,
        }
    }
}

// Public function to create new environment
pub fn new_env() -> ValRef {
    env_new()
}
