#![no_std]

use core::cell::RefCell;

// ============================================================================
// Arena Allocator
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ArenaError {
    OutOfMemory,
    InvalidIndex,
}

pub struct Arena<const N: usize> {
    cells: RefCell<[Value; N]>,
    next_free: RefCell<usize>,
}

impl<const N: usize> Arena<N> {
    pub const fn new() -> Self {
        Arena {
            cells: RefCell::new([Value::Nil; N]),
            next_free: RefCell::new(0),
        }
    }

    pub fn alloc(&self, value: Value) -> Result<ValRef, ArenaError> {
        let mut next = self.next_free.borrow_mut();
        if *next >= N {
            return Err(ArenaError::OutOfMemory);
        }
        let index = *next;
        *next += 1;
        self.cells.borrow_mut()[index] = value;
        Ok(ValRef(index))
    }

    pub fn get(&self, index: ValRef) -> Value {
        self.cells.borrow()[index.0]
    }

    pub fn set(&self, index: ValRef, value: Value) {
        self.cells.borrow_mut()[index.0] = value;
    }

    pub fn reset(&self) {
        *self.next_free.borrow_mut() = 0;
        for cell in self.cells.borrow_mut().iter_mut() {
            *cell = Value::Nil;
        }
    }
    pub fn capacity(&self) -> usize {
        N
    }

    pub fn available(&self) -> usize {
        N - *self.next_free.borrow()
    }
    pub fn used(&self) -> usize {
        *self.next_free.borrow()
    }
}

// ============================================================================
// Value Types
// ============================================================================

// Store builtin function index instead of function pointer
pub type BuiltinId = u8;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Value {
    Number(i64),
    Symbol(StrRef),
    Bool(bool),
    Char(char),
    Cons(ValRef, ValRef),
    Builtin(BuiltinId),
    Lambda {
        params: ValRef,
        body: ValRef,
        env: EnvRef,
    },
    Nil,
}

// ============================================================================
// String representation as index into string storage
// ============================================================================

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StrRef(usize);

const MAX_STRINGS: usize = 512;
const MAX_STRING_LEN: usize = 64;

pub struct StringTable {
    strings: RefCell<[[u8; MAX_STRING_LEN]; MAX_STRINGS]>,
    lengths: RefCell<[usize; MAX_STRINGS]>,
    next_free: RefCell<usize>,
}

impl StringTable {
    pub const fn new() -> Self {
        StringTable {
            strings: RefCell::new([[0; MAX_STRING_LEN]; MAX_STRINGS]),
            lengths: RefCell::new([0; MAX_STRINGS]),
            next_free: RefCell::new(0),
        }
    }

    pub fn intern(&self, s: &str) -> Result<StrRef, ()> {
        let bytes = s.as_bytes();
        if bytes.len() > MAX_STRING_LEN {
            return Err(());
        }

        // Check if string already exists
        let next = *self.next_free.borrow();
        let strings = self.strings.borrow();
        let lengths = self.lengths.borrow();

        for i in 0..next {
            if lengths[i] == bytes.len() {
                let mut matches = true;
                for j in 0..bytes.len() {
                    if strings[i][j] != bytes[j] {
                        matches = false;
                        break;
                    }
                }
                if matches {
                    return Ok(StrRef(i));
                }
            }
        }

        // Add new string
        drop(strings);
        drop(lengths);

        let mut next = self.next_free.borrow_mut();
        if *next >= MAX_STRINGS {
            return Err(());
        }

        let index = *next;
        *next += 1;

        let mut strings = self.strings.borrow_mut();
        let mut lengths = self.lengths.borrow_mut();

        for (i, &b) in bytes.iter().enumerate() {
            strings[index][i] = b;
        }
        lengths[index] = bytes.len();

        Ok(StrRef(index))
    }

    pub fn get<'a>(&self, s: StrRef, buf: &'a mut [u8]) -> Result<&'a str, ()> {
        let strings = self.strings.borrow();
        let lengths = self.lengths.borrow();
        let len = lengths[s.0];

        if len > buf.len() {
            return Err(());
        }

        for i in 0..len {
            buf[i] = strings[s.0][i];
        }

        core::str::from_utf8(&buf[..len]).map_err(|_| ())
    }

    pub fn eq(&self, a: StrRef, b: StrRef) -> bool {
        a.0 == b.0
    }

    pub fn reset(&self) {
        *self.next_free.borrow_mut() = 0;
    }
}

// ============================================================================
// ValRef - Index into arena
// ============================================================================

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ValRef(usize);

impl ValRef {
    pub const fn nil() -> Self {
        ValRef(usize::MAX)
    }

    pub fn is_nil(self) -> bool {
        self.0 == usize::MAX
    }
}

// ============================================================================
// Environment
// ============================================================================

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EnvRef(pub usize);

const MAX_ENVS: usize = 256;

#[derive(Clone, Copy, Debug)]
struct Env {
    bindings: ValRef,
    parent: Option<EnvRef>,
}

pub struct EnvTable {
    envs: RefCell<[Env; MAX_ENVS]>,
    next_free: RefCell<usize>,
}

impl EnvTable {
    pub const fn new() -> Self {
        EnvTable {
            envs: RefCell::new(
                [Env {
                    bindings: ValRef::nil(),
                    parent: None,
                }; MAX_ENVS],
            ),
            next_free: RefCell::new(0),
        }
    }

    pub fn alloc(&self, bindings: ValRef, parent: Option<EnvRef>) -> Result<EnvRef, ()> {
        let mut next = self.next_free.borrow_mut();
        if *next >= MAX_ENVS {
            return Err(());
        }
        let index = *next;
        *next += 1;
        self.envs.borrow_mut()[index] = Env { bindings, parent };
        Ok(EnvRef(index))
    }

    pub fn get_bindings(&self, e: EnvRef) -> ValRef {
        self.envs.borrow()[e.0].bindings
    }

    pub fn set_bindings(&self, e: EnvRef, bindings: ValRef) {
        self.envs.borrow_mut()[e.0].bindings = bindings;
    }

    pub fn get_parent(&self, e: EnvRef) -> Option<EnvRef> {
        self.envs.borrow()[e.0].parent
    }

    pub fn reset(&self) {
        *self.next_free.borrow_mut() = 0;
    }
}

// ============================================================================
// Interpreter Context
// ============================================================================

pub struct Interpreter<const N: usize> {
    pub arena: Arena<N>,
    pub strings: StringTable,
    pub envs: EnvTable,
}

impl<const N: usize> Interpreter<N> {
    pub const fn new() -> Self {
        Interpreter {
            arena: Arena::new(),
            strings: StringTable::new(),
            envs: EnvTable::new(),
        }
    }

    pub fn reset(&self) {
        self.arena.reset();
        self.strings.reset();
        self.envs.reset();
    }

    // Helper to create a cons cell
    pub fn cons(&self, car: ValRef, cdr: ValRef) -> Result<ValRef, ArenaError> {
        self.arena.alloc(Value::Cons(car, cdr))
    }

    // Helper to get list length
    pub fn list_len(&self, mut list: ValRef) -> usize {
        let mut count = 0;
        while !list.is_nil() {
            match self.arena.get(list) {
                Value::Cons(_, cdr) => {
                    count += 1;
                    list = cdr;
                }
                _ => break,
            }
        }
        count
    }

    // Helper to get nth element
    pub fn list_nth(&self, mut list: ValRef, n: usize) -> Option<ValRef> {
        let mut idx = 0;
        while !list.is_nil() {
            match self.arena.get(list) {
                Value::Cons(car, cdr) => {
                    if idx == n {
                        return Some(car);
                    }
                    idx += 1;
                    list = cdr;
                }
                _ => return None,
            }
        }
        None
    }

    // Reverse a list
    pub fn reverse_list(&self, mut list: ValRef) -> Result<ValRef, ArenaError> {
        let mut result = ValRef::nil();
        while !list.is_nil() {
            match self.arena.get(list) {
                Value::Cons(car, cdr) => {
                    result = self.cons(car, result)?;
                    list = cdr;
                }
                _ => break,
            }
        }
        Ok(result)
    }

    // Create environment with builtins
    pub fn create_global_env(&self) -> Result<EnvRef, ()> {
        let env = self.envs.alloc(ValRef::nil(), None)?;

        self.env_set(env, "nil", ValRef::nil())?;
        self.env_set(
            env,
            "+",
            self.arena.alloc(Value::Builtin(0)).map_err(|_| ())?,
        )?;
        self.env_set(
            env,
            "-",
            self.arena.alloc(Value::Builtin(1)).map_err(|_| ())?,
        )?;
        self.env_set(
            env,
            "*",
            self.arena.alloc(Value::Builtin(2)).map_err(|_| ())?,
        )?;
        self.env_set(
            env,
            "/",
            self.arena.alloc(Value::Builtin(3)).map_err(|_| ())?,
        )?;
        self.env_set(
            env,
            "=",
            self.arena.alloc(Value::Builtin(4)).map_err(|_| ())?,
        )?;
        self.env_set(
            env,
            "<",
            self.arena.alloc(Value::Builtin(5)).map_err(|_| ())?,
        )?;
        self.env_set(
            env,
            ">",
            self.arena.alloc(Value::Builtin(6)).map_err(|_| ())?,
        )?;
        self.env_set(
            env,
            "list",
            self.arena.alloc(Value::Builtin(7)).map_err(|_| ())?,
        )?;
        self.env_set(
            env,
            "car",
            self.arena.alloc(Value::Builtin(8)).map_err(|_| ())?,
        )?;
        self.env_set(
            env,
            "cdr",
            self.arena.alloc(Value::Builtin(9)).map_err(|_| ())?,
        )?;
        self.env_set(
            env,
            "cons",
            self.arena.alloc(Value::Builtin(10)).map_err(|_| ())?,
        )?;
        self.env_set(
            env,
            "null?",
            self.arena.alloc(Value::Builtin(11)).map_err(|_| ())?,
        )?;
        self.env_set(
            env,
            "cons?",
            self.arena.alloc(Value::Builtin(12)).map_err(|_| ())?,
        )?;
        self.env_set(
            env,
            "length",
            self.arena.alloc(Value::Builtin(13)).map_err(|_| ())?,
        )?;
        self.env_set(
            env,
            "append",
            self.arena.alloc(Value::Builtin(14)).map_err(|_| ())?,
        )?;
        self.env_set(
            env,
            "reverse",
            self.arena.alloc(Value::Builtin(15)).map_err(|_| ())?,
        )?;

        Ok(env)
    }

    pub fn env_set(&self, env: EnvRef, name: &str, value: ValRef) -> Result<(), ()> {
        let name_str = self.strings.intern(name)?;
        let binding = self
            .cons(
                self.arena.alloc(Value::Symbol(name_str)).map_err(|_| ())?,
                value,
            )
            .map_err(|_| ())?;

        let bindings = self.envs.get_bindings(env);
        let new_bindings = self.cons(binding, bindings).map_err(|_| ())?;
        self.envs.set_bindings(env, new_bindings);
        Ok(())
    }

    pub fn env_get(&self, env: EnvRef, name: StrRef) -> Option<ValRef> {
        let mut bindings = self.envs.get_bindings(env);

        while !bindings.is_nil() {
            match self.arena.get(bindings) {
                Value::Cons(binding, rest) => {
                    if let Value::Cons(key, value) = self.arena.get(binding) {
                        if let Value::Symbol(s) = self.arena.get(key) {
                            if self.strings.eq(s, name) {
                                return Some(value);
                            }
                        }
                    }
                    bindings = rest;
                }
                _ => break,
            }
        }

        if let Some(parent) = self.envs.get_parent(env) {
            self.env_get(parent, name)
        } else {
            None
        }
    }
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
        (true, 1)
    } else if bytes[0] == b'+' {
        (false, 1)
    } else {
        (false, 0)
    };

    if start >= bytes.len() {
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

impl<const N: usize> Interpreter<N> {
    pub fn tokenize(&self, input: &str) -> Result<ValRef, StrRef> {
        let mut result = ValRef::nil();
        let mut chars = input.chars().peekable();

        while let Some(&ch) = chars.peek() {
            match ch {
                ' ' | '\t' | '\n' | '\r' => {
                    chars.next();
                }
                '(' => {
                    let s = self.strings.intern("(").map_err(|_| StrRef(0))?;
                    result = self
                        .cons(
                            self.arena.alloc(Value::Symbol(s)).map_err(|_| StrRef(0))?,
                            result,
                        )
                        .map_err(|_| StrRef(0))?;
                    chars.next();
                }
                ')' => {
                    let s = self.strings.intern(")").map_err(|_| StrRef(0))?;
                    result = self
                        .cons(
                            self.arena.alloc(Value::Symbol(s)).map_err(|_| StrRef(0))?,
                            result,
                        )
                        .map_err(|_| StrRef(0))?;
                    chars.next();
                }
                '\'' => {
                    let s = self.strings.intern("'").map_err(|_| StrRef(0))?;
                    result = self
                        .cons(
                            self.arena.alloc(Value::Symbol(s)).map_err(|_| StrRef(0))?,
                            result,
                        )
                        .map_err(|_| StrRef(0))?;
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
                            result = self
                                .cons(
                                    self.arena.alloc(Value::Bool(true)).map_err(|_| StrRef(0))?,
                                    result,
                                )
                                .map_err(|_| StrRef(0))?;
                            chars.next();
                        }
                        Some(&'f') => {
                            result = self
                                .cons(
                                    self.arena
                                        .alloc(Value::Bool(false))
                                        .map_err(|_| StrRef(0))?,
                                    result,
                                )
                                .map_err(|_| StrRef(0))?;
                            chars.next();
                        }
                        _ => {
                            return Err(self
                                .strings
                                .intern("Invalid boolean")
                                .unwrap_or(StrRef(0)));
                        }
                    }
                }
                _ => {
                    let mut atom = [0u8; 128];
                    let mut idx = 0;

                    while let Some(&c) = chars.peek() {
                        if c.is_whitespace() || c == '(' || c == ')' || c == '\'' {
                            break;
                        }
                        let mut buf = [0u8; 4];
                        let s = c.encode_utf8(&mut buf);
                        for &b in s.as_bytes() {
                            if idx >= atom.len() {
                                return Err(self
                                    .strings
                                    .intern("Atom too long")
                                    .unwrap_or(StrRef(0)));
                            }
                            atom[idx] = b;
                            idx += 1;
                        }
                        chars.next();
                    }

                    if idx == 0 {
                        continue;
                    }

                    let atom_str = core::str::from_utf8(&atom[..idx]).map_err(|_| StrRef(0))?;

                    if let Ok(num) = parse_i64(atom_str) {
                        result = self
                            .cons(
                                self.arena
                                    .alloc(Value::Number(num))
                                    .map_err(|_| StrRef(0))?,
                                result,
                            )
                            .map_err(|_| StrRef(0))?;
                    } else {
                        let s = self.strings.intern(atom_str).map_err(|_| StrRef(0))?;
                        result = self
                            .cons(
                                self.arena.alloc(Value::Symbol(s)).map_err(|_| StrRef(0))?,
                                result,
                            )
                            .map_err(|_| StrRef(0))?;
                    }
                }
            }
        }

        self.reverse_list(result).map_err(|_| StrRef(0))
    }
}

// ============================================================================
// Parser
// ============================================================================

impl<const N: usize> Interpreter<N> {
    pub fn parse_tokens(&self, tokens: ValRef) -> Result<(ValRef, ValRef), StrRef> {
        if tokens.is_nil() {
            return Err(self.strings.intern("Unexpected end").unwrap_or(StrRef(0)));
        }

        match self.arena.get(tokens) {
            Value::Cons(first, rest) => match self.arena.get(first) {
                Value::Number(_) | Value::Bool(_) => Ok((first, rest)),
                Value::Symbol(s) => {
                    let mut buf = [0u8; 32];
                    let s_str = self.strings.get(s, &mut buf).map_err(|_| StrRef(0))?;

                    if s_str == "'" {
                        match self.arena.get(rest) {
                            Value::Cons(next_expr, remaining) => {
                                let next_tokens =
                                    self.cons(next_expr, remaining).map_err(|_| StrRef(0))?;
                                let (val, consumed) = self.parse_tokens(next_tokens)?;
                                let quote_sym =
                                    self.strings.intern("quote").map_err(|_| StrRef(0))?;
                                let quoted = self
                                    .cons(
                                        self.arena
                                            .alloc(Value::Symbol(quote_sym))
                                            .map_err(|_| StrRef(0))?,
                                        self.cons(val, ValRef::nil()).map_err(|_| StrRef(0))?,
                                    )
                                    .map_err(|_| StrRef(0))?;
                                Ok((quoted, consumed))
                            }
                            _ => Err(self.strings.intern("Quote needs expr").unwrap_or(StrRef(0))),
                        }
                    } else if s_str == "(" {
                        let mut items = ValRef::nil();
                        let mut pos = rest;

                        loop {
                            if pos.is_nil() {
                                return Err(self
                                    .strings
                                    .intern("Unmatched (")
                                    .unwrap_or(StrRef(0)));
                            }

                            match self.arena.get(pos) {
                                Value::Cons(token, rest_tokens) => {
                                    if let Value::Symbol(tok_s) = self.arena.get(token) {
                                        let mut tok_buf = [0u8; 32];
                                        let tok_str = self
                                            .strings
                                            .get(tok_s, &mut tok_buf)
                                            .map_err(|_| StrRef(0))?;
                                        if tok_str == ")" {
                                            return Ok((
                                                self.reverse_list(items).map_err(|_| StrRef(0))?,
                                                rest_tokens,
                                            ));
                                        }
                                    }
                                    let (val, consumed) = self.parse_tokens(pos)?;
                                    items = self.cons(val, items).map_err(|_| StrRef(0))?;
                                    pos = consumed;
                                }
                                _ => return Err(StrRef(0)),
                            }
                        }
                    } else if s_str == ")" {
                        Err(self.strings.intern("Unexpected )").unwrap_or(StrRef(0)))
                    } else {
                        Ok((first, rest))
                    }
                }
                _ => Err(StrRef(0)),
            },
            _ => Err(StrRef(0)),
        }
    }

    pub fn parse(&self, input: &str) -> Result<ValRef, StrRef> {
        let tokens = self.tokenize(input)?;
        if tokens.is_nil() {
            return Err(self.strings.intern("Empty input").unwrap_or(StrRef(0)));
        }
        let (val, remaining) = self.parse_tokens(tokens)?;
        if !remaining.is_nil() {
            return Err(self.strings.intern("Extra tokens").unwrap_or(StrRef(0)));
        }
        Ok(val)
    }
}

// ============================================================================
// Evaluator
// ============================================================================

pub enum EvalResult {
    Done(ValRef),
    TailCall(ValRef, EnvRef),
}

impl<const N: usize> Interpreter<N> {
    pub fn eval(&self, mut expr: ValRef, mut env: EnvRef) -> Result<ValRef, StrRef> {
        loop {
            match self.eval_step(expr, env)? {
                EvalResult::Done(val) => return Ok(val),
                EvalResult::TailCall(new_expr, new_env) => {
                    expr = new_expr;
                    env = new_env;
                }
            }
        }
    }

    fn eval_step(&self, expr: ValRef, env: EnvRef) -> Result<EvalResult, StrRef> {
        if expr.is_nil() {
            return Ok(EvalResult::Done(ValRef::nil()));
        }

        match self.arena.get(expr) {
            Value::Number(_)
            | Value::Bool(_)
            | Value::Char(_)
            | Value::Builtin(_)
            | Value::Lambda { .. } => Ok(EvalResult::Done(expr)),
            Value::Symbol(s) => {
                let mut buf = [0u8; 32];
                let s_str = self.strings.get(s, &mut buf).map_err(|_| StrRef(0))?;
                if s_str == "nil" {
                    return Ok(EvalResult::Done(ValRef::nil()));
                }
                self.env_get(env, s)
                    .map(EvalResult::Done)
                    .ok_or_else(|| self.strings.intern("Unbound symbol").unwrap_or(StrRef(0)))
            }
            Value::Cons(car, _cdr) => {
                if let Value::Symbol(sym) = self.arena.get(car) {
                    let mut buf = [0u8; 32];
                    let sym_str = self.strings.get(sym, &mut buf).map_err(|_| StrRef(0))?;

                    match sym_str {
                        "define" => {
                            let len = self.list_len(expr);
                            if len != 3 {
                                return Err(self
                                    .strings
                                    .intern("define needs 2 args")
                                    .unwrap_or(StrRef(0)));
                            }
                            let name_val = self.list_nth(expr, 1).ok_or(StrRef(0))?;
                            let name = match self.arena.get(name_val) {
                                Value::Symbol(s) => s,
                                _ => {
                                    return Err(self
                                        .strings
                                        .intern("define needs symbol")
                                        .unwrap_or(StrRef(0)));
                                }
                            };
                            let body_val = self.list_nth(expr, 2).ok_or(StrRef(0))?;
                            let val = self.eval(body_val, env)?;

                            let mut buf = [0u8; 32];
                            let name_str =
                                self.strings.get(name, &mut buf).map_err(|_| StrRef(0))?;
                            self.env_set(env, name_str, val).map_err(|_| StrRef(0))?;
                            return Ok(EvalResult::Done(val));
                        }
                        "lambda" => {
                            let len = self.list_len(expr);
                            if len != 3 {
                                return Err(self
                                    .strings
                                    .intern("lambda needs 2 args")
                                    .unwrap_or(StrRef(0)));
                            }
                            let params = self.list_nth(expr, 1).ok_or(StrRef(0))?;
                            let body = self.list_nth(expr, 2).ok_or(StrRef(0))?;

                            let lambda = self
                                .arena
                                .alloc(Value::Lambda { params, body, env })
                                .map_err(|_| StrRef(0))?;
                            return Ok(EvalResult::Done(lambda));
                        }
                        "if" => {
                            let len = self.list_len(expr);
                            if len != 4 {
                                return Err(self
                                    .strings
                                    .intern("if needs 3 args")
                                    .unwrap_or(StrRef(0)));
                            }
                            let cond_expr = self.list_nth(expr, 1).ok_or(StrRef(0))?;
                            let cond = self.eval(cond_expr, env)?;
                            let is_true = match self.arena.get(cond) {
                                Value::Bool(b) => b,
                                Value::Nil => false,
                                _ => true,
                            };
                            let branch_idx = if is_true { 2 } else { 3 };
                            let branch = self.list_nth(expr, branch_idx).ok_or(StrRef(0))?;
                            return Ok(EvalResult::TailCall(branch, env));
                        }
                        "quote" => {
                            let len = self.list_len(expr);
                            if len != 2 {
                                return Err(self
                                    .strings
                                    .intern("quote needs 1 arg")
                                    .unwrap_or(StrRef(0)));
                            }
                            let quoted = self.list_nth(expr, 1).ok_or(StrRef(0))?;
                            return Ok(EvalResult::Done(quoted));
                        }
                        _ => {}
                    }
                }

                // Function application
                let func = self.eval(car, env)?;

                // Evaluate arguments
                let mut args = ValRef::nil();
                let mut current = match self.arena.get(expr) {
                    Value::Cons(_, cdr) => cdr,
                    _ => return Err(StrRef(0)),
                };

                while !current.is_nil() {
                    match self.arena.get(current) {
                        Value::Cons(arg_car, arg_cdr) => {
                            let evaled = self.eval(arg_car, env)?;
                            args = self.cons(evaled, args).map_err(|_| StrRef(0))?;
                            current = arg_cdr;
                        }
                        _ => break,
                    }
                }
                args = self.reverse_list(args).map_err(|_| StrRef(0))?;

                match self.arena.get(func) {
                    Value::Builtin(id) => Ok(EvalResult::Done(self.call_builtin(id, args)?)),
                    Value::Lambda {
                        params,
                        body,
                        env: lambda_env,
                    } => {
                        let param_count = self.list_len(params);
                        let arg_count = self.list_len(args);

                        if arg_count != param_count {
                            return Err(self
                                .strings
                                .intern("Arg count mismatch")
                                .unwrap_or(StrRef(0)));
                        }

                        let call_env = self
                            .envs
                            .alloc(ValRef::nil(), Some(lambda_env))
                            .map_err(|_| StrRef(0))?;

                        let mut param_cur = params;
                        let mut arg_cur = args;

                        while !param_cur.is_nil() {
                            match (self.arena.get(param_cur), self.arena.get(arg_cur)) {
                                (Value::Cons(p_car, p_cdr), Value::Cons(a_car, a_cdr)) => {
                                    if let Value::Symbol(param_name) = self.arena.get(p_car) {
                                        let mut buf = [0u8; 32];
                                        let name_str = self
                                            .strings
                                            .get(param_name, &mut buf)
                                            .map_err(|_| StrRef(0))?;
                                        self.env_set(call_env, name_str, a_car)
                                            .map_err(|_| StrRef(0))?;
                                    }
                                    param_cur = p_cdr;
                                    arg_cur = a_cdr;
                                }
                                _ => break,
                            }
                        }

                        Ok(EvalResult::TailCall(body, call_env))
                    }
                    _ => Err(self.strings.intern("Not callable").unwrap_or(StrRef(0))),
                }
            }
            Value::Nil => Ok(EvalResult::Done(ValRef::nil())),
        }
    }

    pub fn eval_str(&self, input: &str, env: EnvRef) -> Result<[u8; 4096], StrRef> {
        let expr = self.parse(input)?;
        let result = self.eval(expr, env)?;
        self.value_to_string(result)
    }

    fn call_builtin(&self, id: BuiltinId, args: ValRef) -> Result<ValRef, StrRef> {
        match id {
            0 => builtin_add(self, args),
            1 => builtin_sub(self, args),
            2 => builtin_mul(self, args),
            3 => builtin_div(self, args),
            4 => builtin_eq(self, args),
            5 => builtin_lt(self, args),
            6 => builtin_gt(self, args),
            7 => builtin_list(self, args),
            8 => builtin_car(self, args),
            9 => builtin_cdr(self, args),
            10 => builtin_cons_fn(self, args),
            11 => builtin_null(self, args),
            12 => builtin_cons_p(self, args),
            13 => builtin_length(self, args),
            14 => builtin_append(self, args),
            15 => builtin_reverse(self, args),
            _ => Err(self.strings.intern("Unknown builtin").unwrap_or(StrRef(0))),
        }
    }

    fn value_to_string(&self, val: ValRef) -> Result<[u8; 4096], StrRef> {
        let mut buf = [0u8; 4096];
        let len = self
            .value_to_str_buf(val, &mut buf)
            .map_err(|_| StrRef(0))?;
        buf[len] = 0; // null terminator
        Ok(buf)
    }

    fn value_to_str_buf(&self, val: ValRef, buf: &mut [u8]) -> Result<usize, ()> {
        if val.is_nil() {
            let s = b"nil";
            if s.len() > buf.len() {
                return Err(());
            }
            buf[..s.len()].copy_from_slice(s);
            return Ok(s.len());
        }

        match self.arena.get(val) {
            Value::Number(n) => {
                let mut temp = [0u8; 32];
                let mut idx = 0;
                let mut num = n;
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
                    buf[1..=idx].copy_from_slice(&temp[..idx]);
                    Ok(idx + 1)
                } else {
                    if idx > buf.len() {
                        return Err(());
                    }
                    buf[..idx].copy_from_slice(&temp[..idx]);
                    Ok(idx)
                }
            }
            Value::Symbol(s) => {
                let mut temp = [0u8; MAX_STRING_LEN];
                let s_str = self.strings.get(s, &mut temp)?;
                let bytes = s_str.as_bytes();
                if bytes.len() > buf.len() {
                    return Err(());
                }
                buf[..bytes.len()].copy_from_slice(bytes);
                Ok(bytes.len())
            }
            Value::Bool(b) => {
                let s = if b { b"#t" } else { b"#f" };
                if s.len() > buf.len() {
                    return Err(());
                }
                buf[..s.len()].copy_from_slice(s);
                Ok(s.len())
            }
            Value::Char(c) => {
                let mut char_buf = [0u8; 4];
                let s = c.encode_utf8(&mut char_buf);
                let bytes = s.as_bytes();
                if bytes.len() > buf.len() {
                    return Err(());
                }
                buf[..bytes.len()].copy_from_slice(bytes);
                Ok(bytes.len())
            }
            Value::Cons(_, _) => self.list_to_str_buf(val, buf),
            Value::Builtin(_) => {
                let s = b"<builtin>";
                if s.len() > buf.len() {
                    return Err(());
                }
                buf[..s.len()].copy_from_slice(s);
                Ok(s.len())
            }
            Value::Lambda { .. } => {
                let s = b"<lambda>";
                if s.len() > buf.len() {
                    return Err(());
                }
                buf[..s.len()].copy_from_slice(s);
                Ok(s.len())
            }
            Value::Nil => {
                let s = b"nil";
                if s.len() > buf.len() {
                    return Err(());
                }
                buf[..s.len()].copy_from_slice(s);
                Ok(s.len())
            }
        }
    }

    fn list_to_str_buf(&self, mut list: ValRef, buf: &mut [u8]) -> Result<usize, ()> {
        let mut idx = 0;
        if idx >= buf.len() {
            return Err(());
        }
        buf[idx] = b'(';
        idx += 1;

        let mut first = true;

        while !list.is_nil() {
            match self.arena.get(list) {
                Value::Cons(car, cdr) => {
                    if !first {
                        if idx >= buf.len() {
                            return Err(());
                        }
                        buf[idx] = b' ';
                        idx += 1;
                    }
                    first = false;

                    let item_len = self.value_to_str_buf(car, &mut buf[idx..])?;
                    idx += item_len;

                    list = cdr;
                }
                _ => {
                    if !first {
                        if idx + 3 > buf.len() {
                            return Err(());
                        }
                        buf[idx] = b' ';
                        idx += 1;
                        buf[idx] = b'.';
                        idx += 1;
                        buf[idx] = b' ';
                        idx += 1;
                    }

                    let item_len = self.value_to_str_buf(list, &mut buf[idx..])?;
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

        Ok(idx)
    }
}

// ============================================================================
// Built-in Functions
// ============================================================================

fn builtin_add<const N: usize>(interp: &Interpreter<N>, args: ValRef) -> Result<ValRef, StrRef> {
    let mut result: i64 = 0;
    let mut current = args;

    while !current.is_nil() {
        match interp.arena.get(current) {
            Value::Cons(car, cdr) => {
                match interp.arena.get(car) {
                    Value::Number(n) => {
                        result = result.checked_add(n).ok_or(StrRef(0))?;
                    }
                    _ => {
                        return Err(interp
                            .strings
                            .intern("+ needs numbers")
                            .unwrap_or(StrRef(0)));
                    }
                }
                current = cdr;
            }
            _ => break,
        }
    }

    interp
        .arena
        .alloc(Value::Number(result))
        .map_err(|_| StrRef(0))
}

fn builtin_sub<const N: usize>(interp: &Interpreter<N>, args: ValRef) -> Result<ValRef, StrRef> {
    let len = interp.list_len(args);
    if len == 0 {
        return Err(interp
            .strings
            .intern("- needs 1+ args")
            .unwrap_or(StrRef(0)));
    }

    let first = interp.list_nth(args, 0).ok_or(StrRef(0))?;
    let first_num = match interp.arena.get(first) {
        Value::Number(n) => n,
        _ => {
            return Err(interp
                .strings
                .intern("- needs numbers")
                .unwrap_or(StrRef(0)));
        }
    };

    if len == 1 {
        return interp
            .arena
            .alloc(Value::Number(-first_num))
            .map_err(|_| StrRef(0));
    }

    let mut result = first_num;
    let mut current = match interp.arena.get(args) {
        Value::Cons(_, cdr) => cdr,
        _ => return Err(StrRef(0)),
    };

    while !current.is_nil() {
        match interp.arena.get(current) {
            Value::Cons(car, cdr) => {
                match interp.arena.get(car) {
                    Value::Number(n) => {
                        result = result.checked_sub(n).ok_or(StrRef(0))?;
                    }
                    _ => {
                        return Err(interp
                            .strings
                            .intern("- needs numbers")
                            .unwrap_or(StrRef(0)));
                    }
                }
                current = cdr;
            }
            _ => break,
        }
    }

    interp
        .arena
        .alloc(Value::Number(result))
        .map_err(|_| StrRef(0))
}

fn builtin_mul<const N: usize>(interp: &Interpreter<N>, args: ValRef) -> Result<ValRef, StrRef> {
    let mut result: i64 = 1;
    let mut current = args;

    while !current.is_nil() {
        match interp.arena.get(current) {
            Value::Cons(car, cdr) => {
                match interp.arena.get(car) {
                    Value::Number(n) => {
                        result = result.checked_mul(n).ok_or(StrRef(0))?;
                    }
                    _ => {
                        return Err(interp
                            .strings
                            .intern("* needs numbers")
                            .unwrap_or(StrRef(0)));
                    }
                }
                current = cdr;
            }
            _ => break,
        }
    }

    interp
        .arena
        .alloc(Value::Number(result))
        .map_err(|_| StrRef(0))
}

fn builtin_div<const N: usize>(interp: &Interpreter<N>, args: ValRef) -> Result<ValRef, StrRef> {
    let len = interp.list_len(args);
    if len < 2 {
        return Err(interp
            .strings
            .intern("/ needs 2+ args")
            .unwrap_or(StrRef(0)));
    }

    let first = interp.list_nth(args, 0).ok_or(StrRef(0))?;
    let mut result = match interp.arena.get(first) {
        Value::Number(n) => n,
        _ => {
            return Err(interp
                .strings
                .intern("/ needs numbers")
                .unwrap_or(StrRef(0)));
        }
    };

    let mut current = match interp.arena.get(args) {
        Value::Cons(_, cdr) => cdr,
        _ => return Err(StrRef(0)),
    };

    while !current.is_nil() {
        match interp.arena.get(current) {
            Value::Cons(car, cdr) => {
                match interp.arena.get(car) {
                    Value::Number(n) => {
                        if n == 0 {
                            return Err(interp.strings.intern("Div by zero").unwrap_or(StrRef(0)));
                        }
                        result = result.checked_div(n).ok_or(StrRef(0))?;
                    }
                    _ => {
                        return Err(interp
                            .strings
                            .intern("/ needs numbers")
                            .unwrap_or(StrRef(0)));
                    }
                }
                current = cdr;
            }
            _ => break,
        }
    }

    interp
        .arena
        .alloc(Value::Number(result))
        .map_err(|_| StrRef(0))
}

fn builtin_eq<const N: usize>(interp: &Interpreter<N>, args: ValRef) -> Result<ValRef, StrRef> {
    if interp.list_len(args) != 2 {
        return Err(interp.strings.intern("= needs 2 args").unwrap_or(StrRef(0)));
    }
    let a = interp.list_nth(args, 0).ok_or(StrRef(0))?;
    let b = interp.list_nth(args, 1).ok_or(StrRef(0))?;

    let (a_num, b_num) = match (interp.arena.get(a), interp.arena.get(b)) {
        (Value::Number(x), Value::Number(y)) => (x, y),
        _ => {
            return Err(interp
                .strings
                .intern("= needs numbers")
                .unwrap_or(StrRef(0)));
        }
    };

    interp
        .arena
        .alloc(Value::Bool(a_num == b_num))
        .map_err(|_| StrRef(0))
}

fn builtin_lt<const N: usize>(interp: &Interpreter<N>, args: ValRef) -> Result<ValRef, StrRef> {
    if interp.list_len(args) != 2 {
        return Err(interp.strings.intern("< needs 2 args").unwrap_or(StrRef(0)));
    }
    let a = interp.list_nth(args, 0).ok_or(StrRef(0))?;
    let b = interp.list_nth(args, 1).ok_or(StrRef(0))?;

    let (a_num, b_num) = match (interp.arena.get(a), interp.arena.get(b)) {
        (Value::Number(x), Value::Number(y)) => (x, y),
        _ => {
            return Err(interp
                .strings
                .intern("< needs numbers")
                .unwrap_or(StrRef(0)));
        }
    };

    interp
        .arena
        .alloc(Value::Bool(a_num < b_num))
        .map_err(|_| StrRef(0))
}

fn builtin_gt<const N: usize>(interp: &Interpreter<N>, args: ValRef) -> Result<ValRef, StrRef> {
    if interp.list_len(args) != 2 {
        return Err(interp.strings.intern("> needs 2 args").unwrap_or(StrRef(0)));
    }
    let a = interp.list_nth(args, 0).ok_or(StrRef(0))?;
    let b = interp.list_nth(args, 1).ok_or(StrRef(0))?;

    let (a_num, b_num) = match (interp.arena.get(a), interp.arena.get(b)) {
        (Value::Number(x), Value::Number(y)) => (x, y),
        _ => {
            return Err(interp
                .strings
                .intern("> needs numbers")
                .unwrap_or(StrRef(0)));
        }
    };

    interp
        .arena
        .alloc(Value::Bool(a_num > b_num))
        .map_err(|_| StrRef(0))
}

fn builtin_list<const N: usize>(interp: &Interpreter<N>, args: ValRef) -> Result<ValRef, StrRef> {
    Ok(args)
}

fn builtin_car<const N: usize>(interp: &Interpreter<N>, args: ValRef) -> Result<ValRef, StrRef> {
    if interp.list_len(args) != 1 {
        return Err(interp
            .strings
            .intern("car needs 1 arg")
            .unwrap_or(StrRef(0)));
    }
    let list = interp.list_nth(args, 0).ok_or(StrRef(0))?;
    match interp.arena.get(list) {
        Value::Cons(car, _) => Ok(car),
        _ => Err(interp.strings.intern("car needs cons").unwrap_or(StrRef(0))),
    }
}

fn builtin_cdr<const N: usize>(interp: &Interpreter<N>, args: ValRef) -> Result<ValRef, StrRef> {
    if interp.list_len(args) != 1 {
        return Err(interp
            .strings
            .intern("cdr needs 1 arg")
            .unwrap_or(StrRef(0)));
    }
    let list = interp.list_nth(args, 0).ok_or(StrRef(0))?;
    match interp.arena.get(list) {
        Value::Cons(_, cdr) => Ok(cdr),
        _ => Err(interp.strings.intern("cdr needs cons").unwrap_or(StrRef(0))),
    }
}

fn builtin_cons_fn<const N: usize>(
    interp: &Interpreter<N>,
    args: ValRef,
) -> Result<ValRef, StrRef> {
    if interp.list_len(args) != 2 {
        return Err(interp
            .strings
            .intern("cons needs 2 args")
            .unwrap_or(StrRef(0)));
    }
    let car = interp.list_nth(args, 0).ok_or(StrRef(0))?;
    let cdr = interp.list_nth(args, 1).ok_or(StrRef(0))?;
    interp.cons(car, cdr).map_err(|_| StrRef(0))
}

fn builtin_null<const N: usize>(interp: &Interpreter<N>, args: ValRef) -> Result<ValRef, StrRef> {
    if interp.list_len(args) != 1 {
        return Err(interp
            .strings
            .intern("null? needs 1 arg")
            .unwrap_or(StrRef(0)));
    }
    let val = interp.list_nth(args, 0).ok_or(StrRef(0))?;
    interp
        .arena
        .alloc(Value::Bool(val.is_nil()))
        .map_err(|_| StrRef(0))
}

fn builtin_cons_p<const N: usize>(interp: &Interpreter<N>, args: ValRef) -> Result<ValRef, StrRef> {
    if interp.list_len(args) != 1 {
        return Err(interp
            .strings
            .intern("cons? needs 1 arg")
            .unwrap_or(StrRef(0)));
    }
    let val = interp.list_nth(args, 0).ok_or(StrRef(0))?;
    let is_cons = matches!(interp.arena.get(val), Value::Cons(_, _));
    interp
        .arena
        .alloc(Value::Bool(is_cons))
        .map_err(|_| StrRef(0))
}

fn builtin_length<const N: usize>(interp: &Interpreter<N>, args: ValRef) -> Result<ValRef, StrRef> {
    if interp.list_len(args) != 1 {
        return Err(interp
            .strings
            .intern("length needs 1 arg")
            .unwrap_or(StrRef(0)));
    }
    let list = interp.list_nth(args, 0).ok_or(StrRef(0))?;
    let len = interp.list_len(list);
    interp
        .arena
        .alloc(Value::Number(len as i64))
        .map_err(|_| StrRef(0))
}

fn builtin_append<const N: usize>(interp: &Interpreter<N>, args: ValRef) -> Result<ValRef, StrRef> {
    let mut result = ValRef::nil();
    let mut current = args;

    while !current.is_nil() {
        match interp.arena.get(current) {
            Value::Cons(list, rest) => {
                let mut list_cur = list;
                while !list_cur.is_nil() {
                    match interp.arena.get(list_cur) {
                        Value::Cons(item, item_rest) => {
                            result = interp.cons(item, result).map_err(|_| StrRef(0))?;
                            list_cur = item_rest;
                        }
                        _ => break,
                    }
                }
                current = rest;
            }
            _ => break,
        }
    }

    interp.reverse_list(result).map_err(|_| StrRef(0))
}

fn builtin_reverse<const N: usize>(
    interp: &Interpreter<N>,
    args: ValRef,
) -> Result<ValRef, StrRef> {
    if interp.list_len(args) != 1 {
        return Err(interp
            .strings
            .intern("reverse needs 1 arg")
            .unwrap_or(StrRef(0)));
    }
    let list = interp.list_nth(args, 0).ok_or(StrRef(0))?;
    interp.reverse_list(list).map_err(|_| StrRef(0))
}
