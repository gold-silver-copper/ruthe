#![no_std]

use core::cell::RefCell;

// ============================================================================
// Core Data Types
// ============================================================================

pub type BuiltinFn<const N: usize, const MAX_ROOTS: usize> =
    fn(&Arena<N, MAX_ROOTS>, usize) -> Result<usize, usize>;

#[derive(Clone, Copy)]
pub enum LispValue<const N: usize, const MAX_ROOTS: usize> {
    Nil,
    Number(i32),
    Symbol(u32),
    Cons(usize, usize),
    Bool(bool),
    Char(char),
    Builtin(BuiltinFn<N, MAX_ROOTS>),
    Lambda {
        params: usize,
        body: usize,
        env: usize,
    },
}

impl<const N: usize, const MAX_ROOTS: usize> core::fmt::Debug for LispValue<N, MAX_ROOTS> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Nil => write!(f, "Nil"),
            Self::Number(n) => write!(f, "Number({})", n),
            Self::Symbol(s) => write!(f, "Symbol({})", s),
            Self::Cons(car, cdr) => write!(f, "Cons({}, {})", car, cdr),
            Self::Bool(b) => write!(f, "Bool({})", b),
            Self::Char(c) => write!(f, "Char({})", c),
            Self::Builtin(_) => write!(f, "Builtin(<fn>)"),
            Self::Lambda { params, body, env } => {
                write!(
                    f,
                    "Lambda {{ params: {}, body: {}, env: {} }}",
                    params, body, env
                )
            }
        }
    }
}

impl<const N: usize, const MAX_ROOTS: usize> PartialEq for LispValue<N, MAX_ROOTS> {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Nil, Self::Nil) => true,
            (Self::Number(a), Self::Number(b)) => a == b,
            (Self::Symbol(a), Self::Symbol(b)) => a == b,
            (Self::Cons(a1, a2), Self::Cons(b1, b2)) => a1 == b1 && a2 == b2,
            (Self::Bool(a), Self::Bool(b)) => a == b,
            (Self::Char(a), Self::Char(b)) => a == b,
            (
                Self::Lambda {
                    params: p1,
                    body: b1,
                    env: e1,
                },
                Self::Lambda {
                    params: p2,
                    body: b2,
                    env: e2,
                },
            ) => p1 == p2 && b1 == b2 && e1 == e2,
            _ => false,
        }
    }
}

// ============================================================================
// Arena with Automatic GC
// ============================================================================

pub struct Arena<const N: usize, const MAX_ROOTS: usize> {
    cells: RefCell<[LispValue<N, MAX_ROOTS>; N]>,
    allocated: RefCell<[bool; N]>,
    marked: RefCell<[bool; N]>,
    free_list: RefCell<[usize; N]>,
    free_count: RefCell<usize>,
    next_free: RefCell<usize>,
    gc_roots: RefCell<[usize; MAX_ROOTS]>,
    gc_roots_count: RefCell<usize>,
}

impl<const N: usize, const MAX_ROOTS: usize> Arena<N, MAX_ROOTS> {
    pub const fn new() -> Self {
        Arena {
            cells: RefCell::new([LispValue::Nil; N]),
            allocated: RefCell::new([false; N]),
            marked: RefCell::new([false; N]),
            free_list: RefCell::new([0; N]),
            free_count: RefCell::new(0),
            next_free: RefCell::new(0),
            gc_roots: RefCell::new([0; MAX_ROOTS]),
            gc_roots_count: RefCell::new(0),
        }
    }

    pub fn push_root(&self, root: usize) {
        let mut count = self.gc_roots_count.borrow_mut();
        if *count < MAX_ROOTS {
            self.gc_roots.borrow_mut()[*count] = root;
            *count += 1;
        }
    }

    pub fn pop_root(&self) {
        let mut count = self.gc_roots_count.borrow_mut();
        if *count > 0 {
            *count -= 1;
        }
    }

    pub fn alloc(&self, value: LispValue<N, MAX_ROOTS>) -> Result<usize, ArenaError> {
        let mut free_count = self.free_count.borrow_mut();
        if *free_count > 0 {
            *free_count -= 1;
            let index = self.free_list.borrow()[*free_count];
            self.cells.borrow_mut()[index] = value;
            self.allocated.borrow_mut()[index] = true;
            return Ok(index);
        }
        drop(free_count);

        let mut next = self.next_free.borrow_mut();
        if *next >= N {
            // Try GC before giving up
            drop(next);
            self.auto_collect();

            // Try again after GC
            let mut free_count = self.free_count.borrow_mut();
            if *free_count > 0 {
                *free_count -= 1;
                let index = self.free_list.borrow()[*free_count];
                self.cells.borrow_mut()[index] = value;
                self.allocated.borrow_mut()[index] = true;
                return Ok(index);
            }
            drop(free_count);

            let mut next = self.next_free.borrow_mut();
            if *next >= N {
                return Err(ArenaError::OutOfMemory);
            }
            let index = *next;
            *next += 1;
            self.cells.borrow_mut()[index] = value;
            self.allocated.borrow_mut()[index] = true;
            return Ok(index);
        }
        let index = *next;
        *next += 1;
        self.cells.borrow_mut()[index] = value;
        self.allocated.borrow_mut()[index] = true;
        Ok(index)
    }

    pub fn get(&self, index: usize) -> Result<LispValue<N, MAX_ROOTS>, ArenaError> {
        if index >= N || !self.allocated.borrow()[index] {
            return Err(ArenaError::InvalidIndex);
        }
        Ok(self.cells.borrow()[index])
    }

    pub fn set(&self, index: usize, value: LispValue<N, MAX_ROOTS>) -> Result<(), ArenaError> {
        if index >= N || !self.allocated.borrow()[index] {
            return Err(ArenaError::InvalidIndex);
        }
        self.cells.borrow_mut()[index] = value;
        Ok(())
    }

    fn mark(&self, index: usize) {
        if index >= N || self.marked.borrow()[index] || !self.allocated.borrow()[index] {
            return;
        }

        self.marked.borrow_mut()[index] = true;

        match self.cells.borrow()[index] {
            LispValue::Cons(car, cdr) => {
                self.mark(car);
                self.mark(cdr);
            }
            LispValue::Lambda { params, body, env } => {
                self.mark(params);
                self.mark(body);
                self.mark(env);
            }
            _ => {}
        }
    }

    fn sweep(&self) {
        let next = *self.next_free.borrow();
        let mut free_list = self.free_list.borrow_mut();
        let mut free_count = self.free_count.borrow_mut();
        let mut allocated = self.allocated.borrow_mut();
        let mut marked = self.marked.borrow_mut();

        *free_count = 0;

        for i in 0..next {
            if allocated[i] && !marked[i] {
                allocated[i] = false;
                free_list[*free_count] = i;
                *free_count += 1;
            }
            marked[i] = false;
        }
    }

    fn auto_collect(&self) {
        let count = *self.gc_roots_count.borrow();
        for i in 0..count {
            let root = self.gc_roots.borrow()[i];
            self.mark(root);
        }
        self.sweep();
    }

    pub fn collect(&self, roots: &[usize]) {
        for &root in roots {
            self.mark(root);
        }
        self.sweep();
    }

    pub fn used(&self) -> usize {
        self.allocated.borrow().iter().filter(|&&x| x).count()
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ArenaError {
    OutOfMemory,
    InvalidIndex,
}

// ============================================================================
// String/Symbol Utilities (stored as cons lists of chars)
// ============================================================================

fn str_to_list<const N: usize, const MAX_ROOTS: usize>(
    arena: &Arena<N, MAX_ROOTS>,
    s: &str,
) -> Result<usize, ArenaError> {
    let mut result = arena.alloc(LispValue::Nil)?;
    for ch in s.chars().rev() {
        let ch_val = arena.alloc(LispValue::Char(ch))?;
        result = arena.alloc(LispValue::Cons(ch_val, result))?;
    }
    Ok(result)
}

pub fn hash_str(s: &str) -> u32 {
    let mut hash: u32 = 5381;
    for byte in s.bytes() {
        hash = hash.wrapping_mul(33).wrapping_add(byte as u32);
    }
    hash
}

fn list_to_str<'a, const N: usize, const MAX_ROOTS: usize>(
    arena: &Arena<N, MAX_ROOTS>,
    idx: usize,
    buf: &'a mut [u8],
) -> Result<&'a str, ()> {
    let mut pos = 0;
    let mut current = idx;

    loop {
        match arena.get(current) {
            Ok(LispValue::Cons(car, cdr)) => {
                if let Ok(LispValue::Char(ch)) = arena.get(car) {
                    let mut char_buf = [0u8; 4];
                    let s = ch.encode_utf8(&mut char_buf);
                    for &b in s.as_bytes() {
                        if pos >= buf.len() {
                            return Err(());
                        }
                        buf[pos] = b;
                        pos += 1;
                    }
                    current = cdr;
                } else {
                    return Err(());
                }
            }
            Ok(LispValue::Nil) => break,
            _ => return Err(()),
        }
    }

    core::str::from_utf8(&buf[..pos]).map_err(|_| ())
}

fn reverse_list<const N: usize, const MAX_ROOTS: usize>(
    arena: &Arena<N, MAX_ROOTS>,
    list: usize,
) -> Result<usize, ArenaError> {
    let mut result = arena.alloc(LispValue::Nil)?;
    let mut current = list;

    loop {
        match arena.get(current)? {
            LispValue::Cons(car, cdr) => {
                result = arena.alloc(LispValue::Cons(car, result))?;
                current = cdr;
            }
            LispValue::Nil => break,
            _ => break,
        }
    }

    Ok(result)
}

fn list_len<const N: usize, const MAX_ROOTS: usize>(
    arena: &Arena<N, MAX_ROOTS>,
    list: usize,
) -> usize {
    let mut count = 0;
    let mut current = list;

    loop {
        match arena.get(current) {
            Ok(LispValue::Cons(_, cdr)) => {
                count += 1;
                current = cdr;
            }
            _ => break,
        }
    }

    count
}

fn list_nth<const N: usize, const MAX_ROOTS: usize>(
    arena: &Arena<N, MAX_ROOTS>,
    list: usize,
    n: usize,
) -> Option<usize> {
    let mut current = list;
    let mut idx = 0;

    loop {
        match arena.get(current) {
            Ok(LispValue::Cons(car, cdr)) => {
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

// ============================================================================
// Environment (stored in arena as cons list of bindings)
// ============================================================================

const NO_PARENT: usize = usize::MAX;

pub fn env_new<const N: usize, const MAX_ROOTS: usize>(
    arena: &Arena<N, MAX_ROOTS>,
) -> Result<usize, ArenaError> {
    let bindings = arena.alloc(LispValue::Nil)?;
    let env = arena.alloc(LispValue::Cons(bindings, NO_PARENT))?;
    arena.push_root(env);
    Ok(env)
}

fn env_with_parent<const N: usize, const MAX_ROOTS: usize>(
    arena: &Arena<N, MAX_ROOTS>,
    parent: usize,
) -> Result<usize, ArenaError> {
    let bindings = arena.alloc(LispValue::Nil)?;
    arena.alloc(LispValue::Cons(bindings, parent))
}

fn env_set<const N: usize, const MAX_ROOTS: usize>(
    arena: &Arena<N, MAX_ROOTS>,
    env: usize,
    name: u32,
    value: usize,
) -> Result<(), ArenaError> {
    if let Ok(LispValue::Cons(bindings, parent)) = arena.get(env) {
        let name_val = arena.alloc(LispValue::Symbol(name))?;
        let binding = arena.alloc(LispValue::Cons(name_val, value))?;
        let new_bindings = arena.alloc(LispValue::Cons(binding, bindings))?;
        arena.set(env, LispValue::Cons(new_bindings, parent))?;
    }
    Ok(())
}

fn env_get<const N: usize, const MAX_ROOTS: usize>(
    arena: &Arena<N, MAX_ROOTS>,
    env: usize,
    name: u32,
) -> Option<usize> {
    if let Ok(LispValue::Cons(bindings, parent)) = arena.get(env) {
        let mut current = bindings;

        loop {
            match arena.get(current) {
                Ok(LispValue::Cons(binding, rest)) => {
                    if let Ok(LispValue::Cons(key, value)) = arena.get(binding) {
                        if let Ok(LispValue::Symbol(sym)) = arena.get(key) {
                            if sym == name {
                                return Some(value);
                            }
                        }
                    }
                    current = rest;
                }
                _ => break,
            }
        }

        if parent != NO_PARENT {
            return env_get(arena, parent, name);
        }
    }

    None
}

// ============================================================================
// Tokenizer
// ============================================================================

fn tokenize<const N: usize, const MAX_ROOTS: usize>(
    arena: &Arena<N, MAX_ROOTS>,
    input: &str,
) -> Result<usize, usize> {
    let mut result = arena
        .alloc(LispValue::Nil)
        .map_err(|_| str_to_list(arena, "Out of memory").unwrap_or(0))?;
    let mut chars = input.chars().peekable();

    while let Some(&ch) = chars.peek() {
        match ch {
            ' ' | '\t' | '\n' | '\r' => {
                chars.next();
            }
            '(' => {
                let sym = arena
                    .alloc(LispValue::Symbol(hash_str("(")))
                    .map_err(|_| str_to_list(arena, "Out of memory").unwrap_or(0))?;
                result = arena
                    .alloc(LispValue::Cons(sym, result))
                    .map_err(|_| str_to_list(arena, "Out of memory").unwrap_or(0))?;
                chars.next();
            }
            ')' => {
                let sym = arena
                    .alloc(LispValue::Symbol(hash_str(")")))
                    .map_err(|_| str_to_list(arena, "Out of memory").unwrap_or(0))?;
                result = arena
                    .alloc(LispValue::Cons(sym, result))
                    .map_err(|_| str_to_list(arena, "Out of memory").unwrap_or(0))?;
                chars.next();
            }
            '\'' => {
                let sym = arena
                    .alloc(LispValue::Symbol(hash_str("'")))
                    .map_err(|_| str_to_list(arena, "Out of memory").unwrap_or(0))?;
                result = arena
                    .alloc(LispValue::Cons(sym, result))
                    .map_err(|_| str_to_list(arena, "Out of memory").unwrap_or(0))?;
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
                        let val = arena
                            .alloc(LispValue::Bool(true))
                            .map_err(|_| str_to_list(arena, "Out of memory").unwrap_or(0))?;
                        result = arena
                            .alloc(LispValue::Cons(val, result))
                            .map_err(|_| str_to_list(arena, "Out of memory").unwrap_or(0))?;
                        chars.next();
                    }
                    Some(&'f') => {
                        let val = arena
                            .alloc(LispValue::Bool(false))
                            .map_err(|_| str_to_list(arena, "Out of memory").unwrap_or(0))?;
                        result = arena
                            .alloc(LispValue::Cons(val, result))
                            .map_err(|_| str_to_list(arena, "Out of memory").unwrap_or(0))?;
                        chars.next();
                    }
                    _ => return Err(str_to_list(arena, "Invalid boolean").unwrap_or(0)),
                }
            }
            _ => {
                let mut atom = [0u8; 64];
                let mut len = 0;

                while let Some(&c) = chars.peek() {
                    if c.is_whitespace() || c == '(' || c == ')' || c == '\'' || c == ';' {
                        break;
                    }
                    if len >= atom.len() {
                        return Err(str_to_list(arena, "Atom too long").unwrap_or(0));
                    }
                    let mut buf = [0u8; 4];
                    let s = c.encode_utf8(&mut buf);
                    for &b in s.as_bytes() {
                        if len >= atom.len() {
                            break;
                        }
                        atom[len] = b;
                        len += 1;
                    }
                    chars.next();
                }

                if len == 0 {
                    continue;
                }

                let atom_str = core::str::from_utf8(&atom[..len])
                    .map_err(|_| str_to_list(arena, "Invalid UTF-8").unwrap_or(0))?;

                let val = if let Ok(num) = parse_i32(atom_str) {
                    arena
                        .alloc(LispValue::Number(num))
                        .map_err(|_| str_to_list(arena, "Out of memory").unwrap_or(0))?
                } else {
                    arena
                        .alloc(LispValue::Symbol(hash_str(atom_str)))
                        .map_err(|_| str_to_list(arena, "Out of memory").unwrap_or(0))?
                };

                result = arena
                    .alloc(LispValue::Cons(val, result))
                    .map_err(|_| str_to_list(arena, "Out of memory").unwrap_or(0))?;
            }
        }
    }

    reverse_list(arena, result).map_err(|_| str_to_list(arena, "Out of memory").unwrap_or(0))
}

fn parse_i32(s: &str) -> Result<i32, ()> {
    let bytes = s.as_bytes();
    if bytes.is_empty() {
        return Err(());
    }

    let (negative, start) = if bytes[0] == b'-' {
        (true, 1)
    } else {
        (false, 0)
    };

    if start >= bytes.len() {
        return Err(());
    }

    let mut result: i32 = 0;
    for &b in &bytes[start..] {
        if !(b'0'..=b'9').contains(&b) {
            return Err(());
        }
        let digit = (b - b'0') as i32;
        result = result.checked_mul(10).ok_or(())?;
        result = result.checked_add(digit).ok_or(())?;
    }

    if negative {
        result.checked_neg().ok_or(())
    } else {
        Ok(result)
    }
}

// ============================================================================
// Parser
// ============================================================================

fn parse_tokens<const N: usize, const MAX_ROOTS: usize>(
    arena: &Arena<N, MAX_ROOTS>,
    tokens: usize,
) -> Result<(usize, usize), usize> {
    match arena.get(tokens) {
        Ok(LispValue::Nil) => Err(str_to_list(arena, "Unexpected end").unwrap_or(0)),
        Ok(LispValue::Cons(first, rest)) => match arena.get(first) {
            Ok(LispValue::Number(_)) | Ok(LispValue::Bool(_)) => Ok((first, rest)),
            Ok(LispValue::Symbol(sym)) => {
                if sym == hash_str("'") {
                    // Handle quote
                    let (val, consumed) = parse_tokens(arena, rest)?;
                    let quote_sym = arena
                        .alloc(LispValue::Symbol(hash_str("quote")))
                        .map_err(|_| str_to_list(arena, "Out of memory").unwrap_or(0))?;
                    let nil = arena
                        .alloc(LispValue::Nil)
                        .map_err(|_| str_to_list(arena, "Out of memory").unwrap_or(0))?;
                    let quoted_list = arena
                        .alloc(LispValue::Cons(val, nil))
                        .map_err(|_| str_to_list(arena, "Out of memory").unwrap_or(0))?;
                    let quoted = arena
                        .alloc(LispValue::Cons(quote_sym, quoted_list))
                        .map_err(|_| str_to_list(arena, "Out of memory").unwrap_or(0))?;
                    Ok((quoted, consumed))
                } else if sym == hash_str("(") {
                    // Handle list
                    let mut items = arena
                        .alloc(LispValue::Nil)
                        .map_err(|_| str_to_list(arena, "Out of memory").unwrap_or(0))?;
                    let mut pos = rest;

                    loop {
                        match arena.get(pos) {
                            Ok(LispValue::Nil) => {
                                return Err(str_to_list(arena, "Unmatched (").unwrap_or(0));
                            }
                            Ok(LispValue::Cons(token, rest_tokens)) => {
                                if let Ok(LispValue::Symbol(s)) = arena.get(token) {
                                    if s == hash_str(")") {
                                        // Empty list or end of list
                                        let result = reverse_list(arena, items).map_err(|_| {
                                            str_to_list(arena, "Out of memory").unwrap_or(0)
                                        })?;
                                        return Ok((result, rest_tokens));
                                    }
                                }
                                let (val, consumed) = parse_tokens(arena, pos)?;
                                items = arena.alloc(LispValue::Cons(val, items)).map_err(|_| {
                                    str_to_list(arena, "Out of memory").unwrap_or(0)
                                })?;
                                pos = consumed;
                            }
                            _ => return Err(str_to_list(arena, "Invalid tokens").unwrap_or(0)),
                        }
                    }
                } else if sym == hash_str(")") {
                    Err(str_to_list(arena, "Unexpected )").unwrap_or(0))
                } else {
                    // Regular symbol
                    Ok((first, rest))
                }
            }
            _ => Err(str_to_list(arena, "Unexpected token").unwrap_or(0)),
        },
        _ => Err(str_to_list(arena, "Invalid tokens").unwrap_or(0)),
    }
}

pub fn parse<const N: usize, const MAX_ROOTS: usize>(
    arena: &Arena<N, MAX_ROOTS>,
    input: &str,
) -> Result<usize, usize> {
    let tokens = tokenize(arena, input)?;
    if let Ok(LispValue::Nil) = arena.get(tokens) {
        // Allow empty input, return nil
        return arena
            .alloc(LispValue::Nil)
            .map_err(|_| str_to_list(arena, "Out of memory").unwrap_or(0));
    }
    let (val, remaining) = parse_tokens(arena, tokens)?;
    if let Ok(LispValue::Nil) = arena.get(remaining) {
        Ok(val)
    } else {
        Err(str_to_list(arena, "Extra tokens").unwrap_or(0))
    }
}

// ============================================================================
// Evaluator with TCO
// ============================================================================

enum EvalResult {
    Done(usize),
    TailCall(usize, usize),
}

fn eval_step<const N: usize, const MAX_ROOTS: usize>(
    arena: &Arena<N, MAX_ROOTS>,
    expr: usize,
    env: usize,
) -> Result<EvalResult, usize> {
    match arena.get(expr) {
        Ok(LispValue::Number(_))
        | Ok(LispValue::Bool(_))
        | Ok(LispValue::Builtin(_))
        | Ok(LispValue::Lambda { .. }) => Ok(EvalResult::Done(expr)),
        Ok(LispValue::Symbol(sym)) => {
            if sym == hash_str("nil") {
                return Ok(EvalResult::Done(arena.alloc(LispValue::Nil).unwrap()));
            }
            env_get(arena, env, sym)
                .map(EvalResult::Done)
                .ok_or_else(|| str_to_list(arena, "Unbound symbol").unwrap_or(0))
        }
        Ok(LispValue::Cons(car, _)) => {
            if let Ok(LispValue::Symbol(sym)) = arena.get(car) {
                // Special forms
                if sym == hash_str("define") {
                    let len = list_len(arena, expr);
                    if len != 3 {
                        return Err(str_to_list(arena, "define needs 2 args").unwrap_or(0));
                    }
                    let name_val = list_nth(arena, expr, 1).unwrap();
                    let name = if let Ok(LispValue::Symbol(s)) = arena.get(name_val) {
                        s
                    } else {
                        return Err(str_to_list(arena, "define needs symbol").unwrap_or(0));
                    };
                    let body = list_nth(arena, expr, 2).unwrap();
                    arena.push_root(body);
                    let val = eval(arena, body, env)?;
                    arena.pop_root(); // body
                    env_set(arena, env, name, val).unwrap();
                    return Ok(EvalResult::Done(val));
                } else if sym == hash_str("lambda") {
                    let len = list_len(arena, expr);
                    if len != 3 {
                        return Err(str_to_list(arena, "lambda needs 2 args").unwrap_or(0));
                    }
                    let params = list_nth(arena, expr, 1).unwrap();
                    let body = list_nth(arena, expr, 2).unwrap();

                    // Validate params are symbols or nil
                    let mut cur = params;
                    loop {
                        match arena.get(cur) {
                            Ok(LispValue::Cons(p, rest)) => {
                                if let Ok(LispValue::Symbol(_)) = arena.get(p) {
                                    cur = rest;
                                } else {
                                    return Err(str_to_list(
                                        arena,
                                        "lambda params must be symbols",
                                    )
                                    .unwrap_or(0));
                                }
                            }
                            Ok(LispValue::Nil) => break,
                            _ => {
                                return Err(
                                    str_to_list(arena, "lambda params must be list").unwrap_or(0)
                                );
                            }
                        }
                    }

                    let lambda = arena
                        .alloc(LispValue::Lambda { params, body, env })
                        .unwrap();
                    return Ok(EvalResult::Done(lambda));
                } else if sym == hash_str("if") {
                    let len = list_len(arena, expr);
                    if len != 4 {
                        return Err(str_to_list(arena, "if needs 3 args").unwrap_or(0));
                    }
                    let cond_expr = list_nth(arena, expr, 1).unwrap();
                    arena.push_root(cond_expr);
                    let cond = eval(arena, cond_expr, env)?;
                    arena.pop_root(); // cond_expr
                    let is_true = match arena.get(cond) {
                        Ok(LispValue::Bool(b)) => b,
                        Ok(LispValue::Nil) => false,
                        _ => true,
                    };
                    let branch = list_nth(arena, expr, if is_true { 2 } else { 3 }).unwrap();
                    return Ok(EvalResult::TailCall(branch, env));
                } else if sym == hash_str("quote") {
                    let len = list_len(arena, expr);
                    if len != 2 {
                        return Err(str_to_list(arena, "quote needs 1 arg").unwrap_or(0));
                    }
                    let quoted = list_nth(arena, expr, 1).unwrap();
                    return Ok(EvalResult::Done(quoted));
                }
            }

            // Function application
            arena.push_root(car);
            let func = eval(arena, car, env)?;
            arena.pop_root(); // car
            arena.push_root(func); // Protect func from GC

            // Evaluate arguments
            let mut args = arena.alloc(LispValue::Nil).unwrap();
            arena.push_root(args); // Protect args list from GC

            let cdr = if let Ok(LispValue::Cons(_, c)) = arena.get(expr) {
                c
            } else {
                arena.pop_root(); // args
                arena.pop_root(); // func
                return Err(str_to_list(arena, "Invalid call").unwrap_or(0));
            };
            let mut current = cdr;
            loop {
                match arena.get(current) {
                    Ok(LispValue::Cons(arg, rest)) => {
                        arena.push_root(arg);
                        arena.push_root(rest);
                        let evaled = eval(arena, arg, env)?;
                        arena.pop_root(); // rest
                        arena.pop_root(); // arg
                        args = arena.alloc(LispValue::Cons(evaled, args)).unwrap();
                        // Update the args root
                        let root_count = *arena.gc_roots_count.borrow();
                        if root_count > 0 {
                            arena.gc_roots.borrow_mut()[root_count - 1] = args;
                        }
                        current = rest;
                    }
                    Ok(LispValue::Nil) => break,
                    _ => {
                        arena.pop_root(); // args
                        arena.pop_root(); // func
                        return Err(str_to_list(arena, "Bad args").unwrap_or(0));
                    }
                }
            }
            args = reverse_list(arena, args).unwrap();
            // Update root after reverse
            let root_count = *arena.gc_roots_count.borrow();
            if root_count > 0 {
                arena.gc_roots.borrow_mut()[root_count - 1] = args;
            }

            let result = match arena.get(func) {
                Ok(LispValue::Builtin(f)) => {
                    let res = f(arena, args);
                    arena.pop_root(); // args
                    arena.pop_root(); // func
                    Ok(EvalResult::Done(res?))
                }
                Ok(LispValue::Lambda {
                    params,
                    body,
                    env: lambda_env,
                }) => {
                    let param_count = list_len(arena, params);
                    let arg_count = list_len(arena, args);

                    if param_count != arg_count {
                        arena.pop_root(); // args
                        arena.pop_root(); // func
                        return Err(str_to_list(arena, "Arg count mismatch").unwrap_or(0));
                    }

                    let call_env = env_with_parent(arena, lambda_env).unwrap();
                    arena.push_root(call_env); // Protect new env

                    let mut p_cur = params;
                    let mut a_cur = args;
                    loop {
                        match (arena.get(p_cur), arena.get(a_cur)) {
                            (Ok(LispValue::Cons(p, p_rest)), Ok(LispValue::Cons(a, a_rest))) => {
                                if let Ok(LispValue::Symbol(name)) = arena.get(p) {
                                    env_set(arena, call_env, name, a).unwrap();
                                }
                                p_cur = p_rest;
                                a_cur = a_rest;
                            }
                            (Ok(LispValue::Nil), Ok(LispValue::Nil)) => break,
                            _ => {
                                arena.pop_root(); // call_env
                                arena.pop_root(); // args
                                arena.pop_root(); // func
                                return Err(str_to_list(arena, "Param/arg mismatch").unwrap_or(0));
                            }
                        }
                    }

                    arena.pop_root(); // call_env (not needed, we're passing it in tail call)
                    arena.pop_root(); // args
                    arena.pop_root(); // func
                    Ok(EvalResult::TailCall(body, call_env))
                }
                _ => {
                    arena.pop_root(); // args
                    arena.pop_root(); // func
                    Err(str_to_list(arena, "Not callable").unwrap_or(0))
                }
            };

            result
        }
        Ok(LispValue::Nil) => Ok(EvalResult::Done(expr)),
        _ => Err(str_to_list(arena, "Invalid expr").unwrap_or(0)),
    }
}

pub fn eval<const N: usize, const MAX_ROOTS: usize>(
    arena: &Arena<N, MAX_ROOTS>,
    mut expr: usize,
    mut env: usize,
) -> Result<usize, usize> {
    arena.push_root(expr);
    arena.push_root(env);

    let result = loop {
        // Update roots to current values
        let root_count = *arena.gc_roots_count.borrow();
        if root_count >= 2 {
            let mut roots = arena.gc_roots.borrow_mut();
            roots[root_count - 2] = expr;
            roots[root_count - 1] = env;
        }

        match eval_step(arena, expr, env) {
            Ok(EvalResult::Done(val)) => break Ok(val),
            Ok(EvalResult::TailCall(new_expr, new_env)) => {
                expr = new_expr;
                env = new_env;
            }
            Err(e) => break Err(e),
        }
    };

    arena.pop_root(); // env
    arena.pop_root(); // expr
    result
}

// ============================================================================
// Builtins
// ============================================================================

fn builtin_add<const N: usize, const MAX_ROOTS: usize>(
    arena: &Arena<N, MAX_ROOTS>,
    args: usize,
) -> Result<usize, usize> {
    let mut result: i32 = 0;
    let mut current = args;
    loop {
        match arena.get(current) {
            Ok(LispValue::Cons(car, cdr)) => {
                if let Ok(LispValue::Number(n)) = arena.get(car) {
                    result = result
                        .checked_add(n)
                        .ok_or_else(|| str_to_list(arena, "Overflow").unwrap_or(0))?;
                    current = cdr;
                } else {
                    return Err(str_to_list(arena, "+ needs numbers").unwrap_or(0));
                }
            }
            Ok(LispValue::Nil) => break,
            _ => return Err(str_to_list(arena, "Bad args").unwrap_or(0)),
        }
    }
    arena
        .alloc(LispValue::Number(result))
        .map_err(|_| str_to_list(arena, "Out of memory").unwrap_or(0))
}

fn builtin_sub<const N: usize, const MAX_ROOTS: usize>(
    arena: &Arena<N, MAX_ROOTS>,
    args: usize,
) -> Result<usize, usize> {
    let len = list_len(arena, args);
    if len == 0 {
        return Err(str_to_list(arena, "- needs args").unwrap_or(0));
    }
    let first = list_nth(arena, args, 0).unwrap();
    let first_num = if let Ok(LispValue::Number(n)) = arena.get(first) {
        n
    } else {
        return Err(str_to_list(arena, "- needs numbers").unwrap_or(0));
    };

    if len == 1 {
        return arena
            .alloc(LispValue::Number(-first_num))
            .map_err(|_| str_to_list(arena, "Out of memory").unwrap_or(0));
    }

    let mut result = first_num;
    let mut current = if let Ok(LispValue::Cons(_, rest)) = arena.get(args) {
        rest
    } else {
        return Err(str_to_list(arena, "Bad args").unwrap_or(0));
    };

    loop {
        match arena.get(current) {
            Ok(LispValue::Cons(car, cdr)) => {
                if let Ok(LispValue::Number(n)) = arena.get(car) {
                    result = result
                        .checked_sub(n)
                        .ok_or_else(|| str_to_list(arena, "Overflow").unwrap_or(0))?;
                    current = cdr;
                } else {
                    return Err(str_to_list(arena, "- needs numbers").unwrap_or(0));
                }
            }
            Ok(LispValue::Nil) => break,
            _ => return Err(str_to_list(arena, "Bad args").unwrap_or(0)),
        }
    }
    arena
        .alloc(LispValue::Number(result))
        .map_err(|_| str_to_list(arena, "Out of memory").unwrap_or(0))
}

fn builtin_mul<const N: usize, const MAX_ROOTS: usize>(
    arena: &Arena<N, MAX_ROOTS>,
    args: usize,
) -> Result<usize, usize> {
    let mut result: i32 = 1;
    let mut current = args;
    loop {
        match arena.get(current) {
            Ok(LispValue::Cons(car, cdr)) => {
                if let Ok(LispValue::Number(n)) = arena.get(car) {
                    result = result
                        .checked_mul(n)
                        .ok_or_else(|| str_to_list(arena, "Overflow").unwrap_or(0))?;
                    current = cdr;
                } else {
                    return Err(str_to_list(arena, "* needs numbers").unwrap_or(0));
                }
            }
            Ok(LispValue::Nil) => break,
            _ => return Err(str_to_list(arena, "Bad args").unwrap_or(0)),
        }
    }
    arena
        .alloc(LispValue::Number(result))
        .map_err(|_| str_to_list(arena, "Out of memory").unwrap_or(0))
}

fn builtin_lt<const N: usize, const MAX_ROOTS: usize>(
    arena: &Arena<N, MAX_ROOTS>,
    args: usize,
) -> Result<usize, usize> {
    if list_len(arena, args) != 2 {
        return Err(str_to_list(arena, "< needs 2 args").unwrap_or(0));
    }
    let a = list_nth(arena, args, 0).unwrap();
    let b = list_nth(arena, args, 1).unwrap();
    let an = if let Ok(LispValue::Number(n)) = arena.get(a) {
        n
    } else {
        return Err(str_to_list(arena, "< needs numbers").unwrap_or(0));
    };
    let bn = if let Ok(LispValue::Number(n)) = arena.get(b) {
        n
    } else {
        return Err(str_to_list(arena, "< needs numbers").unwrap_or(0));
    };
    arena
        .alloc(LispValue::Bool(an < bn))
        .map_err(|_| str_to_list(arena, "Out of memory").unwrap_or(0))
}

fn builtin_eq<const N: usize, const MAX_ROOTS: usize>(
    arena: &Arena<N, MAX_ROOTS>,
    args: usize,
) -> Result<usize, usize> {
    if list_len(arena, args) != 2 {
        return Err(str_to_list(arena, "= needs 2 args").unwrap_or(0));
    }
    let a = list_nth(arena, args, 0).unwrap();
    let b = list_nth(arena, args, 1).unwrap();

    let result = match (arena.get(a), arena.get(b)) {
        (Ok(LispValue::Number(an)), Ok(LispValue::Number(bn))) => an == bn,
        (Ok(LispValue::Nil), Ok(LispValue::Nil)) => true,
        (Ok(LispValue::Symbol(sa)), Ok(LispValue::Symbol(sb))) => sa == sb,
        (Ok(LispValue::Bool(ba)), Ok(LispValue::Bool(bb))) => ba == bb,
        _ => a == b, // Pointer equality for other types
    };

    arena
        .alloc(LispValue::Bool(result))
        .map_err(|_| str_to_list(arena, "Out of memory").unwrap_or(0))
}

fn builtin_cons<const N: usize, const MAX_ROOTS: usize>(
    arena: &Arena<N, MAX_ROOTS>,
    args: usize,
) -> Result<usize, usize> {
    if list_len(arena, args) != 2 {
        return Err(str_to_list(arena, "cons needs 2 args").unwrap_or(0));
    }
    let car = list_nth(arena, args, 0).unwrap();
    let cdr = list_nth(arena, args, 1).unwrap();
    arena
        .alloc(LispValue::Cons(car, cdr))
        .map_err(|_| str_to_list(arena, "Out of memory").unwrap_or(0))
}

fn builtin_car<const N: usize, const MAX_ROOTS: usize>(
    arena: &Arena<N, MAX_ROOTS>,
    args: usize,
) -> Result<usize, usize> {
    if list_len(arena, args) != 1 {
        return Err(str_to_list(arena, "car needs 1 arg").unwrap_or(0));
    }
    let list = list_nth(arena, args, 0).unwrap();
    match arena.get(list) {
        Ok(LispValue::Cons(car, _)) => Ok(car),
        Ok(LispValue::Nil) => Err(str_to_list(arena, "car of empty list").unwrap_or(0)),
        _ => Err(str_to_list(arena, "car needs list").unwrap_or(0)),
    }
}

fn builtin_cdr<const N: usize, const MAX_ROOTS: usize>(
    arena: &Arena<N, MAX_ROOTS>,
    args: usize,
) -> Result<usize, usize> {
    if list_len(arena, args) != 1 {
        return Err(str_to_list(arena, "cdr needs 1 arg").unwrap_or(0));
    }
    let list = list_nth(arena, args, 0).unwrap();
    match arena.get(list) {
        Ok(LispValue::Cons(_, cdr)) => Ok(cdr),
        Ok(LispValue::Nil) => Err(str_to_list(arena, "cdr of empty list").unwrap_or(0)),
        _ => Err(str_to_list(arena, "cdr needs list").unwrap_or(0)),
    }
}

pub fn init_env<const N: usize, const MAX_ROOTS: usize>(
    arena: &Arena<N, MAX_ROOTS>,
) -> Result<usize, ArenaError> {
    let env = env_new(arena)?;
    let nil = arena.alloc(LispValue::Nil)?;
    env_set(arena, env, hash_str("nil"), nil)?;
    env_set(
        arena,
        env,
        hash_str("+"),
        arena.alloc(LispValue::Builtin(builtin_add))?,
    )?;
    env_set(
        arena,
        env,
        hash_str("-"),
        arena.alloc(LispValue::Builtin(builtin_sub))?,
    )?;
    env_set(
        arena,
        env,
        hash_str("*"),
        arena.alloc(LispValue::Builtin(builtin_mul))?,
    )?;
    env_set(
        arena,
        env,
        hash_str("<"),
        arena.alloc(LispValue::Builtin(builtin_lt))?,
    )?;
    env_set(
        arena,
        env,
        hash_str("="),
        arena.alloc(LispValue::Builtin(builtin_eq))?,
    )?;
    env_set(
        arena,
        env,
        hash_str("cons"),
        arena.alloc(LispValue::Builtin(builtin_cons))?,
    )?;
    env_set(
        arena,
        env,
        hash_str("car"),
        arena.alloc(LispValue::Builtin(builtin_car))?,
    )?;
    env_set(
        arena,
        env,
        hash_str("cdr"),
        arena.alloc(LispValue::Builtin(builtin_cdr))?,
    )?;
    Ok(env)
}

pub fn eval_str<const N: usize, const MAX_ROOTS: usize>(
    arena: &Arena<N, MAX_ROOTS>,
    input: &str,
    env: usize,
) -> Result<i32, ()> {
    let expr = parse(arena, input).map_err(|_| ())?;
    let result = eval(arena, expr, env).map_err(|_| ())?;
    if let Ok(LispValue::Number(n)) = arena.get(result) {
        Ok(n)
    } else {
        Err(())
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_arithmetic() {
        let arena: Arena<1024, 256> = Arena::new();
        let env = init_env(&arena).unwrap();

        assert_eq!(eval_str(&arena, "(+ 1 2)", env).unwrap(), 3);
        assert_eq!(eval_str(&arena, "(- 10 3)", env).unwrap(), 7);
        assert_eq!(eval_str(&arena, "(* 4 5)", env).unwrap(), 20);
    }

    #[test]
    fn test_nested_arithmetic() {
        let arena: Arena<1024, 256> = Arena::new();
        let env = init_env(&arena).unwrap();

        assert_eq!(eval_str(&arena, "(+ (* 2 3) 4)", env).unwrap(), 10);
        assert_eq!(eval_str(&arena, "(- (* 5 4) (+ 2 3))", env).unwrap(), 15);
    }

    #[test]
    fn test_define() {
        let arena: Arena<1024, 256> = Arena::new();
        let env = init_env(&arena).unwrap();

        let expr = parse(&arena, "(define x 42)").unwrap();
        let _ = eval(&arena, expr, env).unwrap();

        assert_eq!(eval_str(&arena, "x", env).unwrap(), 42);
    }

    #[test]
    fn test_lambda() {
        let arena: Arena<1024, 256> = Arena::new();
        let env = init_env(&arena).unwrap();

        let expr = parse(&arena, "(define double (lambda (x) (* x 2)))").unwrap();
        let _ = eval(&arena, expr, env).unwrap();

        assert_eq!(eval_str(&arena, "(double 21)", env).unwrap(), 42);
    }

    #[test]
    fn test_if() {
        let arena: Arena<1024, 256> = Arena::new();
        let env = init_env(&arena).unwrap();

        assert_eq!(eval_str(&arena, "(if #t 1 2)", env).unwrap(), 1);
        assert_eq!(eval_str(&arena, "(if #f 1 2)", env).unwrap(), 2);
        assert_eq!(eval_str(&arena, "(if (< 3 5) 10 20)", env).unwrap(), 10);
    }

    #[test]
    fn test_factorial_recursive() {
        let arena: Arena<1024, 256> = Arena::new();
        let env = init_env(&arena).unwrap();

        let code = "
            (define fact 
              (lambda (n) 
                (if (= n 0) 
                    1 
                    (* n (fact (- n 1))))))
        ";
        let expr = parse(&arena, code).unwrap();
        let _ = eval(&arena, expr, env).unwrap();

        assert_eq!(eval_str(&arena, "(fact 5)", env).unwrap(), 120);
    }

    #[test]
    fn test_fibonacci_tail_recursive() {
        let arena: Arena<4000, 256> = Arena::new();
        let env = init_env(&arena).unwrap();

        let code = "
            (define fib-iter
              (lambda (a b count)
                (if (= count 0)
                    a
                    (fib-iter b (+ a b) (- count 1)))))
        ";
        let expr = parse(&arena, code).unwrap();
        let _ = eval(&arena, expr, env).unwrap();

        let code2 = "
            (define fib 
              (lambda (n) 
                (fib-iter 0 1 n)))
        ";
        let expr2 = parse(&arena, code2).unwrap();
        let _ = eval(&arena, expr2, env).unwrap();

        assert_eq!(eval_str(&arena, "(fib 10)", env).unwrap(), 55);
        assert_eq!(eval_str(&arena, "(fib 20)", env).unwrap(), 6765);
    }

    #[test]
    fn test_tco_deep_recursion() {
        let arena: Arena<8000, 256> = Arena::new();
        let env = init_env(&arena).unwrap();

        // This would stack overflow without TCO
        let code = "
            (define count-down
              (lambda (n)
                (if (= n 0)
                    0
                    (count-down (- n 1)))))
        ";
        let expr = parse(&arena, code).unwrap();
        let _ = eval(&arena, expr, env).unwrap();

        // This should work with TCO even for large n
        assert_eq!(eval_str(&arena, "(count-down 100)", env).unwrap(), 0);
        assert_eq!(eval_str(&arena, "(count-down 500)", env).unwrap(), 0);
    }

    #[test]
    fn test_gc_with_fibonacci() {
        let arena: Arena<4000, 256> = Arena::new();
        let env = init_env(&arena).unwrap();

        let code = "
            (define fib-iter
              (lambda (a b count)
                (if (= count 0)
                    a
                    (fib-iter b (+ a b) (- count 1)))))
        ";
        let expr = parse(&arena, code).unwrap();
        let _fib_iter = eval(&arena, expr, env).unwrap();

        let code2 = "
            (define fib 
              (lambda (n) 
                (fib-iter 0 1 n)))
        ";
        let expr2 = parse(&arena, code2).unwrap();
        let _fib = eval(&arena, expr2, env).unwrap();

        // Collect garbage, keeping only env which has our definitions
        arena.collect(&[env]);

        let used_after_gc = arena.used();

        // Should still be able to call fib after GC
        assert_eq!(eval_str(&arena, "(fib 15)", env).unwrap(), 610);

        // Memory usage should be reasonable
        assert!(used_after_gc < 500);
    }

    #[test]
    fn test_cons_car_cdr() {
        let arena: Arena<1024, 256> = Arena::new();
        let env = init_env(&arena).unwrap();

        // Test cons
        let code = "(define lst (cons 1 (cons 2 (cons 3 nil))))";
        let expr = parse(&arena, code).unwrap();
        let _ = eval(&arena, expr, env).unwrap();

        // Test car
        assert_eq!(eval_str(&arena, "(car lst)", env).unwrap(), 1);

        // Test cdr and nested car
        let code2 = "(car (cdr lst))";
        assert_eq!(eval_str(&arena, code2, env).unwrap(), 2);

        // Test nested cdr
        let code3 = "(car (cdr (cdr lst)))";
        assert_eq!(eval_str(&arena, code3, env).unwrap(), 3);
    }

    #[test]
    fn test_quote() {
        let arena: Arena<1024, 256> = Arena::new();
        let env = init_env(&arena).unwrap();

        // Quote should return the expression unevaluated
        let expr = parse(&arena, "'(1 2 3)").unwrap();
        let result = eval(&arena, expr, env).unwrap();

        // Result should be a list
        match arena.get(result) {
            Ok(LispValue::Cons(_, _)) => {} // Success
            _ => panic!("Quote should return a list"),
        }
    }

    #[test]
    fn test_boolean_operations() {
        let arena: Arena<1024, 256> = Arena::new();
        let env = init_env(&arena).unwrap();

        assert_eq!(eval_str(&arena, "(if #t 1 2)", env).unwrap(), 1);
        assert_eq!(eval_str(&arena, "(if #f 1 2)", env).unwrap(), 2);
        assert_eq!(eval_str(&arena, "(if (< 3 5) 1 2)", env).unwrap(), 1);
        assert_eq!(eval_str(&arena, "(if (< 5 3) 1 2)", env).unwrap(), 2);
    }

    #[test]
    fn test_comparison_operators() {
        let arena: Arena<1024, 256> = Arena::new();
        let env = init_env(&arena).unwrap();

        assert_eq!(eval_str(&arena, "(if (< 3 5) 1 0)", env).unwrap(), 1);
        assert_eq!(eval_str(&arena, "(if (< 5 3) 1 0)", env).unwrap(), 0);
        assert_eq!(eval_str(&arena, "(if (= 5 5) 1 0)", env).unwrap(), 1);
        assert_eq!(eval_str(&arena, "(if (= 5 3) 1 0)", env).unwrap(), 0);
    }

    #[test]
    fn test_nested_lambdas() {
        let arena: Arena<2048, 256> = Arena::new();
        let env = init_env(&arena).unwrap();

        let code = "
            (define make-adder
              (lambda (x)
                (lambda (y) (+ x y))))
        ";
        let expr = parse(&arena, code).unwrap();
        let _ = eval(&arena, expr, env).unwrap();

        let code2 = "(define add5 (make-adder 5))";
        let expr2 = parse(&arena, code2).unwrap();
        let _ = eval(&arena, expr2, env).unwrap();

        assert_eq!(eval_str(&arena, "(add5 10)", env).unwrap(), 15);
        assert_eq!(eval_str(&arena, "(add5 3)", env).unwrap(), 8);
    }

    #[test]
    fn test_higher_order_functions() {
        let arena: Arena<2048, 256> = Arena::new();
        let env = init_env(&arena).unwrap();

        let code = "
            (define apply-twice
              (lambda (f x)
                (f (f x))))
        ";
        let expr = parse(&arena, code).unwrap();
        let _ = eval(&arena, expr, env).unwrap();

        let code2 = "(define double (lambda (x) (* x 2)))";
        let expr2 = parse(&arena, code2).unwrap();
        let _ = eval(&arena, expr2, env).unwrap();

        assert_eq!(eval_str(&arena, "(apply-twice double 3)", env).unwrap(), 12);
    }

    #[test]
    fn test_multiple_definitions() {
        let arena: Arena<2048, 256> = Arena::new();
        let env = init_env(&arena).unwrap();

        let expr = parse(&arena, "(define x 10)").unwrap();
        let _ = eval(&arena, expr, env).unwrap();

        let expr = parse(&arena, "(define y 20)").unwrap();
        let _ = eval(&arena, expr, env).unwrap();

        let expr = parse(&arena, "(define z (+ x y))").unwrap();
        let _ = eval(&arena, expr, env).unwrap();

        assert_eq!(eval_str(&arena, "z", env).unwrap(), 30);
    }

    #[test]
    fn test_gc_preserves_live_data() {
        let arena: Arena<2048, 256> = Arena::new();
        let env = init_env(&arena).unwrap();

        // Create some values
        let expr = parse(&arena, "(define x 42)").unwrap();
        let _ = eval(&arena, expr, env).unwrap();

        let expr = parse(&arena, "(define y 100)").unwrap();
        let _ = eval(&arena, expr, env).unwrap();

        // Do GC
        arena.collect(&[env]);

        // Values should still be accessible
        assert_eq!(eval_str(&arena, "x", env).unwrap(), 42);
        assert_eq!(eval_str(&arena, "y", env).unwrap(), 100);
        assert_eq!(eval_str(&arena, "(+ x y)", env).unwrap(), 142);
    }
}
