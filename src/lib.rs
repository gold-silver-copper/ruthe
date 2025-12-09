#![no_std]

use core::cell::Cell;

// ============================================================================
// Arena Allocator with Reference Counting
// ============================================================================

const ARENA_SIZE: usize = 2048;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct ArenaRef(pub u16);

impl ArenaRef {
    const NULL: ArenaRef = ArenaRef(u16::MAX);

    fn is_null(self) -> bool {
        self.0 == u16::MAX
    }
}

#[derive(Clone, Copy)]
pub enum Value {
    Number(i64),
    Bool(bool),
    Char(char),
    Cons(ArenaRef, ArenaRef),
    Symbol(ArenaRef),                     // Points to char list
    Lambda(ArenaRef, ArenaRef, ArenaRef), // params, body, env
    Builtin(u8),                          // Index into builtin table
    Nil,
    Free, // Marks freed slot
}

pub struct Arena {
    // Separate arrays for better initialization
    values: [Value; ARENA_SIZE],
    refcounts: [Cell<u16>; ARENA_SIZE],
    next_free: Cell<usize>,
}

impl Arena {
    pub fn new() -> Self {
        Arena {
            values: [Value::Free; ARENA_SIZE],
            refcounts: [const { Cell::new(0) }; ARENA_SIZE],
            next_free: Cell::new(0),
        }
    }

    pub fn alloc(&mut self, value: Value) -> ArenaRef {
        let start = self.next_free.get();

        // Search for free slot starting from next_free
        for i in 0..ARENA_SIZE {
            let idx = (start + i) % ARENA_SIZE;
            if matches!(self.values[idx], Value::Free) {
                self.values[idx] = value;
                self.refcounts[idx].set(1);
                self.next_free.set((idx + 1) % ARENA_SIZE);
                return ArenaRef(idx as u16);
            }
        }

        ArenaRef::NULL
    }

    pub fn get(&self, r: ArenaRef) -> Option<&Value> {
        if r.is_null() {
            return None;
        }
        let value = &self.values[r.0 as usize];
        if matches!(value, Value::Free) {
            None
        } else {
            Some(value)
        }
    }

    pub fn incref(&self, r: ArenaRef) {
        if r.is_null() {
            return;
        }
        let idx = r.0 as usize;
        if !matches!(self.values[idx], Value::Free) {
            let rc = self.refcounts[idx].get();
            if rc < u16::MAX {
                self.refcounts[idx].set(rc + 1);
            }
        }
    }

    pub fn decref(&mut self, r: ArenaRef) {
        if r.is_null() {
            return;
        }

        let idx = r.0 as usize;
        if matches!(self.values[idx], Value::Free) {
            return;
        }

        let rc = self.refcounts[idx].get();
        if rc > 1 {
            self.refcounts[idx].set(rc - 1);
        } else if rc == 1 {
            // Free this node
            let value = self.values[idx];
            self.values[idx] = Value::Free;
            self.refcounts[idx].set(0);

            // Recursively decref children
            match value {
                Value::Cons(car, cdr) => {
                    self.decref(car);
                    self.decref(cdr);
                }
                Value::Symbol(s) => self.decref(s),
                Value::Lambda(params, body, env) => {
                    self.decref(params);
                    self.decref(body);
                    self.decref(env);
                }
                _ => {}
            }
        }
    }

    // Constructors
    pub fn nil(&mut self) -> ArenaRef {
        self.alloc(Value::Nil)
    }

    pub fn number(&mut self, n: i64) -> ArenaRef {
        self.alloc(Value::Number(n))
    }

    pub fn bool_val(&mut self, b: bool) -> ArenaRef {
        self.alloc(Value::Bool(b))
    }

    pub fn char_val(&mut self, c: char) -> ArenaRef {
        self.alloc(Value::Char(c))
    }

    pub fn cons(&mut self, car: ArenaRef, cdr: ArenaRef) -> ArenaRef {
        self.incref(car);
        self.incref(cdr);
        self.alloc(Value::Cons(car, cdr))
    }

    pub fn symbol(&mut self, s: ArenaRef) -> ArenaRef {
        self.incref(s);
        self.alloc(Value::Symbol(s))
    }

    pub fn lambda(&mut self, params: ArenaRef, body: ArenaRef, env: ArenaRef) -> ArenaRef {
        self.incref(params);
        self.incref(body);
        self.incref(env);
        self.alloc(Value::Lambda(params, body, env))
    }

    pub fn builtin(&mut self, idx: u8) -> ArenaRef {
        self.alloc(Value::Builtin(idx))
    }

    // String helpers
    pub fn str_to_list(&mut self, s: &str) -> ArenaRef {
        let mut result = self.nil();
        for ch in s.chars().rev() {
            let char_ref = self.char_val(ch);
            let new_cons = self.cons(char_ref, result);
            self.decref(char_ref);
            self.decref(result);
            result = new_cons;
        }
        result
    }

    pub fn list_to_str<'a>(&self, list: ArenaRef, buf: &'a mut [u8]) -> Option<&'a str> {
        let mut idx = 0;
        let mut current = list;

        loop {
            match self.get(current)? {
                Value::Char(ch) => {
                    let mut temp = [0u8; 4];
                    let s = ch.encode_utf8(&mut temp);
                    for &b in s.as_bytes() {
                        if idx >= buf.len() {
                            return None;
                        }
                        buf[idx] = b;
                        idx += 1;
                    }
                    break;
                }
                Value::Cons(car, cdr) => {
                    if let Value::Char(ch) = self.get(*car)? {
                        let mut temp = [0u8; 4];
                        let s = ch.encode_utf8(&mut temp);
                        for &b in s.as_bytes() {
                            if idx >= buf.len() {
                                return None;
                            }
                            buf[idx] = b;
                            idx += 1;
                        }
                        current = *cdr;
                    } else {
                        return None;
                    }
                }
                Value::Nil => break,
                _ => return None,
            }
        }

        core::str::from_utf8(&buf[..idx]).ok()
    }

    pub fn list_len(&self, mut list: ArenaRef) -> usize {
        let mut count = 0;
        loop {
            match self.get(list) {
                Some(Value::Cons(_, cdr)) => {
                    count += 1;
                    list = *cdr;
                }
                _ => break,
            }
        }
        count
    }

    pub fn list_nth(&self, mut list: ArenaRef, n: usize) -> Option<ArenaRef> {
        let mut idx = 0;
        loop {
            match self.get(list) {
                Some(Value::Cons(car, cdr)) => {
                    if idx == n {
                        return Some(*car);
                    }
                    idx += 1;
                    list = *cdr;
                }
                _ => return None,
            }
        }
    }

    pub fn reverse_list(&mut self, mut list: ArenaRef) -> ArenaRef {
        let mut result = self.nil();
        loop {
            // Extract values first to avoid borrow conflicts
            let pair = match self.get(list) {
                Some(Value::Cons(car, cdr)) => Some((*car, *cdr)),
                _ => None,
            };

            match pair {
                Some((car, cdr)) => {
                    let new_result = self.cons(car, result);
                    self.decref(result);
                    result = new_result;
                    list = cdr;
                }
                None => break,
            }
        }
        result
    }

    pub fn str_eq(&self, mut s1: ArenaRef, mut s2: ArenaRef) -> bool {
        loop {
            match (self.get(s1), self.get(s2)) {
                (Some(Value::Cons(c1, r1)), Some(Value::Cons(c2, r2))) => {
                    if let (Some(Value::Char(ch1)), Some(Value::Char(ch2))) =
                        (self.get(*c1), self.get(*c2))
                    {
                        if ch1 != ch2 {
                            return false;
                        }
                        s1 = *r1;
                        s2 = *r2;
                    } else {
                        return false;
                    }
                }
                (Some(Value::Nil), Some(Value::Nil)) => return true,
                _ => return false,
            }
        }
    }

    // Set cons cell in place (for environment mutation)
    pub fn set_cons(&mut self, cons_ref: ArenaRef, new_car: ArenaRef, new_cdr: ArenaRef) {
        if let Some(Value::Cons(old_car, old_cdr)) = self.get(cons_ref).copied() {
            self.incref(new_car);
            self.incref(new_cdr);
            self.values[cons_ref.0 as usize] = Value::Cons(new_car, new_cdr);
            self.decref(old_car);
            self.decref(old_cdr);
        }
    }
}

// ============================================================================
// Environment Operations
// ============================================================================

pub fn env_new(arena: &mut Arena) -> ArenaRef {
    // Create intermediate values to avoid multiple mutable borrows
    let nil1 = arena.nil();
    let nil2 = arena.nil();
    let env = arena.cons(nil1, nil2);
    arena.decref(nil1);
    arena.decref(nil2);
    register_builtins(arena, env);
    env
}

pub fn env_with_parent(arena: &mut Arena, parent: ArenaRef) -> ArenaRef {
    let nil = arena.nil();
    let result = arena.cons(nil, parent);
    arena.decref(nil);
    result
}

pub fn env_set(arena: &mut Arena, env: ArenaRef, name: ArenaRef, value: ArenaRef) {
    if let Some(Value::Cons(bindings, parent)) = arena.get(env).copied() {
        let sym = arena.symbol(name);
        let new_binding = arena.cons(sym, value);
        let new_bindings = arena.cons(new_binding, bindings);
        arena.set_cons(env, new_bindings, parent);
        arena.decref(sym);
        arena.decref(new_binding);
        arena.decref(new_bindings);
    }
}

pub fn env_get(arena: &Arena, mut env: ArenaRef, name: ArenaRef) -> Option<ArenaRef> {
    loop {
        match arena.get(env).copied() {
            Some(Value::Cons(bindings, parent)) => {
                let mut bindings_list = bindings;
                loop {
                    match arena.get(bindings_list).copied() {
                        Some(Value::Cons(binding, rest)) => {
                            if let Some(Value::Cons(key, value)) = arena.get(binding).copied() {
                                if let Some(Value::Symbol(s)) = arena.get(key).copied() {
                                    if arena.str_eq(s, name) {
                                        return Some(value);
                                    }
                                }
                            }
                            bindings_list = rest;
                        }
                        _ => break,
                    }
                }

                match arena.get(parent) {
                    Some(Value::Nil) => return None,
                    _ => env = parent,
                }
            }
            _ => return None,
        }
    }
}

// ============================================================================
// Builtin Functions
// ============================================================================

const BUILTIN_ADD: u8 = 0;
const BUILTIN_SUB: u8 = 1;
const BUILTIN_MUL: u8 = 2;
const BUILTIN_DIV: u8 = 3;
const BUILTIN_EQ: u8 = 4;
const BUILTIN_LT: u8 = 5;
const BUILTIN_GT: u8 = 6;
const BUILTIN_CAR: u8 = 7;
const BUILTIN_CDR: u8 = 8;
const BUILTIN_CONS: u8 = 9;
const BUILTIN_LIST: u8 = 10;
const BUILTIN_NULL: u8 = 11;

type BuiltinFn = fn(&mut Arena, ArenaRef) -> Result<ArenaRef, ArenaRef>;

const BUILTINS: [BuiltinFn; 12] = [
    builtin_add,
    builtin_sub,
    builtin_mul,
    builtin_div,
    builtin_eq,
    builtin_lt,
    builtin_gt,
    builtin_car,
    builtin_cdr,
    builtin_cons_fn,
    builtin_list,
    builtin_null,
];

fn call_builtin(arena: &mut Arena, idx: u8, args: ArenaRef) -> Result<ArenaRef, ArenaRef> {
    if (idx as usize) < BUILTINS.len() {
        BUILTINS[idx as usize](arena, args)
    } else {
        Err(arena.str_to_list("Unknown builtin"))
    }
}

fn builtin_add(arena: &mut Arena, args: ArenaRef) -> Result<ArenaRef, ArenaRef> {
    let mut result = 0i64;
    let mut current = args;

    loop {
        match arena.get(current).copied() {
            Some(Value::Cons(car, cdr)) => {
                if let Some(Value::Number(n)) = arena.get(car) {
                    result = result
                        .checked_add(*n)
                        .ok_or_else(|| arena.str_to_list("Overflow"))?;
                    current = cdr;
                } else {
                    return Err(arena.str_to_list("+ requires numbers"));
                }
            }
            Some(Value::Nil) => break,
            _ => return Err(arena.str_to_list("Invalid args")),
        }
    }

    Ok(arena.number(result))
}

fn builtin_sub(arena: &mut Arena, args: ArenaRef) -> Result<ArenaRef, ArenaRef> {
    let len = arena.list_len(args);
    if len == 0 {
        return Err(arena.str_to_list("- requires args"));
    }

    let first = arena.list_nth(args, 0).unwrap();
    let first_num = if let Some(Value::Number(n)) = arena.get(first) {
        *n
    } else {
        return Err(arena.str_to_list("- requires numbers"));
    };

    if len == 1 {
        return Ok(arena.number(-first_num));
    }

    let mut result = first_num;
    let mut current = if let Some(Value::Cons(_, cdr)) = arena.get(args).copied() {
        cdr
    } else {
        return Err(arena.str_to_list("Invalid args"));
    };

    loop {
        match arena.get(current).copied() {
            Some(Value::Cons(car, cdr)) => {
                if let Some(Value::Number(n)) = arena.get(car) {
                    result = result
                        .checked_sub(*n)
                        .ok_or_else(|| arena.str_to_list("Overflow"))?;
                    current = cdr;
                } else {
                    return Err(arena.str_to_list("- requires numbers"));
                }
            }
            Some(Value::Nil) => break,
            _ => return Err(arena.str_to_list("Invalid args")),
        }
    }

    Ok(arena.number(result))
}

fn builtin_mul(arena: &mut Arena, args: ArenaRef) -> Result<ArenaRef, ArenaRef> {
    let mut result = 1i64;
    let mut current = args;

    loop {
        match arena.get(current).copied() {
            Some(Value::Cons(car, cdr)) => {
                if let Some(Value::Number(n)) = arena.get(car) {
                    result = result
                        .checked_mul(*n)
                        .ok_or_else(|| arena.str_to_list("Overflow"))?;
                    current = cdr;
                } else {
                    return Err(arena.str_to_list("* requires numbers"));
                }
            }
            Some(Value::Nil) => break,
            _ => return Err(arena.str_to_list("Invalid args")),
        }
    }

    Ok(arena.number(result))
}

fn builtin_div(arena: &mut Arena, args: ArenaRef) -> Result<ArenaRef, ArenaRef> {
    let len = arena.list_len(args);
    if len < 2 {
        return Err(arena.str_to_list("/ requires 2+ args"));
    }

    let first = arena.list_nth(args, 0).unwrap();
    let mut result = if let Some(Value::Number(n)) = arena.get(first) {
        *n
    } else {
        return Err(arena.str_to_list("/ requires numbers"));
    };

    let mut current = if let Some(Value::Cons(_, cdr)) = arena.get(args).copied() {
        cdr
    } else {
        return Err(arena.str_to_list("Invalid args"));
    };

    loop {
        match arena.get(current).copied() {
            Some(Value::Cons(car, cdr)) => {
                if let Some(Value::Number(n)) = arena.get(car) {
                    if *n == 0 {
                        return Err(arena.str_to_list("Division by zero"));
                    }
                    result = result
                        .checked_div(*n)
                        .ok_or_else(|| arena.str_to_list("Overflow"))?;
                    current = cdr;
                } else {
                    return Err(arena.str_to_list("/ requires numbers"));
                }
            }
            Some(Value::Nil) => break,
            _ => return Err(arena.str_to_list("Invalid args")),
        }
    }

    Ok(arena.number(result))
}

fn builtin_eq(arena: &mut Arena, args: ArenaRef) -> Result<ArenaRef, ArenaRef> {
    if arena.list_len(args) != 2 {
        return Err(arena.str_to_list("= requires 2 args"));
    }
    let a = arena.list_nth(args, 0).unwrap();
    let b = arena.list_nth(args, 1).unwrap();

    let eq = match (arena.get(a), arena.get(b)) {
        (Some(Value::Number(x)), Some(Value::Number(y))) => x == y,
        _ => return Err(arena.str_to_list("= requires numbers")),
    };

    Ok(arena.bool_val(eq))
}

fn builtin_lt(arena: &mut Arena, args: ArenaRef) -> Result<ArenaRef, ArenaRef> {
    if arena.list_len(args) != 2 {
        return Err(arena.str_to_list("< requires 2 args"));
    }
    let a = arena.list_nth(args, 0).unwrap();
    let b = arena.list_nth(args, 1).unwrap();

    let lt = match (arena.get(a), arena.get(b)) {
        (Some(Value::Number(x)), Some(Value::Number(y))) => x < y,
        _ => return Err(arena.str_to_list("< requires numbers")),
    };

    Ok(arena.bool_val(lt))
}

fn builtin_gt(arena: &mut Arena, args: ArenaRef) -> Result<ArenaRef, ArenaRef> {
    if arena.list_len(args) != 2 {
        return Err(arena.str_to_list("> requires 2 args"));
    }
    let a = arena.list_nth(args, 0).unwrap();
    let b = arena.list_nth(args, 1).unwrap();

    let gt = match (arena.get(a), arena.get(b)) {
        (Some(Value::Number(x)), Some(Value::Number(y))) => x > y,
        _ => return Err(arena.str_to_list("> requires numbers")),
    };

    Ok(arena.bool_val(gt))
}

fn builtin_car(arena: &mut Arena, args: ArenaRef) -> Result<ArenaRef, ArenaRef> {
    if arena.list_len(args) != 1 {
        return Err(arena.str_to_list("car requires 1 arg"));
    }
    let list = arena.list_nth(args, 0).unwrap();
    if let Some(Value::Cons(car, _)) = arena.get(list) {
        let result = *car;
        arena.incref(result);
        Ok(result)
    } else {
        Err(arena.str_to_list("car requires cons"))
    }
}

fn builtin_cdr(arena: &mut Arena, args: ArenaRef) -> Result<ArenaRef, ArenaRef> {
    if arena.list_len(args) != 1 {
        return Err(arena.str_to_list("cdr requires 1 arg"));
    }
    let list = arena.list_nth(args, 0).unwrap();
    if let Some(Value::Cons(_, cdr)) = arena.get(list) {
        let result = *cdr;
        arena.incref(result);
        Ok(result)
    } else {
        Err(arena.str_to_list("cdr requires cons"))
    }
}

fn builtin_cons_fn(arena: &mut Arena, args: ArenaRef) -> Result<ArenaRef, ArenaRef> {
    if arena.list_len(args) != 2 {
        return Err(arena.str_to_list("cons requires 2 args"));
    }
    let car = arena.list_nth(args, 0).unwrap();
    let cdr = arena.list_nth(args, 1).unwrap();
    Ok(arena.cons(car, cdr))
}

fn builtin_list(arena: &mut Arena, args: ArenaRef) -> Result<ArenaRef, ArenaRef> {
    arena.incref(args);
    Ok(args)
}

fn builtin_null(arena: &mut Arena, args: ArenaRef) -> Result<ArenaRef, ArenaRef> {
    if arena.list_len(args) != 1 {
        return Err(arena.str_to_list("null? requires 1 arg"));
    }
    let val = arena.list_nth(args, 0).unwrap();
    let is_nil = matches!(arena.get(val), Some(Value::Nil));
    Ok(arena.bool_val(is_nil))
}

fn register_builtins(arena: &mut Arena, env: ArenaRef) {
    let builtins = [
        ("+", BUILTIN_ADD),
        ("-", BUILTIN_SUB),
        ("*", BUILTIN_MUL),
        ("/", BUILTIN_DIV),
        ("=", BUILTIN_EQ),
        ("<", BUILTIN_LT),
        (">", BUILTIN_GT),
        ("car", BUILTIN_CAR),
        ("cdr", BUILTIN_CDR),
        ("cons", BUILTIN_CONS),
        ("list", BUILTIN_LIST),
        ("null?", BUILTIN_NULL),
    ];

    for (name, idx) in builtins {
        let name_ref = arena.str_to_list(name);
        let builtin_ref = arena.builtin(idx);
        env_set(arena, env, name_ref, builtin_ref);
        arena.decref(name_ref);
        arena.decref(builtin_ref);
    }
}

// ============================================================================
// Tokenizer
// ============================================================================

fn is_delimiter(ch: char) -> bool {
    ch.is_whitespace() || ch == '(' || ch == ')' || ch == '\'' || ch == ';'
}

pub fn tokenize(arena: &mut Arena, input: &str) -> Result<ArenaRef, ArenaRef> {
    let mut result = arena.nil();
    let mut chars = input.chars().peekable();

    while let Some(&ch) = chars.peek() {
        match ch {
            ' ' | '\t' | '\n' | '\r' => {
                chars.next();
            }
            '(' => {
                let tok = arena.str_to_list("(");
                let sym = arena.symbol(tok);
                let new_result = arena.cons(sym, result);
                arena.decref(tok);
                arena.decref(sym);
                arena.decref(result);
                result = new_result;
                chars.next();
            }
            ')' => {
                let tok = arena.str_to_list(")");
                let sym = arena.symbol(tok);
                let new_result = arena.cons(sym, result);
                arena.decref(tok);
                arena.decref(sym);
                arena.decref(result);
                result = new_result;
                chars.next();
            }
            '\'' => {
                let tok = arena.str_to_list("'");
                let sym = arena.symbol(tok);
                let new_result = arena.cons(sym, result);
                arena.decref(tok);
                arena.decref(sym);
                arena.decref(result);
                result = new_result;
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
                        let val = arena.bool_val(true);
                        let new_result = arena.cons(val, result);
                        arena.decref(val);
                        arena.decref(result);
                        result = new_result;
                        chars.next();
                    }
                    Some(&'f') => {
                        let val = arena.bool_val(false);
                        let new_result = arena.cons(val, result);
                        arena.decref(val);
                        arena.decref(result);
                        result = new_result;
                        chars.next();
                    }
                    _ => return Err(arena.str_to_list("Invalid boolean")),
                }
            }
            _ => {
                let mut atom = arena.nil();
                while let Some(&c) = chars.peek() {
                    if is_delimiter(c) {
                        break;
                    }
                    let ch_val = arena.char_val(c);
                    let new_atom = arena.cons(ch_val, atom);
                    arena.decref(ch_val);
                    arena.decref(atom);
                    atom = new_atom;
                    chars.next();
                }

                atom = arena.reverse_list(atom);

                // Try to parse as number
                let mut buf = [0u8; 64];
                if let Some(s) = arena.list_to_str(atom, &mut buf) {
                    if let Ok(num) = parse_i64(s) {
                        let num_val = arena.number(num);
                        let new_result = arena.cons(num_val, result);
                        arena.decref(num_val);
                        arena.decref(atom);
                        arena.decref(result);
                        result = new_result;
                    } else {
                        let sym = arena.symbol(atom);
                        let new_result = arena.cons(sym, result);
                        arena.decref(sym);
                        arena.decref(atom);
                        arena.decref(result);
                        result = new_result;
                    }
                } else {
                    arena.decref(atom);
                    return Err(arena.str_to_list("Atom too long"));
                }
            }
        }
    }

    Ok(arena.reverse_list(result))
}

fn parse_i64(s: &str) -> Result<i64, ()> {
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

    let mut result = 0i64;
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

// ============================================================================
// Parser
// ============================================================================

pub fn parse(arena: &mut Arena, tokens: ArenaRef) -> Result<ArenaRef, ArenaRef> {
    let (expr, remaining) = parse_one(arena, tokens)?;
    if !matches!(arena.get(remaining), Some(Value::Nil)) {
        return Err(arena.str_to_list("Extra tokens"));
    }
    Ok(expr)
}

fn parse_one(arena: &mut Arena, tokens: ArenaRef) -> Result<(ArenaRef, ArenaRef), ArenaRef> {
    match arena.get(tokens).copied() {
        Some(Value::Nil) => Err(arena.str_to_list("Unexpected EOF")),
        Some(Value::Cons(first, rest)) => match arena.get(first) {
            Some(Value::Number(_)) | Some(Value::Bool(_)) => {
                arena.incref(first);
                arena.incref(rest);
                Ok((first, rest))
            }
            Some(Value::Symbol(s)) => {
                let mut buf = [0u8; 32];
                if let Some(s_str) = arena.list_to_str(*s, &mut buf) {
                    if s_str == "'" {
                        let (expr, consumed) = parse_one(arena, rest)?;
                        let quote_sym = arena.str_to_list("quote");
                        let quoted_sym = arena.symbol(quote_sym);
                        let nil = arena.nil();
                        let list = arena.cons(expr, nil);
                        let quoted = arena.cons(quoted_sym, list);
                        arena.decref(quote_sym);
                        arena.decref(quoted_sym);
                        arena.decref(expr);
                        arena.decref(nil);
                        arena.decref(list);
                        Ok((quoted, consumed))
                    } else if s_str == "(" {
                        parse_list(arena, rest)
                    } else if s_str == ")" {
                        Err(arena.str_to_list("Unexpected )"))
                    } else {
                        arena.incref(first);
                        arena.incref(rest);
                        Ok((first, rest))
                    }
                } else {
                    Err(arena.str_to_list("Symbol too long"))
                }
            }
            _ => Err(arena.str_to_list("Invalid token")),
        },
        _ => Err(arena.str_to_list("Invalid tokens")),
    }
}

fn parse_list(arena: &mut Arena, mut tokens: ArenaRef) -> Result<(ArenaRef, ArenaRef), ArenaRef> {
    let mut items = arena.nil();

    loop {
        match arena.get(tokens).copied() {
            Some(Value::Nil) => return Err(arena.str_to_list("Unmatched (")),
            Some(Value::Cons(tok, rest)) => {
                if let Some(Value::Symbol(s)) = arena.get(tok) {
                    let mut buf = [0u8; 32];
                    if let Some(s_str) = arena.list_to_str(*s, &mut buf) {
                        if s_str == ")" {
                            let result = arena.reverse_list(items);
                            arena.decref(items);
                            arena.incref(rest);
                            return Ok((result, rest));
                        }
                    }
                }

                let (expr, consumed) = parse_one(arena, tokens)?;
                let new_items = arena.cons(expr, items);
                arena.decref(expr);
                arena.decref(items);
                items = new_items;
                tokens = consumed;
            }
            _ => return Err(arena.str_to_list("Invalid tokens")),
        }
    }
}

// ============================================================================
// Evaluator
// ============================================================================

pub fn eval(arena: &mut Arena, expr: ArenaRef, env: ArenaRef) -> Result<ArenaRef, ArenaRef> {
    match arena.get(expr) {
        Some(Value::Number(_))
        | Some(Value::Bool(_))
        | Some(Value::Builtin(_))
        | Some(Value::Lambda(..)) => {
            arena.incref(expr);
            Ok(expr)
        }
        Some(Value::Symbol(s)) => {
            let s = *s;
            let nil_str = arena.str_to_list("nil");
            let is_nil = arena.str_eq(s, nil_str);
            arena.decref(nil_str);

            if is_nil {
                return Ok(arena.nil());
            }

            if let Some(val) = env_get(arena, env, s) {
                arena.incref(val);
                Ok(val)
            } else {
                Err(arena.str_to_list("Unbound symbol"))
            }
        }
        Some(Value::Cons(car, _)) => {
            let car = *car;

            // Check for special forms
            if let Some(Value::Symbol(sym)) = arena.get(car) {
                let sym = *sym;
                let mut buf = [0u8; 32];
                if let Some(sym_str) = arena.list_to_str(sym, &mut buf) {
                    match sym_str {
                        "quote" => {
                            if let Some(quoted) = arena.list_nth(expr, 1) {
                                arena.incref(quoted);
                                return Ok(quoted);
                            }
                        }
                        "if" => {
                            return eval_if(arena, expr, env);
                        }
                        "lambda" => {
                            return eval_lambda(arena, expr, env);
                        }
                        "define" => {
                            return eval_define(arena, expr, env);
                        }
                        _ => {}
                    }
                }
            }

            // Function application
            let func = eval(arena, car, env)?;
            let args = eval_args(arena, expr, env)?;
            let result = apply(arena, func, args, env)?;
            arena.decref(func);
            arena.decref(args);
            Ok(result)
        }
        Some(Value::Nil) => Ok(arena.nil()),
        _ => Err(arena.str_to_list("Invalid expression")),
    }
}

fn eval_if(arena: &mut Arena, expr: ArenaRef, env: ArenaRef) -> Result<ArenaRef, ArenaRef> {
    if arena.list_len(expr) != 4 {
        return Err(arena.str_to_list("if requires 3 args"));
    }

    let cond_expr = arena.list_nth(expr, 1).unwrap();
    let cond = eval(arena, cond_expr, env)?;

    let is_true = match arena.get(cond) {
        Some(Value::Bool(b)) => *b,
        Some(Value::Nil) => false,
        _ => true,
    };

    arena.decref(cond);

    let branch = if is_true {
        arena.list_nth(expr, 2).unwrap()
    } else {
        arena.list_nth(expr, 3).unwrap()
    };

    eval(arena, branch, env)
}

fn eval_lambda(arena: &mut Arena, expr: ArenaRef, env: ArenaRef) -> Result<ArenaRef, ArenaRef> {
    if arena.list_len(expr) != 3 {
        return Err(arena.str_to_list("lambda requires 2 args"));
    }

    let params = arena.list_nth(expr, 1).unwrap();
    let body = arena.list_nth(expr, 2).unwrap();

    Ok(arena.lambda(params, body, env))
}

fn eval_define(arena: &mut Arena, expr: ArenaRef, env: ArenaRef) -> Result<ArenaRef, ArenaRef> {
    if arena.list_len(expr) != 3 {
        return Err(arena.str_to_list("define requires 2 args"));
    }

    let name = arena.list_nth(expr, 1).unwrap();
    let name_sym = if let Some(Value::Symbol(s)) = arena.get(name) {
        *s
    } else {
        return Err(arena.str_to_list("define needs symbol"));
    };

    let value_expr = arena.list_nth(expr, 2).unwrap();
    let value = eval(arena, value_expr, env)?;

    env_set(arena, env, name_sym, value);
    arena.incref(value);
    Ok(value)
}

fn eval_args(arena: &mut Arena, expr: ArenaRef, env: ArenaRef) -> Result<ArenaRef, ArenaRef> {
    let mut result = arena.nil();
    let mut current = if let Some(Value::Cons(_, cdr)) = arena.get(expr).copied() {
        cdr
    } else {
        return Ok(result);
    };

    loop {
        match arena.get(current).copied() {
            Some(Value::Cons(car, cdr)) => {
                let val = eval(arena, car, env)?;
                let new_result = arena.cons(val, result);
                arena.decref(val);
                arena.decref(result);
                result = new_result;
                current = cdr;
            }
            Some(Value::Nil) => break,
            _ => {
                arena.decref(result);
                return Err(arena.str_to_list("Invalid args"));
            }
        }
    }

    Ok(arena.reverse_list(result))
}

fn apply(
    arena: &mut Arena,
    func: ArenaRef,
    args: ArenaRef,
    _env: ArenaRef,
) -> Result<ArenaRef, ArenaRef> {
    match arena.get(func).copied() {
        Some(Value::Builtin(idx)) => call_builtin(arena, idx, args),
        Some(Value::Lambda(params, body, lambda_env)) => {
            let call_env = env_with_parent(arena, lambda_env);

            let mut p = params;
            let mut a = args;

            loop {
                match (arena.get(p).copied(), arena.get(a).copied()) {
                    (Some(Value::Cons(p_car, p_cdr)), Some(Value::Cons(a_car, a_cdr))) => {
                        if let Some(Value::Symbol(name)) = arena.get(p_car) {
                            env_set(arena, call_env, *name, a_car);
                        }
                        p = p_cdr;
                        a = a_cdr;
                    }
                    (Some(Value::Nil), Some(Value::Nil)) => break,
                    _ => {
                        arena.decref(call_env);
                        return Err(arena.str_to_list("Arg count mismatch"));
                    }
                }
            }

            let result = eval(arena, body, call_env)?;
            arena.decref(call_env);
            Ok(result)
        }
        _ => Err(arena.str_to_list("Not callable")),
    }
}

// ============================================================================
// High-level API
// ============================================================================

pub fn eval_string(arena: &mut Arena, input: &str, env: ArenaRef) -> Result<ArenaRef, ArenaRef> {
    let tokens = tokenize(arena, input)?;
    let expr = parse(arena, tokens)?;
    let result = eval(arena, expr, env)?;
    arena.decref(tokens);
    arena.decref(expr);
    Ok(result)
}

// ============================================================================
// Example/Test
// ============================================================================

pub fn run_example() -> Result<(), ()> {
    let mut arena = Arena::new();
    let env = env_new(&mut arena);

    // Test: (+ 1 2 3)
    let result = eval_string(&mut arena, "(+ 1 2 3)", env);
    if let Ok(val) = result {
        if let Some(Value::Number(_n)) = arena.get(val) {
            // Should be 6
            arena.decref(val);
        }
    }

    // Test: (define x 42)
    let _ = eval_string(&mut arena, "(define x 42)", env);

    // Test: (* x 2)
    let result = eval_string(&mut arena, "(* x 2)", env);
    if let Ok(val) = result {
        if let Some(Value::Number(_n)) = arena.get(val) {
            // Should be 84
            arena.decref(val);
        }
    }

    arena.decref(env);
    Ok(())
}
