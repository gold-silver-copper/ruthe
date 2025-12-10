#![no_std]

use core::cell::Cell;
use core::ops::Deref;

// ============================================================================
// Arena Allocator with AUTOMATIC Reference Counting
// ============================================================================

pub const ARENA_SIZE: usize = 10000;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct ArenaRef(pub usize);

impl ArenaRef {
    pub const NULL: ArenaRef = ArenaRef(usize::MAX);

    pub fn is_null(self) -> bool {
        self.0 == usize::MAX
    }
}

#[derive(Clone, Copy, Debug)]
pub enum Value {
    Number(i64),
    Bool(bool),
    Char(char),
    Cons(ArenaRef, ArenaRef),
    Symbol(ArenaRef),
    Lambda(ArenaRef, ArenaRef, ArenaRef),
    Builtin(u8),
    Nil,
    Free,
}

#[derive(Debug)]
pub struct Arena {
    pub values: [Cell<Value>; ARENA_SIZE],
    pub refcounts: [Cell<usize>; ARENA_SIZE],
    pub next_free: Cell<usize>,
}

impl Arena {
    pub fn new() -> Self {
        Arena {
            values: [const { Cell::new(Value::Free) }; ARENA_SIZE],
            refcounts: [const { Cell::new(0) }; ARENA_SIZE],
            next_free: Cell::new(0),
        }
    }

    fn alloc(&self, value: Value) -> ArenaRef {
        let start = self.next_free.get();

        for i in 0..ARENA_SIZE {
            let idx = (start + i) % ARENA_SIZE;
            if matches!(self.values[idx].get(), Value::Free) {
                self.values[idx].set(value);
                self.refcounts[idx].set(1);
                self.next_free.set((idx + 1) % ARENA_SIZE);
                return ArenaRef(idx as usize);
            }
        }

        ArenaRef::NULL
    }

    pub fn get(&self, r: ArenaRef) -> Option<Value> {
        if r.is_null() {
            return None;
        }
        let value = self.values[r.0 as usize].get();
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
        if !matches!(self.values[idx].get(), Value::Free) {
            let rc = self.refcounts[idx].get();
            if rc < usize::MAX {
                self.refcounts[idx].set(rc + 1);
            }
        }
    }

    pub fn decref(&self, r: ArenaRef) {
        if r.is_null() {
            return;
        }

        let idx = r.0 as usize;
        let value = self.values[idx].get();
        if matches!(value, Value::Free) {
            return;
        }

        let rc = self.refcounts[idx].get();
        if rc > 1 {
            self.refcounts[idx].set(rc - 1);
        } else if rc == 1 {
            self.values[idx].set(Value::Free);
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

    // High-level API that returns RAII-wrapped references
    pub fn nil(&self) -> Ref {
        Ref::from_alloc(self, self.alloc(Value::Nil))
    }

    pub fn number(&self, n: i64) -> Ref {
        Ref::from_alloc(self, self.alloc(Value::Number(n)))
    }

    pub fn bool_val(&self, b: bool) -> Ref {
        Ref::from_alloc(self, self.alloc(Value::Bool(b)))
    }

    pub fn char_val(&self, c: char) -> Ref {
        Ref::from_alloc(self, self.alloc(Value::Char(c)))
    }

    pub fn cons(&self, car: &Ref, cdr: &Ref) -> Ref {
        self.incref(**car);
        self.incref(**cdr);
        Ref::from_alloc(self, self.alloc(Value::Cons(**car, **cdr)))
    }

    pub fn symbol(&self, s: &Ref) -> Ref {
        self.incref(**s);
        Ref::from_alloc(self, self.alloc(Value::Symbol(**s)))
    }

    pub fn lambda(&self, params: &Ref, body: &Ref, env: &Ref) -> Ref {
        self.incref(**params);
        self.incref(**body);
        self.incref(**env);
        Ref::from_alloc(self, self.alloc(Value::Lambda(**params, **body, **env)))
    }

    pub fn builtin(&self, idx: u8) -> Ref {
        Ref::from_alloc(self, self.alloc(Value::Builtin(idx)))
    }

    pub fn set_cons(&self, cons_ref: &Ref, new_car: &Ref, new_cdr: &Ref) {
        if let Some(Value::Cons(old_car, old_cdr)) = self.get(**cons_ref) {
            self.incref(**new_car);
            self.incref(**new_cdr);
            self.values[cons_ref.inner.0 as usize].set(Value::Cons(**new_car, **new_cdr));
            self.decref(old_car);
            self.decref(old_cdr);
        }
    }
}

// ============================================================================
// RAII Reference Wrapper - Automatic Reference Counting!
// ============================================================================
#[derive(Debug)]
pub struct Ref {
    arena: *const Arena,
    inner: ArenaRef,
}

impl Ref {
    pub fn new(arena: &Arena, r: ArenaRef) -> Self {
        arena.incref(r);
        Ref {
            arena: arena as *const Arena,
            inner: r,
        }
    }

    fn from_alloc(arena: &Arena, r: ArenaRef) -> Self {
        Ref {
            arena: arena as *const Arena,
            inner: r,
        }
    }

    pub fn raw(&self) -> ArenaRef {
        self.inner
    }

    pub fn get(&self) -> Option<Value> {
        unsafe { (*self.arena).get(self.inner) }
    }

    pub fn is_null(&self) -> bool {
        self.inner.is_null()
    }

    fn arena(&self) -> &Arena {
        unsafe { &*self.arena }
    }
}

impl Clone for Ref {
    fn clone(&self) -> Self {
        self.arena().incref(self.inner);
        Ref {
            arena: self.arena,
            inner: self.inner,
        }
    }
}

impl Drop for Ref {
    fn drop(&mut self) {
        self.arena().decref(self.inner);
    }
}

impl Deref for Ref {
    type Target = ArenaRef;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

// ============================================================================
// Helper Functions with Automatic Reference Counting
// ============================================================================

impl Arena {
    pub fn str_to_list(&self, s: &str) -> Ref {
        let mut result = self.nil();
        for ch in s.chars().rev() {
            let char_ref = self.char_val(ch);
            result = self.cons(&char_ref, &result);
        }
        result
    }

    pub fn list_to_str<'a>(&self, list: &Ref, buf: &'a mut [u8]) -> Option<&'a str> {
        let mut idx = 0;
        let mut current = list.clone();

        loop {
            match self.get(*current)? {
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
                    if let Some(Value::Char(ch)) = self.get(car) {
                        let mut temp = [0u8; 4];
                        let s = ch.encode_utf8(&mut temp);
                        for &b in s.as_bytes() {
                            if idx >= buf.len() {
                                return None;
                            }
                            buf[idx] = b;
                            idx += 1;
                        }
                        current = Ref::new(self, cdr);
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

    pub fn list_len(&self, list: &Ref) -> usize {
        let mut count = 0;
        let mut current = list.clone();
        loop {
            match self.get(*current) {
                Some(Value::Cons(_, cdr)) => {
                    count += 1;
                    current = Ref::new(self, cdr);
                }
                _ => break,
            }
        }
        count
    }

    pub fn list_nth(&self, list: &Ref, n: usize) -> Option<Ref> {
        let mut idx = 0;
        let mut current = list.clone();
        loop {
            match self.get(*current) {
                Some(Value::Cons(car, cdr)) => {
                    if idx == n {
                        return Some(Ref::new(self, car));
                    }
                    idx += 1;
                    current = Ref::new(self, cdr);
                }
                _ => return None,
            }
        }
    }

    pub fn reverse_list(&self, list: &Ref) -> Ref {
        let mut result = self.nil();
        let mut current = list.clone();
        loop {
            match self.get(*current) {
                Some(Value::Cons(car, cdr)) => {
                    let car_ref = Ref::new(self, car);
                    result = self.cons(&car_ref, &result);
                    current = Ref::new(self, cdr);
                }
                _ => break,
            }
        }
        result
    }

    pub fn str_eq(&self, s1: &Ref, s2: &Ref) -> bool {
        let mut c1 = s1.clone();
        let mut c2 = s2.clone();
        loop {
            match (self.get(*c1), self.get(*c2)) {
                (Some(Value::Cons(car1, cdr1)), Some(Value::Cons(car2, cdr2))) => {
                    if let (Some(Value::Char(ch1)), Some(Value::Char(ch2))) =
                        (self.get(car1), self.get(car2))
                    {
                        if ch1 != ch2 {
                            return false;
                        }
                        c1 = Ref::new(self, cdr1);
                        c2 = Ref::new(self, cdr2);
                    } else {
                        return false;
                    }
                }
                (Some(Value::Nil), Some(Value::Nil)) => return true,
                _ => return false,
            }
        }
    }
}

// ============================================================================
// Environment Operations
// ============================================================================

pub fn env_new(arena: &Arena) -> Ref {
    let nil1 = arena.nil();
    let nil2 = arena.nil();
    let env = arena.cons(&nil1, &nil2);
    register_builtins(arena, &env);
    env
}

pub fn env_with_parent(arena: &Arena, parent: &Ref) -> Ref {
    let nil = arena.nil();
    arena.cons(&nil, parent)
}

pub fn env_set(arena: &Arena, env: &Ref, name: &Ref, value: &Ref) {
    if let Some(Value::Cons(bindings, parent)) = arena.get(**env) {
        let bindings_ref = Ref::new(arena, bindings);
        let parent_ref = Ref::new(arena, parent);

        let sym = arena.symbol(name);
        let new_binding = arena.cons(&sym, value);
        let new_bindings = arena.cons(&new_binding, &bindings_ref);

        arena.set_cons(env, &new_bindings, &parent_ref);
    }
}

pub fn env_get(arena: &Arena, env: &Ref, name: &Ref) -> Option<Ref> {
    let mut current_env = env.clone();
    loop {
        match arena.get(*current_env) {
            Some(Value::Cons(bindings, parent)) => {
                let mut bindings_list = Ref::new(arena, bindings);
                loop {
                    match arena.get(*bindings_list) {
                        Some(Value::Cons(binding, rest)) => {
                            if let Some(Value::Cons(key, value)) = arena.get(binding) {
                                if let Some(Value::Symbol(s)) = arena.get(key) {
                                    let key_sym = Ref::new(arena, s);
                                    if arena.str_eq(&key_sym, name) {
                                        return Some(Ref::new(arena, value));
                                    }
                                }
                            }
                            bindings_list = Ref::new(arena, rest);
                        }
                        _ => break,
                    }
                }

                match arena.get(parent) {
                    Some(Value::Nil) => return None,
                    _ => current_env = Ref::new(arena, parent),
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
const BUILTIN_LENGTH: u8 = 12;
const BUILTIN_APPEND: u8 = 13;
const BUILTIN_REVERSE: u8 = 14;

type BuiltinFn = fn(&Arena, &Ref) -> Result<Ref, Ref>;

const BUILTINS: [BuiltinFn; 15] = [
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
    builtin_length,
    builtin_append,
    builtin_reverse,
];

fn call_builtin(arena: &Arena, idx: u8, args: &Ref) -> Result<Ref, Ref> {
    if (idx as usize) < BUILTINS.len() {
        BUILTINS[idx as usize](arena, args)
    } else {
        Err(arena.str_to_list("Unknown builtin"))
    }
}

fn builtin_add(arena: &Arena, args: &Ref) -> Result<Ref, Ref> {
    let mut result = 0i64;
    let mut current = args.clone();

    loop {
        match arena.get(*current) {
            Some(Value::Cons(car, cdr)) => {
                if let Some(Value::Number(n)) = arena.get(car) {
                    result = result
                        .checked_add(n)
                        .ok_or_else(|| arena.str_to_list("Overflow"))?;
                    current = Ref::new(arena, cdr);
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

fn builtin_sub(arena: &Arena, args: &Ref) -> Result<Ref, Ref> {
    let len = arena.list_len(args);
    if len == 0 {
        return Err(arena.str_to_list("- requires args"));
    }

    let first = arena.list_nth(args, 0).unwrap();
    let first_num = if let Some(Value::Number(n)) = arena.get(*first) {
        n
    } else {
        return Err(arena.str_to_list("- requires numbers"));
    };

    if len == 1 {
        return Ok(arena.number(-first_num));
    }

    let mut result = first_num;
    let mut current = if let Some(Value::Cons(_, cdr)) = arena.get(**args) {
        Ref::new(arena, cdr)
    } else {
        return Err(arena.str_to_list("Invalid args"));
    };

    loop {
        match arena.get(*current) {
            Some(Value::Cons(car, cdr)) => {
                if let Some(Value::Number(n)) = arena.get(car) {
                    result = result
                        .checked_sub(n)
                        .ok_or_else(|| arena.str_to_list("Overflow"))?;
                    current = Ref::new(arena, cdr);
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

fn builtin_mul(arena: &Arena, args: &Ref) -> Result<Ref, Ref> {
    let mut result = 1i64;
    let mut current = args.clone();

    loop {
        match arena.get(*current) {
            Some(Value::Cons(car, cdr)) => {
                if let Some(Value::Number(n)) = arena.get(car) {
                    result = result
                        .checked_mul(n)
                        .ok_or_else(|| arena.str_to_list("Overflow"))?;
                    current = Ref::new(arena, cdr);
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

fn builtin_div(arena: &Arena, args: &Ref) -> Result<Ref, Ref> {
    let len = arena.list_len(args);
    if len < 2 {
        return Err(arena.str_to_list("/ requires 2+ args"));
    }

    let first = arena.list_nth(args, 0).unwrap();
    let mut result = if let Some(Value::Number(n)) = arena.get(*first) {
        n
    } else {
        return Err(arena.str_to_list("/ requires numbers"));
    };

    let mut current = if let Some(Value::Cons(_, cdr)) = arena.get(**args) {
        Ref::new(arena, cdr)
    } else {
        return Err(arena.str_to_list("Invalid args"));
    };

    loop {
        match arena.get(*current) {
            Some(Value::Cons(car, cdr)) => {
                if let Some(Value::Number(n)) = arena.get(car) {
                    if n == 0 {
                        return Err(arena.str_to_list("Division by zero"));
                    }
                    result = result
                        .checked_div(n)
                        .ok_or_else(|| arena.str_to_list("Overflow"))?;
                    current = Ref::new(arena, cdr);
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

fn builtin_eq(arena: &Arena, args: &Ref) -> Result<Ref, Ref> {
    if arena.list_len(args) != 2 {
        return Err(arena.str_to_list("= requires 2 args"));
    }
    let a = arena.list_nth(args, 0).unwrap();
    let b = arena.list_nth(args, 1).unwrap();

    let eq = match (arena.get(*a), arena.get(*b)) {
        (Some(Value::Number(x)), Some(Value::Number(y))) => x == y,
        _ => return Err(arena.str_to_list("= requires numbers")),
    };

    Ok(arena.bool_val(eq))
}

fn builtin_lt(arena: &Arena, args: &Ref) -> Result<Ref, Ref> {
    if arena.list_len(args) != 2 {
        return Err(arena.str_to_list("< requires 2 args"));
    }
    let a = arena.list_nth(args, 0).unwrap();
    let b = arena.list_nth(args, 1).unwrap();

    let lt = match (arena.get(*a), arena.get(*b)) {
        (Some(Value::Number(x)), Some(Value::Number(y))) => x < y,
        _ => return Err(arena.str_to_list("< requires numbers")),
    };

    Ok(arena.bool_val(lt))
}

fn builtin_gt(arena: &Arena, args: &Ref) -> Result<Ref, Ref> {
    if arena.list_len(args) != 2 {
        return Err(arena.str_to_list("> requires 2 args"));
    }
    let a = arena.list_nth(args, 0).unwrap();
    let b = arena.list_nth(args, 1).unwrap();

    let gt = match (arena.get(*a), arena.get(*b)) {
        (Some(Value::Number(x)), Some(Value::Number(y))) => x > y,
        _ => return Err(arena.str_to_list("> requires numbers")),
    };

    Ok(arena.bool_val(gt))
}

fn builtin_car(arena: &Arena, args: &Ref) -> Result<Ref, Ref> {
    if arena.list_len(args) != 1 {
        return Err(arena.str_to_list("car requires 1 arg"));
    }
    let list = arena.list_nth(args, 0).unwrap();
    if let Some(Value::Cons(car, _)) = arena.get(*list) {
        Ok(Ref::new(arena, car))
    } else {
        Err(arena.str_to_list("car requires cons"))
    }
}

fn builtin_cdr(arena: &Arena, args: &Ref) -> Result<Ref, Ref> {
    if arena.list_len(args) != 1 {
        return Err(arena.str_to_list("cdr requires 1 arg"));
    }
    let list = arena.list_nth(args, 0).unwrap();
    if let Some(Value::Cons(_, cdr)) = arena.get(*list) {
        Ok(Ref::new(arena, cdr))
    } else {
        Err(arena.str_to_list("cdr requires cons"))
    }
}

fn builtin_cons_fn(arena: &Arena, args: &Ref) -> Result<Ref, Ref> {
    if arena.list_len(args) != 2 {
        return Err(arena.str_to_list("cons requires 2 args"));
    }
    let car = arena.list_nth(args, 0).unwrap();
    let cdr = arena.list_nth(args, 1).unwrap();
    Ok(arena.cons(&car, &cdr))
}

fn builtin_list(_arena: &Arena, args: &Ref) -> Result<Ref, Ref> {
    Ok(args.clone())
}

fn builtin_null(arena: &Arena, args: &Ref) -> Result<Ref, Ref> {
    if arena.list_len(args) != 1 {
        return Err(arena.str_to_list("null? requires 1 arg"));
    }
    let val = arena.list_nth(args, 0).unwrap();
    let is_nil = matches!(arena.get(*val), Some(Value::Nil));
    Ok(arena.bool_val(is_nil))
}

fn builtin_length(arena: &Arena, args: &Ref) -> Result<Ref, Ref> {
    if arena.list_len(args) != 1 {
        return Err(arena.str_to_list("length requires 1 arg"));
    }
    let list = arena.list_nth(args, 0).unwrap();
    let len = arena.list_len(&list);
    Ok(arena.number(len as i64))
}

fn builtin_append(arena: &Arena, args: &Ref) -> Result<Ref, Ref> {
    let mut result = arena.nil();
    let mut current = args.clone();

    loop {
        match arena.get(*current) {
            Some(Value::Cons(list, rest)) => {
                let mut list_cur = Ref::new(arena, list);
                loop {
                    match arena.get(*list_cur) {
                        Some(Value::Cons(item, item_rest)) => {
                            let item_ref = Ref::new(arena, item);
                            result = arena.cons(&item_ref, &result);
                            list_cur = Ref::new(arena, item_rest);
                        }
                        Some(Value::Nil) => break,
                        _ => break,
                    }
                }
                current = Ref::new(arena, rest);
            }
            Some(Value::Nil) => break,
            _ => {
                return Err(arena.str_to_list("Invalid argument list"));
            }
        }
    }

    Ok(arena.reverse_list(&result))
}

fn builtin_reverse(arena: &Arena, args: &Ref) -> Result<Ref, Ref> {
    if arena.list_len(args) != 1 {
        return Err(arena.str_to_list("reverse requires 1 arg"));
    }
    let list = arena.list_nth(args, 0).unwrap();
    Ok(arena.reverse_list(&list))
}

fn register_builtins(arena: &Arena, env: &Ref) {
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
        ("length", BUILTIN_LENGTH),
        ("append", BUILTIN_APPEND),
        ("reverse", BUILTIN_REVERSE),
    ];

    for (name, idx) in builtins {
        let name_ref = arena.str_to_list(name);
        let builtin_ref = arena.builtin(idx);
        env_set(arena, env, &name_ref, &builtin_ref);
    }
}

// ============================================================================
// Tokenizer
// ============================================================================

fn is_delimiter(ch: char) -> bool {
    ch.is_whitespace() || ch == '(' || ch == ')' || ch == '\'' || ch == ';'
}

pub fn tokenize(arena: &Arena, input: &str) -> Result<Ref, Ref> {
    let mut result = arena.nil();
    let mut chars = input.chars().peekable();

    while let Some(&ch) = chars.peek() {
        match ch {
            ' ' | '\t' | '\n' | '\r' => {
                chars.next();
            }
            '(' => {
                let tok = arena.str_to_list("(");
                let sym = arena.symbol(&tok);
                result = arena.cons(&sym, &result);
                chars.next();
            }
            ')' => {
                let tok = arena.str_to_list(")");
                let sym = arena.symbol(&tok);
                result = arena.cons(&sym, &result);
                chars.next();
            }
            '\'' => {
                let tok = arena.str_to_list("'");
                let sym = arena.symbol(&tok);
                result = arena.cons(&sym, &result);
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
                        result = arena.cons(&val, &result);
                        chars.next();
                    }
                    Some(&'f') => {
                        let val = arena.bool_val(false);
                        result = arena.cons(&val, &result);
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
                    atom = arena.cons(&ch_val, &atom);
                    chars.next();
                }

                atom = arena.reverse_list(&atom);

                let mut buf = [0u8; 64];
                if let Some(s) = arena.list_to_str(&atom, &mut buf) {
                    if let Ok(num) = parse_i64(s) {
                        let num_val = arena.number(num);
                        result = arena.cons(&num_val, &result);
                    } else {
                        let sym = arena.symbol(&atom);
                        result = arena.cons(&sym, &result);
                    }
                } else {
                    return Err(arena.str_to_list("Atom too long"));
                }
            }
        }
    }

    Ok(arena.reverse_list(&result))
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

pub fn parse(arena: &Arena, tokens: &Ref) -> Result<Ref, Ref> {
    let (expr, remaining) = parse_one(arena, tokens)?;
    if !matches!(arena.get(*remaining), Some(Value::Nil)) {
        return Err(arena.str_to_list("Extra tokens"));
    }
    Ok(expr)
}

fn parse_one(arena: &Arena, tokens: &Ref) -> Result<(Ref, Ref), Ref> {
    match arena.get(**tokens) {
        Some(Value::Nil) => Err(arena.str_to_list("Unexpected EOF")),
        Some(Value::Cons(first, rest)) => {
            let first_ref = Ref::new(arena, first);
            let rest_ref = Ref::new(arena, rest);

            match arena.get(first) {
                Some(Value::Number(_)) | Some(Value::Bool(_)) => Ok((first_ref, rest_ref)),
                Some(Value::Symbol(s)) => {
                    let mut buf = [0u8; 32];
                    if let Some(s_str) = arena.list_to_str(&Ref::new(arena, s), &mut buf) {
                        if s_str == "'" {
                            let (expr, consumed) = parse_one(arena, &rest_ref)?;
                            let quote_sym = arena.str_to_list("quote");
                            let quoted_sym = arena.symbol(&quote_sym);
                            let nil = arena.nil();
                            let list = arena.cons(&expr, &nil);
                            let quoted = arena.cons(&quoted_sym, &list);
                            Ok((quoted, consumed))
                        } else if s_str == "(" {
                            parse_list(arena, &rest_ref)
                        } else if s_str == ")" {
                            Err(arena.str_to_list("Unexpected )"))
                        } else {
                            Ok((first_ref, rest_ref))
                        }
                    } else {
                        Err(arena.str_to_list("Symbol too long"))
                    }
                }
                _ => Err(arena.str_to_list("Invalid token")),
            }
        }
        _ => Err(arena.str_to_list("Invalid tokens")),
    }
}

fn parse_list(arena: &Arena, tokens: &Ref) -> Result<(Ref, Ref), Ref> {
    let mut items = arena.nil();
    let mut current = tokens.clone();

    loop {
        match arena.get(*current) {
            Some(Value::Nil) => return Err(arena.str_to_list("Unmatched (")),
            Some(Value::Cons(tok, rest)) => {
                if let Some(Value::Symbol(s)) = arena.get(tok) {
                    let mut buf = [0u8; 32];
                    if let Some(s_str) = arena.list_to_str(&Ref::new(arena, s), &mut buf) {
                        if s_str == ")" {
                            let result = arena.reverse_list(&items);
                            return Ok((result, Ref::new(arena, rest)));
                        }
                    }
                }

                let (expr, consumed) = parse_one(arena, &current)?;
                items = arena.cons(&expr, &items);
                current = consumed;
            }
            _ => return Err(arena.str_to_list("Invalid tokens")),
        }
    }
}

// ============================================================================
// Evaluator with PROPER Tail Call Optimization
// ============================================================================

pub fn eval(arena: &Arena, expr: &Ref, env: &Ref) -> Result<Ref, Ref> {
    let mut current_expr = expr.clone();
    let mut current_env = env.clone();

    loop {
        match eval_step(arena, &current_expr, &current_env)? {
            EvalResult::Done(val) => return Ok(val),
            EvalResult::TailCall(new_expr, new_env) => {
                current_expr = new_expr;
                current_env = new_env;
            }
        }
    }
}

pub enum EvalResult {
    Done(Ref),
    TailCall(Ref, Ref),
}

fn eval_step(arena: &Arena, expr: &Ref, env: &Ref) -> Result<EvalResult, Ref> {
    match arena.get(**expr) {
        Some(Value::Number(_))
        | Some(Value::Bool(_))
        | Some(Value::Builtin(_))
        | Some(Value::Lambda(..)) => Ok(EvalResult::Done(expr.clone())),
        Some(Value::Symbol(s)) => {
            let s_ref = Ref::new(arena, s);
            let nil_str = arena.str_to_list("nil");
            let is_nil = arena.str_eq(&s_ref, &nil_str);

            if is_nil {
                return Ok(EvalResult::Done(arena.nil()));
            }

            if let Some(val) = env_get(arena, env, &s_ref) {
                Ok(EvalResult::Done(val))
            } else {
                Err(arena.str_to_list("Unbound symbol"))
            }
        }
        Some(Value::Cons(car, _)) => {
            let car_ref = Ref::new(arena, car);

            if let Some(Value::Symbol(sym)) = arena.get(car) {
                let sym_ref = Ref::new(arena, sym);
                let mut buf = [0u8; 32];
                if let Some(sym_str) = arena.list_to_str(&sym_ref, &mut buf) {
                    match sym_str {
                        "quote" => {
                            if let Some(quoted) = arena.list_nth(expr, 1) {
                                return Ok(EvalResult::Done(quoted));
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

            let func = eval(arena, &car_ref, env)?;
            let args = eval_args(arena, expr, env)?;

            apply(arena, &func, &args)
        }
        Some(Value::Nil) => Ok(EvalResult::Done(arena.nil())),
        _ => Err(arena.str_to_list("Invalid expression")),
    }
}

fn eval_if(arena: &Arena, expr: &Ref, env: &Ref) -> Result<EvalResult, Ref> {
    if arena.list_len(expr) != 4 {
        return Err(arena.str_to_list("if requires 3 args"));
    }

    let cond_expr = arena.list_nth(expr, 1).unwrap();
    let cond = eval(arena, &cond_expr, env)?;

    let is_true = match arena.get(*cond) {
        Some(Value::Bool(b)) => b,
        Some(Value::Nil) => false,
        _ => true,
    };

    let branch = if is_true {
        arena.list_nth(expr, 2).unwrap()
    } else {
        arena.list_nth(expr, 3).unwrap()
    };

    Ok(EvalResult::TailCall(branch, env.clone()))
}

fn eval_lambda(arena: &Arena, expr: &Ref, env: &Ref) -> Result<EvalResult, Ref> {
    if arena.list_len(expr) != 3 {
        return Err(arena.str_to_list("lambda requires 2 args"));
    }

    let params = arena.list_nth(expr, 1).unwrap();
    let body = arena.list_nth(expr, 2).unwrap();

    Ok(EvalResult::Done(arena.lambda(&params, &body, env)))
}

fn eval_define(arena: &Arena, expr: &Ref, env: &Ref) -> Result<EvalResult, Ref> {
    if arena.list_len(expr) != 3 {
        return Err(arena.str_to_list("define requires 2 args"));
    }

    let name = arena.list_nth(expr, 1).unwrap();
    let name_sym = if let Some(Value::Symbol(s)) = arena.get(*name) {
        Ref::new(arena, s)
    } else {
        return Err(arena.str_to_list("define needs symbol"));
    };

    let value_expr = arena.list_nth(expr, 2).unwrap();
    let value = eval(arena, &value_expr, env)?;

    env_set(arena, env, &name_sym, &value);
    Ok(EvalResult::Done(value))
}

fn eval_args(arena: &Arena, expr: &Ref, env: &Ref) -> Result<Ref, Ref> {
    let mut result = arena.nil();
    let mut current = if let Some(Value::Cons(_, cdr)) = arena.get(**expr) {
        Ref::new(arena, cdr)
    } else {
        return Ok(result);
    };

    loop {
        match arena.get(*current) {
            Some(Value::Cons(car, cdr)) => {
                let car_ref = Ref::new(arena, car);
                let val = eval(arena, &car_ref, env)?;
                result = arena.cons(&val, &result);
                current = Ref::new(arena, cdr);
            }
            Some(Value::Nil) => break,
            _ => {
                return Err(arena.str_to_list("Invalid args"));
            }
        }
    }

    Ok(arena.reverse_list(&result))
}

fn apply(arena: &Arena, func: &Ref, args: &Ref) -> Result<EvalResult, Ref> {
    match arena.get(**func) {
        Some(Value::Builtin(idx)) => {
            let result = call_builtin(arena, idx, args)?;
            Ok(EvalResult::Done(result))
        }
        Some(Value::Lambda(params, body, lambda_env)) => {
            let lambda_env_ref = Ref::new(arena, lambda_env);
            let call_env = env_with_parent(arena, &lambda_env_ref);

            let mut p = Ref::new(arena, params);
            let mut a = args.clone();

            loop {
                match (arena.get(*p), arena.get(*a)) {
                    (Some(Value::Cons(p_car, p_cdr)), Some(Value::Cons(a_car, a_cdr))) => {
                        if let Some(Value::Symbol(name)) = arena.get(p_car) {
                            let name_ref = Ref::new(arena, name);
                            let a_car_ref = Ref::new(arena, a_car);
                            env_set(arena, &call_env, &name_ref, &a_car_ref);
                        }
                        p = Ref::new(arena, p_cdr);
                        a = Ref::new(arena, a_cdr);
                    }
                    (Some(Value::Nil), Some(Value::Nil)) => break,
                    _ => {
                        return Err(arena.str_to_list("Arg count mismatch"));
                    }
                }
            }

            let body_ref = Ref::new(arena, body);
            Ok(EvalResult::TailCall(body_ref, call_env))
        }
        _ => Err(arena.str_to_list("Not callable")),
    }
}

// ============================================================================
// High-level API
// ============================================================================

pub fn eval_string(arena: &Arena, input: &str, env: &Ref) -> Result<Ref, Ref> {
    let tokens = tokenize(arena, input)?;
    let expr = parse(arena, &tokens)?;
    eval(arena, &expr, env)
}

// ============================================================================
// Example/Test
// ============================================================================

pub fn run_example() -> Result<(), ()> {
    let arena = Arena::new();
    let env = env_new(&arena);

    // Test: (+ 1 2 3)
    let result = eval_string(&arena, "(+ 1 2 3)", &env);
    if let Ok(val) = result {
        if let Some(Value::Number(n)) = arena.get(*val) {
            if n != 6 {
                return Err(());
            }
        }
    }

    // Test: (define x 42)
    let _ = eval_string(&arena, "(define x 42)", &env);

    // Test: (* x 2)
    let result = eval_string(&arena, "(* x 2)", &env);
    if let Ok(val) = result {
        if let Some(Value::Number(n)) = arena.get(*val) {
            if n != 84 {
                return Err(());
            }
        }
    }

    // Test: (define factorial ...)
    let factorial_def = r#"
        (define factorial
          (lambda (n)
            (if (= n 0)
                1
                (* n (factorial (- n 1))))))
    "#;
    let _ = eval_string(&arena, factorial_def, &env);

    // Test: (factorial 5)
    let result = eval_string(&arena, "(factorial 5)", &env);
    if let Ok(val) = result {
        if let Some(Value::Number(n)) = arena.get(*val) {
            if n != 120 {
                return Err(());
            }
        }
    }

    Ok(())
}
