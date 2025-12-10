#![no_std]

use core::cell::Cell;

// ============================================================================
// Error System - Does NOT allocate in arena
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorCode {
    UnboundSymbol,
    InvalidExpression,
    UnexpectedEOF,
    UnexpectedCloseParen,
    InvalidToken,
    SymbolTooLong,
    AtomTooLong,
    InvalidBoolean,
    ExtraTokens,
    UnmatchedOpenParen,
    TypeError,
    ArityError,
    DivisionByZero,
    Overflow,
    InvalidArgs,
    NotCallable,
    RequiresNumbers,
    RequiresCons,
    InvalidBegin,
    InvalidDefine,
    InvalidSet,
    InvalidLambda,
    InvalidIf,
    UnknownBuiltin,
    InvalidBindings,
    ArenaExhausted,
}

impl ErrorCode {
    pub fn message(&self) -> &'static str {
        match self {
            ErrorCode::UnboundSymbol => "Unbound symbol",
            ErrorCode::InvalidExpression => "Invalid expression",
            ErrorCode::UnexpectedEOF => "Unexpected EOF",
            ErrorCode::UnexpectedCloseParen => "Unexpected )",
            ErrorCode::InvalidToken => "Invalid token",
            ErrorCode::SymbolTooLong => "Symbol too long",
            ErrorCode::AtomTooLong => "Atom too long",
            ErrorCode::InvalidBoolean => "Invalid boolean",
            ErrorCode::ExtraTokens => "Extra tokens",
            ErrorCode::UnmatchedOpenParen => "Unmatched (",
            ErrorCode::TypeError => "Type error",
            ErrorCode::ArityError => "Arity error",
            ErrorCode::DivisionByZero => "Division by zero",
            ErrorCode::Overflow => "Overflow",
            ErrorCode::InvalidArgs => "Invalid args",
            ErrorCode::NotCallable => "Not callable",
            ErrorCode::RequiresNumbers => "Requires numbers",
            ErrorCode::RequiresCons => "Requires cons",
            ErrorCode::InvalidBegin => "Invalid begin",
            ErrorCode::InvalidDefine => "Invalid define",
            ErrorCode::InvalidSet => "Invalid set!",
            ErrorCode::InvalidLambda => "Invalid lambda",
            ErrorCode::InvalidIf => "Invalid if",
            ErrorCode::UnknownBuiltin => "Unknown builtin",
            ErrorCode::InvalidBindings => "Invalid bindings",
            ErrorCode::ArenaExhausted => "Arena exhausted",
        }
    }
}

// ============================================================================
// Arena Allocator with AUTOMATIC Reference Counting
// ============================================================================

pub const DEFAULT_ARENA_SIZE: usize = 10000;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct ArenaRef(pub u32);

impl ArenaRef {
    pub const NULL: ArenaRef = ArenaRef(u32::MAX);

    pub fn is_null(self) -> bool {
        self.0 == u32::MAX
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
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
pub struct Arena<const N: usize> {
    pub values: [Cell<Value>; N],
    pub refcounts: [Cell<u32>; N],
    pub next_free: Cell<usize>,
}

impl<const N: usize> Arena<N> {
    pub fn new() -> Self {
        Arena {
            values: [const { Cell::new(Value::Free) }; N],
            refcounts: [const { Cell::new(0) }; N],
            next_free: Cell::new(0),
        }
    }

    fn alloc(&self, value: Value) -> Result<ArenaRef, ErrorCode> {
        let start = self.next_free.get();

        for i in 0..N {
            let idx = (start + i) % N;
            if matches!(self.values[idx].get(), Value::Free) {
                self.values[idx].set(value);
                self.refcounts[idx].set(1);
                self.next_free.set((idx + 1) % N);
                return Ok(ArenaRef(idx as u32));
            }
        }

        Err(ErrorCode::ArenaExhausted)
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
            if rc < u32::MAX {
                self.refcounts[idx].set(rc + 1);
            }
        }
    }

    pub fn decref(&self, r: ArenaRef) {
        if r.is_null() {
            return;
        }

        // Use iterative approach with explicit stack to avoid stack overflow
        let mut stack = [ArenaRef::NULL; 128];
        let mut stack_len = 0;

        stack[0] = r;
        stack_len = 1;

        while stack_len > 0 {
            stack_len -= 1;
            let current = stack[stack_len];

            if current.is_null() {
                continue;
            }

            let idx = current.0 as usize;
            let value = self.values[idx].get();

            if matches!(value, Value::Free) {
                continue;
            }

            let rc = self.refcounts[idx].get();
            if rc > 1 {
                self.refcounts[idx].set(rc - 1);
            } else if rc == 1 {
                self.values[idx].set(Value::Free);
                self.refcounts[idx].set(0);

                // Push children onto stack instead of recursing
                match value {
                    Value::Cons(car, cdr) => {
                        if stack_len + 2 <= stack.len() {
                            stack[stack_len] = car;
                            stack[stack_len + 1] = cdr;
                            stack_len += 2;
                        } else {
                            // Stack full - fall back to recursion (rare case)
                            self.decref(car);
                            self.decref(cdr);
                        }
                    }
                    Value::Symbol(s) => {
                        if stack_len < stack.len() {
                            stack[stack_len] = s;
                            stack_len += 1;
                        } else {
                            self.decref(s);
                        }
                    }
                    Value::Lambda(params, body, env) => {
                        if stack_len + 3 <= stack.len() {
                            stack[stack_len] = params;
                            stack[stack_len + 1] = body;
                            stack[stack_len + 2] = env;
                            stack_len += 3;
                        } else {
                            self.decref(params);
                            self.decref(body);
                            self.decref(env);
                        }
                    }
                    _ => {}
                }
            }
        }
    }

    // High-level API that returns RAII-wrapped references
    pub fn nil(&self) -> Result<Ref<N>, ErrorCode> {
        Ok(Ref::from_alloc(self, self.alloc(Value::Nil)?))
    }

    pub fn number(&self, n: i64) -> Result<Ref<N>, ErrorCode> {
        Ok(Ref::from_alloc(self, self.alloc(Value::Number(n))?))
    }

    pub fn bool_val(&self, b: bool) -> Result<Ref<N>, ErrorCode> {
        Ok(Ref::from_alloc(self, self.alloc(Value::Bool(b))?))
    }

    pub fn char_val(&self, c: char) -> Result<Ref<N>, ErrorCode> {
        Ok(Ref::from_alloc(self, self.alloc(Value::Char(c))?))
    }

    pub fn cons(&self, car: &Ref<N>, cdr: &Ref<N>) -> Result<Ref<N>, ErrorCode> {
        self.incref(car.inner);
        self.incref(cdr.inner);
        Ok(Ref::from_alloc(
            self,
            self.alloc(Value::Cons(car.inner, cdr.inner))?,
        ))
    }

    pub fn symbol(&self, s: &Ref<N>) -> Result<Ref<N>, ErrorCode> {
        self.incref(s.inner);
        Ok(Ref::from_alloc(self, self.alloc(Value::Symbol(s.inner))?))
    }

    pub fn lambda(
        &self,
        params: &Ref<N>,
        body: &Ref<N>,
        env: &Ref<N>,
    ) -> Result<Ref<N>, ErrorCode> {
        self.incref(params.inner);
        self.incref(body.inner);
        self.incref(env.inner);
        Ok(Ref::from_alloc(
            self,
            self.alloc(Value::Lambda(params.inner, body.inner, env.inner))?,
        ))
    }

    pub fn builtin(&self, idx: u8) -> Result<Ref<N>, ErrorCode> {
        Ok(Ref::from_alloc(self, self.alloc(Value::Builtin(idx))?))
    }

    pub fn set_cons(&self, cons_ref: &Ref<N>, new_car: &Ref<N>, new_cdr: &Ref<N>) {
        if let Some(Value::Cons(old_car, old_cdr)) = self.get(cons_ref.inner) {
            self.incref(new_car.inner);
            self.incref(new_cdr.inner);
            self.values[cons_ref.inner.0 as usize].set(Value::Cons(new_car.inner, new_cdr.inner));
            self.decref(old_car);
            self.decref(old_cdr);
        }
    }
}

// ============================================================================
// RAII Reference Wrapper - Automatic Reference Counting!
// ============================================================================
#[derive(Debug)]
pub struct Ref<'arena, const N: usize> {
    arena: &'arena Arena<N>,
    pub inner: ArenaRef,
}

impl<'arena, const N: usize> Ref<'arena, N> {
    pub fn new(arena: &'arena Arena<N>, r: ArenaRef) -> Self {
        arena.incref(r);
        Ref { arena, inner: r }
    }

    fn from_alloc(arena: &'arena Arena<N>, r: ArenaRef) -> Self {
        Ref { arena, inner: r }
    }

    pub fn raw(&self) -> ArenaRef {
        self.inner
    }

    pub fn get(&self) -> Option<Value> {
        self.arena.get(self.inner)
    }

    pub fn is_null(&self) -> bool {
        self.inner.is_null()
    }

    pub fn arena(&self) -> &'arena Arena<N> {
        self.arena
    }
}

impl<'arena, const N: usize> Clone for Ref<'arena, N> {
    fn clone(&self) -> Self {
        self.arena.incref(self.inner);
        Ref {
            arena: self.arena,
            inner: self.inner,
        }
    }
}

impl<'arena, const N: usize> Drop for Ref<'arena, N> {
    fn drop(&mut self) {
        self.arena.decref(self.inner);
    }
}

// ============================================================================
// Helper Functions with Automatic Reference Counting
// ============================================================================

impl<const N: usize> Arena<N> {
    pub fn str_to_list(&self, s: &str) -> Result<Ref<N>, ErrorCode> {
        let mut result = self.nil()?;
        for ch in s.chars().rev() {
            let char_ref = self.char_val(ch)?;
            result = self.cons(&char_ref, &result)?;
        }
        Ok(result)
    }

    pub fn list_to_str<'a>(&self, list: &Ref<N>, buf: &'a mut [u8]) -> Option<&'a str> {
        let mut idx = 0;
        let mut current = list.clone();

        loop {
            match self.get(current.inner)? {
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

    pub fn list_len(&self, list: &Ref<N>) -> usize {
        let mut count = 0;
        let mut current = list.clone();
        loop {
            match self.get(current.inner) {
                Some(Value::Cons(_, cdr)) => {
                    count += 1;
                    current = Ref::new(self, cdr);
                }
                _ => break,
            }
        }
        count
    }

    pub fn list_nth(&self, list: &Ref<N>, n: usize) -> Option<Ref<N>> {
        let mut idx = 0;
        let mut current = list.clone();
        loop {
            match self.get(current.inner) {
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

    pub fn reverse_list(&self, list: &Ref<N>) -> Result<Ref<N>, ErrorCode> {
        let mut result = self.nil()?;
        let mut current = list.clone();
        loop {
            match self.get(current.inner) {
                Some(Value::Cons(car, cdr)) => {
                    let car_ref = Ref::new(self, car);
                    result = self.cons(&car_ref, &result)?;
                    current = Ref::new(self, cdr);
                }
                _ => break,
            }
        }
        Ok(result)
    }

    pub fn str_eq(&self, s1: &Ref<N>, s2: &Ref<N>) -> bool {
        let mut c1 = s1.clone();
        let mut c2 = s2.clone();
        loop {
            match (self.get(c1.inner), self.get(c2.inner)) {
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

pub fn env_new<'arena, const N: usize>(
    arena: &'arena Arena<N>,
) -> Result<Ref<'arena, N>, ErrorCode> {
    let nil1 = arena.nil()?;
    let nil2 = arena.nil()?;
    let env = arena.cons(&nil1, &nil2)?;
    register_builtins(arena, &env)?;
    Ok(env)
}

pub fn env_with_parent<'arena, const N: usize>(
    arena: &'arena Arena<N>,
    parent: &Ref<'arena, N>,
) -> Result<Ref<'arena, N>, ErrorCode> {
    let nil = arena.nil()?;
    arena.cons(&nil, parent)
}

pub fn env_set<'arena, const N: usize>(
    arena: &'arena Arena<N>,
    env: &Ref<'arena, N>,
    name: &Ref<'arena, N>,
    value: &Ref<'arena, N>,
) -> Result<(), ErrorCode> {
    if let Some(Value::Cons(bindings, parent)) = arena.get(env.inner) {
        let bindings_ref = Ref::new(arena, bindings);
        let parent_ref = Ref::new(arena, parent);

        // Check if variable already exists in current environment
        // If it does, remove the old binding to prevent memory leak
        let filtered_bindings = remove_binding(arena, &bindings_ref, name)?;

        let sym = arena.symbol(name)?;
        let new_binding = arena.cons(&sym, value)?;
        let new_bindings = arena.cons(&new_binding, &filtered_bindings)?;

        arena.set_cons(env, &new_bindings, &parent_ref);
    }
    Ok(())
}

pub fn env_update<'arena, const N: usize>(
    arena: &'arena Arena<N>,
    env: &Ref<'arena, N>,
    name: &Ref<'arena, N>,
    value: &Ref<'arena, N>,
) -> Result<(), ErrorCode> {
    let mut current_env = env.clone();
    loop {
        match arena.get(current_env.inner) {
            Some(Value::Cons(bindings, parent)) => {
                let mut bindings_list = Ref::new(arena, bindings);
                loop {
                    match arena.get(bindings_list.inner) {
                        Some(Value::Cons(binding, rest)) => {
                            if let Some(Value::Cons(key, _old_value)) = arena.get(binding) {
                                if let Some(Value::Symbol(s)) = arena.get(key) {
                                    let key_sym = Ref::new(arena, s);
                                    if arena.str_eq(&key_sym, name) {
                                        // Found it! Update the binding in place
                                        let key_ref = Ref::new(arena, key);
                                        let binding_ref = Ref::new(arena, binding);
                                        arena.set_cons(&binding_ref, &key_ref, value);
                                        return Ok(());
                                    }
                                }
                            }
                            bindings_list = Ref::new(arena, rest);
                        }
                        _ => break,
                    }
                }

                match arena.get(parent) {
                    Some(Value::Nil) => return Err(ErrorCode::UnboundSymbol),
                    _ => current_env = Ref::new(arena, parent),
                }
            }
            _ => return Err(ErrorCode::UnboundSymbol),
        }
    }
}

// Helper function to remove a binding from the binding list
fn remove_binding<'arena, const N: usize>(
    arena: &'arena Arena<N>,
    bindings: &Ref<'arena, N>,
    name: &Ref<'arena, N>,
) -> Result<Ref<'arena, N>, ErrorCode> {
    match arena.get(bindings.inner) {
        Some(Value::Nil) => arena.nil(),
        Some(Value::Cons(binding, rest)) => {
            let rest_ref = Ref::new(arena, rest);

            // Check if this binding matches the name we're looking for
            if let Some(Value::Cons(key, _)) = arena.get(binding) {
                if let Some(Value::Symbol(s)) = arena.get(key) {
                    let key_sym = Ref::new(arena, s);
                    if arena.str_eq(&key_sym, name) {
                        // Found it! Skip this binding and continue with the rest
                        return remove_binding(arena, &rest_ref, name);
                    }
                }
            }

            // Not a match, keep this binding
            let binding_ref = Ref::new(arena, binding);
            let filtered_rest = remove_binding(arena, &rest_ref, name)?;
            arena.cons(&binding_ref, &filtered_rest)
        }
        _ => arena.nil(),
    }
}

pub fn env_get<'arena, const N: usize>(
    arena: &'arena Arena<N>,
    env: &Ref<'arena, N>,
    name: &Ref<'arena, N>,
) -> Option<Ref<'arena, N>> {
    let mut current_env = env.clone();
    loop {
        match arena.get(current_env.inner) {
            Some(Value::Cons(bindings, parent)) => {
                let mut bindings_list = Ref::new(arena, bindings);
                loop {
                    match arena.get(bindings_list.inner) {
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

fn call_builtin<'arena, const N: usize>(
    arena: &'arena Arena<N>,
    idx: u8,
    args: &Ref<'arena, N>,
) -> Result<Ref<'arena, N>, ErrorCode> {
    match idx {
        BUILTIN_ADD => builtin_add(arena, args),
        BUILTIN_SUB => builtin_sub(arena, args),
        BUILTIN_MUL => builtin_mul(arena, args),
        BUILTIN_DIV => builtin_div(arena, args),
        BUILTIN_EQ => builtin_eq(arena, args),
        BUILTIN_LT => builtin_lt(arena, args),
        BUILTIN_GT => builtin_gt(arena, args),
        BUILTIN_CAR => builtin_car(arena, args),
        BUILTIN_CDR => builtin_cdr(arena, args),
        BUILTIN_CONS => builtin_cons_fn(arena, args),
        BUILTIN_LIST => builtin_list(arena, args),
        BUILTIN_NULL => builtin_null(arena, args),
        BUILTIN_LENGTH => builtin_length(arena, args),
        BUILTIN_APPEND => builtin_append(arena, args),
        BUILTIN_REVERSE => builtin_reverse(arena, args),
        _ => Err(ErrorCode::UnknownBuiltin),
    }
}

fn builtin_add<'arena, const N: usize>(
    arena: &'arena Arena<N>,
    args: &Ref<'arena, N>,
) -> Result<Ref<'arena, N>, ErrorCode> {
    let mut result = 0i64;
    let mut current = args.clone();

    loop {
        match arena.get(current.inner) {
            Some(Value::Cons(car, cdr)) => {
                if let Some(Value::Number(n)) = arena.get(car) {
                    result = result.checked_add(n).ok_or(ErrorCode::Overflow)?;
                    current = Ref::new(arena, cdr);
                } else {
                    return Err(ErrorCode::RequiresNumbers);
                }
            }
            Some(Value::Nil) => break,
            _ => return Err(ErrorCode::InvalidArgs),
        }
    }

    arena.number(result)
}

fn builtin_sub<'arena, const N: usize>(
    arena: &'arena Arena<N>,
    args: &Ref<'arena, N>,
) -> Result<Ref<'arena, N>, ErrorCode> {
    let len = arena.list_len(args);
    if len == 0 {
        return Err(ErrorCode::ArityError);
    }

    let first = arena.list_nth(args, 0).unwrap();
    let first_num = if let Some(Value::Number(n)) = arena.get(first.inner) {
        n
    } else {
        return Err(ErrorCode::RequiresNumbers);
    };

    if len == 1 {
        return arena.number(-first_num);
    }

    let mut result = first_num;
    let mut current = if let Some(Value::Cons(_, cdr)) = arena.get(args.inner) {
        Ref::new(arena, cdr)
    } else {
        return Err(ErrorCode::InvalidArgs);
    };

    loop {
        match arena.get(current.inner) {
            Some(Value::Cons(car, cdr)) => {
                if let Some(Value::Number(n)) = arena.get(car) {
                    result = result.checked_sub(n).ok_or(ErrorCode::Overflow)?;
                    current = Ref::new(arena, cdr);
                } else {
                    return Err(ErrorCode::RequiresNumbers);
                }
            }
            Some(Value::Nil) => break,
            _ => return Err(ErrorCode::InvalidArgs),
        }
    }

    arena.number(result)
}

fn builtin_mul<'arena, const N: usize>(
    arena: &'arena Arena<N>,
    args: &Ref<'arena, N>,
) -> Result<Ref<'arena, N>, ErrorCode> {
    let mut result = 1i64;
    let mut current = args.clone();

    loop {
        match arena.get(current.inner) {
            Some(Value::Cons(car, cdr)) => {
                if let Some(Value::Number(n)) = arena.get(car) {
                    result = result.checked_mul(n).ok_or(ErrorCode::Overflow)?;
                    current = Ref::new(arena, cdr);
                } else {
                    return Err(ErrorCode::RequiresNumbers);
                }
            }
            Some(Value::Nil) => break,
            _ => return Err(ErrorCode::InvalidArgs),
        }
    }

    arena.number(result)
}

fn builtin_div<'arena, const N: usize>(
    arena: &'arena Arena<N>,
    args: &Ref<'arena, N>,
) -> Result<Ref<'arena, N>, ErrorCode> {
    let len = arena.list_len(args);
    if len < 2 {
        return Err(ErrorCode::ArityError);
    }

    let first = arena.list_nth(args, 0).unwrap();
    let mut result = if let Some(Value::Number(n)) = arena.get(first.inner) {
        n
    } else {
        return Err(ErrorCode::RequiresNumbers);
    };

    let mut current = if let Some(Value::Cons(_, cdr)) = arena.get(args.inner) {
        Ref::new(arena, cdr)
    } else {
        return Err(ErrorCode::InvalidArgs);
    };

    loop {
        match arena.get(current.inner) {
            Some(Value::Cons(car, cdr)) => {
                if let Some(Value::Number(n)) = arena.get(car) {
                    if n == 0 {
                        return Err(ErrorCode::DivisionByZero);
                    }
                    result = result.checked_div(n).ok_or(ErrorCode::Overflow)?;
                    current = Ref::new(arena, cdr);
                } else {
                    return Err(ErrorCode::RequiresNumbers);
                }
            }
            Some(Value::Nil) => break,
            _ => return Err(ErrorCode::InvalidArgs),
        }
    }

    arena.number(result)
}

fn builtin_eq<'arena, const N: usize>(
    arena: &'arena Arena<N>,
    args: &Ref<'arena, N>,
) -> Result<Ref<'arena, N>, ErrorCode> {
    if arena.list_len(args) != 2 {
        return Err(ErrorCode::ArityError);
    }
    let a = arena.list_nth(args, 0).unwrap();
    let b = arena.list_nth(args, 1).unwrap();

    let eq = match (arena.get(a.inner), arena.get(b.inner)) {
        (Some(Value::Number(x)), Some(Value::Number(y))) => x == y,
        _ => return Err(ErrorCode::RequiresNumbers),
    };

    arena.bool_val(eq)
}

fn builtin_lt<'arena, const N: usize>(
    arena: &'arena Arena<N>,
    args: &Ref<'arena, N>,
) -> Result<Ref<'arena, N>, ErrorCode> {
    if arena.list_len(args) != 2 {
        return Err(ErrorCode::ArityError);
    }
    let a = arena.list_nth(args, 0).unwrap();
    let b = arena.list_nth(args, 1).unwrap();

    let lt = match (arena.get(a.inner), arena.get(b.inner)) {
        (Some(Value::Number(x)), Some(Value::Number(y))) => x < y,
        _ => return Err(ErrorCode::RequiresNumbers),
    };

    arena.bool_val(lt)
}

fn builtin_gt<'arena, const N: usize>(
    arena: &'arena Arena<N>,
    args: &Ref<'arena, N>,
) -> Result<Ref<'arena, N>, ErrorCode> {
    if arena.list_len(args) != 2 {
        return Err(ErrorCode::ArityError);
    }
    let a = arena.list_nth(args, 0).unwrap();
    let b = arena.list_nth(args, 1).unwrap();

    let gt = match (arena.get(a.inner), arena.get(b.inner)) {
        (Some(Value::Number(x)), Some(Value::Number(y))) => x > y,
        _ => return Err(ErrorCode::RequiresNumbers),
    };

    arena.bool_val(gt)
}

fn builtin_car<'arena, const N: usize>(
    arena: &'arena Arena<N>,
    args: &Ref<'arena, N>,
) -> Result<Ref<'arena, N>, ErrorCode> {
    if arena.list_len(args) != 1 {
        return Err(ErrorCode::ArityError);
    }
    let list = arena.list_nth(args, 0).unwrap();
    if let Some(Value::Cons(car, _)) = arena.get(list.inner) {
        Ok(Ref::new(arena, car))
    } else {
        Err(ErrorCode::RequiresCons)
    }
}

fn builtin_cdr<'arena, const N: usize>(
    arena: &'arena Arena<N>,
    args: &Ref<'arena, N>,
) -> Result<Ref<'arena, N>, ErrorCode> {
    if arena.list_len(args) != 1 {
        return Err(ErrorCode::ArityError);
    }
    let list = arena.list_nth(args, 0).unwrap();
    if let Some(Value::Cons(_, cdr)) = arena.get(list.inner) {
        Ok(Ref::new(arena, cdr))
    } else {
        Err(ErrorCode::RequiresCons)
    }
}

fn builtin_cons_fn<'arena, const N: usize>(
    arena: &'arena Arena<N>,
    args: &Ref<'arena, N>,
) -> Result<Ref<'arena, N>, ErrorCode> {
    if arena.list_len(args) != 2 {
        return Err(ErrorCode::ArityError);
    }
    let car = arena.list_nth(args, 0).unwrap();
    let cdr = arena.list_nth(args, 1).unwrap();
    arena.cons(&car, &cdr)
}

fn builtin_list<'arena, const N: usize>(
    _arena: &'arena Arena<N>,
    args: &Ref<'arena, N>,
) -> Result<Ref<'arena, N>, ErrorCode> {
    Ok(args.clone())
}

fn builtin_null<'arena, const N: usize>(
    arena: &'arena Arena<N>,
    args: &Ref<'arena, N>,
) -> Result<Ref<'arena, N>, ErrorCode> {
    if arena.list_len(args) != 1 {
        return Err(ErrorCode::ArityError);
    }
    let val = arena.list_nth(args, 0).unwrap();
    let is_nil = matches!(arena.get(val.inner), Some(Value::Nil));
    arena.bool_val(is_nil)
}

fn builtin_length<'arena, const N: usize>(
    arena: &'arena Arena<N>,
    args: &Ref<'arena, N>,
) -> Result<Ref<'arena, N>, ErrorCode> {
    if arena.list_len(args) != 1 {
        return Err(ErrorCode::ArityError);
    }
    let list = arena.list_nth(args, 0).unwrap();
    let len = arena.list_len(&list);
    arena.number(len as i64)
}

fn builtin_append<'arena, const N: usize>(
    arena: &'arena Arena<N>,
    args: &Ref<'arena, N>,
) -> Result<Ref<'arena, N>, ErrorCode> {
    let mut result = arena.nil()?;
    let mut current = args.clone();

    loop {
        match arena.get(current.inner) {
            Some(Value::Cons(list, rest)) => {
                let mut list_cur = Ref::new(arena, list);
                loop {
                    match arena.get(list_cur.inner) {
                        Some(Value::Cons(item, item_rest)) => {
                            let item_ref = Ref::new(arena, item);
                            result = arena.cons(&item_ref, &result)?;
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
                return Err(ErrorCode::InvalidArgs);
            }
        }
    }

    arena.reverse_list(&result)
}

fn builtin_reverse<'arena, const N: usize>(
    arena: &'arena Arena<N>,
    args: &Ref<'arena, N>,
) -> Result<Ref<'arena, N>, ErrorCode> {
    if arena.list_len(args) != 1 {
        return Err(ErrorCode::ArityError);
    }
    let list = arena.list_nth(args, 0).unwrap();
    arena.reverse_list(&list)
}

fn register_builtins<'arena, const N: usize>(
    arena: &'arena Arena<N>,
    env: &Ref<'arena, N>,
) -> Result<(), ErrorCode> {
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
        let name_ref = arena.str_to_list(name)?;
        let builtin_ref = arena.builtin(idx)?;
        env_set(arena, env, &name_ref, &builtin_ref)?;
    }

    Ok(())
}

// ============================================================================
// Tokenizer
// ============================================================================

fn is_delimiter(ch: char) -> bool {
    ch.is_whitespace() || ch == '(' || ch == ')' || ch == '\'' || ch == ';'
}

pub fn tokenize<'arena, const N: usize>(
    arena: &'arena Arena<N>,
    input: &str,
) -> Result<Ref<'arena, N>, ErrorCode> {
    let mut result = arena.nil()?;
    let mut chars = input.chars().peekable();

    while let Some(&ch) = chars.peek() {
        match ch {
            ' ' | '\t' | '\n' | '\r' => {
                chars.next();
            }
            '(' => {
                let tok = arena.str_to_list("(")?;
                let sym = arena.symbol(&tok)?;
                result = arena.cons(&sym, &result)?;
                chars.next();
            }
            ')' => {
                let tok = arena.str_to_list(")")?;
                let sym = arena.symbol(&tok)?;
                result = arena.cons(&sym, &result)?;
                chars.next();
            }
            '\'' => {
                let tok = arena.str_to_list("'")?;
                let sym = arena.symbol(&tok)?;
                result = arena.cons(&sym, &result)?;
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
                        let val = arena.bool_val(true)?;
                        result = arena.cons(&val, &result)?;
                        chars.next();
                    }
                    Some(&'f') => {
                        let val = arena.bool_val(false)?;
                        result = arena.cons(&val, &result)?;
                        chars.next();
                    }
                    _ => return Err(ErrorCode::InvalidBoolean),
                }
            }
            _ => {
                let mut atom = arena.nil()?;
                while let Some(&c) = chars.peek() {
                    if is_delimiter(c) {
                        break;
                    }
                    let ch_val = arena.char_val(c)?;
                    atom = arena.cons(&ch_val, &atom)?;
                    chars.next();
                }

                atom = arena.reverse_list(&atom)?;

                let mut buf = [0u8; 64];
                if let Some(s) = arena.list_to_str(&atom, &mut buf) {
                    if let Ok(num) = parse_i64(s) {
                        let num_val = arena.number(num)?;
                        result = arena.cons(&num_val, &result)?;
                    } else {
                        let sym = arena.symbol(&atom)?;
                        result = arena.cons(&sym, &result)?;
                    }
                } else {
                    return Err(ErrorCode::AtomTooLong);
                }
            }
        }
    }

    arena.reverse_list(&result)
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

pub fn parse<'arena, const N: usize>(
    arena: &'arena Arena<N>,
    tokens: &Ref<'arena, N>,
) -> Result<Ref<'arena, N>, ErrorCode> {
    let (expr, remaining) = parse_one(arena, tokens)?;
    if !matches!(arena.get(remaining.inner), Some(Value::Nil)) {
        return Err(ErrorCode::ExtraTokens);
    }
    Ok(expr)
}

fn parse_one<'arena, const N: usize>(
    arena: &'arena Arena<N>,
    tokens: &Ref<'arena, N>,
) -> Result<(Ref<'arena, N>, Ref<'arena, N>), ErrorCode> {
    match arena.get(tokens.inner) {
        Some(Value::Nil) => Err(ErrorCode::UnexpectedEOF),
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
                            let quote_sym = arena.str_to_list("quote")?;
                            let quoted_sym = arena.symbol(&quote_sym)?;
                            let nil = arena.nil()?;
                            let list = arena.cons(&expr, &nil)?;
                            let quoted = arena.cons(&quoted_sym, &list)?;
                            Ok((quoted, consumed))
                        } else if s_str == "(" {
                            parse_list(arena, &rest_ref)
                        } else if s_str == ")" {
                            Err(ErrorCode::UnexpectedCloseParen)
                        } else {
                            Ok((first_ref, rest_ref))
                        }
                    } else {
                        Err(ErrorCode::SymbolTooLong)
                    }
                }
                _ => Err(ErrorCode::InvalidToken),
            }
        }
        _ => Err(ErrorCode::InvalidToken),
    }
}

fn parse_list<'arena, const N: usize>(
    arena: &'arena Arena<N>,
    tokens: &Ref<'arena, N>,
) -> Result<(Ref<'arena, N>, Ref<'arena, N>), ErrorCode> {
    let mut items = arena.nil()?;
    let mut current = tokens.clone();

    loop {
        match arena.get(current.inner) {
            Some(Value::Nil) => return Err(ErrorCode::UnmatchedOpenParen),
            Some(Value::Cons(tok, rest)) => {
                if let Some(Value::Symbol(s)) = arena.get(tok) {
                    let mut buf = [0u8; 32];
                    if let Some(s_str) = arena.list_to_str(&Ref::new(arena, s), &mut buf) {
                        if s_str == ")" {
                            let result = arena.reverse_list(&items)?;
                            return Ok((result, Ref::new(arena, rest)));
                        }
                    }
                }

                let (expr, consumed) = parse_one(arena, &current)?;
                items = arena.cons(&expr, &items)?;
                current = consumed;
            }
            _ => return Err(ErrorCode::InvalidToken),
        }
    }
}

// ============================================================================
// Evaluator with PROPER Tail Call Optimization
// ============================================================================

pub fn eval<'arena, const N: usize>(
    arena: &'arena Arena<N>,
    expr: &Ref<'arena, N>,
    env: &Ref<'arena, N>,
) -> Result<Ref<'arena, N>, ErrorCode> {
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

pub enum EvalResult<'arena, const N: usize> {
    Done(Ref<'arena, N>),
    TailCall(Ref<'arena, N>, Ref<'arena, N>),
}

fn eval_step<'arena, const N: usize>(
    arena: &'arena Arena<N>,
    expr: &Ref<'arena, N>,
    env: &Ref<'arena, N>,
) -> Result<EvalResult<'arena, N>, ErrorCode> {
    match arena.get(expr.inner) {
        Some(Value::Number(_))
        | Some(Value::Bool(_))
        | Some(Value::Builtin(_))
        | Some(Value::Lambda(..)) => Ok(EvalResult::Done(expr.clone())),
        Some(Value::Symbol(s)) => {
            let s_ref = Ref::new(arena, s);
            let nil_str = arena.str_to_list("nil")?;
            let is_nil = arena.str_eq(&s_ref, &nil_str);

            if is_nil {
                return Ok(EvalResult::Done(arena.nil()?));
            }

            if let Some(val) = env_get(arena, env, &s_ref) {
                Ok(EvalResult::Done(val))
            } else {
                Err(ErrorCode::UnboundSymbol)
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
                        "set!" => {
                            return eval_set(arena, expr, env);
                        }
                        "begin" => {
                            return eval_begin(arena, expr, env);
                        }
                        _ => {}
                    }
                }
            }

            let func = eval(arena, &car_ref, env)?;
            let args = eval_args(arena, expr, env)?;

            apply(arena, &func, &args)
        }
        Some(Value::Nil) => Ok(EvalResult::Done(arena.nil()?)),
        _ => Err(ErrorCode::InvalidExpression),
    }
}

fn eval_if<'arena, const N: usize>(
    arena: &'arena Arena<N>,
    expr: &Ref<'arena, N>,
    env: &Ref<'arena, N>,
) -> Result<EvalResult<'arena, N>, ErrorCode> {
    if arena.list_len(expr) != 4 {
        return Err(ErrorCode::InvalidIf);
    }

    let cond_expr = arena.list_nth(expr, 1).unwrap();
    let cond = eval(arena, &cond_expr, env)?;

    let is_true = match arena.get(cond.inner) {
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

fn eval_lambda<'arena, const N: usize>(
    arena: &'arena Arena<N>,
    expr: &Ref<'arena, N>,
    env: &Ref<'arena, N>,
) -> Result<EvalResult<'arena, N>, ErrorCode> {
    let len = arena.list_len(expr);
    if len < 3 {
        return Err(ErrorCode::InvalidLambda);
    }

    let params = arena.list_nth(expr, 1).unwrap();

    // If there's only one body expression, use it directly
    // Otherwise, wrap multiple body expressions in a begin
    let body = if len == 3 {
        arena.list_nth(expr, 2).unwrap()
    } else {
        // Multiple body expressions - wrap in begin
        let begin_sym = arena.str_to_list("begin")?;
        let begin_symbol = arena.symbol(&begin_sym)?;

        // Collect all body expressions (skip 'lambda' and params)
        let mut body_exprs = arena.nil()?;
        let mut current = expr.clone();

        // Skip lambda keyword and params
        if let Some(Value::Cons(_, rest1)) = arena.get(current.inner) {
            if let Some(Value::Cons(_, rest2)) = arena.get(rest1) {
                current = Ref::new(arena, rest2);
            }
        }

        // Collect body expressions
        loop {
            match arena.get(current.inner) {
                Some(Value::Cons(car, cdr)) => {
                    let car_ref = Ref::new(arena, car);
                    body_exprs = arena.cons(&car_ref, &body_exprs)?;
                    current = Ref::new(arena, cdr);
                }
                _ => break,
            }
        }

        body_exprs = arena.reverse_list(&body_exprs)?;
        arena.cons(&begin_symbol, &body_exprs)?
    };

    Ok(EvalResult::Done(arena.lambda(&params, &body, env)?))
}

fn eval_begin<'arena, const N: usize>(
    arena: &'arena Arena<N>,
    expr: &Ref<'arena, N>,
    env: &Ref<'arena, N>,
) -> Result<EvalResult<'arena, N>, ErrorCode> {
    let len = arena.list_len(expr);
    if len < 2 {
        return Err(ErrorCode::InvalidBegin);
    }

    // Skip the 'begin' symbol
    let mut current = if let Some(Value::Cons(_, cdr)) = arena.get(expr.inner) {
        Ref::new(arena, cdr)
    } else {
        return Err(ErrorCode::InvalidBegin);
    };

    let mut result = arena.nil()?;

    loop {
        match arena.get(current.inner) {
            Some(Value::Cons(car, cdr)) => {
                let car_ref = Ref::new(arena, car);

                // Check if this is the last expression
                let is_last = matches!(arena.get(cdr), Some(Value::Nil));

                if is_last {
                    // Tail call the last expression
                    return Ok(EvalResult::TailCall(car_ref, env.clone()));
                } else {
                    // Evaluate non-last expressions for side effects
                    result = eval(arena, &car_ref, env)?;
                    current = Ref::new(arena, cdr);
                }
            }
            Some(Value::Nil) => {
                // Empty begin or already handled
                return Ok(EvalResult::Done(result));
            }
            _ => return Err(ErrorCode::InvalidBegin),
        }
    }
}

fn eval_define<'arena, const N: usize>(
    arena: &'arena Arena<N>,
    expr: &Ref<'arena, N>,
    env: &Ref<'arena, N>,
) -> Result<EvalResult<'arena, N>, ErrorCode> {
    if arena.list_len(expr) != 3 {
        return Err(ErrorCode::InvalidDefine);
    }

    let name = arena.list_nth(expr, 1).unwrap();
    let name_sym = if let Some(Value::Symbol(s)) = arena.get(name.inner) {
        Ref::new(arena, s)
    } else {
        return Err(ErrorCode::InvalidDefine);
    };

    let value_expr = arena.list_nth(expr, 2).unwrap();
    let value = eval(arena, &value_expr, env)?;

    env_set(arena, env, &name_sym, &value)?;
    Ok(EvalResult::Done(value))
}

fn eval_set<'arena, const N: usize>(
    arena: &'arena Arena<N>,
    expr: &Ref<'arena, N>,
    env: &Ref<'arena, N>,
) -> Result<EvalResult<'arena, N>, ErrorCode> {
    if arena.list_len(expr) != 3 {
        return Err(ErrorCode::InvalidSet);
    }

    let name = arena.list_nth(expr, 1).unwrap();
    let name_sym = if let Some(Value::Symbol(s)) = arena.get(name.inner) {
        Ref::new(arena, s)
    } else {
        return Err(ErrorCode::InvalidSet);
    };

    let value_expr = arena.list_nth(expr, 2).unwrap();
    let value = eval(arena, &value_expr, env)?;

    env_update(arena, env, &name_sym, &value)?;

    Ok(EvalResult::Done(value))
}

fn eval_args<'arena, const N: usize>(
    arena: &'arena Arena<N>,
    expr: &Ref<'arena, N>,
    env: &Ref<'arena, N>,
) -> Result<Ref<'arena, N>, ErrorCode> {
    let mut result = arena.nil()?;
    let mut current = if let Some(Value::Cons(_, cdr)) = arena.get(expr.inner) {
        Ref::new(arena, cdr)
    } else {
        return Ok(result);
    };

    loop {
        match arena.get(current.inner) {
            Some(Value::Cons(car, cdr)) => {
                let car_ref = Ref::new(arena, car);
                let val = eval(arena, &car_ref, env)?;
                result = arena.cons(&val, &result)?;
                current = Ref::new(arena, cdr);
            }
            Some(Value::Nil) => break,
            _ => {
                return Err(ErrorCode::InvalidArgs);
            }
        }
    }

    arena.reverse_list(&result)
}

fn apply<'arena, const N: usize>(
    arena: &'arena Arena<N>,
    func: &Ref<'arena, N>,
    args: &Ref<'arena, N>,
) -> Result<EvalResult<'arena, N>, ErrorCode> {
    match arena.get(func.inner) {
        Some(Value::Builtin(idx)) => {
            let result = call_builtin(arena, idx, args)?;
            Ok(EvalResult::Done(result))
        }
        Some(Value::Lambda(params, body, lambda_env)) => {
            let lambda_env_ref = Ref::new(arena, lambda_env);
            let call_env = env_with_parent(arena, &lambda_env_ref)?;

            let mut p = Ref::new(arena, params);
            let mut a = args.clone();

            loop {
                match (arena.get(p.inner), arena.get(a.inner)) {
                    (Some(Value::Cons(p_car, p_cdr)), Some(Value::Cons(a_car, a_cdr))) => {
                        if let Some(Value::Symbol(name)) = arena.get(p_car) {
                            let name_ref = Ref::new(arena, name);
                            let a_car_ref = Ref::new(arena, a_car);
                            env_set(arena, &call_env, &name_ref, &a_car_ref)?;
                        }
                        p = Ref::new(arena, p_cdr);
                        a = Ref::new(arena, a_cdr);
                    }
                    (Some(Value::Nil), Some(Value::Nil)) => break,
                    _ => {
                        return Err(ErrorCode::ArityError);
                    }
                }
            }

            let body_ref = Ref::new(arena, body);
            Ok(EvalResult::TailCall(body_ref, call_env))
        }
        _ => Err(ErrorCode::NotCallable),
    }
}

// ============================================================================
// High-level API
// ============================================================================

pub fn eval_string<'arena, const N: usize>(
    arena: &'arena Arena<N>,
    input: &str,
    env: &Ref<'arena, N>,
) -> Result<Ref<'arena, N>, ErrorCode> {
    let tokens = tokenize(arena, input)?;
    let expr = parse(arena, &tokens)?;
    eval(arena, &expr, env)
}

// ============================================================================
// Example/Test
// ============================================================================

pub fn run_example() -> Result<(), ErrorCode> {
    let arena = Arena::<DEFAULT_ARENA_SIZE>::new();
    let env = env_new(&arena)?;

    // Test: (+ 1 2 3)
    let result = eval_string(&arena, "(+ 1 2 3)", &env);
    if let Ok(val) = result {
        if let Some(Value::Number(n)) = arena.get(val.inner) {
            if n != 6 {
                return Err(ErrorCode::InvalidExpression);
            }
        }
    }

    // Test: (define x 42)
    let _ = eval_string(&arena, "(define x 42)", &env);

    // Test: (* x 2)
    let result = eval_string(&arena, "(* x 2)", &env);
    if let Ok(val) = result {
        if let Some(Value::Number(n)) = arena.get(val.inner) {
            if n != 84 {
                return Err(ErrorCode::InvalidExpression);
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
        if let Some(Value::Number(n)) = arena.get(val.inner) {
            if n != 120 {
                return Err(ErrorCode::InvalidExpression);
            }
        }
    }

    // Test: set! for mutation
    let _ = eval_string(&arena, "(define counter 0)", &env);
    let _ = eval_string(&arena, "(set! counter 42)", &env);
    let result = eval_string(&arena, "counter", &env);
    if let Ok(val) = result {
        if let Some(Value::Number(n)) = arena.get(val.inner) {
            if n != 42 {
                return Err(ErrorCode::InvalidExpression);
            }
        }
    }

    // Test: set! should fail on unbound variable
    let result = eval_string(&arena, "(set! nonexistent 123)", &env);
    if result.is_ok() {
        return Err(ErrorCode::InvalidExpression); // Should have failed
    }

    // Test: begin with multiple expressions
    let result = eval_string(&arena, "(begin 1 2 3)", &env);
    if let Ok(val) = result {
        if let Some(Value::Number(n)) = arena.get(val.inner) {
            if n != 3 {
                return Err(ErrorCode::InvalidExpression);
            }
        }
    }

    // Test: lambda with multiple body expressions
    let _ = eval_string(&arena, "(define x 0)", &env);
    let _ = eval_string(
        &arena,
        "(define inc-twice (lambda () (set! x (+ x 1)) (set! x (+ x 1)) x))",
        &env,
    );
    let result = eval_string(&arena, "(inc-twice)", &env);
    if let Ok(val) = result {
        if let Some(Value::Number(n)) = arena.get(val.inner) {
            if n != 2 {
                return Err(ErrorCode::InvalidExpression);
            }
        }
    }

    Ok(())
}
