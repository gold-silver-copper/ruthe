#![no_std]

use core::cell::Cell;

// ============================================================================
// String Interning System - NO heap allocation!
// ============================================================================

pub const STRING_TABLE_SIZE: usize = 2048; // Total bytes for string data
pub const MAX_STRINGS: usize = 256; // Maximum number of unique strings

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StringId(pub u8);

impl StringId {
    pub const INVALID: StringId = StringId(u8::MAX);

    pub fn is_valid(self) -> bool {
        self.0 != u8::MAX
    }
}

pub struct StringTable {
    // Raw string data storage
    data: [Cell<u8>; STRING_TABLE_SIZE],
    // (start_offset, length) for each interned string
    strings: [Cell<(u16, u16)>; MAX_STRINGS],
    next_string: Cell<usize>,
    data_pos: Cell<usize>,
}

impl StringTable {
    pub const fn new() -> Self {
        StringTable {
            data: [const { Cell::new(0u8) }; STRING_TABLE_SIZE],
            strings: [const { Cell::new((0, 0)) }; MAX_STRINGS],
            next_string: Cell::new(0),
            data_pos: Cell::new(0),
        }
    }

    /// Intern a string, returning its ID. Returns existing ID if already interned.
    pub fn intern(&self, s: &str) -> Result<StringId, ErrorCode> {
        let bytes = s.as_bytes();

        // Check if already interned
        for i in 0..self.next_string.get() {
            let (start, len) = self.strings[i].get();
            let start = start as usize;
            let len = len as usize;

            if len == bytes.len() {
                let mut matches = true;
                for j in 0..len {
                    if self.data[start + j].get() != bytes[j] {
                        matches = false;
                        break;
                    }
                }
                if matches {
                    return Ok(StringId(i as u8));
                }
            }
        }

        // Not found, need to intern
        let next = self.next_string.get();
        if next >= MAX_STRINGS {
            return Err(ErrorCode::StringTableFull);
        }

        let pos = self.data_pos.get();
        if pos + bytes.len() > STRING_TABLE_SIZE {
            return Err(ErrorCode::StringTableFull);
        }

        // Copy string data
        for (i, &b) in bytes.iter().enumerate() {
            self.data[pos + i].set(b);
        }

        // Record string location
        self.strings[next].set((pos as u16, bytes.len() as u16));
        self.next_string.set(next + 1);
        self.data_pos.set(pos + bytes.len());

        Ok(StringId(next as u8))
    }

    /// Get a string by its ID (copies to provided buffer)
    pub fn get_to_buf<'a>(&self, id: StringId, buf: &'a mut [u8]) -> Option<&'a str> {
        if !id.is_valid() || (id.0 as usize) >= self.next_string.get() {
            return None;
        }

        let (start, len) = self.strings[id.0 as usize].get();
        let start = start as usize;
        let len = len as usize;

        if len > buf.len() {
            return None;
        }

        for i in 0..len {
            buf[i] = self.data[start + i].get();
        }

        core::str::from_utf8(&buf[..len]).ok()
    }

    /// Check if a StringId matches a given string literal (O(n) but avoids allocation)
    pub fn matches(&self, id: StringId, s: &str) -> bool {
        if !id.is_valid() || (id.0 as usize) >= self.next_string.get() {
            return false;
        }

        let (start, len) = self.strings[id.0 as usize].get();
        let start = start as usize;
        let len = len as usize;

        let bytes = s.as_bytes();
        if len != bytes.len() {
            return false;
        }

        for i in 0..len {
            if self.data[start + i].get() != bytes[i] {
                return false;
            }
        }

        true
    }

    /// Get a string by its ID (for debugging/display - limited to 64 bytes)
    pub fn get(&self, id: StringId) -> Option<&'static str> {
        // This is a limitation of using Cell - we can't return a direct reference
        // This method should primarily be used for debugging
        // For comparisons, use eq() or matches() instead
        None
    }

    /// Compare two string IDs (O(1) operation!)
    pub fn eq(&self, a: StringId, b: StringId) -> bool {
        a.0 == b.0
    }
}

// ============================================================================
// Error System
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
    StringTableFull,
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
            ErrorCode::StringTableFull => "String table full",
        }
    }
}

// ============================================================================
// Arena Allocator with Automatic Reference Counting
// ============================================================================

pub const DEFAULT_ARENA_SIZE: usize = 10000;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct ArenaRef(pub u16);

impl ArenaRef {
    pub const NULL: ArenaRef = ArenaRef(u16::MAX);

    pub fn is_null(self) -> bool {
        self.0 == u16::MAX
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Value {
    Number(i64),
    Bool(bool),
    Cons(ArenaRef, ArenaRef),
    Symbol(StringId), // NOW USING STRING INTERNING!
    Lambda(ArenaRef, ArenaRef, ArenaRef),
    Builtin(u8),
    Nil,
    Free,
}

#[derive(Debug)]
pub struct Arena<const N: usize> {
    pub values: [Cell<Value>; N],
    pub refcounts: [Cell<u16>; N],
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
                return Ok(ArenaRef(idx as u16));
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
            if rc < u16::MAX {
                self.refcounts[idx].set(rc + 1);
            }
        }
    }

    pub fn decref(&self, r: ArenaRef) {
        if r.is_null() {
            return;
        }

        let mut stack = [ArenaRef::NULL; 128];
        let mut stack_len;

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

                match value {
                    Value::Cons(car, cdr) => {
                        if stack_len + 2 <= stack.len() {
                            stack[stack_len] = car;
                            stack[stack_len + 1] = cdr;
                            stack_len += 2;
                        } else {
                            self.decref(car);
                            self.decref(cdr);
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

    pub fn nil(&self) -> Result<Ref<N>, ErrorCode> {
        Ok(Ref::from_alloc(self, self.alloc(Value::Nil)?))
    }

    pub fn number(&self, n: i64) -> Result<Ref<N>, ErrorCode> {
        Ok(Ref::from_alloc(self, self.alloc(Value::Number(n))?))
    }

    pub fn bool_val(&self, b: bool) -> Result<Ref<N>, ErrorCode> {
        Ok(Ref::from_alloc(self, self.alloc(Value::Bool(b))?))
    }

    pub fn cons(&self, car: &Ref<N>, cdr: &Ref<N>) -> Result<Ref<N>, ErrorCode> {
        self.incref(car.inner);
        self.incref(cdr.inner);
        Ok(Ref::from_alloc(
            self,
            self.alloc(Value::Cons(car.inner, cdr.inner))?,
        ))
    }

    pub fn symbol(&self, id: StringId) -> Result<Ref<N>, ErrorCode> {
        Ok(Ref::from_alloc(self, self.alloc(Value::Symbol(id))?))
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
// RAII Reference Wrapper
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
// Helper Functions
// ============================================================================

impl<const N: usize> Arena<N> {
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
}

// ============================================================================
// Environment Operations
// ============================================================================

pub fn env_new<'arena, const N: usize>(
    arena: &'arena Arena<N>,
    strings: &StringTable,
) -> Result<Ref<'arena, N>, ErrorCode> {
    let nil1 = arena.nil()?;
    let nil2 = arena.nil()?;
    let env = arena.cons(&nil1, &nil2)?;
    register_builtins(arena, strings, &env)?;
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
    strings: &StringTable,
    env: &Ref<'arena, N>,
    name: StringId,
    value: &Ref<'arena, N>,
) -> Result<(), ErrorCode> {
    if let Some(Value::Cons(bindings, parent)) = arena.get(env.inner) {
        let bindings_ref = Ref::new(arena, bindings);
        let parent_ref = Ref::new(arena, parent);

        let filtered_bindings = remove_binding(arena, strings, &bindings_ref, name)?;

        let sym = arena.symbol(name)?;
        let new_binding = arena.cons(&sym, value)?;
        let new_bindings = arena.cons(&new_binding, &filtered_bindings)?;

        arena.set_cons(env, &new_bindings, &parent_ref);
    }
    Ok(())
}

pub fn env_update<'arena, const N: usize>(
    arena: &'arena Arena<N>,
    strings: &StringTable,
    env: &Ref<'arena, N>,
    name: StringId,
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
                                if let Some(Value::Symbol(key_id)) = arena.get(key) {
                                    if strings.eq(key_id, name) {
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

fn remove_binding<'arena, const N: usize>(
    arena: &'arena Arena<N>,
    strings: &StringTable,
    bindings: &Ref<'arena, N>,
    name: StringId,
) -> Result<Ref<'arena, N>, ErrorCode> {
    match arena.get(bindings.inner) {
        Some(Value::Nil) => arena.nil(),
        Some(Value::Cons(binding, rest)) => {
            let rest_ref = Ref::new(arena, rest);

            if let Some(Value::Cons(key, _)) = arena.get(binding) {
                if let Some(Value::Symbol(key_id)) = arena.get(key) {
                    if strings.eq(key_id, name) {
                        return remove_binding(arena, strings, &rest_ref, name);
                    }
                }
            }

            let binding_ref = Ref::new(arena, binding);
            let filtered_rest = remove_binding(arena, strings, &rest_ref, name)?;
            arena.cons(&binding_ref, &filtered_rest)
        }
        _ => arena.nil(),
    }
}

pub fn env_get<'arena, const N: usize>(
    arena: &'arena Arena<N>,
    strings: &StringTable,
    env: &Ref<'arena, N>,
    name: StringId,
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
                                if let Some(Value::Symbol(key_id)) = arena.get(key) {
                                    if strings.eq(key_id, name) {
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
            _ => return Err(ErrorCode::InvalidArgs),
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
    strings: &StringTable,
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
        let name_id = strings.intern(name)?;
        let builtin_ref = arena.builtin(idx)?;
        env_set(arena, strings, env, name_id, &builtin_ref)?;
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
    strings: &StringTable,
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
                let id = strings.intern("(")?;
                let sym = arena.symbol(id)?;
                result = arena.cons(&sym, &result)?;
                chars.next();
            }
            ')' => {
                let id = strings.intern(")")?;
                let sym = arena.symbol(id)?;
                result = arena.cons(&sym, &result)?;
                chars.next();
            }
            '\'' => {
                let id = strings.intern("'")?;
                let sym = arena.symbol(id)?;
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
                let mut atom_buf = [0u8; 64];
                let mut atom_len = 0;

                while let Some(&c) = chars.peek() {
                    if is_delimiter(c) {
                        break;
                    }
                    if atom_len >= atom_buf.len() {
                        return Err(ErrorCode::AtomTooLong);
                    }
                    let mut char_buf = [0u8; 4];
                    let char_str = c.encode_utf8(&mut char_buf);
                    for &b in char_str.as_bytes() {
                        if atom_len >= atom_buf.len() {
                            return Err(ErrorCode::AtomTooLong);
                        }
                        atom_buf[atom_len] = b;
                        atom_len += 1;
                    }
                    chars.next();
                }

                if let Ok(s) = core::str::from_utf8(&atom_buf[..atom_len]) {
                    if let Ok(num) = parse_i64(s) {
                        let num_val = arena.number(num)?;
                        result = arena.cons(&num_val, &result)?;
                    } else {
                        let id = strings.intern(s)?;
                        let sym = arena.symbol(id)?;
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
    strings: &StringTable,
    tokens: &Ref<'arena, N>,
) -> Result<Ref<'arena, N>, ErrorCode> {
    let (expr, remaining) = parse_one(arena, strings, tokens)?;
    if !matches!(arena.get(remaining.inner), Some(Value::Nil)) {
        return Err(ErrorCode::ExtraTokens);
    }
    Ok(expr)
}

fn parse_one<'arena, const N: usize>(
    arena: &'arena Arena<N>,
    strings: &StringTable,
    tokens: &Ref<'arena, N>,
) -> Result<(Ref<'arena, N>, Ref<'arena, N>), ErrorCode> {
    match arena.get(tokens.inner) {
        Some(Value::Nil) => Err(ErrorCode::UnexpectedEOF),
        Some(Value::Cons(first, rest)) => {
            let first_ref = Ref::new(arena, first);
            let rest_ref = Ref::new(arena, rest);

            match arena.get(first) {
                Some(Value::Number(_)) | Some(Value::Bool(_)) => Ok((first_ref, rest_ref)),
                Some(Value::Symbol(id)) => {
                    if strings.matches(id, "'") {
                        let (expr, consumed) = parse_one(arena, strings, &rest_ref)?;
                        let quote_id = strings.intern("quote")?;
                        let quoted_sym = arena.symbol(quote_id)?;
                        let nil = arena.nil()?;
                        let list = arena.cons(&expr, &nil)?;
                        let quoted = arena.cons(&quoted_sym, &list)?;
                        Ok((quoted, consumed))
                    } else if strings.matches(id, "(") {
                        parse_list(arena, strings, &rest_ref)
                    } else if strings.matches(id, ")") {
                        Err(ErrorCode::UnexpectedCloseParen)
                    } else {
                        Ok((first_ref, rest_ref))
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
    strings: &StringTable,
    tokens: &Ref<'arena, N>,
) -> Result<(Ref<'arena, N>, Ref<'arena, N>), ErrorCode> {
    let mut items = arena.nil()?;
    let mut current = tokens.clone();

    loop {
        match arena.get(current.inner) {
            Some(Value::Nil) => return Err(ErrorCode::UnmatchedOpenParen),
            Some(Value::Cons(tok, rest)) => {
                if let Some(Value::Symbol(id)) = arena.get(tok) {
                    if strings.matches(id, ")") {
                        let result = arena.reverse_list(&items)?;
                        return Ok((result, Ref::new(arena, rest)));
                    }
                }

                let (expr, consumed) = parse_one(arena, strings, &current)?;
                items = arena.cons(&expr, &items)?;
                current = consumed;
            }
            _ => return Err(ErrorCode::InvalidToken),
        }
    }
}

// ============================================================================
// Evaluator with Tail Call Optimization
// ============================================================================

pub fn eval<'arena, const N: usize>(
    arena: &'arena Arena<N>,
    strings: &StringTable,
    expr: &Ref<'arena, N>,
    env: &Ref<'arena, N>,
) -> Result<Ref<'arena, N>, ErrorCode> {
    let mut current_expr = expr.clone();
    let mut current_env = env.clone();

    loop {
        match eval_step(arena, strings, &current_expr, &current_env)? {
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
    strings: &StringTable,
    expr: &Ref<'arena, N>,
    env: &Ref<'arena, N>,
) -> Result<EvalResult<'arena, N>, ErrorCode> {
    match arena.get(expr.inner) {
        Some(Value::Number(_))
        | Some(Value::Bool(_))
        | Some(Value::Builtin(_))
        | Some(Value::Lambda(..)) => Ok(EvalResult::Done(expr.clone())),
        Some(Value::Symbol(id)) => {
            let nil_id = strings.intern("nil")?;
            let is_nil = strings.eq(id, nil_id);

            if is_nil {
                return Ok(EvalResult::Done(arena.nil()?));
            }

            if let Some(val) = env_get(arena, strings, env, id) {
                Ok(EvalResult::Done(val))
            } else {
                Err(ErrorCode::UnboundSymbol)
            }
        }
        Some(Value::Cons(car, _)) => {
            let car_ref = Ref::new(arena, car);

            if let Some(Value::Symbol(sym_id)) = arena.get(car) {
                if strings.matches(sym_id, "quote") {
                    if let Some(quoted) = arena.list_nth(expr, 1) {
                        return Ok(EvalResult::Done(quoted));
                    }
                } else if strings.matches(sym_id, "if") {
                    return eval_if(arena, strings, expr, env);
                } else if strings.matches(sym_id, "lambda") {
                    return eval_lambda(arena, strings, expr, env);
                } else if strings.matches(sym_id, "define") {
                    return eval_define(arena, strings, expr, env);
                } else if strings.matches(sym_id, "set!") {
                    return eval_set(arena, strings, expr, env);
                } else if strings.matches(sym_id, "begin") {
                    return eval_begin(arena, strings, expr, env);
                }
            }

            let func = eval(arena, strings, &car_ref, env)?;
            let args = eval_args(arena, strings, expr, env)?;

            apply(arena, strings, &func, &args)
        }
        Some(Value::Nil) => Ok(EvalResult::Done(arena.nil()?)),
        _ => Err(ErrorCode::InvalidExpression),
    }
}

fn eval_if<'arena, const N: usize>(
    arena: &'arena Arena<N>,
    strings: &StringTable,
    expr: &Ref<'arena, N>,
    env: &Ref<'arena, N>,
) -> Result<EvalResult<'arena, N>, ErrorCode> {
    if arena.list_len(expr) != 4 {
        return Err(ErrorCode::InvalidIf);
    }

    let cond_expr = arena.list_nth(expr, 1).unwrap();
    let cond = eval(arena, strings, &cond_expr, env)?;

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
    strings: &StringTable,
    expr: &Ref<'arena, N>,
    env: &Ref<'arena, N>,
) -> Result<EvalResult<'arena, N>, ErrorCode> {
    let len = arena.list_len(expr);
    if len < 3 {
        return Err(ErrorCode::InvalidLambda);
    }

    let params = arena.list_nth(expr, 1).unwrap();

    let body = if len == 3 {
        arena.list_nth(expr, 2).unwrap()
    } else {
        let begin_id = strings.intern("begin")?;
        let begin_symbol = arena.symbol(begin_id)?;

        let mut body_exprs = arena.nil()?;
        let mut current = expr.clone();

        if let Some(Value::Cons(_, rest1)) = arena.get(current.inner) {
            if let Some(Value::Cons(_, rest2)) = arena.get(rest1) {
                current = Ref::new(arena, rest2);
            }
        }

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
    strings: &StringTable,
    expr: &Ref<'arena, N>,
    env: &Ref<'arena, N>,
) -> Result<EvalResult<'arena, N>, ErrorCode> {
    let len = arena.list_len(expr);
    if len < 2 {
        return Err(ErrorCode::InvalidBegin);
    }

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

                let is_last = matches!(arena.get(cdr), Some(Value::Nil));

                if is_last {
                    return Ok(EvalResult::TailCall(car_ref, env.clone()));
                } else {
                    result = eval(arena, strings, &car_ref, env)?;
                    current = Ref::new(arena, cdr);
                }
            }
            Some(Value::Nil) => {
                return Ok(EvalResult::Done(result));
            }
            _ => return Err(ErrorCode::InvalidBegin),
        }
    }
}

fn eval_define<'arena, const N: usize>(
    arena: &'arena Arena<N>,
    strings: &StringTable,
    expr: &Ref<'arena, N>,
    env: &Ref<'arena, N>,
) -> Result<EvalResult<'arena, N>, ErrorCode> {
    if arena.list_len(expr) != 3 {
        return Err(ErrorCode::InvalidDefine);
    }

    let name = arena.list_nth(expr, 1).unwrap();
    let name_id = if let Some(Value::Symbol(id)) = arena.get(name.inner) {
        id
    } else {
        return Err(ErrorCode::InvalidDefine);
    };

    let value_expr = arena.list_nth(expr, 2).unwrap();
    let value = eval(arena, strings, &value_expr, env)?;

    env_set(arena, strings, env, name_id, &value)?;
    Ok(EvalResult::Done(value))
}

fn eval_set<'arena, const N: usize>(
    arena: &'arena Arena<N>,
    strings: &StringTable,
    expr: &Ref<'arena, N>,
    env: &Ref<'arena, N>,
) -> Result<EvalResult<'arena, N>, ErrorCode> {
    if arena.list_len(expr) != 3 {
        return Err(ErrorCode::InvalidSet);
    }

    let name = arena.list_nth(expr, 1).unwrap();
    let name_id = if let Some(Value::Symbol(id)) = arena.get(name.inner) {
        id
    } else {
        return Err(ErrorCode::InvalidSet);
    };

    let value_expr = arena.list_nth(expr, 2).unwrap();
    let value = eval(arena, strings, &value_expr, env)?;

    env_update(arena, strings, env, name_id, &value)?;

    Ok(EvalResult::Done(value))
}

fn eval_args<'arena, const N: usize>(
    arena: &'arena Arena<N>,
    strings: &StringTable,
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
                let val = eval(arena, strings, &car_ref, env)?;
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
    strings: &StringTable,
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
                        if let Some(Value::Symbol(name_id)) = arena.get(p_car) {
                            let a_car_ref = Ref::new(arena, a_car);
                            env_set(arena, strings, &call_env, name_id, &a_car_ref)?;
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
    strings: &StringTable,
    input: &str,
    env: &Ref<'arena, N>,
) -> Result<Ref<'arena, N>, ErrorCode> {
    let tokens = tokenize(arena, strings, input)?;
    let expr = parse(arena, strings, &tokens)?;
    eval(arena, strings, &expr, env)
}

// ============================================================================
// Example/Test
// ============================================================================

pub fn run_example() -> Result<(), ErrorCode> {
    let arena = Arena::<DEFAULT_ARENA_SIZE>::new();
    let strings = StringTable::new();
    let env = env_new(&arena, &strings)?;

    // Test: (+ 1 2 3)
    let result = eval_string(&arena, &strings, "(+ 1 2 3)", &env);
    if let Ok(val) = result {
        if let Some(Value::Number(n)) = arena.get(val.inner) {
            if n != 6 {
                return Err(ErrorCode::InvalidExpression);
            }
        }
    }

    // Test: (define x 42)
    let _ = eval_string(&arena, &strings, "(define x 42)", &env);

    // Test: (* x 2)
    let result = eval_string(&arena, &strings, "(* x 2)", &env);
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
    let _ = eval_string(&arena, &strings, factorial_def, &env);

    // Test: (factorial 5)
    let result = eval_string(&arena, &strings, "(factorial 5)", &env);
    if let Ok(val) = result {
        if let Some(Value::Number(n)) = arena.get(val.inner) {
            if n != 120 {
                return Err(ErrorCode::InvalidExpression);
            }
        }
    }

    // Test: set! for mutation
    let _ = eval_string(&arena, &strings, "(define counter 0)", &env);
    let _ = eval_string(&arena, &strings, "(set! counter 42)", &env);
    let result = eval_string(&arena, &strings, "counter", &env);
    if let Ok(val) = result {
        if let Some(Value::Number(n)) = arena.get(val.inner) {
            if n != 42 {
                return Err(ErrorCode::InvalidExpression);
            }
        }
    }

    // Test: set! should fail on unbound variable
    let result = eval_string(&arena, &strings, "(set! nonexistent 123)", &env);
    if result.is_ok() {
        return Err(ErrorCode::InvalidExpression);
    }

    // Test: begin with multiple expressions
    let result = eval_string(&arena, &strings, "(begin 1 2 3)", &env);
    if let Ok(val) = result {
        if let Some(Value::Number(n)) = arena.get(val.inner) {
            if n != 3 {
                return Err(ErrorCode::InvalidExpression);
            }
        }
    }

    // Test: lambda with multiple body expressions
    let _ = eval_string(&arena, &strings, "(define x 0)", &env);
    let _ = eval_string(
        &arena,
        &strings,
        "(define inc-twice (lambda () (set! x (+ x 1)) (set! x (+ x 1)) x))",
        &env,
    );
    let result = eval_string(&arena, &strings, "(inc-twice)", &env);
    if let Ok(val) = result {
        if let Some(Value::Number(n)) = arena.get(val.inner) {
            if n != 2 {
                return Err(ErrorCode::InvalidExpression);
            }
        }
    }

    Ok(())
}
