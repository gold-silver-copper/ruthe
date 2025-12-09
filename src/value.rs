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

    pub fn list_to_display_str<'a>(&self, buf: &'a mut [u8]) -> Result<&'a str, ()> {
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

    pub fn list_len(&self) -> usize {
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

    pub fn list_nth(&self, n: usize) -> Option<ValRef> {
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

pub fn reverse_list(list: ValRef) -> ValRef {
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
