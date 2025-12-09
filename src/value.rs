use alloc::rc::Rc;

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
