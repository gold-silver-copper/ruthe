use crate::value::*;

use crate::builtins::*;

// ============================================================================
// Environment Operations - Mutable cons list manipulation
// ============================================================================

/// Environment structure:
/// Cons((bindings . parent_env))
/// where bindings is a cons list of (symbol . value) pairs

pub fn env_new() -> ValRef {
    let env = ValRef::cons(ValRef::nil(), ValRef::nil());
    register_builtins(&env);
    env
}

pub fn env_with_parent(parent: ValRef) -> ValRef {
    ValRef::cons(ValRef::nil(), parent)
}

pub fn env_set(env: &ValRef, name: ValRef, value: ValRef) {
    // env is Cons(bindings, parent)
    if let Some(cell) = env.as_cons() {
        let (bindings, parent) = cell.borrow().clone();
        let new_binding = ValRef::cons(ValRef::symbol(name), value);
        let new_bindings = ValRef::cons(new_binding, bindings);
        *cell.borrow_mut() = (new_bindings, parent);
    }
}

pub fn env_get(env: &ValRef, name: &ValRef) -> Option<ValRef> {
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
