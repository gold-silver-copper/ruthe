use crate::env::*;
use crate::value::*;
pub fn register_builtins(env: &ValRef) {
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
