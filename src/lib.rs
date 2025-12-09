#![no_std]

extern crate alloc;
mod value;
pub use value::*;
mod env;
pub use env::*;
mod builtins;
pub use builtins::*;
mod parser;
pub use parser::*;

// ============================================================================
// Evaluator - With Proper Tail Call Optimization via Trampoline
// ============================================================================

fn eval_step(expr: ValRef, env: &ValRef) -> Result<EvalResult, ValRef> {
    match expr.as_ref() {
        Value::Number(_)
        | Value::Bool(_)
        | Value::Char(_)
        | Value::Builtin(_)
        | Value::Lambda { .. } => Ok(EvalResult::Done(expr.clone())),
        Value::Symbol(s) => {
            let mut buf = [0u8; 32];
            let s_str = s
                .to_str_buf(&mut buf)
                .map_err(|_| ValRef::new_str("Symbol too long"))?;

            if s_str == "nil" {
                return Ok(EvalResult::Done(ValRef::nil()));
            }
            env_get(env, s)
                .map(EvalResult::Done)
                .ok_or_else(|| ValRef::new_str("Unbound symbol"))
        }
        Value::Cons(cell) => {
            let (car, cdr) = cell.borrow().clone();
            if let Value::Symbol(sym) = car.as_ref() {
                let mut buf = [0u8; 32];
                let sym_str = sym
                    .to_str_buf(&mut buf)
                    .map_err(|_| ValRef::new_str("Symbol too long"))?;

                match sym_str {
                    "define" => {
                        let len = expr.as_ref().list_len();
                        if len != 3 {
                            return Err(ValRef::new_str("define requires 2 arguments"));
                        }
                        let name_val = expr
                            .as_ref()
                            .list_nth(1)
                            .ok_or(ValRef::new_str("define missing name"))?;
                        let name = name_val
                            .as_symbol()
                            .ok_or(ValRef::new_str("define requires symbol as first arg"))?
                            .clone();
                        let body_val = expr
                            .as_ref()
                            .list_nth(2)
                            .ok_or(ValRef::new_str("define missing body"))?;
                        let val = eval(body_val, env)?;
                        env_set(env, name, val.clone());
                        return Ok(EvalResult::Done(val));
                    }
                    "lambda" => {
                        let len = expr.as_ref().list_len();
                        if len != 3 {
                            return Err(ValRef::new_str(
                                "lambda requires 2 arguments (params body)",
                            ));
                        }

                        let params_list = expr
                            .as_ref()
                            .list_nth(1)
                            .ok_or(ValRef::new_str("lambda missing params"))?;

                        let mut current = params_list.clone();
                        loop {
                            match current.as_ref() {
                                Value::Cons(param_cell) => {
                                    let (param, rest) = param_cell.borrow().clone();
                                    if param.as_symbol().is_none() {
                                        return Err(ValRef::new_str(
                                            "lambda params must be symbols",
                                        ));
                                    }
                                    current = rest;
                                }
                                Value::Nil => break,
                                _ => return Err(ValRef::new_str("lambda params must be a list")),
                            }
                        }

                        let body = expr
                            .as_ref()
                            .list_nth(2)
                            .ok_or(ValRef::new_str("lambda missing body"))?;

                        return Ok(EvalResult::Done(ValRef::lambda(
                            params_list,
                            body,
                            env.clone(),
                        )));
                    }
                    "if" => {
                        let len = expr.as_ref().list_len();
                        if len != 4 {
                            return Err(ValRef::new_str("if requires 3 arguments"));
                        }
                        let cond_expr = expr
                            .as_ref()
                            .list_nth(1)
                            .ok_or(ValRef::new_str("if missing condition"))?;
                        let cond = eval(cond_expr, env)?;
                        let is_true = match cond.as_ref() {
                            Value::Bool(b) => *b,
                            Value::Nil => false,
                            _ => true,
                        };
                        let branch_idx = if is_true { 2 } else { 3 };
                        let branch = expr
                            .as_ref()
                            .list_nth(branch_idx)
                            .ok_or(ValRef::new_str("if missing branch"))?;
                        return Ok(EvalResult::TailCall(branch, env.clone()));
                    }
                    "quote" => {
                        let len = expr.as_ref().list_len();
                        if len != 2 {
                            return Err(ValRef::new_str("quote requires 1 argument"));
                        }
                        let quoted = expr
                            .as_ref()
                            .list_nth(1)
                            .ok_or(ValRef::new_str("quote missing argument"))?;
                        return Ok(EvalResult::Done(quoted));
                    }
                    _ => {}
                }
            }

            let func = eval(car, env)?;

            let mut args = ValRef::nil();
            let mut current = cdr;
            loop {
                match current.as_ref() {
                    Value::Cons(arg_cell) => {
                        let (arg_car, arg_cdr) = arg_cell.borrow().clone();
                        let evaled = eval(arg_car, env)?;
                        args = ValRef::cons(evaled, args);
                        current = arg_cdr;
                    }
                    Value::Nil => break,
                    _ => return Err(ValRef::new_str("Malformed argument list")),
                }
            }
            args = reverse_list(args);

            match func.as_ref() {
                Value::Builtin(f) => Ok(EvalResult::Done(f(&args)?)),
                Value::Lambda {
                    params,
                    body,
                    env: lambda_env,
                } => {
                    let param_count = params.as_ref().list_len();
                    let arg_count = args.as_ref().list_len();

                    if arg_count != param_count {
                        return Err(ValRef::new_str("Lambda argument count mismatch"));
                    }

                    let call_env = env_with_parent(lambda_env.clone());

                    let mut param_cur = params.clone();
                    let mut arg_cur = args.clone();

                    loop {
                        match (param_cur.as_ref(), arg_cur.as_ref()) {
                            (Value::Cons(p_cell), Value::Cons(a_cell)) => {
                                let (p_car, p_cdr) = p_cell.borrow().clone();
                                let (a_car, a_cdr) = a_cell.borrow().clone();
                                if let Value::Symbol(param_name) = p_car.as_ref() {
                                    env_set(&call_env, param_name.clone(), a_car);
                                }
                                param_cur = p_cdr;
                                arg_cur = a_cdr;
                            }
                            (Value::Nil, Value::Nil) => break,
                            _ => {
                                return Err(ValRef::new_str("Parameter/argument mismatch"));
                            }
                        }
                    }

                    Ok(EvalResult::TailCall(body.clone(), call_env))
                }
                _ => Err(ValRef::new_str("Cannot call non-function")),
            }
        }
        Value::Nil => Ok(EvalResult::Done(ValRef::nil())),
    }
}

pub fn eval(mut expr: ValRef, env: &ValRef) -> Result<ValRef, ValRef> {
    let mut current_env = env.clone();

    loop {
        match eval_step(expr, &current_env)? {
            EvalResult::Done(val) => return Ok(val),
            EvalResult::TailCall(new_expr, new_env) => {
                expr = new_expr;
                current_env = new_env;
            }
        }
    }
}

// ============================================================================
// Public API
// ============================================================================

pub fn eval_str(input: &str, env: &ValRef) -> Result<ValRef, ValRef> {
    let expr = parse(input)?;
    let result = eval(expr, env)?;
    Ok(result)
}

pub fn eval_str_multiple(input: &str, env: &ValRef) -> Result<ValRef, ValRef> {
    let expressions = parse_multiple(input)?;
    if expressions.is_nil() {
        return Err(ValRef::new_str("No expressions to evaluate"));
    }

    let mut last_result = ValRef::nil();
    let mut current = expressions;

    loop {
        match current.as_ref() {
            Value::Cons(cell) => {
                let (expr, rest) = cell.borrow().clone();
                last_result = eval(expr, env)?;
                current = rest;
            }
            Value::Nil => break,
            _ => return Err(ValRef::new_str("Invalid expression list")),
        }
    }

    Ok(last_result)
}

// Public function to create new environment
pub fn new_env() -> ValRef {
    env_new()
}
