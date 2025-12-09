pub use crate::builtins::*;
pub use crate::env::*;
pub use crate::value::*;

// ============================================================================
// Tokenizer
// ============================================================================

fn parse_i64(s: &str) -> Result<i64, ()> {
    let bytes = s.as_bytes();
    if bytes.is_empty() {
        return Err(());
    }

    let (negative, start) = if bytes[0] == b'-' {
        if bytes.len() == 1 {
            return Err(());
        }
        (true, 1)
    } else if bytes[0] == b'+' {
        if bytes.len() == 1 {
            return Err(());
        }
        (false, 1)
    } else {
        (false, 0)
    };

    if bytes[start..].is_empty() {
        return Err(());
    }

    let mut result: i64 = 0;
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

fn tokenize(input: &str) -> Result<ValRef, ValRef> {
    let mut result = ValRef::nil();
    let mut chars = input.chars().peekable();

    while let Some(&ch) = chars.peek() {
        match ch {
            ' ' | '\t' | '\n' | '\r' => {
                chars.next();
            }
            '(' => {
                result = ValRef::cons(ValRef::symbol(ValRef::new_str("(")), result);
                chars.next();
            }
            ')' => {
                result = ValRef::cons(ValRef::symbol(ValRef::new_str(")")), result);
                chars.next();
            }
            '\'' => {
                result = ValRef::cons(ValRef::symbol(ValRef::new_str("'")), result);
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
                        result = ValRef::cons(ValRef::bool_val(true), result);
                        chars.next();
                    }
                    Some(&'f') => {
                        result = ValRef::cons(ValRef::bool_val(false), result);
                        chars.next();
                    }
                    _ => return Err(ValRef::new_str("Invalid boolean literal")),
                }
            }
            _ => {
                let mut atom_chars = ValRef::nil();
                while let Some(&c) = chars.peek() {
                    if c.is_whitespace() || c == '(' || c == ')' || c == '\'' {
                        break;
                    }
                    atom_chars = ValRef::cons(ValRef::char_val(c), atom_chars);
                    chars.next();
                }

                if atom_chars.is_nil() {
                    continue;
                }

                atom_chars = reverse_list(atom_chars);

                let mut buf = [0u8; 128];
                let mut idx = 0;
                let mut cur = atom_chars;
                loop {
                    match cur.as_ref() {
                        Value::Cons(cell) => {
                            let (car, cdr) = cell.borrow().clone();
                            if let Value::Char(ch) = car.as_ref() {
                                let mut char_buf = [0u8; 4];
                                let s = ch.encode_utf8(&mut char_buf);
                                for &b in s.as_bytes() {
                                    if idx >= buf.len() {
                                        return Err(ValRef::new_str("Atom too long"));
                                    }
                                    buf[idx] = b;
                                    idx += 1;
                                }
                            }
                            cur = cdr;
                        }
                        Value::Nil => break,
                        _ => break,
                    }
                }

                let atom_str = core::str::from_utf8(&buf[..idx])
                    .map_err(|_| ValRef::new_str("Invalid UTF-8"))?;

                if let Ok(num) = parse_i64(atom_str) {
                    result = ValRef::cons(ValRef::number(num), result);
                } else {
                    result = ValRef::cons(ValRef::symbol(ValRef::new_str(atom_str)), result);
                }
            }
        }
    }

    Ok(reverse_list(result))
}

// ============================================================================
// Parser
// ============================================================================

fn parse_tokens(tokens: ValRef) -> Result<(ValRef, ValRef), ValRef> {
    match tokens.as_ref() {
        Value::Nil => Err(ValRef::new_str("Unexpected end of input")),
        Value::Cons(cell) => {
            let (first, rest) = cell.borrow().clone();
            match first.as_ref() {
                Value::Number(_) | Value::Bool(_) => Ok((first, rest)),
                Value::Symbol(s) => {
                    let mut buf = [0u8; 32];
                    let s_str = s
                        .to_str_buf(&mut buf)
                        .map_err(|_| ValRef::new_str("Symbol too long"))?;

                    if s_str == "'" {
                        if let Value::Cons(next_cell) = rest.as_ref() {
                            let (next_expr, remaining) = next_cell.borrow().clone();
                            let (val, consumed) = parse_tokens(ValRef::cons(next_expr, remaining))?;
                            let quoted = ValRef::cons(
                                ValRef::symbol(ValRef::new_str("quote")),
                                ValRef::cons(val, ValRef::nil()),
                            );
                            Ok((quoted, consumed))
                        } else {
                            Err(ValRef::new_str("Quote requires an expression"))
                        }
                    } else if s_str == "(" {
                        let mut items = ValRef::nil();
                        let mut pos = rest;

                        loop {
                            match pos.as_ref() {
                                Value::Nil => return Err(ValRef::new_str("Unmatched '('")),
                                Value::Cons(token_cell) => {
                                    let (token, rest_tokens) = token_cell.borrow().clone();
                                    if let Value::Symbol(tok_s) = token.as_ref() {
                                        let mut tok_buf = [0u8; 32];
                                        let tok_str = tok_s
                                            .to_str_buf(&mut tok_buf)
                                            .map_err(|_| ValRef::new_str("Symbol too long"))?;
                                        if tok_str == ")" {
                                            return Ok((reverse_list(items), rest_tokens));
                                        }
                                    }
                                    let (val, consumed) = parse_tokens(pos)?;
                                    items = ValRef::cons(val, items);
                                    pos = consumed;
                                }
                                _ => return Err(ValRef::new_str("Invalid token stream")),
                            }
                        }
                    } else if s_str == ")" {
                        Err(ValRef::new_str("Unexpected ')'"))
                    } else {
                        Ok((first, rest))
                    }
                }
                _ => Err(ValRef::new_str("Unexpected token type")),
            }
        }
        _ => Err(ValRef::new_str("Invalid token stream")),
    }
}

pub fn parse(input: &str) -> Result<ValRef, ValRef> {
    let tokens = tokenize(input)?;
    if tokens.is_nil() {
        return Err(ValRef::new_str("Empty input"));
    }
    let (val, remaining) = parse_tokens(tokens)?;
    if !remaining.is_nil() {
        return Err(ValRef::new_str("Unexpected tokens after expression"));
    }
    Ok(val)
}

pub fn parse_multiple(input: &str) -> Result<ValRef, ValRef> {
    let tokens = tokenize(input)?;
    if tokens.is_nil() {
        return Err(ValRef::new_str("Empty input"));
    }

    let mut expressions = ValRef::nil();
    let mut pos = tokens;

    loop {
        if pos.is_nil() {
            break;
        }
        let (val, consumed) = parse_tokens(pos)?;
        expressions = ValRef::cons(val, expressions);
        pos = consumed;
    }

    Ok(reverse_list(expressions))
}
