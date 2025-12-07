#![no_std]

use core::cell::RefCell;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LispValue {
    Nil,
    Number(i32),
    Symbol(u32),
    Cons(usize, usize),
    Bool(bool),
}

pub struct Arena<const N: usize> {
    cells: RefCell<[LispValue; N]>,
    next_free: RefCell<usize>,
}

impl<const N: usize> Arena<N> {
    pub const fn new() -> Self {
        Arena {
            cells: RefCell::new([LispValue::Nil; N]),
            next_free: RefCell::new(0),
        }
    }

    pub fn alloc(&self, value: LispValue) -> Result<usize, ArenaError> {
        let mut next = self.next_free.borrow_mut();

        if *next >= N {
            return Err(ArenaError::OutOfMemory);
        }

        let index = *next;
        *next += 1;

        self.cells.borrow_mut()[index] = value;

        Ok(index)
    }

    pub fn get(&self, index: usize) -> Result<LispValue, ArenaError> {
        if index >= N {
            return Err(ArenaError::InvalidIndex);
        }

        Ok(self.cells.borrow()[index])
    }

    pub fn set(&self, index: usize, value: LispValue) -> Result<(), ArenaError> {
        if index >= N {
            return Err(ArenaError::InvalidIndex);
        }

        self.cells.borrow_mut()[index] = value;
        Ok(())
    }

    pub fn reset(&self) {
        *self.next_free.borrow_mut() = 0;
        for cell in self.cells.borrow_mut().iter_mut() {
            *cell = LispValue::Nil;
        }
    }

    pub fn capacity(&self) -> usize {
        N
    }

    pub fn used(&self) -> usize {
        *self.next_free.borrow()
    }

    pub fn available(&self) -> usize {
        N - *self.next_free.borrow()
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ArenaError {
    OutOfMemory,
    InvalidIndex,
}

// Example usage
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_allocation() {
        let arena: Arena<100> = Arena::new();

        let idx1 = arena.alloc(LispValue::Number(42)).unwrap();
        let idx2 = arena.alloc(LispValue::Bool(true)).unwrap();

        assert_eq!(arena.get(idx1).unwrap(), LispValue::Number(42));
        assert_eq!(arena.get(idx2).unwrap(), LispValue::Bool(true));
    }

    #[test]
    fn test_cons_cells() {
        let arena: Arena<100> = Arena::new();

        let car = arena.alloc(LispValue::Number(1)).unwrap();
        let cdr = arena.alloc(LispValue::Number(2)).unwrap();
        let cons = arena.alloc(LispValue::Cons(car, cdr)).unwrap();

        if let LispValue::Cons(c, d) = arena.get(cons).unwrap() {
            assert_eq!(arena.get(c).unwrap(), LispValue::Number(1));
            assert_eq!(arena.get(d).unwrap(), LispValue::Number(2));
        } else {
            panic!("Expected Cons cell");
        }
    }

    #[test]
    fn test_out_of_memory() {
        let arena: Arena<2> = Arena::new();

        arena.alloc(LispValue::Number(1)).unwrap();
        arena.alloc(LispValue::Number(2)).unwrap();

        let result = arena.alloc(LispValue::Number(3));
        assert_eq!(result, Err(ArenaError::OutOfMemory));
    }

    #[test]
    fn test_reset() {
        let arena: Arena<10> = Arena::new();

        arena.alloc(LispValue::Number(1)).unwrap();
        arena.alloc(LispValue::Number(2)).unwrap();

        assert_eq!(arena.used(), 2);

        arena.reset();

        assert_eq!(arena.used(), 0);
        assert_eq!(arena.get(0).unwrap(), LispValue::Nil);
    }
}
