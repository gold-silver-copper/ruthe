#![no_std]

extern crate alloc;
use alloc::vec::Vec;
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
    allocated: RefCell<[bool; N]>,
    marked: RefCell<[bool; N]>,
    free_list: RefCell<Vec<usize>>,
    next_free: RefCell<usize>,
}

impl<const N: usize> Arena<N> {
    pub const fn new() -> Self {
        Arena {
            cells: RefCell::new([LispValue::Nil; N]),
            allocated: RefCell::new([false; N]),
            marked: RefCell::new([false; N]),
            free_list: RefCell::new(Vec::new()),
            next_free: RefCell::new(0),
        }
    }

    pub fn alloc(&self, value: LispValue) -> Result<usize, ArenaError> {
        // Try to use a cell from the free list first
        if let Some(index) = self.free_list.borrow_mut().pop() {
            self.cells.borrow_mut()[index] = value;
            self.allocated.borrow_mut()[index] = true;
            return Ok(index);
        }

        // Otherwise, allocate from the next free position
        let mut next = self.next_free.borrow_mut();
        if *next >= N {
            return Err(ArenaError::OutOfMemory);
        }
        let index = *next;
        *next += 1;
        self.cells.borrow_mut()[index] = value;
        self.allocated.borrow_mut()[index] = true;
        Ok(index)
    }

    pub fn get(&self, index: usize) -> Result<LispValue, ArenaError> {
        if index >= N {
            return Err(ArenaError::InvalidIndex);
        }
        if !self.allocated.borrow()[index] {
            return Err(ArenaError::InvalidIndex);
        }
        Ok(self.cells.borrow()[index])
    }

    pub fn set(&self, index: usize, value: LispValue) -> Result<(), ArenaError> {
        if index >= N {
            return Err(ArenaError::InvalidIndex);
        }
        if !self.allocated.borrow()[index] {
            return Err(ArenaError::InvalidIndex);
        }
        self.cells.borrow_mut()[index] = value;
        Ok(())
    }

    pub fn reset(&self) {
        *self.next_free.borrow_mut() = 0;
        self.free_list.borrow_mut().clear();
        for cell in self.cells.borrow_mut().iter_mut() {
            *cell = LispValue::Nil;
        }
        for alloc in self.allocated.borrow_mut().iter_mut() {
            *alloc = false;
        }
    }

    pub fn capacity(&self) -> usize {
        N
    }

    pub fn used(&self) -> usize {
        self.allocated.borrow().iter().filter(|&&x| x).count()
    }

    pub fn available(&self) -> usize {
        N - self.used()
    }

    // Mark phase: recursively mark all reachable cells from roots
    fn mark(&self, index: usize) {
        // Bounds check
        if index >= N {
            return;
        }

        // Skip if already marked or not allocated
        if self.marked.borrow()[index] || !self.allocated.borrow()[index] {
            return;
        }

        // Mark this cell
        self.marked.borrow_mut()[index] = true;

        // Recursively mark children if this is a Cons cell
        let value = self.cells.borrow()[index];
        if let LispValue::Cons(car, cdr) = value {
            self.mark(car);
            self.mark(cdr);
        }
    }

    // Sweep phase: collect unmarked cells and add them to free list
    fn sweep(&self) {
        let next = *self.next_free.borrow();
        let mut free_list = self.free_list.borrow_mut();
        let mut allocated = self.allocated.borrow_mut();
        let mut marked = self.marked.borrow_mut();

        for i in 0..next {
            if allocated[i] && !marked[i] {
                // This cell is allocated but not marked - collect it
                allocated[i] = false;
                free_list.push(i);
            }
            // Reset mark bit for next collection
            marked[i] = false;
        }
    }

    // Perform mark-and-sweep garbage collection
    pub fn collect(&self, roots: &[usize]) {
        // Mark phase: mark all reachable cells from roots
        for &root in roots {
            self.mark(root);
        }

        // Sweep phase: collect unmarked cells
        self.sweep();
    }

    // Helper for tests: check if a cell is allocated
    pub fn is_allocated(&self, index: usize) -> bool {
        if index >= N {
            return false;
        }
        self.allocated.borrow()[index]
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ArenaError {
    OutOfMemory,
    InvalidIndex,
}

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
        assert_eq!(arena.get(0).unwrap_err(), ArenaError::InvalidIndex);
    }

    // GC Tests

    #[test]
    fn test_gc_simple_collection() {
        let arena: Arena<10> = Arena::new();

        // Allocate some values
        let v1 = arena.alloc(LispValue::Number(1)).unwrap();
        let v2 = arena.alloc(LispValue::Number(2)).unwrap();
        let v3 = arena.alloc(LispValue::Number(3)).unwrap();

        assert_eq!(arena.used(), 3);

        // Only root v1 and v3, v2 should be collected
        arena.collect(&[v1, v3]);

        assert_eq!(arena.used(), 2);
        assert!(arena.is_allocated(v1));
        assert!(!arena.is_allocated(v2));
        assert!(arena.is_allocated(v3));
    }

    #[test]
    fn test_gc_no_roots() {
        let arena: Arena<10> = Arena::new();

        // Allocate several values
        arena.alloc(LispValue::Number(1)).unwrap();
        arena.alloc(LispValue::Number(2)).unwrap();
        arena.alloc(LispValue::Number(3)).unwrap();

        assert_eq!(arena.used(), 3);

        // Collect with no roots - everything should be freed
        arena.collect(&[]);

        assert_eq!(arena.used(), 0);
    }

    #[test]
    fn test_gc_cons_chain() {
        let arena: Arena<20> = Arena::new();

        // Create a list: (1 . (2 . (3 . nil)))
        let n3 = arena.alloc(LispValue::Number(3)).unwrap();
        let nil = arena.alloc(LispValue::Nil).unwrap();
        let cons3 = arena.alloc(LispValue::Cons(n3, nil)).unwrap();

        let n2 = arena.alloc(LispValue::Number(2)).unwrap();
        let cons2 = arena.alloc(LispValue::Cons(n2, cons3)).unwrap();

        let n1 = arena.alloc(LispValue::Number(1)).unwrap();
        let cons1 = arena.alloc(LispValue::Cons(n1, cons2)).unwrap();

        // Also allocate garbage
        let garbage1 = arena.alloc(LispValue::Number(99)).unwrap();
        let garbage2 = arena.alloc(LispValue::Number(100)).unwrap();

        assert_eq!(arena.used(), 9);

        // Root only the head of the list
        arena.collect(&[cons1]);

        // All list elements should be preserved, garbage collected
        assert_eq!(arena.used(), 7);
        assert!(arena.is_allocated(cons1));
        assert!(arena.is_allocated(n1));
        assert!(arena.is_allocated(cons2));
        assert!(arena.is_allocated(n2));
        assert!(arena.is_allocated(cons3));
        assert!(arena.is_allocated(n3));
        assert!(arena.is_allocated(nil));
        assert!(!arena.is_allocated(garbage1));
        assert!(!arena.is_allocated(garbage2));
    }

    #[test]
    fn test_gc_circular_reference() {
        let arena: Arena<20> = Arena::new();

        // Create circular structure: cons1 -> cons2 -> cons1
        let placeholder = arena.alloc(LispValue::Nil).unwrap();
        let cons1 = arena
            .alloc(LispValue::Cons(placeholder, placeholder))
            .unwrap();
        let cons2 = arena.alloc(LispValue::Cons(placeholder, cons1)).unwrap();

        // Update cons1 to point to cons2 (creating cycle)
        arena
            .set(cons1, LispValue::Cons(placeholder, cons2))
            .unwrap();

        // Allocate garbage
        let garbage = arena.alloc(LispValue::Number(999)).unwrap();

        assert_eq!(arena.used(), 4);

        // Root cons1, which should preserve the cycle
        arena.collect(&[cons1]);

        assert_eq!(arena.used(), 3);
        assert!(arena.is_allocated(cons1));
        assert!(arena.is_allocated(cons2));
        assert!(arena.is_allocated(placeholder));
        assert!(!arena.is_allocated(garbage));
    }

    #[test]
    fn test_gc_multiple_roots() {
        let arena: Arena<20> = Arena::new();

        // Create two separate structures
        let a1 = arena.alloc(LispValue::Number(1)).unwrap();
        let a2 = arena.alloc(LispValue::Number(2)).unwrap();
        let list_a = arena.alloc(LispValue::Cons(a1, a2)).unwrap();

        let b1 = arena.alloc(LispValue::Number(10)).unwrap();
        let b2 = arena.alloc(LispValue::Number(20)).unwrap();
        let list_b = arena.alloc(LispValue::Cons(b1, b2)).unwrap();

        // Garbage
        let garbage = arena.alloc(LispValue::Number(999)).unwrap();

        assert_eq!(arena.used(), 7);

        // Root both lists
        arena.collect(&[list_a, list_b]);

        assert_eq!(arena.used(), 6);
        assert!(arena.is_allocated(list_a));
        assert!(arena.is_allocated(list_b));
        assert!(!arena.is_allocated(garbage));
    }

    #[test]
    fn test_gc_reuse_freed_cells() {
        let arena: Arena<5> = Arena::new();

        // Fill the arena
        let v1 = arena.alloc(LispValue::Number(1)).unwrap();
        let v2 = arena.alloc(LispValue::Number(2)).unwrap();
        let v3 = arena.alloc(LispValue::Number(3)).unwrap();
        let v4 = arena.alloc(LispValue::Number(4)).unwrap();
        let v5 = arena.alloc(LispValue::Number(5)).unwrap();

        // Arena is full
        assert_eq!(arena.used(), 5);
        assert!(arena.alloc(LispValue::Number(6)).is_err());

        // Collect, keeping only v1 and v5
        arena.collect(&[v1, v5]);

        assert_eq!(arena.used(), 2);

        // Now we should be able to allocate again (using freed cells)
        let v6 = arena.alloc(LispValue::Number(6)).unwrap();
        let v7 = arena.alloc(LispValue::Number(7)).unwrap();
        let v8 = arena.alloc(LispValue::Number(8)).unwrap();

        assert_eq!(arena.used(), 5);

        // Verify the new allocations
        assert_eq!(arena.get(v6).unwrap(), LispValue::Number(6));
        assert_eq!(arena.get(v7).unwrap(), LispValue::Number(7));
        assert_eq!(arena.get(v8).unwrap(), LispValue::Number(8));

        // Original rooted values should still be there
        assert_eq!(arena.get(v1).unwrap(), LispValue::Number(1));
        assert_eq!(arena.get(v5).unwrap(), LispValue::Number(5));

        // Freed cells should not be accessible
        assert_eq!(arena.get(v2).unwrap_err(), ArenaError::InvalidIndex);
        assert_eq!(arena.get(v3).unwrap_err(), ArenaError::InvalidIndex);
        assert_eq!(arena.get(v4).unwrap_err(), ArenaError::InvalidIndex);
    }

    #[test]
    fn test_gc_deep_structure() {
        let arena: Arena<50> = Arena::new();

        // Create a deep binary tree
        // Level 3 (leaves)
        let l1 = arena.alloc(LispValue::Number(1)).unwrap();
        let l2 = arena.alloc(LispValue::Number(2)).unwrap();
        let l3 = arena.alloc(LispValue::Number(3)).unwrap();
        let l4 = arena.alloc(LispValue::Number(4)).unwrap();

        // Level 2
        let n1 = arena.alloc(LispValue::Cons(l1, l2)).unwrap();
        let n2 = arena.alloc(LispValue::Cons(l3, l4)).unwrap();

        // Level 1 (root)
        let root = arena.alloc(LispValue::Cons(n1, n2)).unwrap();

        // Add some garbage
        for i in 0..10 {
            arena.alloc(LispValue::Number(100 + i)).unwrap();
        }

        let used_before = arena.used();
        assert_eq!(used_before, 17); // 7 tree nodes + 10 garbage

        // Collect, rooting only the tree root
        arena.collect(&[root]);

        // All tree nodes should be preserved, garbage collected
        assert_eq!(arena.used(), 7);
        assert!(arena.is_allocated(root));
        assert!(arena.is_allocated(n1));
        assert!(arena.is_allocated(n2));
        assert!(arena.is_allocated(l1));
        assert!(arena.is_allocated(l2));
        assert!(arena.is_allocated(l3));
        assert!(arena.is_allocated(l4));
    }

    #[test]
    fn test_gc_partial_structure() {
        let arena: Arena<20> = Arena::new();

        // Create a structure where only part is rooted
        let a = arena.alloc(LispValue::Number(1)).unwrap();
        let b = arena.alloc(LispValue::Number(2)).unwrap();
        let cons_ab = arena.alloc(LispValue::Cons(a, b)).unwrap();

        let c = arena.alloc(LispValue::Number(3)).unwrap();
        let d = arena.alloc(LispValue::Number(4)).unwrap();
        let cons_cd = arena.alloc(LispValue::Cons(c, d)).unwrap();

        assert_eq!(arena.used(), 6);

        // Root only cons_ab, cons_cd should be collected
        arena.collect(&[cons_ab]);

        assert_eq!(arena.used(), 3);
        assert!(arena.is_allocated(cons_ab));
        assert!(arena.is_allocated(a));
        assert!(arena.is_allocated(b));
        assert!(!arena.is_allocated(cons_cd));
        assert!(!arena.is_allocated(c));
        assert!(!arena.is_allocated(d));
    }

    #[test]
    fn test_gc_multiple_collections() {
        let arena: Arena<10> = Arena::new();

        // First round
        let v1 = arena.alloc(LispValue::Number(1)).unwrap();
        let v2 = arena.alloc(LispValue::Number(2)).unwrap();
        arena.collect(&[v1]);
        assert_eq!(arena.used(), 1);

        // Second round - allocate more
        let v3 = arena.alloc(LispValue::Number(3)).unwrap();
        let v4 = arena.alloc(LispValue::Number(4)).unwrap();
        arena.collect(&[v1, v3]);
        assert_eq!(arena.used(), 2);

        // Third round
        let v5 = arena.alloc(LispValue::Number(5)).unwrap();
        arena.collect(&[v5]);
        assert_eq!(arena.used(), 1);
        assert!(arena.is_allocated(v5));
        assert!(!arena.is_allocated(v1));
        assert!(!arena.is_allocated(v3));
    }

    #[test]
    fn test_gc_preserves_all_rooted() {
        let arena: Arena<20> = Arena::new();

        // Allocate many values
        let mut roots = Vec::new();
        for i in 0..10 {
            roots.push(arena.alloc(LispValue::Number(i)).unwrap());
        }

        // Add garbage
        arena.alloc(LispValue::Number(999)).unwrap();
        arena.alloc(LispValue::Number(998)).unwrap();

        assert_eq!(arena.used(), 12);

        // Root all the values we created
        arena.collect(&roots);

        // All rooted values should remain
        assert_eq!(arena.used(), 10);
        for &root in &roots {
            assert!(arena.is_allocated(root));
        }
    }

    #[test]
    fn test_gc_empty_arena() {
        let arena: Arena<10> = Arena::new();

        // Collect on empty arena should not panic
        arena.collect(&[]);
        assert_eq!(arena.used(), 0);

        // Should still be able to allocate after
        let v = arena.alloc(LispValue::Number(42)).unwrap();
        assert_eq!(arena.get(v).unwrap(), LispValue::Number(42));
    }

    #[test]
    fn test_gc_shared_structure() {
        let arena: Arena<20> = Arena::new();

        // Create a shared leaf
        let shared = arena.alloc(LispValue::Number(999)).unwrap();

        // Two different cons cells pointing to the same shared value
        let cons1 = arena.alloc(LispValue::Cons(shared, shared)).unwrap();
        let cons2 = arena.alloc(LispValue::Cons(shared, shared)).unwrap();

        // Garbage
        let garbage = arena.alloc(LispValue::Number(123)).unwrap();

        assert_eq!(arena.used(), 4);

        // Root only cons1
        arena.collect(&[cons1]);

        // cons1 and shared should survive, cons2 and garbage should be collected
        assert_eq!(arena.used(), 2);
        assert!(arena.is_allocated(cons1));
        assert!(arena.is_allocated(shared));
        assert!(!arena.is_allocated(cons2));
        assert!(!arena.is_allocated(garbage));
    }
}
