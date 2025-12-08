use core::cell::RefCell;
use core::marker::PhantomData;
use core::ops::Deref;

// ============================================================================
// Arena Configuration
// ============================================================================

const VALUE_ARENA_SIZE: usize = 4096;
const ENV_ARENA_SIZE: usize = 512;

// ============================================================================
// Generic Slot Structure
// ============================================================================

struct Slot<T> {
    value: Option<T>,
    ref_count: usize,
}

impl<T> Slot<T> {
    const fn new() -> Self {
        Slot {
            value: None,
            ref_count: 0,
        }
    }
}

// ============================================================================
// Generic Arena
// ============================================================================

pub struct Arena<T, const N: usize> {
    slots: RefCell<[Slot<T>; N]>,
}

impl<T, const N: usize> Arena<T, N> {
    pub const fn new() -> Self {
        Arena {
            slots: RefCell::new([const { Slot::new() }; N]),
        }
    }

    fn allocate(&self, value: T) -> Option<usize> {
        let mut slots = self.slots.borrow_mut();
        for (idx, slot) in slots.iter_mut().enumerate() {
            if slot.value.is_none() {
                slot.value = Some(value);
                slot.ref_count = 1;
                return Some(idx);
            }
        }
        None
    }

    fn increment_ref(&self, idx: usize) {
        let mut slots = self.slots.borrow_mut();
        if idx < slots.len() {
            slots[idx].ref_count += 1;
        }
    }

    fn decrement_ref(&self, idx: usize) {
        let mut slots = self.slots.borrow_mut();
        if idx < slots.len() {
            slots[idx].ref_count = slots[idx].ref_count.saturating_sub(1);
            if slots[idx].ref_count == 0 {
                slots[idx].value = None;
            }
        }
    }

    fn get_ref<'a>(&'a self, idx: usize) -> Option<impl Deref<Target = T> + 'a> {
        let borrow = self.slots.borrow();
        if idx < borrow.len() && borrow[idx].value.is_some() {
            Some(OwningRef {
                _borrow: borrow,
                idx,
                _phantom: PhantomData,
            })
        } else {
            None
        }
    }
}

// ============================================================================
// OwningRef - Safely holds a Ref while providing access to inner value
// ============================================================================

struct OwningRef<'a, T, const N: usize> {
    _borrow: core::cell::Ref<'a, [Slot<T>; N]>,
    idx: usize,
    _phantom: PhantomData<&'a T>,
}

impl<'a, T, const N: usize> Deref for OwningRef<'a, T, N> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        // Safety: We hold a Ref, the index is valid (checked in get_ref),
        // and ref_count > 0 while this OwningRef exists
        let ptr = &self._borrow[self.idx].value as *const Option<T>;
        unsafe { (*ptr).as_ref().unwrap_unchecked() }
    }
}

// ============================================================================
// Rc - Reference Counted Pointer
// ============================================================================

pub struct Rc<T, const N: usize> {
    arena: &'static Arena<T, N>,
    index: usize,
    _phantom: PhantomData<T>,
}

impl<T, const N: usize> Rc<T, N> {
    pub fn new_in(arena: &'static Arena<T, N>, value: T) -> Option<Self> {
        arena.allocate(value).map(|index| Rc {
            arena,
            index,
            _phantom: PhantomData,
        })
    }

    pub fn strong_count(&self) -> usize {
        let slots = self.arena.slots.borrow();
        slots[self.index].ref_count
    }

    pub fn ptr_eq(this: &Self, other: &Self) -> bool {
        core::ptr::eq(this.arena, other.arena) && this.index == other.index
    }
}

impl<T, const N: usize> Clone for Rc<T, N> {
    fn clone(&self) -> Self {
        self.arena.increment_ref(self.index);
        Rc {
            arena: self.arena,
            index: self.index,
            _phantom: PhantomData,
        }
    }
}

impl<T, const N: usize> Drop for Rc<T, N> {
    fn drop(&mut self) {
        self.arena.decrement_ref(self.index);
    }
}

impl<T, const N: usize> Deref for Rc<T, N> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        // Get a temporary borrow and extract the pointer
        let slots = self.arena.slots.borrow();
        let ptr = slots[self.index].value.as_ref().unwrap() as *const T;
        drop(slots); // Explicitly drop the borrow
        // Safety: The value is kept alive by our ref_count > 0
        unsafe { &*ptr }
    }
}

impl<T: core::fmt::Debug, const N: usize> core::fmt::Debug for Rc<T, N> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let slots = self.arena.slots.borrow();
        if let Some(ref val) = slots[self.index].value {
            val.fmt(f)
        } else {
            write!(f, "<dropped>")
        }
    }
}

// ============================================================================
// Global Arenas for Lisp Interpreter
// ============================================================================

pub static VALUE_ARENA: Arena<crate::Value, VALUE_ARENA_SIZE> = Arena::new();
pub static ENV_ARENA: Arena<RefCell<crate::Env>, ENV_ARENA_SIZE> = Arena::new();

// ============================================================================
// Type Aliases for Drop-in Replacement
// ============================================================================

/// Drop-in replacement for alloc::rc::Rc<Value>
pub type ValueRc = Rc<crate::Value, VALUE_ARENA_SIZE>;

/// Drop-in replacement for alloc::rc::Rc<RefCell<Env>>
pub type EnvRc = Rc<RefCell<crate::Env>, ENV_ARENA_SIZE>;

// ============================================================================
// Convenience constructors that match alloc::rc::Rc::new
// ============================================================================

impl ValueRc {
    /// Creates a new Rc<Value> in the global arena
    /// This is a drop-in replacement for alloc::rc::Rc::new(value)
    pub fn new(value: crate::Value) -> Self {
        Self::new_in(&VALUE_ARENA, value)
            .expect("Value arena exhausted - increase VALUE_ARENA_SIZE")
    }
}

impl EnvRc {
    /// Creates a new Rc<RefCell<Env>> in the global arena
    /// This is a drop-in replacement for alloc::rc::Rc::new(value)
    pub fn new(value: RefCell<crate::Env>) -> Self {
        Self::new_in(&ENV_ARENA, value).expect("Env arena exhausted - increase ENV_ARENA_SIZE")
    }
}

// ============================================================================
// Example Integration
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_rc() {
        static TEST_ARENA: Arena<i32, 16> = Arena::new();

        let rc1 = Rc::new_in(&TEST_ARENA, 42).unwrap();
        assert_eq!(*rc1, 42);
        assert_eq!(rc1.strong_count(), 1);

        let rc2 = rc1.clone();
        assert_eq!(*rc2, 42);
        assert_eq!(rc1.strong_count(), 2);
        assert_eq!(rc2.strong_count(), 2);

        drop(rc1);
        assert_eq!(*rc2, 42);
        assert_eq!(rc2.strong_count(), 1);
    }

    #[test]
    fn test_ptr_eq() {
        static TEST_ARENA: Arena<i32, 16> = Arena::new();

        let rc1 = Rc::new_in(&TEST_ARENA, 100).unwrap();
        let rc2 = rc1.clone();
        let rc3 = Rc::new_in(&TEST_ARENA, 100).unwrap();

        assert!(Rc::ptr_eq(&rc1, &rc2));
        assert!(!Rc::ptr_eq(&rc1, &rc3));
    }

    #[test]
    fn test_arena_reuse() {
        static TEST_ARENA: Arena<i32, 2> = Arena::new();

        let rc1 = Rc::new_in(&TEST_ARENA, 1).unwrap();
        let rc2 = Rc::new_in(&TEST_ARENA, 2).unwrap();

        // Arena is full
        assert!(Rc::<i32, 2>::new_in(&TEST_ARENA, 3).is_none());

        drop(rc1); // Free up a slot

        // Now we can allocate again
        let rc3 = Rc::new_in(&TEST_ARENA, 3).unwrap();
        assert_eq!(*rc3, 3);
        assert_eq!(*rc2, 2);
    }
}
