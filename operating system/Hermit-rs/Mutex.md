#kernel #mutex

## 1. Overview
Ordinary Rust (and C/C++, Python, ...) mutexes are implemented on top of OS
services - futexes on Linux, `SRWLOCK` on Windows, etc. Hermit is a
unikernel and has no underlying OS to call into, so it implements its own
mutex from scratch, in the `hermit-sync` crate. Hermit's kernel then uses
this crate's types throughout (e.g. for its physical/virtual memory
freelists, semaphores, futex, recursive mutex).

## 2. API Layer
- `hermit-sync` implements the `lock_api::RawMutex` trait, then uses
  `lock_api::Mutex<R, T>` / `lock_api::MutexGuard` to generate the standard
  lock/unlock/guard API - so consumer code uses it exactly like `std::sync::Mutex`,
  with no unsafe code required at call sites.

## 3. Inner Locking Mechanism - Ticket Lock

The core algorithm is a **ticket lock** (a fair spin lock), defined in
`hermit-sync`'s `RawTicketMutex`:

```rust
// hermit-sync-0.1.6/src/mutex/ticket.rs
pub struct RawTicketMutex {
    next_ticket: AtomicUsize,
    next_serving: AtomicUsize,
}

unsafe impl RawMutex for RawTicketMutex {
    fn lock(&self) {
        let ticket = self.next_ticket.fetch_add(1, Ordering::Relaxed);
        let mut backoff = Backoff::default();
        while self.next_serving.load(Ordering::Acquire) != ticket {
            backoff.relax();
        }
    }

    unsafe fn unlock(&self) {
        self.next_serving.fetch_add(1, Ordering::Release);
    }
}
````

- **Lock:** atomically take a ticket number (`next_ticket.fetch_add`), then spin until `next_serving` equals your ticket - this guarantees FIFO fairness (no thread can be starved by newer arrivals).

- **Unlock:** increment `next_serving`, which admits exactly the next waiting ticket holder.

### 3.1 Exponential backoff while spinning

While waiting, the thread doesn't just hammer the cache line - it backs off exponentially, capped at an upper bound (`spinning_top::relax::Backoff`):

```rust
// spinning_top-0.3.0/src/relax.rs
pub struct Backoff { step: u8 }
impl Backoff {
    const YIELD_LIMIT: u8 = 10;
}
impl Relax for Backoff {
    fn relax(&mut self) {
        for _ in 0..1_u16 << self.step {
            core::hint::spin_loop();
        }
        if self.step <= Self::YIELD_LIMIT {
            self.step += 1;
        }
    }
}
```

- Each retry spins `2^step` times before rechecking, and `step` stops growing once it hits `YIELD_LIMIT = 10` - bounding the maximum backoff.

## 4. Interrupt Safety Wrapper

On top of the ticket lock, `RawInterruptMutex<I>` disables interrupts for the duration of the critical section, saving/restoring the prior interrupt state (analogous to Linux's `spin_lock_irqsave`/`spin_unlock_irqrestore`):

```rust
// interrupt-mutex-0.1.0/src/lib.rs
pub struct RawInterruptMutex<I> {
    inner: I,
    interrupt_guard: UnsafeCell<MaybeUninit<interrupts::Guard>>,
}

fn lock(&self) {
    let guard = interrupts::disable();  // save + disable interrupts
    self.inner.lock();                  // then attempt the ticket lock
    self.interrupt_guard.get().write(MaybeUninit::new(guard));
}

unsafe fn unlock(&self) {
    let guard = self.interrupt_guard.get().replace(MaybeUninit::uninit());
    self.inner.unlock();
    drop(guard);   // dropping the guard restores prior interrupt state
}
```

- Interrupts are disabled **before** attempting to acquire the inner lock, and restored only after the inner lock is released - this prevents deadlock against an interrupt handler that would otherwise try to take the same lock on the same core.

## 5. Composed Type Used by Hermit Kernel

Hermit combines these two layers into `InterruptTicketMutex<T>`, which is what most of the kernel actually uses (e.g. the physical/virtual memory freelists):

```rust
pub type RawInterruptTicketMutex = RawInterruptMutex<RawTicketMutex>;
pub type InterruptTicketMutex<T> = lock_api::Mutex<RawInterruptTicketMutex, T>;
```

```rust
// kernel/src/mm/physicalmem.rs
static PHYSICAL_FREE_LIST: InterruptTicketMutex<FreeList<16>> =
    InterruptTicketMutex::new(FreeList::new());
```

- Note: `hermit-sync` also provides a plain `RawSpinMutex` (test-and-test-and-set spin lock, not ticket-based) and non-interrupt-safe `TicketMutex`/`SpinMutex` variants, for cases that don't need interrupt safety (e.g. `kernel/src/synch/recmutex.rs` uses plain `TicketMutex`). `InterruptTicketMutex` is simply the most common combination, not the only one.

## 6. Key File References

- `hermit-sync-0.1.6/src/mutex/ticket.rs` - `RawTicketMutex` (ticket lock algorithm)
- `hermit-sync-0.1.6/src/mutex/mod.rs` - type aliases combining raw mutexes
- `hermit-sync-0.1.6/src/lib.rs` - crate overview, public type table
- `interrupt-mutex-0.1.0/src/lib.rs` - `RawInterruptMutex` (interrupt disable/restore wrapper)
- `spinning_top-0.3.0/src/relax.rs` - `Backoff` (exponential backoff relax strategy)
- `kernel/src/mm/physicalmem.rs`, `kernel/src/mm/virtualmem.rs`, `kernel/src/synch/semaphore.rs`, `kernel/src/synch/futex.rs` - kernel usage sites