#kernel #RISCV

# 1. After booted

All CPU Cores goto a fixed address and execute the same code.

# 2. `_entry`

- Setup stack for these "hardware thread" 
- These stack is defined in memory of the kernel object file data.    

> **Note:** hardware thread is not user thread (which is manage by the kernel). Hardware thread is just execution of codes.

- jump to `start`.
    
# 3. `start`

- All CPU cores continue to execute the `start` function in machine privileges. 
- It setup the CPU core. 
- It use hardware stack (setup in `_entry`) to store local variable. 
     => Each CPU always use its own hardware stack when control is returned to hardware thread.  
- After all, `mret` is call to downgrade privilege to supervisor and control is jumped to `main`. 

# 4. `main`

The function is still executed by all CPU cores. However, common work is done by only CPU 0. It still use hardware stack to store/read local variable.

**List of works done by `main`:**

- Initialize physical page allocator
- Create kernel page table 
- Turn on paging (virtual address mode) 
- Initialize process table 
- Initialize file system / disks 
- Create 1st user process (init proc). 

> **Note:** All of memory page is identity-mapped in kernel-page table. => The kernel can switch on/off paging with no problem. trampoline page is mapped with 2 virtual addresses.

- Finally, `main` call scheduler and never return.

# 5. `scheduler`

- All CPU cores call `scheduler` after finish setting up.   
- The function is executed in supervisor mode using hardware stack. 
- The function loop infinitely to check for runnable process to run. 
- If none available, it goto sleep waiting for interrupt. 
- If a runnable thread is found, the scheduler function call `swtch` to switch to the process context and run.

# 6. `swtch`

Function signature: `void swtch(struct context* old_ctx, struct context* new_ctx);`

The function is implemented in assembly code which:

- Saves `ra`, `sp`, and callee-saved registers (`s0`-`s11`) of the **old** context to `old_ctx`.
- Loads `ra`, `sp`, and callee-saved registers from `new_ctx`.
- Executes `ret` → jumps to the loaded `ra` of the new context.

> **Note:** `swtch` only saves callee-saved registers because it is called as a normal C function. The caller already follows calling convention and has saved any caller-saved registers it cares about. This is in contrast to `uservec` (see below), which must save **all** registers.

# 7. User Process Runs

After `scheduler` calls `swtch`, the CPU is now running the user process's **kernel thread**. Eventually the kernel returns to user space via `userret` + `sret` (see section 10), and the user process executes its instructions in user privilege.

# 8. Interrupt from User Process → `uservec`

When a user process is interrupted (timer, syscall, page fault, etc.):

- Hardware automatically switches privilege to supervisor mode.
- `stvec` register (set to point to `uservec` in trampoline) takes control.
- `uservec` runs in supervisor mode **but still using the user page table**.

**What `uservec` does:**

- Saves **all** user registers (`ra`, `sp`, `gp`, `tp`, `t0`-`t6`, `a0`-`a7`, `s0`-`s11`) to the process's `trapframe`.

> **Note:** All registers must be saved here (not just callee-saved) because the interrupt does not follow the C calling convention — any register could have been in use at the moment of interruption.

- Loads kernel stack pointer, kernel page table, and kernel `hartid` from the trap frame (values were stored there before returning to user last time).
- Switches to the kernel page table.
- Calls `usertrap`.

# 9. `usertrap`

Now executing in full kernel context (kernel page table, kernel stack):

- Switches `stvec` to `kernelvec` (trap handler for kernel-mode traps).
- Saves `sepc` (user program counter) to `trapframe->epc`. 
- Identifies the cause of the trap (`scause`): 
    - `8` → system call → calls `syscall()`
    - timer interrupt (`which_dev == 2`) → calls `yield()`
    - page fault → calls `vmfault()`
    - otherwise → marks process as killed
- After handling, calls `prepare_return()` which sets up `sstatus`, `sepc`, and `stvec` for the return to user space. 
- Returns `satp` (user page table address) → this return lands in `userret` (see below).
 
# 10. `yield` + `sched` → Back to Scheduler

For a timer interrupt, `usertrap` calls `yield`:

```
usertrap()
  → yield()         # acquires p->lock, sets state = RUNNABLE
    → sched()       # validates kernel invariants
      → swtch(&p->context, &cpu->context)
                    # saves process's kernel context (ra points back into sched)
                    # loads scheduler's context
                    # ret → jumps into scheduler()
```

Control is now back in `scheduler()`, right after its own `swtch` call. The scheduler loop continues looking for the next runnable process.

# 11. Returning to User Space

When the scheduler picks a process and calls `swtch(&cpu->context, &p->context)`, the destination depends on whether this is the process's first scheduling:

**First time the process is scheduled** (`p->context.ra = forkret`, set by `allocproc`):

```
scheduler()
  → swtch lands in forkret()     # does first-time setup (e.g. filesystem init for init proc)
    → usertrapret() / prepare_return()
      → userret (trampoline.S)
```

**Every subsequent scheduling** (`p->context.ra` = saved return address inside `sched()`):

```
scheduler()
  → swtch lands inside sched()   # resumes where the process last called swtch
    → sched() returns
      → yield() releases p->lock and returns
        → usertrap() resumes after yield() call
          → prepare_return()
          → usertrap() returns satp
            → userret (trampoline.S)
```

> **Note:** `userret` is reached via the normal C `return` from `usertrap`. This works because `uservec` called `usertrap` with `jalr t0`, which stored the address of `userret` (the very next label in trampoline.S) into `ra`.

**What `userret` does:**

- Switches back to the user page table (using the returned `satp`).
- Restores **all** registers from `trapframe`.
- Executes `sret` → privilege drops to user mode, PC jumps to `trapframe->epc` (the user's next instruction).

The full round-trip for a timer interrupt (after first scheduling) looks like:

```
user process
  → [timer interrupt] → uservec → usertrap → yield → sched → swtch
                                                                 ↓
                                                          scheduler loop
                                                                 ↓
                                              swtch → sched → yield → usertrap → prepare_return → userret → sret
                                                                                                               ↓
                                                                     
```