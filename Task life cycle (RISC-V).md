#kernel #hermit-rs 

## 1. Birth - `spawn`

A task starts its life when it is spawned. The Hermit kernel allocates its
stacks - on RISC-V that's **three** regions: a kernel/interrupt stack, a
default stack, and the task's main ("user") stack, plus guard pages.

Because Hermit is a unikernel, spawning usually just means grabbing an
already in-memory function pointer and building a task around it - no ELF/
Mach-O loading, no segment mapping, no dynamic linking like a traditional OS.

`create_stack_frame()` crafts a `State` on the kernel stack:
- `ra = task_start`
- `a0 = func`, `a1 = arg`
- `a2 = task-stack top`

The new task is pushed to the **ready queue**, waiting for the next round of
scheduling.

## 2. Running

When the scheduler picks the task, `switch_to_task` restores the crafted
`State` and `ret`s into `task_start`, which switches `sp` to the task stack
and jumps to `task_entry → func(arg)`.

The task runs on the same core until control leaves it. Note Hermit is **not**
an always-awake watchdog over the task - it hands the core to the task and
lets it run. Control returns to kernel/scheduler code in one of two ways:

- **(A) Preemptive:** a hardware (one-shot) timer interrupt fires.
- **(B) Cooperative:** the task calls a blocking/yielding operation (mutex,
  `sleep`, `join`, explicit yield) which invokes `reschedule()` directly - no
  hardware interrupt needed.

## 3. Trap entry (preemptive path)

On interrupt, control transfers to `stvec`, i.e. `trap_entry` in the
**trapframe** crate. It:

1. Saves 34 slots on the kernel stack - all GPRs (`x1..x31`, `sp` handled
   specially), plus `sstatus` and `sepc`.
2. Sees `sstatus.SPP == 1` (trap came from S-mode) → takes the
   "from kernel" path.
3. Sets `ra = trap_return` (so the callee will return to `trap_return`), then jumps to `trap_handler` (implemented in the Hermit kernel).

`trap_handler` inspects the cause. For a timer interrupt it calls `timer_handler → scheduler()`. If the scheduler decides another task should run, it does bookkeeping (task status, FPU save, `current = new`) and calls `switch_to_task`, saving the current kernel context and switching to the new task's kernel stack.

## 4. Resume

When the task is picked again, it resumes **inside its own previous
`switch_to_task`** and unwinds back up the exact call chain:

switch_to_task → scheduler() → timer_handler → trap_handler
              → (ra) trap_return → sret

`trap_return` restores `sstatus`, `sepc`, and all general purpose registers, then executes `sret` to jump to user task.

> **`sret` does NOT switch to user mode here.** `sret` returns to whatever is
> in `sstatus.SPP`. Since the trap came from S-mode (`SPP == 1`), it returns
> to **Supervisor mode** - no privilege change. It also restores `SPIE→SIE`
> and jumps to `sepc`. (`sret` only drops to User mode when `SPP == 0`.)

The task then continues exactly where it was interrupted.

## 5. Termination - `exit`

The cycle continues until the task calls `exit`, which sets the task status to
`Finished` and reschedules. The task is **not** removed synchronously: the
scheduler later marks it `Invalid`, moves it to `finished_tasks`, and
`cleanup_tasks()` frees it in a later pass.


> **NOTE**: default Hermit unikernel mode on RISC-V. Tasks run entirely in
> Supervisor (S) mode - there is no user/kernel privilege boundary.
> (Real user mode only exists under the separate `common-os` feature.)

## Diagram

```plaintext
   TASK LIFE CYCLE (Hermit RISC-V, S-mode only)
   ============================================

  spawn(func, arg)
    | alloc 3 stacks (kernel + default + task) + guards
    | craft State: ra=task_start, a0=func,
    |              a1=arg, a2=task-stack top
    v
  +-----------+  push   +-------------+
  | new_tasks |-------->| ready_queue |
  +-----------+         +------+------+
                               | pick highest prio
                               v
                  switch_to_task(old, new)
                    restore State, `ret`
                               |
                               v
                         task_start
                    mv sp,a2 ; j task_entry
                               |
                               v  func(arg)
                  +------------------------+
                  |  RUNNING (Supervisor)  |<----+
                  +-----------+------------+      |
                              |                   | resumed
        +---------------------+--------------+    |  later
        |                     |              |    |
   (A) timer IRQ       (B) cooperative   (C) exit()
     (preemptive)        reschedule       /return
        |              (mutex/sleep/join)   |
        v                     |             v
  trap_entry (trap.S)         |       status=Finished
    save 34 regs              |       -> reschedule()
    (x1..x31,sstatus,sepc)    |
    SPP==1 (from S-mode)      |
    ra = trap_return          |
        |                     |
        v                     |
  trap_handler                |
    cause = Timer             |
        v                     |
  timer_handler --------------+
    set_oneshot_timer(None)
        v
  scheduler()  pick new task?
    yes: cur->Ready, push queue
         save FPU, current=new
        v
  switch_to_task(cur, new)
    save cur kernel ctx,
    switch sp to new stack
        |  (runs other task; this task
        |   resumes here when re-picked)
        v
  -- RESUME PATH ----------------------
   switch_to_task > scheduler >
   timer_handler > trap_handler
        v (ra)
   trap_return
     restore sstatus, sepc, GPRs
        v
   sret  -> SPP==1 => back to S-mode
          (NO privilege change)
        v
   continue where interrupted -> RUNNING

  -- TERMINATION (C) ------------------
   exit > Finished > scheduler marks
   Invalid > finished_tasks >
   cleanup_tasks() frees later
```
## Key Takeaways

- Everything runs in **Supervisor mode**; the "user stack" is just the task's main stack.
- The scheduler is reached via **either** a HW timer interrupt **or** a cooperative `reschedule()` - not only interrupts.
- A task always **resumes inside its own previous `switch_to_task`** and unwinds back up to `trap_return`/`sret`.
- `sret` returns to `sstatus.SPP`; here `SPP == 1`, so it stays in Supervisor mode.
Note: the whole thing above is inside one outer code fence for easy copying. When pasting into Obsidian, drop the outermost  ``markdown  fence and its closing  ``` ` so the headings render - keep the two inner fenced blocks (the resume chain and the diagram).