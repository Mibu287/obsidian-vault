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

# 6. `swtch`

Function signature
`void swtch(struct context* cpu_ctx, struct context* proc_ctx);`

The function is implemented in assembly code which 