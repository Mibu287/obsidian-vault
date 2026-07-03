#hermit-rs #kernel 

## 1. Overview

Hermit's kernel bootstraps physical and virtual memory tracking using a free-list
allocator, before its own global heap allocator exists. The early boot path is
carefully designed to avoid ever triggering a heap allocation, since none is
available yet. This note covers the RISCV64 target specifically.

## 2. Free List Data Structure

- Implemented via the external `free-list` crate.
- Used for both physical memory (`kernel/src/mm/physicalmem.rs`) and the
  kernel's virtual address range (`kernel/src/mm/virtualmem.rs`).
- Internally backed by a `SmallVec<[PageRange; 16]>` - holds up to **16**
  disjoint free ranges inline before it would need to grow onto the heap.
- `allocate()` finds the first range that fits (`PageRange::fit`), splitting
  a chunk to carve out a region of the requested size/alignment.

## 3. Boot-Time Initialization

### 3.1 Physical memory freelist

- Populated from FDT `/memory` regions (`physicalmem.rs::detect_from_fdt`),
  producing several disjoint free chunks.
- Reserved regions are then carved out: FDT memory-reservations, the kernel
  image region, and the FDT blob region itself.

### 3.2 Virtual memory freelist (kernel address range)

- RISCV64 kernel heap end constant, and the range handed to the freelist
  (`kernel/src/mm/virtualmem.rs:52-74`):

```rust
pub fn kernel_heap_end() -> VirtAddr {
    // RISCV64: 256 GiB
    VirtAddr::new(0x0040_0000_0000 - 1)
}

unsafe fn init() {
    let range = PageRange::new(
        kernel_heap_end().as_usize().div_ceil(2),
        kernel_heap_end().as_usize() + 1,
    ).unwrap();
    unsafe { PageAlloc::deallocate(range); }
}
````

- Result: freelist seeded with **128 GiB – 256 GiB**.
- Note: this freelist covers only the _kernel's own_ virtual address range, not all of virtual memory - user-space task memory is a separate region.

## 4. The Chicken-and-Egg Problem

- The global allocator (`ALLOCATOR: Talck`) isn't claimed until the very end of `mm::init()` - nothing can use `Box`/`Vec` before that point.
- Danger: allocating/freeing physical or virtual ranges fragments the freelist. If fragmentation ever exceeds the 16-slot inline `SmallVec` capacity, the freelist itself would need to grow - which means calling the (not-yet-ready) global allocator.
- The real safety margin is that 16-slot inline capacity, not "it's a Vec so it might reallocate" in the abstract - early boot just needs to keep fragmentation low enough to stay inline.

## 5. Huge Pages at Startup (RISCV64 specifics)

- RISCV64 uses SV39 paging (3 levels): root = `L2Table`, then `L1Table`, then `L0Table`. `HugePageSize` = 1 GiB, with `MAP_LEVEL == 2` - same level as the root table.
- Mapping logic branches on whether the page size's level matches the current table's level (`kernel/src/arch/RISCV64/mm/paging.rs:412-451`):

```rust
fn map_page<S: PageSize>(&mut self, page: Page<S>, ...) {
    if L::LEVEL > S::MAP_LEVEL {
        // subtable doesn't exist yet -> allocate a frame from the freelist
        if !self.entries[index].is_present() {
            let frame_range = FrameAlloc::allocate(frame_layout).unwrap();
            self.entries[index].set(new_entry, PageTableEntryFlags::BLANK);
        }
        subtable.map_page::<S>(page, physical_address, flags);
    } else {
        // L::LEVEL == S::MAP_LEVEL: write directly into this table, no allocation
        self.map_page_in_this_table::<S>(page, physical_address, flags);
    }
}
```

- For `HugePageSize`, `L::LEVEL == S::MAP_LEVEL` at the root table, so it takes the `else` branch - **zero** frame allocations. `LargePageSize` (2 MiB) and `BasePageSize` (4 KiB) take the allocating branch instead.
- This is why the heap-mapping cascade in `mm::init()` tries huge pages first (`kernel/src/mm/mod.rs:233-248`):

## 6. Heap Setup & Allocator Handoff

1. Virtual heap range is allocated from the kernel virtual freelist (`PageAlloc`), removing it from the free list.
2. The range is mapped via the huge → large → base fallback cascade above.
3. Once mapped, the arena is claimed by the global allocator (`kernel/src/mm/mod.rs:281-284`):

```rust
let arena = Span::new(heap_start_addr.as_mut_ptr(), heap_end_addr.as_mut_ptr());
unsafe { ALLOCATOR.lock().claim(arena).unwrap(); }
```

This is the exact moment dynamic allocation (`Box`, `Vec`, etc.) becomes available.

## 7. Virtual Memory Translation (Sv39 Enable & Runtime)

### 7.1. Boot starts without translation

- Full chain on RISCV64: hardware reset → OpenSBI (M-mode firmware) →
  **hermit-loader** (a separate project, S-mode, not part of this repo) →
  Hermit kernel `_start` (`kernel/src/arch/RISCV64/kernel/start.rs`).
- OpenSBI explicitly zeroes `satp` (Bare mode) during its own hart-init
  routine - this is a deliberate write, not just reliance on a reset
  default - before handing off to the loader in S-mode.
- Hermit's own `_start`/`pre_init` contains no MMU/`satp` code at all; it
  implicitly assumes Bare mode was left in place by the loader/OpenSBI. The
  kernel only touches `satp` once, later, in its own `enable_page_table()`.
- So: everything before that single call - device tree parsing, physical
  frame allocator init - runs with loads/stores going directly to physical
  memory, no translation.

### 7.2. Building the page table before translation is on

- Regions (e.g. PLIC, UART, virtio devices found via the device tree) are
  identity-mapped with `HugePageSize` *before* `crate::mm::init()` even runs
  the physical frame allocator init - only possible because, as established
  in §5, huge pages need none of dynamic memory allocation:

- `identity_map` builds the virtual address directly from the physical one (`kernel/src/arch/RISCV64/mm/paging.rs:626-638`) - `virt == phys`.

### 7.3. Turning translation on

- A single call writes the root table's physical page number into `satp` with `Mode::Sv39`, then flushes the TLB globally (`kernel/src/arch/RISCV64/mm/paging.rs:640-653`):

```rust
fn enable_page_table() {
    satp::set(satp::Mode::Sv39, asid, ppn); // this write IS what enables translation
    asm::sfence_vma_all();                  // safety flush, not a separate "enable" step
}
```

- Because the switched-on mapping is identity (`virt == phys`) for the memory the kernel is currently executing from, control flow is unaffected by the switch, i.e. no pointer fix up needed.

### 7.4. Physical ↔ virtual cardinality

- Default build: strictly **1:1** - every mapped physical page has exactly one identity-mapped virtual address.
- The page table code does support one physical range being mapped at a _second_ virtual address if needed

### 7.5. TLB maintenance after enabling

- Once translation is live, the CPU caches translations (TLB) during page table walks. When the kernel overwrites an already-present PTE, it must explicitly flush that entry:

```rust
// only needed when overwriting an existing (already-present) entry
page.flush_from_tlb(); // emits sfence.vma for that page
```

- New (previously-not-present) entries need no flush - only modifications to existing translations require it.

## 8. Key File References

- `kernel/src/mm/physicalmem.rs` - physical freelist, FDT region detection
- `kernel/src/mm/virtualmem.rs` - kernel virtual freelist, heap-end constants
- `kernel/src/mm/mod.rs` - `init()` sequence, heap mapping cascade, allocator claim
- `kernel/src/arch/RISCV64/mm/paging.rs` - Sv39 levels, `HugePageSize` = 1 GiB, `map_page` allocation branching (lines 412-451)
- `kernel/src/arch/RISCV64/kernel/processor.rs` - `has_1gib_pages`