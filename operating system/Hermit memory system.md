# Hermit OS Memory Management (Boot-Time, riscv64)

## Overview
Hermit's kernel bootstraps physical and virtual memory tracking using a free-list
allocator, before its own global heap allocator exists. The early boot path is
carefully designed to avoid ever triggering a heap allocation, since none is
available yet. This note covers the riscv64 target specifically.

## Free List Data Structure
- Implemented via the external `free-list` crate (`FreeList<16>`).
- Used for both physical memory (`kernel/src/mm/physicalmem.rs`) and the
  kernel's virtual address range (`kernel/src/mm/virtualmem.rs`).
- Internally backed by a `SmallVec<[PageRange; 16]>` — holds up to **16**
  disjoint free ranges inline before it would need to grow onto the heap.
- `allocate()` finds the first range that fits (`PageRange::fit`), splitting
  a chunk to carve out a region of the requested size/alignment.

## Boot-Time Initialization

### Physical memory freelist
- Populated from FDT `/memory` regions (`physicalmem.rs::detect_from_fdt`),
  producing several disjoint free chunks.
- Reserved regions are then carved out: FDT memory-reservations, the kernel
  image region, and the FDT blob region itself.

### Virtual memory freelist (kernel address range)
- riscv64 kernel heap end constant: `0x0040_0000_0000 - 1` = 256 GiB − 1.
- Freelist seeded as `[kernel_heap_end/2 .. kernel_heap_end+1]`
  → **128 GiB – 256 GiB**, matching the "max virtual memory 256 GB" figure.
- Note: this freelist covers only the *kernel's own* virtual address range,
  not all of virtual memory — user-space task memory is a separate region.

## The Chicken-and-Egg Problem
- The global allocator (`ALLOCATOR: Talck`) isn't claimed until the very end
  of `mm::init()` — nothing can use `Box`/`Vec` before that point.
- Danger: allocating/freeing physical or virtual ranges fragments the
  freelist. If fragmentation ever exceeds the 16-slot inline `SmallVec`
  capacity, the freelist itself would need to grow — which means calling the
  (not-yet-ready) global allocator.
- The real safety margin is that 16-slot inline capacity, not "it's a Vec so
  it might reallocate" in the abstract — early boot just needs to keep
  fragmentation low enough to stay inline.

## Huge Pages at Startup (riscv64 specifics)
- riscv64 uses **Sv39** paging (3 levels): root = `L2Table`, then `L1Table`,
  then `L0Table`.
- Hermit's `HugePageSize` = 1 GiB gigapage, with `MAP_LEVEL == 2` — the same
  level as the root table itself.
- Because the huge page's map level equals the root table's level, mapping a
  huge page writes a PTE **directly into the pre-existing root table**, with
  no intermediate table allocation at all.
- Contrast with `LargePageSize` (2 MiB) and `BasePageSize` (4 KiB): both have
  `MAP_LEVEL < 2`, so mapping with them requires allocating intermediate
  L1/L0 tables via `FrameAlloc::allocate` — which pulls from the physical
  freelist and risks the fragmentation problem above.
- This is why Hermit maps the heap region with `HugePageSize` first at boot:
  it is the *only* mapping granularity that requires **zero** additional
  physical frame allocations, fully sidestepping the freelist-growth risk
  before the allocator exists.
- Boot call site: `paging::map_heap::<HugePageSize>(...)` in `mm::init()`,
  gated on `has_1gib_pages` (always `true` on riscv64).

## Heap Setup & Allocator Handoff
1. Virtual heap range is allocated from the kernel virtual freelist
   (`PageAlloc`), removing it from the free list.
2. The range is mapped using a fallback cascade: `HugePageSize` (1 GiB) →
   `LargePageSize` (2 MiB) → `BasePageSize` (4 KiB) — huge pages preferred
   for the zero-allocation property above.
3. Once mapped, `ALLOCATOR.lock().claim(arena)` is called — this is the
   exact moment dynamic allocation (`Box`, `Vec`, etc.) becomes available.

## Key File References
- `kernel/src/mm/physicalmem.rs` — physical freelist, FDT region detection
- `kernel/src/mm/virtualmem.rs` — kernel virtual freelist, heap-end constants
- `kernel/src/mm/mod.rs` — `init()` sequence, heap mapping cascade, allocator claim
- `kernel/src/arch/riscv64/mm/paging.rs` — Sv39 levels, `HugePageSize` = 1 GiB,
  `map_page` allocation branching (line ~412-451)
- `kernel/src/arch/riscv64/kernel/processor.rs` — `has_1gib_pages`
