// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only
//
// Unit tests of the `ps5hle::ampr` module.

use super::command_buffer as cbuf;
use super::*;
use crate::context::CpuRegister::*;
use crate::registry::HleCallback;
use crate::sync::equeue;
use std::sync::atomic::{AtomicU64, Ordering};

const BASE: u64 = 0x5C63_0000_0000;
const MAP: u64 = 0x10000;

/// Per-test guest window (tests run in parallel).
fn window(tag: u64) -> u64 {
    BASE + tag * MAP
}

fn with_vm(tag: u64, f: impl FnOnce(&mut CpuContext, u64)) {
    let vm = ps5vm::VirtualMemory::new();
    let base = window(tag);
    vm.map(base, MAP, ps5vm::MemProtect::RW).unwrap();
    let mut ctx = CpuContext::new(&vm);
    f(&mut ctx, base);
    let _ = vm.unmap(base, MAP);
}

/// Call an export with integer arguments; arguments past the sixth go to
/// the stack (`stack` is the guest rsp, its return slot left free).
fn call(ctx: &mut CpuContext, stack: u64, f: HleCallback, args: &[u64]) -> HleResult {
    for (r, v) in [Rdi, Rsi, Rdx, Rcx, R8, R9].iter().zip(args) {
        ctx.set(*r, *v);
    }
    for (i, v) in args.iter().skip(6).enumerate() {
        ctx.write_u64(stack + 8 + i as u64 * 8, *v);
    }
    ctx.set(Rsp, stack);
    f(ctx)
}

const EINVAL: HleResult = ERR_INVALID_ARGUMENT;

#[test]
fn nids_match_names() {
    let mut seen = std::collections::HashSet::new();
    for &(nid, name, _) in EXPORTS.iter().chain(KERNEL_EXPORTS) {
        assert_eq!(ps5nid::compute_nid(name), nid, "NID mismatch for {name}");
        assert!(seen.insert(nid), "duplicate NID {nid}");
    }
}

#[test]
fn buffer_management() {
    with_vm(0, |ctx, base| {
        let (cb, buf, stack) = (base, base + 0x100, base + 0x8000);
        assert_eq!(call(ctx, stack, cb_constructor, &[0]), OK);
        ctx.write_bytes(cb, &[0xFF; 0x28]);
        assert_eq!(call(ctx, stack, cb_constructor, &[cb]), OK);
        assert_eq!(ctx.read_bytes(cb, 0x18).unwrap(), vec![0; 0x18]);
        assert_eq!(ctx.read_u64(cb + 0x18), Some(u64::MAX));
        assert_eq!(call(ctx, stack, cb_apr_constructor, &[cb, 0, 0]), OK);
        assert_eq!(ctx.read_u64(cb + 0x18), Some(0));
        assert_eq!(ctx.read_u64(cb + 0x20), Some(0));

        // SetBuffer: argument checks, then EBUSY while a buffer is attached.
        assert_eq!(call(ctx, stack, cb_set_buffer, &[0, buf, 0x400]), EINVAL);
        assert_eq!(call(ctx, stack, cb_set_buffer, &[cb, 0, 0x400]), EINVAL);
        assert_eq!(call(ctx, stack, cb_set_buffer, &[cb, buf + 2, 0x400]), EINVAL);
        assert_eq!(call(ctx, stack, cb_set_buffer, &[cb, buf, 0]), EINVAL);
        assert_eq!(call(ctx, stack, cb_set_buffer, &[cb, buf, 0x402]), EINVAL);
        assert_eq!(call(ctx, stack, cb_set_buffer, &[cb, buf, cbuf::BUFFER_SIZE_MAX + 4]), EINVAL);
        assert_eq!(call(ctx, stack, cb_set_buffer, &[cb, buf, 0x400]), OK);
        assert_eq!(call(ctx, stack, cb_set_buffer, &[cb, buf, 0x400]), ERR_BUSY);
        assert_eq!(call(ctx, stack, cb_get_size, &[cb]), 0x400);
        assert_eq!(call(ctx, stack, cb_get_buffer_base_address, &[cb]), buf as i64);

        // Nops: measured sizes match what the writers append.
        assert_eq!(call(ctx, stack, cb_nop, &[cb, 0]), EINVAL);
        assert_eq!(call(ctx, stack, cb_nop, &[cb, 17]), EINVAL);
        assert_eq!(call(ctx, stack, cb_nop, &[cb, 3]), OK);
        assert_eq!(call(ctx, stack, cb_measure_nop, &[3]), 12);
        assert_eq!(call(ctx, stack, cb_measure_nop, &[0]), 4);
        assert_eq!(call(ctx, stack, cb_nop_with_data, &[cb, 16, 0]), EINVAL);
        assert_eq!(call(ctx, stack, cb_nop_with_data, &[cb, 2, 0]), OK);
        assert_eq!(call(ctx, stack, cb_measure_nop_with_data, &[2, 0]), 12);
        ctx.write_bytes(base + 0x7000, b"label\0");
        assert_eq!(call(ctx, stack, cb_measure_marker, &[base + 0x7000]), 12);
        assert_eq!(call(ctx, stack, cb_measure_marker_with_color, &[base + 0x7000, 1]), 16);
        assert_eq!(call(ctx, stack, cb_measure_marker, &[0]), 4);
        assert_eq!(call(ctx, stack, cb_marker_with_color, &[cb, base + 0x7000, 1]), OK);
        assert_eq!(call(ctx, stack, cb_pop_marker, &[cb]), OK);
        assert_eq!(call(ctx, stack, cb_construct_nop, &[cb, 0, 0, 5, 0]), OK);
        assert_eq!(call(ctx, stack, cb_get_current_offset, &[cb]), 12 + 12 + 16 + 4 + 12);
        assert_eq!(call(ctx, stack, cb_get_num_commands, &[cb]), 5);

        // Map brackets keep the map-active type bit.
        assert_eq!(call(ctx, stack, cb_apr_map_end, &[cb]), ERR_PERMISSION_DENIED);
        assert_eq!(call(ctx, stack, cb_measure_apr_map_begin, &[0x4000, 0x4000, 0, 0]), 0x0C);
        assert_eq!(call(ctx, stack, cb_measure_apr_map_begin, &[0x4000, 0x2000, 0, 0]), EINVAL);
        assert_eq!(call(ctx, stack, cb_apr_map_direct_begin, &[cb, 0x4000, 0x2000, 0x4000, 0, 0]), EINVAL);
        assert_eq!(call(ctx, stack, cb_apr_map_begin, &[cb, 0x4000, 0x4000, 0, 0]), OK);
        assert_eq!(call(ctx, stack, cb_get_type, &[cb]) as u32 & cbuf::TYPE_MAP_ACTIVE, cbuf::TYPE_MAP_ACTIVE);
        assert_eq!(call(ctx, stack, cb_apr_map_end, &[cb]), OK);
        assert_eq!(call(ctx, stack, cb_get_type, &[cb]), 0);

        // Reset rewinds; ClearBuffer detaches and returns the buffer.
        assert_eq!(call(ctx, stack, cb_reset, &[0]), ERR_PERMISSION_DENIED);
        assert_eq!(call(ctx, stack, cb_reset, &[cb]), OK);
        assert_eq!(call(ctx, stack, cb_get_current_offset, &[cb]), 0);
        assert_eq!(call(ctx, stack, cb_get_num_commands, &[cb]), 0);
        assert_eq!(call(ctx, stack, cb_clear_buffer, &[cb]), buf as i64);
        assert_eq!(call(ctx, stack, cb_clear_buffer, &[cb]), 0);
        assert_eq!(call(ctx, stack, cb_reset, &[cb]), ERR_PERMISSION_DENIED);
        assert_eq!(call(ctx, stack, cb_get_size, &[0]), 0);
        assert_eq!(call(ctx, stack, cb_destructor, &[cb]), OK);
    });
}

#[test]
fn read_records_and_gather_scatter() {
    with_vm(1, |ctx, base| {
        let (cb, buf, stack, dst) = (base, base + 0x100, base + 0x8000, base + 0x4000);
        assert_eq!(call(ctx, stack, cb_constructor, &[cb]), OK);
        assert_eq!(call(ctx, stack, cb_set_buffer, &[cb, buf, 0x400]), OK);

        // Continuations need a previous read.
        assert_eq!(call(ctx, stack, cb_apr_read_file_gather, &[cb, 0, 0, 0x10, 0]), EINVAL);
        assert_eq!(call(ctx, stack, cb_apr_read_file_scatter, &[cb, 0, 0, dst, 0x10]), EINVAL);
        // Offsets at or past 2^40 (including -1) and empty reads are invalid.
        assert_eq!(call(ctx, stack, cb_apr_read_file, &[cb, 0, 0, 7, dst, 0x10, u64::MAX]), EINVAL);
        assert_eq!(call(ctx, stack, cb_apr_read_file, &[cb, 0, 0, 7, dst, 0, 0]), EINVAL);
        assert_eq!(call(ctx, stack, cb_measure_read_file, &[0, dst, 0x10, 1 << 40]), MEASURE_INVALID);

        assert_eq!(call(ctx, stack, cb_apr_read_file, &[cb, 0, 0, 7, dst, 0x10, 0x20]), OK);
        assert_eq!(ctx.read_bytes(buf, 0x14).unwrap()[0], cbuf::OP_READ_FILE);
        assert_eq!(call(ctx, stack, cb_measure_read_file, &[0, dst, 0x10, 0x20]), 0x14);
        assert_eq!(call(ctx, stack, cb_measure_read_file, &[0, dst, 0x10, 1 << 32]), 0x18);
        assert_eq!(call(ctx, stack, cb_get_type, &[cb]) as u32, cbuf::TYPE_GATHER_SCATTER_VALID);
        assert_eq!(call(ctx, stack, cb_apr_read_file_gather, &[cb, 0, 0, 0x8, 0x40000]), OK);
        assert_eq!(call(ctx, stack, cb_measure_read_file_gather, &[0x8, 0x40000]), 0x0C);
        assert_eq!(call(ctx, stack, cb_apr_read_file_scatter, &[cb, 0, 0, dst + 0x100, 0x4]), OK);
        assert_eq!(call(ctx, stack, cb_apr_read_file_gather_scatter, &[cb, 0, 0, dst + 0x200, 0x4, 0]), OK);
        let st = cbuf::try_get_state(ctx, cb).unwrap();
        let reads: Vec<_> = st
            .commands
            .iter()
            .map(|c| match c.data {
                CommandData::ReadFile { file_id, destination, size, file_offset } => {
                    (c.record_offset, file_id, destination, size, file_offset)
                }
                other => panic!("unexpected {other:?}"),
            })
            .collect();
        assert_eq!(
            reads,
            vec![
                (0x00, 7, dst, 0x10, 0x20),
                (0x14, 7, dst + 0x10, 0x8, 0x40000),
                (0x20, 7, dst + 0x100, 0x4, 0x40008),
                (0x2C, 7, dst + 0x200, 0x4, 0),
            ]
        );
        assert_eq!(call(ctx, stack, cb_apr_reset_gather_scatter_state, &[cb, 0, 0]), OK);
        assert_eq!(call(ctx, stack, cb_get_type, &[cb]), 0);
        assert_eq!(call(ctx, stack, cb_apr_read_file_scatter, &[cb, 0, 0, dst, 0x10]), EINVAL);
        assert_eq!(call(ctx, stack, cb_get_current_offset, &[cb]), 0x40);

        // A full buffer rejects further records without moving.
        assert_eq!(call(ctx, stack, cb_reset, &[cb]), OK);
        for _ in 0..0x400 / 0x40 {
            assert_eq!(call(ctx, stack, cb_nop, &[cb, 16]), OK);
        }
        assert_eq!(call(ctx, stack, cb_nop, &[cb, 1]), ERR_MEMORY_FAULT);
        assert_eq!(call(ctx, stack, cb_write_address, &[cb, dst, 1]), ERR_BUSY);
        assert_eq!(call(ctx, stack, cb_get_current_offset, &[cb]), 0x400);
        assert_eq!(call(ctx, stack, cb_destructor, &[cb]), OK);
    });
}

/// Guest addresses of the ordering scenario.
struct Ordering4 {
    headers: [u64; 4],
    data: [u64; 4],
    stack: u64,
    vars: u64,
    output: u64,
    results: u64,
    ids: u64,
}

impl Ordering4 {
    fn new(base: u64) -> Self {
        let headers = [0, 1, 2, 3].map(|i| base + i * 0x800);
        Self {
            headers,
            data: headers.map(|h| h + 0x100),
            stack: base + 0x8000,
            vars: base + 0x9000,
            output: base + 0x9800,
            results: base + 0x9900,
            ids: base + 0x9A00,
        }
    }

    fn var(&self, i: u64) -> u64 {
        self.vars + i * 8
    }

    /// Construct every buffer: the first `apr_count` as APR buffers (state
    /// words in the object), the rest as AMM buffers.
    fn reset(&self, ctx: &mut CpuContext, apr_count: usize) {
        for (i, &h) in self.headers.iter().enumerate() {
            assert_eq!(call(ctx, self.stack, cb_constructor, &[h]), OK);
            if i < apr_count {
                assert_eq!(call(ctx, self.stack, cb_apr_constructor, &[h, h + 0x18, h + 0x20]), OK);
            } else {
                assert_eq!(call(ctx, self.stack, cb_noop_destructor, &[h]), OK);
            }
            assert_eq!(call(ctx, self.stack, cb_set_buffer, &[h, self.data[i], 0x400]), OK);
        }
    }

    fn offset(&self, ctx: &mut CpuContext, i: usize) -> u64 {
        call(ctx, self.stack, cb_get_current_offset, &[self.headers[i]]) as u64
    }
}

fn load(ctx: &CpuContext, addr: u64) -> u64 {
    // SAFETY: test addresses are aligned and mapped.
    let _ = ctx;
    unsafe { AtomicU64::from_ptr(addr as *mut u64) }.load(Ordering::Acquire)
}

/// Wait conditions block their submission queue, not the engine; later
/// submissions (APR or AMM, address fences or shared counters) release
/// them; reads and completion events run only once their dependency is met.
#[test]
fn submission_ordering_counters_and_events() {
    let root = crate::filesystem::tests::temp_root("ampr-ordering");
    std::fs::write(root.join("ampr.dat"), b"APR").unwrap();
    let _app0 = crate::filesystem::APP0_TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    crate::filesystem::set_app0_root(&root);
    let file_id = crate::filesystem::apr_resolve_path("/app0/ampr.dat").expect("resolve fixture");

    with_vm(2, |ctx, base| {
        let o = Ordering4::new(base);
        let s = o.stack;
        let read = |ctx: &mut CpuContext, i: usize| {
            let h = o.headers[i];
            call(ctx, s, cb_apr_read_file, &[h, h + 0x18, h + 0x20, file_id as u64, o.output, 3, 0])
        };
        let result = |ctx: &CpuContext, i: u64| (ctx.read_u32(o.results + i * 8).unwrap() as i32, ctx.read_u32(o.results + i * 8 + 4).unwrap());
        let id = |ctx: &CpuContext, i: u64| ctx.read_u32(o.ids + i * 4).unwrap() as u64;
        let submit_apr = |ctx: &mut CpuContext, i: usize, prio: u64, r: u64| {
            call(ctx, s, cb_apr_submit_and_get_result, &[o.headers[i], prio, o.results + r * 8, o.ids + r * 4])
        };
        let submit_amm = |ctx: &mut CpuContext, i: usize, prio: u64, r: u64| {
            let bytes = o.offset(ctx, i);
            call(ctx, s, cb_amm_submit_and_get_id, &[o.data[i], bytes, prio, o.ids + r * 4])
        };
        let wait_apr = |ctx: &mut CpuContext, r: u64| {
            let id = id(ctx, r);
            call(ctx, s, cb_apr_wait, &[id])
        };
        let wait_amm = |ctx: &mut CpuContext, r: u64| {
            let id = id(ctx, r);
            call(ctx, s, cb_amm_wait, &[id])
        };

        // Compare order: ==, unsigned >/<, !=, wrapped >=, signed >/<.
        let comparisons: [(u64, u64, u64, u64); 8] = [
            (0, 1, 2, 2),
            (1, 0x4000_0000_0000_19c3, 0x4000_0000_0000_19c3, 0x4000_0000_0000_19c4),
            (1, 0, i64::MAX as u64, 1 << 63),
            (2, u64::MAX, 1 << 63, i64::MAX as u64),
            (3, 2, 2, 3),
            (4, u64::MAX - 1, u64::MAX, 0),
            (5, u64::MAX, 0, 1),
            (6, 0, 0, u64::MAX),
        ];
        for (compare, blocked, reference, released) in comparisons {
            o.reset(ctx, 3);
            let (fence, blocked_done, read_done, lower_done) = (o.var(0), o.var(1), o.var(2), o.var(3));
            for (a, v) in [(fence, blocked), (blocked_done, 0), (read_done, 0), (lower_done, 0)] {
                ctx.write_u64(a, v);
            }
            ctx.write_bytes(o.output, &[0; 3]);
            for r in 0..3 {
                ctx.write_u32(o.results + r * 8, 1234);
                ctx.write_u32(o.results + r * 8 + 4, 5678);
            }
            let [blocked_cb, reader, lower, producer] = o.headers;
            assert_eq!(call(ctx, s, cb_wait_on_address_dispatch(), &[blocked_cb, fence, reference, compare, 0]), OK);
            assert_eq!(call(ctx, s, cb_write_address, &[blocked_cb, blocked_done, 1]), OK);
            assert_eq!(read(ctx, 1), OK);
            assert_eq!(call(ctx, s, cb_write_address, &[reader, read_done, 2]), OK);
            assert_eq!(call(ctx, s, cb_write_address, &[lower, lower_done, 3]), OK);
            assert_eq!(call(ctx, s, cb_write_address, &[producer, fence, released]), OK);

            // The APR submit returns while blocked; a lower priority completes.
            assert_eq!(submit_apr(ctx, 0, 3, 0), OK);
            assert_eq!(submit_apr(ctx, 1, 3, 1), OK);
            assert_eq!(submit_apr(ctx, 2, 4, 2), OK);
            assert_eq!(wait_apr(ctx, 2), OK);
            assert_eq!(load(ctx, lower_done), 3);
            assert_eq!((load(ctx, blocked_done), load(ctx, read_done)), (0, 0), "compare {compare}");
            assert_eq!(ctx.read_bytes(o.output, 3).unwrap(), vec![0; 3]);
            assert_eq!(result(ctx, 0).0, 1234);
            assert_eq!(result(ctx, 1).0, 1234);

            // A later AMM submission releases the APR dependency.
            assert_eq!(submit_amm(ctx, 3, 1, 3), OK);
            assert_eq!(wait_amm(ctx, 3), OK);
            assert_eq!(wait_apr(ctx, 0), OK);
            assert_eq!(wait_apr(ctx, 1), OK);
            assert_eq!((load(ctx, blocked_done), load(ctx, read_done)), (1, 2));
            assert_eq!(ctx.read_bytes(o.output, 3).unwrap(), b"APR".to_vec());
            for r in 0..3 {
                assert_eq!(result(ctx, r), (0, 0), "submission {r} completes with OK");
            }
            // A retired id is gone.
            assert_eq!(wait_apr(ctx, 0), -1);
        }

        const COUNTER: u64 = 127;
        let counter_comparisons: [(u64, u64, u64, u64); 4] =
            [(1, 0, 0, 1), (1, 0, 0, 1), (4, 0x7fff_fffe, 0x7fff_ffff, 0x8000_0000), (5, 0xffff_ffff, 0, 1)];
        for (compare, blocked, reference, released) in counter_comparisons {
            o.reset(ctx, 1);
            let [producer, consumer, lower, initializer] = o.headers;
            // Invalid counter operations fail without appending a command.
            assert_eq!(call(ctx, s, cb_write_counter, &[producer, 128, 1, 1, 0, 0]), EINVAL);
            assert_eq!(call(ctx, s, cb_write_counter, &[producer, COUNTER, 8, 1, 0, 0]), EINVAL);
            assert_eq!(call(ctx, s, cb_write_counter, &[producer, COUNTER, 1, 1, 5, 0]), EINVAL);
            assert_eq!(call(ctx, s, cb_write_counter, &[producer, COUNTER, 1, 1, 0, 2]), EINVAL);
            assert_eq!(call(ctx, s, cb_wait_on_counter, &[consumer, COUNTER, 8, 0, 1, 0, 0, 0]), EINVAL);
            assert_eq!(call(ctx, s, cb_wait_on_counter, &[consumer, COUNTER, 1, 0, 7, 0, 0, 0]), EINVAL);
            assert_eq!(call(ctx, s, cb_wait_on_counter, &[consumer, COUNTER, 1, 0, 1, 2, 0, 0]), EINVAL);
            assert_eq!(call(ctx, s, cb_wait_on_counter, &[consumer, COUNTER, 1, 0, 1, 0, 0, 2]), EINVAL);
            assert_eq!((o.offset(ctx, 0), o.offset(ctx, 1)), (0, 0));

            let (retired, lower_done) = (o.var(4), o.var(5));
            ctx.write_u64(retired, 0);
            ctx.write_u64(lower_done, 0);
            ctx.write_bytes(o.output, &[0; 3]);
            ctx.write_u32(o.results, 1234);
            if blocked != 0 {
                assert_eq!(call(ctx, s, cb_write_counter, &[initializer, COUNTER, 1, blocked, 0, 0]), OK);
                assert_eq!(submit_amm(ctx, 3, 0, 3), OK);
                assert_eq!(wait_amm(ctx, 3), OK);
            }
            assert_eq!(read(ctx, 0), OK);
            assert_eq!(call(ctx, s, cb_write_counter, &[producer, COUNTER, 1, released, 0, 0]), OK);
            assert_eq!(call(ctx, s, cb_wait_on_counter, &[consumer, COUNTER, 1, reference, compare, 0, 0, 0]), OK);
            assert_eq!(call(ctx, s, cb_write_counter, &[consumer, COUNTER, 1, 0, 0, 0]), OK);
            assert_eq!(call(ctx, s, cb_write_address, &[consumer, retired, 1]), OK);
            assert_eq!(call(ctx, s, cb_write_address, &[lower, lower_done, 1]), OK);

            // The counter wait blocks AMM retirement while its lower priority progresses.
            assert_eq!(submit_amm(ctx, 1, 0, 1), OK);
            assert_eq!(submit_amm(ctx, 2, 1, 2), OK);
            assert_eq!(wait_amm(ctx, 2), OK);
            assert_eq!((load(ctx, lower_done), load(ctx, retired)), (1, 0), "counter compare {compare}");

            // The APR completion releases AMM only after its read, then resets the counter.
            assert_eq!(submit_apr(ctx, 0, 3, 0), OK);
            assert_eq!(wait_amm(ctx, 1), OK);
            assert_eq!(wait_apr(ctx, 0), OK);
            assert_eq!(result(ctx, 0).0, 0);
            assert_eq!(load(ctx, retired), 1);
            assert_eq!(ctx.read_bytes(o.output, 3).unwrap(), b"APR".to_vec());
            assert_eq!(engine::counter(COUNTER as u8), 0);
        }

        // An APR read submits its completion event to the AMPR registration;
        // a user event with the same ident is independent.
        o.reset(ctx, 1);
        let eq = equeue::create();
        let udata = 0xAB_CDEF;
        assert!(equeue::register_event(eq, 42, equeue::FILTER_AMPR, udata, equeue::EV_CLEAR, 0));
        assert!(equeue::register_event(eq, 42, equeue::FILTER_USER, 0, equeue::EV_CLEAR, 0));
        ctx.write_bytes(o.output, &[0; 3]);
        ctx.write_u32(o.results, 1234);
        let reader = o.headers[0];
        assert_eq!(read(ctx, 0), OK);
        assert_eq!(call(ctx, s, cb_write_kernel_event_queue, &[reader, eq, 42, 0x1234_5678_9abc, 0, 0]), OK);
        assert_eq!(call(ctx, s, cb_measure_kernel_event_queue, &[]), o.offset(ctx, 0) as i64 - 0x14);
        assert_eq!(submit_apr(ctx, 0, 3, 0), OK);
        assert_eq!(wait_apr(ctx, 0), OK);
        assert_eq!(result(ctx, 0).0, 0);
        let events = equeue::wait(eq, 2, Some(0)).unwrap();
        assert_eq!(events.len(), 1);
        let e = events[0];
        assert_eq!((e.ident, e.filter, e.data, e.udata), (42, equeue::FILTER_AMPR, 0x1234_5678_9abc, udata));
        assert_eq!(ctx.read_bytes(o.output, 3).unwrap(), b"APR".to_vec());
        assert!(equeue::delete_registered_event(eq, 42, equeue::FILTER_AMPR));
        assert!(!equeue::trigger_ampr_event(eq, 42, 0));
        assert!(equeue::trigger_event(eq, 42, equeue::FILTER_USER, 0));
        let events = equeue::wait(eq, 2, Some(0)).unwrap();
        assert_eq!((events.len(), events[0].filter), (1, equeue::FILTER_USER));
        assert!(equeue::delete(eq));

        // A failing command stops its submission and reports its offset.
        o.reset(ctx, 1);
        let h = o.headers[0];
        ctx.write_u32(o.results, 1234);
        assert_eq!(call(ctx, s, cb_write_address, &[h, o.var(6), 1]), OK);
        assert_eq!(call(ctx, s, cb_apr_read_file, &[h, 0, 0, 0x7FFF_FFF0, o.output, 3, 0]), OK);
        assert_eq!(call(ctx, s, cb_write_address, &[h, o.var(7), 1]), OK);
        assert_eq!(submit_apr(ctx, 0, 0, 0), OK);
        assert_eq!(wait_apr(ctx, 0), OK);
        assert_eq!(result(ctx, 0), (crate::sync::ERR_NOT_FOUND as i32, 0x20));
        assert_eq!((load(ctx, o.var(6)), load(ctx, o.var(7))), (1, 0));
        // Submission argument checks.
        assert_eq!(call(ctx, s, cb_apr_submit, &[0, 0]), -1);
        assert_eq!(call(ctx, s, cb_apr_submit, &[h, 7]), -1);
        assert_eq!(call(ctx, s, cb_apr_submit_and_get_id, &[h, 0, 0]), -1);
        assert_eq!(call(ctx, s, cb_amm_submit, &[o.data[1], 0, 0]), EINVAL);
        assert_eq!(call(ctx, s, cb_amm_submit, &[o.data[0], 0x24, 3]), EINVAL);
        assert_eq!(call(ctx, s, cb_amm_submit, &[o.data[0], 0x1000, 0]), EINVAL);
        assert_eq!(call(ctx, s, cb_amm_wait, &[0xFFFF_FFF0]), engine::ERR_NO_SUCH_SUBMISSION);

        // AMM: a lower priority progresses while its high priority waits on
        // a CPU fence, which is observed without a submission notification.
        o.reset(ctx, 0);
        let (cpu_fence, amm_done, lower_done) = (o.var(8), o.var(9), o.var(10));
        for a in [cpu_fence, amm_done, lower_done] {
            ctx.write_u64(a, 0);
        }
        assert_eq!(call(ctx, s, cb_wait_on_address_dispatch(), &[o.headers[0], cpu_fence, 0, 1, 0]), OK);
        assert_eq!(call(ctx, s, cb_write_address, &[o.headers[0], amm_done, 1]), OK);
        assert_eq!(call(ctx, s, cb_write_address, &[o.headers[1], lower_done, 1]), OK);
        assert_eq!(submit_amm(ctx, 0, 0, 0), OK);
        assert_eq!(submit_amm(ctx, 1, 1, 1), OK);
        assert_eq!(wait_amm(ctx, 1), OK);
        assert_eq!((load(ctx, lower_done), load(ctx, amm_done)), (1, 0));
        // SAFETY: mapped, aligned test address.
        unsafe { AtomicU64::from_ptr(cpu_fence as *mut u64) }.store(1, Ordering::Release);
        assert_eq!(wait_amm(ctx, 0), OK);
        assert_eq!(load(ctx, amm_done), 1);

        for h in o.headers {
            assert_eq!(call(ctx, s, cb_destructor, &[h]), OK);
        }
    });
    let _ = std::fs::remove_dir_all(&root);
}

/// Both WaitOnAddress exports share one body.
fn cb_wait_on_address_dispatch() -> HleCallback {
    let (_, _, a) = *EXPORTS.iter().find(|e| e.0 == "DLfoNxTFNVk").unwrap();
    let (_, _, b) = *EXPORTS.iter().find(|e| e.0 == "V7GQTEeUfhw").unwrap();
    assert_eq!(a as usize, b as usize);
    a
}

#[test]
fn wait_and_write_argument_checks() {
    with_vm(3, |ctx, base| {
        let (cb, buf, stack) = (base, base + 0x100, base + 0x8000);
        assert_eq!(call(ctx, stack, cb_constructor, &[cb]), OK);
        assert_eq!(call(ctx, stack, cb_set_buffer, &[cb, buf, 0x400]), OK);
        let w = cb_wait_on_address_dispatch();
        assert_eq!(call(ctx, stack, w, &[cb, 0, 0, 0, 0]), EINVAL);
        assert_eq!(call(ctx, stack, w, &[cb, base + 4, 0, 0, 0]), EINVAL);
        assert_eq!(call(ctx, stack, w, &[cb, base + 8, 0, 7, 0]), EINVAL);
        assert_eq!(call(ctx, stack, cb_write_address, &[cb, base + 4, 0]), EINVAL);
        assert_eq!(call(ctx, stack, cb_write_address_zero, &[cb, 0, 0]), EINVAL);
        assert_eq!(call(ctx, stack, cb_get_current_offset, &[cb]), 0);
        assert_eq!(call(ctx, stack, w, &[cb, base + 8, 5, 6, 1]), OK);
        assert_eq!(call(ctx, stack, cb_write_address_zero, &[cb, base + 8, 0, 0]), OK);
        assert_eq!(call(ctx, stack, cb_write_counter_on_completion, &[cb, 0, 0, 0, 0]), OK);
        assert_eq!(call(ctx, stack, cb_get_current_offset, &[cb]), 3 * cbuf::FIXED32_SIZE as i64);
        assert_eq!(call(ctx, stack, cb_measure_fixed32, &[]), 0x20);
        let st = cbuf::try_get_state(ctx, cb).unwrap();
        assert_eq!(
            st.commands.iter().map(|c| c.data).collect::<Vec<_>>(),
            vec![
                CommandData::Wait { source: base + 8, value: 5, compare: 6, is_counter: false },
                CommandData::WriteAddress { address: base + 8, value: 0 },
            ]
        );
        // AMM map records validate page alignment.
        assert_eq!(call(ctx, stack, cb_amm_map, &[cb, 0x4000, 0x2000, 0, 0x33]), EINVAL);
        assert_eq!(call(ctx, stack, cb_amm_map_direct, &[cb, 0x4000, 0x2000, 0x4000, 0, 0x33]), EINVAL);
        assert_eq!(call(ctx, stack, cb_amm_map_direct_with_gpu_mask_id, &[cb, 0x4000, 0x8000, 0x4000, 0, 0x33, 2]), OK);
        assert_eq!(call(ctx, stack, cb_amm_unmap, &[cb, 0x4000, 0x4000]), OK);
        let st = cbuf::try_get_state(ctx, cb).unwrap();
        assert_eq!(st.write_offset, 3 * 0x20 + 0x30 + 0x20);
        assert_eq!(
            st.commands[2].data,
            CommandData::AmmMap(AmmMap {
                kind: AmmKind::MapDirect,
                va: 0x4000,
                dmem_offset: 0x8000,
                size: 0x4000,
                mtype: 0,
                prot: 0x33,
                gpu_mask_id: 2,
                ..Default::default()
            })
        );
        // Usage stats are written up to the caller's size.
        let stats = base + 0x7000;
        ctx.write_bytes(stats, &[0xEE; 0x20]);
        ctx.write_u64(stats, 0x10);
        assert_eq!(call(ctx, stack, cb_amm_get_usage_stats_data, &[stats]), OK);
        assert_eq!(ctx.read_u32(stats + 8), Some(512));
        assert_eq!(ctx.read_u32(stats + 0x10), Some(0xEEEE_EEEE));
        ctx.write_u64(stats, 0x19);
        assert_eq!(call(ctx, stack, cb_amm_get_usage_stats_data, &[stats]), EINVAL);
        assert_eq!(call(ctx, stack, cb_amm_get_virtual_address_ranges, &[stats, stats + 8, 0, 0]), 0);
        assert_eq!(ctx.read_u64(stats), Some(amm::VA_BASE));
        assert_eq!(ctx.read_u64(stats + 8), Some(amm::VA_BASE + amm::VA_SIZE));
        assert_eq!(call(ctx, stack, cb_destructor, &[cb]), OK);
    });
}

/// A zero-filled command buffer object that no constructor ran on still
/// records and submits once a buffer is attached and reset.
#[test]
fn unconstructed_buffer_submits() {
    let root = crate::filesystem::tests::temp_root("ampr-unconstructed");
    let bytes: Vec<u8> = (0..256u32).map(|i| (i * 7 + 3) as u8).collect();
    std::fs::write(root.join("zero.dat"), &bytes).unwrap();
    let _app0 = crate::filesystem::APP0_TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    crate::filesystem::set_app0_root(&root);
    let file_id = crate::filesystem::apr_resolve_path("/app0/zero.dat").expect("resolve fixture");

    with_vm(4, |ctx, base| {
        let (cb, buf, data, done, result, id, stack) =
            (base, base + 0x100, base + 0x1000, base + 0x1100, base + 0x1200, base + 0x1210, base + 0x8000);
        ctx.write_bytes(cb, &[0; 0x40]);
        ctx.write_bytes(data, &[0; 32]);
        ctx.write_u64(done, 0);
        assert_eq!(call(ctx, stack, cb_set_buffer, &[cb, buf, 0x400]), OK);
        assert_eq!(call(ctx, stack, cb_reset, &[cb]), OK);
        assert_eq!(call(ctx, stack, cb_apr_read_file, &[cb, base + 0x40, base + 0x48, file_id as u64, data, 32, 64]), OK);
        assert_eq!(call(ctx, stack, cb_write_address, &[cb, done, 0x1234567]), OK);
        assert_eq!(call(ctx, stack, cb_apr_submit_and_get_result, &[cb, 1, result, id]), OK);
        let submission = ctx.read_u32(id).unwrap() as u64;
        assert_eq!(call(ctx, stack, cb_apr_wait, &[submission]), OK);
        assert_eq!(ctx.read_bytes(data, 32).unwrap(), bytes[64..96].to_vec());
        assert_eq!(load(ctx, done), 0x1234567);
    });
}

/// A read that reaches the end of the file fills the rest of the
/// destination with zeros, also when it starts at or past the end.
#[test]
fn read_past_end_of_file_zero_fills() {
    let root = crate::filesystem::tests::temp_root("ampr-past-end");
    let bytes: Vec<u8> = (0..64u32).map(|i| (i * 5 + 1) as u8).collect();
    std::fs::write(root.join("settings.ini"), &bytes).unwrap();
    let _app0 = crate::filesystem::APP0_TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    crate::filesystem::set_app0_root(&root);
    let file_id = crate::filesystem::apr_resolve_path("/app0/settings.ini").expect("resolve fixture");
    // The file shrinks by one byte after its id was resolved.
    std::fs::write(root.join("settings.ini"), &bytes[..63]).unwrap();

    with_vm(5, |ctx, base| {
        let (cb, buf, whole, at_end, past_end) = (base, base + 0x100, base + 0x1000, base + 0x1100, base + 0x1200);
        let (result, id, stack) = (base + 0x1300, base + 0x1310, base + 0x8000);
        ctx.write_bytes(cb, &[0; 0x40]);
        ctx.write_bytes(whole, &[0xA5; 0x300]);
        assert_eq!(call(ctx, stack, cb_set_buffer, &[cb, buf, 0x400]), OK);
        assert_eq!(call(ctx, stack, cb_reset, &[cb]), OK);
        let read = |ctx: &mut CpuContext, dst: u64, size: u64, offset: u64| {
            call(ctx, stack, cb_apr_read_file, &[cb, base + 0x40, base + 0x48, file_id as u64, dst, size, offset])
        };
        assert_eq!(read(ctx, whole, 64, 0), OK);
        assert_eq!(read(ctx, at_end, 16, 63), OK);
        assert_eq!(read(ctx, past_end, 16, 4096), OK);
        assert_eq!(call(ctx, stack, cb_apr_submit_and_get_result, &[cb, 1, result, id]), OK);
        let submission = ctx.read_u32(id).unwrap() as u64;
        assert_eq!(call(ctx, stack, cb_apr_wait, &[submission]), OK);
        let mut expect = bytes[..63].to_vec();
        expect.push(0);
        assert_eq!(ctx.read_bytes(whole, 64).unwrap(), expect);
        assert_eq!(ctx.read_bytes(whole + 64, 1).unwrap(), vec![0xA5], "no byte past the read is written");
        assert_eq!(ctx.read_bytes(at_end, 16).unwrap(), vec![0; 16]);
        assert_eq!(ctx.read_bytes(past_end, 16).unwrap(), vec![0; 16]);
    });
}

const PAGE: u64 = cbuf::AMM_PAGE_SIZE;

/// Remap/MultiMap records: NULL object EINVAL, no command memory EPERM,
/// then both ranges and the protection are checked; the record carries the
/// source and the protection.
#[test]
fn amm_remap_and_multimap_records() {
    with_vm(6, |ctx, base| {
        let (cb, buf, stack) = (base, base + 0x100, base + 0x8000);
        assert_eq!(call(ctx, stack, cb_constructor, &[cb]), OK);
        assert_eq!(call(ctx, stack, cb_amm_remap, &[0, PAGE, 2 * PAGE, PAGE, 3]), EINVAL);
        assert_eq!(call(ctx, stack, cb_amm_remap, &[cb, PAGE, 2 * PAGE, PAGE, 3]), ERR_PERMISSION_DENIED);
        assert_eq!(call(ctx, stack, cb_set_buffer, &[cb, buf, 0x50]), OK);
        assert_eq!(call(ctx, stack, cb_amm_remap, &[cb, PAGE, 0x2000, PAGE, 3]), EINVAL, "unaligned source");
        assert_eq!(call(ctx, stack, cb_amm_remap, &[cb, PAGE, 2 * PAGE, PAGE, 0x400]), EINVAL, "invalid protection");
        assert_eq!(call(ctx, stack, cb_amm_multi_map, &[cb, 0, 2 * PAGE, PAGE, 3]), EINVAL);
        assert_eq!(call(ctx, stack, cb_amm_remap_with_gpu_mask_id, &[cb, PAGE, 2 * PAGE, PAGE, 0x33, 5]), OK);
        assert_eq!(call(ctx, stack, cb_amm_multi_map, &[cb, 4 * PAGE, 2 * PAGE, PAGE, 3]), OK);
        assert_eq!(call(ctx, stack, cb_amm_multi_map_with_gpu_mask_id, &[cb, 4 * PAGE, 2 * PAGE, PAGE, 3, 1]), ERR_BUSY, "full");
        assert_eq!(call(ctx, stack, cb_get_current_offset, &[cb]), 2 * cbuf::AMM_REMAP_SIZE as i64);
        let st = cbuf::try_get_state(ctx, cb).unwrap();
        let expect = |kind, prot, gpu_mask_id, va| AmmMap { kind, va, source: 2 * PAGE, size: PAGE, prot, gpu_mask_id, ..Default::default() };
        assert_eq!(
            st.commands.iter().map(|c| c.data).collect::<Vec<_>>(),
            vec![
                CommandData::AmmMap(expect(AmmKind::Remap, 0x33, 5, PAGE)),
                CommandData::AmmMap(expect(AmmKind::MultiMap, 3, 0, 4 * PAGE)),
            ]
        );
        assert_eq!(call(ctx, stack, cb_measure_amm_remap, &[PAGE, 2 * PAGE, PAGE, 3]), 0x28);
        assert_eq!(call(ctx, stack, cb_measure_amm_remap, &[PAGE, 2 * PAGE, PAGE + 1, 3]), EINVAL);
    });
}

/// ModifyProtect/ModifyMtypeProtect records (the gpu mask id of the
/// mtype variant on the stack) and their measures.
#[test]
fn amm_protect_records() {
    with_vm(7, |ctx, base| {
        let (cb, buf, stack) = (base, base + 0x100, base + 0x8000);
        assert_eq!(call(ctx, stack, cb_constructor, &[cb]), OK);
        assert_eq!(call(ctx, stack, cb_amm_modify_protect, &[cb, PAGE, PAGE, 3, 3]), ERR_PERMISSION_DENIED);
        assert_eq!(call(ctx, stack, cb_set_buffer, &[cb, buf, 0x400]), OK);
        assert_eq!(call(ctx, stack, cb_amm_modify_protect, &[cb, PAGE, PAGE, 3, 0x800]), EINVAL, "invalid mask");
        assert_eq!(call(ctx, stack, cb_amm_modify_protect, &[cb, PAGE, 0, 3, 3]), EINVAL, "empty range");
        assert_eq!(call(ctx, stack, cb_amm_modify_protect, &[cb, PAGE, PAGE, 1, 2]), OK);
        assert_eq!(call(ctx, stack, cb_amm_modify_protect_with_gpu_mask_id, &[cb, PAGE, PAGE, 1, 2, 4]), OK);
        assert_eq!(call(ctx, stack, cb_amm_modify_mtype_protect, &[cb, PAGE, PAGE, 3, 0x32, 0x33]), OK);
        assert_eq!(call(ctx, stack, cb_amm_modify_mtype_protect_with_gpu_mask_id, &[cb, PAGE, PAGE, 3, 0x32, 0x33, 6]), OK);
        let st = cbuf::try_get_state(ctx, cb).unwrap();
        let p = |kind, mtype, prot, mask, gpu_mask_id| {
            CommandData::AmmMap(AmmMap { kind, va: PAGE, size: PAGE, mtype, prot, mask, gpu_mask_id, ..Default::default() })
        };
        assert_eq!(
            st.commands.iter().map(|c| c.data).collect::<Vec<_>>(),
            vec![
                p(AmmKind::ModifyProtect, 0, 1, 2, 0),
                p(AmmKind::ModifyProtect, 0, 1, 2, 4),
                p(AmmKind::ModifyMtypeProtect, 3, 0x32, 0x33, 0),
                p(AmmKind::ModifyMtypeProtect, 3, 0x32, 0x33, 6),
            ]
        );
        assert_eq!(st.write_offset, 4 * cbuf::AMM_PROTECT_SIZE);
        assert_eq!(call(ctx, stack, cb_measure_amm_modify_protect, &[PAGE, PAGE, 1, 2]), 0x28);
        assert_eq!(call(ctx, stack, cb_measure_amm_modify_protect, &[PAGE, PAGE, 1, 0x1000]), EINVAL);
        assert_eq!(call(ctx, stack, cb_measure_amm_modify_mtype_protect, &[PAGE, PAGE, 99, 1, 2]), 0x28);
        assert_eq!(call(ctx, stack, cb_measure_amm_modify_mtype_protect, &[PAGE, PAGE, 99, 0x400, 2]), EINVAL);
    });
}

/// PRT records: a full buffer is EBUSY before missing command memory is
/// EPERM; AllocatePaForPrt checks the protection first and records the
/// allocation mask.
#[test]
fn amm_prt_records() {
    with_vm(8, |ctx, base| {
        let (cb, buf, stack) = (base, base + 0x100, base + 0x8000);
        assert_eq!(call(ctx, stack, cb_constructor, &[cb]), OK);
        assert_eq!(call(ctx, stack, cb_amm_map_as_prt, &[cb, PAGE, PAGE]), ERR_BUSY, "no command memory: no room");
        assert_eq!(call(ctx, stack, cb_amm_allocate_pa_for_prt, &[0, PAGE, PAGE, 0, 0x400]), EINVAL, "protection first");
        assert_eq!(call(ctx, stack, cb_amm_allocate_pa_for_prt, &[0, PAGE, PAGE, 0, 3]), EINVAL);
        assert_eq!(call(ctx, stack, cb_set_buffer, &[cb, buf, 0x48]), OK);
        assert_eq!(call(ctx, stack, cb_amm_map_as_prt, &[cb, PAGE, PAGE + 4]), EINVAL);
        assert_eq!(call(ctx, stack, cb_amm_map_as_prt, &[cb, PAGE, PAGE]), OK);
        assert_eq!(call(ctx, stack, cb_amm_allocate_pa_for_prt, &[cb, PAGE, PAGE, 3, 0x33]), OK);
        assert_eq!(call(ctx, stack, cb_amm_map_as_prt, &[cb, PAGE, PAGE]), ERR_BUSY);
        let st = cbuf::try_get_state(ctx, cb).unwrap();
        assert_eq!(
            st.commands.iter().map(|c| c.data).collect::<Vec<_>>(),
            vec![
                CommandData::AmmMap(AmmMap { kind: AmmKind::MapAsPrt, va: PAGE, size: PAGE, ..Default::default() }),
                CommandData::AmmMap(AmmMap {
                    kind: AmmKind::AllocatePaForPrt,
                    va: PAGE,
                    size: PAGE,
                    mtype: 3,
                    prot: 0x33,
                    mask: 1019,
                    ..Default::default()
                }),
            ]
        );
        assert_eq!(st.write_offset, cbuf::AMM_MAP_AS_PRT_SIZE + cbuf::AMM_PROTECT_SIZE);
        assert_eq!(call(ctx, stack, cb_measure_amm_map_as_prt, &[PAGE, PAGE]), 0x20);
        assert_eq!(call(ctx, stack, cb_measure_amm_map_as_prt, &[0, PAGE]), EINVAL);
        assert_eq!(call(ctx, stack, cb_measure_amm_allocate_pa_for_prt, &[PAGE, PAGE, 0, 3]), 0x28);
        assert_eq!(call(ctx, stack, cb_measure_amm_allocate_pa_for_prt, &[PAGE, PAGE, 0, 4]), EINVAL);
    });
}

/// WaitOnCounter (32-bit reference): counter below 128, compare below 4,
/// flush 0 or 1; records a counter wait.
#[test]
fn wait_on_counter_32_records() {
    with_vm(9, |ctx, base| {
        let (cb, buf, stack) = (base, base + 0x100, base + 0x8000);
        assert_eq!(call(ctx, stack, cb_constructor, &[cb]), OK);
        assert_eq!(call(ctx, stack, cb_wait_on_counter_32, &[cb, 1, 5, 0, 0]), ERR_PERMISSION_DENIED);
        assert_eq!(call(ctx, stack, cb_set_buffer, &[cb, buf, 0x400]), OK);
        assert_eq!(call(ctx, stack, cb_wait_on_counter_32, &[0, 1, 5, 0, 0]), EINVAL);
        assert_eq!(call(ctx, stack, cb_wait_on_counter_32, &[cb, 128, 5, 0, 0]), EINVAL);
        assert_eq!(call(ctx, stack, cb_wait_on_counter_32, &[cb, 1, 5, 4, 0]), EINVAL);
        assert_eq!(call(ctx, stack, cb_wait_on_counter_32, &[cb, 1, 5, 0, 2]), EINVAL);
        assert_eq!(call(ctx, stack, cb_wait_on_counter_32, &[cb, 127, 0x1_0000_0005, 3, 1]), OK);
        let st = cbuf::try_get_state(ctx, cb).unwrap();
        assert_eq!(st.commands[0].data, CommandData::Wait { source: 127, value: 5, compare: 3, is_counter: true });
        assert_eq!(st.write_offset, cbuf::FIXED32_SIZE);
    });
}
