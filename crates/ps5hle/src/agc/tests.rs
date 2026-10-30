// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only
//
// Unit tests of the `ps5hle::agc` module.

use super::pm4::interp::{self, Interpreter};
use super::pm4::queue::{GpuQueue, SliceResult};
use super::pm4::{self as p4, nop, op};
use super::*;
use crate::context::{CpuContext, CpuRegister};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

/// Every libSceAgc / libSceAgcDriver NID the test title imports.
const GAME_NIDS: &[&str] = &[
    "KT-hTp-Ch14", "ewobAQeMo5k", "hcIxS8pmXF4", "XKKuA6VkSRc", "da1Sm8-QDoU", "qyM2bxYFPAk", "ozKzBP4aki4", "qzMN2XKGA4k",
    "CbQh3DKMSno", "j3EtxFkSIhQ", "PxKWV2fVAps", "-RnpfpxIhec", "M0ttm8h7SKA", "cFazmnXpJOE", "Y-5vneiBtzk", "e1DFTg+Sd8U",
    "b-oySn+G2tE", "q4VuU-QsLOE", "6mFxkVqdmbQ", "szG7hz2yEhA", "eCjKaqeeQ5s", "cpCILPya5Zk", "F8NLhWvFemI", "JrtiDtKeS38",
    "DwICrVxerkY", "0ZOG0jc9nRg", "ebixW91gpPw", "xAeBOa0A3kk", "opR1JeJZCBU", "rVOmPz2RBlg", "idlaArvdXEs", "htn36gPnBk4",
    "GPbUp9jXQa8", "eZ4+17OQz4Q", "3ZWa3AoyWZQ", "k-JpyR2dYAM", "eWaWyFegzgQ", "GXBlM-ekzrI", "w1KFAHVqpaU", "uZW-mqsxkrM",
    "7toV+elXqNM", "FuVbkyKlf+s", "k3GhuSNmBLU", "Abendgtz+3o", "LtTouSCZjHM", "t7PlZ9nt5Lc", "hL7C0IRpWZI", "wr23dPKyWc0",
    "n2fD4A+pb+g", "bxGoVxpdSPQ", "UZbQjYAwwXM", "yUBESvCCJ4I", "MDLD5Ly94Xk", "JOWmDrl+j20", "03RZmELWWzw", "TGEZzUWLbrc",
    "YWTKOju587o", "ORWsxIbk4TE", "HV4j+E0MBHE", "D9sr1xGUriE", "f3dg2CSgRKY", "57labkp+rSQ", "-vnlTPPXPrw", "1tB0xkLNjcw",
    "1-gUn1PI4Sw", "oz6zQq1JwCE", "ms1xVoZ-Vwc", "BIPexNBSGog", "ou16V5hh5sg", "H6vHS5cidSA", "1rZSWUv1IRc", "b5u0Jzm8TF8",
    "CtB+A9-VxO0", "w8HVkEeXPv8", "WmAc2MEj6Io", "2ccJz9LQI+w", "q88lQ+GP5Yk", "Yw0jKSqop+E", "WrdP9Zxx3lQ", "6ee9Hd3EWXQ",
    "t1vNu082-jM", "mStuvI0zOtc", "ypVBz4uPKcQ", "r98I08t+LOg", "Rlx+bykm0r0", "mR9j7+SfM34", "B+aG9DUnTKA", "qMlfB1ZhMDc",
    "1q1titRBL6o", "cxPZ4Wgvdj8", "kUlvghKs-mA", "pYoKs3lPy88", "P1CugZ99Uzc", "aJf+j5yntiU", "C4l9fB17t8w", "vuSXe69VILM",
    "rUuVjyR+Rd4", "xSAR0LTcRKM", "VEGu4dixjUg", "G0jrLdvEqDw", "H7uZqCoNuWk", "jt3pl7EN17o", "KjPeVduz6jU", "+kSrjIVxKFE",
    "zg6u-N6Otxs", "TRO721eVt4g", "zfcxg-ewMK8", "QIXCsbipds0", "9S4noWrUI0s", "MMlmJAL7N5w", "RmaJwLtc8rY", "yheJGN-ay+A",
    "LHFXRrlTPD8", "1DeUNpRIDDA", "ZvwO9euwYzc", "GBCh3zCihoU", "YUeqkyT7mEQ", "l4fM9K-Lyks", "j4emHHndCPY", "8N2tmT3jmC8",
    "mljzuGDZRQ4", "0o3VDdtA6nM", "AFIh8SQkYlQ", "GIIW2J37e70", "ca4KPvp0qLQ", "QhCbS4X9Rl8", "tSBxhAPyytQ", "6DFuRKT4C9w",
    "bbFueFP+J4k", "vLrBL8DQiz8", "pFLArOT53+w", "QhPDD513V0w", "-HOOCn0JY48", "nNlUtdDDvZ0", "w4-d0n60hdo", "aP1Ki9G3++4",
    "hvUfkUIQcOE", "UQGTw4xRlcM", "hEK26Wdny6s", "LFSPFmGc9Hg", "XN+Iuu7XsM8", "u2T2DiA5hRI", "+u6dKSLWM2o", "43WJ08sSugE",
    "VmW0Tdpy420", "MWiElSNE8j8", "i1jyy49AjXU", "p9tI+yTvx68", "IxYiarKlXxM", "cdDRpqcFGbU", "F0ZXt5q0ZTA", "uJziRsODk1c",
    "U9ueyEhSkF4", "V++UgBtQhn0", "Lkf86B98qPc", "2JtWUUiYBXs", "wRbq6ZjNop4", "23LRUSvYu1M", "0fWWK5uG9rQ", "MlEw1feXcjg",
    "J8YCgfKAMQs", "T9fjQIINoeE", "ziVA3whp3p4", "d-6uF9sZDIU", "vcmNN+AAXnY", "whb1RL7K4Ss", "K2mciNVxUCE", "w6Dj1VJt5qY",
    "n8vgpaQg6dA", "z2duB-hHQSM", "Qrj4c+61z4A", "nCUgItdN2ms", "vRoArM9zaIk", "6lNcCp+fxi4", "fRG-JOH5+sI", "h9z6+0hEydk",
    "qj7QZpgr9Uw", "Y3ymLfZ1384", "3KDcnM3lrcU", "n485EBnIWmk", "7nOoijNPvEU", "w2rJhmD+dsE", "AhGvpITrf4M", "+T8Xo6LtFJI",
    "DL2RXaXOy88", "5l3IfCFJxBs", "Zw7uUVPulbw", "5CdQTZIQPxM", "LepGrgk77sM", "NghWEUXp1qM", "M9yBzRKkjPc", "mXn+K9E-wOA",
    "rI9lNAXPMIw", "ls4jfY576lw", "F0Y42t-3e18", "Ddwk4gLT5j0", "qspAL8bgcBY", "+TN0oRTBxJQ", "AOLcoIkQDgM", "emP3ckeS2uo",
    "X-Nm5KLREeg", "W5z4eZrjEas", "3AyTaWcF-H8", "SAfhzJPcjuk", "FOwvmNlFLjM", "VOMSpd9+vxU", "gSRnr79F8tQ", "UglJIZjGssM",
    "HF3YllT3mXU", "6UzEidRZwkg", "Xq5WmbwPTnQ", "SCoAN5fYlUM", "ZLJk9r2+2Aw", "pWLG7WOpVcw", "n5ElQVYsU1A",
];

// ---------------------------------------------------------------------------
// Test environment: a tracked host arena standing in for guest memory.
// ---------------------------------------------------------------------------

const ARENA: u64 = 0x80000;
const CB: u64 = 0x0; // command buffer object
const STACK: u64 = 0x1000; // rsp
const DATA: u64 = 0x2000; // scratch data
const LABELS: u64 = 0x8000; // label area
const CMD: u64 = 0x10000; // command space
const CMD_END: u64 = 0x70000;

struct Env {
    base: u64,
    vm: ps5vm::VirtualMemory,
}

impl Env {
    fn new() -> Self {
        let buf: &'static mut [u8] = Box::leak(vec![0u8; ARENA as usize + 0x1000].into_boxed_slice());
        let base = (buf.as_ptr() as u64 + 0xFFF) & !0xFFF;
        ps5vm::ranges::track(base, ARENA, ps5vm::MemProtect::RW);
        let e = Env { base, vm: ps5vm::VirtualMemory::new() };
        e.reset_cb();
        e
    }

    fn a(&self, off: u64) -> u64 {
        self.base + off
    }

    fn cb(&self) -> u64 {
        self.a(CB)
    }

    fn reset_cb(&self) {
        assert!(mem::w64(self.a(CB + CB_CURSOR_UP), self.a(CMD)));
        assert!(mem::w64(self.a(CB + CB_CURSOR_DOWN), self.a(CMD_END)));
        assert!(mem::w64(self.a(CB + CB_CALLBACK), 0));
        assert!(mem::w64(self.a(CB + CB_USER_DATA), 0));
        assert!(mem::w32(self.a(CB + CB_RESERVED_DW), 0));
    }

    fn cursor(&self) -> u64 {
        mem::r64(self.a(CB + CB_CURSOR_UP)).unwrap()
    }

    fn call(&self, name: &str, args: &[u64], stack: &[u64]) -> i64 {
        let (_, _, _, cb) = all_exports().into_iter().find(|e| e.1 == name).unwrap_or_else(|| panic!("no export {name}"));
        let mut ctx = CpuContext::new(&self.vm);
        const REGS: [CpuRegister; 6] =
            [CpuRegister::Rdi, CpuRegister::Rsi, CpuRegister::Rdx, CpuRegister::Rcx, CpuRegister::R8, CpuRegister::R9];
        for (i, v) in args.iter().enumerate() {
            ctx.set(REGS[i], *v);
        }
        let rsp = self.a(STACK);
        ctx.set(CpuRegister::Rsp, rsp);
        for (i, v) in stack.iter().enumerate() {
            assert!(mem::w64(rsp + 8 * (i as u64 + 1), *v));
        }
        cb(&mut ctx)
    }
}

/// Walk the packets in `[start, end)`; every header must be type 3 (or the
/// filler) and the lengths must tile the range exactly.
fn parse_range(start: u64, end: u64) -> Vec<u32> {
    let mut headers = Vec::new();
    let mut p = start;
    while p < end {
        let h = mem::r32(p).unwrap();
        let n = if h == p4::FILLER {
            1
        } else {
            assert_eq!(p4::packet_type(h), 3, "header 0x{h:08x} at +0x{:x}", p - start);
            p4::packet_size(h)
        };
        headers.push(h);
        p += n as u64 * 4;
    }
    assert_eq!(p, end, "packets overrun the emitted range");
    headers
}

// ---------------------------------------------------------------------------
// NIDs
// ---------------------------------------------------------------------------

#[test]
fn nids_match_names() {
    let exports = all_exports();
    for (nid, name, _, _) in &exports {
        if LABELED_NIDS.contains(nid) {
            continue;
        }
        assert_eq!(ps5nid::compute_nid(name), *nid, "NID mismatch for {name}");
    }
    let mut seen = std::collections::HashSet::new();
    for (nid, name, _, _) in &exports {
        assert!(seen.insert(*nid), "duplicate NID {nid} ({name})");
    }
}

#[test]
fn every_game_import_is_implemented() {
    let exports: std::collections::HashSet<&str> = all_exports().iter().map(|e| e.0).collect();
    assert_eq!(GAME_NIDS.len(), 215);
    for nid in GAME_NIDS {
        assert!(exports.contains(nid), "game import {nid} not implemented");
    }
    let mut r = HleRegistry::new();
    register(&mut r);
    for nid in GAME_NIDS {
        assert!(r.is_implemented(nid), "{nid} not registered");
    }
}

// ---------------------------------------------------------------------------
// Builder <-> GetSize consistency
// ---------------------------------------------------------------------------

struct Case {
    builder: &'static str,
    args: Vec<u64>,
    stack: Vec<u64>,
    size: Option<(&'static str, Vec<u64>)>,
}

fn case(builder: &'static str, args: Vec<u64>, stack: Vec<u64>, size: Option<(&'static str, Vec<u64>)>) -> Case {
    Case { builder, args, stack, size }
}

fn builder_cases(e: &Env) -> Vec<Case> {
    let cb = e.cb();
    let label = e.a(LABELS);
    let label2 = e.a(LABELS + 0x100);
    let data = e.a(DATA);
    let target = e.a(CMD_END - 0x400);
    // Scratch inputs: dwords, register pairs, marker string, workload ids.
    mem::write_dwords(data, &[1, 2, 3, 4, 5, 6, 7, 8]).then_some(()).unwrap();
    let regs = e.a(DATA + 0x100);
    mem::write_dwords(regs, &[0x10, 0xAA, 0x11, 0xBB, 0x30, 0xCC]).then_some(()).unwrap();
    let s = e.a(DATA + 0x200);
    mem::write(s, b"marker\0").then_some(()).unwrap();
    let ids = e.a(DATA + 0x300);
    mem::write_dwords(ids, &[2, 5]).then_some(()).unwrap();
    let packed = (0x1234_5678u64 << 32) | 0x10;
    // The workload builders need a registered stream.
    let stream = e.a(DATA + 0x380);
    let _ = e.call("sceAgcDriverRegisterWorkloadStream", &[1, stream], &[]);
    let g = |n: &'static str, a: Vec<u64>| Some((n, a));
    vec![
        case("sceAgcCbNop", vec![cb, 5], vec![], g("sceAgcCbNopGetSize", vec![5])),
        case("sceAgcCbNop", vec![cb, 2], vec![], g("sceAgcCbNopGetSize", vec![2])),
        case("sceAgcCbNop", vec![cb, 1], vec![], g("sceAgcCbNopGetSize", vec![1])),
        case("sceAgcCbDispatch", vec![cb, 4, 2, 1, 0], vec![], g("sceAgcCbDispatchGetSize", vec![])),
        case("sceAgcAcbResetQueue", vec![cb], vec![], None),
        case("sceAgcDcbResetQueue", vec![cb, 0x3FF, 0], vec![], None),
        case("sceAgcDcbStallCommandBufferParser", vec![cb, 0, label, 1], vec![], g("sceAgcDcbStallCommandBufferParserGetSize", vec![])),
        case("sceAgcAcbAcquireMem", vec![cb, 0, 0x10000, 0x1000, 400], vec![], g("sceAgcAcbAcquireMemGetSize", vec![])),
        case("sceAgcDcbAcquireMem", vec![cb, 1, 0, 0, 0x10000, u64::MAX], vec![400], g("sceAgcDcbAcquireMemGetSize", vec![])),
        case("sceAgcDcbWaitRegMem", vec![cb, 0, 3, 0, 0, label], vec![1, 0xFFFF_FFFF, 160], g("sceAgcDcbWaitOnAddressGetSize", vec![0])),
        case("sceAgcDcbWaitRegMem", vec![cb, 1, 3, 1, 0, label], vec![1, u64::MAX, 160], g("sceAgcDcbWaitOnAddressGetSize", vec![1])),
        case("sceAgcAcbWaitRegMem", vec![cb, 0, 3, 0, label, 1], vec![0xFFFF_FFFF, 160], g("sceAgcAcbWaitOnAddressGetSize", vec![0])),
        case("sceAgcDcbWriteData", vec![cb, 5, 0, label, data, 3], vec![0, 1], g("sceAgcDcbWriteDataGetSize", vec![3])),
        case("sceAgcAcbWriteData", vec![cb, 5, 0, label, data, 8], vec![0, 0], g("sceAgcDcbWriteDataGetSize", vec![8])),
        case("sceAgcCbReleaseMem", vec![cb, 0x28, 0, 1, 0, label], vec![2, 7, 0, 0, 2, 9], g("sceAgcCbQueueEndOfPipeActionGetSize", vec![])),
        case("sceAgcCbCondWrite", vec![cb, 3, 1, label, 5, label2], vec![0, 0xFFFF_FFFF], g("sceAgcCbCondWriteGetSize", vec![])),
        case("sceAgcDcbGetLodStats", vec![cb, 0, label, 64, 0xFF, 0], vec![1, 0], g("sceAgcDcbGetLodStatsGetSize", vec![1])),
        case("sceAgcDcbPushMarker", vec![cb, s, 0], vec![], None),
        case("sceAgcAcbPushMarker", vec![cb, s, 0], vec![], None),
        case("sceAgcDcbPopMarker", vec![cb], vec![], None),
        case("sceAgcAcbPopMarker", vec![cb], vec![], None),
        case("sceAgcDcbSetMarker", vec![cb, s, 0xFF00FF], vec![], None),
        case("sceAgcAcbSetMarker", vec![cb, s, 0xFF00FF], vec![], None),
        case("sceAgcDcbSetWorkloadsActive", vec![cb, 1, ids, 2], vec![], None),
        case("sceAgcAcbSetWorkloadsActive", vec![cb, 1, ids, 2], vec![], None),
        case("sceAgcDcbSetWorkloadComplete", vec![cb, 1, 5], vec![], None),
        case("sceAgcAcbSetWorkloadComplete", vec![cb, 1, 5], vec![], None),
        case("sceAgcDcbRewind", vec![cb, 1], vec![], g("sceAgcDcbRewindGetSize", vec![])),
        case("sceAgcAcbRewind", vec![cb, 0], vec![], g("sceAgcAcbRewindGetSize", vec![])),
        case("sceAgcDcbJump", vec![cb, 1, 0, target, 16], vec![], g("sceAgcDcbJumpGetSize", vec![])),
        case("sceAgcAcbJump", vec![cb, 0, target, 16], vec![], g("sceAgcAcbJumpGetSize", vec![])),
        case("sceAgcAcbJump", vec![cb, 1, target, 16], vec![], g("sceAgcAcbJumpGetSize", vec![])),
        case("sceAgcCbBranch", vec![cb, 2, 3, label, u64::MAX, 1], vec![0, target, 8, 0, target, 8], g("sceAgcCbBranchGetSize", vec![])),
        case("sceAgcDcbCondExec", vec![cb, label, 4], vec![], g("sceAgcDcbCondExecGetSize", vec![])),
        case("sceAgcAcbCondExec", vec![cb, label, 4], vec![], g("sceAgcAcbCondExecGetSize", vec![])),
        case("sceAgcDcbSetPredication", vec![cb, 0, 3, 1, label], vec![], g("sceAgcDcbSetBoolPredicationEnableGetSize", vec![])),
        case("sceAgcDcbSetPredication", vec![cb, 1, 3, 0, label], vec![], g("sceAgcDcbSetZPassPredicationEnableGetSize", vec![])),
        case("sceAgcDcbSetPredication", vec![cb, 0, 0, 0, 0], vec![], g("sceAgcDcbSetPredicationDisableGetSize", vec![])),
        case("sceAgcDcbCopyData", vec![cb, 5, 0, label, 5, 0], vec![label2, 1, 0], g("sceAgcDcbCopyDataGetSize", vec![])),
        case("sceAgcAcbCopyData", vec![cb, 1, 0, label, 1, 0], vec![label2, 0, 1], g("sceAgcAcbCopyDataGetSize", vec![])),
        case("sceAgcDcbDmaData", vec![cb, 0, 0, 0, label, 0], vec![0, label2, 16, 1, 0, 0], g("sceAgcDcbDmaDataGetSize", vec![])),
        case("sceAgcAcbDmaData", vec![cb, 0, 0, label, 0, 0], vec![label2, 16, 1, 0], g("sceAgcAcbDmaDataGetSize", vec![])),
        case("sceAgcDcbAtomicMem", vec![cb, 15, 0, 0, label, 1], vec![0, 0], g("sceAgcDcbAtomicMemGetSize", vec![])),
        case("sceAgcAcbAtomicMem", vec![cb, 0x2F, 0, 0, label, 1], vec![0, 0], g("sceAgcAcbAtomicMemGetSize", vec![])),
        case("sceAgcDcbMemSemaphore", vec![cb, label, 6, 0, 0], vec![], None),
        case("sceAgcAcbMemSemaphore", vec![cb, label, 7, 0, 1], vec![], None),
        case("sceAgcDcbPrimeUtcl2", vec![cb, 1, 0, 0x10000, 4], vec![], g("sceAgcDcbPrimeUtcl2GetSize", vec![])),
        case("sceAgcAcbPrimeUtcl2", vec![cb, 1, 0, 0x10000, 4], vec![], g("sceAgcAcbPrimeUtcl2GetSize", vec![])),
        case("sceAgcDcbEventWrite", vec![cb, 0x16, 0], vec![], g("sceAgcDcbEventWriteGetSize", vec![0x16])),
        case("sceAgcDcbEventWrite", vec![cb, 0x39, label], vec![], g("sceAgcDcbEventWriteGetSize", vec![0x39])),
        case("sceAgcAcbEventWrite", vec![cb, 0x07, 0], vec![], g("sceAgcAcbEventWriteGetSize", vec![0x07])),
        case("sceAgcAcbEventWrite", vec![cb, 0x38, label], vec![], g("sceAgcAcbEventWriteGetSize", vec![0x38])),
        case("sceAgcDcbWaitUntilSafeForRendering", vec![cb, 1, 0], vec![], None),
        case("sceAgcAcbWaitUntilSafeForRendering", vec![cb, 1, 1], vec![], None),
        case("sceAgcDcbSetFlip", vec![cb, 1, 0, 1, 7], vec![], None),
        case("sceAgcAcbSetFlip", vec![cb, 1, 1, 1, 7], vec![], None),
        case("sceAgcDcbSetIndexSize", vec![cb, 1, 0], vec![], g("sceAgcDcbSetIndexSizeGetSize", vec![])),
        case("sceAgcDcbSetIndexSize_0600", vec![cb, 1, 0, 1], vec![], g("sceAgcDcbSetIndexSizeGetSize", vec![])),
        case("sceAgcDcbSetIndexCount", vec![cb, 30], vec![], g("sceAgcDcbSetIndexCountGetSize", vec![])),
        case("sceAgcDcbSetNumInstances", vec![cb, 3], vec![], g("sceAgcDcbSetNumInstancesGetSize", vec![])),
        case("sceAgcDcbSetIndexBuffer", vec![cb, data], vec![], g("sceAgcDcbSetIndexBufferGetSize", vec![])),
        case("sceAgcDcbDrawIndex", vec![cb, 3, data, 0x4000_0000], vec![], g("sceAgcDcbDrawIndexGetSize", vec![])),
        case("sceAgcDcbDrawIndexAuto", vec![cb, 3, 0x4000_0000], vec![], g("sceAgcDcbDrawIndexAutoGetSize", vec![])),
        case("sceAgcDcbDrawIndexOffset", vec![cb, 6, 3, 0], vec![], g("sceAgcDcbDrawIndexOffsetGetSize", vec![])),
        case("sceAgcDcbDrawIndirect", vec![cb, 0x20, 0x205], vec![], g("sceAgcDcbDrawIndirectGetSize", vec![])),
        case("sceAgcDcbDrawIndexIndirect", vec![cb, 0x20, 0x207], vec![], g("sceAgcDcbDrawIndexIndirectGetSize", vec![])),
        case("sceAgcDcbDrawIndexIndirectMulti", vec![cb, 0, 1, 4, label, 20], vec![0], g("sceAgcDcbDrawIndexIndirectMultiGetSize", vec![])),
        case("sceAgcDcbDrawIndirectMulti", vec![cb, 0, 0, 4, 0, 16], vec![8], g("sceAgcDcbDrawIndirectMultiGetSize", vec![])),
        case("sceAgcDcbDrawIndexMultiInstanced", vec![cb, 3, data, ids, 2, 0], vec![], g("sceAgcDcbDrawIndexMultiInstancedGetSize", vec![])),
        case("sceAgcDcbSetBaseIndirectArgs", vec![cb, 0, label], vec![], g("sceAgcDcbSetBaseDrawIndirectArgsGetSize", vec![])),
        case("sceAgcDcbSetBaseIndirectArgs", vec![cb, 1, label], vec![], g("sceAgcDcbSetBaseDispatchIndirectArgsGetSize", vec![])),
        case("sceAgcDcbSetIndexIndirectArgs", vec![cb, label & !0xF, 0x44], vec![], g("sceAgcDcbSetIndexIndirectArgsGetSize", vec![])),
        case("sceAgcDcbDispatchIndirect", vec![cb, 0, 0], vec![], g("sceAgcDcbDispatchIndirectGetSize", vec![])),
        case("sceAgcAcbDispatchIndirect", vec![cb, label, 0], vec![], g("sceAgcAcbDispatchIndirectGetSize", vec![])),
        case("sceAgcDcbSetShRegisterDirect", vec![cb, packed], vec![], g("sceAgcDcbSetShRegisterDirectGetSize", vec![])),
        case("sceAgcDcbSetUcRegisterDirect", vec![cb, packed], vec![], g("sceAgcDcbSetUcRegisterDirectGetSize", vec![])),
        case("sceAgcDcbSetCxRegisterDirect", vec![cb, packed], vec![], g("sceAgcDcbSetCxRegisterDirectGetSize", vec![])),
        case("sceAgcDcbSetCxRegistersIndirect", vec![cb, regs, 3], vec![], g("sceAgcDcbSetCxRegistersIndirectGetSize", vec![])),
        case("sceAgcDcbSetShRegistersIndirect", vec![cb, regs, 3], vec![], g("sceAgcDcbSetShRegistersIndirectGetSize", vec![])),
        case("sceAgcDcbSetUcRegistersIndirect", vec![cb, regs, 3], vec![], g("sceAgcDcbSetUcRegistersIndirectGetSize", vec![])),
        case("sceAgcCbSetShRegistersDirect", vec![cb, regs, 3], vec![], None),
        case("sceAgcCbSetUcRegistersDirect", vec![cb, regs, 2], vec![], None),
        case("sceAgcCbSetShRegisterRangeDirect", vec![cb, 0x10, data, 4], vec![], g("sceAgcCbSetShRegisterRangeDirectGetSize", vec![4])),
        case("sceAgcCbSetUcRegisterRangeDirect", vec![cb, 0x10, data, 4], vec![], g("sceAgcCbSetUcRegisterRangeDirectGetSize", vec![4])),
        case("sceAgcDcbSetCfRegisterRangeDirect", vec![cb, 0x10, data, 4], vec![], None),
        case("sceAgcDcbContextStateOp_0100", vec![cb, 0], vec![], g("sceAgcDcbContextStateOpGetSize", vec![0])),
        case("sceAgcDcbContextStateOp_0100", vec![cb, 1], vec![], g("sceAgcDcbContextStateOpGetSize", vec![1])),
        case("sceAgcDcbContextStateOp_0100", vec![cb, 2], vec![], g("sceAgcDcbContextStateOpGetSize", vec![2])),
        case("sceAgcDcbContextStateOp_0100", vec![cb, 3], vec![], g("sceAgcDcbContextStateOpGetSize", vec![3])),
    ]
}

#[test]
fn builders_match_get_size_and_parse() {
    let e = Env::new();
    let mut checked_sizes = 0;
    for c in builder_cases(&e) {
        e.reset_cb();
        let before = e.cursor();
        let r = e.call(c.builder, &c.args, &c.stack);
        let after = e.cursor();
        assert_ne!(r, 0, "{} returned null", c.builder);
        assert!((r as u64) >= before && (r as u64) < after, "{} returned 0x{r:x} outside its packets", c.builder);
        parse_range(before, after);
        if let Some((size_fn, size_args)) = &c.size {
            let size = e.call(size_fn, size_args, &[]);
            assert_eq!(size as u64, after - before, "{} emitted {} bytes, {size_fn} says {size}", c.builder, after - before);
            checked_sizes += 1;
        }
    }
    assert!(checked_sizes > 60);
}

/// Every `*GetSize` export is covered by a builder case or is a size-only
/// export whose builder is not exported.
/// Consecutive register offsets share one SET packet; each run is its own
/// packet.
#[test]
fn register_direct_builders_coalesce_runs() {
    let e = Env::new();
    let regs = e.a(DATA + 0x100);
    mem::write_dwords(regs, &[0x10, 0xAA, 0x11, 0xBB, 0x30, 0xCC, 0x31, 0xDD, 0x32, 0xEE]).then_some(()).unwrap();
    let before = e.cursor();
    let r = e.call("sceAgcCbSetShRegistersDirect", &[e.cb(), regs, 5], &[]) as u64;
    assert_eq!(r, before);
    let after = e.cursor();
    assert_eq!(parse_range(before, after), vec![p4::make(4, op::SET_SH_REG, 0), p4::make(5, op::SET_SH_REG, 0)]);
    assert_eq!(mem::read_dwords(before, 9).unwrap(), vec![p4::make(4, op::SET_SH_REG, 0), 0x10, 0xAA, 0xBB, p4::make(5, op::SET_SH_REG, 0), 0x30, 0xCC, 0xDD, 0xEE]);
    assert!(after - before <= e.call("sceAgcCbSetShRegistersDirectGetSize", &[5], &[]) as u64);
    e.reset_cb();
    let before = e.cursor();
    e.call("sceAgcCbSetUcRegistersDirect", &[e.cb(), regs, 3], &[]);
    assert_eq!(parse_range(before, e.cursor()), vec![p4::make(4, op::SET_UCONFIG_REG, 0), p4::make(3, op::SET_UCONFIG_REG, 0)]);
    // An empty list writes nothing and returns the cursor.
    let before = e.cursor();
    assert_eq!(e.call("sceAgcCbSetUcRegistersDirect", &[e.cb(), regs, 0], &[]) as u64, before);
    assert_eq!(e.cursor(), before);
}

#[test]
fn every_get_size_is_checked() {
    let e = Env::new();
    let covered: std::collections::HashSet<&str> = builder_cases(&e).iter().filter_map(|c| c.size.as_ref().map(|s| s.0)).collect();
    let size_only = [
        "sceAgcDcbWaitOnAddressGetSize",
        "sceAgcAcbWaitOnAddressGetSize",
        "sceAgcDcbAtomicGdsGetSize",
        "sceAgcAcbAtomicGdsGetSize",
        "sceAgcDcbQueueEndOfShaderActionGetSize",
        "sceAgcAcbQueueEndOfShaderActionGetSize",
        "sceAgcDcbBeginOcclusionQueryGetSize",
        "sceAgcDcbEndOcclusionQueryGetSize",
        // Upper bounds: runs of consecutive registers share a packet.
        "sceAgcCbSetShRegistersDirectGetSize",
        "sceAgcCbSetUcRegistersDirectGetSize",
    ];
    for (_, name, _, _) in all_exports() {
        if name.ends_with("GetSize") {
            assert!(covered.contains(name) || size_only.contains(&name), "{name} has no consistency case");
        }
    }
    assert_eq!(e.call("sceAgcDcbWaitOnAddressGetSize", &[0], &[]), 56);
    assert_eq!(e.call("sceAgcDcbWaitOnAddressGetSize", &[1], &[]), 64);
    assert_eq!(e.call("sceAgcDcbAtomicGdsGetSize", &[], &[]), 44);
}

#[test]
fn buffer_full_invokes_guest_callback() {
    static GROWN: AtomicU64 = AtomicU64::new(0);
    fn grow(_func: u64, args: [u64; 6]) -> u64 {
        // Move the end cursor out by 0x100 bytes.
        let cb = args[0];
        let down = mem::r64(cb + CB_CURSOR_DOWN).unwrap();
        mem::w64(cb + CB_CURSOR_DOWN, down + 0x100);
        GROWN.fetch_add(1, Ordering::SeqCst);
        1
    }
    let _g = HOOK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let e = Env::new();
    let start = e.a(CMD);
    mem::w64(e.a(CB + CB_CURSOR_DOWN), start + 8);
    mem::w64(e.a(CB + CB_CALLBACK), 0x1234);
    set_guest_call(grow);
    let r = e.call("sceAgcCbNop", &[e.cb(), 4], &[]);
    set_guest_call(|f, a| ps5cpu::exec::call_guest_nested(f, a));
    assert_eq!(r as u64, start);
    assert_eq!(GROWN.load(Ordering::SeqCst), 1);
    assert_eq!(e.cursor(), start + 16);
}

/// A buffer whose cursorDown is 0 (a title sizes the buffer beforehand and
/// leaves the end cursor unset) counts as unbounded, as in the library's
/// unsigned arithmetic: no callback, the packet is written.
#[test]
fn cursor_down_below_cursor_up_never_calls_the_callback() {
    static CALLED: AtomicU64 = AtomicU64::new(0);
    fn fail(_func: u64, _args: [u64; 6]) -> u64 {
        CALLED.fetch_add(1, Ordering::SeqCst);
        0
    }
    let _g = HOOK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let e = Env::new();
    let start = e.a(CMD);
    mem::w64(e.a(CB + CB_CURSOR_DOWN), 0);
    mem::w64(e.a(CB + CB_CALLBACK), 0x1234);
    set_guest_call(fail);
    let r = e.call("sceAgcDcbResetQueue", &[e.cb(), 0x43f, 0], &[]);
    set_guest_call(|f, a| ps5cpu::exec::call_guest_nested(f, a));
    assert_eq!(r as u64, start);
    assert_eq!(CALLED.load(Ordering::SeqCst), 0);
    assert!(e.cursor() > start);
}

// ---------------------------------------------------------------------------
// Interpreter
// ---------------------------------------------------------------------------

/// Serializes tests that touch process-wide hooks.
static HOOK_LOCK: Mutex<()> = Mutex::new(());

fn run_queue(q: &GpuQueue) {
    for _ in 0..1000 {
        let r = q.process_one();
        // What the command processor queued for the recording thread
        // (labels, transfers) has happened once the queue is idle.
        if let Some(ex) = crate::agc::render_executor() {
            ex.drain();
        }
        match r {
            SliceResult::NoWork | SliceResult::AllBlocked => return,
            _ => {}
        }
    }
}

fn submit_cb(e: &Env, q: &GpuQueue) {
    let start = e.a(CMD);
    let dwords = ((e.cursor() - start) / 4) as u32;
    q.submit_graphics(start, dwords);
}

#[test]
fn pm4_parses_builder_stream_into_register_banks() {
    let e = Env::new();
    let regs = e.a(DATA + 0x100);
    mem::write_dwords(regs, &[0x20, 0x11, 0x21, 0x22]).then_some(()).unwrap();
    let vals = e.a(DATA);
    mem::write_dwords(vals, &[9, 8, 7]).then_some(()).unwrap();
    e.call("sceAgcDcbSetCxRegisterDirect", &[e.cb(), (0xABCDu64 << 32) | 0x5], &[]);
    e.call("sceAgcDcbSetShRegisterDirect", &[e.cb(), (0x77u64 << 32) | 0x48], &[]);
    e.call("sceAgcCbSetShRegisterRangeDirect", &[e.cb(), 0x100, vals, 3], &[]);
    e.call("sceAgcDcbSetCxRegistersIndirect", &[e.cb(), regs, 2], &[]);
    e.call("sceAgcDcbSetIndexSize", &[e.cb(), 1, 0], &[]);
    e.call("sceAgcDcbDrawIndexAuto", &[e.cb(), 3, 0x4000_0000], &[]);
    e.call("sceAgcDcbSetIndexBuffer", &[e.cb(), vals], &[]);
    e.call("sceAgcDcbDrawIndex", &[e.cb(), 6, vals, 0x4000_0000], &[]);
    e.call("sceAgcCbDispatch", &[e.cb(), 1, 1, 1, 0], &[]);
    let q = GpuQueue::new(1);
    submit_cb(&e, &q);
    run_queue(&q);
    assert_eq!(q.pending(), 0);
    q.with_interpreter(0, |i: &mut Interpreter| {
        assert_eq!(i.regs.context[5], 0xABCD);
        assert_eq!(i.regs.context[0x20], 0x11);
        assert_eq!(i.regs.context[0x21], 0x22);
        assert_eq!(i.regs.shader[0x48], 0x77);
        assert_eq!(&i.regs.shader[0x100..0x103], &[9, 8, 7]);
        assert_eq!(i.index_type & 3, 1);
        assert_eq!(i.draws, 2);
        assert_eq!(i.dispatches, 1);
        assert_eq!(i.index_base, vals);
    });
}

/// Register tables captured from the test title (render-target and depth
/// groups built from its register templates once AGC init resolved them):
/// every entry lands on its context register.
#[test]
fn pm4_captured_title_register_tables() {
    let e = Env::new();
    #[rustfmt::skip]
    let depth: [u32; 32] = [
        0x10, 0xa000_0183, 0x11, 0x2000_0181, 0x12, 0x5010_a600, 0x13, 0x5010_a800,
        0x14, 0x5010_a600, 0x15, 0x5010_a800, 0x1a, 0, 0x1b, 0,
        0x1c, 0, 0x1d, 0, 0x1e, 0, 0x02, 0,
        0x05, 0x5010_a700, 0x07, 0, 0x0b, 0, 0x0a, 0,
    ];
    // Slot 1 colour target (CB_COLOR1_*: slot stride 0xF, *_EXT/ATTRIB2/3 stride 1).
    #[rustfmt::skip]
    let color1: [u32; 32] = [
        0x327, 0x5020_9500, 0x32a, 0, 0x32b, 0x1004_0730, 0x32c, 0,
        0x32d, 0x0018_0028, 0x32e, 0, 0x330, 0, 0x332, 0,
        0x333, 0, 0x334, 0, 0x391, 0, 0x399, 0,
        0x3a1, 0, 0x3a9, 0, 0x3b1, 0x00ff_c3ff, 0x3b9, 0x4dc6_c000,
    ];
    let t0 = e.a(DATA);
    let t1 = e.a(DATA + 0x100);
    mem::write_dwords(t0, &depth).then_some(()).unwrap();
    mem::write_dwords(t1, &color1).then_some(()).unwrap();
    e.call("sceAgcDcbSetCxRegistersIndirect", &[e.cb(), t0, 16], &[]);
    e.call("sceAgcDcbSetCxRegistersIndirect", &[e.cb(), t1, 16], &[]);
    let q = GpuQueue::new(1);
    submit_cb(&e, &q);
    run_queue(&q);
    q.with_interpreter(0, |i: &mut Interpreter| {
        for t in [&depth[..], &color1[..]] {
            for p in t.chunks_exact(2) {
                assert_eq!(i.regs.context[p[0] as usize], p[1], "register 0x{:x}", p[0]);
            }
        }
    });
}

/// Templates left unresolved (AGC init skipped) carry the 0x1000_0000
/// placeholder plus the slot stride, as captured before the fix: such
/// entries name no register and must not clobber the low context bank.
#[test]
fn pm4_register_table_selector_bits_are_stripped() {
    let e = Env::new();
    #[rustfmt::skip]
    let table: [u32; 12] = [
        0x1000_0000, 0x5030_2700, 0x1000_0000, 0x1004_0730, 0x1000_000f, 0x5025_1300,
        0x1000_001e, 0x5027_1700, 0xFFFF_FFFF, 0xFFFF_FFFF, 0x0318, 0x5010_9e00,
    ];
    let t = e.a(DATA);
    mem::write_dwords(t, &table).then_some(()).unwrap();
    e.call("sceAgcDcbSetCxRegistersIndirect", &[e.cb(), t, 6], &[]);
    let q = GpuQueue::new(1);
    submit_cb(&e, &q);
    run_queue(&q);
    q.with_interpreter(0, |i: &mut Interpreter| {
        // Bits 28..30 are selectors: the entry names the register below them.
        assert_eq!(i.regs.context[0], 0x1004_0730);
        assert_eq!(i.regs.context[0xf], 0x5025_1300);
        assert_eq!(i.regs.context[0x1e], 0x5027_1700);
        assert_eq!(i.regs.context[0x318], 0x5010_9e00);
    });
}

#[test]
fn pm4_label_writes_and_eop_interrupt() {
    let e = Env::new();
    let label = e.a(LABELS);
    let label64 = e.a(LABELS + 0x10);
    let data = e.a(DATA);
    mem::write_dwords(data, &[0xDEAD_BEEF, 0x1234]).then_some(()).unwrap();
    // WRITE_DATA (private form) of two dwords.
    e.call("sceAgcDcbWriteData", &[e.cb(), 5, 0, label + 0x40, data, 2], &[0, 0]);
    // RELEASE_MEM 64-bit with interrupt (selector 2), context id 0x55.
    e.call("sceAgcCbReleaseMem", &[e.cb(), 0x28, 0, 1, 0, label64], &[2, 0x1122_3344_5566_7788, 0, 0, 2, 0x55]);
    // 32-bit release without interrupt.
    e.call("sceAgcCbReleaseMem", &[e.cb(), 0x28, 0, 1, 0, label], &[1, 0x99, 0, 0, 0, 0]);
    // Native EVENT_WRITE_EOP: data_sel 1, int_sel 0.
    let eop = [p4::make(6, op::EVENT_WRITE_EOP, 0), 0x04 | (5 << 8), (label as u32) + 8, ((label >> 32) as u32 & 0xFFFF) | (1 << 29), 0x4242, 0];
    let a = alloc_dwords(e.cb(), 6).unwrap();
    mem::write_dwords(a, &eop).then_some(()).unwrap();

    let eq = crate::sync::equeue::create();
    assert!(crate::sync::equeue::register_event(eq, 0, crate::sync::equeue::FILTER_GRAPHICS, 0xABC, crate::sync::equeue::EV_CLEAR, 0));
    let q = GpuQueue::new(1);
    submit_cb(&e, &q);
    run_queue(&q);
    // WRITE_DATA lands when the command processor reaches it.
    assert_eq!(mem::r32(label + 0x40), Some(0xDEAD_BEEF));
    assert_eq!(mem::r32(label + 0x44), Some(0x1234));
    // End-of-pipe labels land once the work before them completed.
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    while mem::r32(label + 8) != Some(0x4242) && std::time::Instant::now() < deadline {
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
    assert_eq!(mem::r64(label64), Some(0x1122_3344_5566_7788));
    assert_eq!(mem::r32(label), Some(0x99));
    assert_eq!(mem::r32(label + 8), Some(0x4242));
    // Interrupts are delivered once the GPU timeline retires the work, on
    // another thread: wait for it with a bound.
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    while crate::sync::equeue::pending_count(eq) == 0 && std::time::Instant::now() < deadline {
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
    assert!(crate::sync::equeue::pending_count(eq) >= 1, "graphics interrupt not delivered");
    crate::sync::equeue::delete(eq);
}

#[test]
fn pm4_wait_reg_mem_suspends_until_label() {
    let e = Env::new();
    let label = e.a(LABELS + 0x20);
    let out = e.a(LABELS + 0x30);
    mem::w32(label, 0);
    let one = e.a(DATA);
    mem::w32(one, 7);
    // Wait until *label == 1, then write 7 to out.
    e.call("sceAgcDcbWaitRegMem", &[e.cb(), 0, 3, 0, 0, label], &[1, 0xFFFF_FFFF, 16]);
    e.call("sceAgcDcbWriteData", &[e.cb(), 5, 0, out, one, 1], &[0, 0]);
    let q = GpuQueue::new(1);
    submit_cb(&e, &q);
    run_queue(&q);
    assert_eq!(q.pending(), 1, "wait should block");
    assert_eq!(mem::r32(out), Some(0));
    mem::w32(label, 1);
    // Blocked heads are retried after an unblock (worker does it on a timer).
    q.retry_blocked();
    run_queue(&q);
    assert_eq!(q.pending(), 0);
    assert_eq!(mem::r32(out), Some(7));
}

#[test]
fn pm4_indirect_buffer_cond_exec_and_atomics() {
    let e = Env::new();
    let label = e.a(LABELS + 0x40);
    let pred = e.a(LABELS + 0x48);
    let ctr = e.a(LABELS + 0x50);
    let v = e.a(DATA);
    mem::w32(v, 5);
    mem::w32(pred, 0);
    mem::w32(ctr, 10);
    // Child buffer at CMD_END-0x400: one WRITE_DATA of 5 to label.
    let child = e.a(CMD_END - 0x400);
    let child_words = [p4::make(5, op::WRITE_DATA, 0), 0x500, label as u32, (label >> 32) as u32, 5];
    mem::write_dwords(child, &child_words).then_some(()).unwrap();
    e.call("sceAgcDcbJump", &[e.cb(), 0, 0, child, 5], &[]); // call
    // COND_EXEC with zero predicate skips the next 5 dwords (an atomic add would follow).
    e.call("sceAgcDcbCondExec", &[e.cb(), pred, 9], &[]);
    e.call("sceAgcDcbAtomicMem", &[e.cb(), 15, 0, 0, ctr, 100], &[0, 0]); // skipped (9 dwords)
    e.call("sceAgcDcbAtomicMem", &[e.cb(), 15, 0, 0, ctr, 1], &[0, 0]); // executed: 10 + 1
    let q = GpuQueue::new(1);
    submit_cb(&e, &q);
    run_queue(&q);
    assert_eq!(q.pending(), 0);
    assert_eq!(mem::r32(label), Some(5));
    assert_eq!(mem::r32(ctr), Some(11));
}

static FLIP_CALLS: Mutex<Vec<(i32, i32, u32, i64)>> = Mutex::new(Vec::new());
static FLIP_DONE: AtomicU64 = AtomicU64::new(0);

fn test_flip(h: i32, i: i32, m: u32, a: i64) -> Result<(), i64> {
    FLIP_CALLS.lock().unwrap().push((h, i, m, a));
    Ok(())
}

fn test_flip_done(_h: i32, _i: i32) -> bool {
    FLIP_DONE.load(Ordering::SeqCst) != 0
}

#[test]
fn flip_packets_route_through_hooks() {
    let _g = HOOK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    set_flip_hooks(test_flip, test_flip_done);
    FLIP_CALLS.lock().unwrap().clear();
    FLIP_DONE.store(0, Ordering::SeqCst);
    let e = Env::new();
    let out = e.a(LABELS + 0x60);
    let v = e.a(DATA);
    mem::w32(v, 3);
    e.call("sceAgcDcbSetFlip", &[e.cb(), 1, 2, 1, 0x1_0000_0007], &[]);
    e.call("sceAgcDcbWaitUntilSafeForRendering", &[e.cb(), 1, 2], &[]);
    e.call("sceAgcDcbWriteData", &[e.cb(), 5, 0, out, v, 1], &[0, 0]);
    let q = GpuQueue::new(1);
    submit_cb(&e, &q);
    run_queue(&q);
    assert_eq!(FLIP_CALLS.lock().unwrap().as_slice(), &[(1, 2, 1, 0x1_0000_0007)]);
    assert_eq!(q.pending(), 1, "WaitFlipDone should block while the buffer is in use");
    assert_eq!(mem::r32(out), Some(0));
    FLIP_DONE.store(1, Ordering::SeqCst);
    q.retry_blocked();
    run_queue(&q);
    assert_eq!(q.pending(), 0);
    assert_eq!(mem::r32(out), Some(3));
    // The flip was not re-issued when the submission resumed.
    assert_eq!(FLIP_CALLS.lock().unwrap().len(), 1);
    set_flip_hooks(crate::videoout::gpu_flip, crate::videoout::is_flip_done);
}

/// A flip packet raises the graphics end-of-pipe event once the work
/// before it completed (a title waiting on the event after every flip,
/// with no interrupt requested by its releases, keeps presenting).
#[test]
fn flip_raises_the_graphics_end_of_pipe_event() {
    let _g = HOOK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    set_flip_hooks(test_flip, test_flip_done);
    FLIP_CALLS.lock().unwrap().clear();
    let e = Env::new();
    let eq = crate::sync::equeue::create();
    assert!(crate::sync::equeue::register_event(eq, 0, crate::sync::equeue::FILTER_GRAPHICS, 0xF11, crate::sync::equeue::EV_CLEAR, 0));
    let q = bare_queue();
    run_words(&q, e.a(CMD + 0x800), &[p4::make(7, op::NOP, nop::FLIP), 7, -2i32 as u32, 1, 0, 0, 0]);
    assert_eq!(FLIP_CALLS.lock().unwrap().as_slice(), &[(7, -2, 1, 0)]);
    assert!(crate::sync::equeue::pending_count(eq) >= 1, "the flip raised no graphics event");
    crate::sync::equeue::delete(eq);
    set_flip_hooks(crate::videoout::gpu_flip, crate::videoout::is_flip_done);
}

/// A queue that delivered a release interrupt raises no end-of-pipe event
/// for its next four flips; the fifth clears the latch and raises it. The
/// latched queue still delivers its own releases (reference
/// agc_driver_eop_interrupts).
#[test]
fn flip_event_is_latched_off_by_a_queue_release_interrupt() {
    use crate::sync::equeue::{self as eqm, EV_CLEAR, FILTER_GRAPHICS};
    let _g = HOOK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    set_flip_hooks(test_flip, test_flip_done);
    let e = Env::new();
    // An ident no other test raises.
    const IDENT: u64 = 0x7F3;
    let eq = eqm::create();
    let fresh = || {
        eqm::delete_registered_event(eq, IDENT, FILTER_GRAPHICS);
        assert!(eqm::register_event(eq, IDENT, FILTER_GRAPHICS, 0, EV_CLEAR, 0));
    };
    let q = bare_queue();
    q.with_interpreter(0, |i| i.event_id = IDENT);
    let flip = [p4::make(7, op::NOP, nop::FLIP), 7, -2i32 as u32, 1, 0, 0, 0];
    // RELEASE_MEM: interrupt only (data selector 0, INT_SEL 1).
    let release = [p4::make(8, op::RELEASE_MEM, 0), 0x28 | (5 << 8), 1 << 24, 0, 0, 0, 0, 0];
    fresh();
    run_words(&q, e.a(CMD), &flip);
    assert_eq!(eqm::pending_count(eq), 1, "an unlatched flip raises the event");
    fresh();
    run_words(&q, e.a(CMD), &release);
    assert_eq!(eqm::pending_count(eq), 1, "the release interrupt is delivered");
    for n in 1..=4 {
        fresh();
        run_words(&q, e.a(CMD), &flip);
        assert_eq!(eqm::pending_count(eq), 0, "latched flip {n} raised the event");
    }
    fresh();
    run_words(&q, e.a(CMD), &flip);
    assert_eq!(eqm::pending_count(eq), 1, "the fifth flip clears the latch");
    fresh();
    run_words(&q, e.a(CMD), &flip);
    assert_eq!(eqm::pending_count(eq), 1, "the latch stays clear");
    // A release followed by two flips raises none for the flips.
    fresh();
    run_words(&q, e.a(CMD), &release);
    fresh();
    run_words(&q, e.a(CMD), &flip);
    run_words(&q, e.a(CMD), &flip);
    assert_eq!(eqm::pending_count(eq), 0);
    eqm::delete(eq);
    set_flip_hooks(crate::videoout::gpu_flip, crate::videoout::is_flip_done);
}

/// Async rewind: the ACB emitter adds the offload bit, a failed allocation
/// leaves the buffer alone, the state patch only checks the opcode, and a
/// pending rewind holds the queue until it is patched valid.
#[test]
fn rewind_async_emit_patch_and_resume() {
    let e = Env::new();
    const SENTINEL: u32 = 0x1234_5678;
    let start = e.a(CMD);
    for state in [0u64, 1] {
        for offload in [0u64, 1] {
            mem::write_dwords(start, &[SENTINEL; 4]);
            mem::w64(e.a(CB + CB_CURSOR_UP), start + 4);
            mem::w64(e.a(CB + CB_CURSOR_DOWN), start + 12);
            let p = e.call("sceAgcAcbRewind", &[e.cb(), state, offload], &[]) as u64;
            assert_eq!(p, start + 4);
            assert_eq!(e.cursor(), start + 12);
            let want = [SENTINEL, 0xC000_5900, (state as u32) << 31 | (offload as u32) << 24, SENTINEL];
            for (i, w) in want.iter().enumerate() {
                assert_eq!(mem::r32(start + 4 * i as u64), Some(*w), "state {state} offload {offload} dword {i}");
            }
        }
    }
    // Bounded allocation: one reserved dword leaves no room for the packet.
    mem::w64(e.a(CB + CB_CURSOR_UP), start + 4);
    mem::w32(e.a(CB + CB_RESERVED_DW), 1);
    let before: Vec<_> = (0..4).map(|i| mem::r32(start + 4 * i).unwrap()).collect();
    assert_eq!(e.call("sceAgcAcbRewind", &[e.cb(), 1, 1], &[]), 0);
    assert_eq!(e.call("sceAgcAcbRewind", &[0, 1, 1], &[]), 0);
    assert_eq!(e.cursor(), start + 4);
    assert_eq!((0..4).map(|i| mem::r32(start + 4 * i).unwrap()).collect::<Vec<_>>(), before);
    e.reset_cb();

    // The patch keeps the header and the low 31 bits, offload included.
    let cmd = e.a(DATA);
    mem::write_dwords(cmd, &[0xA5FF_591F, 0x0155_AA55]);
    assert_eq!(e.call("sceAgcAsyncRewindPatchSetRewindState", &[cmd, 1], &[]), 0);
    assert_eq!((mem::r32(cmd), mem::r32(cmd + 4)), (Some(0xA5FF_591F), Some(0x8155_AA55)));
    assert_eq!(e.call("sceAgcAsyncRewindPatchSetRewindState", &[cmd, 0], &[]), 0);
    assert_eq!((mem::r32(cmd), mem::r32(cmd + 4)), (Some(0xA5FF_591F), Some(0x0155_AA55)));
    mem::w32(cmd, 0xC000_1000);
    assert_eq!(e.call("sceAgcAsyncRewindPatchSetRewindState", &[cmd, 1], &[]), ERR_INVALID_PACKET);
    assert_eq!((mem::r32(cmd), mem::r32(cmd + 4)), (Some(0xC000_1000), Some(0x0155_AA55)));

    // A pending rewind blocks the queue until the patch makes it valid.
    for (emitter, patch, offload) in [
        ("sceAgcDcbRewind", "sceAgcRewindPatchSetRewindState", 0u64),
        ("sceAgcAcbRewind", "sceAgcAsyncRewindPatchSetRewindState", 0),
        ("sceAgcAcbRewind", "sceAgcAsyncRewindPatchSetRewindState", 1),
    ] {
        e.reset_cb();
        let out = e.a(LABELS + 0x40);
        let value = e.a(DATA + 0x40);
        mem::w32(out, 0);
        mem::w32(value, 33);
        let r = e.call(emitter, &[e.cb(), 0, offload], &[]) as u64;
        assert_eq!(r, start);
        e.call("sceAgcDcbWriteData", &[e.cb(), 5, 0, out, value, 1], &[0, 0]);
        let q = GpuQueue::new(1);
        submit_cb(&e, &q);
        run_queue(&q);
        assert_eq!((q.pending(), mem::r32(out)), (1, Some(0)), "{emitter}: pending rewind blocks");
        q.retry_blocked();
        run_queue(&q);
        assert_eq!((q.pending(), mem::r32(out)), (1, Some(0)), "{emitter}: still pending before the patch");
        assert_eq!(e.call(patch, &[r, 1], &[]), 0);
        assert_eq!(mem::r32(r + 4), Some(0x8000_0000 | (offload as u32) << 24));
        q.retry_blocked();
        run_queue(&q);
        assert_eq!((q.pending(), mem::r32(out)), (0, Some(33)), "{emitter}: resumes after the patch");
    }
}

#[test]
fn patch_helpers_edit_packets() {
    let e = Env::new();
    let label = e.a(LABELS);
    let w = e.call("sceAgcDcbWaitRegMem", &[e.cb(), 0, 3, 0, 0, label], &[1, 0xFF, 16]) as u64;
    assert_eq!(e.call("sceAgcWaitRegMemPatchReference", &[w, 0x55], &[]), 0);
    assert_eq!(e.call("sceAgcWaitRegMemPatchCompareFunction", &[w, 4], &[]), 0);
    assert_eq!(e.call("sceAgcWaitRegMemPatchAddress", &[w, label + 0x10], &[]), 0);
    // Begin tag (4 dwords), then the native wait.
    let wait = w + 16;
    assert_eq!(mem::r32(wait + 16), Some(0x55));
    assert_eq!(mem::r32(wait + 4).unwrap() & 7, 4);
    assert_eq!(mem::r32(wait + 8), Some((label + 0x10) as u32));
    assert_eq!(mem::r32(w + 12), Some((label + 0x10) as u32));
    let r = e.call("sceAgcCbReleaseMem", &[e.cb(), 0x28, 0, 1, 0, label], &[1, 1, 0, 0, 0, 0]) as u64;
    assert_eq!(e.call("sceAgcQueueEndOfPipeActionPatchData", &[r, 0x77], &[]), 0);
    assert_eq!(e.call("sceAgcQueueEndOfPipeActionPatchType", &[r, 2], &[]), 0);
    assert_eq!(mem::r64(r + 20), Some(0x77));
    assert_eq!(mem::r32(r + 8).unwrap() >> 29, 2);
    let c = e.call("sceAgcDcbCondExec", &[e.cb(), label, 0], &[]) as u64;
    assert_eq!(e.call("sceAgcCondExecPatchSetEnd", &[c, c + 20 + 12], &[]), 0);
    assert_eq!(mem::r32(c + 16), Some(3));
    assert_eq!(e.call("sceAgcGetPacketSize", &[c], &[]), 5);
    let ind = e.call("sceAgcDcbSetShRegistersIndirect", &[e.cb(), label, 2], &[]) as u64;
    assert_eq!(e.call("sceAgcSetShRegIndirectPatchAddRegisters", &[ind, 3], &[]), 0);
    assert_eq!(mem::r32(ind + 16), Some(5));
    assert_eq!(e.call("sceAgcSetShRegIndirectPatchSetNumRegisters", &[ind, 1], &[]), 0);
    assert_eq!(mem::r32(ind + 16), Some(1));
    assert_eq!(e.call("sceAgcSetPacketPredication", &[ind, 1], &[]), 0);
    assert_eq!(mem::r32(ind).unwrap() & 1, 1);
}

/// The asynchronous WRITE_DATA setters: the destination is the compute
/// DST_SEL (bits 8..11), values outside their field are refused without
/// changing the packet, destination 0 with write confirmation is refused,
/// and a packet of another opcode is refused.
#[test]
fn async_write_data_patch_setters() {
    let e = Env::new();
    let label = e.a(LABELS);
    let data = e.a(DATA);
    mem::write_dwords(data, &[0x1111_2222]).then_some(()).unwrap();
    // ACB WRITE_DATA to DST_SEL 1 with write confirmation (the builder keeps
    // the confirmation only for a non-zero destination).
    let w = e.call("sceAgcAcbWriteData", &[e.cb(), 1, 0, label, data, 1], &[0, 1]) as u64;
    let control = || mem::r32(w + 4).unwrap();
    assert_eq!((control() >> 8) & 0xF, 1);
    assert_ne!(control() & (1 << 20), 0);

    let rest = control() & !(0xF << 8);
    assert_eq!(e.call("sceAgcAsyncWriteDataPatchSetDst", &[w, 5], &[]), 0);
    assert_eq!((control() >> 8) & 0xF, 5);
    assert_eq!(control() & !(0xF << 8), rest, "only DST_SEL changes");
    assert_eq!(e.call("sceAgcAsyncWriteDataPatchSetDst", &[w, 0x10], &[]), ERR_INVALID_ARGUMENT);
    assert_eq!((control() >> 8) & 0xF, 5, "a refused value changes nothing");
    assert_eq!(e.call("sceAgcAsyncWriteDataPatchSetDst", &[w, 0], &[]), ERR_INVALID_ARGUMENT);
    assert_eq!((control() >> 8) & 0xF, 5);

    assert_eq!(e.call("sceAgcAsyncWriteDataPatchSetCachePolicy", &[w, 3], &[]), 0);
    assert_eq!((control() >> 25) & 3, 3);
    assert_eq!(e.call("sceAgcAsyncWriteDataPatchSetCachePolicy", &[w, 4], &[]), ERR_INVALID_ARGUMENT);
    assert_eq!((control() >> 25) & 3, 3);

    assert_eq!(e.call("sceAgcAsyncWriteDataPatchSetAddressOrOffset", &[w, 0x1_2345_6780], &[]), 0);
    assert_eq!(mem::r32(w + 8), Some(0x2345_6780));
    assert_eq!(mem::r32(w + 12), Some(1));

    let wait = e.call("sceAgcDcbWaitRegMem", &[e.cb(), 0, 3, 0, 0, label], &[1, 0xFF, 16]) as u64;
    assert_eq!(e.call("sceAgcAsyncWriteDataPatchSetDst", &[wait, 1], &[]), ERR_INVALID_PACKET);
    assert_eq!(e.call("sceAgcAsyncWriteDataPatchSetCachePolicy", &[wait, 1], &[]), ERR_INVALID_PACKET);
    assert_eq!(e.call("sceAgcAsyncWriteDataPatchSetAddressOrOffset", &[wait, 0], &[]), ERR_INVALID_PACKET);
}

#[test]
fn register_defaults_blob_layout() {
    for v in 0..=13 {
        let (public, internal) = defaults::select(v).unwrap();
        for d in [public, internal] {
            let base = 0x1000_0000u64;
            let blob = defaults::build_blob(d, base).unwrap();
            let r32 = |o: usize| u32::from_le_bytes(blob[o..o + 4].try_into().unwrap());
            let r64 = |o: usize| u64::from_le_bytes(blob[o..o + 8].try_into().unwrap());
            let types = r64(0x30);
            assert!(types >= base && ((types - base) as usize) < blob.len());
            assert_eq!(r32(0x38) as usize * 3 * 4 + (types - base) as usize, blob.len());
            for t in 0..4 {
                let count = r32(0x20 + t * 4);
                let table = r64(t * 8);
                assert_eq!(table == 0, count == 0);
            }
        }
    }
    // Later versions fall back to the newest.
    let (p13, i13) = defaults::select(13).unwrap();
    let (p99, i99) = defaults::select(99).unwrap();
    assert!(std::ptr::eq(p13, p99) && std::ptr::eq(i13, i99));
    let reg = |t: &[u32], off: u32| t.chunks_exact(2).find(|e| e[0] == off).map(|e| e[1]);
    // Version 11 and later force the table-0 bit 16 of register 0x201.
    for v in [11, 12, 13] {
        assert_eq!(reg(defaults::select(v).unwrap().0.t0, 0x201).unwrap() & 0x10000, 0x10000, "v{v}");
    }
    assert_eq!(reg(defaults::select(10).unwrap().0.t0, 0x201).unwrap() & 0x10000, 0);
    // Version 13's internal table 1 differs from version 11's in two values.
    assert_eq!(reg(i13.t1, 0x81), Some(1));
    assert_eq!(reg(i13.t1, 0x101), Some(3));
    assert_eq!(reg(defaults::select(11).unwrap().1.t1, 0x81), Some(0));
    // Versions 0..3 share one dataset, 5 and 6 another.
    assert!(std::ptr::eq(defaults::select(0).unwrap().0, defaults::select(3).unwrap().0));
    assert!(std::ptr::eq(defaults::select(5).unwrap().1, defaults::select(6).unwrap().1));
}

#[test]
fn interpreter_event_ids() {
    assert_eq!(Interpreter::new(0).event_id, 0);
    assert_eq!(Interpreter::new(1).event_id, interp::COMPUTE_QUEUE_BASE);
    assert_eq!(Interpreter::new(56).event_id, 0x57);
}

// ---------------------------------------------------------------------------
// Draw state resolution (PM4 -> render executor)
// ---------------------------------------------------------------------------

#[derive(Default)]
struct Recorder {
    draws: Mutex<Vec<ps5gpu::render::ResolvedDraw>>,
    dispatches: Mutex<Vec<ps5gpu::render::ResolvedDispatch>>,
    eops: AtomicU64,
    fills: Mutex<Vec<(u64, u64, u32)>>,
}

impl ps5gpu::render::RenderExecutor for Recorder {
    fn draw(&self, d: &ps5gpu::render::ResolvedDraw) {
        self.draws.lock().unwrap().push(d.clone());
    }
    fn dispatch(&self, d: &ps5gpu::render::ResolvedDispatch) {
        self.dispatches.lock().unwrap().push(*d);
    }
    fn end_of_pipe(&self, done: Box<dyn FnOnce() + Send>) {
        self.eops.fetch_add(1, Ordering::SeqCst);
        done();
    }
    fn memory_filled(&self, addr: u64, size: u64, value: u32) {
        self.fills.lock().unwrap().push((addr, size, value));
    }
}

#[test]
fn pm4_draw_state_resolves_from_builder_stream() {
    use ps5gpu::ash::vk;
    use ps5gpu::render::{regs::cx, regs::sh, regs::uc, DrawKind, IndexType};
    let e = Env::new();
    let cx_reg = |off: u32, v: u32| {
        e.call("sceAgcDcbSetCxRegisterDirect", &[e.cb(), ((v as u64) << 32) | off as u64], &[]);
    };
    let sh_reg = |off: u32, v: u32| {
        e.call("sceAgcDcbSetShRegisterDirect", &[e.cb(), ((v as u64) << 32) | off as u64], &[]);
    };
    let uc_reg = |off: u32, v: u32| {
        e.call("sceAgcDcbSetUcRegisterDirect", &[e.cb(), ((v as u64) << 32) | off as u64], &[]);
    };
    let (w, h) = (1920u32, 1080u32);
    // 64 KiB-aligned guest addresses (never dereferenced).
    let color = 0x12_3450_0000u64;
    let dcc = 0x12_3460_0000u64;
    let depth = 0x12_3470_0000u64;
    let ps = 0x9_8765_4300u64;
    let gs = 0x9_8765_8800u64;
    // Color target 0: 8_8_8_8 SRGB, alternate order, R_X, DCC.
    cx_reg(cx::CB_COLOR0_BASE, (color >> 8) as u32);
    cx_reg(cx::CB_COLOR0_BASE_EXT, (color >> 40) as u32);
    cx_reg(cx::CB_COLOR0_INFO, (10 << 2) | (6 << 8) | (1 << 11) | (1 << 28));
    cx_reg(cx::CB_COLOR0_ATTRIB2, (h - 1) | ((w - 1) << 14));
    cx_reg(cx::CB_COLOR0_ATTRIB3, (27 << 14) | (1 << 24));
    cx_reg(cx::CB_COLOR0_DCC_BASE, (dcc >> 8) as u32);
    cx_reg(cx::CB_COLOR0_CLEAR_WORD0, 0xFF20_4060);
    cx_reg(cx::CB_TARGET_MASK, 0xF);
    cx_reg(cx::CB_SHADER_MASK, 0xF);
    cx_reg(cx::CB_COLOR_CONTROL, (1 << 4) | (0xCC << 16));
    cx_reg(cx::CB_BLEND0_CONTROL, (1 << 30) | 4 | (5 << 8));
    // Depth: Z32F, test + write, LESS_EQUAL, clear to 1.0.
    cx_reg(cx::DB_Z_INFO, 3);
    cx_reg(cx::DB_Z_READ_BASE, (depth >> 8) as u32);
    cx_reg(cx::DB_Z_WRITE_BASE, (depth >> 8) as u32);
    cx_reg(cx::DB_DEPTH_SIZE_XY, (w - 1) | ((h - 1) << 16));
    cx_reg(cx::DB_DEPTH_CONTROL, 2 | 4 | (3 << 4));
    cx_reg(cx::DB_RENDER_CONTROL, 1);
    cx_reg(cx::DB_DEPTH_CLEAR, 1.0f32.to_bits());
    // Viewport 0 covering the target (Y flipped), DX clip space.
    for (i, v) in [w as f32 / 2.0, w as f32 / 2.0, -(h as f32) / 2.0, h as f32 / 2.0, 1.0, 0.0].iter().enumerate() {
        cx_reg(cx::PA_CL_VPORT_XSCALE + i as u32, v.to_bits());
    }
    cx_reg(cx::PA_CL_CLIP_CNTL, 1 << 19);
    cx_reg(cx::PA_SC_SCREEN_SCISSOR_TL, 0);
    cx_reg(cx::PA_SC_SCREEN_SCISSOR_BR, w | (h << 16));
    cx_reg(cx::PA_SC_WINDOW_SCISSOR_TL, 0x8000_0000);
    cx_reg(cx::PA_SC_WINDOW_SCISSOR_BR, 0x4000 | (0x4000 << 16));
    cx_reg(cx::PA_SC_GENERIC_SCISSOR_TL, 0x8000_0000 | 16 | (8 << 16));
    cx_reg(cx::PA_SC_GENERIC_SCISSOR_BR, 1000 | (500 << 16));
    cx_reg(cx::PA_SU_SC_MODE_CNTL, 2 | 4);
    cx_reg(cx::VGT_SHADER_STAGES_EN, 0x0200_2000);
    // Shaders: PS with 3 user SGPRs, GS (primitive shader).
    sh_reg(sh::SPI_SHADER_PGM_LO_PS, (ps >> 8) as u32);
    sh_reg(sh::SPI_SHADER_PGM_LO_PS + 1, (ps >> 40) as u32);
    sh_reg(sh::SPI_SHADER_PGM_RSRC1_PS + 1, 3 << 1);
    for i in 0..3 {
        sh_reg(sh::SPI_SHADER_USER_DATA_PS_0 + i, 0x100 + i);
    }
    sh_reg(sh::SPI_SHADER_PGM_LO_GS, (gs >> 8) as u32);
    sh_reg(sh::SPI_SHADER_PGM_LO_GS + 1, (gs >> 40) as u32);
    sh_reg(sh::SPI_SHADER_USER_DATA_GS_0, 0xAAAA);
    uc_reg(uc::VGT_PRIMITIVE_TYPE, 4);
    // Compute shader for a dispatch.
    sh_reg(sh::COMPUTE_PGM_LO, 0x1234);
    sh_reg(sh::COMPUTE_NUM_THREAD_X, 64);
    sh_reg(sh::COMPUTE_USER_DATA_0 + 2, 0x55);
    e.call("sceAgcDcbSetIndexSize", &[e.cb(), 1, 0], &[]);
    e.call("sceAgcDcbSetNumInstances", &[e.cb(), 2], &[]);
    let indices = e.a(DATA + 0x800);
    e.call("sceAgcDcbDrawIndex", &[e.cb(), 36, indices, 0], &[]);
    e.call("sceAgcDcbDrawIndexAuto", &[e.cb(), 3, 0], &[]);
    e.call("sceAgcCbDispatch", &[e.cb(), 8, 4, 1, 0], &[]);
    // A label with an interrupt goes through end of pipe.
    e.call("sceAgcCbReleaseMem", &[e.cb(), 0x28, 0, 1, 0, e.a(LABELS)], &[2, 7, 0, 0, 2, 0x1]);

    let rec = std::sync::Arc::new(Recorder::default());
    let q = GpuQueue::new(1);
    let ex: std::sync::Arc<dyn ps5gpu::render::RenderExecutor> = rec.clone();
    q.with_interpreter(0, |i| i.executor = Some(ex));
    submit_cb(&e, &q);
    run_queue(&q);
    assert_eq!(q.pending(), 0);

    let draws = rec.draws.lock().unwrap();
    assert_eq!(draws.len(), 2);
    let d = &draws[0];
    assert_eq!(d.packet.kind, DrawKind::Index);
    assert_eq!(d.packet.count, 36);
    assert_eq!(d.packet.instance_count, 2);
    let ib = d.index.unwrap();
    assert_eq!((ib.address, ib.kind, ib.count), (indices, IndexType::U32, 36));
    assert_eq!(d.primitive_type, 4);
    assert_eq!(d.topology, Some(vk::PrimitiveTopology::TRIANGLE_LIST));
    assert_eq!(d.color_targets.len(), 1);
    let c = &d.color_targets[0];
    assert_eq!(c.base, color);
    assert_eq!((c.width, c.height), (w, h));
    assert_eq!(c.format.host, vk::Format::B8G8R8A8_SRGB);
    assert_eq!(c.swizzle, 27);
    assert_eq!(c.pitch, 1920);
    assert_eq!(c.slice_size, 15 * 9 * 65536);
    assert!(c.dcc_enabled);
    assert_eq!(c.dcc_address, dcc);
    assert!(c.dcc_slice_size() > 0);
    assert!(c.blend.enable);
    assert_eq!((c.blend.color_src, c.blend.color_dst), (4, 5));
    assert_eq!(d.color_mode, 1);
    assert_eq!(d.rop3, 0xCC);
    let z = d.depth.unwrap();
    assert_eq!(z.z_read_base, depth);
    assert_eq!(z.host_format, vk::Format::D32_SFLOAT);
    assert_eq!((z.width, z.height, z.swizzle), (w, h, 24));
    assert!(d.depth_state.depth_test && d.depth_state.depth_write && d.depth_state.depth_clear_enable);
    assert_eq!(d.depth_state.depth_compare, 3);
    assert_eq!(d.depth_state.depth_clear_value, 1.0);
    let vp = d.viewports[0];
    assert_eq!((vp.x, vp.y, vp.width, vp.height), (0.0, h as f32, w as f32, -(h as f32)));
    assert_eq!((vp.min_depth, vp.max_depth), (0.0, 1.0));
    assert_eq!(d.scissor, ps5gpu::render::Rect { x0: 16, y0: 8, x1: 1000, y1: 500 });
    assert!(d.raster.cull_back && d.raster.front_face_cw && !d.raster.cull_front);
    assert_eq!(d.shaders.ps.address, ps);
    assert_eq!(d.shaders.ps.user_sgpr_count(), 3);
    assert_eq!(&d.shaders.ps.user_data[..3], &[0x100, 0x101, 0x102]);
    assert_eq!(d.shaders.gs.address, gs);
    assert_eq!(d.shaders.gs.user_data[0], 0xAAAA);
    assert_eq!(d.shaders.stages, 0x0200_2000);
    let a = &draws[1];
    assert_eq!(a.packet.kind, DrawKind::Auto);
    assert_eq!(a.packet.count, 3);
    assert!(a.index.is_none());
    drop(draws);

    let disp = rec.dispatches.lock().unwrap();
    assert_eq!(disp.len(), 1);
    assert_eq!(disp[0].packet.groups, [8, 4, 1]);
    assert_eq!(disp[0].shader.address, 0x1234 << 8);
    assert_eq!(disp[0].threads[0], 64);
    assert_eq!(disp[0].shader.user_data[2], 0x55);
    drop(disp);
    assert_eq!(rec.eops.load(Ordering::SeqCst), 1, "the interrupt went through end of pipe");
    assert_eq!(mem::r32(e.a(LABELS)), Some(7));
}

// ---------------------------------------------------------------------------
// Command-processor semantics
// ---------------------------------------------------------------------------

/// A queue whose graphics processor has no render executor (labels and
/// GDS stay on the CPU, whatever executor other tests installed).
fn bare_queue() -> GpuQueue {
    let q = GpuQueue::new(1);
    q.with_interpreter(0, |i| i.executor = None);
    q
}

/// Write `words` at `at` and run them as one graphics submission.
fn run_words(q: &GpuQueue, at: u64, words: &[u32]) {
    assert!(mem::write_dwords(at, words));
    q.submit_graphics(at, words.len() as u32);
    run_queue(q);
}

/// Native WRITE_DATA of one dword to memory.
fn write_data(addr: u64, v: u32) -> [u32; 5] {
    [p4::make(5, op::WRITE_DATA, 0), 5 << 8, addr as u32, (addr >> 32) as u32, v]
}

fn indirect_buffer(addr: u64, dwords: u32, chain: bool) -> [u32; 4] {
    [p4::make(4, op::INDIRECT_BUFFER, 0), addr as u32, (addr >> 32) as u32, 0x0F20_0000 | ((chain as u32) << 20) | dwords]
}

/// A call returns to the packet after it; a chain abandons the rest of the
/// calling buffer.
#[test]
fn cp_indirect_call_returns_and_chain_replaces() {
    let e = Env::new();
    let (l0, l1, l2) = (e.a(LABELS), e.a(LABELS + 8), e.a(LABELS + 16));
    let child = e.a(CMD_END - 0x400);
    assert!(mem::write_dwords(child, &write_data(l0, 1)));
    let q = bare_queue();
    let mut w = Vec::new();
    w.extend_from_slice(&indirect_buffer(child, 5, false));
    w.extend_from_slice(&write_data(l1, 2));
    run_words(&q, e.a(CMD), &w);
    assert_eq!((mem::r32(l0), mem::r32(l1)), (Some(1), Some(2)));

    let mut w = Vec::new();
    w.extend_from_slice(&indirect_buffer(child, 5, true));
    w.extend_from_slice(&write_data(l2, 3));
    mem::w32(l0, 0);
    run_words(&q, e.a(CMD + 0x100), &w);
    assert_eq!(mem::r32(l0), Some(1));
    assert_eq!(mem::r32(l2), Some(0), "a chain does not return");
    assert_eq!(q.pending(), 0);
}

/// A taken branch calls its buffer and returns to the packet after it
/// (COND_INDIRECT_BUFFER has no CHAIN bit; reference Submission.cpp); a
/// branch not taken falls through (mode 1) or calls the else buffer
/// (mode 2).
#[test]
fn cp_branch_calls_then_or_else_and_returns() {
    let e = Env::new();
    let (cmp, then_l, else_l, after_l) = (e.a(LABELS), e.a(LABELS + 8), e.a(LABELS + 16), e.a(LABELS + 24));
    let (then_buf, else_buf) = (e.a(CMD_END - 0x400), e.a(CMD_END - 0x200));
    assert!(mem::write_dwords(then_buf, &write_data(then_l, 1)));
    assert!(mem::write_dwords(else_buf, &write_data(else_l, 1)));
    let branch = |mode: u32, reference: u64| -> Vec<u32> {
        let mut w = vec![
            p4::make(14, op::INDIRECT_BUFFER, 0),
            mode | (3 << 8), // equal
            cmp as u32,
            (cmp >> 32) as u32,
            u32::MAX,
            u32::MAX,
            reference as u32,
            (reference >> 32) as u32,
            then_buf as u32,
            (then_buf >> 32) as u32,
            5,
            else_buf as u32,
            (else_buf >> 32) as u32,
            5,
        ];
        w.extend_from_slice(&write_data(after_l, 1));
        w
    };
    let q = bare_queue();
    assert!(mem::w64(cmp, 7));
    run_words(&q, e.a(CMD), &branch(2, 7));
    assert_eq!((mem::r32(then_l), mem::r32(else_l), mem::r32(after_l)), (Some(1), Some(0), Some(1)), "then called, then the packet after");
    assert!(mem::write_dwords(then_l, &[0; 8]));
    run_words(&q, e.a(CMD + 0x100), &branch(2, 8));
    assert_eq!((mem::r32(then_l), mem::r32(else_l), mem::r32(after_l)), (Some(0), Some(1), Some(1)), "else called, then the packet after");
    assert!(mem::write_dwords(then_l, &[0; 8]));
    run_words(&q, e.a(CMD + 0x200), &branch(1, 8));
    assert_eq!((mem::r32(then_l), mem::r32(else_l), mem::r32(after_l)), (Some(0), Some(0), Some(1)), "mode 1 not taken falls through");
}

/// sceAgcDcbSetBaseIndirectArgs: SET_BASE index 1 with the address; a
/// NULL base clears it (titles unbind with 0) and the shader-type bit
/// selects compute (reference Commands.cpp testIndirectBase).
#[test]
fn set_base_indirect_args_accepts_a_null_base() {
    let e = Env::new();
    let args = e.a(DATA + 0x100);
    let bound = e.call("sceAgcDcbSetBaseIndirectArgs", &[e.cb(), 0, args], &[]) as u64;
    assert_eq!(mem::read_dwords(bound, 4).unwrap(), vec![p4::make(4, op::SET_BASE, 0), 1, args as u32, (args >> 32) as u32]);
    let cleared = e.call("sceAgcDcbSetBaseIndirectArgs", &[e.cb(), 1, 0], &[]) as u64;
    assert_eq!(cleared, bound + 16);
    let w = mem::read_dwords(cleared, 4).unwrap();
    assert!(w[0] & 2 != 0, "compute shader type");
    assert_eq!(&w[1..], &[1, 0, 0]);
}

/// sceAgcSetNop(packet) turns a packet in place into a NOP of the same
/// length: the opcode and the private custom code go, the count and the
/// predicate bits stay, a marker signature in the first payload dword is
/// cleared and the rest of the payload is untouched. The disabled packets
/// do not execute; the packet after them does.
#[test]
fn set_nop_disables_packets_in_place() {
    let e = Env::new();
    let (returned, released, selected) = (e.a(LABELS + 0x100), e.a(LABELS + 0x108), e.a(LABELS + 0x110));
    // An end-of-pipe release as the library encodes it (custom NOP code).
    let release_at = e.cursor();
    let r = e.call("sceAgcCbReleaseMem", &[e.cb(), 0x28, 0, 0, 0, released], &[2, 77, 0, 0, 0, 0]);
    assert_eq!(r as u64, release_at);
    let release_words = mem::read_dwords(release_at, 8).unwrap();
    assert_eq!(release_words[0], p4::make(8, op::NOP, nop::RELEASE_MEM));
    let mut w = Vec::new();
    w.extend_from_slice(&write_data(returned, 99)); // 0..5, predicated below
    w[0] |= 1;
    w.extend_from_slice(&[p4::make(2, op::NUM_INSTANCES, 0), 0x6875_0123]); // 5..7
    w.extend_from_slice(&release_words); // 7..15
    w.extend_from_slice(&write_data(selected, 88)); // 15..20
    let at = e.a(CMD + 0x3000);
    assert!(mem::write_dwords(at, &w));
    for i in [0u64, 5, 7] {
        assert_eq!(e.call("sceAgcSetNop", &[at + i * 4], &[0xDEAD]), 0, "packet at dword {i}");
    }
    let got = mem::read_dwords(at, w.len()).unwrap();
    assert_eq!(got[0], (w[0] & 0xFFFF_00FF) | (op::NOP << 8), "count and predicate bit kept");
    assert_eq!(got[5], p4::make(2, op::NOP, nop::ZERO));
    assert_eq!(got[6], 0x123, "marker signature cleared");
    assert_eq!(got[7], p4::make(8, op::NOP, nop::ZERO), "custom code cleared");
    assert_eq!(&got[1..5], &w[1..5], "payload untouched");
    assert_eq!(&got[8..], &w[8..], "payload and next packet untouched");
    for (i, n) in [(0usize, 5u32), (5, 2), (7, 8)] {
        assert_eq!(e.call("sceAgcGetPacketSize", &[at + i as u64 * 4], &[]), n as i64);
    }
    // A header-only one-dword NOP stays one dword; a null packet is refused.
    let one = e.a(CMD + 0x3100);
    assert!(mem::w32(one, 0xFFFF_1000));
    assert_eq!(e.call("sceAgcSetNop", &[one], &[]), 0);
    assert_eq!(mem::r32(one), Some(0xFFFF_1000));
    assert_ne!(e.call("sceAgcSetNop", &[0], &[]), 0);
    // Executed: nothing of the disabled packets runs, the last one does.
    let q = bare_queue();
    q.submit_graphics(at, w.len() as u32);
    run_queue(&q);
    assert_eq!((mem::r32(returned), mem::r32(released), mem::r32(selected)), (Some(0), Some(0), Some(88)));
}

/// sceAgcCbNop reserves its dwords without touching the payload; a
/// one-dword NOP is walked, predicated and executed as one dword.
#[test]
fn cp_one_dword_nop_reservations() {
    let e = Env::new();
    let start = e.cursor();
    assert_eq!(e.call("sceAgcCbNop", &[e.cb(), 0], &[]), 0);
    assert_eq!((e.cursor(), mem::r32(start)), (start, Some(0)), "a zero-sized reservation changed the buffer");
    assert_eq!(e.call("sceAgcCbNop", &[e.cb(), 1], &[]) as u64, start);
    assert_eq!((e.cursor(), mem::r32(start)), (start + 4, Some(0xFFFF_1000)));
    assert_eq!(e.call("sceAgcCbNopGetSize", &[1], &[]), 4);
    assert_eq!(e.call("sceAgcGetPacketSize", &[start], &[]), 1);
    // The payload of a longer reservation is the caller's.
    let at = e.cursor();
    assert!(mem::write_dwords(at, &[0, 0x1234_5678, 0x9ABC_DEF0]));
    assert_eq!(e.call("sceAgcCbNop", &[e.cb(), 3], &[]) as u64, at);
    assert_eq!(mem::read_dwords(at, 3), Some(vec![0xC001_1000, 0x1234_5678, 0x9ABC_DEF0]));

    // NOP(1), a completion write, NOP(1): range predication touches every
    // header and no payload, and the write is reached either way.
    let label = e.a(LABELS);
    let words = e.a(CMD + 0x1000);
    let mut w = vec![0xFFFF_1000];
    w.extend_from_slice(&write_data(label, 7));
    w.push(0xFFFF_1000);
    assert!(mem::write_dwords(words, &w));
    let q = bare_queue();
    for predication in [1u32, 0] {
        assert_eq!(e.call("sceAgcSetRangePredication", &[words, words + 7 * 4, predication as u64], &[]), 0);
        let got = mem::read_dwords(words, 7).unwrap();
        assert_eq!((got[0], got[6]), (0xFFFF_1000 | predication, 0xFFFF_1000 | predication));
        assert_eq!((got[1], got[5]), (w[1] | predication, 7), "range predication skipped a packet or changed a payload");
        assert!(mem::w32(label, 0));
        q.submit_graphics(words, 7);
        run_queue(&q);
        assert_eq!(mem::r32(label), Some(7), "the one-dword NOP hid the completion write");
    }
    assert_eq!(q.pending(), 0);
}

/// The branch and indirect-context packets match their size queries; the
/// then-target patch keeps the condition, the else target and the reserved
/// bits, and only applies to an INDIRECT_BUFFER packet.
#[test]
fn branch_then_target_patch() {
    let e = Env::new();
    // CB_TARGET_MASK through an indirect context register table.
    let regs = e.a(DATA + 0x100);
    assert!(mem::write_dwords(regs, &[0x8E, 0x7654_3210]));
    let cx = e.cursor();
    assert_eq!(e.call("sceAgcDcbSetCxRegistersIndirect", &[e.cb(), regs, 1], &[]) as u64, cx);
    assert_eq!(e.cursor() - cx, e.call("sceAgcDcbSetCxRegistersIndirectGetSize", &[], &[]) as u64);
    assert_eq!(e.cursor() - cx, 20);

    let (cond, else_buf) = (e.a(LABELS), e.a(CMD_END - 0x200));
    assert!(mem::write_dwords(else_buf, &write_data(e.a(LABELS + 8), 33)));
    let branch = e.cursor();
    let made = e.call("sceAgcCbBranch", &[e.cb(), 1, 0, cond, 0, 0], &[0, 0, 0, 3, else_buf, 5]) as u64;
    assert_eq!(made, branch);
    assert_eq!(e.cursor() - branch, e.call("sceAgcCbBranchGetSize", &[], &[]) as u64);
    assert_eq!(e.cursor() - branch, 56);
    let w8 = branch + 8 * 4;
    assert!(mem::w32(w8, 3) && mem::w32(w8 + 8, 0xC5A0_0000));
    let unpatched = mem::read_dwords(branch, 14).unwrap();
    // The plain form takes (packet, target, dwords) and keeps the cache
    // policy; the 3.00 form takes the policy first and replaces it.
    assert_eq!(e.call("sceAgcBranchPatchSetThenTarget", &[branch, cx, 5], &[]), 0);
    let patched = mem::read_dwords(branch, 14).unwrap();
    assert_eq!(patched[8], (cx as u32 & !3) | 3);
    assert_eq!(patched[9], (cx >> 32) as u32);
    assert_eq!(patched[10], 0xC5A0_0005);
    assert_eq!(patched[..8], unpatched[..8], "the plain form changed the condition");
    assert_eq!(patched[11..], unpatched[11..], "the plain form changed the else target");
    assert_eq!(e.call("sceAgcBranchPatchSetThenTarget_0300", &[branch, 2, cx, 5], &[]), 0);
    let patched = mem::read_dwords(branch, 14).unwrap();
    assert_eq!(patched[8], (cx as u32 & !3) | 3);
    assert_eq!(patched[9], (cx >> 32) as u32);
    assert_eq!(patched[10], 0xE5A0_0005);
    assert_eq!(patched[..8], unpatched[..8], "the 3.00 form changed the condition");
    assert_eq!(patched[11..], unpatched[11..], "the 3.00 form changed the else target");
    // The else forms write dwords 11..13 only.
    let before_else = mem::read_dwords(branch, 14).unwrap();
    assert_eq!(e.call("sceAgcBranchPatchSetElseTarget_0300", &[branch, 1, else_buf, 7], &[]), 0);
    let patched = mem::read_dwords(branch, 14).unwrap();
    assert_eq!(patched[11] & !3, else_buf as u32);
    assert_eq!(patched[12], (else_buf >> 32) as u32);
    assert_eq!(patched[13] & 0x300F_FFFF, 0x1000_0007);
    assert_eq!(patched[..11], before_else[..11], "the else form changed the then target");
    assert_eq!(e.call("sceAgcBranchPatchSetElseTarget", &[branch, else_buf, 5], &[]), 0);
    let patched = mem::read_dwords(branch, 14).unwrap();
    assert_eq!(patched[13] & 0x300F_FFFF, 0x1000_0005, "the plain else form changed the cache policy");
    // Refused before anything is written: a policy above 3, an unaligned
    // or null target, a size over 20 bits.
    let kept = mem::read_dwords(branch, 14).unwrap();
    assert_eq!(e.call("sceAgcBranchPatchSetThenTarget_0300", &[branch, 4, cx, 5], &[]), ERR_INVALID_ARGUMENT);
    assert_eq!(e.call("sceAgcBranchPatchSetThenTarget", &[branch, cx + 2, 5], &[]), ERR_INVALID_ARGUMENT);
    assert_eq!(e.call("sceAgcBranchPatchSetThenTarget", &[branch, 0, 5], &[]), ERR_INVALID_ARGUMENT);
    assert_eq!(e.call("sceAgcBranchPatchSetElseTarget", &[branch, else_buf, 0x10_0000], &[]), ERR_INVALID_ARGUMENT);
    assert_eq!(mem::read_dwords(branch, 14).unwrap(), kept);
    assert_eq!(e.call("sceAgcBranchPatchSetThenTarget", &[branch, cx, 5], &[]), 0);

    // The patched branch fetches and applies the indirect context packet.
    let q = bare_queue();
    q.submit_graphics(branch, 14);
    run_queue(&q);
    assert_eq!(q.pending(), 0);
    q.with_interpreter(0, |i: &mut Interpreter| assert_eq!(i.regs.context[0x8E], 0x7654_3210));

    // Another packet type is refused and left alone.
    assert!(mem::w32(branch, p4::make(14, op::NOP, 0)));
    let mismatched = mem::read_dwords(branch, 14).unwrap();
    assert_eq!(e.call("sceAgcBranchPatchSetThenTarget", &[branch, else_buf, 5], &[]), ERR_INVALID_PACKET);
    assert_eq!(e.call("sceAgcBranchPatchSetElseTarget_0300", &[branch, 0, else_buf, 5], &[]), ERR_INVALID_PACKET);
    assert_eq!(mem::read_dwords(branch, 14).unwrap(), mismatched);
    assert_eq!(e.call("sceAgcBranchPatchSetThenTarget", &[0, else_buf, 5], &[]), ERR_INVALID_PACKET);
}

/// A wait inside a called buffer suspends there; resuming continues at the
/// wait without re-running anything before it, then returns to the caller.
#[test]
fn cp_suspension_inside_a_call_resumes_in_place() {
    let e = Env::new();
    let (flag, before, after, back) = (e.a(LABELS), e.a(LABELS + 8), e.a(LABELS + 16), e.a(LABELS + 24));
    let child = e.a(CMD_END - 0x400);
    let mut c = Vec::new();
    c.extend_from_slice(&[p4::make(9, op::ATOMIC_MEM, 0), 15, before as u32, (before >> 32) as u32, 1, 0, 0, 0, 0]); // before += 1
    c.extend_from_slice(&[p4::make(7, op::WAIT_REG_MEM, 0), 0x13, flag as u32, (flag >> 32) as u32, 1, 0xFFFF_FFFF, 10]);
    c.extend_from_slice(&write_data(after, 1));
    assert!(mem::write_dwords(child, &c));
    let mut w = Vec::new();
    w.extend_from_slice(&indirect_buffer(child, c.len() as u32, false));
    w.extend_from_slice(&write_data(back, 1));
    let q = bare_queue();
    run_words(&q, e.a(CMD), &w);
    assert_eq!(q.pending(), 1);
    assert_eq!((mem::r32(before), mem::r32(after), mem::r32(back)), (Some(1), Some(0), Some(0)));
    mem::w32(flag, 1);
    q.retry_blocked();
    run_queue(&q);
    assert_eq!(q.pending(), 0);
    assert_eq!((mem::r32(before), mem::r32(after), mem::r32(back)), (Some(1), Some(1), Some(1)));
}

/// A slice that writes a label and then blocks releases the queues waiting
/// on that label at once: the next slice runs them, without the retry timer
/// (`RPS5E_PM4_WAKE`).
#[test]
fn cp_label_written_by_a_blocked_slice_releases_its_waiters() {
    let e = Env::new();
    let (flag, never, out) = (e.a(LABELS + 0x40), e.a(LABELS + 0x48), e.a(LABELS + 0x50));
    mem::w32(flag, 0);
    mem::w32(never, 0);
    mem::w32(out, 0);
    let q = bare_queue();
    q.with_interpreter(1, |i| i.executor = None);
    // Graphics: wait for flag == 1, then out = 1.
    let mut g = vec![p4::make(7, op::WAIT_REG_MEM, 0), 0x13, flag as u32, (flag >> 32) as u32, 1, 0xFFFF_FFFF, 10];
    g.extend_from_slice(&write_data(out, 1));
    run_words(&q, e.a(CMD), &g);
    assert_eq!(q.pending(), 1, "the graphics queue waits for the flag");
    // Compute: flag = 1, then a wait nobody releases.
    let mut c = write_data(flag, 1).to_vec();
    c.extend_from_slice(&[p4::make(7, op::WAIT_REG_MEM, 0), 0x13, never as u32, (never >> 32) as u32, 1, 0xFFFF_FFFF, 10]);
    let at = e.a(CMD + 0x1000);
    assert!(mem::write_dwords(at, &c));
    assert!(q.submit_compute(0x20, at, c.len() as u32));
    assert_eq!(q.process_one(), SliceResult::Progressed, "the compute slice writes the flag and blocks");
    assert_eq!(mem::r32(flag), Some(1));
    assert_eq!(q.process_one(), SliceResult::Completed, "the graphics queue runs without a retry");
    assert_eq!(mem::r32(out), Some(1));
    assert_eq!(q.pending(), 1, "the compute queue still waits");
}

/// EVENT_WRITE_EOS: data selector 2 stores the 32-bit value; selector 1
/// stores GDS words.
#[test]
fn cp_end_of_shader_writes_value_or_gds() {
    let e = Env::new();
    let (value_l, gds_l) = (e.a(LABELS), e.a(LABELS + 0x10));
    let gds_offset = 0x7F00u32; // dwords 0x1FC0.. (unused by other tests)
    let q = bare_queue();
    let eos = |dst: u64, sel: u32, data: u32| {
        [p4::make(5, op::EVENT_WRITE_EOS, 0), 0x0000_0006, dst as u32, ((dst >> 32) as u32 & 0xFFFF) | (sel << 29), data]
    };
    let mut w = Vec::new();
    // Fill two GDS dwords, then read them back with EOS.
    w.extend_from_slice(&[p4::make(7, op::DMA_DATA, 0), (1 << 20) | (2 << 29), 0x1234_5678, 0, gds_offset, 0, 8]);
    w.extend_from_slice(&eos(value_l, 2, 0xABCD));
    w.extend_from_slice(&eos(gds_l, 1, (gds_offset / 4) | (2 << 16)));
    run_words(&q, e.a(CMD), &w);
    assert_eq!(mem::r32(value_l), Some(0xABCD));
    assert_eq!(mem::r64(gds_l), Some(0x1234_5678_1234_5678));
}

/// WRITE_DATA with DST_SEL 3 writes guest memory (a frame label a later
/// WAIT_REG_MEM polls), not GDS: the control word `sceAgcDcbWriteData`
/// builds for its memory destination.
#[test]
fn write_data_destination_3_writes_memory() {
    let e = Env::new();
    let l = e.a(LABELS + 0x200);
    assert!(mem::write_dwords(l, &[0, 0]));
    let q = bare_queue();
    run_words(&q, e.a(CMD), &[p4::make(6, op::WRITE_DATA, 0), 0x0610_0300, l as u32, (l >> 32) as u32, 91, 92]);
    assert_eq!((mem::r32(l), mem::r32(l + 4)), (Some(91), Some(92)));
}

/// WAIT_REG_MEM ignores the low address bits below its width; a
/// WAIT_REG_MEM_64 whose mask has no high bits awaits the dword at a 4-byte
/// aligned address (hardware rules of the reference driver's tests).
#[test]
fn wait_reg_mem_masks_its_address_by_the_awaited_width() {
    let e = Env::new();
    let base = e.a(LABELS + 0x800);
    let out = e.a(LABELS + 0x840);
    assert!(mem::write_dwords(base, &[0x1234_5678, 0x0123_4567, 0x5555_5555, 0xAAAA_AAAA]));
    // 32-bit waits at base + 1..3 compare the dword at base.
    for offset in 0..4u64 {
        let a = base + offset;
        let q = bare_queue();
        mem::w32(out, 0);
        let mut w = vec![p4::make(7, op::WAIT_REG_MEM, 0), 0x13, a as u32, (a >> 32) as u32, 0x1234_5678, 0xFFFF_FFFF, 10];
        w.extend_from_slice(&write_data(out, 1));
        run_words(&q, e.a(CMD), &w);
        assert_eq!((q.pending(), mem::r32(out)), (0, Some(1)), "32-bit wait at +{offset}");
    }
    // 64-bit waits at base + 1..7 compare the qword at base.
    for offset in 0..8u64 {
        let a = base + offset;
        let q = bare_queue();
        mem::w32(out, 0);
        let mut w = vec![p4::make(9, op::WAIT_REG_MEM64, 0), 0x13, a as u32, (a >> 32) as u32, 0x1234_5678, 0x0123_4567, 0xFFFF_FFFF, 0xFFFF_FFFF, 10];
        w.extend_from_slice(&write_data(out, 1));
        run_words(&q, e.a(CMD), &w);
        assert_eq!((q.pending(), mem::r32(out)), (0, Some(1)), "64-bit wait at +{offset}");
    }
    // A 64-bit wait with a zero high mask at base + 4..7 compares the dword
    // at base + 4, not the low half of the qword at base.
    for offset in 4..8u64 {
        let a = base + offset;
        let q = bare_queue();
        mem::w32(out, 0);
        let mut w = vec![p4::make(9, op::WAIT_REG_MEM64, 0), 0x13, a as u32, (a >> 32) as u32, 0x0123_4567, 0, 0xFFFF_FFFF, 0, 10];
        w.extend_from_slice(&write_data(out, 1));
        run_words(&q, e.a(CMD), &w);
        assert_eq!((q.pending(), mem::r32(out)), (0, Some(1)), "narrow 64-bit wait at +{offset}");
    }
}

/// PIXEL_PIPE_STAT_DUMP (EVENT_WRITE ZPASS_DONE, index 1) of an occlusion
/// query's begin (block) or end (block + 8).
fn zpass_dump(addr: u64) -> [u32; 4] {
    [p4::make(4, op::EVENT_WRITE, 0), 0x39 | (1 << 8), addr as u32, (addr >> 32) as u32]
}

/// The guest's view of an occlusion result block: every slot ready, and
/// the sum of end - begin over the depth blocks.
fn occlusion_sum(block: u64) -> Option<u64> {
    let mut sum = 0u64;
    for db in 0..16 {
        let (begin, end) = (mem::r64(block + db * 16)?, mem::r64(block + db * 16 + 8)?);
        if begin >> 63 == 0 || end >> 63 == 0 {
            return None;
        }
        sum = sum.wrapping_add((end & !(1 << 63)).wrapping_sub(begin & !(1 << 63)));
    }
    Some(sum)
}

/// Without host sample counts, a query reports a ready, visible result.
#[test]
fn cp_occlusion_query_without_host_counts_is_visible() {
    let e = Env::new();
    let block = e.a(DATA + 0x1000);
    assert!(mem::write(block, &[0u8; 256]));
    let q = bare_queue();
    let mut w = Vec::new();
    w.extend_from_slice(&zpass_dump(block));
    w.extend_from_slice(&zpass_dump(block + 8));
    run_words(&q, e.a(CMD), &w);
    let sum = occlusion_sum(block).expect("every slot is ready");
    assert!(sum > 0, "the object stays visible");
}

/// An executor counting samples receives both dumps; the interpreter writes
/// nothing itself, and the block the executor publishes is what the guest reads.
#[test]
fn cp_occlusion_query_with_host_counts_goes_to_the_executor() {
    #[derive(Default)]
    struct Counting {
        dumps: Mutex<Vec<u64>>,
        write: Mutex<Option<ps5gpu::render::OcclusionWrite>>,
    }
    impl ps5gpu::render::RenderExecutor for Counting {
        fn draw(&self, _: &ps5gpu::render::ResolvedDraw) {}
        fn dispatch(&self, _: &ps5gpu::render::ResolvedDispatch) {}
        fn occlusion_counter_dump(&self, addr: u64, write: &ps5gpu::render::OcclusionWrite) -> bool {
            self.dumps.lock().unwrap().push(addr);
            *self.write.lock().unwrap() = Some(write.clone());
            true
        }
    }
    let e = Env::new();
    let block = e.a(DATA + 0x1100);
    assert!(mem::write(block, &[0u8; 256]));
    let rec = std::sync::Arc::new(Counting::default());
    let q = GpuQueue::new(1);
    let ex: std::sync::Arc<dyn ps5gpu::render::RenderExecutor> = rec.clone();
    q.with_interpreter(0, |i| i.executor = Some(ex));
    let mut w = Vec::new();
    w.extend_from_slice(&zpass_dump(block));
    w.extend_from_slice(&zpass_dump(block + 8));
    run_words(&q, e.a(CMD), &w);
    assert_eq!(*rec.dumps.lock().unwrap(), vec![block, block + 8]);
    assert_eq!(mem::r64(block), Some(0));
    assert_eq!(mem::r64(block + 8), Some(0));
    assert_eq!(occlusion_sum(block), None, "nothing published yet");
    // The executor publishes 5 + 7 samples counted in two segments.
    let mut result = [0u8; 256];
    for (i, slot) in result.chunks_exact_mut(8).enumerate() {
        let count = match i {
            1 => 5,
            3 => 7,
            _ => 0,
        };
        slot.copy_from_slice(&((1u64 << 63) | count).to_le_bytes());
    }
    let write = rec.write.lock().unwrap().clone().unwrap();
    write(block, &result);
    assert_eq!(occlusion_sum(block), Some(12));
}

/// The private RELEASE_MEM uses the native dword layout.
#[test]
fn cp_private_release_mem_follows_the_native_layout() {
    let e = Env::new();
    let (l32, l64) = (e.a(LABELS), e.a(LABELS + 0x10));
    let q = bare_queue();
    let r32 = e.call("sceAgcCbReleaseMem", &[e.cb(), 0x28, 0, 0, 0, l32], &[1, 0x1122_3344_5566_7788, 0, 0, 0, 0]) as u64;
    let r64 = e.call("sceAgcCbReleaseMem", &[e.cb(), 0x2F, 0x100, 0, 0, l64], &[2, 0x1122_3344_5566_7788, 0, 0, 0, 0]) as u64;
    assert_eq!(mem::r32(r32 + 8).unwrap() >> 29, 1);
    // GL2 invalidate implies the write-back; event index 6 from action 0x2F.
    assert_eq!(mem::r32(r64 + 4), Some(0x2F | (6 << 8) | (0x300 << 12)));
    submit_cb(&e, &q);
    run_queue(&q);
    assert_eq!(mem::r32(l32), Some(0x5566_7788));
    assert_eq!(mem::r32(l32 + 4), Some(0));
    assert_eq!(mem::r64(l64), Some(0x1122_3344_5566_7788));
}

/// RELEASE_MEM data selectors 3 and 4 store a 64-bit counter (GPU clock,
/// system clock): both get the reference clock when the pipe drained.
/// Selector 4 used to be dropped with a warning (the label kept its old
/// value).
#[test]
fn cp_release_mem_counter_selectors_write_the_clock() {
    let e = Env::new();
    let (l3, l4) = (e.a(LABELS), e.a(LABELS + 0x10));
    mem::w64(l3, 0);
    mem::w64(l4, 0);
    let release = |sel: u32, l: u64| [p4::make(8, op::RELEASE_MEM, 0), 0x28, sel << 29, l as u32, (l >> 32) as u32, 0, 0, 0];
    let q = bare_queue();
    let before = interp::reference_clock();
    let mut w = Vec::new();
    w.extend_from_slice(&release(3, l3));
    w.extend_from_slice(&release(4, l4));
    run_words(&q, e.a(CMD), &w);
    let after = interp::reference_clock();
    for (sel, l) in [(3, l3), (4, l4)] {
        let v = mem::r64(l).unwrap();
        assert!(before <= v && v <= after, "selector {sel}: {v} not in [{before}, {after}]");
    }
}

/// Boolean predication reads the value whatever the wait bit; Z-pass
/// predication waits for ready counters (or draws everything with the wait
/// bit set).
#[test]
fn cp_predication_boolean_and_zpass() {
    let e = Env::new();
    let (pred, mark) = (e.a(LABELS), e.a(LABELS + 8));
    let zpass = e.a(DATA + 0x400);
    let set_pred = |flags: u32, addr: u64| [p4::make(4, op::SET_PREDICATION, 0), flags, addr as u32 & 0xFFFF_FFF0, (addr >> 32) as u32];
    let predicated_write = |v: u32| {
        let mut w = write_data(mark, v);
        w[0] |= 1;
        w
    };
    let q = bare_queue();
    // Boolean, condition 0 (skip when the value is non-zero), wait bit set.
    mem::w64(pred, 1);
    let mut w = Vec::new();
    w.extend_from_slice(&set_pred((3 << 16) | (1 << 12), pred));
    w.extend_from_slice(&predicated_write(5));
    w.extend_from_slice(&set_pred(0, 0));
    run_words(&q, e.a(CMD), &w);
    assert_eq!(mem::r32(mark), Some(0), "skipped");
    // Z-pass with counters not ready and no wait: the submission waits.
    for i in 0..32 {
        mem::w64(zpass + i * 8, 0);
    }
    let mut w = Vec::new();
    w.extend_from_slice(&set_pred(1 << 16, zpass));
    w.extend_from_slice(&predicated_write(6));
    run_words(&q, e.a(CMD + 0x100), &w);
    assert_eq!(q.pending(), 1);
    // Ready counters with 3 passing samples: condition 0 skips.
    for db in 0..16u64 {
        mem::w64(zpass + db * 16, 1 << 63);
        mem::w64(zpass + db * 16 + 8, (1 << 63) | if db == 0 { 3 } else { 0 });
    }
    q.retry_blocked();
    run_queue(&q);
    assert_eq!(q.pending(), 0);
    assert_eq!(mem::r32(mark), Some(0));
    // Not ready with the wait bit: predication off, the packet runs.
    mem::w64(zpass, 0);
    let mut w = Vec::new();
    w.extend_from_slice(&set_pred((1 << 16) | (1 << 12), zpass));
    w.extend_from_slice(&predicated_write(7));
    run_words(&q, e.a(CMD + 0x200), &w);
    assert_eq!(mem::r32(mark), Some(7));
    // 32-bit boolean: only the dword counts (the high dword is not read),
    // and the rest of the submission runs (it used to be dropped).
    mem::w64(pred, 0xFFFF_FFFF_0000_0000);
    let mut w = Vec::new();
    w.extend_from_slice(&set_pred((4 << 16) | (1 << 8), pred));
    w.extend_from_slice(&predicated_write(8));
    w.extend_from_slice(&set_pred(0, 0));
    w.extend_from_slice(&write_data(mark + 4, 9));
    run_words(&q, e.a(CMD + 0x300), &w);
    assert_eq!(mem::r32(mark), Some(7), "condition 1 skips when the dword is zero");
    assert_eq!(mem::r32(mark + 4), Some(9), "the packets after it run");
    mem::w32(pred, 1);
    let mut w = Vec::new();
    w.extend_from_slice(&set_pred((4 << 16) | (1 << 8), pred));
    w.extend_from_slice(&predicated_write(10));
    w.extend_from_slice(&set_pred(0, 0));
    run_words(&q, e.a(CMD + 0x400), &w);
    assert_eq!(mem::r32(mark), Some(10), "a non-zero dword draws");
}

/// Context-state operations: a clear restores the reset values; one
/// context can be saved and restored.
#[test]
fn cp_context_state_push_pop_clear() {
    let e = Env::new();
    let q = bare_queue();
    let cx = |off: u32, v: u32| [p4::make(3, op::SET_CONTEXT_REG, 0), off, v];
    let state = |operation: u32| [p4::make(3, op::NOP, nop::CONTEXT_STATE), operation, 0];
    let initial = interp::initial_context_registers();
    let mut w = Vec::new();
    w.extend_from_slice(&cx(0x202, 0x1234)); // CB_COLOR_CONTROL
    w.extend_from_slice(&state(3)); // push + clear
    run_words(&q, e.a(CMD), &w);
    q.with_interpreter(0, |i| assert_eq!(i.regs.context, initial));
    let mut w = Vec::new();
    w.extend_from_slice(&cx(0x202, 0x5678));
    w.extend_from_slice(&state(2)); // pop
    run_words(&q, e.a(CMD + 0x100), &w);
    q.with_interpreter(0, |i| assert_eq!(i.regs.context[0x202], 0x1234));
    run_words(&q, e.a(CMD + 0x200), &[p4::make(2, op::CLEAR_STATE, 0), 0]);
    q.with_interpreter(0, |i| assert_eq!(i.regs.context, initial));
}

/// Retiring a frame boundary resets the graphics processor's state.
#[test]
fn cp_frame_boundary_resets_the_graphics_processor() {
    let e = Env::new();
    let q = bare_queue();
    let mut w = vec![p4::make(3, op::SET_SH_REG, 0), 0x10, 7];
    w.extend_from_slice(&[p4::make(2, op::NUM_INSTANCES, 0), 4]);
    w.extend_from_slice(&[p4::make(3, op::INDEX_BASE, 0), 0x1000, 0]);
    run_words(&q, e.a(CMD), &w);
    q.with_interpreter(0, |i| assert_eq!((i.regs.shader[0x10], i.regs.num_instances, i.index_base), (7, 4, 0x1000)));
    q.suspend_point();
    run_queue(&q);
    q.with_interpreter(0, |i| {
        assert_eq!((i.regs.shader[0x10], i.regs.num_instances, i.index_base), (0, 1, 0));
        assert!(!i.predicate_skip);
    });
}

/// Register runs land in one copy; the bank's nop register and offsets past
/// the bank are skipped; VGT_INDEX_TYPE updates the index type.
#[test]
fn cp_register_writes() {
    let e = Env::new();
    let q = bare_queue();
    let mut w = vec![p4::make(5, op::SET_SH_REG, 0), 0x27F, 1, 2, 3]; // 0x280 is the nop register
    w.extend_from_slice(&[p4::make(4, op::SET_UCONFIG_REG, 0), 0x242, 4, 0x1D]); // prim type, index type
    w.extend_from_slice(&[p4::make(3, op::SET_CONTEXT_REG, 0), 0x3FF, 9]);
    w.extend_from_slice(&[p4::make(3, op::SET_CONTEXT_REG, 0), 0x400, 9]);
    run_words(&q, e.a(CMD), &w);
    q.with_interpreter(0, |i| {
        assert_eq!((i.regs.shader[0x27F], i.regs.shader[0x280], i.regs.shader[0x281]), (1, 0, 3));
        assert_eq!((i.regs.uconfig[0x242], i.regs.uconfig[0x243]), (4, 0x1D));
        assert_eq!(i.index_type, 1);
        assert_eq!(i.regs.context[0x3FF], 9);
    });
}

/// DISPATCH_DRAW_PREAMBLE draws the instance count of dword 6 (NUM_INSTANCES
/// when it is zero).
#[test]
fn cp_multi_instanced_draw_instance_count() {
    use ps5gpu::render::DrawKind;
    let e = Env::new();
    let rec = std::sync::Arc::new(Recorder::default());
    let q = bare_queue();
    let ex: std::sync::Arc<dyn ps5gpu::render::RenderExecutor> = rec.clone();
    q.with_interpreter(0, |i| i.executor = Some(ex));
    let (idx, ids) = (e.a(DATA), e.a(DATA + 0x100));
    e.call("sceAgcDcbSetNumInstances", &[e.cb(), 9], &[]);
    e.call("sceAgcDcbDrawIndexMultiInstanced", &[e.cb(), 6, idx, ids, 3, 0], &[]);
    e.call("sceAgcDcbDrawIndexMultiInstanced", &[e.cb(), 6, idx, ids, 0, 0], &[]);
    submit_cb(&e, &q);
    run_queue(&q);
    let draws = rec.draws.lock().unwrap();
    assert_eq!(draws.len(), 2);
    assert_eq!(draws[0].packet.kind, DrawKind::IndexMultiInstanced);
    assert_eq!((draws[0].packet.instance_count, draws[0].packet.index_address, draws[0].packet.count), (3, idx, 6));
    assert_eq!(draws[1].packet.instance_count, 9);
}

/// The wait-on-address sequence: begin tag, native wait, end tag; the
/// interpreter skips the tags and waits.
#[test]
fn cp_wait_on_address_sequence() {
    let e = Env::new();
    let label = e.a(LABELS);
    let done = e.a(LABELS + 8);
    let w = e.call("sceAgcDcbWaitRegMem", &[e.cb(), 1, 3, 0, 0, label], &[0x1_0000_0002, u64::MAX, 16]) as u64;
    let headers = parse_range(w, e.cursor());
    assert_eq!(headers, vec![p4::make(4, op::SET_UCONFIG_REG, 1), p4::make(9, op::WAIT_REG_MEM64, 0), p4::make(3, op::SET_UCONFIG_REG, 1)]);
    assert_eq!(mem::r32(w + 8).unwrap() & 0xFFFF_0000, p4::WAIT_USER_DATA_BEGIN64_TAG);
    mem::w32(e.a(DATA), 1);
    e.call("sceAgcDcbWriteData", &[e.cb(), 10, 0, done, e.a(DATA), 1], &[0, 0]);
    let q = bare_queue();
    submit_cb(&e, &q);
    run_queue(&q);
    assert_eq!(q.pending(), 1);
    mem::w64(label, 0x1_0000_0002);
    q.retry_blocked();
    run_queue(&q);
    assert_eq!(q.pending(), 0);
    assert_eq!(mem::r32(done), Some(1));
    q.with_interpreter(0, |i| assert_eq!(i.regs.uconfig[p4::INTERNAL_DATA_REGISTER as usize], 0));
}

/// Native DMA_DATA from the builders: memory copy and immediate fill.
#[test]
fn cp_dma_data_copy_and_fill() {
    let e = Env::new();
    let (src, dst, fill) = (e.a(DATA), e.a(DATA + 0x100), e.a(DATA + 0x200));
    assert!(mem::write_dwords(src, &[1, 2, 3, 4]));
    e.call("sceAgcDcbDmaData", &[e.cb(), 0, 0, 0, dst, 0], &[0, src, 16, 0, 0, 0]);
    e.call("sceAgcAcbDmaData", &[e.cb(), 0, 0, fill, 2, 0], &[0xDEAD_BEEF, 8, 0, 0]);
    let p = e.a(CMD);
    assert_eq!(e.call("sceAgcDmaDataPatchSetDstAddressOrOffset", &[p, dst + 4], &[]), 0);
    let q = bare_queue();
    submit_cb(&e, &q);
    run_queue(&q);
    assert_eq!(mem::read_dwords(dst, 5).unwrap(), vec![0, 1, 2, 3, 4]);
    assert_eq!(mem::read_dwords(fill, 2).unwrap(), vec![0xDEAD_BEEF; 2]);
}

/// Unknown NOP custom codes and private packets no builder emits are
/// skipped without stopping the submission.
#[test]
fn cp_unknown_custom_nops_are_skipped() {
    let e = Env::new();
    let l = e.a(LABELS);
    let q = bare_queue();
    let mut w = vec![p4::make(3, op::NOP, 0x3E), 0, 0];
    w.extend_from_slice(&write_data(l, 4));
    run_words(&q, e.a(CMD), &w);
    assert_eq!(mem::r32(l), Some(4));
}

// ---------------------------------------------------------------------------
// Driver and shader exports
// ---------------------------------------------------------------------------

/// Command buffers go to the queue named by the queue context (+4): compute
/// for 0x20..0x57, graphics otherwise.
#[test]
fn driver_submit_command_buffer_routes_by_queue() {
    let e = Env::new();
    let (qctx, packet) = (e.a(DATA), e.a(DATA + 0x20));
    assert_eq!(e.call("sceAgcDriverSubmitCommandBuffer", &[0, packet], &[]), ERR_INVALID_ARGUMENT);
    assert_eq!(e.call("sceAgcDriverSubmitMultiCommandBuffers", &[0, 0, 0, 0], &[]), OK);
    assert_eq!(e.call("sceAgcDriverSubmitMultiCommandBuffers", &[0, 0, 0, 1], &[]), ERR_INVALID_ARGUMENT);
    // Empty buffers are accepted and ignored.
    mem::w32(qctx + 4, 0x21);
    mem::w64(packet, 0);
    mem::w32(packet + 8, 0);
    assert_eq!(e.call("sceAgcDriverSubmitCommandBuffer", &[qctx, packet], &[]), OK);
    assert_eq!(e.call("sceAgcDriverSubmitAcb", &[0x21, 0], &[]), OK);
    assert_eq!(e.call("sceAgcDriverSubmitMultiDcbs", &[0, 0, 0], &[]), OK);
}

/// Event type / context id follow the event's filter.
#[test]
fn driver_eq_event_fields() {
    let e = Env::new();
    let ev = e.a(DATA);
    mem::w64(ev, 0x40);
    mem::write(ev + 8, &(-14i16).to_le_bytes());
    mem::w64(ev + 0x10, 0x0123_4567_89AB);
    assert_eq!(e.call("sceAgcDriverGetEqEventType", &[ev], &[]), 0x40);
    assert_eq!(e.call("sceAgcDriverGetEqContextId", &[ev], &[]), 0x4567_89AB);
    mem::write(ev + 8, &(-11i16).to_le_bytes());
    assert_eq!(e.call("sceAgcDriverGetEqEventType", &[ev], &[]), 0x4567_89AB);
    assert_eq!(e.call("sceAgcDriverGetEqContextId", &[ev], &[]), 0x40);
    assert_eq!(e.call("sceAgcDriverGetEqEventType", &[0], &[]), 0);
    assert_eq!(e.call("sceAgcDriverAddEqEvent", &[0xDEAD_0000, 0, 0], &[]), ERR_BAD_FILE);
}

#[test]
fn data_packet_payload_range() {
    let e = Env::new();
    let (range, packet) = (e.a(DATA), e.a(DATA + 0x40));
    mem::w32(packet, p4::make(6, op::NOP, 0));
    assert_eq!(e.call("sceAgcGetDataPacketPayloadRange", &[range, packet, 0], &[]), 0);
    assert_eq!((mem::r64(range), mem::r64(range + 8)), (Some(packet + 4), Some(20)));
    assert_eq!(e.call("sceAgcGetDataPacketPayloadRange", &[range, packet, 1], &[]), 0);
    assert_eq!((mem::r64(range), mem::r64(range + 8)), (Some(packet + 8), Some(16)));
    mem::w32(packet, 0xFFFF_FFFF);
    assert_eq!(e.call("sceAgcGetDataPacketPayloadRange", &[range, packet, 0], &[]), 0);
    assert_eq!((mem::r64(range), mem::r64(range + 8)), (Some(0), Some(0)));
}

/// A minimal shader object: `regs` as the shader register table.
fn shader_object(e: &Env, at: u64, ty: u8, sh: &[(u32, u32)], cx: &[(u32, u32)], stages: u32) -> u64 {
    let (sh_at, cx_at, specials) = (at + 0x60, at + 0x200, at + 0x300);
    assert!(mem::write(at, &[0u8; 0x60]));
    mem::w32(at, 0x3433_3231);
    mem::w32(at + 4, 0x18);
    // Relative pointers, as the shader binary stores them.
    mem::w64(at + 0x20, sh_at - (at + 0x20));
    mem::w64(at + 0x18, cx_at - (at + 0x18));
    mem::w64(at + 0x28, specials - (at + 0x28));
    mem::w8(at + 0x5A, ty);
    mem::w8(at + 0x5B, cx.len() as u8);
    mem::w8(at + 0x5C, sh.len() as u8);
    for (i, &(o, v)) in sh.iter().enumerate() {
        mem::w32(sh_at + i as u64 * 8, o);
        mem::w32(sh_at + i as u64 * 8 + 4, v);
    }
    for (i, &(o, v)) in cx.iter().enumerate() {
        mem::w32(cx_at + i as u64 * 8, o);
        mem::w32(cx_at + i as u64 * 8 + 4, v);
    }
    mem::w32(specials + 8, 0x2D5);
    mem::w32(specials + 12, stages);
    let _ = e;
    at
}

/// CreateShader relocates the program address pair of the shader type,
/// adding the table's offset and keeping the upper hi bits.
#[test]
fn create_shader_relocates_the_program_address() {
    let e = Env::new();
    let out = e.a(DATA);
    let code = 0x0123_4567_8900u64;
    let ps = shader_object(&e, e.a(DATA + 0x1000), 1, &[(0x08, 0x10), (0x09, 0xAB00)], &[], 0);
    assert_eq!(e.call("sceAgcCreateShader", &[out, ps, code], &[]), OK);
    assert_eq!(mem::r64(out), Some(ps));
    let sh = mem::r64(ps + 0x20).unwrap();
    assert_eq!(sh, ps + 0x60);
    assert_eq!(mem::r32(sh + 4), Some(((code + 0x1000) >> 8) as u32));
    assert_eq!(mem::r32(sh + 12), Some(0xAB00 | 0x01));
    // GsBack patches SPI_SHADER_PGM_LO_GS; a missing hi register fails.
    let back = shader_object(&e, e.a(DATA + 0x2000), 6, &[(0x88, 0), (0x8A, 0)], &[], 0);
    assert_eq!(e.call("sceAgcCreateShader", &[out, back, code], &[]), ERR_INCOMPLETE_SHADER_REGISTERS);
    // Front halves have no program register to patch.
    let front = shader_object(&e, e.a(DATA + 0x3000), 4, &[], &[], 0);
    assert_eq!(e.call("sceAgcCreateShader", &[out, front, code], &[]), OK);
}

/// Fused halves: the back half's registers with the front half's program
/// (LO_ES), checksum, merged resources; the newer export recomputes the
/// shared VGPRs.
#[test]
fn fuse_shader_halves_merges_the_front_half() {
    let e = Env::new();
    let fused = e.a(DATA);
    let scratch = e.a(DATA + 0x100);
    let front_code = 0x5500_0000_1000u64;
    let front = shader_object(
        &e,
        e.a(DATA + 0x1000),
        4,
        &[(0x80, 0xF00D), (0x8A, 0x0000_0007), (0x8B, 0x1000_0004 | 0x0800_0000)],
        &[],
        1 << 22,
    );
    let back = shader_object(&e, e.a(DATA + 0x2000), 6, &[(0x80, 0), (0x88, 0), (0x89, 0), (0x8A, 0x0000_0003), (0x8B, 0x2000_0000), (0xC8, 0), (0xC9, 0)], &[], 1 << 22);
    for h in [front, back] {
        assert_eq!(e.call("sceAgcCreateShader", &[e.a(DATA + 0x80), h, if h == front { front_code } else { 0x66_0000_0000 }], &[]), OK);
    }
    let size = e.a(DATA + 0x90);
    assert_eq!(e.call("sceAgcGetFusedShaderSize_0080", &[size, front, back], &[]), OK);
    assert_eq!((mem::r64(size), mem::r64(size + 8)), (Some(7 * 8), Some(4)));
    assert_eq!(e.call("sceAgcFuseShaderHalves_0200", &[fused, front, back, scratch], &[]), OK);
    assert_eq!(mem::r8(fused + 0x5A), Some(2));
    assert_eq!(mem::r64(fused + 0x20), Some(scratch));
    assert_eq!(mem::r64(fused + 0x08), Some(0));
    let reg = |o: u32| (0..7).map(|i| scratch + i * 8).find(|&a| mem::r32(a) == Some(o)).and_then(|a| mem::r32(a + 4)).unwrap();
    assert_eq!(reg(0x80), 0xF00D);
    assert_eq!(reg(0x88), (0x66_0000_0000u64 >> 8) as u32, "LO_GS keeps the back code");
    assert_eq!(reg(0xC8), (front_code >> 8) as u32);
    assert_eq!(reg(0xC9), 0x55);
    // VGPRs: max(7, 3); shared VGPRs recomputed: front 32+8, back 16+16 ->
    // (40 - 32 + 7) / 64 = 0; user SGPRs (bits 1..5, 27) from the front.
    assert_eq!(reg(0x8A) & 0x3F, 7);
    assert_eq!(reg(0x8B), 0x0800_0004);
    // Halves for different wave sizes do not fuse.
    mem::w32(mem::r64(front + 0x28).unwrap() + 12, 0);
    assert_eq!(e.call("sceAgcFuseShaderHalves_0200", &[fused, front, back, scratch], &[]), 0x8A6C_0008u32 as i32 as i64);
}

/// Oversubscription: no budget leaves both registers clear, an unlimited
/// budget sets both fully, otherwise the target follows the occupancy.
#[test]
fn gs_oversubscription_registers() {
    let e = Env::new();
    let regs = e.a(DATA);
    let gs = shader_object(
        &e,
        e.a(DATA + 0x1000),
        2,
        &[(0xC8, 0), (0xC9, 0)],
        &[(0x291, 64 << 11), (0x2D3, 1), (0x1B1, 0), (0x207, 0), (0x1FF, 64)],
        1 << 22,
    );
    let created = e.a(DATA + 0x80);
    assert_eq!(e.call("sceAgcCreateShader", &[created, gs, 0x10_0000], &[]), OK);
    let read = || mem::read_dwords(regs, 4).unwrap();
    assert_eq!(super::shader::gs_oversubscription(regs, gs, 0, 0.5), OK);
    assert_eq!(read(), vec![0x260, 0, 0x81, 0]);
    assert_eq!(super::shader::gs_oversubscription(regs, gs, u32::MAX, 0.5), OK);
    assert_eq!(read(), vec![0x260, 0x7FF, 0x81, 0x7F_0000]);
    // ow = 2, waves = max(2, 2) = 2: base (1024/2*2, 128*4/2*2) = (1024, 512),
    // expanded (2048, 1528 -> 764*2): limit min(2048, 1528, budget>>5, 1024).
    assert_eq!(super::shader::gs_oversubscription(regs, gs, 0x10000, 1.0), OK);
    let v = read();
    assert_eq!((v[0], v[2]), (0x260, 0x81));
    // target 1024 >= base export 512: export-limited branch.
    assert_eq!(v[1], 0x7FF);
    assert_eq!(v[3], (((1024 - 512) * 127 / (1528 - 512)) as u32) << 16);
}

/// Draws re-resolve only the banks written since the previous draw; every
/// kind of write (direct, table, context-state operation) counts.
#[test]
fn cp_draw_state_follows_every_register_write() {
    let e = Env::new();
    let rec = std::sync::Arc::new(Recorder::default());
    let q = bare_queue();
    let ex: std::sync::Arc<dyn ps5gpu::render::RenderExecutor> = rec.clone();
    q.with_interpreter(0, |i| i.executor = Some(ex));
    let draw = [p4::make(3, op::DRAW_INDEX_AUTO, 0), 3, 2];
    let table = e.a(DATA);
    assert!(mem::write_dwords(table, &[0x8E, 0x3, 0x245, 9]));
    let mut w = Vec::new();
    w.extend_from_slice(&[p4::make(3, op::SET_CONTEXT_REG, 0), 0x8E, 0xF]); // CB_TARGET_MASK
    w.extend_from_slice(&draw);
    w.extend_from_slice(&[p4::make(3, op::SET_SH_REG, 0), 0x0C, 0x77]); // PS user data 0
    w.extend_from_slice(&draw);
    w.extend_from_slice(&[p4::make(5, op::SET_CONTEXT_REG_INDIRECT, 0), table as u32, (table >> 32) as u32, 0x8000_0000, 1]);
    w.extend_from_slice(&draw);
    w.extend_from_slice(&[p4::make(3, op::NOP, nop::CONTEXT_STATE), 3, 0]); // push + clear
    w.extend_from_slice(&draw);
    w.extend_from_slice(&[p4::make(3, op::NOP, nop::CONTEXT_STATE), 2, 0]); // pop
    w.extend_from_slice(&draw);
    run_words(&q, e.a(CMD), &w);
    let draws = rec.draws.lock().unwrap();
    let masks: Vec<u32> = draws.iter().map(|d| d.target_mask).collect();
    assert_eq!(masks, vec![0xF, 0xF, 0x3, 0, 0x3]);
    assert_eq!(draws[0].shaders.ps.user_data[0], 0);
    assert!(draws[1..].iter().all(|d| d.shaders.ps.user_data[0] == 0x77));
}

/// Timing of a 10k-draw frame: building it with the libSceAgc builders and
/// running it through the command processor with a recording executor
/// (`cargo test --release -p ps5hle --lib bench_pm4 -- --ignored --nocapture`).
#[test]
#[ignore]
fn bench_pm4_draw_stream() {
    const DRAWS: usize = 10_000;
    /// Counts draws, keeping only the last target mask (no copy).
    #[derive(Default)]
    struct Count(AtomicU64, AtomicU64);
    impl ps5gpu::render::RenderExecutor for Count {
        fn draw(&self, d: &ps5gpu::render::ResolvedDraw) {
            self.0.fetch_add(1, Ordering::Relaxed);
            self.1.store(d.target_mask as u64 + d.shaders.ps.user_data[0] as u64, Ordering::Relaxed);
        }
        fn dispatch(&self, _: &ps5gpu::render::ResolvedDispatch) {}
    }
    let e = Env::new();
    // A large command space of its own.
    let bytes = 64usize << 20;
    let buf: &'static mut [u8] = Box::leak(vec![0u8; bytes + 0x1000].into_boxed_slice());
    let space = (buf.as_ptr() as u64 + 0xFFF) & !0xFFF;
    ps5vm::ranges::track(space, bytes as u64, ps5vm::MemProtect::RW);
    let cx_table = e.a(DATA);
    let sh_table = e.a(DATA + 0x200);
    let values = e.a(DATA + 0x400);
    let indices = e.a(DATA + 0x800);
    let mut t = Vec::new();
    for i in 0..12u32 {
        t.extend_from_slice(&[0x100 + i, i]);
    }
    assert!(mem::write_dwords(cx_table, &t));
    let mut t = Vec::new();
    for i in 0..6u32 {
        t.extend_from_slice(&[0x08 + i, i]);
    }
    assert!(mem::write_dwords(sh_table, &t));
    assert!(mem::write_dwords(values, &[1, 2, 3, 4, 5, 6, 7, 8]));
    let mut best_build = f64::MAX;
    let mut best_run = f64::MAX;
    let mut best_seq = f64::MAX;
    let mut best_sliced = f64::MAX;
    let mut log_bytes = 0u64;
    let mut packets = 0u64;
    for _ in 0..5 {
        mem::w64(e.a(CB + CB_CURSOR_UP), space);
        mem::w64(e.a(CB + CB_CURSOR_DOWN), space + bytes as u64);
        let t0 = std::time::Instant::now();
        for d in 0..DRAWS {
            e.call("sceAgcCbSetShRegisterRangeDirect", &[e.cb(), 0x0C, values, 8], &[]);
            e.call("sceAgcDcbSetCxRegistersIndirect", &[e.cb(), cx_table, 12], &[]);
            e.call("sceAgcDcbSetShRegistersIndirect", &[e.cb(), sh_table, 6], &[]);
            if d % 50 == 0 {
                e.call("sceAgcDcbSetCxRegisterDirect", &[e.cb(), ((d as u64 & 0xF) << 32) | 0x8E], &[]);
            }
            e.call("sceAgcDcbDrawIndex", &[e.cb(), 36, indices, 0], &[]);
        }
        best_build = best_build.min(t0.elapsed().as_secs_f64());
        let end = mem::r64(e.a(CB + CB_CURSOR_UP)).unwrap();
        let dwords = ((end - space) / 4) as u32;
        let rec = std::sync::Arc::new(Count::default());
        let q = GpuQueue::new(1);
        let ex: std::sync::Arc<dyn ps5gpu::render::RenderExecutor> = rec.clone();
        q.with_interpreter(0, |i| i.executor = Some(ex));
        let before = interp::PACKETS.load(Ordering::Relaxed);
        let t1 = std::time::Instant::now();
        q.submit_graphics(space, dwords);
        run_queue(&q);
        best_run = best_run.min(t1.elapsed().as_secs_f64());
        packets = interp::PACKETS.load(Ordering::Relaxed) - before;
        assert_eq!(rec.0.load(Ordering::Relaxed), DRAWS as u64);
        // Slice mode: the command thread logs, the slice thread resolves.
        let rec = std::sync::Arc::new(Count::default());
        let sx = ps5gpu::slices::SliceExecutor::start(rec.clone(), 128, 4, ps5gpu::slices::Runner::new(false));
        let q = GpuQueue::new(1);
        let ex: std::sync::Arc<dyn ps5gpu::render::RenderExecutor> = sx.clone();
        q.with_interpreter(0, |i| i.executor = Some(ex));
        let log0 = ps5gpu::slices::stats::LOG_BYTES.load(Ordering::Relaxed);
        let t2 = std::time::Instant::now();
        q.submit_graphics(space, dwords);
        run_queue(&q);
        best_seq = best_seq.min(t2.elapsed().as_secs_f64());
        ps5gpu::render::RenderExecutor::drain(&*sx);
        best_sliced = best_sliced.min(t2.elapsed().as_secs_f64());
        assert_eq!(rec.0.load(Ordering::Relaxed), DRAWS as u64);
        log_bytes = ps5gpu::slices::stats::LOG_BYTES.load(Ordering::Relaxed) - log0;
        sx.stop();
    }
    println!(
        "bench_pm4: {DRAWS} draws, {packets} packets: build {:.2} ms ({:.0} ns/draw), process {:.2} ms ({:.0} ns/draw, {:.0} ns/packet)",
        best_build * 1e3,
        best_build * 1e9 / DRAWS as f64,
        best_run * 1e3,
        best_run * 1e9 / DRAWS as f64,
        best_run * 1e9 / packets as f64
    );
    println!(
        "bench_pm4 slice mode: command thread {:.2} ms ({:.0} ns/draw), until the slice thread finished {:.2} ms ({:.0} ns/draw), log {} B/draw",
        best_seq * 1e3,
        best_seq * 1e9 / DRAWS as f64,
        best_sliced * 1e3,
        best_sliced * 1e9 / DRAWS as f64,
        log_bytes / DRAWS as u64
    );
}

/// An indirect draw leaves its last argument record's offsets in the
/// destination shader registers and its instance count as NUM_INSTANCES.
#[test]
fn cp_indirect_draw_updates_draw_state() {
    let e = Env::new();
    let q = bare_queue();
    let base = e.a(DATA);
    // Indexed record at +0x20: indices, instances, first index, base vertex, first instance.
    assert!(mem::write_dwords(base + 0x20, &[36, 5, 7, 11, 13]));
    let modifier = 1 | 2 | 4 | (2 << 9) | (3 << 14) | (4 << 19);
    e.call("sceAgcDcbSetBaseIndirectArgs", &[e.cb(), 0, base], &[]);
    e.call("sceAgcDcbSetNumInstances", &[e.cb(), 2], &[]);
    e.call("sceAgcDcbDrawIndexIndirect", &[e.cb(), 0x20, modifier], &[]);
    submit_cb(&e, &q);
    run_queue(&q);
    q.with_interpreter(0, |i| {
        assert_eq!(i.regs.num_instances, 5);
        assert_eq!((i.regs.shader[0x8E], i.regs.shader[0x8F], i.regs.shader[0x90]), (11, 7, 13));
    });
}

/// A multi-draw with the draw-index destination enabled (modifier bit 3,
/// user SGPR in bits 24..28) leaves the last record's index there, and the
/// draw reaches the back-end with that register.
#[test]
fn cp_multi_draw_leaves_the_draw_index() {
    let e = Env::new();
    let q = bare_queue();
    let base = e.a(DATA);
    for r in 0..3u32 {
        assert!(mem::write_dwords(base + 0x20 + r as u64 * 20, &[36, 1, 7 * r, 0, 0]));
    }
    let modifier: u64 = 8 | (5 << 24);
    e.call("sceAgcDcbSetBaseIndirectArgs", &[e.cb(), 0, base], &[]);
    e.call("sceAgcDcbSetIndexSize", &[e.cb(), 1, 0], &[]);
    e.call("sceAgcDcbDrawIndexIndirectMulti", &[e.cb(), 0x20, 0, 3, 0, 20], &[modifier]);
    submit_cb(&e, &q);
    run_queue(&q);
    q.with_interpreter(0, |i| assert_eq!(i.regs.shader[0x8C + 5], 2, "index of the last record"));
}

/// Indirect dispatches with thread dimensions reach the executor as
/// indirect ones: the back-end reads the record once the GPU work writing
/// it is done (the CP thread would read it too early).
#[test]
fn cp_dispatch_indirect_with_thread_dimensions() {
    let e = Env::new();
    let rec = std::sync::Arc::new(Recorder::default());
    let q = bare_queue();
    let ex: std::sync::Arc<dyn ps5gpu::render::RenderExecutor> = rec.clone();
    q.with_interpreter(0, |i| i.executor = Some(ex));
    let args = e.a(DATA);
    assert!(mem::write_dwords(args, &[64, 2, 1]));
    e.call("sceAgcAcbDispatchIndirect", &[e.cb(), args, 0x20], &[]);
    e.call("sceAgcAcbDispatchIndirect", &[e.cb(), args, 0], &[]);
    submit_cb(&e, &q);
    run_queue(&q);
    let d = rec.dispatches.lock().unwrap();
    assert_eq!((d[0].packet.groups, d[0].packet.indirect_address, d[0].packet.dispatch_initiator & 0x20), ([0, 0, 0], args, 0x20));
    assert_eq!((d[1].packet.groups, d[1].packet.indirect_address), ([0, 0, 0], args));
}

/// In the write-through label mode a label is deferred while an earlier
/// deferred one is unwritten (labels keep their order), and written at once
/// otherwise when no write-through is pending.
#[test]
fn deferred_labels_keep_their_order() {
    use interp::{defer_in_mode, defer_label, LabelMode};
    #[derive(Default)]
    struct Held(Mutex<Vec<Box<dyn FnOnce() + Send>>>);
    impl ps5gpu::render::RenderExecutor for Held {
        fn draw(&self, _: &ps5gpu::render::ResolvedDraw) {}
        fn dispatch(&self, _: &ps5gpu::render::ResolvedDispatch) {}
        fn end_of_pipe(&self, done: Box<dyn FnOnce() + Send>) {
            done();
        }
        fn after_pipe(&self, done: Box<dyn FnOnce() + Send>) {
            self.0.lock().unwrap().push(done);
        }
    }
    let _g = HOOK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let ex = Held::default();
    let through = LabelMode::AfterWriteThroughs;
    assert!(!defer_in_mode(through, &ex), "nothing pending: written at once");
    assert!(defer_in_mode(LabelMode::AfterGpu, &ex));
    assert!(!defer_in_mode(LabelMode::Recorded, &ex));
    let order = std::sync::Arc::new(Mutex::new(Vec::new()));
    let o = order.clone();
    defer_label(&ex, move || o.lock().unwrap().push(1));
    assert!(defer_in_mode(through, &ex), "an older label is still deferred");
    let o = order.clone();
    defer_label(&ex, move || o.lock().unwrap().push(2));
    // The GPU completes: the callbacks run in queue order.
    let held: Vec<_> = std::mem::take(&mut *ex.0.lock().unwrap());
    for f in held {
        f();
    }
    assert_eq!(*order.lock().unwrap(), [1, 2]);
    assert!(!defer_in_mode(through, &ex), "all deferred labels written");
}

#[test]
fn label_modes_parse() {
    use interp::{parse_label_mode, LabelMode};
    assert_eq!(parse_label_mode(None, false), LabelMode::Recorded);
    assert_eq!(parse_label_mode(None, true), LabelMode::AfterGpu);
    assert_eq!(parse_label_mode(Some("1"), false), LabelMode::AfterGpu);
    assert_eq!(parse_label_mode(Some("0"), true), LabelMode::Recorded);
    assert_eq!(parse_label_mode(Some(" through "), false), LabelMode::AfterWriteThroughs);
    assert_eq!(parse_label_mode(Some("yes"), false), LabelMode::Recorded);
}

/// `sceAgcGetDefaultCxStateFlat`: version 8's context defaults as a flat
/// array of 512 registers, copied up to `size` bytes (at most 2048).
#[test]
fn default_cx_state_flat() {
    let e = Env::new();
    let out = e.a(DATA);
    assert!(mem::fill(out, 0x804, 0xCC));
    assert_eq!(e.call("sceAgcGetDefaultCxStateFlat", &[0, 2048], &[]), 0x8089_0001u32 as i32 as i64);
    assert_eq!(e.call("sceAgcGetDefaultCxStateFlat", &[out, 0], &[]), 0);
    assert_eq!(mem::r32(out), Some(0xCCCC_CCCC), "a zero size copies nothing");
    assert_eq!(e.call("sceAgcGetDefaultCxStateFlat", &[out, 0x804], &[]), 0);
    let r = |reg: u64| mem::r32(out + reg * 4);
    assert_eq!((r(0x001), r(0x007), r(0x12d), r(0x1e0)), (Some(0x1100_0100), Some(0xFFFF), Some(0x3F80_0000), Some(0x2001_0001)));
    assert_eq!(r(512), Some(0xCCCC_CCCC), "at most 2048 bytes");
}

/// `sceAgcGetGsPrimPayload`: 8 bytes when a context register sets
/// SPI_SHADER_IDX_FORMAT to 2 components (the low nibble), else 0.
#[test]
fn gs_prim_payload() {
    let e = Env::new();
    let (gs, regs, out) = (e.a(DATA), e.a(DATA + 0x100), e.a(DATA + 0x200));
    let run = |pairs: &[(u32, u32)], count: u8| {
        assert!(mem::fill(gs, 0x60, 0));
        assert!(mem::w64(gs + 0x18, regs));
        assert!(mem::w8(gs + 0x5B, count));
        for (i, &(o, v)) in pairs.iter().enumerate() {
            assert!(mem::w32(regs + i as u64 * 8, o) && mem::w32(regs + i as u64 * 8 + 4, v));
        }
        assert!(mem::w32(out, 0xCCCC_CCCC));
        assert_eq!(e.call("sceAgcGetGsPrimPayload", &[out, gs], &[]), 0);
        mem::r32(out).unwrap()
    };
    assert_eq!(run(&[(0x1C2, 2)], 1), 8);
    assert_eq!(run(&[(0x1C2, 0xFFFF_FFF2)], 1), 8);
    for v in [0, 1, 3, 4, 0x20] {
        assert_eq!(run(&[(0x1C2, v)], 1), 0, "{v:#x}");
    }
    assert_eq!(run(&[(0x1C2, 4), (0x1C2, 2)], 2), 8, "the search goes on past a mismatch");
    assert_eq!(run(&[(0x1C3, 0), (0x1C2, 2)], 1), 0, "only the counted registers");
    assert_ne!(e.call("sceAgcGetGsPrimPayload", &[0, gs], &[]), 0);
}

/// `sceAgcAcbAtomicGds_0900`: the ATOMIC_GDS packet (reference vectors).
#[test]
fn acb_atomic_gds_packet() {
    use super::builders::atomic_gds_packet;
    assert_eq!(
        atomic_gds_packet(0xFFFF_FFFF, 1, 0xFFFF_FFFF, 0xFFFF_FFFF, 0xABCD, 0xFEDC, 0x1234_5678, 0x8765_4321, 0xFEDC_BA98_7654_3210, 0x0123_4567_89AB_CDEF),
        [0xC009_1D00, 0x0015_00FF, 0x13F, 0xABCD, 0xFEDC, 0x0034_0078, 0x8765_4321, 0x7654_3210, 0xFEDC_BA98, 0x89AB_CDEF, 0x0123_4567]
    );
    for (control, mode, op) in [(0, 0, 0), (0, 1, 0x2_0000), (0, 2, 0), (0, 0xFFFF_FFFF, 0), (0x20, 1, 0x4_0020), (0x40, 1, 0x12_0040), (0x200, 1, 0x3_0000), (0xFFFF_FD00, 0, 0), (0xFF, 1, 0x14_00FF)] {
        let w = atomic_gds_packet(control, mode, 0xFFFF_FFFE, 0xFFFF_FFC0, 0, 0, 0xFF00_FF00, 0, 0, 0);
        assert_eq!(w[1], op, "control {control:#x} mode {mode}");
        assert_eq!(&w[2..], &[0; 9]);
    }
    let e = Env::new();
    let at = e.cursor();
    let ret = e.call("sceAgcAcbAtomicGds_0900", &[e.cb(), 0xFF, 1, 1, 5, 0x10], &[2, 0xFFFF_FFFF, 7, 0x1_0000_0002, 3]) as u64;
    assert_eq!(ret, at);
    assert_eq!(mem::read_dwords(at, 11).unwrap(), atomic_gds_packet(0xFF, 1, 1, 5, 0x10, 2, 0xFFFF_FFFF, 7, 0x1_0000_0002, 3).to_vec());
    e.reset_cb();
}

/// Submit validation is unavailable in the retail library; shader
/// debugging reports 1. Workload streams that are not registered (or a
/// null buffer) get no inactive packet.
#[test]
fn validation_queries_and_inactive_workload_streams() {
    let e = Env::new();
    for name in ["sceAgcDriverSetValidationErrorOutputFrequency", "sceAgcDriverSetSubmitValidationMode", "sceAgcDriverGetSubmitValidationConfig", "sceAgcDriverGetSubmitValidationMode", "sceAgcDriverSetSubmitValidationConfig"] {
        assert_eq!(e.call(name, &[e.a(DATA)], &[]), 0x8A6C_1000u32 as i32 as i64, "{name}");
    }
    assert_eq!(e.call("sceAgcDriverGetShaderDebuggingStatus", &[], &[]), 1);
    let at = e.cursor();
    assert_eq!(e.call("sceAgcDcbSetWorkloadStreamInactive", &[e.cb(), 30], &[]), 0);
    assert_eq!(e.call("sceAgcAcbSetWorkloadStreamInactive", &[0, 1], &[]), 0);
    assert_eq!(e.cursor(), at);
}

/// COPY_DATA from a register (SRC_SEL 0) copies the value the queue holds
/// for a shader, context or user config register (two with COUNT_SEL); a
/// dispatch leaves COMPUTE_DISPATCH_INITIATOR and its direct group counts
/// in the shader registers.
#[test]
fn copy_data_from_registers() {
    let e = Env::new();
    let dst = e.a(LABELS + 0x300);
    let copy = |control: u32, reg: u32| [p4::make(6, op::COPY_DATA, 0), control, reg, 0, dst as u32, (dst >> 32) as u32];
    let run = |words: &[u32]| {
        assert!(mem::write_dwords(dst, &[0xDEAD_BEEF, 0xDEAD_BEEF]));
        let q = bare_queue();
        run_words(&q, e.a(CMD + 0x1000), words);
        (mem::r32(dst).unwrap(), mem::r32(dst + 4).unwrap())
    };
    let set_sh = [p4::make(4, op::SET_SH_REG, 0), 0x20C, 0x0275_4800, 2];
    assert_eq!(run(&[&set_sh[..], &copy(0x0010_0200, 0x2E0C)].concat()), (0x0275_4800, 0xDEAD_BEEF));
    assert_eq!(run(&[&set_sh[..], &copy(0x0011_0200, 0x2E0C)].concat()), (0x0275_4800, 2));
    let set_cx = [p4::make(3, op::SET_CONTEXT_REG, 0), 0x10, 0xABC];
    assert_eq!(run(&[&set_cx[..], &copy(0x0010_0200, 0xA010)].concat()), (0xABC, 0xDEAD_BEEF));
    let dispatch = [p4::make(5, op::DISPATCH_DIRECT, 0), 4, 2, 1, 0x8041];
    assert_eq!(run(&[&dispatch[..], &copy(0x0010_0200, 0x2E00)].concat()).0, 0x8041);
    assert_eq!(run(&[&dispatch[..], &copy(0x0011_0200, 0x2E01)].concat()), (4, 2));
}

/// An empty register list writes nothing and returns the cursor; with a
/// null buffer it fails (reference cf914b510).
#[test]
fn empty_register_list_writes_nothing() {
    let e = Env::new();
    let at = e.cursor();
    assert!(mem::w32(at, 0xCCCC_CCCC));
    for name in ["sceAgcCbSetShRegistersDirect", "sceAgcCbSetUcRegistersDirect"] {
        assert_eq!(e.call(name, &[e.cb(), 0, 0], &[]) as u64, at, "{name}");
        assert_eq!(e.cursor(), at, "{name}: the cursor moved");
        assert_eq!(e.call(name, &[0, 0, 0], &[]), 0, "{name}: null buffer");
    }
    assert_eq!(mem::r32(at), Some(0xCCCC_CCCC));
}

/// INDEX_ATTRIBUTES_INDIRECT (0x91) loads the index base, size and type
/// from record `index` of a 16-byte table (reference dd3c13ffd).
#[test]
fn index_attributes_indirect_loads_the_record() {
    let e = Env::new();
    let table = e.a(DATA + 0x400);
    let ia = 0x1234_5678_9A00u64;
    assert!(mem::write_dwords(table, &[0, 0, 0, 0, ia as u32, (ia >> 32) as u32, 12, 1]));
    let q = bare_queue();
    run_words(&q, e.a(CMD + 0x1400), &[p4::make(4, op::INDEX_ATTRIBUTES_INDIRECT, 0), table as u32, (table >> 32) as u32, 1]);
    let (base, size, ty) = q.with_interpreter(0, |i| (i.index_base, i.index_buffer_size, i.index_type));
    assert_eq!((base, size, ty), (ia, 12, 1));
}

/// Any type-2 header is a one-dword filler, not only 0x80000000
/// (reference FillerPacket).
#[test]
fn cp_type2_headers_are_fillers() {
    let e = Env::new();
    let l = e.a(LABELS + 0x400);
    assert!(mem::w32(l, 0));
    let q = bare_queue();
    let mut w = vec![0x8000_1234, 0xBFFF_FFFF];
    w.extend_from_slice(&write_data(l, 1));
    run_words(&q, e.a(CMD + 0x1800), &w);
    assert_eq!(q.pending(), 0);
    assert_eq!(mem::r32(l), Some(1));
}

/// SET_SH_REG drops the selector bits 30:28 of its offset, as the other
/// register banks do.
#[test]
fn cp_set_sh_reg_masks_the_selector_bits() {
    let e = Env::new();
    let q = bare_queue();
    run_words(&q, e.a(CMD + 0x1900), &[p4::make(4, op::SET_SH_REG, 0), 0x1000_0010, 7, 8]);
    q.with_interpreter(0, |i| assert_eq!((i.regs.shader[0x10], i.regs.shader[0x11]), (7, 8)));
}

/// SET_*_REG_INDIRECT with a first register other than 0x80000000 reads
/// that many consecutive values (reference Pm4.cpp ReadIndirectRegisters).
#[test]
fn cp_indirect_registers_contiguous_run() {
    let e = Env::new();
    let values = e.a(DATA + 0x500);
    assert!(mem::write_dwords(values, &[0x11, 0x22, 0x33]));
    let q = bare_queue();
    let mut w = Vec::new();
    for (opcode, first) in [(op::SET_SH_REG_INDIRECT, 0x20), (op::SET_CONTEXT_REG_INDIRECT, 0x30), (op::SET_UCONFIG_REG_INDIRECT, 0x241)] {
        w.extend_from_slice(&[p4::make(5, opcode, 0), values as u32, (values >> 32) as u32, first, 3]);
    }
    run_words(&q, e.a(CMD + 0x1A00), &w);
    q.with_interpreter(0, |i| {
        assert_eq!(&i.regs.shader[0x20..0x23], &[0x11, 0x22, 0x33]);
        assert_eq!(&i.regs.context[0x30..0x33], &[0x11, 0x22, 0x33]);
        assert_eq!(&i.regs.uconfig[0x241..0x244], &[0x11, 0x22, 0x33]);
        assert_eq!(i.index_type, 0x33 & 3, "VGT_INDEX_TYPE written by the run");
    });
}

/// WRITE_DATA with DST_SEL 0 writes registers at absolute offsets
/// (reference Pm4.cpp: user config 0xC340..); WR_ONE_ADDR writes every
/// value to the same register.
#[test]
fn cp_write_data_to_registers() {
    let e = Env::new();
    let q = bare_queue();
    let mut w = vec![p4::make(6, op::WRITE_DATA, 0), 0, 0xC340, 0, 5, 6];
    w.extend_from_slice(&[p4::make(6, op::WRITE_DATA, 0), 1 << 16, 0xC344, 0, 8, 9]);
    run_words(&q, e.a(CMD + 0x1B00), &w);
    q.with_interpreter(0, |i| {
        assert_eq!((i.regs.uconfig[0x340], i.regs.uconfig[0x341]), (5, 6));
        assert_eq!((i.regs.uconfig[0x344], i.regs.uconfig[0x345]), (9, 0));
    });
}

/// COPY_DATA with DST_SEL 0 loads context registers from memory: one
/// dword, two with COUNT_SEL (reference Pm4.cpp CopyDataToRegister).
#[test]
fn cp_copy_data_to_context_registers() {
    let e = Env::new();
    let src = e.a(DATA + 0x600);
    assert!(mem::write_dwords(src, &[0xAAAA, 0xBBBB]));
    let q = bare_queue();
    let copy = |control: u32, reg: u32| [p4::make(6, op::COPY_DATA, 0), control, src as u32, (src >> 32) as u32, reg, 0];
    let w = [copy(0x0000_0001, 0xA010), copy(0x0001_0002, 0xA020)].concat();
    run_words(&q, e.a(CMD + 0x1C00), &w);
    q.with_interpreter(0, |i| {
        assert_eq!((i.regs.context[0x10], i.regs.context[0x11]), (0xAAAA, 0));
        assert_eq!((i.regs.context[0x20], i.regs.context[0x21]), (0xAAAA, 0xBBBB));
    });
}

/// COPY_DATA source selector 1 (memory) read by the PFP engine (bit 30) is
/// memory, not GDS (raw selector 3), both into memory and into registers.
#[test]
fn cp_copy_data_memory_source_on_the_pfp() {
    let e = Env::new();
    let src = e.a(DATA + 0x640);
    let dst = e.a(LABELS + 0x380);
    assert!(mem::write_dwords(src, &[0x1234_5678, 0x9ABC_DEF0]));
    assert!(mem::write_dwords(dst, &[0, 0]));
    let q = bare_queue();
    let to_mem = [p4::make(6, op::COPY_DATA, 0), 0x4001_0201, src as u32, (src >> 32) as u32, dst as u32, (dst >> 32) as u32];
    let to_reg = [p4::make(6, op::COPY_DATA, 0), 0x4000_0001, src as u32, (src >> 32) as u32, 0xA030, 0];
    run_words(&q, e.a(CMD + 0x1D00), &[to_mem, to_reg].concat());
    assert_eq!((mem::r32(dst), mem::r32(dst + 4)), (Some(0x1234_5678), Some(0x9ABC_DEF0)));
    q.with_interpreter(0, |i| assert_eq!(i.regs.context[0x30], 0x1234_5678));
}

/// sceAgcDcbClearState: CLEAR_STATE with the low four command bits
/// (reference Commands.cpp testClearState).
#[test]
fn dcb_clear_state_packet() {
    let e = Env::new();
    let start = e.cursor();
    for command in 0..=0xFu64 {
        let p = e.call("sceAgcDcbClearState", &[e.cb(), command], &[]) as u64;
        assert_eq!(p, start + command * 8);
        assert_eq!(mem::read_dwords(p, 2).unwrap(), vec![0xC000_1200, command as u32]);
    }
    let p = e.call("sceAgcDcbClearState", &[e.cb(), 0x10], &[]) as u64;
    assert_eq!(mem::read_dwords(p, 2).unwrap(), vec![0xC000_1200, 0]);
    let p = e.call("sceAgcDcbClearState", &[e.cb(), 0xFFFF_FFFF], &[]) as u64;
    assert_eq!(mem::read_dwords(p, 2).unwrap(), vec![0xC000_1200, 0xF]);
    assert_eq!(e.cursor(), start + 36 * 4);
    assert_eq!(e.call("sceAgcDcbClearState", &[0, 0], &[]), 0);
}

/// Config registers are written with SET_CONFIG_REG (0x68), not
/// SET_CONTEXT_REG (reference RegisterCommands.cpp testConfig).
#[test]
fn dcb_set_cf_register_packets() {
    let e = Env::new();
    let start = e.cursor();
    assert!(mem::write_dwords(start, &[0xABCD_EF01; 16]));
    let direct = e.call("sceAgcDcbSetCfRegisterDirect", &[e.cb(), 0x1234_5678u64 << 32 | 0x2468], &[]) as u64;
    assert_eq!(direct, start);
    assert_eq!(mem::read_dwords(direct, 3).unwrap(), vec![0xC001_6800, 0x2468, 0x1234_5678]);
    let values = e.a(DATA + 0x700);
    assert!(mem::write_dwords(values, &[1, 2, 3]));
    let range = e.call("sceAgcDcbSetCfRegisterRangeDirect", &[e.cb(), 0x100, values, 3], &[]) as u64;
    assert_eq!(range, direct + 12);
    assert_eq!(mem::read_dwords(range, 5).unwrap(), vec![0xC003_6800, 0x100, 1, 2, 3]);
    let reserved = e.call("sceAgcDcbSetCfRegisterRangeDirect", &[e.cb(), 0x200, 0, 2], &[]) as u64;
    assert_eq!(reserved, range + 20);
    assert_eq!(mem::read_dwords(reserved, 5).unwrap(), vec![0xC002_6800, 0x200, 0xABCD_EF01, 0xABCD_EF01, 0xABCD_EF01]);
    let end = e.cursor();
    assert_eq!(end, reserved + 16);
    for (name, args) in [
        ("sceAgcDcbSetCfRegisterDirect", vec![e.cb(), 0x1_0000]),
        ("sceAgcDcbSetCfRegisterDirect", vec![0, 1u64 << 32]),
        ("sceAgcDcbSetCfRegisterRangeDirect", vec![e.cb(), 0x100, values, 0]),
        ("sceAgcDcbSetCfRegisterRangeDirect", vec![e.cb(), 0xFFFF, values, 2]),
    ] {
        assert_eq!(e.call(name, &args, &[]), 0, "{name} {args:x?}");
    }
    assert_eq!(e.cursor(), end, "a rejected call wrote packets");
    // The command processor consumes the packet without touching a bank.
    let q = bare_queue();
    let mut w = mem::read_dwords(start, ((end - start) / 4) as usize).unwrap();
    w.extend_from_slice(&write_data(e.a(LABELS + 0x500), 1));
    run_words(&q, e.a(CMD + 0x2000), &w);
    assert_eq!(mem::r32(e.a(LABELS + 0x500)), Some(1));
    q.with_interpreter(0, |i| assert_eq!(i.regs.context[0x100], 0));
}

/// sceAgcDcbGetSystemSoftwareVersion reads the first payload dword of a
/// type-3 packet of three or more dwords (reference PacketInfo.cpp).
#[test]
fn dcb_get_system_software_version() {
    let e = Env::new();
    let (out, packet) = (e.a(LABELS + 0x600), e.a(DATA + 0x800));
    assert!(mem::write_dwords(packet, &[p4::make(3, op::NOP, 0), 0x0B00_0001, 0]));
    assert_eq!(e.call("sceAgcDcbGetSystemSoftwareVersion", &[out, packet], &[]), 0);
    assert_eq!(mem::r32(out), Some(0x0B00_0001));
    assert!(mem::write_dwords(packet, &[p4::make(2, op::NOP, 0), 7]));
    assert_ne!(e.call("sceAgcDcbGetSystemSoftwareVersion", &[out, packet], &[]), 0, "a two-dword packet carries no version");
    assert!(mem::write_dwords(packet, &[0x8000_0000, 7, 0]));
    assert_ne!(e.call("sceAgcDcbGetSystemSoftwareVersion", &[out, packet], &[]), 0, "not a type-3 packet");
    assert_ne!(e.call("sceAgcDcbGetSystemSoftwareVersion", &[0, packet], &[]), 0);
    assert_eq!(mem::r32(out), Some(0x0B00_0001));
}

/// sceAgcGetRegisterDefaultsInternal returns the version 0 internal table;
/// sceAgcGetRegisterDefaults its own object, not a versioned one
/// (reference Commands.cpp).
#[test]
fn register_defaults_without_version() {
    let e = Env::new();
    let public = e.call("sceAgcGetRegisterDefaults", &[], &[]);
    let internal = e.call("sceAgcGetRegisterDefaultsInternal", &[], &[]);
    assert_ne!(public, 0);
    assert_ne!(internal, 0);
    assert_ne!(public, internal, "the internal table is not the public one");
    assert_eq!(public, e.call("sceAgcGetRegisterDefaults", &[], &[]), "one object per process");
    assert_ne!(public, e.call("sceAgcGetRegisterDefaults2", &[13], &[]), "the compact versioned table was reused");
    assert_eq!(internal, e.call("sceAgcGetRegisterDefaults2Internal", &[0], &[]));
}

/// Find the MRT0 colour-target block the way a title does: walk the type
/// table as `{key, index << 2 | bank}` pairs, then read the context block
/// (reference Commands.cpp testDefaultsTypes).
fn colour_target_block(root: u64) -> u32 {
    let context_blocks = mem::r64(root).unwrap();
    let types = mem::r64(root + 0x30).unwrap();
    let count = mem::r32(root + 0x38).unwrap();
    assert!(context_blocks != 0 && types != 0 && count > 0);
    let (mut contexts, mut matches, mut block) = (0, 0, 0);
    for entry in 0..count as u64 {
        let key = mem::r32(types + entry * 8).unwrap();
        let bank_index = mem::r32(types + entry * 8 + 4).unwrap();
        if bank_index & 3 == 0 {
            contexts += 1;
        }
        if key != 0x38E9_2C91 {
            continue;
        }
        assert_eq!(bank_index & 3, 0, "MRT0 defaults are not in the context bank");
        block = bank_index >> 2;
        matches += 1;
    }
    assert!(matches == 1 && block < contexts, "MRT0 key missing, repeated or outside the context blocks");
    let group = mem::r64(context_blocks + block as u64 * 8).unwrap();
    assert_ne!(group, 0);
    let offsets = [0x318, 0x31b, 0x31c, 0x31d, 0x31e, 0x31f, 0x321, 0x323, 0x324, 0x325, 0x390, 0x398, 0x3a0, 0x3a8, 0x3b0, 0x3b8];
    for (i, &o) in offsets.iter().enumerate() {
        assert_eq!(mem::r32(group + i as u64 * 8), Some(o), "MRT0 block register {i}");
    }
    block
}

#[test]
fn register_defaults_types_as_key_bank_pairs() {
    let e = Env::new();
    let legacy = e.call("sceAgcGetRegisterDefaults", &[], &[]) as u64;
    assert_eq!(mem::r32(legacy + 0x38), Some(137));
    assert_eq!(colour_target_block(legacy), 78, "MRT0 defaults are block 78 of the 137-entry table");
    // Every version from 4 has a valid MRT0 block in its pairs.
    for v in 4..=13 {
        let pairs = defaults::key_bank_pairs(defaults::select(v).unwrap().0).unwrap();
        let mrt0: Vec<_> = pairs.iter().filter(|p| p[0] == 0x38E9_2C91).collect();
        assert_eq!(mrt0.len(), 1, "v{v}");
    }
    let v10 = defaults::key_bank_pairs(defaults::select(10).unwrap().0).unwrap();
    assert_eq!(v10.iter().find(|p| p[0] == 0x38E9_2C91).unwrap()[1] >> 2, 73);
    // The versioned calls keep their compact 3-dword entries.
    let compact = |root: u64, count: u32, key: u32, packed: u32| {
        let types = mem::r64(root + 0x30).unwrap();
        assert_eq!(mem::r32(root + 0x38), Some(count));
        assert_eq!(mem::read_dwords(types, 3).unwrap(), vec![key, packed, 0]);
    };
    compact(e.call("sceAgcGetRegisterDefaults2", &[0], &[]) as u64, 150, 0xE24F_806D, 0x0004_0400);
    compact(e.call("sceAgcGetRegisterDefaults2", &[10], &[]) as u64, 128, 0xE24F_806D, 0x0004_0400);
    compact(e.call("sceAgcGetRegisterDefaults2Internal", &[0], &[]) as u64, 15, 0xF58F_EA31, 0x0004_0400);
}

/// The unversioned and graphics forms write the packets of their versioned
/// and compute twins; sceAgcCbMemSemaphore writes the console's 4-dword
/// MEM_SEMAPHORE (reference Semaphore.cpp, EndOfShaderActionSize.cpp,
/// ContextStateSize.cpp, PayloadRange.cpp).
#[test]
fn unversioned_and_graphics_twins_write_the_same_packets() {
    let e = Env::new();
    // sceAgcCbMemSemaphore: the address rounded down to 16 bytes, control
    // 0x80100000 whatever the other arguments.
    for (addr, args) in [(0x1111_1111_2222_2222u64, [0x3333_3333u64, 0x4444_4444, 0]), (1, [2, 3, 4]), (e.a(LABELS + 0x18), [0x1234_5678, 0x9ABC_DEF0, 0x55AA_55AA])] {
        let at = e.cursor();
        let p = e.call("sceAgcCbMemSemaphore", &[e.cb(), addr, args[0], args[1], args[2]], &[]) as u64;
        assert_eq!(p, at);
        let a = addr & !0xF;
        assert_eq!(mem::read_dwords(p, 4).unwrap(), vec![0xC002_3900, a as u32, (a >> 32) as u32, 0x8010_0000]);
        assert_eq!(e.cursor(), at + 16);
    }
    // sceAgcDcbAtomicGds(_0900) and sceAgcAcbAtomicGds_0900.
    let stack = [0x200u64, 0x00FF_00FF, 0x1234_5678, 0x1122_3344_5566_7788, 0x99AA_BBCC_DDEE_FF00];
    let mut packets = Vec::new();
    for name in ["sceAgcDcbAtomicGds_0900", "sceAgcDcbAtomicGds", "sceAgcAcbAtomicGds_0900"] {
        let at = e.cursor();
        let p = e.call(name, &[e.cb(), 0x261, 1, 3, 7, 0x100], &stack) as u64;
        assert_eq!(p, at, "{name}");
        assert_eq!(e.cursor() - at, e.call("sceAgcDcbAtomicGdsGetSize", &[], &[]) as u64, "{name}");
        packets.push(mem::read_dwords(p, 11).unwrap());
    }
    assert_eq!(packets[0][0], 0xC009_1D00);
    assert!(packets.iter().all(|p| *p == packets[0]));
    // sceAgcDcbContextStateOp.
    for operation in 0..4u64 {
        let a = e.cursor();
        assert_eq!(e.call("sceAgcDcbContextStateOp_0100", &[e.cb(), operation], &[]) as u64, a);
        let b = e.cursor();
        assert_eq!(e.call("sceAgcDcbContextStateOp", &[e.cb(), operation], &[]) as u64, b);
        let c = e.cursor();
        assert_eq!(c - b, b - a);
        assert_eq!(c - b, e.call("sceAgcDcbContextStateOpGetSize", &[operation], &[]) as u64);
        assert_eq!(mem::read_dwords(a, ((b - a) / 4) as usize), mem::read_dwords(b, ((c - b) / 4) as usize));
    }
    // sceAgcGetDataPacketPayloadAddress.
    let (out, packet) = (e.a(LABELS + 0x40), e.a(DATA + 0x800));
    assert!(mem::write_dwords(packet, &[p4::make(4, op::NOP, 0), 1, 2, 3]));
    for ty in 0..2u64 {
        assert_eq!(e.call("sceAgcGetDataPacketPayloadAddress_0090", &[out, packet, ty], &[]), 0);
        let versioned = mem::r64(out);
        assert_eq!(e.call("sceAgcGetDataPacketPayloadAddress", &[out, packet, ty], &[]), 0);
        assert_eq!(mem::r64(out), versioned);
    }
    assert_ne!(e.call("sceAgcGetDataPacketPayloadAddress", &[0, packet, 1], &[]), 0);
    assert_ne!(e.call("sceAgcGetDataPacketPayloadAddress", &[out, 0, 1], &[]), 0);
}

/// sceAgcSetSubmitMode accepts mode 0 only.
#[test]
fn set_submit_mode_accepts_mode_zero() {
    let e = Env::new();
    assert_eq!(e.call("sceAgcSetSubmitMode", &[0], &[]), 0);
    assert_ne!(e.call("sceAgcSetSubmitMode", &[1], &[]), 0);
}

/// A shader semantic word: id, hardware mapping, is_f16.
fn semantic(id: u32, mapping: u32, f16: u32) -> u32 {
    id | mapping << 8 | f16 << 20
}
const SEM_FLAT: u32 = 1 << 22;
const SEM_CUSTOM: u32 = 1 << 24;
const fn sem_defaults(lo: u32, hi: u32) -> u32 {
    lo << 28 | hi << 30
}

/// A created shader object (absolute pointers) of type `ty` with the given
/// input and output semantics and GS specials at +0x300.
fn semantic_shader(at: u64, ty: u8, inputs: &[u32], outputs: &[u32]) -> u64 {
    let (ins, outs, specials) = (at + 0x80, at + 0x180, at + 0x300);
    assert!(mem::write(at, &[0u8; 0x60]));
    assert!(mem::write_dwords(ins, inputs) || inputs.is_empty());
    assert!(mem::write_dwords(outs, outputs) || outputs.is_empty());
    mem::w64(at + 0x28, specials);
    mem::w64(at + 0x30, if inputs.is_empty() { 0 } else { ins });
    mem::w64(at + 0x38, if outputs.is_empty() { 0 } else { outs });
    mem::w32(at + 0x50, inputs.len() as u32);
    assert!(mem::write(at + 0x56, &(outputs.len() as u16).to_le_bytes()));
    mem::w8(at + 0x5A, ty);
    // GE_CNTL, VGT_SHADER_STAGES_EN, VGT_GS_OUT_PRIM_TYPE, GE_USER_VGPR_EN.
    assert!(mem::write_dwords(specials, &[0x25B, 0x123, 0x2D5, 0, 0, 0, 0, 0, 0x29B, 2, 0x262, 7]));
    at
}

fn interpolant_registers(regs: u64, n: usize) -> Vec<(u32, u32)> {
    mem::read_dwords(regs, n * 2).unwrap().chunks_exact(2).map(|p| (p[0], p[1])).collect()
}

/// sceAgcUpdateInterpolantMapping_10000 writes the second-generation
/// mapping of the PS inputs only (reference InterpolantMapping.cpp
/// testMapping / testUpdateMapping / testIdentity / testRejections).
#[test]
fn update_interpolant_mapping_10000() {
    const SPI_PS_INPUT_CNTL_0: u32 = 0x191;
    let e = Env::new();
    let (gs_at, ps_at, regs, create) = (e.a(DATA + 0x1000), e.a(DATA + 0x1400), e.a(DATA + 0x1800), e.a(DATA + 0x1A00));
    let canary = [0xCCCC_CCCCu32; 64];
    let check_mapping = |inputs: &[u32], outputs: &[u32], expected: &[u32]| {
        let gs = semantic_shader(gs_at, 2, &[], outputs);
        let ps = semantic_shader(ps_at, 1, inputs, &[]);
        assert!(mem::write_dwords(regs, &canary));
        assert_eq!(e.call("sceAgcUpdateInterpolantMapping_10000", &[regs, gs, ps], &[]), 0);
        let got = interpolant_registers(regs, inputs.len());
        for (i, &(offset, value)) in got.iter().enumerate() {
            assert_eq!(offset, SPI_PS_INPUT_CNTL_0 + i as u32);
            assert_eq!(value, expected[i], "input {i}: {value:#010x}");
        }
        assert_eq!(mem::read_dwords(regs + inputs.len() as u64 * 8, 64 - inputs.len() * 2).unwrap(), canary[inputs.len() * 2..].to_vec(), "wrote past its inputs");
        // The same values as the full second-generation mapping.
        assert_eq!(e.call("sceAgcCreateInterpolantMapping_10000", &[create, gs, ps], &[]), 0);
        assert_eq!(interpolant_registers(create, inputs.len()), got);
    };
    let outputs = [semantic(1, 7, 3), semantic(2, 4, 1), semantic(3, 9, 0), semantic(4, 2, 1)];
    let inputs = [
        semantic(1, 0, 2) | SEM_FLAT | sem_defaults(1, 3),
        semantic(2, 0, 2) | sem_defaults(0, 1),
        semantic(5, 0, 2) | SEM_FLAT | sem_defaults(0, 2),
        semantic(3, 0, 0) | SEM_FLAT | sem_defaults(2, 0),
        semantic(4, 0, 3),
    ];
    check_mapping(&inputs, &outputs, &[0x0268_0707, 0x0228_0124, 0x0248_0220, 0x0000_0609, 0x0318_0002]);
    let outputs = [semantic(1, 5, 0), semantic(2, 7, 1), semantic(4, 9, 3)];
    let mut inputs = [
        semantic(1, 0, 0) | SEM_CUSTOM | sem_defaults(2, 0),
        semantic(2, 0, 1) | sem_defaults(1, 1),
        semantic(3, 0, 2) | sem_defaults(0, 2),
        semantic(4, 0, 3) | SEM_CUSTOM | sem_defaults(3, 3),
    ];
    check_mapping(&inputs, &outputs, &[0x0000_0625, 0x0138_0107, 0x0248_0220, 0x0368_0709]);
    // Low-half f16 inputs whose low and high defaults differ are refused.
    for index in [1, 3] {
        let saved = inputs[index];
        inputs[index] &= !sem_defaults(3, 0);
        let gs = semantic_shader(gs_at, 2, &[], &outputs);
        let ps = semantic_shader(ps_at, 1, &inputs, &[]);
        assert!(mem::write_dwords(regs, &canary));
        assert_ne!(e.call("sceAgcUpdateInterpolantMapping_10000", &[regs, gs, ps], &[]), 0);
        assert_eq!(mem::read_dwords(regs, 64).unwrap(), canary.to_vec(), "a refused update wrote registers");
        inputs[index] = saved;
    }
    // No inputs: nothing written, even without a GS.
    let ps = semantic_shader(ps_at, 1, &[], &[]);
    assert!(mem::write_dwords(regs, &canary));
    assert_eq!(e.call("sceAgcUpdateInterpolantMapping_10000", &[regs, 0, ps], &[]), 0);
    assert_eq!(mem::read_dwords(regs, 64).unwrap(), canary.to_vec());
    // Rejections.
    let gs = semantic_shader(gs_at, 2, &[], &[]);
    let ps = semantic_shader(ps_at, 1, &[semantic(1, 0, 0); 33], &[]);
    assert_ne!(e.call("sceAgcUpdateInterpolantMapping_10000", &[regs, gs, ps], &[]), 0);
    let ps = semantic_shader(ps_at, 1, &[semantic(1, 0, 0)], &[]);
    assert_ne!(e.call("sceAgcUpdateInterpolantMapping_10000", &[regs, 0, ps], &[]), 0);
    assert_ne!(e.call("sceAgcUpdateInterpolantMapping_10000", &[0, gs, ps], &[]), 0);
    assert_ne!(e.call("sceAgcUpdateInterpolantMapping_10000", &[regs, gs, 0], &[]), 0);
    assert_eq!(mem::read_dwords(regs, 64).unwrap(), canary.to_vec());
}

/// sceAgcLinkShaders: CreatePrimState's two context and three user config
/// registers, then the 32 interpolant registers of CreateInterpolantMapping
/// (reference LinkShaders.cpp); an invalid program writes nothing.
#[test]
fn link_shaders_combines_prim_state_and_interpolants() {
    let e = Env::new();
    let (gs_at, ps_at, cx, uc) = (e.a(DATA + 0x1000), e.a(DATA + 0x1400), e.a(DATA + 0x1800), e.a(DATA + 0x1A00));
    let (want_cx, want_uc, want_map) = (e.a(DATA + 0x1B00), e.a(DATA + 0x1B40), e.a(DATA + 0x1C00));
    let gs = semantic_shader(gs_at, 2, &[], &[semantic(9, 3, 0)]);
    let ps = semantic_shader(ps_at, 1, &[semantic(9, 0, 0) | SEM_FLAT, semantic(4, 0, 0)], &[]);
    assert_eq!(e.call("sceAgcCreatePrimState", &[want_cx, want_uc, 0, gs, 7], &[]), 0);
    assert_eq!(e.call("sceAgcCreateInterpolantMapping_0100", &[want_map, gs, ps], &[]), 0);
    let mut expected = mem::read_dwords(want_cx, 4).unwrap();
    expected.extend(mem::read_dwords(want_map, 64).unwrap());
    assert_eq!(expected[..4], [0x2D5, 0, 0x29B, 3]);
    assert!(mem::write_dwords(cx, &[0xDEAD_BEEF; 68]) && mem::write_dwords(uc, &[0xDEAD_BEEF; 6]));
    let link = |cx: u64, uc: u64, reserved: u64, gs: u64, ps: u64| e.call("sceAgcLinkShaders", &[cx, uc, reserved, gs, ps, 7], &[]);
    // Rejected: nothing written.
    for (reserved, g, p) in [(1, gs, ps), (0, ps, ps), (0, gs, gs), (0, 0, ps)] {
        assert_eq!(link(cx, uc, reserved, g, p), 0x8A6C_0005u32 as i32 as i64);
    }
    assert_eq!(link(0, uc, 0, gs, ps), 0x8A6C_0005u32 as i32 as i64);
    assert_eq!(mem::read_dwords(cx, 68).unwrap(), vec![0xDEAD_BEEF; 68]);
    assert_eq!(mem::read_dwords(uc, 6).unwrap(), vec![0xDEAD_BEEF; 6]);
    assert_eq!(link(cx, uc, 0, gs, ps), 0);
    assert_eq!(mem::read_dwords(cx, 68).unwrap(), expected);
    assert_eq!(mem::read_dwords(uc, 6).unwrap(), mem::read_dwords(want_uc, 6).unwrap());
    assert_eq!(mem::read_dwords(uc, 6).unwrap(), vec![0x25B, 0x123, 0x262, 7, 0x242, 7]);
    // Without a pixel shader the interpolants are the identity.
    assert_eq!(link(cx, uc, 0, gs, 0), 0);
    let identity: Vec<u32> = (0..32).flat_map(|i| [0x191 + i, i]).collect();
    assert_eq!(mem::read_dwords(cx + 16, 64).unwrap(), identity);
}

/// The tessellation factor ring reads back the console default until a
/// ring is set; an empty ring and null outputs are refused (reference
/// TessellationFactorRing.cpp).
#[test]
fn driver_tessellation_factor_ring() {
    let e = Env::new();
    let (base, size) = (e.a(LABELS + 0x700), e.a(LABELS + 0x708));
    let ring = |e: &Env| {
        assert!(mem::w64(base, u64::MAX) && mem::w32(size, 0) && mem::w32(size + 4, 0xA5A5_A5A5));
        assert_eq!(e.call("sceAgcDriverGetTFRing", &[base, size], &[]), 0);
        assert_eq!(mem::r32(size + 4), Some(0xA5A5_A5A5));
        (mem::r64(base).unwrap(), mem::r32(size).unwrap())
    };
    assert_eq!(ring(&e), (0xF_F000_0000, 0x2_0000));
    let r = e.a(DATA + 0x2000);
    assert_eq!(e.call("sceAgcDriverSetTFRing", &[r, 0x1B000], &[]), 0);
    assert_eq!(ring(&e), (r, 0x1B000));
    assert_ne!(e.call("sceAgcDriverSetTFRing", &[0, 0x2000], &[]), 0);
    assert_ne!(e.call("sceAgcDriverSetTFRing", &[r, 0], &[]), 0);
    assert_eq!(ring(&e), (r, 0x1B000));
    assert_ne!(e.call("sceAgcDriverGetTFRing", &[0, size], &[]), 0);
    assert_ne!(e.call("sceAgcDriverGetTFRing", &[base, 0], &[]), 0);
}

/// sceAgcDriverWaitUntilSafeForRendering writes the 4-dword rendering
/// wait at the cursor and advances it (reference Submit/Dcb.cpp); the
/// command processor holds the packet until the buffer's flip is done.
#[test]
fn driver_rendering_wait_packet() {
    let _g = HOOK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let e = Env::new();
    assert_eq!(e.call("sceAgcDriverGetWaitRenderingPacketSizeInDwords", &[], &[]), 4);
    // The flip packet size is what sceAgcDcbSetFlip writes.
    let flip_at = e.cursor();
    assert_ne!(e.call("sceAgcDcbSetFlip", &[e.cb(), 1, 2, 0, 0], &[]), 0);
    assert_eq!(e.call("sceAgcDriverGetSetFlipPacketSizeInDwords", &[], &[]) as u64 * 4, e.cursor() - flip_at);
    let (cursor, at) = (e.a(LABELS + 0x710), e.a(CMD + 0x3000));
    assert!(mem::w64(cursor, at) && mem::write_dwords(at, &[0xCCCC_CCCC; 4]));
    for args in [[0, 4, 0, 1, 2], [cursor, 3, 0, 1, 2], [cursor, 4, 1, 1, 2], [cursor, 4, 0, 1, u32::MAX as u64]] {
        assert_ne!(e.call("sceAgcDriverWaitUntilSafeForRendering", &args, &[]), 0, "{args:x?}");
    }
    assert_eq!((mem::r64(cursor), mem::r32(at)), (Some(at), Some(0xCCCC_CCCC)), "a refused call wrote");
    assert_eq!(e.call("sceAgcDriverWaitUntilSafeForRendering", &[cursor, 8, 0, 1, 2], &[]), 0);
    assert_eq!(mem::read_dwords(at, 4).unwrap(), vec![0xC002_1018, 1, 2, 0]);
    assert_eq!(mem::r64(cursor), Some(at + 16));
    set_flip_hooks(test_flip, test_flip_done);
    FLIP_DONE.store(0, Ordering::SeqCst);
    let l = e.a(LABELS + 0x718);
    assert!(mem::w32(l, 0) && mem::write_dwords(at + 16, &write_data(l, 1)));
    let q = bare_queue();
    q.submit_graphics(at, 9);
    run_queue(&q);
    assert_eq!((q.pending(), mem::r32(l)), (1, Some(0)), "the wait holds while the buffer is in use");
    FLIP_DONE.store(1, Ordering::SeqCst);
    q.retry_blocked();
    run_queue(&q);
    assert_eq!((q.pending(), mem::r32(l)), (0, Some(1)));
    set_flip_hooks(crate::videoout::gpu_flip, crate::videoout::is_flip_done);
}

/// sceAgcDcbSetIndexIndirectArgs writes the console's
/// INDEX_ATTRIBUTES_INDIRECT; a misaligned table or a 17-bit index is
/// refused without writing (reference libSceAgc tests/Commands.cpp).
#[test]
fn index_indirect_args_packet() {
    let e = Env::new();
    let start = e.cursor();
    let p = e.call("sceAgcDcbSetIndexIndirectArgs", &[e.cb(), 0x1111_1111_2222_2220, 0x4444], &[]) as u64;
    assert_eq!(p, start);
    assert_eq!(mem::read_dwords(p, 4).unwrap(), vec![0xC002_9100, 0x2222_2220, 0x1111_1111, 0x4444]);
    assert_eq!(e.call("sceAgcDcbSetIndexIndirectArgsGetSize", &[], &[]), 16);
    let cleared = e.call("sceAgcDcbSetIndexIndirectArgs", &[e.cb(), 0, 0], &[]) as u64;
    assert_eq!(mem::read_dwords(cleared, 4).unwrap(), vec![0xC002_9100, 0, 0, 0]);
    let end = e.cursor();
    assert_eq!(e.call("sceAgcDcbSetIndexIndirectArgs", &[e.cb(), 0x1111_1111_2222_2228, 0], &[]), 0);
    assert_eq!(e.call("sceAgcDcbSetIndexIndirectArgs", &[e.cb(), 0x1111_1111_2222_2220, 0x1_0000], &[]), 0);
    assert_eq!(e.cursor(), end, "a refused call wrote");
}

/// The register range writers read their values byte for byte from any
/// address, as the console does (reference libSceAgc
/// tests/RegisterCommands.cpp testRangeFromAnyByte).
#[test]
fn register_ranges_read_values_at_any_byte_address() {
    let e = Env::new();
    let bytes = e.a(DATA + 0x800);
    let pattern: Vec<u8> = (0..32u8).map(|i| 0x10 + i).collect();
    assert!(mem::write(bytes, &pattern));
    let cases: [(u64, [u32; 4]); 4] = [
        (1, [0x1413_1211, 0x1817_1615, 0x1c1b_1a19, 0x201f_1e1d]),
        (2, [0x1514_1312, 0x1918_1716, 0x1d1c_1b1a, 0x2120_1f1e]),
        (3, [0x1615_1413, 0x1a19_1817, 0x1e1d_1c1b, 0x2221_201f]),
        (13, [0x201f_1e1d, 0x2423_2221, 0x2827_2625, 0x2c2b_2a29]),
    ];
    for (name, header) in [
        ("sceAgcCbSetShRegisterRangeDirect", 0xC004_7600u32),
        ("sceAgcCbSetUcRegisterRangeDirect", 0xC004_7900),
        ("sceAgcDcbSetCfRegisterRangeDirect", 0xC004_6800),
    ] {
        for (offset, values) in cases {
            e.reset_cb();
            let start = e.cursor();
            assert!(mem::write_dwords(start, &[0xABCD_EF01; 7]));
            let p = e.call(name, &[e.cb(), 0x40, bytes + offset, 4], &[]) as u64;
            assert_eq!((p, e.cursor()), (start, start + 24), "{name} +{offset}");
            let w = mem::read_dwords(p, 7).unwrap();
            assert_eq!(&w[..2], &[header, 0x40], "{name}");
            assert_eq!(&w[2..6], &values, "{name} +{offset}");
            assert_eq!(w[6], 0xABCD_EF01, "{name} overwrote the next word");
        }
    }
}

/// sceAgcAcbSetFlip writes the same 6-dword flip packet as the draw
/// command buffer's, and a compute queue executes it (reference libSceAgc
/// tests/Commands.cpp testFlip, libSceAgcDriver tests/Flip.cpp
/// testComputeFlip).
#[test]
fn compute_queue_flip() {
    let _g = HOOK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    set_flip_hooks(test_flip, test_flip_done);
    FLIP_CALLS.lock().unwrap().clear();
    let e = Env::new();
    let arg = (-0x0123_4567_89ab_cdefi64) as u64;
    let p = e.call("sceAgcAcbSetFlip", &[e.cb(), 0xfedc_ba98, (-2i64) as u64, 0x1234_5678, arg], &[]) as u64;
    assert_eq!(e.cursor(), p + 24);
    let words = mem::read_dwords(p, 6).unwrap();
    assert_eq!(words, vec![0xC004_105C, 0xfedc_ba98, 0xffff_fffe, 0x1234_5678, arg as u32, (arg >> 32) as u32]);
    let q = bare_queue();
    let at = e.a(CMD + 0x3400);
    let flip = [0xC004_105C, 7, 0xffff_fffe, 1, 0x7654_3211, 0xfedc_ba98];
    assert!(mem::write_dwords(at, &flip));
    assert!(q.submit_compute(0x20, at, flip.len() as u32));
    run_queue(&q);
    assert_eq!(FLIP_CALLS.lock().unwrap().as_slice(), &[(7, -2, 1, -0x0123_4567_89ab_cdef)]);
    set_flip_hooks(crate::videoout::gpu_flip, crate::videoout::is_flip_done);
}

/// The retail library's answers to the debug, capture, GPU printf and GPU
/// reset calls and the workload packet sizes (reference
/// libSceAgcDriver Unimplemented.cpp, Submit/Dcb.cpp).
#[test]
fn driver_retail_results_and_packet_sizes() {
    let e = Env::new();
    let debug = 0x8A6C_1000u32 as i32 as i64;
    let feature = 0x8A6D_0001u32 as i32 as i64;
    for name in ["sceAgcDriverSetValidationDisabled", "sceAgcDriverGetCaptureConfig", "sceAgcDriverSetCaptureConfig"] {
        assert_eq!(e.call(name, &[], &[]), debug, "{name}");
    }
    for name in [
        "sceAgcDriverGetGpuPrintfWorkArea",
        "sceAgcDriverSetGpuPrintfEnabled",
        "sceAgcDriverDbgWriteRegister",
        "sceAgcDriverGetGpuRefClks",
        "sceAgcDriverIsSuspendPointInFlight",
    ] {
        assert_eq!(e.call(name, &[], &[]), feature, "{name}");
    }
    for name in [
        "sceAgcDriverDbgReadRegister",
        "sceAgcDriverIsPaDebug",
        "sceAgcDriverGetExceptionStatus",
        "sceAgcDriverGetTraceInitiator",
        "sceAgcDriverRegisterGpuResetCallbacks",
        "sceAgcDriverUnregisterGpuResetCallbacks",
        "sceAgcDriverSysSubmitFlipHandleProxy",
        "sceAgcDriverUserDataImmediateWrite",
        "sceAgcDriverUserDataImmediateWritePacket",
        "sceAgcDriverSetResourceRegistrationInfo",
    ] {
        assert_eq!(e.call(name, &[], &[]), 0, "{name}");
    }
    assert_eq!(e.call("sceAgdResetGpu", &[], &[]), 0x8A6C_0045u32 as i32 as i64);
    assert_eq!(e.call("sceAgcDriverRegisterOwnerForSystem", &[], &[]), 0x8A6C_9018u32 as i32 as i64);
    assert_eq!(e.call("sceAgcDriverInitResourceRegistrationInterface", &[], &[]), 0x8A6C_9018u32 as i32 as i64);
    let out = e.a(LABELS + 0x760);
    assert!(mem::w32(out, 7));
    assert_eq!(e.call("sceAgcDriverGetCoredumpMode", &[0], &[]), 0x8A6D_0003u32 as i32 as i64);
    assert_eq!(e.call("sceAgcDriverGetCoredumpMode", &[out], &[]), 0);
    assert_eq!(mem::r32(out), Some(0));
    let range = e.a(LABELS + 0x770);
    assert!(mem::fill(range, 16, 0xAB));
    assert_eq!(e.call("sceAgcDriverAllocateToolMemoryForGpuReset", &[1, 2, 3, range], &[]), feature);
    assert_eq!(mem::read_dwords(range, 4).unwrap(), vec![0, 0, 0, 0xABAB_ABAB]);
    assert!(mem::fill(range, 12, 0xAB));
    assert_eq!(e.call("sceAgcDriverGetAllocatedToolMemoryForGpuReset", &[1, range], &[]), feature);
    assert_eq!(mem::read_dwords(range, 3).unwrap(), vec![0, 0, 0]);
    assert_eq!(e.call("sceAgcDriverAllocateToolMemoryForGpuReset", &[1, 2, 3, 0], &[]), feature);
    assert_eq!(e.call("sceAgcDriverGetSetWorkloadsActivePacketSize", &[], &[]), 18);
    assert_eq!(e.call("sceAgcDriverGetSetWorkloadCompletePacketSize", &[], &[]), 12);
    assert_eq!(e.call("sceAgcDriverGetSetWorkloadStreamInactivePacketSize", &[], &[]), 9);
    for (bytes, nop, user) in [(0u64, 2, 3), (1, 3, 4), (4, 3, 4), (5, 4, 9), (8, 4, 9), (13, 6, 11)] {
        assert_eq!(e.call("sceAgcDriverNopDataGetPacketSize", &[bytes], &[]), nop, "nop {bytes}");
        assert_eq!(e.call("sceAgcDriverUserDataGetPacketSize", &[bytes], &[]), user, "user {bytes}");
    }
}

/// Resource lookups and multi-registrations report registration
/// unavailable (reference Resource/Registration.cpp).
#[test]
fn driver_resource_lookup_unavailable() {
    let e = Env::new();
    let out = e.a(LABELS + 0x720);
    assert!(mem::w32(out, 0x1234));
    assert_eq!(e.call("sceAgcDriverFindResourcesPublic", &[0, 0], &[]), 0x8A6C_9018u32 as i32 as i64);
    assert_eq!(e.call("sceAgcDriverRegisterMultipleResources", &[out, 1, e.a(DATA), 2], &[]), 0x8A6C_9018u32 as i32 as i64);
    assert_eq!(mem::r32(out), Some(0x1234));
}

const EOP_TEST_TOKEN: u32 = 0x0400_0123;
static EOP_CLAIMED: AtomicU64 = AtomicU64::new(0);
static EOP_FLIPS: Mutex<Vec<u64>> = Mutex::new(Vec::new());

fn test_claim_eop_flip(token: u32) -> Option<u64> {
    (token == EOP_TEST_TOKEN && EOP_CLAIMED.swap(1, Ordering::SeqCst) == 0).then_some(77)
}

fn test_eop_flip(id: u64) -> Result<(), i64> {
    EOP_FLIPS.lock().unwrap().push(id);
    Ok(())
}

/// An 8-dword graphics RELEASE_MEM with interrupt selector 2 whose context
/// id is a reserved EOP flip token writes its label, then performs the
/// flip once; other releases never claim one (reference Submission.cpp
/// EopRelease / reserveOutputs).
#[test]
fn eop_release_performs_the_reserved_flip() {
    let _g = HOOK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    crate::agc::set_eop_flip_hooks(Some((test_claim_eop_flip, test_eop_flip)));
    EOP_CLAIMED.store(0, Ordering::SeqCst);
    EOP_FLIPS.lock().unwrap().clear();
    let e = Env::new();
    let l = e.a(LABELS + 0x740);
    assert!(mem::w32(l, 0));
    let release = |int_sel: u32, token: u32| -> [u32; 8] {
        // [event, data_sel | int_sel | dst_sel, address, data, context id]
        [p4::make(8, op::RELEASE_MEM, 0), 0x28, 1 << 29 | int_sel << 24, l as u32, (l >> 32) as u32, 9, 0, token]
    };
    let q = bare_queue();
    // Other selectors and unknown tokens are plain releases.
    run_words(&q, e.a(CMD + 0x3100), &[release(1, EOP_TEST_TOKEN), release(2, EOP_TEST_TOKEN + 1)].concat());
    assert_eq!(EOP_CLAIMED.load(Ordering::SeqCst), 0);
    assert!(EOP_FLIPS.lock().unwrap().is_empty());
    assert!(mem::w32(l, 0));
    run_words(&q, e.a(CMD + 0x3200), &release(2, EOP_TEST_TOKEN));
    assert_eq!(mem::r32(l), Some(9));
    assert_eq!(EOP_FLIPS.lock().unwrap().as_slice(), &[77]);
    // The token is single use.
    run_words(&q, e.a(CMD + 0x3300), &release(2, EOP_TEST_TOKEN));
    assert_eq!(EOP_FLIPS.lock().unwrap().as_slice(), &[77]);
    // The private release packet sceAgcCbReleaseMem writes claims the flip
    // too.
    let mut private = release(2, EOP_TEST_TOKEN);
    private[0] = p4::make(8, op::NOP, crate::agc::pm4::nop::RELEASE_MEM);
    EOP_CLAIMED.store(0, Ordering::SeqCst);
    run_words(&q, e.a(CMD + 0x3400), &private);
    assert_eq!(EOP_FLIPS.lock().unwrap().as_slice(), &[77, 77]);
    crate::agc::set_eop_flip_hooks(None);
}
