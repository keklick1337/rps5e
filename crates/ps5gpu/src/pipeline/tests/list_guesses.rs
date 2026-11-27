// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! The prewarm list and the compile lookahead's guesses: a module and a
//! compute pipeline the lookahead made for an upcoming dispatch enter the
//! list when a dispatch first asks for them, not when they are finished;
//! what a dispatch starts itself is listed at once, as is everything with
//! `RPS5E_LIST_GUESSES=1`.
//!
//! Run with `cargo test -p ps5gpu --release -- --ignored list_guesses`.

use super::*;
use crate::pipeline::provider::GuessCounts;

const S_WAITCNT_0: u32 = 0xBF8C_0000;

/// A store program with `pad` no-ops before its store (each `pad` is a
/// program of its own, with a module of its own).
fn store_program(g: &mut Guest, pad: usize) -> u64 {
    let store = asm::buffer_store_dword(1, 0, 0);
    let mut words = vec![asm::v_mov_from_sgpr(1, 4)];
    words.extend(vec![S_WAITCNT_0; pad]);
    words.extend([store[0], store[1], asm::S_ENDPGM]);
    g.program(&words)
}

/// One group of 64 threads of `cs` storing through a V# over `base`.
fn store_dispatch(cs: u64, base: u64) -> ResolvedDispatch {
    let mut user = vec![base as u32, (base >> 32) as u32, 64 * 4, (3 << 28) | (20 << 12) | 0xFAC];
    user.push(7);
    let mut shader = stage(cs, &user);
    shader.rsrc2 = (user.len() as u32) << 1;
    ResolvedDispatch { packet: DispatchPacket { groups: [1, 1, 1], dispatch_initiator: 1 | (1 << 15), ..DispatchPacket::default() }, shader, rsrc3: 0, threads: [64, 1, 1], start: [0; 3] }
}

/// Three store programs, their compiler and a dispatch of each.
fn three_programs(g: &mut Guest) -> (TestCompiler, [ResolvedDispatch; 3]) {
    let base = g.zeroed(8192);
    let cs = [store_program(g, 0), store_program(g, 1), store_program(g, 2)];
    let compiler = cs.iter().fold(TestCompiler::new(), |c, &a| c.with(a, TestProgram::StoreCompute { local_size: 64 }));
    (compiler, cs.map(|a| store_dispatch(a, base)))
}

fn fresh_cache(name: &str) -> std::path::PathBuf {
    let path = temp_cache(name);
    remove_cache(&path);
    path
}

fn remove_cache(path: &std::path::Path) {
    let list = prewarm::list_path(path);
    for (_, journal) in prewarm::journal_files(&list) {
        let _ = std::fs::remove_file(journal);
    }
    let _ = std::fs::remove_file(list);
    let _ = std::fs::remove_file(path);
}

/// The module keys and the compute pipeline count of the list saved next
/// to `path` (its journals merged in).
fn saved(path: &std::path::Path) -> (Vec<u64>, usize) {
    let list = prewarm::PrewarmList::load(Some(path), 0);
    (list.modules.iter().map(|m| m.key).collect(), list.compute.len())
}

#[test]
#[ignore]
fn list_guesses_enter_the_list_at_their_first_use() {
    let mut g = Guest::new(16 << 20);
    let (compiler, [a, b, c]) = three_programs(&mut g);
    let compiler = Arc::new(compiler);
    let path = fresh_cache("list-guesses-use");
    let mut rig = Rig::new(&g, compiler.clone(), None, Some(path.clone()));
    rig.provider.set_async_compile(true);
    rig.provider.set_list_guesses(false);

    // What a dispatch starts is listed when it is made.
    assert_eq!(rig.dispatch(&a), DrawOutcome::Recorded);
    let (first, pipelines) = rig.provider.listed();
    assert_eq!((first.len(), pipelines), (1, 1), "a dispatch's own module and pipeline are listed");

    // The lookahead makes the modules and pipelines of two dispatches it
    // found ahead: there for the session, not in the list.
    rig.provider.guess_ahead(rig.buffers.tracker.clone(), vec![b.clone(), c.clone()]);
    assert_eq!(compiler.compiled(), 3, "the guesses were compiled");
    let s = rig.provider.stats();
    assert_eq!((s.modules, s.compute_pipelines), (3, 3), "the guesses' modules and pipelines are made");
    assert_eq!(rig.provider.guess_counts(), GuessCounts { modules: 2, modules_used: 0, pipelines: 2, pipelines_used: 0 });
    assert_eq!(rig.provider.listed(), (first.clone(), 1), "a guess nothing asked for is not listed");

    // A dispatch asks for one of them: listed now, without another compile.
    assert_eq!(rig.dispatch(&b), DrawOutcome::Recorded);
    assert_eq!(compiler.compiled(), 3, "the dispatch took the guess's module");
    let (second, pipelines) = rig.provider.listed();
    assert_eq!((second.len(), pipelines), (2, 2), "a guess is listed at its first use");
    assert_eq!(second[0], first[0]);
    assert_eq!(rig.provider.guess_counts(), GuessCounts { modules: 2, modules_used: 1, pipelines: 2, pipelines_used: 1 });

    // Once: later uses add nothing.
    assert_eq!(rig.dispatch(&b), DrawOutcome::Recorded);
    assert_eq!(rig.dispatch(&a), DrawOutcome::Recorded);
    assert_eq!(rig.provider.listed(), (second.clone(), 2));
    assert_eq!(rig.provider.guess_counts().modules_used, 1);
    assert!(rig.provider.summary().unwrap().contains("guesses=[modules 2 made 1 used 1 unlisted; pipelines 2 made 1 used 1 unlisted]"));

    // The file (list and journal) holds what the list held.
    drop(rig);
    let (mut on_disk, pipelines) = saved(&path);
    let mut expected = second.clone();
    on_disk.sort_unstable();
    expected.sort_unstable();
    assert_eq!((on_disk, pipelines), (expected, 2), "the unused guess is not in the saved list");

    // The next start builds the two listed modules; the third program is
    // compiled when its dispatch comes, and listed then.
    let mut rig = Rig::new(&g, compiler.clone(), None, Some(path.clone()));
    rig.provider.set_async_compile(true);
    rig.provider.set_list_guesses(false);
    assert_eq!(rig.dispatch(&c), DrawOutcome::Recorded);
    let (third, pipelines) = rig.provider.listed();
    assert_eq!((third.len(), pipelines), (3, 3), "a real compile is listed");
    assert_eq!(rig.provider.guess_counts(), GuessCounts::default());
    drop(rig);
    remove_cache(&path);
}

#[test]
#[ignore]
fn list_guesses_a_dispatch_waits_for_are_listed_when_made() {
    use super::testing::CompileGate;
    let mut g = Guest::new(16 << 20);
    let (compiler, [a, b, _]) = three_programs(&mut g);
    let path = fresh_cache("list-guesses-wait");
    // The first dispatch of the process decides how stages are prepared
    // ahead: one that nothing holds back.
    {
        let ungated = Arc::new(TestCompiler::new().with(a.shader.address, TestProgram::StoreCompute { local_size: 64 }));
        let mut rig = Rig::new(&g, ungated, None, None);
        assert_eq!(rig.dispatch(&a), DrawOutcome::Recorded);
    }
    let gate = Arc::new(CompileGate::default());
    let compiler = Arc::new(compiler.gated(gate.clone()));
    let mut rig = Rig::new(&g, compiler, None, Some(path.clone()));
    rig.provider.set_async_compile(true);
    rig.provider.set_list_guesses(false);
    // The lookahead queues the module; its compile is held back.
    rig.provider.queue_guesses(rig.buffers.tracker.clone(), vec![b.clone()]);
    assert_eq!(rig.provider.listed(), (vec![], 0));
    // The dispatch itself comes while the module compiles: it waits for
    // it, and the module is no guess any more when it is finished.
    let outcome = std::thread::scope(|s| {
        s.spawn(|| {
            std::thread::sleep(std::time::Duration::from_millis(300));
            gate.open();
        });
        rig.dispatch(&b)
    });
    assert_eq!(outcome, DrawOutcome::Recorded);
    let (modules, pipelines) = rig.provider.listed();
    assert_eq!((modules.len(), pipelines), (1, 1), "the module a dispatch waited for is listed once");
    let c = rig.provider.guess_counts();
    assert_eq!((c.modules, c.pipelines), (c.modules_used, c.pipelines_used), "nothing made stays unlisted");
    drop(rig);
    let (on_disk, pipelines) = saved(&path);
    assert_eq!((on_disk, pipelines), (modules, 1));
    remove_cache(&path);
}

#[test]
#[ignore]
fn list_guesses_switch_lists_every_guess() {
    let mut g = Guest::new(16 << 20);
    let (compiler, [a, b, c]) = three_programs(&mut g);
    let path = fresh_cache("list-guesses-switch");
    let mut rig = Rig::new(&g, Arc::new(compiler), None, Some(path.clone()));
    rig.provider.set_async_compile(true);
    rig.provider.set_list_guesses(true);
    assert_eq!(rig.dispatch(&a), DrawOutcome::Recorded);
    rig.provider.guess_ahead(rig.buffers.tracker.clone(), vec![b.clone(), c]);
    let (modules, pipelines) = rig.provider.listed();
    assert_eq!((modules.len(), pipelines), (3, 3), "RPS5E_LIST_GUESSES=1 lists a guess when it is made");
    assert_eq!(rig.provider.guess_counts(), GuessCounts { modules: 2, modules_used: 0, pipelines: 2, pipelines_used: 0 });
    // A use is still counted, and adds nothing.
    assert_eq!(rig.dispatch(&b), DrawOutcome::Recorded);
    assert_eq!(rig.provider.listed(), (modules, 3));
    assert_eq!(rig.provider.guess_counts(), GuessCounts { modules: 2, modules_used: 1, pipelines: 2, pipelines_used: 1 });
    assert!(rig.provider.summary().unwrap().contains("unlisted; all listed]"));
    drop(rig);
    assert_eq!(saved(&path).0.len(), 3);
    remove_cache(&path);
}
