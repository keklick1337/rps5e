// RustPS5Emu (RPS5E)
// Copyright (C) 2025-2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Process runtime: the loader state that stays alive after start-up so
//! that modules can be loaded and unloaded while the game runs
//! (`sceKernelLoadStartModule` / `sceKernelStopUnloadModule`).

use crate::{dispatcher::CoreDispatcher, linker, relocator, stubs::StubTable, LoadedGame, LoadedModule};
use ps5hle::HleRegistry;
use ps5vm::VirtualMemory;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};

const ENOENT: i64 = 0x8002_0002u32 as i32 as i64;
const ENOEXEC: i64 = 0x8002_0008u32 as i32 as i64;
const ENOMEM: i64 = 0x8002_000Cu32 as i32 as i64;
const EINVAL: i64 = 0x8002_0016u32 as i32 as i64;

struct LoaderState {
    stubs: StubTable,
    programs: Vec<LoadedModule>,
    lle_imports: HashMap<String, u64>,
    hle_data: crate::HleData,
    next_base: u64,
}

pub struct Runtime {
    pub vm: Arc<VirtualMemory>,
    pub registry: Arc<HleRegistry>,
    stub_map: Arc<RwLock<HashMap<u64, String>>>,
    exit_handler: u64,
    state: Mutex<LoaderState>,
}

impl Runtime {
    /// Take ownership of a loaded game.
    pub fn new(game: LoadedGame) -> Arc<Self> {
        let mut stubs = game.import_stubs;
        // The process exit handler handed to the entry point is an import
        // stub of the module registry's exit callback.
        let exit_handler = stubs.ensure_stub(&game.vm, ps5hle::modules::EXIT_HANDLER_NID).unwrap_or(ps5cpu::exec::EXIT_HANDLER_STUB);
        let stub_map = Arc::new(RwLock::new(stubs.addr_to_nid_map()));
        // Modules loaded at run time go after the last module loaded at start-up.
        let end = game.programs.iter().map(|p| p.base + p.size).max().unwrap_or(game.image_base + game.image_size);
        let next_base = crate::next_module_base(end);
        Arc::new(Self {
            vm: Arc::new(game.vm),
            registry: Arc::new(game.hle),
            stub_map,
            exit_handler,
            state: Mutex::new(LoaderState {
                stubs,
                programs: game.programs,
                lle_imports: game.lle_imports,
                hle_data: game.hle_data,
                next_base,
            }),
        })
    }

    pub fn dispatcher(&self) -> CoreDispatcher {
        CoreDispatcher::new(self.stub_map.clone(), self.registry.clone(), self.vm.clone())
    }

    /// Install this runtime as the loader behind the kernel module calls.
    pub fn install_module_loader(self: &Arc<Self>) {
        ps5hle::modules::set_module_loader(Box::new(ModuleLoaderImpl(self.clone())));
    }

    /// Exit handler passed to the main entry point (stops every module).
    pub fn exit_handler(&self) -> u64 {
        self.exit_handler
    }

    fn state(&self) -> std::sync::MutexGuard<'_, LoaderState> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// (name, base, DT_INIT offset) of every loaded PRX module, in load order.
    pub fn module_inits(&self) -> Vec<(String, u64, u64)> {
        self.state().programs.iter().filter(|m| m.is_shared).map(|m| (m.name.clone(), m.base, m.init_func)).collect()
    }

    fn publish_stubs(&self, stubs: &StubTable) {
        let mut map = self.stub_map.write().unwrap();
        for (nid, &addr) in stubs.nid_to_addr_iter() {
            map.entry(addr).or_insert_with(|| nid.clone());
        }
    }

    /// Load and relocate a PRX given its guest path; returns the module
    /// handle. The caller starts it on the requesting guest thread.
    pub fn load_module(&self, guest_path: &str) -> Result<i32, i64> {
        let host = ps5hle::filesystem::resolve_guest_path(guest_path).filter(|h| h.is_file()).ok_or(ENOENT)?;
        let mut guard = self.state();
        let st = &mut *guard;
        let base = st.next_base;
        let (module, _) = crate::map_program(&self.vm, &host, guest_path, base).map_err(|e| {
            log::warn!("[runtime] load {guest_path}: {e}");
            ENOEXEC
        })?;
        let next = crate::next_module_base(module.base + module.size);
        if next > crate::MODULE_LIMIT {
            let _ = self.vm.unmap(module.base, module.size);
            return Err(ENOMEM);
        }
        st.next_base = next;
        st.programs.push(module);
        // Resolve what the new module provides to everybody, and its own imports.
        relocator::relocate_all(&mut st.programs, &self.vm, &mut st.stubs, &st.hle_data);
        let idx = st.programs.len() - 1;
        crate::register_program(&self.vm, &mut st.programs[idx], false);
        let linked = linker::link_native_imports(&self.vm, &st.stubs, &st.programs, &self.registry);
        st.lle_imports.extend(linked);
        if let Err(e) = crate::game_patch::apply_pending(&self.vm, &st.programs[idx]) {
            log::error!("Game cheat error: {e}");
            log::error!("Failed to apply pending game cheat");
            // The caller stops the run; a process exit here would skip the
            // shutdown of the other guest threads.
            return Err(EINVAL);
        }
        crate::finish_program(&self.vm, &st.programs[idx]);
        self.publish_stubs(&st.stubs);
        let m = &st.programs[idx];
        log::info!("[runtime] loaded {} at {:#x} size={:#x} handle={}", m.name, m.base, m.size, m.handle);
        Ok(m.handle)
    }

    /// Unmap a stopped module: references of other programs into it are
    /// rebound, imports linked to its code return to HLE.
    pub fn unload_module(&self, handle: i32) {
        let mut guard = self.state();
        let st = &mut *guard;
        let Some(idx) = st.programs.iter().position(|p| p.handle == handle && p.is_shared) else { return };
        let (lo, hi) = (st.programs[idx].base, st.programs[idx].base + st.programs[idx].size);
        relocator::unbind(idx, &mut st.programs, &self.vm, &mut st.stubs, &st.hle_data);
        let gone: Vec<String> = st.lle_imports.iter().filter(|(_, &t)| t >= lo && t < hi).map(|(n, _)| n.clone()).collect();
        for nid in gone {
            st.lle_imports.remove(&nid);
            if let Err(e) = st.stubs.unredirect(&self.vm, &nid) {
                log::warn!("[runtime] cannot restore import stub {nid}: {e}");
            }
        }
        let p = st.programs.remove(idx);
        // Another module may provide what the departed one did.
        let linked = linker::link_native_imports(&self.vm, &st.stubs, &st.programs, &self.registry);
        st.lle_imports.extend(linked);
        if p.tls_id != 0 {
            ps5hle::tls::unregister_module(p.tls_id);
        }
        ps5cpu::exec::remove_guest_code_ranges(p.base, p.size);
        ps5hle::modules::unregister_module(p.handle);
        if let Err(e) = self.vm.unmap(p.base, p.size) {
            log::warn!("[runtime] unmap {}: {e}", p.name);
        }
        log::info!("[runtime] unloaded {} (handle {})", p.name, p.handle);
    }
}

/// Log the HLE import rate every 10 seconds.
pub fn start_stats_reporter() {
    let _ = std::thread::Builder::new().name("hle-stats".into()).spawn(|| {
        let mut last = (std::time::Instant::now(), ps5cpu::exec::total_imports());
        loop {
            std::thread::sleep(std::time::Duration::from_secs(10));
            let now = (std::time::Instant::now(), ps5cpu::exec::total_imports());
            let rate = (now.1 - last.1) as f64 / now.0.duration_since(last.0).as_secs_f64();
            let emulated: Vec<String> = ps5cpu::x64emu::emulated_counts().iter().map(|(n, c)| format!("{n}={c}")).collect();
            if emulated.is_empty() {
                log::info!("[hle] imports={} rate={rate:.0}/s", now.1);
            } else {
                log::info!("[hle] imports={} rate={rate:.0}/s emulated instructions: {}", now.1, emulated.join(" "));
            }
            last = now;
            for line in crate::dispatcher::stuck::report(std::time::Duration::from_secs(5)) {
                log::info!("[hle] stuck: {line}");
            }
            if crate::dispatcher::stuck::enabled() {
                for line in crate::dispatcher::stuck::busy_report() {
                    log::info!("[hle] busy: {line}");
                }
            }
        }
    });
}

struct ModuleLoaderImpl(Arc<Runtime>);

impl ps5hle::modules::ModuleLoader for ModuleLoaderImpl {
    fn load(&self, guest_path: &str) -> Result<i32, i64> {
        self.0.load_module(guest_path)
    }

    fn unload(&self, handle: i32) {
        self.0.unload_module(handle)
    }
}
