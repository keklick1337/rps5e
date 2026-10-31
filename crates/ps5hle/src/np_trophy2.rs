// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! libSceNpTrophy2 HLE: offline trophy service.
//!
//! Trophy definitions are read from the application's trophy package
//! (`/app0/sce_sys/trophy2/trophyNN.ucp`, NN = service label), UDS stat rules
//! from `/app0/sce_sys/uds/udsNN.ucp`. Unlocks persist per title, user and
//! service label under the emulator data directory (see `package.rs`).
//!
//! Trophies are unlocked through libSceNpUniversalDataSystem: the
//! `_UnlockTrophy` event (property `_trophy_id`) or stat events matched
//! against the UDS extraction rules. Completing every trophy that names a
//! platinum unlocks that platinum.

pub mod package;

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex, MutexGuard};

use crate::context::{CpuContext, CpuRegister};
use crate::registry::{HleRegistry, HleResult};
use package::{Package, Trophy, UdsInteger, UnlockData};

const OK: HleResult = 0;

const fn err(code: u32) -> HleResult {
    code as i32 as HleResult
}

pub(crate) const ERR_INVALID_ARGUMENT: HleResult = err(0x8055_3904);
pub(crate) const ERR_INVALID_CONTEXT: HleResult = err(0x8055_3909);
pub(crate) const ERR_INVALID_TROPHY_ID: HleResult = err(0x8055_390A);
pub(crate) const ERR_INVALID_GROUP_ID: HleResult = err(0x8055_390B);
pub(crate) const ERR_CONTEXT_ALREADY_EXISTS: HleResult = err(0x8055_3910);
pub(crate) const ERR_ICON_FILE_NOT_FOUND: HleResult = err(0x8055_3911);
pub(crate) const ERR_CONTEXT_EXCEEDS_MAX: HleResult = err(0x8055_391B);
pub(crate) const ERR_TITLE_CONF_NOT_INSTALLED: HleResult = err(0x8055_391E);
pub(crate) const ERR_NOT_REGISTERED: HleResult = err(0x8055_3920);
pub(crate) const ERR_ALREADY_REGISTERED: HleResult = err(0x8055_3921);

const MAX_CONTEXTS: usize = 8;

/// Group id used for whole-game counts.
const ALL_GROUPS: i32 = -2;

// Guest structure sizes.
const GAME_DETAILS_SIZE: usize = 152;
const GAME_DATA_SIZE: usize = 24;
const GROUP_DETAILS_SIZE: usize = 152;
const GROUP_DATA_SIZE: usize = 32;
const TROPHY_DETAILS_SIZE: usize = 1312;
const TROPHY_DATA_SIZE: usize = 32;

/// (user id, service label)
pub(crate) type TrophyKey = (i32, u32);

/// Where packages, unlock files, the title id and the language come from.
enum Env {
    /// The running application (`/app0`, data dir, param.json, system language).
    Live,
    #[cfg_attr(not(test), allow(dead_code))]
    Fixed { app0: PathBuf, data: PathBuf, title: String, language: i32 },
}

impl Env {
    fn app_file(&self, relative: &str) -> Option<PathBuf> {
        match self {
            Env::Live => crate::filesystem::resolve_guest_path(&format!("/app0/{relative}")),
            Env::Fixed { app0, .. } => Some(app0.join(relative)),
        }
    }

    fn data_root(&self) -> PathBuf {
        match self {
            Env::Live => crate::filesystem::user_data_dir(),
            Env::Fixed { data, .. } => data.clone(),
        }
    }

    fn title_id(&self) -> String {
        match self {
            Env::Live => crate::app_info::title_id().unwrap_or_default(),
            Env::Fixed { title, .. } => title.clone(),
        }
    }

    fn language(&self) -> i32 {
        match self {
            Env::Live => crate::app_info::system_language(),
            Env::Fixed { language, .. } => *language,
        }
    }
}

struct TrophyState {
    package: Arc<Package>,
    unlocks: UnlockData,
    path: Option<PathBuf>,
}

struct TrophyContext {
    key: TrophyKey,
    registered: bool,
}

pub(crate) struct TrophyService {
    env: Env,
    packages: BTreeMap<u32, Arc<Package>>,
    states: BTreeMap<TrophyKey, TrophyState>,
    contexts: BTreeMap<i32, TrophyContext>,
    next_context: i32,
}

/// Writes `text` into a fixed char field, truncated with a terminating NUL.
fn put_cstr(buf: &mut [u8], offset: usize, capacity: usize, text: &str) {
    let n = text.len().min(capacity - 1);
    buf[offset..offset + n].copy_from_slice(&text.as_bytes()[..n]);
    buf[offset + n] = 0;
}

fn put_u32(buf: &mut [u8], offset: usize, v: u32) {
    buf[offset..offset + 4].copy_from_slice(&v.to_le_bytes());
}

fn put_u64(buf: &mut [u8], offset: usize, v: u64) {
    buf[offset..offset + 8].copy_from_slice(&v.to_le_bytes());
}

/// Counts shared by game/group details (at +4) and data (at `data_base`).
fn fill_counts(state: &TrophyState, group: i32, details: &mut [u8], data: &mut [u8], data_base: usize) {
    let progress = package::get_progress(
        &state.package,
        &state.unlocks,
        if group == ALL_GROUPS { None } else { Some(group) },
    );
    let (total, earned) = (progress.total_grade, progress.earned_grade);
    put_u32(details, 4, progress.total);
    for grade in 1..5 {
        put_u32(details, 4 + 4 * grade, total[grade]);
    }
    put_u32(data, data_base, progress.earned);
    for grade in 1..5 {
        put_u32(data, data_base + 4 * grade, earned[grade]);
    }
    put_u32(data, data_base + 20, progress.percentage());
}

fn game_info_bytes(state: &TrophyState) -> (Vec<u8>, Vec<u8>) {
    let mut details = vec![0u8; GAME_DETAILS_SIZE];
    let mut data = vec![0u8; GAME_DATA_SIZE];
    fill_counts(state, ALL_GROUPS, &mut details, &mut data, 0);
    put_u32(&mut details, 0, state.package.groups.len() as u32);
    put_cstr(&mut details, 24, 128, &state.package.title);
    (details, data)
}

fn group_info_bytes(state: &TrophyState, group_id: i32, title: &str) -> (Vec<u8>, Vec<u8>) {
    let mut details = vec![0u8; GROUP_DETAILS_SIZE];
    let mut data = vec![0u8; GROUP_DATA_SIZE];
    fill_counts(state, group_id, &mut details, &mut data, 4);
    put_u32(&mut details, 0, group_id as u32);
    put_cstr(&mut details, 24, 128, title);
    put_u32(&mut data, 0, group_id as u32);
    (details, data)
}

fn trophy_info_bytes(state: &TrophyState, trophy: &Trophy) -> (Vec<u8>, Vec<u8>) {
    let mut details = vec![0u8; TROPHY_DETAILS_SIZE];
    put_u32(&mut details, 0, trophy.id as u32);
    put_u32(&mut details, 4, trophy.grade as u32);
    put_u32(&mut details, 8, trophy.group_id as u32);
    details[12] = trophy.hidden as u8;
    details[13] = trophy.has_reward as u8;
    let target = trophy.target.filter(|_| trophy.progressive);
    if let Some(target) = target {
        put_u32(&mut details, 16, 1);
        put_u64(&mut details, 24, target);
    }
    put_cstr(&mut details, 32, 128, &trophy.name);
    put_cstr(&mut details, 160, 1024, &trophy.description);
    put_cstr(&mut details, 1184, 128, &trophy.reward);

    let mut data = vec![0u8; TROPHY_DATA_SIZE];
    let unlocked = state.unlocks.unlocked.contains(&trophy.id);
    put_u32(&mut data, 0, trophy.id as u32);
    data[4] = unlocked as u8;
    if let Some(target) = target {
        put_u32(&mut data, 8, 1);
        put_u64(&mut data, 16, if unlocked { target } else { 0 });
    }
    if let Some(&tick) = state.unlocks.timestamps.get(&trophy.id) {
        put_u64(&mut data, 24, tick);
    }
    (details, data)
}

fn now_tick() -> u64 {
    let micros = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_micros() as u64)
        .unwrap_or(0);
    package::UNIX_EPOCH_TICK + micros
}

impl TrophyService {
    const fn new(env: Env) -> Self {
        Self { env, packages: BTreeMap::new(), states: BTreeMap::new(), contexts: BTreeMap::new(), next_context: 1 }
    }

    /// Loads (once) the package and unlocks for `key`. None when the title
    /// has no usable trophy package.
    fn load_state(&mut self, key: TrophyKey) -> Option<&mut TrophyState> {
        if self.states.contains_key(&key) {
            return self.states.get_mut(&key);
        }
        let title_id = self.env.title_id();
        let label = key.1;
        let loaded = self.packages.get(&label).filter(|p| !p.trophies.is_empty()).cloned();
        let package = match loaded {
            Some(p) => p,
            None => {
                let relative = package::package_path(label);
                let mut package = self
                    .env
                    .app_file(&relative)
                    .map(|p| package::load_package(&p, self.env.language()))
                    .unwrap_or_default();
                if package.trophies.is_empty() {
                    log::warn!("[Trophy] could not load trophy package /app0/{relative} for {title_id}");
                    return None;
                }
                log::info!(
                    "[Trophy] loaded {} trophy definitions for {title_id} (service {label})",
                    package.trophies.len()
                );
                let uds = package::uds_package_path(label);
                package.event_rules =
                    self.env.app_file(&uds).map(|p| package::load_uds_rules(&p)).unwrap_or_default();
                if package.event_rules.is_empty() {
                    log::info!("[Trophy] no supported UDS extraction rules in /app0/{uds}");
                } else {
                    log::info!(
                        "[Trophy] loaded UDS extraction rules for {} events from /app0/{uds}",
                        package.event_rules.len()
                    );
                }
                let package = Arc::new(package);
                self.packages.insert(label, package.clone());
                package
            }
        };
        let path = package::unlocks_path(&self.env.data_root(), &title_id, key.0, label);
        let unlocks = path.as_deref().map(package::load_unlock_data).unwrap_or_default();
        Some(self.states.entry(key).or_insert(TrophyState { package, unlocks, path }))
    }

    fn create_context(&mut self, user_id: i32, service_label: u32, options: u64) -> Result<i32, HleResult> {
        if options != 0 || user_id < 0 {
            return Err(ERR_INVALID_ARGUMENT);
        }
        let key = (user_id, service_label);
        if self.contexts.values().any(|c| c.key == key) {
            return Err(ERR_CONTEXT_ALREADY_EXISTS);
        }
        if self.contexts.len() == MAX_CONTEXTS {
            return Err(ERR_CONTEXT_EXCEEDS_MAX);
        }
        let id = self.next_context;
        self.next_context += 1;
        self.contexts.insert(id, TrophyContext { key, registered: false });
        Ok(id)
    }

    fn register_context(&mut self, context: i32, options: u64) -> HleResult {
        let Some(ctx) = self.contexts.get(&context) else {
            return ERR_INVALID_CONTEXT;
        };
        if options != 0 {
            return ERR_INVALID_ARGUMENT;
        }
        if ctx.registered {
            return ERR_ALREADY_REGISTERED;
        }
        let key = ctx.key;
        if self.load_state(key).is_none() {
            return ERR_TITLE_CONF_NOT_INSTALLED;
        }
        if let Some(ctx) = self.contexts.get_mut(&context) {
            ctx.registered = true;
        }
        OK
    }

    fn destroy_context(&mut self, context: i32) -> HleResult {
        if self.contexts.remove(&context).is_some() {
            OK
        } else {
            ERR_INVALID_CONTEXT
        }
    }

    fn state(&self, context: i32) -> Result<&TrophyState, HleResult> {
        let ctx = self.contexts.get(&context).ok_or(ERR_INVALID_CONTEXT)?;
        if !ctx.registered {
            return Err(ERR_NOT_REGISTERED);
        }
        self.states.get(&ctx.key).ok_or(ERR_NOT_REGISTERED)
    }

    fn game_info(&self, context: i32) -> Result<(Vec<u8>, Vec<u8>), HleResult> {
        Ok(game_info_bytes(self.state(context)?))
    }

    fn group_info(&self, context: i32, group_id: i32) -> Result<(Vec<u8>, Vec<u8>), HleResult> {
        let state = self.state(context)?;
        let title = state.package.groups.get(&group_id).ok_or(ERR_INVALID_GROUP_ID)?;
        Ok(group_info_bytes(state, group_id, title))
    }

    fn group_info_array(&self, context: i32, offset: u32, limit: u32) -> Result<Vec<(Vec<u8>, Vec<u8>)>, HleResult> {
        let state = self.state(context)?;
        Ok(state
            .package
            .groups
            .iter()
            .skip(offset as usize)
            .take(limit as usize)
            .map(|(&id, title)| group_info_bytes(state, id, title))
            .collect())
    }

    fn trophy_info(&self, context: i32, trophy_id: i32) -> Result<(Vec<u8>, Vec<u8>), HleResult> {
        let state = self.state(context)?;
        let trophy = state.package.trophies.get(&trophy_id).ok_or(ERR_INVALID_TROPHY_ID)?;
        Ok(trophy_info_bytes(state, trophy))
    }

    fn trophy_info_array(&self, context: i32, offset: u32, limit: u32) -> Result<Vec<(Vec<u8>, Vec<u8>)>, HleResult> {
        let state = self.state(context)?;
        Ok(state
            .package
            .trophies
            .values()
            .skip(offset as usize)
            .take(limit as usize)
            .map(|t| trophy_info_bytes(state, t))
            .collect())
    }

    /// Records an unlock (plus the platinum it completes). Platinum and,
    /// unless driven by a stat, progressive trophies cannot be unlocked
    /// directly. Returns the newly earned trophy ids.
    fn record_unlock(&mut self, key: TrophyKey, trophy_id: i32, from_stat: bool) -> Vec<i32> {
        let Some(state) = self.load_state(key) else {
            return Vec::new();
        };
        let Some(trophy) = state.package.trophies.get(&trophy_id) else {
            log::warn!("[Trophy] ignored unknown trophy ID {trophy_id}");
            return Vec::new();
        };
        if trophy.grade == 1 || (!from_stat && trophy.progressive) || !state.unlocks.unlocked.insert(trophy_id) {
            return Vec::new();
        }
        let timestamp = now_tick();
        state.unlocks.timestamps.entry(trophy_id).or_insert(timestamp);
        let mut earned = vec![trophy_id];
        let platinum = trophy.platinum_id;
        let package = state.package.clone();
        if platinum >= 0
            && package.trophies.contains_key(&platinum)
            && package
                .trophies
                .iter()
                .all(|(id, t)| t.platinum_id != platinum || state.unlocks.unlocked.contains(id))
            && state.unlocks.unlocked.insert(platinum)
        {
            state.unlocks.timestamps.entry(platinum).or_insert(timestamp);
            earned.push(platinum);
        }
        let saved = state.path.as_deref().is_some_and(|p| package::save_unlock_data(p, &state.unlocks));
        if saved {
            log::info!("[Trophy] recorded unlock {trophy_id}: {}", package.trophies[&trophy_id].name);
        } else {
            log::warn!("[Trophy] could not save unlocks to {:?}", state.path);
        }
        for id in &earned {
            let t = &package.trophies[id];
            let grade = ["", "platinum", "gold", "silver", "bronze"].get(t.grade as usize).copied().unwrap_or("");
            log::info!("[Trophy] trophy unlocked ({grade}): {}", t.name);
        }
        earned
    }

    fn uds_stat_event(&mut self, key: TrophyKey, event: &str, properties: &BTreeMap<String, UdsInteger>) -> Vec<i32> {
        let Some(state) = self.load_state(key) else {
            return Vec::new();
        };
        let ids = package::find_uds_trophies(&state.package, event, properties);
        ids.into_iter().flat_map(|id| self.record_unlock(key, id, true)).collect()
    }
}

static SERVICE: Mutex<TrophyService> = Mutex::new(TrophyService::new(Env::Live));

fn service() -> MutexGuard<'static, TrophyService> {
    SERVICE.lock().unwrap_or_else(|e| e.into_inner())
}

/// `_UnlockTrophy` UDS event for the given user/service label.
pub(crate) fn uds_unlock_trophy(key: TrophyKey, trophy_id: i32) {
    service().record_unlock(key, trophy_id, false);
}

/// Stat-bearing UDS event: unlocks every trophy whose rule is satisfied.
pub(crate) fn uds_stat_event(key: TrophyKey, event: &str, properties: &BTreeMap<String, UdsInteger>) {
    service().uds_stat_event(key, event, properties);
}

// ---------------------------------------------------------------------------
// HLE entry points
// ---------------------------------------------------------------------------

fn stack_arg(ctx: &CpuContext, i: u64) -> u64 {
    ctx.read_u64(ctx.get(CpuRegister::Rsp) + 8 + 8 * i).unwrap_or(0)
}

/// Writes the optional details/data outputs.
fn write_pair(ctx: &CpuContext, details: u64, data: u64, pair: &(Vec<u8>, Vec<u8>)) -> HleResult {
    if (details != 0 && !ctx.write_bytes(details, &pair.0)) || (data != 0 && !ctx.write_bytes(data, &pair.1)) {
        return ERR_INVALID_ARGUMENT;
    }
    OK
}

fn cb_create_handle(ctx: &mut CpuContext) -> HleResult {
    let out = ctx.rdi();
    if out == 0 || !ctx.write_u32(out, 1) {
        return ERR_INVALID_ARGUMENT;
    }
    OK
}

/// (int* context, userId, serviceLabel, uint64 options)
fn cb_create_context(ctx: &mut CpuContext) -> HleResult {
    let out = ctx.rdi();
    if out == 0 {
        return ERR_INVALID_ARGUMENT;
    }
    let mut svc = service();
    match svc.create_context(ctx.rsi() as i32, ctx.rdx() as u32, ctx.rcx()) {
        Ok(id) => {
            if ctx.write_u32(out, id as u32) {
                OK
            } else {
                svc.contexts.remove(&id);
                ERR_INVALID_ARGUMENT
            }
        }
        Err(e) => e,
    }
}

/// (context, handle, uint64 options)
fn cb_register_context(ctx: &mut CpuContext) -> HleResult {
    service().register_context(ctx.rdi() as i32, ctx.rdx())
}

fn cb_destroy_context(ctx: &mut CpuContext) -> HleResult {
    service().destroy_context(ctx.rdi() as i32)
}

/// (context, handle, SceNpTrophy2GameDetails*, SceNpTrophy2GameData*)
fn cb_get_game_info(ctx: &mut CpuContext) -> HleResult {
    let (details, data) = (ctx.rdx(), ctx.rcx());
    if details == 0 && data == 0 {
        return ERR_INVALID_ARGUMENT;
    }
    match service().game_info(ctx.rdi() as i32) {
        Ok(pair) => write_pair(ctx, details, data, &pair),
        Err(e) => e,
    }
}

/// (context, handle, groupId, SceNpTrophy2GroupDetails*, SceNpTrophy2GroupData*)
fn cb_get_group_info(ctx: &mut CpuContext) -> HleResult {
    let (details, data) = (ctx.rcx(), ctx.r8());
    if details == 0 && data == 0 {
        return ERR_INVALID_ARGUMENT;
    }
    match service().group_info(ctx.rdi() as i32, ctx.rdx() as i32) {
        Ok(pair) => write_pair(ctx, details, data, &pair),
        Err(e) => e,
    }
}

/// (context, handle, trophyId, SceNpTrophy2Details*, SceNpTrophy2Data*)
fn cb_get_trophy_info(ctx: &mut CpuContext) -> HleResult {
    let (details, data) = (ctx.rcx(), ctx.r8());
    if details == 0 && data == 0 {
        return ERR_INVALID_ARGUMENT;
    }
    match service().trophy_info(ctx.rdi() as i32, ctx.rdx() as i32) {
        Ok(pair) => write_pair(ctx, details, data, &pair),
        Err(e) => e,
    }
}

type ArrayQuery = fn(&TrophyService, i32, u32, u32) -> Result<Vec<(Vec<u8>, Vec<u8>)>, HleResult>;

/// (context, handle, offset, limit, details[], data[], uint32* count)
fn info_array(ctx: &CpuContext, query: ArrayQuery, details_size: usize, data_size: usize) -> HleResult {
    let (details, data, count) = (ctx.r8(), ctx.r9(), stack_arg(ctx, 0));
    if count == 0 || (details == 0 && data == 0) {
        return ERR_INVALID_ARGUMENT;
    }
    let entries = match query(&service(), ctx.rdi() as i32, ctx.rdx() as u32, ctx.rcx() as u32) {
        Ok(entries) => entries,
        Err(e) => return e,
    };
    for (i, pair) in entries.iter().enumerate() {
        let d = if details != 0 { details + (i * details_size) as u64 } else { 0 };
        let v = if data != 0 { data + (i * data_size) as u64 } else { 0 };
        let r = write_pair(ctx, d, v, pair);
        if r != OK {
            return r;
        }
    }
    if !ctx.write_u32(count, entries.len() as u32) {
        return ERR_INVALID_ARGUMENT;
    }
    OK
}

fn cb_get_group_info_array(ctx: &mut CpuContext) -> HleResult {
    info_array(ctx, TrophyService::group_info_array, GROUP_DETAILS_SIZE, GROUP_DATA_SIZE)
}

fn cb_get_trophy_info_array(ctx: &mut CpuContext) -> HleResult {
    info_array(ctx, TrophyService::trophy_info_array, TROPHY_DETAILS_SIZE, TROPHY_DATA_SIZE)
}

/// Icons are not served: the size output is zeroed and ICON_FILE_NOT_FOUND
/// is returned.
fn icon_not_found(ctx: &CpuContext, size: u64) -> HleResult {
    if size != 0 {
        ctx.write_u64(size, 0);
    }
    ERR_ICON_FILE_NOT_FOUND
}

/// (context, handle, void* buffer, size_t* size)
fn cb_get_game_icon(ctx: &mut CpuContext) -> HleResult {
    icon_not_found(ctx, ctx.rcx())
}

/// (context, handle, groupId / trophyId / rewardId, void* buffer, size_t* size)
fn cb_get_id_icon(ctx: &mut CpuContext) -> HleResult {
    icon_not_found(ctx, ctx.r8())
}

fn cb_ok(_ctx: &mut CpuContext) -> HleResult {
    OK
}

type Export = (&'static str, &'static str, fn(&mut CpuContext) -> HleResult);

const EXPORTS: &[Export] = &[
    ("Bagshr7OQ6Q", "sceNpTrophy2CreateContext", cb_create_context),
    ("Gz1rmUZpROM", "sceNpTrophy2CreateHandle", cb_create_handle),
    ("bIDov3wBu5Q", "sceNpTrophy2RegisterContext", cb_register_context),
    ("4IzqhhUQ3nk", "sceNpTrophy2GetGameInfo", cb_get_game_info),
    ("DoZWauG8mu0", "sceNpTrophy2GetGroupInfo", cb_get_group_info),
    ("+PDSI6WgPRc", "sceNpTrophy2GetGroupInfoArray", cb_get_group_info_array),
    ("EwNylPdWUTM", "sceNpTrophy2GetTrophyInfo", cb_get_trophy_info),
    ("y3zHpdZO6ME", "sceNpTrophy2GetTrophyInfoArray", cb_get_trophy_info_array),
    ("2QgUy+xJqS0", "sceNpTrophy2GetGameIcon", cb_get_game_icon),
    ("6IjXJUy6ZnA", "sceNpTrophy2GetGroupIcon", cb_get_id_icon),
    ("-9LLVU0uvs8", "sceNpTrophy2GetTrophyIcon", cb_get_id_icon),
    ("BsE-m8JxIOg", "sceNpTrophy2GetRewardIcon", cb_get_id_icon),
    ("sUXGfNMalIo", "sceNpTrophy2RegisterUnlockCallback", cb_ok),
    ("wVqxM58sIKs", "sceNpTrophy2UnregisterUnlockCallback", cb_ok),
    ("EHQEDVXZ0TI", "sceNpTrophy2ShowTrophyList", cb_ok),
    ("fYapWA9xVmA", "sceNpTrophy2AbortHandle", cb_ok),
    ("d8P11CI40KE", "sceNpTrophy2DestroyHandle", cb_ok),
    ("sysY2FHYff4", "sceNpTrophy2DestroyContext", cb_destroy_context),
];

pub fn register(reg: &mut HleRegistry) {
    for &(nid, name, cb) in EXPORTS {
        reg.register_fn(nid, name, "libSceNpTrophy2", cb);
    }
}

#[cfg(test)]
mod tests {
    use super::package::tests::{temp_dir, write_package, TEST_CONF, TEST_EXTRACTION, TEST_META, TEST_STATS};
    use super::*;
    use crate::context::CpuRegister::{R8, R9, Rcx, Rdi, Rdx, Rsi, Rsp};

    const BASE: u64 = 0x5D20_0000_0000;

    /// Test package: trophies 1..=3 name platinum 0; trophy 5 is progressive.
    const PLAT_CONF: &str = r#"{"defaultLanguage":"en-US","trophies":[
        {"id":0,"grade":"P"},
        {"id":1,"grade":"G","platinumTrophyId":0,"unlockCondition":{"udsStatId":10,"comparator":"ge","targetValue":10}},
        {"id":2,"grade":"S","platinumTrophyId":0,"groupId":1,"hidden":true},
        {"id":3,"grade":"B","platinumTrophyId":0},
        {"id":5,"grade":"B","unlockCondition":{"progressive":true,"udsStatId":30,"comparator":"le","targetValue":5}}
    ]}"#;
    const PLAT_META: &str = r#"{"metadata":{"titleMetadata":{"name":"Test game"},
        "groupMetadata":[{"id":1,"name":"Extra"}],
        "trophyMetadata":[{"id":1,"name":"Find All Coins","detail":"Find all coins in the level.","reward":"Hat"}]}}"#;

    fn fixture(tag: &str, conf: &str, meta: &str) -> (TrophyService, PathBuf) {
        let dir = temp_dir(tag);
        let app0 = dir.join("app0");
        std::fs::create_dir_all(app0.join("sce_sys/trophy2")).unwrap();
        std::fs::create_dir_all(app0.join("sce_sys/uds")).unwrap();
        write_package(
            &app0.join(package::package_path(0)),
            &[("tropconf.json", conf.as_bytes()), ("tropmeta_en-US.json", meta.as_bytes())],
        );
        write_package(
            &app0.join(package::uds_package_path(0)),
            &[
                ("stats_definition.json", TEST_STATS.as_bytes()),
                ("stats_extraction.json", TEST_EXTRACTION.as_bytes()),
            ],
        );
        let env = Env::Fixed { app0, data: dir.join("data"), title: "PPSA00001".into(), language: 1 };
        (TrophyService::new(env), dir)
    }

    #[test]
    fn nids_match_names() {
        for &(nid, name, _) in EXPORTS {
            assert_eq!(ps5nid::compute_nid(name), nid, "NID mismatch for {name}");
        }
    }

    #[test]
    fn context_lifecycle_and_errors() {
        let (mut s, dir) = fixture("ctx", TEST_CONF, TEST_META);
        assert_eq!(s.create_context(-1, 0, 0), Err(ERR_INVALID_ARGUMENT));
        assert_eq!(s.create_context(1, 0, 1), Err(ERR_INVALID_ARGUMENT));
        assert_eq!(s.create_context(1, 0, 0), Ok(1));
        assert_eq!(s.create_context(1, 0, 0), Err(ERR_CONTEXT_ALREADY_EXISTS));
        for label in 1..8 {
            assert!(s.create_context(1, label, 0).is_ok());
        }
        assert_eq!(s.create_context(2, 0, 0), Err(ERR_CONTEXT_EXCEEDS_MAX));
        assert_eq!(s.game_info(1).unwrap_err(), ERR_NOT_REGISTERED);
        assert_eq!(s.game_info(99).unwrap_err(), ERR_INVALID_CONTEXT);
        assert_eq!(s.register_context(99, 0), ERR_INVALID_CONTEXT);
        assert_eq!(s.register_context(1, 1), ERR_INVALID_ARGUMENT);
        assert_eq!(s.register_context(1, 0), OK);
        assert_eq!(s.register_context(1, 0), ERR_ALREADY_REGISTERED);
        // Service label 1 has no package.
        assert_eq!(s.register_context(2, 0), ERR_TITLE_CONF_NOT_INSTALLED);
        assert_eq!(s.destroy_context(2), OK);
        assert_eq!(s.destroy_context(2), ERR_INVALID_CONTEXT);
        assert_eq!(s.create_context(2, 0, 0), Ok(9));
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn info_structures() {
        let (mut s, dir) = fixture("info", PLAT_CONF, PLAT_META);
        let c = s.create_context(1, 0, 0).unwrap();
        assert_eq!(s.register_context(c, 0), OK);
        let (details, data) = s.game_info(c).unwrap();
        let u32_at = |b: &[u8], o: usize| u32::from_le_bytes(b[o..o + 4].try_into().unwrap());
        // groups -1 and 1; 5 trophies: 1 P, 1 G, 1 S, 2 B.
        assert_eq!((0..6).map(|i| u32_at(&details, i * 4)).collect::<Vec<_>>(), vec![2, 5, 1, 1, 1, 2]);
        assert_eq!(&details[24..34], b"Test game\0");
        assert_eq!(data, vec![0u8; GAME_DATA_SIZE]);
        assert_eq!(s.group_info(c, 7).unwrap_err(), ERR_INVALID_GROUP_ID);
        let (gd, gdata) = s.group_info(c, 1).unwrap();
        assert_eq!(u32_at(&gd, 0) as i32, 1);
        assert_eq!(u32_at(&gd, 4), 1);
        assert_eq!(u32_at(&gd, 16), 1); // silver
        assert_eq!(&gd[24..30], b"Extra\0");
        assert_eq!(u32_at(&gdata, 0), 1);
        let groups = s.group_info_array(c, 0, 10).unwrap();
        assert_eq!(groups.len(), 2);
        assert_eq!(u32_at(&groups[0].0, 0) as i32, -1);
        assert_eq!(s.group_info_array(c, 5, 10).unwrap().len(), 0);
        assert_eq!(s.trophy_info(c, 4).unwrap_err(), ERR_INVALID_TROPHY_ID);
        let (td, _) = s.trophy_info(c, 1).unwrap();
        assert_eq!((u32_at(&td, 0), u32_at(&td, 4), u32_at(&td, 8) as i32), (1, 2, -1));
        assert_eq!(u32_at(&td, 16), 0, "non-progressive trophy has no target");
        assert_eq!(&td[32..47], b"Find All Coins\0");
        assert_eq!(&td[160..189], b"Find all coins in the level.\0");
        assert_eq!(&td[1184..1188], b"Hat\0");
        let (td, tdata) = s.trophy_info(c, 5).unwrap();
        assert_eq!((u32_at(&td, 16), td[24]), (1, 5));
        assert_eq!((u32_at(&tdata, 8), tdata[16]), (1, 0));
        assert_eq!(s.trophy_info(c, 2).unwrap().0[12], 1, "hidden flag");
        let trophies = s.trophy_info_array(c, 1, 2).unwrap();
        assert_eq!(trophies.iter().map(|t| u32_at(&t.0, 0)).collect::<Vec<_>>(), vec![1, 2]);
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn unlocks_platinum_and_persist() {
        let (mut s, dir) = fixture("unlock", PLAT_CONF, PLAT_META);
        let key = (1, 0);
        assert!(s.record_unlock(key, 0, false).is_empty(), "platinum cannot be unlocked directly");
        assert!(s.record_unlock(key, 5, false).is_empty(), "progressive trophy needs its stat");
        assert!(s.record_unlock(key, 77, false).is_empty());
        assert_eq!(s.record_unlock(key, 1, false), vec![1]);
        assert!(s.record_unlock(key, 1, false).is_empty(), "already unlocked");
        assert_eq!(s.record_unlock(key, 2, false), vec![2]);
        assert_eq!(s.record_unlock(key, 3, false), vec![3, 0], "last member unlocks platinum");
        let path = dir.join("data/_Trophies/PPSA00001/trophies_1_0.json");
        let saved = package::load_unlock_data(&path);
        assert_eq!(saved.unlocked, [0, 1, 2, 3].into_iter().collect());
        assert_eq!(saved.timestamps[&3], saved.timestamps[&0]);
        // Stat path: TIME.seconds (uint32) <= 5 unlocks progressive trophy 5.
        let props: BTreeMap<String, UdsInteger> = [("seconds".to_string(), UdsInteger::U32(4))].into();
        assert_eq!(s.uds_stat_event(key, "TIME", &props), vec![5]);
        // A fresh service reloads the persisted unlocks.
        let env = Env::Fixed {
            app0: dir.join("app0"),
            data: dir.join("data"),
            title: "PPSA00001".into(),
            language: 1,
        };
        let mut fresh = TrophyService::new(env);
        let c = fresh.create_context(1, 0, 0).unwrap();
        assert_eq!(fresh.register_context(c, 0), OK);
        let (_, data) = fresh.game_info(c).unwrap();
        let u32_at = |o: usize| u32::from_le_bytes(data[o..o + 4].try_into().unwrap());
        assert_eq!((u32_at(0), u32_at(4), u32_at(20)), (5, 1, 100));
        let (_, tdata) = fresh.trophy_info(c, 5).unwrap();
        assert_eq!((tdata[4], tdata[16]), (1, 5), "unlocked progressive reports its target");
        assert!(u64::from_le_bytes(tdata[24..32].try_into().unwrap()) > package::UNIX_EPOCH_TICK);
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn guest_entry_points() {
        let vm = ps5vm::VirtualMemory::new();
        vm.map(BASE, 0x4000, ps5vm::MemProtect::RW).unwrap();
        let mut ctx = CpuContext::new(&vm);
        ctx.set(Rdi, 0);
        assert_eq!(cb_create_handle(&mut ctx), ERR_INVALID_ARGUMENT);
        ctx.set(Rdi, BASE);
        assert_eq!(cb_create_handle(&mut ctx), OK);
        assert_eq!(ctx.read_u32(BASE), Some(1));
        // Both outputs null.
        ctx.set(Rdi, 1);
        ctx.set(Rdx, 0);
        ctx.set(Rcx, 0);
        assert_eq!(cb_get_game_info(&mut ctx), ERR_INVALID_ARGUMENT);
        ctx.set(R8, 0);
        assert_eq!(cb_get_trophy_info(&mut ctx), ERR_INVALID_ARGUMENT);
        assert_eq!(cb_get_group_info(&mut ctx), ERR_INVALID_ARGUMENT);
        // Array without count pointer.
        ctx.write_u64(BASE + 0x100, 0);
        ctx.set(Rsp, BASE + 0xF8);
        ctx.set(R8, BASE + 0x200);
        ctx.set(R9, 0);
        assert_eq!(cb_get_trophy_info_array(&mut ctx), ERR_INVALID_ARGUMENT);
        // Icons: size zeroed, not found.
        ctx.write_u64(BASE + 0x300, 0x55);
        ctx.set(Rcx, BASE + 0x300);
        assert_eq!(cb_get_game_icon(&mut ctx), ERR_ICON_FILE_NOT_FOUND);
        assert_eq!(ctx.read_u64(BASE + 0x300), Some(0));
        ctx.write_u64(BASE + 0x300, 0x55);
        ctx.set(R8, BASE + 0x300);
        assert_eq!(cb_get_id_icon(&mut ctx), ERR_ICON_FILE_NOT_FOUND);
        assert_eq!(ctx.read_u64(BASE + 0x300), Some(0));
        // Reward icons take the trophy icon's arguments (size pointer in r8).
        let reward = EXPORTS.iter().find(|e| e.1 == "sceNpTrophy2GetRewardIcon").expect("sceNpTrophy2GetRewardIcon is exported");
        ctx.write_u64(BASE + 0x300, 0x55);
        assert_eq!((reward.2)(&mut ctx), ERR_ICON_FILE_NOT_FOUND);
        assert_eq!(ctx.read_u64(BASE + 0x300), Some(0));
        // CreateContext validation (no service state touched).
        ctx.set(Rdi, 0);
        assert_eq!(cb_create_context(&mut ctx), ERR_INVALID_ARGUMENT);
        ctx.set(Rdi, BASE);
        ctx.set(Rsi, (-5i32) as u32 as u64);
        assert_eq!(cb_create_context(&mut ctx), ERR_INVALID_ARGUMENT);
    }

    #[test]
    fn info_array_writes_strided_entries() {
        // Drive the array writer with a fixed service via the shared helper.
        let (mut s, dir) = fixture("array", PLAT_CONF, PLAT_META);
        let c = s.create_context(1, 0, 0).unwrap();
        assert_eq!(s.register_context(c, 0), OK);
        let vm = ps5vm::VirtualMemory::new();
        const BASE: u64 = 0x5D21_0000_0000;
        vm.map(BASE, 0x4000, ps5vm::MemProtect::RW).unwrap();
        let mut ctx = CpuContext::new(&vm);
        ctx.write_u64(BASE + 8, BASE + 0x10);
        ctx.set(Rsp, BASE);
        ctx.set(Rdi, c as u64);
        ctx.set(Rdx, 0);
        ctx.set(Rcx, 10);
        ctx.set(R8, 0);
        ctx.set(R9, BASE + 0x100);
        let entries = s.trophy_info_array(c, 0, 10).unwrap();
        assert_eq!(entries.len(), 5);
        for (i, pair) in entries.iter().enumerate() {
            assert_eq!(write_pair(&ctx, 0, BASE + 0x100 + (i * TROPHY_DATA_SIZE) as u64, pair), OK);
        }
        assert_eq!(ctx.read_u32(BASE + 0x100 + 4 * TROPHY_DATA_SIZE as u64), Some(5));
        let _ = std::fs::remove_dir_all(dir);
    }
}
