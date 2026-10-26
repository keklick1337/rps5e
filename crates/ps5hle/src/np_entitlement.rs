// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! libSceNpEntitlementAccess HLE: offline entitlements.
//!
//! The installed main package is reported as a full SKU, and a fixed set of
//! entitlement labels that represent locally installed base-game content is
//! reported as owned (some titles gate their main mode on these labels).
//! Every other add-on / edition remains unowned.

use crate::context::{CpuContext, CpuRegister};
use crate::registry::{HleRegistry, HleResult};
use std::collections::BTreeMap;
use std::sync::{Mutex, MutexGuard};

const OK: HleResult = 0;
const ERR_PARAMETER: HleResult = 0x817D_0002u32 as i32 as i64;
const ERR_NO_ENTITLEMENT: HleResult = 0x817D_0007u32 as i32 as i64;

const BOOT_PARAM_CLEAR_SIZE: usize = 0x20;
/// SceNpUnifiedEntitlementLabel: 16 chars + NUL.
const LABEL_SIZE: usize = 17;
const LABEL_PADDING: usize = 3;
/// SceNpEntitlementAccessAddcontEntitlementInfo:
/// label[17], pad[3], packageType u32 @0x14, downloadStatus u32 @0x18.
const ADDCONT_INFO_SIZE: usize = LABEL_SIZE + LABEL_PADDING + 4 + 4;

/// Package type PSAL (license-style add-on).
const PACKAGE_TYPE_PSAL: u32 = 3;
/// Download status INSTALLED.
const DOWNLOAD_STATUS_INSTALLED: u32 = 4;
/// SKU flag FULL.
const SKU_FLAG_FULL: u32 = 3;

struct Entitlement {
    label: &'static str,
    package_type: u32,
    download_status: u32,
}

const fn installed(label: &'static str) -> Entitlement {
    Entitlement { label, package_type: PACKAGE_TYPE_PSAL, download_status: DOWNLOAD_STATUS_INSTALLED }
}

/// Base-game entitlements owned by any locally installed main package
/// (GTA V Enhanced story-mode labels, Ghost of Yotei app/base labels).
static OWNED: [Entitlement; 5] = [
    installed("85y-je"),
    installed("5d5c48"),
    installed("_mtqu6"),
    installed("GHOST2APP0000000"),
    installed("GHOST2BASE000000"),
];

fn serialize_info(e: &Entitlement) -> [u8; ADDCONT_INFO_SIZE] {
    let mut info = [0u8; ADDCONT_INFO_SIZE];
    let label = e.label.as_bytes();
    let n = label.len().min(LABEL_SIZE - 1);
    info[..n].copy_from_slice(&label[..n]);
    let off = LABEL_SIZE + LABEL_PADDING;
    info[off..off + 4].copy_from_slice(&e.package_type.to_le_bytes());
    info[off + 4..off + 8].copy_from_slice(&e.download_status.to_le_bytes());
    info
}

/// Label bytes up to the first NUL.
fn label_of(bytes: &[u8]) -> &[u8] {
    let len = bytes.iter().position(|&b| b == 0).unwrap_or(bytes.len());
    &bytes[..len]
}

fn find_owned(label: &[u8]) -> Option<&'static Entitlement> {
    OWNED.iter().find(|e| e.label.as_bytes() == label)
}

fn cb_initialize(ctx: &mut CpuContext) -> HleResult {
    let init_param = ctx.rdi();
    let boot_param = ctx.rsi();
    if init_param == 0 || boot_param == 0 {
        return ERR_PARAMETER;
    }
    let mut st = requests();
    if !ctx.write_bytes(boot_param, &[0u8; BOOT_PARAM_CLEAR_SIZE]) {
        return ERR_PARAMETER;
    }
    st.initialized = true;
    log::debug!("np.entitlement: initialize init={init_param:#018x} boot={boot_param:#018x}");
    OK
}

fn cb_get_sku_flag(ctx: &mut CpuContext) -> HleResult {
    let addr = ctx.rdi();
    if addr == 0 {
        return ERR_PARAMETER;
    }
    if !ctx.write_u32(addr, SKU_FLAG_FULL) {
        return ERR_PARAMETER;
    }
    log::debug!("np.entitlement: get_sku_flag -> {SKU_FLAG_FULL}");
    OK
}

/// sceNpEntitlementAccessGetGameTrialsFlag(u32* flag): 0, the full game
/// (not a trial).
fn cb_get_game_trials_flag(ctx: &mut CpuContext) -> HleResult {
    let addr = ctx.rdi();
    if let Err(e) = requests().check() {
        return e;
    }
    if addr == 0 || !ctx.write_u32(addr, 0) {
        return ERR_PARAMETER;
    }
    OK
}

fn cb_get_addcont_info_list(ctx: &mut CpuContext) -> HleResult {
    // (serviceLabel, info list*, listNum, hitNum*)
    let list = ctx.rsi();
    let list_num = ctx.rdx() as u32;
    let hit_num_addr = ctx.rcx();
    if hit_num_addr == 0 || (list == 0 && list_num != 0) {
        return ERR_PARAMETER;
    }
    let hit_num = OWNED.len() as u32;
    if !ctx.write_u32(hit_num_addr, hit_num) {
        return ERR_PARAMETER;
    }
    if list != 0 && list_num != 0 {
        // The cleared region must fit in an i32 byte count.
        let Some(clear_len) = (list_num as usize)
            .checked_mul(ADDCONT_INFO_SIZE)
            .filter(|&n| n <= i32::MAX as usize)
        else {
            return ERR_PARAMETER;
        };
        if !ctx.write_bytes(list, &vec![0u8; clear_len]) {
            return ERR_PARAMETER;
        }
        for (i, e) in OWNED.iter().take(list_num.min(hit_num) as usize).enumerate() {
            let addr = list + (i * ADDCONT_INFO_SIZE) as u64;
            if !ctx.write_bytes(addr, &serialize_info(e)) {
                return ERR_PARAMETER;
            }
        }
    }
    log::debug!(
        "np.entitlement: get_addcont_info_list service={} list={list:#018x} list_num={list_num} hit_num={hit_num}",
        ctx.rdi()
    );
    OK
}

fn cb_get_addcont_info(ctx: &mut CpuContext) -> HleResult {
    // (serviceLabel, entitlementLabel*, info*)
    let label_addr = ctx.rsi();
    let info_addr = ctx.rdx();
    if label_addr == 0 || info_addr == 0 {
        return ERR_PARAMETER;
    }
    let Some(raw) = ctx.read_bytes(label_addr, LABEL_SIZE) else {
        return ERR_PARAMETER;
    };
    let label = label_of(&raw);
    if !ctx.write_bytes(info_addr, &[0u8; ADDCONT_INFO_SIZE]) {
        return ERR_PARAMETER;
    }
    match find_owned(label) {
        Some(e) => {
            if !ctx.write_bytes(info_addr, &serialize_info(e)) {
                return ERR_PARAMETER;
            }
            log::debug!("np.entitlement: get_addcont_info label='{}' -> owned", e.label);
            OK
        }
        None => {
            log::debug!(
                "np.entitlement: get_addcont_info label='{}' -> no entitlement",
                String::from_utf8_lossy(label)
            );
            ERR_NO_ENTITLEMENT
        }
    }
}

/// sceNpEntitlementAccessGetEntitlementKey(serviceLabel, entitlementLabel*,
/// key*): no entitlement carries a key, the key is left untouched.
fn cb_get_entitlement_key(ctx: &mut CpuContext) -> HleResult {
    if ctx.rsi() == 0 || ctx.rdx() == 0 {
        return ERR_PARAMETER;
    }
    ERR_NO_ENTITLEMENT
}

// ---------------------------------------------------------------------------
// Requests. The console is offline: every request completes when it is
// created and keeps its result until the guest deletes it. Lists are empty
// (result 0, no hits); a single unified entitlement and a consume request
// answer NO_ENTITLEMENT through the poll's result.
// ---------------------------------------------------------------------------

const ERR_NOT_INITIALIZED: HleResult = 0x817D_0001u32 as i32 as i64;
const ERR_OUT_OF_MEMORY: HleResult = 0x817D_0010u32 as i32 as i64;
const ERR_REQUEST_NOT_FOUND: HleResult = 0x817D_0015u32 as i32 as i64;
const NO_ENTITLEMENT: i32 = 0x817D_0007u32 as i32;
/// SCE_NP_ENTITLEMENT_ACCESS_INVALID_OFFSET.
const INVALID_OFFSET: i32 = -1;
const INFO_LIST_MAX_SIZE: i32 = 100;
/// SceNpEntitlementAccessRequestEntitlementInfoListParam: size u64,
/// entitlementType u32, offset i32, limit i32, sort u32, direction u32,
/// packageType u32.
const LIST_PARAM_SIZE: usize = 32;
/// SceNpEntitlementAccessTransactionId: data[65] + padding[7].
const TRANSACTION_ID_SIZE: usize = 72;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum RequestKind {
    UnifiedInfo,
    ServiceInfo,
    UnifiedList,
    ServiceList,
    /// Consumption of a unified or service entitlement (one poll serves both).
    Consume,
}

struct Requests {
    initialized: bool,
    next_id: i64,
    next_transaction: u64,
    live: BTreeMap<i64, (RequestKind, i32)>,
}

impl Requests {
    const fn new() -> Self {
        Self { initialized: false, next_id: 1, next_transaction: 1, live: BTreeMap::new() }
    }

    fn check(&self) -> Result<(), HleResult> {
        if self.initialized { Ok(()) } else { Err(ERR_NOT_INITIALIZED) }
    }

    fn create(&mut self, kind: RequestKind, result: i32) -> Result<i64, HleResult> {
        self.check()?;
        if self.next_id == i64::MAX {
            return Err(ERR_OUT_OF_MEMORY);
        }
        let id = self.next_id;
        self.next_id += 1;
        self.live.insert(id, (kind, result));
        Ok(id)
    }

    fn poll(&self, id: i64, kind: RequestKind) -> Result<i32, HleResult> {
        self.check()?;
        match self.live.get(&id) {
            None => Err(ERR_REQUEST_NOT_FOUND),
            Some(&(k, _)) if k != kind => Err(ERR_PARAMETER),
            Some(&(_, result)) => Ok(result),
        }
    }

    /// Offline requests finish at creation: there is no work to abort.
    fn abort(&self, id: i64) -> HleResult {
        match self.check() {
            Err(e) => e,
            Ok(()) if self.live.contains_key(&id) => OK,
            Ok(()) => ERR_REQUEST_NOT_FOUND,
        }
    }

    fn delete(&mut self, id: i64) -> HleResult {
        match self.check() {
            Err(e) => e,
            Ok(()) if self.live.remove(&id).is_some() => OK,
            Ok(()) => ERR_REQUEST_NOT_FOUND,
        }
    }

    /// The 64-character transaction id text: "6b797479" + 24 hex digits.
    fn transaction_id(&mut self) -> Result<String, HleResult> {
        self.check()?;
        if self.next_transaction == 0 {
            return Err(ERR_OUT_OF_MEMORY);
        }
        let id = format!("6b797479{:024x}", self.next_transaction);
        self.next_transaction = self.next_transaction.wrapping_add(1);
        Ok(id)
    }
}

static REQUESTS: Mutex<Requests> = Mutex::new(Requests::new());

fn requests() -> MutexGuard<'static, Requests> {
    REQUESTS.lock().unwrap_or_else(|e| e.into_inner())
}

/// Validates a list request parameter (the bytes of the 32-byte struct).
fn list_param_valid(kind: RequestKind, p: &[u8]) -> bool {
    let u32_at = |o: usize| u32::from_le_bytes(p[o..o + 4].try_into().unwrap());
    let size = u64::from_le_bytes(p[0..8].try_into().unwrap());
    let (ty, offset, limit) = (u32_at(8), u32_at(12) as i32, u32_at(16) as i32);
    let (sort, direction, package) = (u32_at(20), u32_at(24), u32_at(28));
    if size != LIST_PARAM_SIZE as u64
        || offset < 0
        || !(1..=INFO_LIST_MAX_SIZE).contains(&limit)
        || sort > 1
        || direction > 2
    {
        return false;
    }
    match kind {
        RequestKind::UnifiedList => ty == 2 && (package == 0 || (4..=6).contains(&package)),
        _ => ty == 1 && package == 0,
    }
}

fn write_request_id(ctx: &CpuContext, out: u64, id: Result<i64, HleResult>) -> HleResult {
    match id {
        Ok(id) if ctx.write_u64(out, id as u64) => OK,
        Ok(_) => ERR_PARAMETER,
        Err(e) => e,
    }
}

/// Request{Unified,Service}EntitlementInfoList(userId, serviceLabel, list*,
/// listNum, param*, requestId*).
fn request_list(ctx: &mut CpuContext, kind: RequestKind) -> HleResult {
    let (list, list_num, param, out) = (ctx.rdx(), ctx.rcx() as u32, ctx.r8(), ctx.r9());
    let mut st = requests();
    if let Err(e) = st.check() {
        return e;
    }
    if param == 0 || out == 0 || (list == 0 && list_num != 0) {
        return ERR_PARAMETER;
    }
    let Some(p) = ctx.read_bytes(param, LIST_PARAM_SIZE) else { return ERR_PARAMETER };
    if !list_param_valid(kind, &p) {
        return ERR_PARAMETER;
    }
    let id = st.create(kind, 0);
    write_request_id(ctx, out, id)
}

fn cb_request_unified_list(ctx: &mut CpuContext) -> HleResult {
    request_list(ctx, RequestKind::UnifiedList)
}

fn cb_request_service_list(ctx: &mut CpuContext) -> HleResult {
    request_list(ctx, RequestKind::ServiceList)
}

/// Request{Unified,Service}EntitlementInfo(userId, serviceLabel, label*,
/// requestId*): no entitlement is owned.
fn request_info(ctx: &mut CpuContext, kind: RequestKind) -> HleResult {
    let (label, out) = (ctx.rdx(), ctx.rcx());
    let mut st = requests();
    if let Err(e) = st.check() {
        return e;
    }
    if label == 0 || out == 0 {
        return ERR_PARAMETER;
    }
    if let Some(raw) = ctx.read_bytes(label, LABEL_SIZE) {
        log::debug!("np.entitlement: {kind:?} '{}' -> no entitlement", String::from_utf8_lossy(label_of(&raw)));
    }
    let id = st.create(kind, NO_ENTITLEMENT);
    write_request_id(ctx, out, id)
}

fn cb_request_unified_info(ctx: &mut CpuContext) -> HleResult {
    request_info(ctx, RequestKind::UnifiedInfo)
}

fn cb_request_service_info(ctx: &mut CpuContext) -> HleResult {
    request_info(ctx, RequestKind::ServiceInfo)
}

/// RequestConsume{Unified,Service}Entitlement(userId, serviceLabel, label*,
/// transactionId*, useCount, requestId*).
fn cb_request_consume_unified(ctx: &mut CpuContext) -> HleResult {
    let (label, transaction, use_count, out) = (ctx.rdx(), ctx.rcx(), ctx.r8() as i32, ctx.r9());
    let mut st = requests();
    if let Err(e) = st.check() {
        return e;
    }
    if label == 0 || transaction == 0 || use_count <= 0 || out == 0 {
        return ERR_PARAMETER;
    }
    let id = st.create(RequestKind::Consume, NO_ENTITLEMENT);
    write_request_id(ctx, out, id)
}

/// Poll{Unified,Service}EntitlementInfoList(requestId, result*, list*,
/// listNum, hitNum*, nextOffset*, previousOffset*).
fn poll_list(ctx: &mut CpuContext, kind: RequestKind) -> HleResult {
    let (id, result, list, list_num, hit_num) = (ctx.rdi() as i64, ctx.rsi(), ctx.rdx(), ctx.rcx() as u32, ctx.r8());
    let (next, previous) = (ctx.r9(), stack_arg(ctx, 0));
    let st = requests();
    if let Err(e) = st.check() {
        return e;
    }
    if result == 0 || hit_num == 0 || next == 0 || previous == 0 || (list == 0 && list_num != 0) {
        return ERR_PARAMETER;
    }
    let value = match st.poll(id, kind) {
        Ok(v) => v,
        Err(e) => return e,
    };
    let ok = ctx.write_u32(result, value as u32)
        && ctx.write_u32(hit_num, 0)
        && ctx.write_u32(next, INVALID_OFFSET as u32)
        && ctx.write_u32(previous, INVALID_OFFSET as u32);
    if ok { OK } else { ERR_PARAMETER }
}

fn cb_poll_unified_list(ctx: &mut CpuContext) -> HleResult {
    poll_list(ctx, RequestKind::UnifiedList)
}

fn cb_poll_service_list(ctx: &mut CpuContext) -> HleResult {
    poll_list(ctx, RequestKind::ServiceList)
}

/// Poll{Unified,Service}EntitlementInfo(requestId, result*, info*): the
/// info is left untouched, the result is NO_ENTITLEMENT.
fn poll_info(ctx: &mut CpuContext, kind: RequestKind) -> HleResult {
    let (id, result, info) = (ctx.rdi() as i64, ctx.rsi(), ctx.rdx());
    let st = requests();
    if let Err(e) = st.check() {
        return e;
    }
    if info == 0 || result == 0 {
        return ERR_PARAMETER;
    }
    match st.poll(id, kind) {
        Ok(v) if ctx.write_u32(result, v as u32) => OK,
        Ok(_) => ERR_PARAMETER,
        Err(e) => e,
    }
}

fn cb_poll_unified_info(ctx: &mut CpuContext) -> HleResult {
    poll_info(ctx, RequestKind::UnifiedInfo)
}

fn cb_poll_service_info(ctx: &mut CpuContext) -> HleResult {
    poll_info(ctx, RequestKind::ServiceInfo)
}

/// PollConsumeEntitlement(requestId, result*, useLimit*).
fn cb_poll_consume(ctx: &mut CpuContext) -> HleResult {
    let (id, result, use_limit) = (ctx.rdi() as i64, ctx.rsi(), ctx.rdx());
    let st = requests();
    if let Err(e) = st.check() {
        return e;
    }
    if use_limit == 0 || result == 0 {
        return ERR_PARAMETER;
    }
    match st.poll(id, RequestKind::Consume) {
        Ok(v) => {
            if !ctx.write_u32(result, v as u32) || (v == 0 && !ctx.write_u32(use_limit, 0)) {
                return ERR_PARAMETER;
            }
            OK
        }
        Err(e) => e,
    }
}

fn cb_abort_request(ctx: &mut CpuContext) -> HleResult {
    requests().abort(ctx.rdi() as i64)
}

fn cb_delete_request(ctx: &mut CpuContext) -> HleResult {
    requests().delete(ctx.rdi() as i64)
}

/// GenerateTransactionId(SceNpEntitlementAccessTransactionId*).
fn cb_generate_transaction_id(ctx: &mut CpuContext) -> HleResult {
    let out = ctx.rdi();
    let mut st = requests();
    if let Err(e) = st.check() {
        return e;
    }
    if out == 0 {
        return ERR_PARAMETER;
    }
    let text = match st.transaction_id() {
        Ok(t) => t,
        Err(e) => return e,
    };
    let mut id = [0u8; TRANSACTION_ID_SIZE];
    id[..text.len()].copy_from_slice(text.as_bytes());
    if ctx.write_bytes(out, &id) { OK } else { ERR_PARAMETER }
}

fn stack_arg(ctx: &CpuContext, i: u64) -> u64 {
    ctx.read_u64(ctx.get(CpuRegister::Rsp) + 8 + 8 * i).unwrap_or(0)
}

const REQUEST_EXPORTS: &[(&str, &str, crate::registry::HleCallback)] = &[
    ("brbRxzr7qyI", "sceNpEntitlementAccessRequestServiceEntitlementInfoList", cb_request_service_list),
    ("uCZf2L27th8", "sceNpEntitlementAccessRequestUnifiedEntitlementInfoList", cb_request_unified_list),
    ("jautSRs4OdQ", "sceNpEntitlementAccessRequestUnifiedEntitlementInfo", cb_request_unified_info),
    ("eOxGbG3sPb0", "sceNpEntitlementAccessRequestConsumeUnifiedEntitlement", cb_request_consume_unified),
    ("Aqae0TjLvQU", "sceNpEntitlementAccessRequestConsumeServiceEntitlement", cb_request_consume_unified),
    ("M8icY9OwkKs", "sceNpEntitlementAccessRequestServiceEntitlementInfo", cb_request_service_info),
    ("ShGBFuoSSsQ", "sceNpEntitlementAccessPollServiceEntitlementInfo", cb_poll_service_info),
    ("aFv8qms6XTM", "sceNpEntitlementAccessPollServiceEntitlementInfoList", cb_poll_service_list),
    ("nAEqawEZG5s", "sceNpEntitlementAccessPollUnifiedEntitlementInfoList", cb_poll_unified_list),
    ("M4XlLFnzQaQ", "sceNpEntitlementAccessPollUnifiedEntitlementInfo", cb_poll_unified_info),
    ("69u+XqsoNd0", "sceNpEntitlementAccessPollConsumeEntitlement", cb_poll_consume),
    ("HFcQl9TMcFQ", "sceNpEntitlementAccessAbortRequest", cb_abort_request),
    ("Z0eQj8m7XA8", "sceNpEntitlementAccessDeleteRequest", cb_delete_request),
    ("IQtb-TaIjSM", "sceNpEntitlementAccessGenerateTransactionId", cb_generate_transaction_id),
    ("y7JD1gfwyfA", "sceNpEntitlementAccessGetGameTrialsFlag", cb_get_game_trials_flag),
];

pub fn register(reg: &mut HleRegistry) {
    const LIB: &str = "libSceNpEntitlementAccess";
    for &(nid, name, cb) in REQUEST_EXPORTS {
        reg.register_fn(nid, name, LIB, cb);
    }
    reg.register_fn("jO8DM8oyego", "sceNpEntitlementAccessInitialize", LIB, cb_initialize);
    reg.register_fn("lPDO62PpJIA", "sceNpEntitlementAccessGetSkuFlag", LIB, cb_get_sku_flag);
    reg.register_fn("TFyU+KFBv54", "sceNpEntitlementAccessGetAddcontEntitlementInfoList", LIB, cb_get_addcont_info_list);
    reg.register_fn("xddD23+8TfQ", "sceNpEntitlementAccessGetAddcontEntitlementInfo", LIB, cb_get_addcont_info);
    reg.register_fn("5LiMEPuW0DQ", "sceNpEntitlementAccessGetEntitlementKey", LIB, cb_get_entitlement_key);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_nids_match_names() {
        for &(nid, name, _) in REQUEST_EXPORTS {
            assert_eq!(ps5nid::compute_nid(name), nid, "NID mismatch for {name}");
        }
    }

    /// Requests before sceNpEntitlementAccessInitialize, unknown ids, a poll
    /// of the wrong kind, abort and delete.
    #[test]
    fn request_table_lifecycle() {
        let mut st = Requests::new();
        assert_eq!(st.create(RequestKind::UnifiedInfo, 0), Err(ERR_NOT_INITIALIZED));
        assert_eq!(st.poll(1, RequestKind::UnifiedInfo), Err(ERR_NOT_INITIALIZED));
        assert_eq!(st.abort(1), ERR_NOT_INITIALIZED);
        assert_eq!(st.delete(1), ERR_NOT_INITIALIZED);
        assert_eq!(st.transaction_id(), Err(ERR_NOT_INITIALIZED));
        st.initialized = true;
        let a = st.create(RequestKind::UnifiedInfo, NO_ENTITLEMENT).unwrap();
        let b = st.create(RequestKind::UnifiedList, 0).unwrap();
        assert!(a > 0 && b > a);
        assert_eq!(st.poll(a, RequestKind::UnifiedInfo), Ok(NO_ENTITLEMENT));
        assert_eq!(st.poll(a, RequestKind::UnifiedInfo), Ok(NO_ENTITLEMENT), "a poll does not consume the request");
        assert_eq!(st.poll(b, RequestKind::UnifiedList), Ok(0));
        assert_eq!(st.poll(b, RequestKind::ServiceList), Err(ERR_PARAMETER));
        assert_eq!(st.poll(99, RequestKind::UnifiedList), Err(ERR_REQUEST_NOT_FOUND));
        assert_eq!(st.abort(a), OK);
        assert_eq!(st.delete(a), OK);
        assert_eq!(st.delete(a), ERR_REQUEST_NOT_FOUND);
        assert_eq!(st.abort(a), ERR_REQUEST_NOT_FOUND);
        let t1 = st.transaction_id().unwrap();
        let t2 = st.transaction_id().unwrap();
        assert_eq!(t1, "6b797479000000000000000000000001");
        assert_ne!(t1, t2);
    }

    #[test]
    fn list_param_validation() {
        let param = |size: u64, ty: u32, offset: i32, limit: i32, sort: u32, dir: u32, pkg: u32| {
            let mut p = size.to_le_bytes().to_vec();
            for v in [ty, offset as u32, limit as u32, sort, dir, pkg] {
                p.extend_from_slice(&v.to_le_bytes());
            }
            p
        };
        assert!(list_param_valid(RequestKind::UnifiedList, &param(32, 2, 0, 100, 1, 2, 0)));
        assert!(list_param_valid(RequestKind::UnifiedList, &param(32, 2, 5, 1, 0, 0, 6)));
        assert!(!list_param_valid(RequestKind::UnifiedList, &param(32, 2, 0, 1, 0, 0, 3)));
        assert!(!list_param_valid(RequestKind::UnifiedList, &param(32, 1, 0, 1, 0, 0, 0)));
        assert!(list_param_valid(RequestKind::ServiceList, &param(32, 1, 0, 1, 0, 0, 0)));
        assert!(!list_param_valid(RequestKind::ServiceList, &param(32, 1, 0, 1, 0, 0, 4)));
        assert!(!list_param_valid(RequestKind::ServiceList, &param(24, 1, 0, 1, 0, 0, 0)));
        assert!(!list_param_valid(RequestKind::ServiceList, &param(32, 1, -1, 1, 0, 0, 0)));
        assert!(!list_param_valid(RequestKind::ServiceList, &param(32, 1, 0, 0, 0, 0, 0)));
        assert!(!list_param_valid(RequestKind::ServiceList, &param(32, 1, 0, 101, 0, 0, 0)));
        assert!(!list_param_valid(RequestKind::ServiceList, &param(32, 1, 0, 1, 2, 0, 0)));
        assert!(!list_param_valid(RequestKind::ServiceList, &param(32, 1, 0, 1, 0, 3, 0)));
    }

    /// The guest sequence of a unified entitlement check: request, poll
    /// (result NO_ENTITLEMENT, info untouched), delete; an empty list poll.
    #[test]
    fn unified_info_and_list_through_exports() {
        const BASE: u64 = 0x5D41_0300_0000;
        let vm = ps5vm::VirtualMemory::new();
        vm.map(BASE, 0x1000, ps5vm::MemProtect::RW).unwrap();
        let mut ctx = CpuContext::new(&vm);
        ctx.set(CpuRegister::Rdi, BASE);
        ctx.set(CpuRegister::Rsi, BASE + 0x40);
        assert_eq!(cb_initialize(&mut ctx), OK);

        ctx.write_bytes(BASE + 0x100, b"MSM2DDEUPGRADE00\0");
        ctx.set(CpuRegister::Rdi, 1);
        ctx.set(CpuRegister::Rsi, 0);
        ctx.set(CpuRegister::Rdx, BASE + 0x100);
        ctx.set(CpuRegister::Rcx, BASE + 0x200);
        assert_eq!(cb_request_unified_info(&mut ctx), OK);
        let id = ctx.read_u64(BASE + 0x200).unwrap();
        assert!(id > 0);

        ctx.write_bytes(BASE + 0x300, &[0xAB; 64]);
        ctx.set(CpuRegister::Rdi, id);
        ctx.set(CpuRegister::Rsi, BASE + 0x210);
        ctx.set(CpuRegister::Rdx, BASE + 0x300);
        assert_eq!(cb_poll_unified_info(&mut ctx), OK);
        assert_eq!(ctx.read_u32(BASE + 0x210), Some(NO_ENTITLEMENT as u32));
        assert_eq!(ctx.read_bytes(BASE + 0x300, 64).unwrap(), vec![0xAB; 64]);
        assert_eq!(cb_poll_unified_list(&mut ctx), ERR_PARAMETER, "the list poll needs its outputs");
        assert_eq!(cb_delete_request(&mut ctx), OK);
        assert_eq!(cb_poll_unified_info(&mut ctx), ERR_REQUEST_NOT_FOUND);

        // RequestUnifiedEntitlementInfoList(user, service, list=0, num=0, param, id*).
        let mut p = 32u64.to_le_bytes().to_vec();
        for v in [2u32, 0, 10, 0, 0, 0] {
            p.extend_from_slice(&v.to_le_bytes());
        }
        ctx.write_bytes(BASE + 0x400, &p);
        ctx.set(CpuRegister::Rdx, 0);
        ctx.set(CpuRegister::Rcx, 0);
        ctx.set(CpuRegister::R8, BASE + 0x400);
        ctx.set(CpuRegister::R9, BASE + 0x200);
        assert_eq!(cb_request_unified_list(&mut ctx), OK);
        let id = ctx.read_u64(BASE + 0x200).unwrap();
        // Poll(id, result*, list=0, num=0, hit*, next*, [rsp+8] previous*).
        ctx.write_u64(BASE + 0x808, BASE + 0x230);
        ctx.set(CpuRegister::Rsp, BASE + 0x800);
        ctx.write_bytes(BASE + 0x210, &[0x55; 0x30]);
        ctx.set(CpuRegister::Rdi, id);
        ctx.set(CpuRegister::Rsi, BASE + 0x210);
        ctx.set(CpuRegister::Rdx, 0);
        ctx.set(CpuRegister::Rcx, 0);
        ctx.set(CpuRegister::R8, BASE + 0x220);
        ctx.set(CpuRegister::R9, BASE + 0x228);
        assert_eq!(cb_poll_unified_list(&mut ctx), OK);
        assert_eq!(ctx.read_u32(BASE + 0x210), Some(0));
        assert_eq!(ctx.read_u32(BASE + 0x220), Some(0));
        assert_eq!(ctx.read_u32(BASE + 0x228), Some(u32::MAX));
        assert_eq!(ctx.read_u32(BASE + 0x230), Some(u32::MAX));
        assert_eq!(cb_poll_service_list(&mut ctx), ERR_PARAMETER, "a poll of another kind");
    }

    /// Service entitlement info and consumption: the request completes with
    /// NO_ENTITLEMENT, a unified poll does not accept it, and a service
    /// consume request is polled like a unified one.
    #[test]
    fn service_info_and_consume_through_exports() {
        const BASE: u64 = 0x5FAC_0000_0000;
        let vm = ps5vm::VirtualMemory::new();
        vm.map(BASE, 0x1000, ps5vm::MemProtect::RW).unwrap();
        let mut ctx = CpuContext::new(&vm);
        ctx.set(CpuRegister::Rdi, BASE);
        ctx.set(CpuRegister::Rsi, BASE + 0x40);
        assert_eq!(cb_initialize(&mut ctx), OK);

        ctx.write_bytes(BASE + 0x100, b"SERVICELABEL0000\0");
        ctx.set(CpuRegister::Rdi, 1);
        ctx.set(CpuRegister::Rsi, 0);
        ctx.set(CpuRegister::Rdx, 0);
        ctx.set(CpuRegister::Rcx, BASE + 0x200);
        assert_eq!(cb_request_service_info(&mut ctx), ERR_PARAMETER);
        ctx.set(CpuRegister::Rdx, BASE + 0x100);
        assert_eq!(cb_request_service_info(&mut ctx), OK);
        let id = ctx.read_u64(BASE + 0x200).unwrap();

        ctx.write_bytes(BASE + 0x300, &[0xAB; 64]);
        ctx.set(CpuRegister::Rdi, id);
        ctx.set(CpuRegister::Rsi, BASE + 0x210);
        ctx.set(CpuRegister::Rdx, BASE + 0x300);
        assert_eq!(cb_poll_unified_info(&mut ctx), ERR_PARAMETER, "a poll of another kind");
        assert_eq!(cb_poll_service_info(&mut ctx), OK);
        assert_eq!(ctx.read_u32(BASE + 0x210), Some(NO_ENTITLEMENT as u32));
        assert_eq!(ctx.read_bytes(BASE + 0x300, 64).unwrap(), vec![0xAB; 64]);
        assert_eq!(cb_delete_request(&mut ctx), OK);

        // RequestConsumeServiceEntitlement(user, service, label*, transaction*, useCount, id*).
        ctx.set(CpuRegister::Rdi, 1);
        ctx.set(CpuRegister::Rsi, 0);
        ctx.set(CpuRegister::Rdx, BASE + 0x100);
        ctx.set(CpuRegister::Rcx, BASE + 0x400);
        ctx.set(CpuRegister::R8, 0);
        ctx.set(CpuRegister::R9, BASE + 0x200);
        assert_eq!(cb_request_consume_unified(&mut ctx), ERR_PARAMETER, "use count must be positive");
        ctx.set(CpuRegister::R8, 1);
        assert_eq!(cb_request_consume_unified(&mut ctx), OK);
        let id = ctx.read_u64(BASE + 0x200).unwrap();
        ctx.set(CpuRegister::Rdi, id);
        ctx.set(CpuRegister::Rsi, BASE + 0x210);
        ctx.set(CpuRegister::Rdx, BASE + 0x220);
        assert_eq!(cb_poll_consume(&mut ctx), OK);
        assert_eq!(ctx.read_u32(BASE + 0x210), Some(NO_ENTITLEMENT as u32));
    }

    #[test]
    fn info_layout() {
        assert_eq!(ADDCONT_INFO_SIZE, 28);
        let info = serialize_info(&OWNED[3]);
        assert_eq!(&info[..16], b"GHOST2APP0000000");
        assert_eq!(&info[16..20], &[0, 0, 0, 0]);
        assert_eq!(u32::from_le_bytes(info[20..24].try_into().unwrap()), PACKAGE_TYPE_PSAL);
        assert_eq!(u32::from_le_bytes(info[24..28].try_into().unwrap()), DOWNLOAD_STATUS_INSTALLED);
    }

    #[test]
    fn label_lookup() {
        let mut raw = [0u8; LABEL_SIZE];
        raw[..6].copy_from_slice(b"85y-je");
        assert!(find_owned(label_of(&raw)).is_some());
        assert!(find_owned(label_of(b"unknown\0")).is_none());
        assert!(find_owned(label_of(&[0u8; LABEL_SIZE])).is_none());
    }
}
