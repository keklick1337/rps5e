// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! libSceContentSearch and libSceContentDelete HLE.
//!
//! The host content library is empty: searches find nothing. The libraries
//! accept their init parameters and terminate.

use std::sync::atomic::{AtomicBool, Ordering};

use crate::context::CpuContext;
use crate::registry::{HleCallback, HleRegistry, HleResult};

const OK: HleResult = 0;
const CONTENT_SEARCH_ERROR_INVALID_PARAM: HleResult = 0x809D_1003u32 as i32 as i64;
const CONTENT_DELETE_ERROR_INVALID_PARAM: HleResult = 0x809D_5001u32 as i32 as i64;

static SEARCH_INITIALIZED: AtomicBool = AtomicBool::new(false);
static DELETE_INITIALIZED: AtomicBool = AtomicBool::new(false);

/// sceContentSearchInit(const SceContentSearchInitParam* {memory_size u64})
fn cb_content_search_init(ctx: &mut CpuContext) -> HleResult {
    let param = ctx.rdi();
    if param == 0 {
        return CONTENT_SEARCH_ERROR_INVALID_PARAM;
    }
    log::debug!("[contentsearch] init memory_size={:?}", ctx.read_u64(param));
    SEARCH_INITIALIZED.store(true, Ordering::Release);
    OK
}

/// sceContentDeleteInitialize(const SceContentDeleteInitParam*): reserved[4],
/// heap_size u64 at +8, reserved[32].
fn cb_content_delete_initialize(ctx: &mut CpuContext) -> HleResult {
    let param = ctx.rdi();
    if param == 0 {
        return CONTENT_DELETE_ERROR_INVALID_PARAM;
    }
    log::debug!("[contentdelete] init heap_size={:?}", ctx.read_u64(param + 8));
    DELETE_INITIALIZED.store(true, Ordering::Release);
    OK
}

// ---------------------------------------------------------------------------
// Searches. The content library of the host is empty: every search finds
// nothing, and the library's update id stays 0.
// ---------------------------------------------------------------------------

const CONTENT_SEARCH_ERROR_NOT_INITIALIZED: HleResult = 0x809D_1001u32 as i32 as i64;
const CONTENT_SEARCH_ERROR_LIMIT_TOO_BIG: HleResult = 0x809D_100Au32 as i32 as i64;
const MAX_SEARCH_LIMIT: u32 = 92;
const LIBRARY_UPDATE_ID: u64 = 0;
/// SceContentSearchContentColumnSet: columns*, columnsLength i32,
/// logicalConnector i32.
const COLUMN_SET_SIZE: u64 = 16;
/// SceContentSearchContentColumn: columnType i32, columnOperator i32,
/// value*, logicalConnector i32 (+pad).
const COLUMN_SIZE: u64 = 24;
/// SceContentSearchContentOrderByCondition: columnType i32, sortType i32.
const ORDER_BY_SIZE: u64 = 8;

fn is_content_column(column: i32) -> bool {
    matches!(column, 1..=7 | 9..=12 | 14)
}

fn is_sort_column(column: i32) -> bool {
    matches!(column, 1 | 4..=6 | 9 | 10)
}

fn is_connector(connector: i32) -> bool {
    (0..=2).contains(&connector)
}

/// Checks the column sets of a search: every set has columns with known
/// types, operators and connectors, and a value.
fn column_sets_valid(ctx: &CpuContext, sets: u64, len: u32) -> bool {
    if sets == 0 && len != 0 {
        return false;
    }
    for i in 0..u64::from(len) {
        let set = sets + i * COLUMN_SET_SIZE;
        let (Some(columns), Some(count), Some(connector)) =
            (ctx.read_u64(set), ctx.read_u32(set + 8), ctx.read_u32(set + 12))
        else {
            return false;
        };
        let count = count as i32;
        if columns == 0 || count <= 0 || !is_connector(connector as i32) {
            return false;
        }
        for j in 0..count as u64 {
            let column = columns + j * COLUMN_SIZE;
            let (Some(ty), Some(op), Some(value), Some(conn)) =
                (ctx.read_u32(column), ctx.read_u32(column + 4), ctx.read_u64(column + 8), ctx.read_u32(column + 16))
            else {
                return false;
            };
            if !is_content_column(ty as i32) || !(0..=5).contains(&(op as i32)) || value == 0 || !is_connector(conn as i32) {
                return false;
            }
        }
    }
    true
}

fn order_by_valid(ctx: &CpuContext, conditions: u64, len: u32) -> bool {
    if conditions == 0 && len != 0 {
        return false;
    }
    (0..u64::from(len)).all(|i| {
        let c = conditions + i * ORDER_BY_SIZE;
        matches!((ctx.read_u32(c), ctx.read_u32(c + 4)), (Some(ty), Some(sort)) if is_sort_column(ty as i32) && sort <= 1)
    })
}

fn search_initialized() -> Result<(), HleResult> {
    if SEARCH_INITIALIZED.load(Ordering::Acquire) { Ok(()) } else { Err(CONTENT_SEARCH_ERROR_NOT_INITIALIZED) }
}

fn stack_arg(ctx: &CpuContext, i: u64) -> u64 {
    ctx.read_u64(ctx.get(crate::context::CpuRegister::Rsp) + 8 + 8 * i).unwrap_or(0)
}

/// sceContentSearchSearchContent(columnSet*, columnSetLength, orderBy*,
/// orderByLength, offset, limit, numOfContent*, infos*, lastUpdateId*).
fn cb_search_content(ctx: &mut CpuContext) -> HleResult {
    let (sets, sets_len, order, order_len) = (ctx.rdi(), ctx.rsi() as u32, ctx.rdx(), ctx.rcx() as u32);
    let limit = ctx.r9() as u32;
    let (count_out, infos, update_out) = (stack_arg(ctx, 0), stack_arg(ctx, 1), stack_arg(ctx, 2));
    if let Err(e) = search_initialized() {
        return e;
    }
    if count_out == 0 || update_out == 0 || (infos == 0 && limit != 0) {
        return CONTENT_SEARCH_ERROR_INVALID_PARAM;
    }
    if !column_sets_valid(ctx, sets, sets_len) || !order_by_valid(ctx, order, order_len) {
        return CONTENT_SEARCH_ERROR_INVALID_PARAM;
    }
    if limit > MAX_SEARCH_LIMIT {
        return CONTENT_SEARCH_ERROR_LIMIT_TOO_BIG;
    }
    if !ctx.write_u64(count_out, 0) || !ctx.write_u64(update_out, LIBRARY_UPDATE_ID) {
        return CONTENT_SEARCH_ERROR_INVALID_PARAM;
    }
    OK
}

/// sceContentSearchGetNumOfContent / GetTotalContentSize(columnSet*,
/// columnSetLength, value*, lastUpdateId*): 0 contents, 0 bytes.
fn cb_count_content(ctx: &mut CpuContext) -> HleResult {
    let (sets, sets_len, value_out, update_out) = (ctx.rdi(), ctx.rsi() as u32, ctx.rdx(), ctx.rcx());
    if let Err(e) = search_initialized() {
        return e;
    }
    if value_out == 0 || update_out == 0 || !column_sets_valid(ctx, sets, sets_len) {
        return CONTENT_SEARCH_ERROR_INVALID_PARAM;
    }
    if !ctx.write_u64(value_out, 0) || !ctx.write_u64(update_out, LIBRARY_UPDATE_ID) {
        return CONTENT_SEARCH_ERROR_INVALID_PARAM;
    }
    OK
}

/// sceContentSearchGetContentLastUpdateId(lastUpdateId*).
fn cb_get_last_update_id(ctx: &mut CpuContext) -> HleResult {
    let out = ctx.rdi();
    if let Err(e) = search_initialized() {
        return e;
    }
    if out == 0 || !ctx.write_u64(out, LIBRARY_UPDATE_ID) {
        return CONTENT_SEARCH_ERROR_INVALID_PARAM;
    }
    OK
}

fn cb_content_search_term(_ctx: &mut CpuContext) -> HleResult {
    if SEARCH_INITIALIZED.swap(false, Ordering::AcqRel) { OK } else { CONTENT_SEARCH_ERROR_NOT_INITIALIZED }
}

// ---------------------------------------------------------------------------
// Metadata and deletion. The content library is empty, so no path or
// content id is in it and no metadata id is ever handed out. The console's
// error for an item that is not in the library is not known: those calls
// are logged and refused with INVALID_PARAM.
// ---------------------------------------------------------------------------

/// sceContentSearchOpenMetadata(const char* path, int32* metadataId)
fn cb_open_metadata(ctx: &mut CpuContext) -> HleResult {
    let (path, out) = (ctx.rdi(), ctx.rsi());
    if let Err(e) = search_initialized() {
        return e;
    }
    if path == 0 || out == 0 {
        return CONTENT_SEARCH_ERROR_INVALID_PARAM;
    }
    let p = ctx.read_cstring(path, 1024).unwrap_or_default();
    log::warn!("[contentsearch] OpenMetadata('{p}'): not in the (empty) content library");
    CONTENT_SEARCH_ERROR_INVALID_PARAM
}

/// sceContentSearchOpenMetadataByContentId(int64 contentId, int32* metadataId)
fn cb_open_metadata_by_content_id(ctx: &mut CpuContext) -> HleResult {
    let (id, out) = (ctx.rdi() as i64, ctx.rsi());
    if let Err(e) = search_initialized() {
        return e;
    }
    if out == 0 {
        return CONTENT_SEARCH_ERROR_INVALID_PARAM;
    }
    log::warn!("[contentsearch] OpenMetadataByContentId({id}): not in the (empty) content library");
    CONTENT_SEARCH_ERROR_INVALID_PARAM
}

/// sceContentSearchCloseMetadata(int32 metadataId)
fn cb_close_metadata(ctx: &mut CpuContext) -> HleResult {
    if let Err(e) = search_initialized() {
        return e;
    }
    log::warn!("[contentsearch] CloseMetadata({}): no metadata id was handed out", ctx.rdi() as i32);
    CONTENT_SEARCH_ERROR_INVALID_PARAM
}

/// sceContentSearchGetMetadataFieldInfo(int32 id, const char* field,
/// int32* type, int32* size)
fn cb_get_metadata_field_info(ctx: &mut CpuContext) -> HleResult {
    if let Err(e) = search_initialized() {
        return e;
    }
    if ctx.rsi() == 0 || ctx.rdx() == 0 || ctx.rcx() == 0 {
        return CONTENT_SEARCH_ERROR_INVALID_PARAM;
    }
    log::warn!("[contentsearch] GetMetadataFieldInfo({}): no metadata id was handed out", ctx.rdi() as i32);
    CONTENT_SEARCH_ERROR_INVALID_PARAM
}

/// sceContentSearchGetMetadataValue(int32 id, const char* field, void* value)
fn cb_get_metadata_value(ctx: &mut CpuContext) -> HleResult {
    if let Err(e) = search_initialized() {
        return e;
    }
    if ctx.rsi() == 0 || ctx.rdx() == 0 {
        return CONTENT_SEARCH_ERROR_INVALID_PARAM;
    }
    log::warn!("[contentsearch] GetMetadataValue({}): no metadata id was handed out", ctx.rdi() as i32);
    CONTENT_SEARCH_ERROR_INVALID_PARAM
}

/// sceContentDeleteByPath(const char* path): a NULL path is INVALID_PARAM;
/// before sceContentDeleteInitialize, or for a path (the library is
/// empty), the call is refused the same way and logged.
fn cb_content_delete_by_path(ctx: &mut CpuContext) -> HleResult {
    let path = ctx.rdi();
    if path == 0 {
        return CONTENT_DELETE_ERROR_INVALID_PARAM;
    }
    let p = ctx.read_cstring(path, 1024).unwrap_or_default();
    if !DELETE_INITIALIZED.load(Ordering::Acquire) {
        log::warn!("[contentdelete] DeleteByPath('{p}') before sceContentDeleteInitialize");
    } else {
        log::warn!("[contentdelete] DeleteByPath('{p}'): not in the (empty) content library");
    }
    CONTENT_DELETE_ERROR_INVALID_PARAM
}

/// sceContentDeleteById(int64 contentId): as DeleteByPath.
fn cb_content_delete_by_id(ctx: &mut CpuContext) -> HleResult {
    let id = ctx.rdi() as i64;
    if !DELETE_INITIALIZED.load(Ordering::Acquire) {
        log::warn!("[contentdelete] DeleteById({id}) before sceContentDeleteInitialize");
    } else {
        log::warn!("[contentdelete] DeleteById({id}): not in the (empty) content library");
    }
    CONTENT_DELETE_ERROR_INVALID_PARAM
}

fn cb_content_delete_terminate(_ctx: &mut CpuContext) -> HleResult {
    DELETE_INITIALIZED.store(false, Ordering::Release);
    OK
}

const EXPORTS: &[(&str, &str, &str, HleCallback)] = &[
    ("dPj4ZtRcIWk", "sceContentSearchInit", "libSceContentSearch", cb_content_search_init),
    ("TEW3IKxYfXc", "sceContentSearchSearchContent", "libSceContentSearch", cb_search_content),
    ("o-RBPV0qr8c", "sceContentSearchGetNumOfContent", "libSceContentSearch", cb_count_content),
    ("i2ZMZKwZpWs", "sceContentSearchGetTotalContentSize", "libSceContentSearch", cb_count_content),
    ("PD5bnBTsv6Y", "sceContentSearchGetContentLastUpdateId", "libSceContentSearch", cb_get_last_update_id),
    ("1xSZodB2geA", "sceContentSearchTerm", "libSceContentSearch", cb_content_search_term),
    ("zoxb0wEChEM", "sceContentDeleteInitialize", "libSceContentDelete", cb_content_delete_initialize),
    ("5XLSih32qHA", "sceContentDeleteTerminate", "libSceContentDelete", cb_content_delete_terminate),
    ("G13HF1z-e4o", "sceContentSearchOpenMetadata", "libSceContentSearch", cb_open_metadata),
    ("bjAlYWwRTJA", "sceContentSearchOpenMetadataByContentId", "libSceContentSearch", cb_open_metadata_by_content_id),
    ("-YbpaF0XS-I", "sceContentSearchCloseMetadata", "libSceContentSearch", cb_close_metadata),
    ("EbNufIY0Zvc", "sceContentSearchGetMetadataFieldInfo", "libSceContentSearch", cb_get_metadata_field_info),
    ("ruNe-FgCzO8", "sceContentSearchGetMetadataValue", "libSceContentSearch", cb_get_metadata_value),
    ("qC3QW+AltHs", "sceContentDeleteByPath", "libSceContentDelete", cb_content_delete_by_path),
    ("pXJh3aVk8Ks", "sceContentDeleteById", "libSceContentDelete", cb_content_delete_by_id),
];

pub fn register(reg: &mut HleRegistry) {
    for &(nid, name, lib, cb) in EXPORTS {
        reg.register_fn(nid, name, lib, cb);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ps5vm::{MemProtect, VirtualMemory};

    #[test]
    fn nids_match_names() {
        for &(nid, name, _, _) in EXPORTS {
            assert_eq!(ps5nid::compute_nid(name), nid, "NID mismatch for {name}");
        }
    }

    #[test]
    fn init_requires_params() {
        const BASE: u64 = 0x5CC9_0000_0000;
        let vm = VirtualMemory::new();
        vm.map(BASE, 0x1000, MemProtect::RW).unwrap();
        let mut ctx = CpuContext::new(&vm);
        ctx.set(crate::context::CpuRegister::Rdi, 0);
        assert_eq!(cb_content_search_init(&mut ctx), CONTENT_SEARCH_ERROR_INVALID_PARAM);
        assert_eq!(cb_content_delete_initialize(&mut ctx), CONTENT_DELETE_ERROR_INVALID_PARAM);
        ctx.set(crate::context::CpuRegister::Rdi, BASE);
        assert_eq!(cb_content_search_init(&mut ctx), OK);
        assert_eq!(cb_content_delete_initialize(&mut ctx), OK);
        assert!(SEARCH_INITIALIZED.load(Ordering::Acquire) && DELETE_INITIALIZED.load(Ordering::Acquire));
        metadata_and_delete(&mut ctx, BASE);
        search_after_init(&mut ctx, BASE);
        assert_eq!(cb_content_delete_terminate(&mut ctx), OK);
        assert!(!DELETE_INITIALIZED.load(Ordering::Acquire));
    }

    /// Nothing is in the library: metadata cannot be opened, ids were never
    /// handed out, nothing can be deleted; NULL outputs are INVALID_PARAM.
    fn metadata_and_delete(ctx: &mut CpuContext, base: u64) {
        use crate::context::CpuRegister::*;
        let (path, out) = (base + 0x900, base + 0x980);
        ctx.write_bytes(path, b"/user/av_contents/photo/x.jpg\0");
        let set = |ctx: &mut CpuContext, a: &[u64]| {
            for (r, v) in [Rdi, Rsi, Rdx, Rcx].iter().zip(a) {
                ctx.set(*r, *v);
            }
        };
        set(ctx, &[path, 0]);
        assert_eq!(cb_open_metadata(ctx), CONTENT_SEARCH_ERROR_INVALID_PARAM);
        set(ctx, &[path, out]);
        assert_eq!(cb_open_metadata(ctx), CONTENT_SEARCH_ERROR_INVALID_PARAM);
        set(ctx, &[7, out]);
        assert_eq!(cb_open_metadata_by_content_id(ctx), CONTENT_SEARCH_ERROR_INVALID_PARAM);
        set(ctx, &[1]);
        assert_eq!(cb_close_metadata(ctx), CONTENT_SEARCH_ERROR_INVALID_PARAM);
        set(ctx, &[1, path, out, out + 4]);
        assert_eq!(cb_get_metadata_field_info(ctx), CONTENT_SEARCH_ERROR_INVALID_PARAM);
        set(ctx, &[1, path, out]);
        assert_eq!(cb_get_metadata_value(ctx), CONTENT_SEARCH_ERROR_INVALID_PARAM);
        set(ctx, &[0]);
        assert_eq!(cb_content_delete_by_path(ctx), CONTENT_DELETE_ERROR_INVALID_PARAM);
        set(ctx, &[path]);
        assert_eq!(cb_content_delete_by_path(ctx), CONTENT_DELETE_ERROR_INVALID_PARAM);
        set(ctx, &[5]);
        assert_eq!(cb_content_delete_by_id(ctx), CONTENT_DELETE_ERROR_INVALID_PARAM);
    }

    /// An empty library: searches and counts find nothing, the update id
    /// is 0, bad columns / sort conditions and a too big limit are refused.
    fn search_after_init(ctx: &mut CpuContext, base: u64) {
        use crate::context::CpuRegister::*;
        let (set, column, value, order) = (base + 0x100, base + 0x140, base + 0x180, base + 0x1C0);
        let (count, update, infos, stack) = (base + 0x200, base + 0x208, base + 0x300, base + 0x800);
        ctx.write_u64(set, column);
        ctx.write_u32(set + 8, 1);
        ctx.write_u32(set + 12, 0);
        ctx.write_u32(column, 1);
        ctx.write_u32(column + 4, 0);
        ctx.write_u64(column + 8, value);
        ctx.write_u32(column + 16, 0);
        ctx.write_u32(order, 4);
        ctx.write_u32(order + 4, 1);
        ctx.write_u64(stack + 8, count);
        ctx.write_u64(stack + 16, infos);
        ctx.write_u64(stack + 24, update);
        let search = |ctx: &mut CpuContext, limit: u64| {
            ctx.write_u64(count, 0x55);
            ctx.write_u64(update, 0x55);
            ctx.set(Rdi, set);
            ctx.set(Rsi, 1);
            ctx.set(Rdx, order);
            ctx.set(Rcx, 1);
            ctx.set(R8, 0);
            ctx.set(R9, limit);
            ctx.set(Rsp, stack);
            cb_search_content(ctx)
        };
        assert_eq!(search(ctx, 10), OK);
        assert_eq!((ctx.read_u64(count), ctx.read_u64(update)), (Some(0), Some(0)));
        assert_eq!(search(ctx, 92), OK);
        assert_eq!(search(ctx, 93), CONTENT_SEARCH_ERROR_LIMIT_TOO_BIG);
        ctx.write_u32(column, 8);
        assert_eq!(search(ctx, 10), CONTENT_SEARCH_ERROR_INVALID_PARAM, "column type 8 is not a content column");
        ctx.write_u32(column, 1);
        ctx.write_u32(order, 2);
        assert_eq!(search(ctx, 10), CONTENT_SEARCH_ERROR_INVALID_PARAM, "column 2 does not sort");
        ctx.write_u32(order, 4);

        ctx.set(Rdi, set);
        ctx.set(Rsi, 1);
        ctx.set(Rdx, count);
        ctx.set(Rcx, update);
        ctx.write_u64(count, 0x55);
        assert_eq!(cb_count_content(ctx), OK);
        assert_eq!(ctx.read_u64(count), Some(0));
        ctx.set(Rdx, 0);
        assert_eq!(cb_count_content(ctx), CONTENT_SEARCH_ERROR_INVALID_PARAM);
        ctx.set(Rdi, update);
        assert_eq!(cb_get_last_update_id(ctx), OK);

        assert_eq!(cb_content_search_term(ctx), OK);
        assert_eq!(cb_content_search_term(ctx), CONTENT_SEARCH_ERROR_NOT_INITIALIZED);
        assert_eq!(search(ctx, 10), CONTENT_SEARCH_ERROR_NOT_INITIALIZED);
        assert_eq!(cb_get_last_update_id(ctx), CONTENT_SEARCH_ERROR_NOT_INITIALIZED);
    }
}
