// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! libSceNpUniversalDataSystem — offline telemetry ("UDS") object model.
//!
//! Games build telemetry events out of property objects/arrays and post them
//! to PSN. Offline there is no server, but the full object model is kept so
//! that handle validation, error codes and the memory statistics behave like
//! on hardware. Posted events are snapshotted into a bounded local queue.
//!
//! Posted events also drive trophies: `_UnlockTrophy` (Int32 property
//! `_trophy_id`) unlocks a trophy, and any other event carrying integer
//! properties is matched against the title's UDS stat rules (see
//! `np_trophy2`). An event copies its initial property object.
//!
//! Nodes live in an arena keyed by internal ids; property handles, event roots
//! and posted snapshots reference nodes, and a node may be shared by several
//! parents (setting an existing handle as a child links, it does not copy).
//! Unreachable nodes are swept when handles are destroyed.

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::sync::Mutex;

use crate::context::CpuContext;
use crate::np_trophy2::package::UdsInteger;
use crate::np_trophy2::TrophyKey;
use crate::registry::{HleRegistry, HleResult};

const LIB: &str = "libSceNpUniversalDataSystem";

const ERR_INVALID_ARGUMENT: HleResult = 0x8055_3102u32 as i32 as i64;
const ERR_INVALID_CONTEXT: HleResult = 0x8055_3104u32 as i32 as i64;
const ERR_CONTEXT_EXCEEDS_MAX: HleResult = 0x8055_3105u32 as i32 as i64;
const ERR_INVALID_USER: HleResult = 0x8055_310Au32 as i32 as i64;
const ERR_NOT_REGISTERED: HleResult = 0x8055_3120u32 as i32 as i64;
const ERR_ALREADY_REGISTERED: HleResult = 0x8055_3121u32 as i32 as i64;

const MAX_CONTEXTS: usize = 8;
/// Upper bound for binary property payloads copied from the guest.
const MAX_BINARY_BYTES: usize = 1 << 20;

const MAX_GUEST_STRING_BYTES: usize = 64 * 1024;
const MAX_POSTED_EVENTS: usize = 64;
const EVENT_HANDLE_BASE: u64 = 0xE000_0000_0000_0000;
const OBJECT_HANDLE_BASE: u64 = 0xD000_0000_0000_0000;
const ARRAY_HANDLE_BASE: u64 = 0xC000_0000_0000_0000;

type NodeId = u64;

/// A typed property value. Object/Array values reference arena nodes.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum UdsValue {
    String(String),
    Int32(i32),
    UInt32(u32),
    Int64(i64),
    UInt64(u64),
    Float32(f32),
    Float64(f64),
    Bool(bool),
    Binary(Vec<u8>),
    Object(NodeId),
    Array(NodeId),
}

#[derive(Clone, Debug)]
enum Node {
    /// Insertion-ordered key/value pairs; re-setting a key replaces in place.
    Object(Vec<(String, UdsValue)>),
    Array(Vec<UdsValue>),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct UdsMemoryStat {
    pub pool_size: u64,
    pub max_in_use: u64,
    pub current_in_use: u64,
}

struct UdsContext {
    user_id: i32,
    service_label: u32,
    registered: bool,
}

struct UdsEvent {
    name: String,
    root: NodeId,
}

/// Trophy work requested by a posted event, applied after the UDS lock is
/// released.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum TrophyTrigger {
    None,
    /// `_UnlockTrophy` with `_trophy_id`.
    Unlock { key: TrophyKey, trophy_id: i32 },
    /// Event with integer properties, matched against the UDS stat rules.
    Stats { key: TrophyKey, event: String, properties: BTreeMap<String, UdsInteger> },
}

pub(crate) struct UdsState {
    pool_size: u64,
    max_in_use: u64,
    next_context: i32,
    next_event: u64,
    next_object: u64,
    next_array: u64,
    next_node: NodeId,
    contexts: BTreeMap<i32, UdsContext>,
    events: BTreeMap<u64, UdsEvent>,
    objects: BTreeMap<u64, NodeId>,
    arrays: BTreeMap<u64, NodeId>,
    posted: VecDeque<(String, NodeId)>,
    nodes: BTreeMap<NodeId, Node>,
}

fn next_positive_id(v: &mut i32) -> i32 {
    *v = if *v == i32::MAX { 1 } else { *v + 1 };
    *v
}

impl UdsState {
    pub(crate) const fn new() -> Self {
        Self {
            pool_size: 0,
            max_in_use: 0,
            next_context: 0,
            next_event: 0,
            next_object: 0,
            next_array: 0,
            next_node: 0,
            contexts: BTreeMap::new(),
            events: BTreeMap::new(),
            objects: BTreeMap::new(),
            arrays: BTreeMap::new(),
            posted: VecDeque::new(),
            nodes: BTreeMap::new(),
        }
    }

    /// Records the pool size reported by the memory statistics. Existing
    /// contexts and objects are kept.
    pub(crate) fn initialize(&mut self, pool_size: u64) {
        self.pool_size = pool_size;
    }

    pub(crate) fn create_context(&mut self, user_id: i32, service_label: u32, options: u64) -> Result<i32, HleResult> {
        if options != 0 {
            return Err(ERR_INVALID_ARGUMENT);
        }
        if user_id < 0 {
            return Err(ERR_INVALID_USER);
        }
        if self.contexts.len() == MAX_CONTEXTS {
            return Err(ERR_CONTEXT_EXCEEDS_MAX);
        }
        let id = next_positive_id(&mut self.next_context);
        self.contexts.insert(id, UdsContext { user_id, service_label, registered: false });
        self.update_max();
        Ok(id)
    }

    pub(crate) fn destroy_context(&mut self, id: i32) -> HleResult {
        if self.contexts.remove(&id).is_some() {
            0
        } else {
            ERR_INVALID_CONTEXT
        }
    }

    pub(crate) fn register_context(&mut self, context: i32, options: u64) -> HleResult {
        let Some(ctx) = self.contexts.get_mut(&context) else {
            return ERR_INVALID_CONTEXT;
        };
        if options != 0 {
            return ERR_INVALID_ARGUMENT;
        }
        if ctx.registered {
            return ERR_ALREADY_REGISTERED;
        }
        ctx.registered = true;
        self.update_max();
        0
    }

    fn alloc_node(&mut self, node: Node) -> NodeId {
        self.next_node += 1;
        self.nodes.insert(self.next_node, node);
        self.next_node
    }

    fn object_handle_for(&mut self, node: NodeId) -> u64 {
        self.next_object += 1;
        let handle = OBJECT_HANDLE_BASE | self.next_object;
        self.objects.insert(handle, node);
        handle
    }

    fn array_handle_for(&mut self, node: NodeId) -> u64 {
        self.next_array += 1;
        let handle = ARRAY_HANDLE_BASE | self.next_array;
        self.arrays.insert(handle, node);
        handle
    }

    pub(crate) fn create_object(&mut self) -> Option<u64> {
        let node = self.alloc_node(Node::Object(Vec::new()));
        let handle = self.object_handle_for(node);
        self.update_max();
        Some(handle)
    }

    pub(crate) fn destroy_object(&mut self, handle: u64) -> bool {
        let ok = self.objects.remove(&handle).is_some();
        if ok {
            self.collect_garbage();
        }
        ok
    }

    pub(crate) fn create_array(&mut self) -> Option<u64> {
        let node = self.alloc_node(Node::Array(Vec::new()));
        let handle = self.array_handle_for(node);
        self.update_max();
        Some(handle)
    }

    pub(crate) fn destroy_array(&mut self, handle: u64) -> bool {
        let ok = self.arrays.remove(&handle).is_some();
        if ok {
            self.collect_garbage();
        }
        ok
    }

    pub(crate) fn set_object_value(&mut self, object: u64, key: &str, value: UdsValue) -> bool {
        let Some(&node) = self.objects.get(&object) else {
            return false;
        };
        self.object_insert(node, key, value);
        self.update_max();
        true
    }

    pub(crate) fn add_array_value(&mut self, array: u64, value: UdsValue) -> bool {
        let Some(&node) = self.arrays.get(&array) else {
            return false;
        };
        if let Some(Node::Array(values)) = self.nodes.get_mut(&node) {
            values.push(value);
        }
        self.update_max();
        true
    }

    fn object_insert(&mut self, node: NodeId, key: &str, value: UdsValue) {
        if let Some(Node::Object(entries)) = self.nodes.get_mut(&node) {
            match entries.iter_mut().find(|(k, _)| k == key) {
                Some(entry) => entry.1 = value,
                None => entries.push((key.to_string(), value)),
            }
        }
    }

    /// Resolve (or create) the child node for a SetObject/SetArray call.
    /// Returns the child value and the handle reported back to the caller.
    fn resolve_child(&mut self, child: u64, is_array: bool, create_handle: bool) -> Option<(UdsValue, u64)> {
        if child != 0 {
            let node = if is_array { self.arrays.get(&child) } else { self.objects.get(&child) };
            let node = *node?;
            let value = if is_array { UdsValue::Array(node) } else { UdsValue::Object(node) };
            return Some((value, child));
        }
        if is_array {
            let node = self.alloc_node(Node::Array(Vec::new()));
            let handle = if create_handle { self.array_handle_for(node) } else { 0 };
            Some((UdsValue::Array(node), handle))
        } else {
            let node = self.alloc_node(Node::Object(Vec::new()));
            let handle = if create_handle { self.object_handle_for(node) } else { 0 };
            Some((UdsValue::Object(node), handle))
        }
    }

    pub(crate) fn set_object_node(
        &mut self,
        object: u64,
        key: &str,
        child: u64,
        is_array: bool,
        create_handle: bool,
    ) -> Option<u64> {
        let parent = *self.objects.get(&object)?;
        let (value, handle) = self.resolve_child(child, is_array, create_handle)?;
        self.object_insert(parent, key, value);
        self.update_max();
        Some(handle)
    }

    pub(crate) fn add_array_node(&mut self, array: u64, child: u64, is_array: bool, create_handle: bool) -> Option<u64> {
        let parent = *self.arrays.get(&array)?;
        let (value, handle) = self.resolve_child(child, is_array, create_handle)?;
        if let Some(Node::Array(values)) = self.nodes.get_mut(&parent) {
            values.push(value);
        }
        self.update_max();
        Some(handle)
    }

    /// Creates an event whose properties are a copy of `property` (or
    /// empty). Returns (event handle, handle of the event's own property
    /// object when requested, else 0).
    pub(crate) fn create_event(&mut self, name: &str, property: u64, create_property_handle: bool) -> Option<(u64, u64)> {
        let root = if property != 0 {
            let source = *self.objects.get(&property)?;
            self.clone_node(source, &mut BTreeMap::new())
        } else {
            self.alloc_node(Node::Object(Vec::new()))
        };
        let prop_handle = if create_property_handle { self.object_handle_for(root) } else { 0 };
        self.next_event += 1;
        let handle = EVENT_HANDLE_BASE | self.next_event;
        self.events.insert(handle, UdsEvent { name: name.to_string(), root });
        self.update_max();
        Some((handle, prop_handle))
    }

    pub(crate) fn destroy_event(&mut self, handle: u64) -> bool {
        let ok = self.events.remove(&handle).is_some();
        if ok {
            self.collect_garbage();
        }
        ok
    }

    /// Integer properties of an object node (Int32/UInt32/UInt64 entries;
    /// other value types are not stat inputs).
    fn integer_properties(&self, node: NodeId) -> BTreeMap<String, UdsInteger> {
        let mut out = BTreeMap::new();
        if let Some(Node::Object(entries)) = self.nodes.get(&node) {
            for (key, value) in entries {
                let v = match value {
                    UdsValue::Int32(v) => UdsInteger::I32(*v),
                    UdsValue::UInt32(v) => UdsInteger::U32(*v),
                    UdsValue::UInt64(v) => UdsInteger::U64(*v),
                    _ => continue,
                };
                out.insert(key.clone(), v);
            }
        }
        out
    }

    /// Validates and records a posted event; returns the trophy work it
    /// triggers.
    pub(crate) fn post_event(&mut self, context: i32, event: u64) -> Result<TrophyTrigger, HleResult> {
        if event == 0 {
            return Err(ERR_INVALID_ARGUMENT);
        }
        let ctx = self.contexts.get(&context).ok_or(ERR_INVALID_CONTEXT)?;
        if !ctx.registered {
            return Err(ERR_NOT_REGISTERED);
        }
        let key = (ctx.user_id, ctx.service_label);
        let ev = self.events.get(&event).ok_or(ERR_INVALID_ARGUMENT)?;
        let (name, root) = (ev.name.clone(), ev.root);
        let properties = self.integer_properties(root);
        let trigger = if name == "_UnlockTrophy" {
            match properties.get("_trophy_id") {
                Some(&UdsInteger::I32(trophy_id)) if trophy_id >= 0 => TrophyTrigger::Unlock { key, trophy_id },
                _ => return Err(ERR_INVALID_ARGUMENT),
            }
        } else if !properties.is_empty() {
            TrophyTrigger::Stats { key, event: name.clone(), properties }
        } else {
            TrophyTrigger::None
        };
        log::debug!("[UDS] posted event: {name}");
        let mut memo = BTreeMap::new();
        let snapshot = self.clone_node(root, &mut memo);
        if self.posted.len() == MAX_POSTED_EVENTS {
            self.posted.pop_front();
        }
        self.posted.push_back((name, snapshot));
        self.collect_garbage();
        Ok(trigger)
    }


    pub(crate) fn memory_stat(&self) -> UdsMemoryStat {
        UdsMemoryStat {
            pool_size: self.pool_size,
            max_in_use: self.max_in_use,
            current_in_use: self.in_use(),
        }
    }

    /// Deep copy preserving sharing/cycles (memoised by source node).
    fn clone_node(&mut self, src: NodeId, memo: &mut BTreeMap<NodeId, NodeId>) -> NodeId {
        if let Some(&done) = memo.get(&src) {
            return done;
        }
        let Some(node) = self.nodes.get(&src).cloned() else {
            return src;
        };
        let dst = match &node {
            Node::Object(_) => self.alloc_node(Node::Object(Vec::new())),
            Node::Array(_) => self.alloc_node(Node::Array(Vec::new())),
        };
        memo.insert(src, dst);
        let copied = match node {
            Node::Object(entries) => Node::Object(
                entries.into_iter().map(|(k, v)| (k, self.clone_value(v, memo))).collect(),
            ),
            Node::Array(values) => Node::Array(values.into_iter().map(|v| self.clone_value(v, memo)).collect()),
        };
        self.nodes.insert(dst, copied);
        dst
    }

    fn clone_value(&mut self, v: UdsValue, memo: &mut BTreeMap<NodeId, NodeId>) -> UdsValue {
        match v {
            UdsValue::Object(n) => UdsValue::Object(self.clone_node(n, memo)),
            UdsValue::Array(n) => UdsValue::Array(self.clone_node(n, memo)),
            other => other,
        }
    }

    fn update_max(&mut self) {
        self.max_in_use = self.max_in_use.max(self.in_use());
    }

    /// Bookkeeping size of all live state (contexts, handles, events, and
    /// every reachable node counted once).
    fn in_use(&self) -> u64 {
        let mut size = (self.contexts.len() * 24 + self.events.len() * 32) as u64;
        let mut visited = BTreeSet::new();
        for &n in self.objects.values() {
            size += self.measure_node(n, &mut visited);
        }
        for &n in self.arrays.values() {
            size += self.measure_node(n, &mut visited);
        }
        for ev in self.events.values() {
            size += ev.name.len() as u64 + 1;
            size += self.measure_node(ev.root, &mut visited);
        }
        size += self.contexts.values().filter(|c| c.registered).count() as u64 * 16;
        size
    }

    fn measure_node(&self, id: NodeId, visited: &mut BTreeSet<NodeId>) -> u64 {
        if !visited.insert(id) {
            return 0;
        }
        match self.nodes.get(&id) {
            Some(Node::Object(entries)) => {
                24 + entries.iter().map(|(k, v)| k.len() as u64 + 1 + self.measure_value(v, visited)).sum::<u64>()
            }
            Some(Node::Array(values)) => 24 + values.iter().map(|v| self.measure_value(v, visited)).sum::<u64>(),
            None => 0,
        }
    }

    fn measure_value(&self, v: &UdsValue, visited: &mut BTreeSet<NodeId>) -> u64 {
        match v {
            UdsValue::String(s) => s.len() as u64 + 1,
            UdsValue::Binary(b) => b.len() as u64,
            UdsValue::Int32(_) | UdsValue::UInt32(_) | UdsValue::Float32(_) | UdsValue::Bool(_) => 4,
            UdsValue::Int64(_) | UdsValue::UInt64(_) | UdsValue::Float64(_) => 8,
            UdsValue::Object(n) | UdsValue::Array(n) => self.measure_node(*n, visited),
        }
    }

    /// Drop arena nodes no longer reachable from any handle, event or
    /// posted snapshot.
    fn collect_garbage(&mut self) {
        let mut live = BTreeSet::new();
        let mut stack: Vec<NodeId> = self
            .objects
            .values()
            .chain(self.arrays.values())
            .copied()
            .chain(self.events.values().map(|e| e.root))
            .chain(self.posted.iter().map(|p| p.1))
            .collect();
        while let Some(id) = stack.pop() {
            if !live.insert(id) {
                continue;
            }
            let children: Vec<&UdsValue> = match self.nodes.get(&id) {
                Some(Node::Object(entries)) => entries.iter().map(|(_, v)| v).collect(),
                Some(Node::Array(values)) => values.iter().collect(),
                None => Vec::new(),
            };
            for v in children {
                if let UdsValue::Object(n) | UdsValue::Array(n) = v {
                    stack.push(*n);
                }
            }
        }
        self.nodes.retain(|id, _| live.contains(id));
    }

    #[cfg(test)]
    fn posted_count(&self) -> usize {
        self.posted.len()
    }
}

static STATE: Mutex<UdsState> = Mutex::new(UdsState::new());

fn state() -> std::sync::MutexGuard<'static, UdsState> {
    STATE.lock().unwrap_or_else(|e| e.into_inner())
}

// ---------------------------------------------------------------------------
// Guest helpers
// ---------------------------------------------------------------------------

/// Read a NUL-terminated UTF-8 string of at most `capacity` bytes, in small
/// chunks so a string ending right before unmapped memory still reads. If no
/// terminator is found within `capacity`, the first `capacity` bytes are used.
pub(crate) fn read_guest_utf8(ctx: &CpuContext, addr: u64, capacity: usize) -> Option<String> {
    if addr == 0 || capacity == 0 {
        return None;
    }
    const CHUNK: usize = 128;
    let mut bytes = Vec::new();
    while bytes.len() < capacity {
        let chunk = CHUNK.min(capacity - bytes.len());
        let base = addr + bytes.len() as u64;
        let mut buf = vec![0u8; chunk];
        if ctx.vm.read(base, &mut buf).is_ok() {
            if let Some(t) = buf.iter().position(|&b| b == 0) {
                bytes.extend_from_slice(&buf[..t]);
                return Some(String::from_utf8_lossy(&bytes).into_owned());
            }
            bytes.extend_from_slice(&buf);
            continue;
        }
        for i in 0..chunk {
            let mut b = [0u8; 1];
            ctx.vm.read(base + i as u64, &mut b).ok()?;
            if b[0] == 0 {
                return Some(String::from_utf8_lossy(&bytes).into_owned());
            }
            bytes.push(b[0]);
        }
    }
    Some(String::from_utf8_lossy(&bytes).into_owned())
}

/// Null pointer -> invalid argument, unreadable -> memory fault.
fn read_string_arg(ctx: &CpuContext, addr: u64) -> Result<String, HleResult> {
    if addr == 0 {
        return Err(ERR_INVALID_ARGUMENT);
    }
    read_guest_utf8(ctx, addr, MAX_GUEST_STRING_BYTES).ok_or(ERR_INVALID_ARGUMENT)
}

fn write_i32(ctx: &CpuContext, addr: u64, v: i32) -> bool {
    ctx.write_bytes(addr, &v.to_le_bytes())
}

fn status(ok: bool) -> HleResult {
    if ok {
        0
    } else {
        ERR_INVALID_ARGUMENT
    }
}

// ---------------------------------------------------------------------------
// Library lifetime / contexts / handles
// ---------------------------------------------------------------------------

/// (const InitParam* {size_t size; size_t poolSize;})
fn cb_initialize(ctx: &mut CpuContext) -> HleResult {
    let param = ctx.rdi();
    if param == 0 {
        return ERR_INVALID_ARGUMENT;
    }
    let pool_size = ctx.read_u64(param + 8).unwrap_or(0);
    state().initialize(pool_size);
    0
}

fn cb_terminate(_ctx: &mut CpuContext) -> HleResult {
    0
}

/// (int* context, userId, serviceLabel, uint64 options)
fn cb_create_context(ctx: &mut CpuContext) -> HleResult {
    let out = ctx.rdi();
    if out == 0 {
        return ERR_INVALID_ARGUMENT;
    }
    let mut st = state();
    match st.create_context(ctx.rsi() as i32, ctx.rdx() as u32, ctx.rcx()) {
        Ok(id) => {
            if write_i32(ctx, out, id) {
                0
            } else {
                st.contexts.remove(&id);
                ERR_INVALID_ARGUMENT
            }
        }
        Err(e) => e,
    }
}

fn cb_destroy_context(ctx: &mut CpuContext) -> HleResult {
    state().destroy_context(ctx.rdi() as i32)
}

/// Service handles carry no state: every handle is 1.
fn cb_create_handle(ctx: &mut CpuContext) -> HleResult {
    let out = ctx.rdi();
    if out == 0 || !write_i32(ctx, out, 1) {
        return ERR_INVALID_ARGUMENT;
    }
    0
}

fn cb_destroy_handle(_ctx: &mut CpuContext) -> HleResult {
    0
}

fn cb_abort_handle(_ctx: &mut CpuContext) -> HleResult {
    0
}

/// (context, handle, uint64 options)
fn cb_register_context(ctx: &mut CpuContext) -> HleResult {
    state().register_context(ctx.rdi() as i32, ctx.rdx())
}

/// (context, StorageStat*): seven size_t counters, all zero offline.
fn cb_get_storage_stat(ctx: &mut CpuContext) -> HleResult {
    let out = ctx.rsi();
    if out == 0 || !ctx.write_bytes(out, &[0u8; 56]) {
        return ERR_INVALID_ARGUMENT;
    }
    0
}

// ---------------------------------------------------------------------------
// Property objects / arrays / events
// ---------------------------------------------------------------------------

fn create_handle_out(ctx: &CpuContext, create: fn(&mut UdsState) -> Option<u64>) -> HleResult {
    let out = ctx.rdi();
    if out == 0 {
        return ERR_INVALID_ARGUMENT;
    }
    if !ctx.write_u64(out, 0) {
        return ERR_INVALID_ARGUMENT;
    }
    let Some(handle) = create(&mut state()) else {
        return ERR_INVALID_ARGUMENT;
    };
    if ctx.write_u64(out, handle) {
        0
    } else {
        ERR_INVALID_ARGUMENT
    }
}

/// Destroying a null handle is a no-op.
fn destroy_handle(handle: u64, destroy: fn(&mut UdsState, u64) -> bool) -> HleResult {
    if handle == 0 {
        return 0;
    }
    status(destroy(&mut state(), handle))
}

fn cb_create_property_object(ctx: &mut CpuContext) -> HleResult {
    create_handle_out(ctx, UdsState::create_object)
}

fn cb_destroy_property_object(ctx: &mut CpuContext) -> HleResult {
    destroy_handle(ctx.rdi(), UdsState::destroy_object)
}

fn cb_create_property_array(ctx: &mut CpuContext) -> HleResult {
    create_handle_out(ctx, UdsState::create_array)
}

fn cb_destroy_property_array(ctx: &mut CpuContext) -> HleResult {
    destroy_handle(ctx.rdi(), UdsState::destroy_array)
}

/// (const char* name, const PropertyObject* prop, Event** event, PropertyObject** propPtr)
fn cb_create_event(ctx: &mut CpuContext) -> HleResult {
    let (event_out, prop_out) = (ctx.rdx(), ctx.rcx());
    if ctx.rdi() == 0 || event_out == 0 {
        return ERR_INVALID_ARGUMENT;
    }
    let name = match read_string_arg(ctx, ctx.rdi()) {
        Ok(s) => s,
        Err(e) => return e,
    };
    if !ctx.write_u64(event_out, 0) || (prop_out != 0 && !ctx.write_u64(prop_out, 0)) {
        return ERR_INVALID_ARGUMENT;
    }
    let Some((event, prop)) = state().create_event(&name, ctx.rsi(), prop_out != 0) else {
        return ERR_INVALID_ARGUMENT;
    };
    if !ctx.write_u64(event_out, event) || (prop_out != 0 && !ctx.write_u64(prop_out, prop)) {
        return ERR_INVALID_ARGUMENT;
    }
    0
}

fn cb_destroy_event(ctx: &mut CpuContext) -> HleResult {
    destroy_handle(ctx.rdi(), UdsState::destroy_event)
}

/// (context, handle, const Event* event, uint64 options)
fn cb_post_event(ctx: &mut CpuContext) -> HleResult {
    let trigger = match state().post_event(ctx.rdi() as i32, ctx.rdx()) {
        Ok(t) => t,
        Err(e) => return e,
    };
    // The UDS lock is released before the trophy service runs.
    match trigger {
        TrophyTrigger::None => {}
        TrophyTrigger::Unlock { key, trophy_id } => crate::np_trophy2::uds_unlock_trophy(key, trophy_id),
        TrophyTrigger::Stats { key, event, properties } => {
            crate::np_trophy2::uds_stat_event(key, &event, &properties)
        }
    }
    0
}

/// Text form of an event (the payload is not rendered offline).
const EVENT_STRING: &[u8] = b"{}\0";

/// (const Event* event, size_t* size)
fn cb_event_estimate_size(ctx: &mut CpuContext) -> HleResult {
    let (event, size) = (ctx.rdi(), ctx.rsi());
    if event == 0 || size == 0 || !ctx.write_u64(size, EVENT_STRING.len() as u64) {
        return ERR_INVALID_ARGUMENT;
    }
    0
}

/// (const Event* event, char* buf, size_t bufSize, size_t* stringSize)
fn cb_event_to_string(ctx: &mut CpuContext) -> HleResult {
    let (event, buf, buf_size, string_size) = (ctx.rdi(), ctx.rsi(), ctx.rdx(), ctx.rcx());
    if event == 0 {
        return ERR_INVALID_ARGUMENT;
    }
    if string_size != 0 && !ctx.write_u64(string_size, EVENT_STRING.len() as u64) {
        return ERR_INVALID_ARGUMENT;
    }
    if buf != 0 && buf_size > 0 {
        let n = (EVENT_STRING.len() - 1).min(buf_size as usize - 1);
        let mut text = EVENT_STRING[..n].to_vec();
        text.push(0);
        if !ctx.write_bytes(buf, &text) {
            return ERR_INVALID_ARGUMENT;
        }
    }
    0
}

fn cb_get_memory_stat(ctx: &mut CpuContext) -> HleResult {
    let out = ctx.rdi();
    if out == 0 {
        return ERR_INVALID_ARGUMENT;
    }
    let stat = state().memory_stat();
    let mut buf = [0u8; 24];
    buf[0..8].copy_from_slice(&stat.pool_size.to_le_bytes());
    buf[8..16].copy_from_slice(&stat.max_in_use.to_le_bytes());
    buf[16..24].copy_from_slice(&stat.current_in_use.to_le_bytes());
    if ctx.write_bytes(out, &buf) {
        0
    } else {
        ERR_INVALID_ARGUMENT
    }
}

/// Binary payload (value, size): null value only with size 0.
fn read_binary(ctx: &CpuContext, value: u64, size: u64) -> Result<Vec<u8>, HleResult> {
    if size == 0 {
        return Ok(Vec::new());
    }
    if value == 0 || size as usize > MAX_BINARY_BYTES {
        return Err(ERR_INVALID_ARGUMENT);
    }
    ctx.read_bytes(value, size as usize).ok_or(ERR_INVALID_ARGUMENT)
}

// --- EventPropertyObjectSet*: (object, const char* key, value) -----------

fn set_object(ctx: &CpuContext, value: UdsValue) -> HleResult {
    let key = match read_string_arg(ctx, ctx.rsi()) {
        Ok(k) => k,
        Err(e) => return e,
    };
    status(state().set_object_value(ctx.rdi(), &key, value))
}

fn cb_object_set_string(ctx: &mut CpuContext) -> HleResult {
    let key = match read_string_arg(ctx, ctx.rsi()) {
        Ok(k) => k,
        Err(e) => return e,
    };
    let value = match read_string_arg(ctx, ctx.rdx()) {
        Ok(v) => v,
        Err(e) => return e,
    };
    status(state().set_object_value(ctx.rdi(), &key, UdsValue::String(value)))
}

fn cb_object_set_int32(ctx: &mut CpuContext) -> HleResult {
    set_object(ctx, UdsValue::Int32(ctx.rdx() as i32))
}

fn cb_object_set_uint32(ctx: &mut CpuContext) -> HleResult {
    set_object(ctx, UdsValue::UInt32(ctx.rdx() as u32))
}

fn cb_object_set_int64(ctx: &mut CpuContext) -> HleResult {
    set_object(ctx, UdsValue::Int64(ctx.rdx() as i64))
}

fn cb_object_set_uint64(ctx: &mut CpuContext) -> HleResult {
    set_object(ctx, UdsValue::UInt64(ctx.rdx()))
}

// Float arguments arrive in xmm0, which the HLE register snapshot does not
// capture; the value is telemetry-only (never read back), so the typed slot
// is recorded with a zero payload. Size accounting depends only on the type.
fn cb_object_set_float32(ctx: &mut CpuContext) -> HleResult {
    set_object(ctx, UdsValue::Float32(0.0))
}

fn cb_object_set_float64(ctx: &mut CpuContext) -> HleResult {
    set_object(ctx, UdsValue::Float64(0.0))
}

fn cb_object_set_bool(ctx: &mut CpuContext) -> HleResult {
    set_object(ctx, UdsValue::Bool(ctx.rdx() as u8 != 0))
}

/// (object, key, const void* value, size_t size)
fn cb_object_set_binary(ctx: &mut CpuContext) -> HleResult {
    if ctx.rdi() == 0 || ctx.rsi() == 0 {
        return ERR_INVALID_ARGUMENT;
    }
    match read_binary(ctx, ctx.rdx(), ctx.rcx()) {
        Ok(bytes) => set_object(ctx, UdsValue::Binary(bytes)),
        Err(e) => e,
    }
}

/// SetObject/SetArray: (object, key, child handle or 0, out child handle*).
fn set_object_node(ctx: &CpuContext, is_array: bool) -> HleResult {
    let key = match read_string_arg(ctx, ctx.rsi()) {
        Ok(k) => k,
        Err(e) => return e,
    };
    let out = ctx.rcx();
    if out != 0 && !ctx.write_u64(out, 0) {
        return ERR_INVALID_ARGUMENT;
    }
    let Some(handle) = state().set_object_node(ctx.rdi(), &key, ctx.rdx(), is_array, out != 0) else {
        return ERR_INVALID_ARGUMENT;
    };
    if out == 0 || ctx.write_u64(out, handle) {
        0
    } else {
        ERR_INVALID_ARGUMENT
    }
}

fn cb_object_set_object(ctx: &mut CpuContext) -> HleResult {
    set_object_node(ctx, false)
}

fn cb_object_set_array(ctx: &mut CpuContext) -> HleResult {
    set_object_node(ctx, true)
}

// --- EventPropertyArraySet*: (array, value) — appends --------------------

fn add_array(ctx: &CpuContext, value: UdsValue) -> HleResult {
    status(state().add_array_value(ctx.rdi(), value))
}

fn cb_array_set_string(ctx: &mut CpuContext) -> HleResult {
    let value = match read_string_arg(ctx, ctx.rsi()) {
        Ok(v) => v,
        Err(e) => return e,
    };
    add_array(ctx, UdsValue::String(value))
}

fn cb_array_set_int32(ctx: &mut CpuContext) -> HleResult {
    add_array(ctx, UdsValue::Int32(ctx.rsi() as i32))
}

fn cb_array_set_uint32(ctx: &mut CpuContext) -> HleResult {
    add_array(ctx, UdsValue::UInt32(ctx.rsi() as u32))
}

fn cb_array_set_int64(ctx: &mut CpuContext) -> HleResult {
    add_array(ctx, UdsValue::Int64(ctx.rsi() as i64))
}

fn cb_array_set_uint64(ctx: &mut CpuContext) -> HleResult {
    add_array(ctx, UdsValue::UInt64(ctx.rsi()))
}

// See the note on the object float setters (xmm0 not captured).
fn cb_array_set_float32(ctx: &mut CpuContext) -> HleResult {
    add_array(ctx, UdsValue::Float32(0.0))
}

fn cb_array_set_float64(ctx: &mut CpuContext) -> HleResult {
    add_array(ctx, UdsValue::Float64(0.0))
}

fn cb_array_set_bool(ctx: &mut CpuContext) -> HleResult {
    add_array(ctx, UdsValue::Bool(ctx.rsi() as u8 != 0))
}

/// (array, const void* value, size_t size)
fn cb_array_set_binary(ctx: &mut CpuContext) -> HleResult {
    if ctx.rdi() == 0 {
        return ERR_INVALID_ARGUMENT;
    }
    match read_binary(ctx, ctx.rsi(), ctx.rdx()) {
        Ok(bytes) => add_array(ctx, UdsValue::Binary(bytes)),
        Err(e) => e,
    }
}

/// SetObject/SetArray on arrays: (array, child handle or 0, out child handle*).
fn add_array_node(ctx: &CpuContext, is_array: bool) -> HleResult {
    let out = ctx.rdx();
    if out != 0 && !ctx.write_u64(out, 0) {
        return ERR_INVALID_ARGUMENT;
    }
    let Some(handle) = state().add_array_node(ctx.rdi(), ctx.rsi(), is_array, out != 0) else {
        return ERR_INVALID_ARGUMENT;
    };
    if out == 0 || ctx.write_u64(out, handle) {
        0
    } else {
        ERR_INVALID_ARGUMENT
    }
}

fn cb_array_set_object(ctx: &mut CpuContext) -> HleResult {
    add_array_node(ctx, false)
}

fn cb_array_set_array(ctx: &mut CpuContext) -> HleResult {
    add_array_node(ctx, true)
}

type Export = (&'static str, &'static str, fn(&mut CpuContext) -> HleResult);

const EXPORTS: &[Export] = &[
    ("sjaobBgqeB4", "sceNpUniversalDataSystemInitialize", cb_initialize),
    ("47UAEuQl+iI", "sceNpUniversalDataSystemTerminate", cb_terminate),
    ("5zBnau1uIEo", "sceNpUniversalDataSystemCreateContext", cb_create_context),
    ("wB7IWzGp2v0", "sceNpUniversalDataSystemDestroyContext", cb_destroy_context),
    ("hT0IAEvN+M0", "sceNpUniversalDataSystemCreateHandle", cb_create_handle),
    ("AUIHb7jUX3I", "sceNpUniversalDataSystemDestroyHandle", cb_destroy_handle),
    ("jZCqWFgMehE", "sceNpUniversalDataSystemAbortHandle", cb_abort_handle),
    ("tpFJ8LIKvPw", "sceNpUniversalDataSystemRegisterContext", cb_register_context),
    ("KmN62tT4U8A", "sceNpUniversalDataSystemGetStorageStat", cb_get_storage_stat),
    ("su7jW3VDDb4", "sceNpUniversalDataSystemGetMemoryStat", cb_get_memory_stat),
    ("s6W4Zl4Slgk", "sceNpUniversalDataSystemCreateEventPropertyObject", cb_create_property_object),
    ("kKUH0Viib3c", "sceNpUniversalDataSystemDestroyEventPropertyObject", cb_destroy_property_object),
    ("Hm7qubT3b70", "sceNpUniversalDataSystemCreateEventPropertyArray", cb_create_property_array),
    ("W-0xwY0ZMjw", "sceNpUniversalDataSystemDestroyEventPropertyArray", cb_destroy_property_array),
    ("p+GcLqwpL9M", "sceNpUniversalDataSystemCreateEvent", cb_create_event),
    ("wG+84pnNIuo", "sceNpUniversalDataSystemDestroyEvent", cb_destroy_event),
    ("CzkKf7ahIyU", "sceNpUniversalDataSystemPostEvent", cb_post_event),
    ("+s14jq-KGYw", "sceNpUniversalDataSystemEventEstimateSize", cb_event_estimate_size),
    ("vj6CQGWtEBg", "sceNpUniversalDataSystemEventToString", cb_event_to_string),
    ("MfDb+4Nln64", "sceNpUniversalDataSystemEventPropertyObjectSetString", cb_object_set_string),
    ("YE4dbtbz6OE", "sceNpUniversalDataSystemEventPropertyObjectSetInt32", cb_object_set_int32),
    ("AzD4irAcKE4", "sceNpUniversalDataSystemEventPropertyObjectSetUInt32", cb_object_set_uint32),
    ("56QLTqx911s", "sceNpUniversalDataSystemEventPropertyObjectSetInt64", cb_object_set_int64),
    ("xvsP5Yz6FmY", "sceNpUniversalDataSystemEventPropertyObjectSetUInt64", cb_object_set_uint64),
    ("lbPlT4+QVcE", "sceNpUniversalDataSystemEventPropertyObjectSetFloat32", cb_object_set_float32),
    ("4Fu8tHW+u-k", "sceNpUniversalDataSystemEventPropertyObjectSetFloat64", cb_object_set_float64),
    ("Fidd8vWgyVE", "sceNpUniversalDataSystemEventPropertyObjectSetBool", cb_object_set_bool),
    ("wAcxBDLHj1M", "sceNpUniversalDataSystemEventPropertyObjectSetBinary", cb_object_set_binary),
    ("74ASEqxSnkM", "sceNpUniversalDataSystemEventPropertyObjectSetObject", cb_object_set_object),
    ("Wxbg5x3pTXA", "sceNpUniversalDataSystemEventPropertyObjectSetArray", cb_object_set_array),
    ("4llLk7YJRTE", "sceNpUniversalDataSystemEventPropertyArraySetString", cb_array_set_string),
    ("BypQuF113-k", "sceNpUniversalDataSystemEventPropertyArraySetInt32", cb_array_set_int32),
    ("yMi0xAOpmXM", "sceNpUniversalDataSystemEventPropertyArraySetUInt32", cb_array_set_uint32),
    ("viVXAwmmYrY", "sceNpUniversalDataSystemEventPropertyArraySetInt64", cb_array_set_int64),
    ("Qo9qR7v5zO4", "sceNpUniversalDataSystemEventPropertyArraySetUInt64", cb_array_set_uint64),
    ("JmgwKm96Lq4", "sceNpUniversalDataSystemEventPropertyArraySetFloat32", cb_array_set_float32),
    ("sbSYZLR5AiE", "sceNpUniversalDataSystemEventPropertyArraySetFloat64", cb_array_set_float64),
    ("0+l4QSWCM4E", "sceNpUniversalDataSystemEventPropertyArraySetBool", cb_array_set_bool),
    ("IEdUCV9j2Cw", "sceNpUniversalDataSystemEventPropertyArraySetBinary", cb_array_set_binary),
    ("XY14n3jNIpE", "sceNpUniversalDataSystemEventPropertyArraySetObject", cb_array_set_object),
    ("rdi9BAfDLq8", "sceNpUniversalDataSystemEventPropertyArraySetArray", cb_array_set_array),
];

pub fn register(reg: &mut HleRegistry) {
    for &(nid, name, cb) in EXPORTS {
        reg.register_fn(nid, name, LIB, cb);
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    fn init() -> UdsState {
        let mut s = UdsState::new();
        s.initialize(0x10000);
        s
    }

    fn registered(s: &mut UdsState, user: i32, label: u32) -> i32 {
        let ctx = s.create_context(user, label, 0).unwrap();
        assert_eq!(s.register_context(ctx, 0), 0);
        ctx
    }

    #[test]
    fn contexts_and_errors() {
        // No Initialize required.
        let mut s = UdsState::new();
        assert_eq!(s.create_context(1, 0, 1), Err(ERR_INVALID_ARGUMENT));
        assert_eq!(s.create_context(-1, 0, 0), Err(ERR_INVALID_USER));
        assert_eq!(s.create_context(1, 0, 0), Ok(1));
        assert_eq!(s.create_context(1, 0, 0), Ok(2));
        for _ in 0..6 {
            assert!(s.create_context(1, 0, 0).is_ok());
        }
        assert_eq!(s.create_context(1, 0, 0), Err(ERR_CONTEXT_EXCEEDS_MAX));
        assert_eq!(s.register_context(99, 0), ERR_INVALID_CONTEXT);
        assert_eq!(s.register_context(1, 5), ERR_INVALID_ARGUMENT);
        assert_eq!(s.register_context(1, 0), 0);
        assert_eq!(s.register_context(1, 0), ERR_ALREADY_REGISTERED);
        assert_eq!(s.destroy_context(1), 0);
        assert_eq!(s.destroy_context(1), ERR_INVALID_CONTEXT);
        assert!(s.create_object().is_some());
    }

    #[test]
    fn handle_bases() {
        let mut s = init();
        assert_eq!(s.create_object(), Some(OBJECT_HANDLE_BASE | 1));
        assert_eq!(s.create_array(), Some(ARRAY_HANDLE_BASE | 1));
        let (ev, prop) = s.create_event("boot", 0, true).unwrap();
        assert_eq!(ev, EVENT_HANDLE_BASE | 1);
        assert_eq!(prop, OBJECT_HANDLE_BASE | 2);
        let (_, prop) = s.create_event("x", 0, false).unwrap();
        assert_eq!(prop, 0);
    }

    #[test]
    fn typed_sets_and_sizes() {
        let mut s = init();
        let obj = s.create_object().unwrap();
        // object node: 24
        assert_eq!(s.memory_stat().current_in_use, 24);
        assert!(s.set_object_value(obj, "a", UdsValue::Int32(5))); // 1+1+4
        assert!(s.set_object_value(obj, "bb", UdsValue::UInt64(5))); // 2+1+8
        assert!(s.set_object_value(obj, "s", UdsValue::String("hey".into()))); // 1+1+4
        assert_eq!(s.memory_stat().current_in_use, 24 + 6 + 11 + 6);
        // Replacing a key keeps one entry.
        assert!(s.set_object_value(obj, "a", UdsValue::Float64(1.0)));
        assert_eq!(s.memory_stat().current_in_use, 24 + 10 + 11 + 6);
        assert!(!s.set_object_value(obj + 1, "a", UdsValue::Bool(true)));
        let arr = s.create_array().unwrap();
        assert!(s.add_array_value(arr, UdsValue::Bool(true)));
        assert!(s.add_array_value(arr, UdsValue::Bool(true)));
        assert!(!s.add_array_value(obj, UdsValue::Bool(true)));
        assert_eq!(s.memory_stat().current_in_use, 51 + 24 + 8);
        assert_eq!(s.memory_stat().max_in_use, 51 + 24 + 8);
        assert_eq!(s.memory_stat().pool_size, 0x10000);
        assert!(s.add_array_value(arr, UdsValue::Binary(vec![1, 2, 3])));
        assert_eq!(s.memory_stat().current_in_use, 51 + 24 + 8 + 3);
    }

    #[test]
    fn child_nodes_shared_and_created() {
        let mut s = init();
        let parent = s.create_object().unwrap();
        let child = s.create_array().unwrap();
        // Linking an existing array returns the same handle.
        assert_eq!(s.set_object_node(parent, "arr", child, true, true), Some(child));
        // Wrong kind is rejected.
        assert_eq!(s.set_object_node(parent, "x", child, false, true), None);
        assert!(s.add_array_value(child, UdsValue::Int64(1)));
        // Shared node is counted once: obj 24 + "arr"(4) + arr(24+8).
        assert_eq!(s.memory_stat().current_in_use, 24 + 4 + 32);
        // Fresh child with a returned handle.
        let h = s.set_object_node(parent, "o", 0, false, true).unwrap();
        assert_eq!(h, OBJECT_HANDLE_BASE | 2);
        assert!(s.set_object_value(h, "k", UdsValue::Bool(false)));
        // Fresh child without handle.
        assert_eq!(s.add_array_node(child, 0, false, false), Some(0));
        let ah = s.add_array_node(child, 0, true, true).unwrap();
        assert_eq!(ah, ARRAY_HANDLE_BASE | 2);
        // Destroying a handle keeps the node alive while still referenced.
        assert!(s.destroy_array(child));
        assert!(!s.destroy_array(child));
        assert!(s.add_array_value(ah, UdsValue::Bool(true)));
        assert_eq!(s.nodes.len(), 5);
        assert!(s.destroy_object(parent));
        // Only the nodes still owned by handles (h, ah) survive the sweep.
        assert_eq!(s.nodes.len(), 2);
        assert!(s.set_object_value(h, "k2", UdsValue::Bool(false)));
    }

    #[test]
    fn cycles_are_safe() {
        let mut s = init();
        let a = s.create_object().unwrap();
        let b = s.create_object().unwrap();
        assert!(s.set_object_node(a, "b", b, false, false).is_some());
        assert!(s.set_object_node(b, "a", a, false, false).is_some());
        let size = s.memory_stat().current_in_use;
        assert_eq!(size, 24 + 2 + 24 + 2);
        let ctx = registered(&mut s, 1, 0);
        let (ev, _) = s.create_event("cyc", a, false).unwrap();
        assert_eq!(s.post_event(ctx, ev), Ok(TrophyTrigger::None));
        assert!(s.destroy_object(a));
        assert!(s.destroy_object(b));
        assert!(s.destroy_event(ev));
        // Only the posted snapshot (two cloned nodes) remains.
        assert_eq!(s.nodes.len(), 2);
    }

    #[test]
    fn create_event_copies_properties() {
        let mut s = init();
        let obj = s.create_object().unwrap();
        assert!(s.set_object_value(obj, "n", UdsValue::Int32(1)));
        let (_, own) = s.create_event("e", obj, true).unwrap();
        assert_ne!(own, obj, "event exposes its own property object");
        assert!(s.set_object_value(obj, "later", UdsValue::Int32(2)));
        match s.nodes.get(&s.objects[&own]) {
            Some(Node::Object(e)) => assert_eq!(e.len(), 1),
            _ => panic!("event properties missing"),
        }
        assert_eq!(s.create_event("e", obj + 100, false), None);
    }

    #[test]
    fn post_requires_registration() {
        let mut s = init();
        let ctx = s.create_context(1, 0, 0).unwrap();
        let (ev, prop) = s.create_event("e", 0, true).unwrap();
        assert!(s.set_object_value(prop, "n", UdsValue::String("v".into())));
        assert_eq!(s.post_event(ctx, 0), Err(ERR_INVALID_ARGUMENT));
        assert_eq!(s.post_event(99, ev), Err(ERR_INVALID_CONTEXT));
        assert_eq!(s.post_event(ctx, ev), Err(ERR_NOT_REGISTERED));
        assert_eq!(s.register_context(ctx, 0), 0);
        assert!(s.post_event(ctx, ev).is_ok());
        assert_eq!(s.post_event(ctx, ev + 1), Err(ERR_INVALID_ARGUMENT));
        // Snapshot is independent of later edits.
        assert!(s.set_object_value(prop, "m", UdsValue::Int32(2)));
        let snap = s.posted.back().unwrap().1;
        match s.nodes.get(&snap) {
            Some(Node::Object(e)) => assert_eq!(e.len(), 1),
            _ => panic!("snapshot missing"),
        }
    }

    #[test]
    fn post_triggers_trophies() {
        let mut s = init();
        let ctx = registered(&mut s, 7, 3);
        let (ev, prop) = s.create_event("_UnlockTrophy", 0, true).unwrap();
        assert_eq!(s.post_event(ctx, ev), Err(ERR_INVALID_ARGUMENT), "missing _trophy_id");
        assert!(s.set_object_value(prop, "_trophy_id", UdsValue::UInt32(4)));
        assert_eq!(s.post_event(ctx, ev), Err(ERR_INVALID_ARGUMENT), "_trophy_id must be int32");
        assert!(s.set_object_value(prop, "_trophy_id", UdsValue::Int32(-1)));
        assert_eq!(s.post_event(ctx, ev), Err(ERR_INVALID_ARGUMENT));
        assert!(s.set_object_value(prop, "_trophy_id", UdsValue::Int32(4)));
        assert_eq!(s.post_event(ctx, ev), Ok(TrophyTrigger::Unlock { key: (7, 3), trophy_id: 4 }));

        let (ev, prop) = s.create_event("COLLECT", 0, true).unwrap();
        assert_eq!(s.post_event(ctx, ev), Ok(TrophyTrigger::None));
        assert!(s.set_object_value(prop, "label", UdsValue::String("x".into())));
        assert!(s.set_object_value(prop, "big", UdsValue::Int64(5)));
        assert_eq!(s.post_event(ctx, ev), Ok(TrophyTrigger::None), "only int32/uint32/uint64 are stat inputs");
        assert!(s.set_object_value(prop, "counter", UdsValue::UInt64(10)));
        assert!(s.set_object_value(prop, "seconds", UdsValue::UInt32(2)));
        let expected: BTreeMap<String, UdsInteger> =
            [("counter".to_string(), UdsInteger::U64(10)), ("seconds".to_string(), UdsInteger::U32(2))].into();
        assert_eq!(
            s.post_event(ctx, ev),
            Ok(TrophyTrigger::Stats { key: (7, 3), event: "COLLECT".into(), properties: expected })
        );
    }

    #[test]
    fn posted_queue_bounded() {
        let mut s = init();
        let ctx = registered(&mut s, 1, 0);
        let (ev, _) = s.create_event("e", 0, false).unwrap();
        for _ in 0..100 {
            assert!(s.post_event(ctx, ev).is_ok());
        }
        assert_eq!(s.posted_count(), MAX_POSTED_EVENTS);
        // Event root + 64 snapshots.
        assert_eq!(s.nodes.len(), 1 + MAX_POSTED_EVENTS);
    }

    #[test]
    fn event_and_context_accounting() {
        let mut s = init();
        let c = s.create_context(1, 2, 0).unwrap();
        assert_eq!(s.memory_stat().current_in_use, 24);
        assert_eq!(s.register_context(c, 0), 0);
        assert_eq!(s.memory_stat().current_in_use, 40);
        let _ = s.create_event("abc", 0, false).unwrap();
        // event 32 + name 4 + root object 24
        assert_eq!(s.memory_stat().current_in_use, 40 + 32 + 4 + 24);
    }

    #[test]
    fn nids_match_names() {
        for &(nid, name, _) in EXPORTS {
            assert_eq!(ps5nid::compute_nid(name), nid, "NID mismatch for {name}");
        }
        assert_eq!(EXPORTS.len(), 41);
    }

    #[test]
    fn positive_id_wraps() {
        let mut v = i32::MAX;
        assert_eq!(next_positive_id(&mut v), 1);
    }

    #[test]
    fn guest_entry_points() {
        use crate::context::CpuRegister::{Rcx, Rdi, Rdx, Rsi};
        const BASE: u64 = 0x5D22_0000_0000;
        let vm = ps5vm::VirtualMemory::new();
        vm.map(BASE, 0x1000, ps5vm::MemProtect::RW).unwrap();
        let mut ctx = CpuContext::new(&vm);
        // Service handles are always 1; destroy/abort accept anything.
        ctx.set(Rdi, BASE);
        assert_eq!(cb_create_handle(&mut ctx), 0);
        assert_eq!(ctx.read_u32(BASE), Some(1));
        ctx.set(Rdi, 0);
        assert_eq!(cb_create_handle(&mut ctx), ERR_INVALID_ARGUMENT);
        ctx.set(Rdi, 12345);
        assert_eq!(cb_destroy_handle(&mut ctx), 0);
        assert_eq!(cb_abort_handle(&mut ctx), 0);
        // Null destroys are no-ops.
        ctx.set(Rdi, 0);
        assert_eq!(cb_destroy_event(&mut ctx), 0);
        assert_eq!(cb_destroy_property_object(&mut ctx), 0);
        assert_eq!(cb_destroy_property_array(&mut ctx), 0);
        assert_eq!(cb_initialize(&mut ctx), ERR_INVALID_ARGUMENT);
        // EstimateSize / ToString.
        ctx.set(Rsi, BASE + 0x10);
        assert_eq!(cb_event_estimate_size(&mut ctx), ERR_INVALID_ARGUMENT);
        ctx.set(Rdi, EVENT_HANDLE_BASE | 1);
        assert_eq!(cb_event_estimate_size(&mut ctx), 0);
        assert_eq!(ctx.read_u64(BASE + 0x10), Some(3));
        ctx.write_bytes(BASE + 0x20, &[0xAA; 8]);
        ctx.set(Rsi, BASE + 0x20);
        ctx.set(Rdx, 2);
        ctx.set(Rcx, BASE + 0x30);
        assert_eq!(cb_event_to_string(&mut ctx), 0);
        assert_eq!(ctx.read_bytes(BASE + 0x20, 3).unwrap(), b"{\0\xAA");
        assert_eq!(ctx.read_u64(BASE + 0x30), Some(3));
        ctx.set(Rdx, 16);
        assert_eq!(cb_event_to_string(&mut ctx), 0);
        assert_eq!(ctx.read_bytes(BASE + 0x20, 3).unwrap(), b"{}\0");
        ctx.set(Rdi, 0);
        assert_eq!(cb_event_to_string(&mut ctx), ERR_INVALID_ARGUMENT);
        // GetStorageStat zeroes seven counters.
        ctx.write_bytes(BASE + 0x100, &[0xFF; 64]);
        ctx.set(Rdi, 1);
        ctx.set(Rsi, BASE + 0x100);
        assert_eq!(cb_get_storage_stat(&mut ctx), 0);
        assert_eq!(ctx.read_bytes(BASE + 0x100, 56).unwrap(), vec![0u8; 56]);
        assert_eq!(ctx.read_bytes(BASE + 0x138, 1).unwrap(), vec![0xFF]);
        ctx.set(Rsi, 0);
        assert_eq!(cb_get_storage_stat(&mut ctx), ERR_INVALID_ARGUMENT);
        // Binary payloads: null value only with size 0.
        assert_eq!(read_binary(&ctx, 0, 0), Ok(Vec::new()));
        assert_eq!(read_binary(&ctx, 0, 4), Err(ERR_INVALID_ARGUMENT));
        assert_eq!(read_binary(&ctx, BASE + 0x20, 2), Ok(b"{}".to_vec()));
        ctx.set(Rdi, 0);
        assert_eq!(cb_array_set_binary(&mut ctx), ERR_INVALID_ARGUMENT);
        assert_eq!(cb_object_set_binary(&mut ctx), ERR_INVALID_ARGUMENT);
    }
}
