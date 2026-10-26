// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! libSceJson2: the `sce::Json` C++ API.
//!
//! Object model
//! ------------
//! * `sce::Json::Value` is 0x20 bytes in guest memory and is the source of
//!   truth for its own type and payload: parent Value at +0, root parameter
//!   at +8, the payload union at +0x10 (bool byte / int64 / uint64 / double
//!   bits / String* / Array* / Object*) and the raw type (0 null, 1 bool,
//!   2 int, 3 uint, 4 real, 5 string, 6 array, 7 object; other values are
//!   kept as given) at +0x1C. Scalar getters return a pointer to +0x10.
//! * `sce::Json::String`, `Array` and `Object` are opaque guest objects
//!   whose contents live host-side, keyed by the object's address: string
//!   bytes, element Value addresses, and key -> member block. A member block
//!   is the guest `pair<const String, Value>`: key String at +0, Value at +8.
//! * Values, strings and containers created by the library (parse results,
//!   copies, typed constructors) are allocated guest objects owned here and
//!   released when their owning Value is cleared.
//! * Parsing is RFC 8259 JSON: optional UTF-8 BOM, a NUL byte ends the
//!   input, non-negative integers are unsigned, negative ones signed, and
//!   integers that do not fit (and fractions/exponents) are reals; objects
//!   keep the last of duplicate keys and iterate in key order.
//! * Serialisation is compact, writes strings and keys verbatim and reals
//!   in `%f` form.

#[cfg(windows)]
#[allow(unused_imports)]
use crate::host::c as libc;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::{LazyLock, Mutex, MutexGuard};

use crate::context::CpuContext;
use crate::registry::{HleCallback, HleRegistry, HleResult};
/// Host-side document model used by the emulator's own JSON readers.
pub(crate) use crate::json_doc::{parse_json, JsonElem, JsonKind};

const LIB: &str = "libSceJson";

const VALUE_SIZE: usize = 0x20;
const VALUE_PAYLOAD_OFFSET: u64 = 0x10;
const VALUE_TYPE_OFFSET: u64 = 0x1C;
const INIT_PARAMETER2_SIZE: usize = 0x28;
const INIT_PARAMETER2_FLOAT_FORMAT_OFFSET: u64 = 0x18;
/// Offset of the Value inside an object member block (after the key String).
const MEMBER_VALUE_OFFSET: u64 = 8;
/// Longest C string argument read from the guest.
const MAX_CSTRING: usize = 16 * 1024 * 1024;
/// Nesting bound for the recursive host-side walks.
const MAX_DEPTH: usize = 1024;

const TYPE_NULL: u32 = 0;
const TYPE_BOOLEAN: u32 = 1;
const TYPE_INTEGER: u32 = 2;
const TYPE_UINTEGER: u32 = 3;
const TYPE_REAL: u32 = 4;
const TYPE_STRING: u32 = 5;
const TYPE_ARRAY: u32 = 6;
const TYPE_OBJECT: u32 = 7;

const ERR_JSON_PARSE_INVALID_CHAR: HleResult = 0x8084_8101u32 as i32 as i64;
const ERR_JSON_INVALID_ARGUMENT: HleResult = 0x8084_8120u32 as i32 as i64;

// ---------------------------------------------------------------------------
// Host-side value trees (parse results and copies in flight)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
enum Tree {
    Null,
    Boolean(bool),
    Integer(i64),
    UInteger(u64),
    Real(f64),
    String(Vec<u8>),
    Array(Vec<Tree>),
    Object(BTreeMap<Vec<u8>, Tree>),
    /// A type outside 0..=7 (payload zero).
    Other(u32),
}

struct Parser<'a> {
    s: &'a [u8],
    pos: usize,
}

impl Parser<'_> {
    fn peek(&self) -> Option<u8> {
        self.s.get(self.pos).copied()
    }

    fn ws(&mut self) {
        while matches!(self.peek(), Some(b' ' | b'\t' | b'\n' | b'\r')) {
            self.pos += 1;
        }
    }

    fn literal(&mut self, word: &[u8], v: Tree) -> Option<Tree> {
        if self.s[self.pos..].starts_with(word) {
            self.pos += word.len();
            Some(v)
        } else {
            None
        }
    }

    fn value(&mut self, depth: usize) -> Option<Tree> {
        self.ws();
        match self.peek()? {
            b'n' => self.literal(b"null", Tree::Null),
            b't' => self.literal(b"true", Tree::Boolean(true)),
            b'f' => self.literal(b"false", Tree::Boolean(false)),
            b'"' => self.string().map(Tree::String),
            b'[' => {
                if depth >= MAX_DEPTH {
                    return None;
                }
                self.pos += 1;
                let mut items = Vec::new();
                self.ws();
                if self.peek() == Some(b']') {
                    self.pos += 1;
                    return Some(Tree::Array(items));
                }
                loop {
                    items.push(self.value(depth + 1)?);
                    self.ws();
                    match self.peek()? {
                        b',' => self.pos += 1,
                        b']' => {
                            self.pos += 1;
                            return Some(Tree::Array(items));
                        }
                        _ => return None,
                    }
                }
            }
            b'{' => {
                if depth >= MAX_DEPTH {
                    return None;
                }
                self.pos += 1;
                let mut members = BTreeMap::new();
                self.ws();
                if self.peek() == Some(b'}') {
                    self.pos += 1;
                    return Some(Tree::Object(members));
                }
                loop {
                    self.ws();
                    if self.peek() != Some(b'"') {
                        return None;
                    }
                    let key = self.string()?;
                    self.ws();
                    if self.peek() != Some(b':') {
                        return None;
                    }
                    self.pos += 1;
                    let v = self.value(depth + 1)?;
                    // The last of duplicate keys wins.
                    members.insert(key, v);
                    self.ws();
                    match self.peek()? {
                        b',' => self.pos += 1,
                        b'}' => {
                            self.pos += 1;
                            return Some(Tree::Object(members));
                        }
                        _ => return None,
                    }
                }
            }
            b'-' | b'0'..=b'9' => self.number(),
            _ => None,
        }
    }

    fn digits(&mut self) -> usize {
        let start = self.pos;
        while matches!(self.peek(), Some(b'0'..=b'9')) {
            self.pos += 1;
        }
        self.pos - start
    }

    fn number(&mut self) -> Option<Tree> {
        let start = self.pos;
        let negative = self.peek() == Some(b'-');
        if negative {
            self.pos += 1;
        }
        match self.peek()? {
            b'0' => self.pos += 1,
            b'1'..=b'9' => {
                self.digits();
            }
            _ => return None,
        }
        let mut integral = true;
        if self.peek() == Some(b'.') {
            integral = false;
            self.pos += 1;
            if self.digits() == 0 {
                return None;
            }
        }
        if matches!(self.peek(), Some(b'e' | b'E')) {
            integral = false;
            self.pos += 1;
            if matches!(self.peek(), Some(b'+' | b'-')) {
                self.pos += 1;
            }
            if self.digits() == 0 {
                return None;
            }
        }
        let text = std::str::from_utf8(&self.s[start..self.pos]).ok()?;
        if integral {
            if negative {
                if let Ok(v) = text.parse::<i64>() {
                    return Some(Tree::Integer(v));
                }
            } else if let Ok(v) = text.parse::<u64>() {
                return Some(Tree::UInteger(v));
            }
        }
        let v: f64 = text.parse().ok()?;
        v.is_finite().then_some(Tree::Real(v))
    }

    fn hex4(&mut self) -> Option<u32> {
        let h = self.s.get(self.pos..self.pos + 4)?;
        if !h.iter().all(u8::is_ascii_hexdigit) {
            return None;
        }
        self.pos += 4;
        u32::from_str_radix(std::str::from_utf8(h).ok()?, 16).ok()
    }

    /// String token at `pos` (the opening quote) decoded to UTF-8 bytes.
    fn string(&mut self) -> Option<Vec<u8>> {
        self.pos += 1;
        let mut out = Vec::new();
        loop {
            let c = self.peek()?;
            match c {
                b'"' => {
                    self.pos += 1;
                    return Some(out);
                }
                b'\\' => {
                    self.pos += 1;
                    let e = self.peek()?;
                    self.pos += 1;
                    let ch = match e {
                        b'"' => '"',
                        b'\\' => '\\',
                        b'/' => '/',
                        b'b' => '\u{8}',
                        b'f' => '\u{c}',
                        b'n' => '\n',
                        b'r' => '\r',
                        b't' => '\t',
                        b'u' => {
                            let hi = self.hex4()?;
                            let cp = match hi {
                                0xD800..=0xDBFF => {
                                    if self.s.get(self.pos..self.pos + 2) != Some(b"\\u") {
                                        return None;
                                    }
                                    self.pos += 2;
                                    let lo = self.hex4()?;
                                    if !(0xDC00..=0xDFFF).contains(&lo) {
                                        return None;
                                    }
                                    0x10000 + ((hi - 0xD800) << 10) + (lo - 0xDC00)
                                }
                                0xDC00..=0xDFFF => return None,
                                cp => cp,
                            };
                            char::from_u32(cp)?
                        }
                        _ => return None,
                    };
                    let mut buf = [0u8; 4];
                    out.extend_from_slice(ch.encode_utf8(&mut buf).as_bytes());
                }
                0..=0x1F => return None,
                0x20..=0x7F => {
                    out.push(c);
                    self.pos += 1;
                }
                _ => {
                    let len = match c {
                        0xC2..=0xDF => 2,
                        0xE0..=0xEF => 3,
                        0xF0..=0xF4 => 4,
                        _ => return None,
                    };
                    let seq = self.s.get(self.pos..self.pos + len)?;
                    std::str::from_utf8(seq).ok()?;
                    out.extend_from_slice(seq);
                    self.pos += len;
                }
            }
        }
    }
}

/// Parse a JSON document; the input ends at its first NUL byte. None = invalid.
fn parse_document(bytes: &[u8]) -> Option<Tree> {
    let end = bytes.iter().position(|&b| b == 0).unwrap_or(bytes.len());
    let s = &bytes[..end];
    let mut p = Parser { s, pos: 0 };
    if s.first() == Some(&0xEF) {
        if !s.starts_with(&[0xEF, 0xBB, 0xBF]) {
            return None;
        }
        p.pos = 3;
    }
    let v = p.value(0)?;
    p.ws();
    (p.pos == s.len()).then_some(v)
}

/// `%f` text of a double.
fn format_real(v: f64) -> String {
    if v.is_nan() {
        if v.is_sign_negative() { "-nan" } else { "nan" }.to_string()
    } else if v.is_infinite() {
        if v < 0.0 { "-inf" } else { "inf" }.to_string()
    } else {
        format!("{v:.6}")
    }
}

/// Compact text of a tree: strings and keys verbatim, reals as `%f`.
fn serialize_tree(t: &Tree, out: &mut Vec<u8>) {
    match t {
        Tree::Null | Tree::Other(_) => out.extend_from_slice(b"null"),
        Tree::Boolean(b) => out.extend_from_slice(if *b { b"true" } else { b"false" }),
        Tree::Integer(i) => out.extend_from_slice(i.to_string().as_bytes()),
        Tree::UInteger(u) => out.extend_from_slice(u.to_string().as_bytes()),
        Tree::Real(r) => out.extend_from_slice(format_real(*r).as_bytes()),
        Tree::String(s) => {
            out.push(b'"');
            out.extend_from_slice(s);
            out.push(b'"');
        }
        Tree::Array(items) => {
            out.push(b'[');
            for (i, v) in items.iter().enumerate() {
                if i > 0 {
                    out.push(b',');
                }
                serialize_tree(v, out);
            }
            out.push(b']');
        }
        Tree::Object(members) => {
            out.push(b'{');
            for (i, (k, v)) in members.iter().enumerate() {
                if i > 0 {
                    out.push(b',');
                }
                out.push(b'"');
                out.extend_from_slice(k);
                out.extend_from_slice(b"\":");
                serialize_tree(v, out);
            }
            out.push(b'}');
        }
    }
}

// ---------------------------------------------------------------------------
// Host-side object state
// ---------------------------------------------------------------------------

#[derive(Default)]
struct StringState {
    value: Vec<u8>,
    /// Guest buffer handed out by c_str() and its capacity.
    buffer: u64,
    capacity: usize,
}

/// Shared read-only objects returned for wrong-type accesses.
#[derive(Default)]
struct Statics {
    null_value: u64,
    empty_array: u64,
    empty_object: u64,
    empty_string: u64,
    /// Zeroed slot returned by the scalar getters of other types.
    zero: u64,
}

#[derive(Default)]
struct JsonState {
    strings: HashMap<u64, StringState>,
    /// `sce::Json::Array` at an address -> its element Values.
    arrays: HashMap<u64, Vec<u64>>,
    /// `sce::Json::Object` at an address -> key -> member block.
    objects: HashMap<u64, BTreeMap<Vec<u8>, u64>>,
    /// Container -> (guest table of element/member pointers, capacity) that
    /// iterators walk.
    iter_tables: HashMap<u64, (u64, usize)>,
    /// Guest objects allocated by this module (freed when released).
    owned: HashSet<u64>,
    statics: Statics,
    /// Value -> its per-Value handlers (setTypeMismatchHandler and the
    /// other two setters).
    handlers: HashMap<u64, Handlers>,
}

/// A Value's handlers: (function, context) pairs. A setter installs its
/// handler on the Value and every value it holds; a parse into the Value
/// gives every value it creates the Value's handlers; content changes keep
/// them; a copy does not carry them.
#[derive(Clone, Copy, Default, PartialEq, Debug)]
struct Handlers {
    type_mismatch: (u64, u64),
    element_access_failure: (u64, u64),
    special_float: (u64, u64),
}

static STATE: LazyLock<Mutex<JsonState>> = LazyLock::new(|| Mutex::new(JsonState::default()));

fn state() -> MutexGuard<'static, JsonState> {
    STATE.lock().unwrap_or_else(|p| p.into_inner())
}

fn new_owned(st: &mut JsonState, ctx: &CpuContext) -> Option<u64> {
    let a = crate::pthread::alloc_object(ctx)?;
    st.owned.insert(a);
    Some(a)
}

fn release(st: &mut JsonState, addr: u64) {
    if st.owned.remove(&addr) {
        st.iter_tables.remove(&addr);
        st.handlers.remove(&addr);
        crate::pthread::free_object(addr);
    }
}

/// (type, payload) of the guest Value at `v`.
fn read_head(ctx: &CpuContext, v: u64) -> Option<(u32, u64)> {
    if v == 0 {
        return None;
    }
    let payload = ctx.read_u64(v + VALUE_PAYLOAD_OFFSET)?;
    let ty = ctx.read_u32(v + VALUE_TYPE_OFFSET)?;
    Some((ty, payload))
}

fn value_type(ctx: &CpuContext, v: u64) -> u32 {
    read_head(ctx, v).map_or(TYPE_NULL, |(t, _)| t)
}

/// Payload of the Value at `v` when it has type `ty`.
fn payload_if(ctx: &CpuContext, v: u64, ty: u32) -> Option<u64> {
    match read_head(ctx, v)? {
        (t, p) if t == ty => Some(p),
        _ => None,
    }
}

fn write_head(ctx: &CpuContext, v: u64, ty: u32, payload: u64) {
    ctx.write_u64(v + VALUE_PAYLOAD_OFFSET, payload);
    ctx.write_u32(v + VALUE_TYPE_OFFSET, ty);
}

/// Value initialisation: the whole object zeroed, type null.
fn init_value(ctx: &CpuContext, v: u64) {
    if v != 0 {
        ctx.write_bytes(v, &[0u8; VALUE_SIZE]);
    }
}

/// Release what the Value at `v` holds and reset it to null; the parent and
/// root links are reset too unless `keep_links`.
fn clear_value(st: &mut JsonState, ctx: &CpuContext, v: u64, keep_links: bool) {
    let Some((ty, p)) = read_head(ctx, v) else { return };
    match ty {
        TYPE_STRING if st.owned.contains(&p) => {
            st.strings.remove(&p);
            release(st, p);
        }
        TYPE_ARRAY if st.owned.contains(&p) => {
            destroy_array(st, ctx, p);
            release(st, p);
        }
        TYPE_OBJECT if st.owned.contains(&p) => {
            destroy_object(st, ctx, p);
            release(st, p);
        }
        _ => {}
    }
    if !keep_links {
        ctx.write_bytes(v, &[0u8; 0x10]);
    }
    write_head(ctx, v, TYPE_NULL, 0);
}

/// Free an Array's elements and forget it (the object itself stays).
fn destroy_array(st: &mut JsonState, ctx: &CpuContext, c: u64) {
    for e in st.arrays.remove(&c).unwrap_or_default() {
        clear_value(st, ctx, e, false);
        release(st, e);
    }
    st.iter_tables.remove(&c);
}

/// Free an Object's members and forget it (the object itself stays).
fn destroy_object(st: &mut JsonState, ctx: &CpuContext, c: u64) {
    for (_, m) in st.objects.remove(&c).unwrap_or_default() {
        drop_member(st, ctx, m);
    }
    st.iter_tables.remove(&c);
}

fn drop_member(st: &mut JsonState, ctx: &CpuContext, m: u64) {
    clear_value(st, ctx, m + MEMBER_VALUE_OFFSET, false);
    st.handlers.remove(&(m + MEMBER_VALUE_OFFSET));
    st.strings.remove(&m);
    release(st, m);
}

/// Deep host copy of the Value at `v`.
fn snapshot(st: &JsonState, ctx: &CpuContext, v: u64, depth: usize) -> Tree {
    let Some((ty, p)) = read_head(ctx, v) else { return Tree::Null };
    if depth > MAX_DEPTH {
        return Tree::Null;
    }
    match ty {
        TYPE_NULL => Tree::Null,
        TYPE_BOOLEAN => Tree::Boolean(p & 0xFF != 0),
        TYPE_INTEGER => Tree::Integer(p as i64),
        TYPE_UINTEGER => Tree::UInteger(p),
        TYPE_REAL => Tree::Real(f64::from_bits(p)),
        TYPE_STRING => Tree::String(st.strings.get(&p).map(|s| s.value.clone()).unwrap_or_default()),
        TYPE_ARRAY => Tree::Array(snapshot_array(st, ctx, p, depth)),
        TYPE_OBJECT => Tree::Object(snapshot_object(st, ctx, p, depth)),
        other => Tree::Other(other),
    }
}

fn snapshot_array(st: &JsonState, ctx: &CpuContext, c: u64, depth: usize) -> Vec<Tree> {
    st.arrays.get(&c).into_iter().flatten().map(|&e| snapshot(st, ctx, e, depth + 1)).collect()
}

fn snapshot_object(st: &JsonState, ctx: &CpuContext, c: u64, depth: usize) -> BTreeMap<Vec<u8>, Tree> {
    st.objects
        .get(&c)
        .into_iter()
        .flatten()
        .map(|(k, &m)| (k.clone(), snapshot(st, ctx, m + MEMBER_VALUE_OFFSET, depth + 1)))
        .collect()
}

/// Store `t` into the (null) Value at `v`; children get `v` as parent.
fn install(st: &mut JsonState, ctx: &CpuContext, v: u64, t: &Tree, parent: u64) {
    if v == 0 {
        return;
    }
    ctx.write_u64(v, parent);
    let (ty, payload) = match t {
        Tree::Null => (TYPE_NULL, 0),
        Tree::Boolean(b) => (TYPE_BOOLEAN, u64::from(*b)),
        Tree::Integer(i) => (TYPE_INTEGER, *i as u64),
        Tree::UInteger(u) => (TYPE_UINTEGER, *u),
        Tree::Real(r) => (TYPE_REAL, r.to_bits()),
        Tree::String(s) => match new_owned(st, ctx) {
            Some(a) => {
                st.strings.insert(a, StringState { value: s.clone(), ..Default::default() });
                (TYPE_STRING, a)
            }
            None => (TYPE_NULL, 0),
        },
        Tree::Array(items) => match new_owned(st, ctx) {
            Some(c) => {
                let elems = install_items(st, ctx, items, v);
                st.arrays.insert(c, elems);
                (TYPE_ARRAY, c)
            }
            None => (TYPE_NULL, 0),
        },
        Tree::Object(members) => match new_owned(st, ctx) {
            Some(c) => {
                let m = install_members(st, ctx, members, v);
                st.objects.insert(c, m);
                (TYPE_OBJECT, c)
            }
            None => (TYPE_NULL, 0),
        },
        Tree::Other(t) => (*t, 0),
    };
    write_head(ctx, v, ty, payload);
}

fn install_items(st: &mut JsonState, ctx: &CpuContext, items: &[Tree], parent: u64) -> Vec<u64> {
    let mut out = Vec::with_capacity(items.len());
    for it in items {
        let Some(e) = new_owned(st, ctx) else { break };
        install(st, ctx, e, it, parent);
        out.push(e);
    }
    out
}

fn install_members(
    st: &mut JsonState,
    ctx: &CpuContext,
    members: &BTreeMap<Vec<u8>, Tree>,
    parent: u64,
) -> BTreeMap<Vec<u8>, u64> {
    let mut out = BTreeMap::new();
    for (k, it) in members {
        let Some(m) = new_owned(st, ctx) else { break };
        st.strings.insert(m, StringState { value: k.clone(), ..Default::default() });
        install(st, ctx, m + MEMBER_VALUE_OFFSET, it, parent);
        out.insert(k.clone(), m);
    }
    out
}

/// Replace the Value at `v` with `t` (taken before the old payload goes).
fn replace_value(st: &mut JsonState, ctx: &CpuContext, v: u64, t: &Tree) {
    if v == 0 {
        return;
    }
    clear_value(st, ctx, v, false);
    install(st, ctx, v, t, 0);
}

/// Value assignment: deep copy of `src` into `dst`; nothing happens for a
/// NULL operand or self-assignment.
fn copy_value(st: &mut JsonState, ctx: &CpuContext, dst: u64, src: u64) {
    if dst == 0 || src == 0 || dst == src {
        return;
    }
    let t = snapshot(st, ctx, src, 0);
    replace_value(st, ctx, dst, &t);
}

/// Read-only shared object (allocated once, never released).
fn static_object(ctx: &CpuContext, slot: &mut u64) -> u64 {
    if *slot == 0 {
        *slot = crate::pthread::alloc_object(ctx).unwrap_or(0);
    }
    *slot
}

fn static_null_value(st: &mut JsonState, ctx: &CpuContext) -> u64 {
    let a = static_object(ctx, &mut st.statics.null_value);
    if a != 0 {
        ctx.write_u32(a + VALUE_TYPE_OFFSET, TYPE_NULL);
    }
    a
}

fn static_empty_array(st: &mut JsonState, ctx: &CpuContext) -> u64 {
    let a = static_object(ctx, &mut st.statics.empty_array);
    st.arrays.entry(a).or_default();
    a
}

fn static_empty_object(st: &mut JsonState, ctx: &CpuContext) -> u64 {
    let a = static_object(ctx, &mut st.statics.empty_object);
    st.objects.entry(a).or_default();
    a
}

fn static_empty_string(st: &mut JsonState, ctx: &CpuContext) -> u64 {
    let a = static_object(ctx, &mut st.statics.empty_string);
    st.strings.entry(a).or_default();
    a
}

fn static_zero(st: &mut JsonState, ctx: &CpuContext) -> u64 {
    static_object(ctx, &mut st.statics.zero)
}

/// Replace a String's text, keeping its c_str() buffer.
fn assign_string(st: &mut JsonState, addr: u64, text: Vec<u8>) {
    st.strings.entry(addr).or_default().value = text;
}

fn string_or_empty(st: &JsonState, addr: u64) -> Vec<u8> {
    st.strings.get(&addr).map(|s| s.value.clone()).unwrap_or_default()
}

/// NUL-terminated C string argument; NULL reads as empty.
fn read_text_arg(ctx: &CpuContext, addr: u64) -> Vec<u8> {
    ctx.read_cstr_bytes(addr, MAX_CSTRING).unwrap_or_default()
}

/// The double argument in xmm0 of the import being dispatched.
fn xmm0_f64() -> f64 {
    f64::from_bits(ps5cpu::exec::trap_xmm_low(0).unwrap_or(0))
}

/// Rewrite the guest pointer table iterators of `container` walk; returns
/// (table, entry count).
fn iter_table(st: &mut JsonState, ctx: &CpuContext, container: u64, entries: &[u64]) -> Option<(u64, usize)> {
    let n = entries.len();
    let table = match st.iter_tables.get(&container) {
        Some(&(t, cap)) if cap >= n.max(1) => t,
        _ => {
            let cap = n.max(4).next_power_of_two();
            let t = crate::pthread::alloc_guest_block(ctx, (cap * 8) as u64)?;
            st.iter_tables.insert(container, (t, cap));
            t
        }
    };
    let bytes: Vec<u8> = entries.iter().flat_map(|e| e.to_le_bytes()).collect();
    if !bytes.is_empty() && !ctx.write_bytes(table, &bytes) {
        return None;
    }
    Some((table, n))
}

/// The null access callback (`const Value& (*)(ValueType, const Value*
/// parent, void* context)`) and its context, set by
/// Initializer::setGlobalNullAccessCallback and Value::setNullAccessCallBack.
static NULL_ACCESS: Mutex<(u64, u64)> = Mutex::new((0, 0));

/// Ask `handler` (callback, context) for the value to hand out when a
/// `requested`-typed access on `parent` finds nothing; its answer is used
/// when it is non-NULL and of the requested type (any type for a null
/// request). `invoke` runs the guest callback.
fn null_access_via(
    ctx: &CpuContext,
    handler: (u64, u64),
    requested: u32,
    parent: u64,
    invoke: &mut dyn FnMut(u64, [u64; 6]) -> u64,
) -> Option<u64> {
    let (callback, context) = handler;
    if callback == 0 {
        return None;
    }
    let result = invoke(callback, [u64::from(requested), parent, context, 0, 0, 0]);
    (result != 0 && (requested == TYPE_NULL || value_type(ctx, result) == requested)).then_some(result)
}

/// The title's null access answer (see [`null_access_via`]); the state lock
/// must not be held, the callback is guest code.
fn null_access(ctx: &CpuContext, requested: u32, parent: u64) -> Option<u64> {
    let handler = *NULL_ACCESS.lock().unwrap_or_else(|p| p.into_inner());
    null_access_via(ctx, handler, requested, parent, &mut |f, args| ps5cpu::exec::call_guest_nested(f, args))
}

/// The answer for a typed getter on a Value of another type: a non-null
/// Value with a type mismatch handler asks it (`handler(requested, this,
/// NULL, context)`, an answer of another type counts as none); otherwise
/// the null access callback is asked. The state lock must not be held.
fn mismatch_or_null(ctx: &CpuContext, requested: u32, this: u64) -> Option<u64> {
    let handler = state().handlers.get(&this).map_or((0, 0), |h| h.type_mismatch);
    if handler.0 != 0 && value_type(ctx, this) != TYPE_NULL {
        return type_mismatch_via(ctx, handler, requested, this, &mut |f, args| ps5cpu::exec::call_guest_nested(f, args));
    }
    null_access(ctx, requested, this)
}

fn type_mismatch_via(
    ctx: &CpuContext,
    (callback, context): (u64, u64),
    requested: u32,
    this: u64,
    invoke: &mut dyn FnMut(u64, [u64; 6]) -> u64,
) -> Option<u64> {
    let result = invoke(callback, [u64::from(requested), this, 0, context, 0, 0]);
    (result != 0 && value_type(ctx, result) == requested).then_some(result)
}

/// Apply `f` to the handlers of the Value at `v` and of every value it holds.
fn set_handlers(st: &mut JsonState, ctx: &CpuContext, v: u64, depth: usize, f: &dyn Fn(&mut Handlers)) {
    if v == 0 || depth > MAX_DEPTH {
        return;
    }
    f(st.handlers.entry(v).or_default());
    let children: Vec<u64> = match read_head(ctx, v) {
        Some((TYPE_ARRAY, c)) => st.arrays.get(&c).cloned().unwrap_or_default(),
        Some((TYPE_OBJECT, c)) => st.objects.get(&c).map(|o| o.values().map(|m| m + MEMBER_VALUE_OFFSET).collect()).unwrap_or_default(),
        _ => Vec::new(),
    };
    for child in children {
        set_handlers(st, ctx, child, depth + 1, f);
    }
}

/// Value::setTypeMismatchHandler(handler, context): asked by the typed
/// getters on a value of another type.
fn cb_value_set_type_mismatch_handler(ctx: &mut CpuContext) -> HleResult {
    let (this, h) = (ctx.rdi(), (ctx.rsi(), ctx.rdx()));
    set_handlers(&mut state(), ctx, this, 0, &|x| x.type_mismatch = h);
    0
}

/// Value::setElementAccessFailureHandler(handler, context): kept; what the
/// console passes it is not known, so a missing element still answers as
/// before.
fn cb_value_set_element_access_failure_handler(ctx: &mut CpuContext) -> HleResult {
    let (this, h) = (ctx.rdi(), (ctx.rsi(), ctx.rdx()));
    set_handlers(&mut state(), ctx, this, 0, &|x| x.element_access_failure = h);
    0
}

/// Value::setSpecialFloatHandler(handler, context): kept; the function type
/// the console passes is not known, so serialisation is unchanged.
fn cb_value_set_special_float_handler(ctx: &mut CpuContext) -> HleResult {
    let (this, h) = (ctx.rdi(), (ctx.rsi(), ctx.rdx()));
    set_handlers(&mut state(), ctx, this, 0, &|x| x.special_float = h);
    0
}

/// The shared null value, or the title's null access answer.
fn missing_value(ctx: &CpuContext, parent: u64) -> HleResult {
    match null_access(ctx, TYPE_NULL, parent) {
        Some(v) => v as i64,
        None => static_null_value(&mut state(), ctx) as i64,
    }
}

// ---------------------------------------------------------------------------
// MemAllocator / Initializer / InitParameter2
// ---------------------------------------------------------------------------

fn cb_return_this(ctx: &mut CpuContext) -> HleResult {
    ctx.rdi() as i64
}

fn cb_return_void(_ctx: &mut CpuContext) -> HleResult {
    0
}

/// Initializer::initialize (both parameter versions) and
/// setGlobalSpecialFloatHandler: accepted, always 0.
fn cb_return_ok(_ctx: &mut CpuContext) -> HleResult {
    0
}

/// InitParameter2::InitParameter2(): zero the 0x28-byte object.
fn cb_init_parameter2_ctor(ctx: &mut CpuContext) -> HleResult {
    let this = ctx.rdi();
    if this != 0 {
        ctx.write_bytes(this, &[0u8; INIT_PARAMETER2_SIZE]);
    }
    this as i64
}

/// InitParameter2::setAllocator(MemAllocator*, void* userData)
fn cb_init_parameter2_set_allocator(ctx: &mut CpuContext) -> HleResult {
    let this = ctx.rdi();
    if this != 0 {
        ctx.write_u64(this, ctx.rsi());
        ctx.write_u64(this + 8, ctx.rdx());
    }
    0
}

/// InitParameter2::setFileBufferSize(size_t)
fn cb_init_parameter2_set_file_buffer_size(ctx: &mut CpuContext) -> HleResult {
    let this = ctx.rdi();
    if this != 0 {
        ctx.write_u64(this + 0x10, ctx.rsi());
    }
    0
}

/// Initializer::setGlobalNullAccessCallback(callback, context) (this in
/// rdi): kept for the getters; always 0.
fn cb_set_global_null_access_callback(ctx: &mut CpuContext) -> HleResult {
    *NULL_ACCESS.lock().unwrap_or_else(|p| p.into_inner()) = (ctx.rsi(), ctx.rdx());
    0
}

/// Initializer::terminate(): the null access callback is forgotten.
fn cb_initializer_terminate(_ctx: &mut CpuContext) -> HleResult {
    *NULL_ACCESS.lock().unwrap_or_else(|p| p.into_inner()) = (0, 0);
    0
}

/// static Value::setNullAccessCallBack(callback, context): the same
/// callback the Initializer sets.
fn cb_value_set_null_access_callback(ctx: &mut CpuContext) -> HleResult {
    *NULL_ACCESS.lock().unwrap_or_else(|p| p.into_inner()) = (ctx.rdi(), ctx.rsi());
    0
}

/// InitParameterRtti::InitParameterRtti(AllocParamRtti*, void* userData,
/// size_t fileBufferSize): the three fields in argument order.
fn cb_init_parameter_rtti_ctor(ctx: &mut CpuContext) -> HleResult {
    let this = ctx.rdi();
    if this != 0 {
        ctx.write_u64(this, ctx.rsi());
        ctx.write_u64(this + 8, ctx.rdx());
        ctx.write_u64(this + 0x10, ctx.rcx());
    }
    this as i64
}

/// MemAllocator::notifyError(int error, size_t size, void* userData): the
/// base class's virtual does nothing (the library never allocates through
/// the title's allocator, so it never reports a failure itself).
fn cb_mem_allocator_notify_error(_ctx: &mut CpuContext) -> HleResult {
    0
}

/// InitParameter2::setSpecialFloatFormatType(SpecialFloatFormatType)
fn cb_init_parameter2_set_special_float_format_type(ctx: &mut CpuContext) -> HleResult {
    let this = ctx.rdi();
    if this != 0 {
        ctx.write_u32(this + INIT_PARAMETER2_FLOAT_FORMAT_OFFSET, ctx.rsi() as u32);
    }
    0
}

// ---------------------------------------------------------------------------
// Parser
// ---------------------------------------------------------------------------

/// Parser::parse(Value&, const char* buf, size_t size)
fn cb_parser_parse(ctx: &mut CpuContext) -> HleResult {
    let (dst, src, size) = (ctx.rdi(), ctx.rsi(), ctx.rdx());
    if dst == 0 || src == 0 {
        return ERR_JSON_INVALID_ARGUMENT;
    }
    // The input ends at the first NUL byte or after `size` bytes.
    let max = usize::try_from(size).unwrap_or(usize::MAX);
    let Some(bytes) = ctx.read_cstr_bytes(src, max) else {
        return ERR_JSON_PARSE_INVALID_CHAR;
    };
    parse_into_value(ctx, dst, &bytes)
}

/// Parse `bytes` into the Value at `dst`, which keeps its value on failure;
/// the values created take `dst`'s handlers.
fn parse_into_value(ctx: &CpuContext, dst: u64, bytes: &[u8]) -> HleResult {
    match parse_document(bytes) {
        Some(tree) => {
            let mut st = state();
            replace_value(&mut st, ctx, dst, &tree);
            if let Some(h) = st.handlers.get(&dst).copied() {
                set_handlers(&mut st, ctx, dst, 0, &|x| *x = h);
            }
            0
        }
        None => ERR_JSON_PARSE_INVALID_CHAR,
    }
}

/// Characters a read callback may deliver before the document is refused.
const MAX_READ_CALLBACK_TEXT: usize = MAX_CSTRING;

/// The text a parse read callback delivers: `read` gives the callback's
/// result and character; characters are taken until it returns non-zero or
/// gives NUL. None when the text exceeds the input bound.
fn read_callback_text(mut read: impl FnMut() -> (i32, u8)) -> Option<Vec<u8>> {
    let mut text = Vec::new();
    loop {
        let (result, ch) = read();
        if result != 0 || ch == 0 {
            return Some(text);
        }
        if text.len() == MAX_READ_CALLBACK_TEXT {
            return None;
        }
        text.push(ch);
    }
}

/// Parser::parse(Value&, int (*read)(char&, void*), void* context): the
/// callback returns 0 with the next character and, at the end of its data,
/// writes NUL and returns -1. The text read is then parsed as by the
/// `(text, size)` overload.
fn cb_parser_parse_callback(ctx: &mut CpuContext) -> HleResult {
    let (dst, read, context) = (ctx.rdi(), ctx.rsi(), ctx.rdx());
    if dst == 0 || read == 0 {
        log::error!("[json] Parser::parse(value={dst:#x}, read={read:#x}): NULL argument");
        return ERR_JSON_INVALID_ARGUMENT;
    }
    let cell = crate::libc::heap_block(ctx, 8);
    if cell == 0 {
        return ERR_JSON_INVALID_ARGUMENT;
    }
    let text = read_callback_text(|| {
        ctx.write_u64(cell, 0);
        let result = ps5cpu::exec::call_guest_nested(read, [cell, context, 0, 0, 0, 0]) as i32;
        (result, ctx.read_bytes(cell, 1).map_or(0, |b| b[0]))
    });
    crate::libc::heap_release(cell);
    match text {
        Some(bytes) => parse_into_value(ctx, dst, &bytes),
        None => {
            log::error!("[json] Parser::parse: the read callback delivered more than {MAX_READ_CALLBACK_TEXT} bytes");
            ERR_JSON_PARSE_INVALID_CHAR
        }
    }
}

// ---------------------------------------------------------------------------
// Value
// ---------------------------------------------------------------------------

/// Value::Value(): null value.
fn cb_value_ctor(ctx: &mut CpuContext) -> HleResult {
    let this = ctx.rdi();
    init_value(ctx, this);
    state().handlers.remove(&this);
    this as i64
}

/// Value::~Value()
fn cb_value_dtor(ctx: &mut CpuContext) -> HleResult {
    let mut st = state();
    clear_value(&mut st, ctx, ctx.rdi(), false);
    st.handlers.remove(&ctx.rdi());
    0
}

/// Value::clear(): back to null, keeping the parent and root links.
fn cb_value_clear(ctx: &mut CpuContext) -> HleResult {
    clear_value(&mut state(), ctx, ctx.rdi(), true);
    0
}

/// Value::Value(const Value&)
fn cb_value_copy_ctor(ctx: &mut CpuContext) -> HleResult {
    let (this, src) = (ctx.rdi(), ctx.rsi());
    init_value(ctx, this);
    let mut st = state();
    st.handlers.remove(&this);
    copy_value(&mut st, ctx, this, src);
    this as i64
}

/// Value::operator=(const Value&) / set(const Value&)
fn cb_value_assign(ctx: &mut CpuContext) -> HleResult {
    let (this, src) = (ctx.rdi(), ctx.rsi());
    copy_value(&mut state(), ctx, this, src);
    this as i64
}

/// Initialise `this` (constructor) or clear it (setter), then store `t`.
fn ctor_or_set(ctx: &CpuContext, ctor: bool, t: Tree) -> HleResult {
    let this = ctx.rdi();
    if this != 0 {
        let mut st = state();
        if ctor {
            init_value(ctx, this);
        } else {
            clear_value(&mut st, ctx, this, false);
        }
        install(&mut st, ctx, this, &t, 0);
    }
    this as i64
}

/// Tree of the empty value of a raw type (Value(ValueType) / set(ValueType)).
fn empty_of_type(ty: u32) -> Tree {
    match ty {
        TYPE_NULL => Tree::Null,
        TYPE_BOOLEAN => Tree::Boolean(false),
        TYPE_INTEGER => Tree::Integer(0),
        TYPE_UINTEGER => Tree::UInteger(0),
        TYPE_REAL => Tree::Real(0.0),
        TYPE_STRING => Tree::String(Vec::new()),
        TYPE_ARRAY => Tree::Array(Vec::new()),
        TYPE_OBJECT => Tree::Object(BTreeMap::new()),
        other => Tree::Other(other),
    }
}

fn cb_value_bool_ctor(ctx: &mut CpuContext) -> HleResult {
    ctor_or_set(ctx, true, Tree::Boolean(ctx.rsi() & 0xFF != 0))
}
fn cb_value_int_ctor(ctx: &mut CpuContext) -> HleResult {
    ctor_or_set(ctx, true, Tree::Integer(ctx.rsi() as i64))
}
fn cb_value_uint_ctor(ctx: &mut CpuContext) -> HleResult {
    ctor_or_set(ctx, true, Tree::UInteger(ctx.rsi()))
}
fn cb_value_double_ctor(ctx: &mut CpuContext) -> HleResult {
    ctor_or_set(ctx, true, Tree::Real(xmm0_f64()))
}
fn cb_value_cstring_ctor(ctx: &mut CpuContext) -> HleResult {
    let text = read_text_arg(ctx, ctx.rsi());
    ctor_or_set(ctx, true, Tree::String(text))
}
fn cb_value_string_ctor(ctx: &mut CpuContext) -> HleResult {
    let text = string_or_empty(&state(), ctx.rsi());
    ctor_or_set(ctx, true, Tree::String(text))
}
fn cb_value_type_ctor(ctx: &mut CpuContext) -> HleResult {
    ctor_or_set(ctx, true, empty_of_type(ctx.rsi() as u32))
}
fn cb_value_set_bool(ctx: &mut CpuContext) -> HleResult {
    ctor_or_set(ctx, false, Tree::Boolean(ctx.rsi() & 0xFF != 0));
    0
}
fn cb_value_set_int(ctx: &mut CpuContext) -> HleResult {
    ctor_or_set(ctx, false, Tree::Integer(ctx.rsi() as i64));
    0
}
fn cb_value_set_uint(ctx: &mut CpuContext) -> HleResult {
    ctor_or_set(ctx, false, Tree::UInteger(ctx.rsi()));
    0
}
fn cb_value_set_double(ctx: &mut CpuContext) -> HleResult {
    ctor_or_set(ctx, false, Tree::Real(xmm0_f64()));
    0
}
fn cb_value_set_cstring(ctx: &mut CpuContext) -> HleResult {
    let text = read_text_arg(ctx, ctx.rsi());
    ctor_or_set(ctx, false, Tree::String(text));
    0
}
fn cb_value_set_string(ctx: &mut CpuContext) -> HleResult {
    let text = string_or_empty(&state(), ctx.rsi());
    ctor_or_set(ctx, false, Tree::String(text));
    0
}
fn cb_value_set_type(ctx: &mut CpuContext) -> HleResult {
    ctor_or_set(ctx, false, empty_of_type(ctx.rsi() as u32));
    0
}

/// Value::Value(const Array&): deep copy, elements parented to `this`.
fn cb_value_array_ctor(ctx: &mut CpuContext) -> HleResult {
    let items = snapshot_array(&state(), ctx, ctx.rsi(), 0);
    ctor_or_set(ctx, true, Tree::Array(items))
}

/// Value::Value(const Object&)
fn cb_value_object_ctor(ctx: &mut CpuContext) -> HleResult {
    let members = snapshot_object(&state(), ctx, ctx.rsi(), 0);
    ctor_or_set(ctx, true, Tree::Object(members))
}

/// Value::set(const Array&): the source may live inside this value.
fn cb_value_set_array(ctx: &mut CpuContext) -> HleResult {
    let items = snapshot_array(&state(), ctx, ctx.rsi(), 0);
    ctor_or_set(ctx, false, Tree::Array(items));
    0
}

/// Value::set(const Object&): the source may live inside this value.
fn cb_value_set_object(ctx: &mut CpuContext) -> HleResult {
    let members = snapshot_object(&state(), ctx, ctx.rsi(), 0);
    ctor_or_set(ctx, false, Tree::Object(members));
    0
}

fn cb_value_get_type(ctx: &mut CpuContext) -> HleResult {
    value_type(ctx, ctx.rdi()) as i64
}

/// Value::count(): elements of an array / members of an object, else 0.
fn cb_value_count(ctx: &mut CpuContext) -> HleResult {
    let st = state();
    match read_head(ctx, ctx.rdi()) {
        Some((TYPE_ARRAY, c)) => st.arrays.get(&c).map_or(0, Vec::len) as i32 as i64,
        Some((TYPE_OBJECT, c)) => st.objects.get(&c).map_or(0, BTreeMap::len) as i32 as i64,
        _ => 0,
    }
}

/// Value::referArray(): the value's own array, NULL for other types.
fn cb_value_refer_array(ctx: &mut CpuContext) -> HleResult {
    payload_if(ctx, ctx.rdi(), TYPE_ARRAY).unwrap_or(0) as i64
}

/// Value::referObject(): the value's own object, NULL for other types.
fn cb_value_refer_object(ctx: &mut CpuContext) -> HleResult {
    payload_if(ctx, ctx.rdi(), TYPE_OBJECT).unwrap_or(0) as i64
}

/// Value::getArray() const: the title's null access answer or a shared
/// empty array for other types.
fn cb_value_get_array(ctx: &mut CpuContext) -> HleResult {
    let this = ctx.rdi();
    if let Some(c) = payload_if(ctx, this, TYPE_ARRAY).or_else(|| mismatch_or_null(ctx, TYPE_ARRAY, this).and_then(|v| payload_if(ctx, v, TYPE_ARRAY))) {
        return c as i64;
    }
    static_empty_array(&mut state(), ctx) as i64
}

/// Value::getObject() const: the title's null access answer or a shared
/// empty object for other types.
fn cb_value_get_object(ctx: &mut CpuContext) -> HleResult {
    let this = ctx.rdi();
    if let Some(c) = payload_if(ctx, this, TYPE_OBJECT).or_else(|| mismatch_or_null(ctx, TYPE_OBJECT, this).and_then(|v| payload_if(ctx, v, TYPE_OBJECT))) {
        return c as i64;
    }
    static_empty_object(&mut state(), ctx) as i64
}

/// Value::getString() const: the title's null access answer or a shared
/// empty string for other types.
fn cb_value_get_string(ctx: &mut CpuContext) -> HleResult {
    let this = ctx.rdi();
    if let Some(s) = payload_if(ctx, this, TYPE_STRING).or_else(|| mismatch_or_null(ctx, TYPE_STRING, this).and_then(|v| payload_if(ctx, v, TYPE_STRING))) {
        return s as i64;
    }
    static_empty_string(&mut state(), ctx) as i64
}

/// Pointer to the payload when the value has one of `types`; otherwise to
/// the payload of the title's null access answer for `requested`, else to a
/// shared zero.
fn scalar_getter(ctx: &CpuContext, types: &[u32], requested: u32) -> HleResult {
    let this = ctx.rdi();
    if read_head(ctx, this).is_some_and(|(t, _)| types.contains(&t)) {
        return (this + VALUE_PAYLOAD_OFFSET) as i64;
    }
    if let Some(v) = mismatch_or_null(ctx, requested, this) {
        return (v + VALUE_PAYLOAD_OFFSET) as i64;
    }
    static_zero(&mut state(), ctx) as i64
}

fn cb_value_get_boolean(ctx: &mut CpuContext) -> HleResult {
    scalar_getter(ctx, &[TYPE_BOOLEAN], TYPE_BOOLEAN)
}
fn cb_value_get_integer(ctx: &mut CpuContext) -> HleResult {
    scalar_getter(ctx, &[TYPE_INTEGER, TYPE_UINTEGER], TYPE_INTEGER)
}
fn cb_value_get_uinteger(ctx: &mut CpuContext) -> HleResult {
    scalar_getter(ctx, &[TYPE_INTEGER, TYPE_UINTEGER], TYPE_UINTEGER)
}
fn cb_value_get_real(ctx: &mut CpuContext) -> HleResult {
    scalar_getter(ctx, &[TYPE_REAL], TYPE_REAL)
}

/// The member Value of `key` when the Value at `this` is an object.
fn member_of(ctx: &CpuContext, this: u64, key: &[u8]) -> Option<u64> {
    let st = state();
    payload_if(ctx, this, TYPE_OBJECT)
        .and_then(|c| st.objects.get(&c).and_then(|o| o.get(key)).copied())
        .map(|m| m + MEMBER_VALUE_OFFSET)
}

/// The element Value at `index` when the Value at `this` is an array.
fn element_of(ctx: &CpuContext, this: u64, index: u64) -> Option<u64> {
    let st = state();
    payload_if(ctx, this, TYPE_ARRAY).and_then(|c| st.arrays.get(&c).and_then(|a| a.get(usize::try_from(index).ok()?)).copied())
}

/// Value::operator[](const char*) const: the member, or (missing key, not
/// an object) the title's null access answer or the shared null value.
fn cb_value_index_key(ctx: &mut CpuContext) -> HleResult {
    let (this, key) = (ctx.rdi(), read_text_arg(ctx, ctx.rsi()));
    match member_of(ctx, this, &key) {
        Some(m) => m as i64,
        None => missing_value(ctx, this),
    }
}

/// Value::getValue(const String&) const: as operator[](const char*) with
/// the String's text; a NULL key finds nothing.
fn cb_value_get_value_key(ctx: &mut CpuContext) -> HleResult {
    let (this, key) = (ctx.rdi(), ctx.rsi());
    let found = if key == 0 {
        None
    } else {
        let key = string_or_empty(&state(), key);
        member_of(ctx, this, &key)
    };
    match found {
        Some(m) => m as i64,
        None => missing_value(ctx, this),
    }
}

/// Value::operator[](size_t) const / getValue(size_t): the element, or (out
/// of range, not an array) the title's null access answer or the shared
/// null value.
fn cb_value_index_position(ctx: &mut CpuContext) -> HleResult {
    let this = ctx.rdi();
    match element_of(ctx, this, ctx.rsi()) {
        Some(e) => e as i64,
        None => missing_value(ctx, this),
    }
}

/// Value::referValue(size_t): the element, NULL when out of range or not an
/// array.
fn cb_value_refer_value_index(ctx: &mut CpuContext) -> HleResult {
    element_of(ctx, ctx.rdi(), ctx.rsi()).unwrap_or(0) as i64
}

/// Value::referValue(const String& key): the member, NULL when missing or
/// not an object.
fn cb_value_refer_value(ctx: &mut CpuContext) -> HleResult {
    let st = state();
    let key = string_or_empty(&st, ctx.rsi());
    payload_if(ctx, ctx.rdi(), TYPE_OBJECT)
        .and_then(|c| st.objects.get(&c).and_then(|o| o.get(&key)).copied())
        .map_or(0, |m| (m + MEMBER_VALUE_OFFSET) as i64)
}

/// Value::toString(String&): a string value's text, otherwise the
/// serialised value.
fn cb_value_to_string(ctx: &mut CpuContext) -> HleResult {
    let (this, dst) = (ctx.rdi(), ctx.rsi());
    if dst == 0 {
        return 0;
    }
    let mut st = state();
    let text = match payload_if(ctx, this, TYPE_STRING) {
        Some(s) => string_or_empty(&st, s),
        None => {
            let mut out = Vec::new();
            serialize_tree(&snapshot(&st, ctx, this, 0), &mut out);
            out
        }
    };
    assign_string(&mut st, dst, text);
    0
}

/// Value::serialize(String&): compact text; always 0.
fn cb_value_serialize(ctx: &mut CpuContext) -> HleResult {
    let (this, dst) = (ctx.rdi(), ctx.rsi());
    if dst != 0 {
        let mut st = state();
        let mut out = Vec::new();
        serialize_tree(&snapshot(&st, ctx, this, 0), &mut out);
        assign_string(&mut st, dst, out);
    }
    0
}

// ---------------------------------------------------------------------------
// String
// ---------------------------------------------------------------------------

fn set_string(addr: u64, value: Vec<u8>) {
    if addr != 0 {
        state().strings.insert(addr, StringState { value, ..Default::default() });
    }
}

fn cb_string_ctor(ctx: &mut CpuContext) -> HleResult {
    let this = ctx.rdi();
    set_string(this, Vec::new());
    this as i64
}

fn cb_string_cstring_ctor(ctx: &mut CpuContext) -> HleResult {
    let this = ctx.rdi();
    set_string(this, read_text_arg(ctx, ctx.rsi()));
    this as i64
}

fn cb_string_copy_ctor(ctx: &mut CpuContext) -> HleResult {
    let this = ctx.rdi();
    let text = string_or_empty(&state(), ctx.rsi());
    set_string(this, text);
    this as i64
}

/// String::~String(): drop the text and clear the object's pointer.
fn cb_string_dtor(ctx: &mut CpuContext) -> HleResult {
    let this = ctx.rdi();
    if this != 0 {
        state().strings.remove(&this);
        ctx.write_u64(this, 0);
    }
    0
}

/// String::String() base-object constructor: also clears the data pointer.
fn cb_string_base_ctor(ctx: &mut CpuContext) -> HleResult {
    let this = ctx.rdi();
    if this != 0 {
        set_string(this, Vec::new());
        ctx.write_u64(this, 0);
    }
    this as i64
}

/// String::c_str(): the text in a guest buffer (empty for NULL).
fn cb_string_c_str(ctx: &mut CpuContext) -> HleResult {
    let this = ctx.rdi();
    let mut st = state();
    if this == 0 {
        return static_zero(&mut st, ctx) as i64;
    }
    let s = st.strings.entry(this).or_default();
    let mut bytes = s.value.clone();
    bytes.push(0);
    if s.buffer == 0 || s.capacity < bytes.len() {
        match crate::pthread::alloc_guest_block(ctx, bytes.len() as u64) {
            Some(b) => {
                s.buffer = b;
                s.capacity = bytes.len().div_ceil(0x100) * 0x100;
            }
            None => return 0,
        }
    }
    let buffer = s.buffer;
    if !ctx.write_bytes(buffer, &bytes) {
        return 0;
    }
    ctx.write_u64(this, buffer);
    buffer as i64
}

/// String::operator=(const String&)
fn cb_string_assign(ctx: &mut CpuContext) -> HleResult {
    let this = ctx.rdi();
    if this != 0 {
        let mut st = state();
        let text = string_or_empty(&st, ctx.rsi());
        assign_string(&mut st, this, text);
    }
    this as i64
}

/// String::length() (bytes).
fn cb_string_length(ctx: &mut CpuContext) -> HleResult {
    state().strings.get(&ctx.rdi()).map_or(0, |s| s.value.len()) as i64
}

fn cb_string_empty(ctx: &mut CpuContext) -> HleResult {
    state().strings.get(&ctx.rdi()).is_none_or(|s| s.value.is_empty()) as i64
}

/// String::operator==(const char*) const
fn cb_string_eq_cstr(ctx: &mut CpuContext) -> HleResult {
    let Some(other) = ctx.read_cstr_bytes(ctx.rsi(), MAX_CSTRING) else {
        return 0;
    };
    (string_or_empty(&state(), ctx.rdi()) == other) as i64
}

/// String::npos.
const NPOS: u64 = u64::MAX;

/// String::clear()
fn cb_string_clear(ctx: &mut CpuContext) -> HleResult {
    let this = ctx.rdi();
    if this != 0 {
        assign_string(&mut state(), this, Vec::new());
    }
    0
}

/// String::reserve(size_t): capacity only, the text is unchanged.
fn cb_string_reserve(ctx: &mut CpuContext) -> HleResult {
    let (this, capacity) = (ctx.rdi(), ctx.rsi());
    if this != 0 {
        let mut st = state();
        let s = st.strings.entry(this).or_default();
        let extra = usize::try_from(capacity).unwrap_or(0).min(MAX_CSTRING).saturating_sub(s.value.len());
        s.value.reserve(extra);
    }
    0
}

/// String::append(const String&) -> *this; appending a string to itself
/// doubles it.
fn cb_string_append_string(ctx: &mut CpuContext) -> HleResult {
    let (this, other) = (ctx.rdi(), ctx.rsi());
    if this != 0 {
        let mut st = state();
        let tail = string_or_empty(&st, other);
        st.strings.entry(this).or_default().value.extend_from_slice(&tail);
    }
    this as i64
}

/// String::append(const char*) -> *this; a NULL text is a title bug and
/// leaves the string unchanged.
fn cb_string_append_cstr(ctx: &mut CpuContext) -> HleResult {
    let (this, text) = (ctx.rdi(), ctx.rsi());
    if text == 0 {
        log::error!("[json] String::append({this:#x}, NULL): null text");
        return this as i64;
    }
    let tail = read_text_arg(ctx, text);
    if this != 0 {
        state().strings.entry(this).or_default().value.extend_from_slice(&tail);
    }
    this as i64
}

/// std::string::find(char, pos) on `s`.
fn find_byte(s: &[u8], ch: u8, pos: u64) -> u64 {
    let start = usize::try_from(pos).unwrap_or(usize::MAX);
    s.get(start..).and_then(|t| t.iter().position(|&b| b == ch)).map_or(NPOS, |i| (start + i) as u64)
}

/// std::string::rfind(char, pos) on `s`: the last match at or before `pos`.
fn rfind_byte(s: &[u8], ch: u8, pos: u64) -> u64 {
    if s.is_empty() {
        return NPOS;
    }
    let end = usize::try_from(pos).unwrap_or(usize::MAX).min(s.len() - 1);
    s[..=end].iter().rposition(|&b| b == ch).map_or(NPOS, |i| i as u64)
}

/// String::find(char, size_t pos) const
fn cb_string_find(ctx: &mut CpuContext) -> HleResult {
    find_byte(&string_or_empty(&state(), ctx.rdi()), ctx.rsi() as u8, ctx.rdx()) as i64
}

/// String::rfind(char, size_t pos) const
fn cb_string_rfind(ctx: &mut CpuContext) -> HleResult {
    rfind_byte(&string_or_empty(&state(), ctx.rdi()), ctx.rsi() as u8, ctx.rdx()) as i64
}

/// String::substr(size_t pos, size_t len) const: the String is returned
/// through the hidden result pointer in rdi, `this` in rsi. A position past
/// the end is a title bug (std::out_of_range): the result is empty.
fn cb_string_substr(ctx: &mut CpuContext) -> HleResult {
    let (result, this, pos, len) = (ctx.rdi(), ctx.rsi(), ctx.rdx(), ctx.rcx());
    let s = string_or_empty(&state(), this);
    let part = match usize::try_from(pos).ok().filter(|&p| p <= s.len()) {
        Some(p) => {
            let n = usize::try_from(len).unwrap_or(usize::MAX).min(s.len() - p);
            s[p..p + n].to_vec()
        }
        None => {
            log::error!("[json] String::substr({this:#x}, {pos}, {len}): position past the end ({})", s.len());
            Vec::new()
        }
    };
    if result != 0 {
        ctx.write_u64(result, 0);
        set_string(result, part);
    }
    result as i64
}

/// String::compare(const char*) const: <0, 0 or >0 as the bytes compare; a
/// NULL text is a title bug and compares unequal.
fn cb_string_compare(ctx: &mut CpuContext) -> HleResult {
    let (this, text) = (ctx.rdi(), ctx.rsi());
    if text == 0 {
        log::error!("[json] String::compare({this:#x}, NULL): null text");
        return 1;
    }
    let other = read_text_arg(ctx, text);
    match string_or_empty(&state(), this).cmp(&other) {
        std::cmp::Ordering::Less => -1,
        std::cmp::Ordering::Equal => 0,
        std::cmp::Ordering::Greater => 1,
    }
}

// ---------------------------------------------------------------------------
// Array
// ---------------------------------------------------------------------------

fn cb_array_ctor(ctx: &mut CpuContext) -> HleResult {
    let this = ctx.rdi();
    if this != 0 {
        let mut st = state();
        destroy_array(&mut st, ctx, this);
        st.arrays.insert(this, Vec::new());
    }
    this as i64
}

fn cb_array_dtor(ctx: &mut CpuContext) -> HleResult {
    let this = ctx.rdi();
    if this != 0 {
        destroy_array(&mut state(), ctx, this);
    }
    0
}

/// Array::push_back(const Value&): appends a copy (a null value for NULL).
fn cb_array_push_back(ctx: &mut CpuContext) -> HleResult {
    let (this, src) = (ctx.rdi(), ctx.rsi());
    if this == 0 {
        return 0;
    }
    let mut st = state();
    let t = snapshot(&st, ctx, src, 0);
    if let Some(e) = new_owned(&mut st, ctx) {
        install(&mut st, ctx, e, &t, 0);
        st.arrays.entry(this).or_default().push(e);
    }
    this as i64
}

/// Array::back(): the last element (the shared null value when empty).
fn cb_array_back(ctx: &mut CpuContext) -> HleResult {
    let mut st = state();
    match st.arrays.get(&ctx.rdi()).and_then(|a| a.last().copied()) {
        Some(v) => v as i64,
        None => static_null_value(&mut st, ctx) as i64,
    }
}

fn cb_array_size(ctx: &mut CpuContext) -> HleResult {
    state().arrays.get(&ctx.rdi()).map_or(0, Vec::len) as i64
}

fn cb_array_empty(ctx: &mut CpuContext) -> HleResult {
    state().arrays.get(&ctx.rdi()).is_none_or(Vec::is_empty) as i64
}

/// begin()/end() of an Array (`object == false`) or Object: the iterator
/// (one pointer into the table) is returned through the hidden result
/// pointer in rdi; `this` is in rsi.
fn container_iterator(ctx: &CpuContext, object: bool, at_end: bool) -> HleResult {
    let (result, this) = (ctx.rdi(), ctx.rsi());
    if result == 0 {
        return 0;
    }
    let mut st = state();
    let entries: Vec<u64> = if object {
        st.objects.get(&this).map(|m| m.values().copied().collect()).unwrap_or_default()
    } else {
        st.arrays.get(&this).cloned().unwrap_or_default()
    };
    let pos = match iter_table(&mut st, ctx, this, &entries) {
        Some((t, n)) if at_end => t + 8 * n as u64,
        Some((t, _)) => t,
        None => 0,
    };
    ctx.write_u64(result, pos);
    result as i64
}

fn cb_array_begin(ctx: &mut CpuContext) -> HleResult {
    container_iterator(ctx, false, false)
}
fn cb_array_end(ctx: &mut CpuContext) -> HleResult {
    container_iterator(ctx, false, true)
}
fn cb_object_begin(ctx: &mut CpuContext) -> HleResult {
    container_iterator(ctx, true, false)
}
fn cb_object_end(ctx: &mut CpuContext) -> HleResult {
    container_iterator(ctx, true, true)
}

/// iterator::operator++()
fn cb_iterator_increment(ctx: &mut CpuContext) -> HleResult {
    let this = ctx.rdi();
    if let Some(pos) = ctx.read_u64(this) {
        ctx.write_u64(this, pos + 8);
    }
    this as i64
}

/// iterator::operator*(): the element Value (Array) or the
/// `pair<const String, Value>` member block (Object).
fn cb_iterator_deref(ctx: &mut CpuContext) -> HleResult {
    ctx.read_u64(ctx.rdi()).and_then(|pos| ctx.read_u64(pos)).unwrap_or(0) as i64
}

/// iterator::operator!=(const iterator&)
fn cb_iterator_ne(ctx: &mut CpuContext) -> HleResult {
    (ctx.read_u64(ctx.rdi()) != ctx.read_u64(ctx.rsi())) as i64
}

/// Array::iterator::iterator(): singular (no position).
fn cb_iterator_default_ctor(ctx: &mut CpuContext) -> HleResult {
    let this = ctx.rdi();
    if this != 0 {
        ctx.write_u64(this, 0);
    }
    this as i64
}

/// Array::iterator copy constructor and operator=: the position is copied.
fn cb_iterator_copy(ctx: &mut CpuContext) -> HleResult {
    let (this, other) = (ctx.rdi(), ctx.rsi());
    if this != 0 {
        ctx.write_u64(this, ctx.read_u64(other).unwrap_or(0));
    }
    this as i64
}

/// Array::iterator::operator++(int): the old position is returned through
/// the hidden result pointer in rdi, the iterator (in rsi) steps on.
fn cb_iterator_post_increment(ctx: &mut CpuContext) -> HleResult {
    let (result, this) = (ctx.rdi(), ctx.rsi());
    let pos = ctx.read_u64(this).unwrap_or(0);
    ctx.write_u64(this, pos.wrapping_add(8));
    if result != 0 {
        ctx.write_u64(result, pos);
    }
    result as i64
}

/// Array::iterator::advance(size_t n): n elements on.
fn cb_iterator_advance(ctx: &mut CpuContext) -> HleResult {
    let (this, n) = (ctx.rdi(), ctx.rsi());
    if let Some(pos) = ctx.read_u64(this) {
        ctx.write_u64(this, pos.wrapping_add(n.wrapping_mul(8)));
    }
    this as i64
}

/// Array::Array(const Array&): deep copy without a parent Value.
fn cb_array_copy_ctor(ctx: &mut CpuContext) -> HleResult {
    let (this, src) = (ctx.rdi(), ctx.rsi());
    if this != 0 {
        let mut st = state();
        let items = snapshot_array(&st, ctx, src, 0);
        let copy = install_items(&mut st, ctx, &items, 0);
        st.arrays.insert(this, copy);
    }
    this as i64
}

/// Array::clear(): the elements are released, the array stays.
fn cb_array_clear(ctx: &mut CpuContext) -> HleResult {
    let this = ctx.rdi();
    if this != 0 {
        let mut st = state();
        destroy_array(&mut st, ctx, this);
        st.arrays.insert(this, Vec::new());
    }
    0
}

/// Array::front(): the first element; an empty array is a title bug and
/// gets the shared null value, as back() does.
fn cb_array_front(ctx: &mut CpuContext) -> HleResult {
    let this = ctx.rdi();
    let mut st = state();
    match st.arrays.get(&this).and_then(|a| a.first().copied()) {
        Some(v) => v as i64,
        None => {
            log::error!("[json] Array::front({this:#x}): empty array");
            static_null_value(&mut st, ctx) as i64
        }
    }
}

// ---------------------------------------------------------------------------
// Object
// ---------------------------------------------------------------------------

fn cb_object_ctor(ctx: &mut CpuContext) -> HleResult {
    let this = ctx.rdi();
    if this != 0 {
        let mut st = state();
        destroy_object(&mut st, ctx, this);
        st.objects.insert(this, BTreeMap::new());
    }
    this as i64
}

/// Object::Object(const Object&): deep copy without a parent Value.
fn cb_object_copy_ctor(ctx: &mut CpuContext) -> HleResult {
    let (this, src) = (ctx.rdi(), ctx.rsi());
    if this != 0 {
        let mut st = state();
        let members = snapshot_object(&st, ctx, src, 0);
        let copy = install_members(&mut st, ctx, &members, 0);
        st.objects.insert(this, copy);
    }
    this as i64
}

/// Object::operator=(const Object&): copy first, then drop the old members.
fn cb_object_assign(ctx: &mut CpuContext) -> HleResult {
    let (this, src) = (ctx.rdi(), ctx.rsi());
    if this != 0 {
        let mut st = state();
        let members = snapshot_object(&st, ctx, src, 0);
        let copy = install_members(&mut st, ctx, &members, 0);
        destroy_object(&mut st, ctx, this);
        st.objects.insert(this, copy);
    }
    this as i64
}

fn cb_object_dtor(ctx: &mut CpuContext) -> HleResult {
    let this = ctx.rdi();
    if this != 0 {
        destroy_object(&mut state(), ctx, this);
    }
    0
}

fn cb_object_clear(ctx: &mut CpuContext) -> HleResult {
    let this = ctx.rdi();
    if this != 0 {
        let mut st = state();
        destroy_object(&mut st, ctx, this);
        st.objects.insert(this, BTreeMap::new());
    }
    0
}

/// Object::size() const
fn cb_object_size(ctx: &mut CpuContext) -> HleResult {
    state().objects.get(&ctx.rdi()).map_or(0, BTreeMap::len) as i64
}

/// Object::empty() const
fn cb_object_empty(ctx: &mut CpuContext) -> HleResult {
    state().objects.get(&ctx.rdi()).is_none_or(BTreeMap::is_empty) as i64
}

/// Object::operator[](const String&): the member Value, inserted as null
/// when missing (the shared null value for a NULL object).
fn cb_object_index(ctx: &mut CpuContext) -> HleResult {
    let this = ctx.rdi();
    let mut st = state();
    if this == 0 {
        return static_null_value(&mut st, ctx) as i64;
    }
    let key = string_or_empty(&st, ctx.rsi());
    if let Some(&m) = st.objects.get(&this).and_then(|o| o.get(&key)) {
        return (m + MEMBER_VALUE_OFFSET) as i64;
    }
    let Some(m) = new_owned(&mut st, ctx) else {
        return static_null_value(&mut st, ctx) as i64;
    };
    st.strings.insert(m, StringState { value: key.clone(), ..Default::default() });
    st.objects.entry(this).or_default().insert(key, m);
    (m + MEMBER_VALUE_OFFSET) as i64
}

/// Object::find(const String&) const: the iterator at the member, or end(),
/// through the hidden result pointer in rdi; `this` in rsi, the key in rdx.
fn cb_object_find(ctx: &mut CpuContext) -> HleResult {
    let (result, this, key) = (ctx.rdi(), ctx.rsi(), ctx.rdx());
    if result == 0 {
        return 0;
    }
    let mut st = state();
    let key = string_or_empty(&st, key);
    let (entries, index) = match st.objects.get(&this) {
        Some(m) => (m.values().copied().collect::<Vec<_>>(), m.keys().position(|k| *k == key).unwrap_or(m.len())),
        None => (Vec::new(), 0),
    };
    let pos = iter_table(&mut st, ctx, this, &entries).map_or(0, |(t, _)| t + 8 * index as u64);
    ctx.write_u64(result, pos);
    result as i64
}

/// Object::Pair::Pair(const String& key, const Value& value): deep copies
/// in the member block layout iteration hands out (key String at +0, Value
/// at +8).
fn cb_pair_ctor(ctx: &mut CpuContext) -> HleResult {
    let (this, key, value) = (ctx.rdi(), ctx.rsi(), ctx.rdx());
    if this == 0 {
        return 0;
    }
    let mut st = state();
    let key = string_or_empty(&st, key);
    let copy = snapshot(&st, ctx, value, 0);
    ctx.write_u64(this, 0);
    st.strings.insert(this, StringState { value: key, ..Default::default() });
    init_value(ctx, this + MEMBER_VALUE_OFFSET);
    install(&mut st, ctx, this + MEMBER_VALUE_OFFSET, &copy, 0);
    this as i64
}

/// Object::Pair::~Pair(): the key and the value are released.
fn cb_pair_dtor(ctx: &mut CpuContext) -> HleResult {
    let this = ctx.rdi();
    if this != 0 {
        let mut st = state();
        clear_value(&mut st, ctx, this + MEMBER_VALUE_OFFSET, false);
        st.strings.remove(&this);
        ctx.write_u64(this, 0);
    }
    0
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

const EXPORTS: &[(&str, &str, HleCallback)] = &[
    ("-hJRce8wn1U", "_ZN3sce4Json12MemAllocatorC2Ev", cb_return_this),
    ("OcAgPxcq5Vk", "_ZN3sce4Json12MemAllocatorD2Ev", cb_return_void),
    ("cK6bYHf-Q5E", "_ZN3sce4Json11InitializerC1Ev", cb_return_this),
    ("RujUxbr3haM", "_ZN3sce4Json11InitializerD1Ev", cb_return_void),
    ("PR5k1penBLM", "_ZN3sce4Json11Initializer9terminateEv", cb_initializer_terminate),
    ("Cxwy7wHq4J0", "_ZN3sce4Json11Initializer10initializeEPKNS0_13InitParameterE", cb_return_ok),
    ("IXW-z8pggfg", "_ZN3sce4Json11Initializer10initializeEPKNS0_14InitParameter2E", cb_return_ok),
    (
        "+drDFyAS6u4",
        "_ZN3sce4Json11Initializer27setGlobalNullAccessCallbackEPFRKNS0_5ValueENS0_9ValueTypeEPS3_PvES7_",
        cb_set_global_null_access_callback,
    ),
    (
        "00oCq0RwSAY",
        "_ZN3sce4Json11Initializer27setGlobalNullAccessCallBackEPFRKNS0_5ValueENS0_9ValueTypeEPS3_PvES7_",
        cb_set_global_null_access_callback,
    ),
    (
        "Xbl-LYVFNEE",
        "_ZN3sce4Json5Value21setNullAccessCallBackEPFRKS1_NS0_9ValueTypeEPS2_PvES6_",
        cb_value_set_null_access_callback,
    ),
    (
        "ukQodCmg99s",
        "_ZN3sce4Json5Value22setTypeMismatchHandlerEPFRKS1_NS0_9ValueTypeEPS2_S5_PvES6_",
        cb_value_set_type_mismatch_handler,
    ),
    (
        "6tTbrt1O1SY",
        "_ZN3sce4Json5Value30setElementAccessFailureHandlerEPFviPKNS0_6StringEPKmPKS1_S8_PvES9_",
        cb_value_set_element_access_failure_handler,
    ),
    (
        "bJWBpvGCIJU",
        "_ZN3sce4Json5Value22setSpecialFloatHandlerEPFKS1_NS0_12FunctionTypeEdPS2_PvPbES5_",
        cb_value_set_special_float_handler,
    ),
    ("i-XwZjw0OOY", "_ZN3sce4Json12MemAllocator11notifyErrorEimPv", cb_mem_allocator_notify_error),
    ("6i18OJSvFWk", "_ZN3sce4Json17InitParameterRttiC1EPNS0_14AllocParamRttiEPvm", cb_init_parameter_rtti_ctor),
    ("WSOuge5IsCg", "_ZN3sce4Json14InitParameter2C1Ev", cb_init_parameter2_ctor),
    ("GvGvswb0v34", "_ZN3sce4Json14InitParameter2C2Ev", cb_init_parameter2_ctor),
    ("I2QC8PYhJWY", "_ZN3sce4Json14InitParameter212setAllocatorEPNS0_12MemAllocatorEPv", cb_init_parameter2_set_allocator),
    (
        "W72B9ylU2JA",
        "_ZN3sce4Json18InitParameterRtti216setAllocatorRttiEPNS0_14AllocParamRttiEPv",
        cb_init_parameter2_set_allocator,
    ),
    ("Eu95jmqn5Rw", "_ZN3sce4Json14InitParameter217setFileBufferSizeEm", cb_init_parameter2_set_file_buffer_size),
    (
        "WVZBP4IyM+E",
        "_ZN3sce4Json14InitParameter225setSpecialFloatFormatTypeENS0_22SpecialFloatFormatTypeE",
        cb_init_parameter2_set_special_float_format_type,
    ),
    ("S5JxQnoGF3E", "_ZN3sce4Json6Parser5parseERNS0_5ValueEPKcm", cb_parser_parse),
    ("itqj2YmuAa8", "_ZN3sce4Json6Parser5parseERNS0_5ValueEPFiRcPvES5_", cb_parser_parse_callback),
    // Value.
    ("qBMjqyBn3OM", "_ZN3sce4Json5ValueC1Ev", cb_value_ctor),
    ("-wa17B7TGnw", "_ZN3sce4Json5ValueC2Ev", cb_value_ctor),
    ("WTtYf+cNnXI", "_ZN3sce4Json5ValueD1Ev", cb_value_dtor),
    ("0eUrW9JAxM0", "_ZN3sce4Json5ValueD2Ev", cb_value_dtor),
    ("FIjXN2TkuTs", "_ZN3sce4Json5Value5clearEv", cb_value_clear),
    ("fSb2oQTNrgA", "_ZN3sce4Json5ValueC1ERKS1_", cb_value_copy_ctor),
    ("4zrm6VrgIAw", "_ZN3sce4Json5ValueaSERKS1_", cb_value_assign),
    ("XL8+BUqjB1w", "_ZN3sce4Json5Value3setERKS1_", cb_value_assign),
    ("UeuWT+yNdCQ", "_ZN3sce4Json5ValueC1Eb", cb_value_bool_ctor),
    ("0lLK8+kDqmE", "_ZN3sce4Json5ValueC1El", cb_value_int_ctor),
    ("x4AUdbhpRB0", "_ZN3sce4Json5ValueC1Em", cb_value_uint_ctor),
    ("sOmU4vnx3s0", "_ZN3sce4Json5ValueC1Ed", cb_value_double_ctor),
    ("b9V6fmppLXY", "_ZN3sce4Json5ValueC1EPKc", cb_value_cstring_ctor),
    ("sZIoMRGO+jk", "_ZN3sce4Json5ValueC1ERKNS0_6StringE", cb_value_string_ctor),
    ("CbrT3dwDILo", "_ZN3sce4Json5ValueC1ENS0_9ValueTypeE", cb_value_type_ctor),
    ("iZeYfOxtMRg", "_ZN3sce4Json5ValueC1ERKNS0_5ArrayE", cb_value_array_ctor),
    ("3xUXnmUkXfo", "_ZN3sce4Json5ValueC1ERKNS0_6ObjectE", cb_value_object_ctor),
    ("5yHuiWXo2gg", "_ZN3sce4Json5Value3setEb", cb_value_set_bool),
    ("QxVVYhP-mvg", "_ZN3sce4Json5Value3setEl", cb_value_set_int),
    ("SIe1ZmW7e7s", "_ZN3sce4Json5Value3setEm", cb_value_set_uint),
    ("BSmWDIkV4w4", "_ZN3sce4Json5Value3setEd", cb_value_set_double),
    ("n6FC+l9DU70", "_ZN3sce4Json5Value3setEPKc", cb_value_set_cstring),
    ("6l3Bv2gysNc", "_ZN3sce4Json5Value3setERKNS0_6StringE", cb_value_set_string),
    ("IKQimvG9Wqs", "_ZN3sce4Json5Value3setENS0_9ValueTypeE", cb_value_set_type),
    ("195ad-jAsTU", "_ZN3sce4Json5Value3setERKNS0_5ArrayE", cb_value_set_array),
    ("dFCphqnd+a4", "_ZN3sce4Json5Value3setERKNS0_6ObjectE", cb_value_set_object),
    ("SHtAad20YYM", "_ZNK3sce4Json5Value7getTypeEv", cb_value_get_type),
    ("RBw+4NukeGQ", "_ZNK3sce4Json5Value5countEv", cb_value_count),
    ("nM5XqdeXFPw", "_ZN3sce4Json5Value10referArrayEv", cb_value_refer_array),
    ("-NxEk7XLkDY", "_ZN3sce4Json5Value11referObjectEv", cb_value_refer_object),
    ("ONT8As5R1ug", "_ZNK3sce4Json5Value8getArrayEv", cb_value_get_array),
    ("IlsmvBtMkak", "_ZNK3sce4Json5Value9getObjectEv", cb_value_get_object),
    ("epJ6x2LV0kU", "_ZNK3sce4Json5Value9getStringEv", cb_value_get_string),
    ("zTwZdI8AZ5Y", "_ZNK3sce4Json5Value10getBooleanEv", cb_value_get_boolean),
    ("DIxvoy7Ngvk", "_ZNK3sce4Json5Value10getIntegerEv", cb_value_get_integer),
    ("sn4HNCtNRzY", "_ZNK3sce4Json5Value11getUIntegerEv", cb_value_get_uinteger),
    ("3qrge7L-AU4", "_ZNK3sce4Json5Value7getRealEv", cb_value_get_real),
    ("HwDt5lD9Bfo", "_ZNK3sce4Json5ValueixEPKc", cb_value_index_key),
    ("XlWbvieLj2M", "_ZNK3sce4Json5ValueixEm", cb_value_index_position),
    ("0YqYAoO-+Uo", "_ZNK3sce4Json5Value8getValueEm", cb_value_index_position),
    ("wLsJlmgEIaI", "_ZN3sce4Json5Value10referValueERKNS0_6StringE", cb_value_refer_value),
    ("gLzCc67aTbw", "_ZN3sce4Json5Value10referValueEm", cb_value_refer_value_index),
    ("MsMOdxWfbwQ", "_ZNK3sce4Json5Value8getValueERKNS0_6StringE", cb_value_get_value_key),
    ("Ncel8t2Rrpc", "_ZNK3sce4Json5Value8toStringERNS0_6StringE", cb_value_to_string),
    ("R7FDWtcN6f8", "_ZN3sce4Json5Value9serializeERNS0_6StringE", cb_value_serialize),
    // String.
    ("qSmqLXXCPas", "_ZN3sce4Json6StringC1Ev", cb_string_ctor),
    ("eG9E9M6XvTM", "_ZN3sce4Json6StringC2Ev", cb_string_base_ctor),
    ("9KUZFjI1IxA", "_ZN3sce4Json6StringC1EPKc", cb_string_cstring_ctor),
    ("0CAesfH963Q", "_ZN3sce4Json6StringC1ERKS1_", cb_string_copy_ctor),
    ("cG1VE2HMl6c", "_ZN3sce4Json6StringD1Ev", cb_string_dtor),
    ("Ui7YFnSTCBw", "_ZN3sce4Json6StringD2Ev", cb_string_dtor),
    ("L1KAkYWml-M", "_ZNK3sce4Json6String5c_strEv", cb_string_c_str),
    ("cn9svYGWKDQ", "_ZN3sce4Json6StringaSERKS1_", cb_string_assign),
    ("EUH+EmT-v9E", "_ZNK3sce4Json6String6lengthEv", cb_string_length),
    ("wM4LO2iK3s8", "_ZNK3sce4Json6String5emptyEv", cb_string_empty),
    ("VbFjEs--uiA", "_ZNK3sce4Json6StringeqEPKc", cb_string_eq_cstr),
    ("A+FhGENHV-E", "_ZN3sce4Json6String5clearEv", cb_string_clear),
    ("Va5FJ6TmNCw", "_ZN3sce4Json6String6appendEPKc", cb_string_append_cstr),
    ("Rz00kiTKypo", "_ZN3sce4Json6String6appendERKS1_", cb_string_append_string),
    ("Z9tyi1-LrQg", "_ZN3sce4Json6String7reserveEm", cb_string_reserve),
    ("x4SuxMR26I4", "_ZNK3sce4Json6String4findEcm", cb_string_find),
    ("t+Dx3eQtBPQ", "_ZNK3sce4Json6String5rfindEcm", cb_string_rfind),
    ("8OCexW9qJgQ", "_ZNK3sce4Json6String6substrEmm", cb_string_substr),
    ("Yfw63FVJgtU", "_ZNK3sce4Json6String7compareEPKc", cb_string_compare),
    // Array.
    ("JP-PtKMiI1E", "_ZN3sce4Json5ArrayC1Ev", cb_array_ctor),
    ("HJ8GpRT1aiw", "_ZN3sce4Json5ArrayD1Ev", cb_array_dtor),
    ("t6AHhkWu4mY", "_ZN3sce4Json5ArrayC2Ev", cb_array_ctor),
    ("MIyMPw7YTGM", "_ZN3sce4Json5ArrayD2Ev", cb_array_dtor),
    ("bI5AGFMydrA", "_ZN3sce4Json5ArrayC1ERKS1_", cb_array_copy_ctor),
    ("qVOSuDRHCpA", "_ZN3sce4Json5Array5clearEv", cb_array_clear),
    ("zcfZQyM7eAE", "_ZNK3sce4Json5Array5frontEv", cb_array_front),
    ("zQtLRTqceMY", "_ZN3sce4Json5Array9push_backERKNS0_5ValueE", cb_array_push_back),
    ("bAM9Qwofus0", "_ZNK3sce4Json5Array4backEv", cb_array_back),
    ("rQGJeNjOuUk", "_ZNK3sce4Json5Array4sizeEv", cb_array_size),
    ("9uP25i6ipno", "_ZNK3sce4Json5Array5emptyEv", cb_array_empty),
    ("bcH5EnFE2xY", "_ZNK3sce4Json5Array5beginEv", cb_array_begin),
    ("WXF2ihRF+B8", "_ZNK3sce4Json5Array3endEv", cb_array_end),
    ("w5+VCznos5E", "_ZN3sce4Json5Array8iteratorppEv", cb_iterator_increment),
    ("wcgr5mte7T8", "_ZNK3sce4Json5Array8iteratordeEv", cb_iterator_deref),
    ("5AZPp99ogrc", "_ZNK3sce4Json5Array8iteratorneERKS2_", cb_iterator_ne),
    ("9yLjn46Ypfs", "_ZN3sce4Json5Array8iteratorD1Ev", cb_return_void),
    ("9niNrXsF3Bc", "_ZN3sce4Json5Array8iteratorD2Ev", cb_return_void),
    ("TAPTWYaYf8Q", "_ZN3sce4Json5Array8iteratorC1Ev", cb_iterator_default_ctor),
    ("JSg1MSo7NKE", "_ZN3sce4Json5Array8iteratorC2Ev", cb_iterator_default_ctor),
    ("IaIvX3kgyb0", "_ZN3sce4Json5Array8iteratorC1ERKS2_", cb_iterator_copy),
    ("Hdc3DSIsG2I", "_ZN3sce4Json5Array8iteratorC2ERKS2_", cb_iterator_copy),
    ("T3WReElX4BY", "_ZN3sce4Json5Array8iteratoraSERKS2_", cb_iterator_copy),
    ("GEylwSsiGG8", "_ZN3sce4Json5Array8iteratorppEi", cb_iterator_post_increment),
    ("43MESLndGvQ", "_ZN3sce4Json5Array8iterator7advanceEm", cb_iterator_advance),
    ("iAIYn4oAWvI", "_ZNK3sce4Json5Array8iteratorptEv", cb_iterator_deref),
    // Object.
    ("OJPTonqdg0I", "_ZN3sce4Json6ObjectC1Ev", cb_object_ctor),
    ("a+W7HHlwpBs", "_ZN3sce4Json6ObjectC1ERKS1_", cb_object_copy_ctor),
    ("urOpESTBZmo", "_ZN3sce4Json6ObjectaSERKS1_", cb_object_assign),
    ("5JmzZt8twAo", "_ZN3sce4Json6ObjectD1Ev", cb_object_dtor),
    ("oH8aBmLU+fc", "_ZN3sce4Json6Object5clearEv", cb_object_clear),
    ("fSGHm9RjN5U", "_ZNK3sce4Json6Object4sizeEv", cb_object_size),
    ("i2l3IYvQ9UE", "_ZNK3sce4Json6Object5emptyEv", cb_object_empty),
    ("ERuf9y0DY84", "_ZN3sce4Json6ObjectixERKNS0_6StringE", cb_object_index),
    ("xhAcaIwnrgk", "_ZNK3sce4Json6Object5beginEv", cb_object_begin),
    ("ivMCitpSQNk", "_ZNK3sce4Json6Object3endEv", cb_object_end),
    ("DlWmn2ZQuWY", "_ZN3sce4Json6Object8iteratorppEv", cb_iterator_increment),
    ("ZCd6IYoD3Bc", "_ZNK3sce4Json6Object8iteratordeEv", cb_iterator_deref),
    ("+isUKw4zud4", "_ZNK3sce4Json6Object8iteratorneERKS2_", cb_iterator_ne),
    ("hoINmSMlYjI", "_ZN3sce4Json6Object8iteratorD1Ev", cb_return_void),
    ("EnesV4gXDtE", "_ZNK3sce4Json6Object8iteratorptEv", cb_iterator_deref),
    ("ehN1Biw3tw8", "_ZNK3sce4Json6Object4findERKNS0_6StringE", cb_object_find),
    ("+hYsagaV-Ug", "_ZN3sce4Json6Object4PairC1ERKNS0_6StringERKNS0_5ValueE", cb_pair_ctor),
    ("Q0ng4sWj2E8", "_ZN3sce4Json6Object4PairD1Ev", cb_pair_dtor),
    (
        "i1393UBWu1U",
        "_ZN3sce4Json11Initializer28setGlobalSpecialFloatHandlerEPFKNS0_5ValueENS0_12FunctionTypeEdPS3_PvPbES6_",
        cb_return_ok,
    ),
];

pub fn register(reg: &mut HleRegistry) {
    for &(nid, name, cb) in EXPORTS {
        reg.register_fn(nid, name, LIB, cb);
    }
}

#[cfg(test)]
mod tests {
    /// The library state is global: tests that create and destroy values run
    /// one at a time (a freed string address may be reused by another test).
    static SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());

    fn serial() -> std::sync::MutexGuard<'static, ()> {
        SERIAL.lock().unwrap_or_else(|e| e.into_inner())
    }

    use super::*;
    use crate::context::CpuRegister::*;

    /// Call `f` with rdi/rsi set.
    fn call(ctx: &mut CpuContext, f: HleCallback, rdi: u64, rsi: u64) -> u64 {
        ctx.set(Rdi, rdi);
        ctx.set(Rsi, rsi);
        f(ctx) as u64
    }

    fn text(addr: u64) -> String {
        String::from_utf8(string_or_empty(&state(), addr)).unwrap()
    }

    fn serialized(ctx: &mut CpuContext, value: u64, out: u64) -> String {
        call(ctx, cb_string_ctor, out, 0);
        assert_eq!(call(ctx, cb_value_serialize, value, out), 0);
        text(out)
    }

    fn parse_into(ctx: &mut CpuContext, value: u64, buf: u64, doc: &[u8]) -> i64 {
        ctx.write_bytes(buf, doc);
        ctx.set(Rdx, doc.len() as u64);
        call(ctx, cb_parser_parse, value, buf) as i64
    }

    /// Per-Value handlers: the setters return 0 and install on the Value and
    /// what it holds; a parse into the Value hands them to the new values;
    /// a copy does not carry them; the mismatch answer must have the
    /// requested type.
    #[test]
    fn per_value_handlers() {
        let _serial = serial();
        const BASE: u64 = 0x5DB9_0000_0000;
        let vm = json_vm(BASE);
        let mut ctx = CpuContext::new(&vm);
        let (root, buf, copy, answer, key) = (BASE, BASE + 0x100, BASE + 0x40, BASE + 0x80, BASE + 0x200);
        call(&mut ctx, cb_value_ctor, root, 0);
        ctx.set(Rdx, 0xC0E7);
        assert_eq!(call(&mut ctx, cb_value_set_type_mismatch_handler, root, 0xC0DE), 0);
        ctx.set(Rdx, 0);
        assert_eq!(call(&mut ctx, cb_value_set_element_access_failure_handler, root, 0xE1), 0);
        assert_eq!(call(&mut ctx, cb_value_set_special_float_handler, root, 0xF1), 0);
        assert_eq!(parse_into(&mut ctx, root, buf, b"{\"a\":\"text\",\"b\":[1]}\0"), 0);
        ctx.write_bytes(key, b"a\0");
        let a = call(&mut ctx, cb_value_index_key, root, key);
        let h = state().handlers.get(&a).copied().unwrap();
        assert_eq!((h.type_mismatch, h.element_access_failure, h.special_float), ((0xC0DE, 0xC0E7), (0xE1, 0), (0xF1, 0)));
        assert_eq!(state().handlers.get(&root).unwrap().type_mismatch, (0xC0DE, 0xC0E7), "kept across the parse");
        // The handler's answer counts only with the requested type.
        call(&mut ctx, cb_value_int_ctor, answer, 42);
        let mut calls = Vec::new();
        let mut invoke = |f: u64, args: [u64; 6]| {
            calls.push((f, args[0], args[1], args[2], args[3]));
            answer
        };
        assert_eq!(type_mismatch_via(&ctx, h.type_mismatch, TYPE_INTEGER, a, &mut invoke), Some(answer));
        assert_eq!(type_mismatch_via(&ctx, h.type_mismatch, TYPE_REAL, a, &mut invoke), None);
        assert_eq!(calls, vec![(0xC0DE, 2, a, 0, 0xC0E7), (0xC0DE, 4, a, 0, 0xC0E7)]);
        assert_eq!(type_mismatch_via(&ctx, h.type_mismatch, TYPE_INTEGER, a, &mut |_, _| 0), None);
        // A copy has no handlers; destroying a Value drops them.
        call(&mut ctx, cb_value_copy_ctor, copy, root);
        assert!(!state().handlers.contains_key(&copy));
        call(&mut ctx, cb_value_dtor, copy, 0);
        call(&mut ctx, cb_value_dtor, answer, 0);
        call(&mut ctx, cb_value_dtor, root, 0);
        assert!(!state().handlers.contains_key(&root));
        assert!(!state().handlers.contains_key(&a));
    }

    #[test]
    fn nids_match_names() {
        let _serial = serial();
        for &(nid, name, _) in EXPORTS {
            assert_eq!(ps5nid::compute_nid(name), nid, "NID mismatch for {name}");
        }
    }

    #[test]
    fn parser_number_types_and_syntax() {
        let _serial = serial();
        let p = |s: &str| parse_document(s.as_bytes());
        // Non-negative integers are unsigned, negative ones signed,
        // integers that do not fit and fractions are reals.
        assert_eq!(p("42"), Some(Tree::UInteger(42)));
        assert_eq!(p("-42"), Some(Tree::Integer(-42)));
        assert_eq!(p("-0"), Some(Tree::Integer(0)));
        assert_eq!(p("18446744073709551615"), Some(Tree::UInteger(u64::MAX)));
        assert_eq!(p("18446744073709551616"), Some(Tree::Real(18446744073709551616.0)));
        assert_eq!(p("-9223372036854775809"), Some(Tree::Real(-9223372036854775809.0)));
        assert_eq!(p("1e3"), Some(Tree::Real(1000.0)));
        assert_eq!(p("2.5"), Some(Tree::Real(2.5)));
        assert_eq!(p("1e400"), None);
        for bad in ["", "01", "1.", ".5", "+1", "-", "[1,]", "{\"a\":1,}", "tru", "nul", "[1 2]", "'a'"] {
            assert_eq!(p(bad), None, "{bad}");
        }
        // A NUL ends the input; a UTF-8 BOM is skipped, a broken one is not.
        assert_eq!(parse_document(b"{\"a\":1}\0garbage"), p("{\"a\":1}"));
        assert_eq!(parse_document(b"\0"), None);
        assert_eq!(parse_document(b"[1,\0 2]"), None);
        assert_eq!(parse_document(b"\xEF\xBB\xBF[true]"), Some(Tree::Array(vec![Tree::Boolean(true)])));
        assert_eq!(parse_document(b"\xEF\xBB[true]"), None);
        // Nesting deeper than 64 levels is fine.
        let deep = format!("{}{}", "[".repeat(200), "]".repeat(200));
        assert!(p(&deep).is_some());
    }

    #[test]
    fn parser_strings_and_duplicates() {
        let _serial = serial();
        let s = |doc: &[u8]| match parse_document(doc) {
            Some(Tree::String(v)) => Some(v),
            other => {
                assert_eq!(other, None);
                None
            }
        };
        assert_eq!(s(br#""a\"\\\/\b\f\n\r\t""#).unwrap(), b"a\"\\/\x08\x0c\n\r\t");
        assert_eq!(s(br#""\u00e9\uD83D\uDE00\u0000""#).unwrap(), "\u{e9}\u{1F600}\0".as_bytes());
        assert_eq!(s(br#""\uD83D""#), None); // lone high surrogate
        assert_eq!(s(br#""\uDE00""#), None); // lone low surrogate
        assert_eq!(s(br#""\uD83Dx""#), None);
        assert_eq!(s(b"\"\x01\""), None);
        assert_eq!(s(b"\"\xC3\""), None); // truncated UTF-8
        assert_eq!(s(b"\"\xC0\xAF\""), None); // overlong
        assert_eq!(s(br#""\x""#), None);
        // Last duplicate wins; members are kept in key order.
        let v = parse_document(br#"{"b": 1, "a": 2, "b": [3]}"#).unwrap();
        let Tree::Object(m) = v else { panic!() };
        assert_eq!(m.keys().cloned().collect::<Vec<_>>(), vec![b"a".to_vec(), b"b".to_vec()]);
        assert_eq!(m[&b"b".to_vec()], Tree::Array(vec![Tree::UInteger(3)]));
    }

    #[test]
    fn serialization_format() {
        let _serial = serial();
        let mut out = Vec::new();
        let t = parse_document(br#"{"z": -1, "a": [1.5, 2, true, null, "q\"x"], "m": {}}"#).unwrap();
        serialize_tree(&t, &mut out);
        // Keys sorted, reals `%f`, strings verbatim (no re-escaping).
        assert_eq!(String::from_utf8(out).unwrap(), r#"{"a":[1.500000,2,true,null,"q"x"],"m":{},"z":-1}"#);
        assert_eq!(format_real(1e20), "100000000000000000000.000000");
        assert_eq!(format_real(-0.0), "-0.000000");
        assert_eq!(format_real(f64::INFINITY), "inf");
        assert_eq!(format_real(f64::NAN), "nan");
        let mut out = Vec::new();
        serialize_tree(&Tree::Other(9), &mut out);
        assert_eq!(out, b"null");
    }

    #[test]
    fn parse_into_guest_values() {
        let _serial = serial();
        let vm = ps5vm::VirtualMemory::new();
        const BASE: u64 = 0x5D80_0000_0000;
        vm.map(BASE, 0x4000, ps5vm::MemProtect::RW).unwrap();
        let mut ctx = CpuContext::new(&vm);
        let (value, buf, key, out) = (BASE, BASE + 0x1000, BASE + 0x100, BASE + 0x200);
        call(&mut ctx, cb_value_ctor, value, 0);
        let doc = b"{\"version\": 42, \"neg\": -3, \"name\": \"silent\", \"ok\": true, \"list\": [10, 2.5]}\0";
        assert_eq!(parse_into(&mut ctx, value, buf, doc), 0);
        assert_eq!(ctx.read_u32(value + 0x1C), Some(TYPE_OBJECT));
        assert_eq!(call(&mut ctx, cb_value_count, value, 0), 5);

        // operator[](const char*) hands out the member itself.
        ctx.write_bytes(BASE + 0x800, b"version\0");
        let ver = call(&mut ctx, cb_value_index_key, value, BASE + 0x800);
        assert_eq!(call(&mut ctx, cb_value_index_key, value, BASE + 0x800), ver);
        assert_eq!(call(&mut ctx, cb_value_get_type, ver, 0), u64::from(TYPE_UINTEGER));
        assert_eq!(ctx.read_u64(ver), Some(value)); // parent link
        let slot = call(&mut ctx, cb_value_get_integer, ver, 0);
        assert_eq!(slot, ver + 0x10);
        assert_eq!(ctx.read_u64(slot), Some(42));
        // Wrong-type scalar getters point at a shared zero.
        let zero = call(&mut ctx, cb_value_get_real, ver, 0);
        assert_ne!(zero, ver + 0x10);
        assert_eq!(ctx.read_u64(zero), Some(0));
        assert_eq!(call(&mut ctx, cb_value_get_boolean, ver, 0), zero);

        ctx.write_bytes(BASE + 0x800, b"neg\0");
        let neg = call(&mut ctx, cb_value_index_key, value, BASE + 0x800);
        assert_eq!(call(&mut ctx, cb_value_get_type, neg, 0), u64::from(TYPE_INTEGER));

        // Missing keys give the shared null value; referValue gives NULL.
        ctx.write_bytes(BASE + 0x800, b"missing\0");
        let missing = call(&mut ctx, cb_value_index_key, value, BASE + 0x800);
        assert_eq!(call(&mut ctx, cb_value_get_type, missing, 0), 0);
        call(&mut ctx, cb_string_cstring_ctor, key, BASE + 0x800);
        assert_eq!(call(&mut ctx, cb_value_refer_value, value, key), 0);
        ctx.write_bytes(BASE + 0x800, b"ok\0");
        call(&mut ctx, cb_string_cstring_ctor, key, BASE + 0x800);
        let ok = call(&mut ctx, cb_value_refer_value, value, key);
        assert_eq!(ctx.read_u32(ok + 0x1C), Some(TYPE_BOOLEAN));
        let b = call(&mut ctx, cb_value_get_boolean, ok, 0);
        assert_eq!(ctx.read_u64(b), Some(1));

        // Array elements by position; out of range -> shared null.
        ctx.write_bytes(BASE + 0x800, b"list\0");
        let list = call(&mut ctx, cb_value_index_key, value, BASE + 0x800);
        let real = call(&mut ctx, cb_value_index_position, list, 1);
        assert_eq!(ctx.read_u32(real + 0x1C), Some(TYPE_REAL));
        let r = call(&mut ctx, cb_value_get_real, real, 0);
        assert_eq!(ctx.read_u64(r), Some(2.5f64.to_bits()));
        assert_eq!(call(&mut ctx, cb_value_index_position, list, 9), missing);

        // toString: string text verbatim, other values serialised.
        ctx.write_bytes(BASE + 0x800, b"name\0");
        let name = call(&mut ctx, cb_value_index_key, value, BASE + 0x800);
        call(&mut ctx, cb_string_ctor, out, 0);
        assert_eq!(call(&mut ctx, cb_value_to_string, name, out), 0);
        let p = call(&mut ctx, cb_string_c_str, out, 0);
        assert_eq!(ctx.read_cstring(p, 32).unwrap(), "silent");
        assert_eq!(call(&mut ctx, cb_value_to_string, list, out), 0);
        assert_eq!(text(out), "[10,2.500000]");
        assert_eq!(call(&mut ctx, cb_value_serialize, value, 0), 0);

        // Errors: NULL arguments, invalid text, empty input; the target
        // keeps its value on failure.
        assert_eq!(parse_into(&mut ctx, 0, buf, b"1"), ERR_JSON_INVALID_ARGUMENT);
        ctx.set(Rdx, 1);
        assert_eq!(call(&mut ctx, cb_parser_parse, value, 0) as i64, ERR_JSON_INVALID_ARGUMENT);
        assert_eq!(parse_into(&mut ctx, value, buf, b"{oops}"), ERR_JSON_PARSE_INVALID_CHAR);
        assert_eq!(parse_into(&mut ctx, value, buf, b""), ERR_JSON_PARSE_INVALID_CHAR);
        assert_eq!(ctx.read_u32(value + 0x1C), Some(TYPE_OBJECT));
        // A size past the terminator stops at the NUL.
        ctx.write_bytes(buf, b"[7]\0");
        ctx.set(Rdx, 0x100);
        assert_eq!(call(&mut ctx, cb_parser_parse, value, buf), 0);
        assert_eq!(serialized(&mut ctx, value, out), "[7]");
        call(&mut ctx, cb_value_dtor, value, 0);
        assert_eq!(ctx.read_bytes(value, 0x20).unwrap(), vec![0; 0x20]);
    }

    #[test]
    fn constructors_setters_and_copies() {
        let _serial = serial();
        let vm = ps5vm::VirtualMemory::new();
        const BASE: u64 = 0x5D81_0000_0000;
        vm.map(BASE, 0x2000, ps5vm::MemProtect::RW).unwrap();
        let mut ctx = CpuContext::new(&vm);
        let (v, w, out) = (BASE, BASE + 0x40, BASE + 0x100);
        // Constructors write the 32-byte value.
        ctx.write_bytes(v, &[0xAA; 0x20]);
        assert_eq!(call(&mut ctx, cb_value_int_ctor, v, (-7i64) as u64), v);
        assert_eq!(ctx.read_u64(v + 0x10), Some((-7i64) as u64));
        assert_eq!(ctx.read_u32(v + 0x1C), Some(TYPE_INTEGER));
        assert_eq!(ctx.read_u64(v), Some(0));
        // Raw types outside 0..=7 are kept and serialise as null.
        call(&mut ctx, cb_value_set_type, v, 9);
        assert_eq!(call(&mut ctx, cb_value_get_type, v, 0), 9);
        assert_eq!(serialized(&mut ctx, v, out), "null");
        call(&mut ctx, cb_value_set_type, v, u64::from(TYPE_REAL));
        assert_eq!(serialized(&mut ctx, v, out), "0.000000");
        call(&mut ctx, cb_value_set_type, v, u64::from(TYPE_STRING));
        assert_eq!(serialized(&mut ctx, v, out), "\"\"");

        // Assignment from NULL leaves the target alone; copy construction
        // from NULL yields null.
        call(&mut ctx, cb_value_set_uint, v, 5);
        assert_eq!(call(&mut ctx, cb_value_assign, v, 0), v);
        assert_eq!(serialized(&mut ctx, v, out), "5");
        ctx.write_bytes(w, &[0xAA; 0x20]);
        call(&mut ctx, cb_value_copy_ctor, w, 0);
        assert_eq!(ctx.read_bytes(w, 0x20).unwrap(), vec![0; 0x20]);
        call(&mut ctx, cb_value_copy_ctor, w, v);
        assert_eq!(serialized(&mut ctx, w, out), "5");

        // clear() keeps the parent and root links.
        ctx.write_u64(w, 0x1234);
        ctx.write_u64(w + 8, 0x5678);
        call(&mut ctx, cb_value_clear, w, 0);
        assert_eq!(ctx.read_u64(w), Some(0x1234));
        assert_eq!(ctx.read_u64(w + 8), Some(0x5678));
        assert_eq!(ctx.read_u32(w + 0x1C), Some(TYPE_NULL));
        call(&mut ctx, cb_value_dtor, w, 0);
        assert_eq!(ctx.read_u64(w), Some(0));

        // getString on another type -> shared empty string; on a string
        // value -> the value's own String.
        let empty = call(&mut ctx, cb_value_get_string, v, 0);
        assert_eq!(call(&mut ctx, cb_string_length, empty, 0), 0);
        ctx.write_bytes(BASE + 0x800, b"hi\0");
        call(&mut ctx, cb_value_cstring_ctor, v, BASE + 0x800);
        let s = call(&mut ctx, cb_value_get_string, v, 0);
        assert_eq!(ctx.read_u64(v + 0x10), Some(s));
        assert_eq!(text(s), "hi");
        call(&mut ctx, cb_value_dtor, v, 0);
        assert!(!state().strings.contains_key(&s));

        // InitParameter2 layout and NULL tolerance.
        let p = BASE + 0x200;
        ctx.write_bytes(p, &[0xFF; 0x28]);
        assert_eq!(call(&mut ctx, cb_init_parameter2_ctor, p, 0), p);
        assert_eq!(ctx.read_bytes(p, 0x28).unwrap(), vec![0; 0x28]);
        ctx.set(Rdx, 0x2222);
        call(&mut ctx, cb_init_parameter2_set_allocator, p, 0x1111);
        call(&mut ctx, cb_init_parameter2_set_file_buffer_size, p, 0x8000);
        call(&mut ctx, cb_init_parameter2_set_special_float_format_type, p, 2);
        assert_eq!(ctx.read_u64(p), Some(0x1111));
        assert_eq!(ctx.read_u64(p + 8), Some(0x2222));
        assert_eq!(ctx.read_u64(p + 0x10), Some(0x8000));
        assert_eq!(ctx.read_u32(p + 0x18), Some(2));
        assert_eq!(call(&mut ctx, cb_init_parameter2_ctor, 0, 0), 0);
        assert_eq!(call(&mut ctx, cb_return_ok, 0, 0), 0);
        // c_str of NULL is an empty string.
        let z = call(&mut ctx, cb_string_c_str, 0, 0);
        assert_eq!(ctx.read_cstring(z, 4).unwrap(), "");
    }

    #[test]
    fn build_and_iterate_containers() {
        let _serial = serial();
        let vm = ps5vm::VirtualMemory::new();
        const BASE: u64 = 0x5D82_0000_0000;
        vm.map(BASE, 0x4000, ps5vm::MemProtect::RW).unwrap();
        let mut ctx = CpuContext::new(&vm);
        let (obj, arr, val, tmp, key, out) = (BASE, BASE + 0x100, BASE + 0x200, BASE + 0x300, BASE + 0x400, BASE + 0x500);

        // Object o; o["name"] = Value("x"); o["n"] = Value(-3);
        call(&mut ctx, cb_object_ctor, obj, 0);
        ctx.write_bytes(BASE + 0x1000, b"name\0x\0n\0list\0");
        call(&mut ctx, cb_string_cstring_ctor, key, BASE + 0x1000);
        assert_eq!(call(&mut ctx, cb_object_empty, obj, 0), 1);
        assert_eq!(call(&mut ctx, cb_object_size, obj, 0), 0);
        let slot = call(&mut ctx, cb_object_index, obj, key);
        assert_eq!(call(&mut ctx, cb_object_empty, obj, 0), 0);
        assert_eq!(call(&mut ctx, cb_object_index, obj, key), slot);
        assert_eq!(ctx.read_u32(slot + 0x1C), Some(TYPE_NULL));
        call(&mut ctx, cb_value_cstring_ctor, tmp, BASE + 0x1005);
        call(&mut ctx, cb_value_assign, slot, tmp);
        assert_eq!(ctx.read_u32(slot + 0x1C), Some(TYPE_STRING));
        call(&mut ctx, cb_string_cstring_ctor, key, BASE + 0x1007);
        let n = call(&mut ctx, cb_object_index, obj, key);
        call(&mut ctx, cb_value_int_ctor, tmp, (-3i64) as u64);
        call(&mut ctx, cb_value_assign, n, tmp);
        assert_eq!(call(&mut ctx, cb_object_size, obj, 0), 2);
        // Object::operator[] on NULL gives the shared null value.
        let null = call(&mut ctx, cb_object_index, 0, key);
        assert_eq!(call(&mut ctx, cb_value_get_type, null, 0), 0);

        // Array a; a.push_back(Value(true)); a.push_back(Value(o)); a.push_back(NULL)
        call(&mut ctx, cb_array_ctor, arr, 0);
        assert_eq!(call(&mut ctx, cb_array_empty, arr, 0), 1);
        call(&mut ctx, cb_value_bool_ctor, tmp, 1);
        call(&mut ctx, cb_array_push_back, arr, tmp);
        call(&mut ctx, cb_value_object_ctor, tmp, obj);
        call(&mut ctx, cb_array_push_back, arr, tmp);
        call(&mut ctx, cb_array_push_back, arr, 0);
        assert_eq!(call(&mut ctx, cb_array_size, arr, 0), 3);
        let back = call(&mut ctx, cb_array_back, arr, 0);
        assert_eq!(call(&mut ctx, cb_value_get_type, back, 0), 0);

        // o["list"] = Value(a); Value v(o);
        call(&mut ctx, cb_string_cstring_ctor, key, BASE + 0x1009);
        let list = call(&mut ctx, cb_object_index, obj, key);
        call(&mut ctx, cb_value_array_ctor, tmp, arr);
        call(&mut ctx, cb_value_assign, list, tmp);
        call(&mut ctx, cb_value_object_ctor, val, obj);
        let expected = r#"{"list":[true,{"n":-3,"name":"x"},null],"n":-3,"name":"x"}"#;
        assert_eq!(serialized(&mut ctx, val, out), expected);
        assert_eq!(call(&mut ctx, cb_value_count, val, 0), 3);

        // Copies are deep.
        call(&mut ctx, cb_object_clear, obj, 0);
        call(&mut ctx, cb_object_dtor, obj, 0);
        call(&mut ctx, cb_array_dtor, arr, 0);
        assert_eq!(serialized(&mut ctx, val, out), expected);

        // Iterate the value's object: sorted keys, member blocks.
        let o = call(&mut ctx, cb_value_refer_object, val, 0);
        assert_eq!(call(&mut ctx, cb_value_get_object, val, 0), o);
        assert_eq!(call(&mut ctx, cb_value_refer_array, val, 0), 0);
        let empty = call(&mut ctx, cb_value_get_array, val, 0);
        assert_eq!(call(&mut ctx, cb_array_size, empty, 0), 0);
        let (it, end) = (BASE + 0x600, BASE + 0x608);
        call(&mut ctx, cb_object_begin, it, o);
        call(&mut ctx, cb_object_end, end, o);
        let mut keys = Vec::new();
        while call(&mut ctx, cb_iterator_ne, it, end) == 1 {
            let pair = call(&mut ctx, cb_iterator_deref, it, 0);
            let p = call(&mut ctx, cb_string_c_str, pair, 0);
            keys.push(ctx.read_cstring(p, 16).unwrap());
            call(&mut ctx, cb_iterator_increment, it, 0);
        }
        assert_eq!(keys, vec!["list", "n", "name"]);

        // set(Object) from a member of the value itself.
        ctx.write_bytes(BASE + 0x1000, b"list\0");
        let inner = call(&mut ctx, cb_value_index_key, val, BASE + 0x1000);
        let arr2 = call(&mut ctx, cb_value_refer_array, inner, 0);
        let (ai, ae) = (BASE + 0x700, BASE + 0x708);
        call(&mut ctx, cb_array_begin, ai, arr2);
        call(&mut ctx, cb_array_end, ae, arr2);
        let mut types = Vec::new();
        while call(&mut ctx, cb_iterator_ne, ai, ae) == 1 {
            let e = call(&mut ctx, cb_iterator_deref, ai, 0);
            assert_eq!(ctx.read_u64(e), Some(inner)); // parent link
            types.push(call(&mut ctx, cb_value_get_type, e, 0));
            call(&mut ctx, cb_iterator_increment, ai, 0);
        }
        assert_eq!(types, vec![1, 7, 0]);
        let e1 = call(&mut ctx, cb_value_index_position, inner, 1);
        let o1 = call(&mut ctx, cb_value_refer_object, e1, 0);
        call(&mut ctx, cb_value_set_object, val, o1);
        assert_eq!(serialized(&mut ctx, val, out), r#"{"n":-3,"name":"x"}"#);
        call(&mut ctx, cb_value_dtor, val, 0);
        assert!(!state().objects.contains_key(&o));
    }

    /// The u64 behind the pointer `f(rdi)` returns.
    fn read_at(ctx: &mut CpuContext, f: HleCallback, rdi: u64) -> Option<u64> {
        let p = call(ctx, f, rdi, 0);
        ctx.read_u64(p)
    }

    fn json_vm(base: u64) -> ps5vm::VirtualMemory {
        let vm = ps5vm::VirtualMemory::new();
        vm.map(base, 0x4000, ps5vm::MemProtect::RW).unwrap();
        vm
    }

    /// A String constructed from `text` at `addr` (text staged at `scratch`).
    fn make_string(ctx: &mut CpuContext, addr: u64, scratch: u64, text: &str) {
        let mut bytes = text.as_bytes().to_vec();
        bytes.push(0);
        ctx.write_bytes(scratch, &bytes);
        call(ctx, cb_string_cstring_ctor, addr, scratch);
    }

    /// find/rfind/substr/append/reserve/clear/compare follow std::string
    /// (reference GuestJson StringMethods and ObjectFind).
    #[test]
    fn string_methods() {
        let _serial = serial();
        const BASE: u64 = 0x5DB0_0000_0000;
        let vm = json_vm(BASE);
        let mut ctx = CpuContext::new(&vm);
        let (base, suffix, part, scratch) = (BASE, BASE + 0x10, BASE + 0x20, BASE + 0x1000);
        make_string(&mut ctx, base, scratch, "a/b/c");
        make_string(&mut ctx, suffix, scratch, "-tail");
        let find = |ctx: &mut CpuContext, f: HleCallback, ch: u8, pos: u64| {
            ctx.set(Rdx, pos);
            call(ctx, f, base, u64::from(ch))
        };
        assert_eq!(find(&mut ctx, cb_string_find, b'/', 0), 1);
        assert_eq!(find(&mut ctx, cb_string_find, b'/', 2), 3);
        assert_eq!(find(&mut ctx, cb_string_find, b'z', 0), NPOS);
        assert_eq!(find(&mut ctx, cb_string_find, b'/', 99), NPOS);
        assert_eq!(find(&mut ctx, cb_string_rfind, b'/', NPOS), 3);
        assert_eq!(find(&mut ctx, cb_string_rfind, b'/', 2), 1);
        assert_eq!(find(&mut ctx, cb_string_rfind, b'a', 0), 0);
        assert_eq!(find(&mut ctx, cb_string_rfind, b'z', NPOS), NPOS);

        let substr = |ctx: &mut CpuContext, pos: u64, len: u64| {
            ctx.set(Rdx, pos);
            ctx.set(Rcx, len);
            assert_eq!(call(ctx, cb_string_substr, part, base), part);
            text(part)
        };
        assert_eq!(substr(&mut ctx, 2, 3), "b/c");
        assert_eq!(substr(&mut ctx, 1, NPOS), "/b/c");
        assert_eq!(substr(&mut ctx, 5, NPOS), "");
        assert_eq!(substr(&mut ctx, 6, 1), "", "a position past the end is refused");

        call(&mut ctx, cb_string_reserve, base, 64);
        assert_eq!(text(base), "a/b/c");
        assert_eq!(call(&mut ctx, cb_string_append_string, base, suffix), base);
        ctx.write_bytes(scratch, b"!\0");
        assert_eq!(call(&mut ctx, cb_string_append_cstr, base, scratch), base);
        assert_eq!(text(base), "a/b/c-tail!");
        call(&mut ctx, cb_string_append_string, base, base);
        assert_eq!(text(base), "a/b/c-tail!a/b/c-tail!");
        assert_eq!(call(&mut ctx, cb_string_append_cstr, base, 0), base, "a NULL text changes nothing");
        assert_eq!(text(base), "a/b/c-tail!a/b/c-tail!");
        call(&mut ctx, cb_string_clear, base, 0);
        assert_eq!(text(base), "");
        assert_eq!(call(&mut ctx, cb_string_empty, base, 0), 1);

        make_string(&mut ctx, base, scratch, "beta");
        for (other, sign) in [("beta", 0i64), ("betb", -1), ("bet", 1), ("alpha", 1)] {
            make_string(&mut ctx, part, scratch, other);
            let got = call(&mut ctx, cb_string_compare, base, scratch) as i64 as i32;
            assert_eq!(i64::from(got.signum()), sign, "compare with {other}");
        }
        assert_ne!(call(&mut ctx, cb_string_compare, base, 0), 0, "NULL never compares equal");
        for s in [base, suffix, part] {
            call(&mut ctx, cb_string_dtor, s, 0);
        }
    }

    /// Default iterators are singular; assignment, postfix ++, advance and
    /// operator-> walk the elements (reference GuestJson ArrayIteratorCopies).
    #[test]
    fn array_iterator_copies() {
        let _serial = serial();
        const BASE: u64 = 0x5DB1_0000_0000;
        let vm = json_vm(BASE);
        let mut ctx = CpuContext::new(&vm);
        let (arr, item, it, begin, end, prev) = (BASE, BASE + 0x40, BASE + 0x100, BASE + 0x108, BASE + 0x110, BASE + 0x118);
        call(&mut ctx, cb_array_ctor, arr, 0);
        for i in 1..=3u64 {
            call(&mut ctx, cb_value_int_ctor, item, i);
            call(&mut ctx, cb_array_push_back, arr, item);
            call(&mut ctx, cb_value_dtor, item, 0);
        }
        ctx.write_u64(it, 0x5A5A_5A5A_5A5A_5A5A);
        assert_eq!(call(&mut ctx, cb_iterator_default_ctor, it, 0), it);
        assert_eq!(ctx.read_u64(it), Some(0));
        call(&mut ctx, cb_array_begin, begin, arr);
        call(&mut ctx, cb_array_end, end, arr);
        assert_eq!(call(&mut ctx, cb_iterator_copy, it, begin), it);
        assert_eq!(call(&mut ctx, cb_iterator_ne, it, begin), 0);
        let mut seen = Vec::new();
        while call(&mut ctx, cb_iterator_ne, it, end) == 1 {
            assert_eq!(call(&mut ctx, cb_iterator_post_increment, prev, it), prev);
            let v = call(&mut ctx, cb_iterator_deref, prev, 0);
            assert_eq!(call(&mut ctx, cb_iterator_deref, prev, 0), v, "operator-> is operator*");
            seen.push(read_at(&mut ctx, cb_value_get_integer, v).unwrap());
            assert_eq!(call(&mut ctx, cb_iterator_ne, prev, it), 1);
        }
        assert_eq!(seen, vec![1, 2, 3]);
        let last = call(&mut ctx, cb_iterator_deref, prev, 0);
        assert_eq!(read_at(&mut ctx, cb_value_get_integer, last), Some(3));
        // advance(n) and the copy constructor.
        call(&mut ctx, cb_iterator_copy, it, begin);
        call(&mut ctx, cb_iterator_advance, it, 2);
        assert_eq!(call(&mut ctx, cb_iterator_ne, it, prev), 0);
        call(&mut ctx, cb_iterator_advance, it, 1);
        assert_eq!(call(&mut ctx, cb_iterator_ne, it, end), 0);
        assert_eq!(call(&mut ctx, cb_return_void, it, 0), 0);
        call(&mut ctx, cb_array_dtor, arr, 0);
    }

    /// Array copies are deep; clear empties only the cleared array; front
    /// is the first element and, on an empty array, the shared null value.
    #[test]
    fn array_copy_clear_front() {
        let _serial = serial();
        const BASE: u64 = 0x5DB2_0000_0000;
        let vm = json_vm(BASE);
        let mut ctx = CpuContext::new(&vm);
        let (arr, copy, item, it) = (BASE, BASE + 0x10, BASE + 0x40, BASE + 0x100);
        call(&mut ctx, cb_array_ctor, arr, 0);
        call(&mut ctx, cb_value_bool_ctor, item, 1);
        call(&mut ctx, cb_array_push_back, arr, item);
        ctx.set_return_f64(0.0);
        call(&mut ctx, cb_value_uint_ctor, item, 7);
        call(&mut ctx, cb_array_push_back, arr, item);
        let front = call(&mut ctx, cb_array_front, arr, 0);
        assert_eq!(read_at(&mut ctx, cb_value_get_boolean, front).map(|b| b & 0xFF), Some(1));
        call(&mut ctx, cb_array_begin, it, arr);
        assert_eq!(call(&mut ctx, cb_iterator_deref, it, 0), front);
        assert_eq!(call(&mut ctx, cb_array_copy_ctor, copy, arr), copy);
        assert_eq!(call(&mut ctx, cb_array_clear, arr, 0), 0);
        assert_eq!(call(&mut ctx, cb_array_size, arr, 0), 0);
        let null = call(&mut ctx, cb_array_front, arr, 0);
        assert_eq!(call(&mut ctx, cb_value_get_type, null, 0), u64::from(TYPE_NULL));
        assert_eq!(call(&mut ctx, cb_array_size, copy, 0), 2);
        let f = call(&mut ctx, cb_array_front, copy, 0);
        assert_ne!(f, front);
        assert_eq!(call(&mut ctx, cb_value_get_type, f, 0), u64::from(TYPE_BOOLEAN));
        let b = call(&mut ctx, cb_array_back, copy, 0);
        assert_eq!(read_at(&mut ctx, cb_value_get_uinteger, b), Some(7));
        call(&mut ctx, cb_array_dtor, copy, 0);
        call(&mut ctx, cb_array_dtor, arr, 0);
        assert!(!state().arrays.contains_key(&copy));
    }

    /// find gives the member's iterator or end(); operator-> is operator*
    /// (reference GuestJson ObjectFind).
    #[test]
    fn object_find() {
        let _serial = serial();
        const BASE: u64 = 0x5DB3_0000_0000;
        let vm = json_vm(BASE);
        let mut ctx = CpuContext::new(&vm);
        let (obj, alpha, beta, gamma, found, end, tmp, scratch) =
            (BASE, BASE + 0x10, BASE + 0x20, BASE + 0x30, BASE + 0x100, BASE + 0x108, BASE + 0x200, BASE + 0x1000);
        call(&mut ctx, cb_object_ctor, obj, 0);
        make_string(&mut ctx, alpha, scratch, "alpha");
        make_string(&mut ctx, beta, scratch, "beta");
        make_string(&mut ctx, gamma, scratch, "gamma");
        call(&mut ctx, cb_object_end, end, obj);
        ctx.set(Rdx, alpha);
        call(&mut ctx, cb_object_find, found, obj);
        assert_eq!(call(&mut ctx, cb_iterator_ne, found, end), 0, "empty object");

        let first = call(&mut ctx, cb_object_index, obj, alpha);
        call(&mut ctx, cb_value_set_int, first, 1);
        let second = call(&mut ctx, cb_object_index, obj, beta);
        call(&mut ctx, cb_value_cstring_ctor, tmp, scratch);
        call(&mut ctx, cb_value_assign, second, tmp);
        call(&mut ctx, cb_object_end, end, obj);
        ctx.set(Rdx, beta);
        assert_eq!(call(&mut ctx, cb_object_find, found, obj), found);
        assert_eq!(call(&mut ctx, cb_iterator_ne, found, end), 1);
        let pair = call(&mut ctx, cb_iterator_deref, found, 0);
        assert_eq!(pair + MEMBER_VALUE_OFFSET, second);
        assert_eq!(text(pair), "beta");
        call(&mut ctx, cb_iterator_increment, found, 0);
        assert_eq!(call(&mut ctx, cb_iterator_ne, found, end), 0);
        ctx.set(Rdx, alpha);
        call(&mut ctx, cb_object_find, found, obj);
        assert_eq!(call(&mut ctx, cb_iterator_deref, found, 0) + MEMBER_VALUE_OFFSET, first);
        ctx.set(Rdx, gamma);
        call(&mut ctx, cb_object_find, found, obj);
        assert_eq!(call(&mut ctx, cb_iterator_ne, found, end), 0, "missing key");
        assert_eq!(call(&mut ctx, cb_object_size, obj, 0), 2);
        call(&mut ctx, cb_value_dtor, tmp, 0);
        call(&mut ctx, cb_object_dtor, obj, 0);
    }

    /// Pair(key, value) deep-copies both into the member block layout; the
    /// destructor releases them (reference GuestJson ObjectPair).
    #[test]
    fn object_pair() {
        let _serial = serial();
        const BASE: u64 = 0x5DB4_0000_0000;
        let vm = json_vm(BASE);
        let mut ctx = CpuContext::new(&vm);
        let (key, value, pair, scratch) = (BASE, BASE + 0x40, BASE + 0x100, BASE + 0x1000);
        make_string(&mut ctx, key, scratch, "answer");
        ctx.write_bytes(scratch + 0x100, b"forty-two\0");
        call(&mut ctx, cb_value_cstring_ctor, value, scratch + 0x100);
        ctx.write_bytes(pair, &[0x5A; 0x28]);
        ctx.set(Rdx, value);
        assert_eq!(call(&mut ctx, cb_pair_ctor, pair, key), pair);
        let pv = pair + MEMBER_VALUE_OFFSET;
        assert_eq!(text(pair), "answer");
        assert_eq!(call(&mut ctx, cb_value_get_type, pv, 0), u64::from(TYPE_STRING));
        let s = call(&mut ctx, cb_value_get_string, pv, 0);
        assert_ne!(s, call(&mut ctx, cb_value_get_string, value, 0), "the value is copied");
        assert_eq!(text(s), "forty-two");
        call(&mut ctx, cb_value_set_int, value, 7);
        call(&mut ctx, cb_string_dtor, key, 0);
        call(&mut ctx, cb_value_dtor, value, 0);
        assert_eq!(text(pair), "answer");
        assert_eq!(call(&mut ctx, cb_value_get_type, pv, 0), u64::from(TYPE_STRING));
        assert_eq!(call(&mut ctx, cb_pair_dtor, pair, 0), 0);
        assert!(!state().strings.contains_key(&pair) && !state().strings.contains_key(&s));
        assert_eq!(call(&mut ctx, cb_value_get_type, pv, 0), u64::from(TYPE_NULL));
    }

    /// getValue(String) / referValue(size_t) on a parsed document (reference
    /// GuestJson ValueAccess).
    #[test]
    fn value_access_by_string_and_index() {
        let _serial = serial();
        const BASE: u64 = 0x5DB5_0000_0000;
        let vm = json_vm(BASE);
        let mut ctx = CpuContext::new(&vm);
        let (root, key, missing, buf, scratch) = (BASE, BASE + 0x40, BASE + 0x50, BASE + 0x1000, BASE + 0x2000);
        call(&mut ctx, cb_value_ctor, root, 0);
        assert_eq!(parse_into(&mut ctx, root, buf, br#"{"list":[10,20,30],"name":"value","flag":true}"#), 0);
        make_string(&mut ctx, key, scratch, "list");
        let list = call(&mut ctx, cb_value_get_value_key, root, key);
        assert_eq!(call(&mut ctx, cb_value_get_type, list, 0), u64::from(TYPE_ARRAY));
        let second = call(&mut ctx, cb_value_index_position, list, 1);
        assert_eq!(read_at(&mut ctx, cb_value_get_integer, second), Some(20));
        let mutable_list = call(&mut ctx, cb_value_refer_value, root, key);
        assert_eq!(mutable_list, list);
        let element = call(&mut ctx, cb_value_refer_value_index, mutable_list, 0);
        assert_ne!(element, 0);
        call(&mut ctx, cb_value_set_int, element, 15);
        let first = call(&mut ctx, cb_value_index_position, list, 0);
        assert_eq!(read_at(&mut ctx, cb_value_get_integer, first), Some(15));
        assert_eq!(call(&mut ctx, cb_value_refer_value_index, mutable_list, 3), 0, "out of range");
        assert_eq!(call(&mut ctx, cb_value_refer_value_index, root, 0), 0, "not an array");
        make_string(&mut ctx, missing, scratch, "missing");
        let null = call(&mut ctx, cb_value_get_value_key, root, missing);
        assert_eq!(call(&mut ctx, cb_value_get_type, null, 0), u64::from(TYPE_NULL));
        assert_eq!(call(&mut ctx, cb_value_get_value_key, root, 0), null, "NULL key");
        assert_eq!(call(&mut ctx, cb_value_get_value_key, list, key), null, "not an object");
        call(&mut ctx, cb_value_dtor, root, 0);
    }

    /// The null access answer is used only when non-NULL and of the
    /// requested type (any type for a null request); the callback gets the
    /// requested type, the parent and the context. The setters store the
    /// callback and terminate forgets it (reference GuestJson NullAccess).
    #[test]
    fn null_access_callback() {
        let _serial = serial();
        const BASE: u64 = 0x5DB6_0000_0000;
        let vm = json_vm(BASE);
        let mut ctx = CpuContext::new(&vm);
        let (fallback, parent) = (BASE, BASE + 0x40);
        call(&mut ctx, cb_value_int_ctor, fallback, 99);
        let mut calls = Vec::new();
        let mut answer = |f: u64, args: [u64; 6]| {
            calls.push((f, args[0], args[1], args[2]));
            fallback
        };
        let handler = (0xC0DE, 0xC0E7);
        assert_eq!(null_access_via(&ctx, handler, TYPE_NULL, parent, &mut answer), Some(fallback));
        assert_eq!(null_access_via(&ctx, handler, TYPE_INTEGER, parent, &mut answer), Some(fallback));
        assert_eq!(null_access_via(&ctx, handler, TYPE_BOOLEAN, parent, &mut answer), None, "wrong type");
        assert_eq!(calls, vec![(0xC0DE, 0, parent, 0xC0E7), (0xC0DE, 2, parent, 0xC0E7), (0xC0DE, 1, parent, 0xC0E7)]);
        assert_eq!(null_access_via(&ctx, handler, TYPE_NULL, parent, &mut |_, _| 0), None, "NULL answer");
        assert_eq!(null_access_via(&ctx, (0, 0xC0E7), TYPE_NULL, parent, &mut |_, _| panic!("no callback")), None);

        // Only a NULL callback is ever installed here: other tests run
        // getters concurrently.
        call(&mut ctx, cb_value_set_null_access_callback, 0, 0x1234);
        assert_eq!(*NULL_ACCESS.lock().unwrap(), (0, 0x1234));
        ctx.set(Rdx, 0x5678);
        assert_eq!(call(&mut ctx, cb_set_global_null_access_callback, BASE + 0x100, 0), 0);
        assert_eq!(*NULL_ACCESS.lock().unwrap(), (0, 0x5678));
        assert_eq!(call(&mut ctx, cb_initializer_terminate, BASE + 0x100, 0), 0);
        assert_eq!(*NULL_ACCESS.lock().unwrap(), (0, 0));
        call(&mut ctx, cb_value_dtor, fallback, 0);
    }

    /// The read callback's characters are taken until it fails or gives
    /// NUL, then parsed as a whole document; trailing text is refused and
    /// the target keeps its value.
    #[test]
    fn parse_through_read_callback() {
        let _serial = serial();
        const BASE: u64 = 0x5DB7_0000_0000;
        let vm = json_vm(BASE);
        let mut ctx = CpuContext::new(&vm);
        let (value, out) = (BASE, BASE + 0x100);
        let feed = |text: &'static [u8], end_result: i32| {
            let mut pos = 0;
            read_callback_text(move || {
                let r = match text.get(pos) {
                    Some(&c) => (0, c),
                    None => (end_result, 0),
                };
                pos += 1;
                r
            })
        };
        assert_eq!(feed(br#"{"a":[1,2]}"#, -1).unwrap(), br#"{"a":[1,2]}"#);
        assert_eq!(feed(b"[1]\0[2]", -1).unwrap(), b"[1]", "NUL ends the text");
        assert_eq!(feed(b"[3]", 0).unwrap(), b"[3]", "a NUL character with result 0 ends it too");
        call(&mut ctx, cb_value_ctor, value, 0);
        assert_eq!(parse_into_value(&ctx, value, &feed(br#"{"a":[1,2]}"#, -1).unwrap()), 0);
        assert_eq!(serialized(&mut ctx, value, out), r#"{"a":[1,2]}"#);
        assert_eq!(parse_into_value(&ctx, value, &feed(b"[1] x", -1).unwrap()), ERR_JSON_PARSE_INVALID_CHAR);
        assert_eq!(parse_into_value(&ctx, value, &feed(b"", -1).unwrap()), ERR_JSON_PARSE_INVALID_CHAR);
        assert_eq!(serialized(&mut ctx, value, out), r#"{"a":[1,2]}"#);
        // NULL value or callback: refused before any guest call.
        ctx.set(Rdx, 0);
        assert_eq!(call(&mut ctx, cb_parser_parse_callback, value, 0) as i64, ERR_JSON_INVALID_ARGUMENT);
        assert_eq!(call(&mut ctx, cb_parser_parse_callback, 0, 0xC0DE) as i64, ERR_JSON_INVALID_ARGUMENT);
        call(&mut ctx, cb_value_dtor, value, 0);
    }

    /// InitParameterRtti stores its arguments in order; notifyError of the
    /// base allocator touches nothing (reference GuestJson).
    #[test]
    fn init_parameter_rtti_and_notify_error() {
        let _serial = serial();
        const BASE: u64 = 0x5DB8_0000_0000;
        let vm = json_vm(BASE);
        let mut ctx = CpuContext::new(&vm);
        ctx.write_bytes(BASE, &[0xFF; 0x20]);
        ctx.set(Rdx, 0x2222);
        ctx.set(Rcx, 0x8000);
        assert_eq!(call(&mut ctx, cb_init_parameter_rtti_ctor, BASE, 0x1111), BASE);
        assert_eq!((ctx.read_u64(BASE), ctx.read_u64(BASE + 8), ctx.read_u64(BASE + 0x10)), (Some(0x1111), Some(0x2222), Some(0x8000)));
        assert_eq!(ctx.read_u64(BASE + 0x18), Some(u64::MAX), "nothing past the three fields");
        ctx.write_u64(BASE + 0x100, 0x1234);
        ctx.set(Rdx, 64);
        ctx.set(Rcx, BASE + 0x108);
        assert_eq!(call(&mut ctx, cb_mem_allocator_notify_error, BASE + 0x100, (-1i64) as u64), 0);
        assert_eq!(ctx.read_u64(BASE + 0x100), Some(0x1234));
    }
}
