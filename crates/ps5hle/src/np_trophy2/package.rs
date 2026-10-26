// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Trophy package (`sce_sys/trophy2/trophyNN.ucp`) and UDS package
//! (`sce_sys/uds/udsNN.ucp`) parsing, unlock persistence and progress.
//!
//! Both packages share one container format (all integers big-endian):
//!   +0x00 magic 0xB228C60A, +0x04 version 1, +0x08 u64 total size,
//!   +0x10 u32 file count, +0x14 u32 table offset; entries start at
//!   table + 0x20, 0x40 bytes each: name[0x20], u64 offset, u64 length.
//!
//! Trophy definitions come from `tropconf.json`, localized texts from
//! `tropmeta_<locale>.json`, icons from `tropNNNN.png`. UDS rules come from
//! `stats_definition.json` + `stats_extraction.json`.
//!
//! Unlocks are stored per user/service label as
//! `<root>/_Trophies/<TITLE>/trophies_<user>_<label>.json`:
//! `{"unlockedAt":{"<id>":<tick>,...},"unlockedTrophies":[<id>,...]}` where a
//! tick is microseconds since 0001-01-01 UTC.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use crate::json::{parse_json, JsonElem, JsonKind};

pub const PACKAGE_DIRECTORY: &str = "sce_sys/trophy2";
pub const UNIX_EPOCH_TICK: u64 = 62_135_596_800_000_000;
const MAX_TICK: u64 = 315_537_897_599_999_999;
const MAX_UNLOCKS_SIZE: u64 = 1 << 20;
const MAX_PACKAGE_SIZE: u64 = 1 << 30;
const PACKAGE_MAGIC: u64 = 0xB228_C60A;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Comparison {
    #[default]
    None,
    GreaterEqual,
    Greater,
    LessEqual,
    Less,
}

#[derive(Clone, Debug)]
pub struct Trophy {
    pub id: i32,
    pub group_id: i32,
    pub platinum_id: i32,
    /// 1 platinum, 2 gold, 3 silver, 4 bronze.
    pub grade: i32,
    pub target: Option<u64>,
    pub uds_stat_id: Option<u64>,
    pub comparison: Comparison,
    pub progressive: bool,
    pub name: String,
    pub description: String,
    pub reward: String,
    pub icon_png: Vec<u8>,
    pub hidden: bool,
    pub has_reward: bool,
}

impl Default for Trophy {
    fn default() -> Self {
        Self {
            id: 0,
            group_id: -1,
            platinum_id: -1,
            grade: 0,
            target: None,
            uds_stat_id: None,
            comparison: Comparison::None,
            progressive: false,
            name: String::new(),
            description: String::new(),
            reward: String::new(),
            icon_png: Vec::new(),
            hidden: false,
            has_reward: false,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct UdsRule {
    pub stat_id: u64,
    pub input: String,
    pub is_uint32: bool,
}

pub type UdsRules = BTreeMap<String, Vec<UdsRule>>;

/// Integer property value of a UDS event (the only types stats consume).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UdsInteger {
    I32(i32),
    U32(u32),
    U64(u64),
}

#[derive(Clone, Debug, Default)]
pub struct Package {
    pub title: String,
    pub groups: BTreeMap<i32, String>,
    pub trophies: BTreeMap<i32, Trophy>,
    pub event_rules: UdsRules,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct UnlockData {
    pub unlocked: BTreeSet<i32>,
    pub timestamps: BTreeMap<i32, u64>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Progress {
    pub total_grade: [u32; 5],
    pub earned_grade: [u32; 5],
    pub total: u32,
    pub earned: u32,
}

impl Progress {
    /// Gold, silver and bronze weigh 6:2:1; platinum is excluded.
    pub fn percentage(&self) -> u32 {
        let points = self.total_grade[2] * 6 + self.total_grade[3] * 2 + self.total_grade[4];
        let earned = self.earned_grade[2] * 6 + self.earned_grade[3] * 2 + self.earned_grade[4];
        if points == 0 {
            0
        } else {
            earned * 100 / points
        }
    }
}

// ---------------------------------------------------------------------------
// JSON helpers
// ---------------------------------------------------------------------------

/// Result of reading one JSON member file of a package.
enum Doc {
    /// File not present (behaves as JSON null).
    Absent,
    /// Present but not valid JSON.
    Invalid,
    Value(JsonElem),
}

impl Doc {
    fn is_null(&self) -> bool {
        match self {
            Doc::Absent => true,
            Doc::Invalid => false,
            Doc::Value(v) => matches!(v.kind, JsonKind::Null),
        }
    }

    fn value(&self) -> Option<&JsonElem> {
        match self {
            Doc::Value(v) => Some(v),
            _ => None,
        }
    }
}

fn parse_bytes(bytes: &[u8]) -> Option<JsonElem> {
    let bytes = bytes.strip_prefix(b"\xEF\xBB\xBF").unwrap_or(bytes);
    parse_json(bytes)
}

fn read_doc(files: &Files<'_>, name: &str) -> Doc {
    match files.get(name.as_bytes()) {
        None => Doc::Absent,
        Some(data) => parse_bytes(data).map_or(Doc::Invalid, Doc::Value),
    }
}

/// Integral JSON number (no fraction/exponent) as i64 / u64.
enum JsonInt {
    Signed(i64),
    Unsigned(u64),
}

fn json_int(v: &JsonElem) -> Option<JsonInt> {
    if !matches!(v.kind, JsonKind::Number) || v.raw().contains(['.', 'e', 'E']) {
        return None;
    }
    let raw = v.raw();
    if raw.starts_with('-') {
        raw.parse::<i64>().ok().map(JsonInt::Signed)
    } else {
        raw.parse::<u64>().ok().map(JsonInt::Unsigned)
    }
}

/// Decimal text with nothing else (no sign for unsigned, optional '-' for signed).
fn parse_signed_text(text: &str) -> Option<i32> {
    let digits = text.strip_prefix('-').unwrap_or(text);
    if digits.is_empty() || !digits.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    text.parse::<i32>().ok()
}

fn parse_unsigned_text(text: &str) -> Option<u64> {
    if text.is_empty() || !text.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    text.parse::<u64>().ok()
}

/// Non-negative id from an integer or a decimal string; -1 when invalid.
fn read_id(v: &JsonElem) -> i32 {
    match &v.kind {
        JsonKind::Number => match json_int(v) {
            Some(JsonInt::Signed(n)) if (0..=i32::MAX as i64).contains(&n) => n as i32,
            Some(JsonInt::Unsigned(n)) if n <= i32::MAX as u64 => n as i32,
            _ => -1,
        },
        JsonKind::String(s) => read_id_text(s),
        _ => -1,
    }
}

fn read_id_text(s: &str) -> i32 {
    match parse_signed_text(s) {
        Some(id) if id >= 0 => id,
        _ => -1,
    }
}

fn read_unsigned(v: &JsonElem) -> Option<u64> {
    match &v.kind {
        JsonKind::Number => match json_int(v)? {
            JsonInt::Unsigned(n) => Some(n),
            JsonInt::Signed(n) => u64::try_from(n).ok(),
        },
        JsonKind::String(s) => parse_unsigned_text(s),
        _ => None,
    }
}

fn is_object(v: &JsonElem) -> bool {
    matches!(v.kind, JsonKind::Object(_))
}

fn as_array(v: &JsonElem) -> Option<&Vec<JsonElem>> {
    match &v.kind {
        JsonKind::Array(a) => Some(a),
        _ => None,
    }
}

fn as_str(v: &JsonElem) -> Option<&str> {
    match &v.kind {
        JsonKind::String(s) => Some(s),
        _ => None,
    }
}

fn is_true(v: Option<JsonElem>) -> bool {
    v.is_some_and(|v| matches!(v.kind, JsonKind::True))
}

fn str_eq(v: &JsonElem, text: &str) -> bool {
    as_str(v) == Some(text)
}

/// Members of an object with duplicate keys resolved (last one wins),
/// in key order.
fn object_members(v: &JsonElem) -> BTreeMap<String, JsonElem> {
    match &v.kind {
        JsonKind::Object(m) => m.iter().map(|(k, v)| (k.clone(), v.clone())).collect(),
        _ => BTreeMap::new(),
    }
}

fn object_len(v: &JsonElem) -> usize {
    object_members(v).len()
}

// ---------------------------------------------------------------------------
// Container
// ---------------------------------------------------------------------------

type Files<'a> = BTreeMap<Vec<u8>, &'a [u8]>;

fn read_be(data: &[u8], offset: usize, size: usize) -> u64 {
    data[offset..offset + size].iter().fold(0u64, |v, &b| (v << 8) | b as u64)
}

/// Returns the file table and the declared container size.
fn read_files(data: &[u8]) -> Option<(Files<'_>, u64)> {
    if data.len() < 0x40
        || data.len() as u64 > MAX_PACKAGE_SIZE
        || read_be(data, 0, 4) != PACKAGE_MAGIC
        || read_be(data, 4, 4) != 1
    {
        return None;
    }
    let size = read_be(data, 8, 8);
    let count = read_be(data, 0x10, 4);
    let toc = read_be(data, 0x14, 4);
    if size < 0x40 || size > data.len() as u64 || count > 4096 || toc > size || 0x20 + count * 0x40 > size - toc {
        return None;
    }
    let mut files = Files::new();
    for i in 0..count {
        let entry = (toc + 0x20 + i * 0x40) as usize;
        let name = &data[entry..entry + 0x20];
        let name = &name[..name.iter().position(|&b| b == 0).unwrap_or(0x20)];
        let offset = read_be(data, entry + 0x20, 8);
        let length = read_be(data, entry + 0x28, 8);
        if offset > size || length > size - offset {
            return None;
        }
        files.entry(name.to_vec()).or_insert(&data[offset as usize..(offset + length) as usize]);
    }
    Some((files, size))
}

fn read_package_file(path: &Path) -> Option<Vec<u8>> {
    let meta = fs::metadata(path).ok()?;
    if !meta.is_file() || meta.len() > MAX_PACKAGE_SIZE {
        return None;
    }
    fs::read(path).ok()
}

// ---------------------------------------------------------------------------
// Trophy package
// ---------------------------------------------------------------------------

const LOCALES: [&str; 30] = [
    "ja-JP", "en-US", "fr-FR", "es-ES", "de-DE", "it-IT", "nl-NL", "pt-PT", "ru-RU", "ko-KR", "zh-Hant", "zh-Hans",
    "fi-FI", "sv-SE", "da-DK", "no-NO", "pl-PL", "pt-BR", "en-GB", "tr-TR", "es-419", "ar-AE", "fr-CA", "cs-CZ",
    "hu-HU", "el-GR", "ro-RO", "th-TH", "vi-VN", "id-ID",
];

pub fn parse_package(data: &[u8], console_language: i32) -> Package {
    parse_package_inner(data, console_language).unwrap_or_default()
}

fn parse_package_inner(data: &[u8], console_language: i32) -> Option<Package> {
    let (files, size) = read_files(data)?;
    let conf = read_doc(&files, "tropconf.json");
    let conf = conf.value().filter(|c| is_object(c))?;
    let definitions = conf.property("trophies")?;
    let definitions = as_array(&definitions)?.clone();
    if definitions.len() > 1000 {
        return None;
    }
    let default_language = conf.property("defaultLanguage")?;
    let default_language = as_str(&default_language)?.to_string();
    let locale = usize::try_from(console_language)
        .ok()
        .filter(|&l| l < LOCALES.len())
        .map_or(LOCALES[1], |l| LOCALES[l]);
    let mut meta = read_doc(&files, &format!("tropmeta_{locale}.json"));
    if meta.is_null() {
        meta = read_doc(&files, &format!("tropmeta_{default_language}.json"));
    }
    let meta = meta.value().filter(|m| is_object(m))?;
    let texts = meta.property("metadata").filter(is_object)?;
    let title_meta = texts.property("titleMetadata").filter(is_object)?;
    let title = title_meta.property("name")?;
    let title = as_str(&title)?.to_string();
    let trophy_meta = texts.property("trophyMetadata")?;
    let trophy_meta = as_array(&trophy_meta)?.clone();

    let mut package = Package { title: title.clone(), ..Package::default() };
    package.groups.insert(-1, title);
    let mut icon_bytes = 0u64;
    for definition in &definitions {
        if !is_object(definition) {
            return None;
        }
        let id = definition.property("id")?;
        let grade = definition.property("grade")?;
        let grade = as_str(&grade)?;
        let mut trophy = Trophy { id: read_id(&id), ..Trophy::default() };
        let grade_index = "PGSB".find(grade);
        if trophy.id < 0 || grade.len() != 1 {
            return None;
        }
        trophy.grade = grade_index? as i32 + 1;
        if let Some(group) = definition.property("groupId") {
            trophy.group_id = read_id(&group);
            if trophy.group_id < 0 {
                return None;
            }
        }
        if let Some(platinum) = definition.property("platinumTrophyId") {
            trophy.platinum_id = read_id(&platinum);
        }
        trophy.hidden = is_true(definition.property("hidden"));
        trophy.has_reward = is_true(definition.property("hasReward"));
        if let Some(condition) = definition.property("unlockCondition").filter(is_object) {
            trophy.progressive = is_true(condition.property("progressive"));
            if let Some(stat) = condition.property("udsStatId") {
                trophy.uds_stat_id = Some(read_unsigned(&stat)?);
            }
            if let Some(target) = condition.property("targetValue") {
                trophy.target = read_unsigned(&target);
            }
            if let Some(comparator) = condition.property("comparator") {
                trophy.comparison = match as_str(&comparator) {
                    Some("ge") => Comparison::GreaterEqual,
                    Some("gt") => Comparison::Greater,
                    Some("le") => Comparison::LessEqual,
                    Some("lt") => Comparison::Less,
                    _ => Comparison::None,
                };
            }
        }
        if let Some(icon) = files.get(format!("trop{:04}.png", trophy.id).as_bytes()) {
            if icon.len() as u64 > size - icon_bytes {
                return None;
            }
            trophy.icon_png = icon.to_vec();
            icon_bytes += icon.len() as u64;
        }
        package.groups.entry(trophy.group_id).or_default();
        if package.groups.len() > 50 || package.trophies.contains_key(&trophy.id) {
            return None;
        }
        package.trophies.insert(trophy.id, trophy);
    }
    if let Some(groups) = texts.property("groupMetadata") {
        for group in as_array(&groups).into_iter().flatten() {
            if !is_object(group) {
                continue;
            }
            let (Some(id), Some(name)) = (group.property("id"), group.property("name")) else {
                continue;
            };
            let Some(name) = as_str(&name) else { continue };
            if let Some(slot) = package.groups.get_mut(&read_id(&id)) {
                *slot = name.to_string();
            }
        }
    }
    for text in &trophy_meta {
        if !is_object(text) {
            continue;
        }
        let Some(id) = text.property("id") else { continue };
        let Some(trophy) = package.trophies.get_mut(&read_id(&id)) else {
            continue;
        };
        for (key, field) in [
            ("name", &mut trophy.name),
            ("detail", &mut trophy.description),
            ("reward", &mut trophy.reward),
        ] {
            if let Some(s) = text.property(key).as_ref().and_then(as_str) {
                *field = s.to_string();
            }
        }
    }
    Some(package)
}

pub fn load_package(path: &Path, console_language: i32) -> Package {
    match read_package_file(path) {
        Some(data) => parse_package(&data, console_language),
        None => Package::default(),
    }
}

/// `sce_sys/trophy2/trophyNN.ucp` relative to the application root.
pub fn package_path(service_label: u32) -> String {
    format!("{PACKAGE_DIRECTORY}/trophy{service_label:02}.ucp")
}

/// `sce_sys/uds/udsNN.ucp` relative to the application root.
pub(crate) fn uds_package_path(service_label: u32) -> String {
    format!("sce_sys/uds/uds{service_label:02}.ucp")
}

// ---------------------------------------------------------------------------
// UDS rules
// ---------------------------------------------------------------------------

/// Only direct unsigned latest-value stats are evaluated; aggregated stats
/// and filtered/nested extraction rules are skipped.
pub(crate) fn load_uds_rules(path: &Path) -> UdsRules {
    read_package_file(path).map(|d| parse_uds_rules(&d)).unwrap_or_default()
}

pub(crate) fn parse_uds_rules(data: &[u8]) -> UdsRules {
    parse_uds_rules_inner(data).unwrap_or_default()
}

fn parse_uds_rules_inner(data: &[u8]) -> Option<UdsRules> {
    let (files, _) = read_files(data)?;
    let definitions = read_doc(&files, "stats_definition.json");
    let extraction = read_doc(&files, "stats_extraction.json");
    let definitions = definitions.value().filter(|d| is_object(d))?;
    let extraction = extraction.value().filter(|e| is_object(e))?;
    let definitions = definitions.property("statDefinitionArray")?;
    let definitions = as_array(&definitions)?;
    let rules = extraction.property("statsExtractionRuleArray")?;
    let rules = as_array(&rules)?;

    let mut stats: BTreeMap<u64, bool> = BTreeMap::new();
    for definition in definitions {
        if !is_object(definition) {
            continue;
        }
        let (Some(id), Some(aggregation), Some(data_type)) = (
            definition.property("statId"),
            definition.property("aggregation"),
            definition.property("dataType"),
        ) else {
            continue;
        };
        if !str_eq(&aggregation, "latest") {
            continue;
        }
        let Some(id) = read_unsigned(&id) else { continue };
        let is_uint32 = str_eq(&data_type, "uint32");
        if !is_uint32 && !str_eq(&data_type, "uint64") {
            continue;
        }
        let maximum = if is_uint32 { u32::MAX as u64 } else { u64::MAX };
        if definition.property("minValue").is_some_and(|v| read_unsigned(&v) != Some(0))
            || definition.property("maxValue").is_some_and(|v| read_unsigned(&v) != Some(maximum))
        {
            continue;
        }
        stats.entry(id).or_insert(is_uint32);
    }

    let mut events = UdsRules::new();
    for rule in rules {
        if !is_object(rule) {
            continue;
        }
        let (Some(condition), Some(action)) = (rule.property("condition"), rule.property("action")) else {
            continue;
        };
        if !is_object(&condition) || object_len(&condition) != 1 || !is_object(&action) {
            continue;
        }
        let Some(event_name) = condition.property("eventName") else { continue };
        let Some(event_name) = as_str(&event_name) else { continue };
        let Some(input) = action.property("input") else { continue };
        let Some(input) = as_str(&input) else { continue };
        let Some(output) = action.property("output").filter(is_object) else {
            continue;
        };
        let Some(stat_id) = output.property("statId") else { continue };
        let Some(id) = read_unsigned(&stat_id) else { continue };
        let Some(&is_uint32) = stats.get(&id) else { continue };
        let Some(name) = input.strip_prefix("$.") else { continue };
        if name.is_empty() || !name.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_') {
            continue;
        }
        events
            .entry(event_name.to_string())
            .or_default()
            .push(UdsRule { stat_id: id, input: name.to_string(), is_uint32 });
    }
    Some(events)
}

/// Non-platinum trophies whose stat condition is satisfied by `properties`
/// of the named event, in trophy id order.
pub(crate) fn find_uds_trophies(
    package: &Package,
    event_name: &str,
    properties: &BTreeMap<String, UdsInteger>,
) -> Vec<i32> {
    let Some(rules) = package.event_rules.get(event_name) else {
        return Vec::new();
    };
    let mut ids = Vec::new();
    for (&id, trophy) in &package.trophies {
        let (Some(stat), Some(target)) = (trophy.uds_stat_id, trophy.target) else {
            continue;
        };
        if trophy.grade == 1 {
            continue;
        }
        for rule in rules {
            let Some(value) = properties.get(&rule.input) else { continue };
            if rule.stat_id != stat {
                continue;
            }
            let number = match (rule.is_uint32, value) {
                (true, UdsInteger::U32(v)) => *v as u64,
                (false, UdsInteger::U64(v)) => *v,
                _ => continue,
            };
            let matched = match trophy.comparison {
                Comparison::GreaterEqual => number >= target,
                Comparison::Greater => number > target,
                Comparison::LessEqual => number <= target,
                Comparison::Less => number < target,
                Comparison::None => false,
            };
            if matched {
                ids.push(id);
                break;
            }
        }
    }
    ids
}

pub fn get_progress(package: &Package, unlocks: &UnlockData, group: Option<i32>) -> Progress {
    let mut progress = Progress::default();
    for (id, trophy) in &package.trophies {
        if group.is_some_and(|g| trophy.group_id != g) {
            continue;
        }
        let grade = trophy.grade as usize;
        progress.total += 1;
        progress.total_grade[grade] += 1;
        if unlocks.unlocked.contains(id) {
            progress.earned += 1;
            progress.earned_grade[grade] += 1;
        }
    }
    progress
}

// ---------------------------------------------------------------------------
// Unlock persistence
// ---------------------------------------------------------------------------

/// `<root>/_Trophies/<title>/trophies_<user>_<label>.json`; None for an
/// empty or unsafe title id.
pub fn unlocks_path(root: &Path, title_id: &str, user_id: i32, service_label: u32) -> Option<PathBuf> {
    if title_id.is_empty() || !title_id.bytes().all(|c| c.is_ascii_alphanumeric() || c == b'_' || c == b'-') {
        return None;
    }
    Some(root.join("_Trophies").join(title_id).join(format!("trophies_{user_id}_{service_label}.json")))
}

pub fn load_unlock_data(path: &Path) -> UnlockData {
    let Ok(meta) = fs::metadata(path) else {
        return UnlockData::default();
    };
    if !meta.is_file() || meta.len() > MAX_UNLOCKS_SIZE {
        return UnlockData::default();
    }
    let Some(json) = fs::read(path).ok().and_then(|b| parse_bytes(&b)) else {
        return UnlockData::default();
    };
    if !is_object(&json) {
        return UnlockData::default();
    }
    let Some(list) = json.property("unlockedTrophies") else {
        return UnlockData::default();
    };
    let Some(list) = as_array(&list) else {
        return UnlockData::default();
    };
    let mut unlocks = UnlockData::default();
    for value in list {
        let id = read_id(value);
        if id >= 0 {
            unlocks.unlocked.insert(id);
        }
    }
    if let Some(at) = json.property("unlockedAt").filter(is_object) {
        for (key, value) in object_members(&at) {
            let id = read_id_text(&key);
            if id >= 0 && unlocks.unlocked.contains(&id) {
                if let Some(tick) = read_unsigned(&value).filter(|t| (UNIX_EPOCH_TICK..=MAX_TICK).contains(t)) {
                    unlocks.timestamps.entry(id).or_insert(tick);
                }
            }
        }
    }
    unlocks
}

fn unlocks_json(unlocks: &UnlockData) -> String {
    // Object keys are written in sorted order (as strings).
    let stamps: BTreeMap<String, u64> = unlocks
        .timestamps
        .iter()
        .filter(|(id, _)| unlocks.unlocked.contains(id))
        .map(|(id, tick)| (id.to_string(), *tick))
        .collect();
    let stamps: Vec<String> = stamps.iter().map(|(k, v)| format!("\"{k}\":{v}")).collect();
    let ids: Vec<String> = unlocks.unlocked.iter().map(i32::to_string).collect();
    format!("{{\"unlockedAt\":{{{}}},\"unlockedTrophies\":[{}]}}", stamps.join(","), ids.join(","))
}

/// Atomic save: write `<path>.tmp`, flush, rename over `path`. The
/// temporary file is removed on failure.
pub(crate) fn save_unlock_data(path: &Path, unlocks: &UnlockData) -> bool {
    let Some(parent) = path.parent().filter(|_| !path.as_os_str().is_empty()) else {
        return false;
    };
    if fs::create_dir_all(parent).is_err() {
        return false;
    }
    let text = unlocks_json(unlocks);
    if text.len() as u64 > MAX_UNLOCKS_SIZE {
        return false;
    }
    let mut temporary = path.as_os_str().to_owned();
    temporary.push(".tmp");
    let temporary = PathBuf::from(temporary);
    let Ok(mut file) = fs::File::create(&temporary) else {
        return false;
    };
    let complete = file.write_all(text.as_bytes()).is_ok() && file.sync_all().is_ok();
    drop(file);
    if complete && fs::rename(&temporary, path).is_ok() {
        return true;
    }
    let _ = fs::remove_file(&temporary);
    false
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    pub(crate) fn temp_dir(tag: &str) -> PathBuf {
        use std::sync::atomic::{AtomicU32, Ordering};
        static N: AtomicU32 = AtomicU32::new(0);
        let dir = std::env::temp_dir().join(format!(
            "rps5e_trophy_test_{tag}_{}_{}",
            std::process::id(),
            N.fetch_add(1, Ordering::Relaxed)
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    pub(crate) fn write_package(path: &Path, files: &[(&str, &[u8])]) {
        let mut sorted: Vec<_> = files.to_vec();
        sorted.sort_by(|a, b| a.0.cmp(b.0));
        let mut offset = 0x40 + sorted.len() * 0x40;
        let size = offset + sorted.iter().map(|f| f.1.len()).sum::<usize>();
        let mut bytes = vec![0u8; size];
        let put = |b: &mut Vec<u8>, at: usize, v: u64, len: usize| {
            for i in 0..len {
                b[at + len - 1 - i] = (v >> (8 * i)) as u8;
            }
        };
        put(&mut bytes, 0, 0xB228_C60A, 4);
        put(&mut bytes, 4, 1, 4);
        put(&mut bytes, 8, size as u64, 8);
        put(&mut bytes, 0x10, sorted.len() as u64, 4);
        put(&mut bytes, 0x14, 0x20, 4);
        let mut entry = 0x40;
        for (name, contents) in &sorted {
            assert!(name.len() < 0x20);
            bytes[entry..entry + name.len()].copy_from_slice(name.as_bytes());
            put(&mut bytes, entry + 0x20, offset as u64, 8);
            put(&mut bytes, entry + 0x28, contents.len() as u64, 8);
            bytes[offset..offset + contents.len()].copy_from_slice(contents);
            offset += contents.len();
            entry += 0x40;
        }
        fs::write(path, bytes).unwrap();
    }

    pub(crate) const TEST_CONF: &str = r#"{"defaultLanguage":"en-US","trophies":[
        {"id":"0","grade":"P"},
        {"id":"1","grade":"G","unlockCondition":{"udsStatId":"10","comparator":"ge","targetValue":"10","progressive":false}},
        {"id":"2","grade":"S","unlockCondition":{"udsStatId":"20","comparator":"gt","targetValue":"100","progressive":false}},
        {"id":"3","grade":"B","groupId":"1","unlockCondition":{"udsStatId":"30","comparator":"le","targetValue":"5","progressive":false}},
        {"id":"4","grade":"B","unlockCondition":{"udsStatId":"10","comparator":"lt","targetValue":"-1","progressive":false}}
    ]}"#;
    pub(crate) const TEST_META: &str = r#"{"metadata":{"titleMetadata":{"name":"Test game"},"trophyMetadata":[
        {"id":"1","name":"Find All Coins","detail":"Find all coins in the level."}
    ]}}"#;
    pub(crate) const TEST_STATS: &str = r#"{"statDefinitionArray":[
        {"statId":10,"dataType":"uint64","aggregation":"latest","minValue":"0"},
        {"statId":20,"dataType":"uint64","aggregation":"latest"},
        {"statId":30,"dataType":"uint32","aggregation":"latest"},
        {"statId":40,"dataType":"uint64","aggregation":"sum"},
        {"statId":50,"dataType":"uint64","aggregation":"count"},
        {"statId":60,"dataType":"uint64","aggregation":"max"},
        {"statId":70,"dataType":"uint64","aggregation":"latest","maxValue":"3"}
    ]}"#;
    pub(crate) const TEST_EXTRACTION: &str = r#"{"statsExtractionRuleArray":[
        {"condition":{"eventName":"COLLECT"},"action":{"input":"$.counter","output":{"statId":10}}},
        {"condition":{"eventName":"COLLECT"},"action":{"input":"$.counterBase","output":{"statId":20}}},
        {"condition":{"eventName":"TIME"},"action":{"input":"$.seconds","output":{"statId":30}}},
        {"condition":{"eventName":"FILTERED","property":{"path":"$.level","comparator":"==","value":1}},"action":{"input":"$.counter","output":{"statId":10}}},
        {"condition":{"eventName":"NESTED"},"action":{"input":"$.player.counter","output":{"statId":10}}},
        {"condition":{"eventName":"SUM"},"action":{"input":"$.counter","output":{"statId":40}}},
        {"condition":{"eventName":"COUNT"},"action":{"output":{"statId":50}}},
        {"condition":{"eventName":"MAX"},"action":{"input":"$.counter","output":{"statId":60}}},
        {"condition":{"eventName":"BOUNDED"},"action":{"input":"$.counter","output":{"statId":70}}},
        {"bad":1}
    ]}"#;

    fn make_package(dir: &Path) -> Package {
        let path = dir.join("trophy00.ucp");
        write_package(
            &path,
            &[("tropconf.json", TEST_CONF.as_bytes()), ("tropmeta_en-US.json", TEST_META.as_bytes())],
        );
        load_package(&path, 1)
    }

    fn props(entries: &[(&str, UdsInteger)]) -> BTreeMap<String, UdsInteger> {
        entries.iter().map(|(k, v)| (k.to_string(), *v)).collect()
    }

    use UdsInteger::{I32, U32, U64};

    #[test]
    fn event_extraction() {
        let dir = temp_dir("extract");
        let mut package = make_package(&dir);
        assert_eq!(package.trophies.len(), 5, "unsupported signed target does not discard trophy metadata");
        let t1 = &package.trophies[&1];
        assert!(
            t1.uds_stat_id == Some(10) && t1.target == Some(10) && !t1.progressive,
            "non-progressive trophy retains its stat target"
        );
        assert_eq!(t1.name, "Find All Coins");
        assert!(package.trophies[&4].target.is_none(), "negative target is not converted to unsigned");
        assert!(
            find_uds_trophies(&package, "FIND_ALL_COINS", &props(&[("counter", U64(100))])).is_empty(),
            "missing mapping never guesses from trophy text"
        );
        let path = dir.join("uds00.ucp");
        write_package(
            &path,
            &[
                ("stats_definition.json", TEST_STATS.as_bytes()),
                ("stats_extraction.json", TEST_EXTRACTION.as_bytes()),
            ],
        );
        package.event_rules = load_uds_rules(&path);
        assert_eq!(package.event_rules.len(), 2, "skip unsupported extraction and aggregation");
        assert!(
            find_uds_trophies(&package, "COLLECT", &props(&[("counter", U64(9)), ("counterBase", U64(100))]))
                .is_empty(),
            "do not add independent counters or treat gt as ge"
        );
        assert_eq!(
            find_uds_trophies(&package, "COLLECT", &props(&[("counter", U64(10)), ("counterBase", U64(100))])),
            vec![1],
            "each stat reads its configured property"
        );
        assert_eq!(
            find_uds_trophies(&package, "COLLECT", &props(&[("counter", U64(0)), ("counterBase", U64(101))])),
            vec![2],
            "strict comparison requires greater value"
        );
        assert_eq!(
            find_uds_trophies(&package, "COLLECT", &props(&[("counter", U64(10)), ("counterBase", U64(101))])),
            vec![1, 2],
            "one event can unlock multiple mapped trophies"
        );
        assert_eq!(
            find_uds_trophies(&package, "TIME", &props(&[("seconds", U32(5))])),
            vec![3],
            "less-equal condition accepts its boundary"
        );
        assert!(
            find_uds_trophies(&package, "TIME", &props(&[("seconds", U32(6))])).is_empty(),
            "less-equal condition rejects larger input"
        );
        assert!(
            find_uds_trophies(&package, "TIME", &props(&[("counter", U64(0))])).is_empty(),
            "missing property is not zero"
        );
        package.trophies.get_mut(&3).unwrap().comparison = Comparison::Less;
        assert!(
            find_uds_trophies(&package, "TIME", &props(&[("seconds", U32(5))])).is_empty(),
            "strict less rejects its boundary"
        );
        assert_eq!(
            find_uds_trophies(&package, "TIME", &props(&[("seconds", U32(4))])),
            vec![3],
            "strict less accepts lower input"
        );
        package.trophies.get_mut(&3).unwrap().comparison = Comparison::GreaterEqual;
        assert!(
            find_uds_trophies(&package, "TIME", &props(&[("seconds", U64(u32::MAX as u64 + 1))])).is_empty(),
            "uint64 input cannot update a uint32 stat"
        );
        assert!(
            find_uds_trophies(&package, "COLLECT", &props(&[("counter", I32(100))])).is_empty(),
            "signed inputs cannot update unsigned stats"
        );
        assert!(
            find_uds_trophies(&package, "COLLECT", &props(&[("counter", U32(100))])).is_empty(),
            "unsigned input width must match the stat"
        );
        package.trophies.get_mut(&1).unwrap().comparison = Comparison::None;
        assert!(
            find_uds_trophies(&package, "COLLECT", &props(&[("counter", U64(100))])).is_empty(),
            "unknown comparator cannot unlock"
        );
        write_package(&path, &[("stats_extraction.json", b"{}")]);
        assert!(load_uds_rules(&path).is_empty(), "missing stat definitions cannot infer aggregation");
        assert!(load_uds_rules(&dir.join("missing.ucp")).is_empty(), "missing UDS package has no rules");
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn progress() {
        let dir = temp_dir("progress");
        let package = make_package(&dir);
        let unlocks = UnlockData { unlocked: [0, 1, 999].into_iter().collect(), ..UnlockData::default() };
        let progress = get_progress(&package, &unlocks, None);
        assert!(progress.total == 5 && progress.earned == 2, "ignore unlock IDs absent from package");
        assert!(
            progress.earned_grade[1] == 1 && progress.earned_grade[2] == 1 && progress.percentage() == 60,
            "percentage uses 6:2:1 grade points and excludes platinum"
        );
        let group = get_progress(&package, &unlocks, Some(1));
        assert!(group.total == 1 && group.earned == 0 && group.percentage() == 0, "group counts filter trophies");
        assert_eq!(
            get_progress(&Package::default(), &UnlockData::default(), None).percentage(),
            0,
            "empty package progress is zero"
        );
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn unlock_persistence() {
        let dir = temp_dir("persist");
        let path = unlocks_path(&dir, "PPSA00001", 1000, 0).unwrap();
        assert_eq!(path, dir.join("_Trophies/PPSA00001/trophies_1000_0.json"), "store trophies outside save data");
        assert!(unlocks_path(&dir, "../bad", 1000, 0).is_none(), "reject unsafe title IDs");
        assert!(load_unlock_data(&path).unlocked.is_empty(), "missing unlock file loads as empty");
        let mut unlocks = UnlockData { unlocked: [4, 42].into_iter().collect(), ..UnlockData::default() };
        let tick = UNIX_EPOCH_TICK + 1_791_288_000_123_456;
        unlocks.timestamps.insert(42, tick);
        assert!(save_unlock_data(&path, &unlocks), "save timestamped unlocks");
        let mut loaded = load_unlock_data(&path);
        assert_eq!(loaded, unlocks, "persist full UTC microsecond timestamp");
        loaded.unlocked.insert(43);
        loaded.timestamps.insert(43, tick + 1_000_000);
        assert!(save_unlock_data(&path, &loaded), "replace existing unlock file");
        assert_eq!(load_unlock_data(&path).timestamps[&42], tick, "retain original unlock time");

        let blocked = dir.join("blocked.json");
        fs::create_dir(&blocked).unwrap();
        fs::write(blocked.join("keep"), b"").unwrap();
        assert!(
            !save_unlock_data(&blocked, &loaded) && blocked.join("keep").exists(),
            "failed replacement leaves destination intact"
        );
        assert!(!dir.join("blocked.json.tmp").exists(), "failed replacement removes temporary file");
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn unlock_file_format() {
        let unlocks = UnlockData {
            unlocked: [4, 10, 42].into_iter().collect(),
            timestamps: [(10, UNIX_EPOCH_TICK), (42, UNIX_EPOCH_TICK + 5), (7, UNIX_EPOCH_TICK)]
                .into_iter()
                .collect(),
        };
        assert_eq!(
            unlocks_json(&unlocks),
            format!(
                "{{\"unlockedAt\":{{\"10\":{},\"42\":{}}},\"unlockedTrophies\":[4,10,42]}}",
                UNIX_EPOCH_TICK,
                UNIX_EPOCH_TICK + 5
            )
        );
        let dir = temp_dir("format");
        let path = dir.join("u.json");
        fs::write(
            &path,
            br#"{"unlockedTrophies":[1,"2",-3,"x",1.5],"unlockedAt":{"1":"62135596800000001","2":5,"9":62135596800000000}}"#,
        )
        .unwrap();
        let loaded = load_unlock_data(&path);
        assert_eq!(loaded.unlocked, [1, 2].into_iter().collect());
        assert_eq!(loaded.timestamps, [(1, UNIX_EPOCH_TICK + 1)].into_iter().collect());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn container_and_metadata_rules() {
        let dir = temp_dir("container");
        let path = dir.join("trophy00.ucp");
        // Falls back to the default language; group names and icons.
        let conf = br#"{"defaultLanguage":"fr-FR","trophies":[
            {"id":0,"grade":"P","hidden":true},
            {"id":1,"grade":"B","groupId":"2","platinumTrophyId":"0","hasReward":true,
             "unlockCondition":{"progressive":true,"targetValue":5}}]}"#;
        let meta = br#"{"metadata":{"titleMetadata":{"name":"Jeu"},
            "groupMetadata":[{"id":"2","name":"DLC"},{"id":"9","name":"none"}],
            "trophyMetadata":[{"id":1,"name":"N","detail":"D","reward":"R"}]}}"#;
        write_package(
            &path,
            &[("tropconf.json", conf), ("tropmeta_fr-FR.json", meta), ("trop0001.png", b"PNG")],
        );
        let p = load_package(&path, 1);
        assert_eq!(p.title, "Jeu");
        assert_eq!(p.groups.iter().map(|(k, v)| (*k, v.as_str())).collect::<Vec<_>>(), vec![(-1, "Jeu"), (2, "DLC")]);
        let t = &p.trophies[&1];
        assert!(t.progressive && t.target == Some(5) && t.has_reward && t.platinum_id == 0 && t.grade == 4);
        assert_eq!((t.name.as_str(), t.description.as_str(), t.reward.as_str()), ("N", "D", "R"));
        assert_eq!(t.icon_png, b"PNG");
        assert!(p.trophies[&0].hidden && p.trophies[&0].grade == 1);
        // Invalid locale file present: no fallback.
        write_package(
            &path,
            &[("tropconf.json", conf), ("tropmeta_en-US.json", b"{"), ("tropmeta_fr-FR.json", meta)],
        );
        assert!(load_package(&path, 1).trophies.is_empty());
        // Bad grade / duplicate id reject the package.
        let bad = br#"{"defaultLanguage":"en-US","trophies":[{"id":0,"grade":"X"}]}"#;
        write_package(&path, &[("tropconf.json", bad), ("tropmeta_en-US.json", TEST_META.as_bytes())]);
        assert!(load_package(&path, 1).trophies.is_empty());
        let dup = br#"{"defaultLanguage":"en-US","trophies":[{"id":0,"grade":"B"},{"id":"0","grade":"B"}]}"#;
        write_package(&path, &[("tropconf.json", dup), ("tropmeta_en-US.json", TEST_META.as_bytes())]);
        assert!(load_package(&path, 1).trophies.is_empty());
        // Corrupt header.
        let mut data = fs::read(&path).unwrap();
        data[0] = 0;
        assert!(parse_package(&data, 1).trophies.is_empty());
        assert_eq!(package_path(1), "sce_sys/trophy2/trophy01.ucp");
        assert_eq!(uds_package_path(12), "sce_sys/uds/uds12.ucp");
        let _ = fs::remove_dir_all(&dir);
    }

    /// ASTRO BOT's packages parse (ASTRO BOT PPSA21567 under `GAMES_DIR`; skipped when unset or absent).
    #[test]
    fn installed_game_packages() {
        let Some(root) = ps5vm::test_games::game_dir("PPSA21567") else { return };
        let root = PathBuf::from(root);
        let trophy = root.join(package_path(0));
        if !trophy.is_file() {
            return;
        }
        let p = load_package(&trophy, 1);
        assert!(!p.trophies.is_empty());
        assert!(p.trophies.values().any(|t| t.grade == 1));
        let _ = load_uds_rules(&root.join(uds_package_path(0)));
    }
}
