// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Title metadata: `param.sfo` (PS4 style, also present in PS4-format
//! packages) and `sce_sys/param.json` (PS5).

use crate::io::{le_u16, le_u32};
use std::collections::BTreeMap;

/// A param.sfo value.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SfoValue {
    Str(String),
    Int(u32),
    Bytes(Vec<u8>),
}

impl SfoValue {
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::Str(s) => Some(s),
            _ => None,
        }
    }
    pub fn as_int(&self) -> Option<u32> {
        match self {
            Self::Int(v) => Some(*v),
            _ => None,
        }
    }
}

pub const SFO_MAGIC: u32 = 0x4653_5000; // "\0PSF"
const SFO_FMT_UTF8_SPECIAL: u16 = 0x0004;
const SFO_FMT_UTF8: u16 = 0x0204;
const SFO_FMT_INT32: u16 = 0x0404;

/// Parse a param.sfo. Returns None for anything that is not one.
///
/// Layout: `u32 magic, u32 version, u32 key table, u32 data table,
/// u32 count`, then `count` x `{u16 key offset, u16 format, u32 length,
/// u32 max length, u32 data offset}`.
pub fn parse_sfo(b: &[u8]) -> Option<BTreeMap<String, SfoValue>> {
    if b.len() < 0x14 || le_u32(b, 0) != SFO_MAGIC {
        return None;
    }
    let (keys, data, count) = (le_u32(b, 8) as usize, le_u32(b, 0x0C) as usize, le_u32(b, 0x10) as usize);
    if count > 4096 || 0x14 + count * 16 > b.len() {
        return None;
    }
    let mut out = BTreeMap::new();
    for i in 0..count {
        let e = 0x14 + i * 16;
        let (koff, fmt, len, doff) =
            (le_u16(b, e) as usize, le_u16(b, e + 2), le_u32(b, e + 4) as usize, le_u32(b, e + 12) as usize);
        let kstart = keys.checked_add(koff)?;
        let key_bytes = b.get(kstart..)?;
        let key = String::from_utf8_lossy(key_bytes.split(|&c| c == 0).next()?).into_owned();
        let dstart = data.checked_add(doff)?;
        let raw = b.get(dstart..dstart.checked_add(len)?)?;
        let v = match fmt {
            SFO_FMT_INT32 if len >= 4 => SfoValue::Int(le_u32(raw, 0)),
            SFO_FMT_UTF8 | SFO_FMT_UTF8_SPECIAL => {
                SfoValue::Str(String::from_utf8_lossy(raw.split(|&c| c == 0).next().unwrap_or(raw)).into_owned())
            }
            _ => SfoValue::Bytes(raw.to_vec()),
        };
        out.insert(key, v);
    }
    Some(out)
}

/// Build a param.sfo (tests and tools).
pub fn build_sfo(entries: &[(&str, SfoValue)]) -> Vec<u8> {
    let mut keys = Vec::new();
    let mut data = Vec::new();
    let mut table = Vec::new();
    for (k, v) in entries {
        let koff = keys.len() as u16;
        keys.extend_from_slice(k.as_bytes());
        keys.push(0);
        let (fmt, bytes, max): (u16, Vec<u8>, usize) = match v {
            SfoValue::Int(i) => (SFO_FMT_INT32, i.to_le_bytes().to_vec(), 4),
            SfoValue::Str(s) => {
                let mut b = s.as_bytes().to_vec();
                b.push(0);
                let max = b.len().next_multiple_of(4);
                (SFO_FMT_UTF8, b, max)
            }
            SfoValue::Bytes(b) => (0, b.clone(), b.len().next_multiple_of(4)),
        };
        let doff = data.len() as u32;
        data.extend_from_slice(&bytes);
        data.resize(doff as usize + max, 0);
        table.extend_from_slice(&koff.to_le_bytes());
        table.extend_from_slice(&fmt.to_le_bytes());
        table.extend_from_slice(&(bytes.len() as u32).to_le_bytes());
        table.extend_from_slice(&(max as u32).to_le_bytes());
        table.extend_from_slice(&doff.to_le_bytes());
    }
    while keys.len() % 4 != 0 {
        keys.push(0);
    }
    let key_off = 0x14 + table.len();
    let data_off = key_off + keys.len();
    let mut out = Vec::new();
    out.extend_from_slice(&SFO_MAGIC.to_le_bytes());
    out.extend_from_slice(&0x101u32.to_le_bytes());
    out.extend_from_slice(&(key_off as u32).to_le_bytes());
    out.extend_from_slice(&(data_off as u32).to_le_bytes());
    out.extend_from_slice(&(entries.len() as u32).to_le_bytes());
    out.extend_from_slice(&table);
    out.extend_from_slice(&keys);
    out.extend_from_slice(&data);
    out
}

/// Fields the emulator and the launcher care about, from either format.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct TitleParams {
    pub title_id: Option<String>,
    pub content_id: Option<String>,
    pub name: Option<String>,
    pub version: Option<String>,
    /// param.sfo CATEGORY (`gd` game, `gp` patch, `ac` additional content),
    /// or the PS5 application category type as a decimal string.
    pub category: Option<String>,
}

fn non_empty(s: Option<&str>) -> Option<String> {
    s.map(str::trim).filter(|s| !s.is_empty()).map(str::to_string)
}

impl TitleParams {
    pub fn from_sfo(sfo: &BTreeMap<String, SfoValue>) -> Self {
        let s = |k: &str| non_empty(sfo.get(k).and_then(SfoValue::as_str));
        Self {
            title_id: s("TITLE_ID"),
            content_id: s("CONTENT_ID"),
            name: s("TITLE"),
            version: s("APP_VER").or_else(|| s("VERSION")),
            category: s("CATEGORY"),
        }
    }

    /// Parse a PS5 `param.json`. The title name is taken from the default
    /// language of `localizedParameters`, else English, else any entry.
    pub fn from_json(text: &str) -> Option<Self> {
        let v: serde_json::Value = serde_json::from_str(text.trim_start_matches('\u{feff}')).ok()?;
        let s = |k: &str| non_empty(v.get(k).and_then(|x| x.as_str()));
        let name = v.get("localizedParameters").and_then(|lp| {
            let pick = |lang: &str| non_empty(lp.get(lang)?.get("titleName")?.as_str());
            let default = lp.get("defaultLanguage").and_then(|d| d.as_str()).unwrap_or("en-US");
            pick(default).or_else(|| pick("en-US")).or_else(|| pick("en-GB")).or_else(|| {
                lp.as_object()?.values().find_map(|x| non_empty(x.get("titleName")?.as_str()))
            })
        });
        let category = v.get("applicationCategoryType").and_then(|c| c.as_u64()).map(|c| c.to_string());
        Some(Self {
            title_id: s("titleId"),
            content_id: s("contentId"),
            name,
            version: s("contentVersion").or_else(|| s("masterVersion")),
            category,
        })
    }

    /// Fill fields missing here from `other`.
    pub fn merge(&mut self, other: &TitleParams) {
        let fill = |a: &mut Option<String>, b: &Option<String>| {
            if a.is_none() {
                a.clone_from(b);
            }
        };
        fill(&mut self.title_id, &other.title_id);
        fill(&mut self.content_id, &other.content_id);
        fill(&mut self.name, &other.name);
        fill(&mut self.version, &other.version);
        fill(&mut self.category, &other.category);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sfo_roundtrip() {
        let b = build_sfo(&[
            ("APP_VER", SfoValue::Str("01.00".into())),
            ("CATEGORY", SfoValue::Str("gd".into())),
            ("TITLE", SfoValue::Str("Demo Game".into())),
            ("TITLE_ID", SfoValue::Str("CUSA00001".into())),
            ("PARENTAL_LEVEL", SfoValue::Int(5)),
        ]);
        let m = parse_sfo(&b).unwrap();
        assert_eq!(m["TITLE"], SfoValue::Str("Demo Game".into()));
        assert_eq!(m["PARENTAL_LEVEL"].as_int(), Some(5));
        let p = TitleParams::from_sfo(&m);
        assert_eq!(p.title_id.as_deref(), Some("CUSA00001"));
        assert_eq!(p.version.as_deref(), Some("01.00"));
        assert_eq!(p.category.as_deref(), Some("gd"));
        assert!(parse_sfo(b"nope").is_none());
    }

    #[test]
    fn param_json_fields() {
        let j = r#"{"titleId":"PPSA12345","contentId":"UP0000-PPSA12345_00-DEMO000000000000",
            "contentVersion":"01.000.002","applicationCategoryType":0,
            "localizedParameters":{"defaultLanguage":"ja-JP","en-US":{"titleName":"Demo"},"ja-JP":{"titleName":"デモ"}}}"#;
        let p = TitleParams::from_json(j).unwrap();
        assert_eq!(p.title_id.as_deref(), Some("PPSA12345"));
        assert_eq!(p.name.as_deref(), Some("デモ"));
        assert_eq!(p.version.as_deref(), Some("01.000.002"));
        assert_eq!(p.category.as_deref(), Some("0"));
    }
}
