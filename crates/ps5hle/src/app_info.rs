// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Application identity shared by the system-service family of HLE
//! libraries: the /app0 root on the host, the title id from
//! `sce_sys/param.json` and the console language.

use std::path::{Path, PathBuf};
use std::sync::Mutex;

/// Title id reported when param.json is missing or unreadable.
pub const FALLBACK_TITLE_ID: &str = "PPSA00000";
/// Default system language (1 = English US).
pub const DEFAULT_SYSTEM_LANGUAGE: i32 = 1;

struct AppInfo {
    app0_root: Option<PathBuf>,
    title_id: Option<String>,
    content_version: Option<String>,
    system_language: i32,
}

static INFO: Mutex<AppInfo> = Mutex::new(AppInfo {
    app0_root: None,
    title_id: None,
    content_version: None,
    system_language: DEFAULT_SYSTEM_LANGUAGE,
});

/// Configure from the path of the main executable. `/app0` is the eboot's
/// directory unless `RPS5E_APP0_DIR` overrides it; the language comes from
/// `RPS5E_SYSTEM_LANGUAGE` (numeric PS5 language id).
pub fn configure_from_eboot(eboot_path: &Path) {
    let app0 = std::env::var("RPS5E_APP0_DIR")
        .ok()
        .filter(|s| !s.trim().is_empty())
        .map(PathBuf::from)
        .or_else(|| eboot_path.parent().map(Path::to_path_buf));
    let language = parse_system_language(std::env::var("RPS5E_SYSTEM_LANGUAGE").ok().as_deref());
    configure(app0, language);
}

/// Console language numbers (SCE_SYSTEM_PARAM_LANG_JAPANESE ..
/// SCE_SYSTEM_PARAM_LANG_UKRAINIAN).
const SYSTEM_LANGUAGES: std::ops::RangeInclusive<i32> = 0..=30;

/// The console language a `RPS5E_SYSTEM_LANGUAGE` value selects: unset or
/// empty is English (US); otherwise a decimal language number 0..=30.
/// Anything else is reported and English (US) is used.
fn parse_system_language(value: Option<&str>) -> i32 {
    let Some(text) = value.map(str::trim).filter(|t| !t.is_empty()) else {
        return DEFAULT_SYSTEM_LANGUAGE;
    };
    match text.parse::<i32>() {
        Ok(n) if text.bytes().all(|b| b.is_ascii_digit()) && SYSTEM_LANGUAGES.contains(&n) => n,
        _ => {
            log::error!(
                "[APP] RPS5E_SYSTEM_LANGUAGE must be a console language number from {} to {}, not {text:?}; using {DEFAULT_SYSTEM_LANGUAGE}",
                SYSTEM_LANGUAGES.start(),
                SYSTEM_LANGUAGES.end()
            );
            DEFAULT_SYSTEM_LANGUAGE
        }
    }
}

/// Configure explicitly (also used by tests).
pub fn configure(app0_root: Option<PathBuf>, system_language: i32) {
    let json = app0_root
        .as_ref()
        .and_then(|r| std::fs::read_to_string(r.join("sce_sys").join("param.json")).ok());
    let title_id = json
        .as_deref()
        .and_then(|j| param_json_string(j, "titleId"))
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty());
    let content_version = json.as_deref().and_then(|j| {
        param_json_string(j, "contentVersion")
            .or_else(|| param_json_string(j, "masterVersion"))
            .or_else(|| param_json_string(j, "targetContentVersion"))
    });
    // The GPU pipeline cache is kept per game: title id, else content id.
    let cache_name = title_id.clone().or_else(|| json.as_deref().and_then(|j| param_json_string(j, "contentId")).map(|s| s.trim().to_string()));
    if let Some(name) = cache_name.filter(|s| !s.is_empty()) {
        ps5gpu::pipeline::cache::set_title_id(&name);
    }
    if let Some(size) = json.as_deref().and_then(param_json_flexible_memory_size) {
        crate::memory::set_flexible_memory_size(size);
    }
    // Process start: the memory layout is fixed now, and the native AGC
    // driver takes its memory before the title allocates any.
    crate::agc::defaults::reserve_driver_memory();
    log::info!(
        "[APP] app0={:?} titleId={:?} version={:?} language={}",
        app0_root, title_id, content_version, system_language
    );
    {
        let mut info = INFO.lock().unwrap();
        info.app0_root = app0_root;
        info.title_id = title_id.clone();
        info.content_version = content_version;
        info.system_language = system_language;
    }
    if let Some(root) = app0_root_dir() {
        crate::playgo::set_app_root(root);
    }
    crate::savedata::configure_application_info(title_id.as_deref());
}

pub fn app0_root_dir() -> Option<PathBuf> {
    INFO.lock().unwrap().app0_root.clone()
}

/// Title id from param.json, if known.
pub fn title_id() -> Option<String> {
    INFO.lock().unwrap().title_id.clone()
}

pub fn content_version() -> Option<String> {
    INFO.lock().unwrap().content_version.clone()
}

pub fn system_language() -> i32 {
    INFO.lock().unwrap().system_language
}

/// Read `sce_sys/param.json` from the configured app root.
pub fn read_param_json() -> Option<String> {
    let root = app0_root_dir()?;
    std::fs::read_to_string(root.join("sce_sys").join("param.json")).ok()
}

// ---------------------------------------------------------------------------
// Minimal JSON reader for top-level scalar members of param.json.
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
pub enum JsonScalar {
    Str(String),
    Num(f64),
    Bool(bool),
    Null,
    /// Object or array (not decoded).
    Composite,
}

/// Top-level string member of a JSON object.
pub fn param_json_string(json: &str, key: &str) -> Option<String> {
    match param_json_member(json, key)? {
        JsonScalar::Str(s) => Some(s),
        _ => None,
    }
}

/// Top-level member that is an integral number fitting in i32.
pub fn param_json_i32(json: &str, key: &str) -> Option<i32> {
    match param_json_member(json, key)? {
        JsonScalar::Num(n) if n.fract() == 0.0 && n >= i32::MIN as f64 && n <= i32::MAX as f64 => {
            Some(n as i32)
        }
        _ => None,
    }
}

/// Look up a member of the root object. Returns None on malformed JSON.
pub fn param_json_member(json: &str, key: &str) -> Option<JsonScalar> {
    param_json_raw(json, key).map(|(v, _)| v)
}

/// Raw JSON text of an object/array member of the root object.
pub fn param_json_composite<'a>(json: &'a str, key: &str) -> Option<&'a str> {
    match param_json_raw(json, key)? {
        (JsonScalar::Composite, raw) => Some(raw),
        _ => None,
    }
}

/// `kernel.flexibleMemorySize` from param.json.
pub fn param_json_flexible_memory_size(json: &str) -> Option<u64> {
    let kernel = param_json_composite(json, "kernel")?;
    match param_json_member(kernel, "flexibleMemorySize")? {
        JsonScalar::Num(n) if n.fract() == 0.0 && n >= 0.0 && n <= u64::MAX as f64 => Some(n as u64),
        _ => None,
    }
}

fn param_json_raw<'a>(json: &'a str, key: &str) -> Option<(JsonScalar, &'a str)> {
    let mut p = Parser { s: json.as_bytes(), i: 0 };
    p.ws();
    if p.peek()? != b'{' {
        return None;
    }
    p.i += 1;
    loop {
        p.ws();
        match p.peek()? {
            b'}' => return None,
            b',' => {
                p.i += 1;
                continue;
            }
            b'"' => {}
            _ => return None,
        }
        let name = p.string()?;
        p.ws();
        if p.peek()? != b':' {
            return None;
        }
        p.i += 1;
        p.ws();
        let start = p.i;
        let value = p.value()?;
        if name == key {
            return Some((value, json.get(start..p.i)?));
        }
    }
}

struct Parser<'a> {
    s: &'a [u8],
    i: usize,
}

impl Parser<'_> {
    fn peek(&self) -> Option<u8> {
        self.s.get(self.i).copied()
    }

    fn ws(&mut self) {
        while matches!(self.peek(), Some(b' ' | b'\t' | b'\r' | b'\n')) {
            self.i += 1;
        }
    }

    fn string(&mut self) -> Option<String> {
        // Assumes current byte is the opening quote.
        self.i += 1;
        let mut out: Vec<u8> = Vec::new();
        loop {
            let c = self.peek()?;
            self.i += 1;
            match c {
                b'"' => return String::from_utf8(out).ok(),
                b'\\' => {
                    let e = self.peek()?;
                    self.i += 1;
                    match e {
                        b'"' | b'\\' | b'/' => out.push(e),
                        b'b' => out.push(8),
                        b'f' => out.push(12),
                        b'n' => out.push(b'\n'),
                        b'r' => out.push(b'\r'),
                        b't' => out.push(b'\t'),
                        b'u' => {
                            let hex = std::str::from_utf8(self.s.get(self.i..self.i + 4)?).ok()?;
                            let cp = u32::from_str_radix(hex, 16).ok()?;
                            self.i += 4;
                            let ch = char::from_u32(cp).unwrap_or('\u{FFFD}');
                            let mut buf = [0u8; 4];
                            out.extend_from_slice(ch.encode_utf8(&mut buf).as_bytes());
                        }
                        _ => return None,
                    }
                }
                _ => out.push(c),
            }
        }
    }

    fn value(&mut self) -> Option<JsonScalar> {
        match self.peek()? {
            b'"' => self.string().map(JsonScalar::Str),
            b'{' | b'[' => {
                self.skip_composite()?;
                Some(JsonScalar::Composite)
            }
            b't' => self.literal(b"true", JsonScalar::Bool(true)),
            b'f' => self.literal(b"false", JsonScalar::Bool(false)),
            b'n' => self.literal(b"null", JsonScalar::Null),
            _ => {
                let start = self.i;
                while matches!(self.peek(), Some(b'-' | b'+' | b'.' | b'e' | b'E' | b'0'..=b'9')) {
                    self.i += 1;
                }
                let text = std::str::from_utf8(&self.s[start..self.i]).ok()?;
                text.parse::<f64>().ok().map(JsonScalar::Num)
            }
        }
    }

    fn literal(&mut self, lit: &[u8], v: JsonScalar) -> Option<JsonScalar> {
        if self.s.get(self.i..self.i + lit.len())? == lit {
            self.i += lit.len();
            Some(v)
        } else {
            None
        }
    }

    fn skip_composite(&mut self) -> Option<()> {
        let mut depth = 0usize;
        loop {
            match self.peek()? {
                b'"' => {
                    self.string()?;
                    continue;
                }
                b'{' | b'[' => depth += 1,
                b'}' | b']' => {
                    depth -= 1;
                    if depth == 0 {
                        self.i += 1;
                        return Some(());
                    }
                }
                _ => {}
            }
            self.i += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = r#"{
      "addcont": { "titleId": "NESTED", "list": [1, {"a": "}"}] },
      "applicationCategoryType": 0,
      "contentVersion": "01.000.001",
      "esc": "a\"bA",
      "titleId": "PPSA10112",
      "userDefinedParam1": 3,
      "neg": -7,
      "flag": true
    }"#;

    /// RPS5E_SYSTEM_LANGUAGE: unset or empty is English (US); language
    /// numbers 0..=30 select that language; anything else falls back.
    #[test]
    fn system_language_values() {
        assert_eq!(parse_system_language(None), 1);
        assert_eq!(parse_system_language(Some("")), 1);
        assert_eq!(parse_system_language(Some("21")), 21);
        assert_eq!(parse_system_language(Some("0")), 0);
        assert_eq!(parse_system_language(Some("30")), 30);
        for bad in ["31", "-1", "+21", "21a", "0x15", "arabic", "4294967317"] {
            assert_eq!(parse_system_language(Some(bad)), DEFAULT_SYSTEM_LANGUAGE, "{bad}");
        }
    }

    #[test]
    fn top_level_members_only() {
        assert_eq!(param_json_string(SAMPLE, "titleId").as_deref(), Some("PPSA10112"));
        assert_eq!(param_json_string(SAMPLE, "contentVersion").as_deref(), Some("01.000.001"));
        assert_eq!(param_json_string(SAMPLE, "esc").as_deref(), Some("a\"bA"));
        assert_eq!(param_json_i32(SAMPLE, "userDefinedParam1"), Some(3));
        assert_eq!(param_json_i32(SAMPLE, "neg"), Some(-7));
        assert_eq!(param_json_i32(SAMPLE, "userDefinedParam2"), None);
        assert_eq!(param_json_member(SAMPLE, "flag"), Some(JsonScalar::Bool(true)));
        assert_eq!(param_json_member(SAMPLE, "addcont"), Some(JsonScalar::Composite));
    }

    #[test]
    fn kernel_flexible_memory_size() {
        let j = r#"{"titleId": "X", "kernel": {"cpuPageTableSize": 1, "flexibleMemorySize": 536870912}}"#;
        assert_eq!(param_json_flexible_memory_size(j), Some(512 << 20));
        assert_eq!(param_json_flexible_memory_size(SAMPLE), None);
        assert_eq!(param_json_composite(SAMPLE, "titleId"), None);
    }

    #[test]
    fn malformed_json() {
        assert_eq!(param_json_string("[1,2]", "titleId"), None);
        assert_eq!(param_json_string("{\"titleId\": ", "titleId"), None);
    }
}
