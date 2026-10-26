// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Cheat files: per-game mods JSON files under `<data dir>/_Patches/
//! <TITLE_ID>.json`, the selection of enabled mods, imports from the online
//! collection (opt-in) and their validation. The emulator applies the file
//! at load (`rps5e-cli run --game-patch`).

use serde_json::Value;
use std::path::{Path, PathBuf};

pub const PATCHES_DIR: &str = "_Patches";

const INDEX_LIMIT: u64 = 2 * 1024 * 1024;
const FILE_LIMIT: u64 = 8 * 1024 * 1024;

pub const REPOSITORY_URL: &str = "https://github.com/TeeKay87/HEN-Cheats-Collection";
const BASE_URL: &str = "https://raw.githubusercontent.com/TeeKay87/HEN-Cheats-Collection/master/cheats/";
pub const FORMATS: [&str; 3] = ["json", "mc4", "shn"];

/// Cheats are supported for PS5 titles (`PPSA00000`).
pub fn is_supported_title_id(title_id: &str) -> bool {
    let id = title_id.trim().to_ascii_uppercase();
    id.len() == 9 && id.starts_with("PPSA") && id[4..].bytes().all(|c| c.is_ascii_digit())
}

/// `<data dir>/_Patches/<TITLE_ID>.json`.
pub fn plan_path(data_dir: &Path, title_id: &str) -> Option<PathBuf> {
    is_supported_title_id(title_id)
        .then(|| data_dir.join(PATCHES_DIR).join(format!("{}.json", title_id.trim().to_ascii_uppercase())))
}

fn validate_document(doc: &Value) -> Result<(), String> {
    let root = doc.as_object().ok_or("Invalid cheat JSON.")?;
    let mods = root.get("mods").and_then(Value::as_array).ok_or("Cheat JSON has no \"mods\" array.")?;
    for m in mods {
        let ok = m.as_object().is_some_and(|o| {
            o.get("name").and_then(Value::as_str).is_some_and(|n| !n.is_empty())
                && o.get("enabled").is_none_or(Value::is_boolean)
        });
        if !ok {
            return Err("Invalid cheat entry.".into());
        }
    }
    Ok(())
}

/// Parse a cheat file (structure only).
pub fn parse_document(data: &[u8]) -> Result<Value, String> {
    let doc: Value = serde_json::from_slice(data).map_err(|_| "Invalid cheat JSON.".to_string())?;
    validate_document(&doc)?;
    Ok(doc)
}

/// One mod as listed: name and whether it is enabled (default on).
pub fn mods(doc: &Value) -> Vec<(String, bool)> {
    doc.get("mods")
        .and_then(Value::as_array)
        .map(|a| {
            a.iter()
                .map(|m| {
                    let name = m.get("name").and_then(Value::as_str).unwrap_or("").to_string();
                    (name, m.get("enabled").and_then(Value::as_bool).unwrap_or(true))
                })
                .collect()
        })
        .unwrap_or_default()
}

fn is_hex_offset(s: &str) -> bool {
    let t = s.strip_prefix("0x").or_else(|| s.strip_prefix("0X")).unwrap_or(s);
    (1..=16).contains(&t.len()) && t.bytes().all(|c| c.is_ascii_hexdigit())
}

fn is_hex_bytes(s: &str) -> bool {
    !s.is_empty() && s.len() % 2 == 0 && s.bytes().all(|c| c.is_ascii_hexdigit())
}

/// Checks for a downloaded file: it must be for this game (title id,
/// version, executable) and use only supported features.
pub fn validate_import(doc: &Value, title_id: &str, version: &str, process: &str) -> Result<(), String> {
    validate_document(doc)?;
    let root = doc.as_object().ok_or("Invalid cheat JSON.")?;
    let s = |k: &str| root.get(k).and_then(Value::as_str).unwrap_or("");
    if !is_supported_title_id(title_id) || !s("id").eq_ignore_ascii_case(title_id.trim()) {
        return Err("The cheat file does not match this game's title ID.".into());
    }
    if version.is_empty() || s("version") != version {
        return Err(format!("The cheat file does not match this game's version ({version})."));
    }
    if process.is_empty() || !s("process").eq_ignore_ascii_case(process) {
        return Err(format!("The cheat file does not match this game's executable ({process})."));
    }
    let mods = root.get("mods").and_then(Value::as_array).cloned().unwrap_or_default();
    if mods.is_empty() {
        return Err("The cheat file contains no cheats.".into());
    }
    if root.contains_key("master") {
        return Err("This JSON uses master codes, which are not supported yet.".into());
    }
    for m in &mods {
        let name = m.get("name").and_then(Value::as_str).unwrap_or("");
        let unsupported = || format!("Unsupported memory entries in cheat \"{name}\".");
        if m.get("module_name").is_some() {
            return Err("This JSON uses module-specific cheats, which are not supported yet.".into());
        }
        let memory = m.get("memory").and_then(Value::as_array).filter(|a| !a.is_empty()).ok_or_else(unsupported)?;
        for entry in memory {
            let w = entry.as_object();
            let f = |k: &str| w.and_then(|o| o.get(k)).and_then(Value::as_str).unwrap_or("");
            let (offset, off, on) = (f("offset"), f("off"), f("on"));
            let absolute = w.and_then(|o| o.get("absolute"));
            let expected = w.and_then(|o| o.get("expected"));
            let addressing = absolute.is_some_and(|a| a.as_bool() != Some(false))
                || expected.is_some_and(|e| e.as_str().is_none_or(|x| !x.eq_ignore_ascii_case(off)))
                || ["section", "sectionName", "sectionProtection", "logicalOffset", "sectionOffset"]
                    .iter()
                    .any(|k| w.is_some_and(|o| o.contains_key(*k)));
            if addressing {
                return Err("This JSON uses memory addressing features that are not supported yet.".into());
            }
            if w.is_none() || !is_hex_offset(offset) || !is_hex_bytes(off) || !is_hex_bytes(on) || off.len() != on.len() {
                return Err(unsupported());
            }
        }
    }
    Ok(())
}

/// The document with `enabled` set per mod (`None` when the counts differ).
pub fn with_selection(doc: &Value, enabled: &[bool]) -> Option<Value> {
    let mut doc = doc.clone();
    let mods = doc.get_mut("mods")?.as_array_mut()?;
    if mods.len() != enabled.len() {
        return None;
    }
    for (m, &on) in mods.iter_mut().zip(enabled) {
        m.as_object_mut()?.insert("enabled".into(), Value::Bool(on));
    }
    Some(doc)
}

/// Save `doc` at `path`, refusing to replace a file changed since it was
/// loaded as `expected` (empty `expected`: the file must not exist).
pub fn save_document(path: &Path, doc: &Value, expected: &[u8]) -> Result<Vec<u8>, String> {
    if !doc.is_object() {
        return Err("Invalid cheat file.".into());
    }
    match std::fs::read(path) {
        Ok(current) if current != expected => return Err("Cheat file changed; reload it.".into()),
        Ok(_) => {}
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            if !expected.is_empty() {
                return Err("Cheat file changed; reload it.".into());
            }
        }
        Err(e) => return Err(format!("Could not read cheat file: {e}")),
    }
    let data = serde_json::to_vec_pretty(doc).map_err(|e| e.to_string())?;
    let text = String::from_utf8(data.clone()).map_err(|e| e.to_string())?;
    crate::settings::write_atomic(path, &text).map_err(|e| format!("Could not save cheat file: {e}"))?;
    Ok(data)
}

/// Credits line of a cheat file: `credits` list, else `author`.
pub fn credits(doc: &Value) -> String {
    let list: Vec<&str> =
        doc.get("credits").and_then(Value::as_array).map(|a| a.iter().filter_map(Value::as_str).collect()).unwrap_or_default();
    if list.is_empty() {
        doc.get("author").and_then(Value::as_str).unwrap_or("").to_string()
    } else {
        list.join(", ")
    }
}

// ---------------------------------------------------------------------------
// Online collection (opt-in)
// ---------------------------------------------------------------------------

/// A file of the online collection.
#[derive(Clone, Debug, PartialEq)]
pub struct RemoteFile {
    pub name: String,
    pub title: String,
    pub version: String,
    pub format: String,
}

/// `AAAA00000_<version>_<rest>.<format>`: (title id, version, format).
fn match_name(name: &str) -> Option<(&str, &str, &str)> {
    let (stem, ext) = name.rsplit_once('.')?;
    if !FORMATS.contains(&ext) {
        return None;
    }
    let (id, rest) = stem.split_once('_')?;
    let (version, tail) = rest.split_once('_')?;
    let id_ok = id.len() == 9 && id[..4].bytes().all(|c| c.is_ascii_uppercase()) && id[4..].bytes().all(|c| c.is_ascii_digit());
    let mut parts = version.split('.');
    let ver_ok = version.contains('.') && parts.all(|p| !p.is_empty() && p.bytes().all(|c| c.is_ascii_digit()));
    let tail_ok = !tail.is_empty() && tail.bytes().all(|c| c.is_ascii_alphanumeric() || matches!(c, b'_' | b'.' | b'-'));
    (id_ok && ver_ok && tail_ok).then_some((id, version, ext))
}

impl RemoteFile {
    pub fn url(&self) -> Option<String> {
        let (_, _, ext) = match_name(&self.name)?;
        (FORMATS.contains(&self.format.as_str()) && ext == self.format).then(|| format!("{BASE_URL}{}/{}", self.format, self.name))
    }

    /// Only JSON files can be imported.
    pub fn is_supported(&self) -> bool {
        self.format == "json" && self.url().is_some()
    }
}

/// Parse an index file (`<name>=<title>` lines) for one format, keeping the
/// files of `title_id`. Returns the files and an error note.
pub fn parse_index(data: &[u8], format: &str, title_id: &str) -> (Vec<RemoteFile>, Option<String>) {
    if !FORMATS.contains(&format) || data.is_empty() || data.len() as u64 > INDEX_LIMIT {
        return (Vec::new(), Some("Invalid cheat index.".into()));
    }
    let id = title_id.trim().to_ascii_uppercase();
    let text = String::from_utf8_lossy(data);
    let text = text.trim_start_matches('\u{feff}');
    let mut files: Vec<RemoteFile> = Vec::new();
    let mut invalid = 0;
    for line in text.lines() {
        let t = line.trim();
        if t.is_empty() {
            continue;
        }
        let Some((name, title)) = t.split_once('=') else {
            invalid += 1;
            continue;
        };
        let Some((fid, version, ext)) = match_name(name).filter(|m| m.2 == format) else {
            invalid += 1;
            continue;
        };
        if fid != id || files.iter().any(|f| f.name == name) {
            continue;
        }
        files.push(RemoteFile { name: name.to_string(), title: title.trim().to_string(), version: version.to_string(), format: ext.to_string() });
    }
    let err = (invalid != 0).then(|| format!("Skipped {invalid} invalid cheat index entries."));
    (files, err)
}

/// Sort: newest version first, then format, then name; the installed
/// version first of all.
pub fn sort_files(files: &mut [RemoteFile], installed: &str) {
    files.sort_by(|a, b| b.version.cmp(&a.version).then(a.format.cmp(&b.format)).then(a.name.cmp(&b.name)));
    files.sort_by_key(|f| f.version != installed);
}

/// Load the catalog of every format for `title_id` (network).
pub fn load_catalog(title_id: &str) -> (Vec<RemoteFile>, String) {
    let mut files = Vec::new();
    let mut errors = Vec::new();
    for format in FORMATS {
        let res = crate::net::fetch(&format!("{BASE_URL}{format}.txt"), INDEX_LIMIT);
        let err = match res {
            Ok(data) => {
                let (f, e) = parse_index(&data, format, title_id);
                files.extend(f);
                e
            }
            Err(e) => Some(e),
        };
        if let Some(e) = err {
            errors.push(format!("{}: {e}", format.to_ascii_uppercase()));
        }
    }
    (files, errors.join("\n"))
}

/// Download one file of the collection (network).
pub fn fetch_file(file: &RemoteFile) -> Result<Vec<u8>, String> {
    if !file.is_supported() {
        return Err("Only JSON cheat files are supported.".into());
    }
    crate::net::fetch(&file.url().unwrap_or_default(), FILE_LIMIT)
}

#[cfg(test)]
mod tests {
    use super::*;

    const DOC: &str = r#"{"name": "Game Cheats", "id": "PPSA01234", "version": "01.000.000", "process": "eboot.bin",
        "credits": ["a", "b"],
        "mods": [{"name": "One", "memory": [{"offset": "0x10", "on": "90", "off": "C3"}]},
                 {"name": "Two", "enabled": false, "memory": [{"offset": "20", "on": "0101", "off": "0000"}]}]}"#;

    #[test]
    fn title_ids_and_paths() {
        assert!(is_supported_title_id(" ppsa01234 "));
        assert!(!is_supported_title_id("CUSA01234"));
        assert!(!is_supported_title_id("PPSA1234"));
        assert_eq!(plan_path(Path::new("/d"), "ppsa01234"), Some(PathBuf::from("/d/_Patches/PPSA01234.json")));
        assert_eq!(plan_path(Path::new("/d"), "../x"), None);
    }

    #[test]
    fn documents_and_validation() {
        let doc = parse_document(DOC.as_bytes()).unwrap();
        assert_eq!(mods(&doc), [("One".to_string(), true), ("Two".to_string(), false)]);
        assert_eq!(credits(&doc), "a, b");
        assert!(validate_import(&doc, "PPSA01234", "01.000.000", "EBOOT.BIN").is_ok());
        assert!(validate_import(&doc, "PPSA01235", "01.000.000", "eboot.bin").unwrap_err().contains("title ID"));
        assert!(validate_import(&doc, "PPSA01234", "01.000.001", "eboot.bin").unwrap_err().contains("version"));
        assert!(validate_import(&doc, "PPSA01234", "01.000.000", "x.elf").unwrap_err().contains("executable"));
        assert!(parse_document(b"{}").unwrap_err().contains("mods"));
        assert!(parse_document(br#"{"mods":[{"name":""}]}"#).is_err());
        assert!(parse_document(br#"{"mods":[{"name":"x","enabled":"yes"}]}"#).is_err());
        assert!(parse_document(b"nope").is_err());
        let bad = |patch: &str| {
            let text = DOC.replace(r#""offset": "0x10""#, patch);
            validate_import(&parse_document(text.as_bytes()).unwrap(), "PPSA01234", "01.000.000", "eboot.bin")
        };
        assert!(bad(r#""offset": "0x10", "absolute": true"#).unwrap_err().contains("addressing"));
        assert!(bad(r#""offset": "0x10", "absolute": false"#).is_ok());
        assert!(bad(r#""offset": "0x10", "expected": "c3""#).is_ok());
        assert!(bad(r#""offset": "0x10", "expected": "00""#).unwrap_err().contains("addressing"));
        assert!(bad(r#""offset": "0x10", "section": 1"#).unwrap_err().contains("addressing"));
        assert!(bad(r#""offset": "xyz""#).unwrap_err().contains("Unsupported"));
        let master = DOC.replacen('{', r#"{"master": [],"#, 1);
        assert!(validate_import(&parse_document(master.as_bytes()).unwrap(), "PPSA01234", "01.000.000", "eboot.bin")
            .unwrap_err()
            .contains("master"));
    }

    #[test]
    fn selection_and_safe_save() {
        let doc = parse_document(DOC.as_bytes()).unwrap();
        let sel = with_selection(&doc, &[false, true]).unwrap();
        assert_eq!(mods(&sel), [("One".to_string(), false), ("Two".to_string(), true)]);
        assert!(with_selection(&doc, &[true]).is_none());

        let dir = std::env::temp_dir().join(format!("rps5e-gui-cheats-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let path = plan_path(&dir, "PPSA01234").unwrap();
        let saved = save_document(&path, &sel, b"").unwrap();
        assert_eq!(std::fs::read(&path).unwrap(), saved);
        // Stale expectation: refused.
        assert!(save_document(&path, &doc, b"").unwrap_err().contains("changed"));
        let again = save_document(&path, &doc, &saved).unwrap();
        assert_eq!(mods(&parse_document(&again).unwrap())[1].1, false);
        let _ = std::fs::remove_dir_all(&dir);
        assert!(save_document(&path, &doc, b"x").unwrap_err().contains("changed"));
    }

    #[test]
    fn index_parsing() {
        let index = "\u{feff}PPSA01234_01.000.000_Game.json=Game Cheats\nPPSA01234_01.002.000_Game.json=Newer\n\
                     PPSA09999_01.000.000_Other.json=Other\nbroken line\nPPSA01234_1_x.json=bad version\n\
                     PPSA01234_01.000.000_Game.json=dup\nPPSA01234_01.000.000_Game.mc4=wrong format\n";
        let (mut files, err) = parse_index(index.as_bytes(), "json", "ppsa01234");
        assert_eq!(files.len(), 2);
        assert_eq!(err.as_deref(), Some("Skipped 3 invalid cheat index entries."));
        assert_eq!(files[0].title, "Game Cheats");
        assert_eq!(
            files[0].url().as_deref(),
            Some("https://raw.githubusercontent.com/TeeKay87/HEN-Cheats-Collection/master/cheats/json/PPSA01234_01.000.000_Game.json")
        );
        assert!(files[0].is_supported());
        sort_files(&mut files, "01.000.000");
        assert_eq!(files[0].version, "01.000.000");
        sort_files(&mut files, "none");
        assert_eq!(files[0].version, "01.002.000");
        let mc4 = RemoteFile { name: "PPSA01234_01.000.000_G.mc4".into(), title: String::new(), version: "01.000.000".into(), format: "mc4".into() };
        assert!(mc4.url().is_some() && !mc4.is_supported());
        assert!(parse_index(b"", "json", "PPSA01234").1.is_some());
        assert!(parse_index(b"x=y", "exe", "PPSA01234").1.is_some());
    }
}
