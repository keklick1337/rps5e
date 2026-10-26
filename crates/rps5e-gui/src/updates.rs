// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Update checker (opt-in): reads a JSON feed with the latest release tag
//! (`tag`, else `tag_name`) and its https page (`html_url`) and compares the
//! tag with this build's version.

/// This build's release tag.
pub fn current_tag() -> String {
    format!("v{}", env!("CARGO_PKG_VERSION"))
}

#[derive(Clone, Debug, PartialEq)]
pub struct UpdateInfo {
    pub tag: String,
    pub page_url: String,
}

pub fn parse_update_info(data: &[u8]) -> Result<UpdateInfo, String> {
    let doc: serde_json::Value = serde_json::from_slice(data).map_err(|_| "Invalid update feed".to_string())?;
    let root = doc.as_object().ok_or("Invalid update feed")?;
    let s = |k: &str| root.get(k).and_then(|v| v.as_str()).unwrap_or("").trim().to_string();
    let tag = Some(s("tag")).filter(|t| !t.is_empty()).unwrap_or_else(|| s("tag_name"));
    let page_url = s("html_url");
    if tag.is_empty() || !page_url.starts_with("https://") || page_url.len() <= "https://".len() {
        return Err("Incomplete update feed".into());
    }
    Ok(UpdateInfo { tag, page_url })
}

/// Whether `tag` names this build (`v` prefix optional).
pub fn is_current(tag: &str) -> bool {
    tag.trim().trim_start_matches('v') == current_tag().trim_start_matches('v')
}

/// The outcome shown to the user.
#[derive(Clone, Debug, PartialEq)]
pub enum UpdateResult {
    UpToDate(String),
    Available(UpdateInfo),
    Failed(String),
}

/// Check `feed_url` (network).
pub fn check(feed_url: &str) -> UpdateResult {
    if feed_url.trim().is_empty() {
        return UpdateResult::Failed("no update feed is configured".into());
    }
    match crate::net::fetch(feed_url.trim(), 1 << 20).and_then(|d| parse_update_info(&d)) {
        Ok(info) if is_current(&info.tag) => UpdateResult::UpToDate(info.tag),
        Ok(info) => UpdateResult::Available(info),
        Err(e) => UpdateResult::Failed(e),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn feeds() {
        let i = parse_update_info(br#"{"tag": "v1.2.0", "html_url": "https://example.org/r/1.2.0"}"#).unwrap();
        assert_eq!(i, UpdateInfo { tag: "v1.2.0".into(), page_url: "https://example.org/r/1.2.0".into() });
        let i = parse_update_info(br#"{"tag_name": "v9", "html_url": "https://github.com/x/y/releases/v9"}"#).unwrap();
        assert_eq!(i.tag, "v9");
        assert!(parse_update_info(br#"{"tag": "v1", "html_url": "http://insecure"}"#).is_err());
        assert!(parse_update_info(br#"{"html_url": "https://x"}"#).is_err());
        assert!(parse_update_info(b"[]").is_err());
        assert!(is_current(&current_tag()));
        assert!(is_current(env!("CARGO_PKG_VERSION")));
        assert!(!is_current("v999.0.0"));
        assert!(matches!(check(""), UpdateResult::Failed(_)));
    }
}
