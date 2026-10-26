// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Host-side JSON document model: strict RFC 8259 parsing (no comments,
//! no trailing commas, no BOM, nesting limited to 64 levels) into shared
//! nodes that remember their exact source text. Used for configuration
//! files the emulator reads itself (trophy packages); the guest-visible
//! `sce::Json` objects live in `json.rs`.

use std::sync::Arc;

const MAX_DEPTH: usize = 64;

#[derive(Debug)]
pub(crate) enum JsonKind {
    Null,
    True,
    False,
    /// Raw number text is the node's source slice.
    Number,
    String(String),
    Array(Vec<JsonElem>),
    Object(Vec<(String, JsonElem)>),
}

/// One parsed value plus the exact source text it came from.
#[derive(Debug)]
pub(crate) struct JsonNode {
    pub kind: JsonKind,
    doc: Arc<str>,
    start: usize,
    end: usize,
}

pub(crate) type JsonElem = Arc<JsonNode>;

impl JsonNode {
    /// Source text of this value.
    pub fn raw(&self) -> &str {
        &self.doc[self.start..self.end]
    }

    /// Object member lookup; the last duplicate key wins.
    pub fn property(&self, key: &str) -> Option<JsonElem> {
        match &self.kind {
            JsonKind::Object(m) => m.iter().rev().find(|(k, _)| k == key).map(|(_, v)| v.clone()),
            _ => None,
        }
    }

}

struct Parser<'a> {
    s: &'a [u8],
    pos: usize,
    doc: Arc<str>,
}

impl Parser<'_> {
    fn ws(&mut self) {
        while self.pos < self.s.len() && matches!(self.s[self.pos], b' ' | b'\t' | b'\n' | b'\r') {
            self.pos += 1;
        }
    }

    fn node(&self, kind: JsonKind, start: usize) -> JsonElem {
        Arc::new(JsonNode { kind, doc: self.doc.clone(), start, end: self.pos })
    }

    fn literal(&mut self, word: &[u8]) -> Option<()> {
        if self.s[self.pos..].starts_with(word) {
            self.pos += word.len();
            Some(())
        } else {
            None
        }
    }

    fn value(&mut self, depth: usize) -> Option<JsonElem> {
        self.ws();
        let start = self.pos;
        match *self.s.get(self.pos)? {
            b'n' => {
                self.literal(b"null")?;
                Some(self.node(JsonKind::Null, start))
            }
            b't' => {
                self.literal(b"true")?;
                Some(self.node(JsonKind::True, start))
            }
            b'f' => {
                self.literal(b"false")?;
                Some(self.node(JsonKind::False, start))
            }
            b'"' => {
                let text = self.string()?;
                Some(self.node(JsonKind::String(text), start))
            }
            b'[' => {
                if depth >= MAX_DEPTH {
                    return None;
                }
                self.pos += 1;
                let mut items = Vec::new();
                self.ws();
                if self.s.get(self.pos) == Some(&b']') {
                    self.pos += 1;
                } else {
                    loop {
                        items.push(self.value(depth + 1)?);
                        self.ws();
                        match self.s.get(self.pos)? {
                            b',' => self.pos += 1,
                            b']' => {
                                self.pos += 1;
                                break;
                            }
                            _ => return None,
                        }
                    }
                }
                Some(self.node(JsonKind::Array(items), start))
            }
            b'{' => {
                if depth >= MAX_DEPTH {
                    return None;
                }
                self.pos += 1;
                let mut members = Vec::new();
                self.ws();
                if self.s.get(self.pos) == Some(&b'}') {
                    self.pos += 1;
                } else {
                    loop {
                        self.ws();
                        if self.s.get(self.pos) != Some(&b'"') {
                            return None;
                        }
                        let key = self.string()?;
                        self.ws();
                        if self.s.get(self.pos) != Some(&b':') {
                            return None;
                        }
                        self.pos += 1;
                        let v = self.value(depth + 1)?;
                        members.push((key, v));
                        self.ws();
                        match self.s.get(self.pos)? {
                            b',' => self.pos += 1,
                            b'}' => {
                                self.pos += 1;
                                break;
                            }
                            _ => return None,
                        }
                    }
                }
                Some(self.node(JsonKind::Object(members), start))
            }
            b'-' | b'0'..=b'9' => {
                self.number()?;
                Some(self.node(JsonKind::Number, start))
            }
            _ => None,
        }
    }

    fn digits(&mut self) -> usize {
        let start = self.pos;
        while self.pos < self.s.len() && self.s[self.pos].is_ascii_digit() {
            self.pos += 1;
        }
        self.pos - start
    }

    fn number(&mut self) -> Option<()> {
        if self.s[self.pos] == b'-' {
            self.pos += 1;
        }
        match self.s.get(self.pos)? {
            b'0' => self.pos += 1,
            b'1'..=b'9' => {
                self.digits();
            }
            _ => return None,
        }
        if self.s.get(self.pos) == Some(&b'.') {
            self.pos += 1;
            if self.digits() == 0 {
                return None;
            }
        }
        if matches!(self.s.get(self.pos), Some(b'e' | b'E')) {
            self.pos += 1;
            if matches!(self.s.get(self.pos), Some(b'+' | b'-')) {
                self.pos += 1;
            }
            if self.digits() == 0 {
                return None;
            }
        }
        Some(())
    }

    fn hex4(&mut self) -> Option<u32> {
        let h = self.s.get(self.pos..self.pos + 4)?;
        let t = std::str::from_utf8(h).ok()?;
        if !t.bytes().all(|b| b.is_ascii_hexdigit()) {
            return None;
        }
        self.pos += 4;
        u32::from_str_radix(t, 16).ok()
    }

    fn string(&mut self) -> Option<String> {
        self.pos += 1; // opening quote
        let mut out: Vec<u16> = Vec::new();
        let mut buf = String::new();
        loop {
            let c = *self.s.get(self.pos)?;
            match c {
                b'"' => {
                    self.pos += 1;
                    if !out.is_empty() {
                        buf.push_str(&String::from_utf16_lossy(&out));
                    }
                    return Some(buf);
                }
                b'\\' => {
                    self.pos += 1;
                    let e = *self.s.get(self.pos)?;
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
                            out.push(self.hex4()? as u16);
                            continue;
                        }
                        _ => return None,
                    };
                    if !out.is_empty() {
                        buf.push_str(&String::from_utf16_lossy(&out));
                        out.clear();
                    }
                    buf.push(ch);
                }
                0..=0x1F => return None,
                _ => {
                    if !out.is_empty() {
                        buf.push_str(&String::from_utf16_lossy(&out));
                        out.clear();
                    }
                    // Copy one UTF-8 sequence (the document is valid UTF-8).
                    let len = match c {
                        0x00..=0x7F => 1,
                        0xC0..=0xDF => 2,
                        0xE0..=0xEF => 3,
                        _ => 4,
                    };
                    buf.push_str(std::str::from_utf8(self.s.get(self.pos..self.pos + len)?).ok()?);
                    self.pos += len;
                }
            }
        }
    }
}

/// Parse a complete JSON document. None = invalid.
pub(crate) fn parse_json(bytes: &[u8]) -> Option<JsonElem> {
    let text = std::str::from_utf8(bytes).ok()?;
    let doc: Arc<str> = Arc::from(text);
    let mut p = Parser { s: doc.as_bytes(), pos: 0, doc: doc.clone() };
    let v = p.value(0)?;
    p.ws();
    (p.pos == p.s.len()).then_some(v)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_documents_and_keeps_source_text() {
        let v = parse_json(br#" {"a": [1, -2.5e3, "x\u00e9"], "b": true, "c": null} "#).unwrap();
        let JsonKind::Object(m) = &v.kind else { panic!("not an object") };
        assert_eq!(m.len(), 3);
        let JsonKind::Array(a) = &m[0].1.kind else { panic!("not an array") };
        assert_eq!(a[1].raw(), "-2.5e3");
        assert!(matches!(&a[2].kind, JsonKind::String(s) if s == "x\u{e9}"));
        assert!(matches!(m[1].1.kind, JsonKind::True));
        assert!(matches!(m[2].1.kind, JsonKind::Null));
        for bad in [&b"[1,]"[..], b"{\"a\" 1}", b"01", b"\"\x01\"", b"[1] x", b""] {
            assert!(parse_json(bad).is_none(), "{:?} accepted", String::from_utf8_lossy(bad));
        }
    }
}
