// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Trophy viewer data: the game's trophy packages
//! (`sce_sys/trophy2/trophyNN.ucp`) and the unlocks the emulator saved
//! under the data directory (`_Trophies/<TITLE_ID>/trophies_<user>_<NN>.json`).

use ps5hle::np_trophy2::package::{self, Package, Progress, Trophy, UnlockData};
use std::path::{Path, PathBuf};

/// The signed-in user whose unlocks are shown.
pub const USER_ID: i32 = ps5hle::user_service::PRIMARY_USER_ID;

/// One trophy package of a game with its unlocks.
pub struct TrophySet {
    pub label: u32,
    pub package: Package,
    pub unlocks: UnlockData,
}

impl TrophySet {
    pub fn progress(&self) -> Progress {
        package::get_progress(&self.package, &self.unlocks, None)
    }
}

/// `trophyNN.ucp` files of a game folder, sorted, with their service label.
pub fn find_trophy_files(game_dir: &Path) -> Vec<(u32, PathBuf)> {
    let Ok(rd) = std::fs::read_dir(game_dir.join(package::PACKAGE_DIRECTORY)) else { return Vec::new() };
    let mut files: Vec<(u32, PathBuf)> = rd
        .flatten()
        .filter(|e| e.file_type().is_ok_and(|t| t.is_file()))
        .filter_map(|e| {
            let name = e.file_name().to_string_lossy().into_owned();
            let digits = name.strip_prefix("trophy")?.strip_suffix(".ucp")?;
            if digits.is_empty() || !digits.bytes().all(|c| c.is_ascii_digit()) {
                return None;
            }
            Some((digits.parse().ok()?, e.path()))
        })
        .collect();
    files.sort_by(|a, b| a.1.cmp(&b.1));
    files
}

pub fn has_trophy_data(game_dir: &Path) -> bool {
    !find_trophy_files(game_dir).is_empty()
}

/// Load every trophy package of a game. Errors name unreadable packages.
pub fn load_game(game_dir: &Path, title_id: &str, data_dir: &Path, language: i32) -> (Vec<TrophySet>, Vec<String>) {
    let mut sets = Vec::new();
    let mut errors = Vec::new();
    for (label, file) in find_trophy_files(game_dir) {
        let pkg = package::load_package(&file, language);
        if pkg.trophies.is_empty() {
            errors.push(format!("Could not read trophy package {}.", file.file_name().unwrap_or_default().to_string_lossy()));
            continue;
        }
        let unlocks = package::unlocks_path(data_dir, title_id, USER_ID, label)
            .map(|p| package::load_unlock_data(&p))
            .unwrap_or_default();
        sets.push(TrophySet { label, package: pkg, unlocks });
    }
    (sets, errors)
}

/// Progress over all packages of a game.
pub fn game_progress(sets: &[TrophySet]) -> Progress {
    let mut total = Progress::default();
    for s in sets {
        let p = s.progress();
        total.total += p.total;
        total.earned += p.earned;
        for g in 1..5 {
            total.total_grade[g] += p.total_grade[g];
            total.earned_grade[g] += p.earned_grade[g];
        }
    }
    total
}

pub fn grade_name(grade: i32) -> &'static str {
    match grade {
        1 => "Platinum",
        2 => "Gold",
        3 => "Silver",
        4 => "Bronze",
        _ => "",
    }
}

/// Cup colour of a grade (RGB).
pub fn grade_color(grade: i32) -> [u8; 3] {
    match grade {
        1 => [0x7f, 0xc8, 0xff],
        2 => [0xf5, 0xc5, 0x42],
        3 => [0xc4, 0xca, 0xd2],
        _ => [0xcd, 0x7f, 0x4f],
    }
}

/// Name shown for a trophy (hidden ones stay masked until earned).
pub fn trophy_name(id: i32, t: &Trophy, masked: bool) -> String {
    if masked {
        "Hidden trophy".into()
    } else if t.name.is_empty() {
        format!("Trophy {id}")
    } else {
        t.name.clone()
    }
}

/// `yyyy-MM-dd HH:mm:ss` (UTC) of an unlock tick (microseconds since year 1).
pub fn format_tick(tick: u64) -> Option<String> {
    let secs = (tick.checked_sub(package::UNIX_EPOCH_TICK)? / 1_000_000) as i64;
    let (days, rem) = (secs.div_euclid(86_400), secs.rem_euclid(86_400));
    // Civil date from days since 1970-01-01.
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + i64::from(m <= 2);
    Some(format!("{y:04}-{m:02}-{d:02} {:02}:{:02}:{:02}", rem / 3600, rem % 3600 / 60, rem % 60))
}

/// When trophy `id` was earned.
pub fn earned_date(id: i32, unlocks: &UnlockData) -> Option<String> {
    unlocks.timestamps.get(&id).and_then(|&t| format_tick(t))
}

/// Grey copy of an RGBA image (locked trophies).
pub fn grayscale(rgba: &mut [u8]) {
    for px in rgba.chunks_exact_mut(4) {
        let g = ((px[0] as u32 * 11 + px[1] as u32 * 16 + px[2] as u32 * 5) / 32) as u8;
        px[0] = g;
        px[1] = g;
        px[2] = g;
    }
}

/// Decode PNG bytes into RGBA8.
pub fn decode_png(bytes: &[u8]) -> Option<(u32, u32, Vec<u8>)> {
    crate::library::decode_png_rgba(std::io::Cursor::new(bytes))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_package(path: &Path, files: &[(&str, &[u8])]) {
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
            bytes[entry..entry + name.len()].copy_from_slice(name.as_bytes());
            put(&mut bytes, entry + 0x20, offset as u64, 8);
            put(&mut bytes, entry + 0x28, contents.len() as u64, 8);
            bytes[offset..offset + contents.len()].copy_from_slice(contents);
            offset += contents.len();
            entry += 0x40;
        }
        std::fs::write(path, bytes).unwrap();
    }

    const CONF: &str = r#"{"defaultLanguage":"en-US","trophies":[
        {"id":"0","grade":"P"}, {"id":"1","grade":"G"}, {"id":"2","grade":"S","hidden":true}, {"id":"3","grade":"B"}]}"#;
    const META: &str = r#"{"metadata":{"titleMetadata":{"name":"Test game"},"trophyMetadata":[
        {"id":"1","name":"Find All Coins","detail":"Find all coins."}]}}"#;

    #[test]
    fn game_packages_and_unlocks() {
        let root = std::env::temp_dir().join(format!("rps5e-gui-trophy-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let game = root.join("game");
        let data = root.join("data");
        std::fs::create_dir_all(game.join("sce_sys/trophy2")).unwrap();
        write_package(&game.join("sce_sys/trophy2/trophy00.ucp"), &[("tropconf.json", CONF.as_bytes()), ("tropmeta_en-US.json", META.as_bytes())]);
        std::fs::write(game.join("sce_sys/trophy2/trophy01.ucp"), b"broken").unwrap();
        std::fs::write(game.join("sce_sys/trophy2/npbind.dat"), b"x").unwrap();
        assert_eq!(find_trophy_files(&game).iter().map(|f| f.0).collect::<Vec<_>>(), [0, 1]);
        assert!(has_trophy_data(&game) && !has_trophy_data(&data));

        // Unlocks as the emulator writes them.
        let unlocks = data.join(format!("_Trophies/PPSA00001/trophies_{USER_ID}_0.json"));
        std::fs::create_dir_all(unlocks.parent().unwrap()).unwrap();
        let tick = package::UNIX_EPOCH_TICK + 1_700_000_000 * 1_000_000;
        std::fs::write(&unlocks, format!(r#"{{"unlockedAt":{{"1":{tick}}},"unlockedTrophies":[1,3]}}"#)).unwrap();

        let (sets, errors) = load_game(&game, "PPSA00001", &data, 1);
        assert_eq!(errors.len(), 1);
        assert_eq!(sets.len(), 1);
        let s = &sets[0];
        assert_eq!(s.package.title, "Test game");
        let p = game_progress(&sets);
        assert_eq!((p.total, p.earned), (4, 2));
        assert_eq!((p.earned_grade[2], p.earned_grade[4], p.total_grade[1]), (1, 1, 1));
        assert_eq!(p.percentage(), 7 * 100 / 9);
        assert_eq!(earned_date(1, &s.unlocks).as_deref(), Some("2023-11-14 22:13:20"));
        assert_eq!(earned_date(3, &s.unlocks), None);
        let t2 = &s.package.trophies[&2];
        assert!(t2.hidden);
        assert_eq!(trophy_name(2, t2, true), "Hidden trophy");
        assert_eq!(trophy_name(1, &s.package.trophies[&1], false), "Find All Coins");
        assert_eq!(trophy_name(3, &s.package.trophies[&3], false), "Trophy 3");
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn dates_and_colours() {
        assert_eq!(format_tick(package::UNIX_EPOCH_TICK).as_deref(), Some("1970-01-01 00:00:00"));
        assert_eq!(format_tick(package::UNIX_EPOCH_TICK + 951_782_400 * 1_000_000).as_deref(), Some("2000-02-29 00:00:00"));
        assert_eq!(format_tick(0), None);
        let mut px = [255, 0, 0, 7];
        grayscale(&mut px);
        assert_eq!(px, [87, 87, 87, 7]);
        assert_eq!(grade_name(1), "Platinum");
        assert_ne!(grade_color(2), grade_color(4));
    }

    /// ASTRO BOT's trophy packages (found under `GAMES_DIR`; skipped when
    /// absent).
    #[test]
    fn real_trophy_packages() {
        let Some(dir) = ps5vm::test_games::game_dir("PPSA21567") else { return };
        let (sets, errors) = load_game(&dir, "PPSA21567", &std::env::temp_dir(), 1);
        assert!(errors.is_empty(), "{errors:?}");
        assert!(!sets.is_empty() && sets[0].package.trophies.len() > 10);
    }
}
