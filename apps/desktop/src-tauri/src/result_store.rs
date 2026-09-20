use chrono::{SecondsFormat, Utc};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use uuid::Uuid;

const RESULT_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, Copy)]
pub enum ResultKind {
    Recognition,
    Similarity,
}

impl ResultKind {
    fn directory_name(self) -> &'static str {
        match self {
            Self::Recognition => "recognition",
            Self::Similarity => "similarity",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct StoredResult<T> {
    schema_version: u32,
    result_id: String,
    result_type: String,
    directory: String,
    completed_at: String,
    payload: T,
}

#[derive(Debug, Clone)]
pub struct SavedResult {
    pub result_id: String,
    pub completed_at: String,
    pub path: PathBuf,
}

pub fn now_rfc3339() -> String {
    Utc::now().to_rfc3339_opts(SecondsFormat::Millis, true)
}

fn normalize_directory(directory: &str) -> String {
    let normalized = directory.replace('\\', "/");
    let normalized = if normalized
        .get(..8)
        .is_some_and(|prefix| prefix.eq_ignore_ascii_case("//?/UNC/"))
    {
        format!("//{}", &normalized[8..])
    } else if normalized.starts_with("//?/") {
        normalized[4..].to_string()
    } else {
        normalized
    };
    let normalized = normalized.trim_end_matches('/');
    if cfg!(windows) {
        normalized.to_lowercase()
    } else {
        normalized.to_string()
    }
}

fn legacy_directory_key(directory: &str) -> String {
    let normalized = directory.replace('\\', "/");
    let normalized = normalized.trim_end_matches('/');
    let normalized = if cfg!(windows) {
        normalized.to_lowercase()
    } else {
        normalized.to_string()
    };
    stable_directory_key(&normalized)
}

fn stable_directory_key(directory: &str) -> String {
    let mut hash = 0xcbf29ce484222325u64;
    for byte in directory.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    format!("{hash:016x}")
}

/// Compare two library directories tolerating verbatim prefixes and Windows
/// short (8.3) names.
///
/// Saved results store the canonical path returned by `fs::canonicalize`,
/// while the desktop may query with the path the user selected. On Windows the
/// same directory can be spelled `C:\Users\RUNNER~1\...` and
/// `\\?\C:\Users\runneradmin\...`, so a plain string comparison is not enough.
fn directories_match(left: &str, right: &str) -> bool {
    if normalize_directory(left) == normalize_directory(right) {
        return true;
    }
    match (fs::canonicalize(left), fs::canonicalize(right)) {
        (Ok(left), Ok(right)) => {
            normalize_directory(&left.to_string_lossy())
                == normalize_directory(&right.to_string_lossy())
        }
        _ => false,
    }
}

fn directory_key(directory: &str) -> String {
    // Stable FNV-1a key keeps private absolute paths out of result directory names.
    stable_directory_key(&normalize_directory(directory))
}

fn result_directory(data_dir: &Path, kind: ResultKind, directory: &str) -> PathBuf {
    data_dir
        .join("scan-results")
        .join(kind.directory_name())
        .join(directory_key(directory))
}

fn result_files(data_dir: &Path, kind: ResultKind, directory: &str) -> io::Result<Vec<PathBuf>> {
    let kind_root = data_dir.join("scan-results").join(kind.directory_name());
    let mut keys = HashSet::from([directory_key(directory), legacy_directory_key(directory)]);
    if let Ok(canonical) = fs::canonicalize(directory) {
        let canonical = canonical.to_string_lossy();
        // Cover both the current and the legacy key style for the canonical
        // spelling, otherwise a short (8.3) or relative query cannot find
        // results that were saved under the canonical path.
        keys.insert(directory_key(&canonical));
        keys.insert(legacy_directory_key(&canonical));
    }

    let mut files = Vec::new();
    for key in keys {
        let root = kind_root.join(key);
        if !root.is_dir() {
            continue;
        }
        files.extend(
            fs::read_dir(root)?
                .filter_map(Result::ok)
                .map(|entry| entry.path())
                .filter(|path| path.extension().and_then(|value| value.to_str()) == Some("json")),
        );
    }
    files.sort_by(|left, right| right.file_name().cmp(&left.file_name()));
    Ok(files)
}

fn atomic_write(path: &Path, bytes: &[u8]) -> io::Result<()> {
    let parent = path
        .parent()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "结果文件缺少父目录"))?;
    fs::create_dir_all(parent)?;
    let temporary = parent.join(format!(".{}.tmp", Uuid::new_v4()));
    fs::write(&temporary, bytes)?;
    if let Err(error) = fs::rename(&temporary, path) {
        let _ = fs::remove_file(&temporary);
        return Err(error);
    }
    Ok(())
}

pub fn save_versioned<T: Serialize>(
    data_dir: &Path,
    kind: ResultKind,
    directory: &str,
    completed_at: &str,
    payload: &T,
) -> io::Result<SavedResult> {
    let result_id = Uuid::new_v4().to_string();
    let timestamp = Utc::now()
        .timestamp_nanos_opt()
        .unwrap_or_else(|| Utc::now().timestamp_millis() * 1_000_000);
    let path = result_directory(data_dir, kind, directory)
        .join(format!("{timestamp:020}-{result_id}.json"));
    let record = StoredResult {
        schema_version: RESULT_SCHEMA_VERSION,
        result_id: result_id.clone(),
        result_type: kind.directory_name().to_string(),
        directory: directory.to_string(),
        completed_at: completed_at.to_string(),
        payload,
    };
    let bytes = serde_json::to_vec_pretty(&record)
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
    atomic_write(&path, &bytes)?;
    Ok(SavedResult {
        result_id,
        completed_at: completed_at.to_string(),
        path,
    })
}

pub fn load_latest<T: DeserializeOwned>(
    data_dir: &Path,
    kind: ResultKind,
    directory: &str,
) -> io::Result<Option<T>> {
    for path in result_files(data_dir, kind, directory)? {
        let bytes = match fs::read(&path) {
            Ok(bytes) => bytes,
            Err(_) => continue,
        };
        let record: StoredResult<T> = match serde_json::from_slice(&bytes) {
            Ok(record) => record,
            Err(_) => continue,
        };
        if record.schema_version == RESULT_SCHEMA_VERSION
            && directories_match(&record.directory, directory)
        {
            return Ok(Some(record.payload));
        }
    }
    Ok(None)
}

pub fn update_latest<T: Serialize + DeserializeOwned>(
    data_dir: &Path,
    kind: ResultKind,
    directory: &str,
    update: impl FnOnce(&mut T),
) -> io::Result<bool> {
    for path in result_files(data_dir, kind, directory)? {
        let bytes = match fs::read(&path) {
            Ok(bytes) => bytes,
            Err(_) => continue,
        };
        let mut record: StoredResult<T> = match serde_json::from_slice(&bytes) {
            Ok(record) => record,
            Err(_) => continue,
        };
        if record.schema_version != RESULT_SCHEMA_VERSION
            || !directories_match(&record.directory, directory)
        {
            continue;
        }
        update(&mut record.payload);
        let updated = serde_json::to_vec_pretty(&record)
            .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
        atomic_write(&path, &updated)?;
        return Ok(true);
    }
    Ok(false)
}

pub fn delete_latest(data_dir: &Path, kind: ResultKind, directory: &str) -> io::Result<bool> {
    let Some(path) = result_files(data_dir, kind, directory)?.into_iter().next() else {
        return Ok(false);
    };
    fs::remove_file(path)?;
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn versioned_results_keep_history_and_load_latest_per_directory() {
        let root = std::env::temp_dir().join(format!("anime-result-store-{}", Uuid::new_v4()));
        fs::create_dir_all(&root).unwrap();
        save_versioned(
            &root,
            ResultKind::Recognition,
            r"D:\one",
            "first",
            &json!({"value": 1}),
        )
        .unwrap();
        save_versioned(
            &root,
            ResultKind::Recognition,
            r"D:\one",
            "second",
            &json!({"value": 2}),
        )
        .unwrap();
        save_versioned(
            &root,
            ResultKind::Recognition,
            r"D:\two",
            "other",
            &json!({"value": 9}),
        )
        .unwrap();

        let one: serde_json::Value = load_latest(&root, ResultKind::Recognition, "D:/one/")
            .unwrap()
            .unwrap();
        let two: serde_json::Value = load_latest(&root, ResultKind::Recognition, r"D:\two")
            .unwrap()
            .unwrap();
        assert_eq!(one["value"], 2);
        assert_eq!(two["value"], 9);
        assert_eq!(
            result_files(&root, ResultKind::Recognition, r"D:\one")
                .unwrap()
                .len(),
            2
        );
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn windows_verbatim_and_drive_paths_share_the_same_result_key() {
        assert_eq!(
            directory_key(r"\\?\G:\Pictures\Anime"),
            directory_key(r"G:\Pictures\Anime")
        );
        assert_eq!(
            directory_key(r"\\?\UNC\server\share\Anime"),
            directory_key(r"\\server\share\Anime")
        );
    }

    #[cfg(windows)]
    #[test]
    fn loads_results_written_under_the_legacy_verbatim_path_key() {
        let root = std::env::temp_dir().join(format!("anime-result-store-{}", Uuid::new_v4()));
        let library = root.join("library");
        fs::create_dir_all(&library).unwrap();
        let canonical = fs::canonicalize(&library).unwrap();
        let legacy_root = root
            .join("scan-results")
            .join(ResultKind::Similarity.directory_name())
            .join(legacy_directory_key(&canonical.to_string_lossy()));
        fs::create_dir_all(&legacy_root).unwrap();
        let record = StoredResult {
            schema_version: RESULT_SCHEMA_VERSION,
            result_id: "legacy".to_string(),
            result_type: "similarity".to_string(),
            directory: canonical.to_string_lossy().into_owned(),
            completed_at: "legacy".to_string(),
            payload: json!({"value": 7}),
        };
        fs::write(
            legacy_root.join("00000000000000000001-legacy.json"),
            serde_json::to_vec(&record).unwrap(),
        )
        .unwrap();

        let loaded: serde_json::Value =
            load_latest(&root, ResultKind::Similarity, &library.to_string_lossy())
                .unwrap()
                .unwrap();
        assert_eq!(loaded["value"], 7);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    #[cfg(windows)]
    fn directory_keys_are_stable_across_tools() {
        // The key is part of the on-disk contract: headless tooling (and any
        // future importer) has to derive the same folder from the same path.
        // This value was produced by an independent FNV-1a implementation.
        assert_eq!(
            directory_key("D:\\Pictures\\动漫\\Sample-示例角色"),
            "cd5517e004c380a8"
        );
        // Verbatim prefixes and trailing separators must not change the key.
        assert_eq!(
            directory_key("\\\\?\\D:\\Pictures\\动漫\\Sample-示例角色\\"),
            "cd5517e004c380a8"
        );
    }

    #[test]
    fn loads_results_when_the_same_directory_is_spelled_differently() {
        let root = std::env::temp_dir().join(format!("anime-result-store-{}", Uuid::new_v4()));
        let library = root.join("library");
        fs::create_dir_all(&library).unwrap();
        let canonical = fs::canonicalize(&library).unwrap();
        save_versioned(
            &root,
            ResultKind::Similarity,
            &canonical.to_string_lossy(),
            "first",
            &json!({"value": 9}),
        )
        .unwrap();

        // Equivalent spelling (trailing "." segment) must still resolve, which
        // is what happens on Windows when a short 8.3 path meets a canonical one.
        let noisy = format!(
            "{}{}.",
            library.to_string_lossy(),
            std::path::MAIN_SEPARATOR
        );
        let loaded: serde_json::Value = load_latest(&root, ResultKind::Similarity, &noisy)
            .unwrap()
            .unwrap();

        assert_eq!(loaded["value"], 9);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn updating_latest_preserves_older_result() {
        let root = std::env::temp_dir().join(format!("anime-result-store-{}", Uuid::new_v4()));
        fs::create_dir_all(&root).unwrap();
        save_versioned(
            &root,
            ResultKind::Similarity,
            "/images",
            "first",
            &json!({"decision": "old"}),
        )
        .unwrap();
        save_versioned(
            &root,
            ResultKind::Similarity,
            "/images",
            "second",
            &json!({"decision": "new"}),
        )
        .unwrap();
        assert!(update_latest::<serde_json::Value>(
            &root,
            ResultKind::Similarity,
            "/images",
            |payload| {
                payload["decision"] = json!("keep");
            }
        )
        .unwrap());
        let latest: serde_json::Value = load_latest(&root, ResultKind::Similarity, "/images")
            .unwrap()
            .unwrap();
        assert_eq!(latest["decision"], "keep");
        assert_eq!(
            result_files(&root, ResultKind::Similarity, "/images")
                .unwrap()
                .len(),
            2
        );
        fs::remove_dir_all(root).unwrap();
    }
}
