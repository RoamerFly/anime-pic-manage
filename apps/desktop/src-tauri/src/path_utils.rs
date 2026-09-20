//! Path helpers shared by the desktop command boundary.
//!
//! Windows' `std::fs::canonicalize` returns verbatim paths such as
//! `\\?\D:\Pictures`. That spelling is useful to the OS, but it is an
//! implementation detail and should never leak into the UI, saved scan
//! records, or the Python Worker protocol.

use std::io;
use std::path::{Path, PathBuf};

/// Canonicalize an existing path and remove Windows' verbatim-path prefix.
pub fn canonicalize_for_user(path: impl AsRef<Path>) -> io::Result<PathBuf> {
    std::fs::canonicalize(path).map(strip_verbatim_prefix)
}

/// Return a user-facing spelling while preserving normal and non-Windows paths.
pub fn strip_verbatim_prefix(path: PathBuf) -> PathBuf {
    let text = path.to_string_lossy();
    if let Some(rest) = text.strip_prefix(r"\\?\UNC\") {
        return PathBuf::from(format!(r"\\{rest}"));
    }
    if let Some(rest) = text.strip_prefix(r"\\?\") {
        return PathBuf::from(rest);
    }
    path
}

pub fn display_path(path: &Path) -> String {
    strip_verbatim_prefix(path.to_path_buf())
        .to_string_lossy()
        .into_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn removes_windows_drive_verbatim_prefix() {
        assert_eq!(
            strip_verbatim_prefix(PathBuf::from(r"\\?\F:\P\涩图\画师")),
            PathBuf::from(r"F:\P\涩图\画师")
        );
    }

    #[test]
    fn converts_windows_unc_verbatim_prefix() {
        assert_eq!(
            strip_verbatim_prefix(PathBuf::from(r"\\?\UNC\server\share\images")),
            PathBuf::from(r"\\server\share\images")
        );
    }

    #[test]
    fn keeps_normal_path_unchanged() {
        assert_eq!(
            strip_verbatim_prefix(PathBuf::from(r"D:\Pictures")),
            PathBuf::from(r"D:\Pictures")
        );
    }
}
