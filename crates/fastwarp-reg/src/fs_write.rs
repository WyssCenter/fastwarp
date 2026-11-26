// SPDX-FileCopyrightText: 2026 Wyss Center for Bio and Neuroengineering
// SPDX-License-Identifier: Apache-2.0

//! Writing a file through a temporary name in its folder, renamed when complete.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use crate::error::{RegError, Result};

/// A temporary file that is removed when dropped, unless it was renamed.
struct PartFile {
    path: PathBuf,
    renamed: bool,
}

impl Drop for PartFile {
    fn drop(&mut self) {
        if !self.renamed {
            let _ = std::fs::remove_file(&self.path);
        }
    }
}

/// Renames `from` to `to`, trying again a few times after `PermissionDenied` or the Windows
/// errors 5 and 32 (access denied, sharing violation: a virus scanner reading the new file or a
/// reader holding the old one).
fn rename_patiently(from: &Path, to: &Path) -> std::io::Result<()> {
    let mut delay = Duration::from_millis(50);
    for _ in 0..5 {
        match std::fs::rename(from, to) {
            Err(e)
                if e.kind() == std::io::ErrorKind::PermissionDenied
                    || (cfg!(windows) && matches!(e.raw_os_error(), Some(5 | 32))) =>
            {
                std::thread::sleep(delay);
                delay = (delay * 2).min(Duration::from_millis(200));
            }
            other => return other,
        }
    }
    std::fs::rename(from, to)
}

/// Writes `path` through `write` into a temporary file in the same folder and renames it to
/// `path` when it is complete, so a failure or a panic leaves neither a truncated file nor the
/// temporary one.
pub(crate) fn write_complete(
    path: &Path,
    write: impl FnOnce(&mut BufWriter<File>) -> Result<()>,
) -> Result<()> {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let io = |source| RegError::Io {
        path: path.to_path_buf(),
        source,
    };
    let name = path
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();
    let n = NEXT.fetch_add(1, Ordering::Relaxed);
    let mut part = PartFile {
        path: path.with_file_name(format!(".{name}.{}.{n}.part", std::process::id())),
        renamed: false,
    };
    let mut out = BufWriter::with_capacity(1 << 20, File::create(&part.path).map_err(io)?);
    write(&mut out)?;
    out.flush().map_err(io)?;
    drop(out);
    rename_patiently(&part.path, path).map_err(io)?;
    part.renamed = true;
    Ok(())
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::error::invalid;

    fn dir(name: &str) -> PathBuf {
        let d =
            std::env::temp_dir().join(format!("fastwarp_fs_write_{}_{name}", std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn a_complete_write_replaces_the_target_and_leaves_no_part_file() {
        let d = dir("ok");
        let target = d.join("out.bin");
        std::fs::write(&target, b"old").unwrap();
        write_complete(&target, |out| {
            out.write_all(b"new").map_err(|source| RegError::Io {
                path: target.clone(),
                source,
            })
        })
        .unwrap();
        assert_eq!(std::fs::read(&target).unwrap(), b"new");
        assert_eq!(std::fs::read_dir(&d).unwrap().count(), 1);
        std::fs::remove_dir_all(&d).unwrap();
    }

    #[test]
    fn a_failed_write_keeps_the_old_target_and_removes_the_part_file() {
        let d = dir("err");
        let target = d.join("out.bin");
        std::fs::write(&target, b"old").unwrap();
        let r = write_complete(&target, |out| {
            out.write_all(b"partial").unwrap();
            Err(invalid("test", "fails"))
        });
        assert!(r.is_err());
        assert_eq!(std::fs::read(&target).unwrap(), b"old");
        assert_eq!(std::fs::read_dir(&d).unwrap().count(), 1);
        std::fs::remove_dir_all(&d).unwrap();
    }

    #[test]
    fn a_panic_while_writing_removes_the_part_file() {
        let d = dir("panic");
        let target = d.join("out.bin");
        let r = std::panic::catch_unwind(|| {
            let _ = write_complete(&target, |_| panic!("boom"));
        });
        assert!(r.is_err());
        assert_eq!(std::fs::read_dir(&d).unwrap().count(), 0);
        std::fs::remove_dir_all(&d).unwrap();
    }
}
