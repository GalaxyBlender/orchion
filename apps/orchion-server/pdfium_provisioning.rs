use std::path::{Path, PathBuf};

pub(crate) fn is_pdfium_install_reusable(
    expected_source_sha256: &str,
    installed_source_sha256: Option<&str>,
    recorded_library_sha256: Option<&str>,
    actual_library_sha256: Option<&str>,
) -> bool {
    installed_source_sha256.map(str::trim) == Some(expected_source_sha256)
        && recorded_library_sha256.map(str::trim) == actual_library_sha256
        && actual_library_sha256.is_some()
}

pub(crate) fn target_profile_dir(
    manifest_dir: &Path,
    configured_target_dir: Option<&Path>,
    profile: &str,
    target: &str,
    host: &str,
    explicit_target: bool,
) -> PathBuf {
    let mut profile_dir = configured_target_dir
        .map(Path::to_path_buf)
        .unwrap_or_else(|| manifest_dir.join("../../target"));
    if explicit_target || target != host {
        profile_dir.push(target);
    }
    profile_dir.push(profile);
    profile_dir
}
