#[path = "../pdfium_provisioning.rs"]
mod pdfium_provisioning;

use pdfium_provisioning::is_pdfium_install_reusable;
use std::path::Path;

const SOURCE_SHA256: &str = "source-checksum";
const LIBRARY_SHA256: &str = "library-checksum";

#[test]
fn matching_source_and_library_checksums_are_reusable() {
    assert!(is_pdfium_install_reusable(
        SOURCE_SHA256,
        Some(SOURCE_SHA256),
        Some(LIBRARY_SHA256),
        Some(LIBRARY_SHA256),
    ));
}

#[test]
fn tampered_library_is_not_reusable() {
    assert!(!is_pdfium_install_reusable(
        SOURCE_SHA256,
        Some(SOURCE_SHA256),
        Some(LIBRARY_SHA256),
        Some("tampered-library-checksum"),
    ));
}

#[test]
fn legacy_source_only_sidecar_is_not_reusable() {
    assert!(!is_pdfium_install_reusable(
        SOURCE_SHA256,
        Some(SOURCE_SHA256),
        None,
        None,
    ));
}

#[test]
fn default_target_dir_is_separate_from_shared_build_dir() {
    assert_eq!(
        pdfium_provisioning::target_profile_dir(
            Path::new("/workspace/apps/orchion-server"),
            None,
            "debug",
            "aarch64-apple-darwin",
            "aarch64-apple-darwin",
            false,
        ),
        Path::new("/workspace/apps/orchion-server/../../target/debug"),
    );
}

#[test]
fn configured_target_dir_is_used_for_cross_compilation() {
    assert_eq!(
        pdfium_provisioning::target_profile_dir(
            Path::new("/workspace/apps/orchion-server"),
            Some(Path::new("/shared/target")),
            "release",
            "x86_64-unknown-linux-gnu",
            "aarch64-apple-darwin",
            false,
        ),
        Path::new("/shared/target/x86_64-unknown-linux-gnu/release"),
    );
}

#[test]
fn explicit_host_target_uses_target_triple_directory() {
    assert_eq!(
        pdfium_provisioning::target_profile_dir(
            Path::new("/workspace/apps/orchion-server"),
            None,
            "debug",
            "aarch64-apple-darwin",
            "aarch64-apple-darwin",
            true,
        ),
        Path::new("/workspace/apps/orchion-server/../../target/aarch64-apple-darwin/debug"),
    );
}
