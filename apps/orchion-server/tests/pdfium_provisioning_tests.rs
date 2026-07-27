#[path = "../pdfium_provisioning.rs"]
mod pdfium_provisioning;

use pdfium_provisioning::is_pdfium_install_reusable;

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
