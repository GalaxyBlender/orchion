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
