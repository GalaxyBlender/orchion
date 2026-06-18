use orchion_server::logging::{default_rust_log, rust_log_directive_from_sources};

#[test]
fn default_log_filter_enables_server_info_logs() {
    assert_eq!(
        default_rust_log(),
        "orchion_server=info,orchion=info,tower_http=info"
    );
}

#[test]
fn explicit_rust_log_overrides_dotenv_files() {
    let root = tempfile::tempdir().unwrap();
    let exe_dir = root.path().join("bin");
    let work_dir = root.path().join("work");
    std::fs::create_dir_all(&exe_dir).unwrap();
    std::fs::create_dir_all(&work_dir).unwrap();
    std::fs::write(exe_dir.join(".env"), "RUST_LOG=orchion_server=debug\n").unwrap();
    std::fs::write(work_dir.join(".env"), "RUST_LOG=orchion_server=trace\n").unwrap();

    let directive = rust_log_directive_from_sources(
        &exe_dir.join("orchion-server"),
        &work_dir,
        Some("orchion_server=warn"),
    )
    .unwrap();

    assert_eq!(directive, "orchion_server=warn");
}

#[test]
fn dotenv_prefers_executable_dir_before_current_workdir() {
    let root = tempfile::tempdir().unwrap();
    let exe_dir = root.path().join("bin");
    let work_dir = root.path().join("work");
    std::fs::create_dir_all(&exe_dir).unwrap();
    std::fs::create_dir_all(&work_dir).unwrap();
    std::fs::write(exe_dir.join(".env"), "RUST_LOG=orchion_server=debug\n").unwrap();
    std::fs::write(work_dir.join(".env"), "RUST_LOG=orchion_server=trace\n").unwrap();

    let directive =
        rust_log_directive_from_sources(&exe_dir.join("orchion-server"), &work_dir, None).unwrap();

    assert_eq!(directive, "orchion_server=debug");
}

#[test]
fn dotenv_uses_current_workdir_when_executable_dir_has_no_env_file() {
    let root = tempfile::tempdir().unwrap();
    let exe_dir = root.path().join("bin");
    let work_dir = root.path().join("work");
    std::fs::create_dir_all(&exe_dir).unwrap();
    std::fs::create_dir_all(&work_dir).unwrap();
    std::fs::write(work_dir.join(".env"), "RUST_LOG=orchion_server=trace\n").unwrap();

    let directive =
        rust_log_directive_from_sources(&exe_dir.join("orchion-server"), &work_dir, None).unwrap();

    assert_eq!(directive, "orchion_server=trace");
}
