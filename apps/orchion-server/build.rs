use std::env;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitStatus};

const FRONTEND_INPUTS: &[&str] = &[
    "../../web/package.json",
    "../../web/bun.lock",
    "../../web/bun.lockb",
    "../../web/index.html",
    "../../web/tsconfig.json",
    "../../web/tsconfig.app.json",
    "../../web/tsconfig.node.json",
    "../../web/vite.config.ts",
];

fn main() {
    for path in FRONTEND_INPUTS {
        println!("cargo:rerun-if-changed={path}");
    }

    let manifest_dir = PathBuf::from(
        env::var_os("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR is unavailable"),
    );
    let web_src = manifest_dir.join("../../web/src");
    if web_src.exists() {
        println!("cargo:rerun-if-changed=../../web/src");
        print_source_rerun_if_changed(&manifest_dir, &web_src);
    }

    if env::var("PROFILE").as_deref() != Ok("release") {
        return;
    }

    let web_dir = manifest_dir.join("../../web");
    run_bun_command(&web_dir, &["install"], "bun install");
    run_bun_command(&web_dir, &["run", "build"], "bun run build");

    let dist_dir = web_dir.join("dist");
    if !dist_dir.is_dir() {
        panic!(
            "web/dist does not exist after `bun run build`: {}",
            dist_dir.display()
        );
    }

    let out_dir = PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR is unavailable"));
    let target_dir = out_dir.join("ui-dist");
    remove_existing(&target_dir).unwrap_or_else(|error| {
        panic!(
            "failed to remove previous UI dist at {}: {error}",
            target_dir.display()
        );
    });
    copy_dir_recursive(&dist_dir, &target_dir).unwrap_or_else(|error| {
        panic!(
            "failed to copy UI dist from {} to {}: {error}",
            dist_dir.display(),
            target_dir.display()
        );
    });
}

fn print_source_rerun_if_changed(manifest_dir: &Path, directory: &Path) {
    for entry in sorted_entries(directory).unwrap_or_else(|error| {
        panic!(
            "failed to read frontend source directory {}: {error}",
            directory.display()
        );
    }) {
        let path = entry.path();
        let file_type = entry.file_type().unwrap_or_else(|error| {
            panic!(
                "failed to read frontend source entry type for {}: {error}",
                path.display()
            );
        });

        if file_type.is_dir() {
            print_source_rerun_if_changed(manifest_dir, &path);
        } else if file_type.is_file() {
            print_rerun_if_changed(manifest_dir, &path);
        }
    }
}

fn print_rerun_if_changed(manifest_dir: &Path, path: &Path) {
    let relative_path = path.strip_prefix(manifest_dir).unwrap_or_else(|error| {
        panic!(
            "failed to make frontend path {} relative to {}: {error}",
            path.display(),
            manifest_dir.display()
        );
    });
    let cargo_path = relative_path
        .to_string_lossy()
        .replace(std::path::MAIN_SEPARATOR, "/");
    println!("cargo:rerun-if-changed={cargo_path}");
}

fn sorted_entries(directory: &Path) -> io::Result<Vec<fs::DirEntry>> {
    let mut entries = fs::read_dir(directory)?.collect::<io::Result<Vec<_>>>()?;
    entries.sort_by_key(fs::DirEntry::path);
    Ok(entries)
}

fn run_bun_command(web_dir: &Path, args: &[&str], command_name: &str) {
    let status = Command::new("bun")
        .args(args)
        .current_dir(web_dir)
        .status()
        .unwrap_or_else(|error| {
            panic!(
                "failed to run `{command_name}` in {}: {error}",
                web_dir.display()
            );
        });

    if !status.success() {
        panic!(
            "`{command_name}` failed in {} with status {}",
            web_dir.display(),
            display_status(status)
        );
    }
}

fn display_status(status: ExitStatus) -> String {
    status.code().map_or_else(
        || "terminated by signal".to_string(),
        |code| code.to_string(),
    )
}

fn remove_existing(path: &Path) -> io::Result<()> {
    match fs::symlink_metadata(path) {
        Ok(metadata) if metadata.is_dir() => fs::remove_dir_all(path),
        Ok(_) => fs::remove_file(path),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error),
    }
}

fn copy_dir_recursive(source: &Path, destination: &Path) -> io::Result<()> {
    fs::create_dir_all(destination)?;
    for entry in sorted_entries(source)? {
        let source_path = entry.path();
        let destination_path = destination.join(entry.file_name());
        let file_type = entry.file_type()?;

        if file_type.is_dir() {
            copy_dir_recursive(&source_path, &destination_path)?;
        } else if file_type.is_file() {
            fs::copy(&source_path, destination_path)?;
        } else {
            return Err(io::Error::new(
                io::ErrorKind::Unsupported,
                format!("unsupported UI dist entry {}", source_path.display()),
            ));
        }
    }
    Ok(())
}
