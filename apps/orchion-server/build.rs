use std::env;
use std::ffi::OsStr;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitStatus};

use flate2::read::GzDecoder;

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

const PDFIUM_RELEASE_BASE_URL: &str =
    "https://github.com/bblanchon/pdfium-binaries/releases/download/chromium/7891";

struct PdfiumAsset {
    archive_name: &'static str,
    library_name: &'static str,
}

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

    provision_pdfium(&manifest_dir).unwrap_or_else(|error| {
        panic!("failed to provision PDFium for build: {error}");
    });

    let web_dir = manifest_dir.join("../../web");
    run_bun_command(&web_dir, &["install"], "bun install");
    run_bun_command(&web_dir, &["run", "build"], "bun run build");

    if env::var("PROFILE").as_deref() != Ok("release") {
        return;
    }

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

fn pdfium_asset() -> io::Result<PdfiumAsset> {
    let target_os = env::var("CARGO_CFG_TARGET_OS").map_err(io_other)?;
    let target_arch = env::var("CARGO_CFG_TARGET_ARCH").map_err(io_other)?;

    match (target_os.as_str(), target_arch.as_str()) {
        ("macos", "aarch64") => Ok(PdfiumAsset {
            archive_name: "pdfium-mac-arm64.tgz",
            library_name: "libpdfium.dylib",
        }),
        ("macos", "x86_64") => Ok(PdfiumAsset {
            archive_name: "pdfium-mac-x64.tgz",
            library_name: "libpdfium.dylib",
        }),
        ("linux", "x86_64") => Ok(PdfiumAsset {
            archive_name: "pdfium-linux-x64.tgz",
            library_name: "libpdfium.so",
        }),
        ("windows", "x86_64") => Ok(PdfiumAsset {
            archive_name: "pdfium-win-x64.tgz",
            library_name: "pdfium.dll",
        }),
        _ => Err(io::Error::new(
            io::ErrorKind::Unsupported,
            format!("unsupported PDFium target platform: {target_os}-{target_arch}"),
        )),
    }
}

fn target_profile_dir() -> io::Result<PathBuf> {
    let out_dir = PathBuf::from(
        env::var_os("OUT_DIR")
            .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "OUT_DIR is unavailable"))?,
    );

    out_dir
        .ancestors()
        .nth(3)
        .map(Path::to_path_buf)
        .ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "failed to derive Cargo target profile directory from {}",
                    out_dir.display()
                ),
            )
        })
}

fn provision_pdfium(_manifest_dir: &Path) -> io::Result<()> {
    let asset = pdfium_asset()?;
    let target_library_path = target_profile_dir()?.join(asset.library_name);
    if target_library_path.is_file() {
        println!(
            "cargo:warning=using existing PDFium library {}",
            target_library_path.display()
        );
        return Ok(());
    }

    let out_dir = PathBuf::from(
        env::var_os("OUT_DIR")
            .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "OUT_DIR is unavailable"))?,
    );
    let cache_dir = out_dir.join("pdfium");
    fs::create_dir_all(&cache_dir)?;

    let archive_path = cache_dir.join(asset.archive_name);
    if archive_path.exists() {
        println!(
            "cargo:warning=using cached PDFium archive {}",
            archive_path.display()
        );
    } else {
        let url = format!("{PDFIUM_RELEASE_BASE_URL}/{}", asset.archive_name);
        download_pdfium_archive(&url, &archive_path)?;
    }

    let extract_dir = cache_dir.join(format!("extract-{}", asset.archive_name));
    remove_existing(&extract_dir)?;
    fs::create_dir_all(&extract_dir)?;

    println!(
        "cargo:warning=extracting PDFium archive {}",
        archive_path.display()
    );
    extract_pdfium_archive(&archive_path, &extract_dir)?;

    let library_path = find_file_by_name(&extract_dir, asset.library_name)?.ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::NotFound,
            format!(
                "PDFium archive {} did not contain {}",
                archive_path.display(),
                asset.library_name
            ),
        )
    })?;
    copy_file_overwrite(&library_path, &target_library_path)?;
    println!(
        "cargo:warning=copied PDFium library to {}",
        target_library_path.display()
    );

    Ok(())
}

fn download_pdfium_archive(url: &str, archive_path: &Path) -> io::Result<()> {
    println!("cargo:warning=downloading PDFium archive from {url}");
    let temporary_path = archive_path.with_extension("download");
    remove_existing(&temporary_path)?;

    let response = ureq::get(url).call().map_err(io_other)?;
    let (_, body) = response.into_parts();
    let mut reader = body.into_reader();
    let mut file = fs::File::create(&temporary_path)?;
    io::copy(&mut reader, &mut file)?;
    fs::rename(&temporary_path, archive_path)
}

fn extract_pdfium_archive(archive_path: &Path, extract_dir: &Path) -> io::Result<()> {
    let file = fs::File::open(archive_path)?;
    let decoder = GzDecoder::new(file);
    let mut archive = tar::Archive::new(decoder);
    archive.unpack(extract_dir)
}

fn find_file_by_name(directory: &Path, file_name: &str) -> io::Result<Option<PathBuf>> {
    for entry in sorted_entries(directory)? {
        let path = entry.path();
        let file_type = entry.file_type()?;

        if file_type.is_dir() {
            if let Some(path) = find_file_by_name(&path, file_name)? {
                return Ok(Some(path));
            }
        } else if file_type.is_file() && path.file_name() == Some(OsStr::new(file_name)) {
            return Ok(Some(path));
        }
    }

    Ok(None)
}

fn copy_file_overwrite(source: &Path, destination: &Path) -> io::Result<()> {
    if let Some(parent) = destination.parent() {
        fs::create_dir_all(parent)?;
    }

    let file_name = destination.file_name().ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("destination has no file name: {}", destination.display()),
        )
    })?;
    let temporary_path = destination.with_file_name(format!("{}.tmp", file_name.to_string_lossy()));
    remove_existing(&temporary_path)?;
    fs::copy(source, &temporary_path)?;
    remove_existing(destination)?;
    fs::rename(&temporary_path, destination)
}

fn io_other(error: impl std::fmt::Display) -> io::Error {
    io::Error::other(error.to_string())
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
