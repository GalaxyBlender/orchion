use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::SystemTime;

const CMAKE_INPUT_ENV: &[&str] = &[
    "GGML_METAL",
    "GGML_CUDA",
    "GGML_OPENMP",
    "CMAKE_BUILD_TYPE",
    "CMAKE_GENERATOR",
    "CMAKE_OSX_DEPLOYMENT_TARGET",
    "MACOSX_DEPLOYMENT_TARGET",
    "CMAKE_TOOLCHAIN_FILE",
    "CUDA_COMPUTE_CAP",
    "LLAMA_BUILD_SHARED_LIBS",
];

const RESOLVED_KEYS: &[&str] = &[
    "CMAKE_BUILD_TYPE",
    "CMAKE_GENERATOR",
    "CMAKE_OSX_DEPLOYMENT_TARGET",
    "BUILD_SHARED_LIBS",
    "GGML_METAL",
    "GGML_OPENMP",
    "GGML_CUDA",
    "GGML_VULKAN",
    "GGML_NATIVE",
];

fn main() {
    for name in ["TARGET", "PROFILE"] {
        emit_env(
            name,
            &std::env::var(name).unwrap_or_else(|_| "unavailable".to_string()),
        );
    }
    for name in CMAKE_INPUT_ENV {
        println!("cargo:rerun-if-env-changed={name}");
        emit_env(
            &format!("ORCHION_BUILD_INPUT_{name}"),
            &std::env::var(name).unwrap_or_else(|_| "unset".to_string()),
        );
    }

    let mut features = std::env::vars()
        .filter_map(|(name, _)| name.strip_prefix("CARGO_FEATURE_").map(str::to_owned))
        .map(|name| name.to_ascii_lowercase().replace('_', "-"))
        .collect::<Vec<_>>();
    features.sort();
    emit_env("ORCHION_LLAMA_CARGO_FEATURES", &features.join(","));

    let rustc = std::env::var("RUSTC").unwrap_or_else(|_| "rustc".to_string());
    emit_command_output("ORCHION_RUSTC_VERSION", &rustc, &["--version"]);
    emit_command_output("ORCHION_RUSTC_VERBOSE_VERSION", &rustc, &["-vV"]);
    emit_command_output(
        "ORCHION_RUST_TOOLCHAIN",
        "rustup",
        &["show", "active-toolchain"],
    );

    let out_dir = PathBuf::from(std::env::var_os("OUT_DIR").expect("Cargo sets OUT_DIR"));
    let build_root = out_dir
        .parent()
        .and_then(Path::parent)
        .expect("crate OUT_DIR is under the Cargo profile build root");
    let cache = select_cmake_cache(build_root, &features).unwrap_or_else(|error| {
        panic!("failed to locate resolved llama-cpp-sys-2 CMake cache: {error}")
    });
    println!("cargo:rerun-if-changed={}", cache.path.display());
    emit_resolved_cache(build_root, &cache);
}

struct CacheCandidate {
    path: PathBuf,
    values: BTreeMap<String, String>,
    modified: SystemTime,
    contents: Vec<u8>,
}

fn select_cmake_cache(build_root: &Path, features: &[String]) -> Result<CacheCandidate, String> {
    let entries = std::fs::read_dir(build_root)
        .map_err(|error| format!("read {}: {error}", build_root.display()))?;
    let mut candidates = Vec::new();
    for entry in entries.flatten() {
        let file_name = entry.file_name();
        if !file_name.to_string_lossy().starts_with("llama-cpp-sys-2-") {
            continue;
        }
        let path = entry.path().join("out/build/CMakeCache.txt");
        let Ok(contents) = std::fs::read(&path) else {
            continue;
        };
        let values = parse_cmake_cache(&contents);
        let target_is_apple = std::env::var("TARGET").is_ok_and(|target| target.contains("apple"));
        if !RESOLVED_KEYS
            .iter()
            .filter(|key| target_is_apple || **key != "CMAKE_OSX_DEPLOYMENT_TARGET")
            .all(|key| values.contains_key(*key))
            || !cache_matches_features(&values, features)
        {
            continue;
        }
        let modified = std::fs::metadata(&path)
            .and_then(|metadata| metadata.modified())
            .unwrap_or(SystemTime::UNIX_EPOCH);
        candidates.push(CacheCandidate {
            path,
            values,
            modified,
            contents,
        });
    }
    candidates.sort_by(|left, right| {
        right
            .modified
            .cmp(&left.modified)
            .then_with(|| left.path.cmp(&right.path))
    });
    candidates.into_iter().next().ok_or_else(|| {
        format!(
            "no valid cache in {} matching features {}",
            build_root.display(),
            features.join(",")
        )
    })
}

fn parse_cmake_cache(contents: &[u8]) -> BTreeMap<String, String> {
    String::from_utf8_lossy(contents)
        .lines()
        .filter(|line| !line.starts_with("//") && !line.starts_with('#'))
        .filter_map(|line| {
            let (key_and_type, value) = line.split_once('=')?;
            let (key, _) = key_and_type.split_once(':')?;
            Some((key.to_string(), value.trim().to_string()))
        })
        .collect()
}

fn cache_matches_features(values: &BTreeMap<String, String>, features: &[String]) -> bool {
    for (feature, key) in [
        ("cpu-openmp", "GGML_OPENMP"),
        ("cuda", "GGML_CUDA"),
        ("vulkan", "GGML_VULKAN"),
    ] {
        let expected = if features.iter().any(|value| value == feature) {
            "ON"
        } else {
            "OFF"
        };
        if values.get(key).map(String::as_str) != Some(expected) {
            return false;
        }
    }
    if let Ok(expected) = std::env::var("GGML_METAL")
        && values.get("GGML_METAL").map(String::as_str) != Some(expected.as_str())
    {
        return false;
    }
    true
}

fn emit_resolved_cache(build_root: &Path, cache: &CacheCandidate) {
    for key in RESOLVED_KEYS {
        emit_env(
            &format!("ORCHION_BUILD_RESOLVED_{key}"),
            cache
                .values
                .get(*key)
                .map_or("not-applicable", String::as_str),
        );
    }
    let relative = cache
        .path
        .strip_prefix(build_root.parent().unwrap_or(build_root))
        .unwrap_or(&cache.path)
        .to_string_lossy();
    emit_env("ORCHION_BUILD_CMAKE_CACHE_RELATIVE_PATH", &relative);
    emit_env(
        "ORCHION_BUILD_CMAKE_CACHE_SHA256",
        &format!("{:x}", Sha256::digest(&cache.contents)),
    );

    let cmake_root = cache.path.parent().expect("CMake cache has a parent");
    for language in ["C", "CXX"] {
        let compiler = cache
            .values
            .get(&format!("CMAKE_{language}_COMPILER"))
            .map(Path::new)
            .and_then(Path::file_name)
            .and_then(|name| name.to_str())
            .unwrap_or("unavailable");
        emit_env(&format!("ORCHION_BUILD_{language}_COMPILER"), compiler);
        let metadata = find_compiler_metadata(cmake_root, language);
        emit_env(
            &format!("ORCHION_BUILD_{language}_COMPILER_ID"),
            metadata
                .as_ref()
                .and_then(|values| values.get(&format!("CMAKE_{language}_COMPILER_ID")))
                .map_or("unavailable", String::as_str),
        );
        emit_env(
            &format!("ORCHION_BUILD_{language}_COMPILER_VERSION"),
            metadata
                .as_ref()
                .and_then(|values| values.get(&format!("CMAKE_{language}_COMPILER_VERSION")))
                .map_or("unavailable", String::as_str),
        );
    }
}

fn find_compiler_metadata(root: &Path, language: &str) -> Option<BTreeMap<String, String>> {
    let name = format!("CMake{language}Compiler.cmake");
    let cmake_files = std::fs::read_dir(root.join("CMakeFiles")).ok()?;
    let mut paths = cmake_files
        .flatten()
        .map(|version| version.path().join(&name))
        .collect::<Vec<_>>();
    paths.sort();
    for path in paths {
        let Ok(contents) = std::fs::read_to_string(path) else {
            continue;
        };
        let values = contents
            .lines()
            .filter_map(|line| {
                let rest = line.strip_prefix("set(")?;
                let (key, value) = rest.split_once(' ')?;
                Some((
                    key.to_string(),
                    value.trim_end_matches(')').trim_matches('"').to_string(),
                ))
            })
            .collect();
        return Some(values);
    }
    None
}

fn emit_env(name: &str, value: &str) {
    println!(
        "cargo:rustc-env={name}={}",
        value.replace(['\r', '\n'], " ")
    );
}

fn emit_command_output(name: &str, program: &str, args: &[&str]) {
    let value = Command::new(program)
        .args(args)
        .output()
        .ok()
        .filter(|output| output.status.success())
        .map(|output| {
            String::from_utf8_lossy(&output.stdout)
                .trim()
                .replace('\n', " | ")
        })
        .filter(|output| !output.is_empty())
        .unwrap_or_else(|| "unavailable".to_string());
    emit_env(name, &value);
}
