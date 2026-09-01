use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;

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

    println!("cargo:rerun-if-env-changed=DEP_LLAMA_GGML_CMAKE_DIR");
    let sys_out_dir = sys_out_dir().unwrap_or_else(|error| {
        panic!("failed to derive llama-cpp-sys-2 OUT_DIR from DEP_LLAMA_GGML_CMAKE_DIR: {error}")
    });
    let cache_path = sys_out_dir.join("build/CMakeCache.txt");
    let cache = load_cmake_cache(cache_path, &features).unwrap_or_else(|error| {
        panic!("failed to load resolved llama-cpp-sys-2 CMake cache: {error}")
    });
    println!("cargo:rerun-if-changed={}", cache.path.display());
    emit_resolved_cache(&sys_out_dir, &cache);
    compile_common_chat_bridge(&cache);
}

struct CacheCandidate {
    path: PathBuf,
    values: BTreeMap<String, String>,
    contents: Vec<u8>,
}

fn sys_out_dir() -> Result<PathBuf, String> {
    let cmake_dir = PathBuf::from(
        std::env::var_os("DEP_LLAMA_GGML_CMAKE_DIR")
            .ok_or_else(|| "environment variable is missing".to_string())?,
    );
    if cmake_dir.file_name().and_then(|name| name.to_str()) != Some("cmake") {
        return Err(format!(
            "unexpected CMake package directory {}",
            cmake_dir.display()
        ));
    }
    let lib_dir = cmake_dir
        .parent()
        .ok_or_else(|| format!("{} has no parent", cmake_dir.display()))?;
    if !matches!(
        lib_dir.file_name().and_then(|name| name.to_str()),
        Some("lib" | "lib64")
    ) {
        return Err(format!(
            "expected lib or lib64 parent, got {}",
            lib_dir.display()
        ));
    }
    Ok(lib_dir
        .parent()
        .ok_or_else(|| format!("{} has no OUT_DIR parent", lib_dir.display()))?
        .to_path_buf())
}

fn load_cmake_cache(path: PathBuf, features: &[String]) -> Result<CacheCandidate, String> {
    let contents =
        std::fs::read(&path).map_err(|error| format!("read {}: {error}", path.display()))?;
    let values = parse_cmake_cache(&contents);
    let target_is_apple = std::env::var("TARGET").is_ok_and(|target| target.contains("apple"));
    if !RESOLVED_KEYS
        .iter()
        .filter(|key| target_is_apple || **key != "CMAKE_OSX_DEPLOYMENT_TARGET")
        .all(|key| values.contains_key(*key))
        || !cache_matches_features(&values, features)
    {
        return Err(format!(
            "{} is incomplete or does not match Cargo features {}",
            path.display(),
            features.join(",")
        ));
    }
    Ok(CacheCandidate {
        path,
        values,
        contents,
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

fn emit_resolved_cache(sys_out_dir: &Path, cache: &CacheCandidate) {
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
        .strip_prefix(sys_out_dir.parent().unwrap_or(sys_out_dir))
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

fn compile_common_chat_bridge(cache: &CacheCandidate) {
    let source = cache
        .values
        .get("CMAKE_HOME_DIRECTORY")
        .map(PathBuf::from)
        .filter(|path| path.join("common/chat.h").is_file())
        .unwrap_or_else(|| {
            panic!("CMAKE_HOME_DIRECTORY does not identify a llama.cpp source tree")
        });
    for path in [
        "native/common_chat_bridge.h",
        "native/common_chat_bridge.cpp",
    ] {
        println!("cargo:rerun-if-changed={path}");
    }
    let mut build = cc::Build::new();
    build
        .cpp(true)
        .file("native/common_chat_bridge.cpp")
        .include("native")
        .include(&source)
        .include(source.join("common"))
        .include(source.join("include"))
        .include(source.join("ggml/include"))
        .include(source.join("vendor"))
        .flag_if_supported("-std=c++17")
        .flag_if_supported("-Wno-unused-function")
        .pic(true);
    if std::env::var("TARGET").is_ok_and(|target| target.ends_with("-windows-msvc")) {
        build.flag("/std:c++17");
    }
    build.compile("orchion_common_chat_bridge");
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
