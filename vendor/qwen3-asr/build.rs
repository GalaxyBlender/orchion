#[cfg(not(feature = "cuda"))]
fn main() {}

#[cfg(feature = "cuda")]
fn main() -> cudaforge::Result<()> {
    use std::path::PathBuf;

    let out_dir = PathBuf::from(std::env::var_os("OUT_DIR").unwrap());
    cudaforge::KernelBuilder::new()
        .source_files(["kernels/swiglu.cu"])
        .arg("-O3")
        .build_ptx()?
        .write(out_dir.join("swiglu_ptx.rs"))
}
