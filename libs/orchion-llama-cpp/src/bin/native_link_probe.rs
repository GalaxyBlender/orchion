use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::mtmd::MtmdBitmap;

fn main() {
    if std::env::args().any(|argument| argument == "--metadata-json") {
        println!("{}", orchion_llama_cpp::build_metadata_json());
        return;
    }
    let backend = LlamaBackend::init().expect("llama.cpp backend init must succeed");
    std::hint::black_box(backend.supports_mmap());
    drop(backend);
    std::hint::black_box(MtmdBitmap::from_audio_data as fn(&[f32]) -> _);
    println!(
        "llama-cpp-2={} llama.cpp={} binding_features={} cargo_features={}",
        orchion_llama_cpp::BINDING_REVISION,
        orchion_llama_cpp::LLAMA_CPP_REVISION,
        orchion_llama_cpp::build_metadata().binding_features,
        orchion_llama_cpp::build_metadata().cargo_features,
    );
}
