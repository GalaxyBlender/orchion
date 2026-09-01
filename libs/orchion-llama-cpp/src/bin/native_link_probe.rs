use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::mtmd::{MtmdBitmap, MtmdContext, MtmdInputChunks};

fn main() {
    if std::env::args().any(|argument| argument == "--metadata-json") {
        println!("{}", orchion_llama_cpp::build_metadata_json());
        return;
    }
    let backend = LlamaBackend::init().expect("llama.cpp backend init must succeed");
    std::hint::black_box(backend.supports_mmap());
    drop(backend);
    std::hint::black_box(MtmdBitmap::from_audio_data as fn(&[f32]) -> _);
    std::hint::black_box(MtmdBitmap::from_buffer);
    std::hint::black_box(MtmdContext::init_from_file);
    std::hint::black_box(MtmdContext::support_vision as fn(&MtmdContext) -> bool);
    std::hint::black_box(MtmdContext::tokenize);
    std::hint::black_box(MtmdInputChunks::total_tokens as fn(&MtmdInputChunks) -> usize);
    std::hint::black_box(MtmdInputChunks::total_positions as fn(&MtmdInputChunks) -> i32);
    std::hint::black_box(MtmdInputChunks::eval_chunks);
    std::hint::black_box(llama_cpp_2::mtmd::mtmd_default_marker());
    println!(
        "llama-cpp-2={} llama.cpp={} binding_features={} cargo_features={}",
        orchion_llama_cpp::BINDING_REVISION,
        orchion_llama_cpp::LLAMA_CPP_REVISION,
        orchion_llama_cpp::build_metadata().binding_features,
        orchion_llama_cpp::build_metadata().cargo_features,
    );
}
