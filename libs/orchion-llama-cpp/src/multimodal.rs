use std::ffi::CString;
use std::path::Path;

use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::model::LlamaModel;
use llama_cpp_2::mtmd::{
    MtmdBitmap, MtmdContext, MtmdContextParams, MtmdInputChunks, MtmdInputText,
};
use llama_cpp_2::token::data_array::LlamaTokenDataArray;

use crate::common_chat::MEDIA_MARKER;
use crate::contract::{Error, ImageFormat, ImageInput};

pub(crate) struct Projector {
    context: MtmdContext,
}

pub(crate) struct PreparedMedia {
    chunks: MtmdInputChunks,
    _bitmaps: Vec<MtmdBitmap>,
    total_tokens: usize,
    total_positions: i32,
    text_tokens: Vec<llama_cpp_2::token::LlamaToken>,
}

impl Projector {
    pub(crate) fn load(
        path: &Path,
        model: &LlamaModel,
        threads: i32,
        use_gpu: bool,
    ) -> Result<Self, Error> {
        let path = path
            .to_str()
            .ok_or_else(|| Error::InvalidConfig("mmproj path must be valid UTF-8".to_string()))?;
        let params = MtmdContextParams {
            use_gpu,
            print_timings: false,
            n_threads: threads,
            media_marker: CString::new(MEDIA_MARKER).expect("fixed marker has no NUL"),
            image_min_tokens: -1,
            image_max_tokens: -1,
        };
        let context = MtmdContext::init_from_file(path, model, &params)
            .map_err(|error| Error::InvalidConfig(format!("load mmproj: {error}")))?;
        if !context.support_vision() {
            return Err(Error::InvalidConfig(
                "configured mmproj does not support vision".to_string(),
            ));
        }
        Ok(Self { context })
    }

    pub(crate) fn prepare(
        &self,
        prompt: String,
        images: &[ImageInput],
    ) -> Result<PreparedMedia, Error> {
        let marker_count = prompt.matches(MEDIA_MARKER).count();
        if marker_count != images.len() {
            return Err(Error::InvalidConfig(format!(
                "rendered media marker count {marker_count} does not match image count {}",
                images.len()
            )));
        }
        let mut bitmaps = Vec::with_capacity(images.len());
        for image in images {
            let magic_matches = match image.format {
                ImageFormat::Png => image.bytes.starts_with(b"\x89PNG\r\n\x1a\n"),
                ImageFormat::Jpeg => image.bytes.starts_with(&[0xff, 0xd8, 0xff]),
            };
            if !magic_matches {
                return Err(Error::InvalidConfig(
                    "image bytes do not match the declared PNG/JPEG format".to_string(),
                ));
            }
            let bitmap = MtmdBitmap::from_buffer(&self.context, &image.bytes, false)
                .map_err(|error| Error::InvalidConfig(format!("decode image: {error}")))?;
            if bitmap.is_audio() || bitmap.nx() != image.width || bitmap.ny() != image.height {
                return Err(Error::InvalidConfig(
                    "decoded image dimensions or media type do not match validated input"
                        .to_string(),
                ));
            }
            bitmaps.push(bitmap);
        }
        let refs = bitmaps.iter().collect::<Vec<_>>();
        let chunks = self
            .context
            .tokenize(
                MtmdInputText {
                    text: prompt,
                    add_special: true,
                    parse_special: true,
                },
                &refs,
            )
            .map_err(|error| Error::InvalidConfig(format!("tokenize media input: {error}")))?;
        let total_tokens = chunks.total_tokens();
        let total_positions = chunks.total_positions();
        if total_tokens == 0 || total_positions <= 0 {
            return Err(Error::InvalidConfig(
                "media tokenization produced no tokens or positions".to_string(),
            ));
        }
        let mut text_tokens = Vec::new();
        for index in 0..chunks.len() {
            let chunk = chunks.get(index).ok_or_else(|| {
                Error::Generation(format!("media tokenization omitted chunk {index}"))
            })?;
            if let Some(tokens) = chunk.text_tokens() {
                text_tokens.extend_from_slice(tokens);
            }
        }
        Ok(PreparedMedia {
            chunks,
            _bitmaps: bitmaps,
            total_tokens,
            total_positions,
            text_tokens,
        })
    }

    pub(crate) fn eval(
        &self,
        prepared: &PreparedMedia,
        context: &LlamaContext<'_>,
        sequence: i32,
        batch: i32,
    ) -> Result<i32, Error> {
        prepared
            .chunks
            .eval_chunks(&self.context, context, 0, sequence, batch, true)
            .map_err(|error| Error::Generation(format!("multimodal prefill failed: {error}")))
    }
}

impl PreparedMedia {
    pub(crate) const fn total_tokens(&self) -> usize {
        self.total_tokens
    }

    pub(crate) const fn total_positions(&self) -> i32 {
        self.total_positions
    }

    pub(crate) fn text_tokens(&self) -> &[llama_cpp_2::token::LlamaToken] {
        &self.text_tokens
    }

    pub(crate) fn last_logits(context: &LlamaContext<'_>) -> LlamaTokenDataArray {
        context.token_data_array()
    }
}
