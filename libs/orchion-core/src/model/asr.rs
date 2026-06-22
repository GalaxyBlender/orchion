use super::{ModelCategory, ModelSpec};
use crate::{OrchionError, Result};
use std::str::FromStr;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AsrModel {
    Qwen3Asr06B,
    Qwen3Asr17B,
}

impl ModelSpec for AsrModel {
    fn category(self) -> ModelCategory {
        ModelCategory::Asr
    }

    fn huggingface_repo(self) -> &'static str {
        match self {
            Self::Qwen3Asr06B => "Qwen/Qwen3-ASR-0.6B",
            Self::Qwen3Asr17B => "Qwen/Qwen3-ASR-1.7B",
        }
    }

    fn modelscope_repo(self) -> &'static str {
        self.huggingface_repo()
    }

    fn required_files(self) -> &'static [&'static str] {
        &["config.json", "tokenizer.json"]
    }
}

impl FromStr for AsrModel {
    type Err = OrchionError;

    fn from_str(value: &str) -> Result<Self> {
        match value {
            "Qwen/Qwen3-ASR-0.6B" => Ok(Self::Qwen3Asr06B),
            "Qwen/Qwen3-ASR-1.7B" => Ok(Self::Qwen3Asr17B),
            _ => Err(OrchionError::ModelLoad {
                source: anyhow::anyhow!("unknown ASR model `{value}`"),
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_asr_model_names_and_repositories() {
        let model = AsrModel::from_str("Qwen/Qwen3-ASR-0.6B").unwrap();

        assert_eq!(model, AsrModel::Qwen3Asr06B);
        assert_eq!(model.huggingface_repo(), "Qwen/Qwen3-ASR-0.6B");
        assert_eq!(model.modelscope_repo(), "Qwen/Qwen3-ASR-0.6B");
    }

    #[test]
    fn rejects_legacy_asr_aliases() {
        assert!(AsrModel::from_str("qwen3-asr-0.6b").is_err());
    }

    #[test]
    fn asr_models_expose_stable_metadata() {
        assert_eq!(AsrModel::Qwen3Asr06B.category(), ModelCategory::Asr);
        assert_eq!(
            AsrModel::Qwen3Asr17B.huggingface_repo(),
            "Qwen/Qwen3-ASR-1.7B"
        );
    }
}
