use super::{ModelCategory, ModelId, ModelSpec, ParseModelIdError};
use std::fmt;
use std::str::FromStr;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct AsrModel {
    id: ModelId,
}

impl AsrModel {
    pub fn parse(value: &str) -> Result<Self, ParseModelIdError> {
        Ok(Self {
            id: ModelId::parse(value)?,
        })
    }

    pub fn as_str(&self) -> &str {
        self.id.as_str()
    }
}

impl AsRef<str> for AsrModel {
    fn as_ref(&self) -> &str {
        self.as_str()
    }
}

impl fmt::Display for AsrModel {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

impl ModelSpec for AsrModel {
    fn category(&self) -> ModelCategory {
        ModelCategory::Asr
    }

    fn huggingface_repo(&self) -> &str {
        self.as_str()
    }

    fn modelscope_repo(&self) -> &str {
        self.huggingface_repo()
    }

    fn required_files(&self) -> &'static [&'static str] {
        &["config.json", "tokenizer.json"]
    }
}

impl FromStr for AsrModel {
    type Err = ParseModelIdError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        Self::parse(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_asr_model_names_and_repositories() {
        let model = AsrModel::from_str("Qwen/Qwen3-ASR-0.6B").unwrap();

        assert_eq!(model, AsrModel::parse("Qwen/Qwen3-ASR-0.6B").unwrap());
        assert_eq!(model.huggingface_repo(), "Qwen/Qwen3-ASR-0.6B");
        assert_eq!(model.modelscope_repo(), "Qwen/Qwen3-ASR-0.6B");
    }

    #[test]
    fn accepts_custom_asr_model_ids() {
        let model = AsrModel::from_str("Acme/New-ASR").unwrap();

        assert_eq!(model.huggingface_repo(), "Acme/New-ASR");
        assert_eq!(model.modelscope_repo(), "Acme/New-ASR");
    }

    #[test]
    fn rejects_invalid_asr_model_ids() {
        assert!(AsrModel::from_str("qwen3-asr-0.6b").is_err());
    }

    #[test]
    fn asr_models_expose_stable_metadata() {
        assert_eq!(
            AsrModel::parse("Qwen/Qwen3-ASR-0.6B").unwrap().category(),
            ModelCategory::Asr
        );
        assert_eq!(
            AsrModel::parse("Qwen/Qwen3-ASR-1.7B")
                .unwrap()
                .huggingface_repo(),
            "Qwen/Qwen3-ASR-1.7B"
        );
    }
}
