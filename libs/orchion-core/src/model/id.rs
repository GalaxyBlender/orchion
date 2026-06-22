use std::fmt;
use std::str::FromStr;

#[cfg(feature = "serde")]
use serde::{Deserialize, Deserializer, Serialize};
#[cfg(feature = "schema")]
use utoipa::ToSchema;

#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[cfg_attr(feature = "serde", derive(Serialize), serde(transparent))]
#[cfg_attr(feature = "schema", derive(ToSchema))]
pub struct ModelId(String);

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("invalid model id `{value}`; expected {{vendor}}/{{name}}")]
pub struct ParseModelIdError {
    pub value: String,
}

impl ModelId {
    pub fn parse(value: &str) -> Result<Self, ParseModelIdError> {
        let mut parts = value.split('/');
        let Some(vendor) = parts.next() else {
            return Err(ParseModelIdError {
                value: value.to_string(),
            });
        };
        let Some(name) = parts.next() else {
            return Err(ParseModelIdError {
                value: value.to_string(),
            });
        };
        if parts.next().is_some() || !valid_segment(vendor) || !valid_segment(name) {
            return Err(ParseModelIdError {
                value: value.to_string(),
            });
        }
        Ok(Self(value.to_string()))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }

    pub fn vendor(&self) -> &str {
        self.0.split_once('/').map_or("", |(vendor, _)| vendor)
    }

    pub fn name(&self) -> &str {
        self.0.split_once('/').map_or("", |(_, name)| name)
    }
}

impl fmt::Display for ModelId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

impl FromStr for ModelId {
    type Err = ParseModelIdError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        Self::parse(value)
    }
}

impl AsRef<str> for ModelId {
    fn as_ref(&self) -> &str {
        self.as_str()
    }
}

#[cfg(feature = "serde")]
impl<'de> Deserialize<'de> for ModelId {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        Self::parse(&value).map_err(serde::de::Error::custom)
    }
}

fn valid_segment(segment: &str) -> bool {
    !segment.is_empty()
        && segment.chars().all(|character| {
            character.is_ascii_alphanumeric() || matches!(character, '-' | '_' | '.')
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepts_vendor_name_model_ids() {
        let id = ModelId::parse("PaddlePaddle/PaddleOCR-VL-1.6").unwrap();
        assert_eq!(id.as_str(), "PaddlePaddle/PaddleOCR-VL-1.6");
        assert_eq!(id.vendor(), "PaddlePaddle");
        assert_eq!(id.name(), "PaddleOCR-VL-1.6");
    }

    #[test]
    fn rejects_non_repo_model_ids() {
        for value in [
            "",
            "pp-ocrv6",
            "a/b/c",
            "/name",
            "vendor/",
            "vendor/name with space",
            " PaddlePaddle/PaddleOCR-VL-1.6",
            "PaddlePaddle/PaddleOCR-VL-1.6 ",
        ] {
            assert!(ModelId::parse(value).is_err(), "{value} should be rejected");
        }
    }

    #[cfg(feature = "serde")]
    #[test]
    fn model_id_deserialization_validates_input() {
        let id = serde_json::from_str::<ModelId>("\"PaddlePaddle/PaddleOCR-VL-1.6\"").unwrap();
        assert_eq!(id.as_str(), "PaddlePaddle/PaddleOCR-VL-1.6");

        assert!(serde_json::from_str::<ModelId>("\"bad id with space\"").is_err());
    }
}
