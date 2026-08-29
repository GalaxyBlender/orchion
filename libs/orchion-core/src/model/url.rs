use std::fmt;
use std::str::FromStr;

#[cfg(feature = "serde")]
use serde::{Deserialize, Deserializer, Serialize, Serializer};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ModelUrlSource {
    Neutral,
    HuggingFace,
    ModelScope,
    File,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ModelUrl {
    value: String,
    source: ModelUrlSource,
    owner: Option<String>,
    repository: Option<String>,
    path: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("invalid model URL `{value}`: {message}")]
pub struct ParseModelUrlError {
    pub value: String,
    pub message: &'static str,
}

impl ModelUrl {
    pub fn parse(value: &str) -> Result<Self, ParseModelUrlError> {
        if value.contains('%') {
            return Err(invalid(value, "percent encoding is not allowed"));
        }
        if value.contains('?') {
            return Err(invalid(value, "query strings are not allowed"));
        }
        if value.contains('#') {
            return Err(invalid(value, "fragments are not allowed"));
        }
        if value.contains('\\') {
            return Err(invalid(value, "backslashes are not allowed"));
        }
        if value.chars().any(char::is_control) {
            return Err(invalid(value, "control characters are not allowed"));
        }

        if let Some(path) = value.strip_prefix("file://") {
            return Self::parse_file(value, path);
        }
        if let Some(locator) = value.strip_prefix("hf://") {
            return Self::parse_hub(value, locator, ModelUrlSource::HuggingFace);
        }
        if let Some(locator) = value.strip_prefix("ms://") {
            return Self::parse_hub(value, locator, ModelUrlSource::ModelScope);
        }
        if let Some(locator) = value.strip_prefix("//") {
            return Self::parse_hub(value, locator, ModelUrlSource::Neutral);
        }
        Err(invalid(
            value,
            "expected //, hf://, ms://, or file:/// locator",
        ))
    }

    fn parse_hub(
        value: &str,
        locator: &str,
        source: ModelUrlSource,
    ) -> Result<Self, ParseModelUrlError> {
        let segments = validate_segments(value, locator)?;
        if segments.len() < 2 {
            return Err(invalid(value, "hub locators require owner and repository"));
        }
        let owner = segments[0];
        let repository = segments[1];
        if owner.contains(['@', ':']) || repository.contains(['@', ':']) {
            return Err(invalid(
                value,
                "hub owner and repository cannot contain @ or :",
            ));
        }
        if owner.chars().any(char::is_whitespace)
            || repository.chars().any(char::is_whitespace)
            || segments[2..]
                .iter()
                .any(|segment| segment.chars().any(char::is_whitespace))
        {
            return Err(invalid(value, "whitespace is not allowed"));
        }
        Ok(Self {
            value: value.to_string(),
            source,
            owner: Some(owner.to_string()),
            repository: Some(repository.to_string()),
            path: (!segments[2..].is_empty()).then(|| segments[2..].join("/")),
        })
    }

    fn parse_file(value: &str, path: &str) -> Result<Self, ParseModelUrlError> {
        if !path.starts_with('/') {
            return Err(invalid(
                value,
                "file URLs require an empty authority and absolute path",
            ));
        }
        let raw_path = path.strip_prefix('/').expect("absolute path checked");
        let segments = validate_segments(value, raw_path)?;
        if segments.is_empty() {
            return Err(invalid(value, "file URLs require a non-root absolute path"));
        }
        if segments
            .iter()
            .any(|segment| segment.chars().any(char::is_whitespace))
        {
            return Err(invalid(value, "whitespace is not allowed"));
        }
        Ok(Self {
            value: value.to_string(),
            source: ModelUrlSource::File,
            owner: None,
            repository: None,
            path: Some(path.to_string()),
        })
    }

    #[must_use]
    pub const fn source(&self) -> ModelUrlSource {
        self.source
    }

    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.value
    }

    #[must_use]
    pub fn owner(&self) -> Option<&str> {
        self.owner.as_deref()
    }

    #[must_use]
    pub fn repository(&self) -> Option<&str> {
        self.repository.as_deref()
    }

    #[must_use]
    pub fn path(&self) -> Option<&str> {
        self.path.as_deref()
    }
}

fn validate_segments<'a>(value: &str, path: &'a str) -> Result<Vec<&'a str>, ParseModelUrlError> {
    if path.is_empty() {
        return Ok(Vec::new());
    }
    let segments = path.split('/').collect::<Vec<_>>();
    if segments
        .iter()
        .any(|segment| segment.is_empty() || matches!(*segment, "." | ".."))
    {
        return Err(invalid(
            value,
            "empty, dot, and dot-dot segments are not allowed",
        ));
    }
    Ok(segments)
}

fn invalid(value: &str, message: &'static str) -> ParseModelUrlError {
    ParseModelUrlError {
        value: value.to_string(),
        message,
    }
}

impl fmt::Display for ModelUrl {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

impl FromStr for ModelUrl {
    type Err = ParseModelUrlError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        Self::parse(value)
    }
}

impl AsRef<str> for ModelUrl {
    fn as_ref(&self) -> &str {
        self.as_str()
    }
}

#[cfg(feature = "serde")]
impl Serialize for ModelUrl {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(self.as_str())
    }
}

#[cfg(feature = "serde")]
impl<'de> Deserialize<'de> for ModelUrl {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        Self::parse(&value).map_err(serde::de::Error::custom)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_supported_hub_locators_and_accessors() {
        let cases = [
            ("//owner/repo", ModelUrlSource::Neutral),
            ("hf://owner/repo/file.onnx", ModelUrlSource::HuggingFace),
            ("ms://owner/repo/path/file.onnx", ModelUrlSource::ModelScope),
        ];
        for (value, source) in cases {
            let url = ModelUrl::parse(value).unwrap();
            assert_eq!(url.source(), source);
            assert_eq!(url.owner(), Some("owner"));
            assert_eq!(url.repository(), Some("repo"));
            assert_eq!(url.to_string(), value);
        }
        assert_eq!(
            ModelUrl::parse("hf://owner/repo/file.onnx").unwrap().path(),
            Some("file.onnx")
        );
    }

    #[test]
    fn parses_absolute_file_urls_with_empty_authority() {
        let url = ModelUrl::parse("file:///var/models/model@v1.onnx").unwrap();
        assert_eq!(url.source(), ModelUrlSource::File);
        assert_eq!(url.owner(), None);
        assert_eq!(url.repository(), None);
        assert_eq!(url.path(), Some("/var/models/model@v1.onnx"));
    }

    #[test]
    fn rejects_ambiguous_or_unsafe_locators() {
        for value in [
            "",
            "https://owner/repo",
            "ftp://owner/repo",
            "//owner",
            "//owner/repo/",
            "//owner//repo",
            "//owner/repo/./file",
            "//owner/repo/../file",
            "//owner/repo/%2Ffile",
            "//owner/repo?x=1",
            "//owner/repo#part",
            "//owner/repo\\file",
            "//user@owner/repo",
            "//owner:443/repo",
            "//owner/repo:name",
            "//owner/repo/\nfile",
            "file://host/path",
            "file://relative/path",
            "file:///var//model",
            "file:///var/../model",
            "file:///var/model%20name",
        ] {
            assert!(
                ModelUrl::parse(value).is_err(),
                "{value} should be rejected"
            );
        }
    }

    #[cfg(feature = "serde")]
    #[test]
    fn serde_round_trip_preserves_explicit_source() {
        let url = ModelUrl::parse("ms://owner/repo/path/model.onnx").unwrap();
        let encoded = serde_json::to_string(&url).unwrap();
        assert_eq!(encoded, "\"ms://owner/repo/path/model.onnx\"");
        assert_eq!(serde_json::from_str::<ModelUrl>(&encoded).unwrap(), url);
        assert!(serde_json::from_str::<ModelUrl>("\"https://owner/repo\"").is_err());
    }
}
