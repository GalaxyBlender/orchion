use std::fmt;
use std::str::FromStr;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DevicePreference {
    Auto,
    Cpu,
    Metal,
    Cuda(Option<usize>),
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("invalid device `{value}`; expected auto, cpu, metal, metal0, cuda, cuda0, cuda:0, ...")]
pub struct ParseDevicePreferenceError {
    pub value: String,
}

impl FromStr for DevicePreference {
    type Err = ParseDevicePreferenceError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        let trimmed = value.trim();
        let normalized = trimmed.to_ascii_lowercase();

        match normalized.as_str() {
            "auto" => Ok(Self::Auto),
            "cpu" => Ok(Self::Cpu),
            "metal" | "metal0" => Ok(Self::Metal),
            "cuda" => Ok(Self::Cuda(None)),
            _ => parse_cuda_index(&normalized)
                .map(|index| Self::Cuda(Some(index)))
                .ok_or_else(|| ParseDevicePreferenceError {
                    value: value.to_owned(),
                }),
        }
    }
}

impl fmt::Display for DevicePreference {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Auto => formatter.write_str("auto"),
            Self::Cpu => formatter.write_str("cpu"),
            Self::Metal => formatter.write_str("metal"),
            Self::Cuda(None) => formatter.write_str("cuda"),
            Self::Cuda(Some(index)) => write!(formatter, "cuda{index}"),
        }
    }
}

fn parse_cuda_index(value: &str) -> Option<usize> {
    value
        .strip_prefix("cuda:")
        .or_else(|| value.strip_prefix("cuda"))
        .filter(|index| {
            !index.is_empty() && index.chars().all(|character| character.is_ascii_digit())
        })
        .and_then(|index| index.parse().ok())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_supported_device_preferences() {
        assert_eq!(
            "auto".parse::<DevicePreference>().unwrap(),
            DevicePreference::Auto
        );
        assert_eq!(
            " cpu ".parse::<DevicePreference>().unwrap(),
            DevicePreference::Cpu
        );
        assert_eq!(
            "metal".parse::<DevicePreference>().unwrap(),
            DevicePreference::Metal
        );
        assert_eq!(
            "metal0".parse::<DevicePreference>().unwrap(),
            DevicePreference::Metal
        );
        assert_eq!(
            "cuda".parse::<DevicePreference>().unwrap(),
            DevicePreference::Cuda(None)
        );
        assert_eq!(
            "cuda0".parse::<DevicePreference>().unwrap(),
            DevicePreference::Cuda(Some(0))
        );
        assert_eq!(
            "cuda:0".parse::<DevicePreference>().unwrap(),
            DevicePreference::Cuda(Some(0))
        );
        assert_eq!(
            "CUDA:12".parse::<DevicePreference>().unwrap(),
            DevicePreference::Cuda(Some(12))
        );
    }

    #[test]
    fn rejects_malformed_device_preferences() {
        for value in ["", "gpu", "cuda:", "cuda-x", "metal1", "cuda:-1"] {
            assert!(
                value.parse::<DevicePreference>().is_err(),
                "{value} should be rejected"
            );
        }
    }

    #[test]
    fn formats_device_preferences_for_logs() {
        assert_eq!(DevicePreference::Auto.to_string(), "auto");
        assert_eq!(DevicePreference::Cpu.to_string(), "cpu");
        assert_eq!(DevicePreference::Metal.to_string(), "metal");
        assert_eq!(DevicePreference::Cuda(None).to_string(), "cuda");
        assert_eq!(DevicePreference::Cuda(Some(1)).to_string(), "cuda1");
    }
}
