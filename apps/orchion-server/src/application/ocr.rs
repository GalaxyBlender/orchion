use super::{RuntimeError, UseCaseError};
use orchion::{ModelId, OcrLimits, OcrOptions, OcrResponseFormat, OcrResult, OcrTask};
use std::future::Future;
use std::path::PathBuf;
use std::pin::Pin;

pub type OcrFuture<'a> =
    Pin<Box<dyn Future<Output = Result<Option<OcrResult>, RuntimeError>> + Send + 'a>>;

#[derive(Debug, Clone)]
pub struct OcrServicePolicy {
    pub active: bool,
    pub default_model: Option<ModelId>,
    pub available_models: Vec<ModelId>,
    pub layout_default_model: Option<ModelId>,
    pub layout_available_models: Vec<ModelId>,
    pub format: OcrResponseFormat,
    pub max_pixels: u64,
}

#[derive(Debug, Clone)]
pub struct OcrVlServicePolicy {
    pub active: bool,
    pub default_model: Option<ModelId>,
    pub available_models: Vec<ModelId>,
    pub layout_default_model: Option<ModelId>,
    pub layout_available_models: Vec<ModelId>,
    pub format: OcrResponseFormat,
    pub max_tokens: usize,
    pub max_pixels: u64,
}

#[derive(Debug, Clone)]
pub struct OcrPolicy {
    pub ocr: OcrServicePolicy,
    pub ocr_vl: OcrVlServicePolicy,
}

pub trait OcrRuntime: Send + Sync {
    fn ocr_policy(&self) -> OcrPolicy;

    fn recognize(
        &self,
        choice: OcrServiceChoice,
        image_path: PathBuf,
        options: OcrOptions,
        limits: OcrLimits,
    ) -> OcrFuture<'_>;
}

#[derive(Debug)]
pub struct OcrCommand {
    pub image_path: PathBuf,
    pub model: Option<String>,
    pub response_format: Option<OcrResponseFormat>,
    pub task: OcrTask,
    pub layout_model: Option<ModelId>,
    pub max_tokens: Option<usize>,
}

#[derive(Debug)]
pub struct OcrUseCaseResult {
    pub format: OcrResponseFormat,
    pub result: OcrResult,
}

/// # Errors
///
/// Returns [`UseCaseError`] when request validation, image validation, or recognition fails.
pub async fn recognize(
    runtime: &impl OcrRuntime,
    command: OcrCommand,
) -> Result<OcrUseCaseResult, UseCaseError> {
    let policy = runtime.ocr_policy();
    let choice =
        resolve_service_choice(&policy, command.model.as_deref(), command.response_format)?;
    let response_format = resolve_response_format(&policy, &choice, command.response_format);
    let layout_model = resolve_layout_model(&policy, &choice, command.layout_model);
    validate_parameters(
        &choice,
        response_format,
        command.task,
        layout_model.as_ref(),
        command.max_tokens,
        &policy,
    )?;
    let max_tokens = resolve_max_tokens(&choice, command.max_tokens, &policy);
    let max_pixels = match &choice {
        OcrServiceChoice::Ocr { .. } => policy.ocr.max_pixels,
        OcrServiceChoice::OcrVl { .. } => policy.ocr_vl.max_pixels,
    };
    let image_path = command.image_path;
    let validation_path = image_path.clone();
    tokio::task::spawn_blocking(move || {
        orchion::validate_ocr_image_file(&validation_path, max_pixels)
    })
    .await
    .map_err(|error| UseCaseError::Internal(error.to_string()))??;

    let options = OcrOptions {
        response_format,
        task: command.task,
        layout_model,
        max_tokens,
    };
    let unavailable_model = choice.model().to_string();
    let result = runtime
        .recognize(
            choice,
            image_path,
            options,
            OcrLimits {
                max_pixels: Some(max_pixels),
            },
        )
        .await
        .map_err(UseCaseError::from)?
        .ok_or(UseCaseError::ModelNotAvailable(unavailable_model))?;

    Ok(OcrUseCaseResult {
        format: response_format,
        result,
    })
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OcrServiceChoice {
    Ocr { model: ModelId },
    OcrVl { model: ModelId },
}

impl OcrServiceChoice {
    const fn ocr(model: ModelId) -> Self {
        Self::Ocr { model }
    }

    const fn ocr_vl(model: ModelId) -> Self {
        Self::OcrVl { model }
    }

    #[must_use]
    pub const fn model(&self) -> &ModelId {
        match self {
            Self::Ocr { model } | Self::OcrVl { model } => model,
        }
    }

    #[must_use]
    pub const fn is_ocr_vl(&self) -> bool {
        matches!(self, Self::OcrVl { .. })
    }
}

/// # Errors
///
/// Returns [`UseCaseError`] when no compatible configured OCR service can be selected.
pub fn resolve_service_choice(
    policy: &OcrPolicy,
    model: Option<&str>,
    response_format: Option<OcrResponseFormat>,
) -> Result<OcrServiceChoice, UseCaseError> {
    if let Some(model) = model {
        return resolve_explicit_model(policy, model);
    }

    match (policy.ocr.active, policy.ocr_vl.active) {
        (true, false) => Ok(OcrServiceChoice::ocr(default_choice(policy, false)?)),
        (false, true) => Ok(OcrServiceChoice::ocr_vl(default_choice(policy, true)?)),
        (true, true) => resolve_default_model(policy, response_format),
        (false, false) => Err(UseCaseError::invalid(
            "`model` is required",
            Some("model"),
            "missing_required_parameter",
        )),
    }
}

fn resolve_explicit_model(
    policy: &OcrPolicy,
    model: &str,
) -> Result<OcrServiceChoice, UseCaseError> {
    let model_id =
        ModelId::parse(model).map_err(|_| UseCaseError::ModelNotAvailable(model.to_string()))?;
    let ocr_match = policy.ocr.active && policy.ocr.available_models.contains(&model_id);
    let ocr_vl_match = policy.ocr_vl.active && policy.ocr_vl.available_models.contains(&model_id);
    match (ocr_match, ocr_vl_match) {
        (true, _) => Ok(OcrServiceChoice::ocr(model_id)),
        (false, true) => Ok(OcrServiceChoice::ocr_vl(model_id)),
        (false, false) => Err(UseCaseError::ModelNotAvailable(model.to_string())),
    }
}

fn resolve_default_model(
    policy: &OcrPolicy,
    response_format: Option<OcrResponseFormat>,
) -> Result<OcrServiceChoice, UseCaseError> {
    let prefer_ocr_vl = matches!(
        response_format,
        Some(OcrResponseFormat::Markdown | OcrResponseFormat::Html)
    );
    if prefer_ocr_vl {
        if effective_default_model(policy, true).is_some() {
            Ok(OcrServiceChoice::ocr_vl(default_choice(policy, true)?))
        } else {
            Ok(OcrServiceChoice::ocr(default_choice(policy, false)?))
        }
    } else if effective_default_model(policy, false).is_some() {
        Ok(OcrServiceChoice::ocr(default_choice(policy, false)?))
    } else {
        Ok(OcrServiceChoice::ocr_vl(default_choice(policy, true)?))
    }
}

#[must_use]
pub fn resolve_response_format(
    policy: &OcrPolicy,
    choice: &OcrServiceChoice,
    response_format: Option<OcrResponseFormat>,
) -> OcrResponseFormat {
    response_format.unwrap_or(match choice {
        OcrServiceChoice::Ocr { .. } => policy.ocr.format,
        OcrServiceChoice::OcrVl { .. } => policy.ocr_vl.format,
    })
}

#[must_use]
pub fn resolve_layout_model(
    policy: &OcrPolicy,
    choice: &OcrServiceChoice,
    layout_model: Option<ModelId>,
) -> Option<ModelId> {
    layout_model.or_else(|| match choice {
        OcrServiceChoice::Ocr { .. } => policy.ocr.layout_default_model.clone(),
        OcrServiceChoice::OcrVl { .. } => policy.ocr_vl.layout_default_model.clone(),
    })
}

fn default_choice(policy: &OcrPolicy, ocr_vl: bool) -> Result<ModelId, UseCaseError> {
    let Some(default_model) = effective_default_model(policy, ocr_vl) else {
        return Err(UseCaseError::invalid(
            "`model` is required",
            Some("model"),
            "missing_required_parameter",
        ));
    };
    Ok(default_model.clone())
}

fn effective_default_model(policy: &OcrPolicy, ocr_vl: bool) -> Option<&ModelId> {
    if ocr_vl {
        if !policy.ocr_vl.active {
            return None;
        }
        policy
            .ocr_vl
            .default_model
            .as_ref()
            .or_else(|| policy.ocr_vl.available_models.first())
    } else {
        if !policy.ocr.active {
            return None;
        }
        policy
            .ocr
            .default_model
            .as_ref()
            .or_else(|| policy.ocr.available_models.first())
    }
}

/// # Errors
///
/// Returns [`UseCaseError`] when parameters are incompatible with the selected OCR service.
pub fn validate_parameters(
    choice: &OcrServiceChoice,
    response_format: OcrResponseFormat,
    task: OcrTask,
    layout_model: Option<&ModelId>,
    max_tokens: Option<usize>,
    policy: &OcrPolicy,
) -> Result<(), UseCaseError> {
    if choice.is_ocr_vl() && max_tokens.is_some_and(|value| value > policy.ocr_vl.max_tokens) {
        return Err(UseCaseError::invalid(
            format!(
                "`max_tokens` must not exceed the configured limit of {}",
                policy.ocr_vl.max_tokens
            ),
            Some("max_tokens"),
            "max_tokens_exceeded",
        ));
    }
    if !choice.is_ocr_vl() && response_format == OcrResponseFormat::Html {
        return Err(UseCaseError::invalid(
            "selected OCR model does not support HTML responses",
            Some("response_format"),
            "unsupported_response_format",
        ));
    }
    if matches!(
        response_format,
        OcrResponseFormat::Markdown | OcrResponseFormat::Html
    ) && layout_model.is_none()
    {
        return Err(UseCaseError::invalid(
            "selected OCR model does not support structured response format",
            Some("response_format"),
            "unsupported_response_format",
        ));
    }
    if choice.is_ocr_vl() {
        if let Some(layout_model) = layout_model {
            validate_configured_layout_model(
                &policy.ocr_vl.layout_available_models,
                layout_model,
                "OCR-VL",
            )?;
        }
        return Ok(());
    }
    if let Some(layout_model) = layout_model {
        validate_configured_layout_model(&policy.ocr.layout_available_models, layout_model, "OCR")?;
    }
    if task != OcrTask::Ocr || max_tokens.is_some() {
        return Err(UseCaseError::invalid(
            "selected OCR model does not support OCR-VL parameters",
            None,
            "unsupported_ocr_parameter",
        ));
    }
    Ok(())
}

#[must_use]
pub fn resolve_max_tokens(
    choice: &OcrServiceChoice,
    requested: Option<usize>,
    policy: &OcrPolicy,
) -> Option<usize> {
    if choice.is_ocr_vl() {
        Some(
            requested
                .unwrap_or(policy.ocr_vl.max_tokens)
                .min(policy.ocr_vl.max_tokens),
        )
    } else {
        requested
    }
}

fn validate_configured_layout_model(
    available_models: &[ModelId],
    layout_model: &ModelId,
    service_name: &str,
) -> Result<(), UseCaseError> {
    if available_models.contains(layout_model) {
        return Ok(());
    }
    Err(UseCaseError::invalid(
        format!("`layout_model` is not configured for the {service_name} service"),
        Some("layout_model"),
        "model_not_available",
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn traditional_model() -> ModelId {
        ModelId::parse("PaddlePaddle/PP-OCRv6_tiny").unwrap()
    }

    fn ocr_vl_model() -> ModelId {
        ModelId::parse("PaddlePaddle/PaddleOCR-VL-1.6").unwrap()
    }

    fn policy() -> OcrPolicy {
        OcrPolicy {
            ocr: OcrServicePolicy {
                active: true,
                default_model: Some(ModelId::parse("PaddlePaddle/PP-OCRv6_tiny").unwrap()),
                available_models: vec![ModelId::parse("PaddlePaddle/PP-OCRv6_tiny").unwrap()],
                layout_default_model: None,
                layout_available_models: Vec::new(),
                format: OcrResponseFormat::Json,
                max_pixels: 1_000,
            },
            ocr_vl: OcrVlServicePolicy {
                active: true,
                default_model: Some(ModelId::parse("PaddlePaddle/PaddleOCR-VL-1.6").unwrap()),
                available_models: vec![ModelId::parse("PaddlePaddle/PaddleOCR-VL-1.6").unwrap()],
                layout_default_model: None,
                layout_available_models: Vec::new(),
                format: OcrResponseFormat::Markdown,
                max_tokens: 64,
                max_pixels: 2_000,
            },
        }
    }

    #[test]
    fn structured_default_prefers_ocr_vl() {
        let choice =
            resolve_service_choice(&policy(), None, Some(OcrResponseFormat::Markdown)).unwrap();
        assert_eq!(choice.model(), &ocr_vl_model());
    }

    #[test]
    fn traditional_ocr_rejects_vl_parameters() {
        let choice = OcrServiceChoice::Ocr {
            model: traditional_model(),
        };
        let error = validate_parameters(
            &choice,
            OcrResponseFormat::Json,
            OcrTask::Table,
            None,
            None,
            &policy(),
        )
        .unwrap_err();
        assert!(matches!(
            error,
            UseCaseError::InvalidRequest {
                code: "unsupported_ocr_parameter",
                ..
            }
        ));
    }

    #[test]
    fn structured_defaults_prefer_ocr_vl_for_markdown_and_html() {
        for format in [OcrResponseFormat::Markdown, OcrResponseFormat::Html] {
            let choice = resolve_service_choice(&policy(), None, Some(format)).unwrap();
            assert_eq!(choice.model(), &ocr_vl_model());
            assert!(choice.is_ocr_vl());
            assert_eq!(
                resolve_response_format(&policy(), &choice, Some(format)),
                format
            );
        }
    }

    #[test]
    fn active_service_controls_default_choice_and_format() {
        let mut only_vl = policy();
        only_vl.ocr.active = false;
        let choice = resolve_service_choice(&only_vl, None, None).unwrap();
        assert!(choice.is_ocr_vl());
        assert_eq!(
            resolve_response_format(&only_vl, &choice, None),
            OcrResponseFormat::Markdown
        );

        let mut both = policy();
        both.ocr.format = OcrResponseFormat::Text;
        let choice = resolve_service_choice(&both, None, None).unwrap();
        assert_eq!(choice.model(), &traditional_model());
        assert_eq!(
            resolve_response_format(&both, &choice, None),
            OcrResponseFormat::Text
        );
    }

    #[test]
    fn available_model_is_used_when_default_is_omitted() {
        let mut policy = policy();
        policy.ocr.default_model = None;
        policy.ocr_vl.default_model = None;

        assert_eq!(
            resolve_service_choice(&policy, None, None).unwrap().model(),
            &traditional_model()
        );
        assert_eq!(
            resolve_service_choice(&policy, None, Some(OcrResponseFormat::Markdown))
                .unwrap()
                .model(),
            &ocr_vl_model()
        );
    }

    #[test]
    fn layout_defaults_are_resolved_for_each_service() {
        let layout = ModelId::parse("PaddlePaddle/PP-DocLayoutV3").unwrap();
        let mut policy = policy();
        policy.ocr.layout_default_model = Some(layout.clone());
        policy.ocr_vl.layout_default_model = Some(layout.clone());

        assert_eq!(
            resolve_layout_model(
                &policy,
                &OcrServiceChoice::Ocr {
                    model: traditional_model(),
                },
                None,
            ),
            Some(layout.clone())
        );
        assert_eq!(
            resolve_layout_model(
                &policy,
                &OcrServiceChoice::OcrVl {
                    model: ocr_vl_model(),
                },
                None,
            ),
            Some(layout)
        );
    }

    #[test]
    fn traditional_layout_must_be_configured() {
        let layout = ModelId::parse("PaddlePaddle/PP-DocLayoutV3").unwrap();
        let choice = OcrServiceChoice::Ocr {
            model: traditional_model(),
        };
        let mut policy = policy();
        policy.ocr.layout_available_models = vec![layout.clone()];
        validate_parameters(
            &choice,
            OcrResponseFormat::Markdown,
            OcrTask::Ocr,
            Some(&layout),
            None,
            &policy,
        )
        .unwrap();

        policy.ocr.layout_available_models.clear();
        let error = validate_parameters(
            &choice,
            OcrResponseFormat::Json,
            OcrTask::Ocr,
            Some(&layout),
            None,
            &policy,
        )
        .unwrap_err();
        assert!(matches!(
            error,
            UseCaseError::InvalidRequest {
                param: Some("layout_model"),
                code: "model_not_available",
                ..
            }
        ));
    }

    #[test]
    fn ocr_vl_defaults_max_tokens_to_the_policy_limit() {
        let mut policy = policy();
        policy.ocr_vl.max_tokens = 64;
        assert_eq!(
            resolve_max_tokens(
                &OcrServiceChoice::OcrVl {
                    model: ocr_vl_model(),
                },
                None,
                &policy,
            ),
            Some(64)
        );
    }

    #[test]
    fn explicit_model_must_be_available_in_a_compatible_service() {
        let error = resolve_service_choice(
            &policy(),
            Some("Acme/Experimental-OCR"),
            Some(OcrResponseFormat::Json),
        )
        .unwrap_err();
        assert!(matches!(error, UseCaseError::ModelNotAvailable(_)));

        let mut only_vl = policy();
        only_vl.ocr.active = false;
        let error = resolve_service_choice(
            &only_vl,
            Some("PaddlePaddle/PP-OCRv6_tiny"),
            Some(OcrResponseFormat::Json),
        )
        .unwrap_err();
        assert!(matches!(error, UseCaseError::ModelNotAvailable(_)));
    }
}
