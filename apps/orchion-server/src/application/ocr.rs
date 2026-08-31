use super::{
    RuntimeError, UseCaseError, finish_owned_file_operation, protect_owned_file_operation,
};
use orchion::{
    KnownOcrModel, ModelCapabilities, ModelId, OcrLimits, OcrOptions, OcrResponseFormat, OcrResult,
    OcrTask,
};
use std::future::Future;
use std::path::PathBuf;
use std::pin::Pin;

pub type OcrFuture<'a> =
    Pin<Box<dyn Future<Output = Result<Option<OcrResult>, RuntimeError>> + Send + 'a>>;

#[derive(Debug, Clone)]
pub struct OcrServicePolicy {
    pub active: bool,
    pub models: Vec<ModelId>,
    pub model_layouts: Vec<(ModelId, ModelId)>,
    pub format: OcrResponseFormat,
    pub max_pixels: u64,
}

#[derive(Debug, Clone)]
pub struct OcrVlServicePolicy {
    pub active: bool,
    pub models: Vec<ModelId>,
    pub model_layouts: Vec<(ModelId, ModelId)>,
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
    pub model: String,
    pub response_format: Option<OcrResponseFormat>,
    pub task: OcrTask,
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
    let choice = resolve_service_choice(&policy, &command.model)?;
    let response_format = resolve_response_format(&policy, &choice, command.response_format);
    let layout_model = configured_layout_model(&policy, &choice);
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
    if !protect_owned_file_operation() {
        return Err(UseCaseError::Internal("request cancelled".into()));
    }
    let validation = tokio::task::spawn_blocking(move || {
        orchion::validate_ocr_image_file(&validation_path, max_pixels)
    })
    .await;
    if finish_owned_file_operation() {
        return Err(UseCaseError::Internal("request cancelled".into()));
    }
    validation.map_err(|error| UseCaseError::Internal(error.to_string()))??;

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
    model: &str,
) -> Result<OcrServiceChoice, UseCaseError> {
    resolve_explicit_model(policy, model)
}

fn resolve_explicit_model(
    policy: &OcrPolicy,
    model: &str,
) -> Result<OcrServiceChoice, UseCaseError> {
    let model_id =
        ModelId::parse(model).map_err(|_| UseCaseError::ModelNotAvailable(model.to_string()))?;
    let ocr_match = policy.ocr.active && policy.ocr.models.contains(&model_id);
    let ocr_vl_match = policy.ocr_vl.active && policy.ocr_vl.models.contains(&model_id);
    match (ocr_match, ocr_vl_match) {
        (true, _) => Ok(OcrServiceChoice::ocr(model_id)),
        (false, true) => Ok(OcrServiceChoice::ocr_vl(model_id)),
        (false, false) => Err(UseCaseError::ModelNotAvailable(model.to_string())),
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
    let capabilities = effective_capabilities(choice, layout_model.is_some());
    let supports_vision_language = capabilities.contains(ModelCapabilities::OCR_VISION_LANGUAGE);
    if supports_vision_language && max_tokens.is_some_and(|value| value > policy.ocr_vl.max_tokens)
    {
        return Err(UseCaseError::invalid(
            format!(
                "`max_tokens` must not exceed the configured limit of {}",
                policy.ocr_vl.max_tokens
            ),
            Some("max_tokens"),
            "max_tokens_exceeded",
        ));
    }
    let required_format_capability = match response_format {
        OcrResponseFormat::Json | OcrResponseFormat::Text => ModelCapabilities::OCR_TEXT,
        OcrResponseFormat::Markdown => ModelCapabilities::OCR_MARKDOWN,
        OcrResponseFormat::Html => ModelCapabilities::OCR_HTML,
    };
    if !capabilities.contains(required_format_capability) {
        return Err(UseCaseError::invalid(
            "selected OCR deployment does not support the requested response format",
            Some("response_format"),
            "unsupported_response_format",
        ));
    }
    if supports_vision_language {
        return Ok(());
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
    if effective_capabilities(choice, configured_layout_model(policy, choice).is_some())
        .contains(ModelCapabilities::OCR_VISION_LANGUAGE)
    {
        Some(
            requested
                .unwrap_or(policy.ocr_vl.max_tokens)
                .min(policy.ocr_vl.max_tokens),
        )
    } else {
        requested
    }
}

fn effective_capabilities(choice: &OcrServiceChoice, has_layout: bool) -> ModelCapabilities {
    let available = if has_layout {
        ModelCapabilities::OCR_LAYOUT
    } else {
        ModelCapabilities::NONE
    };
    KnownOcrModel::from_model_id(choice.model()).map_or(ModelCapabilities::NONE, |model| {
        model
            .descriptor()
            .effective_capabilities(available)
            .union(available)
    })
}

fn configured_layout_model(policy: &OcrPolicy, choice: &OcrServiceChoice) -> Option<ModelId> {
    let model_layouts = match choice {
        OcrServiceChoice::Ocr { .. } => &policy.ocr.model_layouts,
        OcrServiceChoice::OcrVl { .. } => &policy.ocr_vl.model_layouts,
    };
    model_layouts
        .iter()
        .find_map(|(configured_model, configured_layout)| {
            (configured_model == choice.model()).then(|| configured_layout.clone())
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn traditional_model() -> ModelId {
        ModelId::parse("paddlepaddle/pp-ocrv6-tiny").unwrap()
    }

    fn ocr_vl_model() -> ModelId {
        ModelId::parse("paddlepaddle/paddleocr-vl-1.6").unwrap()
    }

    fn policy() -> OcrPolicy {
        OcrPolicy {
            ocr: OcrServicePolicy {
                active: true,
                models: vec![ModelId::parse("paddlepaddle/pp-ocrv6-tiny").unwrap()],
                model_layouts: Vec::new(),
                format: OcrResponseFormat::Json,
                max_pixels: 1_000,
            },
            ocr_vl: OcrVlServicePolicy {
                active: true,
                models: vec![ModelId::parse("paddlepaddle/paddleocr-vl-1.6").unwrap()],
                model_layouts: Vec::new(),
                format: OcrResponseFormat::Markdown,
                max_tokens: 64,
                max_pixels: 2_000,
            },
        }
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
    fn active_service_controls_default_format() {
        let mut only_vl = policy();
        only_vl.ocr.active = false;
        let choice = resolve_service_choice(&only_vl, ocr_vl_model().as_str()).unwrap();
        assert!(choice.is_ocr_vl());
        assert_eq!(
            resolve_response_format(&only_vl, &choice, None),
            OcrResponseFormat::Markdown
        );

        let mut both = policy();
        both.ocr.format = OcrResponseFormat::Text;
        let choice = resolve_service_choice(&both, traditional_model().as_str()).unwrap();
        assert_eq!(choice.model(), &traditional_model());
        assert_eq!(
            resolve_response_format(&both, &choice, None),
            OcrResponseFormat::Text
        );
    }

    #[test]
    fn traditional_structured_formats_follow_deployment_layout() {
        let layout = ModelId::parse("PaddlePaddle/PP-DocLayoutV3").unwrap();
        let choice = OcrServiceChoice::Ocr {
            model: traditional_model(),
        };
        let mut policy = policy();
        policy.ocr.model_layouts = vec![(traditional_model(), layout.clone())];
        assert_eq!(
            configured_layout_model(&policy, &choice),
            Some(layout.clone())
        );
        validate_parameters(
            &choice,
            OcrResponseFormat::Markdown,
            OcrTask::Ocr,
            Some(&layout),
            None,
            &policy,
        )
        .unwrap();

        let error = validate_parameters(
            &choice,
            OcrResponseFormat::Html,
            OcrTask::Ocr,
            Some(&layout),
            None,
            &policy,
        )
        .unwrap_err();
        assert!(matches!(
            error,
            UseCaseError::InvalidRequest {
                param: Some("response_format"),
                code: "unsupported_response_format",
                ..
            }
        ));

        policy.ocr.model_layouts.clear();
        let error = validate_parameters(
            &choice,
            OcrResponseFormat::Markdown,
            OcrTask::Ocr,
            None,
            None,
            &policy,
        )
        .unwrap_err();
        assert!(matches!(
            error,
            UseCaseError::InvalidRequest {
                code: "unsupported_response_format",
                ..
            }
        ));
    }

    #[test]
    fn layout_configuration_is_scoped_to_the_selected_deployment() {
        let layout = ModelId::parse("PaddlePaddle/PP-DocLayoutV3").unwrap();
        let other = ModelId::parse("paddlepaddle/pp-ocrv6-small").unwrap();
        let mut policy = policy();
        policy.ocr.models.push(other.clone());
        policy.ocr.model_layouts = vec![(traditional_model(), layout.clone())];
        let other_choice = OcrServiceChoice::Ocr { model: other };

        assert_eq!(configured_layout_model(&policy, &other_choice), None);
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
    fn structured_response_requires_deployment_layout_capability() {
        let choice = OcrServiceChoice::OcrVl {
            model: ocr_vl_model(),
        };
        let error = validate_parameters(
            &choice,
            OcrResponseFormat::Markdown,
            OcrTask::Ocr,
            None,
            None,
            &policy(),
        )
        .unwrap_err();

        assert!(matches!(
            error,
            UseCaseError::InvalidRequest {
                param: Some("response_format"),
                code: "unsupported_response_format",
                ..
            }
        ));
    }

    #[test]
    fn explicit_model_must_be_available_in_a_compatible_service() {
        let error = resolve_service_choice(&policy(), "Acme/Experimental-OCR").unwrap_err();
        assert!(matches!(error, UseCaseError::ModelNotAvailable(_)));

        let mut only_vl = policy();
        only_vl.ocr.active = false;
        let error = resolve_service_choice(&only_vl, "paddlepaddle/pp-ocrv6-tiny").unwrap_err();
        assert!(matches!(error, UseCaseError::ModelNotAvailable(_)));
    }
}
