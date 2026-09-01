use encoding_rs::UTF_8;
use llama_cpp_2::model::{AddBos, LlamaChatMessage, LlamaChatTemplate, LlamaModel};
use llama_cpp_2::token::LlamaToken;

use crate::contract::{
    AdvancedRequest, AdvancedSemanticRequest, ContentPart, Error, Input, Message, ReasoningOptions,
    Request, RichMessage, Role, RuntimeConfig, SemanticInput, SemanticRequest, TemplateEngine,
    ToolChoice, ToolDefinition,
};

pub(crate) enum EffectiveTemplate {
    LlamaCpp {
        template: LlamaChatTemplate,
        source: String,
        enable_thinking: bool,
    },
    Jinja {
        source: String,
        enable_thinking: bool,
    },
}

pub(crate) fn effective_template(
    model: &LlamaModel,
    config: &RuntimeConfig,
) -> Result<EffectiveTemplate, String> {
    let source = match config.chat_template.as_deref() {
        Some(template) => template.to_string(),
        None => model
            .chat_template(None)
            .map_err(|error| error.to_string())?
            .to_string()
            .map_err(|error| error.to_string())?,
    };
    let template = match config.template_engine {
        TemplateEngine::LlamaCpp => EffectiveTemplate::LlamaCpp {
            template: LlamaChatTemplate::new(&source).map_err(|error| error.to_string())?,
            source,
            enable_thinking: config.enable_thinking,
        },
        TemplateEngine::Jinja => EffectiveTemplate::Jinja {
            source,
            enable_thinking: config.enable_thinking,
        },
    };
    let canary = [
        Message {
            role: "system".to_string(),
            content: "system".to_string(),
        },
        Message {
            role: "developer".to_string(),
            content: "developer".to_string(),
        },
        Message {
            role: "user".to_string(),
            content: "user".to_string(),
        },
        Message {
            role: "assistant".to_string(),
            content: "assistant".to_string(),
        },
    ];
    apply_effective_template(model, &template, &canary, true)
        .map_err(|error| format!("effective chat template cannot be applied: {error}"))?;
    Ok(template)
}

pub(crate) fn prepare_legacy_messages(
    model: &LlamaModel,
    template: &EffectiveTemplate,
    messages: &[Message],
) -> Result<Vec<LlamaToken>, Error> {
    let semantic = messages
        .iter()
        .map(|message| RichMessage {
            role: Role::from(message.role.clone()),
            content: vec![ContentPart::Text {
                text: message.content.clone(),
            }],
            tool_calls: Vec::new(),
        })
        .collect::<Vec<_>>();
    prepare_semantic_messages(
        model,
        template,
        &semantic,
        &[],
        &ToolChoice::None,
        ReasoningOptions::default(),
    )
}

pub(crate) fn legacy_request_from_semantic(request: SemanticRequest) -> Request {
    Request {
        input: Input::Semantic(Box::new(SemanticInput {
            messages: request.messages,
            tools: request.tools,
            tool_choice: request.tool_choice,
            parallel_tool_calls: request.parallel_tool_calls,
            reasoning: request.reasoning,
        })),
        options: request.options,
    }
}

pub(crate) fn advanced_request_from_semantic(
    request: AdvancedSemanticRequest,
) -> Result<AdvancedRequest, Error> {
    if request.logprobs.is_some()
        && (request.reasoning.enabled == Some(true)
            || (!request.tools.is_empty() && request.tool_choice != ToolChoice::None))
    {
        return unsupported(
            "logprobs",
            "token logprobs cannot be truthfully mapped through parsed reasoning or tool calls",
        );
    }
    Ok(AdvancedRequest {
        input: Input::Semantic(Box::new(SemanticInput {
            messages: request.messages,
            tools: request.tools,
            tool_choice: request.tool_choice,
            parallel_tool_calls: request.parallel_tool_calls,
            reasoning: request.reasoning,
        })),
        options: request.options,
        output: request.output,
        logprobs: request.logprobs,
        logit_bias: request.logit_bias,
        sampling: request.sampling,
        choices: request.choices,
        reasoning_control_id: request.reasoning_control_id,
    })
}

pub(crate) fn prepare_semantic_messages(
    model: &LlamaModel,
    template: &EffectiveTemplate,
    messages: &[RichMessage],
    tools: &[ToolDefinition],
    tool_choice: &ToolChoice,
    reasoning: ReasoningOptions,
) -> Result<Vec<LlamaToken>, Error> {
    let messages = validate_and_flatten(messages, tools, tool_choice, reasoning)?;
    let prompt =
        apply_effective_template(model, template, &messages, true).map_err(Error::Generation)?;
    tokenize_prompt(model, &prompt)
}

pub(crate) fn validate_and_flatten(
    messages: &[RichMessage],
    tools: &[ToolDefinition],
    tool_choice: &ToolChoice,
    reasoning: ReasoningOptions,
) -> Result<Vec<Message>, Error> {
    if !tools.is_empty() {
        return unsupported("tools", "tool definitions require common-chat preparation");
    }
    if *tool_choice != ToolChoice::None {
        return unsupported(
            "tool_choice",
            "tool selection requires common-chat preparation",
        );
    }
    if reasoning.enabled == Some(true) || reasoning.effort.is_some() {
        return unsupported(
            "reasoning",
            "per-request reasoning options require common-chat preparation",
        );
    }

    messages
        .iter()
        .enumerate()
        .map(|(index, message)| {
            if !message.tool_calls.is_empty() {
                return unsupported(
                    "messages.tool_calls",
                    format!("tool calls at message {index}"),
                );
            }
            let mut content = String::new();
            for part in &message.content {
                match part {
                    ContentPart::Text { text } => content.push_str(text),
                    ContentPart::Reasoning { .. } => {
                        return unsupported(
                            "messages.content.reasoning",
                            format!("reasoning content at message {index}"),
                        );
                    }
                    ContentPart::Image(_) => {
                        return unsupported(
                            "messages.content.image",
                            "images require common-chat preparation",
                        );
                    }
                    ContentPart::Media(media) => {
                        return unsupported(
                            "messages.content.media",
                            format!(
                                "{:?} placeholder `{}` at message {index}",
                                media.media_type, media.id
                            ),
                        );
                    }
                    ContentPart::ToolResult(result) => {
                        return unsupported(
                            "messages.content.tool_result",
                            format!("tool result `{}` at message {index}", result.tool_call_id),
                        );
                    }
                }
            }
            Ok(Message {
                role: message.role.as_str().to_string(),
                content,
            })
        })
        .collect()
}

fn unsupported<T>(field: &'static str, detail: impl Into<String>) -> Result<T, Error> {
    Err(Error::Unsupported {
        field,
        detail: detail.into(),
    })
}

pub(crate) fn apply_effective_template(
    model: &LlamaModel,
    template: &EffectiveTemplate,
    messages: &[Message],
    add_generation_prompt: bool,
) -> Result<String, String> {
    let messages = normalize_text_messages(messages);
    match template {
        EffectiveTemplate::LlamaCpp { template, .. } => {
            let messages = messages
                .iter()
                .map(|message| {
                    LlamaChatMessage::new(message.role.clone(), message.content.clone())
                        .map_err(|error| error.to_string())
                })
                .collect::<Result<Vec<_>, _>>()?;
            model
                .apply_chat_template(template, &messages, add_generation_prompt)
                .map_err(|error| error.to_string())
        }
        EffectiveTemplate::Jinja {
            source,
            enable_thinking,
        } => {
            let bos_token = special_token_text(model, model.token_bos())?;
            let eos_token = special_token_text(model, model.token_eos())?;
            render_jinja_template(
                source,
                &messages,
                add_generation_prompt,
                &bos_token,
                &eos_token,
                *enable_thinking,
            )
        }
    }
}

pub(crate) fn common_chat_template_parts<'a>(
    model: &LlamaModel,
    template: &'a EffectiveTemplate,
) -> Result<(&'a str, String, String, bool), Error> {
    let (source, enable_thinking) = match template {
        EffectiveTemplate::LlamaCpp {
            source,
            enable_thinking,
            ..
        }
        | EffectiveTemplate::Jinja {
            source,
            enable_thinking,
        } => (source.as_str(), *enable_thinking),
    };
    Ok((
        source,
        special_token_text(model, model.token_bos()).map_err(Error::Generation)?,
        special_token_text(model, model.token_eos()).map_err(Error::Generation)?,
        enable_thinking,
    ))
}

pub(crate) fn normalize_text_messages(messages: &[Message]) -> Vec<Message> {
    let mut instructions = Vec::new();
    let mut conversation = Vec::new();
    for message in messages {
        if matches!(message.role.as_str(), "system" | "developer") {
            instructions.push(message.content.as_str());
        } else {
            conversation.push(message.clone());
        }
    }
    if !instructions.is_empty() {
        conversation.insert(
            0,
            Message {
                role: "system".to_string(),
                content: instructions.join("\n"),
            },
        );
    }
    conversation
}

fn special_token_text(model: &LlamaModel, token: LlamaToken) -> Result<String, String> {
    let mut decoder = UTF_8.new_decoder();
    model
        .token_to_piece(token, &mut decoder, true, None)
        .map_err(|error| error.to_string())
}

pub(crate) fn render_jinja_template(
    source: &str,
    messages: &[Message],
    add_generation_prompt: bool,
    bos_token: &str,
    eos_token: &str,
    enable_thinking: bool,
) -> Result<String, String> {
    let mut environment = minijinja::Environment::new();
    environment.set_unknown_method_callback(|_state, value, method, args| {
        use minijinja::value::{Value, from_args};
        let text = value
            .as_str()
            .ok_or_else(|| minijinja::Error::from(minijinja::ErrorKind::UnknownMethod))?;
        match method {
            "startswith" => {
                let (needle,): (String,) = from_args(args)?;
                Ok(Value::from(text.starts_with(&needle)))
            }
            "endswith" => {
                let (needle,): (String,) = from_args(args)?;
                Ok(Value::from(text.ends_with(&needle)))
            }
            "split" => {
                let (separator,): (String,) = from_args(args)?;
                Ok(Value::from_serialize(
                    text.split(&separator).collect::<Vec<_>>(),
                ))
            }
            "rstrip" => {
                let (characters,): (String,) = from_args(args)?;
                Ok(Value::from(
                    text.trim_end_matches(|ch| characters.contains(ch)),
                ))
            }
            "lstrip" => {
                let (characters,): (String,) = from_args(args)?;
                Ok(Value::from(
                    text.trim_start_matches(|ch| characters.contains(ch)),
                ))
            }
            _ => Err(minijinja::Error::from(minijinja::ErrorKind::UnknownMethod)),
        }
    });
    environment.add_function(
        "raise_exception",
        |message: String| -> Result<String, minijinja::Error> {
            Err(minijinja::Error::new(
                minijinja::ErrorKind::InvalidOperation,
                message,
            ))
        },
    );
    environment.add_function("strftime_now", |format: String| -> String {
        chrono::Utc::now().format(&format).to_string()
    });
    environment
        .add_template("chat", source)
        .map_err(|error| error.to_string())?;
    environment
        .get_template("chat")
        .map_err(|error| error.to_string())?
        .render(minijinja::context! {
            messages => messages,
            add_generation_prompt => add_generation_prompt,
            bos_token => bos_token,
            eos_token => eos_token,
            tools => Vec::<String>::new(),
            enable_thinking => enable_thinking,
            add_vision_id => false,
        })
        .map_err(|error| error.to_string())
}

pub(crate) fn tokenize_prompt(model: &LlamaModel, prompt: &str) -> Result<Vec<LlamaToken>, Error> {
    model
        .str_to_token(prompt, AddBos::Always)
        .map_err(|error| Error::Generation(error.to_string()))
}
