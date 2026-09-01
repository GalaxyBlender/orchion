#![allow(
    unsafe_code,
    reason = "the private module owns the audited llama-common C ABI boundary"
)]

use std::fmt::Write as _;
use std::marker::PhantomData;
use std::ptr::NonNull;
use std::rc::Rc;
use std::sync::atomic::{AtomicU64, Ordering};

use serde::{Deserialize, Serialize};

use crate::constraints::{grammar_for_constraint, validate_strict_schema};
use crate::contract::{
    ContentPart, Error, OutputConstraint, ReasoningOptions, RichMessage, SemanticDelta, ToolChoice,
    ToolDefinition,
};

static TOOL_CALL_ID: AtomicU64 = AtomicU64::new(1);
pub(crate) const MEDIA_MARKER: &str = "<__orchion_media__>";
const MEDIA_SENTINEL_PREFIX: &str = "<__orchion_media_";
const MEDIA_SENTINEL_SUFFIX: &str = "__>";

#[repr(C)]
struct NativePrepared {
    _private: [u8; 0],
}

#[repr(C)]
struct NativeReasoningControl {
    _private: [u8; 0],
}

#[repr(C)]
#[derive(Default)]
struct NativeBuffer {
    data: *mut u8,
    len: usize,
}

unsafe extern "C" {
    fn orchion_common_chat_prepare(
        template_data: *const u8,
        template_len: usize,
        bos_data: *const u8,
        bos_len: usize,
        eos_data: *const u8,
        eos_len: usize,
        request_json: *const u8,
        request_len: usize,
        prepared: *mut *mut NativePrepared,
        result_json: *mut NativeBuffer,
        error: *mut NativeBuffer,
    ) -> i32;
    fn orchion_common_chat_parse(
        prepared: *const NativePrepared,
        generated: *const u8,
        generated_len: usize,
        is_partial: i32,
        result_json: *mut NativeBuffer,
        error: *mut NativeBuffer,
    ) -> i32;
    fn orchion_common_chat_prepared_free(prepared: *mut NativePrepared);
    fn orchion_reasoning_control_init(
        start_tokens: *const i32,
        start_len: usize,
        end_tokens: *const i32,
        end_tokens_len: usize,
        end_offsets: *const usize,
        end_offsets_len: usize,
        end_count: usize,
        forced_tokens: *const i32,
        forced_len: usize,
        prompt_tokens: *const i32,
        prompt_len: usize,
        control: *mut *mut NativeReasoningControl,
        error: *mut NativeBuffer,
    ) -> i32;
    fn orchion_reasoning_control_apply(
        control: *const NativeReasoningControl,
        token_ids: *const i32,
        logits: *mut f32,
        len: usize,
        error: *mut NativeBuffer,
    ) -> i32;
    fn orchion_reasoning_control_accept(
        control: *mut NativeReasoningControl,
        token: i32,
        error: *mut NativeBuffer,
    ) -> i32;
    fn orchion_reasoning_control_force(
        control: *mut NativeReasoningControl,
        error: *mut NativeBuffer,
    ) -> i32;
    fn orchion_reasoning_control_free(control: *mut NativeReasoningControl);
    fn orchion_common_chat_buffer_free(buffer: NativeBuffer);
}

pub(crate) struct PreparedChat {
    native: NonNull<NativePrepared>,
    pub(crate) metadata: PreparedMetadata,
    tool_names: Vec<String>,
    pub(crate) rendered_image_order: Vec<usize>,
    _not_send_sync: PhantomData<Rc<()>>,
}

pub(crate) struct SemanticParser {
    prepared: PreparedChat,
    generated: String,
    previous: ParsedMessage,
    tool_ids: Vec<String>,
    tool_emissions: Vec<ToolEmission>,
}

pub(crate) struct ReasoningControl {
    native: NonNull<NativeReasoningControl>,
    _not_send_sync: PhantomData<Rc<()>>,
}

#[derive(Debug, Default)]
struct ToolEmission {
    id: String,
    name: Option<String>,
    arguments: String,
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct PreparedMetadata {
    pub(crate) prompt: String,
    pub(crate) grammar: String,
    pub(crate) grammar_lazy: bool,
    pub(crate) grammar_triggers: Vec<GrammarTrigger>,
    pub(crate) preserved_tokens: Vec<String>,
    pub(crate) additional_stops: Vec<String>,
    pub(crate) supports_thinking: bool,
    pub(crate) generation_prompt: String,
    pub(crate) thinking_start_tag: String,
    pub(crate) thinking_end_tags: Vec<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct GrammarTrigger {
    pub(crate) r#type: i32,
    pub(crate) value: String,
    pub(crate) token: i32,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Deserialize)]
pub(crate) struct ParsedMessage {
    #[serde(default)]
    pub(crate) content: String,
    #[serde(default)]
    pub(crate) reasoning_content: String,
    #[serde(default)]
    pub(crate) tool_calls: Vec<ParsedToolCall>,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub(crate) struct ParsedToolCall {
    #[serde(default)]
    pub(crate) id: String,
    pub(crate) function: ParsedFunction,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub(crate) struct ParsedFunction {
    pub(crate) name: String,
    pub(crate) arguments: String,
}

#[derive(Serialize)]
struct BridgeRequest {
    messages: Vec<serde_json::Value>,
    tools: Vec<serde_json::Value>,
    tool_choice: &'static str,
    parallel_tool_calls: bool,
    reasoning_format: &'static str,
    enable_thinking: bool,
    reasoning_effort: Option<&'static str>,
    grammar: String,
    json_schema: String,
}

#[derive(Clone, Copy)]
pub(crate) struct Preparation<'a> {
    pub(crate) template: &'a str,
    pub(crate) bos: &'a str,
    pub(crate) eos: &'a str,
    pub(crate) messages: &'a [RichMessage],
    pub(crate) tools: &'a [ToolDefinition],
    pub(crate) tool_choice: &'a ToolChoice,
    pub(crate) parallel_tool_calls: bool,
    pub(crate) reasoning: ReasoningOptions,
    pub(crate) output: &'a OutputConstraint,
}

impl PreparedChat {
    pub(crate) fn prepare(input: Preparation<'_>) -> Result<Self, Error> {
        let (tools, choice) = selected_tools(input.tools, input.tool_choice)?;
        let (grammar, json_schema) = output_contract(input.output)?;
        let mut image_index = 0;
        let request = BridgeRequest {
            messages: input
                .messages
                .iter()
                .map(|message| message_json(message, &mut image_index))
                .collect::<Result<_, _>>()?,
            tools: tools.iter().map(|tool| tool_json(tool)).collect(),
            tool_choice: choice,
            parallel_tool_calls: input.parallel_tool_calls,
            reasoning_format: if input.reasoning.enabled.unwrap_or(false)
                || input.reasoning.effort.is_some()
            {
                "deepseek"
            } else {
                "none"
            },
            enable_thinking: input.reasoning.enabled.unwrap_or(false)
                || input.reasoning.effort.is_some(),
            reasoning_effort: input.reasoning.effort.map(|effort| match effort {
                crate::contract::ReasoningEffort::Low => "low",
                crate::contract::ReasoningEffort::Medium => "medium",
                crate::contract::ReasoningEffort::High => "high",
            }),
            grammar,
            json_schema,
        };
        let request = serde_json::to_vec(&request).map_err(|error| {
            Error::Generation(format!("serialize common-chat request: {error}"))
        })?;
        let mut native = std::ptr::null_mut();
        let mut result = NativeBuffer::default();
        let mut error = NativeBuffer::default();
        // SAFETY: Every byte slice remains alive for the call, outputs point to writable local
        // storage, and the native function initializes all outputs before returning.
        let status = unsafe {
            orchion_common_chat_prepare(
                input.template.as_ptr(),
                input.template.len(),
                input.bos.as_ptr(),
                input.bos.len(),
                input.eos.as_ptr(),
                input.eos.len(),
                request.as_ptr(),
                request.len(),
                &raw mut native,
                &raw mut result,
                &raw mut error,
            )
        };
        if status != 0 {
            return Err(native_error("prepare common-chat template", error));
        }
        let Some(native) = NonNull::new(native) else {
            free_buffer(result);
            return Err(Error::Generation(
                "prepare common-chat template returned a null handle".to_string(),
            ));
        };
        let mut metadata =
            read_json::<PreparedMetadata>(result, "common-chat preparation metadata").inspect_err(
                |_| {
                    // SAFETY: The successful prepare call transferred one owned handle to Rust.
                    unsafe { orchion_common_chat_prepared_free(native.as_ptr()) };
                },
            )?;
        let rendered_image_order = validate_rendered_media(&mut metadata.prompt, image_index)
            .inspect_err(|_| {
                // SAFETY: The successful prepare call transferred one owned handle to Rust.
                unsafe { orchion_common_chat_prepared_free(native.as_ptr()) };
            })?;
        Ok(Self {
            native,
            metadata,
            tool_names: tools.iter().map(|tool| tool.name.clone()).collect(),
            rendered_image_order,
            _not_send_sync: PhantomData,
        })
    }

    pub(crate) fn parse(&self, generated: &str, is_partial: bool) -> Result<ParsedMessage, Error> {
        let mut result = NativeBuffer::default();
        let mut error = NativeBuffer::default();
        // SAFETY: The handle is valid for self's lifetime, generated remains alive for the call,
        // and both output pointers refer to writable local storage.
        let status = unsafe {
            orchion_common_chat_parse(
                self.native.as_ptr(),
                generated.as_ptr(),
                generated.len(),
                i32::from(is_partial),
                &raw mut result,
                &raw mut error,
            )
        };
        if status != 0 {
            return Err(native_error("parse common-chat output", error));
        }
        read_json(result, "parsed common-chat message")
    }

    pub(crate) fn into_parser(self) -> SemanticParser {
        SemanticParser {
            prepared: self,
            generated: String::new(),
            previous: ParsedMessage::default(),
            tool_ids: Vec::new(),
            tool_emissions: Vec::new(),
        }
    }
}

impl ReasoningControl {
    pub(crate) fn new(
        start: &[i32],
        ends: &[Vec<i32>],
        forced: &[i32],
        prompt: &[i32],
    ) -> Result<Self, Error> {
        let end_offsets = std::iter::once(0)
            .chain(ends.iter().scan(0, |offset, value| {
                *offset += value.len();
                Some(*offset)
            }))
            .collect::<Vec<_>>();
        let end_tokens = ends.iter().flatten().copied().collect::<Vec<_>>();
        validate_reasoning_control_input(start, &end_tokens, &end_offsets, ends.len(), forced)?;
        let mut native = std::ptr::null_mut();
        let mut error = NativeBuffer::default();
        // SAFETY: All slices remain alive for the call and the native bridge initializes the
        // output handle before returning without retaining any slice pointer.
        let status = unsafe {
            orchion_reasoning_control_init(
                start.as_ptr(),
                start.len(),
                end_tokens.as_ptr(),
                end_tokens.len(),
                end_offsets.as_ptr(),
                end_offsets.len(),
                ends.len(),
                forced.as_ptr(),
                forced.len(),
                prompt.as_ptr(),
                prompt.len(),
                &raw mut native,
                &raw mut error,
            )
        };
        if status != 0 {
            return Err(native_error("initialize reasoning control", error));
        }
        let native = NonNull::new(native).ok_or_else(|| {
            Error::Generation("reasoning control returned a null handle".to_string())
        })?;
        Ok(Self {
            native,
            _not_send_sync: PhantomData,
        })
    }

    pub(crate) fn apply(
        &self,
        candidates: &mut llama_cpp_2::token::data_array::LlamaTokenDataArray,
    ) -> Result<(), Error> {
        let ids = candidates
            .data
            .iter()
            .map(|value| value.id().0)
            .collect::<Vec<_>>();
        let mut logits = candidates
            .data
            .iter()
            .map(llama_cpp_2::token::data::LlamaTokenData::logit)
            .collect::<Vec<_>>();
        let mut error = NativeBuffer::default();
        // SAFETY: The opaque handle is live and the bridge only reads ids and mutates the matching
        // logits array during this call.
        let status = unsafe {
            orchion_reasoning_control_apply(
                self.native.as_ptr(),
                ids.as_ptr(),
                logits.as_mut_ptr(),
                ids.len(),
                &raw mut error,
            )
        };
        if status != 0 {
            return Err(native_error("apply reasoning control", error));
        }
        for (candidate, logit) in candidates.data.iter_mut().zip(logits) {
            candidate.set_logit(logit);
        }
        Ok(())
    }

    pub(crate) fn accept(&mut self, token: i32) -> Result<(), Error> {
        let mut error = NativeBuffer::default();
        // SAFETY: The worker-local opaque handle is live and uniquely mutated on this thread.
        let status = unsafe {
            orchion_reasoning_control_accept(self.native.as_ptr(), token, &raw mut error)
        };
        if status == 0 {
            Ok(())
        } else {
            Err(native_error("accept reasoning token", error))
        }
    }

    pub(crate) fn force(&mut self) -> Result<bool, Error> {
        let mut error = NativeBuffer::default();
        // SAFETY: The worker-local opaque handle is live and uniquely mutated on this thread.
        match unsafe { orchion_reasoning_control_force(self.native.as_ptr(), &raw mut error) } {
            0 => Ok(true),
            1 => Ok(false),
            _ => Err(native_error("force reasoning end", error)),
        }
    }
}

fn validate_reasoning_control_input(
    start: &[i32],
    end_tokens: &[i32],
    end_offsets: &[usize],
    end_count: usize,
    forced: &[i32],
) -> Result<(), Error> {
    let expected_offsets = end_count.checked_add(1).ok_or_else(|| {
        Error::InvalidConfig("reasoning end count exceeds addressable offsets".to_string())
    })?;
    if start.is_empty() || forced.is_empty() || end_count == 0 {
        return Err(Error::InvalidConfig(
            "reasoning control token sequences are empty".to_string(),
        ));
    }
    if end_offsets.len() != expected_offsets {
        return Err(Error::InvalidConfig(format!(
            "reasoning end offsets length must equal end_count + 1 ({expected_offsets})"
        )));
    }
    if end_offsets.first() != Some(&0) {
        return Err(Error::InvalidConfig(
            "reasoning end offsets must start at zero".to_string(),
        ));
    }
    if end_offsets.windows(2).any(|pair| pair[1] <= pair[0]) {
        return Err(Error::InvalidConfig(
            "reasoning end offsets must be strictly increasing".to_string(),
        ));
    }
    if end_offsets
        .last()
        .is_none_or(|last| *last > end_tokens.len())
    {
        return Err(Error::InvalidConfig(
            "reasoning end offsets exceed the token buffer".to_string(),
        ));
    }
    Ok(())
}

impl Drop for ReasoningControl {
    fn drop(&mut self) {
        // SAFETY: ReasoningControl uniquely owns the opaque handle and frees it exactly once.
        unsafe { orchion_reasoning_control_free(self.native.as_ptr()) };
    }
}

impl SemanticParser {
    pub(crate) fn push(
        &mut self,
        text: &str,
        is_partial: bool,
    ) -> Result<Vec<SemanticDelta>, Error> {
        self.generated.push_str(text);
        let mut current = self.prepared.parse(&self.generated, is_partial)?;
        stabilize_tool_ids(&mut current, &mut self.tool_ids);
        let deltas = semantic_diffs(
            &self.previous,
            &current,
            is_partial,
            &self.prepared.tool_names,
            &mut self.tool_emissions,
        )?;
        self.previous = current;
        Ok(deltas)
    }

    pub(crate) fn finish(&mut self, text: &str) -> Result<Vec<SemanticDelta>, Error> {
        self.push(text, false)
    }

    pub(crate) fn has_tool_calls(&self) -> bool {
        !self.previous.tool_calls.is_empty()
    }
}

impl Drop for PreparedChat {
    fn drop(&mut self) {
        // SAFETY: PreparedChat uniquely owns the handle and Drop runs exactly once.
        unsafe { orchion_common_chat_prepared_free(self.native.as_ptr()) };
    }
}

fn selected_tools<'a>(
    tools: &'a [ToolDefinition],
    choice: &ToolChoice,
) -> Result<(Vec<&'a ToolDefinition>, &'static str), Error> {
    match choice {
        ToolChoice::None => Ok((Vec::new(), "none")),
        ToolChoice::Auto => Ok((tools.iter().collect(), "auto")),
        ToolChoice::Required => {
            if tools.is_empty() {
                return unsupported("tool_choice", "required needs at least one tool");
            }
            Ok((tools.iter().collect(), "required"))
        }
        ToolChoice::Named(name) => {
            let tool = tools
                .iter()
                .find(|tool| tool.name == *name)
                .ok_or_else(|| Error::Unsupported {
                    field: "tool_choice",
                    detail: format!("named tool `{name}` is not present in tools"),
                })?;
            Ok((vec![tool], "required"))
        }
    }
}

fn stabilize_tool_ids(message: &mut ParsedMessage, ids: &mut Vec<String>) {
    for (index, call) in message.tool_calls.iter_mut().enumerate() {
        if let Some(id) = ids.get(index) {
            call.id.clone_from(id);
            continue;
        }
        if call.id.is_empty() {
            call.id = format!(
                "call_orchion_{:016x}",
                TOOL_CALL_ID.fetch_add(1, Ordering::Relaxed)
            );
        }
        ids.push(call.id.clone());
    }
}

fn semantic_diffs(
    previous: &ParsedMessage,
    current: &ParsedMessage,
    is_partial: bool,
    tool_names: &[String],
    tool_emissions: &mut Vec<ToolEmission>,
) -> Result<Vec<SemanticDelta>, Error> {
    let mut deltas = Vec::new();
    if let Some(delta) = suffix(
        &previous.reasoning_content,
        &current.reasoning_content,
        "reasoning",
    )? {
        deltas.push(SemanticDelta::Reasoning(delta.to_string()));
    }
    if let Some(delta) = suffix(&previous.content, &current.content, "content")? {
        deltas.push(SemanticDelta::Text(delta.to_string()));
    }
    if current.tool_calls.len() < previous.tool_calls.len() {
        return Err(Error::Generation(
            "common-chat parser removed a previously emitted tool call".to_string(),
        ));
    }
    if current.tool_calls.len() < tool_emissions.len() {
        return Err(Error::Generation(
            "common-chat parser removed an emitted tool call".to_string(),
        ));
    }
    for (index, call) in current.tool_calls.iter().enumerate() {
        if index == tool_emissions.len() {
            tool_emissions.push(ToolEmission {
                id: call.id.clone(),
                ..ToolEmission::default()
            });
        }
        let emission = &mut tool_emissions[index];
        if emission.name.is_none() {
            let name_is_stable = tool_names.contains(&call.function.name);
            let arguments_started = !call.function.arguments.is_empty();
            if !is_partial || name_is_stable || arguments_started {
                if call.function.name.is_empty() {
                    return Err(Error::Generation(
                        "common-chat parser finalized a tool call without a name".to_string(),
                    ));
                }
                emission.id.clone_from(&call.id);
                emission.name = Some(call.function.name.clone());
                emission.arguments.clone_from(&call.function.arguments);
                deltas.push(SemanticDelta::ToolCall {
                    index,
                    id: Some(call.id.clone()),
                    name: Some(call.function.name.clone()),
                    arguments: call.function.arguments.clone(),
                });
            }
            continue;
        }
        if emission.id != call.id || emission.name.as_deref() != Some(call.function.name.as_str()) {
            return Err(Error::Generation(
                "common-chat parser revised emitted tool metadata".to_string(),
            ));
        }
        let arguments = suffix(
            &emission.arguments,
            &call.function.arguments,
            "tool arguments",
        )?
        .unwrap_or_default();
        if !arguments.is_empty() {
            emission.arguments.clone_from(&call.function.arguments);
            deltas.push(SemanticDelta::ToolCall {
                index,
                id: None,
                name: None,
                arguments: arguments.to_string(),
            });
        }
    }
    Ok(deltas)
}

fn suffix<'a>(previous: &str, current: &'a str, field: &str) -> Result<Option<&'a str>, Error> {
    if previous == current {
        return Ok(None);
    }
    current.strip_prefix(previous).map(Some).ok_or_else(|| {
        Error::Generation(format!(
            "common-chat parser revised already emitted {field}; streaming would be untruthful"
        ))
    })
}

fn output_contract(output: &OutputConstraint) -> Result<(String, String), Error> {
    match output {
        OutputConstraint::Text => Ok((String::new(), String::new())),
        OutputConstraint::JsonObject => Ok((
            String::new(),
            r#"{"type":"object","additionalProperties":true}"#.to_string(),
        )),
        OutputConstraint::JsonSchema(schema) => Ok((String::new(), {
            validate_strict_schema(schema)?;
            serde_json::to_string(schema)
                .map_err(|error| Error::InvalidConfig(format!("JSON schema: {error}")))?
        })),
        OutputConstraint::Grammar(_) => {
            let grammar = grammar_for_constraint(output)?.ok_or_else(|| {
                Error::InvalidConfig("explicit grammar unexpectedly compiled as empty".to_string())
            })?;
            Ok((grammar, String::new()))
        }
    }
}

fn message_json(
    message: &RichMessage,
    image_index: &mut usize,
) -> Result<serde_json::Value, Error> {
    let mut text = String::new();
    let mut reasoning = String::new();
    let mut tool_call_id = None;
    for part in &message.content {
        match part {
            ContentPart::Text { text: part } => {
                if part.contains(MEDIA_MARKER) || part.contains(MEDIA_SENTINEL_PREFIX) {
                    return unsupported(
                        "messages.content.text",
                        "text collides with the reserved media marker",
                    );
                }
                text.push_str(part);
            }
            ContentPart::Image(_) => {
                if message.role != crate::contract::Role::User {
                    return unsupported(
                        "messages.content.image",
                        "images are only accepted in user messages",
                    );
                }
                write!(
                    text,
                    "{MEDIA_SENTINEL_PREFIX}{:08}{MEDIA_SENTINEL_SUFFIX}",
                    *image_index
                )
                .expect("writing to a String cannot fail");
                *image_index = image_index.checked_add(1).ok_or_else(|| {
                    Error::InvalidConfig("semantic image index overflow".to_string())
                })?;
            }
            ContentPart::Reasoning { text: part } => reasoning.push_str(part),
            ContentPart::ToolResult(result) => {
                if tool_call_id.replace(result.tool_call_id.clone()).is_some() {
                    return unsupported(
                        "messages.content.tool_result",
                        "one tool message cannot contain multiple tool results",
                    );
                }
                text.push_str(&result.content);
            }
            ContentPart::Media(media) => {
                return unsupported(
                    "messages.content.media",
                    format!("{:?} placeholder `{}`", media.media_type, media.id),
                );
            }
        }
    }
    let mut value = serde_json::json!({
        "role": message.role.as_str(),
        "content": text,
    });
    if let Some(tool_call_id) = tool_call_id {
        value["tool_call_id"] = tool_call_id.into();
    }
    if !reasoning.is_empty() {
        value["reasoning_content"] = reasoning.into();
    }
    if !message.tool_calls.is_empty() {
        value["tool_calls"] = message
            .tool_calls
            .iter()
            .map(|call| {
                serde_json::json!({
                    "id": call.id,
                    "type": "function",
                    "function": {"name": call.name, "arguments": call.arguments},
                })
            })
            .collect::<Vec<_>>()
            .into();
    }
    Ok(value)
}

fn validate_rendered_media(prompt: &mut String, expected: usize) -> Result<Vec<usize>, Error> {
    if prompt.contains(MEDIA_MARKER) {
        return Err(Error::InvalidConfig(
            "rendered prompt collides with the reserved media marker".to_string(),
        ));
    }
    let mut rendered = String::with_capacity(prompt.len());
    let mut remainder = prompt.as_str();
    let mut order = Vec::with_capacity(expected);
    let mut seen = vec![false; expected];
    while let Some(offset) = remainder.find(MEDIA_SENTINEL_PREFIX) {
        rendered.push_str(&remainder[..offset]);
        let sentinel = &remainder[offset + MEDIA_SENTINEL_PREFIX.len()..];
        let Some(end) = sentinel.find(MEDIA_SENTINEL_SUFFIX) else {
            return Err(Error::InvalidConfig(
                "rendered prompt contains a malformed media sentinel".to_string(),
            ));
        };
        let index_text = &sentinel[..end];
        if index_text.len() != 8 || !index_text.bytes().all(|byte| byte.is_ascii_digit()) {
            return Err(Error::InvalidConfig(
                "rendered prompt contains a malformed media sentinel".to_string(),
            ));
        }
        let index = index_text.parse::<usize>().map_err(|error| {
            Error::InvalidConfig(format!("rendered media sentinel index is invalid: {error}"))
        })?;
        let Some(was_seen) = seen.get_mut(index) else {
            return Err(Error::InvalidConfig(format!(
                "rendered prompt contains extra media sentinel index {index}"
            )));
        };
        if std::mem::replace(was_seen, true) {
            return Err(Error::InvalidConfig(format!(
                "rendered prompt duplicates media sentinel index {index}"
            )));
        }
        order.push(index);
        rendered.push_str(MEDIA_MARKER);
        remainder = &sentinel[end + MEDIA_SENTINEL_SUFFIX.len()..];
    }
    rendered.push_str(remainder);
    if let Some(index) = seen.iter().position(|seen| !seen) {
        return Err(Error::InvalidConfig(format!(
            "rendered prompt is missing media sentinel index {index}"
        )));
    }
    *prompt = rendered;
    Ok(order)
}

fn tool_json(tool: &ToolDefinition) -> serde_json::Value {
    serde_json::json!({
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters,
        }
    })
}

fn read_json<T: serde::de::DeserializeOwned>(
    buffer: NativeBuffer,
    context: &str,
) -> Result<T, Error> {
    let bytes = copy_buffer(buffer)?;
    serde_json::from_slice(&bytes)
        .map_err(|error| Error::Generation(format!("decode {context}: {error}")))
}

fn native_error(context: &str, buffer: NativeBuffer) -> Error {
    let detail = copy_buffer(buffer)
        .and_then(|bytes| {
            String::from_utf8(bytes)
                .map_err(|error| Error::Generation(format!("native error was not UTF-8: {error}")))
        })
        .unwrap_or_else(|error| error.to_string());
    Error::Generation(format!("{context}: {detail}"))
}

fn copy_buffer(buffer: NativeBuffer) -> Result<Vec<u8>, Error> {
    let result = if buffer.len == 0 {
        Ok(Vec::new())
    } else if buffer.data.is_null() {
        Err(Error::Generation(
            "native bridge returned a null buffer with nonzero length".to_string(),
        ))
    } else {
        // SAFETY: The bridge allocated exactly len initialized bytes and transfers read access
        // until the matching same-side free immediately below.
        Ok(unsafe { std::slice::from_raw_parts(buffer.data, buffer.len) }.to_vec())
    };
    free_buffer(buffer);
    result
}

fn free_buffer(buffer: NativeBuffer) {
    // SAFETY: NativeBuffer values are initialized by the bridge and are freed at most once here.
    unsafe { orchion_common_chat_buffer_free(buffer) };
}

fn unsupported<T>(field: &'static str, detail: impl Into<String>) -> Result<T, Error> {
    Err(Error::Unsupported {
        field,
        detail: detail.into(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::contract::{Role, ToolCall, ToolResult};

    const FIXTURE: &str = r"{% for message in messages %}{{ message.role }}: {{ message.content }}\n{% endfor %}{% if add_generation_prompt %}assistant: {% endif %}";

    #[test]
    fn explicit_template_prepares_and_parses_without_a_model() {
        let prepared = PreparedChat::prepare(Preparation {
            template: FIXTURE,
            bos: "<s>",
            eos: "</s>",
            messages: &[RichMessage {
                role: Role::User,
                content: vec![ContentPart::Text { text: "hi".into() }],
                tool_calls: Vec::new(),
            }],
            tools: &[],
            tool_choice: &ToolChoice::None,
            parallel_tool_calls: false,
            reasoning: ReasoningOptions::default(),
            output: &OutputConstraint::Text,
        })
        .unwrap();
        assert!(prepared.metadata.prompt.contains("user: hi"));
        assert_eq!(prepared.parse("hello", false).unwrap().content, "hello");
    }

    #[test]
    fn rich_history_and_named_tool_are_owned_across_parse_calls() {
        let tool = ToolDefinition {
            name: "weather".into(),
            description: Some("Get weather".into()),
            parameters: serde_json::json!({"type":"object","properties":{}}),
        };
        let messages = vec![
            RichMessage {
                role: Role::Assistant,
                content: Vec::new(),
                tool_calls: vec![ToolCall {
                    id: "call_1".into(),
                    name: "weather".into(),
                    arguments: serde_json::json!({"city":"Paris"}),
                }],
            },
            RichMessage {
                role: Role::Tool,
                content: vec![ContentPart::ToolResult(ToolResult {
                    tool_call_id: "call_1".into(),
                    content: "sunny".into(),
                    is_error: false,
                })],
                tool_calls: Vec::new(),
            },
        ];
        let prepared = PreparedChat::prepare(Preparation {
            template: FIXTURE,
            bos: "",
            eos: "",
            messages: &messages,
            tools: &[tool],
            tool_choice: &ToolChoice::Named("weather".into()),
            parallel_tool_calls: false,
            reasoning: ReasoningOptions::default(),
            output: &OutputConstraint::Text,
        })
        .unwrap();
        assert_eq!(prepared.parse("one", true).unwrap().content, "one");
        assert_eq!(prepared.parse("one two", false).unwrap().content, "one two");
    }

    #[test]
    fn invalid_template_and_named_tool_return_owned_errors() {
        let error = PreparedChat::prepare(Preparation {
            template: "{% broken",
            bos: "",
            eos: "",
            messages: &[],
            tools: &[],
            tool_choice: &ToolChoice::None,
            parallel_tool_calls: false,
            reasoning: ReasoningOptions::default(),
            output: &OutputConstraint::Text,
        })
        .err()
        .unwrap();
        assert!(error.to_string().contains("prepare common-chat template"));

        let error = selected_tools(&[], &ToolChoice::Named("missing".into())).unwrap_err();
        assert!(matches!(error, Error::Unsupported { .. }));
    }

    #[test]
    fn semantic_diffs_separate_reasoning_text_and_stable_tool_calls() {
        let previous = ParsedMessage {
            reasoning_content: "think".into(),
            content: "answer".into(),
            tool_calls: vec![ParsedToolCall {
                id: "call_fixed".into(),
                function: ParsedFunction {
                    name: "weather".into(),
                    arguments: r#"{"city":"Pa"#.into(),
                },
            }],
        };
        let current = ParsedMessage {
            reasoning_content: "thinking".into(),
            content: "answered".into(),
            tool_calls: vec![ParsedToolCall {
                id: "call_fixed".into(),
                function: ParsedFunction {
                    name: "weather".into(),
                    arguments: r#"{"city":"Paris"}"#.into(),
                },
            }],
        };
        let mut emissions = vec![ToolEmission {
            id: "call_fixed".into(),
            name: Some("weather".into()),
            arguments: r#"{"city":"Pa"#.into(),
        }];
        assert_eq!(
            semantic_diffs(&previous, &current, true, &[], &mut emissions).unwrap(),
            [
                SemanticDelta::Reasoning("ing".into()),
                SemanticDelta::Text("ed".into()),
                SemanticDelta::ToolCall {
                    index: 0,
                    id: None,
                    name: None,
                    arguments: "ris\"}".into(),
                },
            ]
        );

        let mut parsed = ParsedMessage {
            tool_calls: vec![ParsedToolCall {
                id: String::new(),
                function: ParsedFunction {
                    name: "weather".into(),
                    arguments: String::new(),
                },
            }],
            ..ParsedMessage::default()
        };
        let mut ids = Vec::new();
        stabilize_tool_ids(&mut parsed, &mut ids);
        let generated = parsed.tool_calls[0].id.clone();
        parsed.tool_calls[0].id = "model_changed_id".into();
        stabilize_tool_ids(&mut parsed, &mut ids);
        assert!(generated.starts_with("call_orchion_"));
        assert_eq!(parsed.tool_calls[0].id, generated);
    }

    #[test]
    fn semantic_parser_state_emits_a_growing_tool_name_once_at_its_final_value() {
        let mut previous = ParsedMessage::default();
        let mut emissions = Vec::new();
        let snapshots = [
            ("w", "", true),
            ("we", "", true),
            ("wea", "", true),
            ("weather", "{", true),
            ("weather", r#"{"city"#, true),
            ("weather", r#"{"city":"Paris"}"#, false),
        ];
        let mut deltas = Vec::new();
        for (name, arguments, partial) in snapshots {
            let current = ParsedMessage {
                tool_calls: vec![ParsedToolCall {
                    id: "call_fixed".into(),
                    function: ParsedFunction {
                        name: name.into(),
                        arguments: arguments.into(),
                    },
                }],
                ..ParsedMessage::default()
            };
            deltas.extend(
                semantic_diffs(
                    &previous,
                    &current,
                    partial,
                    &["weather".to_string()],
                    &mut emissions,
                )
                .unwrap(),
            );
            previous = current;
        }

        let names = deltas
            .iter()
            .filter_map(|delta| match delta {
                SemanticDelta::ToolCall { name, .. } => name.as_deref(),
                SemanticDelta::Text(_) | SemanticDelta::Reasoning(_) => None,
            })
            .collect::<Vec<_>>();
        let arguments = deltas
            .iter()
            .filter_map(|delta| match delta {
                SemanticDelta::ToolCall { arguments, .. } => Some(arguments.as_str()),
                SemanticDelta::Text(_) | SemanticDelta::Reasoning(_) => None,
            })
            .collect::<String>();
        assert_eq!(names, ["weather"]);
        assert_eq!(arguments, r#"{"city":"Paris"}"#);
    }

    #[test]
    fn explicit_template_and_parser_preserve_utf8() {
        let text = "caf\u{00e9}";
        let prepared = PreparedChat::prepare(Preparation {
            template: FIXTURE,
            bos: "",
            eos: "",
            messages: &[RichMessage {
                role: Role::User,
                content: vec![ContentPart::Text { text: text.into() }],
                tool_calls: Vec::new(),
            }],
            tools: &[],
            tool_choice: &ToolChoice::None,
            parallel_tool_calls: false,
            reasoning: ReasoningOptions::default(),
            output: &OutputConstraint::Text,
        })
        .unwrap();
        assert!(prepared.metadata.prompt.contains(text));
        assert_eq!(prepared.parse(text, false).unwrap().content, text);
    }

    fn image_message(text: &str) -> RichMessage {
        RichMessage {
            role: Role::User,
            content: vec![
                ContentPart::Text { text: text.into() },
                ContentPart::Image(crate::contract::ImageInput {
                    bytes: vec![1],
                    format: crate::contract::ImageFormat::Png,
                    width: 1,
                    height: 1,
                }),
            ],
            tool_calls: Vec::new(),
        }
    }

    #[test]
    fn indexed_media_sentinels_follow_custom_template_render_order() {
        let messages = [image_message("first"), image_message("second")];
        let prepared = PreparedChat::prepare(Preparation {
            template: r"{% for message in messages | reverse %}{{ message.content }}{% endfor %}",
            bos: "",
            eos: "",
            messages: &messages,
            tools: &[],
            tool_choice: &ToolChoice::None,
            parallel_tool_calls: false,
            reasoning: ReasoningOptions::default(),
            output: &OutputConstraint::Text,
        })
        .unwrap();
        assert_eq!(prepared.rendered_image_order, [1, 0]);
        assert_eq!(prepared.metadata.prompt.matches(MEDIA_MARKER).count(), 2);
        assert!(
            prepared.metadata.prompt.find("second").unwrap()
                < prepared.metadata.prompt.find("first").unwrap()
        );
    }

    #[test]
    fn malformed_missing_duplicate_and_extra_media_sentinels_are_rejected() {
        let messages = [image_message("image")];
        for template in [
            "constant output",
            "{{ messages[0].content }}{{ messages[0].content }}",
            "<__orchion_media_bad__>{{ messages[0].content }}",
            "<__orchion_media_00000001__>{{ messages[0].content }}",
        ] {
            let error = PreparedChat::prepare(Preparation {
                template,
                bos: "",
                eos: "",
                messages: &messages,
                tools: &[],
                tool_choice: &ToolChoice::None,
                parallel_tool_calls: false,
                reasoning: ReasoningOptions::default(),
                output: &OutputConstraint::Text,
            })
            .err()
            .unwrap();
            assert!(
                error.to_string().contains("media sentinel"),
                "{template}: {error}"
            );
        }
    }

    #[test]
    fn reasoning_control_only_forces_once_while_actively_reasoning() {
        use llama_cpp_2::token::LlamaToken;
        use llama_cpp_2::token::data::LlamaTokenData;
        use llama_cpp_2::token::data_array::LlamaTokenDataArray;

        let mut idle = ReasoningControl::new(&[10], &[vec![12]], &[12], &[]).unwrap();
        assert!(!idle.force().unwrap());

        let mut control = ReasoningControl::new(&[10], &[vec![12]], &[12], &[10]).unwrap();
        assert!(control.force().unwrap());
        assert!(!control.force().unwrap());

        let mut candidates = LlamaTokenDataArray::new(
            vec![
                LlamaTokenData::new(LlamaToken(11), 3.0, 0.0),
                LlamaTokenData::new(LlamaToken(12), 1.0, 0.0),
            ],
            false,
        );
        control.apply(&mut candidates).unwrap();
        assert!(candidates.data[0].logit().is_infinite());
        assert!((candidates.data[1].logit() - 1.0).abs() < f32::EPSILON);

        control.accept(12).unwrap();
        let mut ordinary = LlamaTokenDataArray::new(
            vec![
                LlamaTokenData::new(LlamaToken(11), 3.0, 0.0),
                LlamaTokenData::new(LlamaToken(12), 1.0, 0.0),
            ],
            false,
        );
        control.apply(&mut ordinary).unwrap();
        assert!((ordinary.data[0].logit() - 3.0).abs() < f32::EPSILON);
        assert!((ordinary.data[1].logit() - 1.0).abs() < f32::EPSILON);
        assert!(!control.force().unwrap());
    }

    #[test]
    fn rust_reasoning_control_validation_rejects_malformed_flattened_ends() {
        let start = [10];
        let forced = [12];
        let tokens = [12, 13];
        for result in [
            validate_reasoning_control_input(&start, &tokens, &[0], 1, &forced),
            validate_reasoning_control_input(&start, &tokens, &[1, 2], 1, &forced),
            validate_reasoning_control_input(&start, &tokens, &[0, 2, 1], 2, &forced),
            validate_reasoning_control_input(&start, &tokens[..1], &[0, 2], 1, &forced),
        ] {
            assert!(matches!(result, Err(Error::InvalidConfig(_))));
        }
    }

    #[test]
    fn native_reasoning_control_validates_lengths_offsets_and_null_empty_data() {
        fn initialize(
            end_tokens: &[i32],
            end_offsets: &[usize],
            end_offsets_len: usize,
            end_count: usize,
        ) -> (i32, *mut NativeReasoningControl, NativeBuffer) {
            let start = [10];
            let forced = [12];
            let mut control = std::ptr::null_mut();
            let mut error = NativeBuffer::default();
            // SAFETY: Every non-null input points to the stated live slice. A null prompt with
            // length zero is part of the ABI contract under test, and outputs are writable.
            let status = unsafe {
                orchion_reasoning_control_init(
                    start.as_ptr(),
                    start.len(),
                    end_tokens.as_ptr(),
                    end_tokens.len(),
                    end_offsets.as_ptr(),
                    end_offsets_len,
                    end_count,
                    forced.as_ptr(),
                    forced.len(),
                    std::ptr::null(),
                    0,
                    &raw mut control,
                    &raw mut error,
                )
            };
            (status, control, error)
        }

        let malformed = [
            initialize(&[12], &[0, 1], 1, 1),
            initialize(&[12], &[1, 1], 2, 1),
            initialize(&[12, 13], &[0, 2, 1], 3, 2),
            initialize(&[12], &[0, 2], 2, 1),
        ];
        for (status, control, error) in malformed {
            assert_ne!(status, 0);
            assert!(control.is_null());
            free_buffer(error);
        }

        let (status, control, error) = initialize(&[12], &[0, 1], 2, 1);
        assert_eq!(status, 0);
        assert!(!control.is_null());
        free_buffer(error);
        // SAFETY: A successful initialization returns one owned opaque control handle.
        unsafe { orchion_reasoning_control_free(control) };
    }
}
#[test]
fn media_marker_collision_and_non_user_image_are_rejected() {
    let collision = RichMessage {
        role: crate::contract::Role::User,
        content: vec![ContentPart::Text {
            text: format!("before {MEDIA_MARKER} after"),
        }],
        tool_calls: Vec::new(),
    };
    let mut image_index = 0;
    assert!(message_json(&collision, &mut image_index).is_err());

    let image = RichMessage {
        role: crate::contract::Role::Assistant,
        content: vec![ContentPart::Image(crate::contract::ImageInput {
            bytes: vec![1],
            format: crate::contract::ImageFormat::Png,
            width: 1,
            height: 1,
        })],
        tool_calls: Vec::new(),
    };
    assert!(message_json(&image, &mut image_index).is_err());
}
