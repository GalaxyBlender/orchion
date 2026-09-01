use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use llama_cpp_2::model::LlamaModel;
use llama_cpp_2::sampling::LlamaSampler;
use llama_cpp_2::token::LlamaToken;
use llama_cpp_2::token::data_array::LlamaTokenDataArray;
use llama_cpp_2::token::logit_bias::LlamaLogitBias;
use tokio::sync::{mpsc, oneshot};

use crate::common_chat::{PreparedMetadata, ReasoningControl, SemanticParser};
use crate::constraints::grammar_for_constraint;
use crate::contract::{
    AdvancedRequest, Error, Event, FinishReason, SemanticDelta, Timings, TokenAlternative,
    TokenLogprobs, Usage,
};
use crate::prefix_cache::Compatibility;
use crate::scheduler::ReservationDecision;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct OperationCapability(pub(crate) u64);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct SlotId(u32);

impl SlotId {
    pub(crate) fn new(value: usize) -> Result<Self, Error> {
        u32::try_from(value)
            .map(Self)
            .map_err(|error| Error::InvalidConfig(format!("slot id conversion failed: {error}")))
    }

    pub(crate) fn sequence(self) -> Result<i32, Error> {
        i32::try_from(self.0)
            .map_err(|error| Error::Generation(format!("sequence id conversion failed: {error}")))
    }
}

pub(crate) struct Slot {
    pub(crate) id: SlotId,
    pub(crate) state: SlotState,
}

#[allow(
    clippy::large_enum_variant,
    reason = "scheduler slots keep their active state inline for direct cooperative mutation"
)]
pub(crate) enum SlotState {
    Vacant,
    Reserved(ReservedSlot),
    Active(ActiveSlot),
    Draining(DrainingSlot),
}

pub(crate) struct Lifecycle {
    pub(crate) operation: Option<OperationCapability>,
    pub(crate) cancelled: Arc<AtomicBool>,
    pub(crate) events: Option<mpsc::Sender<Event>>,
    pub(crate) readiness: Option<oneshot::Sender<Result<(), Error>>>,
    pub(crate) acknowledged: Option<oneshot::Sender<Result<(), Error>>>,
    pub(crate) terminal: Option<oneshot::Sender<Result<Event, Error>>>,
}

pub(crate) struct ReservedSlot {
    pub(crate) lifecycle: Lifecycle,
    pub(crate) decision: Option<oneshot::Receiver<ReservationDecision>>,
    pub(crate) committed: Option<AdvancedRequest>,
}

pub(crate) struct ActiveSlot {
    pub(crate) lifecycle: Lifecycle,
    pub(crate) request: AdvancedRequest,
    pub(crate) tokens: Vec<LlamaToken>,
    pub(crate) prefill_cursor: usize,
    pub(crate) cache_n: usize,
    pub(crate) cache_capture_at: Option<usize>,
    pub(crate) cache_compatibility: Arc<Compatibility>,
    pub(crate) pending_decode: Option<LlamaToken>,
    pub(crate) sampler: SamplingState,
    pub(crate) decoder: encoding_rs::Decoder,
    pub(crate) semantic_parser: Option<SemanticParser>,
    pub(crate) preserved_tokens: Vec<LlamaToken>,
    pub(crate) stop_filter: StopFilter,
    pub(crate) pending_semantic: VecDeque<SemanticDelta>,
    pub(crate) pending_tokens: VecDeque<(String, TokenLogprobs)>,
    pub(crate) pending_content: Option<String>,
    pub(crate) completion_tokens: usize,
    pub(crate) reasoning_tokens: usize,
    pub(crate) reasoning_active: bool,
    pub(crate) prompt_tokens: usize,
    pub(crate) next_position: usize,
    pub(crate) prompt_wall: Duration,
    pub(crate) predicted_wall: Duration,
}

pub(crate) struct DrainingSlot {
    pub(crate) lifecycle: Lifecycle,
    pub(crate) pending_tokens: VecDeque<(String, TokenLogprobs)>,
    pub(crate) pending_content: Option<String>,
    pub(crate) pending_semantic: VecDeque<SemanticDelta>,
    pub(crate) outcome: Result<Event, Error>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BatchEntryKind {
    Decode,
    Prefill,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct BatchEntry {
    pub(crate) slot: usize,
    pub(crate) token: LlamaToken,
    pub(crate) position: i32,
    pub(crate) logits: bool,
    pub(crate) kind: BatchEntryKind,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct BatchPlan {
    pub(crate) entries: Vec<BatchEntry>,
    pub(crate) next_prefill_slot: usize,
}

impl Slot {
    pub(crate) fn vacant(id: usize) -> Result<Self, Error> {
        Ok(Self {
            id: SlotId::new(id)?,
            state: SlotState::Vacant,
        })
    }

    pub(crate) fn is_vacant(&self) -> bool {
        matches!(self.state, SlotState::Vacant)
    }

    pub(crate) fn is_occupied(&self) -> bool {
        !self.is_vacant()
    }
}

impl ActiveSlot {
    pub(crate) fn decoding_entry(&self, slot: usize) -> Option<BatchEntry> {
        if self.pending_content.is_some()
            || !self.pending_tokens.is_empty()
            || !self.pending_semantic.is_empty()
        {
            return None;
        }
        let token = self.pending_decode?;
        Some(BatchEntry {
            slot,
            token,
            position: i32::try_from(self.next_position).ok()?,
            logits: true,
            kind: BatchEntryKind::Decode,
        })
    }

    pub(crate) fn remaining_prefill(&self) -> usize {
        if self.pending_content.is_some()
            || !self.pending_tokens.is_empty()
            || !self.pending_semantic.is_empty()
            || self.pending_decode.is_some()
        {
            0
        } else {
            self.cache_capture_at
                .unwrap_or(self.tokens.len())
                .saturating_sub(self.prefill_cursor)
        }
    }

    pub(crate) fn prefill_entry(&self, slot: usize, offset: usize) -> Option<BatchEntry> {
        let index = self.prefill_cursor.checked_add(offset)?;
        let token = *self.tokens.get(index)?;
        Some(BatchEntry {
            slot,
            token,
            position: i32::try_from(index).ok()?,
            logits: index + 1 == self.tokens.len() && self.request.options.max_tokens > 0,
            kind: BatchEntryKind::Prefill,
        })
    }

    pub(crate) fn usage(&self) -> Usage {
        Usage {
            prompt_tokens: self.prompt_tokens,
            completion_tokens: self.completion_tokens,
            reasoning_tokens: self.reasoning_tokens,
            timings: timings_from_wall(
                self.cache_n,
                self.prompt_tokens.saturating_sub(self.cache_n),
                self.prompt_wall,
                self.completion_tokens,
                self.predicted_wall,
            ),
        }
    }

    pub(crate) fn into_draining(
        mut self,
        outcome: Result<Event, Error>,
        flush_filter: bool,
    ) -> DrainingSlot {
        if flush_filter {
            let tokens = self.stop_filter.take_token_flush();
            if tokens.is_empty() {
                if let Some(suffix) = self.stop_filter.take_flush() {
                    self.pending_content
                        .get_or_insert_with(String::new)
                        .push_str(&suffix);
                }
            } else {
                self.pending_tokens.extend(tokens);
            }
        }
        DrainingSlot {
            lifecycle: self.lifecycle,
            pending_tokens: self.pending_tokens,
            pending_content: self.pending_content,
            pending_semantic: self.pending_semantic,
            outcome,
        }
    }
}

pub(crate) fn plan_batch(slots: &[Slot], capacity: usize, prefill_cursor: usize) -> BatchPlan {
    plan_sources(slots, capacity, prefill_cursor)
}

trait BatchSource {
    fn decoding_entry(&self, slot: usize) -> Option<BatchEntry>;
    fn remaining_prefill(&self) -> usize;
    fn prefill_entry(&self, slot: usize, offset: usize) -> Option<BatchEntry>;
}

impl BatchSource for Slot {
    fn decoding_entry(&self, slot: usize) -> Option<BatchEntry> {
        match &self.state {
            SlotState::Active(active) => active.decoding_entry(slot),
            _ => None,
        }
    }

    fn remaining_prefill(&self) -> usize {
        match &self.state {
            SlotState::Active(active) => active.remaining_prefill(),
            _ => 0,
        }
    }

    fn prefill_entry(&self, slot: usize, offset: usize) -> Option<BatchEntry> {
        match &self.state {
            SlotState::Active(active) => active.prefill_entry(slot, offset),
            _ => None,
        }
    }
}

fn plan_sources<S: BatchSource>(slots: &[S], capacity: usize, prefill_cursor: usize) -> BatchPlan {
    let mut entries = Vec::with_capacity(capacity);
    for (slot_index, slot) in slots.iter().enumerate() {
        if let Some(entry) = slot.decoding_entry(slot_index) {
            entries.push(entry);
            if entries.len() == capacity {
                return BatchPlan {
                    entries,
                    next_prefill_slot: prefill_cursor,
                };
            }
        }
    }

    if slots.is_empty() || entries.len() == capacity {
        return BatchPlan {
            entries,
            next_prefill_slot: prefill_cursor,
        };
    }

    let start = prefill_cursor % slots.len();
    let eligible = (0..slots.len())
        .map(|offset| (start + offset) % slots.len())
        .filter(|&slot| slots[slot].remaining_prefill() > 0)
        .collect::<Vec<_>>();
    let remaining_capacity = capacity - entries.len();
    let chunk_size = remaining_capacity.div_ceil(eligible.len().max(1));
    let mut consumed = vec![0; slots.len()];
    while entries.len() < capacity {
        let mut progressed = false;
        for slot_index in eligible.iter().copied() {
            let available = slots[slot_index]
                .remaining_prefill()
                .saturating_sub(consumed[slot_index]);
            let take = available.min(chunk_size).min(capacity - entries.len());
            entries.extend((0..take).filter_map(|offset| {
                slots[slot_index].prefill_entry(slot_index, consumed[slot_index] + offset)
            }));
            consumed[slot_index] += take;
            progressed |= take > 0;
            if entries.len() == capacity {
                break;
            }
        }
        if !progressed {
            break;
        }
    }
    let next_prefill_slot = eligible
        .first()
        .map_or(start, |slot| (slot + 1) % slots.len());
    BatchPlan {
        entries,
        next_prefill_slot,
    }
}

pub(crate) fn logits_targets(plan: &BatchPlan) -> Result<Vec<(i32, usize)>, Error> {
    plan.entries
        .iter()
        .enumerate()
        .filter(|(_, entry)| entry.logits)
        .map(|(raw_index, entry)| {
            i32::try_from(raw_index)
                .map(|raw| (raw, entry.slot))
                .map_err(|error| {
                    Error::Generation(format!("logits index conversion failed: {error}"))
                })
        })
        .collect()
}

pub(crate) struct SamplingState {
    bias: Option<LlamaSampler>,
    penalties: LlamaSampler,
    grammar: Option<LlamaSampler>,
    filters: Option<LlamaSampler>,
    selector: LlamaSampler,
    reasoning_control: Option<ReasoningControl>,
}

pub(crate) fn create_sampler(
    model: &LlamaModel,
    request: &AdvancedRequest,
    context_size: usize,
    tokens: &[LlamaToken],
    common_chat: Option<&PreparedMetadata>,
    reasoning_control: Option<ReasoningControl>,
) -> Result<SamplingState, Error> {
    validate_sampling_request(model, request)?;
    let biases = request
        .logit_bias
        .iter()
        .map(|bias| LlamaLogitBias::new(LlamaToken(bias.token_id), bias.bias))
        .collect::<Vec<_>>();
    let bias = (!biases.is_empty()).then(|| LlamaSampler::logit_bias(model.n_vocab(), &biases));
    let mut penalties = LlamaSampler::penalties(
        model.n_vocab(),
        i32::try_from(context_size).map_err(|error| Error::Generation(error.to_string()))?,
        request.options.repeat_penalty,
        request.options.frequency_penalty,
        request.options.presence_penalty,
    );
    penalties.accept_many(tokens);
    let grammar = match common_chat {
        Some(metadata) => grammar_sampler(model, metadata)?,
        None => grammar_for_constraint(&request.output)?
            .map(|grammar| {
                LlamaSampler::grammar(model, &grammar, "root")
                    .map_err(|error| Error::InvalidConfig(format!("output grammar: {error}")))
            })
            .transpose()?,
    };
    let (filters, selector) = if request.options.temperature > 0.0 {
        let mut filters = vec![
            LlamaSampler::top_k(request.options.top_k),
            LlamaSampler::top_p(request.options.top_p, 1),
            LlamaSampler::min_p(request.options.min_p, 1),
        ];
        if let Some(typical_p) = request.sampling.typical_p {
            filters.push(LlamaSampler::typical(typical_p, 1));
        }
        if let Some(top_n_sigma) = request.sampling.top_n_sigma {
            filters.push(LlamaSampler::top_n_sigma(top_n_sigma));
        }
        filters.push(LlamaSampler::temp(request.options.temperature));
        (
            Some(LlamaSampler::chain_simple(filters)),
            LlamaSampler::dist(request.options.seed),
        )
    } else {
        (None, LlamaSampler::greedy())
    };
    Ok(SamplingState {
        bias,
        penalties,
        grammar,
        filters,
        selector,
        reasoning_control,
    })
}

fn grammar_sampler(
    model: &LlamaModel,
    metadata: &PreparedMetadata,
) -> Result<Option<LlamaSampler>, Error> {
    if metadata.grammar.is_empty() {
        return Ok(None);
    }
    if !metadata.grammar_lazy {
        return LlamaSampler::grammar(model, &metadata.grammar, "root")
            .map(Some)
            .map_err(|error| Error::InvalidConfig(format!("common-chat grammar: {error}")));
    }
    let mut patterns = Vec::new();
    let mut tokens = Vec::new();
    for trigger in &metadata.grammar_triggers {
        match trigger.r#type {
            0 if trigger.token >= 0 => tokens.push(LlamaToken(trigger.token)),
            0 => {
                return Err(Error::Unsupported {
                    field: "grammar_lazy",
                    detail: "template produced an unresolved token trigger".to_string(),
                });
            }
            1 => patterns.push(regex_escape(&trigger.value)),
            2 => patterns.push(trigger.value.clone()),
            3 => patterns.push(anchor_pattern(&trigger.value)),
            other => {
                return Err(Error::Unsupported {
                    field: "grammar_lazy",
                    detail: format!("template produced unknown grammar trigger type {other}"),
                });
            }
        }
    }
    LlamaSampler::grammar_lazy_patterns(model, &metadata.grammar, "root", &patterns, &tokens)
        .map(Some)
        .map_err(|error| Error::InvalidConfig(format!("common-chat lazy grammar: {error}")))
}

fn regex_escape(value: &str) -> String {
    let mut escaped = String::with_capacity(value.len());
    for character in value.chars() {
        if matches!(
            character,
            '.' | '^' | '$' | '*' | '+' | '?' | '(' | ')' | '[' | ']' | '{' | '}' | '\\' | '|'
        ) {
            escaped.push('\\');
        }
        escaped.push(character);
    }
    escaped
}

fn anchor_pattern(value: &str) -> String {
    if value.is_empty() {
        return "^$".to_string();
    }
    format!(
        "{}{}{}",
        if value.starts_with('^') { "" } else { "^" },
        value,
        if value.ends_with('$') { "" } else { "$" }
    )
}

impl SamplingState {
    pub(crate) fn sample(
        &mut self,
        model: &LlamaModel,
        context: &llama_cpp_2::context::LlamaContext<'_>,
        logits_index: i32,
        top_logprobs: Option<usize>,
    ) -> Result<(LlamaToken, Option<TokenLogprobs>), Error> {
        let candidates = context.token_data_array_ith(logits_index);
        self.sample_candidates(model, candidates, top_logprobs)
    }

    pub(crate) fn sample_candidates(
        &mut self,
        model: &LlamaModel,
        mut candidates: LlamaTokenDataArray,
        top_logprobs: Option<usize>,
    ) -> Result<(LlamaToken, Option<TokenLogprobs>), Error> {
        if let Some(bias) = &self.bias {
            bias.apply(&mut candidates);
        }
        if let Some(control) = &self.reasoning_control {
            control.apply(&mut candidates)?;
        }
        self.penalties.apply(&mut candidates);
        if let Some(grammar) = &self.grammar {
            grammar.apply(&mut candidates);
        }
        if let Some(filters) = &self.filters {
            filters.apply(&mut candidates);
        }
        if candidates.data.is_empty() {
            return Err(Error::Generation(
                "sampling processors removed every token candidate".to_string(),
            ));
        }
        let probabilities = top_logprobs
            .map(|top| normalized_logprobs(model, &candidates, top))
            .transpose()?;
        self.selector.apply(&mut candidates);
        let token = candidates.selected_token().ok_or_else(|| {
            Error::Generation("token selector did not select a candidate".to_string())
        })?;
        self.penalties.accept(token);
        if let Some(control) = &mut self.reasoning_control {
            control.accept(token.0)?;
        }
        if let Some(grammar) = &mut self.grammar {
            grammar.try_accept(token).map_err(|error| {
                Error::Generation(format!("grammar acceptance failed: {error}"))
            })?;
        }
        self.selector.accept(token);
        let logprobs = probabilities
            .map(|probabilities| {
                let chosen_logprob = probabilities
                    .all
                    .into_iter()
                    .find_map(|(token_id, logprob)| (token_id == token.0).then_some(logprob))
                    .ok_or_else(|| {
                        Error::Generation(
                            "selected token was absent from processed candidates".to_string(),
                        )
                    })?;
                Ok(TokenLogprobs {
                    chosen: TokenAlternative {
                        token_id: token.0,
                        bytes: token_bytes(model, token)?,
                        logprob: chosen_logprob,
                    },
                    top: probabilities.top,
                })
            })
            .transpose()?;
        Ok((token, logprobs))
    }

    pub(crate) fn force_reasoning_end(&mut self) -> Result<bool, Error> {
        self.reasoning_control
            .as_mut()
            .map_or(Ok(false), ReasoningControl::force)
    }
}

fn validate_sampling_request(model: &LlamaModel, request: &AdvancedRequest) -> Result<(), Error> {
    validate_sampling_bounds(model.n_vocab(), request)
}

fn validate_sampling_bounds(n_vocab: i32, request: &AdvancedRequest) -> Result<(), Error> {
    if request
        .sampling
        .typical_p
        .is_some_and(|value| !value.is_finite() || !(0.0..=1.0).contains(&value))
    {
        return Err(Error::InvalidConfig(
            "typical_p must be finite and in 0..=1".to_string(),
        ));
    }
    if request
        .sampling
        .top_n_sigma
        .is_some_and(|value| !value.is_finite() || value < 0.0)
    {
        return Err(Error::InvalidConfig(
            "top_n_sigma must be finite and nonnegative".to_string(),
        ));
    }
    if request
        .logprobs
        .is_some_and(|options| options.top_logprobs > 20)
    {
        return Err(Error::InvalidConfig(
            "top_logprobs must be in 0..=20".to_string(),
        ));
    }
    if request.logit_bias.len() > 256 {
        return Err(Error::InvalidRequest {
            field: "logit_bias",
            detail: "must contain at most 256 entries".to_string(),
        });
    }
    let mut ids = std::collections::BTreeSet::new();
    for bias in &request.logit_bias {
        if bias.token_id < 0 || bias.token_id >= n_vocab {
            return Err(Error::InvalidRequest {
                field: "logit_bias",
                detail: format!(
                    "token {} is outside model vocabulary 0..{}",
                    bias.token_id, n_vocab
                ),
            });
        }
        if !bias.bias.is_finite() || !(-100.0..=100.0).contains(&bias.bias) {
            return Err(Error::InvalidRequest {
                field: "logit_bias",
                detail: format!(
                    "bias for token {} must be finite and in -100..=100",
                    bias.token_id
                ),
            });
        }
        if !ids.insert(bias.token_id) {
            return Err(Error::InvalidRequest {
                field: "logit_bias",
                detail: format!("duplicate token {}", bias.token_id),
            });
        }
    }
    Ok(())
}

fn normalized_logprobs(
    model: &LlamaModel,
    candidates: &LlamaTokenDataArray,
    top_count: usize,
) -> Result<NormalizedLogprobs, Error> {
    let all = normalize_candidate_logits(candidates)?;
    let top = all
        .iter()
        .take(top_count)
        .map(|(token_id, logprob)| {
            Ok(TokenAlternative {
                token_id: *token_id,
                bytes: token_bytes(model, LlamaToken(*token_id))?,
                logprob: *logprob,
            })
        })
        .collect::<Result<Vec<_>, Error>>()?;
    Ok(NormalizedLogprobs { all, top })
}

fn normalize_candidate_logits(candidates: &LlamaTokenDataArray) -> Result<Vec<(i32, f64)>, Error> {
    let max = candidates
        .data
        .iter()
        .map(llama_cpp_2::token::data::LlamaTokenData::logit)
        .filter(|logit| logit.is_finite())
        .fold(f32::NEG_INFINITY, f32::max);
    if !max.is_finite() {
        return Err(Error::Generation(
            "processed token logits contain no finite candidate".to_string(),
        ));
    }
    let denominator = candidates
        .data
        .iter()
        .map(|candidate| f64::from(candidate.logit() - max).exp())
        .sum::<f64>();
    if !denominator.is_finite() || denominator <= 0.0 {
        return Err(Error::Generation(
            "processed token logits could not be normalized".to_string(),
        ));
    }
    let log_denominator = denominator.ln() + f64::from(max);
    let mut all = candidates
        .data
        .iter()
        .filter(|candidate| candidate.logit().is_finite())
        .map(|candidate| {
            (
                candidate.id().0,
                f64::from(candidate.logit()) - log_denominator,
            )
        })
        .collect::<Vec<_>>();
    all.sort_by(|left, right| {
        right
            .1
            .total_cmp(&left.1)
            .then_with(|| left.0.cmp(&right.0))
    });
    Ok(all)
}

struct NormalizedLogprobs {
    all: Vec<(i32, f64)>,
    top: Vec<TokenAlternative>,
}

fn token_bytes(model: &LlamaModel, token: LlamaToken) -> Result<Vec<u8>, Error> {
    token_bytes_with(|buffer_size, special| {
        model.token_to_piece_bytes(token, buffer_size, special, None)
    })
}

fn token_bytes_with(
    mut convert: impl FnMut(usize, bool) -> Result<Vec<u8>, llama_cpp_2::TokenToStringError>,
) -> Result<Vec<u8>, Error> {
    match token_piece_bytes(&mut convert, false) {
        Ok(bytes) => Ok(bytes),
        Err(llama_cpp_2::TokenToStringError::UnknownTokenType) => {
            match token_piece_bytes(&mut convert, true) {
                Ok(bytes) => Ok(bytes),
                Err(llama_cpp_2::TokenToStringError::UnknownTokenType) => Ok(Vec::new()),
                Err(error) => Err(Error::Generation(error.to_string())),
            }
        }
        Err(error) => Err(Error::Generation(error.to_string())),
    }
}

fn token_piece_bytes(
    convert: &mut impl FnMut(usize, bool) -> Result<Vec<u8>, llama_cpp_2::TokenToStringError>,
    special: bool,
) -> Result<Vec<u8>, llama_cpp_2::TokenToStringError> {
    match convert(256, special) {
        Err(llama_cpp_2::TokenToStringError::InsufficientBufferSpace(required)) => {
            let Some(required) = required
                .checked_neg()
                .and_then(|value| usize::try_from(value).ok())
                .filter(|value| *value <= 1024 * 1024)
            else {
                return Err(llama_cpp_2::TokenToStringError::InsufficientBufferSpace(
                    required,
                ));
            };
            convert(required, special)
        }
        result => result,
    }
}

pub(crate) fn try_flush_content(slot: &mut ActiveSlot) -> Result<(), Error> {
    if let Some(delta) = slot.pending_semantic.pop_front() {
        let Some(events) = slot.lifecycle.events.as_ref() else {
            return Err(Error::Cancelled);
        };
        match events.try_send(Event::Semantic(delta)) {
            Ok(()) => return Ok(()),
            Err(mpsc::error::TrySendError::Full(Event::Semantic(delta))) => {
                slot.pending_semantic.push_front(delta);
                return Ok(());
            }
            Err(mpsc::error::TrySendError::Closed(_)) => return Err(Error::Cancelled),
            Err(mpsc::error::TrySendError::Full(_)) => {
                unreachable!("only semantic delta is flushed")
            }
        }
    }
    if let Some((text, logprobs)) = slot.pending_tokens.pop_front() {
        let Some(events) = slot.lifecycle.events.as_ref() else {
            return Err(Error::Cancelled);
        };
        match events.try_send(Event::Token { text, logprobs }) {
            Ok(()) => {}
            Err(mpsc::error::TrySendError::Full(Event::Token { text, logprobs })) => {
                slot.pending_tokens.push_front((text, logprobs));
                return Ok(());
            }
            Err(mpsc::error::TrySendError::Closed(_)) => return Err(Error::Cancelled),
            Err(mpsc::error::TrySendError::Full(_)) => unreachable!("only token is flushed"),
        }
    }
    let Some(content) = slot.pending_content.take() else {
        return Ok(());
    };
    let Some(events) = slot.lifecycle.events.as_ref() else {
        return Err(Error::Cancelled);
    };
    match events.try_send(Event::Content(content)) {
        Ok(()) => Ok(()),
        Err(mpsc::error::TrySendError::Full(Event::Content(content))) => {
            slot.pending_content = Some(content);
            Ok(())
        }
        Err(mpsc::error::TrySendError::Closed(_)) => Err(Error::Cancelled),
        Err(mpsc::error::TrySendError::Full(_)) => unreachable!("only content is flushed"),
    }
}

pub(crate) fn try_flush_draining(slot: &mut DrainingSlot) -> bool {
    if let Some(delta) = slot.pending_semantic.pop_front() {
        let Some(events) = slot.lifecycle.events.as_ref() else {
            return true;
        };
        match events.try_send(Event::Semantic(delta)) {
            Ok(()) if !slot.pending_semantic.is_empty() => return false,
            Ok(()) => {}
            Err(mpsc::error::TrySendError::Closed(_)) => return true,
            Err(mpsc::error::TrySendError::Full(Event::Semantic(delta))) => {
                slot.pending_semantic.push_front(delta);
                return false;
            }
            Err(mpsc::error::TrySendError::Full(_)) => {
                unreachable!("only semantic delta is flushed")
            }
        }
    }
    if let Some((text, logprobs)) = slot.pending_tokens.pop_front() {
        let Some(events) = slot.lifecycle.events.as_ref() else {
            return true;
        };
        match events.try_send(Event::Token { text, logprobs }) {
            Ok(()) => {}
            Err(mpsc::error::TrySendError::Closed(_)) => return true,
            Err(mpsc::error::TrySendError::Full(Event::Token { text, logprobs })) => {
                slot.pending_tokens.push_front((text, logprobs));
                return false;
            }
            Err(mpsc::error::TrySendError::Full(_)) => unreachable!("only token is flushed"),
        }
    }
    let Some(content) = slot.pending_content.take() else {
        return true;
    };
    let Some(events) = slot.lifecycle.events.as_ref() else {
        return true;
    };
    match events.try_send(Event::Content(content)) {
        Ok(()) | Err(mpsc::error::TrySendError::Closed(_)) => true,
        Err(mpsc::error::TrySendError::Full(Event::Content(content))) => {
            slot.pending_content = Some(content);
            false
        }
        Err(mpsc::error::TrySendError::Full(_)) => unreachable!("only content is flushed"),
    }
}

pub(crate) fn finish_event(active: &ActiveSlot, reason: FinishReason) -> Event {
    Event::Finished {
        reason,
        usage: active.usage(),
    }
}

pub(crate) fn timings_from_wall(
    cache_n: usize,
    prompt_n: usize,
    prompt_elapsed: Duration,
    predicted_n: usize,
    predicted_elapsed: Duration,
) -> Timings {
    // Native timing snapshots are context-global. Shared batches instead charge each participating
    // slot the batch wall time while retaining that slot's own prompt/completion token counts.
    let prompt_ms = finite_nonnegative(prompt_elapsed.as_secs_f64() * 1_000.0);
    let predicted_ms = finite_nonnegative(predicted_elapsed.as_secs_f64() * 1_000.0);
    let (prompt_per_token_ms, prompt_per_second) = timing_rates(prompt_n, prompt_ms);
    let (predicted_per_token_ms, predicted_per_second) = timing_rates(predicted_n, predicted_ms);
    Timings {
        cache_n,
        prompt_n,
        prompt_ms,
        prompt_per_token_ms,
        prompt_per_second,
        predicted_n,
        predicted_ms,
        predicted_per_token_ms,
        predicted_per_second,
    }
}

pub(crate) fn finite_nonnegative(value: f64) -> f64 {
    if value.is_finite() && value >= 0.0 {
        value
    } else {
        0.0
    }
}

pub(crate) fn timing_rates(tokens: usize, milliseconds: f64) -> (f64, f64) {
    if tokens == 0 || milliseconds <= 0.0 {
        return (0.0, 0.0);
    }
    let Ok(tokens) = u32::try_from(tokens) else {
        return (0.0, 0.0);
    };
    let tokens = f64::from(tokens);
    (
        finite_nonnegative(milliseconds / tokens),
        finite_nonnegative(tokens * 1_000.0 / milliseconds),
    )
}

pub(crate) fn send_event(
    events: &mpsc::Sender<Event>,
    mut event: Event,
    cancelled: &AtomicBool,
) -> Result<(), Error> {
    loop {
        if cancelled.load(Ordering::Acquire) {
            return Err(Error::Cancelled);
        }
        match events.try_send(event) {
            Ok(()) => return Ok(()),
            Err(mpsc::error::TrySendError::Closed(_)) => return Err(Error::Cancelled),
            Err(mpsc::error::TrySendError::Full(pending)) => {
                event = pending;
                std::thread::sleep(Duration::from_millis(1));
            }
        }
    }
}

pub(crate) struct StopFilter {
    stops: Vec<String>,
    pending: String,
    pending_tokens: VecDeque<(String, TokenLogprobs)>,
}

impl StopFilter {
    pub(crate) fn new(stops: Vec<String>) -> Self {
        Self {
            stops,
            pending: String::new(),
            pending_tokens: VecDeque::new(),
        }
    }

    pub(crate) fn push_piece(&mut self, piece: &str) -> (bool, Option<String>) {
        self.pending.push_str(piece);
        if let Some(index) = self
            .stops
            .iter()
            .filter_map(|stop| self.pending.find(stop))
            .min()
        {
            let output = self.take_prefix(index);
            self.pending.clear();
            return (true, output);
        }
        let retained = longest_stop_prefix_suffix(&self.pending, &self.stops);
        let emit_len = self.pending.len() - retained;
        (false, self.take_prefix(emit_len))
    }

    pub(crate) fn take_flush(&mut self) -> Option<String> {
        self.take_prefix(self.pending.len())
    }

    pub(crate) fn push_token(
        &mut self,
        piece: String,
        logprobs: TokenLogprobs,
    ) -> (bool, Vec<(String, TokenLogprobs)>) {
        self.pending.push_str(&piece);
        self.pending_tokens.push_back((piece, logprobs));
        if let Some(index) = self
            .stops
            .iter()
            .filter_map(|stop| self.pending.find(stop))
            .min()
        {
            let output = self.take_token_prefix(index, true);
            self.pending.clear();
            self.pending_tokens.clear();
            return (true, output);
        }
        let retained = longest_stop_prefix_suffix(&self.pending, &self.stops);
        let emit_len = self.pending.len() - retained;
        (false, self.take_token_prefix(emit_len, false))
    }

    pub(crate) fn take_token_flush(&mut self) -> Vec<(String, TokenLogprobs)> {
        let len = self.pending.len();
        self.take_token_prefix(len, true)
    }

    #[cfg(test)]
    pub(crate) fn push(
        &mut self,
        piece: &str,
        events: &mpsc::Sender<Event>,
        cancelled: &AtomicBool,
    ) -> Result<bool, Error> {
        let (stopped, output) = self.push_piece(piece);
        if let Some(output) = output {
            send_event(events, Event::Content(output), cancelled)?;
        }
        Ok(stopped)
    }

    fn take_prefix(&mut self, len: usize) -> Option<String> {
        if len == 0 {
            return None;
        }
        let suffix = self.pending.split_off(len);
        let output = std::mem::replace(&mut self.pending, suffix);
        Some(output)
    }

    fn take_token_prefix(
        &mut self,
        limit: usize,
        include_partial: bool,
    ) -> Vec<(String, TokenLogprobs)> {
        let mut consumed = 0;
        let mut output = Vec::new();
        while let Some((piece, mut logprobs)) = self.pending_tokens.pop_front() {
            if piece.is_empty() {
                continue;
            }
            if consumed + piece.len() <= limit {
                consumed += piece.len();
                logprobs.chosen.bytes = piece.as_bytes().to_vec();
                output.push((piece, logprobs));
                continue;
            }
            if include_partial && consumed < limit {
                let visible = piece[..limit - consumed].to_string();
                consumed = limit;
                logprobs.chosen.bytes = visible.as_bytes().to_vec();
                output.push((visible, logprobs));
            } else {
                self.pending_tokens.push_front((piece, logprobs));
            }
            break;
        }
        if consumed > 0 {
            let suffix = self.pending.split_off(consumed);
            self.pending = suffix;
        }
        output
    }
}

fn longest_stop_prefix_suffix(text: &str, stops: &[String]) -> usize {
    stops
        .iter()
        .flat_map(|stop| {
            stop.char_indices()
                .map(|(index, _)| &stop[..index])
                .chain(std::iter::once(stop.as_str()))
        })
        .filter(|prefix| !prefix.is_empty() && text.ends_with(prefix))
        .map(str::len)
        .max()
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_logprobs(bytes: &[u8]) -> TokenLogprobs {
        TokenLogprobs {
            chosen: TokenAlternative {
                token_id: 1,
                bytes: bytes.to_vec(),
                logprob: -0.5,
            },
            top: Vec::new(),
        }
    }

    struct FakeSource {
        decode: Option<LlamaToken>,
        prefill: Vec<LlamaToken>,
        logits_at: Option<usize>,
    }

    impl BatchSource for FakeSource {
        fn decoding_entry(&self, slot: usize) -> Option<BatchEntry> {
            self.decode.map(|token| BatchEntry {
                slot,
                token,
                position: 9,
                logits: true,
                kind: BatchEntryKind::Decode,
            })
        }

        fn remaining_prefill(&self) -> usize {
            self.prefill.len()
        }

        fn prefill_entry(&self, slot: usize, offset: usize) -> Option<BatchEntry> {
            self.prefill.get(offset).copied().map(|token| BatchEntry {
                slot,
                token,
                position: i32::try_from(offset).unwrap(),
                logits: self.logits_at == Some(offset),
                kind: BatchEntryKind::Prefill,
            })
        }
    }

    #[test]
    fn batch_plan_places_every_decode_before_prefill_and_maps_raw_logits_indices() {
        let sources = [
            FakeSource {
                decode: Some(LlamaToken(10)),
                prefill: Vec::new(),
                logits_at: None,
            },
            FakeSource {
                decode: None,
                prefill: vec![LlamaToken(20), LlamaToken(21)],
                logits_at: Some(1),
            },
            FakeSource {
                decode: Some(LlamaToken(30)),
                prefill: Vec::new(),
                logits_at: None,
            },
        ];
        let plan = plan_sources(&sources, 4, 0);
        assert_eq!(
            plan.entries
                .iter()
                .map(|entry| (entry.slot, entry.kind))
                .collect::<Vec<_>>(),
            vec![
                (0, BatchEntryKind::Decode),
                (2, BatchEntryKind::Decode),
                (1, BatchEntryKind::Prefill),
                (1, BatchEntryKind::Prefill),
            ]
        );
        assert_eq!(logits_targets(&plan).unwrap(), vec![(0, 0), (1, 2), (3, 1)]);
    }

    #[test]
    fn rotating_prefill_start_prevents_long_prompt_starvation() {
        let sources = [
            FakeSource {
                decode: None,
                prefill: vec![LlamaToken(1); 12],
                logits_at: None,
            },
            FakeSource {
                decode: None,
                prefill: vec![LlamaToken(2); 12],
                logits_at: None,
            },
            FakeSource {
                decode: None,
                prefill: vec![LlamaToken(3); 12],
                logits_at: None,
            },
        ];
        let first = plan_sources(&sources, 4, 0);
        let second = plan_sources(&sources, 4, first.next_prefill_slot);
        let third = plan_sources(&sources, 4, second.next_prefill_slot);
        assert_eq!(first.entries.first().unwrap().slot, 0);
        assert_eq!(second.entries.first().unwrap().slot, 1);
        assert_eq!(third.entries.first().unwrap().slot, 2);
        for slot in 0..3 {
            assert!(
                first
                    .entries
                    .iter()
                    .chain(&second.entries)
                    .chain(&third.entries)
                    .any(|entry| entry.slot == slot)
            );
        }
    }

    fn advanced_request() -> AdvancedRequest {
        AdvancedRequest {
            input: crate::contract::Input::Prompt("x".to_string()),
            options: crate::contract::GenerationOptions {
                max_tokens: 1,
                temperature: 1.0,
                top_p: 1.0,
                top_k: 0,
                min_p: 0.0,
                presence_penalty: 0.0,
                frequency_penalty: 0.0,
                repeat_penalty: 1.0,
                seed: 1,
                stop: Vec::new(),
            },
            output: crate::contract::OutputConstraint::Text,
            logprobs: Some(crate::contract::LogprobsOptions { top_logprobs: 20 }),
            logit_bias: vec![crate::contract::LogitBias {
                token_id: 3,
                bias: -100.0,
            }],
            sampling: crate::contract::SamplingExtensions {
                typical_p: Some(0.9),
                top_n_sigma: Some(2.0),
            },
            choices: 1,
            reasoning_control_id: None,
        }
    }

    #[test]
    fn advanced_sampling_bounds_accept_edges_and_reject_invalid_values() {
        let valid = advanced_request();
        assert!(validate_sampling_bounds(10, &valid).is_ok());

        let mut invalid = valid.clone();
        invalid.logprobs = Some(crate::contract::LogprobsOptions { top_logprobs: 21 });
        assert!(validate_sampling_bounds(10, &invalid).is_err());
        invalid = valid.clone();
        invalid.logit_bias[0].token_id = 10;
        assert!(matches!(
            validate_sampling_bounds(10, &invalid),
            Err(Error::InvalidRequest {
                field: "logit_bias",
                ..
            })
        ));
        invalid = valid.clone();
        invalid.logit_bias[0].bias = f32::NAN;
        assert!(validate_sampling_bounds(10, &invalid).is_err());
        invalid = valid.clone();
        invalid.logit_bias.push(invalid.logit_bias[0]);
        assert!(validate_sampling_bounds(10, &invalid).is_err());
        invalid = valid.clone();
        invalid.logit_bias = (0..257)
            .map(|token_id| crate::contract::LogitBias {
                token_id,
                bias: 0.0,
            })
            .collect();
        assert!(matches!(
            validate_sampling_bounds(300, &invalid),
            Err(Error::InvalidRequest {
                field: "logit_bias",
                ..
            })
        ));
        invalid = valid.clone();
        invalid.sampling.typical_p = Some(1.1);
        assert!(validate_sampling_bounds(10, &invalid).is_err());
        invalid = valid;
        invalid.sampling.top_n_sigma = Some(-0.1);
        assert!(validate_sampling_bounds(10, &invalid).is_err());
    }

    #[test]
    fn logprobs_are_natural_log_normalized_and_stably_ranked() {
        let candidates = LlamaTokenDataArray::new(
            vec![
                llama_cpp_2::token::data::LlamaTokenData::new(LlamaToken(2), 0.0, 0.0),
                llama_cpp_2::token::data::LlamaTokenData::new(LlamaToken(1), 0.0, 0.0),
                llama_cpp_2::token::data::LlamaTokenData::new(
                    LlamaToken(3),
                    f32::NEG_INFINITY,
                    0.0,
                ),
            ],
            false,
        );
        let values = normalize_candidate_logits(&candidates).unwrap();
        assert_eq!(values.len(), 2);
        assert_eq!(
            values.iter().map(|value| value.0).collect::<Vec<_>>(),
            [1, 2]
        );
        for (_, logprob) in values {
            assert!((logprob + std::f64::consts::LN_2).abs() < 1e-12);
        }
    }

    #[test]
    fn token_piece_conversion_retries_special_tokens_with_the_required_buffer() {
        let mut calls = Vec::new();
        let convert = |buffer_size, special| {
            calls.push((buffer_size, special));
            match calls.len() {
                1 => Err(llama_cpp_2::TokenToStringError::UnknownTokenType),
                2 => Err(llama_cpp_2::TokenToStringError::InsufficientBufferSpace(
                    -300,
                )),
                3 => Ok(b"<special>".to_vec()),
                _ => unreachable!(),
            }
        };

        let bytes = token_bytes_with(convert).unwrap();

        assert_eq!(bytes, b"<special>");
        assert_eq!(calls, [(256, false), (256, true), (300, true)]);
    }

    #[test]
    fn token_piece_conversion_only_degrades_unknown_special_tokens() {
        let bytes = token_bytes_with(|_, _| Err(llama_cpp_2::TokenToStringError::UnknownTokenType))
            .unwrap();
        assert!(bytes.is_empty());

        let utf8_error = String::from_utf8(vec![0xff]).unwrap_err();
        let error = token_bytes_with(|_, _| {
            Err(llama_cpp_2::TokenToStringError::FromUtf8Error(
                utf8_error.clone(),
            ))
        })
        .unwrap_err();
        assert!(matches!(error, Error::Generation(_)));
    }

    #[test]
    fn logprobs_follow_visible_text_and_discard_cross_token_stop_material() {
        let mut filter = StopFilter::new(vec!["END".to_string()]);
        let (stopped, output) = filter.push_token("hello E".to_string(), test_logprobs(b"hello E"));
        assert!(!stopped);
        assert!(output.is_empty());

        let (stopped, output) = filter.push_token("ND".to_string(), test_logprobs(b"ND"));
        assert!(stopped);
        assert_eq!(output.len(), 1);
        assert_eq!(output[0].0, "hello ");
        assert_eq!(output[0].1.chosen.bytes, b"hello ");
        assert!(filter.take_token_flush().is_empty());
    }

    #[test]
    fn logprob_buffer_preserves_split_utf8_and_flushes_unmatched_stop_prefixes() {
        let mut filter = StopFilter::new(vec!["END".to_string()]);
        let (_, first) = filter.push_token(String::new(), test_logprobs(&[0xc3]));
        assert!(first.is_empty());
        let (_, second) = filter.push_token("é".to_string(), test_logprobs(&[0xa9]));
        assert_eq!(second.len(), 1);
        assert_eq!(second[0].0, "é");
        assert_eq!(second[0].1.chosen.bytes, "é".as_bytes());

        let (_, pending) = filter.push_token("EN".to_string(), test_logprobs(b"EN"));
        assert!(pending.is_empty());
        let flushed = filter.take_token_flush();
        assert_eq!(flushed.len(), 1);
        assert_eq!(flushed[0].0, "EN");
        assert_eq!(flushed[0].1.chosen.bytes, b"EN");
    }
}
