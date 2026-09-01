use crate::contract::{ChoiceEvent, EmbeddingOutput, Event};
use crate::worker::{ChoiceGeneration, Engine, Generation};

pub use crate::scheduler::ScriptedControl;
pub use crate::scheduler::{
    SchedulerInstrumentation, reset_scheduler_instrumentation, scheduler_instrumentation,
};

#[must_use]
pub fn deterministic_generation(events: impl IntoIterator<Item = Event>) -> Generation {
    crate::scheduler::deterministic_generation(events)
}

#[must_use]
pub fn deterministic_choice_generation(
    events: impl IntoIterator<Item = ChoiceEvent>,
) -> ChoiceGeneration {
    crate::scheduler::deterministic_choice_generation(events)
}

#[must_use]
pub fn scripted_engine(script: Vec<Event>, command_capacity: usize) -> (Engine, ScriptedControl) {
    let (engine, control) = crate::scheduler::scripted_engine(script, command_capacity);
    (Engine::from_scheduler(engine), control)
}

#[must_use]
pub fn scripted_slow_preparation_engine(
    script: Vec<Event>,
    command_capacity: usize,
) -> (Engine, ScriptedControl) {
    let (engine, control) =
        crate::scheduler::scripted_slow_preparation_engine(script, command_capacity);
    (Engine::from_scheduler(engine), control)
}

#[must_use]
pub fn scripted_preparation_panicking_engine(command_capacity: usize) -> (Engine, ScriptedControl) {
    let (engine, control) =
        crate::scheduler::scripted_preparation_panicking_engine(command_capacity);
    (Engine::from_scheduler(engine), control)
}

#[must_use]
pub fn scripted_embedding_engine(
    output: EmbeddingOutput,
    command_capacity: usize,
) -> (Engine, ScriptedControl) {
    let (engine, control) = crate::scheduler::scripted_embedding_engine(output, command_capacity);
    (Engine::from_scheduler(engine), control)
}

#[must_use]
pub fn scripted_context_limit_engine(
    prompt_tokens: usize,
    max_tokens: usize,
    context_size: usize,
) -> Engine {
    Engine::from_scheduler(crate::scheduler::scripted_context_limit_engine(
        prompt_tokens,
        max_tokens,
        context_size,
    ))
}
