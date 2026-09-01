#include "common_chat_bridge.h"

#include "common/chat.h"
#include "common/reasoning-budget.h"
#include "nlohmann/json.hpp"

#include <algorithm>
#include <cstring>
#include <exception>
#include <memory>
#include <new>
#include <stdexcept>
#include <string>
#include <utility>

using ordered_json = nlohmann::ordered_json;

struct orchion_common_chat_prepared {
    common_chat_templates_ptr templates;
    common_chat_parser_params parser;
};

struct orchion_reasoning_control {
    llama_sampler * sampler = nullptr;

    ~orchion_reasoning_control() {
        llama_sampler_free(sampler);
    }
};

namespace {

std::string read_bytes(const uint8_t * data, size_t len, const char * name) {
    if (data == nullptr && len != 0) {
        throw std::invalid_argument(std::string(name) + " pointer is null");
    }
    if (len == 0) {
        return {};
    }
    return std::string(reinterpret_cast<const char *>(data), len);
}

llama_tokens read_tokens(const int32_t * data, size_t len, const char * name) {
    if (data == nullptr && len != 0) {
        throw std::invalid_argument(std::string(name) + " pointer is null");
    }
    if (len == 0) {
        return {};
    }
    return llama_tokens(data, data + len);
}

void clear(orchion_common_chat_buffer * buffer) noexcept {
    if (buffer != nullptr) {
        buffer->data = nullptr;
        buffer->len = 0;
    }
}

void write(orchion_common_chat_buffer * buffer, const std::string & value) {
    if (buffer == nullptr) {
        throw std::invalid_argument("output buffer pointer is null");
    }
    if (value.empty()) {
        return;
    }
    auto data = std::make_unique<uint8_t[]>(value.size());
    std::memcpy(data.get(), value.data(), value.size());
    buffer->data = data.release();
    buffer->len = value.size();
}

void write_error(orchion_common_chat_buffer * error, const char * message) noexcept {
    try {
        write(error, message == nullptr ? "unknown native bridge error" : std::string(message));
    } catch (...) {
        clear(error);
    }
}

ordered_json prepare_result_json(const common_chat_params & params,
                                 const common_chat_templates * templates,
                                 const common_chat_parser_params & parser) {
    ordered_json triggers = ordered_json::array();
    for (const auto & trigger : params.grammar_triggers) {
        triggers.push_back({
            {"type", static_cast<int>(trigger.type)},
            {"value", trigger.value},
            {"token", trigger.token},
        });
    }
    return {
        {"prompt", params.prompt},
        {"grammar", params.grammar},
        {"grammar_lazy", params.grammar_lazy},
        {"grammar_triggers", std::move(triggers)},
        {"generation_prompt", params.generation_prompt},
        {"preserved_tokens", params.preserved_tokens},
        {"additional_stops", params.additional_stops},
        {"format", common_chat_format_name(params.format)},
        {"supports_thinking", params.supports_thinking},
        {"thinking_start_tag", params.thinking_start_tag},
        {"thinking_end_tags", params.thinking_end_tags},
        {"reasoning_format", common_reasoning_format_name(parser.reasoning_format)},
        {"reasoning_in_content", parser.reasoning_in_content},
        {"parse_tool_calls", parser.parse_tool_calls},
        {"template_caps", common_chat_templates_get_caps(templates)},
    };
}

} // namespace

extern "C" int32_t orchion_common_chat_prepare(
    const uint8_t * template_data,
    size_t template_len,
    const uint8_t * bos_data,
    size_t bos_len,
    const uint8_t * eos_data,
    size_t eos_len,
    const uint8_t * request_json,
    size_t request_len,
    orchion_common_chat_prepared ** prepared,
    orchion_common_chat_buffer * result_json,
    orchion_common_chat_buffer * error) noexcept {
    if (prepared != nullptr) {
        *prepared = nullptr;
    }
    clear(result_json);
    clear(error);
    try {
        if (prepared == nullptr) {
            throw std::invalid_argument("prepared output pointer is null");
        }
        auto request = ordered_json::parse(read_bytes(request_json, request_len, "request JSON"));
        auto templates = common_chat_templates_init(
            nullptr,
            read_bytes(template_data, template_len, "template"),
            read_bytes(bos_data, bos_len, "BOS token"),
            read_bytes(eos_data, eos_len, "EOS token"));
        if (!templates) {
            throw std::runtime_error("common_chat_templates_init returned null");
        }

        common_chat_templates_inputs inputs;
        inputs.messages = common_chat_msgs_parse_oaicompat(request.at("messages"));
        inputs.tools = common_chat_tools_parse_oaicompat(request.value("tools", ordered_json::array()));
        inputs.tool_choice = common_chat_tool_choice_parse_oaicompat(request.value("tool_choice", "none"));
        inputs.parallel_tool_calls = request.value("parallel_tool_calls", false);
        inputs.reasoning_format = common_reasoning_format_from_name(request.value("reasoning_format", "none"));
        inputs.enable_thinking = request.value("enable_thinking", false);
        if (request.contains("reasoning_effort") && !request.at("reasoning_effort").is_null()) {
            inputs.chat_template_kwargs["reasoning_effort"] = request.at("reasoning_effort").dump();
        }
        inputs.grammar = request.value("grammar", "");
        inputs.json_schema = request.value("json_schema", "");
        inputs.add_generation_prompt = true;
        inputs.use_jinja = true;

        auto params = common_chat_templates_apply(templates.get(), inputs);
        common_chat_parser_params parser(params);
        parser.reasoning_format = inputs.reasoning_format;
        parser.reasoning_in_content = false;
        parser.parse_tool_calls = !inputs.tools.empty() && inputs.tool_choice != COMMON_CHAT_TOOL_CHOICE_NONE;
        parser.parser.load(params.parser);

        auto owned = std::make_unique<orchion_common_chat_prepared>();
        owned->templates = std::move(templates);
        owned->parser = std::move(parser);
        write(result_json, prepare_result_json(params, owned->templates.get(), owned->parser).dump());
        *prepared = owned.release();
        return 0;
    } catch (const std::exception & exception) {
        write_error(error, exception.what());
        return 1;
    } catch (...) {
        write_error(error, "unknown C++ exception");
        return 2;
    }
}

extern "C" int32_t orchion_common_chat_parse(
    const orchion_common_chat_prepared * prepared,
    const uint8_t * generated,
    size_t generated_len,
    int32_t is_partial,
    orchion_common_chat_buffer * result_json,
    orchion_common_chat_buffer * error) noexcept {
    clear(result_json);
    clear(error);
    try {
        if (prepared == nullptr) {
            throw std::invalid_argument("prepared handle is null");
        }
        auto message = common_chat_parse(
            read_bytes(generated, generated_len, "generated text"),
            is_partial != 0,
            prepared->parser);
        write(result_json, message.to_json_oaicompat().dump());
        return 0;
    } catch (const std::exception & exception) {
        write_error(error, exception.what());
        return 1;
    } catch (...) {
        write_error(error, "unknown C++ exception");
        return 2;
    }
}

extern "C" int32_t orchion_reasoning_control_init(
    const int32_t * start_tokens,
    size_t start_len,
    const int32_t * end_tokens,
    size_t end_tokens_len,
    const size_t * end_offsets,
    size_t end_offsets_len,
    size_t end_count,
    const int32_t * forced_tokens,
    size_t forced_len,
    const int32_t * prompt_tokens,
    size_t prompt_len,
    orchion_reasoning_control ** control,
    orchion_common_chat_buffer * error) noexcept {
    if (control != nullptr) {
        *control = nullptr;
    }
    clear(error);
    try {
        if (control == nullptr) {
            throw std::invalid_argument("control output pointer is null");
        }
        if (end_count == 0 || start_len == 0 || forced_len == 0) {
            throw std::invalid_argument("reasoning control token sequences are empty");
        }
        if (end_count == SIZE_MAX || end_offsets_len != end_count + 1) {
            throw std::invalid_argument("reasoning end offsets length must equal end_count + 1");
        }
        if (end_offsets == nullptr || (end_tokens == nullptr && end_tokens_len != 0)) {
            throw std::invalid_argument("reasoning end token pointer is null");
        }
        if (end_offsets[0] != 0) {
            throw std::invalid_argument("reasoning end offsets must start at zero");
        }
        std::vector<llama_tokens> ends;
        ends.reserve(end_count);
        for (size_t index = 0; index < end_count; ++index) {
            const size_t begin = end_offsets[index];
            const size_t finish = end_offsets[index + 1];
            if (finish <= begin) {
                throw std::invalid_argument("reasoning end token offsets are invalid");
            }
            if (finish > end_tokens_len) {
                throw std::invalid_argument("reasoning end offsets exceed the token buffer");
            }
            ends.push_back(read_tokens(end_tokens + begin, finish - begin, "reasoning end tokens"));
        }
        auto owned = std::make_unique<orchion_reasoning_control>();
        owned->sampler = common_reasoning_budget_init(
            nullptr,
            {read_tokens(start_tokens, start_len, "reasoning start tokens")},
            ends,
            read_tokens(forced_tokens, forced_len, "forced reasoning end tokens"),
            INT32_MAX);
        if (owned->sampler == nullptr) {
            throw std::runtime_error("common_reasoning_budget_init returned null");
        }
        for (const auto token : read_tokens(prompt_tokens, prompt_len, "generation prompt tokens")) {
            llama_sampler_accept(owned->sampler, token);
        }
        *control = owned.release();
        return 0;
    } catch (const std::exception & exception) {
        write_error(error, exception.what());
        return 2;
    } catch (...) {
        write_error(error, "unknown C++ exception");
        return 3;
    }
}

extern "C" int32_t orchion_reasoning_control_apply(
    const orchion_reasoning_control * control,
    const int32_t * token_ids,
    float * logits,
    size_t len,
    orchion_common_chat_buffer * error) noexcept {
    clear(error);
    try {
        if (control == nullptr || (len != 0 && (token_ids == nullptr || logits == nullptr))) {
            throw std::invalid_argument("invalid reasoning control candidates");
        }
        std::vector<llama_token_data> candidates;
        candidates.reserve(len);
        for (size_t index = 0; index < len; ++index) {
            candidates.push_back({token_ids[index], logits[index], 0.0F});
        }
        llama_token_data_array array{candidates.data(), candidates.size(), -1, false};
        llama_sampler_apply(control->sampler, &array);
        for (size_t index = 0; index < len; ++index) {
            logits[index] = candidates[index].logit;
        }
        return 0;
    } catch (const std::exception & exception) {
        write_error(error, exception.what());
        return 2;
    } catch (...) {
        write_error(error, "unknown C++ exception");
        return 3;
    }
}

extern "C" int32_t orchion_reasoning_control_accept(
    orchion_reasoning_control * control,
    int32_t token,
    orchion_common_chat_buffer * error) noexcept {
    clear(error);
    try {
        if (control == nullptr) {
            throw std::invalid_argument("reasoning control handle is null");
        }
        llama_sampler_accept(control->sampler, token);
        return 0;
    } catch (const std::exception & exception) {
        write_error(error, exception.what());
        return 2;
    } catch (...) {
        write_error(error, "unknown C++ exception");
        return 3;
    }
}

extern "C" int32_t orchion_reasoning_control_force(
    orchion_reasoning_control * control,
    orchion_common_chat_buffer * error) noexcept {
    clear(error);
    try {
        if (control == nullptr) {
            throw std::invalid_argument("reasoning control handle is null");
        }
        return common_reasoning_budget_force(control->sampler) ? 0 : 1;
    } catch (const std::exception & exception) {
        write_error(error, exception.what());
        return 2;
    } catch (...) {
        write_error(error, "unknown C++ exception");
        return 3;
    }
}

extern "C" void orchion_common_chat_prepared_free(orchion_common_chat_prepared * prepared) noexcept {
    delete prepared;
}

extern "C" void orchion_reasoning_control_free(orchion_reasoning_control * control) noexcept {
    delete control;
}

extern "C" void orchion_common_chat_buffer_free(orchion_common_chat_buffer buffer) noexcept {
    delete[] buffer.data;
}
