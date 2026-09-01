#ifndef ORCHION_COMMON_CHAT_BRIDGE_H
#define ORCHION_COMMON_CHAT_BRIDGE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
#define ORCHION_NOEXCEPT noexcept
extern "C" {
#else
#define ORCHION_NOEXCEPT
#endif

typedef struct orchion_common_chat_prepared orchion_common_chat_prepared;
typedef struct orchion_reasoning_control orchion_reasoning_control;

typedef struct orchion_common_chat_buffer {
    uint8_t * data;
    size_t len;
} orchion_common_chat_buffer;

int32_t orchion_common_chat_prepare(
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
    orchion_common_chat_buffer * error) ORCHION_NOEXCEPT;

int32_t orchion_common_chat_parse(
    const orchion_common_chat_prepared * prepared,
    const uint8_t * generated,
    size_t generated_len,
    int32_t is_partial,
    orchion_common_chat_buffer * result_json,
    orchion_common_chat_buffer * error) ORCHION_NOEXCEPT;

int32_t orchion_reasoning_control_init(
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
    orchion_common_chat_buffer * error) ORCHION_NOEXCEPT;

int32_t orchion_reasoning_control_apply(
    const orchion_reasoning_control * control,
    const int32_t * token_ids,
    float * logits,
    size_t len,
    orchion_common_chat_buffer * error) ORCHION_NOEXCEPT;

int32_t orchion_reasoning_control_accept(
    orchion_reasoning_control * control,
    int32_t token,
    orchion_common_chat_buffer * error) ORCHION_NOEXCEPT;

// 0 = forced, 1 = not actively reasoning, 2/3 = invalid/native failure.
int32_t orchion_reasoning_control_force(
    orchion_reasoning_control * control,
    orchion_common_chat_buffer * error) ORCHION_NOEXCEPT;

void orchion_common_chat_prepared_free(orchion_common_chat_prepared * prepared) ORCHION_NOEXCEPT;
void orchion_reasoning_control_free(orchion_reasoning_control * control) ORCHION_NOEXCEPT;
void orchion_common_chat_buffer_free(orchion_common_chat_buffer buffer) ORCHION_NOEXCEPT;

#ifdef __cplusplus
}
#endif

#undef ORCHION_NOEXCEPT

#endif
