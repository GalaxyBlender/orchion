use encoding_rs::{CoderResult, Decoder};

pub(crate) fn decode_piece(decoder: &mut Decoder, bytes: &[u8]) -> String {
    // Include space for an incomplete UTF-8 sequence retained from the previous token.
    let mut output = String::with_capacity(
        decoder
            .max_utf8_buffer_length(bytes.len())
            .expect("token output is too large to decode"),
    );
    let (result, read, _) = decoder.decode_to_string(bytes, &mut output, false);
    assert!(
        matches!(result, CoderResult::InputEmpty) && read == bytes.len(),
        "UTF-8 decoder capacity bound must consume the complete token"
    );
    output
}

#[cfg(test)]
mod tests {
    use super::decode_piece;

    #[test]
    fn preserves_utf8_split_across_token_pieces() {
        let mut decoder = encoding_rs::UTF_8.new_decoder();
        assert_eq!(decode_piece(&mut decoder, &[0xe5, 0x9b]), "");
        assert_eq!(decode_piece(&mut decoder, &[0xb2]), "囲");
        assert_eq!(decode_piece(&mut decoder, &[0xf0, 0x9f]), "");
        assert_eq!(decode_piece(&mut decoder, &[0x98, 0x80]), "😀");
    }

    #[test]
    fn consumes_long_pieces_and_replaces_invalid_utf8() {
        let mut decoder = encoding_rs::UTF_8.new_decoder();
        assert_eq!(decode_piece(&mut decoder, &[0xe5, 0x9b]), "");
        let text = format!("囲{}", "text".repeat(1024));
        assert_eq!(decode_piece(&mut decoder, &text.as_bytes()[2..]), text);
        assert_eq!(decode_piece(&mut decoder, &[0xff, b'a', 0]), "\u{fffd}a\0");
    }
}
