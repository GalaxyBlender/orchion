use orchion::AsrTranscript;
use std::fmt::Write as _;

#[must_use]
pub fn format_srt(transcript: &AsrTranscript) -> String {
    let mut output = String::new();
    let mut cue_index = 1;

    for segment in &transcript.segments {
        let text = normalize_cue_text(&segment.text);
        if text.is_empty() {
            continue;
        }
        if cue_index > 1 {
            output.push('\n');
        }
        let _ = writeln!(output, "{cue_index}");
        let _ = writeln!(
            output,
            "{} --> {}",
            format_timestamp(segment.start),
            format_timestamp(segment.end)
        );
        let _ = writeln!(output, "{text}");
        cue_index += 1;
    }

    output
}

fn normalize_cue_text(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn format_timestamp(seconds: f32) -> String {
    let milliseconds = (seconds.max(0.0) * 1000.0).round() as u64;
    let hours = milliseconds / 3_600_000;
    let minutes = (milliseconds % 3_600_000) / 60_000;
    let seconds = (milliseconds % 60_000) / 1_000;
    let milliseconds = milliseconds % 1_000;

    format!("{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}")
}

#[cfg(test)]
mod tests {
    use orchion::{AsrSegment, AsrTranscript};

    #[test]
    fn format_srt_uses_one_based_cues_and_millisecond_timestamps() {
        let transcript = AsrTranscript {
            text: "hello world".to_string(),
            language: "en".to_string(),
            raw_output: String::new(),
            segments: vec![
                AsrSegment {
                    id: 0,
                    start: 1.25,
                    end: 2.5,
                    text: "hello\nworld".to_string(),
                },
                AsrSegment {
                    id: 1,
                    start: 3_661.001,
                    end: 3_662.0,
                    text: " second   cue ".to_string(),
                },
            ],
        };

        let srt = super::format_srt(&transcript);

        assert_eq!(
            srt,
            "1\n00:00:01,250 --> 00:00:02,500\nhello world\n\n2\n01:01:01,001 --> 01:01:02,000\nsecond cue\n"
        );
    }

    #[test]
    fn format_srt_skips_empty_segment_text() {
        let transcript = AsrTranscript {
            text: String::new(),
            language: String::new(),
            raw_output: String::new(),
            segments: vec![AsrSegment {
                id: 0,
                start: 0.0,
                end: 1.0,
                text: "  \n\t ".to_string(),
            }],
        };

        assert_eq!(super::format_srt(&transcript), "");
    }
}
