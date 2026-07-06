#[derive(Debug, PartialEq, Eq)]
pub(super) struct CaptionTextUpdate<'a> {
    pub(super) segment_final: Option<&'a str>,
    pub(super) partial: &'a str,
}

pub(super) struct CaptionTextSplitter {
    target_segment_ms: u32,
    committed_prefix: String,
    candidate_text: Option<String>,
    stable_count: u8,
}

struct CaptionBoundaryCandidate<'a> {
    text: &'a str,
    has_following_text: bool,
}

impl CaptionTextSplitter {
    pub(super) fn new(target_segment_ms: u32) -> Self {
        Self {
            target_segment_ms,
            committed_prefix: String::new(),
            candidate_text: None,
            stable_count: 0,
        }
    }

    pub(super) fn observe_partial<'a>(
        &mut self,
        text: &'a str,
        duration_ms: u64,
    ) -> CaptionTextUpdate<'a> {
        let uncommitted_text = self.uncommitted_text(text);
        let Some(candidate) = strong_punctuation_candidate_text(uncommitted_text) else {
            self.candidate_text = None;
            self.stable_count = 0;
            return CaptionTextUpdate {
                segment_final: None,
                partial: uncommitted_text,
            };
        };

        if self.candidate_text.as_deref() == Some(candidate.text) {
            self.stable_count = self.stable_count.saturating_add(1);
        } else {
            self.candidate_text = Some(candidate.text.to_string());
            self.stable_count = 1;
        }

        let stable = self.stable_count >= 2;
        let reached_target = duration_ms >= u64::from(self.target_segment_ms);
        let should_commit = stable && (candidate.has_following_text || reached_target);
        if !should_commit {
            return CaptionTextUpdate {
                segment_final: None,
                partial: uncommitted_text,
            };
        }

        let final_text = candidate.text;
        let partial = &uncommitted_text[final_text.len()..];
        self.committed_prefix.push_str(final_text);
        self.candidate_text = None;
        self.stable_count = 0;

        CaptionTextUpdate {
            segment_final: Some(final_text),
            partial,
        }
    }

    pub(super) fn flush<'a>(&mut self, text: &'a str) -> Option<&'a str> {
        let uncommitted_text = self.uncommitted_text(text).trim();
        self.committed_prefix.clear();
        self.candidate_text = None;
        self.stable_count = 0;
        (!uncommitted_text.is_empty()).then_some(uncommitted_text)
    }

    fn uncommitted_text<'a>(&self, text: &'a str) -> &'a str {
        text.trim()
            .strip_prefix(&self.committed_prefix)
            .unwrap_or_else(|| text.trim())
    }
}

fn strong_punctuation_candidate_text(text: &str) -> Option<CaptionBoundaryCandidate<'_>> {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return None;
    }
    let mut terminal_candidate_end = None;
    let mut can_extend_candidate = false;

    for (index, character) in trimmed.char_indices() {
        let character_end = index + character.len_utf8();
        if is_caption_strong_punctuation(character) {
            terminal_candidate_end = Some(character_end);
            can_extend_candidate = true;
            continue;
        }
        if is_caption_closing_character(character) {
            if can_extend_candidate {
                terminal_candidate_end = Some(character_end);
            }
            continue;
        }
        if let Some(candidate_end) = terminal_candidate_end {
            return Some(CaptionBoundaryCandidate {
                text: &trimmed[..candidate_end],
                has_following_text: true,
            });
        }
        can_extend_candidate = false;
    }

    terminal_candidate_end.map(|end| CaptionBoundaryCandidate {
        text: &trimmed[..end],
        has_following_text: false,
    })
}

fn is_caption_strong_punctuation(character: char) -> bool {
    matches!(character, '。' | '！' | '？' | '；' | '.' | '!' | '?' | ';')
}

fn is_caption_closing_character(character: char) -> bool {
    matches!(
        character,
        '"' | '\'' | '”' | '’' | ')' | ']' | '}' | '）' | '】' | '》' | '」' | '』'
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keeps_partial_until_candidate_stabilizes() {
        let mut splitter = CaptionTextSplitter::new(12_000);

        assert_eq!(
            splitter.observe_partial("中俄北京条约。在这个", 13_000),
            CaptionTextUpdate {
                segment_final: None,
                partial: "中俄北京条约。在这个"
            }
        );
        assert_eq!(
            splitter.observe_partial("中俄北京条约。在这个条约里", 13_500),
            CaptionTextUpdate {
                segment_final: Some("中俄北京条约。"),
                partial: "在这个条约里"
            }
        );
    }

    #[test]
    fn emits_next_partial_from_uncommitted_suffix_after_split() {
        let mut splitter = CaptionTextSplitter::new(12_000);

        splitter.observe_partial("中俄北京条约。在这个", 13_000);
        splitter.observe_partial("中俄北京条约。在这个条约里", 13_500);

        assert_eq!(
            splitter.observe_partial("中俄北京条约。在这个条约里，乌苏里江", 14_000),
            CaptionTextUpdate {
                segment_final: None,
                partial: "在这个条约里，乌苏里江"
            }
        );
    }

    #[test]
    fn does_not_commit_revised_terminal_punctuation() {
        let mut splitter = CaptionTextSplitter::new(12_000);

        assert_eq!(
            splitter.observe_partial("一八六零年。", 8_000),
            CaptionTextUpdate {
                segment_final: None,
                partial: "一八六零年。"
            }
        );
        assert_eq!(
            splitter.observe_partial("一八六零年，沙俄", 8_500),
            CaptionTextUpdate {
                segment_final: None,
                partial: "一八六零年，沙俄"
            }
        );
    }

    #[test]
    fn terminal_punctuation_requires_target_duration_and_stability() {
        let mut splitter = CaptionTextSplitter::new(12_000);

        assert_eq!(
            splitter.observe_partial("短句。", 3_000).segment_final,
            None
        );
        assert_eq!(
            splitter.observe_partial("短句。", 12_000).segment_final,
            Some("短句。")
        );
    }

    #[test]
    fn flush_emits_remaining_uncommitted_text() {
        let mut splitter = CaptionTextSplitter::new(12_000);

        splitter.observe_partial("第一句。第二", 13_000);
        splitter.observe_partial("第一句。第二句", 13_500);

        assert_eq!(splitter.flush("第一句。第二句结束"), Some("第二句结束"));
    }

    #[test]
    fn accepts_terminal_closing_quote() {
        let mut splitter = CaptionTextSplitter::new(12_000);

        assert_eq!(
            splitter
                .observe_partial("他说：你好。\"", 12_000)
                .segment_final,
            None
        );
        assert_eq!(
            splitter
                .observe_partial("他说：你好。\"", 12_100)
                .segment_final,
            Some("他说：你好。\"")
        );
    }
}
