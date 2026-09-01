use std::sync::Arc;

use llama_cpp_2::token::LlamaToken;

use crate::contract::{PromptCacheConfig, TemplateEngine};

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum InputCompatibility {
    RawPrompt,
    Template {
        engine: TemplateEngine,
        source: String,
        enable_thinking: bool,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Compatibility {
    pub(crate) load_epoch: u64,
    pub(crate) input: InputCompatibility,
}

#[derive(Debug)]
struct Entry<S> {
    compatibility: Arc<Compatibility>,
    tokens: Vec<LlamaToken>,
    state: S,
    bytes: usize,
    last_used: u64,
}

#[derive(Debug)]
pub(crate) struct PrefixCache<S> {
    config: PromptCacheConfig,
    entries: Vec<Entry<S>>,
    bytes: usize,
    clock: u64,
    enabled: bool,
}

impl<S> PrefixCache<S> {
    pub(crate) fn new(config: PromptCacheConfig) -> Self {
        let enabled = config.enabled;
        Self {
            config,
            entries: Vec::new(),
            bytes: 0,
            clock: 0,
            enabled,
        }
    }

    pub(crate) fn enabled(&self) -> bool {
        self.enabled
    }

    pub(crate) fn min_prefix_tokens(&self) -> usize {
        self.config.min_prefix_tokens
    }

    pub(crate) fn lookup(
        &mut self,
        compatibility: &Compatibility,
        tokens: &[LlamaToken],
    ) -> Option<(usize, &S)> {
        if !self.enabled {
            return None;
        }
        let index = self
            .entries
            .iter()
            .enumerate()
            .filter(|(_, entry)| {
                entry.compatibility.as_ref() == compatibility
                    && entry.tokens.len() < tokens.len()
                    && tokens.starts_with(&entry.tokens)
            })
            .max_by_key(|(_, entry)| entry.tokens.len())
            .map(|(index, _)| index)?;
        self.clock = self.clock.wrapping_add(1);
        self.entries[index].last_used = self.clock;
        Some((self.entries[index].tokens.len(), &self.entries[index].state))
    }

    pub(crate) fn insert(
        &mut self,
        compatibility: Arc<Compatibility>,
        tokens: Vec<LlamaToken>,
        state: S,
        state_bytes: usize,
    ) {
        if !self.enabled || tokens.len() < self.config.min_prefix_tokens {
            return;
        }
        let token_bytes = tokens
            .len()
            .saturating_mul(std::mem::size_of::<LlamaToken>());
        let bytes = state_bytes.saturating_add(token_bytes);
        if bytes > self.config.max_bytes {
            return;
        }
        if let Some(index) = self
            .entries
            .iter()
            .position(|entry| entry.compatibility == compatibility && entry.tokens == tokens)
        {
            self.bytes = self.bytes.saturating_sub(self.entries[index].bytes);
            self.entries.remove(index);
        }
        self.clock = self.clock.wrapping_add(1);
        self.bytes = self.bytes.saturating_add(bytes);
        self.entries.push(Entry {
            compatibility,
            tokens,
            state,
            bytes,
            last_used: self.clock,
        });
        while self.entries.len() > self.config.max_entries || self.bytes > self.config.max_bytes {
            let index = self
                .entries
                .iter()
                .enumerate()
                .min_by_key(|(_, entry)| entry.last_used)
                .map_or(0, |(index, _)| index);
            self.bytes = self.bytes.saturating_sub(self.entries[index].bytes);
            self.entries.remove(index);
        }
    }

    pub(crate) fn disable(&mut self) {
        self.entries.clear();
        self.bytes = 0;
        self.enabled = false;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn compatibility(epoch: u64) -> Arc<Compatibility> {
        Arc::new(Compatibility {
            load_epoch: epoch,
            input: InputCompatibility::RawPrompt,
        })
    }

    fn config(max_entries: usize, max_bytes: usize) -> PromptCacheConfig {
        PromptCacheConfig {
            enabled: true,
            max_entries,
            max_bytes,
            min_prefix_tokens: 1,
        }
    }

    #[test]
    fn lookup_uses_longest_exactly_compatible_prefix() {
        let mut cache = PrefixCache::new(config(4, 1024));
        cache.insert(compatibility(1), vec![LlamaToken(1)], "short", 8);
        cache.insert(
            compatibility(1),
            vec![LlamaToken(1), LlamaToken(2)],
            "long",
            8,
        );
        cache.insert(compatibility(2), vec![LlamaToken(1); 3], "wrong epoch", 8);

        let hit = cache
            .lookup(
                &compatibility(1),
                &[LlamaToken(1), LlamaToken(2), LlamaToken(3)],
            )
            .unwrap();
        assert_eq!(hit, (2, &"long"));

        let template = Compatibility {
            load_epoch: 1,
            input: InputCompatibility::Template {
                engine: TemplateEngine::Jinja,
                source: "template-a".to_string(),
                enable_thinking: false,
            },
        };
        cache.insert(
            Arc::new(template.clone()),
            vec![LlamaToken(7)],
            "template",
            8,
        );
        for incompatible in [
            Compatibility {
                load_epoch: 1,
                input: InputCompatibility::RawPrompt,
            },
            Compatibility {
                load_epoch: 1,
                input: InputCompatibility::Template {
                    engine: TemplateEngine::LlamaCpp,
                    source: "template-a".to_string(),
                    enable_thinking: false,
                },
            },
            Compatibility {
                load_epoch: 1,
                input: InputCompatibility::Template {
                    engine: TemplateEngine::Jinja,
                    source: "template-b".to_string(),
                    enable_thinking: false,
                },
            },
            Compatibility {
                load_epoch: 1,
                input: InputCompatibility::Template {
                    engine: TemplateEngine::Jinja,
                    source: "template-a".to_string(),
                    enable_thinking: true,
                },
            },
        ] {
            assert!(
                cache
                    .lookup(&incompatible, &[LlamaToken(7), LlamaToken(8)])
                    .is_none()
            );
        }
        assert!(
            cache
                .lookup(&template, &[LlamaToken(7), LlamaToken(8)])
                .is_some()
        );
    }

    #[test]
    fn limits_evict_lru_and_oversized_entries_are_not_retained() {
        let mut cache = PrefixCache::new(config(2, 32));
        cache.insert(compatibility(1), vec![LlamaToken(1)], 1, 4);
        cache.insert(compatibility(1), vec![LlamaToken(2)], 2, 4);
        assert!(
            cache
                .lookup(&compatibility(1), &[LlamaToken(1), LlamaToken(9)])
                .is_some()
        );
        cache.insert(compatibility(1), vec![LlamaToken(3)], 3, 4);
        assert!(
            cache
                .lookup(&compatibility(1), &[LlamaToken(2), LlamaToken(9)])
                .is_none()
        );
        cache.insert(compatibility(1), vec![LlamaToken(4)], 4, 64);
        assert!(
            cache
                .lookup(&compatibility(1), &[LlamaToken(4), LlamaToken(9)])
                .is_none()
        );
    }

    #[test]
    fn disable_clears_entries_permanently() {
        let mut cache = PrefixCache::new(config(2, 64));
        cache.insert(compatibility(1), vec![LlamaToken(1)], 1, 4);
        cache.disable();
        assert!(!cache.enabled());
        assert!(
            cache
                .lookup(&compatibility(1), &[LlamaToken(1), LlamaToken(2)])
                .is_none()
        );
    }
}
