# Included upstream source

| Component | Upstream | License / notice |
| --- | --- | --- |
| OpenEvolve | [algorithmicsuperintelligence/openevolve](https://github.com/algorithmicsuperintelligence/openevolve) | [Apache-2.0](integrations/openevolve-takehome/vendor/openevolve/LICENSE) |
| ShinkaEvolve | [SakanaAI/ShinkaEvolve](https://github.com/SakanaAI/ShinkaEvolve) | [Apache-2.0](integrations/shinka-takehome/vendor/shinkaevolve/LICENSE) |
| Anthropic performance takehome | [anthropics/original_performance_takehome](https://github.com/anthropics/original_performance_takehome) | Original source notices retained; [snapshot notes](integrations/openevolve-takehome/vendor/takehome/Readme.md) |
| Pi coding agent | [earendil-works/pi](https://github.com/earendil-works/pi) | MIT; installed separately from the pinned npm dependency |

The bundled OpenEvolve and ShinkaEvolve trees include experiment-specific
controller, tracing, and prompt changes, so they are not claimed to match an
unmodified upstream release. Their original license files and source notices
are preserved. Shinka's WebUI assets and unrelated upstream examples and
documentation are omitted from this code-only distribution.
