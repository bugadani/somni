//! Configuration of the template surface syntax.
//!
//! A [`Syntax`] describes three independent axes:
//!
//! - **Expression interpolation**: always a delimiter pair, defaulting to `{{` … `}}`.
//! - **Block directives**: either a delimiter pair ([`BlockStyle::Paired`], e.g. `{%` … `%}`)
//!   or a line prefix ([`BlockStyle::Line`], e.g. `#`), where a directive occupies a whole
//!   physical line.
//! - **Text line prefix** ([`Syntax::text_prefix`]): an optional line prefix that is stripped;
//!   the remainder of the line is emitted as ordinary template text. Useful when a template is
//!   also a valid program in another language — conditional output can sit behind a comment
//!   marker so it is inert if the file is compiled/run directly.
//!
//! Exactly one [`BlockStyle`] is used per compiled template.
//!
//! Templates may also begin with a `---`-fenced **frontmatter** block that overrides these
//! settings (see [`split_frontmatter`] and [`Syntax::with_frontmatter`]). When present,
//! frontmatter keys take precedence over the [`Syntax`] passed to [`crate::Template::compile`].

use somni_parser::Location;

use crate::error::TemplateError;

/// How block directives (`if`/`else`/`for`/…) are delimited.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BlockStyle {
    /// Directives are wrapped in a delimiter pair, e.g. `{% if x %}`.
    Paired {
        /// Opening delimiter, e.g. `{%`.
        open: String,
        /// Closing delimiter, e.g. `%}`.
        close: String,
    },
    /// Directives occupy a whole line, introduced by a prefix, e.g. `#if x`.
    ///
    /// The entire physical line (leading whitespace through the trailing newline) is
    /// consumed and produces no output.
    Line {
        /// The line prefix that introduces a directive, e.g. `#`.
        prefix: String,
    },
}

/// The configurable surface syntax of a template.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Syntax {
    /// The interpolation delimiter pair, e.g. (`{{`, `}}`).
    pub expr: (String, String),
    /// How block directives are delimited.
    pub block: BlockStyle,
    /// Optional line prefix that is stripped; content after it is emitted as text.
    ///
    /// A physical line whose first non-whitespace characters equal this prefix has the leading
    /// whitespace and the prefix removed. The rest of the line (including a trailing newline)
    /// is ordinary template text and may contain interpolations.
    ///
    /// When both this and a [`BlockStyle::Line`] prefix could match the same line, the longer
    /// prefix wins. They must not be identical (see [`Syntax::validate`]).
    pub text_prefix: Option<String>,
}

impl Default for Syntax {
    fn default() -> Self {
        Self::brackets()
    }
}

/// Frontmatter keys that may appear at most once.
const FRONTMATTER_KEYS: &[&str] = &["expr", "block", "text_prefix"];

impl Syntax {
    /// Bracket style: `{{ expr }}` interpolation and `{% ... %}` block directives.
    pub fn brackets() -> Self {
        Self {
            expr: ("{{".into(), "}}".into()),
            block: BlockStyle::Paired {
                open: "{%".into(),
                close: "%}".into(),
            },
            text_prefix: None,
        }
    }

    /// Line style: `{{ expr }}` interpolation and `#kw ...` line directives.
    pub fn lines() -> Self {
        Self {
            expr: ("{{".into(), "}}".into()),
            block: BlockStyle::Line { prefix: "#".into() },
            text_prefix: None,
        }
    }

    /// Checks that this syntax configuration is internally consistent.
    ///
    /// Currently rejects an empty `text_prefix` and a `text_prefix` identical to the block
    /// line prefix.
    pub fn validate(&self) -> Result<(), TemplateError> {
        if let Some(tp) = &self.text_prefix {
            if tp.is_empty() {
                return Err(TemplateError::new(
                    "`text_prefix` must not be empty",
                    Location { start: 0, end: 0 },
                ));
            }
            if let BlockStyle::Line { prefix } = &self.block {
                if prefix == tp {
                    return Err(TemplateError::new(
                        "`text_prefix` must differ from the block line prefix",
                        Location { start: 0, end: 0 },
                    ));
                }
            }
        }
        Ok(())
    }

    /// Applies frontmatter key/value overrides onto a copy of this syntax.
    ///
    /// Recognized keys (all optional; each may appear at most once):
    ///
    /// - `expr: <open> <close>` — interpolation delimiters.
    /// - `block: paired <open> <close>` — paired block directives.
    /// - `block: line <prefix>` — line directives.
    /// - `text_prefix: <prefix>` — strip this line prefix; remainder is verbatim text.
    ///
    /// Blank lines and lines whose first non-whitespace character is `#` are ignored
    /// (comments). Keys present in `front` replace the corresponding fields of `self`;
    /// omitted keys are left unchanged. Locations in returned errors are relative to `front`.
    pub fn with_frontmatter(&self, front: &str) -> Result<Self, TemplateError> {
        let mut syntax = self.clone();
        let mut offset = 0usize;
        let mut seen = [false; FRONTMATTER_KEYS.len()];

        for line in front.split_inclusive('\n') {
            let line_start = offset;
            offset += line.len();

            let trimmed = line.trim();
            // `trim` also strips a trailing `\r` left by CRLF after splitting on `\n`.
            if trimmed.is_empty() || trimmed.starts_with('#') {
                continue;
            }

            let (key, value) = trimmed.split_once(':').ok_or_else(|| {
                TemplateError::new(
                    format!("frontmatter line is not `key: value`: {trimmed:?}"),
                    line_span(front, line_start, line.len()),
                )
            })?;

            let key = key.trim();
            let Some(idx) = FRONTMATTER_KEYS.iter().position(|&k| k == key) else {
                return Err(TemplateError::new(
                    format!("unknown frontmatter key: {key:?}"),
                    line_span(front, line_start, line.len()),
                ));
            };
            if seen[idx] {
                return Err(TemplateError::new(
                    format!("`{key}` specified more than once"),
                    line_span(front, line_start, line.len()),
                ));
            }
            seen[idx] = true;

            let mut parts = value.split_whitespace();
            match key {
                "expr" => {
                    let open = parts.next().ok_or_else(|| {
                        TemplateError::new(
                            "`expr` needs an opening delimiter",
                            line_span(front, line_start, line.len()),
                        )
                    })?;
                    let close = parts.next().ok_or_else(|| {
                        TemplateError::new(
                            "`expr` needs a closing delimiter",
                            line_span(front, line_start, line.len()),
                        )
                    })?;
                    syntax.expr = (open.to_string(), close.to_string());
                }
                "block" => match parts.next() {
                    Some("line") => {
                        let prefix = parts.next().ok_or_else(|| {
                            TemplateError::new(
                                "`block: line` needs a prefix",
                                line_span(front, line_start, line.len()),
                            )
                        })?;
                        if syntax.text_prefix.as_deref() == Some(prefix) {
                            return Err(TemplateError::new(
                                "`text_prefix` must differ from the block line prefix",
                                line_span(front, line_start, line.len()),
                            ));
                        }
                        syntax.block = BlockStyle::Line {
                            prefix: prefix.to_string(),
                        };
                    }
                    Some("paired") => {
                        let open = parts.next().ok_or_else(|| {
                            TemplateError::new(
                                "`block: paired` needs an opening delimiter",
                                line_span(front, line_start, line.len()),
                            )
                        })?;
                        let close = parts.next().ok_or_else(|| {
                            TemplateError::new(
                                "`block: paired` needs a closing delimiter",
                                line_span(front, line_start, line.len()),
                            )
                        })?;
                        syntax.block = BlockStyle::Paired {
                            open: open.to_string(),
                            close: close.to_string(),
                        };
                    }
                    other => {
                        return Err(TemplateError::new(
                            format!("unknown block style: {other:?}"),
                            line_span(front, line_start, line.len()),
                        ));
                    }
                },
                "text_prefix" => {
                    let prefix = parts.next().ok_or_else(|| {
                        TemplateError::new(
                            "`text_prefix` needs a prefix",
                            line_span(front, line_start, line.len()),
                        )
                    })?;
                    if let BlockStyle::Line {
                        prefix: block_prefix,
                    } = &syntax.block
                    {
                        if block_prefix == prefix {
                            return Err(TemplateError::new(
                                "`text_prefix` must differ from the block line prefix",
                                line_span(front, line_start, line.len()),
                            ));
                        }
                    }
                    syntax.text_prefix = Some(prefix.to_string());
                }
                _ => unreachable!("key matched FRONTMATTER_KEYS"),
            }
        }

        syntax.validate()?;
        Ok(syntax)
    }
}

/// If `source` begins with a `---`-fenced frontmatter block, returns `(front, body)`.
///
/// The opening fence must be at the start of `source` (`---` followed by a newline). The
/// closing fence is a line that is exactly `---`. Without both fences, returns `None` and
/// the whole string is treated as template body.
///
/// Both LF and CRLF newlines are accepted.
pub fn split_frontmatter(source: &str) -> Option<(&str, &str)> {
    let rest = source
        .strip_prefix("---\r\n")
        .or_else(|| source.strip_prefix("---\n"))?;

    // Prefer the LF form; `\r\n---\r\n` still contains `\n---\r\n`.
    // Trim a trailing `\r` so CRLF front matter does not leave a dangling CR in `front`.
    if let Some(end) = rest.find("\n---\n") {
        let front = rest[..end].strip_suffix('\r').unwrap_or(&rest[..end]);
        let body = &rest[end + "\n---\n".len()..];
        return Some((front, body));
    }
    if let Some(end) = rest.find("\n---\r\n") {
        let front = rest[..end].strip_suffix('\r').unwrap_or(&rest[..end]);
        let body = &rest[end + "\n---\r\n".len()..];
        return Some((front, body));
    }
    None
}

/// Resolves the effective [`Syntax`] for a template source.
///
/// When frontmatter is present, its settings overlay `base` (frontmatter wins). Returns the
/// effective syntax, the template body with frontmatter stripped, and the byte offset of the
/// body within the original `source`. Without frontmatter, returns a clone of `base`, the
/// original `source`, and offset `0`.
pub fn resolve_syntax<'a>(
    source: &'a str,
    base: &Syntax,
) -> Result<(Syntax, &'a str, usize), TemplateError> {
    let Some((front, body)) = split_frontmatter(source) else {
        base.validate()?;
        return Ok((base.clone(), source, 0));
    };

    // `front` begins immediately after the opening `---\n` / `---\r\n` fence.
    let front_start = source
        .strip_prefix("---\r\n")
        .or_else(|| source.strip_prefix("---\n"))
        .map(|rest| source.len() - rest.len())
        .expect("split_frontmatter found a fence");

    let syntax = base.with_frontmatter(front).map_err(|err| {
        TemplateError::new(
            err.message,
            Location {
                start: err.location.start + front_start,
                end: err.location.end + front_start,
            },
        )
    })?;
    // `body` is always a suffix of `source`.
    let body_offset = source.len() - body.len();
    Ok((syntax, body, body_offset))
}

fn line_span(front: &str, line_start: usize, line_len: usize) -> Location {
    let end = (line_start + line_len).min(front.len());
    // Prefer pointing at the non-newline contents of the line.
    let content_end = front.as_bytes()[line_start..end]
        .iter()
        .rposition(|&b| b != b'\n' && b != b'\r')
        .map(|i| line_start + i + 1)
        .unwrap_or(line_start);
    Location {
        start: line_start,
        end: content_end.max(line_start),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_no_frontmatter() {
        assert_eq!(split_frontmatter("hello"), None);
        assert_eq!(split_frontmatter("---\nno close"), None);
    }

    #[test]
    fn split_lf_and_crlf() {
        assert_eq!(
            split_frontmatter("---\nblock: line #\n---\nbody"),
            Some(("block: line #", "body"))
        );
        assert_eq!(
            split_frontmatter("---\r\nblock: line #\r\n---\r\nbody"),
            Some(("block: line #", "body"))
        );
    }

    #[test]
    fn frontmatter_overlays_base() {
        let base = Syntax {
            expr: ("[[".into(), "]]".into()),
            block: BlockStyle::Paired {
                open: "{%".into(),
                close: "%}".into(),
            },
            text_prefix: None,
        };
        let merged = base
            .with_frontmatter("block: line //\ntext_prefix: //>")
            .unwrap();
        assert_eq!(merged.expr, ("[[".into(), "]]".into()));
        assert_eq!(
            merged.block,
            BlockStyle::Line {
                prefix: "//".into()
            }
        );
        assert_eq!(merged.text_prefix.as_deref(), Some("//>"));
    }

    #[test]
    fn frontmatter_rejects_duplicate_keys() {
        for (front, duplicate_line) in [
            ("block: line #\nblock: paired {% %}", "block: paired {% %}"),
            ("expr: {{ }}\nexpr: [[ ]]", "expr: [[ ]]"),
        ] {
            let err = Syntax::brackets().with_frontmatter(front).unwrap_err();
            assert!(
                err.message.contains("more than once"),
                "unexpected message for {front:?}: {}",
                err.message
            );
            assert_eq!(err.location.extract(front), duplicate_line);
        }
    }

    #[test]
    fn frontmatter_ignores_hash_comments() {
        let merged = Syntax::brackets()
            .with_frontmatter(
                "# use C-style block comments\n\
                 block: paired /* */\n\
                 # keep default expr\n",
            )
            .unwrap();
        assert_eq!(merged.expr, ("{{".into(), "}}".into()));
        assert_eq!(
            merged.block,
            BlockStyle::Paired {
                open: "/*".into(),
                close: "*/".into(),
            }
        );
    }

    #[test]
    fn resolve_prefers_frontmatter_over_base() {
        let base = Syntax::lines();
        let source = "---\nblock: paired /* */\n---\n/* if x */y/* endif */";
        let (syntax, body, body_offset) = resolve_syntax(source, &base).unwrap();
        assert_eq!(body, "/* if x */y/* endif */");
        assert_eq!(&source[body_offset..], body);
        assert_eq!(
            syntax.block,
            BlockStyle::Paired {
                open: "/*".into(),
                close: "*/".into(),
            }
        );
        // Unmentioned keys keep the Rust-provided value.
        assert_eq!(syntax.expr, base.expr);
    }
}
