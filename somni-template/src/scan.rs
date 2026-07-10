//! Low-level scanner: splits raw template text into a flat list of [`Segment`]s.
//!
//! The scanner is intentionally dumb: it only knows how to find interpolation and block
//! delimiters. Interpreting directive keywords and building the nested tree is the
//! [parser](crate::parse)'s job.

use somni_parser::Location;

use crate::{
    error::TemplateError,
    syntax::{BlockStyle, Syntax},
};

/// A flat lexical unit of a template.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Segment {
    /// Literal text, emitted verbatim. Span is into the template.
    Text(Location),
    /// An interpolation (`{{ expr }}`). Span is the trimmed inner expression.
    Interp(Location),
    /// A block directive. Span is the trimmed inner directive text (keyword + args).
    Directive {
        /// The trimmed inner directive text (e.g. `if x > 0`).
        inner: Location,
        /// The full span of the directive including delimiters/line, for diagnostics.
        full: Location,
    },
}

/// Trims ASCII/Unicode whitespace from both ends of a byte range, returning a [`Location`].
fn trimmed(source: &str, start: usize, end: usize) -> Location {
    let slice = &source[start..end];
    let trimmed = slice.trim();
    if trimmed.is_empty() {
        return Location { start, end: start };
    }
    let offset = trimmed.as_ptr() as usize - slice.as_ptr() as usize;
    Location {
        start: start + offset,
        end: start + offset + trimmed.len(),
    }
}

/// Finds the next occurrence of `needle` in `source[from..]`, returning its absolute start.
fn find_from(source: &str, from: usize, needle: &str) -> Option<usize> {
    source[from..].find(needle).map(|i| from + i)
}

/// Scans a template into segments according to the given [`Syntax`].
pub fn scan(source: &str, syntax: &Syntax) -> Result<Vec<Segment>, TemplateError> {
    match &syntax.block {
        BlockStyle::Paired { open, close } => scan_paired(source, &syntax.expr, open, close),
        BlockStyle::Line { prefix } => scan_lines(source, &syntax.expr, prefix),
    }
}

/// Emits a `Text` segment for `start..end` if the range is non-empty.
fn push_text(out: &mut Vec<Segment>, start: usize, end: usize) {
    if end > start {
        out.push(Segment::Text(Location { start, end }));
    }
}

fn scan_paired(
    source: &str,
    expr: &(String, String),
    block_open: &str,
    block_close: &str,
) -> Result<Vec<Segment>, TemplateError> {
    let (expr_open, expr_close) = expr;
    let mut out = Vec::new();
    let mut pos = 0;

    while pos < source.len() {
        let next_expr = find_from(source, pos, expr_open);
        let next_block = find_from(source, pos, block_open);

        let (at, is_expr) = match (next_expr, next_block) {
            (None, None) => {
                push_text(&mut out, pos, source.len());
                break;
            }
            (Some(e), None) => (e, true),
            (None, Some(b)) => (b, false),
            (Some(e), Some(b)) => {
                if e <= b {
                    (e, true)
                } else {
                    (b, false)
                }
            }
        };

        push_text(&mut out, pos, at);

        if is_expr {
            let inner_start = at + expr_open.len();
            let Some(close_at) = find_from(source, inner_start, expr_close) else {
                return Err(TemplateError::new(
                    format!("unterminated interpolation, expected `{expr_close}`"),
                    Location {
                        start: at,
                        end: source.len(),
                    },
                ));
            };
            out.push(Segment::Interp(trimmed(source, inner_start, close_at)));
            pos = close_at + expr_close.len();
        } else {
            let inner_start = at + block_open.len();
            let Some(close_at) = find_from(source, inner_start, block_close) else {
                return Err(TemplateError::new(
                    format!("unterminated block directive, expected `{block_close}`"),
                    Location {
                        start: at,
                        end: source.len(),
                    },
                ));
            };
            out.push(Segment::Directive {
                inner: trimmed(source, inner_start, close_at),
                full: Location {
                    start: at,
                    end: close_at + block_close.len(),
                },
            });
            pos = close_at + block_close.len();
        }
    }

    Ok(out)
}

fn scan_lines(
    source: &str,
    expr: &(String, String),
    prefix: &str,
) -> Result<Vec<Segment>, TemplateError> {
    let (expr_open, expr_close) = expr;
    let mut out = Vec::new();
    // Offset of the start of the current line.
    let mut line_start = 0;

    while line_start < source.len() {
        // Find the end of this physical line (index of '\n', exclusive) and the index of
        // the first byte of the next line.
        let (line_end, next_line) = match find_from(source, line_start, "\n") {
            Some(nl) => (nl, nl + 1),
            None => (source.len(), source.len()),
        };

        // Is this a directive line? Its first non-whitespace must be `prefix`.
        let line = &source[line_start..line_end];
        let leading_ws = line.len() - line.trim_start().len();
        let content_start = line_start + leading_ws;
        let is_directive = source[content_start..line_end].starts_with(prefix);

        if is_directive {
            let inner_start = content_start + prefix.len();
            out.push(Segment::Directive {
                inner: trimmed(source, inner_start, line_end),
                full: Location {
                    start: line_start,
                    end: next_line,
                },
            });
            // The whole physical line (including the trailing newline) is consumed.
            line_start = next_line;
        } else {
            // A text line: emit text, scanning for interpolations. Include the trailing
            // newline as text.
            scan_text_run(
                source, line_start, next_line, expr_open, expr_close, &mut out,
            )?;
            line_start = next_line;
        }
    }

    Ok(out)
}

/// Scans `source[start..end]` (a text region) for interpolations, emitting Text/Interp.
fn scan_text_run(
    source: &str,
    start: usize,
    end: usize,
    expr_open: &str,
    expr_close: &str,
    out: &mut Vec<Segment>,
) -> Result<(), TemplateError> {
    let mut pos = start;
    while pos < end {
        match find_from(source, pos, expr_open) {
            Some(at) if at < end => {
                push_text(out, pos, at);
                let inner_start = at + expr_open.len();
                let Some(close_at) = find_from(source, inner_start, expr_close) else {
                    return Err(TemplateError::new(
                        format!("unterminated interpolation, expected `{expr_close}`"),
                        Location {
                            start: at,
                            end: source.len(),
                        },
                    ));
                };
                out.push(Segment::Interp(trimmed(source, inner_start, close_at)));
                pos = close_at + expr_close.len();
            }
            _ => {
                push_text(out, pos, end);
                break;
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn text<'s>(source: &'s str, seg: &Segment) -> &'s str {
        match seg {
            Segment::Text(l) | Segment::Interp(l) => l.extract(source),
            Segment::Directive { inner, .. } => inner.extract(source),
        }
    }

    #[test]
    fn plain_text_is_single_text_segment() {
        let src = "hello world";
        let segs = scan(src, &Syntax::brackets()).unwrap();
        assert_eq!(segs, vec![Segment::Text(Location { start: 0, end: 11 })]);
    }

    #[test]
    fn brackets_interpolation() {
        let src = "hi {{ name }}!";
        let segs = scan(src, &Syntax::brackets()).unwrap();
        assert_eq!(segs.len(), 3);
        assert_eq!(text(src, &segs[0]), "hi ");
        assert!(matches!(segs[1], Segment::Interp(_)));
        assert_eq!(text(src, &segs[1]), "name");
        assert_eq!(text(src, &segs[2]), "!");
    }

    #[test]
    fn brackets_directive_and_expr_ordering() {
        let src = "{% if x %}a{{ y }}{% endif %}";
        let segs = scan(src, &Syntax::brackets()).unwrap();
        assert!(matches!(segs[0], Segment::Directive { .. }));
        assert_eq!(text(src, &segs[0]), "if x");
        assert_eq!(text(src, &segs[1]), "a");
        assert_eq!(text(src, &segs[2]), "y");
        assert_eq!(text(src, &segs[3]), "endif");
    }

    #[test]
    fn line_directive_swallows_whole_line() {
        let src = "#if x\nhello\n#endif\n";
        let segs = scan(src, &Syntax::lines()).unwrap();
        // if-directive, "hello\n" text, endif-directive
        assert!(matches!(segs[0], Segment::Directive { .. }));
        assert_eq!(text(src, &segs[0]), "if x");
        assert_eq!(text(src, &segs[1]), "hello\n");
        assert!(matches!(segs[2], Segment::Directive { .. }));
        assert_eq!(text(src, &segs[2]), "endif");
        assert_eq!(segs.len(), 3);
    }

    #[test]
    fn line_style_interpolation_inline() {
        let src = "name: {{ user }}\n";
        let segs = scan(src, &Syntax::lines()).unwrap();
        assert_eq!(text(src, &segs[0]), "name: ");
        assert_eq!(text(src, &segs[1]), "user");
        assert_eq!(text(src, &segs[2]), "\n");
    }

    #[test]
    fn unterminated_interpolation_errors() {
        let src = "hi {{ name";
        let err = scan(src, &Syntax::brackets()).unwrap_err();
        assert!(err.message.contains("unterminated interpolation"));
    }
}
