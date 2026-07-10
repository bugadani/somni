//! Configuration of the template surface syntax.
//!
//! A [`Syntax`] describes two independent axes:
//!
//! - **Expression interpolation**: always a delimiter pair, defaulting to `{{` … `}}`.
//! - **Block directives**: either a delimiter pair ([`BlockStyle::Paired`], e.g. `{%` … `%}`)
//!   or a line prefix ([`BlockStyle::Line`], e.g. `#`), where a directive occupies a whole
//!   physical line.
//!
//! Exactly one [`BlockStyle`] is used per compiled template.

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
}

impl Default for Syntax {
    fn default() -> Self {
        Self::brackets()
    }
}

impl Syntax {
    /// Bracket style: `{{ expr }}` interpolation and `{% ... %}` block directives.
    pub fn brackets() -> Self {
        Self {
            expr: ("{{".into(), "}}".into()),
            block: BlockStyle::Paired {
                open: "{%".into(),
                close: "%}".into(),
            },
        }
    }

    /// Line style: `{{ expr }}` interpolation and `#kw ...` line directives.
    pub fn lines() -> Self {
        Self {
            expr: ("{{".into(), "}}".into()),
            block: BlockStyle::Line { prefix: "#".into() },
        }
    }
}
