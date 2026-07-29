//! Template errors.
//!
//! All errors carry a [`Location`] into the **original template** source, regardless of
//! whether they originated while scanning/parsing the template or while evaluating the
//! generated Somni program (in the latter case the location is translated back through the
//! [source map](crate::transpile)).

use std::fmt::{Debug, Display};

use somni_expr::error::MarkInSource;
use somni_parser::Location;

/// An error produced while compiling or rendering a template.
///
/// The [`location`](TemplateError::location) always refers to the original template source.
#[derive(Clone, PartialEq, Eq)]
pub struct TemplateError {
    /// A human-readable description of the error.
    pub message: Box<str>,
    /// The location in the original template source where the error occurred.
    pub location: Location,
}

impl TemplateError {
    /// Creates a new error at the given template location.
    pub fn new(message: impl Into<Box<str>>, location: Location) -> Self {
        Self {
            message: message.into(),
            location,
        }
    }

    /// Renders the error as a caret-marked snippet against the given template source.
    pub fn display_with<'s>(&'s self, template: &'s str) -> impl Display + 's {
        MarkedTemplateError {
            error: self,
            template,
        }
    }
}

impl Display for TemplateError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Template error: {}", self.message)
    }
}

impl Debug for TemplateError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "TemplateError {{ message: {:?}, location: {:?} }}",
            self.message, self.location
        )
    }
}

/// A [`TemplateError`] paired with the template source, for caret-marked rendering.
struct MarkedTemplateError<'s> {
    error: &'s TemplateError,
    template: &'s str,
}

impl Display for MarkedTemplateError<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        MarkInSource(
            self.template,
            self.error.location,
            "Template error",
            &self.error.message,
        )
        .fmt(f)
    }
}
