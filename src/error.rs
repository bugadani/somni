use std::fmt::Debug;

use somni_expr::error::{ErrorWithLocation, MarkInSource};
use somni_parser::lexer::Location;

#[derive(Clone)]
pub struct CompileError<'s> {
    pub source: &'s str,
    pub location: Location,
    pub error: String,
}

impl<'s> CompileError<'s> {
    pub fn new(source: &'s str, error: impl ErrorWithLocation) -> Self {
        Self {
            source,
            location: error.location(),
            error: error.to_string(),
        }
    }
}

impl Debug for CompileError<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(
            &MarkInSource(self.source, self.location, "Compile error", &self.error),
            f,
        )
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::test::strip_ansi;

    #[test]
    fn test_error() {
        let source = "1 + 2 * foo(3)";
        pretty_assertions::assert_eq!(
            r#"Compile error
 ---> at line 1 column 5
  |
1 | 1 + 2 * foo(3)
  |     ^ Syntax error"#,
            strip_ansi(format!(
                "{:?}",
                CompileError {
                    source,
                    location: Location { start: 4, end: 5 },
                    error: "Syntax error".to_string(),
                }
            ))
        );

        pretty_assertions::assert_eq!(
            r#"Compile error
 ---> at line 1 column 9
  |
1 | 1 + 2 * foo(3)
  |         ^^^^^^ Syntax error"#,
            strip_ansi(format!(
                "{:?}",
                CompileError {
                    source,
                    location: Location { start: 8, end: 14 },
                    error: "Syntax error".to_string(),
                }
            ))
        );
    }
}
