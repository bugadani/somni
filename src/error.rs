use std::fmt::Debug;

use somni_lexer::{LexerError, Location};
use somni_parser::ParserError;

pub trait ErrorWithLocation {
    fn location(&self) -> Location;
}

impl ErrorWithLocation for LexerError {
    fn location(&self) -> Location {
        self.location
    }
}

impl ErrorWithLocation for ParserError {
    fn location(&self) -> Location {
        self.location
    }
}

impl ErrorWithLocation for CompileError<'_> {
    fn location(&self) -> Location {
        self.location
    }
}

#[derive(Clone)]
pub struct CompileError<'s> {
    pub source: &'s str,
    pub location: Location,
    pub error: String,
}

impl Debug for CompileError<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(
            &MarkInSource(self.source, self.location, "Compile error", &self.error),
            f,
        )
    }
}

pub struct MarkInSource<'s>(pub &'s str, pub Location, pub &'s str, pub &'s str);
impl std::fmt::Display for MarkInSource<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let start_loc = location(self.0, self.1.start);
        let end_loc = location(self.0, self.1.end);
        mark_in_source(f, self.0, start_loc, end_loc, self.2, self.3)
    }
}

pub fn mark_in_source(
    f: &mut std::fmt::Formatter<'_>,
    source: &str,
    start_loc: (usize, usize),
    end_loc: (usize, usize),
    message: &str,
    hint: &str,
) -> Result<(), std::fmt::Error> {
    let (start_line, start_col) = start_loc;
    let (end_line, end_col) = end_loc;

    let line = source.lines().nth(start_line - 1).unwrap();
    let line_no_chars = start_line.ilog10() as usize + 1;
    f.write_fmt(format_args!(
        "{bold}{message}{reset}\n{:>width$}{blue}--->{reset} at line {start_line} column {start_col}\n",
        "",
        width = line_no_chars,
        blue = "\x1b[1;36m",
        bold = "\x1b[1m",
        reset = "\x1b[0m",
    ))?;
    f.write_fmt(format_args!(
        "{:>width$} {blue}|{reset}\n",
        "",
        width = line_no_chars,
        blue = "\x1b[1;36m",
        reset = "\x1b[0m",
    ))?;
    f.write_fmt(format_args!(
        "{blue}{start_line} |{reset} {line}\n",
        blue = "\x1b[1;36m",
        reset = "\x1b[0m",
    ))?;

    let arrow_width = if start_line == end_line {
        end_col - start_col
    } else {
        // TODO highlight the whole range
        1
    };

    f.write_fmt(format_args!(
        "{placeholder:>line_no_chars$} {blue}|{reset} {placeholder:>space_width$}{red}{arrow:^<arrow_width$} {hint}{reset}",
        placeholder = "",
        line_no_chars = line_no_chars,
        space_width = start_col - 1,
        arrow = "^",
        arrow_width = arrow_width,
        blue = "\x1b[1;36m",
        red = "\x1b[1;31m",
        reset = "\x1b[0m",
    ))
}

pub fn location(source: &str, offset: usize) -> (usize, usize) {
    let before_offset = &source[..offset];
    let mut line = 1;
    let mut col = 1;

    for char in before_offset.chars() {
        if char == '\n' {
            line += 1;
            col = 1;
        } else if char == '\r' {
            col = 1;
        } else {
            col += char.len_utf8();
        }
    }

    (line, col)
}

#[cfg(test)]
mod tests {
    use crate::test::strip_ansi;

    use super::*;

    #[test]
    fn test_source_location() {
        let test_cases = [
            ((1, 1), 0, "asdf"),
            ((1, 4), 3, "asdf"),
            ((2, 1), 5, "asdf\n"),
            ((3, 4), 17, "asdf\njlk hsdf\nfoo"),
            ((1, 5), 4, "1 + 2 * foo(3)"),
        ];

        for (expected, offset, source) in test_cases {
            assert_eq!(location(source, offset), expected);
        }
    }

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
