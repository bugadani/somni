use crate::error::CompileError;
use std::{fmt::Debug, str::CharIndices};

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub struct Location {
    pub start: usize,
    pub end: usize,
}

impl Location {
    pub const fn dummy() -> Self {
        Location { start: 0, end: 0 }
    }
}

impl Location {
    pub fn extract(self, source: &str) -> &str {
        &source[self.start..self.end]
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum TokenKind {
    Symbol,
    Comment,
    DecimalInteger,
    BinaryInteger,
    HexInteger,
    Float,
    Identifier,
    String,
}

#[derive(Clone, Copy, Debug)]
pub struct Token {
    pub kind: TokenKind,
    pub location: Location,
}
impl Token {
    pub fn source(self, source: &str) -> &str {
        self.location.extract(source)
    }
}

fn ident_char(c: char) -> bool {
    c.is_alphanumeric() || ['-', '_'].contains(&c)
}

struct CharProvider<'s> {
    chars: CharIndices<'s>,
}

impl CharProvider<'_> {
    fn next(&mut self) -> Option<char> {
        self.chars.next().map(|(_, c)| c)
    }

    fn consume_n(&mut self, n: usize) {
        for _ in 0..n {
            self.chars.next();
        }
    }

    fn peek(&self) -> Option<char> {
        self.chars.clone().next().map(|(_, c)| c)
    }

    fn peek_n(&self, n: usize) -> Option<&str> {
        let start = self.chars.offset();
        self.chars
            .clone()
            .nth(n)
            .map(|(idx, _)| &self.chars.as_str()[..idx - start])
    }

    fn offset(&self) -> usize {
        self.chars.offset()
    }
}

pub fn tokenize<'s>(source: &'s str) -> impl Iterator<Item = Result<Token, CompileError<'s>>> {
    let mut chars = CharProvider {
        chars: source.char_indices(),
    };

    let mut location = Location { start: 0, end: 0 };

    std::iter::from_fn(move || {
        loop {
            // Start lexing a new token
            location.start = chars.offset();

            // Comments, prefixed numbers, some symbols
            if let Some(doubles) = chars.peek_n(2) {
                match doubles {
                    "//" => {
                        chars.consume_n(2);
                        while let Some(maybe_newline) = chars.peek() {
                            if maybe_newline == '\n' {
                                break;
                            }

                            chars.next();
                        }
                        location.end = chars.offset();

                        return Some(Ok(Token {
                            kind: TokenKind::Comment,
                            location,
                        }));
                    }
                    "==" | "!=" | "<=" | ">=" | "=>" | "&&" | "||" | "**" | "->" | "<<" | ">>" => {
                        chars.consume_n(2);
                        location.end = chars.offset();
                        return Some(Ok(Token {
                            kind: TokenKind::Symbol,
                            location,
                        }));
                    }
                    "0x" | "0b" => {
                        let kind = if doubles == "0x" {
                            TokenKind::HexInteger
                        } else {
                            TokenKind::BinaryInteger
                        };
                        chars.consume_n(2);

                        let mut has_digit = false;
                        while let Some(maybe_digit) = chars.peek() {
                            if !ident_char(maybe_digit) {
                                break;
                            }
                            has_digit = true;
                            chars.next();
                        }
                        location.end = chars.offset();

                        return if has_digit {
                            Some(Ok(Token { kind, location }))
                        } else {
                            Some(Err(CompileError {
                                source,
                                location,
                                error: String::from("invalid numeric literal"),
                            }))
                        };
                    }
                    _ => {}
                }
            }

            // Single characters
            let Some(next) = chars.next() else {
                break;
            };
            location.end = chars.offset();

            match next {
                c if c.is_whitespace() => {
                    // skip whitespace
                }
                '+' | '-' | '*' | '(' | ')' | '{' | '}' | '[' | ']' | ',' | ';' | '/' | ':'
                | '<' | '>' | '&' | '|' | '^' | '=' | '!' => {
                    return Some(Ok(Token {
                        kind: TokenKind::Symbol,
                        location,
                    }));
                }
                c if c.is_numeric() => {
                    // maybe float
                    let mut is_float = false;
                    while let Some(maybe_boundary) = chars.peek() {
                        if !maybe_boundary.is_numeric() {
                            if maybe_boundary == '.' && is_float {
                                break;
                            }

                            if maybe_boundary == '.' {
                                is_float = true;
                            } else {
                                break;
                            }
                        }
                        chars.next();
                    }
                    location.end = chars.offset();
                    let kind = if is_float {
                        TokenKind::Float
                    } else {
                        TokenKind::DecimalInteger
                    };
                    return Some(Ok(Token { kind, location }));
                }
                c if ident_char(c) => {
                    // Identifier
                    while let Some(maybe_boundary) = chars.peek() {
                        if !ident_char(maybe_boundary) {
                            break;
                        }
                        chars.next();
                    }
                    location.end = chars.offset();

                    return Some(Ok(Token {
                        kind: TokenKind::Identifier,
                        location,
                    }));
                }
                '"' => {
                    // Strings
                    let mut escape_start = None;
                    while let Some(c) = chars.peek() {
                        // Terminating double quote mark?
                        if c == '"' && escape_start.is_none() {
                            chars.next();
                            location.end = chars.offset();

                            return Some(Ok(Token {
                                kind: TokenKind::String,
                                location,
                            }));
                        }

                        if escape_start.take().is_none() && c == '\\' {
                            escape_start = Some(chars.offset());
                        }

                        chars.next();
                    }
                    location.end = source.len();

                    return Some(Err(CompileError {
                        source,
                        location,
                        error: String::from("unterminated string"),
                    }));
                }
                _ => {
                    return Some(Err(CompileError {
                        source,
                        location,
                        error: String::from("unexpected character"),
                    }));
                }
            }
        }

        None
    })
}

#[cfg(test)]
mod test {
    use super::*;

    fn test_tokenizer(
        source: &str,
        expectations: &[Result<(&'static str, TokenKind), &'static str>],
    ) {
        let result = tokenize(source).collect::<Vec<Result<_, _>>>();

        for (idx, expectation) in expectations.iter().enumerate() {
            match expectation.clone() {
                Ok((expected, kind)) => {
                    assert_eq!(
                        expected,
                        result[idx].as_ref().unwrap().location.extract(source)
                    );
                    assert_eq!(kind, result[idx].as_ref().unwrap().kind);
                }
                Err(err) => {
                    assert_eq!(err, result[idx].as_ref().unwrap_err().error);
                }
            }
        }
    }

    #[test]
    fn test_lex_numbers() {
        let source = "2 2. 2.3 2.34 23.4 234 0b00 0b10 0b2 0x123 0xf 0xF";
        let expectations = [
            Ok(("2", TokenKind::DecimalInteger)),
            Ok(("2.", TokenKind::Float)),
            Ok(("2.3", TokenKind::Float)),
            Ok(("2.34", TokenKind::Float)),
            Ok(("23.4", TokenKind::Float)),
            Ok(("234", TokenKind::DecimalInteger)),
            Ok(("0b00", TokenKind::BinaryInteger)),
            Ok(("0b10", TokenKind::BinaryInteger)),
            Ok(("0b2", TokenKind::BinaryInteger)),
            Ok(("0x123", TokenKind::HexInteger)),
            Ok(("0xf", TokenKind::HexInteger)),
            Ok(("0xF", TokenKind::HexInteger)),
        ];

        test_tokenizer(source, &expectations);
    }

    #[test]
    fn test_lexer() {
        let source = "   \n // **a\n2 \n // b\nfoo,ar \"string\\\"\" \"\" () -> {}";

        let expectations = [
            Ok(("// **a", TokenKind::Comment)),
            Ok(("2", TokenKind::DecimalInteger)),
            Ok(("// b", TokenKind::Comment)),
            Ok(("foo", TokenKind::Identifier)),
            Ok((",", TokenKind::Symbol)),
            Ok(("ar", TokenKind::Identifier)),
            Ok(("\"string\\\"\"", TokenKind::String)),
            Ok(("\"\"", TokenKind::String)),
            Ok(("(", TokenKind::Symbol)),
            Ok((")", TokenKind::Symbol)),
            Ok(("->", TokenKind::Symbol)),
            Ok(("{", TokenKind::Symbol)),
            Ok(("}", TokenKind::Symbol)),
        ];

        test_tokenizer(source, &expectations);
    }
}
