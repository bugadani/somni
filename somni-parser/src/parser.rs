//! Grammar parser.
//!
//! This module parses the following grammar (minus comments, which are ignored):
//!
//! ```text
//! program -> item* EOF;
//! item -> function | global | extern_fn; // TODO types.
//!
//! extern_fn -> 'extern' 'fn' identifier '(' function_argument ( ',' function_argument )* ','? ')' return_decl? ;
//! global -> 'var' identifier ':' type '=' static_initializer ';' ;
//!
//! static_initializer -> expression ; // The language does not currently support function calls.
//!
//! function -> 'fn' identifier '(' function_argument ( ',' function_argument )* ','? ')' return_decl? body ;
//! function_argument -> identifier ':' '&'? type ;
//! return_decl -> '->' type ;
//! type -> identifier ;
//!
//! body -> '{' statement* '}' ;
//! statement -> 'var' identifier (':' type)? '=' expression ';'
//!            | 'return' expression? ';'
//!            | 'break' expression? ';'
//!            | 'continue' expression? ';'
//!            | 'if' expression body ( 'else' body )?
//!            | 'loop' body
//!            | 'while' expression body
//!            | expression ';'
//!
//! expression -> binary1 ;
//! binary0 -> binary1 ( '=' binary1 )? ;
//! binary1 -> binary2 ( '||' binary2 )* ;
//! binary2 -> binary3 ( '&&' binary3 )* ;
//! binary3 -> binary4 ( ( '<' | '<=' | '>' | '>=' | '==' | '!=' ) binary4 )* ;
//! binary4 -> binary5 ( '|' binary5 )* ;
//! binary5 -> binary6 ( '^' binary6 )* ;
//! binary6 -> binary7 ( '&' binary7 )* ;
//! binary7 -> binary8 ( ( '<<' | '>>' ) binary8 )* ;
//! binary8 -> binary9 ( ( '+' | '-' ) binary9 )* ;
//! binary9 -> unary ( ( '*' | '/' ) unary )* ;
//! unary -> ('!' | '-' | '&' | '*' )* primary | call ;
//! primary -> ( literal | identifier ( '(' call_arguments ')' )? ) | '(' expression ')' ;
//! call_arguments -> expression ( ',' expression )* ','? ;
//! literal -> NUMBER | STRING | 'true' | 'false' ;
//! ```
//!
//! `NUMBER`: Non-negative integers (binary, decimal, hexadecimal) and floats.
//! `STRING`: Double-quoted strings with escape sequences.

use std::{
    fmt::Debug,
    num::{ParseFloatError, ParseIntError},
};

use crate::{
    ast::{
        Body, Break, Continue, Else, EmptyReturn, Expression, ExternalFunction, Function,
        FunctionArgument, GlobalVariable, If, Item, Literal, LiteralValue, Loop, Program,
        ReturnDecl, ReturnWithValue, Statement, TypeHint, VariableDefinition,
    },
    lexer::{Location, Token, TokenKind},
    parser::private::Sealed,
};

mod private {
    pub trait Sealed {}

    impl Sealed for u32 {}
    impl Sealed for u64 {}
    impl Sealed for u128 {}
    impl Sealed for f32 {}
    impl Sealed for f64 {}
}

/// Parse literals into Somni integers.
pub trait IntParser: Sized + Sealed {
    fn parse(str: &str, radix: u32) -> Result<Self, ParseIntError>;
}

impl IntParser for u32 {
    fn parse(str: &str, radix: u32) -> Result<Self, ParseIntError> {
        u32::from_str_radix(str, radix)
    }
}
impl IntParser for u64 {
    fn parse(str: &str, radix: u32) -> Result<Self, ParseIntError> {
        u64::from_str_radix(str, radix)
    }
}
impl IntParser for u128 {
    fn parse(str: &str, radix: u32) -> Result<Self, ParseIntError> {
        u128::from_str_radix(str, radix)
    }
}

/// Parse literals into Somni floats.
pub trait FloatParser: Sized + Sealed {
    fn parse(str: &str) -> Result<Self, ParseFloatError>;
}

impl FloatParser for f32 {
    fn parse(str: &str) -> Result<Self, ParseFloatError> {
        str.parse::<f32>()
    }
}
impl FloatParser for f64 {
    fn parse(str: &str) -> Result<Self, ParseFloatError> {
        str.parse::<f64>()
    }
}

/// Defines the numeric types used in the parser.
pub trait TypeSet: Clone + Debug + Copy {
    type Integer: IntParser + Clone + Copy + PartialEq + Debug;
    type Float: FloatParser + Clone + Copy + PartialEq + Debug;
}

/// Use 64-bit integers and 64-bit floats (default).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DefaultTypeSet;
impl Sealed for DefaultTypeSet {}

impl TypeSet for DefaultTypeSet {
    type Integer = u64;
    type Float = f64;
}

/// Use 32-bit integers and floats.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TypeSet32;
impl Sealed for TypeSet32 {}
impl TypeSet for TypeSet32 {
    type Integer = u32;
    type Float = f32;
}

/// Use 128-bit integers and 64-bit floats.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TypeSet128;
impl Sealed for TypeSet128 {}
impl TypeSet for TypeSet128 {
    type Integer = u128;
    type Float = f64;
}

#[derive(Debug, Clone, PartialEq)]
pub struct ParserError {
    pub location: Location,
    pub error: String,
}

impl std::fmt::Display for ParserError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Parse error: {}", self.error)
    }
}

impl Program {
    fn parse<'s>(stream: &mut TokenStream<'s>) -> Result<Self, ParserError> {
        let mut items = Vec::new();

        while !stream.end() {
            items.push(Item::parse(stream)?);
        }

        Ok(Program { items })
    }
}

impl Item {
    fn parse<'s>(stream: &mut TokenStream<'s>) -> Result<Self, ParserError> {
        if let Some(global_var) = GlobalVariable::try_parse(stream)? {
            return Ok(Item::GlobalVariable(global_var));
        }
        if let Some(function) = ExternalFunction::try_parse(stream)? {
            return Ok(Item::ExternFunction(function));
        }
        if let Some(function) = Function::try_parse(stream)? {
            return Ok(Item::Function(function));
        }

        Err(stream.error("Expected global variable or function definition"))
    }
}

impl GlobalVariable {
    fn try_parse<'s>(stream: &mut TokenStream<'s>) -> Result<Option<Self>, ParserError> {
        let Some(decl_token) = stream.take_match(TokenKind::Identifier, &["var"]) else {
            return Ok(None);
        };

        let identifier = stream.expect_match(TokenKind::Identifier, &[])?;
        let colon = stream.expect_match(TokenKind::Symbol, &[":"])?;
        let type_token = TypeHint::parse(stream)?;
        let equals_token = stream.expect_match(TokenKind::Symbol, &["="])?;
        let initializer = Expression::parse(stream)?;
        let semicolon = stream.expect_match(TokenKind::Symbol, &[";"])?;

        Ok(Some(GlobalVariable {
            decl_token,
            identifier,
            colon,
            type_token,
            equals_token,
            initializer,
            semicolon,
        }))
    }
}

impl ExternalFunction {
    fn try_parse<'s>(stream: &mut TokenStream<'s>) -> Result<Option<Self>, ParserError> {
        let Some(extern_fn_token) = stream.take_match(TokenKind::Identifier, &["extern"]) else {
            return Ok(None);
        };
        let Some(fn_token) = stream.take_match(TokenKind::Identifier, &["fn"]) else {
            return Ok(None);
        };

        let name = stream.expect_match(TokenKind::Identifier, &[])?;
        let opening_paren = stream.expect_match(TokenKind::Symbol, &["("])?;

        let mut arguments = Vec::new();
        while let Some(arg_name) = stream.take_match(TokenKind::Identifier, &[]) {
            let colon = stream.expect_match(TokenKind::Symbol, &[":"])?;
            let reference_token = stream.take_match(TokenKind::Symbol, &["&"]);
            let type_token = TypeHint::parse(stream)?;

            arguments.push(FunctionArgument {
                name: arg_name,
                colon,
                reference_token,
                arg_type: type_token,
            });

            if stream.take_match(TokenKind::Symbol, &[","]).is_none() {
                break;
            }
        }

        let closing_paren = stream.expect_match(TokenKind::Symbol, &[")"])?;

        let return_decl = if let Some(return_token) = stream.take_match(TokenKind::Symbol, &["->"])
        {
            Some(ReturnDecl {
                return_token,
                return_type: TypeHint::parse(stream)?,
            })
        } else {
            None
        };

        let semicolon = stream.expect_match(TokenKind::Symbol, &[";"])?;

        Ok(Some(ExternalFunction {
            extern_fn_token,
            fn_token,
            name,
            opening_paren,
            arguments,
            closing_paren,
            return_decl,
            semicolon,
        }))
    }
}

impl Function {
    fn try_parse<'s>(stream: &mut TokenStream<'s>) -> Result<Option<Self>, ParserError> {
        let Some(fn_token) = stream.take_match(TokenKind::Identifier, &["fn"]) else {
            return Ok(None);
        };

        let name = stream.expect_match(TokenKind::Identifier, &[])?;
        let opening_paren = stream.expect_match(TokenKind::Symbol, &["("])?;

        let mut arguments = Vec::new();
        while let Some(arg_name) = stream.take_match(TokenKind::Identifier, &[]) {
            let colon = stream.expect_match(TokenKind::Symbol, &[":"])?;
            let reference_token = stream.take_match(TokenKind::Symbol, &["&"]);
            let type_token = TypeHint::parse(stream)?;

            arguments.push(FunctionArgument {
                name: arg_name,
                colon,
                reference_token,
                arg_type: type_token,
            });

            if stream.take_match(TokenKind::Symbol, &[","]).is_none() {
                break;
            }
        }

        let closing_paren = stream.expect_match(TokenKind::Symbol, &[")"])?;

        let return_decl = if let Some(return_token) = stream.take_match(TokenKind::Symbol, &["->"])
        {
            Some(ReturnDecl {
                return_token,
                return_type: TypeHint::parse(stream)?,
            })
        } else {
            None
        };

        let body = Body::parse(stream)?;

        Ok(Some(Function {
            fn_token,
            name,
            opening_paren,
            arguments,
            closing_paren,
            return_decl,
            body,
        }))
    }
}

impl Body {
    fn parse<'s>(stream: &mut TokenStream<'s>) -> Result<Self, ParserError> {
        let opening_brace = stream.expect_match(TokenKind::Symbol, &["{"])?;

        let mut body = Vec::new();
        while Statement::matches(stream) {
            body.push(Statement::parse(stream)?);
        }

        let closing_brace = stream.expect_match(TokenKind::Symbol, &["}"])?;

        Ok(Body {
            opening_brace,
            statements: body,
            closing_brace,
        })
    }
}

impl TypeHint {
    fn parse<'s>(stream: &mut TokenStream<'s>) -> Result<Self, ParserError> {
        let type_name = stream.expect_match(TokenKind::Identifier, &[])?;

        Ok(TypeHint { type_name })
    }
}

impl Statement {
    fn matches(stream: &mut TokenStream<'_>) -> bool {
        !stream.end() && stream.peek_match(TokenKind::Symbol, &["}"]).is_none()
    }

    fn parse<'s>(stream: &mut TokenStream<'s>) -> Result<Self, ParserError> {
        if let Some(return_token) = stream.take_match(TokenKind::Identifier, &["return"]) {
            if let Some(semicolon) = stream.take_match(TokenKind::Symbol, &[";"]) {
                return Ok(Statement::EmptyReturn(EmptyReturn {
                    return_token,
                    semicolon,
                }));
            }

            let expr = Expression::parse(stream)?;
            let semicolon = stream.expect_match(TokenKind::Symbol, &[";"])?;
            return Ok(Statement::Return(ReturnWithValue {
                return_token,
                expression: expr,
                semicolon,
            }));
        }

        if let Some(decl_token) = stream.take_match(TokenKind::Identifier, &["var"]) {
            let identifier = stream.expect_match(TokenKind::Identifier, &[])?;

            let type_token = if stream.take_match(TokenKind::Symbol, &[":"]).is_some() {
                Some(TypeHint::parse(stream)?)
            } else {
                None
            };

            let equals_token = stream.expect_match(TokenKind::Symbol, &["="])?;
            let expression = Expression::parse(stream)?;
            let semicolon = stream.expect_match(TokenKind::Symbol, &[";"])?;

            return Ok(Statement::VariableDefinition(VariableDefinition {
                decl_token,
                identifier,
                type_token,
                equals_token,
                initializer: expression,
                semicolon,
            }));
        }

        if let Some(if_token) = stream.take_match(TokenKind::Identifier, &["if"]) {
            let condition = Expression::parse(stream)?;
            let body = Body::parse(stream)?;

            let else_branch =
                if let Some(else_token) = stream.take_match(TokenKind::Identifier, &["else"]) {
                    let else_body = Body::parse(stream)?;

                    Some(Else {
                        else_token,
                        else_body,
                    })
                } else {
                    None
                };

            return Ok(Statement::If(If {
                if_token,
                condition,
                body,
                else_branch,
            }));
        }

        if let Some(loop_token) = stream.take_match(TokenKind::Identifier, &["loop"]) {
            let body = Body::parse(stream)?;
            return Ok(Statement::Loop(Loop { loop_token, body }));
        }

        if let Some(while_token) = stream.take_match(TokenKind::Identifier, &["while"]) {
            // Desugar while into loop { if condition { loop_body; } else { break; } }
            let condition = Expression::parse(stream)?;
            let body = Body::parse(stream)?;
            return Ok(Statement::Loop(Loop {
                loop_token: while_token,
                body: Body {
                    opening_brace: body.opening_brace,
                    closing_brace: body.closing_brace,
                    statements: vec![Statement::If(If {
                        if_token: while_token,
                        condition: condition.clone(),
                        body: body.clone(),
                        else_branch: Some(Else {
                            else_token: while_token,
                            else_body: Body {
                                opening_brace: body.opening_brace,
                                closing_brace: body.closing_brace,
                                statements: vec![Statement::Break(Break {
                                    break_token: while_token,
                                    semicolon: while_token,
                                })],
                            },
                        }),
                    })],
                },
            }));
        }

        if let Some(break_token) = stream.take_match(TokenKind::Identifier, &["break"]) {
            let semicolon = stream.expect_match(TokenKind::Symbol, &[";"])?;
            return Ok(Statement::Break(Break {
                break_token,
                semicolon,
            }));
        }
        if let Some(continue_token) = stream.take_match(TokenKind::Identifier, &["continue"]) {
            let semicolon = stream.expect_match(TokenKind::Symbol, &[";"])?;
            return Ok(Statement::Continue(Continue {
                continue_token,
                semicolon,
            }));
        }

        let expression = Expression::parse(stream)?;
        let semicolon = stream.expect_match(TokenKind::Symbol, &[";"])?;
        Ok(Statement::Expression {
            expression,
            semicolon,
        })
    }
}

impl<T> Literal<T>
where
    T: TypeSet,
{
    fn parse<'s>(stream: &mut TokenStream<'s>) -> Result<Self, ParserError> {
        let token = stream.peek()?;

        let token_source = stream.source(token.location);
        let location = token.location;

        let literal_value = match token.kind {
            TokenKind::BinaryInteger => Self::parse_integer_literal(&token_source[2..], 2)
                .map_err(|_| stream.error("Invalid binary integer literal"))?,
            TokenKind::DecimalInteger => Self::parse_integer_literal(token_source, 10)
                .map_err(|_| stream.error("Invalid integer literal"))?,
            TokenKind::HexInteger => Self::parse_integer_literal(&token_source[2..], 16)
                .map_err(|_| stream.error("Invalid hexadecimal integer literal"))?,
            TokenKind::Float => <T::Float as FloatParser>::parse(token_source)
                .map(LiteralValue::Float)
                .map_err(|_| stream.error("Invalid float literal"))?,
            TokenKind::String => match unescape(&token_source[1..token_source.len() - 1]) {
                Ok(string) => LiteralValue::String(string),
                Err(offset) => {
                    return Err(ParserError {
                        error: "Invalid escape sequence in string literal".to_string(),
                        location: Location {
                            start: token.location.start + offset,
                            end: token.location.start + offset + 1,
                        },
                    });
                }
            },
            TokenKind::Identifier if token_source == "true" => LiteralValue::Boolean(true),
            TokenKind::Identifier if token_source == "false" => LiteralValue::Boolean(false),
            _ => return Err(stream.error("Expected literal (number, string, or boolean)")),
        };

        stream.expect_match(token.kind, &[])?;
        Ok(Self {
            value: literal_value,
            location,
        })
    }

    fn parse_integer_literal(
        token_source: &str,
        radix: u32,
    ) -> Result<LiteralValue<T>, ParseIntError> {
        <T::Integer as IntParser>::parse(token_source, radix).map(LiteralValue::Integer)
    }
}

fn unescape(s: &str) -> Result<String, usize> {
    let mut result = String::new();
    let mut escaped = false;
    for (i, c) in s.char_indices().peekable() {
        if escaped {
            match c {
                'n' => result.push('\n'),
                't' => result.push('\t'),
                '\\' => result.push('\\'),
                '"' => result.push('"'),
                '\'' => result.push('\''),
                _ => return Err(i), // Invalid escape sequence
            }
            escaped = false;
        } else if c == '\\' {
            escaped = true;
        } else {
            result.push(c);
        }
    }

    Ok(result)
}

enum Associativity {
    Left,
    None,
}

impl<T> Expression<T>
where
    T: TypeSet,
{
    fn parse<'s>(stream: &mut TokenStream<'s>) -> Result<Self, ParserError> {
        // We define the binary operators from the lowest precedence to the highest.
        // Each recursive call to `parse_binary` will handle one level of precedence, and pass
        // the rest to the inner calls of `parse_binary`.
        let operators: &[(Associativity, &[&str])] = &[
            (Associativity::None, &["="]),
            (Associativity::Left, &["||"]),
            (Associativity::Left, &["&&"]),
            (Associativity::Left, &["<", "<=", ">", ">=", "==", "!="]),
            (Associativity::Left, &["|"]),
            (Associativity::Left, &["^"]),
            (Associativity::Left, &["&"]),
            (Associativity::Left, &["<<", ">>"]),
            (Associativity::Left, &["+", "-"]),
            (Associativity::Left, &["*", "/"]),
        ];

        Self::parse_binary(stream, operators)
    }

    fn parse_binary<'s>(
        stream: &mut TokenStream<'s>,
        binary_operators: &[(Associativity, &[&str])],
    ) -> Result<Self, ParserError> {
        let Some(((associativity, current), higher)) = binary_operators.split_first() else {
            unreachable!("At least one operator set is expected");
        };

        let mut expr = if higher.is_empty() {
            Self::parse_unary(stream)?
        } else {
            Self::parse_binary(stream, higher)?
        };

        while let Some(operator) = stream.take_match(TokenKind::Symbol, current) {
            let rhs = if higher.is_empty() {
                Self::parse_unary(stream)?
            } else {
                Self::parse_binary(stream, higher)?
            };

            expr = Self::BinaryOperator {
                name: operator,
                operands: Box::new([expr, rhs]),
            };

            if matches!(associativity, Associativity::None) {
                break;
            }
        }

        Ok(expr)
    }

    fn parse_unary<'s>(stream: &mut TokenStream<'s>) -> Result<Self, ParserError> {
        const UNARY_OPERATORS: &[&str] = &["!", "-", "&", "*"];
        if let Some(operator) = stream.take_match(TokenKind::Symbol, UNARY_OPERATORS) {
            let operand = Self::parse_unary(stream)?;
            Ok(Self::UnaryOperator {
                name: operator,
                operand: Box::new(operand),
            })
        } else {
            Self::parse_primary(stream)
        }
    }

    fn parse_primary<'s>(stream: &mut TokenStream<'s>) -> Result<Self, ParserError> {
        let token = stream.peek()?;

        match token.kind {
            TokenKind::Identifier => {
                // true, false?
                if let Ok(literal) = Literal::<T>::parse(stream) {
                    return Ok(Self::Literal { value: literal });
                }

                Self::parse_call(stream)
            }
            TokenKind::Symbol if stream.source(token.location) == "(" => {
                stream.take_match(token.kind, &[]).unwrap();
                let expr = Self::parse(stream)?;
                stream.expect_match(TokenKind::Symbol, &[")"])?;
                Ok(expr)
            }
            TokenKind::HexInteger
            | TokenKind::DecimalInteger
            | TokenKind::BinaryInteger
            | TokenKind::Float
            | TokenKind::String => Literal::<T>::parse(stream).map(|value| Self::Literal { value }),
            _ => Err(stream.error("Expected variable, literal, or '('")),
        }
    }

    fn parse_call<'s>(stream: &mut TokenStream<'s>) -> Result<Self, ParserError> {
        let token = stream.expect_match(TokenKind::Identifier, &[]).unwrap();

        if stream.take_match(TokenKind::Symbol, &["("]).is_none() {
            return Ok(Self::Variable { variable: token });
        };

        let mut arguments = Vec::new();
        while stream.peek_match(TokenKind::Symbol, &[")"]).is_none() {
            let arg = Self::parse(stream)?;
            arguments.push(arg);

            if stream.take_match(TokenKind::Symbol, &[","]).is_none() {
                break;
            }
        }
        stream.expect_match(TokenKind::Symbol, &[")"])?;

        Ok(Self::FunctionCall {
            name: token,
            arguments: arguments.into_boxed_slice(),
        })
    }
}

struct TokenStream<'s> {
    source: &'s str,
    tokens: &'s [Token],
    position: usize,
}

impl<'s> TokenStream<'s> {
    fn new(source: &'s str, tokens: &'s [Token]) -> Self {
        TokenStream {
            source,
            tokens,
            position: 0,
        }
    }

    /// Returns the position of the next non-comment token.
    fn skip_comments(&self) -> usize {
        let mut position = self.position;
        while self.tokens.get(position).is_some()
            && self.tokens[position].kind == TokenKind::Comment
        {
            position += 1;
        }
        position
    }

    fn end(&self) -> bool {
        let position = self.skip_comments();

        position >= self.tokens.len()
    }

    fn peek(&mut self) -> Result<Token, ParserError> {
        let position = self.skip_comments();

        if position < self.tokens.len() {
            Ok(self.tokens[position])
        } else {
            Err(ParserError {
                error: "Unexpected end of input".to_string(),
                location: self.tokens.get(self.position).map_or(
                    Location {
                        start: self.source.len(),
                        end: self.source.len(),
                    },
                    |t| t.location,
                ),
            })
        }
    }

    fn peek_match(&mut self, token_kind: TokenKind, source: &[&str]) -> Option<Token> {
        let token = self.peek().ok()?;

        if token.kind == token_kind
            && (source.is_empty() || source.contains(&self.source(token.location)))
        {
            Some(token)
        } else {
            None
        }
    }

    fn take_match(&mut self, token_kind: TokenKind, source: &[&str]) -> Option<Token> {
        self.position = self.skip_comments();

        if let Some(token) = self.peek_match(token_kind, source) {
            self.position += 1;
            Some(token)
        } else {
            None
        }
    }

    /// Takes the next token if it matches the expected kind and source.
    /// Returns an error if the token does not match.
    ///
    /// If `source` is empty, it only checks the token kind.
    /// If `source` is not empty, it checks if the token's source matches any of the provided strings.
    fn expect_match(
        &mut self,
        token_kind: TokenKind,
        source: &[&str],
    ) -> Result<Token, ParserError> {
        if let Some(token) = self.take_match(token_kind, source) {
            Ok(token)
        } else {
            let token = self.peek().ok();
            let found = if let Some(token) = token {
                if token.kind == token_kind {
                    format!("found '{}'", self.source(token.location))
                } else {
                    format!("found {:?}", token.kind)
                }
            } else {
                "reached end of input".to_string()
            };
            match source {
                [] => Err(self.error(format!("Expected {token_kind:?}, {found}"))),
                [s] => Err(self.error(format!("Expected '{s}', {found}"))),
                _ => Err(self.error(format!("Expected one of {source:?}, {found}"))),
            }
        }
    }

    fn error(&self, message: impl ToString) -> ParserError {
        ParserError {
            error: message.to_string(),
            location: self
                .tokens
                .get(self.position)
                .map(|t| t.location)
                .unwrap_or(Location {
                    start: self.source.len(),
                    end: self.source.len(),
                }),
        }
    }

    fn source(&self, location: Location) -> &'s str {
        location.extract(self.source)
    }
}

pub fn parse<'s>(source: &'s str, tokens: &'s [Token]) -> Result<Program, ParserError> {
    let mut stream = TokenStream::new(source, tokens);

    Program::parse(&mut stream)
}

pub fn parse_expression<'s, T>(
    source: &'s str,
    tokens: &'s [Token],
) -> Result<Expression<T>, ParserError>
where
    T: TypeSet,
{
    let mut stream = TokenStream::new(source, tokens);

    Expression::<T>::parse(&mut stream)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unescape() {
        assert_eq!(unescape(r#"Hello\nWorld\t!"#).unwrap(), "Hello\nWorld\t!");
        assert_eq!(unescape(r#"Hello\\World"#).unwrap(), "Hello\\World");
        assert_eq!(unescape(r#"Hello\zWorld"#), Err(6)); // Invalid escape sequence
    }

    #[test]
    fn test_out_of_range_literal() {
        let source = "0x100000000";
        let tokens = vec![Token {
            kind: TokenKind::HexInteger,
            location: Location {
                start: 0,
                end: source.len(),
            },
        }];

        let result =
            parse_expression::<TypeSet32>(source, &tokens).expect_err("Parsing should fail");
        assert_eq!("Invalid hexadecimal integer literal", result.error);
    }
}
