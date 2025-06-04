//! Grammar parser.
//!
//! This module parses the following grammar (minus comments, which are ignored):
//!
//! ```text
//! program -> item* EOF;
//! item -> function | const | global ; // TODO types.
//!
//! global -> 'var' identifier ':' type '=' const_initializer ';' ;
//! const -> 'const' identifier ':' type '=' const_initializer ';' ;
//!
//! const_initializer -> literal ; // TODO: const eval expressions
//!
//! function -> 'fn' identifier '(' function_argument ( ',' function_argument )* ','? ')' return_decl? body ;
//! function_argument -> identifier ':' '&'? type ;
//! return_decl -> '->' type ;
//! type -> identifier ;
//!
//! body -> '{' statement* '}' ;
//! statement -> 'var' identifier '=' expression ';'
//!            | 'const' identifier '=' expression ';'
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
//! unary -> ('!' | '-' ) unary | call ;
//! primary -> ( literal | identifier ( '(' call_arguments ')' )? ) | '(' expression ')' ;
//! call_arguments -> expression ( ',' expression )* ','? ;
//! literal -> NUMBER | STRING | "true" | "false" ;
//! ```
use std::num::ParseIntError;

use crate::{
    error::CompileError,
    lexer::{Location, Token, TokenKind},
};

#[derive(Debug)]
pub struct Program {
    pub items: Vec<Item>,
}

impl Program {
    fn parse<'s>(stream: &mut TokenStream<'s>) -> Result<Self, CompileError<'s>> {
        let mut items = Vec::new();

        while !stream.end() {
            items.push(Item::parse(stream)?);
        }

        Ok(Program { items })
    }
}

#[derive(Debug)]
pub enum Item {
    Function(Function),
    GlobalVariable(GlobalVariable),
    Constant(Constant),
}

#[derive(Debug)]
pub struct GlobalVariable {
    pub var_token: Token,
    pub identifier: Token,
    pub colon: Token,
    pub type_token: Type,
    pub equals_token: Token,
    pub initializer: Expression,
    pub semicolon: Token,
}
impl GlobalVariable {
    fn try_parse<'s>(stream: &mut TokenStream<'s>) -> Result<Option<Self>, CompileError<'s>> {
        let Some(var_token) = stream.take_match(TokenKind::Identifier, &["var"]) else {
            return Ok(None);
        };

        let identifier = stream.expect_match(TokenKind::Identifier, &[])?;
        let colon = stream.expect_match(TokenKind::Symbol, &[":"])?;
        let type_token = Type::parse(stream)?;
        let equals_token = stream.expect_match(TokenKind::Symbol, &["="])?;
        let initializer = Expression::Literal {
            value: Literal::parse(stream)?,
        };
        let semicolon = stream.expect_match(TokenKind::Symbol, &[";"])?;

        Ok(Some(GlobalVariable {
            var_token,
            identifier,
            colon,
            type_token,
            equals_token,
            initializer,
            semicolon,
        }))
    }

    pub fn location(&self) -> Location {
        let start = self.var_token.location.start;
        let end = self.semicolon.location.end;
        Location { start, end }
    }
}

#[derive(Debug)]
pub struct Constant {
    pub const_token: Token,
    pub identifier: Token,
    pub colon: Token,
    pub type_token: Type,
    pub equals_token: Token,
    pub value: Expression,
    pub semicolon: Token,
}
impl Constant {
    fn try_parse<'s>(stream: &mut TokenStream<'s>) -> Result<Option<Self>, CompileError<'s>> {
        let Some(const_token) = stream.take_match(TokenKind::Identifier, &["const"]) else {
            return Ok(None);
        };

        let identifier = stream.expect_match(TokenKind::Identifier, &[])?;
        let colon = stream.expect_match(TokenKind::Symbol, &[":"])?;
        let type_token = Type::parse(stream)?;
        let equals_token = stream.expect_match(TokenKind::Symbol, &["="])?;
        let value = Expression::Literal {
            value: Literal::parse(stream)?,
        };
        let semicolon = stream.expect_match(TokenKind::Symbol, &[";"])?;

        Ok(Some(Constant {
            const_token,
            identifier,
            colon,
            type_token,
            equals_token,
            value,
            semicolon,
        }))
    }

    pub fn location(&self) -> Location {
        let start = self.const_token.location.start;
        let end = self.semicolon.location.end;
        Location { start, end }
    }
}

#[derive(Debug)]
pub struct ReturnDecl {
    pub return_token: Token,
    pub return_type: Type,
}

#[derive(Debug)]
pub struct Function {
    pub fn_token: Token,
    pub name: Token,
    pub opening_paren: Token,
    pub arguments: Vec<FunctionArgument>,
    pub closing_paren: Token,
    pub return_decl: Option<ReturnDecl>,
    pub body: Body,
}
impl Function {
    fn try_parse<'s>(stream: &mut TokenStream<'s>) -> Result<Option<Self>, CompileError<'s>> {
        let Some(fn_token) = stream.take_match(TokenKind::Identifier, &["fn"]) else {
            return Ok(None);
        };

        let name = stream.expect_match(TokenKind::Identifier, &[])?;
        let opening_paren = stream.expect_match(TokenKind::Symbol, &["("])?;

        let mut arguments = Vec::new();
        while let Some(arg_name) = stream.take_match(TokenKind::Identifier, &[]) {
            let colon = stream.expect_match(TokenKind::Symbol, &[":"])?;
            let reference_token = stream.take_match(TokenKind::Symbol, &["&"]);
            let type_token = Type::parse(stream)?;

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
                return_type: Type::parse(stream)?,
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

#[derive(Debug, Clone)]
pub struct Body {
    pub opening_brace: Token,
    pub statements: Vec<Statement>,
    pub closing_brace: Token,
}

impl Body {
    fn parse<'s>(stream: &mut TokenStream<'s>) -> Result<Self, CompileError<'s>> {
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

#[derive(Debug)]
pub struct FunctionArgument {
    pub name: Token,
    pub colon: Token,
    pub reference_token: Option<Token>,
    pub arg_type: Type,
}

#[derive(Clone, Copy, Debug)]
pub struct Type {
    pub type_name: Token,
}
impl Type {
    fn parse<'s>(stream: &mut TokenStream<'s>) -> Result<Self, CompileError<'s>> {
        let type_name = stream.expect_match(TokenKind::Identifier, &[])?;

        Ok(Type { type_name })
    }
}

impl Item {
    fn parse<'s>(stream: &mut TokenStream<'s>) -> Result<Self, CompileError<'s>> {
        if let Some(constant) = Constant::try_parse(stream)? {
            return Ok(Item::Constant(constant));
        }
        if let Some(global_var) = GlobalVariable::try_parse(stream)? {
            return Ok(Item::GlobalVariable(global_var));
        }
        if let Some(function) = Function::try_parse(stream)? {
            return Ok(Item::Function(function));
        }

        Err(stream.error("Expected constant, global variable or function definition"))
    }
}

#[derive(Debug, Clone)]
pub struct ConstantDefinition {
    pub const_token: Token,
    pub identifier: Token,
    pub equals_token: Token,
    pub initializer: Expression,
    pub semicolon: Token,
}
impl ConstantDefinition {
    pub fn location(&self) -> Location {
        let start = self.const_token.location.start;
        let end = self.semicolon.location.end;
        Location { start, end }
    }
}

#[derive(Debug, Clone)]
pub struct VariableDefinition {
    pub var_token: Token,
    pub identifier: Token,
    pub equals_token: Token,
    pub initializer: Expression,
    pub semicolon: Token,
}
impl VariableDefinition {
    pub fn location(&self) -> Location {
        let start = self.var_token.location.start;
        let end = self.semicolon.location.end;
        Location { start, end }
    }
}

#[derive(Debug, Clone)]
pub struct ReturnWithValue {
    pub return_token: Token,
    pub expression: Expression,
    pub semicolon: Token,
}
impl ReturnWithValue {
    pub fn location(&self) -> Location {
        let start = self.return_token.location.start;
        let end = self.semicolon.location.end;
        Location { start, end }
    }
}

#[derive(Debug, Clone)]
pub struct EmptyReturn {
    pub return_token: Token,
    pub semicolon: Token,
}
impl EmptyReturn {
    pub fn location(&self) -> Location {
        let start = self.return_token.location.start;
        let end = self.semicolon.location.end;
        Location { start, end }
    }
}

#[derive(Debug, Clone)]
pub struct If {
    pub if_token: Token,
    pub condition: Expression,
    pub body: Body,
    pub else_branch: Option<Else>,
}

#[derive(Debug, Clone)]
pub struct Loop {
    pub loop_token: Token,
    pub body: Body,
}

#[derive(Debug, Clone)]
pub struct While {
    pub while_token: Token,
    pub condition: Expression,
    pub body: Body,
}

#[derive(Debug, Clone)]
pub struct Break {
    pub break_token: Token,
    pub semicolon: Token,
}

#[derive(Debug, Clone)]
pub struct Continue {
    pub continue_token: Token,
    pub semicolon: Token,
}

#[derive(Debug, Clone)]
pub enum Statement {
    VariableDefinition(VariableDefinition),
    ConstantDefinition(ConstantDefinition),
    Return(ReturnWithValue),
    EmptyReturn(EmptyReturn),
    If(If),
    Loop(Loop),
    While(While),
    Break(Break),
    Continue(Continue),
    Expression {
        expression: Expression,
        semicolon: Token,
    },
}

#[derive(Debug, Clone)]
pub struct Else {
    pub else_token: Token,
    pub else_body: Body,
}

impl Statement {
    fn matches(stream: &mut TokenStream<'_>) -> bool {
        !stream.end() && stream.peek_match(TokenKind::Symbol, &["}"]).is_none()
    }

    fn parse<'s>(stream: &mut TokenStream<'s>) -> Result<Self, CompileError<'s>> {
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

        if let Some(var_token) = stream.take_match(TokenKind::Identifier, &["var"]) {
            let identifier = stream.expect_match(TokenKind::Identifier, &[])?;
            let equals_token = stream.expect_match(TokenKind::Symbol, &["="])?;
            let expression = Expression::parse(stream)?;
            let semicolon = stream.expect_match(TokenKind::Symbol, &[";"])?;

            return Ok(Statement::VariableDefinition(VariableDefinition {
                var_token,
                identifier,
                equals_token,
                initializer: expression,
                semicolon,
            }));
        }

        if let Some(const_token) = stream.take_match(TokenKind::Identifier, &["const"]) {
            let identifier = stream.expect_match(TokenKind::Identifier, &[])?;
            let equals_token = stream.expect_match(TokenKind::Symbol, &["="])?;
            let expression = Expression::parse(stream)?;
            let semicolon = stream.expect_match(TokenKind::Symbol, &[";"])?;

            return Ok(Statement::ConstantDefinition(ConstantDefinition {
                const_token,
                identifier,
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
            let condition = Expression::parse(stream)?;
            let body = Body::parse(stream)?;
            return Ok(Statement::While(While {
                while_token,
                condition,
                body,
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

#[derive(Debug, Clone)]
pub enum Expression {
    Variable {
        variable: Token,
    },
    Literal {
        value: Literal,
    },
    UnaryOperator {
        name: Token,
        operand: Box<Expression>,
    },
    BinaryOperator {
        name: Token,
        operands: Box<[Expression; 2]>,
    },
    FunctionCall {
        name: Token,
        arguments: Box<[Expression]>,
    },
}

#[derive(Debug, Clone)]
pub struct Literal {
    pub value: LiteralValue,
    pub location: Location,
}

#[derive(Debug, Clone)]
pub enum LiteralValue {
    Integer(u64),
    Float(f64),
    String(String),
    Boolean(bool),
}

impl Literal {
    fn parse<'s>(stream: &mut TokenStream<'s>) -> Result<Self, CompileError<'s>> {
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
            TokenKind::Float => {
                let value = token_source
                    .parse::<f64>()
                    .map_err(|_| stream.error("Invalid float literal"))?;
                LiteralValue::Float(value)
            }
            TokenKind::String => match unescape(&token_source[1..token_source.len() - 1]) {
                Ok(string) => LiteralValue::String(string),
                Err(offset) => {
                    return Err(CompileError {
                        error: "Invalid escape sequence in string literal".to_string(),
                        location: Location {
                            start: token.location.start + offset,
                            end: token.location.start + offset + 1,
                        },
                        source: stream.source,
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

    fn parse_integer_literal<'s>(
        token_source: &str,
        radix: u32,
    ) -> Result<LiteralValue, ParseIntError> {
        u64::from_str_radix(token_source, radix).map(LiteralValue::Integer)
    }
}

fn unescape(s: &str) -> Result<String, usize> {
    let mut result = String::new();
    let mut chars = s.char_indices().peekable();

    let mut escaped = false;
    while let Some((i, c)) = chars.next() {
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

impl Expression {
    fn parse<'s>(stream: &mut TokenStream<'s>) -> Result<Self, CompileError<'s>> {
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
    ) -> Result<Self, CompileError<'s>> {
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

    fn parse_unary<'s>(stream: &mut TokenStream<'s>) -> Result<Self, CompileError<'s>> {
        const UNARY_OPERATORS: &[&str] = &["!", "-"];
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

    fn parse_primary<'s>(stream: &mut TokenStream<'s>) -> Result<Self, CompileError<'s>> {
        let token = stream.peek()?;

        if let Ok(literal) = Literal::parse(stream) {
            return Ok(Self::Literal { value: literal });
        }

        match token.kind {
            TokenKind::Identifier => Self::parse_call(stream),
            TokenKind::Symbol if stream.source(token.location) == "(" => {
                stream.take_match(token.kind.clone(), &[]).unwrap();
                let expr = Self::parse(stream)?;
                stream.expect_match(TokenKind::Symbol, &[")"])?;
                Ok(expr)
            }
            _ => Err(stream.error("Expected variable, literal, or '('")),
        }
    }

    fn parse_call<'s>(stream: &mut TokenStream<'s>) -> Result<Self, CompileError<'s>> {
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

    pub fn location(&self) -> Location {
        match self {
            Expression::Variable { variable } => variable.location,
            Expression::Literal { value } => value.location,
            Expression::FunctionCall { name, arguments } => {
                let mut location = name.location;
                for arg in arguments {
                    location.start = location.start.min(arg.location().start);
                    location.end = location.end.max(arg.location().end);
                }
                location
            }
            Expression::UnaryOperator { name, operand: rhs } => {
                let mut location = name.location;

                location.start = location.start.min(rhs.location().start);
                location.end = location.end.max(rhs.location().end);

                location
            }
            Expression::BinaryOperator {
                name,
                operands: arguments,
            } => {
                let mut location = name.location;
                for arg in arguments.iter() {
                    location.start = location.start.min(arg.location().start);
                    location.end = location.end.max(arg.location().end);
                }
                location
            }
        }
    }

    pub fn as_variable(&self) -> Option<Token> {
        if let Expression::Variable { variable } = self {
            Some(*variable)
        } else {
            None
        }
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

    fn end(&self) -> bool {
        self.position >= self.tokens.len()
    }

    fn peek(&mut self) -> Result<Token, CompileError<'s>> {
        // Skip comments
        let mut position = self.position;
        while self.tokens.get(position).is_some()
            && self.tokens[position].kind == TokenKind::Comment
        {
            position += 1;
        }

        if position < self.tokens.len() {
            Ok(self.tokens[position])
        } else {
            Err(CompileError {
                error: "Unexpected end of input".to_string(),
                location: self.tokens.get(self.position).map_or(
                    Location {
                        start: self.source.len(),
                        end: self.source.len(),
                    },
                    |t| t.location,
                ),
                source: self.source,
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
        // Skip comments
        while self.tokens.get(self.position).is_some()
            && self.tokens[self.position].kind == TokenKind::Comment
        {
            self.position += 1;
        }

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
    ) -> Result<Token, CompileError<'s>> {
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

    fn error(&self, message: impl ToString) -> CompileError<'s> {
        CompileError {
            error: message.to_string(),
            location: self
                .tokens
                .get(self.position)
                .map(|t| t.location)
                .unwrap_or(Location {
                    start: self.source.len(),
                    end: self.source.len(),
                }),
            source: self.source,
        }
    }

    fn source(&self, location: Location) -> &'s str {
        location.extract(self.source)
    }
}

pub fn parse<'s>(source: &'s str, tokens: &'s [Token]) -> Result<Program, CompileError<'s>> {
    let mut stream = TokenStream::new(source, tokens);

    Program::parse(&mut stream)
}

pub fn parse_expression<'s>(
    source: &'s str,
    tokens: &'s [Token],
) -> Result<Expression, CompileError<'s>> {
    let mut stream = TokenStream::new(source, tokens);

    Expression::parse(&mut stream)
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
    fn test_parser() {
        crate::test::run_parser_tests("tests/parser/");
    }
}
