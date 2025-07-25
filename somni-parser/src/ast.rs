use crate::{lexer::Token, parser::TypeSet, Location};

#[derive(Debug)]
pub struct Program<T>
where
    T: TypeSet,
{
    pub items: Vec<Item<T>>,
}

#[derive(Debug)]
pub enum Item<T>
where
    T: TypeSet,
{
    Function(Function<T>),
    ExternFunction(ExternalFunction),
    GlobalVariable(GlobalVariable<T>),
}

#[derive(Debug)]
pub struct GlobalVariable<T>
where
    T: TypeSet,
{
    pub decl_token: Token,
    pub identifier: Token,
    pub colon: Token,
    pub type_token: TypeHint,
    pub equals_token: Token,
    pub initializer: Expression<T>,
    pub semicolon: Token,
}
impl<T> GlobalVariable<T>
where
    T: TypeSet,
{
    pub fn location(&self) -> Location {
        let start = self.decl_token.location.start;
        let end = self.semicolon.location.end;
        Location { start, end }
    }
}

#[derive(Debug)]
pub struct ReturnDecl {
    pub return_token: Token,
    pub return_type: TypeHint,
}

#[derive(Debug)]
pub struct ExternalFunction {
    pub extern_fn_token: Token,
    pub fn_token: Token,
    pub name: Token,
    pub opening_paren: Token,
    pub arguments: Vec<FunctionArgument>,
    pub closing_paren: Token,
    pub return_decl: Option<ReturnDecl>,
    pub semicolon: Token,
}

#[derive(Debug)]
pub struct Function<T>
where
    T: TypeSet,
{
    pub fn_token: Token,
    pub name: Token,
    pub opening_paren: Token,
    pub arguments: Vec<FunctionArgument>,
    pub closing_paren: Token,
    pub return_decl: Option<ReturnDecl>,
    pub body: Body<T>,
}

#[derive(Debug)]
pub struct Body<T>
where
    T: TypeSet,
{
    pub opening_brace: Token,
    pub statements: Vec<Statement<T>>,
    pub closing_brace: Token,
}

impl<T> Clone for Body<T>
where
    T: TypeSet,
{
    fn clone(&self) -> Self {
        Self {
            opening_brace: self.opening_brace.clone(),
            statements: self.statements.clone(),
            closing_brace: self.closing_brace.clone(),
        }
    }
}

#[derive(Debug)]
pub struct FunctionArgument {
    pub name: Token,
    pub colon: Token,
    pub reference_token: Option<Token>,
    pub arg_type: TypeHint,
}

#[derive(Clone, Copy, Debug)]
pub struct TypeHint {
    pub type_name: Token,
}

#[derive(Debug)]
pub struct VariableDefinition<T>
where
    T: TypeSet,
{
    pub decl_token: Token,
    pub identifier: Token,
    pub type_token: Option<TypeHint>,
    pub equals_token: Token,
    pub initializer: Expression<T>,
    pub semicolon: Token,
}

impl<T> Clone for VariableDefinition<T>
where
    T: TypeSet,
{
    fn clone(&self) -> Self {
        Self {
            decl_token: self.decl_token.clone(),
            identifier: self.identifier.clone(),
            type_token: self.type_token.clone(),
            equals_token: self.equals_token.clone(),
            initializer: self.initializer.clone(),
            semicolon: self.semicolon.clone(),
        }
    }
}

impl<T> VariableDefinition<T>
where
    T: TypeSet,
{
    pub fn location(&self) -> Location {
        let start = self.decl_token.location.start;
        let end = self.semicolon.location.end;
        Location { start, end }
    }
}

#[derive(Debug)]
pub struct ReturnWithValue<T>
where
    T: TypeSet,
{
    pub return_token: Token,
    pub expression: Expression<T>,
    pub semicolon: Token,
}

impl<T> Clone for ReturnWithValue<T>
where
    T: TypeSet,
{
    fn clone(&self) -> Self {
        Self {
            return_token: self.return_token.clone(),
            expression: self.expression.clone(),
            semicolon: self.semicolon.clone(),
        }
    }
}

impl<T> ReturnWithValue<T>
where
    T: TypeSet,
{
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

#[derive(Debug)]
pub struct If<T>
where
    T: TypeSet,
{
    pub if_token: Token,
    pub condition: Expression<T>,
    pub body: Body<T>,
    pub else_branch: Option<Else<T>>,
}

impl<T> Clone for If<T>
where
    T: TypeSet,
{
    fn clone(&self) -> Self {
        Self {
            if_token: self.if_token.clone(),
            condition: self.condition.clone(),
            body: self.body.clone(),
            else_branch: self.else_branch.clone(),
        }
    }
}

#[derive(Debug)]
pub struct Loop<T>
where
    T: TypeSet,
{
    pub loop_token: Token,
    pub body: Body<T>,
}

impl<T> Clone for Loop<T>
where
    T: TypeSet,
{
    fn clone(&self) -> Self {
        Self {
            loop_token: self.loop_token.clone(),
            body: self.body.clone(),
        }
    }
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

#[derive(Debug)]
pub enum Statement<T>
where
    T: TypeSet,
{
    VariableDefinition(VariableDefinition<T>),
    Return(ReturnWithValue<T>),
    EmptyReturn(EmptyReturn),
    If(If<T>),
    Loop(Loop<T>),
    Break(Break),
    Continue(Continue),
    Expression {
        expression: Expression<T>,
        semicolon: Token,
    },
}

impl<T> Clone for Statement<T>
where
    T: TypeSet,
{
    fn clone(&self) -> Self {
        match self {
            Self::VariableDefinition(arg0) => Self::VariableDefinition(arg0.clone()),
            Self::Return(arg0) => Self::Return(arg0.clone()),
            Self::EmptyReturn(arg0) => Self::EmptyReturn(arg0.clone()),
            Self::If(arg0) => Self::If(arg0.clone()),
            Self::Loop(arg0) => Self::Loop(arg0.clone()),
            Self::Break(arg0) => Self::Break(arg0.clone()),
            Self::Continue(arg0) => Self::Continue(arg0.clone()),
            Self::Expression {
                expression,
                semicolon,
            } => Self::Expression {
                expression: expression.clone(),
                semicolon: semicolon.clone(),
            },
        }
    }
}

#[derive(Debug)]
pub struct Else<T>
where
    T: TypeSet,
{
    pub else_token: Token,
    pub else_body: Body<T>,
}

impl<T> Clone for Else<T>
where
    T: TypeSet,
{
    fn clone(&self) -> Self {
        Self {
            else_token: self.else_token.clone(),
            else_body: self.else_body.clone(),
        }
    }
}

#[derive(Debug)]
pub enum Expression<T>
where
    T: TypeSet,
{
    Variable {
        variable: Token,
    },
    Literal {
        value: Literal<T>,
    },
    UnaryOperator {
        name: Token,
        operand: Box<Self>,
    },
    BinaryOperator {
        name: Token,
        operands: Box<[Self; 2]>,
    },
    FunctionCall {
        name: Token,
        arguments: Box<[Self]>,
    },
}

impl<T> Clone for Expression<T>
where
    T: TypeSet,
{
    fn clone(&self) -> Self {
        match self {
            Self::Variable { variable } => Self::Variable {
                variable: variable.clone(),
            },
            Self::Literal { value } => Self::Literal {
                value: value.clone(),
            },
            Self::UnaryOperator { name, operand } => Self::UnaryOperator {
                name: name.clone(),
                operand: operand.clone(),
            },
            Self::BinaryOperator { name, operands } => Self::BinaryOperator {
                name: name.clone(),
                operands: operands.clone(),
            },
            Self::FunctionCall { name, arguments } => Self::FunctionCall {
                name: name.clone(),
                arguments: arguments.clone(),
            },
        }
    }
}
impl<T> Expression<T>
where
    T: TypeSet,
{
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

#[derive(Debug)]
pub struct Literal<T>
where
    T: TypeSet,
{
    pub value: LiteralValue<T>,
    pub location: Location,
}

impl<T> Clone for Literal<T>
where
    T: TypeSet,
{
    fn clone(&self) -> Self {
        Self {
            value: self.value.clone(),
            location: self.location.clone(),
        }
    }
}

#[derive(Debug)]
pub enum LiteralValue<T>
where
    T: TypeSet,
{
    Integer(T::Integer),
    Float(T::Float),
    String(String),
    Boolean(bool),
}

impl<T> Clone for LiteralValue<T>
where
    T: TypeSet,
{
    fn clone(&self) -> Self {
        match self {
            Self::Integer(arg0) => Self::Integer(arg0.clone()),
            Self::Float(arg0) => Self::Float(arg0.clone()),
            Self::String(arg0) => Self::String(arg0.clone()),
            Self::Boolean(arg0) => Self::Boolean(arg0.clone()),
        }
    }
}
