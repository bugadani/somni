use somni_lexer::{Location, Token};

#[derive(Debug)]
pub struct Program {
    pub items: Vec<Item>,
}

#[derive(Debug)]
pub enum Item {
    Function(Function),
    GlobalVariable(GlobalVariable),
}

#[derive(Debug)]
pub struct GlobalVariable {
    pub decl_token: Token,
    pub identifier: Token,
    pub colon: Token,
    pub type_token: TypeHint,
    pub equals_token: Token,
    pub initializer: Expression,
    pub semicolon: Token,
}
impl GlobalVariable {
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
pub struct Function {
    pub fn_token: Token,
    pub name: Token,
    pub opening_paren: Token,
    pub arguments: Vec<FunctionArgument>,
    pub closing_paren: Token,
    pub return_decl: Option<ReturnDecl>,
    pub body: Body,
}

#[derive(Debug, Clone)]
pub struct Body {
    pub opening_brace: Token,
    pub statements: Vec<Statement>,
    pub closing_brace: Token,
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

#[derive(Debug, Clone)]
pub struct VariableDefinition {
    pub decl_token: Token,
    pub identifier: Token,
    pub type_token: Option<TypeHint>,
    pub equals_token: Token,
    pub initializer: Expression,
    pub semicolon: Token,
}

impl VariableDefinition {
    pub fn location(&self) -> Location {
        let start = self.decl_token.location.start;
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
    Return(ReturnWithValue),
    EmptyReturn(EmptyReturn),
    If(If),
    Loop(Loop),
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
impl Expression {
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
