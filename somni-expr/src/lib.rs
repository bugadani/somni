pub mod error;
pub mod string_interner;

use std::fmt::{Debug, Display};

use indexmap::IndexMap;
use somni_parser::{
    ast::{self, Expression},
    lexer::{self, Location},
    parser,
};

use crate::{
    error::MarkInSource,
    string_interner::{StringIndex, StringInterner},
};

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TypedValue {
    Void,
    Int(u64),
    SignedInt(i64),
    Float(f64),
    Bool(bool),
    String(StringIndex),
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub enum OperatorError {
    NotSupported,
    TypeError,
    RuntimeError,
}

impl Display for OperatorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let message = match self {
            OperatorError::NotSupported => "Operation not supported",
            OperatorError::TypeError => "Type error",
            OperatorError::RuntimeError => "Runtime error",
        };

        f.write_str(message)
    }
}

macro_rules! dispatch_binary_to_bool {
    ($method:ident) => {
        pub(crate) fn $method(
            lhs: TypedValue,
            rhs: TypedValue,
        ) -> Result<TypedValue, OperatorError> {
            let result = match (lhs, rhs) {
                (TypedValue::Bool(value), TypedValue::Bool(other)) => {
                    ValueType::$method(value, other)
                }
                (TypedValue::Int(value), TypedValue::Int(other)) => {
                    ValueType::$method(value, other)
                }
                (TypedValue::SignedInt(value), TypedValue::SignedInt(other)) => {
                    ValueType::$method(value, other)
                }
                (TypedValue::Float(value), TypedValue::Float(other)) => {
                    ValueType::$method(value, other)
                }
                (TypedValue::String(value), TypedValue::String(other)) => {
                    ValueType::$method(value, other)
                }
                _ => return Err(OperatorError::TypeError),
            };

            result.map(TypedValue::Bool)
        }
    };
}

macro_rules! dispatch_binary_to_value_type {
    ($method:ident) => {
        pub(crate) fn $method(
            lhs: TypedValue,
            rhs: TypedValue,
        ) -> Result<TypedValue, OperatorError> {
            match (lhs, rhs) {
                (TypedValue::Bool(value), TypedValue::Bool(other)) => {
                    Ok(TypedValue::from(ValueType::$method(value, other)?))
                }
                (TypedValue::Int(value), TypedValue::Int(other)) => {
                    Ok(TypedValue::from(ValueType::$method(value, other)?))
                }
                (TypedValue::SignedInt(value), TypedValue::SignedInt(other)) => {
                    Ok(TypedValue::from(ValueType::$method(value, other)?))
                }
                (TypedValue::Float(value), TypedValue::Float(other)) => {
                    Ok(TypedValue::from(ValueType::$method(value, other)?))
                }
                (TypedValue::String(value), TypedValue::String(other)) => {
                    Ok(TypedValue::from(ValueType::$method(value, other)?))
                }
                _ => return Err(OperatorError::TypeError),
            }
        }
    };
}

impl TypedValue {
    pub fn type_of(&self) -> Type {
        match self {
            TypedValue::Void => Type::Void,
            TypedValue::Int(_) => Type::Int,
            TypedValue::SignedInt(_) => Type::SignedInt,
            TypedValue::Float(_) => Type::Float,
            TypedValue::Bool(_) => Type::Bool,
            TypedValue::String(_) => Type::String,
        }
    }

    dispatch_binary_to_bool!(equals);
    dispatch_binary_to_bool!(less_than);
    dispatch_binary_to_bool!(less_than_or_equal);
    dispatch_binary_to_bool!(not_equals);
    dispatch_binary_to_value_type!(bitwise_or);
    dispatch_binary_to_value_type!(bitwise_xor);
    dispatch_binary_to_value_type!(bitwise_and);
    dispatch_binary_to_value_type!(shift_left);
    dispatch_binary_to_value_type!(shift_right);
    dispatch_binary_to_value_type!(add);
    dispatch_binary_to_value_type!(subtract);
    dispatch_binary_to_value_type!(multiply);
    dispatch_binary_to_value_type!(divide);

    pub fn write(&self, to: &mut [u8]) {
        match self {
            TypedValue::Void => ().write(to),
            TypedValue::Int(value) => value.write(to),
            TypedValue::SignedInt(value) => value.write(to),
            TypedValue::Float(value) => value.write(to),
            TypedValue::Bool(value) => value.write(to),
            TypedValue::String(value) => value.write(to),
        }
    }

    pub fn from_typed_bytes(ty: Type, value: &[u8]) -> TypedValue {
        match ty {
            Type::Void => Self::Void,
            Type::Int => Self::Int(<_>::from_bytes(value)),
            Type::SignedInt => Self::SignedInt(<_>::from_bytes(value)),
            Type::Float => Self::Float(<_>::from_bytes(value)),
            Type::Bool => Self::Bool(<_>::from_bytes(value)),
            Type::String => Self::String(<_>::from_bytes(value)),
        }
    }
}

impl From<u64> for TypedValue {
    fn from(value: u64) -> Self {
        TypedValue::Int(value)
    }
}
impl TryFrom<TypedValue> for u64 {
    type Error = ();

    fn try_from(value: TypedValue) -> Result<Self, Self::Error> {
        if let TypedValue::Int(value) = value {
            Ok(value)
        } else {
            Err(())
        }
    }
}

impl From<i64> for TypedValue {
    fn from(value: i64) -> Self {
        TypedValue::SignedInt(value)
    }
}
impl TryFrom<TypedValue> for i64 {
    type Error = ();

    fn try_from(value: TypedValue) -> Result<Self, Self::Error> {
        if let TypedValue::SignedInt(value) = value {
            Ok(value)
        } else {
            Err(())
        }
    }
}

impl From<f64> for TypedValue {
    fn from(value: f64) -> Self {
        TypedValue::Float(value)
    }
}
impl TryFrom<TypedValue> for f64 {
    type Error = ();

    fn try_from(value: TypedValue) -> Result<Self, Self::Error> {
        if let TypedValue::Float(value) = value {
            Ok(value)
        } else {
            Err(())
        }
    }
}

impl From<bool> for TypedValue {
    fn from(value: bool) -> Self {
        TypedValue::Bool(value)
    }
}
impl TryFrom<TypedValue> for bool {
    type Error = ();

    fn try_from(value: TypedValue) -> Result<Self, Self::Error> {
        if let TypedValue::Bool(value) = value {
            Ok(value)
        } else {
            Err(())
        }
    }
}

impl From<StringIndex> for TypedValue {
    fn from(value: StringIndex) -> Self {
        TypedValue::String(value)
    }
}
impl TryFrom<TypedValue> for StringIndex {
    type Error = ();

    fn try_from(value: TypedValue) -> Result<Self, Self::Error> {
        if let TypedValue::String(str) = value {
            Ok(str)
        } else {
            Err(())
        }
    }
}

impl From<()> for TypedValue {
    fn from(_: ()) -> Self {
        TypedValue::Void
    }
}

impl TryFrom<TypedValue> for () {
    type Error = ();

    fn try_from(value: TypedValue) -> Result<Self, Self::Error> {
        if let TypedValue::Void = value {
            Ok(())
        } else {
            Err(())
        }
    }
}

pub trait ExprContext {
    fn intern_string(&mut self, s: &str) -> StringIndex;
    fn try_load_variable(&self, variable: &str) -> Option<TypedValue>;
    fn address_of(&self, variable: &str) -> TypedValue;
    fn call_function(
        &mut self,
        function_name: &str,
        args: &[TypedValue],
    ) -> Result<TypedValue, FunctionCallError>;
}

pub struct ExpressionVisitor<'a, C> {
    pub context: &'a mut C,
    pub source: &'a str,
}

#[derive(Clone, Debug, PartialEq)]
pub struct EvalError {
    pub message: Box<str>,
    pub location: Location,
}

impl Display for EvalError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Evaluation error: {}", self.message)
    }
}

#[derive(Clone, PartialEq)]
pub struct ExpressionError<'s> {
    error: EvalError,
    source: &'s str,
}

impl Debug for ExpressionError<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let marked = MarkInSource(
            self.source,
            self.error.location,
            "Eval error",
            &self.error.message,
        );
        marked.fmt(f)
    }
}

impl<'a, C> ExpressionVisitor<'a, C>
where
    C: ExprContext,
{
    fn visit_variable(&mut self, variable: &lexer::Token) -> Result<TypedValue, EvalError> {
        let name = variable.source(self.source);
        self.context.try_load_variable(name).ok_or(EvalError {
            message: format!("Variable {name} was not found").into_boxed_str(),
            location: variable.location,
        })
    }

    pub fn visit_expression(&mut self, expression: &Expression) -> Result<TypedValue, EvalError> {
        let result = match expression {
            Expression::Variable { variable } => self.visit_variable(variable)?,
            Expression::Literal { value } => match &value.value {
                ast::LiteralValue::Integer(value) => TypedValue::Int(*value),
                ast::LiteralValue::Float(value) => TypedValue::Float(*value),
                ast::LiteralValue::String(value) => {
                    TypedValue::String(self.context.intern_string(value))
                }
                ast::LiteralValue::Boolean(value) => TypedValue::Bool(*value),
            },
            Expression::UnaryOperator { name, operand } => match name.source(self.source) {
                "!" => match self.visit_expression(operand)? {
                    TypedValue::Bool(b) => TypedValue::Bool(!b),
                    value => {
                        return Err(EvalError {
                            message: format!("Expected boolean, found {}", value.type_of())
                                .into_boxed_str(),
                            location: operand.location(),
                        });
                    }
                },
                "-" => match self.visit_expression(operand)? {
                    // TODO handle overflow errors
                    TypedValue::SignedInt(i) => TypedValue::SignedInt(-i),
                    TypedValue::Int(i) => TypedValue::SignedInt(-(i as i64)),
                    TypedValue::Float(f) => TypedValue::Float(-f),
                    value => {
                        return Err(EvalError {
                            message: format!("Cannot negate {}", value.type_of()).into_boxed_str(),
                            location: operand.location(),
                        });
                    }
                },
                "&" => match operand.as_variable() {
                    Some(variable) => {
                        let name = variable.source(self.source);
                        self.context.address_of(name)
                    }
                    None => {
                        return Err(EvalError {
                            message: String::from("Cannot take address of non-variable expression")
                                .into_boxed_str(),
                            location: operand.location(),
                        });
                    }
                },
                "*" => {
                    return Err(EvalError {
                        message: String::from("Dereference not supported").into_boxed_str(),
                        location: operand.location(),
                    });
                }
                _ => {
                    return Err(EvalError {
                        message: format!("Unknown unary operator: {}", name.source(self.source))
                            .into_boxed_str(),
                        location: expression.location(),
                    });
                }
            },
            Expression::BinaryOperator { name, operands } => {
                let short_circuiting = ["&&", "||"];
                let operator = name.source(self.source);

                if short_circuiting.contains(&operator) {
                    let lhs = self.visit_expression(&operands[0])?;
                    return match operator {
                        "&&" if lhs == TypedValue::Bool(false) => Ok(TypedValue::Bool(false)),
                        "||" if lhs == TypedValue::Bool(true) => Ok(TypedValue::Bool(true)),
                        _ => self.visit_expression(&operands[1]),
                    };
                }

                let lhs = self.visit_expression(&operands[0])?;
                let rhs = self.visit_expression(&operands[1])?;
                let result = match name.source(self.source) {
                    "+" => TypedValue::add(lhs, rhs),
                    "-" => TypedValue::subtract(lhs, rhs),
                    "*" => TypedValue::multiply(lhs, rhs),
                    "/" => TypedValue::divide(lhs, rhs),
                    "<" => TypedValue::less_than(lhs, rhs),
                    ">" => TypedValue::less_than(rhs, lhs),
                    "<=" => TypedValue::less_than_or_equal(lhs, rhs),
                    ">=" => TypedValue::less_than_or_equal(rhs, lhs),
                    "==" => TypedValue::equals(lhs, rhs),
                    "!=" => TypedValue::not_equals(lhs, rhs),
                    "|" => TypedValue::bitwise_or(lhs, rhs),
                    "^" => TypedValue::bitwise_xor(lhs, rhs),
                    "&" => TypedValue::bitwise_and(lhs, rhs),
                    "<<" => TypedValue::shift_left(lhs, rhs),
                    ">>" => TypedValue::shift_right(lhs, rhs),

                    other => {
                        return Err(EvalError {
                            message: format!("Unknown binary operator: {other}").into_boxed_str(),
                            location: expression.location(),
                        });
                    }
                };

                match result {
                    Ok(r) => r,
                    Err(error) => {
                        return Err(EvalError {
                            message: format!("Failed to evaluate expression: {error}")
                                .into_boxed_str(),
                            location: expression.location(),
                        });
                    }
                }
            }
            Expression::FunctionCall { name, arguments } => {
                let function_name = name.source(self.source);
                let mut args = Vec::new();
                for arg in arguments {
                    args.push(self.visit_expression(arg)?);
                }

                match self.context.call_function(function_name, &args) {
                    Ok(result) => result,
                    Err(FunctionCallError::IncorrectArgumentCount { expected }) => {
                        return Err(EvalError {
                            message: format!(
                                "{function_name} takes {expected} arguments, {} given",
                                args.len()
                            )
                            .into_boxed_str(),
                            location: expression.location(),
                        });
                    }
                    Err(FunctionCallError::IncorrectArgumentType { idx, expected }) => {
                        return Err(EvalError {
                            message: format!(
                                "{function_name} expects argument {idx} to be {expected}, got {}",
                                args[idx].type_of()
                            )
                            .into_boxed_str(),
                            location: arguments[idx].location(),
                        });
                    }
                    Err(FunctionCallError::FunctionNotFound) => {
                        return Err(EvalError {
                            message: format!("Function {function_name} is not found")
                                .into_boxed_str(),
                            location: expression.location(),
                        });
                    }
                    Err(FunctionCallError::Other(error)) => {
                        return Err(EvalError {
                            message: format!("Failed to call {function_name}: {error}")
                                .into_boxed_str(),
                            location: expression.location(),
                        });
                    }
                }
            }
        };

        Ok(result)
    }
}
pub trait ValueType: Sized + Copy + TryFrom<TypedValue, Error = ()> + Into<TypedValue> {
    const BYTES: usize;
    const TYPE: Type;

    fn write(&self, to: &mut [u8]);
    fn from_bytes(bytes: &[u8]) -> Self;

    fn equals(a: Self, b: Self) -> Result<bool, OperatorError>;
    fn less_than(a: Self, b: Self) -> Result<bool, OperatorError>;

    fn less_than_or_equal(a: Self, b: Self) -> Result<bool, OperatorError> {
        let less = Self::less_than(a, b)?;
        Ok(less || Self::equals(a, b)?)
    }
    fn not_equals(a: Self, b: Self) -> Result<bool, OperatorError> {
        let equals = Self::equals(a, b)?;
        Ok(!equals)
    }
    fn bitwise_or(_a: Self, _b: Self) -> Result<Self, OperatorError> {
        Err(OperatorError::NotSupported)
    }
    fn bitwise_xor(_a: Self, _b: Self) -> Result<Self, OperatorError> {
        Err(OperatorError::NotSupported)
    }
    fn bitwise_and(_a: Self, _b: Self) -> Result<Self, OperatorError> {
        Err(OperatorError::NotSupported)
    }
    fn shift_left(_a: Self, _b: Self) -> Result<Self, OperatorError> {
        Err(OperatorError::NotSupported)
    }
    fn shift_right(_a: Self, _b: Self) -> Result<Self, OperatorError> {
        Err(OperatorError::NotSupported)
    }
    fn add(_a: Self, _b: Self) -> Result<Self, OperatorError> {
        Err(OperatorError::NotSupported)
    }
    fn subtract(_a: Self, _b: Self) -> Result<Self, OperatorError> {
        Err(OperatorError::NotSupported)
    }
    fn multiply(_a: Self, _b: Self) -> Result<Self, OperatorError> {
        Err(OperatorError::NotSupported)
    }
    fn divide(_a: Self, _b: Self) -> Result<Self, OperatorError> {
        Err(OperatorError::NotSupported)
    }
    fn not(_a: Self) -> Result<Self, OperatorError> {
        Err(OperatorError::NotSupported)
    }
    fn negate(_a: Self) -> Result<Self, OperatorError> {
        Err(OperatorError::NotSupported)
    }
}

impl ValueType for () {
    const BYTES: usize = 0;
    const TYPE: Type = Type::Void;

    fn write(&self, _to: &mut [u8]) {}

    fn from_bytes(_: &[u8]) -> Self {}

    fn less_than(_: Self, _: Self) -> Result<bool, OperatorError> {
        Err(OperatorError::NotSupported)
    }

    fn equals(_: Self, _: Self) -> Result<bool, OperatorError> {
        Err(OperatorError::NotSupported)
    }
}

impl ValueType for u64 {
    const BYTES: usize = 8;
    const TYPE: Type = Type::Int;

    fn write(&self, to: &mut [u8]) {
        to.copy_from_slice(&self.to_le_bytes());
    }

    fn from_bytes(bytes: &[u8]) -> Self {
        u64::from_le_bytes(bytes.try_into().unwrap())
    }

    fn less_than(a: Self, b: Self) -> Result<bool, OperatorError> {
        Ok(a < b)
    }
    fn equals(a: Self, b: Self) -> Result<bool, OperatorError> {
        Ok(a == b)
    }
    fn add(a: Self, b: Self) -> Result<Self, OperatorError> {
        a.checked_add(b).ok_or(OperatorError::RuntimeError)
    }
    fn subtract(a: Self, b: Self) -> Result<Self, OperatorError> {
        a.checked_sub(b).ok_or(OperatorError::RuntimeError)
    }
    fn multiply(a: Self, b: Self) -> Result<Self, OperatorError> {
        a.checked_mul(b).ok_or(OperatorError::RuntimeError)
    }
    fn divide(a: Self, b: Self) -> Result<Self, OperatorError> {
        if b == 0 {
            Err(OperatorError::RuntimeError)
        } else {
            Ok(a / b)
        }
    }
    fn bitwise_or(a: Self, b: Self) -> Result<Self, OperatorError> {
        Ok(a | b)
    }
    fn bitwise_xor(a: Self, b: Self) -> Result<Self, OperatorError> {
        Ok(a ^ b)
    }
    fn bitwise_and(a: Self, b: Self) -> Result<Self, OperatorError> {
        Ok(a & b)
    }
    fn shift_left(a: Self, b: Self) -> Result<Self, OperatorError> {
        if b < 64 {
            Ok(a << b)
        } else {
            Err(OperatorError::RuntimeError)
        }
    }
    fn shift_right(a: Self, b: Self) -> Result<Self, OperatorError> {
        if b < 64 {
            Ok(a >> b)
        } else {
            Err(OperatorError::RuntimeError)
        }
    }
}

impl ValueType for i64 {
    const BYTES: usize = 8;
    const TYPE: Type = Type::SignedInt;

    fn write(&self, to: &mut [u8]) {
        to.copy_from_slice(&self.to_le_bytes());
    }

    fn from_bytes(bytes: &[u8]) -> Self {
        i64::from_le_bytes(bytes.try_into().unwrap())
    }

    fn less_than(a: Self, b: Self) -> Result<bool, OperatorError> {
        Ok(a < b)
    }
    fn equals(a: Self, b: Self) -> Result<bool, OperatorError> {
        Ok(a == b)
    }
    fn add(a: Self, b: Self) -> Result<Self, OperatorError> {
        a.checked_add(b).ok_or(OperatorError::RuntimeError)
    }
    fn subtract(a: Self, b: Self) -> Result<Self, OperatorError> {
        a.checked_sub(b).ok_or(OperatorError::RuntimeError)
    }
    fn multiply(a: Self, b: Self) -> Result<Self, OperatorError> {
        a.checked_mul(b).ok_or(OperatorError::RuntimeError)
    }
    fn divide(a: Self, b: Self) -> Result<Self, OperatorError> {
        if b == 0 {
            Err(OperatorError::RuntimeError)
        } else {
            Ok(a / b)
        }
    }
    fn bitwise_or(a: Self, b: Self) -> Result<Self, OperatorError> {
        Ok(a | b)
    }
    fn bitwise_xor(a: Self, b: Self) -> Result<Self, OperatorError> {
        Ok(a ^ b)
    }
    fn bitwise_and(a: Self, b: Self) -> Result<Self, OperatorError> {
        Ok(a & b)
    }
    fn shift_left(a: Self, b: Self) -> Result<Self, OperatorError> {
        if b < 64 {
            Ok(a << b)
        } else {
            Err(OperatorError::RuntimeError)
        }
    }
    fn shift_right(a: Self, b: Self) -> Result<Self, OperatorError> {
        if b < 64 {
            Ok(a >> b)
        } else {
            Err(OperatorError::RuntimeError)
        }
    }
}

impl ValueType for f64 {
    const BYTES: usize = 8;
    const TYPE: Type = Type::Float;

    fn write(&self, to: &mut [u8]) {
        to.copy_from_slice(&self.to_le_bytes());
    }

    fn from_bytes(bytes: &[u8]) -> Self {
        f64::from_le_bytes(bytes.try_into().unwrap())
    }

    fn less_than(a: Self, b: Self) -> Result<bool, OperatorError> {
        Ok(a < b)
    }
    fn equals(a: Self, b: Self) -> Result<bool, OperatorError> {
        Ok(a == b)
    }
    fn add(a: Self, b: Self) -> Result<Self, OperatorError> {
        Ok(a + b)
    }
    fn subtract(a: Self, b: Self) -> Result<Self, OperatorError> {
        Ok(a - b)
    }
    fn multiply(a: Self, b: Self) -> Result<Self, OperatorError> {
        Ok(a * b)
    }
    fn divide(a: Self, b: Self) -> Result<Self, OperatorError> {
        Ok(a / b)
    }
}

impl ValueType for bool {
    const BYTES: usize = 1;
    const TYPE: Type = Type::Bool;

    fn write(&self, to: &mut [u8]) {
        to.copy_from_slice(&[*self as u8]);
    }

    fn from_bytes(bytes: &[u8]) -> Self {
        bytes[0] != 0
    }

    fn less_than(_: Self, _: Self) -> Result<bool, OperatorError> {
        Err(OperatorError::NotSupported)
    }
    fn equals(a: Self, b: Self) -> Result<bool, OperatorError> {
        Ok(a == b)
    }
}

impl ValueType for StringIndex {
    const BYTES: usize = 8;
    const TYPE: Type = Type::String;

    fn write(&self, to: &mut [u8]) {
        to.copy_from_slice(&self.0.to_le_bytes());
    }
    fn from_bytes(bytes: &[u8]) -> Self {
        StringIndex(u64::from_le_bytes(bytes.try_into().unwrap()) as usize)
    }
    fn less_than(_: Self, _: Self) -> Result<bool, OperatorError> {
        Err(OperatorError::NotSupported)
    }
    fn equals(a: Self, b: Self) -> Result<bool, OperatorError> {
        Ok(a == b)
    }
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub enum Type {
    Void,
    Int,
    SignedInt,
    Float,
    Bool,
    String,
}
impl Type {
    pub fn size_of(&self) -> usize {
        match self {
            Type::Void => <() as ValueType>::BYTES,
            Type::Int => <u64 as ValueType>::BYTES,
            Type::SignedInt => <i64 as ValueType>::BYTES,
            Type::Float => <f64 as ValueType>::BYTES,
            Type::Bool => <bool as ValueType>::BYTES,
            Type::String => <StringIndex as ValueType>::BYTES,
        }
    }
}

impl Display for Type {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Type::Void => write!(f, "void"),
            Type::Int => write!(f, "int"),
            Type::SignedInt => write!(f, "signed"),
            Type::Bool => write!(f, "bool"),
            Type::String => write!(f, "string"),
            Type::Float => write!(f, "float"),
        }
    }
}

#[derive(Default)]
pub struct Context<'ctx> {
    variables: IndexMap<String, TypedValue>,
    functions: IndexMap<String, ExprFn<'ctx>>,
    strings: StringInterner,
}

impl ExprContext for Context<'_> {
    fn intern_string(&mut self, s: &str) -> StringIndex {
        self.strings.intern(s)
    }

    fn try_load_variable(&self, variable: &str) -> Option<TypedValue> {
        self.variables.get(variable).copied()
    }

    fn address_of(&self, variable: &str) -> TypedValue {
        let index = self.variables.get_index_of(variable).unwrap();
        TypedValue::Int(index as u64)
    }

    fn call_function(
        &mut self,
        function_name: &str,
        args: &[TypedValue],
    ) -> Result<TypedValue, FunctionCallError> {
        match self.functions.get(function_name) {
            Some(func) => func.call(args),
            None => Err(FunctionCallError::FunctionNotFound),
        }
    }
}

impl<'ctx> Context<'ctx> {
    pub fn new() -> Self {
        Self {
            variables: IndexMap::new(),
            functions: IndexMap::new(),
            strings: StringInterner::new(),
        }
    }

    fn evaluate_any_impl(&mut self, expression: &str) -> Result<TypedValue, EvalError> {
        // TODO: we can allow new globals to be defined in the expression, but that would require
        // storing a copy of the original globals, so that they can be reset?
        let tokens = match lexer::tokenize(expression).collect::<Result<Vec<_>, _>>() {
            Ok(tokens) => tokens,
            Err(e) => {
                return Err(EvalError {
                    message: format!("Syntax error: {e}").into_boxed_str(),
                    location: e.location,
                });
            }
        };
        let ast = match parser::parse_expression(expression, &tokens) {
            Ok(ast) => ast,
            Err(e) => {
                return Err(EvalError {
                    message: format!("Parser error: {e}").into_boxed_str(),
                    location: e.location,
                });
            }
        };

        let mut visitor = ExpressionVisitor {
            context: self,
            source: expression,
        };

        visitor.visit_expression(&ast)
    }

    pub fn evaluate_any<'s>(
        &mut self,
        expression: &'s str,
    ) -> Result<TypedValue, ExpressionError<'s>> {
        self.evaluate_any_impl(expression)
            .map_err(|error| ExpressionError {
                error,
                source: expression,
            })
    }

    pub fn evaluate<'s, V: ValueType>(
        &mut self,
        expression: &'s str,
    ) -> Result<V, ExpressionError<'s>> {
        let result = self.evaluate_any(expression)?;
        let result_ty = result.type_of();
        result.try_into().map_err(|_| ExpressionError {
            error: EvalError {
                message: format!(
                    "Expression evaluates to {result_ty}, which cannot be converted to {}",
                    V::TYPE
                )
                .into_boxed_str(),
                location: Location::dummy(),
            },
            source: expression,
        })
    }

    pub fn add_variable(&mut self, name: &str, value: TypedValue) {
        self.variables.insert(name.to_string(), value);
    }

    pub fn add_function<F, A>(&mut self, name: &str, func: F)
    where
        F: DynFunction<A> + 'ctx,
    {
        let func = ExprFn::new(func);
        self.functions.insert(name.to_string(), func);
    }
}

pub trait DynFunction<A> {
    fn call(&self, args: &[TypedValue]) -> Result<TypedValue, FunctionCallError>;
}

#[macro_export]
#[doc(hidden)]
macro_rules! for_all_tuples {
    ($pat:tt => $code:tt;) => {
        macro_rules! inner { $pat => $code; }

        inner!();
        inner!(V1);
        inner!(V1, V2);
        inner!(V1, V2, V3);
        inner!(V1, V2, V3, V4);
        inner!(V1, V2, V3, V4, V5);
        inner!(V1, V2, V3, V4, V5, V6);
        inner!(V1, V2, V3, V4, V5, V6, V7);
        inner!(V1, V2, V3, V4, V5, V6, V7, V8);
        inner!(V1, V2, V3, V4, V5, V6, V7, V8, V9);
        inner!(V1, V2, V3, V4, V5, V6, V7, V8, V9, V10);
    };
}

pub enum FunctionCallError {
    FunctionNotFound,
    IncorrectArgumentCount { expected: usize },
    IncorrectArgumentType { idx: usize, expected: Type },
    Other(&'static str),
}

macro_rules! ignore {
    ($arg:tt) => {};
}

for_all_tuples! {
    ($($arg:ident),*) => {
        impl<$($arg,)* R, F> DynFunction<($($arg,)*)> for F
        where
            $($arg: ValueType,)*
            F: Fn($($arg,)*) -> R,
            R: ValueType,
        {
            #[allow(non_snake_case, unused)]
            fn call(&self, args: &[TypedValue]) -> Result<TypedValue, FunctionCallError> {
                let arg_count = 0;
                $(
                    ignore!($arg);
                    let arg_count = arg_count + 1;
                )*

                let idx = 0;
                let mut args = args.iter().copied();
                $(
                    let Some(arg) = args.next() else {
                        return Err(FunctionCallError::IncorrectArgumentCount { expected: arg_count });
                    };
                    let $arg = match <$arg>::try_from(arg) {
                        Ok(arg) => arg,
                        Err(_) => return Err(FunctionCallError::IncorrectArgumentType { idx, expected: $arg::TYPE }),
                    };
                    let idx = idx + 1;
                )*

                if args.next().is_some() {
                    return Err(FunctionCallError::IncorrectArgumentCount { expected: arg_count });
                }

                Ok(self($($arg),*).into())
            }
        }
    };
}

struct ExprFn<'ctx> {
    #[allow(clippy::type_complexity)]
    func: Box<dyn Fn(&[TypedValue]) -> Result<TypedValue, FunctionCallError> + 'ctx>,
}

impl<'ctx> ExprFn<'ctx> {
    fn new<A, F>(func: F) -> Self
    where
        F: DynFunction<A> + 'ctx,
    {
        Self {
            func: Box::new(move |args| func.call(args)),
        }
    }

    fn call(&self, args: &[TypedValue]) -> Result<TypedValue, FunctionCallError> {
        (self.func)(args)
    }
}

#[cfg(test)]
mod test {
    use super::*;

    fn strip_ansi(s: impl AsRef<str>) -> String {
        use ansi_parser::AnsiParser;
        fn text_block(output: ansi_parser::Output<'_>) -> Option<&str> {
            match output {
                ansi_parser::Output::TextBlock(text) => Some(text),
                _ => None,
            }
        }

        s.as_ref()
            .ansi_parse()
            .filter_map(text_block)
            .collect::<String>()
    }

    #[test]
    fn test_evaluating_exprs() {
        let mut ctx = Context::new();

        ctx.add_variable("value", TypedValue::Int(30));
        ctx.add_function("func", |v: u64| 2 * v);
        ctx.add_function("func2", |v1: u64, v2: u64| v1 + v2);

        assert_eq!(ctx.evaluate::<bool>("value / 5 == 6"), Ok(true));
        assert_eq!(ctx.evaluate::<u64>("func(20) / 5"), Ok(8));
        assert_eq!(ctx.evaluate::<u64>("func2(20, 20) / 5"), Ok(8));
    }

    #[test]
    fn test_eval_error() {
        let mut ctx = Context::new();

        ctx.add_function("func", |v1: u64, v2: u64| v1 + v2);

        let err = ctx
            .evaluate::<u64>("func(20, true)")
            .expect_err("Expected expression to return an error");

        pretty_assertions::assert_eq!(
            r#"Eval error
 ---> at line 1 column 10
  |
1 | func(20, true)
  |          ^^^^ func expects argument 1 to be int, got bool"#,
            strip_ansi(format!("{err:?}"))
        );
    }
}
