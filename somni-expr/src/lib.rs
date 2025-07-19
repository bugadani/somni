pub mod string_interner;

use std::fmt::Display;

use somni_parser::{
    ast::{self, Expression},
    lexer,
};

use crate::string_interner::StringIndex;

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
    RuntimeError,
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
                (a, b) => panic!("Cannot compare {} and {}", a.type_of(), b.type_of()),
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
                (a, b) => panic!("Cannot compare {} and {}", a.type_of(), b.type_of()),
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
impl From<i64> for TypedValue {
    fn from(value: i64) -> Self {
        TypedValue::SignedInt(value)
    }
}
impl From<f64> for TypedValue {
    fn from(value: f64) -> Self {
        TypedValue::Float(value)
    }
}
impl From<bool> for TypedValue {
    fn from(value: bool) -> Self {
        TypedValue::Bool(value)
    }
}
impl From<StringIndex> for TypedValue {
    fn from(value: StringIndex) -> Self {
        TypedValue::String(value)
    }
}
impl From<()> for TypedValue {
    fn from(_: ()) -> Self {
        TypedValue::Void
    }
}

pub trait ExprContext {
    fn intern_string(&mut self, s: &str) -> StringIndex;
    fn try_load_variable(&self, variable: &str) -> Option<TypedValue>;
    fn address_of(&self, variable: &str) -> TypedValue;
    fn call_function(&mut self, function_name: &str, args: &[TypedValue]) -> TypedValue;
}

pub struct ExpressionVisitor<'a, C> {
    pub context: &'a mut C,
    pub source: &'a str,
}

#[derive(Clone, Copy, Debug)]
pub struct EvalError;

impl<'a, C> ExpressionVisitor<'a, C>
where
    C: ExprContext,
{
    fn visit_variable(&mut self, variable: &lexer::Token) -> Result<TypedValue, EvalError> {
        let name = variable.source(self.source);
        self.context.try_load_variable(name).ok_or(EvalError)
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
                    value => panic!("Expected boolean, found {}", value.type_of()),
                },
                "-" => match self.visit_expression(operand)? {
                    // TODO handle overflow errors
                    TypedValue::SignedInt(i) => TypedValue::SignedInt(-i),
                    TypedValue::Int(i) => TypedValue::SignedInt(-(i as i64)),
                    TypedValue::Float(f) => TypedValue::Float(-f),
                    value => panic!("Cannot negate {}", value.type_of()),
                },
                "&" => match operand.as_variable() {
                    Some(variable) => {
                        let name = variable.source(self.source);
                        self.context.address_of(name)
                    }
                    None => panic!("Cannot take address of non-variable expression"),
                },
                "*" => panic!("Dereference not supported"),
                _ => panic!("Unknown unary operator: {}", name.source(self.source)),
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
                match name.source(self.source) {
                    "+" => TypedValue::add(lhs, rhs).unwrap(),
                    "-" => TypedValue::subtract(lhs, rhs).unwrap(),
                    "*" => TypedValue::multiply(lhs, rhs).unwrap(),
                    "/" => TypedValue::divide(lhs, rhs).unwrap(),
                    "<" => TypedValue::less_than(lhs, rhs).unwrap(),
                    ">" => TypedValue::less_than(rhs, lhs).unwrap(),
                    "<=" => TypedValue::less_than_or_equal(lhs, rhs).unwrap(),
                    ">=" => TypedValue::less_than_or_equal(rhs, lhs).unwrap(),
                    "==" => TypedValue::equals(lhs, rhs).unwrap(),
                    "!=" => TypedValue::not_equals(lhs, rhs).unwrap(),
                    "|" => TypedValue::bitwise_or(lhs, rhs).unwrap(),
                    "^" => TypedValue::bitwise_xor(lhs, rhs).unwrap(),
                    "&" => TypedValue::bitwise_and(lhs, rhs).unwrap(),
                    "<<" => TypedValue::shift_left(lhs, rhs).unwrap(),
                    ">>" => TypedValue::shift_right(lhs, rhs).unwrap(),

                    other => panic!("Unknown binary operator: {other}"),
                }
            }
            Expression::FunctionCall { name, arguments } => {
                let function_name = name.source(self.source);
                let mut args = Vec::new();
                for arg in arguments {
                    args.push(self.visit_expression(arg)?);
                }

                self.context.call_function(function_name, &args)
            }
        };

        Ok(result)
    }
}
pub trait ValueType: Sized + Copy + Into<TypedValue> {
    const BYTES: usize;

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
