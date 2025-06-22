use somni_parser::ast::{self, Expression};

use crate::{
    codegen::{Type, ValueType},
    ir,
    string_interner::StringIndex,
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

    pub(crate) fn write(&self, to: &mut [u8]) {
        match self {
            TypedValue::Void => ().write(to),
            TypedValue::Int(value) => value.write(to),
            TypedValue::SignedInt(value) => value.write(to),
            TypedValue::Float(value) => value.write(to),
            TypedValue::Bool(value) => value.write(to),
            TypedValue::String(value) => value.write(to),
        }
    }

    pub(crate) fn from_value(value: &ir::Value) -> TypedValue {
        match value {
            ir::Value::Void => Self::from(()),
            ir::Value::Int(value) => Self::from(*value),
            ir::Value::SignedInt(value) => Self::from(*value),
            ir::Value::Float(value) => Self::from(*value),
            ir::Value::Bool(value) => Self::from(*value),
            ir::Value::String(value) => Self::from(*value),
        }
    }

    pub fn from_typed_bytes(ty: Type, value: &[u8]) -> TypedValue {
        match ty {
            Type::Void => Self::Void,
            Type::Int | Type::Address => Self::Int(<_>::from_bytes(value)),
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

impl<'a, C> ExpressionVisitor<'a, C>
where
    C: ExprContext,
{
    fn visit_variable(&mut self, variable: &somni_lexer::Token) -> TypedValue {
        let name = variable.source(self.source);
        self.context.try_load_variable(name).unwrap()
    }

    pub fn visit_expression(&mut self, expression: &Expression) -> TypedValue {
        // TODO: errors
        match expression {
            Expression::Variable { variable } => self.visit_variable(variable),
            Expression::Literal { value } => match &value.value {
                ast::LiteralValue::Integer(value) => TypedValue::Int(*value),
                ast::LiteralValue::Float(value) => TypedValue::Float(*value),
                ast::LiteralValue::String(value) => {
                    TypedValue::String(self.context.intern_string(value))
                }
                ast::LiteralValue::Boolean(value) => TypedValue::Bool(*value),
            },
            Expression::UnaryOperator { name, operand } => match name.source(self.source) {
                "!" => match self.visit_expression(operand) {
                    TypedValue::Bool(b) => TypedValue::Bool(!b),
                    value => panic!("Expected boolean, found {}", value.type_of()),
                },
                "-" => match self.visit_expression(operand) {
                    TypedValue::SignedInt(i) => TypedValue::SignedInt(-i),
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
                    let lhs = self.visit_expression(&operands[0]);
                    return match operator {
                        "&&" if lhs == TypedValue::Bool(false) => TypedValue::Bool(false),
                        "||" if lhs == TypedValue::Bool(true) => TypedValue::Bool(true),
                        _ => self.visit_expression(&operands[1]),
                    };
                }

                let lhs = self.visit_expression(&operands[0]);
                let rhs = self.visit_expression(&operands[1]);
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

                    other => panic!("Unknown binary operator: {}", other),
                }
            }
            Expression::FunctionCall { name, arguments } => {
                let function_name = name.source(self.source);
                let mut args = Vec::new();
                for arg in arguments {
                    args.push(self.visit_expression(arg));
                }

                self.context.call_function(function_name, &args)
            }
        }
    }
}
