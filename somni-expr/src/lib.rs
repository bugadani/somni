//! # Somni expression evaluation Library
//!
//! This library provides tools for evaluating expressions.
//!
//! ## Overview
//!
//! The expression language includes:
//!
//! - Literals: integers, floats, booleans. No strings (yet).
//! - Variables
//! - A basic set of operators
//! - Function calls
//!
//! The expression language does not include:
//!
//! - Control flow (if, loops, etc.)
//! - Complex data structures (arrays, objects, etc.)
//! - Defining functions and variables (these are provided by the context)
//!
//! ## Operators
//!
//! The following binary operators are supported, in order of precedence:
//!
//! - `||`: logical OR, short-circuiting
//! - `&&`: logical AND, short-circuiting
//! - `<`, `<=`, `>`, `>=`, `==`, `!=`: comparison operators
//! - `|`: bitwise OR
//! - `^`: bitwise XOR
//! - `&`: bitwise AND
//! - `<<`, `>>`: bitwise shift
//! - `+`, `-`: addition and subtraction
//! - `*`, `/`: multiplication and division
//!
//! Unary operators include:
//! - `!`: logical NOT
//! - `-`: negation
//!
//! For the full specification of the grammar, see the [`parser`] module's documentation.
//!
//! ## Numeric types
//!
//! The Somni language supports three numeric types:
//!
//! - Integers
//! - Signed integers
//! - Floats
//!
//! By default, the library uses the [`DefaultTypeSet`], which uses `u64`, `i64`, and `f64` for
//! these types. You can use other type sets like [`TypeSet32`] or [`TypeSet128`] to use
//! 32-bit or 128-bit integers and floats. You need to specify the type set when creating
//! the context.
//!
//! ## Usage
//!
//! To evaluate an expression, you need to create a [`Context`] first. You can assign
//! variables and define functions in this context, and then you can use this context
//! to evaluate expressions.
//!
//! ```rust
//! use somni_expr::Context;
//!
//! let mut context = Context::new();
//!
//! // Define a variable
//! context.add_variable::<u64>("x", 42);
//! context.add_function("add_one", |x: u64| { x + 1 });
//! context.add_function("floor", |x: f64| { x.floor() as u64 });
//!
//! // Evaluate an expression - we expect it to evaluate
//! // to a number, which is u64 in the default type set.
//! let result = context.evaluate::<u64>("add_one(x + floor(1.2))");
//!
//! assert_eq!(result, Ok(44));
//! ```
#![warn(missing_docs)]

pub mod error;
#[doc(hidden)]
pub mod string_interner;

use std::{
    collections::HashMap,
    fmt::{Debug, Display},
};

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

pub use somni_parser::parser::{DefaultTypeSet, TypeSet128, TypeSet32};

mod private {
    pub trait Sealed {}
    impl Sealed for u32 {}
    impl Sealed for u64 {}
    impl Sealed for u128 {}
}

use private::Sealed;

/// Defines numeric types in expressions.
pub trait TypeSet: somni_parser::parser::TypeSet + PartialEq
where
    Self::Integer: ValueType,
    Self::Float: ValueType,
{
    /// The type of signed integers in this type set.
    type SignedInteger: ValueType;

    /// Negates a value of this type set, returning `None` if the value cannot be negated.
    fn negate(v: TypedValue<Self>) -> Option<TypedValue<Self>>;
}

impl TypeSet for DefaultTypeSet {
    type SignedInteger = i64;

    fn negate(v: TypedValue<Self>) -> Option<TypedValue<Self>> {
        let v = match v {
            // TODO handle overflow errors
            TypedValue::SignedInt(i) => TypedValue::SignedInt(-i),
            TypedValue::Int(i) => TypedValue::SignedInt(-(i as i64)),
            TypedValue::Float(f) => TypedValue::Float(-f),
            _ => return None,
        };

        Some(v)
    }
}

impl TypeSet for TypeSet32 {
    type SignedInteger = i32;

    fn negate(v: TypedValue<Self>) -> Option<TypedValue<Self>> {
        let v = match v {
            TypedValue::SignedInt(i) => TypedValue::SignedInt(-i),
            TypedValue::Int(i) => TypedValue::SignedInt(-(i as i32)),
            TypedValue::Float(f) => TypedValue::Float(-f),
            _ => return None,
        };

        Some(v)
    }
}

impl TypeSet for TypeSet128 {
    type SignedInteger = i128;

    fn negate(v: TypedValue<Self>) -> Option<TypedValue<Self>> {
        let v = match v {
            TypedValue::SignedInt(i) => TypedValue::SignedInt(-i),
            TypedValue::Int(i) => TypedValue::SignedInt(-(i as i128)),
            TypedValue::Float(f) => TypedValue::Float(-f),
            _ => return None,
        };

        Some(v)
    }
}

#[doc(hidden)]
pub trait Integer: ValueType + Sealed {
    fn from_usize(value: usize) -> Self;
}

impl Integer for u32 {
    fn from_usize(value: usize) -> Self {
        u32::try_from(value).unwrap()
    }
}
impl Integer for u64 {
    fn from_usize(value: usize) -> Self {
        u64::try_from(value).unwrap()
    }
}
impl Integer for u128 {
    fn from_usize(value: usize) -> Self {
        u128::try_from(value).unwrap()
    }
}

/// Represents any value in the expression language.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TypedValue<T = DefaultTypeSet>
where
    T: TypeSet,
    T::Integer: ValueType,
    T::Float: ValueType,
{
    /// Represents no value.
    Void,
    /// Represents an unsigned integer.
    Int(T::Integer),
    /// Represents a signed integer.
    SignedInt(T::SignedInteger),
    /// Represents a floating-point.
    Float(T::Float),
    /// Represents a boolean.
    Bool(bool),
    /// Represents an interned string.
    String(StringIndex),
}

/// Represents an error that can occur during operator evaluation.
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub enum OperatorError {
    /// A type error occurred.
    TypeError,
    /// A runtime error occurred.
    RuntimeError,
}

impl Display for OperatorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let message = match self {
            OperatorError::TypeError => "Type error",
            OperatorError::RuntimeError => "Runtime error",
        };

        f.write_str(message)
    }
}

macro_rules! dispatch_binary {
    ($method:ident) => {
        pub(crate) fn $method(lhs: Self, rhs: Self) -> Result<Self, OperatorError> {
            let result = match (lhs, rhs) {
                (Self::Bool(value), Self::Bool(other)) => {
                    Self::from(ValueType::$method(value, other)?)
                }
                (Self::Int(value), Self::Int(other)) => {
                    Self::from(ValueType::$method(value, other)?)
                }
                (Self::SignedInt(value), Self::SignedInt(other)) => {
                    Self::from(ValueType::$method(value, other)?)
                }
                (Self::Float(value), Self::Float(other)) => {
                    Self::from(ValueType::$method(value, other)?)
                }
                (Self::String(value), Self::String(other)) => {
                    Self::from(ValueType::$method(value, other)?)
                }
                _ => return Err(OperatorError::TypeError),
            };
            Ok(result)
        }
    };
}

macro_rules! dispatch_unary {
    ($method:ident) => {
        pub(crate) fn $method(operand: Self) -> Result<Self, OperatorError> {
            match operand {
                Self::Bool(value) => Ok(Self::from(ValueType::$method(value)?)),
                Self::Int(value) => Ok(Self::from(ValueType::$method(value)?)),
                Self::SignedInt(value) => Ok(Self::from(ValueType::$method(value)?)),
                Self::Float(value) => Ok(Self::from(ValueType::$method(value)?)),
                Self::String(value) => Ok(Self::from(ValueType::$method(value)?)),
                _ => return Err(OperatorError::TypeError),
            }
        }
    };
}

impl<T> TypedValue<T>
where
    T: TypeSet,
    T::Integer: ValueType,
    T::Float: ValueType,
{
    /// Returns the Somni type of this value.
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

    /// Writes the raw bytes of this value to the provided buffer.
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

    /// Creates a `TypedValue` from the provided type and bytes.
    pub fn from_typed_bytes(ty: Type, value: &[u8]) -> TypedValue<T> {
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

impl<T> TypedValue<T>
where
    T: TypeSet,
    T::Integer: ValueType,
    T::Float: ValueType,
    Self: From<T::Integer>,
    Self: From<T::SignedInteger>,
    Self: From<T::Float>,
{
    dispatch_binary!(equals);
    dispatch_binary!(less_than);
    dispatch_binary!(less_than_or_equal);
    dispatch_binary!(not_equals);
    dispatch_binary!(bitwise_or);
    dispatch_binary!(bitwise_xor);
    dispatch_binary!(bitwise_and);
    dispatch_binary!(shift_left);
    dispatch_binary!(shift_right);
    dispatch_binary!(add);
    dispatch_binary!(subtract);
    dispatch_binary!(multiply);
    dispatch_binary!(divide);
    dispatch_unary!(not);
}

macro_rules! convert {
    ($type:ty, $ts_kind:ident, $kind:ident) => {
        impl<T> From<$type> for TypedValue<T>
        where
            T: TypeSet<$ts_kind = $type>,
            T::Integer: ValueType,
            T::Float: ValueType,
        {
            fn from(value: $type) -> Self {
                TypedValue::$kind(value)
            }
        }
        impl<T> TryFrom<TypedValue<T>> for $type
        where
            T: TypeSet<$ts_kind = $type>,
            T::Integer: ValueType,
            T::Float: ValueType,
        {
            type Error = ();

            fn try_from(value: TypedValue<T>) -> Result<Self, Self::Error> {
                if let TypedValue::$kind(value) = value {
                    Ok(value)
                } else {
                    Err(())
                }
            }
        }
    };
}

convert!(u32, Integer, Int);
convert!(u64, Integer, Int);
convert!(u128, Integer, Int);
convert!(i32, SignedInteger, SignedInt);
convert!(i64, SignedInteger, SignedInt);
convert!(i128, SignedInteger, SignedInt);
convert!(f32, Float, Float);
convert!(f64, Float, Float);

impl<T> From<bool> for TypedValue<T>
where
    T: TypeSet,
    T::Integer: ValueType,
    T::Float: ValueType,
{
    fn from(value: bool) -> Self {
        TypedValue::Bool(value)
    }
}
impl<T> TryFrom<TypedValue<T>> for bool
where
    T: TypeSet,
    T::Integer: ValueType,
    T::Float: ValueType,
{
    type Error = ();

    fn try_from(value: TypedValue<T>) -> Result<Self, Self::Error> {
        if let TypedValue::Bool(value) = value {
            Ok(value)
        } else {
            Err(())
        }
    }
}

impl<T> From<StringIndex> for TypedValue<T>
where
    T: TypeSet,
    T::Integer: ValueType,
    T::Float: ValueType,
{
    fn from(value: StringIndex) -> Self {
        TypedValue::String(value)
    }
}
impl<T> TryFrom<TypedValue<T>> for StringIndex
where
    T: TypeSet,
    T::Integer: ValueType,
    T::Float: ValueType,
{
    type Error = ();

    fn try_from(value: TypedValue<T>) -> Result<Self, Self::Error> {
        if let TypedValue::String(str) = value {
            Ok(str)
        } else {
            Err(())
        }
    }
}

impl<T> From<()> for TypedValue<T>
where
    T: TypeSet,
    T::Integer: ValueType,
    T::Float: ValueType,
{
    fn from(_: ()) -> Self {
        TypedValue::Void
    }
}

impl<T> TryFrom<TypedValue<T>> for ()
where
    T: TypeSet,
    T::Integer: ValueType,
    T::Float: ValueType,
{
    type Error = ();

    fn try_from(value: TypedValue<T>) -> Result<Self, Self::Error> {
        if let TypedValue::Void = value {
            Ok(())
        } else {
            Err(())
        }
    }
}

/// An expression context that provides the necessary environment for evaluating expressions.
pub trait ExprContext<T = DefaultTypeSet>
where
    T: TypeSet,
    T::Integer: ValueType,
    T::Float: ValueType,
{
    /// Implements string interning.
    fn intern_string(&mut self, s: &str) -> StringIndex;

    /// Attempts to load a variable from the context.
    fn try_load_variable(&self, variable: &str) -> Option<TypedValue<T>>;

    /// Returns the address of a variable in the context.
    fn address_of(&self, variable: &str) -> TypedValue<T>;

    /// Calls a function in the context.
    fn call_function(
        &mut self,
        function_name: &str,
        args: &[TypedValue<T>],
    ) -> Result<TypedValue<T>, FunctionCallError>;
}

/// A visitor that can process an abstract syntax tree.
pub struct ExpressionVisitor<'a, C, T = DefaultTypeSet> {
    /// The context in which the expression is evaluated.
    pub context: &'a mut C,
    /// The source code from which the expression was parsed.
    pub source: &'a str,
    /// The types of the variables in the context.
    pub _marker: std::marker::PhantomData<T>,
}

/// An error that occurs during evaluation of an expression.
#[derive(Clone, Debug, PartialEq)]
pub struct EvalError {
    /// The error message.
    pub message: Box<str>,
    /// The location in the source code where the error occurred.
    pub location: Location,
}

impl Display for EvalError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Evaluation error: {}", self.message)
    }
}

/// An error that occurs during evaluation.
///
/// Printing this error will show the error message and the location in the source code.
///
/// ```rust
/// use somni_expr::{Context, TypeSet32};
/// let mut ctx = Context::<TypeSet32>::new_with_types();
///
/// let error = ctx.evaluate::<u32>("true + 1").unwrap_err();
///
/// println!("{error:?}");
///
/// // Output:
/// //
/// // Evaluation error
/// // ---> at line 1 column 1
/// //   |
/// // 1 | true + 1
/// //   | ^^^^^^^^ Failed to evaluate expression: Type error
/// ```
#[derive(Clone, PartialEq)]
pub struct ExpressionError<'s> {
    error: EvalError,
    source: &'s str,
}

impl ExpressionError<'_> {
    /// Returns the inner [`EvalError`].
    pub fn into_inner(self) -> EvalError {
        self.error
    }
}

impl Debug for ExpressionError<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let marked = MarkInSource(
            self.source,
            self.error.location,
            "Evaluation error",
            &self.error.message,
        );
        marked.fmt(f)
    }
}

impl<'a, C, T> ExpressionVisitor<'a, C, T>
where
    C: ExprContext<T>,
    T: TypeSet,
    T::Integer: ValueType,
    T::Float: ValueType,
    TypedValue<T>: From<T::Integer>,
    TypedValue<T>: From<T::SignedInteger>,
    TypedValue<T>: From<T::Float>,
{
    fn visit_variable(&mut self, variable: &lexer::Token) -> Result<TypedValue<T>, EvalError> {
        let name = variable.source(self.source);
        self.context.try_load_variable(name).ok_or(EvalError {
            message: format!("Variable {name} was not found").into_boxed_str(),
            location: variable.location,
        })
    }

    /// Visits an expression and evaluates it, returning the result as a `TypedValue`.
    pub fn visit_expression(
        &mut self,
        expression: &Expression<T>,
    ) -> Result<TypedValue<T>, EvalError> {
        let result = match expression {
            Expression::Variable { variable } => self.visit_variable(variable)?,
            Expression::Literal { value } => match &value.value {
                ast::LiteralValue::Integer(value) => TypedValue::<T>::Int(*value),
                ast::LiteralValue::Float(value) => TypedValue::<T>::Float(*value),
                ast::LiteralValue::String(value) => {
                    TypedValue::<T>::String(self.context.intern_string(value))
                }
                ast::LiteralValue::Boolean(value) => TypedValue::<T>::Bool(*value),
            },
            Expression::UnaryOperator { name, operand } => match name.source(self.source) {
                "!" => {
                    let operand = self.visit_expression(operand)?;

                    match TypedValue::<T>::not(operand) {
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

                "-" => {
                    let value = self.visit_expression(operand)?;
                    T::negate(value).ok_or_else(|| EvalError {
                        message: format!("Cannot negate {}", value.type_of()).into_boxed_str(),
                        location: operand.location(),
                    })?
                }
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
                        "&&" if lhs == TypedValue::<T>::Bool(false) => Ok(TypedValue::Bool(false)),
                        "||" if lhs == TypedValue::<T>::Bool(true) => Ok(TypedValue::Bool(true)),
                        _ => self.visit_expression(&operands[1]),
                    };
                }

                let lhs = self.visit_expression(&operands[0])?;
                let rhs = self.visit_expression(&operands[1])?;
                let result = match name.source(self.source) {
                    "+" => TypedValue::<T>::add(lhs, rhs),
                    "-" => TypedValue::<T>::subtract(lhs, rhs),
                    "*" => TypedValue::<T>::multiply(lhs, rhs),
                    "/" => TypedValue::<T>::divide(lhs, rhs),
                    "<" => TypedValue::<T>::less_than(lhs, rhs),
                    ">" => TypedValue::<T>::less_than(rhs, lhs),
                    "<=" => TypedValue::<T>::less_than_or_equal(lhs, rhs),
                    ">=" => TypedValue::<T>::less_than_or_equal(rhs, lhs),
                    "==" => TypedValue::<T>::equals(lhs, rhs),
                    "!=" => TypedValue::<T>::not_equals(lhs, rhs),
                    "|" => TypedValue::<T>::bitwise_or(lhs, rhs),
                    "^" => TypedValue::<T>::bitwise_xor(lhs, rhs),
                    "&" => TypedValue::<T>::bitwise_and(lhs, rhs),
                    "<<" => TypedValue::<T>::shift_left(lhs, rhs),
                    ">>" => TypedValue::<T>::shift_right(lhs, rhs),

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

#[doc(hidden)]
pub trait ValueType: Sized + Copy + PartialEq {
    const BYTES: usize;
    const TYPE: Type;

    fn write(&self, to: &mut [u8]);
    fn from_bytes(bytes: &[u8]) -> Self;

    fn equals(_a: Self, _b: Self) -> Result<bool, OperatorError> {
        unimplemented!("Operation not supported")
    }
    fn less_than(_a: Self, _b: Self) -> Result<bool, OperatorError> {
        unimplemented!("Operation not supported")
    }

    fn less_than_or_equal(a: Self, b: Self) -> Result<bool, OperatorError> {
        let less = Self::less_than(a, b)?;
        Ok(less || Self::equals(a, b)?)
    }
    fn not_equals(a: Self, b: Self) -> Result<bool, OperatorError> {
        let equals = Self::equals(a, b)?;
        Ok(!equals)
    }
    fn bitwise_or(_a: Self, _b: Self) -> Result<Self, OperatorError> {
        unimplemented!("Operation not supported")
    }
    fn bitwise_xor(_a: Self, _b: Self) -> Result<Self, OperatorError> {
        unimplemented!("Operation not supported")
    }
    fn bitwise_and(_a: Self, _b: Self) -> Result<Self, OperatorError> {
        unimplemented!("Operation not supported")
    }
    fn shift_left(_a: Self, _b: Self) -> Result<Self, OperatorError> {
        unimplemented!("Operation not supported")
    }
    fn shift_right(_a: Self, _b: Self) -> Result<Self, OperatorError> {
        unimplemented!("Operation not supported")
    }
    fn add(_a: Self, _b: Self) -> Result<Self, OperatorError> {
        unimplemented!("Operation not supported")
    }
    fn subtract(_a: Self, _b: Self) -> Result<Self, OperatorError> {
        unimplemented!("Operation not supported")
    }
    fn multiply(_a: Self, _b: Self) -> Result<Self, OperatorError> {
        unimplemented!("Operation not supported")
    }
    fn divide(_a: Self, _b: Self) -> Result<Self, OperatorError> {
        unimplemented!("Operation not supported")
    }
    fn not(_a: Self) -> Result<Self, OperatorError> {
        unimplemented!("Operation not supported")
    }
    fn negate(_a: Self) -> Result<Self, OperatorError> {
        unimplemented!("Operation not supported")
    }
}

impl ValueType for () {
    const BYTES: usize = 0;
    const TYPE: Type = Type::Void;

    fn write(&self, _to: &mut [u8]) {}

    fn from_bytes(_: &[u8]) -> Self {}
}

macro_rules! value_type_int {
    ($type:ty, $kind:ident) => {
        impl ValueType for $type {
            const BYTES: usize = std::mem::size_of::<$type>();
            const TYPE: Type = Type::$kind;

            fn write(&self, to: &mut [u8]) {
                to.copy_from_slice(&self.to_le_bytes());
            }

            fn from_bytes(bytes: &[u8]) -> Self {
                <$type>::from_le_bytes(bytes.try_into().unwrap())
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
                if b < std::mem::size_of::<$type>() as Self * 8 {
                    Ok(a << b)
                } else {
                    Err(OperatorError::RuntimeError)
                }
            }
            fn shift_right(a: Self, b: Self) -> Result<Self, OperatorError> {
                if b < std::mem::size_of::<$type>() as Self * 8 {
                    Ok(a >> b)
                } else {
                    Err(OperatorError::RuntimeError)
                }
            }
            fn not(a: Self) -> Result<Self, OperatorError> {
                Ok(!a)
            }
        }
    };
}

value_type_int!(u32, Int);
value_type_int!(u64, Int);
value_type_int!(u128, Int);
value_type_int!(i32, SignedInt);
value_type_int!(i64, SignedInt);
value_type_int!(i128, SignedInt);

impl ValueType for f32 {
    const BYTES: usize = 4;
    const TYPE: Type = Type::Float;

    fn write(&self, to: &mut [u8]) {
        to.copy_from_slice(&self.to_le_bytes());
    }

    fn from_bytes(bytes: &[u8]) -> Self {
        f32::from_le_bytes(bytes.try_into().unwrap())
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

    fn equals(a: Self, b: Self) -> Result<bool, OperatorError> {
        Ok(a == b)
    }

    fn bitwise_and(a: Self, b: Self) -> Result<Self, OperatorError> {
        Ok(a & b)
    }

    fn bitwise_or(a: Self, b: Self) -> Result<Self, OperatorError> {
        Ok(a | b)
    }

    fn bitwise_xor(a: Self, b: Self) -> Result<Self, OperatorError> {
        Ok(a ^ b)
    }

    fn not(a: Self) -> Result<Self, OperatorError> {
        Ok(!a)
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
        unimplemented!("Operation not supported")
    }
    fn equals(a: Self, b: Self) -> Result<bool, OperatorError> {
        Ok(a == b)
    }
}

/// A type in the Somni language.
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub enum Type {
    /// Represents no value, used for e.g. functions that do not return a value.
    Void,
    /// Represents an unsigned integer.
    Int,
    /// Represents a signed integer.
    SignedInt,
    /// Represents a floating point number.
    Float,
    /// Represents a boolean value.
    Bool,
    /// Represents a string value.
    String,
}
impl Type {
    /// Returns the size of the type in bytes.
    pub fn size_of<T>(&self) -> usize
    where
        T: TypeSet,
        T::Integer: ValueType,
        T::Float: ValueType,
    {
        match self {
            Type::Void => <() as ValueType>::BYTES,
            Type::Int => <T::Integer as ValueType>::BYTES,
            Type::SignedInt => <T::SignedInteger as ValueType>::BYTES,
            Type::Float => <T::Float as ValueType>::BYTES,
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

/// The expression context, which holds variables, functions, and other state needed for evaluation.
#[derive(Default)]
pub struct Context<'ctx, T = DefaultTypeSet>
where
    T: TypeSet,
    T::Integer: ValueType,
    T::Float: ValueType,
{
    variables: IndexMap<String, TypedValue<T>>,
    functions: HashMap<String, ExprFn<'ctx, T>>,
    strings: StringInterner,
    marker: std::marker::PhantomData<T>,
}

impl<T> ExprContext<T> for Context<'_, T>
where
    T: TypeSet,
    T::Integer: ValueType + Integer,
    T::Float: ValueType,
{
    fn intern_string(&mut self, s: &str) -> StringIndex {
        self.strings.intern(s)
    }

    fn try_load_variable(&self, variable: &str) -> Option<TypedValue<T>> {
        self.variables.get(variable).copied()
    }

    fn address_of(&self, variable: &str) -> TypedValue<T> {
        let index = self.variables.get_index_of(variable).unwrap();
        TypedValue::Int(<T::Integer as Integer>::from_usize(index))
    }

    fn call_function(
        &mut self,
        function_name: &str,
        args: &[TypedValue<T>],
    ) -> Result<TypedValue<T>, FunctionCallError> {
        match self.functions.remove_entry(function_name) {
            Some((name, func)) => {
                let retval = func.call(self, args);
                self.functions.insert(name, func);

                retval
            }
            None => Err(FunctionCallError::FunctionNotFound),
        }
    }
}

impl<'ctx> Context<'ctx, DefaultTypeSet> {
    /// Creates a new context with [default types][DefaultTypeSet].
    pub fn new() -> Self {
        Self::new_with_types()
    }
}

impl<'ctx, T> Context<'ctx, T>
where
    T: TypeSet,
    T::Integer: Integer,
    T::Float: ValueType,
    TypedValue<T>: From<T::Integer>,
    TypedValue<T>: From<T::SignedInteger>,
    TypedValue<T>: From<T::Float>,
{
    /// Creates a new context. The type set must be specified when using this function.
    ///
    /// ```rust
    /// use somni_expr::{Context, TypeSet32};
    /// let mut ctx = Context::<TypeSet32>::new_with_types();
    /// ```
    pub fn new_with_types() -> Self {
        Self {
            variables: IndexMap::new(),
            functions: HashMap::new(),
            strings: StringInterner::new(),
            marker: std::marker::PhantomData,
        }
    }

    fn evaluate_any_impl(&mut self, expression: &str) -> Result<TypedValue<T>, EvalError> {
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

        let mut visitor = ExpressionVisitor::<Self, T> {
            context: self,
            source: expression,
            _marker: std::marker::PhantomData,
        };

        visitor.visit_expression(&ast)
    }

    /// Evaluates an expression and returns the result as a [`TypedValue<T>`].
    pub fn evaluate_any<'s>(
        &mut self,
        expression: &'s str,
    ) -> Result<TypedValue<T>, ExpressionError<'s>> {
        self.evaluate_any_impl(expression)
            .map_err(|error| ExpressionError {
                error,
                source: expression,
            })
    }

    /// Evaluates an expression and returns the result as a specific value type.
    ///
    /// This function will attempt to convert the result of the expression to the specified type `V`.
    /// If the conversion fails, it will return an `ExpressionError`.
    pub fn evaluate<'s, V: ValueType>(
        &mut self,
        expression: &'s str,
    ) -> Result<V, ExpressionError<'s>>
    where
        V: TryFrom<TypedValue<T>>,
    {
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

    /// Defines a new variable in the context.
    pub fn add_variable<V>(&mut self, name: &str, value: V)
    where
        V: Into<TypedValue<T>>,
    {
        self.variables.insert(name.to_string(), value.into());
    }

    /// Adds a new function to the context.
    pub fn add_function<F, A>(&mut self, name: &str, func: F)
    where
        F: DynFunction<A, T> + 'ctx,
    {
        let func = ExprFn::new(func);
        self.functions.insert(name.to_string(), func);
    }
}

#[doc(hidden)]
pub trait DynFunction<A, T>
where
    T: TypeSet,
    T::Integer: ValueType,
    T::Float: ValueType,
{
    fn call(
        &self,
        ctx: &mut dyn ExprContext<T>,
        args: &[TypedValue<T>],
    ) -> Result<TypedValue<T>, FunctionCallError>;
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

/// An error that occurs when calling a function.
pub enum FunctionCallError {
    /// The function was not found in the context.
    FunctionNotFound,

    /// The number of arguments passed to the function does not match the expected count.
    IncorrectArgumentCount {
        /// The expected number of arguments.
        expected: usize,
    },

    /// The type of an argument does not match the expected type.
    IncorrectArgumentType {
        /// The 0-based index of the argument that has the incorrect type.
        idx: usize,
        /// The expected type of the argument.
        expected: Type,
    },

    /// An error occurred while calling the function.
    Other(&'static str),
}

macro_rules! ignore {
    ($arg:tt) => {};
}

for_all_tuples! {
    ($($arg:ident),*) => {
        impl<$($arg,)* R, F, T> DynFunction<($($arg,)*), T> for F
        where
            $($arg: ValueType + TryFrom<TypedValue<T>>,)*
            F: Fn($($arg,)*) -> R,
            R: ValueType + Into<TypedValue<T>>,
            T: TypeSet,
            T::Integer: ValueType,
            T::Float: ValueType,
        {
            #[allow(non_snake_case, unused)]
            fn call(&self, _ctx: &mut dyn ExprContext<T>, args: &[TypedValue<T>]) -> Result<TypedValue<T>, FunctionCallError> {

                // TODO: while it's great that we can now allow access to the context, it's a bit of a pain to use.
                // The ideal API would load strings in this function, and store the string when returning from the user's function.

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

struct ExprFn<'ctx, T>
where
    T: TypeSet,
    T::Integer: ValueType,
    T::Float: ValueType,
{
    #[allow(clippy::type_complexity)]
    func: Box<
        dyn Fn(
                &mut dyn ExprContext<T>,
                &[TypedValue<T>],
            ) -> Result<TypedValue<T>, FunctionCallError>
            + 'ctx,
    >,
}

impl<'ctx, T> ExprFn<'ctx, T>
where
    T: TypeSet,
    T::Integer: ValueType,
    T::Float: ValueType,
{
    fn new<A, F>(func: F) -> Self
    where
        F: DynFunction<A, T> + 'ctx,
    {
        Self {
            func: Box::new(move |ctx, args| func.call(ctx, args)),
        }
    }

    fn call(
        &self,
        ctx: &mut dyn ExprContext<T>,
        args: &[TypedValue<T>],
    ) -> Result<TypedValue<T>, FunctionCallError> {
        (self.func)(ctx, args)
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

        // TODO: this is a pain point that needs to be resolved
        let five = ctx.strings.intern("five");

        ctx.add_variable("value", TypedValue::Int(30));
        ctx.add_function("func", |v: u64| 2 * v);
        ctx.add_function("func2", |v1: u64, v2: u64| v1 + v2);
        ctx.add_function("five", move || five);

        assert_eq!(ctx.evaluate::<bool>("value / 5 == 6"), Ok(true));
        assert_eq!(ctx.evaluate::<bool>("five() == \"five\""), Ok(true));
        assert_eq!(ctx.evaluate::<u64>("func(20) / 5"), Ok(8));
        assert_eq!(ctx.evaluate::<u64>("func2(20, 20) / 5"), Ok(8));
        assert_eq!(ctx.evaluate::<bool>("true & false"), Ok(false));
        assert_eq!(ctx.evaluate::<bool>("!true"), Ok(false));
        assert_eq!(ctx.evaluate::<bool>("false | false"), Ok(false));
        assert_eq!(ctx.evaluate::<bool>("true ^ true"), Ok(false));
        assert_eq!(ctx.evaluate::<u64>("!0x1111"), Ok(0xFFFF_FFFF_FFFF_EEEE));
    }

    #[test]
    fn test_evaluating_exprs_with_u32() {
        let mut ctx = Context::<TypeSet32>::new_with_types();

        ctx.add_variable("value", TypedValue::Int(30));
        ctx.add_function("func", |v: u32| 2 * v);
        ctx.add_function("func2", |v1: u32, v2: u32| v1 + v2);

        assert_eq!(ctx.evaluate::<bool>("value / 5 == 6"), Ok(true));
        assert_eq!(ctx.evaluate::<u32>("func(20) / 5"), Ok(8));
        assert_eq!(ctx.evaluate::<u32>("func2(20, 20) / 5"), Ok(8));
    }

    #[test]
    fn test_evaluating_exprs_with_u128() {
        let mut ctx = Context::<TypeSet128>::new_with_types();

        ctx.add_variable("value", TypedValue::Int(30));
        ctx.add_function("func", |v: u128| 2 * v);
        ctx.add_function("func2", |v1: u128, v2: u128| v1 + v2);

        assert_eq!(ctx.evaluate::<bool>("value / 5 == 6"), Ok(true));
        assert_eq!(ctx.evaluate::<u128>("func(20) / 5"), Ok(8));
        assert_eq!(ctx.evaluate::<u128>("func2(20, 20) / 5"), Ok(8));
    }

    #[test]
    fn test_eval_error() {
        let mut ctx = Context::new();

        ctx.add_function("func", |v1: u64, v2: u64| v1 + v2);

        let err = ctx
            .evaluate::<u64>("func(20, true)")
            .expect_err("Expected expression to return an error");

        pretty_assertions::assert_eq!(
            strip_ansi(format!("\n{err:?}")),
            r#"
Evaluation error
 ---> at line 1 column 10
  |
1 | func(20, true)
  |          ^^^^ func expects argument 1 to be int, got bool"#,
        );
    }
}
