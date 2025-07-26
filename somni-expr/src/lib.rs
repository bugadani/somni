//! # Somni expression evaluation Library
//!
//! This library provides tools for evaluating expressions.
//!
//! ## Overview
//!
//! The expression language includes:
//!
//! - Literals: integers, floats, booleans, strings.
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
pub mod function;
#[doc(hidden)]
pub mod value;

pub use function::{DynFunction, FunctionCallError};
pub use value::TypedValue;

use std::{
    collections::HashMap,
    fmt::{Debug, Display},
};

use indexmap::IndexMap;
use somni_parser::{
    ast::{self, Expression},
    lexer,
    parser::{self, TypeSet as ParserTypeSet},
    Location,
};

use crate::{
    error::MarkInSource,
    function::ExprFn,
    value::{Load, LoadOwned, Store, ValueType},
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
pub trait TypeSet: Sized + Default + Debug + 'static {
    /// The typeset that will be used to parse source code.
    type Parser: ParserTypeSet<Integer = Self::Integer, Float = Self::Float>;

    /// The type of unsigned integers in this type set.
    type Integer: Copy
        + ValueType<NegateOutput: Load<Self> + Store<Self>>
        + Load<Self>
        + Store<Self>
        + Integer;

    /// The type of signed integers in this type set.
    type SignedInteger: Copy
        + ValueType<NegateOutput: Load<Self> + Store<Self>>
        + Load<Self>
        + Store<Self>;

    /// The type of floating point numbers in this type set.
    type Float: Copy + ValueType<NegateOutput: Load<Self> + Store<Self>> + Load<Self> + Store<Self>;

    /// The type of a string in this type set.
    type String: ValueType<NegateOutput: Load<Self> + Store<Self>> + Load<Self> + Store<Self>;

    /// Converts an unsigned integer into a signed integer.
    fn to_signed(v: Self::Integer) -> Result<Self::SignedInteger, OperatorError>;

    /// Loads a string.
    fn load_string<'s>(&'s self, str: &'s Self::String) -> &'s str;

    /// Stores a string.
    fn store_string(&mut self, str: &str) -> Self::String;
}

impl TypeSet for DefaultTypeSet {
    type Parser = Self;

    type Integer = <Self::Parser as ParserTypeSet>::Integer;
    type SignedInteger = i64;
    type Float = <Self::Parser as ParserTypeSet>::Float;
    type String = String;

    fn to_signed(v: Self::Integer) -> Result<Self::SignedInteger, OperatorError> {
        i64::try_from(v).map_err(|_| OperatorError::RuntimeError)
    }

    fn load_string<'s>(&'s self, str: &'s Self::String) -> &'s str {
        str
    }

    fn store_string(&mut self, str: &str) -> Self::String {
        str.to_string()
    }
}

impl TypeSet for TypeSet32 {
    type Parser = Self;

    type Integer = <Self::Parser as ParserTypeSet>::Integer;
    type SignedInteger = i32;
    type Float = <Self::Parser as ParserTypeSet>::Float;
    type String = String;

    fn to_signed(v: Self::Integer) -> Result<Self::SignedInteger, OperatorError> {
        i32::try_from(v).map_err(|_| OperatorError::RuntimeError)
    }

    fn load_string<'s>(&'s self, str: &'s Self::String) -> &'s str {
        str
    }

    fn store_string(&mut self, str: &str) -> Self::String {
        str.to_string()
    }
}

impl TypeSet for TypeSet128 {
    type Parser = Self;

    type Integer = <Self::Parser as ParserTypeSet>::Integer;
    type SignedInteger = i128;
    type Float = <Self::Parser as ParserTypeSet>::Float;
    type String = String;

    fn to_signed(v: Self::Integer) -> Result<Self::SignedInteger, OperatorError> {
        i128::try_from(v).map_err(|_| OperatorError::RuntimeError)
    }

    fn load_string<'s>(&'s self, str: &'s Self::String) -> &'s str {
        str
    }

    fn store_string(&mut self, str: &str) -> Self::String {
        str.to_string()
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
        pub(crate) fn $method(ctx: &mut T, lhs: Self, rhs: Self) -> Result<Self, OperatorError> {
            let result = match (lhs, rhs) {
                (Self::Bool(value), Self::Bool(other)) => {
                    ValueType::$method(value, other)?.store(ctx)
                }
                (Self::Int(value), Self::Int(other)) => {
                    ValueType::$method(value, other)?.store(ctx)
                }
                (Self::SignedInt(value), Self::SignedInt(other)) => {
                    ValueType::$method(value, other)?.store(ctx)
                }
                (Self::MaybeSignedInt(value), Self::MaybeSignedInt(other)) => {
                    match ValueType::$method(value, other)?.store(ctx) {
                        Self::Int(v) => Self::MaybeSignedInt(v),
                        other => other,
                    }
                }
                (Self::Float(value), Self::Float(other)) => {
                    ValueType::$method(value, other)?.store(ctx)
                }
                (Self::String(value), Self::String(other)) => {
                    ValueType::$method(value, other)?.store(ctx)
                }
                (Self::Int(value), Self::MaybeSignedInt(other)) => {
                    ValueType::$method(value, other)?.store(ctx)
                }
                (Self::MaybeSignedInt(value), Self::Int(other)) => {
                    ValueType::$method(value, other)?.store(ctx)
                }
                (Self::SignedInt(value), Self::MaybeSignedInt(other)) => {
                    ValueType::$method(value, T::to_signed(other)?)?.store(ctx)
                }
                (Self::MaybeSignedInt(value), Self::SignedInt(other)) => {
                    ValueType::$method(T::to_signed(value)?, other)?.store(ctx)
                }
                _ => return Err(OperatorError::TypeError),
            };

            Ok(result)
        }
    };
}

macro_rules! dispatch_unary {
    ($method:ident) => {
        pub(crate) fn $method(ctx: &mut T, operand: Self) -> Result<Self, OperatorError> {
            match operand {
                Self::Bool(value) => Ok(ValueType::$method(value)?.store(ctx)),
                Self::Int(value) | Self::MaybeSignedInt(value) => {
                    Ok(ValueType::$method(value)?.store(ctx))
                }
                Self::SignedInt(value) => Ok(ValueType::$method(value)?.store(ctx)),
                Self::Float(value) => Ok(ValueType::$method(value)?.store(ctx)),
                Self::String(value) => Ok(ValueType::$method(value)?.store(ctx)),
                _ => return Err(OperatorError::TypeError),
            }
        }
    };
}

impl<T> TypedValue<T>
where
    T: TypeSet,
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
    dispatch_unary!(negate);
}

/// An expression context that provides the necessary environment for evaluating expressions.
pub trait ExprContext<T = DefaultTypeSet>
where
    T: TypeSet,
{
    /// Returns a reference to the `TypeSet`.
    fn type_context(&mut self) -> &mut T;

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
        expression: &Expression<T::Parser>,
    ) -> Result<TypedValue<T>, EvalError> {
        let result = match expression {
            Expression::Variable { variable } => self.visit_variable(variable)?,
            Expression::Literal { value } => match &value.value {
                ast::LiteralValue::Integer(value) => TypedValue::<T>::MaybeSignedInt(*value),
                ast::LiteralValue::Float(value) => TypedValue::<T>::Float(*value),
                ast::LiteralValue::String(value) => value.store(self.context.type_context()),
                ast::LiteralValue::Boolean(value) => TypedValue::<T>::Bool(*value),
            },
            Expression::UnaryOperator { name, operand } => match name.source(self.source) {
                "!" => {
                    let operand = self.visit_expression(operand)?;

                    match TypedValue::<T>::not(self.context.type_context(), operand) {
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
                    let ty = value.type_of();
                    TypedValue::<T>::negate(self.context.type_context(), value).map_err(|e| {
                        EvalError {
                            message: format!("Cannot negate {ty}: {e}").into_boxed_str(),
                            location: operand.location(),
                        }
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
                let type_context = self.context.type_context();
                let result = match name.source(self.source) {
                    "+" => TypedValue::<T>::add(type_context, lhs, rhs),
                    "-" => TypedValue::<T>::subtract(type_context, lhs, rhs),
                    "*" => TypedValue::<T>::multiply(type_context, lhs, rhs),
                    "/" => TypedValue::<T>::divide(type_context, lhs, rhs),
                    "<" => TypedValue::<T>::less_than(type_context, lhs, rhs),
                    ">" => TypedValue::<T>::less_than(type_context, rhs, lhs),
                    "<=" => TypedValue::<T>::less_than_or_equal(type_context, lhs, rhs),
                    ">=" => TypedValue::<T>::less_than_or_equal(type_context, rhs, lhs),
                    "==" => TypedValue::<T>::equals(type_context, lhs, rhs),
                    "!=" => TypedValue::<T>::not_equals(type_context, lhs, rhs),
                    "|" => TypedValue::<T>::bitwise_or(type_context, lhs, rhs),
                    "^" => TypedValue::<T>::bitwise_xor(type_context, lhs, rhs),
                    "&" => TypedValue::<T>::bitwise_and(type_context, lhs, rhs),
                    "<<" => TypedValue::<T>::shift_left(type_context, lhs, rhs),
                    ">>" => TypedValue::<T>::shift_right(type_context, lhs, rhs),

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
                let mut args = Vec::with_capacity(arguments.len());
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

/// A type in the Somni language.
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub enum Type {
    /// Represents no value, used for e.g. functions that do not return a value.
    Void,
    /// Represents integer that may be signed or unsigned.
    MaybeSignedInt,
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

impl Display for Type {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Type::Void => write!(f, "void"),
            Type::MaybeSignedInt => write!(f, "{{int/signed}}"),
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
{
    variables: IndexMap<String, TypedValue<T>>,
    functions: HashMap<String, ExprFn<'ctx, T>>,
    marker: std::marker::PhantomData<T>,
    type_context: T,
}

impl<T> ExprContext<T> for Context<'_, T>
where
    T: TypeSet,
{
    fn type_context(&mut self) -> &mut T {
        &mut self.type_context
    }

    fn try_load_variable(&self, variable: &str) -> Option<TypedValue<T>> {
        self.variables.get(variable).cloned()
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
                let retval = func.call(self.type_context(), args);
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
            marker: std::marker::PhantomData,
            type_context: T::default(),
        }
    }

    fn evaluate_any_impl(&mut self, expression: &str) -> Result<TypedValue<T>, EvalError> {
        // TODO: we can allow new globals to be defined in the expression, but that would require
        // storing a copy of the original globals, so that they can be reset?
        let ast = match parser::parse_expression::<T::Parser>(expression) {
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
    pub fn evaluate<'s, V>(
        &'s mut self,
        expression: &'s str,
    ) -> Result<V::Output, ExpressionError<'s>>
    where
        V: LoadOwned<T>,
    {
        let result = self.evaluate_any(expression)?;
        let result_ty = result.type_of();
        V::load(self.type_context(), &result).ok_or_else(|| ExpressionError {
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
        V: Store<T>,
    {
        let stored = value.store(self.type_context());
        self.variables.insert(name.to_string(), stored);
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

        ctx.add_variable::<i64>("signed", 30);
        ctx.add_variable::<u64>("value", 30);
        ctx.add_function("func", |v: u64| 2 * v);
        ctx.add_function("func2", |v1: u64, v2: u64| v1 + v2);
        ctx.add_function("five", || "five");
        ctx.add_function("is_five", |num: &str| num == "five");
        ctx.add_function("concatenate", |a: &str, b: &str| format!("{a}{b}"));

        assert_eq!(ctx.evaluate::<bool>("value / 5 == 6"), Ok(true));
        assert_eq!(ctx.evaluate::<bool>("five() == \"five\""), Ok(true));
        assert_eq!(
            ctx.evaluate::<bool>("is_five(five()) != is_five(\"six\")"),
            Ok(true)
        );
        assert_eq!(ctx.evaluate::<u64>("func(20) / 5"), Ok(8));
        assert_eq!(ctx.evaluate::<u64>("func2(20, 20) / 5"), Ok(8));
        assert_eq!(ctx.evaluate::<bool>("true & false"), Ok(false));
        assert_eq!(ctx.evaluate::<bool>("!true"), Ok(false));
        assert_eq!(ctx.evaluate::<bool>("false | false"), Ok(false));
        assert_eq!(ctx.evaluate::<bool>("true ^ true"), Ok(false));
        assert_eq!(ctx.evaluate::<u64>("!0x1111"), Ok(0xFFFF_FFFF_FFFF_EEEE));
        assert_eq!(
            ctx.evaluate::<String>("concatenate(five(), \"six\")"),
            Ok(String::from("fivesix"))
        );
        assert_eq!(ctx.evaluate::<bool>("signed * 2 == 60"), Ok(true));
    }

    #[test]
    fn test_evaluating_exprs_with_u32() {
        let mut ctx = Context::<TypeSet32>::new_with_types();

        ctx.add_variable::<u32>("value", 30);
        ctx.add_function("func", |v: u32| 2 * v);
        ctx.add_function("func2", |v1: u32, v2: u32| v1 + v2);

        assert_eq!(ctx.evaluate::<bool>("value / 5 == 6"), Ok(true));
        assert_eq!(ctx.evaluate::<u32>("func(20) / 5"), Ok(8));
        assert_eq!(ctx.evaluate::<u32>("func2(20, 20) / 5"), Ok(8));
    }

    #[test]
    fn test_evaluating_exprs_with_u128() {
        let mut ctx = Context::<TypeSet128>::new_with_types();

        ctx.add_variable::<u128>("value", 30);
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
