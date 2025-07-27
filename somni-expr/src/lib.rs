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

macro_rules! for_each {
    // Any parenthesized set of choices, allows multiple matchers in the pattern
    ($(($pattern:tt) in [$( ($($choice:tt)*) ),*] => $code:tt;)*) => {
        $(
            macro_rules! inner { $pattern => $code; }

            $(
                inner!( $($choice)* );
            )*
        )*
    };
    // Single type, single matcher
    ($($pattern:tt in [$($choice:ty),*] => $code:tt;)*) => {
        $(
            macro_rules! inner { $pattern => $code; }

            $(
                inner!($choice);
            )*
        )*
    };
}

pub mod error;
#[doc(hidden)]
pub mod function;
#[doc(hidden)]
pub mod value;
mod visitor;

pub use function::{DynFunction, FunctionCallError};
pub use value::TypedValue;
pub use visitor::ExpressionVisitor;

use std::{
    cell::RefCell,
    collections::HashMap,
    fmt::{Debug, Display},
    rc::Rc,
};

use indexmap::IndexMap;
use somni_parser::{
    ast::{self, Expression, Function, Item, Program},
    parser::{self, parse, TypeSet as ParserTypeSet},
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

/// Defines the backing types for Somni types.
///
/// The [`LoadOwned`], [`Load`] and [`Store`] traits can be used to convert between Rust and Somni types.
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

    /// Converts an unsigned integer into a Rust usize.
    fn to_usize(v: Self::Integer) -> Result<usize, OperatorError>;

    /// Loads a string.
    fn load_string<'s>(&'s self, str: &'s Self::String) -> &'s str;

    /// Stores a string.
    fn store_string(&mut self, str: &str) -> Self::String;
}

for_each! {
    (($name:ident, $signed:ty)) in [(DefaultTypeSet, i64), (TypeSet32, i32), (TypeSet128, i128)] => {
        impl TypeSet for $name {
            type Parser = Self;

            type Integer = <Self::Parser as ParserTypeSet>::Integer;
            type SignedInteger = $signed;
            type Float = <Self::Parser as ParserTypeSet>::Float;
            type String = Box<str>;

            fn to_signed(v: Self::Integer) -> Result<Self::SignedInteger, OperatorError> {
                <$signed>::try_from(v).map_err(|_| OperatorError::RuntimeError)
            }

            fn to_usize(v: Self::Integer) -> Result<usize, OperatorError> {
                usize::try_from(v).map_err(|_| OperatorError::RuntimeError)
            }

            fn load_string<'s>(&'s self, str: &'s Self::String) -> &'s str {
                str
            }

            fn store_string(&mut self, str: &str) -> Self::String {
                str.to_string().into_boxed_str()
            }
        }
    };
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
    fn try_load_variable(&mut self, variable: &str) -> Option<TypedValue<T>>;

    /// Declares a variable in the context.
    fn declare(&mut self, variable: &str, value: TypedValue<T>);

    /// Assigns a new value to a variable in the context.
    fn assign_variable(&mut self, variable: &str, value: &TypedValue<T>) -> Result<(), Box<str>>;

    /// Assigns a new value to a variable in the context.
    fn assign_address(
        &mut self,
        address: TypedValue<T>,
        value: &TypedValue<T>,
    ) -> Result<(), Box<str>>;

    /// Returns the address of a variable in the context.
    fn address_of(&mut self, variable: &str) -> TypedValue<T>;

    /// Calls a function in the context.
    fn call_function(
        &mut self,
        function_name: &str,
        args: &[TypedValue<T>],
    ) -> Result<TypedValue<T>, FunctionCallError>;
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
impl Type {
    fn from_name(source: &str) -> Result<Self, Box<str>> {
        match source {
            "int" => Ok(Type::Int),
            "signed" => Ok(Type::SignedInt),
            "float" => Ok(Type::Float),
            "bool" => Ok(Type::Bool),
            "string" => Ok(Type::String),
            other => Err(format!("Unknown type `{other}`").into_boxed_str()),
        }
    }
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

/// State of an unevaluated global.
enum InitializerState {
    /// Untouched. Contains the item index of the global
    Unevaluated(usize),
    /// The global is being evaluated. This state is used to detect cycles.
    Evaluating,
}

struct Scope<T: TypeSet> {
    start_addr: usize,
    variables: IndexMap<String, TypedValue<T>>,
}

impl<T: TypeSet> Scope<T> {
    fn declare(&mut self, variable: &str, value: TypedValue<T>) {
        self.variables.insert(variable.to_string(), value);
    }
    fn store(&mut self, variable: &str, value: &TypedValue<T>) -> bool {
        if let Some(var) = self.variables.get_mut(variable) {
            *var = value.clone();
            true
        } else {
            false
        }
    }

    fn open(&self) -> Scope<T> {
        Scope {
            start_addr: self.start_addr + self.variables.len(),
            variables: IndexMap::new(),
        }
    }
}

struct ProgramData<'ctx, T: TypeSet> {
    source: &'ctx str,
    program: Program<T::Parser>,
    program_functions: HashMap<&'ctx str, usize>,
    // User-registered functions
    functions: RefCell<HashMap<&'ctx str, ExprFn<'ctx, T>>>,
}

/// The expression context, which holds variables, functions, and other state needed for evaluation.
pub struct Context<'ctx, T = DefaultTypeSet>
where
    T: TypeSet,
{
    program: Rc<ProgramData<'ctx, T>>,
    // Program state
    // ----
    /// Variable stack. Element 0 is the global scope.
    stack: Vec<Scope<T>>,
    // unevaluated globals
    initializers: HashMap<&'ctx str, InitializerState>,
    type_context: T,
}

impl<'ctx> Context<'ctx, DefaultTypeSet> {
    /// Creates a new context with [default types][DefaultTypeSet].
    pub fn new() -> Self {
        Self::new_with_types()
    }

    /// Loads the given program into a new context with [default types][DefaultTypeSet].
    pub fn parse(source: &'ctx str) -> Result<Self, ExpressionError<'ctx>> {
        Self::parse_with_types(source)
    }
}

const GLOBAL_VARIABLE: usize = usize::MAX - usize::MAX / 2;

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
        Self::new_from_program("", Program { items: vec![] })
    }

    /// Parses the given program into a new context. The type set must be specified when using this function.
    ///
    /// ```rust
    /// use somni_expr::{Context, TypeSet32};
    /// let mut ctx = Context::<TypeSet32>::parse_with_types("// program source comes here").unwrap();
    /// ```
    pub fn parse_with_types(source: &'ctx str) -> Result<Self, ExpressionError<'ctx>> {
        let program = parse::<T::Parser>(source).map_err(|e| ExpressionError {
            error: EvalError {
                message: format!("Failed to parse program: {e}").into_boxed_str(),
                location: Location::dummy(),
            },
            source,
        })?;

        Ok(Self::new_from_program(source, program))
    }

    /// Loads the given program into a new context.
    pub fn new_from_program(source: &'ctx str, program: Program<T::Parser>) -> Self {
        let mut program_functions = HashMap::new();
        let mut initializers = HashMap::new();
        // Extract data for O(1) function/initializer lookup
        for (idx, item) in program.items.iter().enumerate() {
            match item {
                ast::Item::Function(function) => {
                    program_functions.insert(function.name.source(source), idx);
                }
                ast::Item::GlobalVariable(global_variable) => {
                    initializers.insert(
                        global_variable.identifier.source(source),
                        InitializerState::Unevaluated(idx),
                    );
                }
                ast::Item::ExternFunction(_) => {}
            }
        }
        Self {
            program: Rc::new(ProgramData {
                source,
                program,
                program_functions,
                functions: RefCell::new(HashMap::new()),
            }),
            stack: vec![Scope {
                start_addr: 0,
                variables: IndexMap::new(),
            }],
            type_context: T::default(),
            initializers,
        }
    }

    fn evaluate_any_function_impl(
        &mut self,
        function_name: &Function<T::Parser>,
        args: &[TypedValue<T>],
    ) -> Result<TypedValue<T>, EvalError> {
        let source = self.program.clone().source;

        let stack_frame = self
            .stack
            .last()
            .expect("The global scope must always be present")
            .open();
        self.stack.push(stack_frame);

        let mut visitor = ExpressionVisitor::<_, T> {
            context: self,
            source,
            _marker: std::marker::PhantomData,
        };

        let result = visitor.visit_function(function_name, args);

        self.stack.pop();

        result
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

        self.evaluate_expr_any(expression, &ast)
    }

    /// Evaluates a parsed expression.
    ///
    /// The `source` should include the whole program source, from which the expression has been extracted.
    pub fn evaluate_expr_any(
        &mut self,
        source: &str,
        ast: &Expression<T::Parser>,
    ) -> Result<TypedValue<T>, EvalError> {
        let mut visitor = ExpressionVisitor::<Self, T> {
            context: self,
            source,
            _marker: std::marker::PhantomData,
        };

        visitor.visit_expression(ast)
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
        V::load_owned(self.type_context(), &result).ok_or_else(|| ExpressionError {
            error: EvalError {
                message: format!(
                    "Expression evaluates to {result_ty}, which cannot be converted to {}",
                    std::any::type_name::<V>()
                )
                .into_boxed_str(),
                location: Location::dummy(),
            },
            source: expression,
        })
    }

    /// Defines a new variable in the context.
    pub fn add_variable<V>(&mut self, name: &'ctx str, value: V)
    where
        V: Store<T>,
    {
        let stored = value.store(self.type_context());
        self.stack[0].declare(name, stored);
    }

    /// Adds a new function to the context.
    pub fn add_function<F, A>(&mut self, name: &'ctx str, func: F)
    where
        F: DynFunction<A, T> + 'ctx,
    {
        self.program
            .functions
            .borrow_mut()
            .insert(name, ExprFn::new(func));
    }

    fn lookup(&mut self, variable: &str) -> Option<(usize, TypedValue<T>)> {
        if self.stack.len() > 1 {
            let frame = self.stack.last().unwrap();
            if let Some((index, _, var)) = frame.variables.get_full(variable) {
                // Already evaluated / user provided
                return Some((index, var.clone()));
            }
        }

        {
            let global_frame = &self.stack[0];
            if let Some((index, _, var)) = global_frame.variables.get_full(variable) {
                // Already evaluated / user provided
                return Some((index | GLOBAL_VARIABLE, var.clone()));
            }
        }

        // Mark as "initializing" to detect potential cycles
        let state = self.initializers.get_mut(variable)?;
        let InitializerState::Unevaluated(idx) =
            std::mem::replace(state, InitializerState::Evaluating)
        else {
            return None;
        };

        // Get a reference to the initializer
        let program = self.program.clone();
        let Some(Item::GlobalVariable(global)) = program.program.items.get(idx) else {
            return None;
        };

        let value = self
            .evaluate_expr_any(self.program.source, &global.initializer)
            .ok()?;

        let global_frame = &mut self.stack[0];
        let (index, _) = global_frame
            .variables
            .insert_full(variable.to_string(), value.clone());

        Some((index | GLOBAL_VARIABLE, value))
    }
}

impl<T> ExprContext<T> for Context<'_, T>
where
    T: TypeSet,
{
    fn type_context(&mut self) -> &mut T {
        &mut self.type_context
    }

    // TODO: return Result
    fn try_load_variable(&mut self, variable: &str) -> Option<TypedValue<T>> {
        self.lookup(variable).map(|(_idx, var)| var)
    }

    fn address_of(&mut self, variable: &str) -> TypedValue<T> {
        let address = self
            .lookup(variable)
            .map(|(address, _var)| address)
            .unwrap();
        TypedValue::Int(<T::Integer as Integer>::from_usize(address))
    }

    /// Declares a variable in the context.
    fn declare(&mut self, variable: &str, value: TypedValue<T>) {
        self.stack.last_mut().unwrap().declare(variable, value);
    }

    /// Assigns a new value to a variable in the context.
    fn assign_variable(&mut self, variable: &str, value: &TypedValue<T>) -> Result<(), Box<str>> {
        if self.stack.last_mut().unwrap().store(variable, &value) {
            return Ok(());
        }
        if self.stack[0].store(variable, &value) {
            return Ok(());
        }

        Err(format!("Variable not found: {variable}").into_boxed_str())
    }

    fn assign_address(
        &mut self,
        address: TypedValue<T>,
        value: &TypedValue<T>,
    ) -> Result<(), Box<str>> {
        let TypedValue::Int(address) = address else {
            return Err(format!("Expected address, got {address:?}").into_boxed_str());
        };

        let address = T::to_usize(address)
            .map_err(|_| format!("Invalid address: {address:?}").into_boxed_str())?;

        if address & GLOBAL_VARIABLE != 0 {
            if let Some((_k, v)) = self.stack[0].variables.get_index_mut(address) {
                v.clone_from(value);
                return Ok(());
            }
        } else {
            for frame in self.stack.iter_mut().rev() {
                if frame.start_addr < address {
                    if let Some((_k, v)) = frame.variables.get_index_mut(address) {
                        v.clone_from(value);
                        return Ok(());
                    }
                    break;
                }
            }
        }

        Err(format!("Not a valid memory address: {address}").into_boxed_str())
    }

    fn call_function(
        &mut self,
        function_name: &str,
        args: &[TypedValue<T>],
    ) -> Result<TypedValue<T>, FunctionCallError> {
        let program = self.program.clone();
        let Some(fn_item) = self.program.program_functions.get(function_name) else {
            // Call out to a Rust function
            return match program.functions.borrow().get(function_name) {
                Some(func) => func.call(self.type_context(), args),
                None => Err(FunctionCallError::FunctionNotFound),
            };
        };

        // Call a Somni function
        let Some(ast::Item::Function(function)) = program.program.items.get(*fn_item) else {
            return Err(FunctionCallError::FunctionNotFound);
        };
        self.evaluate_any_function_impl(function, args)
            .map_err(|err| {
                FunctionCallError::Other(
                    format!(
                        "{:?}",
                        ExpressionError {
                            source: &self.program.source,
                            error: err,
                        }
                    )
                    .into_boxed_str(),
                )
            })
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
    use std::path::Path;

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
        assert_eq!(
            ctx.evaluate::<TypedValue>("func(20) / 5"),
            Ok(TypedValue::Int(8))
        );
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
    fn test_evaluate_function() {
        // TODO: run all the VM eval tests
        let mut ctx =
            Context::parse("fn multiply_with_global(a: int) -> int { return a * global; }")
                .unwrap();

        ctx.add_variable::<u64>("global", 3);

        assert_eq!(
            ctx.evaluate::<bool>("multiply_with_global(2) == 6"),
            Ok(true)
        );
        assert!(ctx
            .evaluate::<bool>("multiply_with_global(\"2\") == 6")
            .is_err());
    }

    #[test]
    fn run_eval_tests() {
        fn filter(path: &Path) -> bool {
            let Ok(env) = std::env::var("TEST_FILTER") else {
                // No filter set, walk folders and somni source files.
                return path.is_dir() || path.extension().map_or(false, |ext| ext == "sm");
            };

            Path::new(&env) == path
        }

        fn walk(dir: &Path, on_file: &impl Fn(&Path)) {
            for entry in std::fs::read_dir(dir)
                .unwrap_or_else(|_| panic!("Folder not found: {}", dir.display()))
                .flatten()
            {
                let path = entry.path();

                if !filter(&path) {
                    continue;
                }

                if path.is_file() {
                    on_file(&path);
                } else {
                    walk(&path, on_file);
                }
            }
        }

        fn run_eval_test(path: &Path) {
            fn parse(source: &str) -> Context<'_> {
                let mut context = Context::parse(&source).unwrap();

                context.add_function("add_from_rust", |a: u64, b: u64| -> i64 { (a + b) as i64 });
                context.add_function("assert", |a: bool| a); // No-op to test calling Rust functions from expressions
                context.add_function("reverse", |s: &str| s.chars().rev().collect::<String>());

                context
            }

            let source = std::fs::read_to_string(path).unwrap();

            let expressions = source
                .lines()
                .filter_map(|line| line.trim().strip_prefix("//@"))
                .collect::<Vec<_>>();

            let mut context = parse(&source);

            for expression in &expressions {
                let expression = if let Some(e) = expression.strip_prefix('+') {
                    // `//@+` preserves VM state (like changes to globals)
                    e.trim()
                } else {
                    // `//@` resets VM state (like changes to globals)
                    context = parse(&source);
                    expression
                };
                println!("Running `{expression}`");
                let value = context
                    .evaluate_any(expression)
                    .unwrap_or_else(|e| panic!("{}: {e:?}", path.display()));
                assert_eq!(
                    value,
                    TypedValue::Bool(true),
                    "Expression `{expression}` evaluated to {value:?}"
                );
            }
        }

        walk("../tests/eval".as_ref(), &|path| {
            run_eval_test(path);
        });
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
  |          ^^^^ func expects argument 1 to be u64, got bool"#,
        );
    }
}
