use somni_parser::parser::DefaultTypeSet;

use crate::{string_interner::StringIndex, ExprContext, OperatorError, Type, TypeSet};

#[doc(hidden)]
pub trait MemoryRepr: Sized + Copy + PartialEq {
    const BYTES: usize;

    fn write(&self, to: &mut [u8]);
    fn from_bytes(bytes: &[u8]) -> Self;
}

#[doc(hidden)]
pub trait ValueType: Sized + Clone + PartialEq + std::fmt::Debug {
    const TYPE: Type;

    type NegateOutput: ValueType;

    fn equals(_a: Self, _b: Self) -> Result<bool, OperatorError> {
        unimplemented!("Operation not supported")
    }
    fn less_than(_a: Self, _b: Self) -> Result<bool, OperatorError> {
        unimplemented!("Operation not supported")
    }

    fn less_than_or_equal(a: Self, b: Self) -> Result<bool, OperatorError> {
        let less = Self::less_than(a.clone(), b.clone())?;
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
    fn negate(_a: Self) -> Result<Self::NegateOutput, OperatorError> {
        unimplemented!("Operation not supported")
    }
}

impl MemoryRepr for () {
    const BYTES: usize = 0;

    fn write(&self, _to: &mut [u8]) {}

    fn from_bytes(_: &[u8]) -> Self {}
}

impl ValueType for () {
    type NegateOutput = Self;
    const TYPE: Type = Type::Void;
}

macro_rules! value_type_int {
    ($type:ty, $negate:ty, $kind:ident) => {
        impl MemoryRepr for $type {
            const BYTES: usize = std::mem::size_of::<$type>();

            fn write(&self, to: &mut [u8]) {
                to.copy_from_slice(&self.to_le_bytes());
            }

            fn from_bytes(bytes: &[u8]) -> Self {
                <$type>::from_le_bytes(bytes.try_into().unwrap())
            }
        }

        impl ValueType for $type {
            const TYPE: Type = Type::$kind;
            type NegateOutput = $negate;

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
            fn negate(a: Self) -> Result<Self::NegateOutput, OperatorError> {
                Ok(-(a as $negate))
            }
        }
    };
}

value_type_int!(u32, i32, Int);
value_type_int!(u64, i64, Int);
value_type_int!(u128, i128, Int);
value_type_int!(i32, i32, SignedInt);
value_type_int!(i64, i64, SignedInt);
value_type_int!(i128, i128, SignedInt);

impl MemoryRepr for f32 {
    const BYTES: usize = 4;

    fn write(&self, to: &mut [u8]) {
        to.copy_from_slice(&self.to_le_bytes());
    }

    fn from_bytes(bytes: &[u8]) -> Self {
        f32::from_le_bytes(bytes.try_into().unwrap())
    }
}
impl ValueType for f32 {
    const TYPE: Type = Type::Float;

    type NegateOutput = Self;

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
    fn negate(a: Self) -> Result<Self::NegateOutput, OperatorError> {
        Ok(-a)
    }
}

impl MemoryRepr for f64 {
    const BYTES: usize = 8;

    fn write(&self, to: &mut [u8]) {
        to.copy_from_slice(&self.to_le_bytes());
    }

    fn from_bytes(bytes: &[u8]) -> Self {
        f64::from_le_bytes(bytes.try_into().unwrap())
    }
}
impl ValueType for f64 {
    const TYPE: Type = Type::Float;

    type NegateOutput = Self;

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
    fn negate(a: Self) -> Result<Self::NegateOutput, OperatorError> {
        Ok(-a)
    }
}

impl MemoryRepr for bool {
    const BYTES: usize = 1;

    fn write(&self, to: &mut [u8]) {
        to.copy_from_slice(&[*self as u8]);
    }

    fn from_bytes(bytes: &[u8]) -> Self {
        bytes[0] != 0
    }
}

impl ValueType for bool {
    const TYPE: Type = Type::Bool;

    type NegateOutput = Self;

    fn equals(a: Self, b: Self) -> Result<bool, OperatorError> {
        Ok(a == b)
    }

    fn bitwise_and(a: Self, b: Self) -> Result<bool, OperatorError> {
        Ok(a & b)
    }

    fn bitwise_or(a: Self, b: Self) -> Result<bool, OperatorError> {
        Ok(a | b)
    }

    fn bitwise_xor(a: Self, b: Self) -> Result<bool, OperatorError> {
        Ok(a ^ b)
    }

    fn not(a: Self) -> Result<bool, OperatorError> {
        Ok(!a)
    }
}

impl MemoryRepr for StringIndex {
    const BYTES: usize = 8;

    fn write(&self, to: &mut [u8]) {
        to.copy_from_slice(&self.0.to_le_bytes());
    }
    fn from_bytes(bytes: &[u8]) -> Self {
        StringIndex(u64::from_le_bytes(bytes.try_into().unwrap()) as usize)
    }
}
impl ValueType for StringIndex {
    const TYPE: Type = Type::String;

    type NegateOutput = Self;

    fn equals(a: Self, b: Self) -> Result<bool, OperatorError> {
        Ok(a == b)
    }
}
impl ValueType for &str {
    const TYPE: Type = Type::String;

    type NegateOutput = Self;

    fn equals(a: Self, b: Self) -> Result<bool, OperatorError> {
        Ok(a == b)
    }
}
impl ValueType for String {
    const TYPE: Type = Type::String;

    type NegateOutput = Self;

    fn equals(a: Self, b: Self) -> Result<bool, OperatorError> {
        Ok(a == b)
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
    /// Represents an integer that may be signed or unsigned.
    MaybeSignedInt(T::Integer),
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
            TypedValue::MaybeSignedInt(_) => Type::MaybeSignedInt,
            TypedValue::SignedInt(_) => Type::SignedInt,
            TypedValue::Float(_) => Type::Float,
            TypedValue::Bool(_) => Type::Bool,
            TypedValue::String(_) => Type::String,
        }
    }
}

pub trait Load<T = DefaultTypeSet>: ValueType
where
    T: TypeSet,
    T::Integer: ValueType,
    T::Float: ValueType,
{
    type Output<'s>
    where
        T: 's;

    fn load(_ctx: &dyn ExprContext<T>, typed: TypedValue<T>) -> Option<Self::Output<'_>>;
}

pub trait Store<T = DefaultTypeSet>: ValueType
where
    T: TypeSet,
    T::Integer: ValueType,
    T::Float: ValueType,
{
    fn store(self, _ctx: &mut dyn ExprContext<T>) -> TypedValue<T>;
}

macro_rules! load {
    ($type:ty, [$($kind:ident),+] $(, $ts_kind:ident)?) => {
        impl<T> Load<T> for $type
        where
            T: TypeSet$(<$ts_kind = $type>)?,
            T::Integer: ValueType,
            T::Float: ValueType,
        {
            type Output<'s> = Self where T: 's;
            fn load(_ctx: &dyn ExprContext<T>, typed: TypedValue<T>) -> Option<Self> {
                $(
                    if let TypedValue::$kind(value) = typed {
                        return Some(value);
                    }
                )+

                None
            }
        }
    };
}
macro_rules! load_signed {
    ($type:ty, $kind:ident $(, $ts_kind:ident)?) => {
        impl<T> Load<T> for $type
        where
            T: TypeSet$(<$ts_kind = $type>)?,
            T::Integer: ValueType,
            T::Float: ValueType,
        {
            type Output<'s> = Self where T: 's;
            fn load(_ctx: &dyn ExprContext<T>, typed: TypedValue<T>) -> Option<Self> {
                if let TypedValue::SignedInt(value) = typed {
                    Some(value)
                } else if let TypedValue::MaybeSignedInt(value) = typed {
                    T::to_signed(value).ok()
                } else {
                    None
                }
            }
        }
    };
}
macro_rules! store {
    ($type:ty, $kind:ident $(, $ts_kind:ident)?) => {
        impl<T> Store<T> for $type
        where
            T: TypeSet$(<$ts_kind = $type>)?,
            T::Integer: ValueType,
            T::Float: ValueType,
        {
            fn store(self, _ctx: &mut dyn ExprContext<T>) -> TypedValue<T> {
                TypedValue::$kind(self)
            }
        }
    };
}

load!(u32, [Int, MaybeSignedInt], Integer);
load!(u64, [Int, MaybeSignedInt], Integer);
load!(u128, [Int, MaybeSignedInt], Integer);
load!(f32, [Float], Float);
load!(f64, [Float], Float);
load!(bool, [Bool]);
load!(StringIndex, [String]);

load_signed!(i32, SignedInt, SignedInteger);
load_signed!(i64, SignedInt, SignedInteger);
load_signed!(i128, SignedInt, SignedInteger);

store!(u32, Int, Integer);
store!(u64, Int, Integer);
store!(u128, Int, Integer);
store!(i32, SignedInt, SignedInteger);
store!(i64, SignedInt, SignedInteger);
store!(i128, SignedInt, SignedInteger);
store!(f32, Float, Float);
store!(f64, Float, Float);
store!(bool, Bool);
store!(StringIndex, String);

impl<T> Load<T> for ()
where
    T: TypeSet,
    T::Integer: ValueType,
    T::Float: ValueType,
{
    type Output<'s>
        = Self
    where
        T: 's;
    fn load(_ctx: &dyn ExprContext<T>, typed: TypedValue<T>) -> Option<Self> {
        if let TypedValue::Void = typed {
            Some(())
        } else {
            None
        }
    }
}
impl<T> Store<T> for ()
where
    T: TypeSet,
    T::Integer: ValueType,
    T::Float: ValueType,
{
    fn store(self, _ctx: &mut dyn ExprContext<T>) -> TypedValue<T> {
        TypedValue::Void
    }
}

impl<T> Store<T> for &str
where
    T: TypeSet,
    T::Integer: ValueType,
    T::Float: ValueType,
{
    fn store(self, ctx: &mut dyn ExprContext<T>) -> TypedValue<T> {
        let idx = ctx.intern_string(self);
        TypedValue::String(idx)
    }
}

impl<T> Store<T> for String
where
    T: TypeSet,
    T::Integer: ValueType,
    T::Float: ValueType,
{
    fn store(self, ctx: &mut dyn ExprContext<T>) -> TypedValue<T> {
        let idx = ctx.intern_string(&self);
        TypedValue::String(idx)
    }
}

impl<T> Load<T> for &str
where
    T: TypeSet,
    T::Integer: ValueType,
    T::Float: ValueType,
{
    type Output<'s>
        = &'s str
    where
        T: 's;

    fn load(ctx: &dyn ExprContext<T>, typed: TypedValue<T>) -> Option<Self::Output<'_>> {
        if let TypedValue::String(index) = typed {
            Some(ctx.load_interned_string(index))
        } else {
            None
        }
    }
}
