use somni_parser::parser::DefaultTypeSet;

use crate::{string_interner::StringIndex, ExprContext, OperatorError, Type, TypeSet};

#[doc(hidden)]
pub trait MemoryRepr: Sized + Copy + PartialEq {
    const BYTES: usize;

    fn write(&self, to: &mut [u8]);
    fn from_bytes(bytes: &[u8]) -> Self;
}

#[doc(hidden)]
pub trait ValueType: Sized + Copy + PartialEq {
    const TYPE: Type;

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

impl MemoryRepr for () {
    const BYTES: usize = 0;

    fn write(&self, _to: &mut [u8]) {}

    fn from_bytes(_: &[u8]) -> Self {}
}

impl ValueType for () {
    const TYPE: Type = Type::Void;
}

macro_rules! value_type_int {
    ($type:ty, $kind:ident) => {
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
    fn load(_ctx: &mut dyn ExprContext<T>, typed: TypedValue<T>) -> Option<Self>;
}

pub trait Store<T = DefaultTypeSet>: ValueType
where
    T: TypeSet,
    T::Integer: ValueType,
    T::Float: ValueType,
{
    fn store(self, _ctx: &mut dyn ExprContext<T>) -> TypedValue<T>;
}

macro_rules! convert {
    ($type:ty, $ts_kind:ident, $kind:ident) => {
        impl<T> Load<T> for $type
        where
            T: TypeSet<$ts_kind = $type>,
            T::Integer: ValueType,
            T::Float: ValueType,
        {
            fn load(_ctx: &mut dyn ExprContext<T>, typed: TypedValue<T>) -> Option<Self> {
                if let TypedValue::$kind(value) = typed {
                    Some(value)
                } else {
                    None
                }
            }
        }
        impl<T> Store<T> for $type
        where
            T: TypeSet<$ts_kind = $type>,
            T::Integer: ValueType,
            T::Float: ValueType,
        {
            fn store(self, _ctx: &mut dyn ExprContext<T>) -> TypedValue<T> {
                TypedValue::$kind(self)
            }
        }
    };

    ($type:ty, $kind:ident) => {
        impl<T> Load<T> for $type
        where
            T: TypeSet,
            T::Integer: ValueType,
            T::Float: ValueType,
        {
            fn load(_ctx: &mut dyn ExprContext<T>, typed: TypedValue<T>) -> Option<Self> {
                if let TypedValue::$kind(value) = typed {
                    Some(value)
                } else {
                    None
                }
            }
        }
        impl<T> Store<T> for $type
        where
            T: TypeSet,
            T::Integer: ValueType,
            T::Float: ValueType,
        {
            fn store(self, _ctx: &mut dyn ExprContext<T>) -> TypedValue<T> {
                TypedValue::$kind(self)
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

convert!(bool, Bool);
convert!(StringIndex, String);

impl<T> Load<T> for ()
where
    T: TypeSet,
    T::Integer: ValueType,
    T::Float: ValueType,
{
    fn load(_ctx: &mut dyn ExprContext<T>, typed: TypedValue<T>) -> Option<Self> {
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
