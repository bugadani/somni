use somni_parser::parser::DefaultTypeSet;

use crate::{string_interner::StringIndex, OperatorError, Type, TypeSet};

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

impl ValueType for StringIndex {
    const BYTES: usize = 8;
    const TYPE: Type = Type::String;

    fn write(&self, to: &mut [u8]) {
        to.copy_from_slice(&self.0.to_le_bytes());
    }
    fn from_bytes(bytes: &[u8]) -> Self {
        StringIndex(u64::from_le_bytes(bytes.try_into().unwrap()) as usize)
    }
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

    ($type:ty, $kind:ident) => {
        impl<T> From<$type> for TypedValue<T>
        where
            T: TypeSet,
            T::Integer: ValueType,
            T::Float: ValueType,
        {
            fn from(value: $type) -> Self {
                TypedValue::$kind(value)
            }
        }
        impl<T> TryFrom<TypedValue<T>> for $type
        where
            T: TypeSet,
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

convert!(bool, Bool);
convert!(StringIndex, String);

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
