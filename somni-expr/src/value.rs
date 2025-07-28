use somni_parser::parser::DefaultTypeSet;

use crate::{OperatorError, Type, TypeSet};

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

impl ValueType for () {
    type NegateOutput = Self;
    const TYPE: Type = Type::Void;
}

macro_rules! value_type_int {
    ($type:ty, $negate:ty, $kind:ident) => {
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

for_each! {
    ($string:ty) in [&str, String, Box<str>] => {
        impl ValueType for $string {
            const TYPE: Type = Type::String;
            type NegateOutput = Self;

            fn equals(a: Self, b: Self) -> Result<bool, OperatorError> {
                Ok(a == b)
            }
        }
    };
}

/// Represents any value in the expression language.
#[derive(Debug)]
pub enum TypedValue<T: TypeSet = DefaultTypeSet> {
    /// Represents no value.
    Void,
    /// Represents an integer that may be signed or unsigned.
    ///
    /// MaybeSignedInt can compare equal with Int and SignedInt.
    MaybeSignedInt(T::Integer),
    /// Represents an unsigned integer.
    Int(T::Integer),
    /// Represents a signed integer.
    SignedInt(T::SignedInteger),
    /// Represents a floating-point.
    Float(T::Float),
    /// Represents a boolean.
    Bool(bool),
    /// Represents a string.
    String(T::String),
}

impl<T: TypeSet> PartialEq for TypedValue<T> {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::MaybeSignedInt(lhs), Self::MaybeSignedInt(rhs) | Self::Int(rhs)) => lhs == rhs,
            (Self::Int(lhs), Self::MaybeSignedInt(rhs) | Self::Int(rhs)) => lhs == rhs,
            (Self::SignedInt(lhs), Self::SignedInt(rhs)) => lhs == rhs,
            (Self::SignedInt(lhs), Self::MaybeSignedInt(rhs)) => {
                T::to_signed(*rhs).map(|rhs| rhs == *lhs).unwrap_or(false)
            }
            (Self::MaybeSignedInt(lhs), Self::SignedInt(rhs)) => {
                T::to_signed(*lhs).map(|lhs| lhs == *rhs).unwrap_or(false)
            }
            (Self::Float(lhs), Self::Float(rhs)) => lhs == rhs,
            (Self::Bool(lhs), Self::Bool(rhs)) => lhs == rhs,
            (Self::String(lhs), Self::String(rhs)) => lhs == rhs,
            _ => core::mem::discriminant(self) == core::mem::discriminant(other),
        }
    }
}

impl<T: TypeSet> Clone for TypedValue<T> {
    fn clone(&self) -> Self {
        match self {
            Self::Void => Self::Void,
            Self::MaybeSignedInt(inner) => Self::MaybeSignedInt(*inner),
            Self::Int(inner) => Self::Int(*inner),
            Self::SignedInt(inner) => Self::SignedInt(*inner),
            Self::Float(inner) => Self::Float(*inner),
            Self::Bool(inner) => Self::Bool(*inner),
            Self::String(inner) => Self::String(inner.clone()),
        }
    }
}

impl<T: TypeSet> TypedValue<T> {
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

/// Loads an owned Rust value from a TypedValue.
pub trait LoadOwned<T: TypeSet = DefaultTypeSet> {
    type Output;

    /// Loads an owned value from the given type context.
    fn load_owned(_ctx: &T, typed: &TypedValue<T>) -> Option<Self::Output>;
}

/// Loads a borrowed Rust value from a TypedValue.
pub trait Load<T: TypeSet = DefaultTypeSet> {
    type Output<'s>
    where
        T: 's;

    /// Loads a borrowed value from the given type context.
    fn load<'s>(_ctx: &'s T, typed: &'s TypedValue<T>) -> Option<Self::Output<'s>>;
}

/// Stores a Rust value into a TypedValue.
pub trait Store<T: TypeSet = DefaultTypeSet> {
    /// Stores a Rust value into a TypedValue using the given type context.
    fn store(&self, _ctx: &mut T) -> TypedValue<T>;
}

macro_rules! load_from_owned {
    ($type:ty $(,$kind:ident)?) => {
        impl<T: TypeSet $(<$kind = Self>)?> Load<T> for $type {
            type Output<'s> = Self;

            fn load(ctx: &T, typed: &TypedValue<T>) -> Option<Self> {
                <Self as LoadOwned<T>>::load_owned(ctx, typed)
            }
        }
    }
}

macro_rules! store {
    ($type:ty, $kind:ident $(, $ts_kind:ident)?) => {
        impl<T: TypeSet$(<$ts_kind = $type>)?> Store<T> for $type {
            fn store(&self, _ctx: &mut T) -> TypedValue<T> {
                TypedValue::$kind(*self)
            }
        }
    };
}

// LoadOwned
impl<T: TypeSet> LoadOwned<T> for TypedValue<T> {
    type Output = Self;
    fn load_owned(_ctx: &T, typed: &TypedValue<T>) -> Option<Self> {
        Some(typed.clone())
    }
}
impl<T: TypeSet> LoadOwned<T> for () {
    type Output = Self;
    fn load_owned(_ctx: &T, typed: &TypedValue<T>) -> Option<Self> {
        if let TypedValue::Void = typed {
            Some(())
        } else {
            None
        }
    }
}
impl<T: TypeSet> LoadOwned<T> for bool {
    type Output = Self;
    fn load_owned(_ctx: &T, typed: &TypedValue<T>) -> Option<Self> {
        if let TypedValue::Bool(value) = typed {
            Some(*value)
        } else {
            None
        }
    }
}

for_each! {
    // Unsigned integers
    ($type:ty) in [u32, u64, u128] => {
        impl<T: TypeSet<Integer = Self>> LoadOwned<T> for $type {
            type Output = Self;
            fn load_owned(_ctx: &T, typed: &TypedValue<T>) -> Option<Self::Output> {
                match typed {
                    TypedValue::MaybeSignedInt(value) => Some(*value),
                    TypedValue::Int(value) => Some(*value),
                    _ => None,
                }
            }
        }
    };

    // Signed integers
    ($type:ty) in [i32, i64, i128] => {
        impl<T: TypeSet<SignedInteger = Self>> LoadOwned<T> for $type {
            type Output = Self;
            fn load_owned(_ctx: &T, typed: &TypedValue<T>) -> Option<Self::Output> {
                match typed {
                    TypedValue::MaybeSignedInt(value) => T::to_signed(*value).ok(),
                    TypedValue::SignedInt(value) => Some(*value),
                    _ => None,
                }
            }
        }
    };

    // Floats
    ($type:ty) in [f32, f64] => {
        impl<T: TypeSet<Float = Self>> LoadOwned<T> for $type {
            type Output = Self;
            fn load_owned(_ctx: &T, typed: &TypedValue<T>) -> Option<Self::Output> {
                match typed {
                    TypedValue::Float(value) => Some(*value),
                    _ => None,
                }
            }
        }
    };

    // Strings
    ($type:ty) in [String, Box<str>] => {
        impl<T: TypeSet> LoadOwned<T> for $type {
            type Output = Self;
            fn load_owned(ctx: &T, typed: &TypedValue<T>) -> Option<Self::Output> {
                <&str as Load<T>>::load(ctx, typed).map(Into::into)
            }
        }
    };
}

// Load
impl<T: TypeSet> Load<T> for &str {
    type Output<'s>
        = &'s str
    where
        T: 's;

    fn load<'s>(ctx: &'s T, typed: &'s TypedValue<T>) -> Option<Self::Output<'s>> {
        if let TypedValue::String(index) = typed {
            Some(ctx.load_string(index))
        } else {
            None
        }
    }
}

for_each! {
    ($type:ty) in [(), bool, TypedValue<T>, String, Box<str>] => { load_from_owned!($type); };

    // Unsigned integers
    ($type:ty) in [u32, u64, u128] => { load_from_owned!($type, Integer); };

    // Signed integers
    ($type:ty) in [i32, i64, i128] => { load_from_owned!($type, SignedInteger); };

    // Floats
    ($type:ty) in [f32, f64] => { load_from_owned!($type, Float); };
}

// Store
impl<T: TypeSet> Store<T> for TypedValue<T> {
    fn store(&self, _ctx: &mut T) -> TypedValue<T> {
        self.clone()
    }
}

impl<T: TypeSet> Store<T> for () {
    fn store(&self, _ctx: &mut T) -> TypedValue<T> {
        TypedValue::Void
    }
}
store!(bool, Bool);

for_each! {
    // Unsigned integers
    ($type:ty) in [u32, u64, u128] => { store!($type, Int, Integer); };

    // Signed integers
    ($type:ty) in [i32, i64, i128] => { store!($type, SignedInt, SignedInteger); };

    // Floats
    ($type:ty) in [f32, f64] => { store!($type, Float, Float); };

    // Strings
    ($type:ty) in [&str, String, Box<str>] => {
        impl<T: TypeSet> Store<T> for $type {
            fn store(&self, ctx: &mut T) -> TypedValue<T> {
                TypedValue::String(ctx.store_string(self))
            }
        }
    };
}
