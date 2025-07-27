use somni_expr::{OperatorError, Type, TypeSet};
use somni_parser::parser::TypeSet as ParserTypeSet;

use crate::string_interner::{StringIndex, StringInterner};

#[doc(hidden)]
pub trait MemoryRepr: Sized + Copy + PartialEq {
    const BYTES: usize = std::mem::size_of::<Self>();

    fn write(&self, to: &mut [u8]);
    fn from_bytes(bytes: &[u8]) -> Self;
}

impl MemoryRepr for () {
    fn write(&self, _to: &mut [u8]) {}

    fn from_bytes(_: &[u8]) -> Self {}
}

impl MemoryRepr for bool {
    fn write(&self, to: &mut [u8]) {
        to.copy_from_slice(&[*self as u8]);
    }

    fn from_bytes(bytes: &[u8]) -> Self {
        bytes[0] != 0
    }
}

impl MemoryRepr for StringIndex {
    fn write(&self, to: &mut [u8]) {
        to.copy_from_slice(&self.0.to_le_bytes());
    }
    fn from_bytes(bytes: &[u8]) -> Self {
        StringIndex(u64::from_le_bytes(bytes.try_into().unwrap()) as usize)
    }
}

macro_rules! mem_repr {
    ($type:ty) => {
        impl MemoryRepr for $type {
            fn write(&self, to: &mut [u8]) {
                to.copy_from_slice(&self.to_le_bytes());
            }

            fn from_bytes(bytes: &[u8]) -> Self {
                <$type>::from_le_bytes(bytes.try_into().unwrap())
            }
        }
    };
}

mem_repr!(u32);
mem_repr!(u64);
mem_repr!(u128);
mem_repr!(i32);
mem_repr!(i64);
mem_repr!(i128);

mem_repr!(f32);
mem_repr!(f64);

pub(crate) trait TypeExt {
    /// Returns the size of the type in bytes in the VM.
    fn vm_size_of(&self) -> usize;
}

impl TypeExt for somni_expr::Type {
    fn vm_size_of(&self) -> usize {
        match self {
            Type::Void => <() as MemoryRepr>::BYTES,
            Type::Int | Type::MaybeSignedInt => {
                <<VmTypeSet as TypeSet>::Integer as MemoryRepr>::BYTES
            }
            Type::SignedInt => <<VmTypeSet as TypeSet>::SignedInteger as MemoryRepr>::BYTES,
            Type::Float => <<VmTypeSet as TypeSet>::Float as MemoryRepr>::BYTES,
            Type::Bool => <bool as MemoryRepr>::BYTES,
            Type::String => <<VmTypeSet as TypeSet>::String as MemoryRepr>::BYTES,
        }
    }
}

/// Represents any value inside the VM.
#[doc(hidden)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TypedValue {
    /// Represents no value.
    Void,
    /// Represents an integer that may be signed or unsigned.
    MaybeSignedInt(u64),
    /// Represents an unsigned integer.
    Int(u64),
    /// Represents a signed integer.
    SignedInt(i64),
    /// Represents a floating-point.
    Float(f64),
    /// Represents a boolean.
    Bool(bool),
    /// Represents a string.
    String(StringIndex),
}

impl From<TypedValue> for somni_expr::TypedValue<VmTypeSet> {
    fn from(value: TypedValue) -> Self {
        match value {
            TypedValue::Void => somni_expr::TypedValue::Void,
            TypedValue::MaybeSignedInt(inner) => somni_expr::TypedValue::MaybeSignedInt(inner),
            TypedValue::Int(inner) => somni_expr::TypedValue::Int(inner),
            TypedValue::SignedInt(inner) => somni_expr::TypedValue::SignedInt(inner),
            TypedValue::Float(inner) => somni_expr::TypedValue::Float(inner),
            TypedValue::Bool(inner) => somni_expr::TypedValue::Bool(inner),
            TypedValue::String(inner) => somni_expr::TypedValue::String(inner),
        }
    }
}
impl From<somni_expr::TypedValue<VmTypeSet>> for TypedValue {
    fn from(value: somni_expr::TypedValue<VmTypeSet>) -> Self {
        match value {
            somni_expr::TypedValue::Void => TypedValue::Void,
            somni_expr::TypedValue::MaybeSignedInt(inner) => TypedValue::MaybeSignedInt(inner),
            somni_expr::TypedValue::Int(inner) => TypedValue::Int(inner),
            somni_expr::TypedValue::SignedInt(inner) => TypedValue::SignedInt(inner),
            somni_expr::TypedValue::Float(inner) => TypedValue::Float(inner),
            somni_expr::TypedValue::Bool(inner) => TypedValue::Bool(inner),
            somni_expr::TypedValue::String(inner) => TypedValue::String(inner),
        }
    }
}

impl TypedValue {
    pub fn type_of(&self) -> somni_expr::Type {
        match self {
            TypedValue::Void => Type::Void,
            TypedValue::MaybeSignedInt(_) => Type::MaybeSignedInt,
            TypedValue::Int(_) => Type::Int,
            TypedValue::SignedInt(_) => Type::SignedInt,
            TypedValue::Float(_) => Type::Float,
            TypedValue::Bool(_) => Type::Bool,
            TypedValue::String(_) => Type::String,
        }
    }

    pub(crate) fn write(&self, to: &mut [u8]) {
        match self {
            TypedValue::Void => ().write(to),
            TypedValue::Int(value) | TypedValue::MaybeSignedInt(value) => value.write(to),
            TypedValue::SignedInt(value) => value.write(to),
            TypedValue::Float(value) => value.write(to),
            TypedValue::Bool(value) => value.write(to),
            TypedValue::String(value) => value.write(to),
        }
    }

    pub(crate) fn from_typed_bytes(ty: Type, value: &[u8]) -> TypedValue {
        match ty {
            Type::Void => Self::Void,
            Type::Int => Self::Int(<_>::from_bytes(value)),
            Type::MaybeSignedInt => Self::MaybeSignedInt(<_>::from_bytes(value)),
            Type::SignedInt => Self::SignedInt(<_>::from_bytes(value)),
            Type::Float => Self::Float(<_>::from_bytes(value)),
            Type::Bool => Self::Bool(<_>::from_bytes(value)),
            Type::String => Self::String(<_>::from_bytes(value)),
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct VmTypeSet(StringInterner);

impl VmTypeSet {
    pub fn new(string_interner: StringInterner) -> Self {
        Self(string_interner)
    }
}

impl ParserTypeSet for VmTypeSet {
    type Integer = u64;
    type Float = f64;
}

impl TypeSet for VmTypeSet {
    type Parser = Self;

    type Integer = u64;
    type SignedInteger = i64;
    type Float = f64;
    type String = StringIndex;

    fn to_signed(v: Self::Integer) -> Result<Self::SignedInteger, OperatorError> {
        i64::try_from(v).map_err(|_| OperatorError::RuntimeError)
    }

    fn to_usize(v: Self::Integer) -> Result<usize, OperatorError> {
        usize::try_from(v).map_err(|_| OperatorError::RuntimeError)
    }

    fn load_string<'s>(&'s self, str: &'s Self::String) -> &'s str {
        self.0.lookup(*str)
    }

    fn store_string(&mut self, str: &str) -> Self::String {
        self.0.intern(str)
    }
}
