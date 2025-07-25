use somni_expr::{string_interner::StringIndex, Type, TypeSet};

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
    /// Returns the size of the type in bytes.
    fn size_of<T>(&self) -> usize
    where
        T: TypeSet,
        T::Integer: MemoryRepr,
        T::SignedInteger: MemoryRepr,
        T::Float: MemoryRepr;
}

impl TypeExt for somni_expr::Type {
    /// Returns the size of the type in bytes.
    fn size_of<T>(&self) -> usize
    where
        T: TypeSet,
        T::Integer: MemoryRepr,
        T::SignedInteger: MemoryRepr,
        T::Float: MemoryRepr,
    {
        match self {
            Type::Void => <() as MemoryRepr>::BYTES,
            Type::Int | Type::MaybeSignedInt => <T::Integer as MemoryRepr>::BYTES,
            Type::SignedInt => <T::SignedInteger as MemoryRepr>::BYTES,
            Type::Float => <T::Float as MemoryRepr>::BYTES,
            Type::Bool => <bool as MemoryRepr>::BYTES,
            Type::String => <StringIndex as MemoryRepr>::BYTES,
        }
    }
}
