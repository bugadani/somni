use std::{cell::RefCell, fmt, rc::Rc};

use somni_expr::{
    OperatorError, Type, TypeSet,
    value::{LoadStore, ValueType},
};
use somni_parser::parser::TypeSet as ParserTypeSet;

use crate::string_interner::{StringIndex, StringInterner};

/// The value type produced by iterators inside the VM.
type ExprValue = somni_expr::TypedValue<VmTypeSet>;

/// A runtime iterator handle backed by a boxed Rust iterator.
///
/// Iterators are opaque, handle-sized values (like strings). The concrete iterator
/// lives in the type context's registry; a [`TypedValue::Iter`] holds the index into it.
#[derive(Clone)]
pub struct SomniIter {
    inner: Rc<RefCell<std::iter::Peekable<Box<dyn Iterator<Item = ExprValue>>>>>,
}

impl SomniIter {
    /// Creates an iterator from anything yielding Somni values.
    pub fn new<I>(iter: I) -> Self
    where
        I: IntoIterator<Item = ExprValue> + 'static,
    {
        let boxed: Box<dyn Iterator<Item = ExprValue>> = Box::new(iter.into_iter());
        Self {
            inner: Rc::new(RefCell::new(boxed.peekable())),
        }
    }

    fn has_next(&self) -> bool {
        self.inner.borrow_mut().peek().is_some()
    }

    fn next_value(&self) -> Option<ExprValue> {
        self.inner.borrow_mut().next()
    }
}

impl fmt::Debug for SomniIter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("<iter>")
    }
}

impl PartialEq for SomniIter {
    fn eq(&self, other: &Self) -> bool {
        Rc::ptr_eq(&self.inner, &other.inner)
    }
}

impl ValueType for SomniIter {
    const TYPE: Type = Type::Iter;
    type NegateOutput = Self;
}

impl LoadStore<VmTypeSet> for SomniIter {
    type Output<'s> = Self;

    fn load<'s>(ctx: &'s VmTypeSet, typed: &'s ExprValue) -> Option<Self::Output<'s>> {
        if let somni_expr::TypedValue::Iter(handle) = typed {
            ctx.get_iterator(*handle)
        } else {
            None
        }
    }

    fn store(&self, ctx: &mut VmTypeSet) -> ExprValue {
        let handle = ctx.store_iterator(self.clone());
        somni_expr::TypedValue::Iter(handle)
    }
}

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
            // Iterators are stored as an 8-byte handle into the iterator registry.
            Type::Iter => <u64 as MemoryRepr>::BYTES,
            // References are represented in the VM as raw 8-byte addresses.
            Type::Ref(_) => <u64 as MemoryRepr>::BYTES,
            // Struct sizes depend on their layout and must be resolved through the
            // `StructRegistry`; they can't be sized from the scalar type alone.
            Type::Struct => panic!("struct sizes must be resolved via the struct registry"),
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
    /// Represents an iterator handle.
    Iter(usize),
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
            TypedValue::Iter(inner) => somni_expr::TypedValue::Iter(inner),
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
            somni_expr::TypedValue::Iter(inner) => TypedValue::Iter(inner),
            // Structs and references are in-language-only in the VM; they never
            // cross the Rust boundary as a scalar `TypedValue`.
            somni_expr::TypedValue::Struct(_) | somni_expr::TypedValue::Ref(_) => {
                panic!("struct and reference values cannot cross the VM boundary")
            }
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
            TypedValue::Iter(_) => Type::Iter,
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
            TypedValue::Iter(value) => (*value as u64).write(to),
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
            Type::Iter => Self::Iter(u64::from_bytes(value) as usize),
            // References are stored as raw 8-byte addresses (like unsigned ints).
            Type::Ref(_) => Self::Int(u64::from_bytes(value)),
            Type::Struct => panic!("struct values cannot be read as a scalar TypedValue"),
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct VmTypeSet {
    strings: StringInterner,
    iterators: Vec<SomniIter>,
}

impl VmTypeSet {
    pub fn new(string_interner: StringInterner) -> Self {
        Self {
            strings: string_interner,
            iterators: Vec::new(),
        }
    }

    /// Drops all live iterators (called when the VM state is reset).
    pub fn clear_iterators(&mut self) {
        self.iterators.clear();
    }

    /// Registers a live iterator, returning its handle.
    pub fn store_iterator(&mut self, iter: SomniIter) -> usize {
        let handle = self.iterators.len();
        self.iterators.push(iter);
        handle
    }

    /// Returns a clone of the iterator handle (they share the same underlying iterator).
    pub fn get_iterator(&self, handle: usize) -> Option<SomniIter> {
        self.iterators.get(handle).cloned()
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
    // The VM stores values in flat memory, so an iterator is represented as a handle
    // into the type context's registry rather than the iterator object itself.
    type Iterator = usize;

    fn to_signed(v: Self::Integer) -> Result<Self::SignedInteger, OperatorError> {
        i64::try_from(v).map_err(|_| OperatorError::RuntimeError)
    }

    fn to_usize(v: Self::Integer) -> Result<usize, OperatorError> {
        usize::try_from(v).map_err(|_| OperatorError::RuntimeError)
    }

    fn int_from_usize(v: usize) -> Self::Integer {
        Self::Integer::try_from(v).unwrap()
    }

    fn load_string<'s>(&'s self, str: &'s Self::String) -> &'s str {
        self.strings.lookup(*str)
    }

    fn store_string(&mut self, str: &str) -> Self::String {
        self.strings.intern(str)
    }

    fn iter_has_next(&self, handle: &Self::Iterator) -> bool {
        self.iterators
            .get(*handle)
            .map(SomniIter::has_next)
            .unwrap_or(false)
    }

    fn iter_next(&self, handle: &Self::Iterator) -> Option<ExprValue> {
        self.iterators.get(*handle).and_then(SomniIter::next_value)
    }
}
