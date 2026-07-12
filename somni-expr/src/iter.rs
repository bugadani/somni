//! Iterator support for type sets.
//!
//! Iterators are opaque values ([`TypedValue::Iter`]). The concrete iterator object
//! is carried directly by the value.
//!
//! The default type sets do not support iteration (their [`TypeSet::Iterator`] is the
//! uninhabited [`NoIterator`](crate::NoIterator)). [`WithIterator`] wraps any
//! [`TypeSet`] and swaps in a real, reference-counted iterator ([`SomniIterator`]),
//! enabling `for` loops (and any host function that produces a [`SomniIterator`]) for
//! the wrapped type set.

use std::{cell::RefCell, fmt, rc::Rc};

use crate::{TypeSet, TypedValue, value::LoadStore};

/// A boxed, peekable iterator yielding values of the *inner* type set.
///
/// The inner type set `T` and its [`WithIterator`] wrapper share all of their
/// scalar associated types, so values convert between the two for free.
type BoxedIter<T> = std::iter::Peekable<Box<dyn Iterator<Item = TypedValue<T>>>>;

/// A live iterator value, backed by a reference-counted Rust iterator.
///
/// This is the [`TypeSet::Iterator`] type used by [`WithIterator`]: a
/// [`TypedValue::Iter`] carries one of these directly (cloning shares the same
/// underlying iterator). Return one from a function registered on a
/// [`Context`](crate::Context) that uses a [`WithIterator`] type set to make it
/// iterable with a `for` loop.
pub struct SomniIterator<T: TypeSet> {
    inner: Rc<RefCell<BoxedIter<T>>>,
}

impl<T: TypeSet> SomniIterator<T> {
    /// Creates an iterator from anything that yields Somni values.
    pub fn new<I>(iter: I) -> Self
    where
        I: IntoIterator<Item = TypedValue<T>> + 'static,
    {
        let boxed: Box<dyn Iterator<Item = TypedValue<T>>> = Box::new(iter.into_iter());
        Self {
            inner: Rc::new(RefCell::new(boxed.peekable())),
        }
    }

    /// Returns whether the iterator can yield another value.
    #[doc(hidden)]
    pub fn has_next(&self) -> bool {
        self.inner.borrow_mut().peek().is_some()
    }

    /// Advances the iterator, returning its next value (in the inner type set).
    #[doc(hidden)]
    pub fn next_value(&self) -> Option<TypedValue<T>> {
        self.inner.borrow_mut().next()
    }
}

impl<T: TypeSet> Clone for SomniIterator<T> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }
}

impl<T: TypeSet> fmt::Debug for SomniIterator<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("<iter>")
    }
}

impl<T: TypeSet> PartialEq for SomniIterator<T> {
    fn eq(&self, other: &Self) -> bool {
        Rc::ptr_eq(&self.inner, &other.inner)
    }
}

impl<T> LoadStore<WithIterator<T>> for SomniIterator<T>
where
    T: TypeSet,
    WithIterator<T>: TypeSet<Iterator = SomniIterator<T>>,
{
    type Output<'s> = Self;

    fn load<'s>(
        _ctx: &'s WithIterator<T>,
        typed: &'s TypedValue<WithIterator<T>>,
    ) -> Option<Self::Output<'s>> {
        if let TypedValue::Iter(iter) = typed {
            Some(iter.clone())
        } else {
            None
        }
    }

    fn store(&self, _ctx: &mut WithIterator<T>) -> TypedValue<WithIterator<T>> {
        TypedValue::Iter(self.clone())
    }
}

/// A [`TypeSet`] wrapper that adds iterator support to the wrapped type set `T`.
///
/// All scalar behavior is delegated to `T`; this only replaces the inert
/// [`NoIterator`](crate::NoIterator) with a real [`SomniIterator`] and implements the
/// [`iter_has_next`](TypeSet::iter_has_next) / [`iter_next`](TypeSet::iter_next)
/// protocol.
pub struct WithIterator<T: TypeSet> {
    /// The inner type set.
    pub inner: T,
}

impl<T: TypeSet> Default for WithIterator<T> {
    fn default() -> Self {
        Self {
            inner: T::default(),
        }
    }
}

impl<T: TypeSet> fmt::Debug for WithIterator<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("WithIterator")
            .field("inner", &self.inner)
            .finish()
    }
}

/// Implements [`TypeSet`] for `WithIterator<$inner>` by delegating scalar behavior to
/// the concrete inner type set and swapping in a real iterator type.
///
/// A generic `impl<T: TypeSet> TypeSet for WithIterator<T>` is not possible: the
/// trait's `Integer: LoadStore<Self>` bounds can only be discharged for a concrete
/// inner type set (otherwise the trait solver cycles through the blanket `LoadStore`
/// impls). Invoke this macro to enable iterators for a custom type set whose iterator
/// type is [`NoIterator`](crate::NoIterator).
#[macro_export]
macro_rules! impl_with_iterator {
    ($inner:ty) => {
        impl $crate::TypeSet for $crate::WithIterator<$inner> {
            type Parser = <$inner as $crate::TypeSet>::Parser;
            type Integer = <$inner as $crate::TypeSet>::Integer;
            type SignedInteger = <$inner as $crate::TypeSet>::SignedInteger;
            type Float = <$inner as $crate::TypeSet>::Float;
            type String = <$inner as $crate::TypeSet>::String;
            type Iterator = $crate::SomniIterator<$inner>;

            fn to_signed(v: Self::Integer) -> Result<Self::SignedInteger, $crate::OperatorError> {
                <$inner as $crate::TypeSet>::to_signed(v)
            }

            fn to_usize(v: Self::Integer) -> Result<usize, $crate::OperatorError> {
                <$inner as $crate::TypeSet>::to_usize(v)
            }

            fn int_from_usize(v: usize) -> Self::Integer {
                <$inner as $crate::TypeSet>::int_from_usize(v)
            }

            fn load_string<'s>(&'s self, str: &'s Self::String) -> &'s str {
                self.inner.load_string(str)
            }

            fn store_string(&mut self, str: &str) -> Self::String {
                self.inner.store_string(str)
            }

            fn iter_has_next(&self, iter: &Self::Iterator) -> bool {
                iter.has_next()
            }

            fn iter_next(&self, iter: &Self::Iterator) -> Option<$crate::TypedValue<Self>> {
                // Rebinds an inner-type-set value to the wrapper type set. All scalar
                // associated types are shared with the inner type set, so those move
                // as-is; structs are converted field-wise; references are type-set
                // independent.
                fn convert(
                    value: $crate::TypedValue<$inner>,
                ) -> $crate::TypedValue<$crate::WithIterator<$inner>> {
                    use $crate::TypedValue;
                    match value {
                        TypedValue::Void => TypedValue::Void,
                        TypedValue::MaybeSignedInt(v) => TypedValue::MaybeSignedInt(v),
                        TypedValue::Int(v) => TypedValue::Int(v),
                        TypedValue::SignedInt(v) => TypedValue::SignedInt(v),
                        TypedValue::Float(v) => TypedValue::Float(v),
                        TypedValue::Bool(v) => TypedValue::Bool(v),
                        TypedValue::String(v) => TypedValue::String(v),
                        TypedValue::Iter(never) => match never {},
                        TypedValue::Struct(s) => {
                            let (name, fields) = s.into_parts();
                            TypedValue::Struct($crate::SomniStruct::new(
                                name,
                                fields.into_iter().map(|(k, v)| (k, convert(v))).collect(),
                            ))
                        }
                        TypedValue::Ref(r) => TypedValue::Ref(r),
                    }
                }
                iter.next_value().map(convert)
            }
        }
    };
}

impl_with_iterator!(crate::DefaultTypeSet);
impl_with_iterator!(crate::TypeSet32);
impl_with_iterator!(crate::TypeSet128);
