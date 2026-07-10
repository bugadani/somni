//! Values that can be handed to a template render.
//!
//! [`IntoValue`] converts common Rust types into Somni values. Scalars convert directly;
//! iterables must be wrapped in [`Iter`] to become a loop source (a Somni iterator).
//!
//! The templating engine uses the [`WithIterator<DefaultTypeSet>`](somni_expr::WithIterator)
//! type set, exposed here as [`TemplateTypes`].

use somni_expr::{DefaultTypeSet, SomniIterator, TypedValue, WithIterator};

/// The Somni type set used by the templating engine.
///
/// It is [`DefaultTypeSet`] (`u64`/`i64`/`f64`) extended with iterator support so that
/// `for` loops can iterate host-provided collections.
pub type TemplateTypes = WithIterator<DefaultTypeSet>;

/// A Rust value that can be registered on an [`Env`](crate::Env).
///
/// Implemented for scalars directly; wrap iterables in [`Iter`] to register a loop source.
pub trait IntoValue {
    /// Converts `self` into a Somni value.
    fn into_value(self) -> TypedValue<TemplateTypes>;
}

impl IntoValue for u64 {
    fn into_value(self) -> TypedValue<TemplateTypes> {
        TypedValue::Int(self)
    }
}
impl IntoValue for i64 {
    fn into_value(self) -> TypedValue<TemplateTypes> {
        TypedValue::SignedInt(self)
    }
}
impl IntoValue for f64 {
    fn into_value(self) -> TypedValue<TemplateTypes> {
        TypedValue::Float(self)
    }
}
impl IntoValue for bool {
    fn into_value(self) -> TypedValue<TemplateTypes> {
        TypedValue::Bool(self)
    }
}
impl IntoValue for &str {
    fn into_value(self) -> TypedValue<TemplateTypes> {
        TypedValue::String(self.into())
    }
}
impl IntoValue for String {
    fn into_value(self) -> TypedValue<TemplateTypes> {
        TypedValue::String(self.into_boxed_str())
    }
}
impl IntoValue for TypedValue<TemplateTypes> {
    fn into_value(self) -> TypedValue<TemplateTypes> {
        self
    }
}

/// Wraps an iterable so it becomes a loop source (a Somni iterator value).
///
/// ```ignore
/// env.value("items", Iter(vec![1u64, 2, 3]));
/// // #for x: int in items ...
/// ```
///
/// Elements must themselves be scalar [`IntoValue`]s; iterators of iterators are not
/// supported (the element type set has no iterator type).
///
/// **Single-pass:** a value registered with `Iter` is backed by a one-shot iterator, so it
/// can only be consumed by a single `for` loop. To iterate the same source repeatedly (e.g.
/// an inner loop nested inside an outer loop), register a host **function** that returns a
/// fresh iterator each call and invoke it in the loop header (`#for x: int in items()`).
pub struct Iter<I>(pub I);

impl<I> IntoValue for Iter<I>
where
    I: IntoIterator,
    I::Item: IntoValue,
{
    fn into_value(self) -> TypedValue<TemplateTypes> {
        let items: Vec<TypedValue<DefaultTypeSet>> = self
            .0
            .into_iter()
            .map(|e| to_inner(e.into_value()))
            .collect();
        TypedValue::Iter(SomniIterator::new(items))
    }
}

/// Down-converts a value from the engine's type set to the inner (non-iterator) type set,
/// used to build iterator elements. Non-scalars collapse to `Void` (unsupported as elements).
fn to_inner(v: TypedValue<TemplateTypes>) -> TypedValue<DefaultTypeSet> {
    match v {
        TypedValue::Void => TypedValue::Void,
        TypedValue::MaybeSignedInt(x) => TypedValue::MaybeSignedInt(x),
        TypedValue::Int(x) => TypedValue::Int(x),
        TypedValue::SignedInt(x) => TypedValue::SignedInt(x),
        TypedValue::Float(x) => TypedValue::Float(x),
        TypedValue::Bool(b) => TypedValue::Bool(b),
        TypedValue::String(s) => TypedValue::String(s),
        // Nested iterators cannot be represented as iterator elements.
        TypedValue::Iter(_) => TypedValue::Void,
    }
}

/// The default `str` conversion registered on every [`Env`](crate::Env).
pub(crate) fn default_str(v: TypedValue<TemplateTypes>) -> String {
    match v {
        TypedValue::Int(i) => i.to_string(),
        TypedValue::MaybeSignedInt(i) => i.to_string(),
        TypedValue::SignedInt(i) => i.to_string(),
        TypedValue::Float(f) => f.to_string(),
        TypedValue::Bool(b) => b.to_string(),
        TypedValue::String(s) => String::from(s),
        TypedValue::Void => String::new(),
        TypedValue::Iter(_) => String::from("<iter>"),
    }
}
