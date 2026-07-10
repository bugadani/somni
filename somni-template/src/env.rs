//! The per-render environment: values, functions and conversions supplied to a template.
//!
//! An [`Env`] is **single-use**: it is consumed by [`Template::render`](crate::Template::render).
//! Registrations are recorded and applied to a fresh evaluation context each render.

use somni_expr::{Context, DynFunction, TypedValue};

use crate::value::{default_str, IntoValue, TemplateTypes};

/// A deferred registration that installs a function onto a context under a given name.
///
/// The name lifetime is bound to the context lifetime, so the callback is lifetime-generic.
type FnApplier = Box<dyn for<'c> FnOnce(&mut Context<'c, TemplateTypes>, &'c str)>;

/// The data supplied to a single template render.
///
/// Register values with [`Env::value`], host functions with [`Env::function`], and
/// string conversions with [`Env::conversion`] (a `str` conversion is pre-registered).
pub struct Env {
    names: Vec<String>,
    values: Vec<(usize, TypedValue<TemplateTypes>)>,
    funcs: Vec<(usize, FnApplier)>,
}

impl Default for Env {
    fn default() -> Self {
        Self::new()
    }
}

impl Env {
    /// Creates a new environment, pre-registering the default `str` conversion.
    pub fn new() -> Self {
        let mut env = Env {
            names: Vec::new(),
            values: Vec::new(),
            funcs: Vec::new(),
        };
        env.conversion("str", default_str);
        env
    }

    /// Registers a value under `name`.
    ///
    /// Scalars are registered directly; wrap iterables in [`Iter`](crate::Iter) to register
    /// a loop source.
    pub fn value(&mut self, name: &str, value: impl IntoValue) -> &mut Self {
        let idx = self.intern(name);
        self.values.push((idx, value.into_value()));
        self
    }

    /// Registers a host function callable from template expressions.
    pub fn function<F, A>(&mut self, name: &str, func: F) -> &mut Self
    where
        F: DynFunction<A, TemplateTypes> + 'static,
    {
        let idx = self.intern(name);
        let applier: FnApplier = Box::new(move |ctx, name| {
            ctx.add_function(name, func);
        });
        self.funcs.push((idx, applier));
        self
    }

    /// Registers a conversion function. A conversion is just a function returning a string;
    /// this is a readability alias for [`Env::function`].
    pub fn conversion<F, A>(&mut self, name: &str, func: F) -> &mut Self
    where
        F: DynFunction<A, TemplateTypes> + 'static,
    {
        self.function(name, func)
    }

    fn intern(&mut self, name: &str) -> usize {
        let idx = self.names.len();
        self.names.push(name.to_string());
        idx
    }

    /// Applies all registrations onto the given context. Names are borrowed from `names`,
    /// which must outlive the context.
    pub(crate) fn apply<'c>(self, ctx: &mut Context<'c, TemplateTypes>, names: &'c [String]) {
        for (idx, value) in self.values {
            ctx.add_variable(names[idx].as_str(), value);
        }
        for (idx, applier) in self.funcs {
            applier(ctx, names[idx].as_str());
        }
    }

    /// Consumes the environment, returning the interned names (which must be kept alive for
    /// the duration of a render) and the registrations to apply.
    pub(crate) fn into_parts(self) -> (Vec<String>, Self) {
        (self.names.clone(), self)
    }
}
