//! # somni-template
//!
//! A small, configurable templating engine built on top of [`somni-expr`](somni_expr).
//!
//! Templates are **transpiled** into a Somni program (a render function plus any include
//! functions) and then executed. Literal text is carried out-of-band (by span into the
//! original template / include arena) and emitted through an internal `emit` function, while
//! `{{ expr }}` interpolations, `if`/`for` conditions, and loop iterables are handed to Somni
//! verbatim.
//!
//! ## Example
//!
//! ```rust
//! use somni_template::{Env, Iter, Syntax, Template};
//!
//! let tmpl = Template::compile(
//!     "#for n in nums\n{{ str(n) }},\n#endfor\n",
//!     &Syntax::lines(),
//! )
//! .unwrap();
//!
//! let mut env = Env::new();
//! env.value("nums", Iter(vec![1u64, 2, 3]));
//!
//! assert_eq!(tmpl.render(env).unwrap(), "1,\n2,\n3,\n");
//! ```
//!
//! ## Syntax
//!
//! - Interpolation: `{{ expr }}` (the expression must evaluate to a `string`; use a
//!   conversion such as `str(x)` for other types).
//! - Directives: `if` / `else if` / `else` / `endif`, `for <var> in <expr>` / `endfor`,
//!   `replace "literal" with <expr>` / `endreplace`, and `include "path"` (optional
//!   `with name: type = expr, …`), in either bracket ([`Syntax::brackets`]) or line
//!   ([`Syntax::lines`]) style. Includes are loaded at compile time via
//!   [`Template::compile_with`]. A bare `include` (no `with`) is expanded in place and
//!   shares the caller’s context; `include … with …` compiles to a separate Somni function
//!   invoked with the listed bindings.
//! - Optional `---`-fenced **frontmatter** at the start of a template may override the
//!   [`Syntax`] passed to [`Template::compile`] (frontmatter wins for keys it sets).
//!
//! See [`Env`] for supplying data, [`IntoValue`]/[`Iter`] for values and loop sources, and
//! [`TemplateError`] for diagnostics (which always point into the original template).

#![warn(missing_docs)]

pub mod error;
pub mod syntax;

mod env;
mod parse;
mod resolve;
mod scan;
mod transpile;
mod value;

use std::{cell::RefCell, marker::PhantomData, rc::Rc};

use somni_expr::{Context, ExpressionVisitor, TypeSet};
use somni_parser::{Location, ast::Item};

pub use env::Env;
pub use error::TemplateError;
pub use somni_expr::{SomniIterator, SomniStruct, TypedValue};
pub use syntax::{BlockStyle, Syntax, resolve_syntax, split_frontmatter};
pub use value::{IntoValue, Iter, TemplateTypes};

use transpile::{EMIT_FN, EMIT_LIT_FN, RENDER_FN, Transpiled};

/// Shared data used by the internal `emit_lit` function to emit literal chunks by index.
struct Literals {
    template: String,
    spans: Vec<Location>,
}

/// A compiled template.
///
/// Compilation is independent of data and reusable across many renders. See
/// [`Template::compile`] and [`Template::render`].
#[derive(Clone, Debug)]
pub struct Template {
    template: String,
    transpiled: Transpiled,
}

impl Template {
    /// Compiles a template from source using the given [`Syntax`].
    ///
    /// Equivalent to [`Template::compile_with`] with a loader that rejects every include.
    ///
    /// If `source` begins with a `---`-fenced frontmatter block, those settings overlay
    /// `syntax` (frontmatter takes precedence for keys it sets) and only the body after the
    /// closing fence is compiled. See [`Syntax::with_frontmatter`].
    ///
    /// Returns a [`TemplateError`] (pointing into the original `source`, including any
    /// frontmatter) on malformed directives or expressions.
    pub fn compile(source: &str, syntax: &Syntax) -> Result<Template, TemplateError> {
        Self::compile_with(source, syntax, &mut |path| {
            Err(format!(
                "include `{path}` requires Template::compile_with with a loader"
            ))
        })
    }

    /// Compiles a template, loading `include` paths through `loader`.
    ///
    /// `loader` is called with the path string from each `include "…"` directive. Returned
    /// source may itself contain frontmatter and nested includes. Cycles are rejected.
    ///
    /// Each distinct `(path, with-parameter signature)` becomes its own Somni function
    /// called with the listed bindings. A bare `include` (no `with`) is inlined instead and
    /// shares the caller’s context. Host functions and Env values are visible in both forms.
    pub fn compile_with(
        source: &str,
        syntax: &Syntax,
        loader: &mut dyn FnMut(&str) -> Result<String, String>,
    ) -> Result<Template, TemplateError> {
        let (syntax, _body, body_offset) = resolve_syntax(source, syntax)?;
        // Scan/parse the full source starting at `body_offset` so locations are absolute
        // (frontmatter-aware) without a post-hoc shift.
        let mut nodes = parse::parse(source, &syntax, body_offset)?;

        let mut arena = source.to_string();
        let modules = resolve::resolve_includes(&mut arena, &mut nodes, &syntax, loader)
            .map_err(|e| e.with_diagnostic_source(arena.clone()))?;
        let transpiled = transpile::transpile(&arena, &nodes, &modules);

        // Validate the generated program so that expression syntax errors surface at compile
        // time, mapped back to the original template.
        if let Err(err) =
            somni_parser::parser::parse::<<TemplateTypes as TypeSet>::Parser>(&transpiled.source)
        {
            let location = transpiled
                .map_location(err.location)
                .unwrap_or(Location {
                    start: body_offset,
                    end: body_offset,
                });
            return Err(TemplateError::new(
                format!("invalid expression: {}", err.error),
                location,
            )
            .with_diagnostic_source(arena));
        }

        Ok(Template {
            template: arena,
            transpiled,
        })
    }

    /// Returns the generated Somni program that this template transpiles to.
    ///
    /// Primarily useful for debugging and snapshotting; the exact output is not a stable
    /// part of the API.
    pub fn generated_program(&self) -> &str {
        &self.transpiled.source
    }

    /// Renders the template with the given (single-use) [`Env`].
    ///
    /// Runtime evaluation errors are mapped back to locations in the original template.
    pub fn render(&self, env: Env) -> Result<String, TemplateError> {
        // `names` must outlive `ctx`, so it is declared first.
        let (names, env) = env.into_parts();

        let buffer = Rc::new(RefCell::new(String::new()));
        let literals = Rc::new(Literals {
            template: self.template.clone(),
            spans: self.transpiled.literals.clone(),
        });

        let mut ctx = Context::<TemplateTypes>::parse_with_types(&self.transpiled.source)
            .expect("generated program is validated during compile");

        {
            let buffer = buffer.clone();
            ctx.add_function(EMIT_FN, move |s: &str| {
                buffer.borrow_mut().push_str(s);
            });
        }
        {
            let buffer = buffer.clone();
            let literals = literals.clone();
            ctx.add_function(EMIT_LIT_FN, move |index: u64| {
                if let Some(span) = literals.spans.get(index as usize) {
                    buffer
                        .borrow_mut()
                        .push_str(&literals.template[span.start..span.end]);
                }
            });
        }

        env.apply(&mut ctx, &names);

        let program = somni_parser::parser::parse::<<TemplateTypes as TypeSet>::Parser>(
            &self.transpiled.source,
        )
        .expect("generated program is validated during compile");
        let render_fn = program
            .items
            .iter()
            .find_map(|item| match item {
                Item::Function(f) if f.name.source(&self.transpiled.source) == RENDER_FN => Some(f),
                _ => None,
            })
            .expect("generated program always defines the render function");

        let result = {
            let mut visitor = ExpressionVisitor::<Context<'_, TemplateTypes>, TemplateTypes> {
                context: &mut ctx,
                source: &self.transpiled.source,
                _marker: PhantomData,
            };
            visitor.visit_function(render_fn, &[])
        };

        result.map_err(|err| {
            let location = self
                .transpiled
                .map_location(err.location)
                .unwrap_or(Location { start: 0, end: 0 });
            TemplateError::new(err.message, location)
                .with_diagnostic_source(self.template.clone())
        })?;

        drop(ctx);

        let output = Rc::try_unwrap(buffer)
            .map(RefCell::into_inner)
            .unwrap_or_else(|rc| rc.borrow().clone());
        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compile_rejects_include_without_loader() {
        let err = Template::compile(r#"{% include "x.tmpl" %}"#, &Syntax::brackets()).unwrap_err();
        assert!(
            err.message.contains("compile_with") || err.message.contains("loader"),
            "unexpected: {}",
            err.message
        );
    }
}
