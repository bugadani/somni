//! # somni-template
//!
//! A small, configurable templating engine built on top of [`somni-expr`](somni_expr).
//!
//! Templates are **transpiled** into a Somni program (a single render function) and then
//! executed. Literal text is carried out-of-band (by span into the original template) and
//! emitted through an internal `emit` function, while `{{ expr }}` interpolations,
//! `if`/`for` conditions, and loop iterables are handed to Somni verbatim.
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
//! - Directives: `if` / `else if` / `else` / `endif`, `for <var> in <expr>` / `endfor`, and
//!   `replace "literal" with <expr>` / `endreplace` (the loop variable may carry an optional
//!   `: <type>` annotation), in either bracket ([`Syntax::brackets`]) or line
//!   ([`Syntax::lines`]) style. A `replace` block rewrites literal text in its body into
//!   prefix / `with`-expression / suffix emit triples at compile time.
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
    /// If `source` begins with a `---`-fenced frontmatter block, those settings overlay
    /// `syntax` (frontmatter takes precedence for keys it sets) and only the body after the
    /// closing fence is compiled. See [`Syntax::with_frontmatter`].
    ///
    /// Returns a [`TemplateError`] (pointing into the original `source`, including any
    /// frontmatter) on malformed directives or expressions.
    pub fn compile(source: &str, syntax: &Syntax) -> Result<Template, TemplateError> {
        let (syntax, _body, body_offset) = resolve_syntax(source, syntax)?;
        // Scan/parse/transpile the full source starting at `body_offset` so locations are
        // absolute (frontmatter-aware) without a post-hoc shift.
        let nodes = parse::parse(source, &syntax, body_offset)?;
        let transpiled = transpile::transpile(source, &nodes);

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
            ));
        }

        Ok(Template {
            template: source.to_string(),
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
        })?;

        drop(ctx);

        let output = Rc::try_unwrap(buffer)
            .map(RefCell::into_inner)
            .unwrap_or_else(|rc| rc.borrow().clone());
        Ok(output)
    }
}
