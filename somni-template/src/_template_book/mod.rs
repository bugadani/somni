//! End-user guide to the somni-template language.
//!
//! Templates mix literal text with interpolations and directives. They are compiled into a
//! Somni program and rendered against an [`Env`](crate::Env).
//!
//! # Chapters
//!
//! - [`_01_syntax`] — delimiters, frontmatter, text line prefix
//! - [`_02_interpolation`] — `{{ expr }}`
//! - [`_03_conditionals`] — `if` / `else if` / `else`
//! - [`_04_loops`] — `for` / `endfor`
//! - [`_05_replace`] — search-and-replace blocks
//! - [`_06_include`] — nested templates
//! - [`_07_expressions`] — what may appear inside directives and interpolations
//!
//! Examples below use [`Syntax::lines`](crate::Syntax::lines) (`#` directives, `{{ }}`
//! interpolations). Other delimiter styles are equivalent; see [`_01_syntax`].

#![allow(dead_code)]

/// Delimiters, frontmatter, and the text line prefix.
///
/// # Delimiters
///
/// Two independent choices:
///
/// | Axis | Default ([`Syntax::brackets`](crate::Syntax::brackets)) | Line style ([`Syntax::lines`](crate::Syntax::lines)) |
/// |------|---------------------------------------------------------|------------------------------------------------------|
/// | Interpolation | `{{` … `}}` | same |
/// | Directives | `{%` … `%}` | whole line starting with `#` |
///
/// Both are configurable via [`Syntax`](crate::Syntax) (and frontmatter). Line directives
/// consume the entire physical line (leading whitespace through the trailing newline) and
/// emit nothing.
///
/// Keywords are case-insensitive (`if` / `IF`, `endfor` / `ENDFOR`, …).
///
/// # Frontmatter
///
/// A template may begin with a `---`-fenced block that overlays the [`Syntax`](crate::Syntax)
/// passed to [`Template::compile`](crate::Template::compile):
///
/// ```text
/// ---
/// expr: {{ }}
/// block: line //
/// text_prefix: //>
/// ---
/// ```
///
/// Keys (each optional, at most once):
///
/// - `expr: <open> <close>`
/// - `block: paired <open> <close>`
/// - `block: line <prefix>`
/// - `text_prefix: <prefix>`
///
/// Lines that are blank or whose first non-whitespace character is `#` are ignored.
/// Frontmatter wins for keys it sets; omitted keys keep the Rust-provided value.
///
/// # Text line prefix
///
/// [`Syntax::text_prefix`](crate::Syntax::text_prefix) (frontmatter `text_prefix`) names a
/// line marker that is stripped: leading whitespace and the prefix are dropped; the rest of
/// the line is ordinary template text (interpolations allowed).
///
/// Use this when the template is also a valid program in another language. Conditional
/// output can sit behind a comment marker so it is inert if the file is compiled or run
/// directly:
///
/// ```text
/// ---
/// block: line //
/// text_prefix: //>
/// ---
/// // if feature
/// //>enable_feature();
/// // endif
/// ```
///
/// Rendered (when `feature` is true): `enable_feature();\n`. As C/C++ source, the `//>` line
/// is only a comment.
///
/// When both `text_prefix` and a line block prefix could match, the longer prefix wins. They
/// must not be identical.
pub mod _01_syntax {}

/// Emitting expression results into the output.
///
/// ```text
/// Hello, {{ name }}!
/// ```
///
/// The expression must evaluate to a `string`. Convert other types explicitly — a default
/// `str` conversion is registered on every [`Env`](crate::Env):
///
/// ```text
/// count = {{ str(count) }}
/// ```
///
/// Host functions and conversions registered on the env are callable the same way.
pub mod _02_interpolation {}

/// Conditional blocks.
///
/// ```text
/// #if online
/// (online)
/// #else if away
/// (away)
/// #else
/// (offline)
/// #endif
/// ```
///
/// Conditions are Somni expressions (see [`_07_expressions`]). Arms may nest arbitrarily with
/// other directives.
pub mod _03_conditionals {}

/// Iteration.
///
/// ```text
/// #for n in nums
/// - {{ str(n) }}
/// #endfor
/// ```
///
/// Optional type annotation on the loop variable:
///
/// ```text
/// #for n: int in nums
/// ```
///
/// When omitted, the type is inferred from uses of the variable.
///
/// The iterable must be provided by the host. Values registered with
/// [`Iter`](crate::Iter) are **single-pass**. To nest loops over the same source, register a
/// function that returns a fresh iterator and call it in the header:
///
/// ```text
/// #for i in items()
/// #for j in items()
/// …
/// #endfor
/// #endfor
/// ```
pub mod _04_loops {}

/// Literal search-and-replace over a body.
///
/// ```text
/// #replace "NAME" with name
/// Hello, NAME!
/// #endreplace
/// ```
///
/// Every occurrence of the quoted literal in the body text is replaced by the value of the
/// `with` expression (which must be a string). The body is otherwise a normal nested
/// template — directives and interpolations still run.
///
/// The literal may contain escapes (`\"`, `\\`, …). It must not be empty. A missing literal
/// is a no-op (body unchanged).
pub mod _05_replace {}

/// Nested templates.
///
/// ```text
/// #include "row.tmpl"
/// #include "cell.tmpl" with label: string = name, n: int = count
/// ```
///
/// Paths are resolved by the loader passed to
/// [`Template::compile_with`](crate::Template::compile_with). [`Template::compile`](crate::Template::compile)
/// rejects every include.
///
/// - **Bare `include`** (no `with`): the body is expanded in place and shares the caller’s
///   locals and env.
/// - **`include … with …`**: compiles to a separate function. Each binding is
///   `name: type = expr`; types are required. Only the listed bindings are in scope inside
///   the include.
///
/// An included file may have its own frontmatter (its own delimiters). Cycles are a compile
/// error.
pub mod _06_include {}

/// Expressions inside interpolations and directives.
///
/// Expression syntax is [Somni](somni_expr)’s. Common forms:
///
/// - Literals, identifiers, field access: `point.x`
/// - Arithmetic and comparisons: `n > 1`, `a + b`
/// - Calls: `str(n)`, `upper(name)`, `items()`
/// - Struct equality in conditions when the host registered structs
///
/// Names come from the [`Env`](crate::Env) (values and functions) plus locals introduced by
/// `for` and `include … with`. There is no assignment and no statement forms — only
/// expressions.
pub mod _07_expressions {}
