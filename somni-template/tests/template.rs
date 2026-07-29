//! End-to-end fixture tests.
//!
//! Each subfolder of `tests/template/` is one test case with these files:
//!
//! - `template` — the template source (required), optionally prefixed with a frontmatter
//!   block that configures the [`Syntax`] (see [`somni_template::split_frontmatter`]).
//! - `output` — the expected rendered output (for successful fixtures).
//! - `compile_error` — if present, compilation must fail; compared against the full
//!   ANSI-stripped [`TemplateError::display_with`] output.
//! - `error` — if present, rendering must fail; compared the same way.
//!
//! `include "path"` directives load sibling files from the fixture directory (the path is
//! joined onto the fixture folder).
//!
//! All fixtures render against the same [`standard_env`], so templates may reference the
//! values and functions registered there. Set `BLESS=1` to (re)generate `output`,
//! `compile_error`, and `error` files from the current run.
//!
//! ## Frontmatter
//!
//! A template file may begin with a `---`-fenced frontmatter block selecting the syntax.
//! Frontmatter keys overlay the Rust-provided base syntax ([`Syntax::brackets`] here):
//!
//! ```text
//! ---
//! expr: {{ }}
//! block: paired /* */
//! ---
//! Status: /* if online */up/* else */down/* endif */ for {{ name }}
//! ```
//!
//! Recognized keys (all optional):
//!
//! - `expr: <open> <close>` — interpolation delimiters.
//! - `block: paired <open> <close>` — paired block directives.
//! - `block: line <prefix>` — line directives.
//!
//! Lines starting with `#` are comments and ignored.
//!
//! Without a leading `---`, the whole file is the template and the bracket style is used.
//!
//! Line endings are normalized to `\n` on both sides so fixtures are stable across platforms.

use std::{fs, path::Path};

use pretty_assertions::assert_eq;
use somni_expr::{Context, ExprContext, somni_struct};
use somni_template::{Env, IntoValue, Iter, Syntax, Template, TemplateError, TemplateTypes};

/// The data available to every fixture template.
fn standard_env() -> Env {
    let mut env = Env::new();

    env.value("name", "Ada");
    env.value("title", "Engineer");
    env.value("count", 3u64);
    env.value("online", true);
    env.value("offline", false);

    // Single-pass collections (each usable by one loop).
    env.value("nums", Iter(vec![1u64, 2, 3]));
    env.value(
        "words",
        Iter(vec![
            "red".to_string(),
            "green".to_string(),
            "blue".to_string(),
        ]),
    );

    // A re-iterable source: a fresh iterator per call (usable in nested loops).
    env.function("seq", || Iter(vec![1u64, 2, 3]).into_value());
    env.function("upper", |s: &str| s.to_uppercase());

    // Structs handed in from the host. Templates read their fields, compare them,
    // and iterate over collections of them. `somni_struct!` builds the values.
    let mut ctx = Context::<TemplateTypes>::new_with_types();
    let tc = ctx.type_context();

    env.value("point", somni_struct!(tc, Point { x: 3u64, y: 4u64 }));
    env.value("same_point", somni_struct!(tc, Point { x: 3u64, y: 4u64 }));
    env.value("other_point", somni_struct!(tc, Point { x: 9u64, y: 9u64 }));

    env.value(
        "points",
        Iter(vec![
            somni_struct!(tc, Point { x: 1u64, y: 2u64 }),
            somni_struct!(tc, Point { x: 3u64, y: 4u64 }),
            somni_struct!(tc, Point { x: 5u64, y: 6u64 }),
        ]),
    );

    // Nested structs must survive the round-trip through the iterator element types.
    env.value(
        "lines",
        Iter(vec![
            somni_struct!(
                tc,
                Line {
                    start: somni_struct!(tc, Point { x: 1u64, y: 2u64 }),
                    end: somni_struct!(tc, Point { x: 3u64, y: 4u64 }),
                }
            ),
            somni_struct!(
                tc,
                Line {
                    start: somni_struct!(tc, Point { x: 5u64, y: 6u64 }),
                    end: somni_struct!(tc, Point { x: 7u64, y: 8u64 }),
                }
            ),
        ]),
    );

    env
}

fn normalize(s: &str) -> String {
    s.replace("\r\n", "\n")
}

fn strip_ansi(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut chars = s.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '\u{1b}' && chars.peek() == Some(&'[') {
            chars.next();
            for ch in chars.by_ref() {
                if ch.is_ascii_alphabetic() {
                    break;
                }
            }
        } else {
            out.push(c);
        }
    }
    out
}

fn rich_error(err: &TemplateError, template: &str) -> String {
    normalize(&strip_ansi(&err.display_with(template).to_string()))
}

fn assert_error_fixture(name: &str, path: &Path, actual: &str, bless: bool) {
    if bless {
        fs::write(path, actual).unwrap();
        return;
    }
    let expected = normalize(
        &fs::read_to_string(path)
            .unwrap_or_else(|e| panic!("[{name}] missing `{}`: {e}", path.display())),
    );
    assert_eq!(actual, expected, "[{name}] error output mismatch");
}

#[test]
fn run_template_fixtures() {
    let bless = std::env::var("BLESS").as_deref() == Ok("1");
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("template");

    let mut ran = 0;
    for entry in fs::read_dir(&root)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", root.display()))
        .flatten()
    {
        let dir = entry.path();
        if !dir.is_dir() {
            continue;
        }
        let template_path = dir.join("template");
        if !template_path.exists() {
            continue;
        }

        let name = dir.file_name().unwrap().to_string_lossy().into_owned();
        let template = normalize(&fs::read_to_string(&template_path).unwrap());

        let compile_error_path = dir.join("compile_error");
        let render_error_path = dir.join("error");

        let compiled = match Template::compile_with(&template, &Syntax::brackets(), &mut |path| {
            let file = dir.join(path);
            fs::read_to_string(&file).map_err(|e| format!("{}: {e}", file.display()))
        }) {
            Ok(t) => {
                if compile_error_path.exists() {
                    panic!("[{name}] expected compile error, but compile succeeded");
                }
                t
            }
            Err(e) => {
                if !compile_error_path.exists() {
                    panic!("[{name}] compile failed:\n{}", rich_error(&e, &template));
                }
                assert_error_fixture(
                    &name,
                    &compile_error_path,
                    &rich_error(&e, &template),
                    bless,
                );
                ran += 1;
                continue;
            }
        };

        // Always save the generated Somni program as an artifact (never compared).
        fs::write(dir.join("program.sm"), compiled.generated_program()).unwrap();

        match compiled.render(standard_env()) {
            Ok(actual) => {
                if render_error_path.exists() {
                    panic!("[{name}] expected render error, but render succeeded:\n{actual}");
                }
                let output_path = dir.join("output");
                if bless {
                    fs::write(&output_path, &actual).unwrap();
                } else {
                    let expected = normalize(
                        &fs::read_to_string(&output_path)
                            .unwrap_or_else(|e| panic!("[{name}] missing `output`: {e}")),
                    );
                    assert_eq!(actual, expected, "[{name}] rendered output mismatch");
                }
            }
            Err(e) => {
                if !render_error_path.exists() {
                    panic!("[{name}] render failed:\n{}", rich_error(&e, &template));
                }
                assert_error_fixture(&name, &render_error_path, &rich_error(&e, &template), bless);
            }
        }
        ran += 1;
    }

    assert!(
        ran > 0,
        "no template fixtures were found in {}",
        root.display()
    );
}
