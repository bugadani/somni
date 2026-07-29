//! End-to-end fixture tests.
//!
//! Each subfolder of `tests/template/` is one test case with these files:
//!
//! - `template` — the template source (required), optionally prefixed with a frontmatter
//!   block that configures the [`Syntax`] (see [`somni_template::split_frontmatter`]).
//! - `output`   — the expected rendered output (required).
//!
//! All fixtures render against the same [`standard_env`], so templates may reference the
//! values and functions registered there. Set `BLESS=1` to (re)generate `output` files from
//! the current rendering.
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
use somni_template::{Env, IntoValue, Iter, Syntax, Template, TemplateTypes};

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

        let compiled = Template::compile(&template, &Syntax::brackets()).unwrap_or_else(|e| {
            panic!("[{name}] compile failed:\n{}", e.display_with(&template));
        });

        // Always save the generated Somni program as an artifact (never compared).
        fs::write(dir.join("program.sm"), compiled.generated_program()).unwrap();

        let actual = compiled.render(standard_env()).unwrap_or_else(|e| {
            panic!("[{name}] render failed:\n{}", e.display_with(&template));
        });

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
        ran += 1;
    }

    assert!(
        ran > 0,
        "no template fixtures were found in {}",
        root.display()
    );
}
