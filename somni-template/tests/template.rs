//! End-to-end fixture tests.
//!
//! Each subfolder of `tests/template/` is one test case with these files:
//!
//! - `template` — the template source (required), optionally prefixed with a frontmatter
//!   block that configures the [`Syntax`] (see [`parse_template_file`]).
//! - `output`   — the expected rendered output (required).
//!
//! All fixtures render against the same [`standard_env`], so templates may reference the
//! values and functions registered there. Set `BLESS=1` to (re)generate `output` files from
//! the current rendering.
//!
//! ## Frontmatter
//!
//! A template file may begin with a `---`-fenced frontmatter block selecting the syntax:
//!
//! ```text
//! ---
//! expr: {{ }}
//! block: paired /* */
//! ---
//! Status: /* if online */up/* else */down/* endif */ for {{ name }}
//! ```
//!
//! Recognized keys (all optional; defaults are the bracket style):
//!
//! - `expr: <open> <close>` — interpolation delimiters (default `{{ }}`).
//! - `block: paired <open> <close>` — paired block directives (default `{% %}`).
//! - `block: line <prefix>` — line directives (e.g. `#`).
//!
//! Without a leading `---`, the whole file is the template and the bracket style is used.
//!
//! Line endings are normalized to `\n` on both sides so fixtures are stable across platforms.

use std::{fs, path::Path};

use somni_expr::{Context, ExprContext, somni_struct};
use somni_template::{BlockStyle, Env, IntoValue, Iter, Syntax, Template, TemplateTypes};

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

/// Splits an (already newline-normalized) template file into its [`Syntax`] and body.
///
/// If the file begins with a `---`-fenced frontmatter block it is parsed for syntax keys;
/// otherwise the bracket style is used and the whole file is the template body.
fn parse_template_file(content: &str) -> (Syntax, String) {
    if let Some(rest) = content.strip_prefix("---\n") {
        if let Some(end) = rest.find("\n---\n") {
            let front = &rest[..end];
            let body = &rest[end + "\n---\n".len()..];
            return (parse_syntax(front), body.to_string());
        }
    }
    (Syntax::brackets(), content.to_string())
}

fn parse_syntax(front: &str) -> Syntax {
    let mut expr = ("{{".to_string(), "}}".to_string());
    let mut block = BlockStyle::Paired {
        open: "{%".to_string(),
        close: "%}".to_string(),
    };

    for line in front.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let (key, value) = line
            .split_once(':')
            .unwrap_or_else(|| panic!("frontmatter line is not `key: value`: {line:?}"));
        let mut parts = value.split_whitespace();

        match key.trim() {
            "expr" => {
                let open = parts.next().expect("`expr` needs an opening delimiter");
                let close = parts.next().expect("`expr` needs a closing delimiter");
                expr = (open.to_string(), close.to_string());
            }
            "block" => match parts.next() {
                Some("line") => {
                    let prefix = parts.next().expect("`block: line` needs a prefix");
                    block = BlockStyle::Line {
                        prefix: prefix.to_string(),
                    };
                }
                Some("paired") => {
                    let open = parts
                        .next()
                        .expect("`block: paired` needs an opening delimiter");
                    let close = parts
                        .next()
                        .expect("`block: paired` needs a closing delimiter");
                    block = BlockStyle::Paired {
                        open: open.to_string(),
                        close: close.to_string(),
                    };
                }
                other => panic!("unknown block style: {other:?}"),
            },
            other => panic!("unknown frontmatter key: {other:?}"),
        }
    }

    Syntax { expr, block }
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
        let raw = normalize(&fs::read_to_string(&template_path).unwrap());
        let (syntax, template) = parse_template_file(&raw);

        let compiled = Template::compile(&template, &syntax)
            .unwrap_or_else(|e| panic!("[{name}] compile failed: {e:?}"));

        // Always save the generated Somni program as an artifact (never compared).
        fs::write(dir.join("program.sm"), compiled.generated_program()).unwrap();

        let actual = compiled
            .render(standard_env())
            .unwrap_or_else(|e| panic!("[{name}] render failed: {e:?}"));

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
