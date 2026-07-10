//! End-to-end rendering tests.

use somni_template::{Env, IntoValue, Iter, Syntax, Template};

fn render(source: &str, syntax: &Syntax, build: impl FnOnce(&mut Env)) -> String {
    let tmpl = Template::compile(source, syntax).expect("compile");
    let mut env = Env::new();
    build(&mut env);
    tmpl.render(env).expect("render")
}

#[test]
fn plain_text_passthrough() {
    assert_eq!(
        render("hello world", &Syntax::brackets(), |_| {}),
        "hello world"
    );
}

#[test]
fn string_interpolation() {
    let out = render("Hi {{ name }}!", &Syntax::brackets(), |env| {
        env.value("name", "Ada");
    });
    assert_eq!(out, "Hi Ada!");
}

#[test]
fn explicit_str_conversion() {
    let out = render("age: {{ str(age) }}", &Syntax::brackets(), |env| {
        env.value("age", 42u64);
    });
    assert_eq!(out, "age: 42");
}

#[test]
fn bare_non_string_interpolation_is_error() {
    let tmpl = Template::compile("{{ age }}", &Syntax::brackets()).unwrap();
    let mut env = Env::new();
    env.value("age", 42u64);
    let err = tmpl.render(env).unwrap_err();
    // Error should point at `age` in the template.
    assert_eq!(err.location.extract("{{ age }}"), "age");
}

#[test]
fn if_else_if_else_chain() {
    let source = "{% if a %}A{% else if b %}B{% else %}C{% endif %}";
    let a = render(source, &Syntax::brackets(), |env| {
        env.value("a", true);
        env.value("b", false);
    });
    assert_eq!(a, "A");

    let b = render(source, &Syntax::brackets(), |env| {
        env.value("a", false);
        env.value("b", true);
    });
    assert_eq!(b, "B");

    let c = render(source, &Syntax::brackets(), |env| {
        env.value("a", false);
        env.value("b", false);
    });
    assert_eq!(c, "C");
}

#[test]
fn for_loop_over_collection() {
    let source = r#"#for n: int in nums
{{ str(n) }},
#endfor
"#;
    let out = render(source, &Syntax::lines(), |env| {
        env.value("nums", Iter(vec![1u64, 2, 3]));
    });
    assert_eq!(out, "1,\n2,\n3,\n");
}

#[test]
fn for_loop_over_strings() {
    let out = render(
        "{% for s: string in words %}[{{ s }}]{% endfor %}",
        &Syntax::brackets(),
        |env| {
            env.value("words", Iter(vec!["a".to_string(), "b".to_string()]));
        },
    );
    assert_eq!(out, "[a][b]");
}

#[test]
fn nested_for_in_if() {
    let source = "{% if show %}{% for x: int in xs %}{{ str(x) }};{% endfor %}{% endif %}done";
    let out = render(source, &Syntax::brackets(), |env| {
        env.value("show", true);
        env.value("xs", Iter(vec![10u64, 20]));
    });
    assert_eq!(out, "10;20;done");
}

#[test]
fn nested_loops() {
    // The inner source is re-iterated once per outer element, so it must be a function that
    // returns a *fresh* iterator each call (value-registered iterators are single-pass).
    let source = r#"#for i: int in outer
#for j: int in inner()
{{ str(i) }}{{ str(j) }} 
#endfor
#endfor
"#;
    let out = render(source, &Syntax::lines(), |env| {
        env.value("outer", Iter(vec![1u64, 2]));
        env.function("inner", || Iter(vec![7u64, 8]).into_value());
    });
    // The body text line is "{{i}}{{j}} \n"; directive lines are swallowed entirely.
    assert_eq!(out, "17 \n18 \n27 \n28 \n");
}

#[test]
fn value_registered_iterator_is_single_pass() {
    // Documents the single-pass semantics: iterating the same value twice yields nothing the
    // second time. Use a function (see `nested_loops`) when re-iteration is needed.
    let source = "{% for x: int in xs %}{{ str(x) }}{% endfor %}|{% for x: int in xs %}{{ str(x) }}{% endfor %}";
    let out = render(source, &Syntax::brackets(), |env| {
        env.value("xs", Iter(vec![1u64, 2]));
    });
    assert_eq!(out, "12|");
}

#[test]
fn host_function() {
    let out = render("{{ upper(name) }}", &Syntax::brackets(), |env| {
        env.value("name", "ada");
        env.function("upper", |s: &str| s.to_uppercase());
    });
    assert_eq!(out, "ADA");
}

#[test]
fn condition_uses_expression() {
    let source = "{% if n > 3 %}big{% else %}small{% endif %}";
    let big = render(source, &Syntax::brackets(), |env| {
        env.value("n", 5u64);
    });
    assert_eq!(big, "big");
    let small = render(source, &Syntax::brackets(), |env| {
        env.value("n", 1u64);
    });
    assert_eq!(small, "small");
}

#[test]
fn literal_text_with_special_characters_roundtrips() {
    // Backslashes, quotes and newlines would break naive string-literal embedding; the
    // span-table approach carries them verbatim.
    let source = r#"line1
"quoted" and a \ backslash {{ x }} end"#;
    let expected = "line1\n\"quoted\" and a \\ backslash X end";
    let out = render(source, &Syntax::brackets(), |env| {
        env.value("x", "X");
    });
    assert_eq!(out, expected);
}

#[test]
fn configurable_delimiters() {
    let syntax = Syntax {
        expr: ("${".into(), "}".into()),
        block: somni_template::BlockStyle::Paired {
            open: "<%".into(),
            close: "%>".into(),
        },
    };
    let out = render("<% if ok %>yes:${ name }<% endif %>", &syntax, |env| {
        env.value("ok", true);
        env.value("name", "Z");
    });
    assert_eq!(out, "yes:Z");
}

#[test]
fn c_style_comment_delimiters() {
    // Block directives delimited by C-style comments; interpolation stays `{{ }}`.
    let syntax = Syntax {
        expr: ("{{".into(), "}}".into()),
        block: somni_template::BlockStyle::Paired {
            open: "/*".into(),
            close: "*/".into(),
        },
    };
    let source = "Hi {{ name }}, you are /* if online */online/* else */offline/* endif */.";
    let up = render(source, &syntax, |env| {
        env.value("name", "Ada");
        env.value("online", true);
    });
    assert_eq!(up, "Hi Ada, you are online.");
    let down = render(source, &syntax, |env| {
        env.value("name", "Ada");
        env.value("online", false);
    });
    assert_eq!(down, "Hi Ada, you are offline.");
}

#[test]
fn c_style_comment_loop() {
    let syntax = Syntax {
        expr: ("{{".into(), "}}".into()),
        block: somni_template::BlockStyle::Paired {
            open: "/*".into(),
            close: "*/".into(),
        },
    };
    let source = "/* for n: int in xs */[{{ str(n) }}]/* endfor */";
    let out = render(source, &syntax, |env| {
        env.value("xs", Iter(vec![1u64, 2, 3]));
    });
    assert_eq!(out, "[1][2][3]");
}

#[test]
fn c_line_comment_directives() {
    // Line directives introduced by `//` (C-style line comments); interpolation is `{{ }}`.
    let syntax = Syntax {
        expr: ("{{".into(), "}}".into()),
        block: somni_template::BlockStyle::Line { prefix: "//".into() },
    };
    let source = r#"User: {{ name }}
// if online
(online)
// else
(offline)
// endif
"#;
    let on = render(source, &syntax, |env| {
        env.value("name", "Ada");
        env.value("online", true);
    });
    assert_eq!(on, "User: Ada\n(online)\n");
    let off = render(source, &syntax, |env| {
        env.value("name", "Ada");
        env.value("online", false);
    });
    assert_eq!(off, "User: Ada\n(offline)\n");
}

#[test]
fn c_line_comment_loop() {
    let syntax = Syntax {
        expr: ("{{".into(), "}}".into()),
        block: somni_template::BlockStyle::Line { prefix: "//".into() },
    };
    let source = r#"// for n: int in xs
* {{ str(n) }}
// endfor
"#;
    let out = render(source, &syntax, |env| {
        env.value("xs", Iter(vec![1u64, 2, 3]));
    });
    assert_eq!(out, "* 1\n* 2\n* 3\n");
}

#[test]
fn runtime_error_maps_to_template_location() {
    // `missing` is not registered -> unknown variable at render time.
    let source = "before {{ missing }} after";
    let tmpl = Template::compile(source, &Syntax::brackets()).unwrap();
    let err = tmpl.render(Env::new()).unwrap_err();
    assert_eq!(err.location.extract(source), "missing");
}

#[test]
fn compile_error_maps_to_template_location() {
    // Malformed expression inside interpolation.
    let source = "value: {{ 1 + }}";
    let err = Template::compile(source, &Syntax::brackets()).unwrap_err();
    // The error should land within the interpolation region, not at offset 0.
    assert!(err.location.start >= "value: {{ ".len(), "{err:?}");
}
