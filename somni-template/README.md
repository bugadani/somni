somni-template
==============

A small, configurable templating engine built on top of [`somni-expr`](../somni-expr).

Templates are transpiled into a Somni program and executed: literal text is emitted verbatim,
while `{{ expr }}` interpolations, `if`/`for` conditions and loop iterables are evaluated by
`somni-expr`.

## Features

- **Configurable syntax.** Interpolation delimiters and block-directive style are both
  configurable. Block directives can be a delimiter pair (e.g. `{% ... %}`, `/* ... */`) or a
  line prefix (e.g. `#`, `//`).
- **Control flow.** `if` / `else if` / `else` and `for <var>: <type> in <iterable>`, arbitrarily
  nestable.
- **Strongly typed output.** Interpolation is string-only; convert other types explicitly
  (a generic `str` conversion is provided, more can be registered).
- **Errors point into the template**, even for evaluation errors in the generated program.

## Example

```rust
use somni_template::{Env, Iter, Syntax, Template};

let tmpl = Template::compile(
    "Hello, {{ name }}!\n#for n: int in nums\n- {{ str(n) }}\n#endfor\n",
    &Syntax::lines(),
)
.unwrap();

let mut env = Env::new();          // ships a default `str` conversion
env.value("name", "Ada");
env.value("nums", Iter(vec![1u64, 2, 3]));

let out = tmpl.render(env).unwrap();
assert_eq!(out, "Hello, Ada!\n- 1\n- 2\n- 3\n");
```

## Notes

- Interpolation must evaluate to a `string`; use `{{ str(x) }}` (or a custom conversion) for
  other types.
- `for` requires the explicit loop-variable type (`for x: int in ...`) and a host-provided
  iterable.
- Iterables registered with [`Iter`] are single-pass. To re-iterate a source (e.g. an inner
  loop nested in an outer loop), register a host function that returns a fresh iterator and
  call it in the loop header (`#for x: int in items()`).

## License

Licensed under either of Apache-2.0 or MIT, at your option.
