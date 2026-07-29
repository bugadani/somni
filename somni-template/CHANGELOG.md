# [0.4.0] - 2026-07-29

- Specify template style options in `---`-fenced frontmatter
- `#replace "SOMETHING" with expression` blocks
- `#include "path"` blocks
- `text_prefix` line marker to output verbatim text with the comment-like syntax in the output language.

# [0.3.2] - 2026-07-15

- Fixed `for ... in` parsing

# [0.3.1] - 2026-07-14

- templates can now use uppercase keywords: `IF`, `ELSE`, `ENDIF`, `FOR`, `ENDFOR`
- Host-provided structs can be used in templates: field access (`value.field`),
  struct equality in conditions, and iterating over collections of structs.
- Re-exported `SomniStruct` and implemented `IntoValue` for it.

[0.3.2]: https://github.com/bugadani/somni/compare/somni-template-v0.3.1...somni-template-v0.3.2
[0.3.1]: https://github.com/bugadani/somni/compare/somni-template-v0.3.0...somni-template-v0.3.1
[0.3.0]: https://github.com/bugadani/somni/releases/tag/somni-template-v0.3.0
