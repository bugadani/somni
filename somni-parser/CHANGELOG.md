# Unreleased

- Hidden the tokenizer function
- Unified lexer/parser errors
- Parser no longer needs the whole source to be tokenized upfront
- Added TypeSet type parameter to the whole AST and the `parse` function
- `Expression::as_variable` has been removed
- Split `Expression` and `RightHandExpression`, added `LeftHandExpression`
- Support free-standing block scopes

# [0.1.0] - 2025-07-24

- Initial release

[0.1.0]: https://github.com/bugadani/somni/releases/tag/somni-parser-v0.1.0
