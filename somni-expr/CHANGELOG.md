# Unreleased

- The TypeSet trait has been reworked for better ergonomics. The parser's TypeSet is now an associated type. Additional associated types and trait bounds have been added.
- The `MemoryRepr` trait has been removed.
- TypedValue's `String` is now also generic.
- The string interner has been removed. The expression engine uses `Box<str>` as storage by default.
- Expressions can be evaluated to `TypedValue`.
- `ExprContext::try_load_variable` and `address_of` now take `&mut self`.
- Added `declare` and `assign_variable` to `ExprContext`.
- `Context::parse` and variants to load a whole program for evaluation.

# [0.1.0] - 2025-07-24

- Initial release

[0.1.0]: https://github.com/bugadani/somni/releases/tag/somni-expr-v0.1.0
