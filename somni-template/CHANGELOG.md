# Unreleased

- templates can now use uppercase keywords: `IF`, `ELSE`, `ENDIF`, `FOR`, `ENDFOR`
- Host-provided structs can be used in templates: field access (`value.field`),
  struct equality in conditions, and iterating over collections of structs.
- Re-exported `SomniStruct` and implemented `IntoValue` for it.
