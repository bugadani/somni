//! Transpilation of a parsed template tree into a Somni program.
//!
//! The generated program is a single function (named [`RENDER_FN`]) whose body drives an
//! output buffer through two internal functions:
//!
//! - [`EMIT_FN`]`(s)` appends a string produced by an interpolation, and
//! - [`EMIT_LIT_FN`]`(i)` appends the `i`-th literal chunk (looked up by span into the
//!   original template — literal text is never embedded in the generated source).
//!
//! Every expression copied verbatim from the template (interpolations, conditions, loop
//! iterables and headers) records a [`MapEntry`] so that evaluation errors — whose locations
//! point into the generated source — can be translated back into the template precisely.

use somni_parser::Location;

use crate::parse::{Arm, Node};

/// Name of the generated render function.
pub const RENDER_FN: &str = "__tmpl_render";
/// Name of the internal string-emitting function.
pub const EMIT_FN: &str = "__tmpl_emit";
/// Name of the internal literal-chunk-emitting function.
pub const EMIT_LIT_FN: &str = "__tmpl_emit_lit";

/// A mapping from a range in the generated source back to a range in the template.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MapEntry {
    /// Start offset in the generated source.
    pub gen_start: usize,
    /// Start offset in the original template.
    pub tmpl_start: usize,
    /// Length of the copied region (identical in both).
    pub len: usize,
}

/// The result of transpiling a template.
#[derive(Clone, Debug)]
pub struct Transpiled {
    /// The generated Somni source.
    pub source: String,
    /// Literal chunk spans into the *original template*, indexed by chunk id.
    pub literals: Vec<Location>,
    /// Source map entries for verbatim-copied expression regions.
    pub map: Vec<MapEntry>,
}

impl Transpiled {
    /// Translates a location in the generated source back into the template.
    ///
    /// Returns `None` if the location does not fall within any verbatim-copied region
    /// (e.g. it points at generated scaffolding).
    pub fn map_location(&self, loc: Location) -> Option<Location> {
        let start = self.map_offset(loc.start)?;
        // Map the end using the same entry if possible; otherwise fall back to start.
        let end = self.map_offset(loc.end).unwrap_or(start);
        Some(Location {
            start,
            end: end.max(start),
        })
    }

    fn map_offset(&self, off: usize) -> Option<usize> {
        for e in &self.map {
            if off >= e.gen_start && off <= e.gen_start + e.len {
                return Some(e.tmpl_start + (off - e.gen_start));
            }
        }
        None
    }
}

struct Writer<'a> {
    template: &'a str,
    out: String,
    literals: Vec<Location>,
    map: Vec<MapEntry>,
}

impl Writer<'_> {
    fn push(&mut self, s: &str) {
        self.out.push_str(s);
    }

    /// Copies a template region verbatim into the output, recording a source-map entry.
    fn push_verbatim(&mut self, loc: Location) {
        let text = loc.extract(self.template);
        self.map.push(MapEntry {
            gen_start: self.out.len(),
            tmpl_start: loc.start,
            len: text.len(),
        });
        self.out.push_str(text);
    }

    fn emit_literal(&mut self, span: Location) {
        let index = self.literals.len();
        self.literals.push(span);
        self.push(&format!("{EMIT_LIT_FN}({index});\n"));
    }

    fn nodes(&mut self, nodes: &[Node]) {
        for node in nodes {
            self.node(node);
        }
    }

    fn node(&mut self, node: &Node) {
        match node {
            Node::Text(span) => self.emit_literal(*span),
            Node::Interp(expr) => {
                self.push(&format!("{EMIT_FN}("));
                self.push_verbatim(*expr);
                self.push(");\n");
            }
            Node::If { arms, otherwise } => self.if_chain(arms, otherwise.as_deref()),
            Node::For {
                var,
                ty,
                iterable,
                body,
            } => {
                self.push("for ");
                self.push_verbatim(*var);
                if let Some(ty) = ty {
                    self.push(": ");
                    self.push_verbatim(*ty);
                }
                self.push(" in (");
                self.push_verbatim(*iterable);
                self.push(") {\n");
                self.nodes(body);
                self.push("}\n");
            }
        }
    }

    fn if_chain(&mut self, arms: &[Arm], otherwise: Option<&[Node]>) {
        let (first, rest) = arms.split_first().expect("if always has >= 1 arm");

        self.push("if (");
        self.push_verbatim(first.cond);
        self.push(") {\n");
        self.nodes(&first.body);
        self.push("}");

        if rest.is_empty() {
            match otherwise {
                Some(body) => {
                    self.push(" else {\n");
                    self.nodes(body);
                    self.push("}\n");
                }
                None => self.push("\n"),
            }
        } else {
            // Chain remaining arms as a nested `else { if ... }`.
            self.push(" else {\n");
            self.if_chain(rest, otherwise);
            self.push("}\n");
        }
    }
}

/// Transpiles a parsed template into a Somni program.
pub fn transpile(template: &str, nodes: &[Node]) -> Transpiled {
    let mut w = Writer {
        template,
        out: String::new(),
        literals: Vec::new(),
        map: Vec::new(),
    };

    w.push(&format!("fn {RENDER_FN}() {{\n"));
    w.nodes(nodes);
    w.push("}\n");

    Transpiled {
        source: w.out,
        literals: w.literals,
        map: w.map,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{parse::parse, syntax::Syntax};

    fn transpile_str(src: &str, syntax: &Syntax) -> Transpiled {
        let nodes = parse(src, syntax).unwrap();
        transpile(src, &nodes)
    }

    #[test]
    fn text_becomes_emit_lit() {
        let t = transpile_str("hello", &Syntax::brackets());
        assert!(t.source.contains(&format!("{EMIT_LIT_FN}(0)")));
        assert_eq!(t.literals.len(), 1);
        assert_eq!(t.literals[0].extract("hello"), "hello");
    }

    #[test]
    fn interp_becomes_emit_verbatim() {
        let src = "{{ str(age) }}";
        let t = transpile_str(src, &Syntax::brackets());
        assert!(
            t.source.contains(&format!("{EMIT_FN}(str(age))")),
            "{}",
            t.source
        );
    }

    #[test]
    fn if_else_if_chain_nests() {
        let src = "{% if a %}A{% else if b %}B{% else %}C{% endif %}";
        let t = transpile_str(src, &Syntax::brackets());
        // Two `if (` and one `else {` for the else-if plus the final else.
        assert_eq!(t.source.matches("if (").count(), 2);
        assert!(t.source.contains("else {"));
    }

    #[test]
    fn for_loop_shape() {
        let src = "#for x: int in xs\n{{ str(x) }}\n#endfor\n";
        let t = transpile_str(src, &Syntax::lines());
        assert!(t.source.contains("for x: int in (xs) {"), "{}", t.source);
    }

    #[test]
    fn source_map_translates_expression_back() {
        let src = "prefix {{ boom }} suffix";
        let t = transpile_str(src, &Syntax::brackets());
        // Find where `boom` sits in the generated source.
        let gen_idx = t.source.find("boom").unwrap();
        let gen_loc = Location {
            start: gen_idx,
            end: gen_idx + "boom".len(),
        };
        let tmpl_loc = t.map_location(gen_loc).unwrap();
        assert_eq!(tmpl_loc.extract(src), "boom");
    }
}
