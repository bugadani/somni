//! Template parser: turns the flat [`Segment`] stream into a nested [`Node`] tree.
//!
//! Directive keywords are fixed (`if` / `else` / `else if` / `endif`, `for` / `endfor`).
//! Expressions and loop iterables are kept as [`Location`]s into the template and handed to
//! Somni verbatim during transpilation.

use somni_parser::Location;

use crate::{error::TemplateError, scan::Segment, syntax::Syntax};

/// A node in the parsed template tree.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Node {
    /// Literal text; span into the template.
    Text(Location),
    /// An interpolation; span is the inner expression.
    Interp(Location),
    /// An `if` / `else if` / `else` chain.
    If {
        /// One or more conditional arms (the first is `if`, the rest are `else if`).
        arms: Vec<Arm>,
        /// The optional `else` body.
        otherwise: Option<Vec<Node>>,
    },
    /// A `for var (: type)? in iterable` loop.
    For {
        /// The loop variable identifier span.
        var: Location,
        /// The optional loop variable type identifier span.
        ty: Option<Location>,
        /// The iterable expression span.
        iterable: Location,
        /// The loop body.
        body: Vec<Node>,
    },
}

/// A single conditional arm: a condition and its body.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Arm {
    /// The condition expression span.
    pub cond: Location,
    /// The body of this arm.
    pub body: Vec<Node>,
}

/// An interpreted directive keyword and its payload.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Directive {
    If(Location),
    ElseIf(Location),
    Else,
    EndIf,
    For {
        var: Location,
        ty: Option<Location>,
        iterable: Location,
    },
    EndFor,
}

/// The first whitespace-delimited word within `span`, and the trimmed remainder span.
fn split_word(source: &str, span: Location) -> (Location, &str, Location) {
    let text = span.extract(source);
    let word_len = text
        .char_indices()
        .find(|(_, c)| c.is_whitespace())
        .map(|(i, _)| i)
        .unwrap_or(text.len());

    let word = &text[..word_len];
    let word_loc = Location {
        start: span.start,
        end: span.start + word_len,
    };

    let rest = &text[word_len..];
    let rest_trimmed = rest.trim_start();
    let rest_offset = rest.len() - rest_trimmed.len();
    let rest_start = span.start + word_len + rest_offset;
    let rest_loc = Location {
        start: rest_start,
        end: span.end,
    };

    (word_loc, word, rest_loc)
}

/// Requires `rest` to be non-empty, producing an error otherwise.
fn require_expr(rest: Location, source: &str, what: &str) -> Result<Location, TemplateError> {
    if rest.extract(source).trim().is_empty() {
        Err(TemplateError::new(
            format!("`{what}` requires an expression"),
            rest,
        ))
    } else {
        Ok(rest)
    }
}

/// Interprets the inner text of a directive segment into a [`Directive`].
fn interpret(source: &str, inner: Location) -> Result<Directive, TemplateError> {
    if inner.extract(source).trim().is_empty() {
        return Err(TemplateError::new("empty directive", inner));
    }

    let (_kw_loc, keyword, rest) = split_word(source, inner);

    match keyword {
        "if" => Ok(Directive::If(require_expr(rest, source, "if")?)),
        "endif" => Ok(Directive::EndIf),
        "for" => parse_for(source, rest),
        "endfor" => Ok(Directive::EndFor),
        "else" => {
            // Could be a bare `else` or `else if <cond>`.
            let (_, second, after) = split_word(source, rest);
            match second {
                "" => Ok(Directive::Else),
                "if" => Ok(Directive::ElseIf(require_expr(after, source, "else if")?)),
                other => Err(TemplateError::new(
                    format!("expected `if` or nothing after `else`, found `{other}`"),
                    rest,
                )),
            }
        }
        other => Err(TemplateError::new(
            format!("unknown directive keyword `{other}`"),
            inner,
        )),
    }
}

/// Parses a `for` header: `var (: type)? in iterable`.
fn parse_for(source: &str, header: Location) -> Result<Directive, TemplateError> {
    // var
    let (var, var_name, rest) = split_word_at_colon(source, header)?;
    if var_name.is_empty() {
        return Err(TemplateError::new("`for` requires a loop variable", header));
    }

    // The `: type` annotation is optional. When omitted, the loop variable's type
    // is inferred from its usage in the body.
    let rest = trim_start_loc(source, rest);
    let (ty, ty_name, after_ty) = if rest.extract(source).starts_with(':') {
        let after_colon = Location {
            start: rest.start + 1,
            end: rest.end,
        };
        let (ty, ty_name, after_ty) = split_word(source, trim_start_loc(source, after_colon));
        if ty_name.is_empty() {
            return Err(TemplateError::new(
                "`for` loop variable requires a type after `:`",
                header,
            ));
        }
        (Some(ty), Some(ty_name), after_ty)
    } else {
        (None, None, rest)
    };

    // `in`
    let (_in_loc, in_kw, iterable) = split_word(source, after_ty);
    if in_kw != "in" {
        let annotated = match ty_name {
            Some(ty_name) => format!("for {var_name}: {ty_name}"),
            None => format!("for {var_name}"),
        };
        return Err(TemplateError::new(
            format!("expected `in` after `{annotated}`, found `{in_kw}`"),
            after_ty,
        ));
    }

    let iterable = require_expr(iterable, source, "for ... in")?;

    Ok(Directive::For { var, ty, iterable })
}

/// Splits off the first word, stopping at whitespace *or* a `:` (so `x:type` also works).
fn split_word_at_colon(
    source: &str,
    span: Location,
) -> Result<(Location, &str, Location), TemplateError> {
    let text = span.extract(source);
    let word_len = text
        .char_indices()
        .find(|(_, c)| c.is_whitespace() || *c == ':')
        .map(|(i, _)| i)
        .unwrap_or(text.len());
    let word = &text[..word_len];
    let word_loc = Location {
        start: span.start,
        end: span.start + word_len,
    };
    let rest_loc = Location {
        start: span.start + word_len,
        end: span.end,
    };
    Ok((word_loc, word, rest_loc))
}

/// Returns `span` with leading whitespace removed.
fn trim_start_loc(source: &str, span: Location) -> Location {
    let text = span.extract(source);
    let trimmed = text.trim_start();
    Location {
        start: span.start + (text.len() - trimmed.len()),
        end: span.end,
    }
}

/// Parses a template source into a node tree.
pub fn parse(source: &str, syntax: &Syntax) -> Result<Vec<Node>, TemplateError> {
    let segments = crate::scan::scan(source, syntax)?;
    let mut parser = Parser {
        source,
        segments: &segments,
        pos: 0,
    };
    let nodes = parser.parse_body(&[])?;
    if parser.pos < parser.segments.len() {
        // A dangling closer (e.g. `endif` without `if`).
        let seg = parser.segments[parser.pos];
        let inner = directive_inner(&seg);
        return Err(TemplateError::new(
            format!("unexpected `{}`", inner.extract(source).trim()),
            inner,
        ));
    }
    let _ = &syntax.block; // BlockStyle is consulted only by the scanner.
    Ok(nodes)
}

fn directive_inner(seg: &Segment) -> Location {
    match seg {
        Segment::Directive { inner, .. } => *inner,
        Segment::Text(l) | Segment::Interp(l) => *l,
    }
}

/// What terminated a body.
enum Stop {
    Eof,
    ElseIf(Location),
    Else,
    EndIf,
    EndFor,
}

struct Parser<'a> {
    source: &'a str,
    segments: &'a [Segment],
    pos: usize,
}

impl Parser<'_> {
    /// Parses nodes until one of `stoppers` (or EOF) is reached, leaving `pos` pointing at
    /// the stopping directive (consumed).
    fn parse_body(
        &mut self,
        stoppers: &[fn(&Directive) -> bool],
    ) -> Result<Vec<Node>, TemplateError> {
        let (nodes, _stop) = self.parse_until(stoppers)?;
        Ok(nodes)
    }

    fn parse_until(
        &mut self,
        stoppers: &[fn(&Directive) -> bool],
    ) -> Result<(Vec<Node>, Stop), TemplateError> {
        let mut nodes = Vec::new();

        while self.pos < self.segments.len() {
            let seg = self.segments[self.pos];
            match seg {
                Segment::Text(loc) => {
                    nodes.push(Node::Text(loc));
                    self.pos += 1;
                }
                Segment::Interp(loc) => {
                    nodes.push(Node::Interp(loc));
                    self.pos += 1;
                }
                Segment::Directive { inner, .. } => {
                    let directive = interpret(self.source, inner)?;

                    if stoppers.iter().any(|s| s(&directive)) {
                        self.pos += 1;
                        return Ok((nodes, stop_of(&directive)));
                    }

                    match directive {
                        Directive::If(cond) => {
                            self.pos += 1;
                            nodes.push(self.parse_if(cond)?);
                        }
                        Directive::For { var, ty, iterable } => {
                            self.pos += 1;
                            nodes.push(self.parse_for(var, ty, iterable)?);
                        }
                        // Any closer/continuation we did not expect here.
                        Directive::EndIf => return Err(unexpected(self.source, inner, "endif")),
                        Directive::EndFor => return Err(unexpected(self.source, inner, "endfor")),
                        Directive::Else => return Err(unexpected(self.source, inner, "else")),
                        Directive::ElseIf(_) => {
                            return Err(unexpected(self.source, inner, "else if"))
                        }
                    }
                }
            }
        }

        Ok((nodes, Stop::Eof))
    }

    fn parse_if(&mut self, first_cond: Location) -> Result<Node, TemplateError> {
        let mut arms = Vec::new();
        let mut cond = first_cond;

        loop {
            let (body, stop) =
                self.parse_until(&[is_else_if as fn(&Directive) -> bool, is_else, is_endif])?;
            arms.push(Arm { cond, body });

            match stop {
                Stop::ElseIf(next) => {
                    cond = next;
                    continue;
                }
                Stop::Else => {
                    let (otherwise, else_stop) =
                        self.parse_until(&[is_endif as fn(&Directive) -> bool])?;
                    return match else_stop {
                        Stop::EndIf => Ok(Node::If {
                            arms,
                            otherwise: Some(otherwise),
                        }),
                        _ => Err(self.unterminated("if")),
                    };
                }
                Stop::EndIf => {
                    return Ok(Node::If {
                        arms,
                        otherwise: None,
                    })
                }
                _ => return Err(self.unterminated("if")),
            }
        }
    }

    fn parse_for(
        &mut self,
        var: Location,
        ty: Option<Location>,
        iterable: Location,
    ) -> Result<Node, TemplateError> {
        let (body, stop) = self.parse_until(&[is_endfor as fn(&Directive) -> bool])?;
        match stop {
            Stop::EndFor => Ok(Node::For {
                var,
                ty,
                iterable,
                body,
            }),
            _ => Err(self.unterminated("for")),
        }
    }

    fn unterminated(&self, what: &str) -> TemplateError {
        let end = self.source.len();
        TemplateError::new(
            format!("unterminated `{what}` block: missing `end{what}`"),
            Location { start: end, end },
        )
    }
}

fn unexpected(source: &str, inner: Location, what: &str) -> TemplateError {
    let _ = source;
    TemplateError::new(format!("unexpected `{what}`"), inner)
}

fn stop_of(d: &Directive) -> Stop {
    match d {
        Directive::ElseIf(c) => Stop::ElseIf(*c),
        Directive::Else => Stop::Else,
        Directive::EndIf => Stop::EndIf,
        Directive::EndFor => Stop::EndFor,
        Directive::If(_) | Directive::For { .. } => unreachable!("not a stopper"),
    }
}

fn is_else_if(d: &Directive) -> bool {
    matches!(d, Directive::ElseIf(_))
}
fn is_else(d: &Directive) -> bool {
    matches!(d, Directive::Else)
}
fn is_endif(d: &Directive) -> bool {
    matches!(d, Directive::EndIf)
}
fn is_endfor(d: &Directive) -> bool {
    matches!(d, Directive::EndFor)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ex(source: &str, loc: Location) -> &str {
        loc.extract(source)
    }

    #[test]
    fn text_and_interp() {
        let src = "a {{ b }} c";
        let nodes = parse(src, &Syntax::brackets()).unwrap();
        assert_eq!(nodes.len(), 3);
        assert!(matches!(nodes[0], Node::Text(_)));
        assert!(matches!(nodes[1], Node::Interp(_)));
        assert!(matches!(nodes[2], Node::Text(_)));
    }

    #[test]
    fn if_else_if_else() {
        let src = "{% if a %}A{% else if b %}B{% else %}C{% endif %}";
        let nodes = parse(src, &Syntax::brackets()).unwrap();
        let Node::If { arms, otherwise } = &nodes[0] else {
            panic!("expected if, got {nodes:?}");
        };
        assert_eq!(arms.len(), 2);
        assert_eq!(ex(src, arms[0].cond), "a");
        assert_eq!(ex(src, arms[1].cond), "b");
        assert!(otherwise.is_some());
    }

    #[test]
    fn for_header_parsing() {
        let src = "#for item: string in items\n{{ item }}\n#endfor\n";
        let nodes = parse(src, &Syntax::lines()).unwrap();
        let Node::For {
            var,
            ty,
            iterable,
            body,
        } = &nodes[0]
        else {
            panic!("expected for, got {nodes:?}");
        };
        assert_eq!(ex(src, *var), "item");
        assert_eq!(ex(src, ty.unwrap()), "string");
        assert_eq!(ex(src, *iterable), "items");
        assert!(body.iter().any(|n| matches!(n, Node::Interp(_))));
    }

    #[test]
    fn for_header_without_type_annotation() {
        let src = "#for item in items\n{{ item }}\n#endfor\n";
        let nodes = parse(src, &Syntax::lines()).unwrap();
        let Node::For {
            var,
            ty,
            iterable,
            body,
        } = &nodes[0]
        else {
            panic!("expected for, got {nodes:?}");
        };
        assert_eq!(ex(src, *var), "item");
        assert!(ty.is_none());
        assert_eq!(ex(src, *iterable), "items");
        assert!(body.iter().any(|n| matches!(n, Node::Interp(_))));
    }

    #[test]
    fn nested_for_in_if() {
        let src = "{% if show %}{% for x in xs %}{{ x }}{% endfor %}{% endif %}";
        let nodes = parse(src, &Syntax::brackets()).unwrap();
        let Node::If { arms, .. } = &nodes[0] else {
            panic!("expected if");
        };
        assert!(matches!(arms[0].body[0], Node::For { .. }));
    }

    #[test]
    fn dangling_endif_errors() {
        let src = "hello{% endif %}";
        let err = parse(src, &Syntax::brackets()).unwrap_err();
        assert!(
            err.message.contains("unexpected `endif`"),
            "{}",
            err.message
        );
    }

    #[test]
    fn unterminated_for_errors() {
        let src = "#for x: int in xs\n{{ x }}\n";
        let err = parse(src, &Syntax::lines()).unwrap_err();
        assert!(
            err.message.contains("unterminated `for`"),
            "{}",
            err.message
        );
    }

    #[test]
    fn omitted_type_annotation_is_allowed() {
        let src = "{% for x in xs %}{% endfor %}";
        let nodes = parse(src, &Syntax::brackets()).unwrap();
        assert!(matches!(nodes[0], Node::For { ty: None, .. }));
    }

    #[test]
    fn empty_type_annotation_still_errors() {
        let src = "{% for x: %}{% endfor %}";
        let err = parse(src, &Syntax::brackets()).unwrap_err();
        assert!(err.message.contains("type"), "{}", err.message);
    }
}
