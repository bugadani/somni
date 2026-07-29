//! Template parser: turns the flat [`Segment`] stream into a nested [`Node`] tree.
//!
//! Directive keywords are fixed (`if` / `else` / `else if` / `endif`, `for` / `endfor`,
//! `replace` / `endreplace`, `include`).
//! Expressions and loop iterables are kept as [`Location`]s into the template and handed to
//! Somni verbatim during transpilation.

use somni_parser::Location;

use crate::{error::TemplateError, scan::Segment, syntax::Syntax};

/// One `name: type = expr` binding on an [`Node::Include`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IncludeArg {
    /// Parameter name span (becomes a Somni function parameter).
    pub name: Location,
    /// Parameter type span (required; Somni function args are typed).
    pub ty: Location,
    /// Call-site expression span passed as that argument.
    pub value: Location,
}

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
    /// A `replace "literal" with expr` … `endreplace` block.
    ///
    /// At transpile time, each occurrence of `literal` in body text is rewritten into a
    /// prefix / `with_expr` / suffix emit triple. The body is otherwise a normal nested
    /// template.
    Replace {
        /// The search literal (already unescaped).
        literal: String,
        /// The `with` expression span (must evaluate to a string at render time).
        with_expr: Location,
        /// Parsed body.
        body: Vec<Node>,
    },
    /// An `include "path"` (optional `with name: type = expr, …`) directive.
    ///
    /// Without `with`, resolve expands the body in place (caller context). With `with`,
    /// resolve fills `module_id` and transpile emits a call to a dedicated include function.
    Include {
        /// Include path (unescaped, without quotes).
        path: String,
        /// Span of the quoted path (for diagnostics).
        path_span: Location,
        /// Explicit bindings passed into the include.
        args: Vec<IncludeArg>,
        /// Index into the resolved module table.
        module_id: Option<usize>,
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
#[derive(Clone, Debug, PartialEq, Eq)]
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
    Replace {
        literal: String,
        with_expr: Location,
    },
    EndReplace,
    Include {
        path: String,
        path_span: Location,
        args: Vec<IncludeArg>,
    },
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
        "if" | "IF" => Ok(Directive::If(require_expr(rest, source, "if")?)),
        "endif" | "ENDIF" => Ok(Directive::EndIf),
        "for" | "FOR" => parse_for(source, rest),
        "endfor" | "ENDFOR" => Ok(Directive::EndFor),
        "replace" | "REPLACE" => parse_replace_header(source, rest),
        "endreplace" | "ENDREPLACE" => Ok(Directive::EndReplace),
        "include" | "INCLUDE" => parse_include_header(source, rest),
        "else" | "ELSE" => {
            // Could be a bare `else` or `else if <cond>`.
            let (_, second, after) = split_word(source, rest);
            match second {
                "" => Ok(Directive::Else),
                "if" | "IF" => Ok(Directive::ElseIf(require_expr(after, source, "else if")?)),
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

/// Parses ` "LITERAL" with <expr> ` after the `replace` keyword.
fn parse_replace_header(source: &str, rest: Location) -> Result<Directive, TemplateError> {
    let rest = trim_start_loc(source, rest);
    let (literal, literal_span, after) = parse_quoted_string(source, rest)?;
    if literal.is_empty() {
        return Err(TemplateError::new(
            "`replace` literal must not be empty",
            literal_span,
        ));
    }
    let (_with_loc, with_kw, expr) = split_word(source, after);
    match with_kw {
        "with" | "WITH" => {
            let with_expr = require_expr(expr, source, "replace ... with")?;
            Ok(Directive::Replace { literal, with_expr })
        }
        "" => Err(TemplateError::new(
            "expected `with` after `replace` literal",
            after,
        )),
        other => Err(TemplateError::new(
            format!("expected `with` after `replace` literal, found `{other}`"),
            after,
        )),
    }
}

/// Parses ` "PATH" (with name: type = expr, …)? ` after the `include` keyword.
fn parse_include_header(source: &str, rest: Location) -> Result<Directive, TemplateError> {
    let rest = trim_start_loc(source, rest);
    let (path, path_span, after) = parse_quoted_string(source, rest)?;
    if path.is_empty() {
        return Err(TemplateError::new(
            "`include` path must not be empty",
            path_span,
        ));
    }
    let after = trim_start_loc(source, after);
    if after.extract(source).trim().is_empty() {
        return Ok(Directive::Include {
            path,
            path_span,
            args: Vec::new(),
        });
    }
    let (_with_loc, with_kw, args_span) = split_word(source, after);
    match with_kw {
        "with" | "WITH" => {
            let args = parse_with_args(source, args_span)?;
            Ok(Directive::Include {
                path,
                path_span,
                args,
            })
        }
        other => Err(TemplateError::new(
            format!("expected `with` after `include` path, found `{other}`"),
            after,
        )),
    }
}

/// Parses `name: type = expr (, name: type = expr)*`.
fn parse_with_args(source: &str, span: Location) -> Result<Vec<IncludeArg>, TemplateError> {
    let mut cur = trim_start_loc(source, span);
    if cur.extract(source).trim().is_empty() {
        return Err(TemplateError::new(
            "`include ... with` requires at least one binding",
            span,
        ));
    }

    let mut args = Vec::new();
    loop {
        let (name, name_str, rest) = split_word_at_colon(source, cur)?;
        if name_str.is_empty() {
            return Err(TemplateError::new(
                "expected a parameter name in `include ... with`",
                cur,
            ));
        }
        let rest = trim_start_loc(source, rest);
        if !rest.extract(source).starts_with(':') {
            return Err(TemplateError::new(
                format!("expected `: <type>` after include parameter `{name_str}`"),
                rest,
            ));
        }
        let after_colon = Location {
            start: rest.start + 1,
            end: rest.end,
        };
        let (ty, ty_str, rest) = split_word(source, trim_start_loc(source, after_colon));
        if ty_str.is_empty() {
            return Err(TemplateError::new(
                format!("include parameter `{name_str}` requires a type after `:`"),
                after_colon,
            ));
        }
        let rest = trim_start_loc(source, rest);
        if !rest.extract(source).starts_with('=') {
            return Err(TemplateError::new(
                format!("expected `=` after include parameter `{name_str}: {ty_str}`"),
                rest,
            ));
        }
        let after_eq = trim_start_loc(
            source,
            Location {
                start: rest.start + 1,
                end: rest.end,
            },
        );
        let (value, after) = split_expr_at_comma(source, after_eq)?;
        if value.extract(source).trim().is_empty() {
            return Err(TemplateError::new(
                format!("include parameter `{name_str}` requires an expression after `=`"),
                after_eq,
            ));
        }
        args.push(IncludeArg { name, ty, value });

        let after = trim_start_loc(source, after);
        if after.extract(source).trim().is_empty() {
            break;
        }
        if !after.extract(source).starts_with(',') {
            return Err(TemplateError::new(
                "expected `,` between include `with` bindings",
                after,
            ));
        }
        cur = trim_start_loc(
            source,
            Location {
                start: after.start + 1,
                end: after.end,
            },
        );
        if cur.extract(source).trim().is_empty() {
            return Err(TemplateError::new(
                "expected another binding after `,` in `include ... with`",
                after,
            ));
        }
    }
    Ok(args)
}

/// Splits `span` into an expression and the remainder at the first top-level comma.
fn split_expr_at_comma(
    source: &str,
    span: Location,
) -> Result<(Location, Location), TemplateError> {
    let text = span.extract(source);
    let mut depth_paren = 0i32;
    let mut depth_brack = 0i32;
    let mut depth_brace = 0i32;
    let mut in_string = false;
    let mut chars = text.char_indices().peekable();
    while let Some((i, ch)) = chars.next() {
        if in_string {
            if ch == '\\' {
                let _ = chars.next();
            } else if ch == '"' {
                in_string = false;
            }
            continue;
        }
        match ch {
            '"' => in_string = true,
            '(' => depth_paren += 1,
            ')' => depth_paren -= 1,
            '[' => depth_brack += 1,
            ']' => depth_brack -= 1,
            '{' => depth_brace += 1,
            '}' => depth_brace -= 1,
            ',' if depth_paren == 0 && depth_brack == 0 && depth_brace == 0 => {
                let value = Location {
                    start: span.start,
                    end: span.start + i,
                };
                let rest = Location {
                    start: span.start + i,
                    end: span.end,
                };
                return Ok((trim_end_loc(source, value), rest));
            }
            _ => {}
        }
    }
    Ok((
        trim_end_loc(source, span),
        Location {
            start: span.end,
            end: span.end,
        },
    ))
}

/// Returns `span` with trailing whitespace removed.
fn trim_end_loc(source: &str, span: Location) -> Location {
    let text = span.extract(source);
    let trimmed = text.trim_end();
    Location {
        start: span.start,
        end: span.start + trimmed.len(),
    }
}

/// Parses a double-quoted string with `\\`, `\"`, `\n`, and `\t` escapes.
///
/// Returns `(unescaped, literal_span_including_quotes, rest_after_closing_quote)`.
fn parse_quoted_string(
    source: &str,
    span: Location,
) -> Result<(String, Location, Location), TemplateError> {
    let text = span.extract(source);
    if !text.starts_with('"') {
        return Err(TemplateError::new(
            "expected a double-quoted string literal",
            span,
        ));
    }

    let mut out = String::new();
    let mut chars = text[1..].char_indices();
    while let Some((i, ch)) = chars.next() {
        match ch {
            '"' => {
                let end = span.start + 1 + i + 1;
                let literal_span = Location {
                    start: span.start,
                    end,
                };
                let after = Location {
                    start: end,
                    end: span.end,
                };
                return Ok((out, literal_span, trim_start_loc(source, after)));
            }
            '\\' => {
                let Some((_, esc)) = chars.next() else {
                    return Err(TemplateError::new(
                        "unterminated escape in string literal",
                        span,
                    ));
                };
                match esc {
                    '\\' => out.push('\\'),
                    '"' => out.push('"'),
                    'n' => out.push('\n'),
                    't' => out.push('\t'),
                    other => {
                        return Err(TemplateError::new(
                            format!("unknown string escape `\\{other}`"),
                            Location {
                                start: span.start + 1 + i,
                                end: span.start + 1 + i + 1 + other.len_utf8(),
                            },
                        ));
                    }
                }
            }
            c => out.push(c),
        }
    }

    Err(TemplateError::new(
        "unterminated string literal",
        span,
    ))
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
    if let "in" | "IN" = in_kw {
        let iterable = require_expr(iterable, source, "for ... in")?;

        Ok(Directive::For { var, ty, iterable })
    } else {
        let annotated = match ty_name {
            Some(ty_name) => format!("for {var_name}: {ty_name}"),
            None => format!("for {var_name}"),
        };
        Err(TemplateError::new(
            format!("expected `in` after `{annotated}`, found `{in_kw}`"),
            after_ty,
        ))
    }
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

/// Parses `source[from..]` into a node tree.
///
/// Locations in the returned tree are absolute into `source` (they include `from`).
pub fn parse(source: &str, syntax: &Syntax, from: usize) -> Result<Vec<Node>, TemplateError> {
    let segments = crate::scan::scan(source, syntax, from)?;
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
    EndReplace,
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
                        Directive::Replace { literal, with_expr } => {
                            self.pos += 1;
                            nodes.push(self.parse_replace(literal, with_expr)?);
                        }
                        Directive::Include {
                            path,
                            path_span,
                            args,
                        } => {
                            self.pos += 1;
                            nodes.push(Node::Include {
                                path,
                                path_span,
                                args,
                                module_id: None,
                            });
                        }
                        // Any closer/continuation we did not expect here.
                        Directive::EndIf => return Err(unexpected(self.source, inner, "endif")),
                        Directive::EndFor => return Err(unexpected(self.source, inner, "endfor")),
                        Directive::EndReplace => {
                            return Err(unexpected(self.source, inner, "endreplace"));
                        }
                        Directive::Else => return Err(unexpected(self.source, inner, "else")),
                        Directive::ElseIf(_) => {
                            return Err(unexpected(self.source, inner, "else if"));
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
                    });
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

    fn parse_replace(
        &mut self,
        literal: String,
        with_expr: Location,
    ) -> Result<Node, TemplateError> {
        let (body, stop) = self.parse_until(&[is_endreplace as fn(&Directive) -> bool])?;
        match stop {
            Stop::EndReplace => Ok(Node::Replace {
                literal,
                with_expr,
                body,
            }),
            _ => Err(self.unterminated("replace")),
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
        Directive::EndReplace => Stop::EndReplace,
        Directive::If(_)
        | Directive::For { .. }
        | Directive::Replace { .. }
        | Directive::Include { .. } => {
            unreachable!("not a stopper")
        }
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
fn is_endreplace(d: &Directive) -> bool {
    matches!(d, Directive::EndReplace)
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
        let nodes = parse(src, &Syntax::brackets(), 0).unwrap();
        assert_eq!(nodes.len(), 3);
        assert!(matches!(nodes[0], Node::Text(_)));
        assert!(matches!(nodes[1], Node::Interp(_)));
        assert!(matches!(nodes[2], Node::Text(_)));
    }

    #[test]
    fn if_else_if_else() {
        let src = "{% if a %}A{% else if b %}B{% else %}C{% endif %}";
        let nodes = parse(src, &Syntax::brackets(), 0).unwrap();
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
        let nodes = parse(src, &Syntax::lines(), 0).unwrap();
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
        let nodes = parse(src, &Syntax::lines(), 0).unwrap();
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
        let nodes = parse(src, &Syntax::brackets(), 0).unwrap();
        let Node::If { arms, .. } = &nodes[0] else {
            panic!("expected if");
        };
        assert!(matches!(arms[0].body[0], Node::For { .. }));
    }

    #[test]
    fn dangling_endif_errors() {
        let src = "hello{% endif %}";
        let err = parse(src, &Syntax::brackets(), 0).unwrap_err();
        assert!(
            err.message.contains("unexpected `endif`"),
            "{}",
            err.message
        );
    }

    #[test]
    fn unterminated_for_errors() {
        let src = "#for x: int in xs\n{{ x }}\n";
        let err = parse(src, &Syntax::lines(), 0).unwrap_err();
        assert!(
            err.message.contains("unterminated `for`"),
            "{}",
            err.message
        );
    }

    #[test]
    fn omitted_type_annotation_is_allowed() {
        let src = "{% for x in xs %}{% endfor %}";
        let nodes = parse(src, &Syntax::brackets(), 0).unwrap();
        assert!(matches!(nodes[0], Node::For { ty: None, .. }));
    }

    #[test]
    fn empty_type_annotation_still_errors() {
        let src = "{% for x: %}{% endfor %}";
        let err = parse(src, &Syntax::brackets(), 0).unwrap_err();
        assert!(err.message.contains("type"), "{}", err.message);
    }

    #[test]
    fn replace_block_parses() {
        let src = "#replace \"X\" with name\nhello X\n#endreplace\n";
        let nodes = parse(src, &Syntax::lines(), 0).unwrap();
        assert_eq!(nodes.len(), 1);
        match &nodes[0] {
            Node::Replace {
                literal,
                with_expr,
                body,
            } => {
                assert_eq!(literal, "X");
                assert_eq!(ex(src, *with_expr), "name");
                assert_eq!(body.len(), 1);
                let Node::Text(span) = &body[0] else {
                    panic!("expected text body");
                };
                assert_eq!(span.extract(src), "hello X\n");
            }
            other => panic!("expected Replace, got {other:?}"),
        }
    }

    #[test]
    fn replace_empty_literal_errors() {
        let src = "#replace \"\" with name\nx\n#endreplace\n";
        let err = parse(src, &Syntax::lines(), 0).unwrap_err();
        assert!(err.message.contains("empty"), "{}", err.message);
    }

    #[test]
    fn include_with_args_parses() {
        let src = r#"{% include "row.tmpl" with item: int = n, label: string = name %}"#;
        let nodes = parse(src, &Syntax::brackets(), 0).unwrap();
        assert_eq!(nodes.len(), 1);
        match &nodes[0] {
            Node::Include {
                path,
                args,
                module_id,
                ..
            } => {
                assert_eq!(path, "row.tmpl");
                assert!(module_id.is_none());
                assert_eq!(args.len(), 2);
                assert_eq!(ex(src, args[0].name), "item");
                assert_eq!(ex(src, args[0].ty), "int");
                assert_eq!(ex(src, args[0].value), "n");
                assert_eq!(ex(src, args[1].name), "label");
                assert_eq!(ex(src, args[1].ty), "string");
                assert_eq!(ex(src, args[1].value), "name");
            }
            other => panic!("expected Include, got {other:?}"),
        }
    }

    #[test]
    fn include_without_with_parses() {
        let src = "#include \"footer.tmpl\"\n";
        let nodes = parse(src, &Syntax::lines(), 0).unwrap();
        match &nodes[0] {
            Node::Include { path, args, .. } => {
                assert_eq!(path, "footer.tmpl");
                assert!(args.is_empty());
            }
            other => panic!("expected Include, got {other:?}"),
        }
    }

    #[test]
    fn include_requires_types_on_with() {
        let src = r#"{% include "x.tmpl" with item = n %}"#;
        let err = parse(src, &Syntax::brackets(), 0).unwrap_err();
        assert!(
            err.message.contains("type") || err.message.contains(':'),
            "{}",
            err.message
        );
    }
}
