//! Compile-time resolution of `include` directives.
//!
//! - **`include "path"`** (no `with`): the included body is expanded in place, sharing the
//!   caller's context.
//! - **`include "path" with name: type = expr, …`**: each distinct `(path, parameter
//!   signature)` is loaded once, compiled to its own Somni function, and call sites pass the
//!   `with` arguments.

use std::collections::HashMap;

use somni_parser::Location;

use crate::{
    error::TemplateError,
    parse::{IncludeArg, Node},
    syntax::{Syntax, resolve_syntax},
};

/// A loaded include template, ready to transpile into a dedicated function.
#[derive(Clone, Debug)]
pub struct ResolvedModule {
    /// Generated Somni function name (`__tmpl_inc_N`).
    pub fn_name: String,
    /// Parameter `(name, type)` pairs for the function header.
    pub params: Vec<(String, String)>,
    /// Parsed body (locations absolute into the shared arena).
    pub nodes: Vec<Node>,
}

/// Host callback that loads an include path to template source text.
pub type IncludeLoader<'a> = dyn FnMut(&str) -> Result<String, String> + 'a;

/// Resolves all `include` nodes under `nodes`, loading nested templates into `arena`.
///
/// `arena` must already contain the main template source (so main-node locations stay valid).
/// Included bodies are appended; their locations are absolute into the arena.
///
/// Bare includes (no `with`) are spliced into `nodes` in place. Includes with `with` get a
/// [`Node::Include::module_id`] and contribute to the returned module list.
pub fn resolve_includes(
    arena: &mut String,
    nodes: &mut Vec<Node>,
    syntax: &Syntax,
    loader: &mut IncludeLoader<'_>,
) -> Result<Vec<ResolvedModule>, TemplateError> {
    let mut state = State {
        modules: Vec::new(),
        by_key: HashMap::new(),
        stack: Vec::new(),
        syntax,
    };
    state.walk(arena, nodes, loader)?;
    Ok(state.modules)
}

struct State<'a> {
    modules: Vec<ResolvedModule>,
    by_key: HashMap<(String, Vec<(String, String)>), usize>,
    stack: Vec<String>,
    syntax: &'a Syntax,
}

impl State<'_> {
    fn walk(
        &mut self,
        arena: &mut String,
        nodes: &mut Vec<Node>,
        loader: &mut IncludeLoader<'_>,
    ) -> Result<(), TemplateError> {
        let mut out = Vec::with_capacity(nodes.len());
        for node in std::mem::take(nodes) {
            self.expand_into(arena, node, loader, &mut out)?;
        }
        *nodes = out;
        Ok(())
    }

    fn expand_into(
        &mut self,
        arena: &mut String,
        node: Node,
        loader: &mut IncludeLoader<'_>,
        out: &mut Vec<Node>,
    ) -> Result<(), TemplateError> {
        match node {
            Node::Text(_) | Node::Interp(_) => {
                out.push(node);
                Ok(())
            }
            Node::If { mut arms, mut otherwise } => {
                for arm in &mut arms {
                    self.walk(arena, &mut arm.body, loader)?;
                }
                if let Some(body) = otherwise.as_mut() {
                    self.walk(arena, body, loader)?;
                }
                out.push(Node::If { arms, otherwise });
                Ok(())
            }
            Node::For {
                var,
                ty,
                iterable,
                mut body,
            } => {
                self.walk(arena, &mut body, loader)?;
                out.push(Node::For {
                    var,
                    ty,
                    iterable,
                    body,
                });
                Ok(())
            }
            Node::Replace {
                literal,
                with_expr,
                mut body,
            } => {
                self.walk(arena, &mut body, loader)?;
                out.push(Node::Replace {
                    literal,
                    with_expr,
                    body,
                });
                Ok(())
            }
            Node::Include {
                path,
                path_span,
                args,
                module_id: _,
            } if args.is_empty() => {
                // Syntactic include: expand the body here so it sees caller locals.
                let mut body = self.load_body(arena, &path, path_span, loader)?;
                self.stack.push(path);
                self.walk(arena, &mut body, loader)?;
                self.stack.pop();
                out.extend(body);
                Ok(())
            }
            Node::Include {
                path,
                path_span,
                args,
                module_id: _,
            } => {
                let id = self.resolve_module(arena, &path, path_span, &args, loader)?;
                out.push(Node::Include {
                    path,
                    path_span,
                    args,
                    module_id: Some(id),
                });
                Ok(())
            }
        }
    }

    fn load_body(
        &mut self,
        arena: &mut String,
        path: &str,
        path_span: Location,
        loader: &mut IncludeLoader<'_>,
    ) -> Result<Vec<Node>, TemplateError> {
        if self.stack.iter().any(|p| p == path) {
            let chain = self
                .stack
                .iter()
                .chain(std::iter::once(&path.to_string()))
                .cloned()
                .collect::<Vec<_>>()
                .join(" -> ");
            return Err(TemplateError::new(
                format!("cyclic include: {chain}"),
                path_span,
            ));
        }

        let content = loader(path).map_err(|e| {
            TemplateError::new(format!("failed to load include `{path}`: {e}"), path_span)
        })?;

        let (inc_syntax, body, _) = resolve_syntax(&content, self.syntax).map_err(|err| {
            TemplateError::new(
                format!("in include `{path}`: {}", err.message),
                path_span,
            )
        })?;

        let base = arena.len();
        // Separate included bodies so diagnostics don't glue onto the previous line.
        if base > 0 && !arena.ends_with('\n') {
            arena.push('\n');
        }
        let base = arena.len();
        arena.push_str(body);

        crate::parse::parse(arena, &inc_syntax, base).map_err(|err| {
            TemplateError::new(
                format!("in include `{path}`: {}", err.message),
                err.location,
            )
        })
    }

    fn resolve_module(
        &mut self,
        arena: &mut String,
        path: &str,
        path_span: Location,
        args: &[IncludeArg],
        loader: &mut IncludeLoader<'_>,
    ) -> Result<usize, TemplateError> {
        let params: Vec<(String, String)> = args
            .iter()
            .map(|a| {
                (
                    a.name.extract(arena).to_string(),
                    a.ty.extract(arena).to_string(),
                )
            })
            .collect();
        let key = (path.to_string(), params.clone());
        if let Some(&id) = self.by_key.get(&key) {
            return Ok(id);
        }

        let mut nodes = self.load_body(arena, path, path_span, loader)?;

        self.stack.push(path.to_string());
        self.walk(arena, &mut nodes, loader)?;
        self.stack.pop();

        let id = self.modules.len();
        let fn_name = format!("__tmpl_inc_{id}");
        self.modules.push(ResolvedModule {
            fn_name,
            params,
            nodes,
        });
        self.by_key.insert(key, id);
        Ok(id)
    }
}
