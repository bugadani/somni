use somni_parser::parser::parse;

use crate::{codegen::Program, error::CompileError, types::VmTypeSet};

pub mod codegen;
pub mod error;
pub mod ir;
pub mod string_interner;
#[cfg(test)]
pub mod test;
pub mod transform_ir;
pub mod types;
pub mod variable_tracker;
pub mod vm;

/// A Somni compiler.
///
/// The compiler is used to turn source code into executable programs. Programs can then be executed using a [`vm::EvalContext`].
pub struct Compiler {}

impl Compiler {
    /// Creates a new compiler instance.
    pub fn new() -> Self {
        Self {}
    }

    /// Compiles Somni source code into a [`Program`].
    pub fn compile(source: &str) -> Result<Program, CompileError<'_>> {
        let ast = match parse::<VmTypeSet>(source) {
            Ok(ast) => ast,
            Err(parse_error) => {
                return Err(CompileError {
                    source,
                    location: parse_error.location,
                    error: parse_error.error,
                })
            }
        };

        let ir = match ir::Program::compile(source, &ast) {
            Ok(program) => program,
            Err(parse_error) => {
                return Err(CompileError {
                    source,
                    location: parse_error.location,
                    error: parse_error.error,
                })
            }
        };

        codegen::compile(source, ast, &ir)
    }
}

#[test]
fn test_parser() {
    test::run_parser_tests("tests/parser/");
}

fn strip_ansi(s: impl AsRef<str>) -> String {
    use ansi_parser::AnsiParser;
    fn text_block(output: ansi_parser::Output<'_>) -> Option<&str> {
        match output {
            ansi_parser::Output::TextBlock(text) => Some(text),
            _ => None,
        }
    }

    s.as_ref()
        .ansi_parse()
        .filter_map(text_block)
        .collect::<String>()
}
