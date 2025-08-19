use std::collections::HashMap;

use somni_parser::parser::parse;

use crate::{
    codegen::Program,
    error::CompileError,
    types::VmTypeSet,
    vm::{NativeFunction, SomniFn},
};

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
pub struct Compiler<'ctx> {
    functions: HashMap<&'ctx str, SomniFn<'ctx>>,
}

impl<'ctx> Compiler<'ctx> {
    /// Creates a new compiler instance.
    pub fn new() -> Self {
        Self {
            functions: HashMap::new(),
        }
    }

    /// Compiles Somni source code into a [`Program`].
    pub fn compile<'s>(&mut self, source: &'s str) -> Result<Program, CompileError<'s>> {
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

        codegen::compile(source, ast, &ir, &self.functions)
    }

    /// Adds a new function to the compiler context.
    ///
    /// These functions are executed if they are needed to evaluate global variable initial values.
    ///
    /// If a program refers to a foreign function that is called both during compilation, and also in runtime,
    /// the function registered to the compiler and the [`vm::EvalContext`] should behave the same.
    ///
    /// ## Example
    ///
    /// ```rust
    /// use somni::Compiler;
    ///
    /// let mut compiler = Compiler::new();
    ///
    /// compiler.add_function("rust_fn", |a: u64| a + 3);
    ///
    /// compiler.compile(r#"
    ///     extern fn rust_fn() -> int;
    ///
    ///     var a: int = foo();
    ///
    ///     fn foo() -> int {
    ///         rust_fn(2) // returns 5 from Rust code
    ///     }
    /// "#).unwrap();
    pub fn add_function<F, A>(&mut self, name: &'ctx str, func: F)
    where
        F: NativeFunction<A> + 'ctx,
    {
        self.functions.insert(name, SomniFn::new(func));
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
