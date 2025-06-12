use std::path::{Path, PathBuf};

use crate::{
    codegen::{self, Value},
    error::CompileError,
    ir,
    lexer::tokenize,
    parser::parse,
    vm::{EvalContext, EvalEvent},
};

fn walk(dir: &Path, on_file: &impl Fn(&Path)) {
    for entry in std::fs::read_dir(dir).unwrap().flatten() {
        let path = entry.path();

        if !filter(&path) {
            continue;
        }

        if path.is_file() {
            on_file(&path);
        } else {
            walk(&path, on_file);
        }
    }
}

pub fn run_parser_tests(dir: impl AsRef<Path>) {
    walk(dir.as_ref(), &|path| {
        run_compile_test(path, false);
    });
}

pub fn run_compile_tests(dir: impl AsRef<Path>) {
    walk(dir.as_ref(), &|path| {
        run_compile_test(path, true);
    });
}

pub fn run_eval_tests(dir: impl AsRef<Path>) {
    walk(dir.as_ref(), &|path| {
        if let Some(program) = run_compile_test(&path, true) {
            run_eval_test(program, path);
        }
    });
}

pub fn run_eval_test(program: codegen::Program, path: impl AsRef<Path>) {
    let ctx = TestContext::from_path(path.as_ref());

    // let expressions = program
    //     .source
    //     .lines()
    //     .filter_map(|line| line.trim().strip_prefix("//@"))
    //     .collect::<Vec<_>>();

    // TODO run expressions against the program
    let mut context = EvalContext::new(
        &program.debug_info.source,
        &program.debug_info.strings,
        &program,
    );

    loop {
        match context.run() {
            EvalEvent::Error(e) => {
                let error = e.mark(&context, "Runtime error");

                ctx.handle_error_string("Runtime error", error.to_string());
                break;
            }
            EvalEvent::Complete(value) => {
                assert!(
                    value == Value::Bool(true),
                    "Test {} exited with an unexpected value: {value:?}",
                    ctx.file.display()
                );
                break;
            }
            EvalEvent::UnknownFunctionCall => {
                let (name, args) = context.unknown_function_info().expect("No function info");
                if name == "assert" {
                    // TODO: generating a type error is not a great way to handle failures
                    match args[0] {
                        Value::Bool(true) => context.set_return_value(Value::Bool(true)).unwrap(),
                        Value::Bool(false) => context.set_return_value(Value::Void).unwrap(),
                        _ => {}
                    }
                }
            }
        }
    }
}

pub fn run_compile_test(file: impl AsRef<Path>, compile_test: bool) -> Option<codegen::Program> {
    let ctx = TestContext::from_path(file.as_ref());

    // First, parse the program.
    let source = std::fs::read_to_string(ctx.file).unwrap();
    let tokens = tokenize(&source).collect::<Result<Vec<_>, _>>().unwrap();
    let ast = match parse(&source, &tokens) {
        Ok(ast) => ast,
        Err(err) => return ctx.handle_error("Failed to parse", err),
    };
    write_out_file(&ctx.out_path.join("ast"), format!("{:#?}", ast));

    if compile_test {
        // Compile the program to IR.
        let ir = match ir::Program::compile(&source, &ast) {
            Ok(ir) => ir,
            Err(err) => return ctx.handle_error("Failed to compile to IR", err),
        };
        write_out_file(&ctx.out_path.join("ir.disasm"), ir.print());

        // Compile the IR to a program.
        let program = match codegen::compile(&source, &ir) {
            Ok(program) => program,
            Err(err) => return ctx.handle_error("Failed to compile", err),
        };
        write_out_file(
            &ctx.out_path.join("unoptimized.disasm"),
            program.disasm(&program.debug_info),
        );

        return Some(program);
    }

    if ctx.fail_expected() {
        panic!(
            "Expected {} to fail, but it compiled successfully.",
            ctx.file.display()
        );
    }

    None
}

struct TestContext<'a> {
    out_path: PathBuf,
    file: &'a Path,
}

impl<'a> TestContext<'a> {
    fn from_path(file: &'a Path) -> Self {
        Self {
            out_path: file.parent().unwrap().join(file.file_stem().unwrap()),
            file,
        }
    }

    fn stderr(&self) -> PathBuf {
        self.out_path.join("stderr")
    }

    fn fail_expected(&self) -> bool {
        self.stderr().exists()
    }

    fn handle_error<T>(&self, failure: &str, err: CompileError<'_>) -> Option<T> {
        self.handle_error_string(failure, format!("{err:?}"));
        None
    }

    fn is_blessed(&self) -> bool {
        std::env::var("BLESS").as_deref() == Ok("1")
    }

    fn handle_error_string(&self, failure: &str, error: String) {
        let error = strip_ansi(error);
        if self.fail_expected() || self.is_blessed() {
            write_out_file(&self.stderr(), error);
        } else if self.fail_expected() {
            let expected_error = std::fs::read_to_string(&self.stderr()).unwrap();
            pretty_assertions::assert_eq!(
                error,
                expected_error,
                "Error did not match expected error. {}",
                self.file.display()
            );
        } else {
            panic!("{}: {failure}\n{error}", self.file.display());
        }
    }
}

fn write_out_file(name: &Path, content: String) {
    _ = std::fs::create_dir_all(&name.parent().unwrap()).unwrap();
    std::fs::write(name, content).unwrap();
}

pub fn strip_ansi(s: impl AsRef<str>) -> String {
    use ansi_parser::AnsiParser;
    fn text_block(output: ansi_parser::Output) -> Option<&str> {
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

fn filter(path: &Path) -> bool {
    let Ok(env) = std::env::var("TEST_FILTER") else {
        // No filter set, walk folders and somni source files.
        return path.is_dir() || path.extension().map_or(false, |ext| ext == "sm");
    };

    Path::new(&env) == path
}
