use std::path::Path;

use crate::{
    codegen::{self, Value},
    ir,
    lexer::tokenize,
    parser::parse,
    vm::{EvalContext, EvalEvent},
};

pub fn run_parser_tests(dir: impl AsRef<Path>) {
    let dir = dir.as_ref();

    for entry in std::fs::read_dir(dir).unwrap() {
        let entry = entry.unwrap();
        let path = entry.path();

        if !filter(&path) {
            continue;
        }

        if path.is_file() && path.extension().map_or(false, |ext| ext == "sm") {
            run_compile_test(&path, false);
        } else if path.is_dir() {
            run_parser_tests(&path);
        }
    }
}

pub fn run_compile_tests(dir: impl AsRef<Path>) {
    let dir = dir.as_ref();

    for entry in std::fs::read_dir(dir).unwrap() {
        let entry = entry.unwrap();
        let path = entry.path();

        if !filter(&path) {
            continue;
        }

        if path.is_file() && path.extension().map_or(false, |ext| ext == "sm") {
            run_compile_test(&path, true);
        } else if path.is_dir() {
            run_compile_tests(&path);
        }
    }
}

pub fn run_eval_tests(dir: impl AsRef<Path>) {
    let bless = std::env::var("BLESS").as_deref() == Ok("1");
    let dir = dir.as_ref();

    for entry in std::fs::read_dir(dir).unwrap() {
        let entry = entry.unwrap();
        let path = entry.path();

        if !filter(&path) {
            continue;
        }

        if path.is_file() && path.extension().map_or(false, |ext| ext == "sm") {
            let program = run_compile_test(&path, true).unwrap();

            run_eval_test(&program, &path, bless);
        } else if path.is_dir() {
            run_compile_tests(&path);
        }
    }
}

pub fn run_eval_test(program: &codegen::Program, path: impl AsRef<Path>, bless: bool) {
    let stderr = path.as_ref().with_extension("stderr");
    let fail_expected = stderr.exists();

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
                let error = e.mark(&context, "Runtime error").to_string();
                if fail_expected && bless {
                    std::fs::write(&stderr, error).unwrap();
                } else if fail_expected {
                    let expected_error = std::fs::read_to_string(&stderr).unwrap();
                    pretty_assertions::assert_eq!(
                        error,
                        expected_error,
                        "Runtime error did not match expected error."
                    );
                } else {
                    panic!("{error} while running {}", path.as_ref().display());
                }
                break;
            }
            EvalEvent::Complete(value) => {
                assert!(
                    value == Value::Bool(true),
                    "Test {} exited with an unexpected value: {value:?}",
                    path.as_ref().display()
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
    let bless = std::env::var("BLESS").as_deref() == Ok("1");
    let file = file.as_ref();
    let source = std::fs::read_to_string(file).unwrap();

    let out_path = file.parent().unwrap().join(file.file_stem().unwrap());

    let write_out_file = |name: &Path, content: String| {
        _ = std::fs::create_dir_all(&name.parent().unwrap()).unwrap();
        std::fs::write(name, strip_ansi(content)).unwrap();
    };

    let stderr = out_path.join("stderr");
    let fail_expected = stderr.exists();

    let tokens = tokenize(&source).collect::<Result<Vec<_>, _>>().unwrap();
    let result = parse(&source, &tokens);

    let ast = match result {
        Ok(ast) => {
            write_out_file(&out_path.join("ast"), format!("{:#?}", ast));
            ast
        }
        Err(err) if fail_expected => {
            compare_error(&stderr, &err, bless, file);
            // Parsing failed as expected, no need to compile
            return None;
        }
        Err(err) => {
            if bless {
                write_out_file(&stderr, format!("{err:?}"));
                // Parsing failed as expected, no need to compile
                return None;
            } else {
                panic!("Failed to parse {}:\n{err:?}", file.display());
            }
        }
    };

    if compile_test {
        let program = ir::Program::compile(&source, &ast).unwrap();

        write_out_file(&out_path.join("ir.disasm"), program.print());

        match codegen::compile(&source, &program) {
            Ok(p) => {
                write_out_file(
                    &out_path.join("unoptimized.disasm"),
                    p.disasm(&p.debug_info),
                );
                return Some(p);
            }
            Err(err) if fail_expected => {
                compare_error(&stderr, &err, bless, file);
                return None;
            }
            Err(err) => {
                if bless {
                    write_out_file(&stderr, format!("{err:?}"));
                    return None;
                } else {
                    panic!("Failed to compile {}:\n{err:?}", file.display());
                }
            }
        }
    }

    if fail_expected {
        panic!(
            "Expected {} to fail, but it compiled successfully.",
            file.display()
        );
    }

    None
}

fn compare_error(stderr: &Path, err: &crate::error::CompileError, bless: bool, file: &Path) {
    let write_out_file = |name: &Path, content: String| {
        _ = std::fs::create_dir_all(&name.parent().unwrap()).unwrap();
        std::fs::write(name, strip_ansi(content)).unwrap();
    };

    let error = strip_ansi(format!("{err:?}"));
    if bless {
        write_out_file(&stderr, error);
    } else {
        let expected_error = std::fs::read_to_string(&stderr).unwrap();
        pretty_assertions::assert_eq!(
            error,
            expected_error,
            "Compilation error did not match expected error. {}",
            file.display()
        );
    }
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
        return true; // No filter set, include all paths
    };

    Path::new(&env) == path
}
