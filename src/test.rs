use std::path::Path;

use crate::{
    compiler::{Value, compile},
    lexer::tokenize,
    parser::parse,
    vm::{EvalContext, EvalEvent},
};

pub fn run_parser_tests(dir: impl AsRef<Path>) {
    let dir = dir.as_ref();

    for entry in std::fs::read_dir(dir).unwrap() {
        let entry = entry.unwrap();
        let path = entry.path();
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
        if path.is_file() && path.extension().map_or(false, |ext| ext == "sm") {
            let program = run_compile_test(&path, true).unwrap();

            run_eval_test(&program, &path, bless);
        } else if path.is_dir() {
            run_compile_tests(&path);
        }
    }
}

pub fn run_eval_test(program: &crate::compiler::Program, path: impl AsRef<Path>, bless: bool) {
    let stderr = path.as_ref().with_extension("stderr");
    let fail_expected = stderr.exists();

    // let expressions = program
    //     .source
    //     .lines()
    //     .filter_map(|line| line.trim().strip_prefix("//@"))
    //     .collect::<Vec<_>>();

    // TODO run expressions against the program
    let mut context = EvalContext::new(&program);

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
            EvalEvent::Complete(_) => break,
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

#[track_caller]
pub fn run_compile_test(
    file: impl AsRef<Path>,
    compile_test: bool,
) -> Option<crate::compiler::Program> {
    let bless = std::env::var("BLESS").as_deref() == Ok("1");
    let file = file.as_ref();
    let source = std::fs::read_to_string(file).unwrap();

    let stderr = file.with_extension("stderr");
    let fail_expected = stderr.exists();

    let tokens = tokenize(&source).collect::<Result<Vec<_>, _>>().unwrap();
    let result = parse(&source, &tokens);

    let ast = match result {
        Ok(ast) => ast,
        Err(err) if fail_expected => {
            let expected_error = std::fs::read_to_string(&stderr).unwrap();
            if bless && format!("{err:?}") != expected_error {
                std::fs::write(&stderr, format!("{err:?}")).unwrap();
            } else {
                pretty_assertions::assert_eq!(
                    format!("{err:?}"),
                    expected_error,
                    "Compilation error did not match expected error."
                );
            }

            // Parsing failed as expected, no need to compile
            return None;
        }
        Err(err) => {
            if bless {
                std::fs::write(stderr, format!("{err:?}")).unwrap();
                // Parsing failed as expected, no need to compile
                return None;
            } else {
                panic!("Failed to parse {}:\n{err:?}", file.display());
            }
        }
    };

    if compile_test {
        match compile(&source, &ast) {
            Ok(p) => return Some(p),
            Err(err) if fail_expected => {
                if bless {
                    std::fs::write(stderr, format!("{err:?}")).unwrap();
                } else {
                    let expected_error = std::fs::read_to_string(&stderr).unwrap();
                    pretty_assertions::assert_eq!(
                        format!("{err:?}"),
                        expected_error,
                        "Compilation error did not match expected error."
                    );
                }
                return None;
            }
            Err(err) => {
                if bless {
                    std::fs::write(stderr, format!("{err:?}")).unwrap();
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
