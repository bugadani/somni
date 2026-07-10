use std::{
    collections::HashMap,
    path::{Path, PathBuf},
};

use crate::{
    codegen,
    error::CompileError,
    ir, strip_ansi,
    transform_ir::transform_ir,
    vm::{EvalContext, TypedValue},
};
use somni_parser::parser::parse;

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
        run_compile_test(path, false, false);
    });
}

pub fn run_compile_tests(dir: impl AsRef<Path>) {
    walk(dir.as_ref(), &|path| {
        run_compile_test(path, true, false);
    });
}

pub fn run_eval_tests(dir: impl AsRef<Path>) {
    walk(dir.as_ref(), &|path| {
        if let Some(program) = run_compile_test(&path, true, true) {
            run_eval_test(program, path);
        }
    });
}

pub fn run_eval_test(program: codegen::Program, path: impl AsRef<Path>) {
    let ctx = TestContext::from_path(path.as_ref());

    let expressions = program
        .debug_info
        .source
        .lines()
        .filter_map(|line| line.trim().strip_prefix("//@"))
        .collect::<Vec<_>>();

    let mut strings = program.debug_info.strings.clone();
    let mut context = EvalContext::new(&program.debug_info.source, &mut strings, &program);

    context.add_function("add_from_rust", |a: u64, b: u64| -> i64 { (a + b) as i64 });
    context.add_function("assert", |a: bool| a); // No-op to test calling Rust functions from expressions
    context.add_function("reverse", |s: &str| s.chars().rev().collect::<String>());
    context.add_function("range", |start: u64, end: u64| {
        crate::types::SomniIter::new((start..end).map(somni_expr::TypedValue::Int))
    });

    for expression in &expressions {
        let expression = if let Some(e) = expression.strip_prefix('+') {
            // `//@+` preserves VM state (like changes to globals)
            e.trim()
        } else {
            // `//@` resets VM state (like changes to globals)
            context.reset();
            expression
        };
        println!("Running `{expression}`");
        let value = context.eval_expression::<TypedValue>(expression);
        match value {
            Ok(TypedValue::Bool(true)) => {}
            Ok(not_true) => {
                ctx.handle_error_string(
                    "Expression did not evaluate to true",
                    format!("Expression `{expression}` evaluated to {not_true:?}"),
                );
            }
            Err(e) => ctx.handle_error_string(
                &format!("Failed to evaluate `{expression}`"),
                e.as_str().to_string(),
            ),
        }
    }
}

pub fn run_compile_test(
    file: impl AsRef<Path>,
    compile_test: bool,
    eval_test: bool,
) -> Option<codegen::Program> {
    println!("Compiling: {}", file.as_ref().display());
    let ctx = TestContext::from_path(file.as_ref());

    // First, parse the program.
    let source = std::fs::read_to_string(ctx.file).unwrap();
    let ast = match parse(&source) {
        Ok(ast) => ast,
        Err(err) => return ctx.handle_error("Failed to parse", CompileError::new(&source, err)),
    };
    write_out_file(&ctx.out_path.join("ast"), format!("{:#?}", ast));

    let mut program = None;
    if compile_test {
        // Compile the program to IR.
        let mut ir = match ir::Program::compile(&source, &ast) {
            Ok(ir) => ir,
            Err(err) => return ctx.handle_error("Failed to compile to IR", err),
        };
        write_out_file(&ctx.out_path.join("ir.disasm"), ir.print());

        if let Err(e) = transform_ir(&source, &mut ir) {
            return ctx.handle_error("Failed to transform IR", e);
        }
        write_out_file(&ctx.out_path.join("ir.transformed.disasm"), ir.print());

        // Compile the IR to a program.
        let p = match codegen::compile(&source, ast, &ir, &HashMap::new()) {
            Ok(program) => program,
            Err(err) => return ctx.handle_error("Failed to compile", err),
        };
        write_out_file(
            &ctx.out_path.join("unoptimized.disasm"),
            p.disasm(&p.debug_info),
        );

        program = Some(p);
    }

    if ctx.fail_expected() && !eval_test {
        panic!(
            "Expected {} to fail, but it compiled successfully.",
            ctx.file.display()
        );
    }

    program
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

    #[track_caller]
    fn handle_error<T>(&self, failure: &str, err: CompileError<'_>) -> Option<T> {
        self.handle_error_string(failure, format!("{err:?}"));
        None
    }

    fn is_blessed(&self) -> bool {
        std::env::var("BLESS").as_deref() == Ok("1")
    }

    #[track_caller]
    fn handle_error_string(&self, failure: &str, error: String) {
        let error = strip_ansi(error);
        if self.is_blessed() {
            write_out_file(&self.stderr(), error);
        } else if self.fail_expected() {
            let expected_error = std::fs::read_to_string(&self.stderr()).unwrap();
            pretty_assertions::assert_eq!(
                expected_error.trim(),
                error.trim(),
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

fn filter(path: &Path) -> bool {
    let Ok(env) = std::env::var("TEST_FILTER") else {
        // No filter set, walk folders and somni source files.
        return path.is_dir() || path.extension().map_or(false, |ext| ext == "sm");
    };

    Path::new(&env) == path
}

/// Verifies that the `somni-expr` tree-walking evaluator (not the VM) can execute
/// `for` loops, using an iterator-registry-backed type set.
#[cfg(test)]
mod expr_for_loop_tests {
    use somni_expr::{Context, TypedValue};

    use crate::types::{SomniIter, VmTypeSet};

    fn run(program: &str, expr: &str) -> TypedValue<VmTypeSet> {
        let mut ctx = Context::<VmTypeSet>::parse_with_types(program).expect("failed to parse");
        ctx.add_function("range", |a: u64, b: u64| {
            SomniIter::new((a..b).map(TypedValue::<VmTypeSet>::Int))
        });
        ctx.evaluate::<TypedValue<VmTypeSet>>(expr)
            .expect("failed to evaluate")
    }

    const PROGRAM: &str = r#"
extern fn range(start: int, end: int) -> iter;

fn sum_range(start: int, end: int) -> int {
    var total = 0;
    for x: int in range(start, end) {
        total = total + x;
    }
    return total;
}

fn skip_and_stop() -> int {
    var total = 0;
    for x: int in range(0, 10) {
        if x == 3 {
            continue;
        }
        if x == 6 {
            break;
        }
        total = total + x;
    }
    return total;
}

fn nested() -> int {
    var total = 0;
    for a: int in range(0, 3) {
        for b: int in range(0, 3) {
            total = total + a * b;
        }
    }
    return total;
}

fn sum_range_inferred(start: int, end: int) -> int {
    var total = 0;
    // No type annotation: `x` is inferred from its use in `total + x`.
    for x in range(start, end) {
        total = total + x;
    }
    return total;
}
"#;

    #[test]
    fn sums_a_range() {
        assert_eq!(run(PROGRAM, "sum_range(0, 5)"), TypedValue::Int(10));
        assert_eq!(run(PROGRAM, "sum_range(3, 3)"), TypedValue::Int(0));
    }

    #[test]
    fn supports_break_and_continue() {
        // 0 + 1 + 2 + (skip 3) + 4 + 5 = 12
        assert_eq!(run(PROGRAM, "skip_and_stop()"), TypedValue::Int(12));
    }

    #[test]
    fn nested_for_loops() {
        assert_eq!(run(PROGRAM, "nested()"), TypedValue::Int(9));
    }

    #[test]
    fn sums_a_range_without_type_annotation() {
        assert_eq!(run(PROGRAM, "sum_range_inferred(0, 5)"), TypedValue::Int(10));
        assert_eq!(run(PROGRAM, "sum_range_inferred(3, 3)"), TypedValue::Int(0));
    }
}
