pub mod codegen;
pub mod error;
pub mod ir;
#[cfg(test)]
pub mod test;
pub mod transform_ir;
pub mod variable_tracker;
pub mod vm;

#[test]
fn test_parser() {
    test::run_parser_tests("tests/parser/");
}
