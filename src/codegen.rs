use std::{
    collections::{HashMap, hash_map::Entry},
    fmt::Display,
    ops::{Add, AddAssign},
};

use indexmap::IndexMap;

use crate::{
    error::CompileError,
    ir::{self, VariableIndex},
    string_interner::{StringIndex, Strings},
    variable_tracker::{self, LocalVariableIndex},
};
use somni_lexer::Location;

// This is just to keep the size of Instruction small enough. Re-evaluate this later.
#[derive(Clone, Copy, Debug)]
pub enum Register {
    /// This register contains the address where the result will be stored.
    Dst,
    /// Contains the address of the left-hand side operand (or the operand in unary operations).
    Lhs,
    /// Contains the address of the right-hand side operand.
    Rhs,
}

#[derive(Clone, Copy, Debug)]
pub enum Instruction {
    /// Return with the value from the eval stack
    Return,
    Jump(CodeAddress),
    JumpIfFalse(MemoryAddress, CodeAddress),
    Copy(MemoryAddress, MemoryAddress),
    DerefCopy(MemoryAddress, MemoryAddress),
    LoadValue(MemoryAddress, Value),
    Call(usize, MemoryAddress),
    CallNamed(StringIndex, MemoryAddress, usize),

    /// Unary operations
    Negate(MemoryAddress, MemoryAddress),
    Not(MemoryAddress, MemoryAddress),
    Dereference(MemoryAddress, MemoryAddress),
    Address(MemoryAddress, MemoryAddress),

    // Binary operations
    TestLessThan(MemoryAddress, MemoryAddress, MemoryAddress),
    TestLessThanOrEqual(MemoryAddress, MemoryAddress, MemoryAddress),
    TestEquals(MemoryAddress, MemoryAddress, MemoryAddress),
    TestNotEquals(MemoryAddress, MemoryAddress, MemoryAddress),
    BitwiseOr(MemoryAddress, MemoryAddress, MemoryAddress),
    BitwiseXor(MemoryAddress, MemoryAddress, MemoryAddress),
    BitwiseAnd(MemoryAddress, MemoryAddress, MemoryAddress),
    ShiftLeft(MemoryAddress, MemoryAddress, MemoryAddress),
    ShiftRight(MemoryAddress, MemoryAddress, MemoryAddress),
    Add(MemoryAddress, MemoryAddress, MemoryAddress),
    Subtract(MemoryAddress, MemoryAddress, MemoryAddress),
    Multiply(MemoryAddress, MemoryAddress, MemoryAddress),
    Divide(MemoryAddress, MemoryAddress, MemoryAddress),
}

impl Instruction {
    fn disasm(&self, debug_info: &DebugInfo) -> String {
        match self {
            Instruction::Return => "return".to_string(),
            Instruction::Jump(address) => format!("jump to {}", address.0),
            Instruction::JumpIfFalse(addr, address) => {
                format!("if {:?} == false jump to {}", addr, address.0)
            }
            Instruction::Copy(dst, src) => format!("copy {dst:?} = {src:?}"),
            Instruction::DerefCopy(dst, src) => format!("copy *{dst:?} = {src:?}"),
            Instruction::LoadValue(dst, value) => format!("load {dst:?} = {value:?}"),
            Instruction::Call(fn_index, stack_frame) => {
                let fn_name = debug_info
                    .function_names
                    .get(*fn_index)
                    .map(|&name| debug_info.strings.lookup(name))
                    .unwrap_or("unknown function");
                format!("call {fn_name}(..) with sp={stack_frame:?}")
            }
            Instruction::CallNamed(name, stack_frame, arg_count) => {
                let fn_name = debug_info.strings.lookup(*name);
                let arg_placeholders = (0..*arg_count)
                    .map(|i| format!("arg{i}"))
                    .collect::<Vec<_>>()
                    .join(", ");
                format!("call {fn_name}({arg_placeholders}) with sp={stack_frame:?}")
            }
            Instruction::Negate(dst, src) => format!("{dst:?} = -{src:?}"),
            Instruction::Not(dst, src) => format!("{dst:?} = !{src:?}"),
            Instruction::Dereference(dst, src) => format!("{dst:?} = *{src:?}"),
            Instruction::Address(dst, src) => format!("{dst:?} = &{src:?}"),
            Instruction::TestLessThan(dst, lhs, rhs) => {
                format!("{dst:?} = {lhs:?} < {rhs:?}")
            }
            Instruction::TestLessThanOrEqual(dst, lhs, rhs) => {
                format!("{dst:?} = {lhs:?} <= {rhs:?}")
            }
            Instruction::TestEquals(dst, lhs, rhs) => {
                format!("{dst:?} = {lhs:?} == {rhs:?}")
            }
            Instruction::TestNotEquals(dst, lhs, rhs) => {
                format!("{dst:?} = {lhs:?} != {rhs:?}")
            }
            Instruction::BitwiseOr(dst, lhs, rhs) => {
                format!("{dst:?} = {lhs:?} | {rhs:?}")
            }
            Instruction::BitwiseXor(dst, lhs, rhs) => {
                format!("{dst:?} = {lhs:?} ^ {rhs:?}")
            }
            Instruction::BitwiseAnd(dst, lhs, rhs) => {
                format!("{dst:?} = {lhs:?} & {rhs:?}")
            }
            Instruction::ShiftLeft(dst, lhs, rhs) => {
                format!("{dst:?} = {lhs:?} << {rhs:?}")
            }
            Instruction::ShiftRight(dst, lhs, rhs) => {
                format!("{dst:?} = {lhs:?} >> {rhs:?}")
            }
            Instruction::Add(dst, lhs, rhs) => {
                format!("{dst:?} = {lhs:?} + {rhs:?}")
            }
            Instruction::Subtract(dst, lhs, rhs) => {
                format!("{dst:?} = {lhs:?} - {rhs:?}")
            }
            Instruction::Multiply(dst, lhs, rhs) => {
                format!("{dst:?} = {lhs:?} * {rhs:?}")
            }
            Instruction::Divide(dst, lhs, rhs) => {
                format!("{dst:?} = {lhs:?} / {rhs:?}")
            }
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Bytecode {
    pub instruction: Instruction,
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub struct CodeAddress(pub usize);

impl Add<usize> for CodeAddress {
    type Output = CodeAddress;

    fn add(self, rhs: usize) -> Self::Output {
        CodeAddress(self.0 + rhs)
    }
}

impl AddAssign<usize> for CodeAddress {
    fn add_assign(&mut self, rhs: usize) {
        self.0 += rhs;
    }
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum MemoryAddress {
    Global(usize),
    Local(usize),
}

impl Add<usize> for MemoryAddress {
    type Output = MemoryAddress;

    fn add(self, rhs: usize) -> Self::Output {
        match self {
            MemoryAddress::Global(addr) => MemoryAddress::Global(addr + rhs),
            MemoryAddress::Local(addr) => MemoryAddress::Local(addr + rhs),
        }
    }
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Value {
    Void,
    Int(u64),
    SignedInt(i64),
    Float(f64),
    Bool(bool),
    String(StringIndex),
    Address(MemoryAddress),
}

impl From<&ir::Value> for Value {
    fn from(value: &ir::Value) -> Self {
        match value {
            ir::Value::Void => Value::Void,
            ir::Value::Int(value) => Value::Int(*value),
            ir::Value::SignedInt(value) => Value::SignedInt(*value),
            ir::Value::Bool(value) => Value::Bool(*value),
            ir::Value::String(value) => Value::String(*value),
            ir::Value::Float(value) => Value::Float(*value),
        }
    }
}

impl Value {
    pub fn type_of(&self) -> Type {
        match self {
            Value::Void => Type::Void,
            Value::Int(_) => Type::Int,
            Value::SignedInt(_) => Type::SignedInt,
            Value::Bool(_) => Type::Bool,
            Value::String(_) => Type::String,
            Value::Float(_) => Type::Float,
            Value::Address(_) => Type::Address,
        }
    }
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub enum Type {
    Void,
    Int,
    SignedInt,
    Float,
    Bool,
    String,
    Address,
}

impl From<ir::Type> for Type {
    fn from(ty: ir::Type) -> Self {
        match ty {
            ir::Type::Void => Type::Void,
            ir::Type::Int => Type::Int,
            ir::Type::SignedInt => Type::SignedInt,
            ir::Type::Float => Type::Float,
            ir::Type::Bool => Type::Bool,
            ir::Type::String => Type::String,
            ir::Type::Address => Type::Address,
        }
    }
}

impl Display for Type {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Type::Void => write!(f, "void"),
            Type::Int => write!(f, "int"),
            Type::SignedInt => write!(f, "signed"),
            Type::Bool => write!(f, "bool"),
            Type::String => write!(f, "string"),
            Type::Float => write!(f, "float"),
            Type::Address => write!(f, "address"),
        }
    }
}

#[derive(Clone, Debug)]
pub struct GlobalVariable {
    initial_value: Value,
}

impl GlobalVariable {
    pub fn value(&self) -> &Value {
        &self.initial_value
    }
}

#[derive(Clone, Debug)]
pub struct Function {
    pub name: StringIndex, // Name of the function
    pub entry_point: CodeAddress,
    pub stack_size: usize,
    pub return_type: Type,
}

#[derive(Clone, Debug, Default)]
pub struct DebugInfo {
    pub source: String,
    pub strings: Strings,
    pub instruction_locations: Vec<Location>,
    pub function_names: Vec<StringIndex>,
}

#[derive(Clone, Debug)]
pub struct Program {
    pub code: Vec<Instruction>,
    pub globals: IndexMap<StringIndex, GlobalVariable>,
    pub functions: IndexMap<StringIndex, Function>,
    pub debug_info: DebugInfo,
}

impl Program {
    pub fn disasm(&self, debug_info: &DebugInfo) -> String {
        let mut output = String::new();
        for (i, instruction) in self.code.iter().enumerate() {
            if let Some(function) = self
                .functions
                .values()
                .position(|func| func.entry_point == CodeAddress(i))
            {
                let (name, data) = self.functions.get_index(function).unwrap();
                output.push_str(&format!(
                    "Function {}(...) -> {} (stack: {})\n",
                    debug_info.strings.lookup(*name),
                    data.return_type,
                    data.stack_size
                ));
            }

            output.push_str(&format!("  {:04}: {}\n", i, instruction.disasm(debug_info)));
        }
        output
    }
}

pub fn compile<'s>(source: &'s str, ir: &ir::Program) -> Result<Program, CompileError<'s>> {
    let mut this = Compiler {
        program: Program {
            code: Vec::new(),
            globals: IndexMap::new(),
            functions: IndexMap::new(),
            debug_info: DebugInfo {
                source: source.to_string(),
                strings: ir.strings.clone(),
                instruction_locations: Vec::new(),
                function_names: ir.functions.iter().map(|(name, _)| *name).collect(),
            },
        },
        strings: ir.strings.clone(),
        block_addresses: HashMap::new(),
        stack_allocator: StackAllocator::new(),
    };

    for (name, global) in ir.globals.iter() {
        match &global.initial_value {
            ir::GlobalInitializer::Value(value) => {
                this.program.globals.insert(
                    *name,
                    GlobalVariable {
                        initial_value: Value::from(value),
                    },
                );
            }
            ir::GlobalInitializer::Expression(_) => {
                todo!("Initializer expressions are not supported yet")
            }
        }
    }

    for (name, func) in ir.functions.iter() {
        this.program.functions.insert(
            *name,
            Function {
                name: *name,
                entry_point: CodeAddress(0),
                stack_size: 0,
                return_type: Type::from(func.return_type),
            },
        );
    }

    for (name, func) in ir.functions.iter() {
        this.block_addresses.clear();
        this.stack_allocator = StackAllocator::new();

        let entry_point = CodeAddress(this.program.code.len());

        this.compile_function(source, func)?;

        let stack_size = this.stack_allocator.depth;

        this.program.functions.insert(
            *name,
            Function {
                name: *name,
                entry_point,
                stack_size,
                return_type: Type::from(func.return_type),
            },
        );
    }

    Ok(this.program)
}

pub struct Compiler {
    program: Program,
    strings: Strings,
    block_addresses: HashMap<usize, CodeAddress>,
    stack_allocator: StackAllocator,
}

impl Compiler {
    fn compile_function<'s>(
        &mut self,
        source: &'s str,
        func: &ir::Function,
    ) -> Result<(), CompileError<'s>> {
        let mut blocks: Vec<(usize, usize, Option<StackAllocator>)> = vec![(0, 0, None)];

        while let Some((current_block_idx, additional, variables)) = blocks.pop() {
            // We may encounter the same block multiple times.
            let address = match self.block_addresses.entry(current_block_idx) {
                Entry::Occupied(occupied_entry) => Some(*occupied_entry.get()),
                Entry::Vacant(vacant_entry) => {
                    vacant_entry.insert(CodeAddress(self.program.code.len()));
                    None
                }
            };

            if let Some(variables) = variables {
                self.stack_allocator.variables = variables.variables;
                self.stack_allocator.depth = variables.depth.max(self.stack_allocator.depth);
            }

            let block = &func.blocks[current_block_idx];
            if address.is_none() {
                self.codegen_block(source, block)?;
            }

            match block.terminator {
                ir::Termination::Return(source_location) => {
                    self.push_instruction(source_location, Instruction::Return);
                }
                ir::Termination::Jump {
                    to,
                    source_location,
                } => {
                    if let Some(&address) = self.block_addresses.get(&to.0) {
                        self.push_instruction(source_location, Instruction::Jump(address));
                    } else {
                        blocks.push((to.0, 0, None));
                    }
                }
                ir::Termination::If {
                    condition,
                    then_block,
                    else_block,
                    source_location,
                } => {
                    // Normally, we'd want to minimize the number of jumps.
                    // For now, we'll lay out one of the branches, and put this block back on the stack
                    // to handle the other branch later.
                    if let Some(address) = address {
                        let len = CodeAddress(self.program.code.len());
                        // Patch the original jump instruction.
                        let Instruction::JumpIfFalse(_, jump_address) =
                            &mut self.program.code[additional]
                        else {
                            unreachable!(
                                "Expected a JumpIfFalse instruction at the start of an if block @ {}",
                                address.0
                            );
                        };

                        // Just start laying out the else block.
                        // TODO: is there some case where we need to add another jump here? The other
                        // branch should have either walked though everything or jumped back.
                        if let Some(addr) = self.block_addresses.get(&else_block.0) {
                            *jump_address = *addr;
                        } else {
                            blocks.push((else_block.0, 0, None));
                            *jump_address = len;
                        }
                    } else {
                        let address = self
                            .address_of_variable(condition)
                            .expect("Variable not found in stack allocator");

                        // We'll have to patch the jump later.
                        blocks.push((
                            current_block_idx,
                            self.program.code.len(),
                            Some(self.stack_allocator.clone()),
                        ));
                        self.push_instruction(
                            source_location,
                            Instruction::JumpIfFalse(address, CodeAddress(0)),
                        );

                        if !self.block_addresses.contains_key(&then_block.0) {
                            blocks.push((then_block.0, 0, None));
                        }
                    }
                }
            }
        }

        Ok(())
    }

    fn codegen_block<'s>(
        &mut self,
        source: &'s str,
        block: &ir::Block,
    ) -> Result<(), CompileError<'s>> {
        for instruction in &block.instructions {
            self.codegen_instruction(source, instruction)?;
        }

        Ok(())
    }

    fn codegen_instruction<'s>(
        &mut self,
        source: &'s str,
        instruction: &ir::IrWithLocation,
    ) -> Result<(), CompileError<'s>> {
        let location = instruction.source_location;
        match &instruction.instruction {
            ir::Ir::Declare(var, value) => self.declare_variable(location, *var, *value),
            ir::Ir::Assign(dst, src) => {
                let dst = self
                    .address_of_variable(*dst)
                    .expect("Variable not found in stack allocator");
                let src = self
                    .address_of_variable(*src)
                    .expect("Variable not found in stack allocator");

                self.push_instruction(location, Instruction::Copy(dst, src));
            }
            ir::Ir::DerefAssign(dst, src) => {
                let dst = self
                    .address_of_variable(*dst)
                    .expect("Variable not found in stack allocator");
                let src = self
                    .address_of_variable(*src)
                    .expect("Variable not found in stack allocator");

                self.push_instruction(location, Instruction::DerefCopy(dst, src));
            }
            ir::Ir::FreeVariable(var) => self.free_variable(*var),
            ir::Ir::Call(fn_name, return_var, args) => {
                if let Some(fn_index) = self.program.functions.get_index_of(fn_name) {
                    self.push_instruction(
                        location,
                        Instruction::Call(
                            fn_index,
                            self.address_of_variable(*return_var)
                                .expect("Variable not found in stack allocator"),
                        ),
                    );
                } else {
                    self.push_instruction(
                        location,
                        Instruction::CallNamed(
                            *fn_name,
                            self.address_of_variable(*return_var)
                                .expect("Variable not found in stack allocator"),
                            args.len(),
                        ),
                    );
                }
            }
            ir::Ir::BinaryOperator(op, dst, lhs, rhs) => {
                let dst = self
                    .address_of_variable(*dst)
                    .expect("Variable not found in stack allocator");
                let mut lhs = self
                    .address_of_variable(*lhs)
                    .expect("Variable not found in stack allocator");
                let mut rhs = self
                    .address_of_variable(*rhs)
                    .expect("Variable not found in stack allocator");

                let (normalized, swap) = match self.strings.lookup(*op) {
                    ">" => ("<", true),
                    ">=" => ("<=", true),
                    other => (other, false),
                };

                if swap {
                    std::mem::swap(&mut rhs, &mut lhs);
                }

                let instruction = match normalized {
                    "<" => Instruction::TestLessThan(dst, lhs, rhs),
                    "<=" => Instruction::TestLessThanOrEqual(dst, lhs, rhs),
                    "==" => Instruction::TestEquals(dst, lhs, rhs),
                    "!=" => Instruction::TestNotEquals(dst, lhs, rhs),
                    "|" => Instruction::BitwiseOr(dst, lhs, rhs),
                    "^" => Instruction::BitwiseXor(dst, lhs, rhs),
                    "&" => Instruction::BitwiseAnd(dst, lhs, rhs),
                    "<<" => Instruction::ShiftLeft(dst, lhs, rhs),
                    ">>" => Instruction::ShiftRight(dst, lhs, rhs),
                    "+" => Instruction::Add(dst, lhs, rhs),
                    "-" => Instruction::Subtract(dst, lhs, rhs),
                    "*" => Instruction::Multiply(dst, lhs, rhs),
                    "/" => Instruction::Divide(dst, lhs, rhs),
                    other => {
                        return Err(CompileError {
                            source,
                            location,
                            error: format!("Unsupported binary operator: {other}"),
                        });
                    }
                };

                self.push_instruction(location, instruction);
            }
            ir::Ir::UnaryOperator(op, dst, operand) => {
                let dst = self
                    .address_of_variable(*dst)
                    .expect("Variable not found in stack allocator");
                let operand = self
                    .address_of_variable(*operand)
                    .expect("Variable not found in stack allocator");

                let instruction = match self.strings.lookup(*op) {
                    "-" => Instruction::Negate(dst, operand),
                    "!" => Instruction::Not(dst, operand),
                    "&" => Instruction::Address(dst, operand),
                    "*" => Instruction::Dereference(dst, operand),
                    other => {
                        return Err(CompileError {
                            source,
                            location,
                            error: format!("Unsupported unary operator: {other}"),
                        });
                    }
                };

                self.push_instruction(location, instruction);
            }
        }
        Ok(())
    }

    fn address_of_variable(&self, variable: VariableIndex) -> Option<MemoryAddress> {
        match variable {
            VariableIndex::Local(index) => self.stack_allocator.find(index),
            VariableIndex::Temporary(index) => self.stack_allocator.find(index),
            VariableIndex::Global(index) => Some(MemoryAddress::Global(index.0)),
        }
    }

    fn declare_variable(
        &mut self,
        location: Location,
        var: ir::VariableDeclaration,
        value: Option<ir::Value>,
    ) {
        let address = self.stack_allocator.allocate_variable(var);

        if let Some(value) = value {
            let value = Value::from(&value);
            self.push_instruction(location, Instruction::LoadValue(address, value));
        }
    }

    fn free_variable(&mut self, var: variable_tracker::LocalVariableIndex) {
        self.stack_allocator.free_variable(var);
    }

    fn push_instruction(&mut self, location: Location, instruction: Instruction) {
        self.program.code.push(instruction);
        self.program.debug_info.instruction_locations.push(location);
    }
}

#[derive(Clone)]
struct StackAllocator {
    pub variables: Vec<Option<LocalVariableIndex>>,
    pub depth: usize,
}

impl StackAllocator {
    pub fn new() -> Self {
        StackAllocator {
            variables: Vec::new(),
            depth: 0,
        }
    }

    pub fn find(&self, index: LocalVariableIndex) -> Option<MemoryAddress> {
        self.variables
            .iter()
            .position(|&v| v == Some(index))
            .map(MemoryAddress::Local)
    }

    fn allocate_variable(&mut self, var: ir::VariableDeclaration) -> MemoryAddress {
        if var.allocation_method == ir::AllocationMethod::FirstFit {
            if let Some(reused) = self.variables.iter().position(|&v| v.is_none()) {
                self.variables[reused] = Some(var.index);
                return MemoryAddress::Local(reused);
            }
        }

        let index = self.variables.len();
        self.variables.push(Some(var.index));
        self.depth = self.depth.max(self.variables.len());
        MemoryAddress::Local(index)
    }

    fn free_variable(&mut self, var: LocalVariableIndex) {
        if let Some(pos) = self.variables.iter().position(|&v| v == Some(var)) {
            self.variables[pos] = None;

            // Trim freed variables from the end of the vector
            while self.variables.pop_if(|v| v.is_none()).is_some() {}
        } else {
            panic!("Attempted to free a variable that is not allocated: {var:?}");
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_compile() {
        crate::test::run_compile_tests("tests/compiler/");
    }
}
