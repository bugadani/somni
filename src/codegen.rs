use std::{
    collections::{HashMap, hash_map::Entry},
    fmt::Display,
    ops::{Add, AddAssign},
};

use indexmap::IndexMap;

use crate::{
    error::CompileError,
    eval::{ExprContext, ExpressionVisitor, OperatorError, TypedValue},
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
pub enum BinaryOperator {
    TestLessThan,
    TestLessThanOrEqual,
    TestEquals,
    TestNotEquals,
    BitwiseOr,
    BitwiseXor,
    BitwiseAnd,
    ShiftLeft,
    ShiftRight,
    Add,
    Subtract,
    Multiply,
    Divide,
}

impl std::fmt::Display for BinaryOperator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let operator = match self {
            BinaryOperator::TestLessThan => "<",
            BinaryOperator::TestLessThanOrEqual => "<=",
            BinaryOperator::TestEquals => "==",
            BinaryOperator::TestNotEquals => "!=",
            BinaryOperator::BitwiseOr => "|",
            BinaryOperator::BitwiseXor => "^",
            BinaryOperator::BitwiseAnd => "&",
            BinaryOperator::ShiftLeft => "<<",
            BinaryOperator::ShiftRight => ">>",
            BinaryOperator::Add => "+",
            BinaryOperator::Subtract => "-",
            BinaryOperator::Multiply => "*",
            BinaryOperator::Divide => "/",
        };
        write!(f, "{}", operator)
    }
}

#[derive(Clone, Copy, Debug)]
pub enum UnaryOperator {
    Negate,
    Not,
}

impl std::fmt::Display for UnaryOperator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let operator = match self {
            UnaryOperator::Negate => "-",
            UnaryOperator::Not => "!",
        };
        write!(f, "{}", operator)
    }
}

#[derive(Clone, Copy, Debug)]
pub enum Instruction {
    /// Return with the value from the eval stack
    Return,
    Jump(CodeAddress),
    JumpIfFalse(MemoryAddress, CodeAddress),
    Copy(Type, MemoryAddress, MemoryAddress),
    DerefCopy(Type, MemoryAddress, MemoryAddress),
    LoadValue(MemoryAddress, TypedValue),
    Call(usize, MemoryAddress),
    CallNamed(StringIndex, Type, MemoryAddress),
    /// Unary operations
    Dereference(Type, MemoryAddress, MemoryAddress),
    AddressOf(MemoryAddress, MemoryAddress),
    UnaryOperator {
        ty: Type,
        operator: UnaryOperator,
        dst: MemoryAddress,
        op: MemoryAddress,
    },

    // Binary operations
    BinaryOperator {
        ty: Type,
        operator: BinaryOperator,
        dst: MemoryAddress,
        lhs: MemoryAddress,
        rhs: MemoryAddress,
    },
}

impl Instruction {
    fn disasm(&self, debug_info: &DebugInfo) -> String {
        match self {
            Instruction::Return => "return".to_string(),
            Instruction::Jump(address) => format!("jump to {}", address.0),
            Instruction::JumpIfFalse(addr, address) => {
                format!("if {:?} == false jump to {}", addr, address.0)
            }
            Instruction::Copy(_, dst, src) => format!("copy {dst:?} = {src:?}"),
            Instruction::DerefCopy(_, dst, src) => format!("copy *{dst:?} = {src:?}"),
            Instruction::LoadValue(dst, value) => format!("load {dst:?} = {value:?}"),
            Instruction::Call(fn_index, stack_frame) => {
                let fn_name = debug_info
                    .function_names
                    .get(*fn_index)
                    .map(|&name| debug_info.strings.lookup(name))
                    .unwrap_or("unknown function");
                format!("call {fn_name}(..) with sp={stack_frame:?}")
            }
            Instruction::CallNamed(name, ret_ty, stack_frame) => {
                let fn_name = debug_info.strings.lookup(*name);
                format!("call {fn_name}(?) -> {ret_ty:?} with sp={stack_frame:?}")
            }
            Instruction::BinaryOperator {
                ty,
                operator,
                dst,
                lhs,
                rhs,
            } => {
                format!("{dst:?} = {lhs:?} {operator} {rhs:?} ({ty})")
            }
            Instruction::UnaryOperator {
                ty,
                operator,
                dst,
                op,
            } => {
                format!("{dst:?} = {operator}{op:?} ({ty})")
            }
            Instruction::Dereference(_, dst, op) => {
                format!("{dst:?} = *{op:?}")
            }
            Instruction::AddressOf(dst, op) => {
                format!("{dst:?} = &{op:?}")
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

pub trait ValueType: Sized + Copy + Into<TypedValue> {
    const BYTES: usize;

    fn write(&self, to: &mut [u8]);
    fn from_bytes(bytes: &[u8]) -> Self;

    fn equals(a: Self, b: Self) -> Result<bool, OperatorError>;
    fn less_than(a: Self, b: Self) -> Result<bool, OperatorError>;

    fn less_than_or_equal(a: Self, b: Self) -> Result<bool, OperatorError> {
        let less = Self::less_than(a, b)?;
        Ok(less || Self::equals(a, b)?)
    }
    fn not_equals(a: Self, b: Self) -> Result<bool, OperatorError> {
        let equals = Self::equals(a, b)?;
        Ok(!equals)
    }
    fn bitwise_or(_a: Self, _b: Self) -> Result<Self, OperatorError> {
        Err(OperatorError::NotSupported)
    }
    fn bitwise_xor(_a: Self, _b: Self) -> Result<Self, OperatorError> {
        Err(OperatorError::NotSupported)
    }
    fn bitwise_and(_a: Self, _b: Self) -> Result<Self, OperatorError> {
        Err(OperatorError::NotSupported)
    }
    fn shift_left(_a: Self, _b: Self) -> Result<Self, OperatorError> {
        Err(OperatorError::NotSupported)
    }
    fn shift_right(_a: Self, _b: Self) -> Result<Self, OperatorError> {
        Err(OperatorError::NotSupported)
    }
    fn add(_a: Self, _b: Self) -> Result<Self, OperatorError> {
        Err(OperatorError::NotSupported)
    }
    fn subtract(_a: Self, _b: Self) -> Result<Self, OperatorError> {
        Err(OperatorError::NotSupported)
    }
    fn multiply(_a: Self, _b: Self) -> Result<Self, OperatorError> {
        Err(OperatorError::NotSupported)
    }
    fn divide(_a: Self, _b: Self) -> Result<Self, OperatorError> {
        Err(OperatorError::NotSupported)
    }
    fn not(_a: Self) -> Result<Self, OperatorError> {
        Err(OperatorError::NotSupported)
    }
    fn negate(_a: Self) -> Result<Self, OperatorError> {
        Err(OperatorError::NotSupported)
    }
}

impl ValueType for () {
    const BYTES: usize = 0;

    fn write(&self, _to: &mut [u8]) {}

    fn from_bytes(_: &[u8]) -> Self {
        ()
    }

    fn less_than(_: Self, _: Self) -> Result<bool, OperatorError> {
        Err(OperatorError::NotSupported)
    }

    fn equals(_: Self, _: Self) -> Result<bool, OperatorError> {
        Err(OperatorError::NotSupported)
    }
}

impl ValueType for u64 {
    const BYTES: usize = 8;

    fn write(&self, to: &mut [u8]) {
        to.copy_from_slice(&self.to_le_bytes());
    }

    fn from_bytes(bytes: &[u8]) -> Self {
        u64::from_le_bytes(bytes.try_into().unwrap())
    }

    fn less_than(a: Self, b: Self) -> Result<bool, OperatorError> {
        Ok(a < b)
    }
    fn equals(a: Self, b: Self) -> Result<bool, OperatorError> {
        Ok(a == b)
    }
    fn add(a: Self, b: Self) -> Result<Self, OperatorError> {
        a.checked_add(b).ok_or(OperatorError::RuntimeError)
    }
    fn subtract(a: Self, b: Self) -> Result<Self, OperatorError> {
        a.checked_sub(b).ok_or(OperatorError::RuntimeError)
    }
    fn multiply(a: Self, b: Self) -> Result<Self, OperatorError> {
        a.checked_mul(b).ok_or(OperatorError::RuntimeError)
    }
    fn divide(a: Self, b: Self) -> Result<Self, OperatorError> {
        if b == 0 {
            Err(OperatorError::RuntimeError)
        } else {
            Ok(a / b)
        }
    }
    fn bitwise_or(a: Self, b: Self) -> Result<Self, OperatorError> {
        Ok(a | b)
    }
    fn bitwise_xor(a: Self, b: Self) -> Result<Self, OperatorError> {
        Ok(a ^ b)
    }
    fn bitwise_and(a: Self, b: Self) -> Result<Self, OperatorError> {
        Ok(a & b)
    }
    fn shift_left(a: Self, b: Self) -> Result<Self, OperatorError> {
        if b < 64 {
            Ok(a << b)
        } else {
            Err(OperatorError::RuntimeError)
        }
    }
    fn shift_right(a: Self, b: Self) -> Result<Self, OperatorError> {
        if b < 64 {
            Ok(a >> b)
        } else {
            Err(OperatorError::RuntimeError)
        }
    }
}

impl ValueType for i64 {
    const BYTES: usize = 8;

    fn write(&self, to: &mut [u8]) {
        to.copy_from_slice(&self.to_le_bytes());
    }

    fn from_bytes(bytes: &[u8]) -> Self {
        i64::from_le_bytes(bytes.try_into().unwrap())
    }

    fn less_than(a: Self, b: Self) -> Result<bool, OperatorError> {
        Ok(a < b)
    }
    fn equals(a: Self, b: Self) -> Result<bool, OperatorError> {
        Ok(a == b)
    }
    fn add(a: Self, b: Self) -> Result<Self, OperatorError> {
        a.checked_add(b).ok_or(OperatorError::RuntimeError)
    }
    fn subtract(a: Self, b: Self) -> Result<Self, OperatorError> {
        a.checked_sub(b).ok_or(OperatorError::RuntimeError)
    }
    fn multiply(a: Self, b: Self) -> Result<Self, OperatorError> {
        a.checked_mul(b).ok_or(OperatorError::RuntimeError)
    }
    fn divide(a: Self, b: Self) -> Result<Self, OperatorError> {
        if b == 0 {
            Err(OperatorError::RuntimeError)
        } else {
            Ok(a / b)
        }
    }
    fn bitwise_or(a: Self, b: Self) -> Result<Self, OperatorError> {
        Ok(a | b)
    }
    fn bitwise_xor(a: Self, b: Self) -> Result<Self, OperatorError> {
        Ok(a ^ b)
    }
    fn bitwise_and(a: Self, b: Self) -> Result<Self, OperatorError> {
        Ok(a & b)
    }
    fn shift_left(a: Self, b: Self) -> Result<Self, OperatorError> {
        if b < 64 {
            Ok(a << b)
        } else {
            Err(OperatorError::RuntimeError)
        }
    }
    fn shift_right(a: Self, b: Self) -> Result<Self, OperatorError> {
        if b < 64 {
            Ok(a >> b)
        } else {
            Err(OperatorError::RuntimeError)
        }
    }
}

impl ValueType for f64 {
    const BYTES: usize = 8;

    fn write(&self, to: &mut [u8]) {
        to.copy_from_slice(&self.to_le_bytes());
    }

    fn from_bytes(bytes: &[u8]) -> Self {
        f64::from_le_bytes(bytes.try_into().unwrap())
    }

    fn less_than(a: Self, b: Self) -> Result<bool, OperatorError> {
        Ok(a < b)
    }
    fn equals(a: Self, b: Self) -> Result<bool, OperatorError> {
        Ok(a == b)
    }
    fn add(a: Self, b: Self) -> Result<Self, OperatorError> {
        Ok(a + b)
    }
    fn subtract(a: Self, b: Self) -> Result<Self, OperatorError> {
        Ok(a - b)
    }
    fn multiply(a: Self, b: Self) -> Result<Self, OperatorError> {
        Ok(a * b)
    }
    fn divide(a: Self, b: Self) -> Result<Self, OperatorError> {
        Ok(a / b)
    }
}

impl ValueType for bool {
    const BYTES: usize = 1;

    fn write(&self, to: &mut [u8]) {
        to.copy_from_slice(&[*self as u8]);
    }

    fn from_bytes(bytes: &[u8]) -> Self {
        bytes[0] != 0
    }

    fn less_than(_: Self, _: Self) -> Result<bool, OperatorError> {
        Err(OperatorError::NotSupported)
    }
    fn equals(a: Self, b: Self) -> Result<bool, OperatorError> {
        Ok(a == b)
    }
}

impl ValueType for StringIndex {
    const BYTES: usize = 8;

    fn write(&self, to: &mut [u8]) {
        to.copy_from_slice(&self.0.to_le_bytes());
    }
    fn from_bytes(bytes: &[u8]) -> Self {
        StringIndex(u64::from_le_bytes(bytes.try_into().unwrap()) as usize)
    }
    fn less_than(_: Self, _: Self) -> Result<bool, OperatorError> {
        Err(OperatorError::NotSupported)
    }
    fn equals(a: Self, b: Self) -> Result<bool, OperatorError> {
        Ok(a == b)
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
impl Type {
    pub fn size_of(&self) -> usize {
        match self {
            Type::Void => <() as ValueType>::BYTES,
            Type::Int => <u64 as ValueType>::BYTES,
            Type::SignedInt => <i64 as ValueType>::BYTES,
            Type::Float => <f64 as ValueType>::BYTES,
            Type::Bool => <bool as ValueType>::BYTES,
            Type::String => <StringIndex as ValueType>::BYTES,
            Type::Address => <u64 as ValueType>::BYTES, // Assuming address is represented as u64
        }
    }
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
        }
    }
}

impl From<ir::Variable> for Type {
    fn from(ty: ir::Variable) -> Self {
        match ty {
            ir::Variable::Value(t) => Self::from(t),
            ir::Variable::Reference(_, _) => Type::Address,
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
    pub address: MemoryAddress,
    pub initial_value: Option<TypedValue>,
}

impl GlobalVariable {
    pub fn value(&self) -> TypedValue {
        self.initial_value.expect("variable is unevaluated")
    }

    pub fn ty(&self) -> Type {
        self.value().type_of()
    }
}

#[derive(Clone, Debug)]
pub struct Function {
    pub name: StringIndex, // Name of the function
    pub entry_point: CodeAddress,
    pub stack_size: usize,
    pub return_type: Type,
    pub arguments: Vec<(MemoryAddress, Type)>,
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

impl ExprContext for Program {
    fn intern_string(&mut self, s: &str) -> StringIndex {
        self.debug_info.strings.find(s).unwrap()
    }

    fn try_load_variable(&self, variable: &str) -> Option<TypedValue> {
        let idx = self.debug_info.strings.find(variable)?;
        self.globals[&idx].initial_value
    }

    fn address_of(&self, variable: &str) -> TypedValue {
        let idx = self.debug_info.strings.find(variable).unwrap();
        let MemoryAddress::Global(addr) = self.globals[&idx].address else {
            unreachable!();
        };

        TypedValue::Int(addr as u64)
    }

    fn call_function(&mut self, _function_name: &str, _args: &[TypedValue]) -> TypedValue {
        todo!("maybe")
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
    };

    let mut global_addr = 0;
    for (name, global) in ir.globals.iter() {
        let ty = Type::from(global.ty);
        this.program.globals.insert(
            *name,
            GlobalVariable {
                address: MemoryAddress::Global(global_addr),
                initial_value: None,
            },
        );
        global_addr += ty.size_of();
    }

    loop {
        let mut made_progress = false;

        for (name, global) in ir.globals.iter() {
            if this.program.globals[name].initial_value.is_some() {
                continue;
            }

            let mut visitor = ExpressionVisitor {
                context: &mut this.program,
                source: source,
            };
            if let Ok(value) = visitor.visit_expression(&global.initializer) {
                this.program.globals[name].initial_value = Some(value);
                made_progress = true;
            }
        }

        if !made_progress {
            break;
        }
    }

    // Check whether every global was evaluated
    for (name, global) in ir.globals.iter() {
        if this.program.globals[name].initial_value.is_none() {
            return Err(CompileError {
                source,
                location: global.initializer.location(),
                error: format!("Failed to evaluate initializer"),
            });
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
                arguments: vec![],
            },
        );
    }

    for (name, func) in ir.functions.iter() {
        let entry_point = CodeAddress(this.program.code.len());

        let mut function_compiler = FunctionCompiler {
            compiler: &mut this,
            source,
            func,
            block_addresses: HashMap::new(),
            stack_allocator: StackAllocator::new(),
        };
        function_compiler.compile()?;

        let stack_size = function_compiler.stack_allocator.depth();
        let arguments = func
            .arguments
            .iter()
            .enumerate()
            .map(|(i, arg)| {
                (
                    function_compiler
                        .address_of_variable(VariableIndex::Local(LocalVariableIndex(i + 1)))
                        .unwrap(),
                    Type::from(*arg),
                )
            })
            .collect();

        this.program.functions.insert(
            *name,
            Function {
                name: *name,
                entry_point,
                stack_size,
                return_type: Type::from(func.return_type),
                arguments,
            },
        );
    }

    Ok(this.program)
}

pub struct Compiler {
    program: Program,
    strings: Strings,
}

struct FunctionCompiler<'s, 'p> {
    compiler: &'p mut Compiler,
    source: &'s str,
    func: &'p ir::Function,
    block_addresses: HashMap<usize, CodeAddress>,
    stack_allocator: StackAllocator,
}

impl<'s> FunctionCompiler<'s, '_> {
    fn compile(&mut self) -> Result<(), CompileError<'s>> {
        let mut blocks: Vec<(usize, usize, Option<StackAllocator>)> = vec![(0, 0, None)];

        while let Some((current_block_idx, additional, variables)) = blocks.pop() {
            // We may encounter the same block multiple times.
            let address = match self.block_addresses.entry(current_block_idx) {
                Entry::Occupied(occupied_entry) => Some(*occupied_entry.get()),
                Entry::Vacant(vacant_entry) => {
                    vacant_entry.insert(CodeAddress(self.compiler.program.code.len()));
                    None
                }
            };

            if let Some(variables) = variables {
                let saved_depth = variables.depth();
                let current_depth = self.stack_allocator.depth();

                self.stack_allocator.allocations = variables.allocations;

                if current_depth > saved_depth {
                    // If the stack allocator has grown, we need to update the current stack depth.
                    let extra_depth = current_depth - saved_depth;
                    if let Some(Allocation::Hole { size }) =
                        self.stack_allocator.allocations.last_mut()
                    {
                        *size += extra_depth;
                    } else {
                        self.stack_allocator
                            .allocations
                            .push(Allocation::Hole { size: extra_depth });
                    }
                }
            }

            let block = &self.func.blocks[current_block_idx];
            if address.is_none() {
                self.codegen_block(block)?;
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
                        let len = CodeAddress(self.compiler.program.code.len());
                        // Patch the original jump instruction.
                        let Instruction::JumpIfFalse(_, jump_address) =
                            &mut self.compiler.program.code[additional]
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
                            self.compiler.program.code.len(),
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

    fn codegen_block(&mut self, block: &ir::Block) -> Result<(), CompileError<'s>> {
        for instruction in &block.instructions {
            self.codegen_instruction(instruction)?;
        }

        Ok(())
    }

    fn codegen_instruction(
        &mut self,
        instruction: &ir::IrWithLocation,
    ) -> Result<(), CompileError<'s>> {
        let location = instruction.source_location;
        match &instruction.instruction {
            ir::Ir::Declare(var, value) => self.declare_variable(location, *var, *value),
            ir::Ir::Assign(dst, src) => {
                let ty = self
                    .type_of_variable(*dst)
                    .expect("Type of destination variable should be known");
                let dst = self
                    .address_of_variable(*dst)
                    .expect("Variable not found in stack allocator");
                let src = self
                    .address_of_variable(*src)
                    .expect("Variable not found in stack allocator");

                self.push_instruction(location, Instruction::Copy(ty, dst, src));
            }
            ir::Ir::DerefAssign(dst, src) => {
                let ty = self
                    .type_of_variable(*dst)
                    .expect("Type of destination variable should be known");
                let dst = self
                    .address_of_variable(*dst)
                    .expect("Variable not found in stack allocator");
                let src = self
                    .address_of_variable(*src)
                    .expect("Variable not found in stack allocator");

                self.push_instruction(location, Instruction::DerefCopy(ty, dst, src));
            }
            ir::Ir::FreeVariable(var) => self.free_variable(*var),
            ir::Ir::Call(fn_name, return_var, _args) => {
                if let Some(fn_index) = self.compiler.program.functions.get_index_of(fn_name) {
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
                            self.type_of_variable(*return_var).unwrap(),
                            self.address_of_variable(*return_var)
                                .expect("Variable not found in stack allocator"),
                        ),
                    );
                }
            }
            ir::Ir::BinaryOperator(op, dst, lhs, rhs) => {
                let dst_addr = self
                    .address_of_variable(*dst)
                    .expect("Variable not found in stack allocator");
                let mut lhs_addr = self
                    .address_of_variable(*lhs)
                    .expect("Variable not found in stack allocator");
                let mut rhs_addr = self
                    .address_of_variable(*rhs)
                    .expect("Variable not found in stack allocator");

                let (normalized, swap) = match self.compiler.strings.lookup(*op) {
                    ">" => ("<", true),
                    ">=" => ("<=", true),
                    other => (other, false),
                };

                if swap {
                    std::mem::swap(&mut rhs_addr, &mut lhs_addr);
                }

                let operator = match normalized {
                    "<" => BinaryOperator::TestLessThan,
                    "<=" => BinaryOperator::TestLessThanOrEqual,
                    "==" => BinaryOperator::TestEquals,
                    "!=" => BinaryOperator::TestNotEquals,
                    "|" => BinaryOperator::BitwiseOr,
                    "^" => BinaryOperator::BitwiseXor,
                    "&" => BinaryOperator::BitwiseAnd,
                    "<<" => BinaryOperator::ShiftLeft,
                    ">>" => BinaryOperator::ShiftRight,
                    "+" => BinaryOperator::Add,
                    "-" => BinaryOperator::Subtract,
                    "*" => BinaryOperator::Multiply,
                    "/" => BinaryOperator::Divide,
                    other => {
                        return Err(CompileError {
                            source: self.source,
                            location,
                            error: format!("Unsupported binary operator: {other}"),
                        });
                    }
                };

                self.push_instruction(
                    location,
                    Instruction::BinaryOperator {
                        ty: self.type_of_variable(*lhs).unwrap(),
                        operator,
                        dst: dst_addr,
                        lhs: lhs_addr,
                        rhs: rhs_addr,
                    },
                );
            }
            ir::Ir::UnaryOperator(op, dst, operand) => {
                let ty = self
                    .type_of_variable(*dst)
                    .expect("Type of destination variable should be known");
                let dst_addr = self
                    .address_of_variable(*dst)
                    .expect("Variable not found in stack allocator");
                let operand_addr = self
                    .address_of_variable(*operand)
                    .expect("Variable not found in stack allocator");

                let instruction = match self.compiler.strings.lookup(*op) {
                    "-" => Instruction::UnaryOperator {
                        ty: self.type_of_variable(*dst).unwrap(),
                        operator: UnaryOperator::Negate,
                        dst: dst_addr,
                        op: operand_addr,
                    },
                    "!" => Instruction::UnaryOperator {
                        ty: self.type_of_variable(*dst).unwrap(),
                        operator: UnaryOperator::Not,
                        dst: dst_addr,
                        op: operand_addr,
                    },
                    "&" => Instruction::AddressOf(dst_addr, operand_addr),
                    "*" => Instruction::Dereference(ty, dst_addr, operand_addr),
                    other => {
                        return Err(CompileError {
                            source: self.source,
                            location,
                            error: format!("Unsupported unary operator: {other}"),
                        });
                    }
                };

                self.push_instruction(location, instruction)
            }
        }
        Ok(())
    }

    fn address_of_variable(&self, variable: VariableIndex) -> Option<MemoryAddress> {
        match variable {
            VariableIndex::Local(index) => self.stack_allocator.find(index),
            VariableIndex::Temporary(index) => self.stack_allocator.find(index),
            VariableIndex::Global(index) => Some(self.compiler.program.globals[index.0].address),
        }
    }

    fn type_of_variable(&self, variable: VariableIndex) -> Option<Type> {
        match variable {
            VariableIndex::Local(index) | VariableIndex::Temporary(index) => self
                .func
                .variables
                .variable(index)
                .map(|v| Type::from(v.ty.unwrap_or(ir::Variable::Value(ir::Type::Void)))), // TODO: revisit after removing unused variables, there should be no unresolved types
            VariableIndex::Global(index) => self
                .compiler
                .program
                .globals
                .get_index(index.0)
                .map(|(_, g)| g.ty()),
        }
    }

    fn declare_variable(
        &mut self,
        location: Location,
        var: ir::VariableDeclaration,
        value: Option<ir::Value>,
    ) {
        let ty = self
            .type_of_variable(VariableIndex::Local(var.index))
            .expect("At this point, the variable type should be known");
        let address = self.stack_allocator.allocate_variable(var, ty.size_of());

        if let Some(value) = value {
            let value = TypedValue::from_value(&value);
            self.push_instruction(location, Instruction::LoadValue(address, value));
        }
    }

    fn free_variable(&mut self, var: variable_tracker::LocalVariableIndex) {
        self.stack_allocator.free_variable(var);
    }

    fn push_instruction(&mut self, location: Location, instruction: Instruction) {
        self.compiler.program.code.push(instruction);
        self.compiler
            .program
            .debug_info
            .instruction_locations
            .push(location);
    }
}

#[derive(Clone, Copy, Debug)]
enum Allocation {
    Hole {
        size: usize,
    },
    Used {
        variable: LocalVariableIndex,
        size: usize,
    },
}

impl Allocation {
    fn size(&self) -> usize {
        match *self {
            Allocation::Hole { size } => size,
            Allocation::Used { size, .. } => size,
        }
    }

    fn free(&self) -> bool {
        match *self {
            Allocation::Hole { .. } => true,
            Allocation::Used { .. } => false,
        }
    }
}

#[derive(Clone)]
struct StackAllocator {
    pub allocations: Vec<Allocation>,
    pub var_addresses: IndexMap<LocalVariableIndex, MemoryAddress>,
}

impl StackAllocator {
    pub fn new() -> Self {
        StackAllocator {
            allocations: Vec::new(),
            var_addresses: IndexMap::new(),
        }
    }

    pub fn find(&self, var: LocalVariableIndex) -> Option<MemoryAddress> {
        self.address_of(var)
    }

    pub fn address_of(&self, var: LocalVariableIndex) -> Option<MemoryAddress> {
        self.var_addresses.get(&var).copied()
    }

    fn allocate_variable(&mut self, var: ir::VariableDeclaration, size: usize) -> MemoryAddress {
        if var.allocation_method == ir::AllocationMethod::FirstFit {
            if let Some(allocation) = self.reuse_hole(var.index, size) {
                let address = self.address_of_allocation(allocation);
                self.var_addresses
                    .insert(var.index, MemoryAddress::Local(address));
                return MemoryAddress::Local(address);
            }
        }

        self.push(var.index, size);
        let address = self.address_of_allocation(self.allocations.len() - 1);
        self.var_addresses
            .insert(var.index, MemoryAddress::Local(address));
        return MemoryAddress::Local(address);
    }

    fn free_variable(&mut self, var: LocalVariableIndex) {
        let Some(pos) = self
            .allocations
            .iter()
            .position(|&v| matches!(v, Allocation::Used { variable, .. } if variable == var))
        else {
            panic!("Attempted to free a variable that is not allocated: {var:?}");
        };

        let alloc = std::mem::replace(&mut self.allocations[pos], Allocation::Hole { size: 0 });
        let mut freed = alloc.size();

        // If the next allocation is a hole, also grab its size.
        if let Some(next) = self.allocations.get_mut(pos + 1) {
            if let Allocation::Hole { size } = next {
                freed += *size;
                *size = 0;
            }
        };

        let mut set_size_of = pos;
        while set_size_of > 0 {
            match &mut self.allocations[set_size_of - 1] {
                // Skip zero-sized holes.
                Allocation::Hole { size } if *size == 0 => {
                    // Check previous allocation.
                    set_size_of -= 1;
                }
                Allocation::Used { .. } | Allocation::Hole { .. } => break,
            }
        }

        // Set the size of the hole to the freed size.
        if let Some(Allocation::Hole { size }) = self.allocations.get_mut(set_size_of) {
            *size += freed;
        } else {
            panic!("Expected a hole allocation at position {set_size_of}");
        }

        // Pop 0-length holes from the end
        while let Some(Allocation::Hole { size }) = self.allocations.last() {
            if *size == 0 {
                self.allocations.pop();
            } else {
                break;
            }
        }
    }

    fn reuse_hole(&mut self, var: LocalVariableIndex, size: usize) -> Option<usize> {
        if let Some(reused) = self
            .allocations
            .iter()
            .position(|&v| v.free() && v.size() >= size)
        {
            let remaining = self.allocations[reused].size() - size;
            // Push remaining size to the right.
            if remaining > 0 {
                match self.allocations.get_mut(reused + 1) {
                    Some(Allocation::Hole { size }) => *size += remaining,
                    Some(_) => self
                        .allocations
                        .insert(reused + 1, Allocation::Hole { size: remaining }),
                    None => self.allocations.push(Allocation::Hole { size: remaining }),
                }
            }
            self.allocations[reused] = Allocation::Used {
                variable: var,
                size,
            };
            return Some(reused);
        }
        None
    }

    fn push(&mut self, var: LocalVariableIndex, var_size: usize) {
        if let Some(alloc @ Allocation::Hole { .. }) = self.allocations.last_mut() {
            let hole_size = alloc.size();

            *alloc = Allocation::Used {
                variable: var,
                size: var_size,
            };

            if hole_size >= var_size {
                let remaining = hole_size - var_size;

                // Push remaining size to the right.
                if remaining > 0 {
                    self.allocations.push(Allocation::Hole { size: remaining });
                }
            }
        } else {
            self.allocations.push(Allocation::Used {
                variable: var,
                size: var_size,
            });
        }
    }

    fn depth(&self) -> usize {
        self.allocations.iter().map(Allocation::size).sum::<usize>()
    }

    fn address_of_allocation(&self, idx: usize) -> usize {
        // TODO this is inefficient
        self.allocations[..idx]
            .iter()
            .map(Allocation::size)
            .sum::<usize>()
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_compile() {
        crate::test::run_compile_tests("tests/compiler/");
    }
}
