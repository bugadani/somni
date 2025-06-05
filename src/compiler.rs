use std::fmt::Display;

use indexmap::{IndexMap, IndexSet};

use crate::{
    error::CompileError,
    lexer::{self, Location},
    parser::{self, Expression},
};

#[derive(Clone, Copy, Debug)]
pub enum Instruction {
    /// Assign the value from the eval stack to an existing variable. The destination address
    /// is loaded from the top eval stack.
    Store,

    /// Load a constant value onto the eval stack
    Push(Value),

    /// Pop the top value from the eval stack
    Pop,

    /// Load a variable value onto the eval stack. The source address is loaded
    /// from the top eval stack.
    Load,

    /// Call a function with a number of arguments popped from the eval stack
    Call(FunctionIndex, usize),

    /// Call a function with a number of arguments popped from the eval stack
    CallByName(StringIndex, usize),

    /// Jump forward by a number of instructions
    JumpForward(usize),

    /// Jump forward by a number of instructions if the top value on the eval stack is true or false
    JumpForwardIf(usize, bool),

    /// Jump back by a number of instructions
    JumpBack(usize),

    // Loop control instructions, should be turned into jumps by an optimizer pass
    LoopStart(usize),
    LoopEnd,
    Continue,
    Break,

    // Reference and dereference instructions
    AddressOf(VarId),
    Dereference(VarId),

    // Operators
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
    Negate,
    InvertBoolean,

    /// Return with the value from the eval stack
    Return,
}
impl Instruction {
    pub fn disasm(&self, program: &Program) -> String {
        match self {
            Instruction::Store => "store".to_string(),
            Instruction::Push(value) => format!("push {:?}", value),
            Instruction::Pop => "pop".to_string(),
            Instruction::Load => "load".to_string(),
            Instruction::Call(function_index, _) => {
                let function = program.functions.get(*function_index).unwrap();
                format!("call {}", function.print_signature(program))
            }
            Instruction::CallByName(name_index, _) => {
                if let Some(function) = program.functions.get_by_name(*name_index) {
                    format!("call_named {}", function.print_signature(program))
                } else {
                    let name = program.strings.lookup_value_by_index(*name_index).unwrap();
                    format!("call_named external {}", name)
                }
            }
            Instruction::JumpForward(offset) => format!("jump_forward {}", offset),
            Instruction::JumpForwardIf(offset, b) => {
                format!("jump_forward_if {} {:?}", offset, b)
            }
            Instruction::JumpBack(offset) => format!("jump_back {}", offset),
            Instruction::LoopStart(size) => format!("loop_start {}", size),
            Instruction::LoopEnd => "loop_end".to_string(),
            Instruction::Continue => "continue".to_string(),
            Instruction::Break => "break".to_string(),
            Instruction::AddressOf(var) => format!("address_of {:?}", var),
            Instruction::Dereference(var) => format!("dereference {:?}", var),
            Instruction::TestLessThan => "less than".to_string(),
            Instruction::TestLessThanOrEqual => "less than or equal".to_string(),
            Instruction::TestEquals => "equals".to_string(),
            Instruction::TestNotEquals => "not equals".to_string(),
            Instruction::BitwiseOr => "bitwise or".to_string(),
            Instruction::BitwiseXor => "bitwise xor".to_string(),
            Instruction::BitwiseAnd => "bitwise and".to_string(),
            Instruction::ShiftLeft => "shift left".to_string(),
            Instruction::ShiftRight => "shift right".to_string(),
            Instruction::Add => "add".to_string(),
            Instruction::Subtract => "subtract".to_string(),
            Instruction::Multiply => "multiply".to_string(),
            Instruction::Divide => "divide".to_string(),
            Instruction::Negate => "negate".to_string(),
            Instruction::InvertBoolean => "not".to_string(),
            Instruction::Return => "return".to_string(),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Bytecode {
    pub instruction: Instruction,
    pub location: LocationIndex,
}

impl Bytecode {
    pub fn disasm(&self, program: &Program) -> String {
        self.instruction.disasm(program)
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
    Address(VarId),
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

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct FunctionSignature {
    pub name: StringIndex,
    pub location: Location,
    pub args: Vec<(VariableIndex, Type)>,
}

pub struct FunctionBody {
    pub code: Vec<Instruction>,
    pub locations: Vec<LocationIndex>,
    pub return_type: Type,
}

#[derive(Clone, Debug)]
pub struct Function {
    pub signature: FunctionSignature,
    pub address: usize,
    pub length: usize,
    pub scope_size: usize,
    pub return_type: Type,
}

impl Function {
    pub fn print_signature(&self, program: &Program) -> String {
        let args = self
            .signature
            .args
            .iter()
            .map(|(_, ty)| ty.to_string())
            .collect::<Vec<_>>()
            .join(", ");
        format!(
            "{}({}) -> {}",
            program
                .strings
                .lookup_value_by_index(self.signature.name)
                .unwrap_or("<unknown>"),
            args,
            self.return_type
        )
    }
}

#[derive(Clone, Debug)]
pub struct Functions {
    functions: IndexMap<StringIndex, Function>,
}
impl Functions {
    fn new() -> Self {
        Functions {
            functions: IndexMap::new(),
        }
    }

    fn declare(&mut self, name: StringIndex) -> Option<FunctionIndex> {
        let (index, old) = self.functions.insert_full(
            name,
            Function {
                signature: FunctionSignature {
                    name,
                    location: Location { start: 0, end: 0 },
                    args: vec![],
                },
                return_type: Type::Void,
                address: 0,
                length: 0,
                scope_size: 0,
            },
        );

        if old.is_some() {
            None
        } else {
            Some(FunctionIndex(index))
        }
    }

    fn push(&mut self, func: Function) -> Option<FunctionIndex> {
        let (index, old) = self
            .functions
            .insert_full(func.signature.name.clone(), func);

        if old.is_some() {
            Some(FunctionIndex(index))
        } else {
            None
        }
    }

    fn index_of(&self, name: StringIndex) -> Option<FunctionIndex> {
        self.functions.get_index_of(&name).map(FunctionIndex)
    }

    pub fn get(&self, index: FunctionIndex) -> Option<&Function> {
        self.functions.get_index(index.0).map(|(_, func)| func)
    }

    pub fn get_by_name(&self, name: StringIndex) -> Option<&Function> {
        self.functions.get(&name)
    }

    pub fn iter(&self) -> impl Iterator<Item = (&StringIndex, &Function)> {
        self.functions.iter()
    }
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub struct FunctionIndex(usize);

#[derive(Clone, Debug)]
pub struct Strings {
    strings: IndexSet<String>,
}
impl Strings {
    fn new() -> Self {
        Strings {
            strings: IndexSet::new(),
        }
    }

    pub fn lookup_value_by_index(&self, idx: StringIndex) -> Option<&str> {
        self.strings.get_index(idx.0).map(|s| s.as_str())
    }

    pub fn lookup_index_by_value(&self, value: &str) -> Option<StringIndex> {
        self.strings.get_index_of(value).map(StringIndex)
    }

    fn intern(&mut self, value: &str) -> StringIndex {
        let (index, _) = self.strings.insert_full(value.to_string());
        StringIndex(index)
    }
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub struct StringIndex(usize);

impl StringIndex {
    pub const fn default() -> Self {
        Self(0)
    }
}

#[derive(Clone, Debug)]
pub struct Locations {
    locations: IndexSet<Location>,
}
impl Locations {
    fn new() -> Self {
        Locations {
            locations: IndexSet::new(),
        }
    }

    pub fn get(&self, index: LocationIndex) -> Option<Location> {
        self.locations.get_index(index.0 as usize).copied()
    }

    pub fn push(&mut self, location: Location) -> LocationIndex {
        let (index, _) = self.locations.insert_full(location);
        LocationIndex(index as u32)
    }
}

#[derive(Default, Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub struct LocationIndex(pub u32);

#[derive(Clone, Copy, Debug)]
pub struct VariableDef {
    pub index: VarId,
    pub initial_value: Value,
    pub type_: Type,
    pub mutable: bool,
}
impl VariableDef {
    pub(crate) fn value(&self) -> Value {
        self.initial_value
    }
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub struct VariableIndex(pub usize);
impl VariableIndex {
    pub(crate) fn index(&self) -> usize {
        self.0
    }
}

#[derive(Clone, Debug)]
pub struct DebugInfo {
    /// Locations of the source code.
    pub locations: Locations,
    pub instruction_locations: Vec<LocationIndex>,
}

#[derive(Clone, Debug)]
pub struct Program {
    pub source: String,
    pub functions: Functions,
    pub strings: Strings,
    pub code: Vec<Instruction>,
    pub debug_info: DebugInfo,
    pub variables: ScopeStack,
}

impl Program {
    pub fn lookup_string_index(&self, name: &str) -> Option<StringIndex> {
        self.strings.lookup_index_by_value(name)
    }

    fn intern_string(&mut self, string: &str) -> StringIndex {
        self.strings.intern(string)
    }

    pub(crate) fn location_at(&self, location: LocationIndex) -> Location {
        self.debug_info
            .locations
            .get(location)
            .expect("Location index out of bounds")
    }

    pub fn disasm(&self) -> String {
        let mut output = String::new();
        for (_, function) in self.functions.iter() {
            output.push_str(&format!("Function: {}\n", function.print_signature(self)));
            output.push_str("  Code:\n");

            let start = function.address;
            let length = function.length;
            let max_i_chars = length.to_string().len();

            for (i, bytecode) in self.code[start..][..length].iter().enumerate() {
                let i_chars = i.to_string().len();
                output.push_str(&format!(
                    "    {}:{padding:>padding_width$} {}\n",
                    i,
                    bytecode.disasm(self),
                    padding = "",
                    padding_width = max_i_chars - i_chars,
                ));
            }
        }
        output
    }
}

pub fn compile<'s>(source: &'s str, ast: &parser::Program) -> Result<Program, CompileError<'s>> {
    let mut this = Compiler {
        program: Program {
            source: source.to_string(),
            strings: Strings::new(),
            debug_info: DebugInfo {
                locations: Locations::new(),
                instruction_locations: Vec::new(),
            },
            functions: Functions::new(),
            variables: ScopeStack::new(),
            code: Vec::new(),
        },
    };

    for item in ast.items.iter() {
        match item {
            parser::Item::Function(func) => this.declare_function(source, func)?,
            _ => {}
        }
    }

    for item in ast.items.iter() {
        match item {
            parser::Item::Function(func) => this.compile_function(source, func)?,
            parser::Item::Constant(constant) => this.compile_global_constant(source, constant)?,
            parser::Item::GlobalVariable(global) => this.compile_global_variable(source, global)?,
        }
    }

    Ok(this.program)
}

pub struct Compiler {
    program: Program,
}

type ScopeIndex = usize;

#[derive(Clone, Debug)]
pub struct ScopeStack {
    scopes: Vec<Scope>,
    tree: ScopeTreeNode,
    current: ScopeIndex,
}

#[derive(Clone, Debug)]
struct ScopeTreeNode {
    parent: usize,
    id: usize,
    children: Vec<ScopeTreeNode>,
}
impl ScopeTreeNode {
    fn is_root(&self) -> bool {
        self.id == 0
    }
}

#[derive(Clone, Debug)]
struct Scope {
    kind: ScopeKind,
    // Currently declared and visible variables.
    variables: IndexMap<StringIndex, VariableDef>,
    // Number of variables declared in this scope. Needed due to shadowing.
    declared_count: usize,
}

impl Scope {
    fn new(kind: ScopeKind) -> Self {
        Scope {
            kind,
            variables: IndexMap::new(),
            declared_count: 0,
        }
    }
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
enum ScopeKind {
    Global,
    Local,
}

impl ScopeStack {
    fn new() -> Self {
        ScopeStack {
            scopes: vec![Scope::new(ScopeKind::Global)],
            tree: ScopeTreeNode {
                parent: 0,
                id: 0,
                children: vec![],
            },
            current: 0,
        }
    }

    fn open_scope(&mut self, kind: ScopeKind) -> Result<ScopeIndex, String> {
        if self.scopes[self.current].kind == ScopeKind::Local && kind == ScopeKind::Global {
            return Err("Cannot open a global scope inside a local scope".to_string());
        }

        // Create new scope
        let new_index = self.scopes.len();
        self.scopes.push(Scope::new(kind));

        // Insert new scope into the tree
        let new_node = ScopeTreeNode {
            parent: self.current,
            id: new_index,
            children: vec![],
        };

        let parent = std::mem::replace(&mut self.current, new_node.id);
        let parent_node = self.node_mut(parent);
        parent_node.children.push(new_node);

        Ok(self.current)
    }

    fn close_scope(&mut self) -> Result<(), String> {
        let current_node = self.node(self.current);
        self.current = current_node.parent;
        Ok(())
    }

    fn node_mut(&mut self, current: ScopeIndex) -> &mut ScopeTreeNode {
        let mut node = &mut self.tree;
        while node.id != current {
            if let Some(child) = node.children.last_mut() {
                node = child;
            } else {
                panic!("Scope not found in the tree");
            }
        }
        node
    }

    fn node(&self, current: ScopeIndex) -> &ScopeTreeNode {
        let mut stack = vec![&self.tree];
        while let Some(current_node) = stack.pop() {
            if current_node.id == current {
                return current_node;
            }
            stack.extend(&current_node.children);
        }

        panic!("Scope not found in the tree");
    }

    fn size_of_scope(&self, index: ScopeIndex) -> usize {
        let scope = &self.scopes[index];
        let mut count = scope.variables.len();

        for child in self.node(index).children.iter() {
            if self.scopes[child.id].kind == scope.kind {
                count += self.size_of_scope(child.id);
            }
        }

        count
    }

    pub fn n_globals(&self) -> usize {
        self.globals().count()
    }

    pub fn globals(&self) -> impl Iterator<Item = &VariableDef> {
        self.scopes
            .iter()
            .filter(|scope| scope.kind == ScopeKind::Global)
            .flat_map(|scope| scope.variables.values())
    }

    fn size_of_current_scope(&self) -> usize {
        self.size_of_scope(self.current)
    }

    /// Find the variable in the current scope or any outer scope.
    fn find(&self, variable_name: StringIndex) -> Option<VariableDef> {
        let mut node = self.node(self.current);
        loop {
            if let Some(def) = self.scopes[node.id].variables.get(&variable_name) {
                return Some(*def);
            }
            if node.is_root() {
                break; // Stop at the global scope
            }
            node = self.node(node.parent);
        }

        None
    }

    fn n_visible_locals(&self) -> usize {
        self.n_visible_locals_of(self.current)
    }

    fn n_visible_locals_of(&self, node: usize) -> usize {
        let mut count = 0;

        let mut node = self.node(node);
        loop {
            let scope = &self.scopes[node.id];
            if scope.kind == ScopeKind::Global {
                break; // Stop at the global scope
            }
            count += scope.declared_count;
            node = self.node(node.parent);
        }

        count
    }

    fn declare_variable(&mut self, name: StringIndex, mut def: VariableDef) -> Option<VarId> {
        let current_scope = &self.scopes[self.current];

        let address = if current_scope.kind == ScopeKind::Global {
            // Globals must be unique in their scope.
            if current_scope.variables.contains_key(&name) {
                return None;
            }

            self.n_globals()
        } else {
            self.n_visible_locals()
        };

        let index = VariableIndex(address);

        let current_scope = &mut self.scopes[self.current];

        def.index = if current_scope.kind == ScopeKind::Global {
            VarId::Global(index)
        } else {
            VarId::Local(index)
        };

        current_scope.variables.insert(name, def);
        current_scope.declared_count += 1;

        let id = if current_scope.kind == ScopeKind::Global {
            VarId::Global(index)
        } else {
            VarId::Local(index)
        };

        Some(id)
    }
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub enum VarId {
    Global(VariableIndex),
    Local(VariableIndex),
}

// TODO: figure out stack allocation and access to globals. Function scopes should be:
// - inner scope sees outer scope variables
// - outer scope does not see inner scope variables
// - global scope is visible from all inner scopes, but is addressed by absolute address
// - inner scopes can define variables that shadow outer scope variables
// - all scopes can define constants using the same rules as variables
// - globals need to be evaulated in compile-time
// - globals should have unique addresses across the whole program, to allow multiple modules
// - only module-globals should be visible by default
// - variable can be defined in the innermost scope only
// - variables are visible from innermost to outermost scope, i.e. shadowing is allowed
// - in local scopes, multiple variables can have the same name, but only the lastly defined variable is visible
//  - due to references, the shadowed variable should not be overwritten - although only functions can take variables by reference currently so this is not a problem yet

// Action items:
// - define references, taking a reference should give an absolute address
// - define variable mutability, constant is just immutable here

impl Compiler {
    fn declare_function<'s>(
        &mut self,
        source: &'s str,
        func: &parser::Function,
    ) -> Result<(), CompileError<'s>> {
        let name = func.name.source(source);
        let name_idx = self.program.intern_string(name);
        self.program.functions.declare(name_idx);
        Ok(())
    }

    fn compile_function<'s>(
        &mut self,
        source: &'s str,
        func: &parser::Function,
    ) -> Result<(), CompileError<'s>> {
        if let Err(error) = self.program.variables.open_scope(ScopeKind::Local) {
            return Err(CompileError {
                source,
                location: func.name.location,
                error: format!("Failed to open scope: {error}"),
            });
        }

        let signature = self.compile_function_signature(source, func)?;
        let body = self.compile_function_body(source, func)?;

        let address = self.program.code.len();
        self.program.code.extend_from_slice(&body.code);
        self.program
            .debug_info
            .instruction_locations
            .extend_from_slice(&body.locations);

        let function = Function {
            signature,
            return_type: body.return_type,
            address,
            length: body.code.len(),
            scope_size: self.program.variables.size_of_current_scope(),
        };
        if let Err(error) = self.program.variables.close_scope() {
            return Err(CompileError {
                source,
                location: func.name.location,
                error: format!("Failed to open scope: {error}"),
            });
        }

        self.program.functions.push(function);
        Ok(())
    }

    fn compile_function_signature<'s>(
        &mut self,
        source: &'s str,
        func: &parser::Function,
    ) -> Result<FunctionSignature, CompileError<'s>> {
        let name = func.name.source(source);
        let name_idx = self.program.intern_string(name);
        let args = func
            .arguments
            .iter()
            .map(|arg| {
                let arg_type = if arg.reference_token.is_some() {
                    Type::Address
                } else {
                    self.compile_type(source, arg.arg_type)?
                };

                let var_name = arg.name.source(source);
                let name_idx = self.program.intern_string(var_name);
                let Some(VarId::Local(variable_index)) = self.program.variables.declare_variable(
                    name_idx,
                    VariableDef {
                        initial_value: Value::Void,
                        index: VarId::Local(VariableIndex(0)),
                        type_: arg_type,
                        mutable: true,
                    },
                ) else {
                    return Err(CompileError {
                        source,
                        location: arg.name.location,
                        error: format!("Variable `{var_name}` is already defined in this scope"),
                    });
                };
                //println!("Allocating {var_name} as {:?}", variable_index);

                Ok((variable_index, arg_type))
            })
            .collect::<Result<Vec<_>, _>>()?;

        Ok(FunctionSignature {
            name: name_idx,
            args,
            location: func.name.location,
        })
    }

    fn compile_function_body<'s>(
        &mut self,
        source: &'s str,
        func: &parser::Function,
    ) -> Result<FunctionBody, CompileError<'s>> {
        let return_type = if let Some(return_decl) = &func.return_decl {
            self.compile_type(source, return_decl.return_type)?
        } else {
            Type::Void
        };
        let mut code = self.compile_body(source, &func.body)?;

        if code
            .last()
            .map_or(false, |c| !matches!(c.instruction, Instruction::Return))
        {
            // If the last instruction is not a return, we need to add a return
            code.push(Bytecode {
                instruction: Instruction::Return,
                location: self.intern_location(func.body.closing_brace.location),
            });
        }

        // Separate bytecode to instructions and locations
        let (instructions, locations): (Vec<_>, Vec<_>) = code
            .iter()
            .map(|bytecode| (bytecode.instruction, bytecode.location))
            .unzip();

        Ok(FunctionBody {
            code: instructions,
            locations,
            return_type,
        })
    }

    fn compile_body<'s>(
        &mut self,
        source: &'s str,
        body: &parser::Body,
    ) -> Result<Vec<Bytecode>, CompileError<'s>> {
        let mut code = Vec::new();

        if let Err(error) = self.program.variables.open_scope(ScopeKind::Local) {
            return Err(CompileError {
                source,
                location: body.opening_brace.location,
                error: format!("Failed to open scope: {error}"),
            });
        }
        for statement in body.statements.iter() {
            let bytecode = self.compile_statement(source, statement)?;
            code.extend_from_slice(&bytecode);
        }
        if let Err(error) = self.program.variables.close_scope() {
            return Err(CompileError {
                source,
                location: body.closing_brace.location,
                error: format!("Failed to close scope: {error}"),
            });
        }

        Ok(code)
    }

    fn compile_statement<'s>(
        &mut self,
        source: &'s str,
        statement: &parser::Statement,
    ) -> Result<Vec<Bytecode>, CompileError<'s>> {
        match statement {
            parser::Statement::VariableDefinition(variable_def) => {
                self.compile_variable_definition(source, variable_def)
            }
            parser::Statement::ConstantDefinition(constant_def) => {
                self.compile_constant_definition(source, constant_def)
            }
            parser::Statement::Return(ret_with_value) => {
                self.compile_return_with_value(source, ret_with_value)
            }
            parser::Statement::EmptyReturn(ret) => self.compile_empty_return(source, ret),
            parser::Statement::If(if_statement) => self.compile_if_statement(source, if_statement),
            parser::Statement::Loop(loop_statement) => {
                self.compile_loop_statement(source, loop_statement)
            }
            parser::Statement::While(while_statement) => {
                self.compile_while_statement(source, while_statement)
            }
            parser::Statement::Break(break_statement) => {
                self.compile_break_statement(source, break_statement)
            }
            parser::Statement::Continue(continue_statement) => {
                self.compile_continue_statement(source, continue_statement)
            }
            parser::Statement::Expression {
                expression,
                semicolon,
            } => self.compile_expression_statement(source, expression, semicolon),
        }
    }

    fn compile_global_constant<'s>(
        &mut self,
        source: &'s str,
        constant: &parser::Constant,
    ) -> Result<(), CompileError<'s>> {
        self.compile_global(
            source,
            constant.identifier,
            constant.type_token,
            &constant.value,
            false,
        )?;
        Ok(())
    }

    fn compile_global_variable<'s>(
        &mut self,
        source: &'s str,
        global: &parser::GlobalVariable,
    ) -> Result<(), CompileError<'s>> {
        self.compile_global(
            source,
            global.identifier,
            global.type_token,
            &global.initializer,
            true,
        )?;
        Ok(())
    }

    fn compile_global<'s>(
        &mut self,
        source: &'s str,
        identifier: lexer::Token,
        type_token: parser::Type,
        value: &parser::Expression,
        mutable: bool,
    ) -> Result<VariableIndex, CompileError<'s>> {
        let parser::Expression::Literal { value } = &value else {
            return Err(CompileError {
                source,
                location: value.location(),
                error: "Only literal values are allowed for globals".to_string(),
            });
        };

        let val_type = self.compile_type(source, type_token)?;
        let value = self.compile_literal_value(source, val_type, value)?;

        let var_name = identifier.source(source);
        let name_idx = self.program.intern_string(var_name);
        let Some(VarId::Global(variable_index)) = self.program.variables.declare_variable(
            name_idx,
            VariableDef {
                initial_value: value,
                index: VarId::Global(VariableIndex(0)),
                type_: val_type,
                mutable: mutable,
            },
        ) else {
            return Err(CompileError {
                source,
                location: identifier.location,
                error: format!("Variable `{var_name}` is already defined in this scope"),
            });
        };

        Ok(variable_index)
    }

    fn compile_literal_value<'s>(
        &mut self,
        source: &'s str,
        literal_type: Type,
        literal: &parser::Literal,
    ) -> Result<Value, CompileError<'s>> {
        let value = match (literal_type, &literal.value) {
            (Type::Int, parser::LiteralValue::Integer(value)) => Value::Int(*value),
            (Type::Float, parser::LiteralValue::Float(value)) => Value::Float(*value),
            (Type::Bool, parser::LiteralValue::Boolean(value)) => Value::Bool(*value),
            (Type::String, parser::LiteralValue::String(value)) => {
                let string_index = self.program.intern_string(value);
                Value::String(string_index)
            }
            (Type::Void | Type::Address, _) => {
                return Err(CompileError {
                    source,
                    location: literal.location,
                    error: format!("{literal_type} is not supported in literals"),
                });
            }
            _ => {
                return Err(CompileError {
                    source,
                    location: literal.location,
                    error: format!(
                        "Expected {literal_type}, found {} literal",
                        match literal.value {
                            parser::LiteralValue::Integer(_) => "integer",
                            parser::LiteralValue::Float(_) => "float",
                            parser::LiteralValue::Boolean(_) => "boolean",
                            parser::LiteralValue::String(_) => "string",
                        }
                    ),
                });
            }
        };

        Ok(value)
    }

    fn compile_type<'s>(
        &mut self,
        source: &'s str,
        type_token: parser::Type,
    ) -> Result<Type, CompileError<'s>> {
        match type_token.type_name.source(source) {
            "int" => Ok(Type::Int),
            "signed" => Ok(Type::SignedInt),
            "float" => Ok(Type::Float),
            "bool" => Ok(Type::Bool),
            "string" => Ok(Type::String),
            other => Err(CompileError {
                source,
                location: type_token.type_name.location,
                error: format!("Unknown type `{other}`"),
            }),
        }
    }

    fn compile_variable_definition<'s>(
        &mut self,
        source: &'s str,
        variable_def: &parser::VariableDefinition,
    ) -> Result<Vec<Bytecode>, CompileError<'s>> {
        let var_name = variable_def.identifier.source(source);
        let name_idx = self.program.intern_string(var_name);

        let type_ = if let Some(type_token) = &variable_def.type_token {
            self.compile_type(source, type_token.clone())?
        } else {
            Type::Void
        };

        let Some(VarId::Local(address)) = self.program.variables.declare_variable(
            name_idx,
            VariableDef {
                initial_value: Value::Void,
                index: VarId::Local(VariableIndex(0)),
                type_,
                mutable: true,
            },
        ) else {
            return Err(CompileError {
                source,
                location: variable_def.identifier.location,
                error: format!("Variable `{var_name}` is already defined in this scope"),
            });
        };
        //println!("Allocating {var_name} as {:?}", var_id);

        let mut code = self.compile_expression(source, &variable_def.initializer)?;
        code.push(Bytecode {
            instruction: Instruction::Push(Value::Address(VarId::Local(address))),
            location: self.intern_location(variable_def.location()),
        });
        code.push(Bytecode {
            instruction: Instruction::Store,
            location: self.intern_location(variable_def.location()),
        });

        Ok(code)
    }

    fn compile_constant_definition<'s>(
        &mut self,
        source: &'s str,
        constant_def: &parser::ConstantDefinition,
    ) -> Result<Vec<Bytecode>, CompileError<'s>> {
        let const_name = constant_def.identifier.source(source);
        let name_idx = self.program.intern_string(const_name);

        let mut initializer_code = vec![];
        let (type_, initial_value) = if let Some(type_token) = &constant_def.type_token {
            let type_ = self.compile_type(source, type_token.clone())?;

            let initial_value =
                if let Expression::Literal { value: literal } = &constant_def.initializer {
                    self.compile_literal_value(source, type_, literal)?
                } else {
                    initializer_code.extend_from_slice(
                        &self.compile_expression(source, &constant_def.initializer)?,
                    );
                    Value::Void
                };

            (type_, initial_value)
        } else {
            (Type::Void, Value::Void)
        };

        let Some(address @ VarId::Local(_)) = self.program.variables.declare_variable(
            name_idx,
            VariableDef {
                initial_value,
                index: VarId::Local(VariableIndex(0)),
                type_,
                mutable: false,
            },
        ) else {
            return Err(CompileError {
                source,
                location: constant_def.identifier.location,
                error: format!("Constant `{const_name}` is already defined in this scope"),
            });
        };

        if !initializer_code.is_empty() {
            initializer_code.push(Bytecode {
                instruction: Instruction::Push(Value::Address(address)),
                location: self.intern_location(constant_def.location()),
            });
            initializer_code.push(Bytecode {
                instruction: Instruction::Store,
                location: self.intern_location(constant_def.location()),
            });
        }

        Ok(initializer_code)
    }

    fn compile_return_with_value<'s>(
        &mut self,
        source: &'s str,
        ret_with_value: &parser::ReturnWithValue,
    ) -> Result<Vec<Bytecode>, CompileError<'s>> {
        let mut expression_code = self.compile_expression(source, &ret_with_value.expression)?;
        expression_code.push(Bytecode {
            instruction: Instruction::Return,
            location: self.intern_location(ret_with_value.location()),
        });
        Ok(expression_code)
    }

    fn compile_empty_return<'s>(
        &mut self,
        _source: &'s str,
        empty_return: &parser::EmptyReturn,
    ) -> Result<Vec<Bytecode>, CompileError<'s>> {
        let location = empty_return.location();
        Ok(vec![Bytecode {
            instruction: Instruction::Return,
            location: self.intern_location(location),
        }])
    }

    fn compile_if_statement<'s>(
        &mut self,
        source: &'s str,
        if_statement: &parser::If,
    ) -> Result<Vec<Bytecode>, CompileError<'s>> {
        // condition
        // if false, jump to else_body
        // then_body
        // jump over else_body
        // else_body

        let mut then_body = self.compile_body(source, &if_statement.body)?;
        let else_body = if let Some(else_statement) = &if_statement.else_branch {
            let else_body = self.compile_body(source, &else_statement.else_body)?;
            // then_body ends with a jump over the else_body
            then_body.push(Bytecode {
                instruction: Instruction::JumpForward(else_body.len()),
                location: self.intern_location(if_statement.if_token.location),
            });
            else_body
        } else {
            Vec::new()
        };

        let mut code = self.compile_expression(source, &if_statement.condition)?;
        code.push(Bytecode {
            instruction: Instruction::JumpForwardIf(then_body.len(), false),
            location: self.intern_location(if_statement.if_token.location),
        });
        code.extend_from_slice(&then_body);
        code.extend_from_slice(&else_body);

        Ok(code)
    }

    fn compile_loop_statement<'s>(
        &mut self,
        source: &'s str,
        loop_statement: &parser::Loop,
    ) -> Result<Vec<Bytecode>, CompileError<'s>> {
        let mut body = vec![];

        body.push(Bytecode {
            instruction: Instruction::LoopStart(0),
            location: self.intern_location(loop_statement.loop_token.location),
        });

        body.extend_from_slice(&self.compile_body(source, &loop_statement.body)?);

        body.push(Bytecode {
            instruction: Instruction::Continue,
            location: self.intern_location(loop_statement.loop_token.location),
        });

        // Patch in the loop size
        body[0].instruction = Instruction::LoopStart(body.len());

        Ok(body)
    }

    fn compile_while_statement<'s>(
        &mut self,
        source: &'s str,
        while_statement: &parser::While,
    ) -> Result<Vec<Bytecode>, CompileError<'s>> {
        let desugared = parser::Statement::Loop(parser::Loop {
            loop_token: while_statement.while_token.clone(),
            body: parser::Body {
                opening_brace: while_statement.body.opening_brace,
                closing_brace: while_statement.body.closing_brace,
                statements: vec![parser::Statement::If(parser::If {
                    if_token: while_statement.while_token,
                    condition: while_statement.condition.clone(),
                    body: while_statement.body.clone(),
                    else_branch: Some(parser::Else {
                        else_token: while_statement.while_token,
                        else_body: parser::Body {
                            opening_brace: while_statement.body.opening_brace,
                            closing_brace: while_statement.body.closing_brace,
                            statements: vec![parser::Statement::Break(parser::Break {
                                break_token: while_statement.while_token,
                                semicolon: while_statement.while_token,
                            })],
                        },
                    }),
                })],
            },
        });

        self.compile_statement(source, &desugared)
    }

    fn compile_break_statement<'s>(
        &mut self,
        _source: &'s str,
        break_statement: &parser::Break,
    ) -> Result<Vec<Bytecode>, CompileError<'s>> {
        Ok(vec![Bytecode {
            instruction: Instruction::Break,
            location: self.intern_location(break_statement.break_token.location),
        }])
    }

    fn compile_continue_statement<'s>(
        &mut self,
        _source: &'s str,
        continue_statement: &parser::Continue,
    ) -> Result<Vec<Bytecode>, CompileError<'s>> {
        Ok(vec![Bytecode {
            instruction: Instruction::Continue,
            location: self.intern_location(continue_statement.continue_token.location),
        }])
    }

    fn compile_expression_statement<'s>(
        &mut self,
        source: &'s str,
        expression: &parser::Expression,
        semicolon: &lexer::Token,
    ) -> Result<Vec<Bytecode>, CompileError<'s>> {
        let mut code = self.compile_expression(source, expression)?;

        code.push(Bytecode {
            instruction: Instruction::Pop,
            location: self.intern_location(semicolon.location),
        });

        Ok(code)
    }

    fn compile_expression<'s>(
        &mut self,
        source: &'s str,
        expression: &parser::Expression,
    ) -> Result<Vec<Bytecode>, CompileError<'s>> {
        let location = self.intern_location(expression.location());
        match &expression {
            parser::Expression::Literal { value } => {
                let val_type = match value.value {
                    parser::LiteralValue::Integer(_) => Type::Int,
                    parser::LiteralValue::Float(_) => Type::Float,
                    parser::LiteralValue::Boolean(_) => Type::Bool,
                    parser::LiteralValue::String(_) => Type::String,
                };

                let value = self.compile_literal_value(source, val_type, value)?;

                Ok(vec![Bytecode {
                    instruction: Instruction::Push(value),
                    location,
                }])
            }
            parser::Expression::Variable { .. } => {
                let variable = self.compile_variable(source, expression)?;
                Ok(vec![
                    Bytecode {
                        instruction: Instruction::Push(Value::Address(variable.index)),
                        location,
                    },
                    Bytecode {
                        instruction: Instruction::Load,
                        location,
                    },
                ])
            }
            parser::Expression::UnaryOperator { name, operand } => {
                let operator = name.source(source);

                let mut code = vec![];

                let instruction = match operator {
                    "!" => {
                        code.extend_from_slice(&self.compile_expression(source, operand)?);
                        Instruction::InvertBoolean
                    }
                    "-" => {
                        code.extend_from_slice(&self.compile_expression(source, operand)?);
                        Instruction::Negate
                    }
                    "&" => {
                        let variable = self.compile_variable(source, operand)?;

                        if !variable.mutable {
                            return Err(CompileError {
                                source,
                                location: expression.location(),
                                error: "Cannot take the address of a constant".to_string(),
                            });
                        }

                        Instruction::AddressOf(variable.index)
                    }
                    "*" => {
                        let variable = self.compile_variable(source, operand)?;
                        Instruction::Dereference(variable.index)
                    }
                    _ => {
                        return Err(CompileError {
                            source,
                            location: name.location,
                            error: format!("Unknown unary operator `{operator}`"),
                        });
                    }
                };

                code.push(Bytecode {
                    instruction,
                    location,
                });

                Ok(code)
            }
            parser::Expression::BinaryOperator { name, operands } => {
                let operator = name.source(source);
                match operator {
                    "=" => {
                        let mut current_operand = &operands[0];
                        let mut derefs = vec![];
                        let mut code = vec![];
                        loop {
                            match current_operand {
                                Expression::Variable { variable } => {
                                    let var_name = variable.source(source);
                                    let name_idx = self.program.intern_string(var_name);
                                    let Some(variable_index) =
                                        self.program.variables.find(name_idx)
                                    else {
                                        return Err(CompileError {
                                            source,
                                            location: variable.location,
                                            error: format!("Variable `{var_name}` is not defined"),
                                        });
                                    };

                                    if !variable_index.mutable {
                                        return Err(CompileError {
                                            source,
                                            location: expression.location(),
                                            error: "Cannot assign to a constant".to_string(),
                                        });
                                    }

                                    code.extend_from_slice(
                                        &self.compile_expression(source, &operands[1])?,
                                    );
                                    code.push(Bytecode {
                                        instruction: Instruction::Push(Value::Address(
                                            variable_index.index,
                                        )),
                                        location,
                                    });
                                    break;
                                }
                                Expression::UnaryOperator { name, operand }
                                    if name.source(source) == "*" =>
                                {
                                    derefs.push(self.intern_location(name.location));
                                    current_operand = operand;
                                    continue;
                                }
                                _ => {
                                    return Err(CompileError {
                                        source,
                                        location: name.location,
                                        error: "Left-hand side of assignment must be a variable, or a dereference of a variable"
                                            .to_string(),
                                    });
                                }
                            }
                        }

                        for loc in derefs.into_iter().rev() {
                            code.push(Bytecode {
                                instruction: Instruction::Load,
                                location: loc,
                            });
                        }

                        code.push(Bytecode {
                            instruction: Instruction::Store,
                            location,
                        });
                        // This value will be popped off by `expression_statement`.
                        // TODO: we should be able to optimize this away.
                        code.push(Bytecode {
                            instruction: Instruction::Push(Value::Void),
                            location,
                        });

                        Ok(code)
                    }

                    "<" | "<=" | ">" | ">=" | "==" | "!=" | "|" | "^" | "&" | "<<" | ">>" | "+"
                    | "-" | "*" | "/" => {
                        let mut lhs = self.compile_expression(source, &operands[0])?;
                        let mut rhs = self.compile_expression(source, &operands[1])?;

                        // Normalize some operators.
                        let operator = match operator {
                            ">" => {
                                std::mem::swap(&mut lhs, &mut rhs);
                                "<"
                            }
                            ">=" => {
                                std::mem::swap(&mut lhs, &mut rhs);
                                "<="
                            }

                            other => other,
                        };

                        let instruction = match operator {
                            "<" => Instruction::TestLessThan,
                            "<=" => Instruction::TestLessThanOrEqual,
                            "==" => Instruction::TestEquals,
                            "!=" => Instruction::TestNotEquals,
                            "|" => Instruction::BitwiseOr,
                            "^" => Instruction::BitwiseXor,
                            "&" => Instruction::BitwiseAnd,
                            "<<" => Instruction::ShiftLeft,
                            ">>" => Instruction::ShiftRight,
                            "+" => Instruction::Add,
                            "-" => Instruction::Subtract,
                            "*" => Instruction::Multiply,
                            "/" => Instruction::Divide,
                            _ => {
                                return Err(CompileError {
                                    source,
                                    location: name.location,
                                    error: format!("Unknown operator `{operator}`"),
                                });
                            }
                        };
                        let mut code = Vec::new();
                        code.append(&mut lhs);
                        code.append(&mut rhs);

                        code.push(Bytecode {
                            instruction,
                            location,
                        });

                        Ok(code)
                    }

                    "&&" | "||" => {
                        let lhs = self.compile_expression(source, &operands[0])?;
                        let mut rhs = self.compile_expression(source, &operands[1])?;

                        let mut code = lhs;

                        match operator {
                            "&&" => {
                                // a && b -> if a { b } else { false }
                                code.push(Bytecode {
                                    instruction: Instruction::JumpForwardIf(rhs.len() + 1, false),
                                    location: self.intern_location(name.location),
                                });
                                code.append(&mut rhs);
                                code.push(Bytecode {
                                    instruction: Instruction::JumpForward(1),
                                    location: self.intern_location(name.location),
                                });
                                code.push(Bytecode {
                                    instruction: Instruction::Push(Value::Bool(false)),
                                    location: self.intern_location(name.location),
                                });
                            }
                            "||" => {
                                // a || b -> if a { true } else { b }
                                code.push(Bytecode {
                                    instruction: Instruction::JumpForwardIf(rhs.len() + 1, true),
                                    location: self.intern_location(name.location),
                                });
                                code.append(&mut rhs);
                                code.push(Bytecode {
                                    instruction: Instruction::JumpForward(1),
                                    location: self.intern_location(name.location),
                                });
                                code.push(Bytecode {
                                    instruction: Instruction::Push(Value::Bool(true)),
                                    location: self.intern_location(name.location),
                                });
                            }
                            _ => unreachable!("Operator {operator} should have been handled above"),
                        };

                        Ok(code)
                    }
                    _ => Err(CompileError {
                        source,
                        location: expression.location(),
                        error: format!("Unknown operator `{operator}`"),
                    }),
                }
            }
            parser::Expression::FunctionCall { name, arguments } => {
                let name_str = name.source(source);
                let name_index = self.program.strings.intern(name_str);

                let function_index = self.program.functions.index_of(name_index);
                let arg_count = arguments.len();

                let instruction = match function_index {
                    Some(index) => Instruction::Call(index, arg_count),
                    None => Instruction::CallByName(name_index, arg_count),
                };

                let mut code = Vec::new();

                for arg in arguments.iter() {
                    let arg_code = self.compile_expression(source, arg)?;
                    code.extend_from_slice(&arg_code);
                }

                code.push(Bytecode {
                    instruction,
                    location,
                });

                Ok(code)
            }
        }
    }

    fn intern_location(&mut self, location: Location) -> LocationIndex {
        self.program.debug_info.locations.push(location)
    }

    fn compile_variable<'p>(
        &mut self,
        source: &'p str,
        expression: &parser::Expression,
    ) -> Result<VariableDef, CompileError<'p>> {
        match expression {
            parser::Expression::Variable { variable } => {
                let var_name = variable.source(source);
                let name_idx = self.program.intern_string(var_name);
                //println!("Looking for {var_name} ({name_idx:?})");
                let Some(variable_index) = self.program.variables.find(name_idx) else {
                    return Err(CompileError {
                        source,
                        location: variable.location,
                        error: format!("Variable `{var_name}` is not defined"),
                    });
                };

                //println!("Loading {var_name} ({name_idx:?}) as {:?}", variable_index);

                Ok(variable_index)
            }
            _ => Err(CompileError {
                source,
                location: expression.location(),
                error: "Expected a variable".to_string(),
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_compiler() {
        crate::test::run_compile_tests("tests/compiler/");
    }
}
