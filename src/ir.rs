use std::fmt::{Debug, Display, Write as _};

use indexmap::IndexMap;

use crate::{
    error::CompileError,
    ir,
    string_interner::{StringIndex, StringInterner},
    types::{TypeExt, TypedValue, VmTypeSet},
    variable_tracker::{LocalVariableIndex, RestorePoint, ScopeData, VariableTracker},
};

use somni_parser::{
    Location,
    ast::{self, FunctionArgument},
    lexer::Token,
};

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Value {
    Void,
    MaybeSignedInt(u64),
    Int(u64),
    SignedInt(i64),
    Float(f64),
    Bool(bool),
    String(StringIndex),
}

impl Value {
    pub(crate) fn into_typed_value(self) -> TypedValue {
        match self {
            ir::Value::Void => TypedValue::Void,
            ir::Value::MaybeSignedInt(value) => TypedValue::MaybeSignedInt(value),
            ir::Value::Int(value) => TypedValue::Int(value),
            ir::Value::SignedInt(value) => TypedValue::SignedInt(value),
            ir::Value::Float(value) => TypedValue::Float(value),
            ir::Value::Bool(value) => TypedValue::Bool(value),
            ir::Value::String(value) => TypedValue::String(value),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Variable {
    Value(Type),
    Reference(usize, Type),
}

impl Display for Variable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Value(ty) => Display::fmt(ty, f),
            Self::Reference(n, ty) => f.write_fmt(format_args!("{amp:n$}{ty}", amp = "&", n = n)),
        }
    }
}
impl Variable {
    pub(crate) fn reference(&self) -> Self {
        match self {
            Self::Value(t) => Self::Reference(1, *t),
            Self::Reference(n, t) => Self::Reference(*n + 1, *t),
        }
    }
    pub(crate) fn dereference(&self) -> Option<Self> {
        match self {
            Self::Value(_) => None,
            Self::Reference(1, t) => Some(Self::Value(*t)),
            Self::Reference(n, t) => Some(Self::Reference(*n - 1, *t)),
        }
    }
    pub(crate) fn maybe_signed_integer(&self) -> bool {
        matches!(
            self,
            Self::Value(Type::MaybeSignedInt) | Self::Reference(_, Type::MaybeSignedInt)
        )
    }

    pub(crate) fn is_integer(&self) -> bool {
        let (Self::Value(t) | Self::Reference(_, t)) = self;
        matches!(t, Type::MaybeSignedInt | Type::Int | Type::SignedInt)
    }
}

impl Value {
    pub fn type_of(&self) -> Type {
        match self {
            Value::Void => Type::Void,
            Value::MaybeSignedInt(_) => Type::MaybeSignedInt,
            Value::Int(_) => Type::Int,
            Value::SignedInt(_) => Type::SignedInt,
            Value::Bool(_) => Type::Bool,
            Value::String(_) => Type::String,
            Value::Float(_) => Type::Float,
        }
    }
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub enum Type {
    Void,
    MaybeSignedInt,
    Int,
    SignedInt,
    Float,
    Bool,
    String,
    Iter,
    /// A struct type, identified by its layout index in [`Program::structs`].
    Struct(StructId),
}

impl Display for Type {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Type::Void => write!(f, "void"),
            Type::MaybeSignedInt => write!(f, "{{int/signed}}"),
            Type::Int => write!(f, "int"),
            Type::SignedInt => write!(f, "signed"),
            Type::Bool => write!(f, "bool"),
            Type::String => write!(f, "string"),
            Type::Float => write!(f, "float"),
            Type::Iter => write!(f, "iter"),
            Type::Struct(id) => write!(f, "struct#{}", id.0),
        }
    }
}

/// An index into [`Program::structs`], identifying a struct's memory layout.
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub struct StructId(pub usize);

/// A field within a [`StructLayout`]: its name, type, and byte offset from the
/// start of the struct.
#[derive(Clone, Debug, PartialEq)]
pub struct StructFieldLayout {
    pub name: StringIndex,
    pub ty: Type,
    pub offset: usize,
}

/// The packed (no-alignment) memory layout of a struct type.
#[derive(Clone, Debug, PartialEq)]
pub struct StructLayout {
    pub name: StringIndex,
    pub fields: Vec<StructFieldLayout>,
    pub size: usize,
}

/// The set of struct layouts in a program, plus a name-to-id index.
#[derive(Clone, Debug, Default)]
pub struct StructRegistry {
    pub layouts: Vec<StructLayout>,
    pub ids: IndexMap<StringIndex, StructId>,
}

impl StructRegistry {
    pub fn layout(&self, id: StructId) -> &StructLayout {
        &self.layouts[id.0]
    }

    pub fn id_of(&self, name: StringIndex) -> Option<StructId> {
        self.ids.get(&name).copied()
    }

    /// Returns the byte size of a type, resolving struct sizes via the registry.
    pub fn size_of(&self, ty: Type) -> usize {
        match ty {
            Type::Struct(id) => self.layout(id).size,
            other => somni_expr::Type::from(other).vm_size_of(),
        }
    }

    /// Looks up a field of a struct by name, returning its layout.
    pub fn field(&self, id: StructId, name: StringIndex) -> Option<&StructFieldLayout> {
        self.layout(id).fields.iter().find(|f| f.name == name)
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct BlockIndex(pub usize);

impl Debug for BlockIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "block {}", self.0)
    }
}

impl BlockIndex {
    const RETURN_BLOCK: Self = BlockIndex(1);
}

// This is a temporary hack for function arguments.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AllocationMethod {
    FirstFit,   // Allocate the first free address that fits
    TopOfStack, // Allocate at the top of the stack
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VariableDeclaration {
    pub index: LocalVariableIndex,
    pub name: StringIndex,
    pub allocation_method: AllocationMethod,
    pub is_argument: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum VariableIndex {
    Local(LocalVariableIndex),
    Temporary(LocalVariableIndex),
    Global(GlobalVariableIndex),
}

impl VariableIndex {
    const RETURN_VALUE: Self = VariableIndex::Local(LocalVariableIndex::RETURN_VALUE);

    fn name<'s>(&self, program: &'s Program, function: &Function) -> &'s str {
        match self {
            VariableIndex::Local(idx) => idx.name(program, function),
            VariableIndex::Temporary(idx) => idx.name(program, function),
            VariableIndex::Global(idx) => idx.name(program),
        }
    }

    pub fn local_index(self) -> Option<LocalVariableIndex> {
        match self {
            VariableIndex::Local(idx) | VariableIndex::Temporary(idx) => Some(idx),
            VariableIndex::Global(_) => None,
        }
    }
}

impl LocalVariableIndex {
    fn name<'s>(&self, program: &'s Program, function: &Function) -> &'s str {
        let var = function
            .implementation
            .as_ref()
            .unwrap()
            .variables
            .variable(*self)
            .expect("Local variable index out of bounds");
        program.strings.lookup(var.name)
    }
}

#[derive(Debug, PartialEq)]
pub struct IrWithLocation {
    pub instruction: Ir,
    pub source_location: Location,
}

impl IrWithLocation {
    fn print(&self, program: &Program, function: &Function) -> String {
        self.instruction.print(program, function)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Ir {
    Declare(VariableDeclaration, Option<Value>), // when allocating, may reuse freed addresses
    Assign(VariableIndex, VariableIndex),
    DerefAssign(VariableIndex, VariableIndex),
    FreeVariable(LocalVariableIndex), // only used for temporaries & scope ends

    Call(StringIndex, VariableIndex, Vec<VariableIndex>), // function name, arguments
    BinaryOperator(StringIndex, VariableIndex, VariableIndex, VariableIndex),
    UnaryOperator(StringIndex, VariableIndex, VariableIndex),

    /// `dst` (bool) = whether `iter` can yield another value.
    IterHasNext(VariableIndex, VariableIndex),
    /// `dst` (element type) = next value from `iter`.
    IterNext(VariableIndex, VariableIndex),

    /// `dst` = the `size`-byte field at `base` + `offset`. When `indirect`, `base`
    /// holds an address (a reference); otherwise `base` is the aggregate itself.
    LoadField {
        dst: VariableIndex,
        base: VariableIndex,
        offset: usize,
        size: usize,
        indirect: bool,
    },
    /// The `size`-byte field at `base` + `offset` = `src`. `indirect` as above.
    StoreField {
        base: VariableIndex,
        offset: usize,
        size: usize,
        indirect: bool,
        src: VariableIndex,
    },
    /// `dst` (a reference) = the address of `base` + `offset`. `indirect` as above.
    AddressOfField {
        dst: VariableIndex,
        base: VariableIndex,
        offset: usize,
        indirect: bool,
    },
}

impl Ir {
    fn print(&self, program: &Program, function: &Function) -> String {
        let mut output = String::new();
        match self {
            Self::Declare(var, init_value) => write!(
                &mut output,
                "var {}: {} = {:?}",
                program.strings.lookup(var.name),
                function
                    .implementation
                    .as_ref()
                    .unwrap()
                    .variables
                    .variable(var.index)
                    .unwrap()
                    .ty
                    .unwrap_or(Variable::Value(Type::Void)),
                init_value
            )
            .unwrap(),
            Self::Assign(dst, src) => {
                let dst = dst.name(program, function);
                let src = src.name(program, function);

                write!(&mut output, "{dst} = {src}").unwrap()
            }
            Self::DerefAssign(dst, src) => {
                let dst = dst.name(program, function);
                let src = src.name(program, function);

                write!(&mut output, "*{dst} = {src}").unwrap()
            }
            Self::FreeVariable(var) => {
                let var = var.name(program, function);

                write!(&mut output, "free({var})").unwrap()
            }
            Self::Call(f, retval, args) => {
                let f = program.strings.lookup(*f);

                let args = args
                    .iter()
                    .map(|arg| arg.name(program, function))
                    .collect::<Vec<_>>()
                    .join(", ");

                let retval = retval.name(program, function);

                write!(&mut output, "{retval} = call {f}({args})").unwrap()
            }
            Self::UnaryOperator(op, dst, operand) => {
                let dst = dst.name(program, function);

                let operand = operand.name(program, function);
                let operator = program.strings.lookup(*op);

                write!(&mut output, "{dst} = {operator}{operand}").unwrap()
            }
            Self::BinaryOperator(op, dst, left, right) => {
                let dst = dst.name(program, function);
                let left = left.name(program, function);
                let right = right.name(program, function);

                let operator = program.strings.lookup(*op);

                write!(&mut output, "{dst} = {left} {operator} {right}").unwrap()
            }
            Self::IterHasNext(dst, iter) => {
                let dst = dst.name(program, function);
                let iter = iter.name(program, function);
                write!(&mut output, "{dst} = has_next({iter})").unwrap()
            }
            Self::IterNext(dst, iter) => {
                let dst = dst.name(program, function);
                let iter = iter.name(program, function);
                write!(&mut output, "{dst} = next({iter})").unwrap()
            }
            Self::LoadField {
                dst,
                base,
                offset,
                size,
                indirect,
            } => {
                let dst = dst.name(program, function);
                let base = base.name(program, function);
                let star = if *indirect { "*" } else { "" };
                write!(&mut output, "{dst} = {star}{base}[{offset}..+{size}]").unwrap()
            }
            Self::StoreField {
                base,
                offset,
                size,
                indirect,
                src,
            } => {
                let base = base.name(program, function);
                let src = src.name(program, function);
                let star = if *indirect { "*" } else { "" };
                write!(&mut output, "{star}{base}[{offset}..+{size}] = {src}").unwrap()
            }
            Self::AddressOfField {
                dst,
                base,
                offset,
                indirect,
            } => {
                let dst = dst.name(program, function);
                let base = base.name(program, function);
                let star = if *indirect { "*" } else { "" };
                write!(&mut output, "{dst} = &{star}{base}[{offset}]").unwrap()
            }
        }
        output
    }
}

#[derive(Debug)]
pub enum Termination {
    Return(Location),
    // Tries to lay out the next block linearly, jumps if its already laid out.
    Jump {
        source_location: Location,
        to: BlockIndex,
    },
    If {
        source_location: Location,
        // The condition is a variable that is true or false
        condition: VariableIndex,
        // If condition == true
        then_block: BlockIndex,
        // If condition == false
        else_block: BlockIndex,
    },
}

impl PartialEq for Termination {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Termination::Return(l1), Termination::Return(l2)) => l1 == l2,
            (Termination::Jump { to: t1, .. }, Termination::Jump { to: t2, .. }) => t1 == t2,
            (
                Termination::If {
                    condition: c1,
                    then_block: tb1,
                    else_block: eb1,
                    ..
                },
                Termination::If {
                    condition: c2,
                    then_block: tb2,
                    else_block: eb2,
                    ..
                },
            ) => c1 == c2 && tb1 == tb2 && eb1 == eb2,
            _ => false,
        }
    }
}

#[derive(Debug, PartialEq)]
pub struct Block {
    pub instructions: Vec<IrWithLocation>,
    pub terminator: Termination,
}

/// The compiler expects all globals to have been evaluated, but the IR compiler can't do that.
pub enum GlobalInitializer {
    Value(Value),
    Expression(Vec<IrWithLocation>),
}

pub struct GlobalVariableInfo {
    pub initializer: ast::Expression<VmTypeSet>,
    pub ty: Type,
}

impl GlobalVariableInfo {
    fn print(&self, _program: &Program) -> String {
        format!("global {}", self.ty)
    }
}

/// An index into the global variables in a program. This uniquely identifies a global, but
/// it is not directly related to a variable's address.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GlobalVariableIndex(pub usize);
impl GlobalVariableIndex {
    fn name<'s>(&self, program: &'s Program) -> &'s str {
        let (name, _) = program
            .globals
            .get_index(self.0)
            .expect("Global variable index out of bounds");
        program.strings.lookup(*name)
    }
}

pub struct Program {
    pub globals: IndexMap<StringIndex, GlobalVariableInfo>,
    pub functions: IndexMap<StringIndex, Function>,
    pub strings: StringInterner,
    pub structs: StructRegistry,
}

/// Resolves a type annotation to an IR type, recognizing struct names via the registry.
fn resolve_type<'s>(
    source: &'s str,
    type_token: &ast::TypeHint,
    strings: &StringInterner,
    structs: &StructRegistry,
) -> Result<Type, CompileError<'s>> {
    let name = type_token.type_name.source(source);
    Ok(match name {
        "int" => Type::Int,
        "signed" => Type::SignedInt,
        "float" => Type::Float,
        "bool" => Type::Bool,
        "string" => Type::String,
        "iter" => Type::Iter,
        other => {
            if let Some(id) = strings.find(other).and_then(|idx| structs.id_of(idx)) {
                Type::Struct(id)
            } else {
                return Err(CompileError {
                    source,
                    location: type_token.type_name.location,
                    error: format!("Unknown type `{other}`").into_boxed_str(),
                });
            }
        }
    })
}

impl Program {
    pub fn print(&self) -> String {
        let mut output = String::new();

        writeln!(&mut output, "Globals:").unwrap();
        for (name, global) in &self.globals {
            let name = self.strings.lookup(*name);
            writeln!(&mut output, "  {name} = {}", global.print(self)).unwrap();
        }

        writeln!(&mut output).unwrap();

        for (_, function) in &self.functions {
            output.push_str(&function.print(self));
            writeln!(&mut output).unwrap();
        }

        output
    }

    pub(crate) fn compile<'s>(
        source: &'s str,
        ast: &ast::Program<VmTypeSet>,
    ) -> Result<Program, CompileError<'s>> {
        let mut strings = StringInterner::new();
        let mut functions = IndexMap::new();
        let mut globals = IndexMap::new();

        // Build struct layouts first, so that struct type names resolve everywhere else.
        let structs = build_struct_registry(source, ast, &mut strings)?;

        // Declare items first
        for item in &ast.items {
            match item {
                ast::Item::Struct(_) => {}
                ast::Item::GlobalVariable(global_variable) => {
                    let name = global_variable.identifier.source(source);
                    let name_idx = strings.intern(name);
                    if globals.contains_key(&name_idx) {
                        return Err(CompileError {
                            source,
                            location: global_variable.identifier.location,
                            error: format!("Global variable `{name}` is already defined")
                                .into_boxed_str(),
                        });
                    }

                    let ty = resolve_type(source, &global_variable.type_token, &strings, &structs)?;
                    if let Type::Struct(_) = ty {
                        return Err(CompileError {
                            source,
                            location: global_variable.type_token.type_name.location,
                            error: "Struct-typed global variables are not supported by the VM"
                                .to_string()
                                .into_boxed_str(),
                        });
                    }

                    globals.insert(
                        name_idx,
                        GlobalVariableInfo {
                            ty,
                            initializer: global_variable.initializer.clone(),
                        },
                    );
                }
                ast::Item::ExternFunction(function) => {
                    let name = function.name.source(source);
                    let name_idx = strings.intern(name);
                    if functions.contains_key(&name_idx) {
                        return Err(CompileError {
                            source,
                            location: function.name.location,
                            error: format!("Function `{name}` is already defined").into_boxed_str(),
                        });
                    }
                    let func =
                        Function::compile_external(source, function, &mut strings, &structs)?;
                    functions.insert(name_idx, func);
                }
                ast::Item::Function(function) => {
                    let name = function.name.source(source);
                    let name_idx = strings.intern(name);
                    if functions.contains_key(&name_idx) {
                        return Err(CompileError {
                            source,
                            location: function.name.location,
                            error: format!("Function `{name}` is already defined").into_boxed_str(),
                        });
                    }
                    functions.insert(name_idx, Function::DUMMY);
                }
            }
        }

        for item in &ast.items {
            if let ast::Item::Function(function) = item {
                let func = Function::compile(source, function, &mut strings, &globals, &structs)?;
                functions.insert(func.name, func);
            }
        }

        Ok(Program {
            globals,
            functions,
            strings,
            structs,
        })
    }
}

/// Builds the [`StructRegistry`] from a program's struct definitions, computing
/// packed layouts and detecting recursive structs.
fn build_struct_registry<'s>(
    source: &'s str,
    ast: &ast::Program<VmTypeSet>,
    strings: &mut StringInterner,
) -> Result<StructRegistry, CompileError<'s>> {
    let mut defs: Vec<&ast::StructDef> = Vec::new();
    let mut ids: IndexMap<StringIndex, StructId> = IndexMap::new();

    for item in &ast.items {
        if let ast::Item::Struct(def) = item {
            let name = def.name.source(source);
            let name_idx = strings.intern(name);
            if ids.contains_key(&name_idx) {
                return Err(CompileError {
                    source,
                    location: def.name.location,
                    error: format!("Struct `{name}` is already defined").into_boxed_str(),
                });
            }
            ids.insert(name_idx, StructId(defs.len()));
            defs.push(def);
        }
    }

    let mut computed: Vec<Option<StructLayout>> = vec![None; defs.len()];
    let mut in_progress = vec![false; defs.len()];
    for i in 0..defs.len() {
        build_struct_layout(
            i,
            &defs,
            source,
            strings,
            &ids,
            &mut computed,
            &mut in_progress,
        )?;
    }

    Ok(StructRegistry {
        layouts: computed.into_iter().map(Option::unwrap).collect(),
        ids,
    })
}

#[allow(clippy::too_many_arguments)]
fn build_struct_layout<'s>(
    idx: usize,
    defs: &[&ast::StructDef],
    source: &'s str,
    strings: &mut StringInterner,
    ids: &IndexMap<StringIndex, StructId>,
    computed: &mut Vec<Option<StructLayout>>,
    in_progress: &mut Vec<bool>,
) -> Result<(), CompileError<'s>> {
    if computed[idx].is_some() {
        return Ok(());
    }
    let def = defs[idx];
    if in_progress[idx] {
        return Err(CompileError {
            source,
            location: def.name.location,
            error: format!("Struct `{}` is recursive", def.name.source(source)).into_boxed_str(),
        });
    }
    in_progress[idx] = true;

    let struct_name_idx = strings.intern(def.name.source(source));
    let mut fields = Vec::new();
    let mut offset = 0;
    let mut seen: std::collections::HashSet<&str> = std::collections::HashSet::new();
    for field in &def.fields {
        let field_name = field.name.source(source);
        if !seen.insert(field_name) {
            return Err(CompileError {
                source,
                location: field.name.location,
                error: format!("Duplicate field `{field_name}`").into_boxed_str(),
            });
        }

        let type_name = field.field_type.type_name.source(source);
        let field_ty = match type_name {
            "int" => Type::Int,
            "signed" => Type::SignedInt,
            "float" => Type::Float,
            "bool" => Type::Bool,
            "string" => Type::String,
            "iter" | "void" => {
                return Err(CompileError {
                    source,
                    location: field.field_type.type_name.location,
                    error: format!("Struct fields cannot be of type `{type_name}`")
                        .into_boxed_str(),
                });
            }
            other => {
                if let Some(dep) = strings.find(other).and_then(|i| ids.get(&i).copied()) {
                    build_struct_layout(dep.0, defs, source, strings, ids, computed, in_progress)?;
                    Type::Struct(dep)
                } else {
                    return Err(CompileError {
                        source,
                        location: field.field_type.type_name.location,
                        error: format!("Unknown type `{other}`").into_boxed_str(),
                    });
                }
            }
        };

        let size = match field_ty {
            Type::Struct(dep) => computed[dep.0].as_ref().unwrap().size,
            other => somni_expr::Type::from(other).vm_size_of(),
        };
        let field_name_idx = strings.intern(field_name);
        fields.push(StructFieldLayout {
            name: field_name_idx,
            ty: field_ty,
            offset,
        });
        offset += size;
    }

    computed[idx] = Some(StructLayout {
        name: struct_name_idx,
        fields,
        size: offset,
    });
    in_progress[idx] = false;
    Ok(())
}

pub struct Function {
    pub name: StringIndex,
    pub arguments: Vec<Variable>,
    pub return_type: Type,
    pub implementation: Option<FunctionImplementation>,
}

pub struct FunctionImplementation {
    pub variables: ScopeData,
    pub blocks: Vec<Block>,
}

impl Function {
    const DUMMY: Self = Function {
        name: StringIndex::dummy(),
        return_type: Type::Void,
        arguments: vec![],
        implementation: None,
    };

    pub fn print(&self, program: &Program) -> String {
        let mut output = String::new();
        writeln!(
            &mut output,
            "Function: {}",
            program.strings.lookup(self.name)
        )
        .unwrap();
        if let Some(implementation) = self.implementation.as_ref() {
            for (idx, block) in implementation.blocks.iter().enumerate() {
                if idx > 0 {
                    writeln!(&mut output).unwrap();
                }

                writeln!(&mut output, "{:?}", BlockIndex(idx)).unwrap();
                for instruction in &block.instructions {
                    writeln!(&mut output, "  {}", instruction.print(program, self)).unwrap();
                }

                match block.terminator {
                    Termination::Return(_) => writeln!(&mut output, "  -> return").unwrap(),
                    Termination::Jump { to, .. } => writeln!(&mut output, "  -> {to:?}").unwrap(),
                    Termination::If {
                        condition,
                        then_block,
                        else_block,
                        ..
                    } => {
                        let condition = condition.name(program, self);
                        writeln!(
                            &mut output,
                            "  if {condition} -> {then_block:?} else {else_block:?}",
                        )
                        .unwrap()
                    }
                }
            }
        } else {
            writeln!(&mut output, "  External function").unwrap();
        }

        output
    }

    pub fn compile<'s>(
        source: &'s str,
        func: &ast::Function<VmTypeSet>,
        strings: &mut StringInterner,
        globals: &IndexMap<StringIndex, GlobalVariableInfo>,
        structs: &StructRegistry,
    ) -> Result<Function, CompileError<'s>> {
        let mut this = FunctionCompiler {
            source,
            structs,
            blocks: Blocks {
                blocks: vec![
                    // Enter function
                    Block {
                        instructions: vec![],
                        terminator: Termination::Jump {
                            source_location: func.name.location,
                            to: BlockIndex::RETURN_BLOCK,
                        }, // Jump to the first block
                    },
                    // Return from function
                    Block {
                        instructions: vec![],
                        terminator: Termination::Return(func.closing_paren.location),
                    },
                ],
                current_block: BlockIndex(0),
            },
            variables: VariableTracker::new(),
            strings,
            loop_stack: Vec::new(),
            globals,
        };

        Self::validate_arg_names(source, &func.arguments)?;

        // Allocate a return variable, if the function has a return type.
        let (return_type, return_token) = if let Some(return_type) = &func.return_decl {
            (
                resolve_type(source, &return_type.return_type, this.strings, this.structs)?,
                return_type.return_type.type_name,
            )
        } else {
            (Type::Void, func.fn_token)
        };

        let rp = this.variables.create_restore_point();
        this.declare_variable(
            "return_value",
            Some(Variable::Value(return_type)),
            return_token.location,
            true,
        )?;

        // allocate variables for arguments
        let mut arguments = vec![];
        for argument in func.arguments.iter() {
            let ty = resolve_type(source, &argument.arg_type, this.strings, this.structs)?;
            let ty = if argument.reference_token.is_some() {
                Variable::Reference(1, ty)
            } else {
                Variable::Value(ty)
            };
            let name = argument.name.source(source);
            this.declare_variable(name, Some(ty), argument.name.location, true)?;
            arguments.push(ty);
        }

        this.compile_body(&func.body, VariableIndex::RETURN_VALUE)?;

        this.rollback_scope(func.closing_paren.location, rp);

        Ok(Function {
            name: this.strings.intern(func.name.source(source)),
            return_type,
            arguments,
            implementation: Some(FunctionImplementation {
                variables: this.variables.finalize(),
                blocks: this.blocks.blocks,
            }),
        })
    }

    pub fn compile_external<'s>(
        source: &'s str,
        func: &ast::ExternalFunction,
        strings: &mut StringInterner,
        structs: &StructRegistry,
    ) -> Result<Function, CompileError<'s>> {
        Self::validate_arg_names(source, &func.arguments)?;

        // Allocate a return variable, if the function has a return type.
        let return_type = if let Some(return_type) = &func.return_decl {
            resolve_type(source, &return_type.return_type, strings, structs)?
        } else {
            Type::Void
        };

        // allocate variables for arguments
        let mut arguments = vec![];
        for argument in func.arguments.iter() {
            let ty = resolve_type(source, &argument.arg_type, strings, structs)?;
            let ty = if argument.reference_token.is_some() {
                Variable::Reference(1, ty)
            } else {
                Variable::Value(ty)
            };
            arguments.push(ty);
        }

        Ok(Function {
            name: strings.intern(func.name.source(source)),
            return_type,
            arguments,
            implementation: None,
        })
    }

    fn validate_arg_names<'s>(
        source: &'s str,
        arguments: &[FunctionArgument],
    ) -> Result<(), CompileError<'s>> {
        // Check that argument names are unique.
        let arg_names = arguments.iter().map(|a| a.name.source(source));
        let mut seen_names = std::collections::HashMap::new();
        const RESERVED_NAMES: &[&str] = &["return_value"];
        for (idx, name) in arg_names.enumerate() {
            if seen_names.insert(name, idx).is_some() {
                return Err(CompileError {
                    source,
                    location: arguments[idx].name.location,
                    error: format!("Duplicate argument name `{name}`").into_boxed_str(),
                });
            }
            if RESERVED_NAMES.contains(&name) {
                return Err(CompileError {
                    source,
                    location: arguments[idx].name.location,
                    error: format!("Argument name `{name}` is reserved").into_boxed_str(),
                });
            }
        }

        Ok(())
    }
}

struct Loop {
    restore_point: RestorePoint,
    body_block: BlockIndex,
    next_block: BlockIndex,
}

struct Blocks {
    blocks: Vec<Block>,
    current_block: BlockIndex,
}

impl Blocks {
    /// Allocates a new block in the function's block list.
    ///
    /// Blocks may be allocated in any order. Codegen will walk through the blocks in order,
    /// and generate code for each block, in order. Codegen will make sure that jumps are
    /// resolved to the correct block.
    fn allocate_block(&mut self, next: BlockIndex) -> BlockIndex {
        let index = BlockIndex(self.blocks.len());
        self.blocks.push(Block {
            instructions: Vec::new(),
            terminator: Termination::Jump {
                source_location: Location::dummy(),
                to: next,
            },
        });
        index
    }

    fn select_block(&mut self, index: BlockIndex) {
        if index.0 >= self.blocks.len() {
            panic!("Block index out of bounds: {}", index.0);
        }
        self.current_block = index;
    }

    fn push_instruction(&mut self, source_location: Location, instruction: Ir) {
        self.blocks[self.current_block.0]
            .instructions
            .push(IrWithLocation {
                instruction,
                source_location,
            });
    }

    fn set_terminator(&mut self, terminator: Termination) {
        self.blocks[self.current_block.0].terminator = terminator;
    }

    fn current(&self) -> BlockIndex {
        self.current_block
    }
}

/// Compiles a function into an intermediate representation (IR).
///
/// The IR is consumed by the bytecode compiler, and the debuginfo compiler.
struct FunctionCompiler<'s, 'c> {
    source: &'s str,
    strings: &'c mut StringInterner,
    globals: &'c IndexMap<StringIndex, GlobalVariableInfo>,
    structs: &'c StructRegistry,

    blocks: Blocks,
    variables: VariableTracker,
    loop_stack: Vec<Loop>,
}

impl<'s> FunctionCompiler<'s, '_> {
    fn declare_variable(
        &mut self,
        name: &str,
        var_ty: Option<Variable>,
        source_location: Location,
        is_argument: bool,
    ) -> Result<LocalVariableIndex, CompileError<'s>> {
        let name_index = self.strings.intern(name);
        let variable = self.variables.declare_variable(name_index, var_ty);
        self.blocks.push_instruction(
            source_location,
            Ir::Declare(
                VariableDeclaration {
                    index: variable,
                    name: name_index,
                    allocation_method: AllocationMethod::FirstFit,
                    is_argument,
                },
                None,
            ),
        );
        Ok(variable)
    }

    /// Declares a fresh temporary variable and emits its `Declare` instruction.
    /// The other `declare_*_temporary` helpers are thin wrappers over this.
    fn declare_temporary_inner(
        &mut self,
        location: Location,
        var_type: Option<Variable>,
        allocation_method: AllocationMethod,
        init_value: Option<Value>,
    ) -> VariableIndex {
        let temp_name = self
            .strings
            .intern(&format!("temp{}", self.variables.len()));
        let temp = self.variables.declare_variable(temp_name, var_type);
        self.blocks.push_instruction(
            location,
            Ir::Declare(
                VariableDeclaration {
                    index: temp,
                    name: temp_name,
                    allocation_method,
                    is_argument: false,
                },
                init_value,
            ),
        );

        VariableIndex::Temporary(temp)
    }

    fn declare_temporary(
        &mut self,
        location: Location,
        init_value: Value,
        allocation_method: AllocationMethod,
    ) -> VariableIndex {
        let init_value = (init_value != Value::Void).then_some(init_value);
        self.declare_temporary_inner(
            location,
            init_value.map(|v| Variable::Value(v.type_of())),
            allocation_method,
            init_value,
        )
    }

    fn declare_typed_temporary(
        &mut self,
        location: Location,
        ty: Type,
        allocation_method: AllocationMethod,
    ) -> VariableIndex {
        self.declare_temporary_inner(location, Some(Variable::Value(ty)), allocation_method, None)
    }

    /// Declares a temporary holding a reference (a pointer) to a value of type
    /// `pointee`. Used to materialize the address of a field or place.
    fn declare_ref_temporary(&mut self, location: Location, pointee: Type) -> VariableIndex {
        self.declare_temporary_inner(
            location,
            Some(Variable::Reference(1, pointee)),
            AllocationMethod::FirstFit,
            None,
        )
    }

    /// Returns the declared type of a variable, if it is known at IR-build time.
    fn var_type(&self, var: VariableIndex) -> Option<Variable> {
        match var {
            VariableIndex::Local(idx) | VariableIndex::Temporary(idx) => {
                self.variables.type_of(idx)
            }
            VariableIndex::Global(GlobalVariableIndex(idx)) => self
                .globals
                .get_index(idx)
                .map(|(_, info)| Variable::Value(info.ty)),
        }
    }

    /// Resolves a struct field access on `base`: returns the field's byte offset,
    /// its type, and whether `base` is a reference (so the access is indirect).
    ///
    /// The base's type must be statically known at IR-build time; if it isn't
    /// (e.g. an un-annotated variable initialized from a function call), a helpful
    /// error is returned.
    fn struct_field_info(
        &self,
        base: VariableIndex,
        field: &Token,
    ) -> Result<(usize, Type, bool), CompileError<'s>> {
        let field_name = field.source(self.source);
        let base_ty = self.var_type(base).ok_or_else(|| CompileError {
            source: self.source,
            location: field.location,
            error: format!(
                "Cannot determine the struct type for field access `.{field_name}`; \
                 add a type annotation"
            )
            .into_boxed_str(),
        })?;

        let (indirect, id) = match base_ty {
            Variable::Value(Type::Struct(id)) => (false, id),
            Variable::Reference(1, Type::Struct(id)) => (true, id),
            other => {
                return Err(CompileError {
                    source: self.source,
                    location: field.location,
                    error: format!(
                        "Cannot access field `{field_name}` of non-struct type `{other}`"
                    )
                    .into_boxed_str(),
                });
            }
        };

        let layout_field = self
            .strings
            .find(field_name)
            .and_then(|idx| self.structs.field(id, idx));
        let Some(layout_field) = layout_field else {
            let struct_name = self.strings.lookup(self.structs.layout(id).name);
            return Err(CompileError {
                source: self.source,
                location: field.location,
                error: format!("Struct `{struct_name}` has no field `{field_name}`")
                    .into_boxed_str(),
            });
        };

        Ok((layout_field.offset, layout_field.ty, indirect))
    }

    /// Looks up a declared variable by its identifier token, erroring if unknown.
    fn lookup_variable(&mut self, token: &Token) -> Result<VariableIndex, CompileError<'s>> {
        let name = token.source(self.source);
        self.find_variable_by_name(name)
            .ok_or_else(|| CompileError {
                source: self.source,
                location: token.location,
                error: format!("Variable `{name}` is not declared").into_boxed_str(),
            })
    }

    /// Materializes a reference to `base.field` by emitting an `AddressOfField`
    /// into a fresh reference temporary, and returns that temporary.
    fn emit_address_of_field(
        &mut self,
        base_var: VariableIndex,
        field: &Token,
    ) -> Result<VariableIndex, CompileError<'s>> {
        let (offset, field_ty, indirect) = self.struct_field_info(base_var, field)?;
        let dst = self.declare_ref_temporary(field.location, field_ty);
        self.blocks.push_instruction(
            field.location,
            Ir::AddressOfField {
                dst,
                base: base_var,
                offset,
                indirect,
            },
        );
        Ok(dst)
    }

    fn compile_statement(
        &mut self,
        statement: &ast::Statement<VmTypeSet>,
        storage: VariableIndex,
    ) -> Result<(), CompileError<'s>> {
        match statement {
            ast::Statement::VariableDefinition(variable_def) => {
                self.compile_variable_definition(variable_def)?;
            }
            ast::Statement::Return(ret_with_value) => {
                self.compile_return_with_value(ret_with_value)?;
            }
            ast::Statement::EmptyReturn(ret) => self.compile_empty_return(ret)?,
            ast::Statement::If(if_statement) => {
                let expect_void = self.declare_typed_temporary(
                    statement.location(),
                    Type::Void,
                    AllocationMethod::FirstFit,
                );
                self.compile_if_statement(if_statement, expect_void)?
            }
            ast::Statement::Loop(loop_statement) => {
                let expect_void = self.declare_typed_temporary(
                    statement.location(),
                    Type::Void,
                    AllocationMethod::FirstFit,
                );
                self.compile_loop_statement(loop_statement, expect_void)?
            }
            ast::Statement::For(for_statement) => {
                let expect_void = self.declare_typed_temporary(
                    statement.location(),
                    Type::Void,
                    AllocationMethod::FirstFit,
                );
                self.compile_for_statement(for_statement, expect_void)?
            }
            ast::Statement::Break(break_statement) => {
                self.compile_break_statement(break_statement)?
            }
            ast::Statement::Continue(continue_statement) => {
                self.compile_continue_statement(continue_statement)?;
            }
            ast::Statement::Scope(body) => return self.compile_free_scope(body, storage),
            ast::Statement::Expression {
                expression,
                semicolon,
            } => self.compile_expression_statement(expression, semicolon)?,
            ast::Statement::ImplicitReturn(expression) => {
                self.compile_implicit_return(expression, storage)?
            }
        }

        Ok(())
    }

    fn compile_variable_definition(
        &mut self,
        variable_def: &ast::VariableDefinition<VmTypeSet>,
    ) -> Result<(), CompileError<'s>> {
        let ident = variable_def.identifier;
        let ty = variable_def.type_token.as_ref();
        let initializer = &variable_def.initializer;

        let name = ident.source(self.source);
        let var_ty = if let Some(type_token) = ty {
            Some(Variable::Value(resolve_type(
                self.source,
                type_token,
                self.strings,
                self.structs,
            )?))
        } else {
            None
        };
        let had_annotation = var_ty.is_some();
        let variable = self.declare_variable(name, var_ty, ident.location, false)?;

        let rp = self.variables.create_restore_point();
        let expr_result = self.compile_right_hand_expression(initializer)?;
        // Struct-typed locals need a concrete type at IR-build time so that later
        // field accesses can resolve their layout. When there's no annotation,
        // adopt the initializer's (build-time-known) struct type.
        if !had_annotation {
            if let Some(
                ty @ (Variable::Value(Type::Struct(_)) | Variable::Reference(_, Type::Struct(_))),
            ) = self.var_type(expr_result)
            {
                self.variables.set_type(variable, ty);
            }
        }
        self.blocks.push_instruction(
            ident.location,
            Ir::Assign(VariableIndex::Local(variable), expr_result),
        );
        self.rollback_scope(initializer.location(), rp);

        Ok(())
    }

    fn compile_if_statement(
        &mut self,
        if_statement: &ast::If<VmTypeSet>,
        if_return_value: VariableIndex,
    ) -> Result<(), CompileError<'s>> {
        let rp = self.variables.create_restore_point();

        // Generate instructions for the condition, allocate then/else blocks and generate conditional jumps
        let condition = self.compile_right_hand_expression(&if_statement.condition)?;

        // Allocate the next block, which is the block that will be executed after the branches.
        // Point to the return block by default.
        let next_block = self.blocks.allocate_block(BlockIndex::RETURN_BLOCK);

        // Save the current block so that we can update it later.
        let condition_block = self.blocks.current();

        // Compile the branches
        let then_block = self.blocks.allocate_block(BlockIndex::RETURN_BLOCK);
        self.blocks.select_block(then_block);
        self.compile_body(&if_statement.body, if_return_value)?;
        self.blocks.set_terminator(Termination::Jump {
            source_location: if_statement.body.closing_brace.location,
            to: next_block,
        });

        let else_block = if let Some(else_branch) = if_statement.else_branch.as_ref() {
            let else_block = self.blocks.allocate_block(BlockIndex::RETURN_BLOCK);
            self.blocks.select_block(else_block);
            self.compile_body(&else_branch.else_body, if_return_value)?;
            self.blocks.set_terminator(Termination::Jump {
                source_location: else_branch.else_body.closing_brace.location,
                to: next_block,
            });
            else_block
        } else {
            // If there is no else branch, we can just jump to the next block.
            next_block
        };

        // Update the condition block's terminator to be a conditional jump.
        self.blocks.select_block(condition_block);
        self.blocks.set_terminator(Termination::If {
            source_location: if_statement.if_token.location,
            condition,
            then_block,
            else_block,
        });

        // Whatever happens, normal execution continues at the next block.
        self.blocks.select_block(next_block);

        self.rollback_scope(if_statement.if_token.location, rp);

        Ok(())
    }

    fn compile_body(
        &mut self,
        body: &ast::Body<VmTypeSet>,
        storage: VariableIndex,
    ) -> Result<(), CompileError<'s>> {
        let restore_point = self.variables.create_restore_point();

        let expect_void =
            self.declare_typed_temporary(body.location(), Type::Void, AllocationMethod::FirstFit);

        let last_idx = body.statements.len().saturating_sub(1);
        for (statement, is_last) in body
            .statements
            .iter()
            .enumerate()
            .map(|(i, stmt)| (stmt, i == last_idx))
        {
            self.compile_statement(statement, if is_last { storage } else { expect_void })?;
        }

        self.rollback_scope(body.closing_brace.location, restore_point);

        Ok(())
    }

    fn compile_loop_statement(
        &mut self,
        loop_statement: &ast::Loop<VmTypeSet>,
        loop_return_value: VariableIndex,
    ) -> Result<(), CompileError<'s>> {
        let next_block = self.blocks.allocate_block(BlockIndex::RETURN_BLOCK);

        // Compile the body of the loop.
        let body_block = self.blocks.allocate_block(next_block);

        // Set up the loop structure.
        self.blocks.set_terminator(Termination::Jump {
            source_location: loop_statement.loop_token.location,
            to: body_block,
        });
        self.loop_stack.push(Loop {
            restore_point: self.variables.create_restore_point(),
            body_block,
            next_block,
        });

        self.blocks.select_block(body_block);
        self.compile_body(&loop_statement.body, loop_return_value)?;
        self.blocks.set_terminator(Termination::Jump {
            source_location: loop_statement.body.closing_brace.location,
            to: body_block,
        });

        self.blocks.select_block(next_block);

        Ok(())
    }

    fn compile_for_statement(
        &mut self,
        for_statement: &ast::For<VmTypeSet>,
        for_return_value: VariableIndex,
    ) -> Result<(), CompileError<'s>> {
        // The loop variable's type comes from the optional annotation. When it is
        // omitted, the variable is left untyped and its type is inferred from how
        // it is used in the loop body (like an untyped variable definition).
        let elem_ty = match &for_statement.var_type {
            Some(var_type) => Some(Variable::Value(resolve_type(
                self.source,
                var_type,
                self.strings,
                self.structs,
            )?)),
            None => None,
        };

        let rp = self.variables.create_restore_point();

        // Obtain the iterator. This temporary persists for the whole loop and is
        // freed when the loop's scope is rolled back.
        let iter_var = self.compile_right_hand_expression(&for_statement.iterable)?;

        // Condition block: re-checked on every iteration (and on `continue`).
        let cond_block = self.blocks.allocate_block(BlockIndex::RETURN_BLOCK);
        self.blocks.set_terminator(Termination::Jump {
            source_location: for_statement.for_token.location,
            to: cond_block,
        });
        self.blocks.select_block(cond_block);

        let has_next = self.declare_typed_temporary(
            for_statement.for_token.location,
            Type::Bool,
            AllocationMethod::FirstFit,
        );
        // Attribute the iterator check to the iterable expression so that a
        // "not an iterator" type error points at the offending operand.
        self.blocks.push_instruction(
            for_statement.iterable.location(),
            Ir::IterHasNext(has_next, iter_var),
        );

        let next_block = self.blocks.allocate_block(BlockIndex::RETURN_BLOCK);
        let body_block = self.blocks.allocate_block(BlockIndex::RETURN_BLOCK);

        self.blocks.set_terminator(Termination::If {
            source_location: for_statement.for_token.location,
            condition: has_next,
            then_block: body_block,
            else_block: next_block,
        });

        // Restore point that captures the state at the start of a body iteration
        // (iterator and `has_next` allocated, loop variable not yet). `break` and
        // `continue` roll back to here so per-iteration variables are freed.
        let body_rp = self.variables.create_restore_point();

        // `break` -> next_block, `continue` -> cond_block (re-check the iterator).
        self.loop_stack.push(Loop {
            restore_point: body_rp,
            body_block: cond_block,
            next_block,
        });

        self.blocks.select_block(body_block);

        // Bind the loop variable to the next value.
        let loop_var = self.declare_variable(
            for_statement.variable.source(self.source),
            elem_ty,
            for_statement.variable.location,
            false,
        )?;
        self.blocks.push_instruction(
            for_statement.variable.location,
            Ir::IterNext(VariableIndex::Local(loop_var), iter_var),
        );

        self.compile_body(&for_statement.body, for_return_value)?;

        // Free the loop variable (and any body temporaries) at the end of each
        // iteration, before looping back to the condition.
        self.rollback_scope(for_statement.body.closing_brace.location, body_rp);
        self.blocks.set_terminator(Termination::Jump {
            source_location: for_statement.body.closing_brace.location,
            to: cond_block,
        });

        self.loop_stack.pop();

        // After the loop, free the iterator and the `has_next` temporary.
        self.blocks.select_block(next_block);
        self.rollback_scope(for_statement.body.closing_brace.location, rp);

        Ok(())
    }

    fn compile_break_statement(
        &mut self,
        break_statement: &ast::Break,
    ) -> Result<(), CompileError<'s>> {
        let Some(loop_entry) = self.loop_stack.last() else {
            return Err(CompileError {
                source: self.source,
                location: break_statement.break_token.location,
                error: "Cannot break outside of a loop."
                    .to_string()
                    .into_boxed_str(),
            });
        };

        self.compile_cleanup_block(
            loop_entry.restore_point,
            loop_entry.next_block,
            break_statement.break_token.location,
        )?;

        Ok(())
    }

    fn compile_continue_statement(
        &mut self,
        continue_statement: &ast::Continue,
    ) -> Result<(), CompileError<'s>> {
        let Some(loop_entry) = self.loop_stack.last() else {
            return Err(CompileError {
                source: self.source,
                location: continue_statement.continue_token.location,
                error: "Cannot continue outside of a loop."
                    .to_string()
                    .into_boxed_str(),
            });
        };

        // The cleanup block jumps to the next block after the loop.
        self.compile_cleanup_block(
            loop_entry.restore_point,
            loop_entry.body_block,
            continue_statement.continue_token.location,
        )?;

        Ok(())
    }

    fn compile_return_with_value(
        &mut self,
        ret: &ast::ReturnWithValue<VmTypeSet>,
    ) -> Result<(), CompileError<'s>> {
        let rp = self.variables.create_restore_point();
        let return_value = self.compile_right_hand_expression(&ret.expression)?;

        // Store variable in the return variable.
        self.blocks.push_instruction(
            ret.location(),
            Ir::Assign(VariableIndex::RETURN_VALUE, return_value),
        );

        self.rollback_scope(ret.semicolon.location, rp);

        self.compile_return(ret.return_token)
    }

    fn compile_implicit_return(
        &mut self,
        expression: &ast::RightHandExpression<VmTypeSet>,
        storage: VariableIndex,
    ) -> Result<(), CompileError<'s>> {
        let return_value = self.compile_right_hand_expression(expression)?;

        // Store variable in the return variable.
        // TODO: each block should have a return temporary, and we should store there.
        self.blocks
            .push_instruction(expression.location(), Ir::Assign(storage, return_value));

        Ok(())
    }

    fn compile_empty_return(&mut self, ret: &ast::EmptyReturn) -> Result<(), CompileError<'s>> {
        self.compile_return(ret.return_token)
    }

    fn compile_return(&mut self, ret: Token) -> Result<(), CompileError<'s>> {
        // The cleanup block jumps to the next block after the loop.
        self.compile_cleanup_block(
            RestorePoint::RETURN_FROM_FN,
            BlockIndex::RETURN_BLOCK,
            ret.location,
        )?;

        Ok(())
    }

    fn compile_cleanup_block(
        &mut self,
        restore_point: RestorePoint,
        next_block: BlockIndex,
        source_location: Location,
    ) -> Result<(), CompileError<'s>> {
        let cleanup_block = self.blocks.allocate_block(next_block);

        self.blocks.set_terminator(Termination::Jump {
            source_location,
            to: cleanup_block,
        });

        self.rollback_scope(source_location, restore_point);

        // The rest of the body is unreachable, so we allocate a new block for the next statements.
        // Nothing will actually jump to this block, but we want to handle compilation errors.
        let next_block = self.blocks.allocate_block(BlockIndex::RETURN_BLOCK);
        self.blocks.select_block(next_block);

        Ok(())
    }

    fn compile_expression_statement(
        &mut self,
        expression: &ast::Expression<VmTypeSet>,
        semicolon: &Token,
    ) -> Result<(), CompileError<'s>> {
        let rp = self.variables.create_restore_point();

        // Compile the expression, which may be a variable assignment, function call,
        // or other expression. Discard the result.
        self.compile_expression(expression)?;

        self.rollback_scope(semicolon.location, rp);

        Ok(())
    }

    fn compile_expression(
        &mut self,
        expression: &ast::Expression<VmTypeSet>,
    ) -> Result<VariableIndex, CompileError<'s>> {
        match &expression {
            ast::Expression::Assignment {
                left_expr,
                operator,
                right_expr,
            } => {
                let rp = self.variables.create_restore_point();

                // Compile the right operand.
                let right = self.compile_right_hand_expression(right_expr)?;

                match left_expr {
                    ast::LeftHandExpression::Deref { operator: _, name } => {
                        let left = self.compile_right_hand_expression(
                            &ast::RightHandExpression::Variable { variable: *name },
                        )?;
                        self.blocks
                            .push_instruction(operator.location, Ir::DerefAssign(left, right));
                    }
                    ast::LeftHandExpression::Name { variable } => {
                        let left = self.compile_right_hand_expression(
                            &ast::RightHandExpression::Variable {
                                variable: *variable,
                            },
                        )?;
                        self.blocks
                            .push_instruction(operator.location, Ir::Assign(left, right));
                    }
                    ast::LeftHandExpression::Field { base, field, .. } => {
                        // Write into a struct field: address the containing place, then
                        // store the field bytes at its offset.
                        let base_var = self.compile_place_base_lhs(base)?;
                        let (offset, field_ty, indirect) =
                            self.struct_field_info(base_var, field)?;
                        let size = self.structs.size_of(field_ty);
                        self.blocks.push_instruction(
                            operator.location,
                            Ir::StoreField {
                                base: base_var,
                                offset,
                                size,
                                indirect,
                                src: right,
                            },
                        );
                    }
                }

                self.rollback_scope(operator.location, rp);

                // Allocate a temporary variable. For assignments this is not used, but
                // we still need to return something.
                Ok(self.declare_temporary(
                    operator.location,
                    Value::Void,
                    AllocationMethod::FirstFit,
                ))
            }
            ast::Expression::Expression { expression } => {
                self.compile_right_hand_expression(expression)
            }
        }
    }

    fn compile_right_hand_expression(
        &mut self,
        expression: &ast::RightHandExpression<VmTypeSet>,
    ) -> Result<VariableIndex, CompileError<'s>> {
        match &expression {
            ast::RightHandExpression::Literal { value } => {
                let val_type = match value.value {
                    ast::LiteralValue::Integer(_) => Type::MaybeSignedInt,
                    ast::LiteralValue::Float(_) => Type::Float,
                    ast::LiteralValue::Boolean(_) => Type::Bool,
                    ast::LiteralValue::String(_) => Type::String,
                };

                let literal_value = self.compile_literal_value(val_type, value)?;
                let temp = self.declare_temporary(
                    value.location,
                    literal_value,
                    AllocationMethod::FirstFit,
                );

                Ok(temp)
            }
            ast::RightHandExpression::Variable { variable } => self.lookup_variable(variable),
            ast::RightHandExpression::UnaryOperator { name, operand } => {
                self.compile_unary_operator(*name, operand)
            }
            ast::RightHandExpression::BinaryOperator { name, operands } => {
                self.compile_binary_operator(*name, operands)
            }
            ast::RightHandExpression::FunctionCall { name, arguments } => {
                self.compile_function_call(*name, arguments)
            }
            ast::RightHandExpression::FieldAccess { base, field, .. } => {
                self.compile_field_access(base, field)
            }
            ast::RightHandExpression::StructLiteral { name, fields, .. } => {
                self.compile_struct_literal(*name, fields)
            }
        }
    }

    /// Reads a struct field: `base.field`. The base is auto-dereferenced when it
    /// is a reference.
    fn compile_field_access(
        &mut self,
        base: &ast::RightHandExpression<VmTypeSet>,
        field: &Token,
    ) -> Result<VariableIndex, CompileError<'s>> {
        let base_var = self.compile_right_hand_expression(base)?;
        let (offset, field_ty, indirect) = self.struct_field_info(base_var, field)?;
        let size = self.structs.size_of(field_ty);
        let dst =
            self.declare_typed_temporary(field.location, field_ty, AllocationMethod::FirstFit);
        self.blocks.push_instruction(
            field.location,
            Ir::LoadField {
                dst,
                base: base_var,
                offset,
                size,
                indirect,
            },
        );
        self.free_if_temporary(base.location(), base_var);
        Ok(dst)
    }

    /// Constructs a struct value from a literal `Name { field: expr, ... }`,
    /// validating field names against the struct's layout and storing each field
    /// at its offset.
    fn compile_struct_literal(
        &mut self,
        name: Token,
        fields: &[ast::StructLiteralField<VmTypeSet>],
    ) -> Result<VariableIndex, CompileError<'s>> {
        let struct_name = name.source(self.source);
        let id = self
            .strings
            .find(struct_name)
            .and_then(|idx| self.structs.id_of(idx))
            .ok_or_else(|| CompileError {
                source: self.source,
                location: name.location,
                error: format!("Unknown struct `{struct_name}`").into_boxed_str(),
            })?;

        let layout = self.structs.layout(id).clone();
        let s = self.declare_typed_temporary(
            name.location,
            Type::Struct(id),
            AllocationMethod::FirstFit,
        );

        let mut provided = vec![false; layout.fields.len()];
        for field in fields {
            let field_name = field.name.source(self.source);
            let pos = self
                .strings
                .find(field_name)
                .and_then(|fi| layout.fields.iter().position(|lf| lf.name == fi));
            let Some(pos) = pos else {
                return Err(CompileError {
                    source: self.source,
                    location: field.name.location,
                    error: format!("Struct `{struct_name}` has no field `{field_name}`")
                        .into_boxed_str(),
                });
            };
            if provided[pos] {
                return Err(CompileError {
                    source: self.source,
                    location: field.name.location,
                    error: format!("Duplicate field `{field_name}` in struct `{struct_name}`")
                        .into_boxed_str(),
                });
            }
            provided[pos] = true;

            let rp = self.variables.create_restore_point();
            let value = self.compile_right_hand_expression(&field.value)?;
            let layout_field = &layout.fields[pos];
            let size = self.structs.size_of(layout_field.ty);
            self.blocks.push_instruction(
                field.name.location,
                Ir::StoreField {
                    base: s,
                    offset: layout_field.offset,
                    size,
                    indirect: false,
                    src: value,
                },
            );
            self.rollback_scope(field.value.location(), rp);
        }

        if let Some(missing) = provided.iter().position(|p| !p) {
            let field_name = self.strings.lookup(layout.fields[missing].name);
            return Err(CompileError {
                source: self.source,
                location: name.location,
                error: format!("Missing field `{field_name}` in struct `{struct_name}`")
                    .into_boxed_str(),
            });
        }

        Ok(s)
    }

    /// Resolves the base of an assignment's field path to a variable usable as a
    /// `StoreField`/`AddressOfField` base: a struct value variable, a reference
    /// variable, or a freshly materialized reference to a sub-place.
    fn compile_place_base_lhs(
        &mut self,
        lhs: &ast::LeftHandExpression,
    ) -> Result<VariableIndex, CompileError<'s>> {
        match lhs {
            ast::LeftHandExpression::Name { variable: token }
            | ast::LeftHandExpression::Deref { name: token, .. } => self.lookup_variable(token),
            ast::LeftHandExpression::Field { base, field, .. } => {
                let base_var = self.compile_place_base_lhs(base)?;
                self.emit_address_of_field(base_var, field)
            }
        }
    }

    /// Resolves the base of an address-of field path (`&base.field`) to a variable
    /// usable as an `AddressOfField` base.
    fn compile_place_base_rhs(
        &mut self,
        expr: &ast::RightHandExpression<VmTypeSet>,
    ) -> Result<VariableIndex, CompileError<'s>> {
        match expr {
            ast::RightHandExpression::Variable { variable } => self.lookup_variable(variable),
            ast::RightHandExpression::FieldAccess { base, field, .. } => {
                let base_var = self.compile_place_base_rhs(base)?;
                self.emit_address_of_field(base_var, field)
            }
            ast::RightHandExpression::UnaryOperator { name, operand }
                if name.source(self.source) == "*" =>
            {
                self.compile_right_hand_expression(operand)
            }
            other => Err(CompileError {
                source: self.source,
                location: other.location(),
                error: "Cannot take the address of this expression".into(),
            }),
        }
    }

    fn compile_literal_value(
        &mut self,
        literal_type: Type,
        literal: &ast::Literal<VmTypeSet>,
    ) -> Result<Value, CompileError<'s>> {
        let value = match (literal_type, &literal.value) {
            (Type::Int, ast::LiteralValue::Integer(value)) => Value::Int(*value),
            (Type::MaybeSignedInt, ast::LiteralValue::Integer(value)) => {
                Value::MaybeSignedInt(*value)
            }
            (Type::Float, ast::LiteralValue::Float(value)) => Value::Float(*value),
            (Type::Bool, ast::LiteralValue::Boolean(value)) => Value::Bool(*value),
            (Type::String, ast::LiteralValue::String(value)) => {
                let string_index = self.strings.intern(value);
                Value::String(string_index)
            }
            (Type::Void, _) => {
                return Err(CompileError {
                    source: self.source,
                    location: literal.location,
                    error: format!("{literal_type} is not supported in literals").into_boxed_str(),
                });
            }
            _ => {
                return Err(CompileError {
                    source: self.source,
                    location: literal.location,
                    error: format!(
                        "Expected {literal_type}, found {} literal",
                        match literal.value {
                            ast::LiteralValue::Integer(_) => "integer",
                            ast::LiteralValue::Float(_) => "float",
                            ast::LiteralValue::Boolean(_) => "boolean",
                            ast::LiteralValue::String(_) => "string",
                        }
                    )
                    .into_boxed_str(),
                });
            }
        };

        Ok(value)
    }

    fn compile_unary_operator(
        &mut self,
        operator: Token,
        operand: &ast::RightHandExpression<VmTypeSet>,
    ) -> Result<VariableIndex, CompileError<'s>> {
        let op = operator.source(self.source);

        // Taking the address of a struct field: `&base.field`. Resolve the field's
        // place directly, rather than taking the address of a temporary copy.
        if op == "&" && matches!(operand, ast::RightHandExpression::FieldAccess { .. }) {
            return self.compile_place_base_rhs(operand);
        }

        // Allocate a temporary variable for the result.
        let temp =
            self.declare_temporary(operator.location, Value::Void, AllocationMethod::FirstFit);

        let rp = self.variables.create_restore_point();

        // Compile the left and right operands.
        let operand_result = self.compile_right_hand_expression(operand)?;

        if let VariableIndex::Local(local_idx) = operand_result {
            if op == "&" {
                self.variables.reference_variable(local_idx);
            }
        }

        // Generate the instruction for the binary operator.
        self.blocks.push_instruction(
            operator.location,
            Ir::UnaryOperator(self.strings.intern(op), temp, operand_result),
        );
        self.rollback_scope(operator.location, rp);

        Ok(temp)
    }

    fn compile_binary_operator(
        &mut self,
        operator: Token,
        operands: &[ast::RightHandExpression<VmTypeSet>; 2],
    ) -> Result<VariableIndex, CompileError<'s>> {
        let op = operator.source(self.source);

        if op == "&&" {
            let temp = self.declare_temporary(
                operator.location,
                Value::Bool(false),
                AllocationMethod::FirstFit,
            );

            let condition = self.compile_right_hand_expression(&operands[0])?;

            // Allocate the next block, which is the block that will be executed after the branches.
            // Point to the return block by default.
            let next_block = self.blocks.allocate_block(BlockIndex::RETURN_BLOCK);

            // Save the current block so that we can update it later.
            let condition_block = self.blocks.current();

            // Compile the then branch
            let then_block = self.blocks.allocate_block(BlockIndex::RETURN_BLOCK);
            self.blocks.select_block(then_block);
            let rp = self.variables.create_restore_point();
            let rhs = self.compile_right_hand_expression(&operands[1])?;
            self.blocks
                .push_instruction(operator.location, Ir::Assign(temp, rhs));
            self.free_if_temporary(operands[1].location(), rhs);

            self.rollback_scope(operator.location, rp);

            self.blocks.set_terminator(Termination::Jump {
                source_location: operator.location,
                to: next_block,
            });

            // Update the condition block's terminator to be a conditional jump.
            self.blocks.select_block(condition_block);
            self.blocks.set_terminator(Termination::If {
                source_location: operator.location,
                condition,
                then_block,
                else_block: next_block,
            });

            // Whatever happens, normal execution continues at the next block.
            self.blocks.select_block(next_block);
            self.free_if_temporary(operands[0].location(), condition);

            Ok(temp)
        } else if op == "||" {
            let temp = self.declare_temporary(
                operator.location,
                Value::Bool(true),
                AllocationMethod::FirstFit,
            );

            let condition = self.compile_right_hand_expression(&operands[0])?;

            // Allocate the next block, which is the block that will be executed after the branches.
            // Point to the return block by default.
            let next_block = self.blocks.allocate_block(BlockIndex::RETURN_BLOCK);

            // Save the current block so that we can update it later.
            let condition_block = self.blocks.current();

            // Compile the else branch
            let else_block = self.blocks.allocate_block(BlockIndex::RETURN_BLOCK);
            self.blocks.select_block(else_block);
            let rp = self.variables.create_restore_point();
            let rhs = self.compile_right_hand_expression(&operands[1])?;
            self.blocks
                .push_instruction(operator.location, Ir::Assign(temp, rhs));
            self.free_if_temporary(operands[1].location(), rhs);

            self.rollback_scope(operator.location, rp);

            self.blocks.set_terminator(Termination::Jump {
                source_location: operator.location,
                to: next_block,
            });

            // Update the condition block's terminator to be a conditional jump.
            self.blocks.select_block(condition_block);
            self.blocks.set_terminator(Termination::If {
                source_location: operator.location,
                condition,
                then_block: next_block,
                else_block,
            });

            // Whatever happens, normal execution continues at the next block.
            self.blocks.select_block(next_block);
            self.free_if_temporary(operands[0].location(), condition);

            Ok(temp)
        } else {
            // Compile the left and right operands.
            let left = self.compile_right_hand_expression(&operands[0])?;
            let right = self.compile_right_hand_expression(&operands[1])?;

            // Allocate a temporary variable for the result.
            let temp =
                self.declare_temporary(operator.location, Value::Void, AllocationMethod::FirstFit);
            // Generate the instruction for the binary operator.
            self.blocks.push_instruction(
                operator.location,
                Ir::BinaryOperator(self.strings.intern(op), temp, left, right),
            );
            self.free_if_temporary(operands[0].location(), left);
            self.free_if_temporary(operands[1].location(), right);

            Ok(temp)
        }
    }

    fn compile_function_call(
        &mut self,
        name: Token,
        arguments: &[ast::RightHandExpression<VmTypeSet>],
    ) -> Result<VariableIndex, CompileError<'s>> {
        // Allocate a temporary variable for the result.
        let temp = self.declare_temporary(name.location, Value::Void, AllocationMethod::FirstFit);
        let rp = self.variables.create_restore_point();

        // Compile the arguments.
        let arguments = arguments
            .iter()
            .map(|arg| {
                self.compile_right_hand_expression(arg)
                    .map(|val| (val, arg.location()))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let function_name = name.source(self.source);
        let function_index = self.strings.intern(function_name);

        // Generate the instruction for the function call. Here we allocate the stack frame, copy
        // arguments into it, then generate the call. The two temporaries above the stack pointer are
        // used by the call/return instructions. This way the stack pointer
        // still points at the return value.
        // TODO: would be better to declare signature as a structure so that we don't have to
        // treat call-related allocations differently.
        let _pc_temporary =
            self.declare_typed_temporary(name.location, Type::Int, AllocationMethod::TopOfStack);
        let _sp_temporary =
            self.declare_typed_temporary(name.location, Type::Int, AllocationMethod::TopOfStack);
        let retval_temporary =
            self.declare_temporary(name.location, Value::Void, AllocationMethod::TopOfStack);
        let arg_temporaries = arguments
            .iter()
            .map(|(arg, location)| {
                let arg_temp =
                    self.declare_temporary(*location, Value::Void, AllocationMethod::TopOfStack);
                self.blocks
                    .push_instruction(*location, Ir::Assign(arg_temp, *arg));
                arg_temp
            })
            .collect::<Vec<_>>();
        self.blocks.push_instruction(
            name.location,
            Ir::Call(function_index, retval_temporary, arg_temporaries),
        );
        // Copy the return value into the temporary variable.
        self.blocks
            .push_instruction(name.location, Ir::Assign(temp, retval_temporary));

        // Free up the temporaries except for the return value.
        self.rollback_scope(name.location, rp);

        Ok(temp)
    }

    fn find_variable_by_name(&mut self, name: &str) -> Option<VariableIndex> {
        let name = self.strings.intern(name);
        if let Some(local) = self.variables.find(name) {
            // If the variable is already declared, we can just return it.
            Some(VariableIndex::Local(local))
        } else {
            self.globals
                .get_index_of(&name)
                .map(|index| VariableIndex::Global(GlobalVariableIndex(index)))
        }
    }

    fn free_if_temporary(&mut self, location: Location, right: VariableIndex) {
        if let VariableIndex::Temporary(operand) = right {
            self.variables.free_variable(operand);
            self.blocks
                .push_instruction(location, Ir::FreeVariable(operand));
        }
    }

    fn rollback_scope(&mut self, location: Location, rp: RestorePoint) {
        for var in self.variables.rollback_to_restore_point(rp) {
            self.blocks
                .push_instruction(location, Ir::FreeVariable(var));
        }
    }

    fn compile_free_scope(
        &mut self,
        body: &ast::Body<VmTypeSet>,
        storage: VariableIndex,
    ) -> Result<(), CompileError<'s>> {
        let next_block = self.blocks.allocate_block(BlockIndex::RETURN_BLOCK);

        // Compile the body of the scope.
        let body_block = self.blocks.allocate_block(next_block);

        self.blocks.set_terminator(Termination::Jump {
            source_location: body.opening_brace.location,
            to: body_block,
        });
        self.blocks.select_block(body_block);
        self.compile_body(body, storage)?;

        self.blocks.set_terminator(Termination::Jump {
            source_location: body.closing_brace.location,
            to: next_block,
        });

        self.blocks.select_block(next_block);
        Ok(())
    }
}
