use std::collections::HashMap;

use indexmap::IndexMap;

use crate::{
    error::CompileError,
    ir::{self, BlockIndex, GlobalInitializer, Type, VariableIndex},
    lexer::Location,
    string_interner::StringIndex,
    variable_tracker::ScopeData,
};

#[derive(Debug, Clone, Copy, PartialEq)]
enum ConstraintKind {
    Equals {
        left: VariableIndex,
        right: VariableIndex,
    },
    Requires {
        variable: VariableIndex,
        required_type: ir::Type,
    },
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct Constraint {
    source_location: Location,
    kind: ConstraintKind,
}

struct FunctionSignature {
    return_type: ir::Type,
    arguments: Vec<ir::Type>,
}

struct TypeResolver<'a, 's> {
    source: &'s str,
    constraints: Vec<Constraint>,
    globals: &'a IndexMap<StringIndex, ir::GlobalVariableInfo>,
    locals: &'a mut ScopeData,
}

impl<'a, 's> TypeResolver<'a, 's> {
    fn new(
        source: &'s str,
        globals: &'a IndexMap<StringIndex, ir::GlobalVariableInfo>,
        locals: &'a mut ScopeData,
    ) -> Self {
        Self {
            source,
            constraints: vec![],
            globals,
            locals,
        }
    }

    /// Require that a variable has a specific type at a given source location.
    fn require(
        &mut self,
        left: VariableIndex,
        ty: ir::Type,
        source_location: Location,
    ) -> Result<(), CompileError<'s>> {
        self.constraints.push(Constraint {
            source_location,
            kind: ConstraintKind::Requires {
                variable: left,
                required_type: ty,
            },
        });
        self.step_resolve()
    }

    /// Add a constraint that two variables must be equal, recording the source location where
    /// this requirement was made.
    fn equate(
        &mut self,
        left: VariableIndex,
        right: VariableIndex,
        source_location: Location,
    ) -> Result<(), CompileError<'s>> {
        self.constraints.push(Constraint {
            source_location,
            kind: ConstraintKind::Equals { left, right },
        });
        self.step_resolve()
    }

    fn current_type_of(&self, variable: VariableIndex) -> Option<ir::Type> {
        match variable {
            VariableIndex::Local(local) | VariableIndex::Temporary(local) => {
                self.locals.variable(local).unwrap().ty
            }
            VariableIndex::Global(global) => {
                let Some((_name, global_info)) = self.globals.get_index(global.0) else {
                    unreachable!();
                };
                match &global_info.initial_value {
                    GlobalInitializer::Value(value) => Some(value.type_of()),
                    GlobalInitializer::Expression(_) => unreachable!(),
                }
            }
        }
    }

    fn step_resolve(&mut self) -> Result<(), CompileError<'s>> {
        let mut i = 0;
        while i < self.constraints.len() {
            let constraint = &self.constraints[i];
            match constraint.kind {
                ConstraintKind::Equals { left, right } => {
                    let left_type = self.current_type_of(left);
                    let right_type = self.current_type_of(right);

                    let made_progress = match (left_type, right_type) {
                        (Some(left_type), Some(right_type)) => {
                            if left_type != right_type {
                                return Err(CompileError {
                                    source: self.source,
                                    location: constraint.source_location,
                                    error: format!(
                                        "Type mismatch: expected {:?}, found {:?}",
                                        left_type, right_type
                                    ),
                                });
                            }
                            true
                        }
                        (Some(ty), None) => {
                            self.locals
                                .variable_mut(right.local_index().unwrap())
                                .unwrap()
                                .ty = Some(ty);
                            true
                        }
                        (None, Some(ty)) => {
                            self.locals
                                .variable_mut(left.local_index().unwrap())
                                .unwrap()
                                .ty = Some(ty);
                            true
                        }
                        (None, None) => {
                            // Cannot make progress with this constraint yet.
                            false
                        }
                    };

                    if made_progress {
                        // If we made progress, we can remove this constraint.
                        self.constraints.swap_remove(i);
                    } else {
                        // If we didn't make progress, just move to the next constraint.
                        i += 1;
                    }
                }
                ConstraintKind::Requires {
                    variable,
                    required_type,
                } => {
                    match variable {
                        VariableIndex::Local(local) | VariableIndex::Temporary(local) => {
                            let local_info = self.locals.variable_mut(local).unwrap();
                            match local_info.ty.as_mut() {
                                Some(ty) => {
                                    if *ty != required_type {
                                        return Err(CompileError {
                                            source: self.source,
                                            location: constraint.source_location,
                                            error: format!(
                                                "Type mismatch: expected {:?}, found {:?}",
                                                required_type, *ty
                                            ),
                                        });
                                    }
                                    *ty = required_type;
                                }
                                None => {
                                    // If the type is not set, we can set it now.
                                    local_info.ty = Some(required_type);
                                }
                            }
                        }
                        VariableIndex::Global(global) => {
                            let Some((_name, _global_info)) = self.globals.get_index(global.0)
                            else {
                                unreachable!();
                            };

                            // TODO
                        }
                    }

                    self.constraints.swap_remove(i);
                }
            }
        }

        Ok(())
    }
}

pub fn transform_ir<'s>(source: &'s str, ir: &mut ir::Program) -> Result<(), CompileError<'s>> {
    // TODO: Jump threading
    // TODO: merge identical blocks
    remove_unreachable_blocks(ir);
    propagate_variable_types(source, ir)?;

    // Remove unnecessary assignments (assignments without reads after them).
    // Remove unused variables.

    Ok(())
}

fn remove_unreachable_blocks(ir: &mut ir::Program) {
    for func in ir.functions.values_mut() {
        let mut reachable = vec![false; func.blocks.len()];
        let mut stack = vec![BlockIndex(0)]; // Start with the first block.

        while let Some(BlockIndex(block_index)) = stack.pop() {
            if reachable[block_index] {
                continue; // Already visited this block.
            }
            reachable[block_index] = true;

            let block = &func.blocks[block_index];

            match block.terminator {
                ir::Termination::Return { .. } => {}
                ir::Termination::Jump { to, .. } => {
                    stack.push(to);
                }
                ir::Termination::If {
                    then_block,
                    else_block,
                    ..
                } => {
                    stack.push(then_block);
                    stack.push(else_block);
                }
            }
        }

        // Renumber reachable blocks.
        let remap = |idx: &BlockIndex| {
            let new_idx = reachable.iter().take(idx.0 + 1).filter(|&&r| r).count() - 1;

            BlockIndex(new_idx)
        };

        for block in &mut func.blocks {
            match &mut block.terminator {
                ir::Termination::Jump { to, .. } => {
                    *to = remap(to);
                }
                ir::Termination::If {
                    then_block,
                    else_block,
                    ..
                } => {
                    *then_block = remap(then_block);
                    *else_block = remap(else_block);
                }
                ir::Termination::Return(..) => {}
            }
        }

        // Remove unreachable blocks.
        let mut current = 0;
        func.blocks.retain(|_| {
            let retain = reachable[current];
            current += 1;
            retain
        });
    }
}

fn propagate_variable_types<'s>(
    source: &'s str,
    ir: &mut ir::Program,
) -> Result<(), CompileError<'s>> {
    let mut func_signatures = HashMap::new();

    // Collect function signatures first
    for (name, function) in &ir.functions {
        let signature = FunctionSignature {
            return_type: function.return_type,
            arguments: function.arguments.clone(),
        };
        func_signatures.insert(*name, signature);
    }

    for (name, function) in &mut ir.functions {
        propagate_variable_types_inner(
            source,
            *name,
            function,
            &ir.strings,
            &ir.globals,
            &func_signatures,
        )?;
    }

    Ok(())
}

fn propagate_variable_types_inner<'s>(
    source: &'s str,
    _name: StringIndex,
    function: &mut ir::Function,
    strings: &crate::string_interner::Strings,
    globals: &IndexMap<StringIndex, ir::GlobalVariableInfo>,
    signatures: &HashMap<StringIndex, FunctionSignature>,
) -> Result<(), CompileError<'s>> {
    let mut resolver = TypeResolver::new(source, globals, &mut function.variables);

    for block in &mut function.blocks {
        for instruction in &block.instructions {
            match &instruction.instruction {
                ir::Ir::Declare(dst, init_value) => {
                    if let Some(init_value) = init_value {
                        // If there's an initial value, we can infer the type from it.
                        resolver.require(
                            VariableIndex::Local(dst.index),
                            init_value.type_of(),
                            instruction.source_location,
                        )?;
                    }
                }
                ir::Ir::Assign(dst, src) => {
                    resolver.equate(*dst, *src, instruction.source_location)?;
                }
                ir::Ir::DerefAssign(dst, _src) => {
                    // TODO: we likely want to encode the type of the dereferenced value
                    resolver.require(*dst, Type::Address, instruction.source_location)?;
                }
                ir::Ir::Call(func, return_value, args) => {
                    if let Some(signature) = signatures.get(func) {
                        // If the function has a signature, we can require the return type.
                        // TODO: require external functions to have a signature.
                        resolver.require(
                            *return_value,
                            signature.return_type,
                            instruction.source_location,
                        )?;

                        if args.len() != signature.arguments.len() {
                            return Err(CompileError {
                                source,
                                location: instruction.source_location,
                                error: format!(
                                    "Function {} expected {} arguments, found {}",
                                    strings.lookup(*func),
                                    signature.arguments.len(),
                                    args.len()
                                ),
                            });
                        }

                        for (arg, arg_type) in args.iter().zip(signature.arguments.iter()) {
                            // TODO: point at variable in error message.
                            resolver.require(*arg, *arg_type, instruction.source_location)?;
                        }
                    }
                }
                ir::Ir::BinaryOperator(op, dst, lhs, rhs) => {
                    match strings.lookup(*op) {
                        "<=" | "<" | ">=" | ">" | "==" | "!=" => {
                            // For comparison operators, the result is a boolean.
                            resolver.require(*dst, ir::Type::Bool, instruction.source_location)?;
                            // The left and right hand sides must be of the same type.
                            resolver.equate(*lhs, *rhs, instruction.source_location)?;
                        }
                        _ => {
                            // For now, all other operators require the
                            // three variables to be of the same type.
                            resolver.equate(*dst, *lhs, instruction.source_location)?;
                            resolver.equate(*dst, *rhs, instruction.source_location)?;
                        }
                    }
                }
                ir::Ir::UnaryOperator(op, dst, src) => {
                    match strings.lookup(*op) {
                        "*" => {
                            resolver.require(
                                *src,
                                ir::Type::Address,
                                instruction.source_location,
                            )?;
                        }
                        "&" => {
                            resolver.require(
                                *dst,
                                ir::Type::Address,
                                instruction.source_location,
                            )?;
                        }
                        _ => {
                            // For now, the destination and source must be of the same type.
                            resolver.equate(*dst, *src, instruction.source_location)?;
                        }
                    }
                }
                ir::Ir::FreeVariable(_) => {}
            }
        }

        if let ir::Termination::If {
            source_location,
            condition,
            ..
        } = block.terminator
        {
            // Condition needs to be a boolean
            resolver.require(condition, ir::Type::Bool, source_location)?;
        }
    }

    let constraints = resolver.constraints.len();
    loop {
        resolver.step_resolve()?;
        if resolver.constraints.len() == constraints {
            // No more progress can be made.
            break;
        }
    }

    Ok(())
}
