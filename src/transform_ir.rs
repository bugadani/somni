use std::collections::{HashMap, HashSet};

use indexmap::IndexMap;

use crate::{
    error::CompileError,
    ir::{self, BlockIndex, VariableIndex},
    string_interner::StringIndex,
    variable_tracker::{LocalVariableIndex, ScopeData},
};
use somni_lexer::Location;

#[derive(Debug, Clone, Copy, PartialEq)]
enum ConstraintKind {
    Equals {
        left: VariableIndex,
        right: VariableIndex,
    },
    Requires {
        variable: VariableIndex,
        required_type: ir::Variable,
    },
    Reference {
        left: VariableIndex,
        right: VariableIndex,
    },
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct Constraint {
    source_location: Location,
    kind: ConstraintKind,
}

struct FunctionSignature {
    return_type: ir::Variable,
    arguments: Vec<ir::Variable>,
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
        ty: ir::Variable,
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

    fn reference(
        &mut self,
        left: VariableIndex,
        right: VariableIndex,
        source_location: Location,
    ) -> Result<(), CompileError<'s>> {
        self.constraints.push(Constraint {
            source_location,
            kind: ConstraintKind::Reference { left, right },
        });
        self.step_resolve()
    }

    fn current_type_of(&self, variable: VariableIndex) -> Option<ir::Variable> {
        match variable {
            VariableIndex::Local(local) | VariableIndex::Temporary(local) => {
                self.locals.variable(local).unwrap().ty
            }
            VariableIndex::Global(global) => {
                let Some((_name, global_info)) = self.globals.get_index(global.0) else {
                    unreachable!();
                };

                Some(ir::Variable::Value(global_info.ty))
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
                        (Some(left_ty), Some(right_ty)) => {
                            if left_ty != right_ty {
                                return Err(CompileError {
                                    source: self.source,
                                    location: constraint.source_location,
                                    error: format!(
                                        "Type mismatch: expected {left_ty}, found {right_ty}",
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
                    required_type: required_ty,
                } => {
                    match variable {
                        VariableIndex::Local(local) | VariableIndex::Temporary(local) => {
                            let local_info = self.locals.variable_mut(local).unwrap();
                            match local_info.ty.as_mut() {
                                Some(ty) => {
                                    if *ty != required_ty {
                                        return Err(CompileError {
                                            source: self.source,
                                            location: constraint.source_location,
                                            error: format!(
                                                "Type mismatch: expected {required_ty}, found {}",
                                                *ty
                                            ),
                                        });
                                    }
                                    *ty = required_ty;
                                }
                                None => {
                                    // If the type is not set, we can set it now.
                                    local_info.ty = Some(required_ty);
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

                ConstraintKind::Reference { left, right } => {
                    let left_type = match left {
                        VariableIndex::Global(_) => {
                            return Err(CompileError {
                                source: self.source,
                                location: constraint.source_location,
                                error: format!("Globals cannot be references"),
                            });
                        }
                        _ => self.current_type_of(left),
                    };
                    let right_type = self.current_type_of(right);

                    let made_progress = match (left_type, right_type) {
                        (Some(left_ty), Some(right_ty)) => {
                            if left_ty.reference() != right_ty {
                                return Err(CompileError {
                                    source: self.source,
                                    location: constraint.source_location,
                                    error: format!(
                                        "Type mismatch: expected {left_ty}, found {right_ty}"
                                    ),
                                });
                            }

                            true
                        }
                        (Some(ty), None) => {
                            let ty = ty.dereference().ok_or_else(|| CompileError {
                                source: self.source,
                                location: constraint.source_location,
                                error: format!("Cannot dereference {ty}"),
                            })?;
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
                                .ty = Some(ty.reference());

                            true
                        }
                        (None, None) => false,
                    };

                    if made_progress {
                        // If we made progress, we can remove this constraint.
                        self.constraints.swap_remove(i);
                    } else {
                        // If we didn't make progress, just move to the next constraint.
                        i += 1;
                    }
                }
            }
        }

        Ok(())
    }
}

pub fn transform_ir<'s>(source: &'s str, ir: &mut ir::Program) -> Result<(), CompileError<'s>> {
    for func in ir.functions.values_mut() {
        bypass_redundant_jump_blocks(func);
        merge_identical_blocks(func);
        remove_unreachable_blocks(func);
    }

    propagate_variable_types(source, ir)?;

    // Remove unnecessary assignments (assignments without reads after them).
    for func in ir.functions.values_mut() {
        // For now we only optimize variables that are declared AND freed in the same block.
        // This is way too conservative, but relaxing it would require a figuring out loops.
        let mut relevant_variables_in_block = HashMap::new();
        for (block_idx, block) in func.blocks.iter().enumerate() {
            let mut declared = HashSet::new();
            let mut freed = HashSet::new();
            for instruction in &block.instructions {
                match instruction.instruction {
                    ir::Ir::Declare(var, _) => {
                        // We also don't allow optimizing TopOfStack variables for now.
                        if var.allocation_method == ir::AllocationMethod::FirstFit {
                            declared.insert(var.index);
                        }
                    }
                    ir::Ir::FreeVariable(var) => {
                        freed.insert(var);
                    }
                    _ => {}
                }
            }
            relevant_variables_in_block.insert(
                block_idx,
                declared
                    .intersection(&freed)
                    .copied()
                    .collect::<HashSet<_>>(),
            );
        }

        propagate_destination(func, &relevant_variables_in_block);
    }
    // Remove unused variables.

    Ok(())
}

/// This function removes blocks that only contain a jump to another block,
/// effectively bypassing them. The bypassed blocks will be removed by `remove_unreachable_blocks`.
fn bypass_redundant_jump_blocks(func: &mut ir::Function) {
    // We will keep track of replacements for blocks that are bypassed.
    let mut replacements = HashMap::new();

    for (index, block) in func.blocks.iter().enumerate() {
        if let ir::Termination::Jump { to, .. } = block.terminator {
            // If the block only contains a jump, we can bypass it.
            if block.instructions.is_empty() {
                // If the target block is already replaced, we use that replacement.
                let new_target = replacements.get(&to).copied().unwrap_or(to);
                // If this block is a target of another replacement, update that replacement.
                if let Some(to_update) =
                    replacements.values_mut().find(|v| **v == BlockIndex(index))
                {
                    *to_update = new_target;
                }

                replacements.insert(BlockIndex(index), new_target);
            }
        }
    }

    let remap = |idx: &BlockIndex| replacements.get(idx).copied().unwrap_or(*idx);

    remap_block_idxs(func, remap);
}

fn remap_block_idxs<F>(func: &mut ir::Function, remap: F)
where
    F: Fn(&BlockIndex) -> BlockIndex,
{
    for block in &mut func.blocks {
        match &mut block.terminator {
            ir::Termination::Jump { to, .. } => *to = remap(to),
            ir::Termination::If {
                then_block,
                else_block,
                ..
            } => {
                *then_block = remap(then_block);
                *else_block = remap(else_block);
            }
            ir::Termination::Return { .. } => {}
        }
    }
}

fn remove_unreachable_blocks(func: &mut ir::Function) {
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

    remap_block_idxs(func, remap);

    // Remove unreachable blocks.
    let mut current = 0;
    func.blocks.retain(|_| {
        let retain = reachable[current];
        current += 1;
        retain
    });
}

fn merge_identical_blocks(func: &mut ir::Function) {
    let mut identical_to = (0..).take(func.blocks.len()).collect::<Vec<_>>();

    for (idx, block) in func.blocks.iter().enumerate() {
        // Check if this block is identical to any previous block.
        for (prev_idx, prev_block) in func.blocks.iter().enumerate().take(idx) {
            if block == prev_block {
                // Mark this block as identical to the previous one.
                identical_to[idx] = prev_idx;
                break;
            }
        }
    }

    let remap = |idx: &BlockIndex| {
        // If the block is identical to another, use the first occurrence.
        BlockIndex(identical_to[idx.0])
    };

    remap_block_idxs(func, remap);

    // remove_unreachable_blocks will trim the blocks
}

fn propagate_variable_types<'s>(
    source: &'s str,
    ir: &mut ir::Program,
) -> Result<(), CompileError<'s>> {
    let mut func_signatures = HashMap::new();

    // Collect function signatures first
    for (name, function) in &ir.functions {
        let signature = FunctionSignature {
            return_type: ir::Variable::Value(function.return_type),
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
                            ir::Variable::Value(init_value.type_of()),
                            instruction.source_location,
                        )?;
                    }
                }
                ir::Ir::Assign(dst, src) => {
                    resolver.equate(*dst, *src, instruction.source_location)?;
                }
                ir::Ir::DerefAssign(dst, src) => {
                    resolver.reference(*src, *dst, instruction.source_location)?;
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
                            resolver.require(
                                *dst,
                                ir::Variable::Value(ir::Type::Bool),
                                instruction.source_location,
                            )?;
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
                            resolver.reference(*src, *dst, instruction.source_location)?;
                        }
                        "&" => {
                            resolver.reference(*dst, *src, instruction.source_location)?;
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
            resolver.require(
                condition,
                ir::Variable::Value(ir::Type::Bool),
                source_location,
            )?;
        }
    }

    let mut constraints = resolver.constraints.len();
    while constraints > 0 {
        resolver.step_resolve()?;
        if resolver.constraints.len() == constraints {
            // No more progress can be made.
            break;
        }
        constraints = resolver.constraints.len();
    }

    Ok(())
}

fn propagate_destination(
    func: &mut ir::Function,
    variables_in_block: &HashMap<usize, HashSet<LocalVariableIndex>>,
) {
    for (block_idx, block) in func.blocks.iter_mut().enumerate() {
        let variables_in_block = &variables_in_block[&block_idx];
        for idx in (1..block.instructions.len()).rev() {
            if let ir::Ir::Assign(dst, src) = block.instructions[idx].instruction {
                // If src is not defined in this block, we can't avoid writing to it.
                if let Some(src) = src.local_index() {
                    // If the source is not in the block, we can skip it.
                    if !variables_in_block.contains(&src) {
                        continue;
                    }
                } else {
                    continue;
                }

                // If we find an assignment, let's try to remove it. We can remove it if we find the
                // src written previously in the block.
                for prev_idx in (0..idx).rev() {
                    let read = match &block.instructions[prev_idx].instruction {
                        ir::Ir::Declare(info, _) => Some(info.index) == dst.local_index(),
                        ir::Ir::Assign(_, assign_source) => *assign_source == src,
                        ir::Ir::Call(_, _, args) => args.contains(&src),
                        ir::Ir::BinaryOperator(_, _, lhs, rhs) => *lhs == src || *rhs == src,
                        ir::Ir::UnaryOperator(_, _, operand) => *operand == src,
                        _ => false,
                    };
                    if read {
                        break;
                    }
                    let written = match &mut block.instructions[prev_idx].instruction {
                        ir::Ir::Assign(dst, _) => dst,
                        ir::Ir::Call(_, dst, _) => dst,
                        ir::Ir::BinaryOperator(_, dst, _, _) => dst,
                        ir::Ir::UnaryOperator(_, dst, _) => dst,
                        _ => continue,
                    };

                    if *written == src {
                        *written = dst; // Replace the written variable with the destination.
                        block.instructions.remove(idx); // Remove the assignment.
                        break;
                    }
                }
            }
        }
    }
}
