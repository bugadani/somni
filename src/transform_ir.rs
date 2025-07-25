use std::collections::{HashMap, HashSet};

use indexmap::IndexMap;

use crate::{
    error::CompileError,
    ir::{self, BlockIndex, VariableIndex},
    string_interner::{self, StringIndex},
    variable_tracker::{LocalVariableIndex, ScopeData},
};
use somni_parser::Location;

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
        self.step_resolve()?;

        Ok(())
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
        self.step_resolve()?;

        Ok(())
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
        self.step_resolve()?;

        Ok(())
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

    fn step_resolve(&mut self) -> Result<bool, CompileError<'s>> {
        let mut i = 0;
        let mut any_changed = false;
        while i < self.constraints.len() {
            let constraint = &self.constraints[i];
            match constraint.kind {
                ConstraintKind::Equals { left, right } => {
                    let left_type = self.current_type_of(left);
                    let right_type = self.current_type_of(right);

                    let made_progress = match (left_type, right_type) {
                        (Some(left_ty), Some(right_ty)) if left_ty == right_ty => {
                            !left_ty.maybe_signed_integer() && !right_ty.maybe_signed_integer()
                        }
                        (Some(left_ty), Some(right_ty)) => {
                            if left_ty.maybe_signed_integer() {
                                self.locals
                                    .variable_mut(left.local_index().unwrap())
                                    .unwrap()
                                    .ty = Some(right_ty);
                                any_changed = true;
                            } else if right_ty.maybe_signed_integer() {
                                self.locals
                                    .variable_mut(right.local_index().unwrap())
                                    .unwrap()
                                    .ty = Some(left_ty);
                                any_changed = true;
                            } else {
                                return Err(CompileError {
                                    source: self.source,
                                    location: constraint.source_location,
                                    error: format!(
                                        "Type mismatch: expected {left_ty}, found {right_ty}",
                                    ),
                                });
                            }

                            !left_ty.maybe_signed_integer() && !right_ty.maybe_signed_integer()
                        }
                        (Some(ty), None) => {
                            self.locals
                                .variable_mut(right.local_index().unwrap())
                                .unwrap()
                                .ty = Some(ty);
                            any_changed = true;
                            !ty.maybe_signed_integer()
                        }
                        (None, Some(ty)) => {
                            self.locals
                                .variable_mut(left.local_index().unwrap())
                                .unwrap()
                                .ty = Some(ty);
                            any_changed = true;
                            !ty.maybe_signed_integer()
                        }
                        (None, None) => {
                            // Cannot make progress with this constraint yet.
                            false
                        }
                    };

                    if made_progress {
                        // If we made progress, we can remove this constraint.
                        any_changed = true;
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
                    let made_progress = match variable {
                        VariableIndex::Local(local) | VariableIndex::Temporary(local) => {
                            let local_info = self.locals.variable_mut(local).unwrap();
                            match local_info.ty.as_mut() {
                                Some(ty) => {
                                    if *ty != required_ty {
                                        any_changed = true;
                                    }

                                    if required_ty.maybe_signed_integer() {
                                        if !ty.is_integer() {
                                            return Err(CompileError {
                                                source: self.source,
                                                location: constraint.source_location,
                                                error: format!(
                                                    "Type mismatch: expected {required_ty}, found {}",
                                                    *ty
                                                ),
                                            });
                                        }
                                        true
                                    } else if ty.maybe_signed_integer() {
                                        if !required_ty.is_integer() {
                                            return Err(CompileError {
                                                source: self.source,
                                                location: constraint.source_location,
                                                error: format!(
                                                    "Type mismatch: expected {required_ty}, found {}",
                                                    *ty
                                                ),
                                            });
                                        }
                                        true
                                    } else if *ty != required_ty {
                                        return Err(CompileError {
                                            source: self.source,
                                            location: constraint.source_location,
                                            error: format!(
                                                "Type mismatch: expected {required_ty}, found {}",
                                                *ty
                                            ),
                                        });
                                    } else {
                                        *ty = required_ty;
                                        !ty.maybe_signed_integer()
                                    }
                                }
                                None => {
                                    // If the type is not set, we can set it now.
                                    local_info.ty = Some(required_ty);
                                    any_changed = true;
                                    true
                                }
                            }
                        }
                        VariableIndex::Global(global) => {
                            let Some((_name, _global_info)) = self.globals.get_index(global.0)
                            else {
                                unreachable!();
                            };

                            // TODO
                            true
                        }
                    };

                    if made_progress {
                        any_changed = true;
                        self.constraints.swap_remove(i);
                    } else {
                        // If we didn't make progress, just move to the next constraint.
                        i += 1;
                    }
                }

                ConstraintKind::Reference { left, right } => {
                    let left_type = match left {
                        VariableIndex::Global(_) => {
                            return Err(CompileError {
                                source: self.source,
                                location: constraint.source_location,
                                error: "Globals cannot be references".to_string(),
                            });
                        }
                        _ => self.current_type_of(left),
                    };
                    let right_type = self.current_type_of(right);

                    let made_progress = match (left_type, right_type) {
                        (Some(left_ty), Some(right_ty)) if left_ty.reference() == right_ty => {
                            !left_ty.maybe_signed_integer()
                        }
                        (Some(left_ty), Some(right_ty))
                            if left_ty.maybe_signed_integer() && right_ty.is_integer() =>
                        {
                            if right_ty.maybe_signed_integer() {
                                false
                            } else {
                                self.locals
                                    .variable_mut(left.local_index().unwrap())
                                    .unwrap()
                                    .ty = Some(right_ty.reference());

                                any_changed = true;

                                !right_ty.maybe_signed_integer()
                            }
                        }
                        (Some(left_ty), Some(right_ty))
                            if right_ty.maybe_signed_integer() && left_ty.is_integer() =>
                        {
                            if left_ty.maybe_signed_integer() {
                                false
                            } else {
                                let left_ty =
                                    left_ty.dereference().ok_or_else(|| CompileError {
                                        source: self.source,
                                        location: constraint.source_location,
                                        error: format!("Cannot dereference {left_ty}"),
                                    })?;
                                self.locals
                                    .variable_mut(right.local_index().unwrap())
                                    .unwrap()
                                    .ty = Some(left_ty);

                                any_changed = true;

                                !left_ty.maybe_signed_integer()
                            }
                        }

                        (Some(left_ty), Some(right_ty)) => {
                            return Err(CompileError {
                                source: self.source,
                                location: constraint.source_location,
                                error: format!(
                                    "Type mismatch: expected {left_ty}, found {right_ty}"
                                ),
                            });
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

                            any_changed = true;

                            !ty.maybe_signed_integer()
                        }
                        (None, Some(ty)) => {
                            self.locals
                                .variable_mut(left.local_index().unwrap())
                                .unwrap()
                                .ty = Some(ty.reference());

                            any_changed = true;

                            !ty.maybe_signed_integer()
                        }
                        (None, None) => false,
                    };

                    if made_progress {
                        // If we made progress, we can remove this constraint.
                        any_changed = true;
                        self.constraints.swap_remove(i);
                    } else {
                        // If we didn't make progress, just move to the next constraint.
                        i += 1;
                    }
                }
            }
        }

        Ok(any_changed)
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
        let Some(implem) = func.implementation.as_mut() else {
            continue;
        };

        // For now we only optimize variables that are declared AND freed in the same block.
        // This is way too conservative, but relaxing it would require a figuring out loops.
        let mut relevant_variables_in_block = HashMap::new();
        for (block_idx, block) in implem.blocks.iter().enumerate() {
            let mut declared = HashMap::new();
            let mut freed = HashSet::new();
            for instruction in &block.instructions {
                match instruction.instruction {
                    ir::Ir::Declare(var, _) => {
                        // We also don't allow optimizing TopOfStack variables for now.
                        if var.allocation_method == ir::AllocationMethod::FirstFit {
                            declared.insert(var.index, var.is_argument);
                        }
                    }
                    ir::Ir::FreeVariable(var) => {
                        freed.insert(var);
                    }
                    _ => {}
                }
            }

            let mut relevant = HashMap::new();
            for freed in freed {
                if let Some((key, value)) = declared.remove_entry(&freed) {
                    relevant.insert(key, value);
                }
            }

            relevant_variables_in_block.insert(block_idx, relevant);
        }

        for (block_idx, block) in implem.blocks.iter_mut().enumerate() {
            let variables_in_block = &relevant_variables_in_block[&block_idx];
            propagate_destination(block, variables_in_block);
            propagate_value(block, variables_in_block);
            remove_unused_assignments(block, variables_in_block);
            remove_unused_variables(block, variables_in_block);
        }
    }
    // Remove unused variables.

    Ok(())
}

/// This function removes blocks that only contain a jump to another block,
/// effectively bypassing them. The bypassed blocks will be removed by `remove_unreachable_blocks`.
fn bypass_redundant_jump_blocks(func: &mut ir::Function) {
    // We will keep track of replacements for blocks that are bypassed.
    let mut replacements = HashMap::new();
    let Some(implem) = func.implementation.as_mut() else {
        return;
    };

    for (index, block) in implem.blocks.iter().enumerate() {
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

    remap_block_idxs(implem, remap);
}

fn remap_block_idxs<F>(implem: &mut ir::FunctionImplementation, remap: F)
where
    F: Fn(&BlockIndex) -> BlockIndex,
{
    for block in &mut implem.blocks {
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
    let Some(implem) = func.implementation.as_mut() else {
        return;
    };
    let mut reachable = vec![false; implem.blocks.len()];
    let mut stack = vec![BlockIndex(0)]; // Start with the first block.

    while let Some(BlockIndex(block_index)) = stack.pop() {
        if reachable[block_index] {
            continue; // Already visited this block.
        }
        reachable[block_index] = true;

        let block = &implem.blocks[block_index];

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

    remap_block_idxs(implem, remap);

    // Remove unreachable blocks.
    let mut current = 0;
    implem.blocks.retain(|_| {
        let retain = reachable[current];
        current += 1;
        retain
    });
}

fn merge_identical_blocks(func: &mut ir::Function) {
    let Some(implem) = func.implementation.as_mut() else {
        return;
    };
    let mut identical_to = (0..).take(implem.blocks.len()).collect::<Vec<_>>();

    for (idx, block) in implem.blocks.iter().enumerate() {
        // Check if this block is identical to any previous block.
        for (prev_idx, prev_block) in implem.blocks.iter().enumerate().take(idx) {
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

    remap_block_idxs(implem, remap);

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
    strings: &string_interner::Strings,
    globals: &IndexMap<StringIndex, ir::GlobalVariableInfo>,
    signatures: &HashMap<StringIndex, FunctionSignature>,
) -> Result<(), CompileError<'s>> {
    let Some(implem) = function.implementation.as_mut() else {
        return Ok(());
    };
    let mut resolver = TypeResolver::new(source, globals, &mut implem.variables);

    for block in &mut implem.blocks {
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

    while !resolver.constraints.is_empty() {
        if !resolver.step_resolve()? {
            // No more progress can be made.
            break;
        }
    }

    Ok(())
}

fn propagate_destination(
    block: &mut ir::Block,
    variables_in_block: &HashMap<LocalVariableIndex, bool>,
) {
    for idx in (1..block.instructions.len()).rev() {
        if let ir::Ir::Assign(dst, src) = block.instructions[idx].instruction {
            // If src is not defined in this block, we can't avoid writing to it.
            if let Some(src) = src.local_index() {
                // If the source is not in the block, we can skip it.
                if !variables_in_block.contains_key(&src) {
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

fn propagate_value(block: &mut ir::Block, _: &HashMap<LocalVariableIndex, bool>) {
    let mut candidates = HashMap::new();
    let mut values = HashMap::new();
    let mut to_delete = vec![];
    for idx in 0..block.instructions.len() {
        match block.instructions[idx].instruction {
            ir::Ir::Declare(new, None) => {
                candidates.insert(new.index, (idx, new));
            }
            ir::Ir::Declare(new, Some(value)) => {
                values.insert(new.index, value);
            }
            ir::Ir::Assign(dst, src) => {
                let Some(dst) = dst.local_index() else {
                    continue;
                };
                if let Some((instruction_idx, decl)) = candidates.remove(&dst) {
                    let Some(src) = src.local_index() else {
                        continue;
                    };
                    if let Some(value) = values.get(&src) {
                        block.instructions[instruction_idx].instruction =
                            ir::Ir::Declare(decl, Some(*value));
                        to_delete.push(idx);
                    }
                }
            }
            _ => {}
        }
    }

    // Remove instructions
    let mut current = 0;
    block.instructions.retain(|_| {
        let delete = to_delete.contains(&current);
        current += 1;
        !delete
    });
}

fn remove_unused_assignments(
    block: &mut ir::Block,
    variables_in_block: &HashMap<LocalVariableIndex, bool>,
) {
    let mut can_remove = variables_in_block
        .iter()
        .filter_map(|(v, is_arg)| if *is_arg { None } else { Some((*v, *is_arg)) })
        .collect::<HashMap<_, _>>();
    for idx in (0..block.instructions.len()).rev() {
        // First, update `can_remove` with the read variables
        match &block.instructions[idx].instruction {
            ir::Ir::Assign(_, src)
            | ir::Ir::DerefAssign(_, src)
            | ir::Ir::UnaryOperator(_, _, src) => {
                if let Some(src) = src.local_index() {
                    can_remove.remove(&src);
                }
            }
            ir::Ir::BinaryOperator(_, _, lhs, rhs) => {
                if let Some(lhs) = lhs.local_index() {
                    can_remove.remove(&lhs);
                }
                if let Some(rhs) = rhs.local_index() {
                    can_remove.remove(&rhs);
                }
            }
            ir::Ir::Call(_, _, args) => {
                for src in args {
                    if let Some(src) = src.local_index() {
                        can_remove.remove(&src);
                    }
                }
            }
            _ => {}
        };

        // Next, try to remove the operation
        match &block.instructions[idx].instruction {
            ir::Ir::Assign(dst, _)
            | ir::Ir::UnaryOperator(_, dst, _)
            | ir::Ir::BinaryOperator(_, dst, _, _) => {
                if let Some(dst) = dst.local_index() {
                    if can_remove.contains_key(&dst) {
                        block.instructions.remove(idx); // Remove the assignment.
                    }
                }
            }
            _ => {}
        };
    }
}

fn remove_unused_variables(
    block: &mut ir::Block,
    variables_in_block: &HashMap<LocalVariableIndex, bool>,
) {
    let mut can_remove = variables_in_block
        .iter()
        .filter_map(|(v, is_arg)| if *is_arg { None } else { Some((*v, *is_arg)) })
        .collect::<HashMap<_, _>>();
    for idx in (0..block.instructions.len()).rev() {
        match &block.instructions[idx].instruction {
            ir::Ir::Assign(_, src) | ir::Ir::DerefAssign(_, src) => {
                if let Some(src) = src.local_index() {
                    can_remove.remove(&src);
                }
            }
            ir::Ir::UnaryOperator(_, dst, src) => {
                if let Some(src) = src.local_index() {
                    can_remove.remove(&src);
                }
                if let Some(dst) = dst.local_index() {
                    can_remove.remove(&dst);
                }
            }
            ir::Ir::BinaryOperator(_, dst, lhs, rhs) => {
                if let Some(lhs) = lhs.local_index() {
                    can_remove.remove(&lhs);
                }
                if let Some(rhs) = rhs.local_index() {
                    can_remove.remove(&rhs);
                }
                if let Some(dst) = dst.local_index() {
                    can_remove.remove(&dst);
                }
            }
            ir::Ir::Call(_, rtv, args) => {
                for src in args {
                    if let Some(src) = src.local_index() {
                        can_remove.remove(&src);
                    }
                }
                if let Some(rtv) = rtv.local_index() {
                    can_remove.remove(&rtv);
                }
            }
            _ => {}
        };
    }
    for idx in (0..block.instructions.len()).rev() {
        match &block.instructions[idx].instruction {
            ir::Ir::Declare(v, _) => {
                if can_remove.contains_key(&v.index) {
                    block.instructions.remove(idx); // Remove the assignment.
                }
            }
            ir::Ir::FreeVariable(v) => {
                if can_remove.contains_key(v) {
                    block.instructions.remove(idx); // Remove the assignment.
                }
            }
            _ => {}
        };
    }
}
