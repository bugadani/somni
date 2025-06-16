use indexmap::{IndexMap, IndexSet};

use crate::{ir::Type, string_interner::StringIndex};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RestorePoint(usize);

impl RestorePoint {
    pub const RETURN_FROM_FN: Self = RestorePoint(0);
}

/// An index into the variables in a program. This uniquely identifies a variable, but
/// it is not directly related to a variable's address.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct LocalVariableIndex(pub usize);

impl LocalVariableIndex {
    pub const RETURN_VALUE: Self = LocalVariableIndex(0);
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Event {
    VariableDeclared(LocalVariableIndex),
    /// Variables without a reference to them can have their memory reused after the last use.
    VariableFreed(LocalVariableIndex),
}

#[derive(Debug, Clone)]
pub struct LocalVariableInfo {
    pub name: StringIndex,
    pub has_reference: bool,
    pub ty: Option<Type>,
}

#[derive(Debug, Clone)]
pub struct ScopeData {
    all_variables: Vec<LocalVariableInfo>,
    // Set of variables that are live at the restore point
    restore_points: Vec<(
        IndexSet<LocalVariableIndex>,
        IndexMap<StringIndex, LocalVariableIndex>,
    )>,
    events: Vec<Event>,
}

impl ScopeData {
    pub const fn empty() -> Self {
        Self {
            all_variables: Vec::new(),
            restore_points: Vec::new(),
            events: Vec::new(),
        }
    }

    pub fn variable(&self, index: LocalVariableIndex) -> Option<&LocalVariableInfo> {
        self.all_variables.get(index.0)
    }

    pub fn variable_mut(&mut self, index: LocalVariableIndex) -> Option<&mut LocalVariableInfo> {
        self.all_variables.get_mut(index.0)
    }
}

#[derive(Debug, Clone)]
pub(crate) struct VariableTracker {
    currently_allocated_variables: IndexSet<LocalVariableIndex>,
    currently_visible_variables: IndexMap<StringIndex, LocalVariableIndex>,
    data: ScopeData,
}

impl VariableTracker {
    pub fn new() -> Self {
        Self {
            currently_allocated_variables: IndexSet::new(),
            currently_visible_variables: IndexMap::new(),
            data: ScopeData {
                all_variables: Vec::new(),
                restore_points: vec![(IndexSet::new(), IndexMap::new())],
                events: Vec::new(),
            },
        }
    }

    pub fn declare_variable(&mut self, name: StringIndex, ty: Option<Type>) -> LocalVariableIndex {
        let index = LocalVariableIndex(self.data.all_variables.len());
        if let Some(old) = self.currently_visible_variables.insert(name, index) {
            if !self.data.all_variables[old.0].has_reference {
                self.currently_allocated_variables.shift_remove(&old);
                self.data.events.push(Event::VariableFreed(old));
            }
        }
        self.currently_allocated_variables.insert(index);
        self.data.all_variables.push(LocalVariableInfo {
            name,
            has_reference: false,
            ty,
        });
        self.data.events.push(Event::VariableDeclared(index));
        index
    }

    pub(crate) fn free_variable(&mut self, var: LocalVariableIndex) {
        assert!(self.is_visible(var), "Variable not visible: {var:?}");
        let var_info = &mut self.data.all_variables[var.0];
        if !var_info.has_reference {
            self.currently_visible_variables.swap_remove(&var_info.name);
            self.currently_allocated_variables.shift_remove(&var);
            self.data.events.push(Event::VariableFreed(var));
        } else {
            panic!("Attempted to free a variable that has a reference: {var:?}");
        }
    }

    pub fn reference_variable(&mut self, var: LocalVariableIndex) {
        assert!(self.is_visible(var), "Variable not visible: {var:?}");
        if let Some(var_info) = self.data.all_variables.get_mut(var.0) {
            var_info.has_reference = true;
        } else {
            panic!("Attempted to create a reference to a variable that does not exist: {var:?}");
        }
    }

    pub fn find(&self, index: StringIndex) -> Option<LocalVariableIndex> {
        self.currently_visible_variables.get(&index).copied()
    }

    pub fn is_visible(&self, index: LocalVariableIndex) -> bool {
        self.data
            .all_variables
            .get(index.0)
            .and_then(|var| self.currently_visible_variables.get(&var.name).copied())
            == Some(index)
    }

    pub fn create_restore_point(&mut self) -> RestorePoint {
        let restore_point_index = self.data.restore_points.len();
        self.data.restore_points.push((
            self.currently_allocated_variables.clone(),
            self.currently_visible_variables.clone(),
        ));
        RestorePoint(restore_point_index)
    }

    pub fn rollback_to_restore_point(
        &mut self,
        restore_point: RestorePoint,
    ) -> Vec<LocalVariableIndex> {
        let Some(vars_after_restore) = self.data.restore_points.get(restore_point.0) else {
            panic!(
                "Attempted to rollback to a restore point that does not exist: {restore_point:?}"
            );
        };

        let (allocated, visible) = vars_after_restore.clone();

        self.restore(allocated, visible)
    }

    pub fn finalize(self) -> ScopeData {
        self.data
    }

    pub fn len(&self) -> usize {
        self.data.all_variables.len()
    }

    fn restore(
        &mut self,
        allocated: IndexSet<LocalVariableIndex>,
        visible: IndexMap<StringIndex, LocalVariableIndex>,
    ) -> Vec<LocalVariableIndex> {
        let freed_vars = self
            .currently_allocated_variables
            .difference(&allocated)
            .copied()
            .rev()
            .collect::<Vec<_>>();

        for var in freed_vars.iter() {
            self.data.events.push(Event::VariableFreed(*var));
        }
        self.currently_allocated_variables = allocated;
        self.currently_visible_variables = visible;

        freed_vars
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::string_interner::StringInterner;

    #[test]
    fn test_variable_tracker() {
        let mut tracker = VariableTracker::new();
        let mut interner = StringInterner::new();

        // Let's simulate this program:
        // ```
        // declare x;
        // declare y;
        // if (some condition) {
        //     declare z;
        //     declare z; // This should reuse the memory of z
        //     declare w; // Reference to z
        //     reference z;
        //     declare z; // This should create a new variable
        // }
        // ```

        tracker.declare_variable(interner.intern("x"), None);
        tracker.declare_variable(interner.intern("y"), None);
        let restore_point = tracker.create_restore_point();
        assert!(tracker.find(interner.intern("x")).is_some());
        assert!(tracker.find(interner.intern("y")).is_some());
        let z = tracker.declare_variable(interner.intern("z"), None);
        let z2 = tracker.declare_variable(interner.intern("z"), None);
        assert_ne!(z, z2);
        assert!(!tracker.is_visible(z));
        let w = tracker.declare_variable(interner.intern("w"), None);
        tracker.reference_variable(z2);
        let z3 = tracker.declare_variable(interner.intern("z"), None);
        assert_ne!(z, z3);
        assert_eq!(tracker.find(interner.intern("z")), Some(z3));
        assert!(tracker.find(interner.intern("w")).is_some());
        assert!(tracker.find(interner.intern("x")).is_some());
        assert!(tracker.find(interner.intern("y")).is_some());
        let freed = tracker.rollback_to_restore_point(restore_point);
        assert_eq!(freed, [z3, w, z2]);
        assert!(tracker.find(interner.intern("x")).is_some());
        assert!(tracker.find(interner.intern("y")).is_some());
        assert!(tracker.find(interner.intern("z")).is_none());
        assert!(tracker.find(interner.intern("w")).is_none());

        tracker.rollback_to_restore_point(RestorePoint::RETURN_FROM_FN);
        let data = tracker.finalize();

        assert_eq!(data.all_variables.len(), 6);
    }
}
