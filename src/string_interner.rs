//! A simple string interner.

use std::collections::HashMap;

use somni_expr::{
    value::{Load, Store, ValueType},
    ExprContext, OperatorError, Type, TypeSet, TypedValue,
};

/// The ID of a string in the interner.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct StringIndex(pub usize);

impl StringIndex {
    /// Creates a dummy `StringIndex` with value 0.
    pub const fn dummy() -> Self {
        Self(0)
    }
}
impl ValueType for StringIndex {
    const TYPE: Type = Type::String;

    type NegateOutput = Self;

    fn equals(a: Self, b: Self) -> Result<bool, OperatorError> {
        Ok(a == b)
    }
}
impl<T> Load<T> for StringIndex
where
    T: TypeSet<String = StringIndex>,
{
    type Output<'s>
        = Self
    where
        T: 's;

    fn load<'s>(
        _ctx: &'s dyn ExprContext<T>,
        typed: &'s TypedValue<T>,
    ) -> Option<Self::Output<'s>> {
        if let TypedValue::String(value) = typed {
            return Some(*value);
        }

        None
    }
}
impl<T> Store<T> for StringIndex
where
    T: TypeSet<String = StringIndex>,
{
    fn store(&self, _ctx: &mut dyn ExprContext<T>) -> TypedValue<T> {
        TypedValue::String(*self)
    }
}

/// A collection of interned strings.
#[derive(Default, Clone, Debug)]
pub struct Strings {
    strings: String,
    positions: Vec<(usize, usize)>, // (start, length) positions of strings in the string
}

impl Strings {
    /// Creates a new `Strings` instance.
    pub fn new() -> Self {
        Strings::default()
    }

    /// Interns a string and returns its index.
    pub fn intern(&mut self, value: &str) -> StringIndex {
        let start = self.strings.len();
        let length = value.len();
        self.strings.push_str(value);

        let index = self.positions.len();
        self.positions.push((start, length));
        StringIndex(index)
    }

    /// Returns the string at the given index.
    pub fn lookup(&self, idx: StringIndex) -> &str {
        let (start, length) = self.positions[idx.0];
        &self.strings[start..start + length]
    }

    /// Returns the index of a particular string, if it exists.
    pub fn find(&self, name: &str) -> Option<StringIndex> {
        for (index, (start, length)) in self.positions.iter().enumerate() {
            if &self.strings[*start..*start + *length] == name {
                return Some(StringIndex(index));
            }
        }

        None
    }
}

/// A string interner that allows for more efficient string storage and retrieval.
#[derive(Default, Clone, Debug)]
pub struct StringInterner {
    strings: Strings,
    reverse_lookup: HashMap<String, StringIndex>, // TODO this doesn't need to be stored
}
impl StringInterner {
    pub fn new() -> Self {
        StringInterner::default()
    }

    pub fn lookup(&self, idx: StringIndex) -> &str {
        self.strings.lookup(idx)
    }

    pub fn lookup_index_by_value(&self, value: &str) -> Option<StringIndex> {
        self.reverse_lookup.get(value).cloned()
    }

    pub fn find(&self, value: &str) -> Option<StringIndex> {
        self.reverse_lookup.get(value).copied()
    }

    pub fn intern(&mut self, value: &str) -> StringIndex {
        if let Some(index) = self.find(value) {
            return index;
        }

        let index = self.strings.intern(value);
        self.reverse_lookup.insert(value.to_string(), index);

        index
    }

    pub fn finalize(self) -> Strings {
        self.strings
    }
}
