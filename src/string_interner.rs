use std::collections::HashMap;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct StringIndex(usize);

impl StringIndex {
    pub const fn dummy() -> Self {
        Self(0)
    }
}

#[derive(Default, Clone, Debug)]
pub struct Strings {
    strings: String,
    positions: Vec<(usize, usize)>, // (start, length) positions of strings in the string
}

impl Strings {
    pub fn new() -> Self {
        Strings::default()
    }

    pub(crate) fn intern(&mut self, value: &str) -> StringIndex {
        let start = self.strings.len();
        let length = value.len();
        self.strings.push_str(value);

        let index = self.positions.len();
        self.positions.push((start, length));
        StringIndex(index)
    }

    pub fn lookup(&self, idx: StringIndex) -> &str {
        let (start, length) = self.positions[idx.0];
        &self.strings[start..start + length]
    }

    pub(crate) fn find(&self, name: &str) -> Option<StringIndex> {
        for (index, (start, length)) in self.positions.iter().enumerate() {
            if &self.strings[*start..*start + *length] == name {
                return Some(StringIndex(index));
            }
        }

        None
    }
}

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

    pub fn intern(&mut self, value: &str) -> StringIndex {
        if let Some(index) = self.reverse_lookup.get(value).copied() {
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
