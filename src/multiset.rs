//! Implements an ordered multiset using `BTreeMap`.

use std::{collections::BTreeMap, iter::FromIterator};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Multiset<T>
where
    T: Ord,
{
    map: BTreeMap<T, usize>,
    len: usize,
}

impl<T> Default for Multiset<T>
where
    T: Ord,
{
    /// Creates an empty multiset.
    fn default() -> Self {
        Self {
            map: BTreeMap::new(),
            len: 0,
        }
    }
}

impl<T> FromIterator<T> for Multiset<T>
where
    T: Ord,
{
    /// Creates a multiset with the elements contained in the `IntoIterator`.
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let mut ms = Multiset::new();
        for value in iter {
            ms.insert(value);
        }

        ms
    }
}

impl<T> From<Vec<T>> for Multiset<T>
where
    T: Ord,
{
    /// Creates a multiset with the elements contained in the `Vec`.
    fn from(value: Vec<T>) -> Self {
        value.into_iter().collect()
    }
}

impl<T> From<BTreeMap<T, usize>> for Multiset<T>
where
    T: Ord,
{
    /// Creates a multiset with the elements contained in the `BTreeMap`.
    ///
    /// Each element of `BTreeMap` must consist of a pair of elements
    /// contained in the multiset and their number.
    fn from(value: BTreeMap<T, usize>) -> Self {
        let len: usize = value.values().sum();

        Self { map: value, len }
    }
}

impl<T> Multiset<T>
where
    T: Ord,
{
    /// Creates an empty multiset.
    pub fn new() -> Self {
        Self {
            map: BTreeMap::new(),
            len: 0,
        }
    }

    /// Returns whether the multiset is empty.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns the number of elements.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Adds a `value` to the set.
    ///
    /// Returns the number of elements equal to `value` in the multiset.
    pub fn insert(&mut self, value: T) -> usize {
        let count = self.map.entry(value).or_default();
        *count += 1;
        self.len += 1;

        *count
    }

    /// If the multiset contains an element equal to `value`, removes it from the multiset.
    ///
    /// Returns the number of elements equal to `value` in the multiset after deletion.
    /// However, if the multiset does not contain any `value`, `None` is returned.
    pub fn remove(&mut self, value: &T) -> Option<usize> {
        if let Some(count) = self.map.get_mut(value) {
            *count -= 1;
            self.len -= 1;

            if *count == 0 {
                self.map.remove(value);

                return Some(0);
            }

            return Some(*count);
        }

        None
    }

    /// Returns whether the multiset contains `value`.
    pub fn contains(&self, value: &T) -> bool {
        self.map.contains_key(value)
    }

    /// Returns the number of elements equal to `value`.
    pub fn count(&self, value: &T) -> usize {
        *self.map.get(value).unwrap_or(&0)
    }

    /// Returns the smallest element contained in the multiset.
    pub fn min_element(&self) -> Option<&T> {
        if let Some((value, _)) = self.map.iter().next() {
            Some(value)
        } else {
            None
        }
    }

    /// Returns the largest element contained in the multiset.
    pub fn max_element(&self) -> Option<&T> {
        if let Some((value, _)) = self.map.iter().next_back() {
            Some(value)
        } else {
            None
        }
    }

    /// Removes all elements.
    pub fn clear(&mut self) {
        self.map.clear();
        self.len = 0;
    }

    /// Removes all elements equal to `value`.
    pub fn remove_all(&mut self, value: &T) {
        self.len -= self.count(value);
        self.map.remove(value);
    }
}
