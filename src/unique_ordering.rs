//! Module for ordering unique elements.

use std::ops::Index;

/// Structure for ordering unique elements.
#[derive(Debug, Clone)]
pub struct UniqueOrdering<T> {
    /// A sequence containing the elements to be ordered.
    seq: Vec<T>,

    /// A flag indicating whether `seq` is sorted and deduplicated.
    organized: bool,
}

impl<T> Default for UniqueOrdering<T>
where
    T: Clone + Ord,
{
    /// Creates a structure for ordering unique elements.
    fn default() -> Self {
        Self::new()
    }
}

impl<T> From<Vec<T>> for UniqueOrdering<T>
where
    T: Clone + Ord,
{
    /// Creates a structure by initializing the elements to be ordered with `seq`.
    fn from(seq: Vec<T>) -> Self {
        Self {
            seq,
            organized: false,
        }
    }
}

impl<T> Index<usize> for UniqueOrdering<T> {
    type Output = T;

    /// Returns the `index`-th (0-based) unique element.
    fn index(&self, index: usize) -> &Self::Output {
        &self.seq[index]
    }
}

impl<T> UniqueOrdering<T>
where
    T: Clone + Ord,
{
    /// Creates a structure for ordering unique elements.
    pub fn new() -> Self {
        Self {
            seq: vec![],
            organized: true,
        }
    }

    /// Adds the elements to be ordered.
    pub fn push(&mut self, x: T) {
        self.seq.push(x);
        self.organized = false;
    }

    /// Appends all elements of the iterator to the elements to be ordered.
    pub fn extend<I>(&mut self, other: I)
    where
        I: IntoIterator<Item = T>,
    {
        self.seq.extend(other);
        self.organized = false;
    }

    /// Sorts the sequence of stored elements in ascending order and removes all duplicates.
    fn organize(&mut self) {
        if !self.organized {
            self.seq.sort_unstable();
            self.seq.dedup();
            self.organized = true;
        }
    }

    /// Returns the `x` position of the unique elements sorted in ascending order.
    pub fn position(&mut self, x: &T) -> usize {
        self.organize();

        self.seq.binary_search(x).unwrap_or_else(|_| {
            panic!("The position of `x` is undefined.");
        })
    }

    /// Returns the `index`-th (0-based) unique element.
    ///
    /// Returns `None` if the `index`-th element does not exist.
    pub fn get(&mut self, index: usize) -> Option<&T> {
        self.organize();

        self.seq.get(index)
    }

    /// Returns the number of unique elements.
    pub fn unique_len(&mut self) -> usize {
        self.organize();

        self.seq.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test() {
        let seq = vec![3, 1, 4, 1, 5, 9];
        let mut uniq_ord = UniqueOrdering::from(seq);

        assert_eq!(uniq_ord.unique_len(), 5);
        assert_eq!(uniq_ord.position(&4), 2);
        assert_eq!(uniq_ord[2], 4);

        uniq_ord.push(1);
        uniq_ord.push(2);

        assert_eq!(uniq_ord.unique_len(), 6);
        assert_eq!(uniq_ord.position(&1), 0);
        assert_eq!(uniq_ord[0], 1);
        assert_eq!(uniq_ord.position(&2), 1);
        assert_eq!(uniq_ord[1], 2);

        uniq_ord.extend(vec![5, 9, 5, 20, 5]);

        assert_eq!(uniq_ord.unique_len(), 7);
        assert_eq!(uniq_ord.position(&20), 6);
        assert_eq!(uniq_ord[6], 20);
    }
}
