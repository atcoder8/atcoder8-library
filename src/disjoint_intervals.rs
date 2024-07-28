use std::{collections::BTreeMap, ops};

/// Data structure that represent a set by the sum of disjoint intervals.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DisjointIntervals<T> {
    /// Map with the start of the interval as key and the end of the interval as value.
    /// Each interval represented as a right half-open interval.
    intervals: BTreeMap<T, T>,

    /// Number of elements belonging to one of the intervals.
    len: usize,
}

impl DisjointIntervals<usize> {
    /// Creates a new empty set.
    pub fn new() -> Self {
        DisjointIntervals {
            intervals: BTreeMap::new(),
            len: 0,
        }
    }

    /// Returns the number of elements in the set.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` if and only if the set is empty.
    pub fn is_empty(&self) -> bool {
        self.intervals.is_empty()
    }

    /// Returns `true` if and only if `value` is in the set.
    pub fn contains(&self, value: usize) -> bool {
        self.intervals
            .range(..=value)
            .next_back()
            .is_some_and(|(_, &right)| right > value)
    }

    /// Returns the number of non-contiguous intervals.
    pub fn number_of_intervals(&self) -> usize {
        self.intervals.len()
    }

    /// Adds the elements contained in the `range`.
    /// Returns the number of newly added elements.
    pub fn insert_range(&mut self, range: ops::Range<usize>) -> usize {
        if range.is_empty() {
            return 0;
        }

        // Finds both ends of the updated intervals such that they contain `range`.
        let insert_left = match self.intervals.range(..=range.start).next_back() {
            Some((&left, &right)) if right >= range.start => left,
            _ => range.start,
        };
        let insert_right = match self.intervals.range(..=range.end).next_back() {
            Some((_, &right)) => right.max(range.end),
            None => range.end,
        };

        // Update the set of intervals and counts the number of newly inserted elements.
        let mut add_element_num = insert_right - insert_left;
        loop {
            match self.intervals.range(insert_left..).next() {
                Some((&left, &right)) if right <= insert_right => {
                    self.intervals.remove(&left);
                    add_element_num -= right - left;
                }
                _ => break,
            }
        }
        self.intervals.insert(insert_left, insert_right);

        // Update number of elements in the set.
        self.len += add_element_num;

        add_element_num
    }

    /// Adds an element.
    /// Returns `true` if and only if the element is newly added.
    pub fn insert(&mut self, value: usize) -> bool {
        self.insert_range(value..value + 1) == 1
    }

    /// Returns the smallest non-negative integer not included in the set.
    pub fn mex(&self) -> usize {
        match self.intervals.first_key_value() {
            Some((&left, &right)) if left == 0 => right,
            _ => 0,
        }
    }

    /// Generates an iterator that traverses the elements contained in the set.
    pub fn range_inclusively(&self, range: ops::Range<usize>) -> RangeInclusively<usize> {
        RangeInclusively {
            intervals: self,
            range,
        }
    }

    /// Generates an iterator that traverses the elements not contained in the set.
    pub fn range_exclusively(&self, range: ops::Range<usize>) -> RangeExclusively<usize> {
        RangeExclusively {
            intervals: self,
            range,
        }
    }
}

#[derive(Debug)]
pub struct RangeInclusively<'a, T> {
    intervals: &'a DisjointIntervals<T>,
    range: ops::Range<usize>,
}

impl<'a> Iterator for RangeInclusively<'a, usize> {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        if self.range.is_empty() {
            return None;
        }

        if !self.intervals.contains(self.range.start) {
            match self.intervals.intervals.range(self.range.start..).next() {
                Some((&left, _)) => self.range.start = left,
                None => {
                    self.range.start = self.range.end;
                    return None;
                }
            }
        }

        self.range.start += 1;

        Some(self.range.start - 1)
    }
}

impl<'a> DoubleEndedIterator for RangeInclusively<'a, usize> {
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.range.is_empty() {
            return None;
        }

        if !self.intervals.contains(self.range.end - 1) {
            match self.intervals.intervals.range(..self.range.end).next_back() {
                Some((_, &right)) => self.range.end = right,
                None => {
                    self.range.end = self.range.start;
                    return None;
                }
            }
        }

        self.range.end -= 1;

        Some(self.range.end)
    }
}

#[derive(Debug)]
pub struct RangeExclusively<'a, T> {
    intervals: &'a DisjointIntervals<T>,
    range: ops::Range<usize>,
}

impl<'a> Iterator for RangeExclusively<'a, usize> {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        if self.range.is_empty() {
            return None;
        }

        if self.intervals.contains(self.range.start) {
            let (_, &right) = self
                .intervals
                .intervals
                .range(..=self.range.start)
                .next_back()
                .unwrap();
            self.range.start = right.min(self.range.end);

            if self.range.is_empty() {
                return None;
            }
        }

        self.range.start += 1;

        Some(self.range.start - 1)
    }
}

impl<'a> DoubleEndedIterator for RangeExclusively<'a, usize> {
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.range.is_empty() {
            return None;
        }

        if self.intervals.contains(self.range.end - 1) {
            let (&left, _) = self
                .intervals
                .intervals
                .range(..self.range.end)
                .next_back()
                .unwrap();
            self.range.end = left.max(self.range.start);

            if self.range.is_empty() {
                return None;
            }
        }

        self.range.end -= 1;

        Some(self.range.end)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_insert() {
        let mut set = DisjointIntervals::new();

        // Expected: {}
        assert!(set.is_empty());
        assert_eq!(set.len(), 0);
        assert_eq!(set.number_of_intervals(), 0);

        // Insert an element.
        // Expected: {3}
        let newly_inserted = set.insert(3);
        assert!(newly_inserted);
        assert!(!set.is_empty());
        assert_eq!(set.len(), 1);
        assert_eq!(set.number_of_intervals(), 1);

        // Insert an existing element.
        // Expected: {3}
        let newly_inserted = set.insert(3);
        assert!(!newly_inserted);
        assert!(!set.is_empty());
        assert_eq!(set.len(), 1);
        assert_eq!(set.number_of_intervals(), 1);

        // Insert an adjacent element.
        // Expected: {3, 4}
        let newly_inserted = set.insert(4);
        assert!(newly_inserted);
        assert!(!set.is_empty());
        assert_eq!(set.len(), 2);
        assert_eq!(set.number_of_intervals(), 1);

        // Insert an adjacent element.
        // Expected: {2, 3, 4}
        let newly_inserted = set.insert(2);
        assert!(newly_inserted);
        assert!(!set.is_empty());
        assert_eq!(set.len(), 3);
        assert_eq!(set.number_of_intervals(), 1);

        // Insert elements within a range.
        // Expected: {2, 3, 4, 8, 9}
        let incremental_number = set.insert_range(8..10);
        assert_eq!(incremental_number, 2);
        assert!(!set.is_empty());
        assert_eq!(set.len(), 5);
        assert_eq!(set.number_of_intervals(), 2);

        // Insert an element.
        // Expected: {2, 3, 4, 6, 8, 9}
        let newly_inserted = set.insert(6);
        assert!(newly_inserted);
        assert!(!set.is_empty());
        assert_eq!(set.len(), 6);
        assert_eq!(set.number_of_intervals(), 3);

        // Insert elements in an overlapping range.
        // Expected: {2, 3, 4, 5, 6, 7, 8, 9}
        let incremental_number = set.insert_range(3..9);
        assert_eq!(incremental_number, 2);
        assert!(!set.is_empty());
        assert_eq!(set.len(), 8);
        assert_eq!(set.number_of_intervals(), 1);
    }

    #[test]
    fn test_mex() {
        let mut set = DisjointIntervals::new();

        // mex({}) = 0
        assert_eq!(set.mex(), 0);

        set.insert_range(8..10);
        set.insert_range(2..4);
        set.insert(1);

        // mex({1, 2, 3, 8, 9}) = 0
        assert_eq!(set.mex(), 0);

        set.insert(0);

        // mex({0, 1, 2, 3, 8, 9}) = 4
        assert_eq!(set.mex(), 4);

        set.insert_range(2..100);

        // mex({0, 1, 2, ... , 99}) = 100
        assert_eq!(set.mex(), 100);
    }

    #[test]
    fn test_range_inclusively() {
        let mut set = DisjointIntervals::new();
        for value in [1, 2, 3, 4, 6, 8, 9, 10, 11] {
            set.insert(value);
        }

        let mut range_inclusively = set.range_inclusively(2..11);
        assert_eq!(range_inclusively.next_back(), Some(10));
        let collected = range_inclusively.collect::<Vec<usize>>();
        assert_eq!(&collected, &[2, 3, 4, 6, 8, 9]);

        let mut range_inclusively = set.range_inclusively(2..11);
        assert_eq!(range_inclusively.next(), Some(2));
        let collected = range_inclusively.rev().collect::<Vec<usize>>();
        assert_eq!(&collected, &[10, 9, 8, 6, 4, 3]);
    }

    #[test]
    fn test_range_exclusively() {
        let mut set = DisjointIntervals::new();
        for value in [1, 2, 3, 4, 6, 8, 9, 10, 11, 14, 15] {
            set.insert(value);
        }

        let mut range_exclusively = set.range_exclusively(0..14);
        assert_eq!(range_exclusively.next_back(), Some(13));
        let collected = range_exclusively.collect::<Vec<usize>>();
        assert_eq!(&collected, &[0, 5, 7, 12]);

        let mut range_exclusively = set.range_exclusively(0..14);
        assert_eq!(range_exclusively.next(), Some(0));
        let collected = range_exclusively.rev().collect::<Vec<usize>>();
        assert_eq!(&collected, &[13, 12, 7, 5]);
    }
}
