//! Processes the following query in `O(log(n))` time
//! for a sequence of numbers with `n` elements:
//! * Update one element
//! * Calculate the sum of the elements of a range

use std::ops::{AddAssign, RangeBounds, Sub};

/// # Examples
///
/// ```
/// use atcoder8_library::fenwick_tree::FenwickTree;
///
/// let ft = FenwickTree::<u32>::new(5);
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FenwickTree<T>(Vec<T>)
where
    T: Clone + Default + Sub<T, Output = T> + AddAssign<T>;

impl<T> From<Vec<T>> for FenwickTree<T>
where
    T: Clone + Default + Sub<T, Output = T> + AddAssign<T>,
{
    /// # Examples
    ///
    /// ```
    /// use atcoder8_library::fenwick_tree::FenwickTree;
    ///
    /// let ft = FenwickTree::from(vec![3, -1, 4, -1, 5, 9, 2]);
    /// assert_eq!(ft.sum(2..6), 17);
    /// ```
    fn from(t: Vec<T>) -> Self {
        let mut ft = FenwickTree::new(t.len());
        for (i, x) in t.into_iter().enumerate() {
            ft.add(i, x);
        }
        ft
    }
}

impl<T> FenwickTree<T>
where
    T: Clone + Default + Sub<T, Output = T> + AddAssign<T>,
{
    /// Constructs a `FenwickTree<T>` with `n` elements.
    ///
    /// Each element is initialized with `T::default()`.
    ///
    /// # Examples
    ///
    /// ```
    /// use atcoder8_library::fenwick_tree::FenwickTree;
    ///
    /// let ft = FenwickTree::<i32>::new(5);
    /// assert_eq!(ft.sum(..), 0);
    /// ```
    pub fn new(n: usize) -> Self {
        FenwickTree(vec![T::default(); n])
    }

    /// Add `x` to the `p`-th element.
    ///
    /// # Examples
    ///
    /// ```
    /// use atcoder8_library::fenwick_tree::FenwickTree;
    ///
    /// let ft = FenwickTree::<u32>::new(5);
    /// ```
    pub fn add(&mut self, p: usize, x: T) {
        let FenwickTree(data) = self;
        let n = data.len();

        assert!(p < n);

        let mut p = p + 1;
        while p <= n {
            data[p - 1] += x.clone();
            p += p & p.overflowing_neg().0;
        }
    }

    /// Compute the sum of the range [0, r).
    fn inner_sum(&self, r: usize) -> T {
        let mut r = r;
        let mut s = T::default();
        while r > 0 {
            s += self.0[r - 1].clone();
            r -= r & r.overflowing_neg().0;
        }
        return s;
    }

    /// Calculate the total of the range.
    /// 
    /// # Examples
    ///
    /// ```
    /// use atcoder8_library::fenwick_tree::FenwickTree;
    ///
    /// let ft = FenwickTree::from(vec![3, -1, 4, -1, 5, 9, 2]);
    /// assert_eq!(ft.sum(2..6), 17);
    /// ```
    pub fn sum<R>(&self, rng: R) -> T
    where
        R: RangeBounds<usize>,
    {
        let n = self.0.len();

        let l = match rng.start_bound() {
            std::ops::Bound::Included(&start_bound) => start_bound,
            std::ops::Bound::Excluded(&start_bound) => start_bound + 1,
            std::ops::Bound::Unbounded => 0,
        };

        assert!(l <= n);

        let r = match rng.end_bound() {
            std::ops::Bound::Included(&end_bound) => end_bound + 1,
            std::ops::Bound::Excluded(&end_bound) => end_bound,
            std::ops::Bound::Unbounded => n,
        };

        assert!(r <= n);

        self.inner_sum(r) - self.inner_sum(l)
    }
}
