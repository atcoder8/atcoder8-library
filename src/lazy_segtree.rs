use std::ops::RangeBounds;

pub trait Monoid {
    /// The identity element.
    fn id() -> Self;

    /// The binary operation.
    fn product(&self, rhs: &Self) -> Self;
}

pub trait Mapping<S>: Monoid
where
    S: Monoid,
{
    fn mapping(&self, s: &S) -> S;
}

#[derive(Debug, Default, Clone)]
pub struct LazySegtree<S, F>
where
    S: Monoid,
    F: Mapping<S>,
{
    n: usize,
    log: u32,
    size: usize,
    data: Vec<S>,
    lazy: Vec<F>,
}

impl<S, F> From<Vec<S>> for LazySegtree<S, F>
where
    S: Clone + Monoid,
    F: Clone + Mapping<S>,
{
    fn from(value: Vec<S>) -> Self {
        if value.is_empty() {
            return Self {
                n: 0,
                log: 0,
                size: 0,
                data: vec![],
                lazy: vec![],
            };
        }

        let n = value.len();
        let log = usize::BITS - (n - 1).leading_zeros();
        let size = 1 << log;
        let mut data = vec![S::id(); size];
        data.extend(value);
        data.extend(vec![S::id(); size - n]);

        let mut seg = Self {
            n,
            log,
            size,
            data,
            lazy: vec![F::id(); size],
        };

        for i in (1..size).rev() {
            seg.up_product(i);
        }

        seg
    }
}

impl<S, F> LazySegtree<S, F>
where
    S: Monoid,
    F: Mapping<S>,
{
    pub fn new(n: usize) -> Self
    where
        S: Clone,
        F: Clone,
    {
        if n == 0 {
            return Self {
                n: 0,
                log: 0,
                size: 0,
                data: vec![],
                lazy: vec![],
            };
        }

        let log = usize::BITS - (n - 1).leading_zeros();
        let size = 1 << log;

        Self {
            n,
            log,
            size,
            data: vec![S::id(); 2 * size],
            lazy: vec![F::id(); size],
        }
    }

    fn get_bounds<R>(&self, rng: R) -> (usize, usize)
    where
        R: RangeBounds<usize>,
    {
        let l = match rng.start_bound() {
            std::ops::Bound::Included(&start_bound) => start_bound,
            std::ops::Bound::Excluded(&start_bound) => start_bound + 1,
            std::ops::Bound::Unbounded => 0,
        };

        let r = match rng.end_bound() {
            std::ops::Bound::Included(&end_bound) => end_bound + 1,
            std::ops::Bound::Excluded(&end_bound) => end_bound,
            std::ops::Bound::Unbounded => self.n,
        };

        (l, r)
    }

    fn up_product(&mut self, k: usize) {
        assert!(1 <= k && k < self.size);

        self.data[k] = self.data[2 * k].product(&self.data[2 * k + 1]);
    }

    fn down_composite(&mut self, k: usize) {
        assert!(1 <= k && k < self.size);

        let left_child = 2 * k;
        self.data[left_child] = self.lazy[k].mapping(&self.data[left_child]);

        let right_child = left_child + 1;
        self.data[right_child] = self.lazy[k].mapping(&self.data[right_child]);

        if left_child < self.size {
            self.lazy[left_child] = self.lazy[left_child].product(&self.lazy[k]);
            self.lazy[right_child] = self.lazy[right_child].product(&self.lazy[k]);
        }

        self.lazy[k] = F::id();
    }

    fn composite(&mut self, p: usize) {
        assert!(p < self.n);

        let p = p + self.size;

        for i in (1..=self.log).rev() {
            self.down_composite(p >> i);
        }
    }

    pub fn get(&mut self, p: usize) -> &S {
        self.composite(p);

        &self.data[p + self.size]
    }

    pub fn set(&mut self, p: usize, x: S) {
        self.composite(p);

        let p = p + self.size;

        self.data[p] = x;
        for i in 1..=self.log {
            self.up_product(p >> i);
        }
    }

    pub fn apply(&mut self, p: usize, f: &F) {
        self.composite(p);

        let p = p + self.size;

        self.data[p] = f.mapping(&self.data[p]);
        for i in 1..=self.log {
            self.up_product(p >> i);
        }
    }

    fn composite_range<R>(&mut self, rng: R)
    where
        R: RangeBounds<usize>,
    {
        let (mut left, mut right) = self.get_bounds(rng);

        assert!(left <= right && right <= self.n);

        left += self.size;
        right += self.size;

        for i in (left.trailing_zeros() + 1..=self.log).rev() {
            self.down_composite(left >> i);
        }

        for i in (right.trailing_zeros() + 1..=self.log).rev() {
            self.down_composite((right - 1) >> i);
        }
    }

    pub fn product_range<R>(&mut self, rng: R) -> S
    where
        R: RangeBounds<usize>,
    {
        let (left, right) = self.get_bounds(rng);

        self.composite_range(left..right);

        let mut left_product = S::id();
        let mut right_product = S::id();

        let mut left = left + self.size;
        let mut right = right + self.size;

        while left < right {
            if left & 1 == 1 {
                left_product = left_product.product(&self.data[left]);
                left += 1;
            }

            if right & 1 == 1 {
                right -= 1;
                right_product = self.data[right].product(&right_product);
            }

            left >>= 1;
            right >>= 1;
        }

        left_product.product(&right_product)
    }

    pub fn product_all(&self) -> &S {
        &self.data[1]
    }

    pub fn apply_segment(&mut self, k: usize, f: &F) {
        self.data[k] = f.mapping(&self.data[k]);
        if k < self.size {
            self.lazy[k] = self.lazy[k].product(f);
        }
    }

    pub fn apply_range<R>(&mut self, rng: R, f: &F)
    where
        R: RangeBounds<usize>,
    {
        let (mut left, mut right) = self.get_bounds(rng);

        self.composite_range(left..right);

        left += self.size;
        right += self.size;

        {
            let mut left = left;
            let mut right = right;

            while left < right {
                if left & 1 == 1 {
                    self.apply_segment(left, &f);
                    left += 1;
                }

                if right & 1 == 1 {
                    right -= 1;
                    self.apply_segment(right, &f);
                }

                left >>= 1;
                right >>= 1;
            }
        }

        for i in left.trailing_zeros() + 1..=self.log {
            self.up_product(left >> i);
        }

        for i in right.trailing_zeros() + 1..=self.log {
            self.up_product((right - 1) >> i);
        }
    }

    pub fn apply_all(&mut self, f: &F) {
        self.apply_segment(1, &f);
    }

    pub fn search_right_boundary<G>(&mut self, left: usize, g: G) -> usize
    where
        G: Fn(&S) -> bool,
    {
        assert!(left <= self.n);

        assert!(
            g(&S::id()),
            "The identity element must be mapped to `true` by `g`."
        );

        if left == self.n {
            return self.n;
        }

        let mut left = left + self.size;

        for i in (1..=self.log).rev() {
            self.down_composite(left >> i);
        }

        let mut prod = S::id();

        loop {
            left >>= left.trailing_zeros();

            let cand_prod = prod.product(&self.data[left]);

            if !g(&cand_prod) {
                break;
            }

            prod = cand_prod;
            left += 1;

            if left.is_power_of_two() {
                return self.n;
            }
        }

        while left < self.size {
            self.down_composite(left);

            left *= 2;

            let cand_prod = prod.product(&self.data[left]);
            if g(&cand_prod) {
                prod = cand_prod;
                left += 1;
            }
        }

        left - self.size
    }

    pub fn search_left_boundary<G>(&mut self, right: usize, g: G) -> usize
    where
        G: Fn(&S) -> bool,
    {
        assert!(right <= self.n);

        assert!(
            &g(&S::id()),
            "The identity element must be mapped to `true` by `g`."
        );

        let mut right = right + self.size;

        for i in (1..=self.log).rev() {
            self.down_composite((right - 1) >> i);
        }

        let mut prod = S::id();

        loop {
            right >>= right.trailing_zeros();

            let cand_prod = prod.product(&self.data[right - 1]);

            if !g(&cand_prod) {
                break;
            }

            prod = cand_prod;
            right -= 1;

            if right.is_power_of_two() {
                return 0;
            }
        }

        while right < self.size {
            self.down_composite(right - 1);

            right *= 2;

            let cand_prod = prod.product(&self.data[right - 1]);
            if g(&cand_prod) {
                prod = cand_prod;
                right -= 1;
            }
        }

        right - 1 - self.size
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test() {
        type Val = i64;

        #[derive(Debug, Clone, Copy)]
        struct S {
            min: Option<Val>,
            max: Option<Val>,
            sum: Val,
            size: usize,
        }

        impl Monoid for S {
            fn id() -> Self {
                Self {
                    min: None,
                    max: None,
                    sum: 0,
                    size: 0,
                }
            }

            fn product(&self, rhs: &Self) -> Self {
                let min = if self.min.is_some() && rhs.min.is_some() {
                    self.min.min(rhs.min)
                } else {
                    self.min.or(rhs.min)
                };

                let max = if self.max.is_some() && rhs.max.is_some() {
                    self.max.max(rhs.max)
                } else {
                    self.max.or(rhs.max)
                };

                Self {
                    min,
                    max,
                    sum: self.sum + rhs.sum,
                    size: self.size + rhs.size,
                }
            }
        }

        impl S {
            fn new(val: Val) -> Self {
                Self {
                    min: Some(val),
                    max: Some(val),
                    sum: val,
                    size: 1,
                }
            }
        }

        #[derive(Debug, Clone, Copy)]
        struct F {
            add: Val,
        }

        impl Monoid for F {
            fn id() -> Self {
                Self { add: 0 }
            }

            fn product(&self, rhs: &Self) -> Self {
                Self {
                    add: self.add + rhs.add,
                }
            }
        }

        impl Mapping<S> for F {
            fn mapping(&self, s: &S) -> S {
                S {
                    min: s.min.map(|min| min + self.add),
                    max: s.max.map(|max| max + self.add),
                    sum: s.sum + s.size as Val + self.add,
                    size: s.size,
                }
            }
        }

        let seq = vec![-3, -1, -4, -1, 0, 5, 9, 2, 6, 5];
        let value: Vec<S> = seq.iter().map(|&x| S::new(x)).collect();
        let lazy_segtree = LazySegtree::<S, F>::from(value);

        assert_eq!(lazy_segtree.product_all().min, seq.iter().min().cloned());
        assert_eq!(lazy_segtree.product_all().max, seq.iter().max().cloned());
        assert_eq!(lazy_segtree.product_all().sum, seq.iter().sum());
        assert_eq!(lazy_segtree.product_all().size, seq.len());
    }
}
