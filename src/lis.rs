use superslice::Ext;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WeeklyLIS<T>
where
    T: Clone + Ord,
{
    dp: Vec<T>,
}

impl<T> WeeklyLIS<T>
where
    T: Clone + Ord,
{
    pub fn new() -> Self {
        Self { dp: vec![] }
    }

    pub fn push(&mut self, x: T) {
        let idx = self.dp.upper_bound(&x);
        if idx < self.dp.len() {
            self.dp[idx] = x;
        } else {
            self.dp.push(x);
        }
    }

    pub fn lis_len(&self) -> usize {
        self.dp.len()
    }
}

impl<T> Default for WeeklyLIS<T>
where
    T: Clone + Ord,
{
    fn default() -> Self {
        WeeklyLIS::new()
    }
}

impl<T> From<Vec<T>> for WeeklyLIS<T>
where
    T: Clone + Ord,
{
    fn from(seq: Vec<T>) -> Self {
        let mut lis = Self::default();
        for x in seq {
            lis.push(x);
        }
        lis
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StronglyLIS<T>
where
    T: Clone + Ord,
{
    dp: Vec<T>,
}

impl<T> StronglyLIS<T>
where
    T: Clone + Ord,
{
    pub fn new() -> Self {
        Self { dp: vec![] }
    }

    pub fn push(&mut self, x: T) {
        let idx = self.dp.lower_bound(&x);
        if idx < self.dp.len() {
            self.dp[idx] = x;
        } else {
            self.dp.push(x);
        }
    }

    pub fn lis_len(&self) -> usize {
        self.dp.len()
    }
}

impl<T> Default for StronglyLIS<T>
where
    T: Clone + Ord,
{
    fn default() -> Self {
        StronglyLIS::new()
    }
}

impl<T> From<Vec<T>> for StronglyLIS<T>
where
    T: Clone + Ord,
{
    fn from(seq: Vec<T>) -> Self {
        let mut lis = Self::default();
        for x in seq {
            lis.push(x);
        }
        lis
    }
}

#[cfg(test)]
mod tests {
    use std::fmt::Debug;

    use itertools::Itertools;
    use rand::Rng;

    use crate::lis::StronglyLIS;

    use super::WeeklyLIS;

    fn create_expected_lis<T>(seq: &Vec<T>, strongly: bool) -> Vec<T>
    where
        T: Clone + Ord,
    {
        let n = seq.len();
        assert!(n <= 30);

        let cmp: fn(x: &[&T]) -> bool = if strongly {
            |x| x[0] < x[1]
        } else {
            |x| x[0] <= x[1]
        };

        let mut lis = vec![];

        for sub_seq_len in 1..(n + 1) {
            let mut min_comb: Option<Vec<&T>> = None;

            for sub_seq in seq.iter().combinations(sub_seq_len) {
                if sub_seq.windows(2).all(cmp) {
                    if let Some(x) = min_comb {
                        min_comb = Some(x.min(sub_seq));
                    } else {
                        min_comb = Some(sub_seq);
                    }
                }
            }

            if let Some(x) = min_comb {
                lis = x;
            } else {
                break;
            }
        }

        lis.into_iter().map(|x| x.clone()).collect()
    }

    fn gen_random_seq() -> Vec<usize> {
        let mut rng = rand::thread_rng();

        let n = rng.gen_range(0..10);

        (0..n).map(|_| rng.gen_range(0..10)).collect()
    }

    fn create_debug_message<T>(
        test_case_number: usize,
        seq: &Vec<T>,
        expected_lis: &Vec<T>,
        actual_lis_length: usize,
    ) -> String
    where
        T: Debug,
    {
        format!(
            "
Wrong Answer on Test #{}

[Input sequence]
{:?}

[Correct LIS]
{:?}

[Expected LIS length]
{}

[Actual LIS length]
{}
",
            test_case_number,
            seq,
            expected_lis,
            expected_lis.len(),
            actual_lis_length,
        )
    }

    #[test]
    fn weekly_random_test() {
        const NUMBER_OF_TESTS: usize = 1000;

        for test_case_number in 0..NUMBER_OF_TESTS {
            let seq = gen_random_seq();

            let expected_lis = create_expected_lis(&seq, false);

            let lis = WeeklyLIS::from(seq.clone());

            assert_eq!(
                lis.lis_len(),
                expected_lis.len(),
                "{}",
                create_debug_message(test_case_number, &seq, &expected_lis, lis.lis_len())
            );
        }
    }

    #[test]
    fn weekly_empty_test() {
        let lis = WeeklyLIS::<i32>::default();
        assert_eq!(lis.lis_len(), 0);
    }

    #[test]
    fn weekly_same_test() {
        let lis = WeeklyLIS::from(vec![1, 1, 1, 1, 1]);
        assert_eq!(lis.lis_len(), 5);
    }

    #[test]
    fn weekly_increasing_test() {
        let lis = WeeklyLIS::from(vec![0, 1, 2, 5, 7, 10]);
        assert_eq!(lis.lis_len(), 6);
    }

    #[test]
    fn weekly_decreasing_test() {
        let lis = WeeklyLIS::from(vec![10, 7, 5, 2, 1, 0]);
        assert_eq!(lis.lis_len(), 1);
    }

    #[test]
    fn strongly_random_test() {
        const NUMBER_OF_TESTS: usize = 1000;

        for test_case_number in 0..NUMBER_OF_TESTS {
            let seq = gen_random_seq();

            let expected_lis = create_expected_lis(&seq, true);

            let lis = StronglyLIS::from(seq.clone());

            assert_eq!(
                lis.lis_len(),
                expected_lis.len(),
                "{}",
                create_debug_message(test_case_number, &seq, &expected_lis, lis.lis_len())
            );
        }
    }

    #[test]
    fn strongly_empty_test() {
        let lis = StronglyLIS::<i32>::default();
        assert_eq!(lis.lis_len(), 0);
    }

    #[test]
    fn strongly_same_test() {
        let lis = StronglyLIS::from(vec![1, 1, 1, 1, 1]);
        assert_eq!(lis.lis_len(), 1);
    }

    #[test]
    fn strongly_increasing_test() {
        let lis = StronglyLIS::from(vec![0, 1, 2, 5, 7, 10]);
        assert_eq!(lis.lis_len(), 6);
    }

    #[test]
    fn strongly_decreasing_test() {
        let lis = StronglyLIS::from(vec![10, 7, 5, 2, 1, 0]);
        assert_eq!(lis.lis_len(), 1);
    }
}
