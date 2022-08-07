use std::ops::{Div, Mul};

pub struct Factorial<T> {
    fac: Vec<T>,
}

impl<T> Default for Factorial<T>
where
    T: Clone + From<usize> + Mul<Output = T> + Div<Output = T>,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<T> Factorial<T>
where
    T: Clone + From<usize> + Mul<Output = T> + Div<Output = T>,
{
    /// Constructs a new, empty `Factorial<T>`.
    pub fn new() -> Self {
        Self {
            fac: vec![T::from(1)],
        }
    }

    /// Returns the factorial of `n`.
    pub fn factorial(&mut self, n: usize) -> T {
        if self.fac.len() < n + 1 {
            for i in (self.fac.len() - 1)..n {
                self.fac.push(self.fac[i].clone() * (i + 1).into());
            }
        }
        self.fac[n].clone()
    }

    /// Returns the number of choices when selecting `k` from `n` and arranging them in a row.
    pub fn permutations(&mut self, n: usize, k: usize) -> T {
        if n < k {
            T::from(0)
        } else {
            self.factorial(n) / self.factorial(n - k)
        }
    }

    /// Returns the number of choices to select `k` from `n`.
    pub fn combinations(&mut self, n: usize, k: usize) -> T {
        if n < k {
            T::from(0)
        } else {
            self.permutations(n, k) / self.factorial(k)
        }
    }

    /// Calculate the number of cases when sample of `k` elements from a set of `n` elements, allowing for duplicates.
    pub fn combinations_with_repetition(&mut self, n: usize, k: usize) -> T {
        if n == 0 {
            if k == 0 {
                T::from(1)
            } else {
                T::from(0)
            }
        } else {
            self.combinations(n + k - 1, k)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Factorial;
    use num::BigUint;
    use std::str::FromStr;

    #[test]
    fn test_factorial_0() {
        assert_eq!(Factorial::<usize>::new().factorial(0), 1);
    }

    #[test]
    fn test_factorial_1() {
        assert_eq!(Factorial::<usize>::new().factorial(1), 1);
    }

    #[test]
    fn test_factorial_10() {
        assert_eq!(Factorial::<usize>::new().factorial(10), 3628800);
    }

    #[test]
    fn test_factorial_100() {
        let factorial_100 = "93326215443944152681699238856266700490715968264381\
            62146859296389521759999322991560894146397615651828\
            62536979208272237582511852109168640000000000000000\
            00000000";
        let factorial_100 = BigUint::from_str(factorial_100).unwrap();
        assert_eq!(Factorial::<BigUint>::new().factorial(100), factorial_100);
    }

    #[test]
    fn test_permutations_0_0() {
        assert_eq!(Factorial::<usize>::new().permutations(0, 0), 1);
    }

    #[test]
    fn test_permutations_0_10() {
        assert_eq!(Factorial::<usize>::new().permutations(0, 10), 0);
    }

    #[test]
    fn test_permutations_3_10() {
        assert_eq!(Factorial::<usize>::new().permutations(3, 10), 0);
    }

    #[test]
    fn test_permutations_7_10() {
        assert_eq!(Factorial::<usize>::new().permutations(7, 10), 0);
    }

    #[test]
    fn test_permutations_10_0() {
        assert_eq!(Factorial::<usize>::new().permutations(10, 0), 1);
    }

    #[test]
    fn test_permutations_10_3() {
        assert_eq!(Factorial::<usize>::new().permutations(10, 3), 720);
    }

    #[test]
    fn test_permutations_10_7() {
        assert_eq!(Factorial::<usize>::new().permutations(10, 7), 604800);
    }

    #[test]
    fn test_permutations_10_10() {
        assert_eq!(Factorial::<usize>::new().permutations(10, 10), 3628800);
    }

    #[test]
    fn test_combinations_0_0() {
        assert_eq!(Factorial::<usize>::new().combinations(0, 0), 1);
    }

    #[test]
    fn test_combinations_0_10() {
        assert_eq!(Factorial::<usize>::new().combinations(0, 10), 0);
    }

    #[test]
    fn test_combinations_3_10() {
        assert_eq!(Factorial::<usize>::new().combinations(3, 10), 0);
    }

    #[test]
    fn test_combinations_7_10() {
        assert_eq!(Factorial::<usize>::new().combinations(7, 10), 0);
    }

    #[test]
    fn test_combinations_10_0() {
        assert_eq!(Factorial::<usize>::new().combinations(10, 0), 1);
    }

    #[test]
    fn test_combinations_10_3() {
        assert_eq!(Factorial::<usize>::new().combinations(10, 3), 120);
    }

    #[test]
    fn test_combinations_10_7() {
        assert_eq!(Factorial::<usize>::new().combinations(10, 7), 120);
    }

    #[test]
    fn test_combinations_10_10() {
        assert_eq!(Factorial::<usize>::new().combinations(10, 10), 1);
    }

    #[test]
    fn test_combinations_with_repetition_0_0() {
        assert_eq!(
            Factorial::<usize>::new().combinations_with_repetition(0, 0),
            1
        );
    }

    #[test]
    fn test_combinations_with_repetition_0_10() {
        assert_eq!(
            Factorial::<usize>::new().combinations_with_repetition(0, 10),
            0
        );
    }

    #[test]
    fn test_combinations_with_repetition_3_10() {
        assert_eq!(
            Factorial::<usize>::new().combinations_with_repetition(3, 10),
            66
        );
    }

    #[test]
    fn test_combinations_with_repetition_7_10() {
        assert_eq!(
            Factorial::<usize>::new().combinations_with_repetition(7, 10),
            8008
        );
    }

    #[test]
    fn test_combinations_with_repetition_10_0() {
        assert_eq!(
            Factorial::<usize>::new().combinations_with_repetition(10, 0),
            1
        );
    }

    #[test]
    fn test_combinations_with_repetition_10_3() {
        assert_eq!(
            Factorial::<usize>::new().combinations_with_repetition(10, 3),
            220
        );
    }

    #[test]
    fn test_combinations_with_repetition_10_7() {
        assert_eq!(
            Factorial::<usize>::new().combinations_with_repetition(10, 7),
            11440
        );
    }

    #[test]
    fn test_combinations_with_repetition_10_10() {
        assert_eq!(
            Factorial::<usize>::new().combinations_with_repetition(10, 10),
            92378
        );
    }
}
