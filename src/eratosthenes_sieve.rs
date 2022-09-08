//! Implements the Sieve of Eratosthenes.
//!
//! Finds the smallest prime factor of each number placed on the sieve,
//! so it can perform Prime Factorization as well as Primality Test.

#[derive(Debug, Clone)]
pub struct EratosthenesSieve {
    sieve: Vec<usize>,
}

impl EratosthenesSieve {
    /// Constructs a Sieve of Eratosthenes.
    ///
    /// # Arguments
    ///
    /// * `upper_limit` - The largest number placed on the sieve.
    ///
    /// # Examples
    ///
    /// ```
    /// use atcoder8_library::eratosthenes_sieve::EratosthenesSieve;
    ///
    /// let sieve = EratosthenesSieve::new(27);
    /// assert_eq!(sieve.prime_factorization(12), vec![(2, 2), (3, 1)]);
    /// ```
    pub fn new(upper_limit: usize) -> Self {
        let mut sieve: Vec<usize> = (0..=upper_limit).collect();

        for p in (2..).take_while(|&i| i * i <= upper_limit) {
            if sieve[p] != p {
                continue;
            }

            for i in ((p * p)..=upper_limit).step_by(p) {
                if sieve[i] == i {
                    sieve[i] = p;
                }
            }
        }

        Self { sieve }
    }

    /// Returns the least divisor of `n`.
    ///
    /// # Examples
    ///
    /// ```
    /// use atcoder8_library::eratosthenes_sieve::EratosthenesSieve;
    ///
    /// let sieve = EratosthenesSieve::new(27);
    /// assert_eq!(sieve.min_divisor(1), 1);
    /// assert_eq!(sieve.min_divisor(2), 2);
    /// assert_eq!(sieve.min_divisor(6), 2);
    /// assert_eq!(sieve.min_divisor(11), 11);
    /// assert_eq!(sieve.min_divisor(27), 3);
    /// ```
    pub fn min_divisor(&self, n: usize) -> usize {
        self.sieve[n]
    }

    /// Determines if `n` is prime.
    ///
    /// # Examples
    ///
    /// ```
    /// use atcoder8_library::eratosthenes_sieve::EratosthenesSieve;
    ///
    /// let sieve = EratosthenesSieve::new(27);
    /// assert!(!sieve.is_prime(1));
    /// assert!(sieve.is_prime(2));
    /// assert!(!sieve.is_prime(6));
    /// assert!(sieve.is_prime(11));
    /// assert!(!sieve.is_prime(27));
    /// ```
    pub fn is_prime(&self, n: usize) -> bool {
        n >= 2 && self.sieve[n] == n
    }

    /// Performs prime factorization of `n`.
    ///
    /// The result of the prime factorization is returned as a
    /// list of prime factor and exponent pairs.
    ///
    /// # Examples
    ///
    /// ```
    /// use atcoder8_library::eratosthenes_sieve::EratosthenesSieve;
    ///
    /// let sieve = EratosthenesSieve::new(27);
    /// assert_eq!(sieve.prime_factorization(1), vec![]);
    /// assert_eq!(sieve.prime_factorization(12), vec![(2, 2), (3, 1)]);
    /// assert_eq!(sieve.prime_factorization(19), vec![(19, 1)]);
    /// assert_eq!(sieve.prime_factorization(27), vec![(3, 3)]);
    /// ```
    pub fn prime_factorization(&self, n: usize) -> Vec<(usize, usize)> {
        let mut factors: Vec<(usize, usize)> = vec![];
        let mut t = n;

        while t != 1 {
            let p = self.sieve[t];

            if factors.is_empty() || factors.last().unwrap().0 != p {
                factors.push((p, 1));
            } else {
                factors.last_mut().unwrap().1 += 1;
            }

            t /= p;
        }

        factors
    }

    /// Creates a list of divisors of `n`.
    ///
    /// The divisors are listed in ascending order.
    ///
    /// # Examples
    ///
    /// ```
    /// use atcoder8_library::eratosthenes_sieve::EratosthenesSieve;
    ///
    /// let sieve = EratosthenesSieve::new(27);
    /// assert_eq!(sieve.create_divisors_list(1), vec![1]);
    /// assert_eq!(sieve.create_divisors_list(12), vec![1, 2, 3, 4, 6, 12]);
    /// assert_eq!(sieve.create_divisors_list(19), vec![1, 19]);
    /// assert_eq!(sieve.create_divisors_list(27), vec![1, 3, 9, 27]);
    /// ```
    pub fn create_divisors_list(&self, n: usize) -> Vec<usize> {
        let mut divisors = vec![];
        let prime_factors = self.prime_factorization(n);
        let mut stack = vec![(vec![0; prime_factors.len()], 0, 1)];

        while let Some((exps, skip, d)) = stack.pop() {
            divisors.push(d);

            for (i, &(p, e)) in prime_factors.iter().enumerate().skip(skip) {
                if exps[i] < e {
                    let mut next_exps = exps.clone();
                    next_exps[i] += 1;

                    stack.push((next_exps, i, d * p));
                }
            }
        }

        divisors.sort_unstable();

        divisors
    }
}
