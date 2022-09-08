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
}
