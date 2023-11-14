//! Library for number-theoretic algorithm.

/// Calculates the sum as shown in the following formula.
///
/// $\sum_{i=0}^{n-1} \left\lfloor \frac{ai+b}{m} \right\rfloor$
pub fn floor_sum(n: i64, m: i64, a: i64, b: i64) -> i64 {
    assert_ne!(m, 0, "`m` must not be zero.");

    if n == 0 {
        return 0;
    }

    let mut sum = 0;
    let (mut n, mut m, mut a, mut b) = (n as i128, m as i128, a as i128, b as i128);
    while n != 0 {
        let (qa, ra) = (a / m, a % m);
        let (qb, rb) = (b / m, b % m);
        let max_numer = ra * (n - 1) + rb;
        let max = max_numer / m;

        sum += n * (n - 1) / 2 * qa + n * qb + max;

        (n, m, a, b) = (max, ra, m, max_numer % m)
    }

    sum as i64
}
