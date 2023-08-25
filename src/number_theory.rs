pub fn gcd(a: i64, b: i64) -> i64 {
    let mut a = a.abs();
    let mut b = b.abs();

    while b != 0 {
        let r = a % b;
        a = b;
        b = r;
    }

    a
}

pub fn lcm(a: i64, b: i64) -> i64 {
    a / gcd(a, b) * b
}

/// Returns a tuple of `(x, y)` and `gcd(a, b)` that satisfy `ax + by = gcd(a, b)` in that order.
///
/// The returned `x`, `y` and `gcd(a, b)` satisfy the following:
///   * Both `|x|` and `|y|` are less than or equal to `max(|a|, |b|)`.
///   * `gcd(a, b)` is non-negative.
///
/// # Examples
///
/// ```
/// # use atcoder8_library::number_theory::ext_gcd;
///
/// let (x, y, g) = ext_gcd(6, 10);
///
/// assert_eq!(g, 2);
/// assert_eq!(6 * x + 10 * y, 2);
/// assert!(x.abs() <= 10 && y.abs() <= 10);
/// ```
pub fn ext_gcd(mut a: i64, mut b: i64) -> (i64, i64, i64) {
    if a == 0 && b == 0 {
        return (0, 0, 0);
    }

    let (mut s, mut t, mut u, mut v) = (1, 0, 0, 1);
    while b != 0 {
        (a, b, s, t, u, v) = (b, a % b, t, s - a / b * t, v, u - a / b * v);
    }

    let sgn = a.signum();
    (sgn * s, sgn * u, sgn * a)
}

pub fn naive_ext_gcd(mut a: i64, mut b: i64) -> (i64, i64, i64) {
    let mut quotients = vec![];
    while b != 0 {
        quotients.push(a / b);
        (a, b) = (b, a % b);
    }

    let (mut x, mut y) = (1, 0);
    for &q in quotients.iter().rev() {
        (x, y) = (y, x - q * y);
    }

    let sgn = a.signum();
    (sgn * x, sgn * y, sgn * a)
}

pub fn crt(rr: &Vec<i64>, mm: &Vec<i64>) -> Option<(i64, i64)> {
    assert_eq!(rr.len(), mm.len());
    assert!(!rr.is_empty());
    assert!(mm.iter().all(|&m| m >= 1));

    let mut acc_r = 0;
    let mut acc_m = 1;

    for (&r, &m) in rr.iter().zip(mm.iter()) {
        let diff_r = r - acc_r;

        let (x, _y, d) = ext_gcd(acc_m, m);

        if diff_r % d != 0 {
            return None;
        }

        let next_acc_m = acc_m / d * m;
        acc_r = (acc_m * diff_r / d * x + acc_r) % next_acc_m;
        acc_m = next_acc_m;
    }

    if acc_r < 0 {
        acc_r += acc_m;
    }

    Some((acc_r, acc_m))
}

#[cfg(test)]
mod tests {
    use num_integer::Integer;

    use super::*;

    fn check_validity_ext_gcd(a: i64, b: i64) {
        let (x, y, g) = ext_gcd(a, b);

        assert_eq!(g, a.gcd(&b));
        assert_eq!(a * x + b * y, g);
        assert!(x.abs().max(y.abs()) <= a.abs().max(b.abs()));

        let boundary = a.abs().max(b.abs());
        assert!(x.abs() <= boundary && y.abs() <= boundary);
    }

    #[test]
    fn test_ext_gcd() {
        for a in -100..=100 {
            for b in -100..=100 {
                check_validity_ext_gcd(a, b);
            }
        }
    }
}
