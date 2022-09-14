/// Creates a list of divisors of `n`.
///
/// The divisors are listed in ascending order.
///
/// # Examples
///
/// ```
/// use atcoder8_library::functions::create_divisors_list;
///
/// assert_eq!(create_divisors_list(1), vec![1]);
/// assert_eq!(create_divisors_list(12), vec![1, 2, 3, 4, 6, 12]);
/// assert_eq!(create_divisors_list(19), vec![1, 19]);
/// assert_eq!(create_divisors_list(27), vec![1, 3, 9, 27]);
/// ```
pub fn create_divisors_list(n: usize) -> Vec<usize> {
    assert_ne!(n, 0, "`n` must be at least 1.");

    let mut divisors: Vec<usize> = (1..)
        .take_while(|&i| i * i <= n)
        .filter(|&i| n % i == 0)
        .collect();

    let mut iter = divisors.iter().rev();
    if divisors.last().unwrap().pow(2) == n {
        iter.next();
    }

    let mut add_divisors: Vec<usize> = iter.map(|&d| n / d).collect();

    divisors.append(&mut add_divisors);

    divisors
}
