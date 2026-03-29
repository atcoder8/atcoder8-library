fn gcd(a: i64, b: i64) -> i64 {
    let mut a = a.abs();
    let mut b = b.abs();

    while b != 0 {
        let t = a % b;
        a = b;
        b = t;
    }

    a
}

/// xy平面上の点について、x軸正方向を基準とした半時計回りの偏角を既約な整数比で表します。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Amplitude {
    x: i64,
    y: i64,
}

impl PartialOrd for Amplitude {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Amplitude {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        let is_upper_half =
            |amplitude: Amplitude| amplitude.y > 0 || (amplitude.y == 0 && amplitude.x > 0);

        match (is_upper_half(*self), is_upper_half(*other)) {
            (true, false) => std::cmp::Ordering::Less,
            (false, true) => std::cmp::Ordering::Greater,
            _ => (other.x * self.y).cmp(&(self.x * other.y)),
        }
    }
}

impl Amplitude {
    pub fn new(x: i64, y: i64) -> Self {
        assert_ne!((x, y), (0, 0), "原点の偏角は定義されません。");

        let g = gcd(x, y);
        Self { x: x / g, y: y / g }
    }

    pub fn from_tuple(tuple: (i64, i64)) -> Self {
        Self::new(tuple.0, tuple.1)
    }

    pub fn x(self) -> i64 {
        self.x
    }

    pub fn y(self) -> i64 {
        self.y
    }

    pub fn fraction(self) -> (i64, i64) {
        (self.x, self.y)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[should_panic]
    fn test_origin() {
        Amplitude::new(0, 0);
    }

    #[test]
    fn test_reduction() {
        assert_eq!(Amplitude::new(2, 0), Amplitude::new(1, 0));
        assert_eq!(Amplitude::new(-4, 6), Amplitude::new(-2, 3));
    }

    #[test]
    fn test_ordering() {
        let fractions = [
            (1, 1),
            (0, -1),
            (-1, 0),
            (1, 2),
            (1, 0),
            (0, 1),
            (1, -1),
            (-1, -1),
            (-1, 1),
            (1, 1),
        ];
        let mut amplitudes = fractions.map(Amplitude::from_tuple);
        amplitudes.sort();

        let expected = [
            (1, 0),
            (1, 1),
            (1, 1),
            (1, 2),
            (0, 1),
            (-1, 1),
            (-1, 0),
            (-1, -1),
            (0, -1),
            (1, -1),
        ];
        let expected_amplitudes = expected.map(Amplitude::from_tuple);
        assert_eq!(amplitudes, expected_amplitudes);
    }
}
