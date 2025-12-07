fn gcd(a: usize, b: usize) -> usize {
    let mut a = a;
    let mut b = b;

    while b != 0 {
        let t = a % b;
        a = b;
        b = t;
    }

    a
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Amplitude {
    pub x: usize,
    pub y: usize,
}

impl PartialOrd for Amplitude {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Amplitude {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        (other.x * self.y).cmp(&(self.x * other.y))
    }
}

impl Amplitude {
    pub fn new(x: usize, y: usize) -> Self {
        assert_ne!((x, y), (0, 0));

        let g = gcd(x, y);

        Self { x: x / g, y: y / g }
    }
}
