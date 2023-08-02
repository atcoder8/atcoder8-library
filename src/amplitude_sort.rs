use std::cmp::Ordering;

type Coord = (i64, i64);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Quadrant {
    Origin,
    XAxisPositive,
    One,
    YAxisPositive,
    Two,
    XAxisNegative,
    Three,
    YAxisNegative,
    Four,
}

impl PartialOrd for Quadrant {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        if self == &Quadrant::Origin || other == &Quadrant::Origin {
            return None;
        }

        self.to_index().partial_cmp(&other.to_index())
    }
}

impl Quadrant {
    fn determine(coord: Coord) -> Self {
        use Quadrant::*;

        let (x, y) = coord;

        if x == 0 && y == 0 {
            return Origin;
        }

        if x == 0 {
            return if y > 0 { YAxisPositive } else { YAxisNegative };
        }

        if y == 0 {
            return if x > 0 { XAxisPositive } else { XAxisNegative };
        }

        return if x > 0 {
            if y > 0 {
                One
            } else {
                Four
            }
        } else {
            if y > 0 {
                Two
            } else {
                Three
            }
        };
    }

    fn to_index(self) -> u8 {
        match self {
            Quadrant::Origin => 255,
            Quadrant::XAxisPositive => 0,
            Quadrant::One => 1,
            Quadrant::YAxisPositive => 2,
            Quadrant::Two => 3,
            Quadrant::XAxisNegative => 4,
            Quadrant::Three => 5,
            Quadrant::YAxisNegative => 6,
            Quadrant::Four => 7,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Amplitude {
    quadrant: Quadrant,
    x: i64,
    y: i64,
}

impl PartialOrd for Amplitude {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        match self.quadrant.partial_cmp(&other.quadrant) {
            Some(core::cmp::Ordering::Equal) => {}
            ord => return ord,
        }

        match (self.y * other.x).partial_cmp(&(other.y * self.x)) {
            Some(core::cmp::Ordering::Equal) => {}
            ord => return ord,
        }

        let self_sq_dist = self.x.pow(2) + self.y.pow(2);
        let other_sq_dist = other.x.pow(2) + other.y.pow(2);

        return self_sq_dist.partial_cmp(&other_sq_dist);
    }
}

impl Ord for Amplitude {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap()
    }
}

impl Amplitude {
    fn new(coord: Coord) -> Self {
        assert_ne!(coord, (0, 0));

        Self {
            quadrant: Quadrant::determine(coord),
            x: coord.0,
            y: coord.1,
        }
    }
}

pub fn amplitude_sort(coords: &mut Vec<Coord>) {
    coords.sort_by_cached_key(|coord| Amplitude::new(*coord));
}

#[cfg(test)]
mod tests {
    use rand::seq::SliceRandom;

    use super::*;

    fn calc_sq_dist(coord: Coord) -> i64 {
        coord.0.pow(2) + coord.1.pow(2)
    }

    fn comp_amplitude(coord1: Coord, coord2: Coord) -> Ordering {
        let (x1, y1) = (coord1.0 as f64, coord1.1 as f64);
        let (x2, y2) = (coord2.0 as f64, coord2.1 as f64);

        let mut rad1 = y1.atan2(x1);
        if rad1 < 0.0 {
            rad1 += 2.0 * std::f64::consts::PI;
        }

        let mut rad2 = y2.atan2(x2);
        if rad2 < 0.0 {
            rad2 += 2.0 * std::f64::consts::PI;
        }

        match rad1.partial_cmp(&rad2).unwrap() {
            Ordering::Equal => {}
            ord => return ord,
        }

        calc_sq_dist(coord1).cmp(&calc_sq_dist(coord2))
    }

    #[test]
    fn test() {
        let mut rng = rand::thread_rng();

        let mut coords = vec![];
        for x in -354..=354 {
            for y in -354..=354 {
                if x == 0 && y == 0 {
                    continue;
                }

                coords.push((x, y));
            }
        }
        coords.shuffle(&mut rng);

        amplitude_sort(&mut coords);

        let is_sorted = coords
            .windows(2)
            .all(|window| comp_amplitude(window[0], window[1]) != Ordering::Greater);
        assert!(is_sorted);
    }
}
