use std::cmp::Ordering;

type Coord = (i64, i64);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Area {
    Origin,
    XAxisPositive,
    QuadrantOne,
    YAxisPositive,
    QuadrantTwo,
    XAxisNegative,
    QuadrantThree,
    YAxisNegative,
    QuadrantFour,
}

impl PartialOrd for Area {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        if let (Some(self_order), Some(other_order)) = (self.order(), other.order()) {
            self_order.partial_cmp(&other_order)
        } else {
            None
        }
    }
}

impl Area {
    fn determine(coord: Coord) -> Self {
        use Area::*;

        let (x, y) = coord;

        match (x == 0, y == 0) {
            (true, true) => return Origin,
            (true, false) => return if y > 0 { YAxisPositive } else { YAxisNegative },
            (false, true) => return if x > 0 { XAxisPositive } else { XAxisNegative },
            (false, false) => {}
        }

        match (x > 0, y > 0) {
            (true, true) => QuadrantOne,
            (false, true) => QuadrantTwo,
            (false, false) => QuadrantThree,
            (true, false) => QuadrantFour,
        }
    }

    fn order(self) -> Option<u8> {
        match self {
            Area::Origin => None,
            Area::XAxisPositive => Some(0),
            Area::QuadrantOne => Some(1),
            Area::YAxisPositive => Some(2),
            Area::QuadrantTwo => Some(3),
            Area::XAxisNegative => Some(4),
            Area::QuadrantThree => Some(5),
            Area::YAxisNegative => Some(6),
            Area::QuadrantFour => Some(7),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Amplitude {
    area: Area,
    x: i64,
    y: i64,
}

impl PartialOrd for Amplitude {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        match self.area.partial_cmp(&other.area) {
            Some(core::cmp::Ordering::Equal) => {}
            ord => return ord,
        }

        match (self.y * other.x).partial_cmp(&(other.y * self.x)) {
            Some(core::cmp::Ordering::Equal) => {}
            ord => return ord,
        }

        let self_sq_dist = self.x.pow(2) + self.y.pow(2);
        let other_sq_dist = other.x.pow(2) + other.y.pow(2);

        self_sq_dist.partial_cmp(&other_sq_dist)
    }
}

impl Ord for Amplitude {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap()
    }
}

impl Amplitude {
    fn new(coord: Coord) -> Self {
        assert_ne!(coord, (0, 0), "Amplitude of the origin is not defined.");

        Self {
            area: Area::determine(coord),
            x: coord.0,
            y: coord.1,
        }
    }
}

/// Sorts a list of coordinates in ascending order with respect to amplitude.
///
/// Coordinates with equal amplitude are sorted in ascending order with
/// respect to their distance from the origin.
///
/// # Examples
///
/// ```
/// use atcoder8_library::amplitude_sort::amplitude_sort;
///
/// let mut coords: Vec<(i64, i64)> = vec![
///     (0, -3), (4, -2), (2, -1), (-2, -2), (1, -4), (-2, 1),
///     (2, 2), (-2, 0), (-1, -4), (0, -2), (1, 0), (0, 1),
///     (-1, 3), (0, 2), (1, 1), (2, 0), (-1, 0), (3, 1),
/// ];
/// amplitude_sort(&mut coords);
///
/// assert_eq!(
///     coords,
///     vec![
///         (1, 0), (2, 0), (3, 1), (1, 1), (2, 2), (0, 1),
///         (0, 2), (-1, 3), (-2, 1), (-1, 0), (-2, 0), (-2, -2),
///         (-1, -4), (0, -2), (0, -3), (1, -4), (2, -1), (4, -2),
///     ]
/// );
/// ```
pub fn amplitude_sort(coords: &mut [Coord]) {
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
        for x in -100..=100 {
            for y in -100..=100 {
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
