//! Library for geometry algorithm.

/// xy平面上の2つの格子点の偏角(x軸正方向を基準とした反時計回りの角度)を比較します。
pub fn compare_by_angle(point1: (i64, i64), point2: (i64, i64)) -> std::cmp::Ordering {
    let is_upper_half = |point: (i64, i64)| point.1 > 0 || (point.1 == 0 && point.0 > 0);
    match (is_upper_half(point1), is_upper_half(point2)) {
        (true, false) => std::cmp::Ordering::Less,
        (false, true) => std::cmp::Ordering::Greater,
        _ => (point2.0 * point1.1).cmp(&(point1.0 * point2.1)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compare_by_angle() {
        let mut points = [
            (2, -2),
            (-2, 1),
            (1, 1),
            (1, -3),
            (-3, 0),
            (2, 0),
            (2, 2),
            (-2, 2),
            (1, 2),
            (-2, -2),
            (0, 2),
            (-3, 1),
            (-1, -2),
            (2, -1),
            (-3, -3),
            (1, -1),
            (1, 0),
            (-3, 2),
            (0, -1),
            (0, -3),
            (-1, -1),
            (0, 1),
            (-2, -1),
            (-3, -1),
            (-3, -2),
            (-1, 0),
            (-1, 1),
            (-1, -3),
            (2, 1),
            (2, -3),
            (-1, 2),
            (-2, 0),
            (1, -2),
            (-2, -3),
            (0, -2),
        ];
        points.sort_by(|&point1, &point2| compare_by_angle(point1, point2));

        let expected = [
            (2, 0),
            (1, 0),
            (2, 1),
            (1, 1),
            (2, 2),
            (1, 2),
            (0, 2),
            (0, 1),
            (-1, 2),
            (-2, 2),
            (-1, 1),
            (-3, 2),
            (-2, 1),
            (-3, 1),
            (-3, 0),
            (-1, 0),
            (-2, 0),
            (-3, -1),
            (-2, -1),
            (-3, -2),
            (-2, -2),
            (-3, -3),
            (-1, -1),
            (-2, -3),
            (-1, -2),
            (-1, -3),
            (0, -1),
            (0, -3),
            (0, -2),
            (1, -3),
            (1, -2),
            (2, -3),
            (2, -2),
            (1, -1),
            (2, -1),
        ];
        assert_eq!(points, expected);
    }
}
