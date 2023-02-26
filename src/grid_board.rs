pub type Coord = (usize, usize);

pub trait GridBoard {
    fn get_size(&self) -> (usize, usize);

    fn get_h(&self) -> usize {
        self.get_size().0
    }

    fn get_w(&self) -> usize {
        self.get_size().1
    }

    fn get_area(&self) -> usize {
        let (h, w) = self.get_size();

        h * w
    }

    fn in_range(&self, coord: Coord) -> bool {
        let (h, w) = self.get_size();

        coord.0 < h && coord.1 < w
    }

    fn coord_to_idx(&self, coord: Coord) -> usize {
        debug_assert!(self.in_range(coord));

        self.get_size().1 * coord.0 + coord.1
    }

    fn idx_to_coord(&self, coord_idx: usize) -> Coord {
        debug_assert!(coord_idx < self.get_area());

        let w = self.get_w();

        (coord_idx / w, coord_idx % w)
    }

    fn get_neighborhood(&self, coord: Coord, diffs: &[Coord]) -> Vec<Coord> {
        let (h, w) = self.get_size();
        let (x, y) = coord;

        diffs
            .iter()
            .filter_map(|&(diff_x, diff_y)| {
                let nei_x = x.wrapping_add(diff_x);
                let nei_y = y.wrapping_add(diff_y);

                if nei_x < h && nei_y < w {
                    Some((nei_x, nei_y))
                } else {
                    None
                }
            })
            .collect()
    }

    fn get_4_neighborhood(&self, coord: Coord) -> Vec<Coord> {
        const DIFFS: [(usize, usize); 4] = [(!0, 0), (0, !0), (0, 1), (1, 0)];

        self.get_neighborhood(coord, &DIFFS)
    }

    fn get_8_neighborhood(&self, coord: Coord) -> Vec<Coord> {
        const DIFFS: [(usize, usize); 8] = [
            (!0, !0),
            (!0, 0),
            (!0, 1),
            (0, !0),
            (0, 1),
            (1, !0),
            (1, 0),
            (1, 1),
        ];

        self.get_neighborhood(coord, &DIFFS)
    }
}
