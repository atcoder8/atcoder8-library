pub type Coord = (usize, usize);

pub trait Grid {
    fn shape(&self) -> (usize, usize);

    fn height(&self) -> usize {
        self.shape().0
    }

    fn width(&self) -> usize {
        self.shape().1
    }

    fn cell_num(&self) -> usize {
        self.height() * self.width()
    }

    fn in_range(&self, coord: Coord) -> bool {
        coord.0 < self.height() && coord.1 < self.width()
    }

    fn coord_to_idx(&self, coord: Coord) -> usize {
        assert!(self.in_range(coord));

        self.width() * coord.0 + coord.1
    }

    fn idx_to_coord(&self, idx: usize) -> Coord {
        assert!(idx < self.cell_num());

        (idx / self.width(), idx % self.width())
    }

    fn neighbor_coords(&self, diffs: &[Coord], coord: Coord) -> Vec<Coord> {
        let (row, col) = coord;

        diffs
            .iter()
            .filter_map(|&(diff_row, diff_col)| {
                let coord = (row.wrapping_add(diff_row), col.wrapping_add(diff_col));
                if self.in_range(coord) {
                    Some(coord)
                } else {
                    None
                }
            })
            .collect()
    }

    fn four_neighbor_coords(&self, coord: Coord) -> Vec<Coord> {
        const DIFFS: [Coord; 4] = [(!0, 0), (0, !0), (0, 1), (1, 0)];

        self.neighbor_coords(&DIFFS, coord)
    }

    fn eight_neighbor_coords(&self, coord: Coord) -> Vec<Coord> {
        const DIFFS: [Coord; 8] = [
            (!0, !0),
            (!0, 0),
            (!0, 1),
            (0, !0),
            (0, 1),
            (1, !0),
            (1, 0),
            (1, 1),
        ];

        self.neighbor_coords(&DIFFS, coord)
    }
}
