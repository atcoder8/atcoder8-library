use fixedbitset::FixedBitSet;

/// Applies the sweep method to the boolean matrix and returns the rank.
///
/// When `extended` is non-zero, it is processed as an augmented matrix.
///
/// # Arguments
///
/// * mat - Matrix with boolean-valued elements.
/// * extended - Number of appended columns.
pub fn bit_sweep(mat: &mut [FixedBitSet], extended: usize) -> usize {
    if mat.is_empty() {
        return 0;
    }

    let (h, w) = (mat.len(), mat[0].len());

    assert!(
        w >= extended,
        "The number of columns in the entire matrix is less than the number of appended columns."
    );

    assert!(
        mat.iter().all(|x| x.len() == w),
        "Length of each bitset must be equal."
    );

    let mut rank = 0;
    for sweep_pos in 0..(w - extended) {
        let pivot = (rank..h).find(|&row| mat[row][sweep_pos]);
        if let Some(pivot) = pivot {
            mat.swap(rank, pivot);

            let pivot_bitset = mat[rank].clone();
            for (row, bitset) in mat.iter_mut().enumerate() {
                if row != rank && bitset[sweep_pos] {
                    bitset.symmetric_difference_with(&pivot_bitset);
                }
            }

            rank += 1;
        }
    }

    rank
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extended() {
        const N: usize = 5;

        // Matrix before processing
        // [[11100|0]
        //  [01110|0]
        //  [10110|1]
        //  [11010|0]
        //  [11001|1]]
        let mut mat: Vec<FixedBitSet> = [0b000111, 0b001110, 0b101101, 0b001011, 0b110011]
            .into_iter()
            .map(|bitset| FixedBitSet::with_capacity_and_blocks(6, [bitset]))
            .collect();

        let rank = bit_sweep(&mut mat, 1);
        assert_eq!(rank, N);

        // Expected matrix after processing
        // [[10000|1]
        //  [01000|0]
        //  [00100|1]
        //  [00010|1]
        //  [00001|0]]
        let expected: Vec<FixedBitSet> = [0b100001, 0b000010, 0b100100, 0b101000, 0b010000]
            .into_iter()
            .map(|bitset| FixedBitSet::with_capacity_and_blocks(6, [bitset]))
            .collect();
        assert_eq!(mat, expected);
    }

    #[test]
    fn test_square_rank_deficiency() {
        // Matrix before processing
        // [[000]
        //  [111]
        //  [000]]
        let mut mat: Vec<FixedBitSet> = [0b000, 0b111, 0b000]
            .into_iter()
            .map(|bitset| FixedBitSet::with_capacity_and_blocks(3, [bitset]))
            .collect();

        let rank = bit_sweep(&mut mat, 0);
        assert_eq!(rank, 1);

        // Expected matrix after processing
        // [[111]
        //  [000]
        //  [000]]
        let expected: Vec<FixedBitSet> = [0b111, 0b000, 0b000]
            .into_iter()
            .map(|bitset| FixedBitSet::with_capacity_and_blocks(3, [bitset]))
            .collect();
        assert_eq!(mat, expected);
    }

    #[test]
    fn test_calc_inverse_matrix() {
        // Matrix before processing
        // [[101|100]
        //  [001|010]
        //  [010|001]]
        let mut mat: Vec<FixedBitSet> = [0b001101, 0b010100, 0b100010]
            .into_iter()
            .map(|bitset| FixedBitSet::with_capacity_and_blocks(6, [bitset]))
            .collect();

        let rank = bit_sweep(&mut mat, 3);
        assert_eq!(rank, 3);

        // Expected matrix after processing
        // [[100|110]
        //  [010|001]
        //  [001|010]]
        let expected: Vec<FixedBitSet> = [0b011001, 0b100010, 0b010100]
            .into_iter()
            .map(|bitset| FixedBitSet::with_capacity_and_blocks(6, [bitset]))
            .collect();
        assert_eq!(mat, expected);
    }

    #[test]
    fn test_zero_matrix() {
        // Matrix
        // [[000]
        //  [000]
        //  [000]]
        let mut mat: Vec<FixedBitSet> = vec![FixedBitSet::with_capacity(3); 3];

        let rank = bit_sweep(&mut mat, 0);
        assert_eq!(rank, 0);

        assert_eq!(mat, vec![FixedBitSet::with_capacity(3); 3]);
    }

    #[test]
    fn test_empty() {
        let mut mat: Vec<FixedBitSet> = vec![];

        let rank = bit_sweep(&mut mat, 0);
        assert_eq!(rank, 0);

        assert_eq!(mat, vec![]);
    }

    #[test]
    fn test_vertically_elongated() {
        // Matrix before processing
        // [[101]
        //  [001]
        //  [100]
        //  [010]]
        let mut mat: Vec<FixedBitSet> = [0b101, 0b100, 0b001, 0b010]
            .into_iter()
            .map(|bitset| FixedBitSet::with_capacity_and_blocks(3, [bitset]))
            .collect();

        let rank = bit_sweep(&mut mat, 0);
        assert_eq!(rank, 3);

        // Expected matrix after processing
        // [[100]
        //  [010]
        //  [001]
        //  [000]]
        let expected: Vec<FixedBitSet> = [0b001, 0b010, 0b100, 0b000]
            .into_iter()
            .map(|bitset| FixedBitSet::with_capacity_and_blocks(3, [bitset]))
            .collect();
        assert_eq!(mat, expected);
    }

    #[test]
    fn test_expanded_and_horizontally_elongated() {
        // Matrix before processing
        // [[11010|0]
        //  [00010|1]]
        let mut mat: Vec<FixedBitSet> = [0b001011, 0b101000]
            .into_iter()
            .map(|bitset| FixedBitSet::with_capacity_and_blocks(6, [bitset]))
            .collect();

        let rank = bit_sweep(&mut mat, 1);
        assert_eq!(rank, 2);

        // Expected matrix after processing
        // [[11000|1]
        //  [00010|1]]
        let expected: Vec<FixedBitSet> = [0b100011, 0b101000]
            .into_iter()
            .map(|bitset| FixedBitSet::with_capacity_and_blocks(6, [bitset]))
            .collect();
        assert_eq!(mat, expected);
    }

    #[test]
    fn test_one_line_identity() {
        // Matrix
        // [[000|0]]
        let mut mat = vec![FixedBitSet::with_capacity_and_blocks(4, [0b0000])];

        let rank = bit_sweep(&mut mat, 1);
        assert_eq!(rank, 0);

        assert_eq!(
            mat,
            vec![FixedBitSet::with_capacity_and_blocks(4, [0b0000])]
        );
    }

    #[test]
    fn test_one_line_impossible() {
        // Matrix
        // [[000|1]]
        let mut mat = vec![FixedBitSet::with_capacity_and_blocks(4, [0b1000])];

        let rank = bit_sweep(&mut mat, 1);
        assert_eq!(rank, 0);

        assert_eq!(
            mat,
            vec![FixedBitSet::with_capacity_and_blocks(4, [0b1000])]
        );
    }

    #[test]
    fn test_indefinite() {
        // Matrix
        // [[00101|1]]
        let mut mat = vec![FixedBitSet::with_capacity_and_blocks(6, [0b110100])];

        let rank = bit_sweep(&mut mat, 1);
        assert_eq!(rank, 1);

        assert_eq!(
            mat,
            vec![FixedBitSet::with_capacity_and_blocks(6, [0b110100])]
        );
    }
}
