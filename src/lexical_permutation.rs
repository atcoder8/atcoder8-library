/// If there is a previous permutation of `seq` with respect to the lexicographic order, replace `seq` with it (return value is `true`).
/// Otherwise (i.e., if `seq` is already in ascending order), it reverts to descending order (return value is `false`).
pub fn prev_permutation<T>(seq: &mut [T]) -> bool
where
    T: Ord,
{
    // If the length of `seq` is 0 or 1, the previous permutation does not exist.
    if seq.len() <= 1 {
        return false;
    }

    // Find the maximum value of `i` such that `seq[i] > seq[i + 1]`.
    // If no such `i` exists, `seq` has already been sorted in descending order.
    let Some(i) = (0..seq.len() - 1).rev().find(|&i| seq[i] > seq[i + 1]) else {
        seq.reverse();
        return false;
    };

    // Find the largest `j` that satisfies `i < j` and `seq[i] > seq[j]`, and exchange `seq[i]` and `seq[j]`.
    let j = (i + 1..seq.len()).rev().find(|&j| seq[i] > seq[j]).unwrap();
    seq.swap(i, j);

    // Sort elements after the `i`-th in descending order to minimum the decrease with respect to lexicographic order.
    seq[i + 1..].reverse();

    true
}

/// If there is a next permutation of `seq` with respect to the lexicographic order, replace `seq` with it (return value is `true`).
/// Otherwise (i.e., if `seq` is already in descending order), it reverts to ascending order (return value is `false`).
pub fn next_permutation<T>(seq: &mut [T]) -> bool
where
    T: Ord,
{
    // If the length of `seq` is 0 or 1, the next permutation does not exist.
    if seq.len() <= 1 {
        return false;
    }

    // Find the maximum value of `i` such that `seq[i] < seq[i + 1]`.
    // If no such `i` exists, `seq` has already been sorted in descending order.
    let Some(i) = (0..seq.len() - 1).rev().find(|&i| seq[i] < seq[i + 1]) else {
        seq.reverse();
        return false;
    };

    // Find the largest `j` that satisfies `i < j` and `seq[i] < seq[j]`, and exchange `seq[i]` and `seq[j]`.
    let j = (i + 1..seq.len()).rev().find(|&j| seq[i] < seq[j]).unwrap();
    seq.swap(i, j);

    // Sort elements after the `i`-th in ascending order to minimize the increase with respect to lexicographic order.
    seq[i + 1..].reverse();

    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prev_permutation() {
        let mut seq = [3, 2, 2, 1];

        let mut permutations = vec![];
        loop {
            permutations.push(seq.clone());

            if !prev_permutation(&mut seq) {
                break;
            }
        }

        let expected = [
            [3, 2, 2, 1],
            [3, 2, 1, 2],
            [3, 1, 2, 2],
            [2, 3, 2, 1],
            [2, 3, 1, 2],
            [2, 2, 3, 1],
            [2, 2, 1, 3],
            [2, 1, 3, 2],
            [2, 1, 2, 3],
            [1, 3, 2, 2],
            [1, 2, 3, 2],
            [1, 2, 2, 3],
        ];
        assert_eq!(permutations, expected);
    }

    #[test]
    fn test_next_permutation() {
        let mut seq = [1, 2, 2, 3];

        let mut permutations = vec![];
        loop {
            permutations.push(seq.clone());

            if !next_permutation(&mut seq) {
                break;
            }
        }

        let expected = [
            [1, 2, 2, 3],
            [1, 2, 3, 2],
            [1, 3, 2, 2],
            [2, 1, 2, 3],
            [2, 1, 3, 2],
            [2, 2, 1, 3],
            [2, 2, 3, 1],
            [2, 3, 1, 2],
            [2, 3, 2, 1],
            [3, 1, 2, 2],
            [3, 2, 1, 2],
            [3, 2, 2, 1],
        ];
        assert_eq!(permutations, expected);
    }
}
