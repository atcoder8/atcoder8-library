use std::ops::{Add, Div, Mul, Sub};

pub trait Field:
    Clone
    + Add<Self, Output = Self>
    + Sub<Self, Output = Self>
    + Mul<Self, Output = Self>
    + Div<Self, Output = Self>
{
    fn is_zero(&self) -> bool;
    fn zero() -> Self;
    fn one() -> Self;
}

pub fn gaussian_elimination<T>(mat: &mut [Vec<T>], extended: usize) -> usize
where
    T: Field,
{
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
        let pivot = match (rank..h).find(|&row| !mat[row][sweep_pos].is_zero()) {
            Some(pivot) => pivot,
            None => continue,
        };

        mat.swap(rank, pivot);

        for col in (sweep_pos + 1)..w {
            let divided = mat[rank][col].clone() / mat[rank][sweep_pos].clone();
            mat[rank][col] = divided;
        }
        mat[rank][sweep_pos] = T::one();

        for row in 0..h {
            if row == rank || mat[row][sweep_pos].is_zero() {
                continue;
            }

            for col in (sweep_pos + 1)..w {
                let subtracted =
                    mat[row][col].clone() - mat[row][sweep_pos].clone() * mat[rank][col].clone();
                mat[row][col] = subtracted;
            }
            mat[row][sweep_pos] = T::zero();
        }

        rank += 1;
    }

    rank
}

#[cfg(test)]
mod tests {
    use super::{super::modint2::Modint1000000007, gaussian_elimination, Field};

    type Mint = Modint1000000007;

    impl Field for Mint {
        fn is_zero(&self) -> bool {
            *self == Mint::new(0)
        }

        fn zero() -> Self {
            Mint::new(0)
        }

        fn one() -> Self {
            Mint::new(1)
        }
    }

    #[test]
    fn test_modint() {
        let mut mat = vec![
            vec![
                Mint::new(3),
                Mint::new(1),
                Mint::new(4),
                Mint::new(1),
                Mint::new(0),
                Mint::new(0),
            ],
            vec![
                Mint::new(1),
                Mint::new(5),
                Mint::new(9),
                Mint::new(0),
                Mint::new(1),
                Mint::new(0),
            ],
            vec![
                Mint::new(2),
                Mint::new(6),
                Mint::new(5),
                Mint::new(0),
                Mint::new(0),
                Mint::new(1),
            ],
        ];

        let rank = gaussian_elimination(&mut mat, 3);
        assert_eq!(rank, 3);

        let expected = vec![
            vec![
                Mint::new(1),
                Mint::new(0),
                Mint::new(0),
                Mint::frac(29, 90),
                Mint::frac(-19, 90),
                Mint::frac(11, 90),
            ],
            vec![
                Mint::new(0),
                Mint::new(1),
                Mint::new(0),
                Mint::frac(-13, 90),
                Mint::frac(-7, 90),
                Mint::frac(23, 90),
            ],
            vec![
                Mint::new(0),
                Mint::new(0),
                Mint::new(1),
                Mint::frac(2, 45),
                Mint::frac(8, 45),
                Mint::frac(-7, 45),
            ],
        ];
        assert_eq!(mat, expected);
    }
}
