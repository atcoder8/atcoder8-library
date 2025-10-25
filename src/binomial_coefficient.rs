//! 動的計画法を用いて二項係数を計算するための構造体を定義するモジュールです。

use std::ops;

pub trait Zero {
    fn zero() -> Self;
}

pub trait One {
    fn one() -> Self;
}

#[derive(Debug, Clone)]
pub struct BinomialCoefficient<T: Clone + ops::Add<Output = T> + Zero + One> {
    coefficients: Vec<T>,
}

/// 二項係数の添字に対応する配列上の添字を返します。
fn calc_index(n: usize, k: usize) -> usize {
    n * (n + 1) / 2 + k
}

impl<T: Clone + ops::Add<Output = T> + Zero + One> BinomialCoefficient<T> {
    /// 動的計画法を用いて二項係数を計算します。
    ///
    /// # Arguments
    ///
    /// `max_n` - 計算する二項係数 `\binom{n}{k}` の添字のうち `n` の最大値
    pub fn new(max_n: usize) -> Self {
        let mut coefficients = vec![T::one(); (max_n + 1) * (max_n + 2) / 2];
        for n in 1..=max_n {
            for k in 1..n {
                coefficients[calc_index(n, k)] = coefficients[calc_index(n - 1, k - 1)].clone()
                    + coefficients[calc_index(n - 1, k)].clone();
            }
        }

        Self { coefficients }
    }

    /// 二項係数 `\binom{n}{k}` の計算結果を返します。
    pub fn binomial_coefficient(&self, n: usize, k: usize) -> T {
        if n >= k {
            self.coefficients[calc_index(n, k)].clone()
        } else {
            T::zero()
        }
    }
}

macro_rules! impl_for_primitive_integer {
    ( $($integer: ty), * ) => {
        $(
            impl Zero for $integer {
                fn zero() -> Self {
                    0
                }
            }

            impl One for $integer {
                fn one() -> Self {
                    1
                }
            }
        )*
    };
}

impl_for_primitive_integer!(i8, i16, i32, i64, i128, isize, u8, u16, u32, u64, u128, usize);
