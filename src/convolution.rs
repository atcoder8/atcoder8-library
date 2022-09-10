use std::f64::consts::PI;

use num::Zero;
use num_complex::Complex64;

/// Calculate ceil(log2(x)).
fn ceil_log2(x: usize) -> usize {
    let mut i = 0;
    while (1 << i) < x {
        i += 1;
    }
    i
}

/// Execute Fast Fourier Transform.
///
/// If inverse is true, execute inverse transformation.
fn fft(a: &mut Vec<Complex64>, inverse: bool) {
    let n = a.len();
    let h = ceil_log2(n);

    assert_eq!(n, 1 << h, "The length of the array must be a power of 2.");

    // Repositioning for butterfly operation
    for i in 0..a.len() {
        let j: usize = (0..h).map(|k| ((i >> k) & 1) << (h - 1 - k)).sum();
        if i < j {
            a.swap(i, j);
        }
    }

    // Butterfly diagram
    for i in 0..h {
        let sub_size = 1 << i;
        for j in 0..sub_size {
            let w = Complex64::new(
                0.0,
                if inverse { 1.0 } else { -1.0 } * PI / sub_size as f64 * j as f64,
            )
            .exp();
            for k in (0..n).step_by(2 * sub_size) {
                let s = a[k + j];
                let t = a[k + sub_size + j] * w;
                a[k + j] = s + t;
                a[k + sub_size + j] = s - t;
            }
        }
    }

    // For inverse transformation, divide each element by n.
    if inverse {
        a.iter_mut().for_each(|x| *x /= n as f64);
    }
}

pub trait Convolution {
    /// Execute convolution.
    fn convolution(&self, other: &Self) -> Self;
}

// Implement convolution for "Vec<Complex64>".
impl Convolution for Vec<Complex64> {
    fn convolution(&self, other: &Self) -> Self {
        assert!(!self.is_empty() && !other.is_empty());

        let s = self.len() + other.len() - 1;
        let n = 1 << ceil_log2(s);

        let mut a = self.clone();
        a.resize(n, Complex64::zero());
        fft(&mut a, false);

        let mut b = other.clone();
        b.resize(n, Complex64::zero());
        fft(&mut b, false);

        for i in 0..n {
            a[i] *= b[i];
        }

        fft(&mut a, true);

        a.resize(s, Complex64::zero());

        a
    }
}

/// The macro implements `Vec<T>` (where T is an integer type) convolution.
macro_rules! impl_for_int_type {
        ($($numeric_type:tt),*) => {
            $(
                impl Convolution for Vec<$numeric_type> {
                    fn convolution(&self, other: &Self) -> Self {
                        assert!(self.len() >= 1 && other.len() >= 1);

                        let a: Vec<_> = self.iter().map(|x| Complex64::from(*x as f64)).collect();
                        let b: Vec<_> = other.iter().map(|x| Complex64::from(*x as f64)).collect();

                        a.convolution(&b).iter().map(|x| x.re.round() as $numeric_type).collect()
                    }
                }
            )*
        };
    }

// Implements `Vec<T>` (where T is integer type) convolution.
impl_for_int_type!(u8, u16, u32, u64, u128, usize, i8, i16, i32, i64, i128, isize);

/// The macro implements `Vec<T>` (where T is a floating-point type) convolution.
macro_rules! impl_for_float_type {
        ($($numeric_type:tt),*) => {
            $(
                impl Convolution for Vec<$numeric_type> {
                    fn convolution(&self, other: &Self) -> Self {
                        assert!(self.len() >= 1 && other.len() >= 1);

                        let a: Vec<_> = self.iter().map(|x| Complex64::from(*x as f64)).collect();
                        let b: Vec<_> = other.iter().map(|x| Complex64::from(*x as f64)).collect();

                        a.convolution(&b).iter().map(|x| x.re as $numeric_type).collect()
                    }
                }
            )*
        };
    }

// Implements `Vec<T>` (where T is floating-point number type) convolution.
impl_for_float_type!(f32, f64);
