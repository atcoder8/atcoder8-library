use std::ops::{Add, AddAssign, Mul, MulAssign, Sub, SubAssign};

use num::{One, Zero};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Matrix<T>
where
    T: Clone,
{
    shape: (usize, usize),
    flattened: Vec<T>,
}

pub trait MatMul<T>
where
    T: Clone + Add<T, Output = T> + Mul<T, Output = T>,
{
    fn mat_mul(&self, rhs: &Self) -> Self;

    fn mat_mul_assign(&mut self, rhs: &Self);
}

pub trait Transpose<T>
where
    T: Clone,
{
    fn transposed(&self) -> Self;
}

pub trait MatPow<T>
where
    T: Clone + Zero + One + MulAssign<T>,
{
    fn mat_pow(&self, exp: usize) -> Self;
}

impl<T> MatMul<T> for Matrix<T>
where
    T: Clone + Add<T, Output = T> + Mul<T, Output = T>,
{
    fn mat_mul(&self, rhs: &Self) -> Self {
        let (h1, w1) = self.shape;
        let (h2, w2) = rhs.shape;

        assert_eq!(w1, h2);

        let calc_elem = |coord: (usize, usize)| {
            let (i, j) = coord;

            let mut elem = self.flattened[self.coord_to_idx((i, 0))].clone()
                * rhs.flattened[rhs.coord_to_idx((0, j))].clone();
            for k in 1..w1 {
                elem = elem
                    + self.flattened[self.coord_to_idx((i, k))].clone()
                        * rhs.flattened[rhs.coord_to_idx((k, j))].clone();
            }

            elem
        };

        let flattened: Vec<T> = (0..(h1 * w2))
            .map(|idx| calc_elem((idx / w2, idx % w2)))
            .collect();

        Self {
            shape: (h1, w2),
            flattened,
        }
    }

    fn mat_mul_assign(&mut self, rhs: &Self) {
        *self = self.mat_mul(rhs);
    }
}

impl<T> Transpose<T> for Matrix<T>
where
    T: Clone,
{
    fn transposed(&self) -> Self {
        let mut flattened = vec![];

        for i in 0..self.elem_num() {
            let coord = (i % self.shape.0, i / self.shape.0);
            flattened.push(self.flattened[self.coord_to_idx(coord)].clone());
        }

        Self {
            shape: (self.shape.1, self.shape.0),
            flattened,
        }
    }
}

impl<T> MatPow<T> for Matrix<T>
where
    T: Clone + Zero + One + MulAssign<T>,
{
    fn mat_pow(&self, exp: usize) -> Self {
        assert!(self.is_square());

        let mut ret = Self::identity(self.shape.0);
        let mut mul = self.clone();
        let mut exp = exp;

        while exp != 0 {
            if exp % 2 == 1 {
                ret.mat_mul_assign(&mul);
            }

            mul = mul.mat_mul(&mul);
            exp /= 2;
        }

        ret
    }
}

impl<T> Matrix<T>
where
    T: Clone,
{
    pub fn new(shape: (usize, usize)) -> Self
    where
        T: Default,
    {
        assert!(shape.0 >= 1 && shape.1 >= 1);

        Self {
            shape,
            flattened: vec![T::default(); shape.0 * shape.1],
        }
    }

    pub fn from_flattened(shape: (usize, usize), flattened: Vec<T>) -> Self {
        assert!(shape.0 >= 1 && shape.1 >= 1);
        assert_eq!(shape.0 * shape.1, flattened.len());

        Self { shape, flattened }
    }

    pub fn filled(shape: (usize, usize), x: T) -> Self {
        Self::from_flattened(shape, vec![x.clone(); shape.0 * shape.1])
    }

    pub fn zero(shape: (usize, usize)) -> Self
    where
        T: Zero,
    {
        Self::from_flattened(shape, vec![T::zero(); shape.0 * shape.1])
    }

    pub fn one(shape: (usize, usize)) -> Self
    where
        T: One,
    {
        Self::from_flattened(shape, vec![T::one(); shape.0 * shape.1])
    }

    pub fn identity(n: usize) -> Self
    where
        T: Zero + One,
    {
        let flattened = (0..n.pow(2))
            .map(|elem_idx| {
                let (i, j) = (elem_idx / n, elem_idx % n);

                if i == j {
                    T::one()
                } else {
                    T::zero()
                }
            })
            .collect();

        Self::from_flattened((n, n), flattened)
    }

    pub fn from_vector(vec: Vec<T>) -> Self {
        Self {
            shape: (vec.len(), 1),
            flattened: vec,
        }
    }

    pub fn shape(&self) -> (usize, usize) {
        self.shape
    }

    pub fn flattened(self) -> Vec<T> {
        self.flattened
    }

    pub fn elem_num(&self) -> usize {
        self.shape.0 * self.shape.1
    }

    pub fn to_vec(self) -> Vec<Vec<T>> {
        let (h, w) = self.shape;

        let mut mat = vec![vec![]; h];
        mat.iter_mut().for_each(|x| x.reserve(w));

        for (i, elem) in self.flattened.into_iter().enumerate() {
            mat[i / w].push(elem)
        }

        mat
    }

    #[allow(unused)]
    fn coord_to_idx(&self, coord: (usize, usize)) -> usize {
        debug_assert!(coord.0 < self.shape.0 && coord.1 < self.shape.1);

        coord.0 * self.shape.1 + coord.1
    }

    #[allow(unused)]
    fn idx_to_coord(&self, idx: usize) -> (usize, usize) {
        debug_assert!(idx < self.elem_num());

        (idx / self.shape.1, idx % self.shape.1)
    }

    pub fn get(&self, coord: (usize, usize)) -> &T {
        let idx = self.coord_to_idx(coord);

        &self.flattened[idx]
    }

    pub fn set(&mut self, coord: (usize, usize), val: T) {
        let idx = self.coord_to_idx(coord);

        self.flattened[idx] = val;
    }

    pub fn is_square(&self) -> bool {
        self.shape.0 == self.shape.1
    }

    pub fn apply_to(&self, vec: &Vector<T>) -> Vector<T>
    where
        T: Clone + Add<T, Output = T> + Mul<T, Output = T>,
    {
        let (h, w) = self.shape;

        assert_eq!(w, vec.len());

        let calc_elem = |i: usize| {
            let mut elem = self.get((i, 0)).clone() * vec.get(0).clone();

            for j in 1..w {
                elem = elem + self.get((i, j)).clone() * vec.get(j).clone();
            }

            elem
        };

        (0..h).map(|i| calc_elem(i)).collect::<Vec<T>>().into()
    }
}

impl<T> From<Vec<Vec<T>>> for Matrix<T>
where
    T: Clone,
{
    fn from(mat: Vec<Vec<T>>) -> Self {
        let h = mat.len();
        assert_ne!(h, 0);

        let w = mat[0].len();
        assert_ne!(w, 0);

        assert!(mat.iter().all(|x| x.len() == w));

        Self {
            shape: (h, w),
            flattened: mat.into_iter().flatten().collect(),
        }
    }
}

impl<T> Add<&Matrix<T>> for &Matrix<T>
where
    T: Clone + Add<T, Output = T>,
{
    type Output = Matrix<T>;

    fn add(self, rhs: &Matrix<T>) -> Self::Output {
        assert_eq!(self.shape, rhs.shape);

        let flattened = self
            .flattened
            .iter()
            .zip(rhs.flattened.iter())
            .map(|(x, y)| x.clone() + y.clone())
            .collect();

        Matrix {
            shape: self.shape,
            flattened,
        }
    }
}

impl<T> AddAssign<&Matrix<T>> for Matrix<T>
where
    T: Clone + AddAssign<T>,
{
    fn add_assign(&mut self, rhs: &Matrix<T>) {
        self.flattened
            .iter_mut()
            .zip(rhs.flattened.iter())
            .for_each(|(x, y)| *x += y.clone());
    }
}

impl<T> Sub<&Matrix<T>> for &Matrix<T>
where
    T: Clone + Sub<T, Output = T>,
{
    type Output = Matrix<T>;

    fn sub(self, rhs: &Matrix<T>) -> Self::Output {
        assert_eq!(self.shape, rhs.shape);

        let flattened = self
            .flattened
            .iter()
            .zip(rhs.flattened.iter())
            .map(|(x, y)| x.clone() - y.clone())
            .collect();

        Self::Output {
            shape: self.shape,
            flattened,
        }
    }
}

impl<T> SubAssign<&Matrix<T>> for Matrix<T>
where
    T: Clone + SubAssign<T>,
{
    fn sub_assign(&mut self, rhs: &Matrix<T>) {
        self.flattened
            .iter_mut()
            .zip(rhs.flattened.iter())
            .for_each(|(x, y)| *x -= y.clone());
    }
}

impl<T> Mul<&Matrix<T>> for &Matrix<T>
where
    T: Clone + Mul<T, Output = T>,
{
    type Output = Matrix<T>;

    fn mul(self, rhs: &Matrix<T>) -> Self::Output {
        assert_eq!(self.shape, rhs.shape);

        let flattened: Vec<T> = self
            .flattened
            .iter()
            .zip(rhs.flattened.iter())
            .map(|(x, y)| x.clone() * y.clone())
            .collect();

        Self::Output {
            shape: self.shape,
            flattened,
        }
    }
}

impl<T> MulAssign<&Matrix<T>> for Matrix<T>
where
    T: Clone + MulAssign<T>,
{
    fn mul_assign(&mut self, rhs: &Matrix<T>) {
        assert_eq!(self.shape, rhs.shape);

        self.flattened
            .iter_mut()
            .zip(rhs.flattened.iter())
            .for_each(|(x, y)| *x *= y.clone());
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Vector<T>
where
    T: Clone,
{
    elements: Vec<T>,
}

pub trait InnerProduct<T>
where
    T: Clone + Add<T, Output = T> + Mul<T, Output = T>,
{
    fn inner_product(&self, rhs: &Vector<T>) -> T;
}

impl<T> InnerProduct<T> for Vector<T>
where
    T: Clone + Add<T, Output = T> + Mul<T, Output = T>,
{
    fn inner_product(&self, rhs: &Vector<T>) -> T {
        assert_eq!(self.len(), rhs.len());

        let mut ret = self.get(0).clone() * rhs.get(0).clone();

        for i in 1..self.len() {
            ret = ret + self.get(i).clone() * rhs.get(i).clone();
        }

        ret
    }
}

impl<T> Vector<T>
where
    T: Clone,
{
    pub fn new(n: usize) -> Self
    where
        T: Default,
    {
        assert_ne!(n, 0);

        vec![T::default(); n].into()
    }

    pub fn len(&self) -> usize {
        self.elements.len()
    }

    pub fn elements(&self) -> &Vec<T> {
        &self.elements
    }

    pub fn to_vec(self) -> Vec<T> {
        self.elements
    }

    pub fn zero(n: usize) -> Self
    where
        T: Zero,
    {
        vec![T::zero(); n].into()
    }

    pub fn one(n: usize) -> Self
    where
        T: One,
    {
        vec![T::one(); n].into()
    }

    pub fn filled(x: T, n: usize) -> Self {
        vec![x.clone(); n].into()
    }

    pub fn get(&self, idx: usize) -> &T {
        &self.elements[idx]
    }

    pub fn apply_from(&self, mat: &Matrix<T>) -> Vector<T>
    where
        T: Clone + Add<T, Output = T> + Mul<T, Output = T>,
    {
        let (h, w) = mat.shape;

        assert_eq!(self.len(), w);

        let calc_elem = |i: usize| {
            let mut elem = mat.get((i, 0)).clone() * self.get(0).clone();

            for j in 1..w {
                elem = elem + mat.get((i, j)).clone() * self.get(j).clone();
            }

            elem
        };

        (0..h).map(|i| calc_elem(i)).collect::<Vec<T>>().into()
    }
}

impl<T> From<Vec<T>> for Vector<T>
where
    T: Clone,
{
    fn from(elements: Vec<T>) -> Self {
        assert!(!elements.is_empty());

        Self { elements }
    }
}

impl<T> Add<&Vector<T>> for &Vector<T>
where
    T: Clone + Add<T, Output = T>,
{
    type Output = Vector<T>;

    fn add(self, rhs: &Vector<T>) -> Self::Output {
        let elements: Vec<T> = self
            .elements
            .iter()
            .zip(rhs.elements.iter())
            .map(|(x, y)| x.clone() + y.clone())
            .collect();

        Self::Output { elements }
    }
}

impl<T> AddAssign<&Vector<T>> for Vector<T>
where
    T: Clone + AddAssign<T>,
{
    fn add_assign(&mut self, rhs: &Vector<T>) {
        self.elements
            .iter_mut()
            .zip(rhs.elements.iter())
            .for_each(|(x, y)| *x += y.clone());
    }
}

impl<T> Sub<&Vector<T>> for &Vector<T>
where
    T: Clone + Sub<T, Output = T>,
{
    type Output = Vector<T>;

    fn sub(self, rhs: &Vector<T>) -> Self::Output {
        let elements: Vec<T> = self
            .elements
            .iter()
            .zip(rhs.elements.iter())
            .map(|(x, y)| x.clone() - y.clone())
            .collect();

        Self::Output { elements }
    }
}

impl<T> SubAssign<&Vector<T>> for Vector<T>
where
    T: Clone + SubAssign<T>,
{
    fn sub_assign(&mut self, rhs: &Vector<T>) {
        self.elements
            .iter_mut()
            .zip(rhs.elements.iter())
            .for_each(|(x, y)| *x -= y.clone());
    }
}

impl<T> Mul<&Vector<T>> for &Vector<T>
where
    T: Clone + Mul<T, Output = T>,
{
    type Output = Vector<T>;

    fn mul(self, rhs: &Vector<T>) -> Self::Output {
        let elements: Vec<T> = self
            .elements
            .iter()
            .zip(rhs.elements.iter())
            .map(|(x, y)| x.clone() * y.clone())
            .collect();

        Self::Output { elements }
    }
}

impl<T> MulAssign<&Vector<T>> for Vector<T>
where
    T: Clone + MulAssign<T>,
{
    fn mul_assign(&mut self, rhs: &Vector<T>) {
        self.elements
            .iter_mut()
            .zip(rhs.elements.iter())
            .for_each(|(x, y)| *x *= y.clone());
    }
}

#[cfg(test)]
mod test {
    mod test_for_matrix {
        use super::super::*;

        #[test]
        fn test_transposed() {
            let mat: Matrix<i32> = vec![vec![3, -1, 4], vec![-1, 5, 9]].into();

            assert_eq!(
                mat.transposed(),
                vec![vec![3, -1], vec![-1, 5], vec![4, 9]].into()
            );
        }

        #[test]
        fn test_mat_mul() {
            let mat1: Matrix<i32> = vec![vec![3, -1, 4], vec![-1, 5, 9]].into();
            let mat2: Matrix<i32> = vec![vec![2, 6], vec![-5, -3], vec![5, 8]].into();

            assert_eq!(mat1.mat_mul(&mat2), vec![vec![31, 53], vec![18, 51]].into());
        }

        #[test]
        fn test_mat_mul_assign() {
            let mut mat1: Matrix<i32> = vec![vec![3, -1, 4], vec![-1, 5, 9]].into();
            let mat2: Matrix<i32> = vec![vec![2, 6], vec![-5, -3], vec![5, 8]].into();

            mat1.mat_mul_assign(&mat2);

            assert_eq!(mat1, vec![vec![31, 53], vec![18, 51]].into());
        }

        #[test]
        fn test_add() {
            let mat1: Matrix<i32> = vec![vec![3, -1, 4], vec![-1, 5, 9]].into();
            let mat2: Matrix<i32> = vec![vec![2, 6, -5], vec![-3, 5, 8]].into();

            assert_eq!(&mat1 + &mat2, vec![vec![5, 5, -1], vec![-4, 10, 17]].into());
        }

        #[test]
        fn test_add_assign() {
            let mut mat1: Matrix<i32> = vec![vec![3, -1, 4], vec![-1, 5, 9]].into();
            let mat2: Matrix<i32> = vec![vec![2, 6, -5], vec![-3, 5, 8]].into();

            mat1 += &mat2;

            assert_eq!(mat1, vec![vec![5, 5, -1], vec![-4, 10, 17]].into());
        }

        #[test]
        fn test_sub() {
            let mat1: Matrix<i32> = vec![vec![3, -1, 4], vec![-1, 5, 9]].into();
            let mat2: Matrix<i32> = vec![vec![2, 6, -5], vec![-3, 5, 8]].into();

            assert_eq!(&mat1 - &mat2, vec![vec![1, -7, 9], vec![2, 0, 1]].into());
        }

        #[test]
        fn test_sub_assign() {
            let mut mat1: Matrix<i32> = vec![vec![3, -1, 4], vec![-1, 5, 9]].into();
            let mat2: Matrix<i32> = vec![vec![2, 6, -5], vec![-3, 5, 8]].into();

            mat1 -= &mat2;

            assert_eq!(mat1, vec![vec![1, -7, 9], vec![2, 0, 1]].into());
        }

        #[test]
        fn test_mul() {
            let mat1: Matrix<i32> = vec![vec![3, -1, 4], vec![-1, 5, 9]].into();
            let mat2: Matrix<i32> = vec![vec![2, 6, -5], vec![-3, 5, 8]].into();

            assert_eq!(
                &mat1 * &mat2,
                vec![vec![6, -6, -20], vec![3, 25, 72]].into()
            );
        }

        #[test]
        fn test_mul_assign() {
            let mut mat1: Matrix<i32> = vec![vec![3, -1, 4], vec![-1, 5, 9]].into();
            let mat2: Matrix<i32> = vec![vec![2, 6, -5], vec![-3, 5, 8]].into();

            mat1 *= &mat2;

            assert_eq!(mat1, vec![vec![6, -6, -20], vec![3, 25, 72]].into());
        }

        #[test]
        fn test_mat_pow() {
            let mat: Matrix<i32> = vec![vec![3, -1, 4], vec![-1, 5, 9], vec![2, 6, -5]].into();

            assert_eq!(mat.mat_pow(0), Matrix::identity(3));

            assert_eq!(mat.mat_pow(1), mat);

            assert_eq!(
                mat.mat_pow(2),
                vec![vec![18, 16, -17], vec![10, 80, -4], vec![-10, -2, 87]].into()
            );

            assert_eq!(
                mat.mat_pow(3),
                vec![vec![4, -40, 301], vec![-58, 366, 780], vec![146, 522, -493]].into()
            );

            assert_eq!(
                Matrix::<i32>::identity(5),
                Matrix::identity(5).mat_pow(1_000_000_000_000_000_000)
            );
        }

        #[test]
        fn test_to_vec() {
            let mat_by_vec = vec![vec![3, -1, 4], vec![-1, 5, 9]];
            let mat = Matrix::from(mat_by_vec.clone());

            assert_eq!(mat.to_vec(), mat_by_vec);
        }

        #[test]
        fn test_shape() {
            let mat: Matrix<i32> = vec![vec![3, -1, 4], vec![-1, 5, 9]].into();

            assert_eq!(mat.shape(), (2, 3));
        }

        #[test]
        fn test_flattened() {
            let mat: Matrix<i32> = vec![vec![3, -1, 4], vec![-1, 5, 9]].into();

            assert_eq!(mat.flattened(), vec![3, -1, 4, -1, 5, 9]);
        }
    }

    mod test_for_vector {
        use super::super::*;

        #[test]
        fn test_inner_product() {
            let vec1: Vector<i32> = vec![3, -1, 4, -1].into();
            let vec2: Vector<i32> = vec![-1, 5, 9, -5].into();

            assert_eq!(vec1.inner_product(&vec2), 33);
        }

        #[test]
        fn test_add() {
            let vec1: Vector<i32> = vec![3, -1, 4, -1].into();
            let vec2: Vector<i32> = vec![-1, 5, 9, -5].into();

            assert_eq!(&vec1 + &vec2, vec![2, 4, 13, -6].into());
        }

        #[test]
        fn test_add_assign() {
            let mut vec1: Vector<i32> = vec![3, -1, 4, -1].into();
            let vec2: Vector<i32> = vec![-1, 5, 9, -5].into();

            vec1 += &vec2;

            assert_eq!(vec1, vec![2, 4, 13, -6].into());
        }

        #[test]
        fn test_sub() {
            let vec1: Vector<i32> = vec![3, -1, 4, -1].into();
            let vec2: Vector<i32> = vec![-1, 5, 9, -5].into();

            assert_eq!(&vec1 - &vec2, vec![4, -6, -5, 4].into());
        }

        #[test]
        fn test_sub_assign() {
            let mut vec1: Vector<i32> = vec![3, -1, 4, -1].into();
            let vec2: Vector<i32> = vec![-1, 5, 9, -5].into();

            vec1 -= &vec2;

            assert_eq!(vec1, vec![4, -6, -5, 4].into());
        }

        #[test]
        fn test_mul() {
            let vec1: Vector<i32> = vec![3, -1, 4, -1].into();
            let vec2: Vector<i32> = vec![-1, 5, 9, -5].into();

            assert_eq!(&vec1 * &vec2, vec![-3, -5, 36, 5].into());
        }

        #[test]
        fn test_mul_assign() {
            let mut vec1: Vector<i32> = vec![3, -1, 4, -1].into();
            let vec2: Vector<i32> = vec![-1, 5, 9, -5].into();

            vec1 *= &vec2;

            assert_eq!(vec1, vec![-3, -5, 36, 5].into());
        }

        #[test]
        fn test_to_vec() {
            let vector_by_vec = vec![3, -1, 4, -1];
            let vec = Vector::from(vector_by_vec.clone());

            assert_eq!(vec.to_vec(), vector_by_vec);
        }
    }

    mod test_for_apply {
        use super::super::*;

        #[test]
        fn test_apply_to() {
            let mat: Matrix<i32> = vec![vec![3, -1, 4], vec![-1, 5, 9]].into();
            let vec: Vector<i32> = vec![2, 6, -5].into();

            assert_eq!(mat.apply_to(&vec), vec![-20, -17].into())
        }

        #[test]
        fn test_apply_from() {
            let mat: Matrix<i32> = vec![vec![3, -1, 4], vec![-1, 5, 9]].into();
            let vec: Vector<i32> = vec![2, 6, -5].into();

            assert_eq!(vec.apply_from(&mat), vec![-20, -17].into())
        }
    }
}
