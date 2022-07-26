use num::{One, Zero};
use num_traits::Pow;
use std::ops::{Add, AddAssign, BitAnd, Mul, MulAssign, Neg, ShrAssign, Sub, SubAssign};

pub type MatByVec<T> = Vec<Vec<T>>;

pub fn shape_matrix<T>(mat: &MatByVec<T>) -> Option<(usize, usize)> {
    if mat.len() == 0 || mat[0].len() == 0 {
        return None;
    }

    if mat.iter().skip(1).all(|x| x.len() == mat[0].len()) {
        Some((mat.len(), mat[0].len()))
    } else {
        None
    }
}

pub fn check_rect<T>(mat: &MatByVec<T>) -> bool {
    shape_matrix(mat).is_some()
}

pub fn check_square<T>(mat: &MatByVec<T>) -> bool {
    if let Some((r, c)) = shape_matrix(mat) {
        r == c
    } else {
        false
    }
}

fn add_vector<T>(lhs: &Vec<T>, rhs: &Vec<T>) -> Vec<T>
where
    T: Clone + Add<Output = T>,
{
    assert!(
        lhs.len() >= 1 && rhs.len() >= 1,
        "The length of the vector must be at least 1."
    );
    assert_eq!(
        lhs.len(),
        rhs.len(),
        "The lengths of the vectors must be the same."
    );

    lhs.iter()
        .zip(rhs.iter())
        .map(|(x, y)| x.clone() + y.clone())
        .collect()
}

fn add_matrix<T>(lhs: &MatByVec<T>, rhs: &MatByVec<T>) -> MatByVec<T>
where
    T: Clone + Add<Output = T>,
{
    if let (Some(lhs_shape), Some(rhs_shape)) = (shape_matrix(lhs), shape_matrix(rhs)) {
        assert_eq!(
            lhs_shape, rhs_shape,
            "The shape of the matrix must be the same."
        );
    } else {
        panic!("The shape of the matrix must be rectangular.");
    }

    lhs.iter()
        .zip(rhs.iter())
        .map(|(x, y)| add_vector(x, y))
        .collect()
}

fn sub_vector<T>(lhs: &Vec<T>, rhs: &Vec<T>) -> Vec<T>
where
    T: Clone + Sub<Output = T>,
{
    assert!(
        lhs.len() >= 1 && rhs.len() >= 1,
        "The length of the vector must be at least 1."
    );
    assert_eq!(
        lhs.len(),
        rhs.len(),
        "The lengths of the vectors must be the same."
    );

    lhs.iter()
        .zip(rhs.iter())
        .map(|(x, y)| x.clone() - y.clone())
        .collect()
}

fn sub_matrix<T>(lhs: &MatByVec<T>, rhs: &MatByVec<T>) -> MatByVec<T>
where
    T: Clone + Sub<Output = T>,
{
    if let (Some(lhs_shape), Some(rhs_shape)) = (shape_matrix(lhs), shape_matrix(rhs)) {
        assert_eq!(
            lhs_shape, rhs_shape,
            "The shape of the matrix must be the same."
        );
    } else {
        panic!("The shape of the matrix must be rectangular.");
    }

    lhs.iter()
        .zip(rhs.iter())
        .map(|(x, y)| sub_vector(x, y))
        .collect()
}

fn mul_vector<T>(lhs: &Vec<T>, rhs: &Vec<T>) -> Vec<T>
where
    T: Clone + Mul<Output = T>,
{
    assert!(
        lhs.len() >= 1 && rhs.len() >= 1,
        "The length of the vector must be at least 1."
    );
    assert_eq!(
        lhs.len(),
        rhs.len(),
        "The lengths of the vectors must be the same."
    );

    lhs.iter()
        .zip(rhs.iter())
        .map(|(x, y)| x.clone() * y.clone())
        .collect()
}

fn mul_matrix<T>(lhs: &MatByVec<T>, rhs: &MatByVec<T>) -> MatByVec<T>
where
    T: Clone + Mul<Output = T>,
{
    if let (Some(lhs_shape), Some(rhs_shape)) = (shape_matrix(lhs), shape_matrix(rhs)) {
        assert_eq!(
            lhs_shape, rhs_shape,
            "The shape of the matrix must be the same."
        );
    } else {
        panic!("The shape of the matrix must be rectangular.");
    }

    lhs.iter()
        .zip(rhs.iter())
        .map(|(x, y)| mul_vector(x, y))
        .collect()
}

fn neg_vector<T>(vec: &Vec<T>) -> Vec<T>
where
    T: Clone + Neg<Output = T>,
{
    assert!(
        vec.len() >= 1,
        "The length of the vector must be at least 1."
    );

    vec.iter().map(|x| -x.clone()).collect()
}

fn neg_matrix<T>(mat: &MatByVec<T>) -> MatByVec<T>
where
    T: Clone + Neg<Output = T>,
{
    assert!(
        check_rect(mat),
        "The shape of the matrix must be rectangular."
    );

    mat.iter().map(|x| neg_vector(x)).collect()
}

fn add_assign_vector<T>(lhs: &mut Vec<T>, rhs: &Vec<T>)
where
    T: Clone + AddAssign<T>,
{
    assert!(
        lhs.len() >= 1 && rhs.len() >= 1,
        "The length of the vector must be at least 1."
    );
    assert_eq!(
        lhs.len(),
        rhs.len(),
        "The lengths of the vectors must be the same."
    );

    lhs.iter_mut()
        .zip(rhs.iter())
        .for_each(|(x, y)| *x += y.clone());
}

fn add_assign_matrix<T>(lhs: &mut MatByVec<T>, rhs: &MatByVec<T>)
where
    T: Clone + AddAssign,
{
    if let (Some(lhs_shape), Some(rhs_shape)) = (shape_matrix(lhs), shape_matrix(rhs)) {
        assert_eq!(
            lhs_shape, rhs_shape,
            "The shape of the matrix must be the same."
        );
    } else {
        panic!("The shape of the matrix must be rectangular.");
    }

    lhs.iter_mut()
        .zip(rhs.iter())
        .for_each(|(x, y)| add_assign_vector(x, y));

    lhs.iter_mut()
        .zip(rhs.iter())
        .for_each(|(x, y)| add_assign_vector(x, y));
}

fn sub_assign_vector<T>(lhs: &mut Vec<T>, rhs: &Vec<T>)
where
    T: Clone + SubAssign<T>,
{
    assert!(
        lhs.len() >= 1 && rhs.len() >= 1,
        "The length of the vector must be at least 1."
    );
    assert_eq!(
        lhs.len(),
        rhs.len(),
        "The lengths of the vectors must be the same."
    );

    lhs.iter_mut()
        .zip(rhs.iter())
        .for_each(|(x, y)| *x -= y.clone());
}

fn sub_assign_matrix<T>(lhs: &mut MatByVec<T>, rhs: &MatByVec<T>)
where
    T: Clone + SubAssign,
{
    if let (Some(lhs_shape), Some(rhs_shape)) = (shape_matrix(lhs), shape_matrix(rhs)) {
        assert_eq!(
            lhs_shape, rhs_shape,
            "The shape of the matrix must be the same."
        );
    } else {
        panic!("The shape of the matrix must be rectangular.");
    }

    lhs.iter_mut()
        .zip(rhs.iter())
        .for_each(|(x, y)| sub_assign_vector(x, y));
}

fn mul_assign_vector<T>(lhs: &mut Vec<T>, rhs: &Vec<T>)
where
    T: Clone + MulAssign<T>,
{
    assert!(
        lhs.len() >= 1 && rhs.len() >= 1,
        "The length of the vector must be at least 1."
    );
    assert_eq!(
        lhs.len(),
        rhs.len(),
        "The lengths of the vectors must be the same."
    );

    lhs.iter_mut()
        .zip(rhs.iter())
        .for_each(|(x, y)| *x *= y.clone());
}

fn mul_assign_matrix<T>(lhs: &mut MatByVec<T>, rhs: &MatByVec<T>)
where
    T: Clone + MulAssign,
{
    if let (Some(lhs_shape), Some(rhs_shape)) = (shape_matrix(lhs), shape_matrix(rhs)) {
        assert_eq!(
            lhs_shape, rhs_shape,
            "The shape of the matrix must be the same."
        );
    } else {
        panic!("The shape of the matrix must be rectangular.");
    }

    lhs.iter_mut()
        .zip(rhs.iter())
        .for_each(|(x, y)| mul_assign_vector(x, y));
}

fn prod_matrix_cell<T>(lhs: &MatByVec<T>, rhs: &MatByVec<T>, pos: (usize, usize)) -> T
where
    T: Clone + Add<Output = T> + Mul<Output = T>,
{
    (0..lhs[0].len())
        .map(|i| lhs[pos.0][i].clone() * rhs[i][pos.1].clone())
        .reduce(Add::add)
        .unwrap()
}

fn prod_matrix_row<T>(lhs: &MatByVec<T>, rhs: &MatByVec<T>, idx: usize) -> Vec<T>
where
    T: Clone + Add<Output = T> + Mul<Output = T>,
{
    (0..rhs[0].len())
        .map(|i| prod_matrix_cell(lhs, rhs, (idx, i)))
        .collect()
}

fn prod_matrix<T>(lhs: &MatByVec<T>, rhs: &MatByVec<T>) -> MatByVec<T>
where
    T: Clone + Add<Output = T> + Mul<Output = T>,
{
    if let (Some((_, c1)), Some((r2, _))) = (shape_matrix(lhs), shape_matrix(rhs)) {
        assert_eq!(
            c1, r2,
            "The number of columns in `lhs` and the number of rows in `rhs` are mismatched."
        );
    } else {
        panic!("The shape of the matrix must be rectangular.");
    };

    (0..lhs.len())
        .map(|i| prod_matrix_row(lhs, rhs, i))
        .collect()
}

fn inner_prod_vector<T>(lhs: &Vec<T>, rhs: &Vec<T>) -> T
where
    T: Clone + Add<Output = T> + Mul<Output = T>,
{
    assert!(
        lhs.len() >= 1 && rhs.len() >= 1,
        "The length of the vector must be at least 1."
    );
    assert_eq!(
        lhs.len(),
        rhs.len(),
        "The lengths of the vectors must be the same."
    );

    lhs.iter()
        .zip(rhs.iter())
        .map(|(x, y)| x.clone() * y.clone())
        .reduce(Add::add)
        .unwrap()
}

fn convert_type_vector<T, U>(vec: &Vec<T>) -> Vec<U>
where
    T: Clone,
    U: From<T>,
{
    vec.iter().map(|x| U::from(x.clone())).collect()
}

fn convert_type_matrix<T, U>(mat: &MatByVec<T>) -> MatByVec<U>
where
    T: Clone,
    U: From<T>,
{
    mat.iter().map(|x| convert_type_vector(x)).collect()
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Vector<T>(Vec<T>);

impl<T> Vector<T> {
    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn get(&self, idx: usize) -> &T {
        &self.0[idx]
    }

    pub fn set(&mut self, idx: usize, elem: T) {
        self.0[idx] = elem;
    }

    pub fn vec(&self) -> &Vec<T> {
        &self.0
    }

    pub fn to_vec(self) -> Vec<T> {
        self.0
    }

    pub fn convert_type<U>(&self) -> Vector<U>
    where
        T: Clone,
        U: From<T>,
    {
        Vector(convert_type_vector(self.vec()))
    }
}

impl<T> From<Vec<T>> for Vector<T> {
    fn from(vec: Vec<T>) -> Self {
        assert_ne!(vec.len(), 0, "The length of the vector must be at least 1.");

        Self(vec)
    }
}

impl<T> Vector<T>
where
    T: Clone + Zero,
{
    pub fn zero(n: usize) -> Self {
        assert_ne!(n, 0, "The length of the vector must be at least 1.");

        Self(vec![T::zero(); n])
    }
}

impl<T> Vector<T>
where
    T: Clone + One,
{
    pub fn one(n: usize) -> Self {
        assert_ne!(n, 0, "The length of the vector must be at least 1.");

        Self(vec![T::one(); n])
    }
}

impl<T> Vector<T>
where
    T: Clone,
{
    pub fn fill(n: usize, elem: &T) -> Self {
        assert_ne!(n, 0, "The length of the vector must be at least 1.");

        Self(vec![elem.clone(); n])
    }
}

impl<T> Vector<T>
where
    T: Clone + Zero + One,
{
    pub fn identity(n: usize, idx: usize) -> Self {
        assert_ne!(n, 0, "The length of the vector must be at least 1.");

        let mut output_mat = vec![T::zero(); n];
        output_mat[idx] = T::one();

        Self(output_mat)
    }
}

impl<T> Add for &Vector<T>
where
    T: Clone + Add<Output = T>,
{
    type Output = Vector<T>;

    fn add(self, rhs: Self) -> Self::Output {
        Vector(add_vector(&self.vec(), rhs.vec()))
    }
}

impl<T> Sub for &Vector<T>
where
    T: Clone + Sub<Output = T>,
{
    type Output = Vector<T>;

    fn sub(self, rhs: Self) -> Self::Output {
        Vector(sub_vector(self.vec(), rhs.vec()))
    }
}

impl<T> Mul for &Vector<T>
where
    T: Clone + Mul<Output = T>,
{
    type Output = Vector<T>;

    fn mul(self, rhs: Self) -> Self::Output {
        Vector(mul_vector(self.vec(), rhs.vec()))
    }
}

impl<T> AddAssign<&Vector<T>> for Vector<T>
where
    T: Clone + AddAssign,
{
    fn add_assign(&mut self, rhs: &Vector<T>) {
        add_assign_vector(&mut self.0, rhs.vec())
    }
}

impl<T> SubAssign<&Vector<T>> for Vector<T>
where
    T: Clone + SubAssign,
{
    fn sub_assign(&mut self, rhs: &Vector<T>) {
        sub_assign_vector(&mut self.0, rhs.vec())
    }
}

impl<T> MulAssign<&Vector<T>> for Vector<T>
where
    T: Clone + MulAssign,
{
    fn mul_assign(&mut self, rhs: &Vector<T>) {
        mul_assign_vector(&mut self.0, rhs.vec())
    }
}

impl<T> Neg for &Vector<T>
where
    T: Clone + Neg<Output = T>,
{
    type Output = Vector<T>;

    fn neg(self) -> Self::Output {
        Vector(neg_vector(&self.0))
    }
}

pub trait VecInnerProd<RHS> {
    type Output;

    fn inner_prod(&self, rhs: RHS) -> Self::Output;
}

impl<T> VecInnerProd<&Vector<T>> for Vector<T>
where
    T: Clone + Add<Output = T> + Mul<Output = T>,
{
    type Output = T;

    fn inner_prod(&self, rhs: &Vector<T>) -> T {
        inner_prod_vector(&self.0, &rhs.0)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Matrix<T>(MatByVec<T>);

impl<T> Matrix<T> {
    pub fn shape(&self) -> (usize, usize) {
        let Matrix(mat) = self;
        (mat.len(), mat[0].len())
    }

    pub fn is_square(&self) -> bool {
        self.0.len() == self.0[0].len()
    }

    pub fn get(&self, indexes: (usize, usize)) -> &T {
        &self.0[indexes.0][indexes.1]
    }

    pub fn set(&mut self, indexes: (usize, usize), elem: T) {
        self.0[indexes.0][indexes.1] = elem;
    }

    pub fn mat_by_vec(&self) -> &MatByVec<T> {
        &self.0
    }

    pub fn to_mat_by_vec(self) -> MatByVec<T> {
        self.0
    }

    pub fn convert_type<U>(&self) -> Matrix<U>
    where
        T: Clone,
        U: From<T>,
    {
        Matrix(convert_type_matrix(self.mat_by_vec()))
    }
}

impl<T> Matrix<T>
where
    T: Clone,
{
    pub fn get_row(&self, idx: usize) -> Vector<T> {
        Vector(self.0[idx].clone())
    }

    pub fn get_col(&self, idx: usize) -> Vector<T> {
        Vector((0..self.0.len()).map(|i| self.0[idx][i].clone()).collect())
    }
}

impl<T> From<MatByVec<T>> for Matrix<T> {
    fn from(mat: MatByVec<T>) -> Self {
        assert!(
            check_rect(&mat),
            "The shape of the matrix must be rectangular."
        );

        Self(mat)
    }
}

impl<T> Matrix<T>
where
    T: Clone + Zero,
{
    pub fn zero(shape: (usize, usize)) -> Self {
        let (r, c) = shape;

        assert!(
            r >= 1 && c >= 1,
            "The number of rows and columns in the matrix must be at least 1."
        );

        Self(vec![vec![T::zero(); c]; r])
    }
}

impl<T> Matrix<T>
where
    T: Clone + One,
{
    pub fn one(shape: (usize, usize)) -> Self {
        let (r, c) = shape;

        assert!(
            r >= 1 && c >= 1,
            "The number of rows and columns in the matrix must be at least 1."
        );

        Self(vec![vec![T::one(); c]; r])
    }
}

impl<T> Matrix<T>
where
    T: Clone,
{
    pub fn fill(shape: (usize, usize), elem: &T) -> Self {
        let (r, c) = shape;

        assert!(
            r >= 1 && c >= 1,
            "The number of rows and columns in the matrix must be at least 1."
        );

        Self(vec![vec![elem.clone(); c]; r])
    }
}

impl<T> Matrix<T>
where
    T: Clone + Zero + One,
{
    pub fn identity(n: usize) -> Self {
        assert!(n >= 1, "The degree of the matrix must be at least 1.");

        let mut mat = vec![vec![T::zero(); n]; n];

        for i in 0..n {
            mat[i][i] = T::one();
        }

        Self(mat)
    }
}

impl<T> Add for &Matrix<T>
where
    T: Clone + Add<Output = T>,
{
    type Output = Matrix<T>;

    fn add(self, rhs: Self) -> Self::Output {
        Matrix(add_matrix(self.mat_by_vec(), rhs.mat_by_vec()))
    }
}

impl<T> Sub for &Matrix<T>
where
    T: Clone + Sub<Output = T>,
{
    type Output = Matrix<T>;

    fn sub(self, rhs: Self) -> Self::Output {
        Matrix(sub_matrix(&self.0, rhs.mat_by_vec()))
    }
}

impl<T> Mul for &Matrix<T>
where
    T: Clone + Mul<Output = T>,
{
    type Output = Matrix<T>;

    fn mul(self, rhs: Self) -> Self::Output {
        Matrix(mul_matrix(self.mat_by_vec(), rhs.mat_by_vec()))
    }
}

pub trait MatProd<RHS> {
    type Output;

    fn prod(&self, rhs: RHS) -> Self::Output;
}

impl<T> MatProd<&Matrix<T>> for Matrix<T>
where
    T: Clone + Add<Output = T> + Mul<Output = T>,
{
    type Output = Matrix<T>;

    fn prod(&self, rhs: &Matrix<T>) -> Self::Output {
        Matrix(prod_matrix(self.mat_by_vec(), rhs.mat_by_vec()))
    }
}

impl<T> AddAssign<&Matrix<T>> for Matrix<T>
where
    T: Clone + AddAssign,
{
    fn add_assign(&mut self, rhs: &Matrix<T>) {
        add_assign_matrix(&mut self.0, rhs.mat_by_vec())
    }
}

impl<T> SubAssign<&Matrix<T>> for Matrix<T>
where
    T: Clone + SubAssign,
{
    fn sub_assign(&mut self, rhs: &Matrix<T>) {
        sub_assign_matrix(&mut self.0, rhs.mat_by_vec())
    }
}

impl<T> MulAssign<&Matrix<T>> for Matrix<T>
where
    T: Clone + MulAssign,
{
    fn mul_assign(&mut self, rhs: &Matrix<T>) {
        mul_assign_matrix(&mut self.0, rhs.mat_by_vec())
    }
}

impl<T> Neg for &Matrix<T>
where
    T: Clone + Neg<Output = T>,
{
    type Output = Matrix<T>;

    fn neg(self) -> Self::Output {
        Matrix(neg_matrix(&self.0))
    }
}

impl<EXP, T> Pow<EXP> for &Matrix<T>
where
    EXP: Clone + PartialOrd + Zero + One + BitAnd<Output = EXP> + ShrAssign,
    T: Clone + Zero + One + Add<Output = T> + Mul<Output = T>,
{
    type Output = Matrix<T>;

    fn pow(self, rhs: EXP) -> Self::Output {
        assert!(
            rhs >= EXP::zero(),
            "The exponent must be greater than or equal to 0."
        );
        assert!(self.is_square(), "It must be a square matrix.");

        let mut output_mat = Matrix::<T>::identity(self.0.len());

        let mut mul = self.clone();

        let mut x = rhs.clone();
        while x > EXP::zero() {
            if !(x.clone() & EXP::one()).is_zero() {
                output_mat = output_mat.prod(&mul);
            }

            mul = mul.prod(&mul);
            x >>= EXP::one();
        }

        output_mat
    }
}

impl<T> MatProd<&Vector<T>> for Matrix<T>
where
    T: Clone + Add<Output = T> + Mul<Output = T>,
{
    type Output = Vector<T>;

    fn prod(&self, rhs: &Vector<T>) -> Self::Output {
        Vector(
            self.0
                .iter()
                .map(|x| inner_prod_vector(x, &rhs.0))
                .collect(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pow_identity_test() {
        let mat = Matrix::<u8>::identity(10);
        assert_eq!(mat.pow(2e18 as u64), mat);
    }

    #[test]
    fn pow_test() {
        let mat = Matrix::from(vec![vec![3, -1, 4], vec![-1, 5, -9], vec![2, 6, 5]]);
        let pow3_mat = Matrix::from(vec![
            vec![120, 308, 133],
            vec![-238, -678, -322],
            vec![-70, 154, -587],
        ]);
        assert_eq!(mat.pow(3), pow3_mat);
    }
}
