//! # Examples
//!
//! ```
//! use atcoder8_library::matrix::{Vector, Matrix};
//!
//! let vec = Vector::from(vec![3, -1, 4]);
//! let mat = Matrix::from(vec![vec![1, -5, 9], vec![2, 6, -5], vec![3, 5, 8]]);
//! assert_eq!(
//!     &mat * &vec,
//!     Vector::from(vec![44, -20, 36])
//! );
//! ```

use num::{One, Zero};
use num_traits::Pow;
use std::ops::{Add, AddAssign, BitAnd, Mul, Neg, ShrAssign, Sub, SubAssign};

pub type MatByVec<T> = Vec<Vec<T>>;

/// # Examples
///
/// ```
/// use atcoder8_library::matrix::Vector;
///
/// let vec = Vector::from(vec![3, -1, 4]);
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Vector<T>(Vec<T>);

impl<T> From<Vec<T>> for Vector<T> {
    /// # Examples
    ///
    /// ```
    /// use atcoder8_library::matrix::Vector;
    ///
    /// let vec = Vector::from(vec![3, -1, 4]);
    /// ```
    fn from(vec: Vec<T>) -> Self {
        assert_ne!(vec.len(), 0, "The length of the vector must be at least 1.");

        Self(vec)
    }
}

impl<T> Vector<T> {
    /// # Examples
    ///
    /// ```
    /// use atcoder8_library::matrix::Vector;
    ///
    /// let vec = Vector::from(vec![3, -1, 4, 1, -5]);
    /// assert_eq!(vec.len(), 5);
    /// ```
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// # Examples
    ///
    /// ```
    /// use atcoder8_library::matrix::Vector;
    ///
    /// let vec = Vector::from(vec![3, -1, 4, 1, -5]);
    /// assert_eq!(vec.get(2), &4);
    /// ```
    pub fn get(&self, idx: usize) -> &T {
        &self.0[idx]
    }

    /// # Examples
    ///
    /// ```
    /// use atcoder8_library::matrix::Vector;
    ///
    /// let mut vec = Vector::from(vec![3, -1, 4, 1, -5]);
    /// vec.set(2, 100);
    /// assert_eq!(vec, Vector::from(vec![3, -1, 100, 1, -5]));
    /// ```
    pub fn set(&mut self, idx: usize, elem: T) {
        self.0[idx] = elem;
    }

    /// # Examples
    ///
    /// ```
    /// use atcoder8_library::matrix::Vector;
    ///
    /// let vec = Vector::from(vec![3, -1, 4, 1, -5]);
    /// assert_eq!(vec.vec(), &vec![3, -1, 4, 1, -5]);
    /// ```
    pub fn vec(&self) -> &Vec<T> {
        &self.0
    }

    /// # Examples
    ///
    /// ```
    /// use atcoder8_library::matrix::Vector;
    ///
    /// let vec = Vector::from(vec![3, -1, 4, 1, -5]);
    /// assert_eq!(vec.to_vec(), vec![3, -1, 4, 1, -5]);
    /// ```
    pub fn to_vec(self) -> Vec<T> {
        self.0
    }

    /// # Examples
    ///
    /// ```
    /// use atcoder8_library::matrix::Vector;
    ///
    /// let vec = Vector::<u32>::from(vec![3, 1, 4, 1, 5]);
    /// let f = |&x: &u32| 2 * x as i32 - 5;
    /// assert_eq!(
    ///     vec.map(&f),
    ///     Vector::from(vec![1, -3, 3, -3, 5])
    /// );
    /// ```
    pub fn map<U, F>(&self, f: &F) -> Vector<U>
    where
        F: Fn(&T) -> U,
    {
        Vector(map_vector(self.vec(), f))
    }
}

impl<T> Vector<T>
where
    T: Zero,
{
    /// # Examples
    ///
    /// ```
    /// use atcoder8_library::matrix::Vector;
    ///
    /// let zero_vec = Vector::from(vec![0, 0, 0]);
    /// assert_eq!(zero_vec.is_zero(), true);
    ///
    /// let not_zero_vec = Vector::from(vec![0, 0, 1]);
    /// assert_eq!(not_zero_vec.is_zero(), false);
    /// ```
    pub fn is_zero(&self) -> bool {
        check_zero_vector(self.vec())
    }
}

impl<T> Vector<T>
where
    T: Clone,
{
    /// # Examples
    ///
    /// ```
    /// use atcoder8_library::matrix::Vector;
    ///
    /// assert_eq!(
    ///     Vector::<u32>::identity(3, 1).convert_type::<u64>(),
    ///     Vector::<u64>::identity(3, 1)
    /// );
    /// ```
    pub fn convert_type<U>(&self) -> Vector<U>
    where
        U: From<T>,
    {
        Vector(convert_type_vector(self.vec()))
    }
}

impl<T> Vector<T>
where
    T: Clone + Zero,
{
    /// # Examples
    ///
    /// ```
    /// use atcoder8_library::matrix::Vector;
    ///
    /// assert_eq!(Vector::<u8>::zero(3), Vector::<u8>::from(vec![0, 0, 0]));
    /// ```
    pub fn zero(n: usize) -> Self {
        assert_ne!(n, 0, "The length of the vector must be at least 1.");

        Self(vec![T::zero(); n])
    }
}

impl<T> Vector<T>
where
    T: Clone + One,
{
    /// # Examples
    ///
    /// ```
    /// use atcoder8_library::matrix::Vector;
    ///
    /// assert_eq!(
    ///     Vector::<u8>::one(3),
    ///     Vector::<u8>::from(vec![1, 1, 1])
    /// );
    /// ```
    pub fn one(n: usize) -> Self {
        assert_ne!(n, 0, "The length of the vector must be at least 1.");

        Self(vec![T::one(); n])
    }
}

impl<T> Vector<T>
where
    T: Clone,
{
    /// # Examples
    ///
    /// ```
    /// use atcoder8_library::matrix::Vector;
    ///
    /// assert_eq!(
    ///     Vector::fill(3, &5),
    ///     Vector::from(vec![5, 5, 5])
    /// );
    /// ```
    pub fn fill(n: usize, elem: &T) -> Self {
        assert_ne!(n, 0, "The length of the vector must be at least 1.");

        Self(vec![elem.clone(); n])
    }
}

impl<T> Vector<T>
where
    T: Clone + Zero + One,
{
    /// # Examples
    ///
    /// ```
    /// use atcoder8_library::matrix::Vector;
    ///
    /// assert_eq!(
    ///     Vector::identity(4, 2),
    ///     Vector::from(vec![0, 0, 1, 0])
    /// );
    /// ```
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

    /// # Examples
    ///
    /// ```
    /// use atcoder8_library::matrix::Vector;
    ///
    /// let vec1 = Vector::from(vec![3, -1, 4]);
    /// let vec2 = Vector::from(vec![1, 5, -9]);
    /// assert_eq!(
    ///     &vec1 + &vec2,
    ///     Vector::from(vec![4, 4, -5])
    /// );
    /// ```
    fn add(self, rhs: Self) -> Self::Output {
        assert_eq!(
            self.len(),
            rhs.len(),
            "The lengths of the vectors must be the same."
        );

        Vector(add_vector(&self.vec(), rhs.vec()))
    }
}

impl<T> Sub for &Vector<T>
where
    T: Clone + Sub<Output = T>,
{
    type Output = Vector<T>;

    /// # Examples
    ///
    /// ```
    /// use atcoder8_library::matrix::Vector;
    ///
    /// let vec1 = Vector::from(vec![3, -1, 4]);
    /// let vec2 = Vector::from(vec![1, 5, -9]);
    /// assert_eq!(
    ///     &vec1 - &vec2,
    ///     Vector::from(vec![2, -6, 13])
    /// );
    /// ```
    fn sub(self, rhs: Self) -> Self::Output {
        assert_eq!(
            self.len(),
            rhs.len(),
            "The lengths of the vectors must be the same."
        );

        Vector(sub_vector(self.vec(), rhs.vec()))
    }
}

impl<T> Mul for &Vector<T>
where
    T: Clone + Add<Output = T> + Mul<Output = T>,
{
    type Output = T;

    /// # Examples
    ///
    /// ```
    /// use atcoder8_library::matrix::Vector;
    ///
    /// let vec1 = Vector::from(vec![3, -1, 4]);
    /// let vec2 = Vector::from(vec![1, 5, -9]);
    /// assert_eq!(&vec1 * &vec2, -38);
    /// ```
    fn mul(self, rhs: Self) -> Self::Output {
        assert_eq!(
            self.len(),
            rhs.len(),
            "The lengths of the vectors must be the same."
        );

        mul_vector(self.vec(), rhs.vec())
    }
}

impl<T> Vector<T>
where
    T: Clone + Mul<Output = T>,
{
    /// # Examples
    ///
    /// ```
    /// use atcoder8_library::matrix::Vector;
    ///
    /// let vec1 = Vector::from(vec![3, -1, 4]);
    /// let vec2 = Vector::from(vec![1, 5, -9]);
    /// assert_eq!(
    ///     vec1.hadamard_prod(&vec2),
    ///     Vector::from(vec![3, -5, -36])
    /// );
    /// ```
    pub fn hadamard_prod(&self, rhs: &Self) -> Self {
        assert_eq!(
            self.len(),
            rhs.len(),
            "The lengths of the vectors must be the same."
        );

        Vector(hadamard_prod_vector(&self.0, &rhs.0))
    }
}

impl<T> Neg for &Vector<T>
where
    T: Clone + Neg<Output = T>,
{
    type Output = Vector<T>;

    /// # Examples
    ///
    /// ```
    /// use atcoder8_library::matrix::Vector;
    ///
    /// let vec = Vector::from(vec![3, -1, 4]);
    /// assert_eq!(-&vec, Vector::from(vec![-3, 1, -4]));
    /// ```
    fn neg(self) -> Self::Output {
        Vector(neg_vector(&self.0))
    }
}

impl<T> AddAssign<&Vector<T>> for Vector<T>
where
    T: Clone + AddAssign,
{
    /// # Examples
    ///
    /// ```
    /// use atcoder8_library::matrix::Vector;
    ///
    /// let mut vec1 = Vector::from(vec![3, -1, 4]);
    /// let vec2 = Vector::from(vec![1, 5, -9]);
    /// vec1 += &vec2;
    /// assert_eq!(vec1, Vector::from(vec![4, 4, -5]));
    /// ```
    fn add_assign(&mut self, rhs: &Vector<T>) {
        assert_eq!(
            self.len(),
            rhs.len(),
            "The lengths of the vectors must be the same."
        );

        add_assign_vector(&mut self.0, rhs.vec())
    }
}

impl<T> SubAssign<&Vector<T>> for Vector<T>
where
    T: Clone + SubAssign,
{
    /// # Examples
    ///
    /// ```
    /// use atcoder8_library::matrix::Vector;
    ///
    /// let mut vec1 = Vector::from(vec![3, -1, 4]);
    /// let vec2 = Vector::from(vec![1, 5, -9]);
    /// vec1 -= &vec2;
    /// assert_eq!(vec1, Vector::from(vec![2, -6, 13]));
    /// ```
    fn sub_assign(&mut self, rhs: &Vector<T>) {
        assert_eq!(
            self.len(),
            rhs.len(),
            "The lengths of the vectors must be the same."
        );

        sub_assign_vector(&mut self.0, rhs.vec())
    }
}

/// # Examples
///
/// ```
/// use atcoder8_library::matrix::Matrix;
///
/// let mat = Matrix::from(vec![vec![3, -1, 4], vec![1, 5, -9]]);
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Matrix<T>(MatByVec<T>);

impl<T> From<MatByVec<T>> for Matrix<T> {
    /// # Examples
    ///
    /// ```
    /// use atcoder8_library::matrix::Matrix;
    ///
    /// let mat = Matrix::from(vec![vec![3, -1, 4], vec![1, 5, -9]]);
    /// ```
    fn from(mat: MatByVec<T>) -> Self {
        assert!(
            check_rect(&mat),
            "The shape of the matrix must be rectangular."
        );

        Self(mat)
    }
}

impl<T> Matrix<T> {
    /// # Examples
    ///
    /// ```
    /// use atcoder8_library::matrix::Matrix;
    ///
    /// let mat = Matrix::from(vec![vec![3, -1, 4], vec![1, 5, -9]]);
    /// assert_eq!(mat.shape(), (2, 3));
    /// ```
    pub fn shape(&self) -> (usize, usize) {
        let Matrix(mat) = self;
        (mat.len(), mat[0].len())
    }

    /// # Examples
    ///
    /// ```
    /// use atcoder8_library::matrix::Matrix;
    ///
    /// let square_mat = Matrix::from(vec![vec![3, -1], vec![4, 1]]);
    /// assert_eq!(square_mat.is_square(), true);
    ///
    /// let not_square_mat = Matrix::from(vec![vec![3, -1, 4], vec![1, 5, -9]]);
    /// assert_eq!(not_square_mat.is_square(), false);
    /// ```
    pub fn is_square(&self) -> bool {
        self.0.len() == self.0[0].len()
    }

    /// # Examples
    ///
    /// ```
    /// use atcoder8_library::matrix::Matrix;
    ///
    /// let mat = Matrix::from(vec![vec![3, -1, 4], vec![1, 5, -9]]);
    /// assert_eq!(
    ///     mat.get((0, 1)),
    ///     &-1
    /// );
    /// ```
    pub fn get(&self, indexes: (usize, usize)) -> &T {
        &self.0[indexes.0][indexes.1]
    }

    /// # Examples
    ///
    /// ```
    /// use atcoder8_library::matrix::Matrix;
    ///
    /// let mut mat = Matrix::from(vec![vec![3, -1, 4], vec![1, 5, -9]]);
    /// mat.set((0, 1), 100);
    /// assert_eq!(
    ///     mat,
    ///     Matrix::from(vec![vec![3, 100, 4], vec![1, 5, -9]])
    /// );
    /// ```
    pub fn set(&mut self, indexes: (usize, usize), elem: T) {
        self.0[indexes.0][indexes.1] = elem;
    }

    /// # Examples
    ///
    /// ```
    /// use atcoder8_library::matrix::Matrix;
    ///
    /// let mat = Matrix::from(vec![vec![3, -1, 4], vec![1, 5, -9]]);
    /// assert_eq!(
    ///     mat.mat_by_vec(),
    ///     &vec![vec![3, -1, 4], vec![1, 5, -9]]
    /// );
    /// ```
    pub fn mat_by_vec(&self) -> &MatByVec<T> {
        &self.0
    }

    /// # Examples
    ///
    /// ```
    /// use atcoder8_library::matrix::Matrix;
    ///
    /// let mat = Matrix::from(vec![vec![3, -1, 4], vec![1, 5, -9]]);
    /// assert_eq!(
    ///     mat.to_mat_by_vec(),
    ///     vec![vec![3, -1, 4], vec![1, 5, -9]]
    /// );
    /// ```
    pub fn to_mat_by_vec(self) -> MatByVec<T> {
        self.0
    }

    /// # Examples
    ///
    /// ```
    /// use atcoder8_library::matrix::Matrix;
    ///
    /// let mat = Matrix::<u32>::from(vec![vec![3, 1, 4], vec![1, 5, 9]]);
    /// let f = |&x: &u32| 2 * x as i32 - 5;
    /// assert_eq!(
    ///     mat.map(&f),
    ///     Matrix::from(vec![vec![1, -3, 3], vec![-3, 5, 13]])
    /// );
    /// ```
    pub fn map<U, F>(&self, f: &F) -> Matrix<U>
    where
        F: Fn(&T) -> U,
    {
        Matrix(map_matrix(self.mat_by_vec(), f))
    }
}

impl<T> Matrix<T>
where
    T: Zero,
{
    /// # Examples
    ///
    /// ```
    /// use atcoder8_library::matrix::Matrix;
    ///
    /// let zero_mat = Matrix::from(vec![vec![0, 0, 0], vec![0, 0, 0]]);
    /// assert_eq!(zero_mat.is_zero(), true);
    ///
    /// let not_zero_mat = Matrix::from(vec![vec![0, 0, 0], vec![0, 1, 0]]);
    /// assert_eq!(not_zero_mat.is_zero(), false);
    /// ```
    pub fn is_zero(&self) -> bool {
        check_zero_matrix(self.mat_by_vec())
    }
}

impl<T> Matrix<T>
where
    T: PartialEq + Zero + One,
{
    /// # Examples
    ///
    /// ```
    /// use atcoder8_library::matrix::Matrix;
    ///
    /// let identity_mat = Matrix::from(vec![vec![1, 0], vec![0, 1]]);
    /// assert_eq!(identity_mat.is_identity(), true);
    ///
    /// let not_identity_mat = Matrix::from(vec![vec![0, 1], vec![1, 0]]);
    /// assert_eq!(not_identity_mat.is_identity(), false);
    /// ```
    pub fn is_identity(&self) -> bool {
        if !self.is_square() {
            return false;
        }

        let n = self.shape().0;

        for i in 0..n {
            for j in 0..n {
                let elem = self.get((i, j));
                if (i == j && !elem.is_one()) || ((i != j) && !elem.is_zero()) {
                    return false;
                }
            }
        }

        true
    }
}

impl<T> Matrix<T>
where
    T: PartialEq,
{
    /// # Examples
    ///
    /// ```
    /// use atcoder8_library::matrix::Matrix;
    ///
    /// let symmetric_mat = Matrix::from(
    ///     vec![
    ///         vec![3, 1, 4],
    ///         vec![1, 1, 5],
    ///         vec![4, 5, 9],
    ///     ]
    /// );
    /// assert_eq!(symmetric_mat.is_symmetric(), true);
    ///
    /// let not_symmetric_mat = Matrix::from(
    ///     vec![
    ///         vec![3, 0, 4],
    ///         vec![1, 1, 5],
    ///         vec![4, 5, 9],
    ///     ]
    /// );
    /// assert_eq!(not_symmetric_mat.is_symmetric(), false);
    /// ```
    pub fn is_symmetric(&self) -> bool {
        if !self.is_square() {
            return false;
        }

        let n = self.shape().0;

        for i in 0..n {
            for j in 0..i {
                if self.get((i, j)) != self.get((j, i)) {
                    return false;
                }
            }
        }

        true
    }
}

impl<T> Matrix<T>
where
    T: Clone,
{
    /// # Examples
    ///
    /// ```
    /// use atcoder8_library::matrix::Matrix;
    ///
    /// assert_eq!(
    ///     Matrix::<u32>::identity(3).convert_type::<u64>(),
    ///     Matrix::<u64>::identity(3)
    /// );
    /// ```
    pub fn convert_type<U>(&self) -> Matrix<U>
    where
        U: From<T>,
    {
        Matrix(convert_type_matrix(self.mat_by_vec()))
    }
}

impl<T> Matrix<T>
where
    T: Clone,
{
    /// # Examples
    ///
    /// ```
    /// use atcoder8_library::matrix::{Vector, Matrix};
    ///
    /// let mat = Matrix::from(vec![vec![3, -1, 4], vec![1, 5, -9]]);
    /// assert_eq!(
    ///     mat.row_vector(0),
    ///     Vector::from(vec![3, -1, 4])
    /// );
    /// ```
    pub fn row_vector(&self, idx: usize) -> Vector<T> {
        Vector(self.0[idx].clone())
    }

    /// # Examples
    ///
    /// ```
    /// use atcoder8_library::matrix::{Vector, Matrix};
    ///
    /// let mat = Matrix::from(vec![vec![3, -1, 4], vec![1, 5, -9]]);
    /// assert_eq!(
    ///     mat.col_vector(2),
    ///     Vector::from(vec![4, -9])
    /// );
    /// ```
    pub fn col_vector(&self, idx: usize) -> Vector<T> {
        Vector((0..self.0.len()).map(|i| self.0[i][idx].clone()).collect())
    }
}

impl<T> Matrix<T>
where
    T: Clone + Zero,
{
    /// # Examples
    ///
    /// ```
    /// use atcoder8_library::matrix::Matrix;
    ///
    /// assert_eq!(
    ///     Matrix::zero((2, 3)),
    ///     Matrix::from(vec![vec![0, 0, 0], vec![0, 0, 0]])
    /// );
    /// ```
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
    /// # Examples
    ///
    /// ```
    /// use atcoder8_library::matrix::Matrix;
    ///
    /// assert_eq!(
    ///     Matrix::one((2, 3)),
    ///     Matrix::from(vec![vec![1, 1, 1], vec![1, 1, 1]]));
    /// ```
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
    /// # Examples
    ///
    /// ```
    /// use atcoder8_library::matrix::Matrix;
    ///
    /// assert_eq!(
    ///     Matrix::fill((2, 3), &5),
    ///     Matrix::from(vec![vec![5, 5, 5], vec![5, 5, 5]])
    /// );
    /// ```
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
    /// # Examples
    ///
    /// ```
    /// use atcoder8_library::matrix::Matrix;
    ///
    /// assert_eq!(
    ///     Matrix::identity(3),
    ///     Matrix::from(vec![vec![1, 0, 0], vec![0, 1, 0], vec![0, 0, 1]])
    /// );
    /// ```
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

    /// Performs the `+` operation.
    ///
    /// # Examples
    ///
    /// ```
    /// use atcoder8_library::matrix::Matrix;
    ///
    /// let mat1 = Matrix::from(vec![vec![3, -1, 4], vec![1, 5, -9]]);
    /// let mat2 = Matrix::from(vec![vec![2, 6, -5], vec![3, 5, -8]]);
    /// assert_eq!(
    ///     &mat1 + &mat2,
    ///     Matrix::from(vec![vec![5, 5, -1], vec![4, 10, -17]])
    /// );
    /// ```
    fn add(self, rhs: Self) -> Self::Output {
        assert_eq!(
            self.shape(),
            rhs.shape(),
            "The shape of the matrix must be the same."
        );

        Matrix(add_matrix(self.mat_by_vec(), rhs.mat_by_vec()))
    }
}

impl<T> Sub for &Matrix<T>
where
    T: Clone + Sub<Output = T>,
{
    type Output = Matrix<T>;

    /// Performs the `-` operation.
    ///
    /// # Examples
    ///
    /// ```
    /// use atcoder8_library::matrix::Matrix;
    ///
    /// let mat1 = Matrix::from(vec![vec![3, -1, 4], vec![1, 5, -9]]);
    /// let mat2 = Matrix::from(vec![vec![2, 6, -5], vec![3, 5, -8]]);
    /// assert_eq!(
    ///     &mat1 - &mat2,
    ///     Matrix::from(vec![vec![1, -7, 9], vec![-2, 0, -1]])
    /// );
    /// ```
    fn sub(self, rhs: Self) -> Self::Output {
        assert_eq!(
            self.shape(),
            rhs.shape(),
            "The shape of the matrix must be the same."
        );

        Matrix(sub_matrix(&self.0, rhs.mat_by_vec()))
    }
}

impl<T> Mul for &Matrix<T>
where
    T: Clone + Add<Output = T> + Mul<Output = T>,
{
    type Output = Matrix<T>;

    /// Performs the `*` operation.
    ///
    /// # Examples
    ///
    /// ```
    /// use atcoder8_library::matrix::Matrix;
    ///
    /// let mat1 = Matrix::from(vec![vec![3, -1], vec![4, 1]]);
    /// let mat2 = Matrix::from(vec![vec![5, 9], vec![2, 6]]);
    /// assert_eq!(
    ///     &mat1 * &mat2,
    ///     Matrix::from(vec![vec![13, 21], vec![22, 42]])
    /// );
    /// ```
    fn mul(self, rhs: Self) -> Self::Output {
        assert_eq!(
            self.shape().1,
            rhs.shape().0,
            "The number of columns in `lhs` and the number of rows in `rhs` are mismatched."
        );

        Matrix(mul_matrix(self.mat_by_vec(), rhs.mat_by_vec()))
    }
}

impl<T> Matrix<T>
where
    T: Clone + Mul<T, Output = T>,
{
    /// # Examples
    ///
    /// ```
    /// use atcoder8_library::matrix::Matrix;
    ///
    /// let mat1 = Matrix::from(vec![vec![3, -1, 4], vec![1, 5, -9]]);
    /// let mat2 = Matrix::from(vec![vec![2, 6, -5], vec![3, 5, -8]]);
    /// assert_eq!(
    ///     mat1.hadamard_prod(&mat2),
    ///     Matrix::from(vec![vec![6, -6, -20], vec![3, 25, 72]])
    /// );
    /// ```
    pub fn hadamard_prod(&self, rhs: &Self) -> Self {
        assert_eq!(
            self.shape(),
            rhs.shape(),
            "The shape of the matrix must be the same."
        );

        Matrix(hadamard_prod_matrix(&self.0, &rhs.0))
    }
}

impl<T> Neg for &Matrix<T>
where
    T: Clone + Neg<Output = T>,
{
    type Output = Matrix<T>;

    /// Performs the unary `-` operation.
    ///
    /// # Examples
    ///
    /// ```
    /// use atcoder8_library::matrix::Matrix;
    ///
    /// let mut mat = Matrix::from(vec![vec![3, -1, 4], vec![1, 5, -9]]);
    /// assert_eq!(
    ///     -&mat,
    ///     Matrix::from(vec![vec![-3, 1, -4], vec![-1, -5, 9]])
    /// );
    /// ```
    fn neg(self) -> Self::Output {
        Matrix(neg_matrix(&self.0))
    }
}

impl<T> AddAssign<&Matrix<T>> for Matrix<T>
where
    T: Clone + AddAssign,
{
    /// # Examples
    ///
    /// ```
    /// use atcoder8_library::matrix::Matrix;
    ///
    /// let mut mat1 = Matrix::from(vec![vec![3, -1, 4], vec![1, 5, -9]]);
    /// let mat2 = Matrix::from(vec![vec![2, 6, -5], vec![3, 5, -8]]);
    /// mat1 += &mat2;
    /// assert_eq!(
    ///     mat1,
    ///     Matrix::from(vec![vec![5, 5, -1], vec![4, 10, -17]])
    /// );
    /// ```
    fn add_assign(&mut self, rhs: &Matrix<T>) {
        assert_eq!(
            self.shape(),
            rhs.shape(),
            "The shape of the matrix must be the same."
        );

        add_assign_matrix(&mut self.0, rhs.mat_by_vec())
    }
}

impl<T> SubAssign<&Matrix<T>> for Matrix<T>
where
    T: Clone + SubAssign,
{
    /// # Examples
    ///
    /// ```
    /// use atcoder8_library::matrix::Matrix;
    ///
    /// let mut mat1 = Matrix::from(vec![vec![3, -1, 4], vec![1, 5, -9]]);
    /// let mat2 = Matrix::from(vec![vec![2, 6, -5], vec![3, 5, -8]]);
    /// mat1 -= &mat2;
    /// assert_eq!(
    ///     mat1,
    ///     Matrix::from(vec![vec![1, -7, 9], vec![-2, 0, -1]])
    /// );
    /// ```
    fn sub_assign(&mut self, rhs: &Matrix<T>) {
        assert_eq!(
            self.shape(),
            rhs.shape(),
            "The shape of the matrix must be the same."
        );

        sub_assign_matrix(&mut self.0, rhs.mat_by_vec())
    }
}

impl<EXP, T> Pow<EXP> for &Matrix<T>
where
    EXP: Clone + PartialOrd + Zero + One + BitAnd<Output = EXP> + ShrAssign,
    T: Clone + Zero + One + Add<Output = T> + Mul<Output = T>,
{
    type Output = Matrix<T>;

    /// # Examples
    ///
    /// ```
    /// use num_traits::Pow;
    /// use atcoder8_library::matrix::Matrix;
    ///
    /// let mat = Matrix::from(vec![vec![3, -1], vec![4, 1]]);
    /// assert_eq!(
    ///     mat.pow(3),
    ///     Matrix::from(vec![vec![-1, -9], vec![36, -19]])
    /// );
    /// ```
    fn pow(self, exp: EXP) -> Self::Output {
        assert!(
            exp >= EXP::zero(),
            "The exponent must be greater than or equal to 0."
        );
        assert!(self.is_square(), "It must be a square matrix.");

        let mut output_mat = Matrix::<T>::identity(self.0.len());

        let mut mul_mat = self.clone();

        let mut x = exp.clone();
        while x > EXP::zero() {
            if !(x.clone() & EXP::one()).is_zero() {
                output_mat = output_mat.mul(&mul_mat);
            }

            mul_mat = mul_mat.mul(&mul_mat);
            x >>= EXP::one();
        }

        output_mat
    }
}

impl<T> Mul<&Vector<T>> for &Matrix<T>
where
    T: Clone + Add<Output = T> + Mul<Output = T>,
{
    type Output = Vector<T>;

    /// # Examples
    ///
    /// ```
    /// use atcoder8_library::matrix::{Matrix, Vector};
    ///
    /// let mat = Matrix::from(vec![vec![3, -1], vec![4, 1]]);
    /// let vec = Vector::from(vec![5, -9]);
    /// assert_eq!(
    ///     &mat * &vec,
    ///     Vector::from(vec![24, 11])
    /// );
    /// ```
    fn mul(self, rhs: &Vector<T>) -> Self::Output {
        Vector(self.0.iter().map(|x| mul_vector(x, &rhs.0)).collect())
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
    lhs.iter()
        .zip(rhs.iter())
        .map(|(x, y)| x.clone() + y.clone())
        .collect()
}

fn sub_vector<T>(lhs: &Vec<T>, rhs: &Vec<T>) -> Vec<T>
where
    T: Clone + Sub<Output = T>,
{
    lhs.iter()
        .zip(rhs.iter())
        .map(|(x, y)| x.clone() - y.clone())
        .collect()
}

fn mul_vector<T>(lhs: &Vec<T>, rhs: &Vec<T>) -> T
where
    T: Clone + Add<Output = T> + Mul<Output = T>,
{
    lhs.iter()
        .zip(rhs.iter())
        .map(|(x, y)| x.clone() * y.clone())
        .reduce(Add::add)
        .unwrap()
}

fn hadamard_prod_vector<T>(lhs: &Vec<T>, rhs: &Vec<T>) -> Vec<T>
where
    T: Clone + Mul<Output = T>,
{
    lhs.iter()
        .zip(rhs.iter())
        .map(|(x, y)| x.clone() * y.clone())
        .collect()
}

fn neg_vector<T>(vec: &Vec<T>) -> Vec<T>
where
    T: Clone + Neg<Output = T>,
{
    vec.iter().map(|x| -x.clone()).collect()
}

fn add_assign_vector<T>(lhs: &mut Vec<T>, rhs: &Vec<T>)
where
    T: Clone + AddAssign<T>,
{
    lhs.iter_mut()
        .zip(rhs.iter())
        .for_each(|(x, y)| *x += y.clone());
}

fn sub_assign_vector<T>(lhs: &mut Vec<T>, rhs: &Vec<T>)
where
    T: Clone + SubAssign<T>,
{
    lhs.iter_mut()
        .zip(rhs.iter())
        .for_each(|(x, y)| *x -= y.clone());
}

fn convert_type_vector<T, U>(vec: &Vec<T>) -> Vec<U>
where
    T: Clone,
    U: From<T>,
{
    vec.iter().map(|x| U::from(x.clone())).collect()
}

fn check_zero_vector<T>(vec: &Vec<T>) -> bool
where
    T: Zero,
{
    vec.iter().all(|x| x.is_zero())
}

fn add_matrix<T>(lhs: &MatByVec<T>, rhs: &MatByVec<T>) -> MatByVec<T>
where
    T: Clone + Add<Output = T>,
{
    lhs.iter()
        .zip(rhs.iter())
        .map(|(x, y)| add_vector(x, y))
        .collect()
}

fn sub_matrix<T>(lhs: &MatByVec<T>, rhs: &MatByVec<T>) -> MatByVec<T>
where
    T: Clone + Sub<Output = T>,
{
    lhs.iter()
        .zip(rhs.iter())
        .map(|(x, y)| sub_vector(x, y))
        .collect()
}

fn hadamard_prod_matrix<T>(lhs: &MatByVec<T>, rhs: &MatByVec<T>) -> MatByVec<T>
where
    T: Clone + Mul<Output = T>,
{
    lhs.iter()
        .zip(rhs.iter())
        .map(|(x, y)| hadamard_prod_vector(x, y))
        .collect()
}

fn neg_matrix<T>(mat: &MatByVec<T>) -> MatByVec<T>
where
    T: Clone + Neg<Output = T>,
{
    mat.iter().map(|x| neg_vector(x)).collect()
}

fn add_assign_matrix<T>(lhs: &mut MatByVec<T>, rhs: &MatByVec<T>)
where
    T: Clone + AddAssign,
{
    lhs.iter_mut()
        .zip(rhs.iter())
        .for_each(|(x, y)| add_assign_vector(x, y));
}

fn sub_assign_matrix<T>(lhs: &mut MatByVec<T>, rhs: &MatByVec<T>)
where
    T: Clone + SubAssign,
{
    lhs.iter_mut()
        .zip(rhs.iter())
        .for_each(|(x, y)| sub_assign_vector(x, y));
}

fn mul_matrix_cell<T>(lhs: &MatByVec<T>, rhs: &MatByVec<T>, pos: (usize, usize)) -> T
where
    T: Clone + Add<Output = T> + Mul<Output = T>,
{
    (0..lhs[0].len())
        .map(|i| lhs[pos.0][i].clone() * rhs[i][pos.1].clone())
        .reduce(Add::add)
        .unwrap()
}

fn mul_matrix_row<T>(lhs: &MatByVec<T>, rhs: &MatByVec<T>, idx: usize) -> Vec<T>
where
    T: Clone + Add<Output = T> + Mul<Output = T>,
{
    (0..rhs[0].len())
        .map(|i| mul_matrix_cell(lhs, rhs, (idx, i)))
        .collect()
}

fn mul_matrix<T>(lhs: &MatByVec<T>, rhs: &MatByVec<T>) -> MatByVec<T>
where
    T: Clone + Add<Output = T> + Mul<Output = T>,
{
    (0..lhs.len())
        .map(|i| mul_matrix_row(lhs, rhs, i))
        .collect()
}

fn convert_type_matrix<T, U>(mat: &MatByVec<T>) -> MatByVec<U>
where
    T: Clone,
    U: From<T>,
{
    mat.iter().map(|x| convert_type_vector(x)).collect()
}

fn check_zero_matrix<T>(mat: &MatByVec<T>) -> bool
where
    T: Zero,
{
    mat.iter().all(|x| check_zero_vector(x))
}

fn map_vector<T, U, F>(vec: &Vec<T>, f: &F) -> Vec<U>
where
    F: Fn(&T) -> U,
{
    vec.iter().map(|x| f(x)).collect()
}

fn map_matrix<T, U, F>(mat: &MatByVec<T>, f: &F) -> MatByVec<U>
where
    F: Fn(&T) -> U,
{
    mat.iter().map(|x| map_vector(x, f)).collect()
}
