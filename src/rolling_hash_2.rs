//! Module for rolling hash.

use std::{collections::VecDeque, iter::zip};

use rand::Rng;

/// The type of the blocks that make up the hash.
pub type HashBlock = u64;

/// Number of integers that make up the hash value.
pub const HASH_BLOCK_NUM: usize = 5;

/// Type of hash value.
///
/// A hash value consists of several integers.
pub type HashValue = [HashBlock; HASH_BLOCK_NUM];

/// Moduli used to calculate hash values.
pub const MODULI: HashValue = [1000002637, 1000011659, 1000012631, 1000017841, 1000018603];

// /// Radixes used to calculate hash values.
// pub const RADIXES: HashValue = [252895580, 406082094, 892791812, 869052263, 261298120];

/// Returns `x` such that `a * x` is equivalent to `1` with `m` as the modulus.
fn modinv(a: u32, m: u32) -> u32 {
    let (mut a, mut b, mut s, mut t) = (a as i64, m as i64, 1, 0);
    while b != 0 {
        let q = a / b;
        a -= q * b;
        std::mem::swap(&mut a, &mut b);
        s -= q * t;
        std::mem::swap(&mut s, &mut t);
    }

    assert_eq!(
        a.abs(),
        1,
        "\
There is no multiplicative inverse of `a` with `m` as the modulus, \
because `a` and `m` are not prime to each other (gcd(a, m) = {}).",
        a.abs()
    );

    ((s % m as i64 + m as i64) % m as i64) as u32
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Hasher {
    radixes: HashValue,
    inverse_radixes: HashValue,
}

impl Hasher {
    /// Generates a `Hasher` using a random number.
    pub fn new() -> Self {
        let mut thread_rng = rand::thread_rng();

        let mut radixes = [0; HASH_BLOCK_NUM];
        let mut inverse_radixes = [0; HASH_BLOCK_NUM];

        for block_idx in 0..HASH_BLOCK_NUM {
            let modulus = MODULI[block_idx];
            radixes[block_idx] = thread_rng.gen_range(0..modulus);
            inverse_radixes[block_idx] =
                modinv(radixes[block_idx] as u32, modulus as u32) as HashBlock;
        }

        Self {
            radixes,
            inverse_radixes,
        }
    }

    /// Converts a single element to a remainder per block.
    fn to_hash_elem<T>(elem: T) -> HashValue
    where
        HashBlock: From<T>,
    {
        let elem = HashBlock::from(elem);

        let mut hash = [0; HASH_BLOCK_NUM];
        zip(&mut hash, MODULI).for_each(|(block, modulus)| *block = elem % modulus);

        hash
    }

    /// Generates a hash value corresponding to an empty sequence.
    pub fn empty_hash(&self) -> RollingHash {
        RollingHash {
            hasher: self,
            hash_elements: VecDeque::new(),
            hash_value: [0; HASH_BLOCK_NUM],
            raised_radixes: [1; HASH_BLOCK_NUM],
        }
    }

    /// Generates a hash value from a iterator of the sequence.
    pub fn hash_from_iter<T, I>(&self, seq: I) -> RollingHash
    where
        HashBlock: From<T>,
        I: IntoIterator<Item = T>,
    {
        let mut hash = self.empty_hash();
        hash.extend(seq);

        hash
    }

    /// Generates a hash value from a slice of the sequence.
    pub fn hash_from_slice<T>(&self, seq: &[T]) -> RollingHash
    where
        HashBlock: From<T>,
        T: Copy,
    {
        self.hash_from_iter(seq.iter().cloned())
    }

    /// Generates a hash value from a string slice.
    pub fn hash_from_str(&self, s: &str) -> RollingHash {
        self.hash_from_iter(s.chars())
    }

    /// Generates a hash value from a slice with elements of type `usize`.
    pub fn hash_from_usize_slice(&self, seq: &[usize]) -> RollingHash {
        self.hash_from_iter(seq.iter().map(|&elem| elem as HashBlock))
    }

    /// Generates a hash value from a sequence with elements of type `usize`.
    pub fn hash_from_usize_iter<I>(&self, seq: I) -> RollingHash
    where
        I: IntoIterator<Item = usize>,
    {
        self.hash_from_iter(seq.into_iter().map(|elem| elem as HashBlock))
    }
}

/// Generates a hash value from a sequence using Rabin-Karp algorithm.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RollingHash<'a> {
    hasher: &'a Hasher,

    /// Hash for each element of the sequence.
    hash_elements: VecDeque<HashValue>,

    /// Hash value corresponding to the sequence.
    hash_value: HashValue,

    /// Sequence length power of the radix.
    /// This array is used to calculate the hash value corresponding to the concatenated sequence.
    raised_radixes: HashValue,
}

impl<'a> RollingHash<'a> {
    /// Returns the length of the sequence.
    pub fn len(&self) -> usize {
        self.hash_elements.len()
    }

    /// Returns whether the sequence is empty or not.
    pub fn is_empty(&self) -> bool {
        self.hash_elements.is_empty()
    }

    fn push_front_hash_elem(&mut self, hash_elem: HashValue) {
        self.hash_elements.push_front(hash_elem);

        for (block_idx, (block, raised_radix)) in
            zip(&mut self.hash_value, &mut self.raised_radixes).enumerate()
        {
            let radix = self.hasher.radixes[block_idx];
            let modulus = MODULI[block_idx];

            *block = (*block * *raised_radix % modulus + hash_elem[block_idx]) % modulus;
            *raised_radix = *raised_radix * radix % modulus;
        }
    }

    /// Adds a hashed element to the sequence.
    fn push_back_hash_elem(&mut self, hash_elem: HashValue) {
        self.hash_elements.push_back(hash_elem);

        for (block_idx, (block, raised_radix)) in
            zip(&mut self.hash_value, &mut self.raised_radixes).enumerate()
        {
            let radix = self.hasher.radixes[block_idx];
            let modulus = MODULI[block_idx];

            *block = (*block * radix % modulus + hash_elem[block_idx]) % modulus;
            *raised_radix = *raised_radix * radix % modulus;
        }
    }

    /// Adds an element to the head of the sequence.
    pub fn push_front<T>(&mut self, elem: T)
    where
        HashBlock: From<T>,
    {
        let hash_elem = Hasher::to_hash_elem(elem);
        self.push_front_hash_elem(hash_elem);
    }

    /// Adds an element to the end of the sequence.
    pub fn push_back<T>(&mut self, elem: T)
    where
        HashBlock: From<T>,
    {
        let hash_elem = Hasher::to_hash_elem(elem);
        self.push_back_hash_elem(hash_elem);
    }

    pub fn pop_front(&mut self) -> Option<HashValue> {
        let hash_elem = self.hash_elements.pop_front()?;

        for (block_idx, (block, raised_radix)) in
            zip(&mut self.hash_value, &mut self.raised_radixes).enumerate()
        {
            let inv_radix = self.hasher.inverse_radixes[block_idx];
            let modulus = MODULI[block_idx];

            *raised_radix = *raised_radix * inv_radix % modulus;
            *block = (*block + modulus - hash_elem[block_idx] * *raised_radix % modulus) % modulus;
        }

        Some(hash_elem)
    }

    /// Removes the last element from a sequence.
    pub fn pop_back(&mut self) -> Option<HashValue> {
        let hash_elem = self.hash_elements.pop_back()?;

        for (block_idx, (block, raised_radix)) in
            zip(&mut self.hash_value, &mut self.raised_radixes).enumerate()
        {
            let inv_radix = self.hasher.inverse_radixes[block_idx];
            let modulus = MODULI[block_idx];

            *raised_radix = *raised_radix * inv_radix % modulus;
            *block = (*block + modulus - hash_elem[block_idx]) % modulus * inv_radix % modulus;
        }

        Some(hash_elem)
    }

    /// Adds some elements to the end of the sequence.
    pub fn extend<T, I>(&mut self, elements: I)
    where
        HashBlock: From<T>,
        I: IntoIterator<Item = T>,
    {
        elements.into_iter().for_each(|elem| self.push_back(elem));
    }

    /// Calculates the hash value corresponding to the concatenated sequence.
    pub fn concat_hash(&self, other: &RollingHash) -> HashValue {
        assert_eq!(
            self.hasher, other.hasher,
            "The hashes to be compared must have been generated by the same `Hasher`."
        );

        let mut hash_value = self.hash_value;
        for (block_idx, block) in hash_value.iter_mut().enumerate() {
            let modulus = MODULI[block_idx];
            *block = (*block * other.raised_radixes[block_idx] % modulus
                + other.hash_value[block_idx])
                % modulus;
        }

        hash_value
    }

    /// Concatenates another sequence behind the sequence.
    pub fn concat(&mut self, other: &RollingHash) {
        assert_eq!(
            self.hasher, other.hasher,
            "The hashes to be compared must have been generated by the same `Hasher`."
        );

        for &hash_elem in &other.hash_elements {
            self.push_back_hash_elem(hash_elem);
        }
    }

    /// Generates a hash value from a chained sequence.
    pub fn chained(&self, other: &RollingHash) -> Self {
        let mut concatenated_hash = self.clone();
        concatenated_hash.concat(other);

        concatenated_hash
    }
}
