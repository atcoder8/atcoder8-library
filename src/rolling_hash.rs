//! Module for rolling hash.

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

/// Radixes used to calculate hash values.
pub const RADIXES: HashValue = [252895580, 406082094, 892791812, 869052263, 261298120];

/// Generates a hash value from a sequence using Rabin-Karp algorithm.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RollingHash {
    /// Length of the sequence.
    len: usize,

    /// Hash value corresponding to the sequence.
    hash_value: HashValue,

    /// Sequence length power of the radix.
    /// This array is used to calculate the hash value corresponding to the concatenated sequence.
    raised_radixes: HashValue,
}

impl Default for RollingHash {
    fn default() -> Self {
        Self {
            len: 0,
            hash_value: [0; HASH_BLOCK_NUM],
            raised_radixes: [1; HASH_BLOCK_NUM],
        }
    }
}

impl<T, I> From<I> for RollingHash
where
    HashBlock: From<T>,
    I: IntoIterator<Item = T>,
{
    /// Generates a hash value from a sequence.
    fn from(seq: I) -> Self {
        let mut hash = RollingHash::new();
        hash.extend(seq);

        hash
    }
}

impl RollingHash {
    /// Generates a hash value corresponding to an empty sequence.
    pub fn new() -> Self {
        Self {
            len: 0,
            raised_radixes: [1; HASH_BLOCK_NUM],
            hash_value: [0; HASH_BLOCK_NUM],
        }
    }

    /// Generates a hash value from a slice of the sequence.
    pub fn from_slice<T>(seq: &[T]) -> Self
    where
        HashBlock: From<T>,
        T: Copy,
    {
        Self::from(seq.iter().cloned())
    }

    /// Generates a hash value from a string slice.
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(s: &str) -> Self {
        Self::from(s.chars())
    }

    /// Generates a hash value from a slice with elements of type `usize`.
    pub fn from_usize_slice(seq: &[usize]) -> Self {
        Self::from(seq.iter().map(|&elem| elem as HashBlock))
    }

    /// Generates a hash value from a sequence with elements of type `usize`.
    pub fn from_usize<I>(seq: I) -> Self
    where
        I: IntoIterator<Item = usize>,
    {
        Self::from(seq.into_iter().map(|elem| elem as HashBlock))
    }

    /// Returns the length of the sequence.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns whether the sequence is empty or not.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Adds an element to the end of the sequence.
    pub fn push<T>(&mut self, elem: T)
    where
        HashBlock: From<T>,
    {
        self.len += 1;

        let elem = HashBlock::from(elem);
        for block_idx in 0..HASH_BLOCK_NUM {
            let radix = RADIXES[block_idx];
            let modulus = MODULI[block_idx];

            let block = &mut self.hash_value[block_idx];
            *block = (*block * radix % modulus + elem % modulus) % modulus;

            let raised_radix = &mut self.raised_radixes[block_idx];
            *raised_radix = *raised_radix * radix % modulus;
        }
    }

    /// Adds some elements to the end of the sequence.
    pub fn extend<T, I>(&mut self, elements: I)
    where
        HashBlock: From<T>,
        I: IntoIterator<Item = T>,
    {
        elements.into_iter().for_each(|elem| self.push(elem));
    }

    /// Concatenates another sequence behind the sequence.
    pub fn concat(&mut self, other: &RollingHash) {
        self.len += other.len;

        for (block_idx, modulus) in MODULI.iter().enumerate() {
            let block = &mut self.hash_value[block_idx];
            *block = (*block * other.raised_radixes[block_idx] % modulus
                + other.hash_value[block_idx])
                % modulus;

            let raised_radix = &mut self.raised_radixes[block_idx];
            *raised_radix = *raised_radix * other.raised_radixes[block_idx] % modulus;
        }
    }

    /// Generates a hash value from a chained sequence.
    pub fn chain(&self, other: &RollingHash) -> Self {
        let mut concatenated_hash = *self;
        concatenated_hash.concat(other);

        concatenated_hash
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chain() {
        let s1 = "Hello, ";
        let s2 = "world!";
        let s3 = "Hello, world!";

        let hash1 = RollingHash::from_str(&s1);
        let hash2 = RollingHash::from_str(&s2);
        let hash3 = RollingHash::from_str(&s3);

        let chained_hash = hash1.chain(&hash2);
        assert_eq!(chained_hash.hash_value, hash3.hash_value);

        assert_eq!(hash1.chain(&hash2), hash3);
        assert_ne!(hash2.chain(&hash1), hash3);
    }

    #[test]
    fn test_multiple_chain() {
        let hash1 = RollingHash::from_str("He");
        let hash2 = RollingHash::from_str("ll");
        let hash3 = RollingHash::from_str("o");

        assert_eq!(
            hash1.chain(&hash2).chain(&hash3),
            RollingHash::from_str("Hello")
        );
    }

    #[test]
    fn test_integer() {
        let seq1: Vec<u64> = vec![
            4968, 8374, 8446, 395, 1388, 9837, 7027, 5723, 1073, 1598, 265, 3190, 119, 6284, 9488,
            4598, 6752, 6639, 7483, 3676,
        ];
        let seq2: Vec<u64> = vec![
            5342, 2733, 3090, 6464, 555, 1967, 7656, 3033, 1110, 4966, 5051, 2489, 2104, 9109,
            9822, 2918, 6309, 4331, 5356, 3155,
        ];
        let chained_seq = seq1.iter().chain(&seq2).cloned().collect::<Vec<u64>>();

        let hash1 = RollingHash::from_slice(&seq1);
        let hash2 = RollingHash::from_slice(&seq2);

        assert_eq!(hash1.len() + hash2.len(), hash1.chain(&hash2).len());
        assert_eq!(hash1.chain(&hash2), RollingHash::from_slice(&chained_seq));
        assert_ne!(hash2.chain(&hash1), RollingHash::from_slice(&chained_seq));
    }

    #[test]
    fn test_large_integer() {
        let seq1: Vec<usize> = vec![
            17095246770412609619,
            2853762817256026923,
            5652629308372694810,
            13266787961747173123,
            158338808912957426,
            14413130188265557047,
            15112893334079214343,
            4140195119943674846,
            17500907723110707926,
            16864396662152713084,
        ];
        let seq2: Vec<usize> = vec![
            867085608603385880,
            14922961579422084500,
            13723401766266717608,
            13899092912652492423,
            10886714825703944663,
            651166826599583269,
            8658613210831637418,
            4275977026533433766,
            10820216705580164009,
            14513572400063478592,
        ];
        let chained_seq = seq1.iter().chain(&seq2).cloned().collect::<Vec<usize>>();

        let hash1 = RollingHash::from_usize_slice(&seq1);
        let hash2 = RollingHash::from_usize_slice(&seq2);

        assert_eq!(hash1.len() + hash2.len(), hash1.chain(&hash2).len());
        assert_eq!(
            hash1.chain(&hash2),
            RollingHash::from_usize_slice(&chained_seq)
        );
        assert_ne!(
            hash2.chain(&hash1),
            RollingHash::from_usize_slice(&chained_seq)
        );
    }

    #[test]
    fn test_same_elements() {
        let seq1: Vec<u64> = vec![3, 3, 3, 3, 3, 3, 3, 3];
        let seq2: Vec<u64> = vec![3, 3, 3];
        let chained_seq = seq1.iter().chain(&seq2).cloned().collect::<Vec<u64>>();

        let hash1 = RollingHash::from_slice(&seq1);
        let hash2 = RollingHash::from_slice(&seq2);

        assert_eq!(hash1.len() + hash2.len(), hash1.chain(&hash2).len());
        assert_eq!(hash1.chain(&hash2), RollingHash::from_slice(&chained_seq));
        assert_eq!(hash2.chain(&hash1), RollingHash::from_slice(&chained_seq));
    }

    #[test]
    fn test_push() {
        let hash1 = RollingHash::from_str("Hello, world!");
        let mut hash2 = RollingHash::from_str("Hello, world");
        hash2.push('!');

        assert_eq!(hash1, hash2);
    }
}
