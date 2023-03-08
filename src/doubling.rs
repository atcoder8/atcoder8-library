//! Takes an array containing the transition destinations and performs doubling.

/// This structure stores the result of doubling.
///
/// # Examples
///
/// ```
/// use atcoder8_library::doubling::Doubling;
///
/// let mut doubling = Doubling::new(&vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 0]);
/// assert_eq!(doubling.get_destination(3, 0), 3);
/// assert_eq!(doubling.get_destination(3, 1), 4);
/// assert_eq!(doubling.get_destination(5, 1000000007), 2);
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Doubling {
    transitions: Vec<Vec<usize>>,
}

impl Doubling {
    /// The structure for doubling is created
    /// from the array storing the destination of the transition.
    ///
    /// # Examples
    ///
    /// ```
    /// use atcoder8_library::doubling::Doubling;
    ///
    /// let mut doubling = Doubling::new(&vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 0]);
    /// assert_eq!(doubling.get_destination(5, 1000000007), 2);
    /// ```
    pub fn new(trans: &[usize]) -> Self {
        Self {
            transitions: vec![trans.to_owned()],
        }
    }

    /// Use doubling to find the destination
    /// after the specified number of transitions have taken place.
    ///
    /// # Arguments
    ///
    /// * start - Initial index.
    /// * trans_num - Number of transitions.
    /// 
    /// # Examples
    ///
    /// ```
    /// use atcoder8_library::doubling::Doubling;
    ///
    /// let mut doubling = Doubling::new(&vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 0]);
    /// assert_eq!(doubling.get_destination(5, 1000000007), 2);
    pub fn get_destination(&mut self, start: usize, trans_num: usize) -> usize {
        let n = self.transitions[0].len();

        let mut dest = start;

        for i in (0..).take_while(|&i| trans_num >> i != 0) {
            if i == self.transitions.len() {
                self.transitions.push(
                    (0..n)
                        .map(|j| self.transitions[i - 1][self.transitions[i - 1][j]])
                        .collect(),
                );
            }

            if (trans_num >> i) & 1 == 1 {
                dest = self.transitions[i][dest];
            }
        }

        dest
    }
}

impl From<Vec<usize>> for Doubling {
    /// Converts the array storing the destination of the transition
    /// into a structure for doubling.
    ///
    /// # Examples
    ///
    /// ```
    /// use atcoder8_library::doubling::Doubling;
    ///
    /// let mut doubling = Doubling::from(vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 0]);
    /// assert_eq!(doubling.get_destination(5, 1000000007), 2);
    /// ```
    fn from(trans: Vec<usize>) -> Self {
        Self {
            transitions: vec![trans],
        }
    }
}

#[cfg(test)]
mod test {
    use super::Doubling;

    fn get_expected_dest(trans: &[usize], start: usize, trans_num: usize) -> usize {
        (0..trans_num).fold(start, |dest, _| trans[dest])
    }

    #[test]
    fn test() {
        let trans = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3];
        let trans_num_seq = [0, 1, 2, 3, 10, 100, 1000];

        for start in 0..trans.len() {
            for trans_num in trans_num_seq {
                let expected_dest = get_expected_dest(&trans, start, trans_num);
                let mut doubling = Doubling::new(&trans);

                assert_eq!(expected_dest, doubling.get_destination(start, trans_num));
            }
        }
    }
}
