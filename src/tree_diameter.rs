//! This module implements an algorithm to find the diameter of a tree.

use std::{collections::VecDeque, mem};

/// Returns the vertex (or one of them if more than one exists)
/// with the maximum distance from the `source_vertex` and its distance;
/// where the distance between two vertices is the number of edges
/// on the shortest path connecting them.
///
/// # Time complexity
///
/// Θ(`n`)
///
/// * `n` - Number of vertices in the tree.
///
/// # Examples
///
/// Take the following tree as an examples.
///
/// ```
/// # /*
///   0
///  / \
/// 1   2
///    / \
///   3   4
///   |
///   5
/// # */
///
/// use atcoder8_library::tree_diameter::find_furthest_vertex;
///
/// let tree = vec![
///     vec![1, 2], vec![0], vec![0, 3, 4],
///     vec![2, 5], vec![2], vec![3]
/// ];
/// assert_eq!(find_furthest_vertex(&tree, 0), (5, 3));
/// assert_eq!(find_furthest_vertex(&tree, 5), (1, 4));
/// ```
pub fn find_furthest_vertex(tree: &[Vec<usize>], source_vertex: usize) -> (usize, usize) {
    let mut que: VecDeque<(Option<usize>, usize, usize)> =
        VecDeque::from(vec![(None, source_vertex, 0)]);
    let mut visited = vec![false; tree.len()];
    visited[source_vertex] = true;

    loop {
        let (pare, curr, dist) = que.pop_front().unwrap();

        for &next in tree[curr].iter().filter(|&&next| Some(next) != pare) {
            assert!(!visited[next], "A closed path exists.");

            que.push_back((Some(curr), next, dist + 1));
            visited[next] = true;
        }

        if que.is_empty() {
            break (curr, dist);
        }
    }
}

/// Returns the pair of vertices (or one of them if more than one exists)
/// such that the distance is maximal, and the distance between them;
/// where the distance between two vertices is the number of edges
/// on the shortest path connecting them.
///
/// If both vertices are returned as `(end1, end2)`, then `end1 <= end2`.
///
/// # Time complexity
///
/// Θ(`n`)
///
/// * `n` - Number of vertices in the tree.
///
/// # Examples
///
/// Take the following tree as an examples.
/// Both ends are 1 and 5 and the diameter is 4.
///
/// ```
/// # /*
///   0
///  / \
/// 1   2
///    / \
///   3   4
///   |
///   5
/// # */
///
/// use atcoder8_library::tree_diameter::find_tree_diameter;
///
/// let tree = vec![
///     vec![1, 2], vec![0], vec![0, 3, 4],
///     vec![2, 5], vec![2], vec![3]
/// ];
/// assert_eq!(find_tree_diameter(&tree), ((1, 5), 4));
/// ```
pub fn find_tree_diameter(tree: &[Vec<usize>]) -> ((usize, usize), usize) {
    assert!(!tree.is_empty(), "Must have 1 or move vertices.");

    let mut end1 = find_furthest_vertex(tree, 0).0;
    let (mut end2, dist) = find_furthest_vertex(tree, end1);

    if end1 > end2 {
        mem::swap(&mut end1, &mut end2);
    }

    ((end1, end2), dist)
}
