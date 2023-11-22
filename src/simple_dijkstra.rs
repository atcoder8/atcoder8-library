//! Calculates the shortest distance from a single node to each node using the Dijkstra's method.

use std::{cmp::Reverse, collections::BinaryHeap, ops::Add};

/// Trait for edge weight.
pub trait Weight: Clone + Ord + Add<Self, Output = Self> {
    /// Additive identity.
    fn zero() -> Self;
}

/// Calculates the shortest distance from a single node to each node using the Dijkstra's method.
///
/// The distance between nodes is defined by the sum of the weights of the edges in the path.
/// Edge weights must be non-negative.
/// Distances to unreachable nodes are denoted by `None`.
///
/// # Arguments
///
/// * `node_num` - Number of nodes.
/// * `edges` - Weighted edges.
/// * `start` - starting point.
/// * `bidirectional` - Specify `true` if the edges are undirected.
///
/// # Examples
///
/// ```
/// # use atcoder8_library::simple_dijkstra::dijkstra_with_edges;
/// #
/// let node_num = 6;
/// let edges = vec![(0, 1, 1), (0, 2, 10), (1, 3, 1), (2, 3, 1), (4, 5, 3)];
///
/// let dists = dijkstra_with_edges::<u32>(node_num, &edges, 0, true);
/// assert_eq!(dists, vec![Some(0), Some(1), Some(3), Some(2), None, None]);
/// ```
///
/// ```
/// # use atcoder8_library::simple_dijkstra::dijkstra_with_edges;
/// #
/// let node_num = 6;
/// let edges = vec![(0, 1, 1), (0, 2, 10), (1, 3, 1), (2, 3, 1), (4, 5, 3)];
///
/// let dists = dijkstra_with_edges::<u32>(node_num, &edges, 0, false);
/// assert_eq!(dists, vec![Some(0), Some(1), Some(10), Some(2), None, None]);
/// ```
pub fn dijkstra_with_edges<W>(
    node_num: usize,
    edges: &[(usize, usize, W)],
    start: usize,
    bidirectional: bool,
) -> Vec<Option<W>>
where
    W: Weight,
{
    let mut graph = vec![vec![]; node_num];
    for (u, v, weight) in edges {
        graph[*u].push((*v, weight.clone()));

        if bidirectional {
            graph[*v].push((*u, weight.clone()));
        }
    }

    dijkstra(&graph, start)
}

/// Calculates the shortest distance from a single node to each node using the Dijkstra's method.
///
/// The distance between nodes is defined by the sum of the weights of the edges in the path.
/// Edge weights must be non-negative.
/// Distances to unreachable nodes are denoted by `None`.
///
/// # Arguments
///
/// * `graphs` - Adjacency list.
/// * `start` - starting point.
///
/// # Examples
///
/// ```
/// # use atcoder8_library::simple_dijkstra::dijkstra;
///
/// let node_num = 6;
/// let graph = vec![vec![(1, 1), (2, 10)], vec![(0, 1), (3, 1)], vec![(0, 10), (3, 1)], vec![(1, 1), (2, 1)], vec![(5, 3)], vec![(4, 3)]];
///
/// let dists = dijkstra::<u32>(&graph, 0);
/// assert_eq!(dists, vec![Some(0), Some(1), Some(3), Some(2), None, None]);
/// ```
pub fn dijkstra<W>(graph: &[Vec<(usize, W)>], start: usize) -> Vec<Option<W>>
where
    W: Weight,
{
    let mut costs = vec![None; graph.len()];
    costs[start] = Some(W::zero());

    let mut heap = BinaryHeap::new();
    heap.push((Reverse(W::zero()), start));

    while let Some((Reverse(cost), cur)) = heap.pop() {
        if costs[cur].as_ref() != Some(&cost) {
            continue;
        }

        for (next, edge_cost) in &graph[cur] {
            let cand_cost = cost.clone() + edge_cost.clone();
            let next_cost = &mut costs[*next];

            if next_cost.is_none() || &cand_cost < next_cost.as_ref().unwrap() {
                *next_cost = Some(cand_cost.clone());
                heap.push((Reverse(cand_cost), *next));
            }
        }
    }

    costs
}

/// Macros to implement `Weight` trait to the built-in integer types.
macro_rules! impl_weight_for_builtin_integer {
    ($($ty: tt), *) => {
        $(
            impl Weight for $ty {
                fn zero() -> Self {
                    0
                }
            }
        )*
    };
}

// Implements the `Weight` trait to the built-in integer types.
impl_weight_for_builtin_integer!(u8, u16, u32, u64, u128, usize, i8, i16, i32, i64, i128, isize);
