use std::{cmp::Reverse, collections::BinaryHeap};

/// For each node, the cost and the previous node of the shortest path
/// from `start_node` are found.
///
/// The element of the list corresponding to the node for which
/// there is no path from `start_node` is `None`.
///
/// # Arguments
///
/// * `graph` - Weighted Graph represented by an adjacency list.
/// `graph[i]` contains the edges stretched from the `i`-th node.
/// Edges are represented by tuples of destination node and cost.
/// * `start_node` - Node to start searching.
///
/// # Examples
///
/// ```
/// use atcoder8_library::dijkstra::min_costs_and_prev_nodes;
///
/// let graph = vec![
///     vec![(1, 7), (2, 1)], vec![(0, 7), (3, 3)], vec![(0, 1), (3, 2)],
///     vec![(1, 3), (2, 2)], vec![]
/// ];
///
/// let (min_costs, prev_nodes) = min_costs_and_prev_nodes(&graph, 0);
/// assert_eq!(min_costs, vec![Some(0), Some(6), Some(1), Some(3), None]);
/// assert_eq!(prev_nodes, vec![None, Some(3), Some(0), Some(2), None]);
/// ```
pub fn min_costs_and_prev_nodes(
    graph: &Vec<Vec<(usize, usize)>>,
    start_node: usize,
) -> (Vec<Option<usize>>, Vec<Option<usize>>) {
    let node_num = graph.len();

    let mut min_costs = vec![None; node_num];
    min_costs[start_node] = Some(0);
    let mut parents = vec![None; node_num];

    let mut heap = BinaryHeap::from(vec![Reverse((0, start_node))]);
    let mut decided = vec![false; node_num];

    while let Some(Reverse((cur_cost, cur_node))) = heap.pop() {
        if decided[cur_node] {
            continue;
        }

        decided[cur_node] = true;

        for &(next_node, edge_cost) in &graph[cur_node] {
            let next_cost = cur_cost + edge_cost;

            if min_costs[next_node].is_none() || next_cost < min_costs[next_node].unwrap() {
                min_costs[next_node] = Some(next_cost);
                parents[next_node] = Some(cur_node);
                heap.push(Reverse((next_cost, next_node)));
            }
        }
    }

    (min_costs, parents)
}

/// Returns the minimum cost and shortest path
/// when going from `start_node` to `end_node` through the edge.
///
/// If no path exists, `None` is returned.
///
/// # Arguments
///
/// * `graph` - Weighted Graph represented by an adjacency list.
/// `graph[i]` contains the edges stretched from the `i`-th node.
/// Edges are represented by tuples of destination node and cost.
/// * `start_node` - Node to start searching.
/// * `end_node` - The destination node of the search.
///
/// # Examples
///
/// ```
/// use atcoder8_library::dijkstra::cost_and_path;
///
/// let graph = vec![
///     vec![(1, 7), (2, 1)], vec![(0, 7), (3, 3)], vec![(0, 1), (3, 2)],
///     vec![(1, 3), (2, 2)], vec![]
/// ];
///
/// let (cost, path) = cost_and_path(&graph, 0, 1).unwrap();
/// assert_eq!(cost, 6);
/// assert_eq!(path, vec![0, 2, 3, 1]);
///
/// assert_eq!(cost_and_path(&graph, 0, 4), None);
/// ```
pub fn cost_and_path(
    graph: &Vec<Vec<(usize, usize)>>,
    start_node: usize,
    end_node: usize,
) -> Option<(usize, Vec<usize>)> {
    let (min_costs, prev_nodes) = min_costs_and_prev_nodes(graph, start_node);

    min_costs[end_node]?;

    let mut path = vec![end_node];
    let mut cur_node = end_node;

    while let Some(par_node) = prev_nodes[cur_node] {
        path.push(par_node);
        cur_node = par_node;
    }

    path.reverse();

    Some((min_costs[end_node].unwrap(), path))
}
