//! Library for searching shortest paths by Dijkstra's algorithm.

use std::{cmp::Reverse, collections::BinaryHeap};

#[derive(Debug, Clone)]
pub struct Dijkstra {
    source: usize,
    min_costs: Vec<Option<usize>>,
    prev_nodes: Vec<Option<usize>>,
}

impl Dijkstra {
    /// Searches the shortest cost and the node immediately before the destination from `source` to each node.
    ///
    /// # Arguments
    ///
    /// * `graph` - Weighted Graph represented by an adjacency list.
    /// `graph[i]` contains the edges stretched from the `i`-th node.
    /// Edges are represented by tuples of destination node and edge cost.
    /// * `source` - Node to start searching.
    ///
    /// # Examples
    ///
    /// ```
    /// use atcoder8_library::dijkstra::Dijkstra;
    ///
    /// let graph = vec![
    ///     vec![(1, 7), (2, 1)], vec![(0, 7), (3, 3)], vec![(0, 1), (3, 2)],
    ///     vec![(1, 3), (2, 2)], vec![]
    /// ];
    ///
    /// let dijkstra = Dijkstra::new(&graph, 0);
    ///
    /// assert_eq!(dijkstra.min_cost(0), Some(0));
    /// assert_eq!(dijkstra.min_cost(1), Some(6));
    /// assert_eq!(dijkstra.min_cost(2), Some(1));
    /// assert_eq!(dijkstra.min_cost(3), Some(3));
    /// assert_eq!(dijkstra.min_cost(4), None);
    /// ```
    pub fn new(graph: &Vec<Vec<(usize, usize)>>, source: usize) -> Self {
        let node_num = graph.len();

        let mut min_costs = vec![None; node_num];
        min_costs[source] = Some(0);
        let mut prev_nodes = vec![None; node_num];

        let mut heap = BinaryHeap::from(vec![Reverse((0, source))]);
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
                    prev_nodes[next_node] = Some(cur_node);
                    heap.push(Reverse((next_cost, next_node)));
                }
            }
        }

        Self {
            source,
            min_costs,
            prev_nodes,
        }
    }

    /// Returns the node where the search started.
    ///
    /// # Examples
    ///
    /// ```
    /// use atcoder8_library::dijkstra::Dijkstra;
    ///
    /// let graph = vec![
    ///     vec![(1, 7), (2, 1)], vec![(0, 7), (3, 3)], vec![(0, 1), (3, 2)],
    ///     vec![(1, 3), (2, 2)], vec![]
    /// ];
    ///
    /// let dijkstra = Dijkstra::new(&graph, 3);
    ///
    /// assert_eq!(dijkstra.source(), 3);
    /// ```
    pub fn source(&self) -> usize {
        self.source
    }

    /// Returns the minimum cost of the path from `source` to `dest`.
    /// If there is no path from `source` to `dest`, returns `None`.
    ///
    /// # Arguments
    ///
    /// * dest - Node of destination.
    ///
    /// # Examples
    ///
    /// ```
    /// use atcoder8_library::dijkstra::Dijkstra;
    ///
    /// let graph = vec![
    ///     vec![(1, 7), (2, 1)], vec![(0, 7), (3, 3)], vec![(0, 1), (3, 2)],
    ///     vec![(1, 3), (2, 2)], vec![]
    /// ];
    ///
    /// let dijkstra = Dijkstra::new(&graph, 0);
    ///
    /// assert_eq!(dijkstra.min_cost(0), Some(0));
    /// assert_eq!(dijkstra.min_cost(1), Some(6));
    /// assert_eq!(dijkstra.min_cost(2), Some(1));
    /// assert_eq!(dijkstra.min_cost(3), Some(3));
    /// assert_eq!(dijkstra.min_cost(4), None);
    /// ```
    pub fn min_cost(&self, dest: usize) -> Option<usize> {
        self.min_costs[dest]
    }

    /// Returns the node immediately before `dest` for the shortest path from `source` to `dest`.
    /// Returns `None` if `dest` == `source` or there is no path from `source` to `dest`.
    ///
    /// # Arguments
    ///
    /// * dest - Node of destination.
    ///
    /// # Examples
    ///
    /// ```
    /// use atcoder8_library::dijkstra::Dijkstra;
    ///
    /// let graph = vec![
    ///     vec![(1, 7), (2, 1)], vec![(0, 7), (3, 3)], vec![(0, 1), (3, 2)],
    ///     vec![(1, 3), (2, 2)], vec![]
    /// ];
    ///
    /// let dijkstra = Dijkstra::new(&graph, 0);
    ///
    /// assert_eq!(dijkstra.prev_node(0), None);
    /// assert_eq!(dijkstra.prev_node(1), Some(3));
    /// assert_eq!(dijkstra.prev_node(2), Some(0));
    /// assert_eq!(dijkstra.prev_node(3), Some(2));
    /// assert_eq!(dijkstra.prev_node(4), None);
    /// ```
    pub fn prev_node(&self, dest: usize) -> Option<usize> {
        self.prev_nodes[dest]
    }

    /// Returns the shortest path from `source` to `dest`.
    /// If there is no path from `source` to `dest`, returns `None`.
    ///
    /// # Examples
    ///
    /// ```
    /// use atcoder8_library::dijkstra::Dijkstra;
    ///
    /// let graph = vec![
    ///     vec![(1, 7), (2, 1)], vec![(0, 7), (3, 3)], vec![(0, 1), (3, 2)],
    ///     vec![(1, 3), (2, 2)], vec![]
    /// ];
    ///
    /// let dijkstra = Dijkstra::new(&graph, 0);
    ///
    /// assert_eq!(dijkstra.shortest_path(0), Some(vec![0]));
    /// assert_eq!(dijkstra.shortest_path(1), Some(vec![0, 2, 3, 1]));
    /// assert_eq!(dijkstra.shortest_path(2), Some(vec![0, 2]));
    /// assert_eq!(dijkstra.shortest_path(3), Some(vec![0, 2, 3]));
    /// assert_eq!(dijkstra.shortest_path(4), None);
    /// ```
    pub fn shortest_path(&self, dest: usize) -> Option<Vec<usize>> {
        self.min_cost(dest)?;

        let mut path = vec![dest];
        let mut cur_node = dest;

        while let Some(par_node) = self.prev_node(cur_node) {
            path.push(par_node);
            cur_node = par_node;
        }

        path.reverse();

        Some(path)
    }
}
