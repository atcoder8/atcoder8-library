use std::{
    collections::VecDeque,
    ops::{Add, Sub},
};

pub trait Zero {
    fn zero() -> Self;

    fn is_zero(&self) -> bool;
}

pub trait MaxValue {
    fn max_value() -> Self;
}

pub trait Capacity:
    std::fmt::Debug
    + Clone
    + Copy
    + Ord
    + Add<Self, Output = Self>
    + Sub<Self, Output = Self>
    + Zero
    + MaxValue
{
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FlowEdge<Cap> {
    /// Node that is the source of the flow.
    pub from: usize,

    /// Node that is the destination of the flow.
    pub to: usize,

    /// Upper limit of flow through this edge.
    pub capacity: Cap,

    /// Current flow for this edge.
    pub flow: Cap,
}

impl<Cap> FlowEdge<Cap>
where
    Cap: Capacity,
{
    /// Create an edge by specifying endpoints and flow limit.
    /// Initial flow amount is zero.
    pub fn new(from: usize, to: usize, capacity: Cap) -> Self {
        Self {
            from,
            to,
            capacity,
            flow: Cap::zero(),
        }
    }

    /// Returns the difference between the upper flow limit and the current flow amount.
    pub fn rem_capacity(&self) -> Cap {
        self.capacity - self.flow
    }
}

#[derive(Debug, Clone)]
pub struct MfGraph<Cap> {
    /// Number of graph edges.
    node_num: usize,

    /// List of edges of this graph.
    edges: Vec<FlowEdge<Cap>>,

    /// For each node, a list containing a list of indices of edges with that node as head.
    graph: Vec<Vec<usize>>,

    /// For each node, a list containing a list of indices of edges with that node as tail.
    inv_graph: Vec<Vec<usize>>,
}

impl<Cap> MfGraph<Cap>
where
    Cap: Capacity,
{
    /// Creates an empty directed graph with `node_num` nodes.
    pub fn new(node_num: usize) -> Self {
        Self {
            node_num,
            graph: vec![vec![]; node_num],
            inv_graph: vec![vec![]; node_num],
            edges: vec![],
        }
    }

    /// Add an edge from the node `from` to the node `to` with flow limit `capacity`.
    pub fn add_edge(&mut self, from: usize, to: usize, capacity: Cap) {
        assert!(
            from < self.node_num && to < self.node_num,
            "`from` and `to` must be smaller than number of nodes."
        );

        assert!(capacity >= Cap::zero(), "`capacity` must be non-negative.");

        let edge_idx = self.edges.len();

        self.edges.push(FlowEdge::new(from, to, capacity));

        self.graph[from].push(edge_idx);
        self.inv_graph[to].push(edge_idx);
    }

    /// Returns the state of the `edge_idx` th (0-based) edge.
    pub fn get_edge(&self, edge_idx: usize) -> FlowEdge<Cap> {
        assert!(
            edge_idx < self.edges.len(),
            "The number of edges is {}, but {} was specified.",
            self.edges.len(),
            edge_idx
        );

        self.edges[edge_idx]
    }

    /// Returns a list of edge states added to the graph.
    pub fn get_edges(&self) -> &Vec<FlowEdge<Cap>> {
        &self.edges
    }

    /// Creates a list of `level` for each node,
    /// where `level` is the distance from `source` using edges or inverses
    /// where the current flow amount is less than the upper limit.
    fn find_levels(&self, source: usize, sink: usize) -> Vec<Option<usize>> {
        let mut levels: Vec<Option<usize>> = vec![None; self.node_num];
        let mut queue = VecDeque::from([(source, 0)]);
        while let Some((cur_node, cand_level)) = queue.pop_front() {
            if levels[cur_node].is_some() {
                continue;
            }

            levels[cur_node] = Some(cand_level);

            if cur_node == sink {
                continue;
            }

            let next_bfs_nodes = self.graph[cur_node]
                .iter()
                .filter(|&&edge_idx| !self.edges[edge_idx].rem_capacity().is_zero())
                .map(|&edge_idx| self.edges[edge_idx].to)
                .chain(
                    self.inv_graph[cur_node]
                        .iter()
                        .filter(|&&edge_idx| !self.edges[edge_idx].flow.is_zero())
                        .map(|&edge_idx| self.edges[edge_idx].from),
                )
                .map(|next_node| (next_node, cand_level + 1));
            queue.extend(next_bfs_nodes);
        }

        levels
    }

    /// Increases the flow from `source` to `sink` by the maximum flow `flow_limit` and returns the increased flow amount.
    fn increase_flow(&mut self, source: usize, sink: usize, flow_limit: Cap) -> Cap {
        let levels = self.find_levels(source, sink);

        let Some(sink_level) = levels[sink] else {
            return Cap::zero();
        };

        let select_edge = |cur_node: usize,
                           edge_progresses: &mut [usize],
                           inv_edge_progresses: &mut [usize],
                           edges: &[FlowEdge<Cap>]| {
            let cur_level = levels[cur_node].unwrap();

            let edge_progress = &mut edge_progresses[cur_node];
            while *edge_progress < self.graph[cur_node].len() {
                let edge_idx = self.graph[cur_node][*edge_progress];
                let edge = edges[edge_idx];

                if levels[edge.to] == Some(cur_level + 1) && !edge.rem_capacity().is_zero() {
                    return Some((edge_idx, false));
                }

                *edge_progress += 1;
            }

            let inv_edge_progress = &mut inv_edge_progresses[cur_node];
            while *inv_edge_progress < self.inv_graph[cur_node].len() {
                let edge_idx = self.inv_graph[cur_node][*inv_edge_progress];
                let edge = edges[edge_idx];

                if levels[edge.from] == Some(cur_level + 1) && !edge.flow.is_zero() {
                    return Some((edge_idx, true));
                }

                *inv_edge_progress += 1;
            }

            None
        };

        let push_dfs_node = |cur_node: usize,
                             edge_progresses: &mut [usize],
                             inv_edge_progresses: &mut [usize],
                             edges: &[FlowEdge<Cap>],
                             stack: &mut Vec<DFSNode<Cap>>| {
            let Some((edge_idx, inverse)) =
                select_edge(cur_node, edge_progresses, inv_edge_progresses, edges)
            else {
                return;
            };
            let edge = edges[edge_idx];
            let (next_node, next_rem_capacity) = if inverse {
                (edge.from, edge.flow)
            } else {
                (edge.to, edge.rem_capacity())
            };

            stack.push(DFSNode::Backward {
                cur_node,
                edge_idx,
                inverse,
            });

            stack.push(DFSNode::Forward {
                cur_node: next_node,
                rem_capacity: next_rem_capacity,
            });
        };

        enum DFSNode<Cap> {
            Forward {
                cur_node: usize,
                rem_capacity: Cap,
            },

            Backward {
                cur_node: usize,
                edge_idx: usize,
                inverse: bool,
            },
        }

        struct FlowState<Cap> {
            flow_limit: Cap,
            flow: Cap,
        }

        let mut edge_progresses = vec![0; self.node_num];
        let mut inv_edge_progresses = vec![0; self.node_num];

        let mut stack = vec![DFSNode::Forward {
            cur_node: source,
            rem_capacity: flow_limit,
        }];
        stack.reserve(2 * sink_level);

        let mut flow_state_stack: Vec<FlowState<Cap>> = Vec::with_capacity(sink_level);

        while let Some(dfs_node) = stack.pop() {
            match dfs_node {
                DFSNode::Forward {
                    cur_node,
                    rem_capacity,
                } => {
                    let cur_flow_limit = match flow_state_stack.last() {
                        Some(prev_flow_state) => prev_flow_state.flow_limit - prev_flow_state.flow,
                        None => flow_limit,
                    }
                    .min(rem_capacity);

                    if cur_node == sink {
                        flow_state_stack.push(FlowState {
                            flow_limit: cur_flow_limit,
                            flow: cur_flow_limit,
                        });

                        continue;
                    }

                    flow_state_stack.push(FlowState {
                        flow_limit: cur_flow_limit,
                        flow: Cap::zero(),
                    });

                    push_dfs_node(
                        cur_node,
                        &mut edge_progresses,
                        &mut inv_edge_progresses,
                        &self.edges,
                        &mut stack,
                    );
                }

                DFSNode::Backward {
                    cur_node,
                    edge_idx,
                    inverse,
                } => {
                    let next_flow_state = flow_state_stack.pop().unwrap();

                    let edge = &mut self.edges[edge_idx];
                    if inverse {
                        edge.flow = edge.flow - next_flow_state.flow;
                    } else {
                        edge.flow = edge.flow + next_flow_state.flow;
                    }

                    let cur_flow_state = flow_state_stack.last_mut().unwrap();
                    cur_flow_state.flow = cur_flow_state.flow + next_flow_state.flow;

                    if cur_flow_state.flow_limit == cur_flow_state.flow {
                        continue;
                    }

                    if inverse {
                        inv_edge_progresses[cur_node] += 1;
                    } else {
                        edge_progresses[cur_node] += 1;
                    }

                    push_dfs_node(
                        cur_node,
                        &mut edge_progresses,
                        &mut inv_edge_progresses,
                        &self.edges,
                        &mut stack,
                    );
                }
            }
        }

        flow_state_stack[0].flow
    }

    /// Flows from `source` to `sink` as much as possible, and returns the flowed amount.
    /// The flow limit is set to the maximum of the values represented by the capacity type.
    pub fn flow(&mut self, source: usize, sink: usize) -> Cap {
        self.flow_with_capacity(source, sink, Cap::max_value())
    }

    /// Flows from `source` to `sink` as much as possible, with the maximum flow as `flow_limit`, and returns the flowed amount.
    pub fn flow_with_capacity(&mut self, source: usize, sink: usize, flow_limit: Cap) -> Cap {
        assert!(
            source < self.node_num && sink < self.node_num,
            "`source` and `sink` must be smaller than number of nodes."
        );

        let mut max_flow = Cap::zero();

        while max_flow < flow_limit {
            let flow = self.increase_flow(source, sink, flow_limit - max_flow);

            if flow.is_zero() {
                break;
            }

            max_flow = max_flow + flow;
        }

        max_flow
    }

    /// Finds for each node if it is reachable from `source` using the edge or the inverse edge,
    /// where the current flow is less than the upper limit.
    pub fn min_cut(&self, source: usize) -> Vec<bool> {
        let mut visited = vec![false; self.node_num];

        let mut queue = VecDeque::from([source]);
        while let Some(cur_node) = queue.pop_front() {
            if visited[cur_node] {
                continue;
            }

            visited[cur_node] = true;

            for &edge_idx in &self.graph[cur_node] {
                let edge = self.edges[edge_idx];
                if !edge.rem_capacity().is_zero() {
                    queue.push_back(edge.to);
                }
            }

            for &edge_idx in &self.inv_graph[cur_node] {
                let edge = self.edges[edge_idx];
                if !edge.flow.is_zero() {
                    queue.push_back(edge.from);
                }
            }
        }

        visited
    }
}

macro_rules! impl_capacity_for_unsigned_integer {
    ( $( $uint: ty ), * ) => {
        $(
            impl Zero for $uint {
                fn zero() -> Self {
                    0
                }

                fn is_zero(&self) -> bool {
                    self == &0
                }
            }

            impl MaxValue for $uint {
                fn max_value() -> Self {
                    <$uint>::MAX
                }
            }

            impl Capacity for $uint {}
        )*
    };
}

impl_capacity_for_unsigned_integer!(u8, u16, u32, u64, u128, usize);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_max_flow_wikipedia() {
        // From https://commons.wikimedia.org/wiki/File:Min_cut.png
        // Under CC BY-SA 3.0 https://creativecommons.org/licenses/by-sa/3.0/deed.en
        let mut graph = MfGraph::<usize>::new(6);
        graph.add_edge(0, 1, 3);
        graph.add_edge(0, 2, 3);
        graph.add_edge(1, 2, 2);
        graph.add_edge(1, 3, 3);
        graph.add_edge(2, 4, 2);
        graph.add_edge(3, 4, 4);
        graph.add_edge(3, 5, 2);
        graph.add_edge(4, 5, 3);

        assert_eq!(graph.flow(0, 5), 5);

        let edges = graph.get_edges();
        {
            #[rustfmt::skip]
            assert_eq!(
                edges,
                &vec![
                    FlowEdge { from: 0, to: 1, capacity: 3, flow: 3 },
                    FlowEdge { from: 0, to: 2, capacity: 3, flow: 2 },
                    FlowEdge { from: 1, to: 2, capacity: 2, flow: 0 },
                    FlowEdge { from: 1, to: 3, capacity: 3, flow: 3 },
                    FlowEdge { from: 2, to: 4, capacity: 2, flow: 2 },
                    FlowEdge { from: 3, to: 4, capacity: 4, flow: 1 },
                    FlowEdge { from: 3, to: 5, capacity: 2, flow: 2 },
                    FlowEdge { from: 4, to: 5, capacity: 3, flow: 3 },
                ]
            );
        }
        // assert_eq!(
        //     graph.min_cut(0),
        //     vec![true, false, true, false, false, false]
        // );
    }

    #[test]
    fn test_max_flow_wikipedia_multiple_edges() {
        // From https://commons.wikimedia.org/wiki/File:Min_cut.png
        // Under CC BY-SA 3.0 https://creativecommons.org/licenses/by-sa/3.0/deed.en
        let mut graph = MfGraph::<usize>::new(6);
        for &(u, v, c) in &[
            (0, 1, 3),
            (0, 2, 3),
            (1, 2, 2),
            (1, 3, 3),
            (2, 4, 2),
            (3, 4, 4),
            (3, 5, 2),
            (4, 5, 3),
        ] {
            for _ in 0..c {
                graph.add_edge(u, v, 1);
            }
        }

        assert_eq!(graph.flow(0, 5), 5);
        // assert_eq!(
        //     graph.min_cut(0),
        //     vec![true, false, true, false, false, false]
        // );
    }

    #[test]
    #[allow(clippy::many_single_char_names)]
    fn test_max_flow_misawa() {
        // Originally by @MiSawa
        // From https://gist.github.com/MiSawa/47b1d99c372daffb6891662db1a2b686
        let n = 100;

        let mut graph = MfGraph::<usize>::new((n + 1) * 2 + 5);
        let (s, a, b, c, t) = (0, 1, 2, 3, 4);
        graph.add_edge(s, a, 1);
        graph.add_edge(s, b, 2);
        graph.add_edge(b, a, 2);
        graph.add_edge(c, t, 2);
        for i in 0..n {
            let i = 2 * i + 5;
            for j in 0..2 {
                for k in 2..4 {
                    graph.add_edge(i + j, i + k, 3);
                }
            }
        }
        for j in 0..2 {
            graph.add_edge(a, 5 + j, 3);
            graph.add_edge(2 * n + 5 + j, c, 3);
        }

        assert_eq!(graph.flow(s, t), 2);
    }

    #[test]
    fn test_dont_repeat_same_phase() {
        let n = 100_000;
        let mut graph = MfGraph::<usize>::new(3);
        graph.add_edge(0, 1, n);
        for _ in 0..n {
            graph.add_edge(1, 2, 1);
        }
        assert_eq!(graph.flow(0, 2), n);
    }
}
