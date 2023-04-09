#[derive(Debug, Clone)]
pub struct SCC {
    graph: Vec<Vec<usize>>,
    inv_graph: Vec<Vec<usize>>,
}

impl SCC {
    pub fn new(n: usize) -> Self {
        Self {
            graph: vec![vec![]; n],
            inv_graph: vec![vec![]; n],
        }
    }

    pub fn add_edge(&mut self, from: usize, to: usize) {
        self.graph[from].push(to);
        self.inv_graph[to].push(from);
    }

    pub fn scc(&self) -> Vec<Vec<usize>> {
        let n = self.graph.len();

        let mut order = vec![];
        let mut visited = vec![false; n];
        for start_node in 0..n {
            if !visited[start_node] {
                order.append(&mut post_order_traversal(
                    &self.graph,
                    &mut visited,
                    start_node,
                ));
            }
        }

        let mut scc = vec![];
        let mut visited = vec![false; n];
        for &start_node in order.iter().rev() {
            if !visited[start_node] {
                scc.push(post_order_traversal(
                    &self.inv_graph,
                    &mut visited,
                    start_node,
                ));
            }
        }

        scc
    }
}

fn post_order_traversal(
    graph: &Vec<Vec<usize>>,
    visited: &mut Vec<bool>,
    start_node: usize,
) -> Vec<usize> {
    let mut post_order = vec![];

    let mut stack = vec![(start_node, false)];

    while let Some((node, back)) = stack.pop() {
        if back {
            post_order.push(node);
        }

        if visited[node] {
            continue;
        }

        visited[node] = true;

        stack.push((node, true));

        stack.extend(graph[node].iter().map(|&next_node| (next_node, false)));
    }

    post_order
}
