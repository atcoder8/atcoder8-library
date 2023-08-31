/// Returns the indices of the `edges` that are bridges, in ascending order.
///
/// # Arguments
///
/// * `node_num` - Number of nodes.
/// * `edges` - Undirected edges connecting two nodes.
///
/// # Examples
///
/// ```
/// # use atcoder8_library::low_link::low_link;
/// let node_num = 5;
/// let edges = vec![(0, 1), (0, 2), (2, 3), (2, 4), (3, 4)];
///
/// assert_eq!(low_link(node_num, &edges), vec![0, 1]);
/// ```
pub fn low_link(node_num: usize, edges: &[(usize, usize)]) -> Vec<usize> {
    let mut graph = vec![vec![]; node_num];
    for (edge_idx, &(u, v)) in edges.iter().enumerate() {
        graph[u].push((v, edge_idx));
        graph[v].push((u, edge_idx));
    }

    let mut bridges = vec![];
    let mut ord: Vec<Option<usize>> = vec![None; node_num];
    let mut low: Vec<Option<usize>> = vec![None; node_num];

    for start in 0..node_num {
        if ord[start].is_some() {
            continue;
        }

        let mut stack: Vec<(Option<usize>, usize, Option<usize>, bool)> =
            vec![(None, start, None, false)];
        while let Some((par, cur, prev_edge_idx, back)) = stack.pop() {
            if back {
                if let Some(par) = par {
                    low[par] = low[par].min(low[cur]);

                    if ord[par] < low[cur] {
                        bridges.push(prev_edge_idx.unwrap());
                    }
                }

                continue;
            }

            if ord[cur].is_some() {
                if let Some(par) = par {
                    low[par] = low[par].min(ord[cur]);
                }

                continue;
            }

            let t = match par {
                Some(par) => ord[par].unwrap() + 1,
                None => 0,
            };

            ord[cur] = Some(t);
            low[cur] = Some(t);

            stack.push((par, cur, prev_edge_idx, true));

            for &(next, edge_idx) in &graph[cur] {
                if Some(edge_idx) != prev_edge_idx {
                    stack.push((Some(cur), next, Some(edge_idx), false));
                }
            }
        }
    }

    bridges.sort_unstable();

    bridges
}

#[cfg(test)]
mod tests {
    use rand::{rngs::StdRng, seq::SliceRandom, Rng};
    use rand_core::SeedableRng;

    use super::*;

    fn count_connected_component(
        graph: &[Vec<(usize, usize)>],
        ignore_edge_idx: Option<usize>,
    ) -> usize {
        let node_num = graph.len();

        let mut comp_cnt = 0;
        let mut visited = vec![false; node_num];
        for start in 0..node_num {
            if visited[start] {
                continue;
            }

            comp_cnt += 1;

            let mut stack: Vec<(usize, Option<usize>)> = vec![(start, None)];
            while let Some((cur, prev_edge_idx)) = stack.pop() {
                if visited[cur] {
                    continue;
                }

                visited[cur] = true;

                for &(next, edge_idx) in &graph[cur] {
                    if Some(edge_idx) != ignore_edge_idx && Some(edge_idx) != prev_edge_idx {
                        stack.push((next, Some(edge_idx)));
                    }
                }
            }
        }

        comp_cnt
    }

    fn naive_find_bridges(node_num: usize, edges: &[(usize, usize)]) -> Vec<usize> {
        let mut graph = vec![vec![]; node_num];
        for (i, &(u, v)) in edges.iter().enumerate() {
            graph[u].push((v, i));
            graph[v].push((u, i));
        }

        let comp_num = count_connected_component(&graph, None);

        (0..edges.len())
            .filter(|&edge_idx| count_connected_component(&graph, Some(edge_idx)) != comp_num)
            .collect()
    }

    #[test]
    fn test_order_zero_graph() {
        assert_eq!(low_link(0, &[]), vec![]);
    }

    #[test]
    fn test_edgeless_graph() {
        assert_eq!(low_link(5, &[]), vec![]);
    }

    #[test]
    fn test_handmade_1() {
        const NODE_NUM: usize = 11;
        const EDGES: [(usize, usize); 13] = [
            (0, 1),
            (0, 2),
            (2, 3),
            (2, 4),
            (3, 4),
            (3, 5),
            (5, 6),
            (5, 7),
            (6, 8),
            (7, 8),
            (7, 10),
            (8, 9),
            (9, 10),
        ];

        assert_eq!(low_link(NODE_NUM, &EDGES), vec![0, 1, 5]);
    }

    #[test]
    fn test_handmade_2() {
        const NODE_NUM: usize = 12;
        const EDGES: [(usize, usize); 11] = [
            (5, 0),
            (2, 11),
            (11, 3),
            (8, 1),
            (8, 4),
            (4, 1),
            (5, 7),
            (9, 3),
            (2, 9),
            (0, 7),
            (6, 8),
        ];

        assert_eq!(low_link(NODE_NUM, &EDGES), vec![10]);
    }

    fn generate_test_case<R>(rng: &mut R) -> (usize, Vec<(usize, usize)>)
    where
        R: Rng,
    {
        let node_num: usize = rng.gen_range(1..20);
        let edge_num: usize = rng.gen_range(1..100);
        let edges = (0..edge_num)
            .map(|_| (rng.gen_range(0..node_num), rng.gen_range(0..node_num)))
            .collect();

        (node_num, edges)
    }

    #[test]
    fn test_random_graph() {
        const TEST_NUM: usize = 1000;

        let mut rng: StdRng = SeedableRng::seed_from_u64(0);

        for _ in 0..TEST_NUM {
            let (node_num, edges) = generate_test_case(&mut rng);

            assert_eq!(
                low_link(node_num, &edges),
                naive_find_bridges(node_num, &edges)
            );
        }
    }

    fn generate_simple_test_case<R>(rng: &mut R) -> (usize, Vec<(usize, usize)>)
    where
        R: Rng,
    {
        let node_num: usize = rng.gen_range(2..20);
        let mut edges = vec![];
        for u in 0..node_num {
            for v in (u + 1)..node_num {
                if rng.gen_bool(0.5) {
                    edges.push((u, v));
                } else {
                    edges.push((v, u));
                }
            }
        }
        let edge_num = rng.gen_range(0..=edges.len());
        let edges = edges.choose_multiple(rng, edge_num).cloned().collect();

        (node_num, edges)
    }

    #[test]
    fn test_random_simple_graph() {
        const TEST_NUM: usize = 1000;

        let mut rng: StdRng = SeedableRng::seed_from_u64(0);

        for _ in 0..TEST_NUM {
            let (node_num, edges) = generate_simple_test_case(&mut rng);

            assert_eq!(
                low_link(node_num, &edges),
                naive_find_bridges(node_num, &edges)
            );
        }
    }
}
