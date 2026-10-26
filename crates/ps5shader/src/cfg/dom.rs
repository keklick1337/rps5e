// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Generic graph analyses over adjacency lists: reverse post-order,
//! dominators (Cooper, Harvey and Kennedy), post-dominators, dominance
//! frontiers, back edges and reducibility.

/// Sentinel for "no node".
pub const NONE: usize = usize::MAX;

/// Reverse post-order of the nodes reachable from `entry` (iterative DFS,
/// successors visited in list order).
pub fn reverse_post_order(succs: &[Vec<usize>], entry: usize) -> Vec<usize> {
    let n = succs.len();
    let mut visited = vec![false; n];
    let mut post = Vec::with_capacity(n);
    let mut stack: Vec<(usize, usize)> = vec![(entry, 0)];
    visited[entry] = true;
    while let Some(&mut (node, ref mut next)) = stack.last_mut() {
        if *next < succs[node].len() {
            let s = succs[node][*next];
            *next += 1;
            if !visited[s] {
                visited[s] = true;
                stack.push((s, 0));
            }
        } else {
            post.push(node);
            stack.pop();
        }
    }
    post.reverse();
    post
}

/// Predecessor lists from successor lists.
pub fn predecessors(succs: &[Vec<usize>]) -> Vec<Vec<usize>> {
    let mut preds = vec![Vec::new(); succs.len()];
    for (b, ss) in succs.iter().enumerate() {
        for &s in ss {
            if !preds[s].contains(&b) {
                preds[s].push(b);
            }
        }
    }
    preds
}

/// Immediate dominators; `idom[entry] == entry`, unreachable nodes `NONE`.
pub fn dominators(succs: &[Vec<usize>], entry: usize) -> Vec<usize> {
    let rpo = reverse_post_order(succs, entry);
    let preds = predecessors(succs);
    idoms_from(&rpo, &preds, succs.len())
}

fn idoms_from(rpo: &[usize], preds: &[Vec<usize>], n: usize) -> Vec<usize> {
    let mut order = vec![NONE; n];
    for (i, &b) in rpo.iter().enumerate() {
        order[b] = i;
    }
    let mut idom = vec![NONE; n];
    let Some(&entry) = rpo.first() else { return idom };
    idom[entry] = entry;
    let intersect = |idom: &[usize], mut a: usize, mut b: usize| {
        while a != b {
            while order[a] > order[b] {
                a = idom[a];
            }
            while order[b] > order[a] {
                b = idom[b];
            }
        }
        a
    };
    let mut changed = true;
    while changed {
        changed = false;
        for &b in &rpo[1..] {
            let mut new = NONE;
            for &p in &preds[b] {
                if order[p] == NONE || idom[p] == NONE {
                    continue;
                }
                new = if new == NONE { p } else { intersect(&idom, p, new) };
            }
            if new != NONE && idom[b] != new {
                idom[b] = new;
                changed = true;
            }
        }
    }
    idom
}

/// Immediate post-dominators. Every node without successors (and, to keep
/// infinite loops well defined, every node that cannot reach one) is joined
/// to a virtual exit, returned as index `succs.len()`; `ipdom[node]` is that
/// index when the node is only post-dominated by the exit.
pub fn post_dominators(succs: &[Vec<usize>], entry: usize) -> Vec<usize> {
    let n = succs.len();
    let exit = n;
    // Reverse graph with the virtual exit as entry.
    let mut rsuccs: Vec<Vec<usize>> = vec![Vec::new(); n + 1];
    for (b, ss) in succs.iter().enumerate() {
        if ss.is_empty() {
            rsuccs[exit].push(b);
        }
        for &s in ss {
            rsuccs[s].push(b);
        }
    }
    // Nodes that cannot reach an exit (infinite loops): connect the loop
    // nodes reachable from entry to the exit so every node has a post
    // dominator.
    let reach = reverse_post_order(&rsuccs, exit);
    let mut reaches = vec![false; n + 1];
    for &b in &reach {
        reaches[b] = true;
    }
    let forward = reverse_post_order(succs, entry);
    for &b in forward.iter().rev() {
        if !reaches[b] {
            rsuccs[exit].push(b);
            for &r in &reverse_post_order(&rsuccs, b) {
                reaches[r] = true;
            }
        }
    }
    let rpo = reverse_post_order(&rsuccs, exit);
    let rpreds = predecessors(&rsuccs);
    let mut ipdom = idoms_from(&rpo, &rpreds, n + 1);
    ipdom.truncate(n);
    ipdom
}

/// Does `a` dominate `b` (walking `b`'s idom chain)?
pub fn dominates(idom: &[usize], a: usize, mut b: usize) -> bool {
    if idom[b] == NONE {
        return false;
    }
    loop {
        if a == b {
            return true;
        }
        let up = idom[b];
        if up == b || up == NONE || up >= idom.len() {
            return false;
        }
        b = up;
    }
}

/// Children lists of the dominator tree.
pub fn dominator_children(idom: &[usize]) -> Vec<Vec<usize>> {
    let mut children = vec![Vec::new(); idom.len()];
    for (b, &d) in idom.iter().enumerate() {
        if d != NONE && d != b && d < idom.len() {
            children[d].push(b);
        }
    }
    children
}

/// Dominance frontiers.
pub fn dominance_frontiers(succs: &[Vec<usize>], idom: &[usize]) -> Vec<Vec<usize>> {
    let preds = predecessors(succs);
    let mut df = vec![Vec::new(); succs.len()];
    for b in 0..succs.len() {
        if idom[b] == NONE {
            continue;
        }
        let reachable_preds: Vec<usize> = preds[b].iter().copied().filter(|&p| idom[p] != NONE).collect();
        if reachable_preds.len() < 2 {
            continue;
        }
        for p in reachable_preds {
            let mut runner = p;
            while runner != idom[b] {
                if !df[runner].contains(&b) {
                    df[runner].push(b);
                }
                if idom[runner] == runner {
                    break;
                }
                runner = idom[runner];
            }
        }
    }
    df
}

/// Edges `(from, to)` that are retreating in the DFS order of
/// [`reverse_post_order`] (target RPO index <= source RPO index).
pub fn retreating_edges(succs: &[Vec<usize>], entry: usize) -> Vec<(usize, usize)> {
    let rpo = reverse_post_order(succs, entry);
    let mut order = vec![NONE; succs.len()];
    for (i, &b) in rpo.iter().enumerate() {
        order[b] = i;
    }
    let mut edges = Vec::new();
    for &b in &rpo {
        for &s in &succs[b] {
            if order[s] <= order[b] {
                edges.push((b, s));
            }
        }
    }
    edges
}

/// The first retreating edge whose target does not dominate its source (the
/// graph is irreducible), if any.
pub fn irreducible_edge(succs: &[Vec<usize>], entry: usize) -> Option<(usize, usize)> {
    let idom = dominators(succs, entry);
    retreating_edges(succs, entry).into_iter().find(|&(from, to)| !dominates(&idom, to, from))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn diamond() {
        // 0 -> 1, 2; 1 -> 3; 2 -> 3
        let g = vec![vec![1, 2], vec![3], vec![3], vec![]];
        let idom = dominators(&g, 0);
        assert_eq!(idom, vec![0, 0, 0, 0]);
        let ipdom = post_dominators(&g, 0);
        assert_eq!(ipdom, vec![3, 3, 3, 4]);
        let df = dominance_frontiers(&g, &idom);
        assert_eq!(df[1], vec![3]);
        assert_eq!(df[2], vec![3]);
        assert!(irreducible_edge(&g, 0).is_none());
    }

    #[test]
    fn irreducible() {
        // 0 -> 1, 2; 1 <-> 2
        let g = vec![vec![1, 2], vec![2], vec![1]];
        assert!(irreducible_edge(&g, 0).is_some());
    }

    /// Immediate dominators from the set-based definition: `Dom(entry) =
    /// {entry}`, `Dom(b) = {b} ∪ ⋂ Dom(p)` over the predecessors reached
    /// so far, iterated to the fixed point; the immediate dominator is the
    /// strict dominator with the largest set.
    fn set_idoms(succs: &[Vec<usize>], entry: usize) -> Vec<usize> {
        let n = succs.len();
        let preds = predecessors(succs);
        let reachable: Vec<bool> = {
            let mut r = vec![false; n];
            for b in reverse_post_order(succs, entry) {
                r[b] = true;
            }
            r
        };
        let all: Vec<usize> = (0..n).collect();
        let mut dom: Vec<Vec<usize>> = (0..n).map(|b| if b == entry { vec![entry] } else { all.clone() }).collect();
        let mut changed = true;
        while changed {
            changed = false;
            for b in 0..n {
                if b == entry || !reachable[b] {
                    continue;
                }
                let mut next: Option<Vec<usize>> = None;
                for &p in preds[b].iter().filter(|&&p| reachable[p]) {
                    next = Some(match next {
                        None => dom[p].clone(),
                        Some(s) => s.into_iter().filter(|x| dom[p].contains(x)).collect(),
                    });
                }
                let mut next = next.unwrap_or_default();
                if !next.contains(&b) {
                    next.push(b);
                    next.sort_unstable();
                }
                if next != dom[b] {
                    dom[b] = next;
                    changed = true;
                }
            }
        }
        (0..n)
            .map(|b| {
                if !reachable[b] {
                    NONE
                } else if b == entry {
                    entry
                } else {
                    *dom[b].iter().filter(|&&d| d != b).max_by_key(|&&d| dom[d].len()).unwrap()
                }
            })
            .collect()
    }

    /// The reversed graph with a virtual exit (index `n`) joined from every
    /// node without successors, as [`post_dominators`] sees it when every
    /// node reaches an exit.
    fn reversed_with_exit(succs: &[Vec<usize>]) -> Vec<Vec<usize>> {
        let n = succs.len();
        let mut r = vec![Vec::new(); n + 1];
        for (b, ss) in succs.iter().enumerate() {
            if ss.is_empty() {
                r[n].push(b);
            }
            for &s in ss {
                r[s].push(b);
            }
        }
        r
    }

    fn require_set_dominance(succs: &[Vec<usize>], entry: usize, what: &str) {
        assert_eq!(dominators(succs, entry), set_idoms(succs, entry), "{what}: dominators");
        let n = succs.len();
        let mut ipdom = set_idoms(&reversed_with_exit(succs), n);
        ipdom.truncate(n);
        assert_eq!(post_dominators(succs, entry), ipdom, "{what}: post-dominators");
    }

    /// The dominator and post-dominator trees agree with the set-based
    /// definition on the shapes where a set intersection comes out empty
    /// and on random DAGs with permuted block ids and a nonzero entry.
    #[test]
    fn dominance_matches_the_set_definition() {
        require_set_dominance(&[vec![1, 2], vec![], vec![]], 0, "disjoint exits with an empty post-dominator intersection");
        require_set_dominance(&[vec![2], vec![2], vec![]], 0, "separate roots with an empty dominator intersection");
        require_set_dominance(&[vec![1], vec![1, 2], vec![]], 0, "a self-loop with an exit");
        require_set_dominance(&[vec![1], vec![], vec![2], vec![0, 4], vec![1]], 3, "a nonzero entry with permuted block ids");
        let mut state = 0x7b19_a5c3u32;
        let mut word = || {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            state
        };
        for trial in 0..24u32 {
            let count = 8 + trial % 9;
            let mut ids: Vec<usize> = (0..count as usize).collect();
            for size in (2..=count as usize).rev() {
                let k = word() as usize % size;
                ids.swap(size - 1, k);
            }
            if ids[0] == 0 {
                let last = ids.len() - 1;
                ids.swap(0, last);
            }
            let mut succs = vec![Vec::new(); count as usize];
            for position in 0..count - 1 {
                let remaining = count - position - 1;
                let degree = match position {
                    0 => 2,
                    1 => 1,
                    _ => (word() % 3).min(remaining),
                };
                if degree == 0 {
                    continue;
                }
                let first = position + 1 + word() % remaining;
                let edges = &mut succs[ids[position as usize]];
                edges.push(ids[first as usize]);
                if degree == 2 {
                    let mut second = position + 1 + word() % (remaining - 1);
                    if second >= first {
                        second += 1;
                    }
                    edges.push(ids[second as usize]);
                }
                edges.sort_unstable();
            }
            require_set_dominance(&succs, ids[0], &format!("deterministic DAG {trial}"));
        }
    }

    #[test]
    fn infinite_loop_post_dominators() {
        // 0 -> 1 -> 1 (no exit)
        let g = vec![vec![1], vec![1]];
        let ipdom = post_dominators(&g, 0);
        assert_eq!(ipdom[0], 1);
    }
}
