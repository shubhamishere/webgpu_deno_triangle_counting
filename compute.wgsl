// --- Storage Buffers ---
// Note: We no longer need the `edges` buffer.

@group(0) @binding(0) var<storage, read> adj_offsets: array<u32>;
@group(0) @binding(1) var<storage, read> adj_data: array<u32>;
@group(0) @binding(2) var<storage, read_write> triangle_count: atomic<u32>;


// --- Helper Function ---

/**
 * Finds the number of common elements between two sorted neighbor lists.
 * This is the core of the triangle counting logic.
 */
fn count_intersection(u_start: u32, u_end: u32, v_start: u32, v_end: u32) -> u32 {
    var count = 0u;
    var i = u_start;
    var j = v_start;

    while (i < u_end && j < v_end) {
        let neighbor_u = adj_data[i];
        let neighbor_v = adj_data[j];

        if (neighbor_u == neighbor_v) {
            count = count + 1u;
            i = i + 1u;
            j = j + 1u;
        } else if (neighbor_u < neighbor_v) {
            i = i + 1u;
        } else {
            j = j + 1u;
        }
    }
    return count;
}


// --- Main Entry Point ---

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // Each thread is now responsible for one NODE, not an edge.
    let u = global_id.x;
    let node_count = arrayLength(&adj_offsets) - 1u;

    if (u >= node_count) {
        return;
    }

    // Get the (oriented) neighbor list for node `u`.
    let u_adj_start = adj_offsets[u];
    let u_adj_end = adj_offsets[u + 1u];

    // For every neighbor `v` of `u`...
    for (var i = u_adj_start; i < u_adj_end; i = i + 1u) {
        let v = adj_data[i];

        // ...find the number of common neighbors between `u` and `v`.
        // This is the number of triangles that include the edge (u,v).
        let v_adj_start = adj_offsets[v];
        let v_adj_end = adj_offsets[v + 1u];
        
        let common_neighbors = count_intersection(u_adj_start, u_adj_end, v_adj_start, v_adj_end);

        if (common_neighbors > 0) {
            // Atomically add the count to the global result.
            // With this algorithm, each triangle is counted exactly once.
            atomicAdd(&triangle_count, common_neighbors);
        }
    }
}
