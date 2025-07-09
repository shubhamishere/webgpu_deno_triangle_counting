import { DelimiterStream } from "jsr:@std/streams/delimiter-stream";

/**
 * Main function to run the triangle counting process.
 * Implements an advanced degree-ordering algorithm for massive graphs.
 */
async function main() {
  const filePath = Deno.args[0];
  if (!filePath) {
    console.error("❌ Please provide the path to the graph file.");
    return;
  }

  // --- 1. Preprocess Graph ---
  console.log(`\n Reading and processing graph from: ${filePath}`);
  const startTime = performance.now();

  const {
    adj_offsets,
    adj_data,
    nodeCountForAllocation,
    officialNodeCount,
    officialEdgeCount,
  } = await processGraph(filePath);

  const processingTime = performance.now() - startTime;
  console.log(` Graph processed in: ${processingTime.toFixed(2)} ms`);
  console.log(`   - Nodes (Official): ${officialNodeCount.toLocaleString()}`);
  console.log(`   - Edges (Official): ${officialEdgeCount.toLocaleString()}`);

  // --- 2. Initialize WebGPU ---
  console.log("\n Initializing WebGPU...");
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) { throw new Error("No WebGPU adapter found."); }
  const device = await adapter.requestDevice();
  console.log(`   - Using device: ${adapter.name}`);

  // --- 3. Create GPU Buffers ---
  const U32_SIZE = 4;
  const offsetBuffer = createGPUBuffer(device, adj_offsets, GPUBufferUsage.STORAGE);
  const adjacencyBuffer = createGPUBuffer(device, adj_data, GPUBufferUsage.STORAGE);
  const resultBuffer = device.createBuffer({
    size: U32_SIZE,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });

  // --- 4. Setup Pipeline and Bind Group ---
  const shaderCode = await Deno.readTextFile("./compute.wgsl");
  const shaderModule = device.createShaderModule({ code: shaderCode });

  const bindGroupLayout = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
    ],
  });

  const bindGroup = device.createBindGroup({
    layout: bindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: offsetBuffer } },
      { binding: 1, resource: { buffer: adjacencyBuffer } },
      { binding: 2, resource: { buffer: resultBuffer } },
    ],
  });

  const pipeline = device.createComputePipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
    compute: { module: shaderModule, entryPoint: "main" },
  });

  // --- 5. Dispatch GPU Compute Job ---
  console.log("\n Dispatching compute shader...");
  const dispatchTime = performance.now();
  const commandEncoder = device.createCommandEncoder();
  const passEncoder = commandEncoder.beginComputePass();
  passEncoder.setPipeline(pipeline);
  passEncoder.setBindGroup(0, bindGroup);
  // Dispatch one workgroup per node
  const workgroupCount = Math.ceil(nodeCountForAllocation / 256);
  passEncoder.dispatchWorkgroups(workgroupCount, 1, 1);
  passEncoder.end();

  // --- 6. Read Result Back from GPU ---
  const stagingBuffer = device.createBuffer({ size: U32_SIZE, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  commandEncoder.copyBufferToBuffer(resultBuffer, 0, stagingBuffer, 0, U32_SIZE);
  device.queue.submit([commandEncoder.finish()]);

  await stagingBuffer.mapAsync(GPUMapMode.READ);
  const gpuTime = performance.now() - dispatchTime;
  console.log(`   - GPU execution time: ${gpuTime.toFixed(2)} ms`);

  const triangleCount = new Uint32Array(stagingBuffer.getMappedRange())[0];

  console.log("\n Result:");
  console.log(`   - Total Triangles Found: ${triangleCount.toLocaleString()}`);
  console.log("------------------------------------------------------\n");

  // --- 7. Cleanup ---
  device.destroy();
}

/**
 * Reads a graph file using a memory-efficient two-pass approach.
 */
async function processGraph(filePath) {
    const decoder = new TextDecoder();
    
    // --- Pass 1: Count edges and find maxNodeId ---
    console.log("   - Pass 1: Counting edges and nodes...");
    let file = await Deno.open(filePath, { read: true });
    let officialEdgeCount = 0;
    const uniqueNodeIds = new Set();
    let maxNodeId = 0;
    let validEdgeCount = 0;

    for await (const lineBytes of file.readable.pipeThrough(new DelimiterStream(new TextEncoder().encode("\n")))) {
        if (lineBytes.length === 0) continue;
        const line = decoder.decode(lineBytes);
        if (line.startsWith("#")) continue;

        officialEdgeCount++;
        const [node1, node2] = line.trim().split(/\s+/).map(Number);
        
        uniqueNodeIds.add(node1);
        uniqueNodeIds.add(node2);
        if (node1 !== node2) validEdgeCount++;

        const maxInPair = Math.max(node1, node2);
        if (maxInPair > maxNodeId) maxNodeId = maxInPair;
    }

    // --- Pass 2: Pre-allocate and fill packed edge array ---
    console.log("   - Pass 2: Building packed edge array...");
    file = await Deno.open(filePath, { read: true });
    const allPackedEdges = new BigUint64Array(validEdgeCount);
    let edgeIndex = 0;

    for await (const lineBytes of file.readable.pipeThrough(new DelimiterStream(new TextEncoder().encode("\n")))) {
        if (lineBytes.length === 0) continue;
        const line = decoder.decode(lineBytes);
        if (line.startsWith("#")) continue;

        const [node1, node2] = line.trim().split(/\s+/).map(Number);
        if (node1 === node2) continue;
        
        const u = BigInt(Math.min(node1, node2));
        const v = BigInt(Math.max(node1, node2));
        allPackedEdges[edgeIndex++] = (u << 32n) | v;
    }

    // --- Stage 3: Deduplicate and build final structures ---
    console.log("   - Stage 3: Deduplicating and building final graph structures...");
    allPackedEdges.sort();
    
    let uniqueEdgeCount = 0;
    if (allPackedEdges.length > 0) {
        uniqueEdgeCount = 1;
        for (let i = 1; i < allPackedEdges.length; i++) {
            if (allPackedEdges[i] !== allPackedEdges[i-1]) {
                uniqueEdgeCount++;
            }
        }
    }

    const uniquePackedEdges = new BigUint64Array(uniqueEdgeCount);
    if (allPackedEdges.length > 0) {
        uniquePackedEdges[0] = allPackedEdges[0];
        let count = 1;
        for (let i = 1; i < allPackedEdges.length; i++) {
            if (allPackedEdges[i] !== allPackedEdges[i-1]) {
                uniquePackedEdges[count++] = allPackedEdges[i];
            }
        }
    }

    const nodeCountForAllocation = maxNodeId + 1;
    const degrees = new Uint32Array(nodeCountForAllocation);
    for (let i = 0; i < uniquePackedEdges.length; i++) {
        const packed = uniquePackedEdges[i];
        degrees[Number(packed >> 32n)]++;
        degrees[Number(packed & 0xFFFFFFFFn)]++;
    }

    const orientedAdj = Array.from({ length: nodeCountForAllocation }, () => []);
    for (let i = 0; i < uniquePackedEdges.length; i++) {
        const packed = uniquePackedEdges[i];
        const u = Number(packed >> 32n);
        const v = Number(packed & 0xFFFFFFFFn);
        if (degrees[u] < degrees[v] || (degrees[u] === degrees[v] && u < v)) {
            orientedAdj[u].push(v);
        } else {
            orientedAdj[v].push(u);
        }
    }

    const adj_offsets = new Uint32Array(nodeCountForAllocation + 1);
    const flat_adj_data = [];
    let currentOffset = 0;

    for (let i = 0; i < nodeCountForAllocation; i++) {
        orientedAdj[i].sort((a, b) => a - b);
        adj_offsets[i] = currentOffset;
        for (const neighbor of orientedAdj[i]) {
            flat_adj_data.push(neighbor);
        }
        currentOffset += orientedAdj[i].length;
    }
    adj_offsets[nodeCountForAllocation] = currentOffset;

    return {
        adj_offsets,
        adj_data: new Uint32Array(flat_adj_data),
        nodeCountForAllocation,
        officialNodeCount: uniqueNodeIds.size,
        officialEdgeCount,
    };
}

/** Helper to create and write data to a GPU buffer */
function createGPUBuffer(device, data, usage) {
    const buffer = device.createBuffer({
        size: data.byteLength,
        usage: usage | GPUBufferUsage.COPY_DST,
        mappedAtCreation: true,
    });
    const Ctor = data.constructor;
    new Ctor(buffer.getMappedRange()).set(data);
    buffer.unmap();
    return buffer;
}

main().catch(console.error);
