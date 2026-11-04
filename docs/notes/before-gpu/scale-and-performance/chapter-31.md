---
title: "Chapter 31: Why matrix add scales but a linked list dies"
---

<div class="bge">
<!-- Title -->
<h1><span class="hl-yellow">Why matrix add scales but a linked list dies</span></h1>
<p class="bge-meta">Part 8: Scale &amp; Performance · Chapter 31 of 56 · notebook date 17 / jul / 26</p>
<span class="annotation">*worked out from first principles*</span>

<!-- Section: Introduction -->
<p style="margin-top: calc(var(--grid-size) * 1.5);">
  Take two tasks that look almost identical on paper. Adding two big matrices together, and walking down a linked list. Both are just memory reads and simple operations. On a GPU, one of them <strong class="hl-green">flies at full speed</strong> and the other <strong class="hl-red">crawls so badly it barely beats a phone</strong>. And the gap between them teaches you the single most important thing about what a GPU actually is.
</p>

<p class="text-blue">
  <span class="arrow">↳</span> A GPU is not a general brain that speeds up any code. It's a machine that runs one specific shape of problem beautifully, and chokes on the opposite shape. <strong class="hl-blue">Matrix addition is that shape. A linked list is its nightmare.</strong>
</p>

<!-- Section: The Dream -->
</div>

## <span class="hl-green">The Dream Shape</span>

<div class="bge">
<p>
  Start with why matrix addition is the dream. It satisfies three properties at once. 
</p>
<p>
  First, <strong class="hl-yellow">one element, one thread</strong>. Each thread computes its own position from a simple formula and goes straight to its data, no searching. Second, <strong class="hl-yellow">zero communication</strong>. The thread handling one cell never waits on the thread handling the neighbor, because each result depends only on its own two inputs. Run them in any order, no coordination needed. Third, <strong class="hl-yellow">the reads line up perfectly</strong>, so 32 neighboring threads pull 32 neighboring values in a single clean memory fetch. 
</p>
<p>
  Independent work, predictable addresses, tidy memory. This is the exact shape the hardware was built for, and it scales almost perfectly, every core busy at once.
</p>

<!-- Section: The Nightmare -->
</div>

## <span class="hl-red">The Nightmare Chain</span>

<div class="bge">
<p>
  Now the linked list, and watch every one of those properties collapse. A linked list is a chain, each node holding the location of the next. And here is the killer: <span class="hl-pink">you cannot reach the hundredth node without first walking through all ninety-nine before it</span>, because the only way to learn where a node lives is to read the one pointing to it. 
</p>
<p>
  So the whole traversal is forced into single file. One thread walks the chain while thousands of cores sit stone cold. There is no parallelism to hand out, because step one hundred literally cannot begin until step ninety-nine has finished. The work is <strong class="hl-red">serial by its very nature</strong>.
</p>

<!-- Section: The Rescue -->
</div>

## <span class="hl-blue">The Failed Rescue</span>

<div class="bge">
<p>
  And the obvious rescue attempt fails. What if you launch a thread per node and let each follow its own pointer? It doesn't work, because thread one wants the second node but has no idea where it is until thread zero has read the first node's pointer. The threads are <span class="hl-blue">chained by a dependency</span>, each unable to start until the previous reveals the next address. The pointer itself is the barrier.
</p>

<!-- Section: Memory Scatter -->
</div>

## <span class="hl-pink">The Memory Scatter</span>

<div class="bge">
<p>
  Now it gets worse, on two fronts that are specific to how a GPU is built. First, memory. The nodes of a linked list were allocated whenever and wherever, so they are <strong class="hl-pink">scattered randomly across memory</strong>. One node here, the next one far away, the one after somewhere else entirely. 
</p>
<p>
  So even when threads do read nodes, each read lands in a completely different region, and the hardware, which likes to deliver neighboring data in one efficient sweep, is instead forced into a separate fetch for every single node, most of each fetch wasted. The tidy one-sweep delivery that made matrix addition fast is impossible when the data is sprinkled all over.
</p>

<!-- Section: No Latency Hiding -->
</div>

## <span class="hl-yellow">The Fatal Stall</span>

<div class="bge">
<p>
  And the second front is the one that truly kills it. A GPU survives long memory waits by having other work ready. When one group of threads stalls waiting for memory, the scheduler instantly swaps in another group whose data is already available, so the cores stay busy and the wait is hidden. 
</p>
<p>
  But in a linked list walk, <span class="hl-yellow">there is no other ready group</span>, because everything is waiting on the same next pointer to be revealed. Every thread is stuck on the same dependency, so there is nothing to switch to. The scheduler looks for a backup and finds none. The cores sit fully exposed to the memory delay, idling through every single hundreds-of-cycles wait with nothing to fill it. The GPU's whole survival trick depends on having independent work lying around, and a pointer chase provides exactly none.
</p>

<p class="text-blue">
  <span class="arrow">↳</span> And notice why the GPU can't even see the problem coming. It cannot fetch the next node early, because it doesn't know where the next node is until it finishes reading the current one. The address is <strong class="hl-blue">hidden inside the data you haven't loaded yet</strong>. So there is no guessing ahead, no running start. Every hop is a fresh cold wait.
</p>

<!-- Section: The Fix -->
</div>

## <span class="hl-green">Flattening the Data</span>

<div class="bge">
<p>
  So are graphs and trees and sparse networks simply hopeless on a GPU, given they are all pointers and links by nature? No, but the fix is not to make the hardware chase pointers faster. It's to <strong class="hl-green">get rid of the pointers entirely</strong> by changing how the data is stored. 
</p>
<p>
  Instead of nodes scattered across memory linked by addresses, you flatten the whole structure into a few flat arrays laid out back to back. One array holds all the values in order, another records where each group of connections begins, so you can jump straight to any row's data by simple arithmetic instead of following a trail. The pointer chasing vanishes. Now different threads can process different rows independently, their reads fall neighboring in memory so they coalesce into clean fetches, and there is plenty of independent work for the scheduler to hide latency behind. The exact same graph, stored two ways, and <span class="hl-green">only the flattened version fits the machine</span>.
</p>

<!-- Takeaway -->
<div style="margin-top: calc(var(--grid-size) * 2); margin-bottom: var(--grid-size); border-left: 3px solid var(--ink-red); padding-left: 20px;">
  <h2 class="text-red" style="font-family: 'Patrick Hand', cursive; font-size: 32px; margin-bottom: var(--grid-size);">
    ↳ The General Takeaway
  </h2>

  <p>
    So the whole picture. <strong class="hl-green">Matrix addition wins</strong> because it is independent work with predictable, neighboring addresses, the native shape of the hardware. 
  </p>
  
  <p>
    <strong class="hl-red">A linked list loses</strong> because it is a forced serial chain of data-dependent, scattered, unpredictable reads, which defeats parallelism, defeats efficient memory delivery, and defeats latency hiding, all three at once. 
  </p>

  <p>
    The deepest truth underneath is that a GPU is not a general-purpose computer, it's a wide array that thrives on regular, flat, independent data and struggles with irregular, pointer-linked, serial data. The CPU is the one built for pointer-heavy tangled structures. 
  </p>

  <p class="handwritten-highlight text-red" style="text-align: center; margin-top: var(--grid-size);">
    The winning move on a GPU is rarely to write cleverer code. <br>
    <span class="hl-yellow" style="font-family: inherit; font-size: inherit;"><strong class="hl-red bold">It's to reshape your data.</strong></span><br>
    Make it stop looking like a chain of pointers and start looking like a flat grid the hardware already loves.
  </p>
</div>
</div>

---

<small>From [Before GPU Engineering](https://github.com/BEFORE-GPU-ENGINEERING/BEFORE-GPU-ENGINEERING) by Ranjan Yadav, used under the [MIT license](../LICENSE.txt).</small>
