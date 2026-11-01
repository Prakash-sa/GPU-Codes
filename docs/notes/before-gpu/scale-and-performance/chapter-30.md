---
title: "30. Why GPU programs actually slow down"
---

<div class="bge">
<!-- Title -->
<h1><span class="hl-yellow">Why GPU programs actually slow down</span></h1>
<p class="bge-meta">Phase 2 · Scale &amp; Performance · Chapter 30 of 56 · notebook date 17 / jul / 26</p>
<span class="annotation">*worked out from first principles*</span>

<!-- Section: The Six Horsemen -->
<p style="margin-top: calc(var(--grid-size) * 1.5);">
  Your GPU is rated for a hundred trillion operations a second. You run your code and it delivers five. Is the chip broken? Do you need cleverer math? Almost never. A GPU is a supercar, and a supercar is rarely slow because the engine is weak. It's slow because it's <strong class="hl-red">stuck in traffic or out of fuel</strong>. Beginners respond to a slow kernel by tuning the arithmetic. Experts know the real fight is almost always somewhere else, in how data moves and how work gets scheduled. There are <strong class="hl-pink">six classic ways a GPU ends up starving</strong>, and knowing all six turns a mysterious slowdown into a diagnosis.
</p>

<p class="text-blue">
  <span class="arrow">↳</span> The thing to hold onto first is what unites all six. Every one of them is a different way of failing to keep the compute cores fed with useful work. The cores themselves are almost never the problem. So each bottleneck is really a story about <strong class="hl-blue">data or scheduling failing to deliver</strong>, and the symptom is always the same: <span class="hl-yellow">cores sitting idle while the clock runs</span>.
</p>

<!-- Horseman 1: Transfer Overhead -->
</div>

## <span class="hl-red">1. Transfer Overhead</span>

<div class="bge">
<p>
  The first horseman is <strong class="hl-red">transfer overhead</strong>, the worst of all. This is when your program spends nearly all its time shipping data back and forth between the main computer and the GPU across the narrow bridge that connects them, for a sliver of actual math. You copy data over, do one tiny operation, copy it back, and the <span class="hl-pink">round trip dwarfs the work</span>. The cores might be busy eight percent of the time. 
</p>
<p>
  <strong class="hl-green">The cure:</strong> Stop the ping-pong. Keep the data living on the GPU across many operations, and batch work into big transfers instead of many tiny ones, so the crossing cost is paid rarely and amortized.
</p>

<!-- Horseman 2: Memory-Bound -->
</div>

## <span class="hl-blue">2. Memory-Bound</span>

<div class="bge">
<p>
  The second is being <strong class="hl-blue">memory-bound</strong>, and this one surprises people who assume the math is the point. The deciding number is how much arithmetic you do per byte you fetch. For something like adding two arrays, you move a dozen bytes to do a single addition. The cores finish that instantly and then sit waiting for the memory system to deliver the next batch, which it simply cannot do fast enough. The cores starve not because they are slow but because the <span class="hl-blue">data pipe is the limit</span>. 
</p>
<p>
  <strong class="hl-green">The fix:</strong> Do more math per byte, by pulling a piece of data into fast on-chip memory once and reusing it many times before letting it go, so each expensive fetch pays off across many operations.
</p>

<!-- Horseman 3: Low Occupancy -->
</div>

## <span class="hl-yellow">3. Low Occupancy</span>

<div class="bge">
<p>
  The third is <strong class="hl-yellow">low occupancy</strong>, which attacks the GPU's core survival trick. The way a GPU hides the long wait for memory is by always having other bundles of threads ready to run while some are stalled. Occupancy is how full that reservoir of ready bundles is. When it's low, a bundle stalls on memory and there is no backup to swap in, so the <span class="hl-yellow">cores go idle waiting</span>. And the usual cause is greed: each thread grabbing too many registers or each block hogging too much fast memory, which leaves room for too few bundles to be resident at once. 
</p>
<p>
  <strong class="hl-green">The fix:</strong> Trim that per-thread appetite so more bundles fit and the scheduler always has a backup ready.
</p>

<!-- Horseman 4: Branch Divergence -->
</div>

## <span class="hl-pink">4. Branch Divergence</span>

<div class="bge">
<p>
  The fourth is <strong class="hl-pink">branch divergence</strong>, which strikes at how a bundle executes. The 32 threads of a bundle move in lockstep, all running the same instruction together. Put a branch in the code where some threads go one way and others go the other, and the hardware cannot run both paths at once. It runs the first path with half the threads active and the rest frozen, then runs the other path with the halves swapped. What should have been one pass becomes two, and the <span class="hl-pink">more tangled the branching, the worse it gets</span>. 
</p>
<p>
  <strong class="hl-green">The fix:</strong> Arrange things so threads in the same bundle tend to take the same path, for instance by grouping similar data together, or by replacing branches with arithmetic that all threads run identically.
</p>

<!-- Horseman 5: Synchronization Cost -->
</div>

## <span class="hl-green">5. Synchronization Cost</span>

<div class="bge">
<p>
  The fifth is <strong class="hl-green">synchronization cost</strong>, the price of making threads wait for each other. A barrier that makes every thread in a block pause until all arrive is necessary sometimes, but it is not free. When you hit one, every thread waits for the slowest straggler, so if one thread is delayed, <span class="hl-green">hundreds sit idle for it</span>. Sprinkle barriers everywhere just to be safe and you turn a smooth parallel flow into stop-and-go traffic. 
</p>
<p>
  <strong class="hl-green">The fix:</strong> Discipline. Sync only exactly where data genuinely crosses between threads, use the lighter bundle-level sync when you only need a bundle to agree, and never scatter barriers as a superstition.
</p>

<!-- Horseman 6: Poor Coalescing -->
</div>

## <span class="hl-red">6. Poor Coalescing</span>

<div class="bge">
<p>
  The sixth is <strong class="hl-red">poor coalescing</strong>, the most common and most invisible killer of all. When a bundle reads memory, the hardware delivers it in fixed-size boxes. If the 32 threads read neighboring slots, they all fit in one box, one fetch. If they read scattered slots, each triggers its own full box fetch, most of which is thrown away, and the bundle drowns in <span class="hl-red">wasteful transactions</span>. The cores stall waiting for a flood of nearly-empty deliveries. 
</p>
<p>
  <strong class="hl-green">The fix:</strong> Arrange data so that neighboring threads read neighboring addresses, choosing layouts where the value all threads want at once sits packed together.
</p>

<!-- Takeaway -->
<div style="margin-top: calc(var(--grid-size) * 2); margin-bottom: var(--grid-size); border-left: 3px solid var(--ink-red); padding-left: 20px;">
  <h2 class="text-red" style="font-family: 'Patrick Hand', cursive; font-size: 32px; margin-bottom: var(--grid-size);">
    ↳ The General Takeaway
  </h2>

  <p>
    Step back and the pattern is unmistakable. Six different points in the journey from data to computation, and each one is a place the flow can choke. 
  </p>
  
  <ul>
    <li><strong class="hl-red">Transfer overhead</strong> means the data isn't even on the GPU yet.</li>
    <li><strong class="hl-red">Memory-bound</strong> means it's there but can't arrive fast enough.</li>
    <li><strong class="hl-red">Low occupancy</strong> means it arrives but no bundle is ready to use it.</li>
    <li><strong class="hl-red">Divergence</strong> means the bundle is ready but wastes half itself on frozen threads.</li>
    <li><strong class="hl-red">Sync cost</strong> means the threads are ready but forced to wait on stragglers.</li>
    <li><strong class="hl-red">Poor coalescing</strong> means the memory requests are ready but multiply into a flood of wasted fetches.</li>
  </ul>

  <p>
    The advertised peak speed of a GPU is a ceiling you approach only when all six of these are handled, never a floor you get for free. So when a kernel runs slow, the useful question is never "how do I make the math faster?". 
  </p>

  <p class="handwritten-highlight text-red" style="text-align: center; margin-top: var(--grid-size);">
    The real question is:<br>
    <span class="hl-yellow" style="font-family: inherit; font-size: inherit;"><strong class="hl-red bold">"Which of these six is starving my cores right now?"</strong></span><br>
    Find the one that's choking, fix that, and the supercar finally gets out of traffic.
  </p>
</div>
</div>

---

<small>From [Before GPU Engineering](https://github.com/BEFORE-GPU-ENGINEERING/BEFORE-GPU-ENGINEERING) by Ranjan Yadav, used under the [MIT license](../LICENSE.txt).</small>
