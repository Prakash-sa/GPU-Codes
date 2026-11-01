---
title: "44. What is memory coalescing? 32 threads, 1 transaction vs 32 transactions."
---

<div class="bge">
<!-- Title -->
<h1><span class="hl-yellow" style="font-size: 0.9em;">What is memory coalescing? 32 threads, 1 transaction vs 32 transactions.</span></h1>
<p class="bge-meta">Phase 3 · Global Memory &amp; Coalescing · Chapter 44 of 56 · notebook date 17 / jul / 26</p>
<span class="annotation">*the tile*</span>

<!-- Section: Introduction -->
<p style="margin-top: calc(var(--grid-size) * 1.5);">
  Multiplying two matrices is the single most important operation a GPU does. It is the heart of every neural network layer, every attention calculation, nearly all the heavy lifting in AI. 
</p>

<p class="text-blue">
  <span class="arrow">↳</span> And the difference between a naive way of doing it and the clever way, called tiling, is not a small tweak. It is often close to a tenfold speedup, from the same arithmetic, on the same hardware. 
</p>

<p>
  Tiling is the flagship example of the one idea that governs almost all GPU performance: that the cost is never the math, it is moving the data. Seeing exactly how it rescues matrix multiplication teaches you the whole philosophy in one concrete picture.
</p>

<!-- Section: The Setup -->
</div>

## <span class="hl-yellow">The Setup</span>

<div class="bge">
<p>
  Start with what multiplying two matrices requires. Every entry in the result is computed by taking one row from the first matrix and one column from the second, marching down them together, multiplying pairs and adding up. 
</p>
<p>
  So to fill in the whole result grid, every row of the first matrix has to meet every column of the second. The total amount of arithmetic is large but perfectly parallel: one thread can compute each output entry independently. So far this looks like an ideal GPU job, tons of independent work.
</p>

<!-- Section: The Fatal Flaw -->
</div>

## <span class="hl-pink">The Fatal Flaw</span>

<div class="bge">
<p>
  Now the naive approach and its fatal flaw. You assign one thread to each output entry, and each thread reads its row from the first matrix and its column from the second, straight out of the slow main memory, and does its multiply-add march. 
</p>

<p>
  And here is the waste. Think about all the threads computing entries in the same row of the result. They all need the exact same row from the first matrix. But each one independently goes out to slow memory and fetches that entire row for itself. The same row gets pulled from slow memory again and again, once per thread that needs it. And the same is true for columns. 
</p>

<p>
  Every piece of the input matrices gets hauled out of slow memory not once, but as many times as there are threads that touch it, which can be hundreds of times over. The actual multiplying is quick, so nearly all the time is spent on this mountain of repeated slow-memory traffic. The operation is badly memory-bound, drowning in redundant fetches.
</p>

<!-- Section: The Fix -->
</div>

## <span class="hl-blue">Cooperative Loading</span>

<div class="bge">
<p class="handwritten-highlight text-red" style="margin-top: var(--grid-size); margin-bottom: var(--grid-size);">
  So the question that points straight at the fix: if the problem is that the same rows and columns are fetched from slow memory over and over, can we fetch each piece just once, put it somewhere fast, and let all the threads that need it share that one copy?
</p>

<p>
  That is tiling, and it uses the fast on-chip scratchpad memory that a group of threads shares. Instead of each thread independently streaming its whole row and column from slow memory, the threads cooperate. They break the matrices into small square blocks called <strong class="hl-blue">tiles</strong>. 
</p>
<p>
  The group of threads works together to load one tile from each input matrix out of slow memory and into the fast shared scratchpad, one single trip. Then, with those tiles sitting in fast memory, every thread does its share of the multiplying by reading from the fast scratchpad, not slow memory. 
</p>
<p>
  Because the tile holds data that many threads need, each value loaded once from slow memory now gets reused by many threads out of fast memory. Once the group has squeezed all the arithmetic it can out of the current pair of tiles, it loads the next pair, and marches across the matrices tile by tile, accumulating the results as it goes.
</p>

<!-- Section: The Dramatic Win -->
</div>

## <span class="hl-green">The Dramatic Win</span>

<div class="bge">
<p>
  Now see why this is such a dramatic win, because it is worth making concrete. In the naive version, each value from the input was dragged out of slow memory once for every thread that used it. In the tiled version, each value is dragged out of slow memory once for the whole tile, and then reused by every thread in the tile from fast memory. 
</p>
<p>
  So if a tile is, say, sixteen by sixteen, then each value loaded serves a whole row or column of threads within that tile, meaning you cut the slow-memory traffic by a factor equal to the tile's width. Load once, reuse many times. The mountain of redundant slow fetches collapses to a fraction. And since the operation was memory-bound, slashing the memory traffic by that factor speeds up the whole thing by nearly the same factor. You moved the bottleneck off of memory and back onto the actual math, which is exactly where you want it.
</p>

<!-- Section: The Tension -->
</div>

## <span class="hl-yellow">The Tension</span>

<div class="bge">
<p class="handwritten-highlight text-blue" style="margin-top: var(--grid-size); margin-bottom: var(--grid-size);">
  Now an objection that gets at the real tradeoff: if bigger tiles mean more reuse and less slow-memory traffic, why not use enormous tiles and reuse everything?
</p>

<p>
  Because the fast scratchpad is tiny. It can only hold so much, and a tile has to fit inside it. Push the tile too big and it simply does not fit. Worse, the more scratchpad each group of threads claims, the fewer groups can run on the hardware at once, which chokes the GPU's ability to hide delays by switching between groups. 
</p>
<p>
  So tile size is a balance: big enough to get strong reuse, small enough to fit in the scratchpad and leave room for enough groups to run. It's the same tension that runs through all of this hardware, the fast memory is precious and scarce, and using it well means using just the right amount.
</p>

<!-- Section: The Subtlety of Loading -->
</div>

## <span class="hl-pink">The Subtlety of Loading</span>

<div class="bge">
<p>
  There is one more subtlety that connects to how memory is delivered. When the group loads a tile from slow memory, it matters that the threads read neighboring locations together, so the hardware can grab them in one efficient sweep rather than many scattered fetches. 
</p>
<p>
  So a well-written tiled multiply is careful to arrange the loading so that consecutive threads pull consecutive values, keeping the tile-loading itself efficient. The tiling gives you reuse, and careful loading gives you efficient fetches, and you need both for the full speedup.
</p>

<!-- Takeaway -->
<div style="margin-top: calc(var(--grid-size) * 2); margin-bottom: var(--grid-size); border-left: 3px solid var(--ink-red); padding-left: 20px;">
  <h2 class="text-red" style="font-family: 'Patrick Hand', cursive; font-size: 32px; margin-bottom: var(--grid-size);">
    ↳ The General Takeaway
  </h2>

  <p>
    So the whole picture. Multiplying matrices has every thread needing the same rows and columns, and the naive version fetches each of those from slow memory once per thread, a crushing pile of redundant traffic that leaves the operation memory-bound. 
  </p>
  
  <p>
    Tiling fixes it by having a group of threads cooperatively load small blocks into the fast shared scratchpad once, then reuse each loaded value across many threads, cutting slow-memory traffic by a factor of the tile width and flipping the operation from memory-bound to compute-bound. The tile size is a balance against the scarce fast memory, and the loading must be arranged for efficient fetches. 
  </p>

  <p class="handwritten-highlight text-red" style="text-align: center; margin-top: var(--grid-size);">
    And the reason this one example matters so much is that this exact pattern:<br>
    <span class="hl-yellow" style="font-family: inherit; font-size: inherit;"><strong class="hl-red bold">stage a chunk in fast memory and reuse it heavily before moving on,</strong></span><br>
    is the beating heart of fast matrix multiplication, which is itself the beating heart of essentially all of modern AI.
  </p>
</div>
</div>

---

<small>From [Before GPU Engineering](https://github.com/BEFORE-GPU-ENGINEERING/BEFORE-GPU-ENGINEERING) by Ranjan Yadav, used under the [MIT license](../LICENSE.txt).</small>
