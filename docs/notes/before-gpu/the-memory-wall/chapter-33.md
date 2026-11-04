---
title: "Chapter 33: Your ALUs are idle. Memory is the real bottleneck."
---

<div class="bge">
<!-- Title -->
<h1><span class="hl-yellow">Your ALUs are idle. Memory is the real bottleneck.</span></h1>
<p class="bge-meta">Part 9: The Memory Wall · Chapter 33 of 56 · notebook date 17 / jul / 26</p>
<span class="annotation">*worked out from first principles*</span>

<!-- Section: Introduction -->
<p style="margin-top: calc(var(--grid-size) * 1.5);">
  Your GPU is rated for a staggering number of operations per second. You run a large language model on it to generate text, and the compute units sit almost completely idle, busy maybe a tenth of the time. The expensive part of the chip you paid a fortune for is <strong class="hl-red">dark, unused, starved.</strong> 
</p>

<p class="text-blue">
  <span class="arrow">↳</span> This is not a bug, not a bad driver, not lazy code. It is the central tension of modern AI hardware, and it is getting worse every single generation. Understanding why is the difference between buying hardware that helps you and buying <strong class="hl-blue">expensive silicon that mostly waits.</strong>
</p>

<!-- Section: The Two Ceilings -->
</div>

## <span class="hl-yellow">The Two Ceilings</span>

<div class="bge">
<p>
  Start with the two ceilings every piece of GPU work lives under. One ceiling is how fast the chip can do math. The other is how fast it can deliver data from memory to the math units. 
</p>
<p>
  Which ceiling you hit depends on a single ratio: <span class="hl-yellow">how much math you do for every byte you fetch.</span> If your work does a lot of math per byte, you bump against the compute ceiling, and the chip is well used. If it does little math per byte, you hit the memory ceiling first, and no amount of extra compute power helps, because the math units finish instantly and then sit waiting for the next byte to arrive. 
</p>
<p>
  This is the whole game. Your performance is decided not by how powerful the chip is, but by whether data can reach the cores fast enough to keep them fed.
</p>

<!-- Section: The Widening Gap -->
</div>

## <span class="hl-pink">The Widening Gap</span>

<div class="bge">
<p>
  Now the trend that makes this urgent, and it is the heart of the matter. Each new generation of GPU grows its compute power dramatically, multiplying the number of math operations it can do. But memory bandwidth, how fast data moves from memory to the cores, <strong class="hl-pink">grows far more slowly.</strong> 
</p>
<p>
  So generation over generation, compute races ahead while bandwidth crawls. The gap between them widens. And here is the consequence people miss: that widening gap means each new chip ships with proportionally more compute that a normal workload cannot use, because the memory cannot feed it fast enough. 
</p>
<p>
  The advertised operations-per-second number keeps climbing, but for a memory-bound job, more of that number is unreachable than ever before. The rating becomes a measure of <span class="hl-pink">theoretical capacity, not of what you will actually get.</span>
</p>

<!-- Section: Fighting Physics -->
</div>

## <span class="hl-blue">Physics Is Not Negotiating</span>

<div class="bge">
<p>
  So why does bandwidth refuse to keep up? Why not just build memory as fast as the compute? Because bandwidth runs into <strong class="hl-blue">hard physical walls.</strong> 
</p>
<p>
  It is limited by how many physical wires you can pack to carry data, by how fast you can push signals down them without corruption, by the heat and power all that movement generates. Compute grows by clever tricks like packing more math units and using lower-precision numbers, which sidestep those walls. Bandwidth has no such escape. It is fighting physics directly, and physics is not negotiating. So the imbalance is structural, baked into what is physically possible, not a temporary engineering lag.
</p>

<!-- Section: The LLM Penalty -->
</div>

## <span class="hl-red">The LLM Penalty</span>

<div class="bge">
<p>
  Now the place this bites hardest, and it explains why your language model felt so slow. Generating text with a large model happens one word at a time, and each word requires <span class="hl-red">reading the model's entire set of parameters out of memory</span> to compute that single step. 
</p>
<p>
  Picture that. Billions of values, hauled from memory across that limited bandwidth, just to produce one word, then hauled again for the next word, and again. The math done on each value is tiny, a quick multiply. So this is the extreme memory-bound case, almost no math per byte, and the cores spend nearly all their time waiting for the flood of parameters to arrive. You bought a machine that can do enormous math, and you are using it as an <strong class="hl-red">expensive device for reading memory one word at a time.</strong>
</p>

<!-- Section: Batching to the Rescue -->
</div>

## <span class="hl-green">Batching to the Rescue</span>

<div class="bge">
<p>
  So how do you rescue those idle cores? And this is the key lever. The problem with generating for a single user is that you read all those parameters just to serve that one user's one word. 
</p>
<p>
  But if you <strong class="hl-green">batch many users together</strong> and generate a word for all of them at once, you read each parameter from memory a single time and immediately use it for every user in the batch. Suddenly you are doing much more math per byte fetched, because the same expensive data delivery now feeds many computations instead of one. The batch pushes your work up off the memory ceiling toward the compute ceiling, and the cores light up. This is why serving systems fight so hard to batch requests together; it is the main way to turn a memory-starved workload into a compute-rich one.
</p>

<!-- Section: The Trade-Offs -->
</div>

## <span class="hl-yellow">The Free Lunch Illusion</span>

<div class="bge">
<p>
  But of course the fix is not free, and the objection is real. If batching more is better, why not batch enormously? Two things break. 
</p>
<p>
  First, holding many users' work in progress consumes a lot of memory, and pile on enough and you simply run out, crashing. Second, the <span class="hl-yellow">very first response gets slower</span>, because the machine is now busy chewing through a big batch before it can return anyone's first word. So you are trading one scarce resource, bandwidth, for another, memory capacity, and trading raw speed against responsiveness. There is no free lunch, only a balance you tune.
</p>

<!-- Section: The Two Phases -->
</div>

## <span class="hl-pink">The Split Personality of LLMs</span>

<div class="bge">
<p>
  Worth naming one more subtlety, because it shows the problem is not uniform. The same model has two very different phases. 
</p>
<p>
  Digesting a user's initial prompt is <strong class="hl-pink">compute-heavy</strong>, because you process all the prompt's words together in parallel, plenty of math per byte, the cores stay busy. But then generating the reply, word by word, flips to <strong class="hl-pink">memory-bound</strong>, one parameter-sweep per word. So a single request lives on both sides of the divide: compute-rich while reading your question, memory-starved while writing its answer. Advanced systems even split these two phases onto different machines, matching each phase to hardware suited for its bottleneck.
</p>

<!-- Takeaway -->
<div style="margin-top: calc(var(--grid-size) * 2); margin-bottom: var(--grid-size); border-left: 3px solid var(--ink-red); padding-left: 20px;">
  <h2 class="text-red" style="font-family: 'Patrick Hand', cursive; font-size: 32px; margin-bottom: var(--grid-size);">
    ↳ The General Takeaway
  </h2>

  <p>
    So the whole picture. Every GPU workload is capped by either compute or memory bandwidth, decided by how much math it does per byte. 
  </p>
  
  <p>
    <strong class="hl-pink">Compute has been racing ahead of bandwidth for years</strong>, and the gap keeps widening, so each new chip carries more compute that ordinary workloads cannot feed—dark silicon by design. Generating text with a large model is the worst case, sweeping the entire model from memory for every single word, leaving the cores idle. The main escape is batching many requests so each fetched byte does more work, though it costs memory and responsiveness. 
  </p>

  <p>
    The deepest truth, and it is the one this whole field keeps circling back to, is that a modern GPU is not really a compute engine you feed data to. 
  </p>

  <p class="handwritten-highlight text-red" style="text-align: center; margin-top: var(--grid-size);">
    It is a memory system with some compute bolted on.<br>
    <span class="hl-yellow" style="font-family: inherit; font-size: inherit;"><strong class="hl-red bold">The scarce resource... is almost never the math.</strong></span><br>
    It is how fast you can move the bytes.
  </p>
</div>
</div>

---

<small>From [Before GPU Engineering](https://github.com/BEFORE-GPU-ENGINEERING/BEFORE-GPU-ENGINEERING) by Ranjan Yadav, used under the [MIT license](../LICENSE.txt).</small>
