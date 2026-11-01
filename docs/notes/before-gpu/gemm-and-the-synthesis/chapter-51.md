---
title: "51. The second lever: hiding latency within a single thread"
---

<div class="bge">
<!-- Title -->
<h1><span class="hl-yellow" style="font-size: 0.9em;">The second lever: hiding latency within a single thread</span></h1>
<p class="bge-meta">Phase 3 · GEMM &amp; The Synthesis · Chapter 51 of 56 · notebook date 27 / aug / 26</p>
<span class="annotation">*instruction-level parallelism*</span>

<!-- Section: Introduction -->
<p style="margin-top: calc(var(--grid-size) * 1.5);">
  Almost everyone learns one way to keep a GPU busy while its slow memory catches up: pile on more threads. When one group of threads stalls waiting for data, the chip switches to another group that is ready, and as long as you have enough groups, the cores never sit idle. This is true, and it's the story you hear everywhere. But it is only half the story, and the missing half explains a genuinely surprising fact: that a GPU running at a fraction of its supposed capacity can sometimes run at full speed anyway. There are two levers for hiding latency, not one, and understanding the second changes how you think about performance entirely.
</p>

<p class="text-blue">
  <span class="arrow">↳</span> The secret is that <strong class="hl-blue">you can hide latency within a single thread, not just across thread groups.</strong>
</p>

<!-- Section: The First Lever -->
</div>

## <span class="hl-yellow">The First Lever: Inter-Block Switching</span>

<div class="bge">
<p>
  Start with the lever everyone knows. The slow part of a GPU is fetching data from far memory, which takes a long time. So when a group of threads asks for data, it stalls, and the chip fills the wait by running other groups that already have their data. The more groups you have resident and ready, the more the chip can switch around and stay busy. How full this pool of resident groups is has a name (occupancy), and people treat filling it up as the whole goal of performance tuning. More groups, more coverage, more hiding. This is real, and it's the first lever.
</p>

<!-- Section: The Accounting Deficit -->
</div>

## <span class="hl-pink">The Accounting Deficit</span>

<div class="bge">
<p>
  But here's the trap in thinking it's the only one. The long wait is genuinely long, and the pool of groups is genuinely limited by the chip's fixed resources. Do the honest accounting and you find that filling the pool to its absolute maximum still often isn't enough on its own to cover the full length of the wait. There simply aren't enough groups allowed at once to paper over how long the memory takes. So if more groups were the only lever, the chip could never fully hide its latency, and yet it clearly does. Something else must be helping.
</p>

<!-- Section: The Second Lever -->
</div>

## <span class="hl-blue">The Second Lever: Intra-Thread Concurrency</span>

<div class="bge">
<p>
  That something is the second lever, and it lives inside a single group of threads rather than across many. So far we imagined each group firing off one memory request and then stalling until it comes back. But what if a single group fires off several independent requests before waiting for any of them? <strong class="hl-blue">It asks for the first piece of data, and instead of immediately waiting, it also asks for the second, third, and fourth</strong>, all of which it will need but none of which depend on each other. Now all those requests are traveling to memory at the same time, overlapping, and the group only waits once for all of them to arrive together rather than paying the full wait four separate times. One group, but several things in flight at once.
</p>

<!-- Section: Multiplying the Coverage -->
</div>

## <span class="hl-green">Multiplying the Coverage</span>

<div class="bge">
<p class="handwritten-highlight text-blue" style="margin-top: var(--grid-size); margin-bottom: var(--grid-size);">
  So the question this raises: if a single group can keep several requests in flight on its own, does it still need a huge pool of other groups to hide the latency?
</p>
<p>
  And the answer is no, and this is the surprise. The total amount of work in flight is what matters for hiding the wait, and that total is the number of groups times how many requests each group juggles. So you can reach the amount you need either way: with many groups each doing one thing, or with fewer groups each doing several things at once. The two levers multiply to give you the coverage you need, so pulling harder on one lets you pull less on the other.
</p>
<p>
  Which means the thing everyone treats as the master dial, how full the pool of groups is, is not actually the goal. It is one way to buy coverage, not the only way. And here is the counterintuitive consequence: a kernel running with only a modest pool of groups, well below the chip's maximum, can completely hide its latency and hit peak speed, as long as each of those groups is juggling several independent operations at once. It looks underutilized by the usual measure, the pool is half empty, and yet it runs flat out, because the second lever made up the difference. Low on the metric everyone watches, full speed in reality.
</p>

<!-- Section: Practical Code Structure -->
</div>

## <span class="hl-yellow">Practical Code Structure</span>

<div class="bge">
<p>
  So how do you actually pull this second lever, since it doesn't come for free? It comes from how you write the inner work of each thread. Instead of writing code that does one thing, waits for it, does the next thing, waits for it, you write code that launches several independent pieces of work up front and only then starts consuming their results. Often this means unrolling a loop so that a few iterations' worth of independent loads all go out together, or arranging the math so a thread holds several running results at once rather than one. The code deliberately creates independent work within a single thread, so the hardware has several things it can have in flight simultaneously. It's a code-structure decision, not a launch-configuration one.
</p>

<!-- Section: The Budget Conflict -->
</div>

## <span class="hl-red">The Budget Conflict</span>

<div class="bge">
<p>
  And this connects to a tension people hit constantly without understanding it. Pulling this second lever costs each thread more of the chip's scratch resources, because holding several things in flight means holding several partial values at once. And using more of those resources per thread means fewer groups can be resident, shrinking the first lever's pool. So the two levers are not just additive alternatives, <strong class="hl-red">they trade against each other through the same limited budget.</strong> Push hard on per-thread parallelism and you naturally get fewer resident groups, which is exactly why a low-pool kernel can still be fast: it spent its budget on the second lever instead of the first. The mistake is to force the pool full by starving each thread, which can leave you worse off than a leaner-looking kernel that hides latency from within.
</p>

<!-- Takeaway -->
<div style="margin-top: calc(var(--grid-size) * 2); margin-bottom: var(--grid-size); border-left: 3px solid var(--ink-red); padding-left: 20px;">
  <h2 class="text-red" style="font-family: 'Patrick Hand', cursive; font-size: 32px; margin-bottom: var(--grid-size);">
    ↳ The General Takeaway
  </h2>

  <p>
    Hiding memory latency needs enough independent work in flight to cover the long wait, and there are two ways to supply it: resident thread groups (lever one) and intra-thread instruction parallelism (lever two).
  </p>
  
  <p>
    <span class="hl-green">These two levers multiply</span>, so you can hit full speed with a big pool of simple groups or a small pool of busy ones, which is why low occupancy can secretly run at 100% throughput. Because registers are finite, the two levers trade off against each other.
  </p>

  <p class="handwritten-highlight text-red" style="text-align: center; margin-top: var(--grid-size);">
    The real target is total independent work in flight...<br>
    <span class="hl-yellow" style="font-family: inherit; font-size: inherit;"><strong class="hl-red bold">the best code pulls the quieter lever within each thread.</strong></span>
  </p>
</div>
</div>

---

<small>From [Before GPU Engineering](https://github.com/BEFORE-GPU-ENGINEERING/BEFORE-GPU-ENGINEERING) by Ranjan Yadav, used under the [MIT license](../LICENSE.txt).</small>
