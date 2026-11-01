---
title: "32. When NOT to use a GPU — and the reduction tree"
---

<div class="bge">
<!-- Title -->
<h1><span class="hl-yellow">When NOT to use a GPU — and the reduction tree</span></h1>
<p class="bge-meta">Phase 2 · Scale &amp; Performance · Chapter 32 of 56 · notebook date 17 / jul / 26</p>
<span class="annotation">*worked out from first principles*</span>

<!-- Section: Introduction -->
<p style="margin-top: calc(var(--grid-size) * 1.5);">
  You have a million numbers and you want their sum. On a machine with thousands of cores, how fast can you go? The obvious answer is embarrassing, and fixing it reveals one of the most elegant patterns in all of parallel computing, the one hiding inside every sum, every average, every max, every dot product a neural network computes. It turns a job that looks stubbornly sequential into one that finishes in the blink of an eye. 
</p>

<p class="text-blue">
  <span class="arrow">↳</span> The trick is to <strong class="hl-blue">stop adding and start running a tournament.</strong>
</p>

<!-- Section: The Trap -->
</div>

## <span class="hl-red">The One-Soldier Trap</span>

<div class="bge">
<p>
  Start with the naive approach, because its trap is instructive. One thread, a loop, add each number to a running total, a million times over. It works. But on a chip with thousands of cores, exactly one is working and all the rest sit idle. <span class="hl-pink">You brought an army and sent one soldier.</span> The total time is a million sequential additions. A plain CPU would do this just as well, so the GPU bought you nothing.
</p>

<!-- Section: The Toll Booth -->
</div>

## <span class="hl-pink">The One-Lane Toll Booth</span>

<div class="bge">
<p>
  So the natural fix: unleash all the threads. Give every one of a million threads a number, and have them all add into a single shared total at once. And this is the trap that catches everyone, because it collapses instantly. 
</p>
<p>
  That shared total can only be safely updated by one thread at a time, otherwise two threads read the same running value, both add their piece, and one update gets stomped. So the hardware forces them to take turns on that one location. A million threads now stand in a single-file line waiting to touch the total, one at a time. <strong class="hl-red">You launched a million workers and built them a one-lane toll booth.</strong> It's as serial as the loop, sometimes worse, because now there is the overhead of all that queueing on top.
</p>

<!-- Section: The Associative Property -->
</div>

## <span class="hl-blue">The Hidden Freedom</span>

<div class="bge">
<p>
  So both extremes fail. One thread doing everything is too serial. A million threads fighting over one spot is also serial, just with a traffic jam. 
</p>
<p>
  The way out comes from a simple property of addition that we normally never think about: <strong class="hl-blue">it doesn't care how you group it</strong>. Adding a to b then adding c gives the same answer as adding a to b and c to d separately and then combining. The order and grouping are free. That freedom is the whole key, because it means you don't have to funnel everything through one running total. You can add lots of independent pairs at the same time.
</p>

<!-- Section: The Tournament -->
</div>

## <span class="hl-green">The Parallel Tournament</span>

<div class="bge">
<p>
  So picture a tournament instead. Round one, take the million numbers and pair them up, and every pair gets added by its own thread, all at the exact same instant. Half a million additions happen in parallel in the time of a single addition. Now you have half a million partial sums. 
</p>
<p>
  Round two, pair those up and add them in parallel too, leaving a quarter million. Round three, an eighth. Each round halves how many numbers remain, and each round takes just one parallel step. Keep halving and you go from a million, to half a million, to a quarter million, and after only about twenty of these halvings you are down to the single final answer. 
</p>
<p>
  That is the magic: <strong class="hl-green">a million numbers summed in roughly twenty parallel steps instead of a million sequential ones.</strong> Halving repeatedly reaches one absurdly fast, which is the same reason a knockout tournament of a thousand players crowns a champion in only ten rounds.
</p>

<!-- Section: Hardware Mapping -->
</div>

## <span class="hl-yellow">The Hardware Reality</span>

<div class="bge">
<p>
  Now how this maps onto the real hardware, which adds one more clever layer. You don't run the whole million-wide tournament in one giant pool, because threads in different corners of the chip can't cheaply talk to each other. 
</p>
<p>
  Instead you <span class="hl-yellow">break the million numbers into many small groups</span>, and each group runs its own little tournament entirely inside the fast on-chip scratchpad memory, where its threads can coordinate quickly. Each group chews its chunk down to a single partial sum. So if you had a million numbers in groups of a thousand, you now have a thousand partial sums, one from each group, computed fully in parallel across the chip.
</p>

<!-- Section: Combining the Rest -->
</div>

## <span class="hl-blue">Combining the Survivors</span>

<div class="bge">
<p>
  And then the obvious question, how do you combine those last thousand partial sums? And here you have a choice that shows the pattern is recursive. 
</p>
<p>
  You could have each group's one partial sum do a single safe add into a global total, and because there are now only a thousand of them instead of a million, the toll booth that was catastrophic before is now trivial, a thousand quick additions with almost no contention. Or, more elegantly, you take those thousand partial sums and run the exact same tournament on them again, a second pass that collapses a thousand to one. The reduction calls itself. Either way, the <strong class="hl-blue">million-way traffic jam became a thousand-way trickle</strong>, which is nothing.
</p>

<!-- Section: The Hidden Flaw -->
</div>

## <span class="hl-pink">The Folding Trick</span>

<div class="bge">
<p>
  There is one subtlety worth knowing, because the naive tournament has a hidden flaw. If in each round you pair up neighbors right next to each other, then as rounds go on the active threads get scattered further and further apart, and this makes the bundles of threads execute inefficiently, with half a bundle working and half idle, and it also causes them to collide while accessing the fast memory. 
</p>
<p>
  The fix is to pair threads that are far apart to begin with, <strong class="hl-pink">folding the data in half each round like folding a strip of paper</strong>, so the still-active threads always stay packed tightly together. Same tournament, same answer, but arranged so the hardware stays fully fed. This is the difference between a textbook reduction and a fast one.
</p>

<!-- Takeaway -->
<div style="margin-top: calc(var(--grid-size) * 2); margin-bottom: var(--grid-size); border-left: 3px solid var(--ink-red); padding-left: 20px;">
  <h2 class="text-red" style="font-family: 'Patrick Hand', cursive; font-size: 32px; margin-bottom: var(--grid-size);">
    ↳ The General Takeaway
  </h2>

  <p>
    So the whole picture. Summing a pile of numbers looks inherently sequential, and both obvious approaches—one thread looping or a million threads brawling over one total—are serial disasters. 
  </p>
  
  <p>
    But because addition can be regrouped freely, you can run it as a tournament, <span class="hl-green">pairing and halving in parallel until one value remains</span>, turning a million steps into about twenty. On real hardware you do this in two tiers, each group reducing itself in fast local memory, then the handful of survivors combined cheaply, with the whole thing arranged so active threads stay clustered for the hardware's sake.
  </p>

  <p>
    And the reason this matters far beyond summing is that this exact pattern—collapse many values into one through parallel pairing—is how a GPU computes <strong class="hl-blue">averages, maximums, dot products, and the normalizing totals inside the attention and softmax layers</strong> of every large model. 
  </p>

  <p class="handwritten-highlight text-red" style="text-align: center; margin-top: var(--grid-size);">
    The humble sum, done right... <br>
    <span class="hl-yellow" style="font-family: inherit; font-size: inherit;"><strong class="hl-red bold">is one of the deepest tricks the machine has.</strong></span>
  </p>
</div>
</div>

---

<small>From [Before GPU Engineering](https://github.com/BEFORE-GPU-ENGINEERING/BEFORE-GPU-ENGINEERING) by Ranjan Yadav, used under the [MIT license](../LICENSE.txt).</small>
