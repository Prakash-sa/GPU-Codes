---
title: "49. Wave quantization: the hidden cost of the last wave"
---

<div class="bge">
<!-- Title -->
<h1><span class="hl-yellow" style="font-size: 0.9em;">Wave quantization: the hidden cost of the last wave</span></h1>
<p class="bge-meta">Phase 3 · GEMM &amp; The Synthesis · Chapter 49 of 56 · notebook date 25 / aug / 26</p>
<span class="annotation">*the mechanics of batch scheduling*</span>

<!-- Section: Introduction -->
<p style="margin-top: calc(var(--grid-size) * 1.5);">
  Here is a result that makes no sense until you understand one thing about how a GPU schedules work: you make your problem slightly bigger, add just a little more work, and it runs in exactly the same time as before. Or worse, a tiny increase in size suddenly makes it noticeably slower, a jump far bigger than the extra work should cost. The culprit is something called wave quantization, and it is one of the most common reasons a GPU runs at half its potential while every other number looks fine. Understanding it explains a whole class of mysterious performance cliffs.
</p>

<p class="text-blue">
  <span class="arrow">↳</span> The secret is that <strong class="hl-blue">the hardware chews through work in fixed-size batches called waves.</strong>
</p>

<!-- Section: Batched Execution -->
</div>

## <span class="hl-yellow">The Batched Execution</span>

<div class="bge">
<p>
  Start with how a GPU actually runs a big pile of work. You divide your work into many independent blocks, and the chip has a fixed number of physical slots to run them, one set of slots per processing unit, and a fixed number of processing units. So the chip cannot run all your blocks at once. It runs as many as fit in its slots simultaneously, and that batch of simultaneously-running blocks is called a wave. When a wave finishes, the next batch of blocks flows in to fill the freed slots, and so on, wave after wave, until all the blocks are done. The chip chews through your work one wave at a time.
</p>

<!-- Section: The Ceiling Function Trap -->
</div>

## <span class="hl-pink">The Ceiling Function Trap</span>

<div class="bge">
<p>
  Now the key fact. Every wave takes essentially the same amount of time, because each wave is the chip running a full set of blocks in parallel, and they all finish together roughly when the slowest one does. So the total time is basically the number of waves times the time for one wave. <span class="hl-pink">And here is the trap hiding in that: the number of waves is your block count divided by how many fit at once, rounded up.</span> Rounded up, because you cannot have a fraction of a wave. If even one block spills past a clean boundary, it needs a whole extra wave to run in.
</p>

<!-- Section: The One-Block Disaster -->
</div>

## <span class="hl-red">The One-Block Disaster</span>

<div class="bge">
<p>
  So walk through the disaster. Suppose the chip can run a certain number of blocks at once, call it a full wave's worth. If your work happens to be exactly two waves' worth of blocks, wonderful: both waves run completely full, every slot busy, the chip fully used the whole time. But now add just one more block. That single extra block cannot squeeze into the two full waves, so it needs a third wave all to itself. And that third wave runs with just one block active while every other slot on the entire chip sits idle. <strong class="hl-red">You added one block's worth of work, but you paid for a whole extra wave of time</strong>, during which the chip was almost entirely empty. Your one percent more work cost you fifty percent more time.
</p>

<!-- Section: The Flat-Step Cliff -->
</div>

## <span class="hl-blue">The Flat-Step Cliff</span>

<div class="bge">
<p class="handwritten-highlight text-blue" style="margin-top: var(--grid-size); margin-bottom: var(--grid-size);">
  So this is the question it answers: why did making my problem slightly bigger not slow it down, or slow it down enormously?
</p>
<p>
  Because what matters is not the exact amount of work, it's how the blocks divide into waves. As long as you stay within the same number of waves, adding work is nearly free, you're just filling slots that were going to run anyway. But the moment you cross a wave boundary, you pay for an entire new wave even if you only spilled into it by a single block. Performance goes up in flat steps, not a smooth line, and the steps are the wave boundaries.
</p>

<!-- Section: The Ragged Tail -->
</div>

## <span class="hl-green">The Ragged Tail</span>

<div class="bge">
<p>
  And the ugliest case is when the last wave is nearly empty, which people call the tail. If your work is one and a bit waves, the first wave runs full and fast, and then the chip spends a second wave running a tiny handful of blocks while almost every processing unit sits idle. <strong class="hl-green">You can easily end up in a situation where half your total runtime is spent on a final wave that is using a small fraction of the chip.</strong> The chip looks busy at the start and then limps to the finish on a nearly-empty last wave. And none of the usual health metrics necessarily scream about this: the work that did run ran fine, there just wasn't enough of it to fill that last wave.
</p>

<!-- Section: Shaping the Work -->
</div>

## <span class="hl-pink">Shaping the Work</span>

<div class="bge">
<p>
  So how do you fix it, and this is where it becomes actionable. The goal is to make your number of blocks a clean multiple of how many fit in a wave, so every wave runs full and there is no wasteful tail. You can often do this by adjusting how you carve up the work, choosing a block size or a tiling shape so that the total block count lands neatly on a wave boundary instead of spilling one over. Sometimes making each block do slightly more or less work, so the block count changes, is enough to eliminate an almost-empty tail wave and reclaim that wasted time. The win comes not from doing less work but from arranging the work so the chip never runs a mostly-idle wave.
</p>

<!-- Section: Why Scale Drowns the Tail -->
</div>

## <span class="hl-yellow">Why Scale Drowns the Tail</span>

<div class="bge">
<p>
  And this is exactly why bigger problems often run more efficiently than small ones, a fact that puzzles people. A small problem might be just a handful of blocks, maybe not even filling a single wave, so a big fraction of the chip is idle the whole time and any tail is a large slice of the total. A huge problem is dozens or hundreds of waves, so even if the last one is ragged and half-empty, it's just one imperfect wave among many, a tiny fraction of the total time. The waste is the same size in absolute terms, one partial wave, but it is a rounding error against a large job and a catastrophe against a small one. Scale doesn't remove the tail, it just drowns it.
</p>

<!-- Takeaway -->
<div style="margin-top: calc(var(--grid-size) * 2); margin-bottom: var(--grid-size); border-left: 3px solid var(--ink-red); padding-left: 20px;">
  <h2 class="text-red" style="font-family: 'Patrick Hand', cursive; font-size: 32px; margin-bottom: var(--grid-size);">
    ↳ The General Takeaway
  </h2>

  <p>
    A GPU runs your blocks in waves, as many as fit at once, and the total time is roughly the number of waves times the time per wave, with the wave count always rounded up.
  </p>
  
  <p>
    <span class="hl-green">Work that spills even slightly past a wave boundary pays for a whole extra, mostly-empty wave</span>, which is why performance climbs in flat steps and why a nearly-empty final wave can waste a huge share of a small job's time. You fix it by shaping your work so the block count lands cleanly on wave boundaries and no wave runs half-empty.
  </p>

  <p class="handwritten-highlight text-red" style="text-align: center; margin-top: var(--grid-size);">
    On hardware that runs things in fixed-size batches...<br>
    <span class="hl-yellow" style="font-family: inherit; font-size: inherit;"><strong class="hl-red bold">the enemy is the ragged remainder.</strong></span>
  </p>
</div>
</div>

---

<small>From [Before GPU Engineering](https://github.com/BEFORE-GPU-ENGINEERING/BEFORE-GPU-ENGINEERING) by Ranjan Yadav, used under the [MIT license](../LICENSE.txt).</small>
