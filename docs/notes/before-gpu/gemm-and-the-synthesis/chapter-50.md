---
title: "50. The traffic equation: throughput, latency, and concurrency"
---

<div class="bge">
<!-- Title -->
<h1><span class="hl-yellow" style="font-size: 0.9em;">The traffic equation: throughput, latency, and concurrency</span></h1>
<p class="bge-meta">Phase 3 · GEMM &amp; The Synthesis · Chapter 50 of 56 · notebook date 26 / aug / 26</p>
<span class="annotation">*the arithmetic of productivity*</span>

<!-- Section: Introduction -->
<p style="margin-top: calc(var(--grid-size) * 1.5);">
  There is a single tiny equation that explains almost everything about why a GPU is built the way it is, why a slow chip can out-produce a fast one, and why batching works. It is not a GPU formula at all, it comes from the study of queues and waiting lines, and it applies to a supermarket checkout as much as to a processor. Once you see it, the whole throughput-versus-latency story stops being a slogan and becomes arithmetic you can actually reason with.
</p>

<p class="text-blue">
  <span class="arrow">↳</span> The equation says that <strong class="hl-blue">throughput equals concurrency divided by latency.</strong>
</p>

<!-- Section: Unpacking the Three Words -->
</div>

## <span class="hl-yellow">Unpacking the Three Words</span>

<div class="bge">
<p>
  Unpack the three words, because they are easy to blur together. Latency is how long one single thing takes from start to finish: one task, one memory fetch, one request. Throughput is how many things you complete per unit of time, the total rate of work coming out the other end. And concurrency is how many things are in flight at the same moment, being worked on simultaneously. The equation ties them together: the rate you finish work equals how many things you have going at once, divided by how long each one takes.
</p>

<!-- Section: Fast vs. Productive -->
</div>

## <span class="hl-pink">Fast vs. Productive</span>

<div class="bge">
<p>
  The reason this is the key to everything is that it separates two ideas people constantly confuse: being fast and being productive. Latency is about being fast at one thing. Throughput is about being productive overall. And the equation shows they are not the same, because throughput has a second lever, concurrency, that has nothing to do with how fast any single thing is. <span class="hl-pink">You can raise your total output either by making each thing faster (shrinking latency), or by having more things in flight at once (raising concurrency).</span> Two completely different paths to more work done.
</p>

<!-- Section: The GPU Bet -->
</div>

## <span class="hl-red">The GPU Bet</span>

<div class="bge">
<p>
  Now watch how this instantly explains the GPU. Take one task in isolation. On a GPU, that single task actually runs slower than on a CPU, it has worse latency, because a GPU core is simpler and slower than a CPU core. So if throughput were only about latency, the GPU would just lose, period. But the GPU plays the other lever, hard. It runs an enormous number of tasks concurrently, thousands in flight at once. So plug it in: throughput equals a huge concurrency divided by a somewhat worse latency. <strong class="hl-red">The huge numerator crushes the slightly worse denominator</strong>, and the GPU's total output ends up far higher than the CPU's, even though every individual task on it was slower. The GPU didn't win by being fast. It won by having a massive amount of work in flight at the same time.
</p>

<!-- Section: Hiding Memory Latency -->
</div>

## <span class="hl-blue">Hiding Memory Latency</span>

<div class="bge">
<p class="handwritten-highlight text-blue" style="margin-top: var(--grid-size); margin-bottom: var(--grid-size);">
  So here's the question that reveals the deepest thing about the hardware: if throughput is concurrency over latency, and memory on a GPU has genuinely high latency, how does the GPU ever stay productive?
</p>
<p>
  It would fail, if concurrency stayed fixed. But the GPU's whole design is to crank concurrency up to match the latency. This is exactly what latency hiding is, seen through the equation. When a task stalls waiting on slow memory, that long latency would normally kill throughput, but the GPU just keeps thousands of other tasks in flight, so the concurrency term rises right along with the latency term, and they cancel. The chip stays productive not by making memory faster, but by having enough work outstanding that the high latency is divided away. So the equation tells you precisely how much concurrency you need: you need enough in-flight work to cover the latency, and if you have less than that, your throughput collapses because the denominator wins. This is the mathematical bones underneath all the intuition about keeping the chip fed.
</p>

<!-- Section: The Arithmetic of Batching -->
</div>

## <span class="hl-green">The Arithmetic of Batching</span>

<div class="bge">
<p>
  And it explains batching in one line. Why does grouping many requests together raise throughput? Because batching is nothing but raising concurrency on purpose. You gather more independent work and put it all in flight at once, pushing the numerator up. And here is the subtle part the equation makes obvious: <strong class="hl-green">batching raises throughput without requiring each individual request to get any faster.</strong> In fact each request might get slightly slower, waiting a moment to be grouped, so latency ticks up a little. But concurrency jumps a lot, so throughput still climbs. You traded a small increase in one thing's latency for a big increase in total output. The equation shows exactly why that trade is usually a win.
</p>

<!-- Section: The Saturation Ceiling -->
</div>

## <span class="hl-yellow">The Saturation Ceiling</span>

<div class="bge">
<p>
  But the same equation also warns you where it stops working, which is just as important. You cannot raise concurrency forever. At some point you hit a physical ceiling: the memory bus is fully saturated, or every compute unit is busy, and there is no room for more work to actually be in flight. Past that point, throwing more requests in doesn't raise real concurrency, it just makes them wait in line, which means their latency climbs while throughput stays flat. So the equation has a regime where raising concurrency buys throughput cheaply, and a regime past saturation where it only buys you worse latency for no gain. Knowing where that line sits, for your specific workload, is most of the practical skill in running these systems.
</p>

<!-- Takeaway -->
<div style="margin-top: calc(var(--grid-size) * 2); margin-bottom: var(--grid-size); border-left: 3px solid var(--ink-red); padding-left: 20px;">
  <h2 class="text-red" style="font-family: 'Patrick Hand', cursive; font-size: 32px; margin-bottom: var(--grid-size);">
    ↳ The General Takeaway
  </h2>

  <p>
    Throughput equals concurrency divided by latency: a law about queues that applies to any system that processes work, including every processor.
  </p>
  
  <p>
    It splits productivity from speed, showing you can win on total output either by making each task faster or by keeping more tasks in flight. <span class="hl-green">The GPU bets entirely on concurrency</span>, accepting worse per-task latency in exchange for thousands of tasks in flight. Batching is deliberately raising concurrency, which lifts throughput at a small cost in latency, right up until hardware saturation.
  </p>

  <p class="handwritten-highlight text-red" style="text-align: center; margin-top: var(--grid-size);">
    Fast and productive are different goals with different math...<br>
    <span class="hl-yellow" style="font-family: inherit; font-size: inherit;"><strong class="hl-red bold">the GPU wins by pushing the concurrency term as high as the hardware allows.</strong></span>
  </p>
</div>
</div>

---

<small>From [Before GPU Engineering](https://github.com/BEFORE-GPU-ENGINEERING/BEFORE-GPU-ENGINEERING) by Ranjan Yadav, used under the [MIT license](../LICENSE.txt).</small>
