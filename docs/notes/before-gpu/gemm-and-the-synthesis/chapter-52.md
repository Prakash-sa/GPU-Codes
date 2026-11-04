---
title: "Chapter 52: Measure, don't guess: the top-down profiling discipline"
---

<div class="bge">
<!-- Title -->
<h1><span class="hl-yellow" style="font-size: 0.9em;">Measure, don't guess: the top-down profiling discipline</span></h1>
<p class="bge-meta">Part 14: GEMM &amp; The Synthesis · Chapter 52 of 56 · notebook date 28 / aug / 26</p>
<span class="annotation">*the empirical mindset*</span>

<!-- Section: Introduction -->
<p style="margin-top: calc(var(--grid-size) * 1.5);">
  There is a story that plays out constantly in GPU programming, and it goes like this: a kernel is slow, you look at the code, you spot something that looks inefficient, you spend a week rewriting it beautifully, and the whole thing gets three percent faster. The week was wasted, not because the rewrite was wrong, but because it fixed something that was never the problem. This is the single most important discipline in making GPU code fast, and it is not a technique at all. It is a refusal to guess. You measure first, find what is actually holding you back, fix only that, and measure again.
</p>

<p class="text-blue">
  <span class="arrow">↳</span> The core rule is <strong class="hl-blue">never touch code until empirical measurement identifies the active bottleneck.</strong>
</p>

<!-- Section: The Multi-Ceiling Trap -->
</div>

## <span class="hl-yellow">The Multi-Ceiling Trap</span>

<div class="bge">
<p>
  Start with why guessing fails so reliably here. A GPU has many separate things that can be the bottleneck: the compute units, the memory bandwidth, the register budget, the fast scratchpad, the rate the scheduler can issue instructions, even the time the main computer spends handing work to the GPU. And at any moment, your kernel is limited by only one or two of these, not all of them. So when you look at the code and your gut says the memory access looks bad, you might be completely right that it looks bad, and completely wrong that it matters, because the kernel was actually limited by something else entirely. Fixing a resource that was not the constraint does nothing, because the real constraint is still sitting there untouched. Intuition cannot tell you which of the several possible limits you are hitting. Only measurement can.
</p>

<!-- Section: The Top-Down Order -->
</div>

## <span class="hl-pink">The Top-Down Order</span>

<div class="bge">
<p>
  So the first rule is to measure before you change anything, and the second, subtler rule is to measure at the right level first. And here is the mistake almost everyone makes: they dive straight into examining the guts of one kernel, studying its every internal detail, before checking whether that kernel even matters. <span class="hl-pink">So the correct order is top down.</span> First look at the whole timeline of your program, the big picture of where the time actually goes. Is the GPU even busy, or is it sitting idle most of the time waiting for the main computer to feed it work? Is time being lost copying data back and forth? Is one phase of the program dwarfing all the others? Only once you have found the part that actually dominates the runtime do you zoom in and study its internals.
</p>
<p>
  Why does this order matter so much? Because a kernel can be internally flawless and still be irrelevant. Imagine you perfect a kernel until it runs at the absolute peak the hardware allows. But the GPU spends most of its time idle between launches, waiting for the slow main computer to prepare the next batch of work. Your perfect kernel runs, then the GPU sits there doing nothing, then it runs again. You made the engine flawless while the car had no wheels. The fix was never inside the kernel at all, it was in how the work was being fed to the GPU. And you would never have known that by staring at the kernel's internals, only by looking at the whole timeline first.
</p>

<!-- Section: Symptoms vs Diagnoses -->
</div>

## <span class="hl-blue">Symptoms vs. Diagnoses</span>

<div class="bge">
<p class="handwritten-highlight text-blue" style="margin-top: var(--grid-size); margin-bottom: var(--grid-size);">
  So here's a question that follows naturally: if the timeline shows the GPU idle a lot, does that tell you the main computer is the problem?
</p>
<p>
  And the answer is a careful no, and it teaches the core discipline. An idle GPU is a symptom, not a diagnosis. It could be the main computer being slow to send work, or it could be data being copied back and forth, or one piece of work waiting on another to finish, or simply not enough work to do. The idle time tells you where to look, not what is wrong. This is the whole mindset in miniature: a measurement points you at the neighborhood, and you still have to investigate to find the actual culprit. The number is evidence, not a verdict.
</p>

<!-- Section: Single-Variable Experiments -->
</div>

## <span class="hl-green">Single-Variable Experiments</span>

<div class="bge">
<p>
  Now the discipline of the fix itself, which is just as important. Once you have found the real bottleneck and formed an idea of the cause, you change one thing. Just one. Because GPU performance is tangled, and the resources all draw from shared, limited budgets, so changing several things at once makes it impossible to know which one helped. If you rewrite the memory access, adjust the work group size, add scratchpad staging, and change the math all at once, and it gets twenty percent faster, you have learned nothing: <strong class="hl-green">you cannot tell which change did the work or whether two of them fought each other.</strong> One change, one hypothesis, then measure. It is controlled experiment, the same reason a scientist changes one variable at a time.
</p>

<!-- Section: Post-Fix Verification -->
</div>

## <span class="hl-red">Post-Fix Verification</span>

<div class="bge">
<p>
  And then the step people skip most of all: measure again after the fix, and check the actual thing you were trying to improve. It is not enough that the whole program got a little faster, you have to confirm that the specific bottleneck you targeted actually moved. Because it is easy to apply a fix, see the runtime dip slightly for some unrelated reason, and declare victory while the real problem is untouched. Verify the metric you aimed at changed, not just that the clock moved.
</p>

<!-- Section: Bottleneck Migration -->
</div>

## <span class="hl-yellow">Bottleneck Migration</span>

<div class="bge">
<p>
  And this is where the deepest and most counterintuitive truth lives. <strong class="hl-red">When you fix the real bottleneck, a new one appears</strong>, and this looks like failure but is exactly success. Say your kernel was starved for memory bandwidth, and you fix it, and now the profile says the compute units are the limit. A beginner sees the new limit and thinks the fix failed. But it didn't: the kernel got faster, and the bottleneck simply moved to the next ceiling. Optimization is precisely this: knocking down the current limit so you hit the next one, over and over, each one a little higher. There is no state with no bottleneck, there is only which ceiling you are currently against.
</p>
<p>
  And the flip side, the trap, is that a fix can make things worse in a way that is not obvious. Because all these resources share the same limited budget, relieving pressure on one can pile it onto another. You can force more parallel work onto the chip and accidentally starve each thread of the resources it needed, ending up slower than before, even though the metric you were chasing improved. This is why measuring after is not optional: the change that looks obviously good on paper can quietly cost you elsewhere, and only measuring the actual runtime tells you the truth. The metric is a clue, but the wall-clock time is the judge.
</p>

<!-- Takeaway -->
<div style="margin-top: calc(var(--grid-size) * 2); margin-bottom: var(--grid-size); border-left: 3px solid var(--ink-red); padding-left: 20px;">
  <h2 class="text-red" style="font-family: 'Patrick Hand', cursive; font-size: 32px; margin-bottom: var(--grid-size);">
    ↳ The General Takeaway
  </h2>

  <p>
    Making GPU code fast is not mainly about knowing clever techniques, it is about the discipline of not guessing.
  </p>
  
  <p>
    <span class="hl-green">Measure top-down first</span> (the whole timeline), because a perfect kernel is worthless if the GPU sits idle between launches. Change exactly one variable at a time, and verify that the target metric moved.
  </p>

  <p class="handwritten-highlight text-red" style="text-align: center; margin-top: var(--grid-size);">
    Optimization is knocking down limits to reach higher ceilings...<br>
    <span class="hl-yellow" style="font-family: inherit; font-size: inherit;"><strong class="hl-red bold">refuse to touch code until measurement tells you where to touch it.</strong></span>
  </p>
</div>
</div>

---

<small>From [Before GPU Engineering](https://github.com/BEFORE-GPU-ENGINEERING/BEFORE-GPU-ENGINEERING) by Ranjan Yadav, used under the [MIT license](../LICENSE.txt).</small>
