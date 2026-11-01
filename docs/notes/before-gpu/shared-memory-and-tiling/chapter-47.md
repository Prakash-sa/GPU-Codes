---
title: "47. Compare-and-swap: The universal atomic primitive."
---

<div class="bge">
<!-- Title -->
<h1><span class="hl-yellow" style="font-size: 0.9em;">Compare-and-swap: The universal atomic primitive.</span></h1>
<p class="bge-meta">Phase 3 · Shared Memory &amp; Tiling · Chapter 47 of 56 · notebook date 23 / aug / 26</p>

<p style="margin-top: calc(var(--grid-size) * 1.5);">
  The hardware gives you a handful of built-in atomic operations, safe ways for many threads to update the same spot without stepping on each other, like atomic addition. But what happens when you need something the hardware doesn't offer, some custom update that isn't just an add. It turns out there is one primitive underneath all of them, a single operation from which you can build any atomic update you can imagine. It is called compare-and-swap, and understanding it is understanding how threads coordinate at the deepest level, the foundation of nearly all lock-free code.
</p>
</div>

## <span class="hl-yellow">The Stale Data Problem</span>

<div class="bge">
<p>Start with the problem it solves. When many threads want to update a shared value in some non-trivial way, you have the classic danger. A thread reads the current value, computes a new one based on it, and writes it back. But between its read and its write, another thread might sneak in, read the same old value, and now they both compute from stale data and one update gets lost. The built-in atomic add avoids this for the specific case of adding, by doing the whole read-add-write as one indivisible step. But addition is only one kind of update. What about all the others, taking a maximum with some complicated condition, updating a value only if it passes some test, modifying a packed structure. The hardware can't provide a dedicated atomic for every possible operation.</p>
</div>

## <span class="hl-pink">The Universal Primitive</span>

<div class="bge">
<p>So what is the one general tool. Compare-and-swap is a single indivisible operation that does exactly this. You hand it three things, a memory location, the value you expect to find there, and the new value you want to put there. Atomically, in one unbreakable step, it checks whether the location still holds the value you expected. If it does, it writes your new value and reports success. If it doesn't, meaning someone else changed it since you looked, it writes nothing and reports failure, handing you back the value it actually found. That's the whole primitive, check if it is what I expect, and only if so, swap in my new value, all as one atomic action that no other thread can interrupt.</p>
</div>

## <span class="hl-blue">The Try-and-Retry Loop</span>

<div class="bge">
<p>Now the beautiful part, how this one operation lets you build any atomic update you want. The pattern is a loop. First, you read the current value of the shared location. Second, you compute your desired new value from it, doing whatever arbitrary calculation you like, no restrictions, because this part is just local work on your own copy. Third, you attempt a compare-and-swap, saying, put my new value here, but only if the location still holds the value I originally read. If it succeeds, great, your update landed cleanly and you are done. If it fails, it means another thread changed the value in the meantime, so your computation was based on stale data, and you simply loop back and try again with the new current value it handed you. You keep retrying until a swap succeeds.</p>
<p>So consider why this loop is correct. The only way your swap succeeds is if the value never changed between your read and your swap, which means your computation was based on current data and your update is valid. If anyone interfered, the compare fails, and you throw away your stale attempt and redo it fresh. So you can never accidentally overwrite someone else's update with a computation based on old data. The compare is the guard that guarantees you only commit when your view of the world was still accurate. This read-compute-tryswap-retry loop can implement literally any atomic operation, which is why compare-and-swap is called universal, every other atomic is really just a specialized, hardware-accelerated version of this pattern.</p>
</div>

## <span class="hl-green">The Cost of Retrying</span>

<div class="bge">
<p class="handwritten-highlight text-red" style="margin-top: var(--grid-size); margin-bottom: var(--grid-size);">
  Now the objection that reveals the cost. If failed attempts just retry, doesn't that mean threads can waste tons of work spinning in the loop, redoing their computation over and over. Yes, and this is the real price. When many threads hammer the same location at once, only one wins each round, and all the others fail their compare and have to loop back and try again. Under heavy contention, threads can retry many times before succeeding, and all that recomputation is wasted effort. So compare-and-swap loops are wonderful when contention is light, threads rarely collide and almost always succeed on the first try, but they degrade badly when a crowd all fights over one spot, everyone spinning and retrying while one at a time squeezes through. It is the same lesson as every shared-location update, the coordination is cheap only when the threads are mostly not actually colliding.
</p>
<p>And this is exactly why, on a gpu with its thousands of threads, you avoid funneling them all onto one location whenever you can. The retry storm from a massive crowd contending on a single spot can cripple performance. So the fast pattern is to let threads work on their own private spots and only combine at the end, or to use the built-in specialized atomics when they fit, and to reach for the compare-and-swap loop mainly for the custom updates that nothing else can express, ideally where the contention is spread out rather than concentrated. The tool is universal and powerful, but its cost scales with how much the threads crowd the same place.</p>

<!-- Takeaway -->
<div style="margin-top: calc(var(--grid-size) * 2); margin-bottom: var(--grid-size); border-left: 3px solid var(--ink-red); padding-left: 20px;">
  <h2 class="text-red" style="font-family: 'Patrick Hand', cursive; font-size: 32px; margin-bottom: var(--grid-size);">
    ↳ The General Takeaway
  </h2>
  <p>So the whole picture. The hardware offers a few fixed atomic operations, but compare-and-swap is the universal one beneath them all, a single indivisible check-and-conditionally-write. From it you build any atomic update through a loop, read the value, compute your new one however you like, then try to swap it in only if nothing changed, and retry if something did. It is correct because the swap only commits when your view was still current, so you never clobber another thread's work with stale data. Its cost is the retrying under contention, which is cheap when collisions are rare and punishing when a crowd fights over one spot. The deep idea is that all coordination between threads ultimately rests on this one tiny atomic promise, I will change this only if it is still what I last saw, and everything else, every lock, every counter, every shared update, is built on top of that single honest guarantee.</p>
</div>
</div>

---

<small>From [Before GPU Engineering](https://github.com/BEFORE-GPU-ENGINEERING/BEFORE-GPU-ENGINEERING) by Ranjan Yadav, used under the [MIT license](../LICENSE.txt).</small>
