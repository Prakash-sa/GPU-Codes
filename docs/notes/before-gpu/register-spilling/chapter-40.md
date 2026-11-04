---
title: "Chapter 40: What is register spilling? The cliff between 1 ms and 50 ms."
---

<div class="bge">
<!-- Title -->
<h1><span class="hl-yellow" style="font-size: 0.9em;">What is register spilling? The cliff between 1 ms and 50 ms.</span></h1>
<p class="bge-meta">Part 11: Register Spilling · Chapter 40 of 56 · notebook date 17 / jul / 26</p>
<span class="annotation">*the rolling batch*</span>

<!-- Section: Introduction -->
<p style="margin-top: calc(var(--grid-size) * 1.5);">
  Serving a language model to many users at once hides a scheduling problem that, solved badly, wastes most of your expensive hardware, and solved well, multiplies your throughput several times over. 
</p>
<p>
  The naive way of grouping users together feels reasonable and quietly cripples the GPU. The fix, <strong class="hl-red">continuous batching</strong>, is one of those ideas that seems obvious in hindsight but took real cleverness to see. Understanding it explains why modern serving systems can pack so many more users onto the same hardware than a straightforward implementation ever could.
</p>

<!-- Section: Why we batch -->
</div>

## <span class="hl-yellow">The Shared Sweep</span>

<div class="bge">
<p>
  Start with why you group users at all. Generating text is bottlenecked on hauling the model's parameters out of memory, and here is the saving grace: if you process many users' requests together, you sweep those parameters from memory once and use them for everybody in the group at the same time. 
</p>
<p>
  <span class="hl-yellow">One memory sweep, many users served.</span> So batching users together is the single most important thing you do to make serving efficient, because it spreads the expensive memory cost across many people instead of paying it fresh for each one. A GPU serving one user at a time is a GPU mostly idle. A GPU serving a big batch is a GPU earning its keep.
</p>

<!-- Section: The flaw -->
</div>

## <span class="hl-pink">The Static Batch Trap</span>

<div class="bge">
<p>
  So the obvious approach, called static batching, is to gather a batch of requests, run them all together until they are all done, then gather the next batch. 
</p>
<p>
  And this is where it quietly falls apart, because of one simple fact: different requests want wildly different amounts of text. One user asks for a one-word answer, another wants a thousand-word essay. In a static batch, they all start together and the batch is not finished until the very last request finishes. So the user who wanted one word sits there, already done, occupying a slot in the batch, while the whole group waits for the essay to complete. That finished request is just holding a seat, contributing nothing, and its share of the GPU is being wasted for the entire time it waits.
</p>

<p>
  Think about how bad that gets. If you batch sixteen requests and fifteen of them finish quickly but one keeps generating, those fifteen slots sit there empty, doing nothing, until the long one is done. Your batch that started at sixteen effectively shrinks to one, and the GPU spends most of its time processing a nearly empty batch, sweeping all those parameters from memory to serve a single straggler. 
</p>
<p>
  You paid for a full batch and got the efficiency of almost no batch. Worse, brand new users who just arrived have to wait for this whole batch to finish before they can even start, because the batch is a locked group. The waste is enormous, and it comes purely from treating the batch as a rigid all-start, all-finish block.
</p>

<!-- Section: Continuous Batching -->
</div>

## <span class="hl-green">The Roster System</span>

<div class="bge">
<p>
  So the question that unlocks the fix. Why do we treat the batch as a locked group at all? Why must a finished request keep holding its seat, and why must a new request wait for the whole batch to clear? What if the batch could change its membership on the fly?
</p>

<p>
  That is <strong class="hl-green">continuous batching</strong>. The key insight is that generation happens one token at a time for everyone in the batch, in lockstep steps, so there is a natural moment between every token where you can rethink the batch. 
</p>
<p>
  So instead of locking the group, you re-evaluate membership at every single step. The moment a request finishes, it leaves the batch immediately and frees its slot, and a waiting new request is slotted in right there to take its place, joining the ongoing batch mid-flight. The batch is no longer a rigid block, it's a rolling roster that continuously evicts the finished and admits the waiting, staying full at all times. This is scheduling at the level of each token rather than each whole batch, and it changes everything.
</p>

<!-- Section: The Payoff -->
</div>

## <span class="hl-blue">The Payoff</span>

<div class="bge">
<p>
  Now the payoff. Because finished requests leave instantly instead of holding dead seats, and new ones join instantly instead of waiting for a clean batch boundary, the GPU is always working on a full batch of active requests. No empty slots, no stragglers holding the group hostage, no new users waiting at the gate. 
</p>
<p>
  The same hardware now serves far more users in the same time, often several times the throughput of the static approach, purely because it stopped wasting slots on requests that were done or absent. You did not make the model faster. You made the machine stop idling.
</p>

<!-- Section: Bookkeeping -->
</div>

## <span class="hl-red">The Bookkeeping Tension</span>

<div class="bge">
<p>
  Now the objection that keeps this honest. If requests are constantly joining and leaving the batch every token, isn't the bookkeeping a nightmare, and doesn't jamming new requests in hurt the users already running? 
</p>
<p>
  It does add real complexity. The scheduler has to manage a roster changing every step, and each active request needs its own growing working memory held on the GPU, so the system needs a flexible way to hand out and reclaim little chunks of memory as requests come and go, rather than reserving fixed large blocks. 
</p>
<p>
  And there is a genuine tension: when a fresh request joins, it first has to digest its whole prompt, which is a heavier one-time step that can briefly delay the smooth token-by-token generation of the users already in flight, causing a small hitch in their output. So the system has to balance admitting new users quickly against keeping existing users flowing smoothly. It's a real scheduling art, not a free lunch, but the throughput gain is so large that it's overwhelmingly worth it.
</p>

<!-- Takeaway -->
<div style="margin-top: calc(var(--grid-size) * 2); margin-bottom: var(--grid-size); border-left: 3px solid var(--ink-red); padding-left: 20px;">
  <h2 class="text-red" style="font-family: 'Patrick Hand', cursive; font-size: 32px; margin-bottom: var(--grid-size);">
    ↳ The General Takeaway
  </h2>

  <p>
    So the whole picture. You batch users together because it shares the expensive memory sweep of the model's parameters across many people at once. The naive static batch locks everyone together from start to finish, so quick requests sit uselessly holding seats while one long request drags the whole group, collapsing your effective batch size and idling the GPU. 
  </p>
  
  <p>
    Continuous batching fixes it by reworking the batch every single token, instantly evicting finished requests and admitting waiting ones, so the batch stays full and the hardware never idles on empty slots. It costs real scheduling and memory-management complexity and a bit of tension between new and existing users, but it multiplies how many people one GPU can serve. 
  </p>

  <p class="handwritten-highlight text-red" style="text-align: center; margin-top: var(--grid-size);">
    And the lesson underneath is a general one about expensive shared resources:<br>
    <span class="hl-yellow" style="font-family: inherit; font-size: inherit;"><strong class="hl-red bold">The enemy is idle capacity.</strong></span><br>
    Often the biggest wins come not from doing each task faster but from never letting the machine sit half empty.
  </p>
</div>
</div>

---

<small>From [Before GPU Engineering](https://github.com/BEFORE-GPU-ENGINEERING/BEFORE-GPU-ENGINEERING) by Ranjan Yadav, used under the [MIT license](../LICENSE.txt).</small>
