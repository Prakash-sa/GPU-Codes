---
title: "39. Why higher occupancy can make performance worse."
---

<div class="bge">
<!-- Title -->
<h1><span class="hl-yellow" style="font-size: 0.9em;">Why higher occupancy can make performance worse.</span></h1>
<p class="bge-meta">Phase 3 · The Register Physics · Chapter 39 of 56 · notebook date 17 / jul / 26</p>
<span class="annotation">*guess and check*</span>

<!-- Section: Introduction -->
<p style="margin-top: calc(var(--grid-size) * 1.5);">
  Generating text from a large model is painfully slow for a reason that feels almost insulting. The model produces one word at a time, and to produce each single word it has to haul its entire enormous set of parameters out of memory. 
</p>

<p class="text-blue">
  <span class="arrow">↳</span> So the expensive compute cores do a tiny bit of work, then wait a long time for the next sweep of the model to arrive, then do a tiny bit more. The cores are mostly idle, starved, waiting on memory. 
</p>

<p>
  <strong class="hl-red">Speculative decoding</strong> is a trick that fills that idle time by turning generation into a game of guess and check, and it can make a model produce text several times faster while giving you the exact same words it would have produced anyway. How it pulls that off is genuinely clever.
</p>

<!-- Section: The Asymmetry -->
</div>

## <span class="hl-yellow">The Key Asymmetry</span>

<div class="bge">
<p>
  Start with why one word at a time is so wasteful. Each time the big model runs, it drags all its parameters from memory just to produce a single word, and the actual math involved is trivial compared to the cost of that memory sweep. So the model has enormous compute capacity sitting unused on every step, because the bottleneck is the data movement, not the arithmetic. 
</p>
<p>
  Here is the key realization hiding in that waste. Running the big model to check five words at once costs <strong class="hl-yellow">almost the same</strong> as running it to produce one word, because either way you pay for one sweep of the parameters from memory, and the extra math to handle five words instead of one is nearly free on those idle cores. 
</p>
<p>
  Producing is serial and slow. Checking many at once is nearly the same price as checking one. That asymmetry is the whole opening.
</p>

<!-- Section: The Draft Model -->
</div>

## <span class="hl-pink">The Cheap Guess</span>

<div class="bge">
<p>
  So the plan. What if you had a second model, a small fast one, that could cheaply guess the next several words? This little model is far less capable, but it's also far smaller, so it can generate words quickly because its parameter sweep from memory is tiny. 
</p>
<p>
  You let this small draft model race ahead and blurt out a guess for, say, the next five words. These are just guesses, some right, some wrong, but they were cheap to produce.
</p>

<!-- Section: The Verification -->
</div>

## <span class="hl-green">The Bulk Verification</span>

<div class="bge">
<p>
  Now the beautiful part. You take those five guessed words and hand them all to the big model at once, and ask it to check them in a single pass. And remember, checking five words costs the big model almost the same as producing one, because it's the same one memory sweep. 
</p>
<p>
  In that single pass, the big model effectively says, for each guessed position, whether that is the word it would have chosen itself. It works through the guesses and accepts them one by one as long as they match what it would have said, and <strong class="hl-green">stops at the first guess it disagrees with.</strong> Everything up to that first disagreement is confirmed correct, exactly what the big model would have produced on its own, and at the point of disagreement the big model supplies its own correct word instead.
</p>

<!-- Section: Why it's brilliant -->
</div>

## <span class="hl-blue">Multiple Words for the Price of One</span>

<div class="bge">
<p>
  So think about what just happened. If the little model guessed five words and the big model agreed with the first three, then in a single expensive pass—the price of normally producing just one word—you produced four correct words: the three accepted guesses plus the one correction. 
</p>
<p>
  You got several words for the price of one memory sweep. The small model did the cheap speculative labor, and the big model, instead of slowly birthing one word, ratified a whole batch in the time it would have taken to make one. The idle compute that was going to waste is now spent verifying guesses, which is nearly free, and it buys you multiple words per expensive step.
</p>

<!-- Section: Does it cheat? -->
</div>

## <span class="hl-pink">Speed Without Compromise</span>

<div class="bge">
<p>
  Now the objection that should be nagging you, because it sounds like it might cheat. If a small dumb model is doing the guessing, aren't you degrading the output? Isn't this a quality shortcut where you get worse words in exchange for speed?
</p>
<p>
  No, and this is the elegant heart of it. The small model never gets the final say on anything. It only proposes. Every single word that survives is one the big model explicitly confirmed it would have chosen itself, and any guess the big model disagrees with is thrown away and replaced with the big model's own choice. 
</p>
<p>
  So the output is word for word identical to what the big model would have produced alone, just arrived at faster. The little model is only there to guess, the big model is the sole authority on what actually gets written. Speed went up, quality did not move a hair. That's what makes the trick honest rather than a shortcut.
</p>

<!-- Section: The Gamble -->
</div>

## <span class="hl-red">The Gamble</span>

<div class="bge">
<p>
  Of course there is a cost, and it lives in the guessing. The whole scheme only pays off when the little model guesses well, because every accepted guess is a word you got nearly for free. 
</p>
<p>
  If the draft model is decent and its guesses are often right, you accept long runs and fly. But if the draft model is bad, or the text is hard to predict, most guesses get rejected, and then you wasted effort running the draft model and checking guesses that got thrown away, and you crawl back toward the original one-word-at-a-time speed, now with extra overhead on top. So the art is in choosing a draft model that's small enough to be cheap but smart enough to be right often, and the actual speedup you get depends entirely on how often its guesses are accepted. Easy, predictable text flies. Hard, surprising text barely benefits.
</p>


<!-- Takeaway -->
<div style="margin-top: calc(var(--grid-size) * 2); margin-bottom: var(--grid-size); border-left: 3px solid var(--ink-red); padding-left: 20px;">
  <h2 class="text-red" style="font-family: 'Patrick Hand', cursive; font-size: 32px; margin-bottom: var(--grid-size);">
    ↳ The General Takeaway
  </h2>

  <p>
    So the whole picture. A big model generating one word per memory sweep leaves its compute cores mostly idle, because the cost is in moving the parameters, not the math, and checking many words costs nearly the same as checking one. 
  </p>
  
  <p>
    Speculative decoding exploits this by letting a small fast model guess several words ahead, then having the big model verify all of them in a single pass, accepting the run of correct guesses and fixing the first wrong one. You get multiple words for the price of one expensive step, and because the big model confirms every surviving word, the output is exactly what it would have produced alone. 
  </p>

  <p class="handwritten-highlight text-red" style="text-align: center; margin-top: var(--grid-size);">
    When your expensive resource is idle because it's waiting on something slow,<br>
    <span class="hl-yellow" style="font-family: inherit; font-size: inherit;"><strong class="hl-red bold">you can often fill that idle time with cheap speculative work and verify it in bulk,</strong></span><br>
    getting real speed from capacity that was otherwise going to waste.
  </p>
</div>
</div>

---

<small>From [Before GPU Engineering](https://github.com/BEFORE-GPU-ENGINEERING/BEFORE-GPU-ENGINEERING) by Ranjan Yadav, used under the [MIT license](../LICENSE.txt).</small>
