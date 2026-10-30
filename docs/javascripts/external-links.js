// Open links to other sites (LeetGPU questions, specs, videos) in a new tab.
// Runs on every page, including after Material's instant navigation swaps the content.
document$.subscribe(() => {
  document.querySelectorAll(".md-content a[href^='http']").forEach((a) => {
    if (a.hostname !== window.location.hostname) {
      a.target = "_blank";
      a.rel = "noopener noreferrer";
    }
  });
});
