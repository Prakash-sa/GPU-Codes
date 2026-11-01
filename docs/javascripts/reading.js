// Reading aids on every page: a progress bar under the header, a tick in the sidebar for
// pages already opened, ← / → for previous / next page, and a "continue" link on the
// Before GPU Engineering contents page. Read state stays in this browser (localStorage).
(function () {
  const KEY = "gpu-codes:read";
  const load = () => {
    try { return JSON.parse(localStorage.getItem(KEY)) || {}; } catch { return {}; }
  };
  const save = (read) => {
    try { localStorage.setItem(KEY, JSON.stringify(read)); } catch { /* private mode */ }
  };
  const pathOf = (href) => new URL(href, location.href).pathname.replace(/index\.html$/, "");

  let bar;
  function progress() {
    const max = document.documentElement.scrollHeight - innerHeight;
    if (bar) bar.style.width = max > 0 ? `${Math.min(100, (scrollY / max) * 100)}%` : "0";
  }
  addEventListener("scroll", progress, { passive: true });
  addEventListener("resize", progress);

  addEventListener("keydown", (e) => {
    const t = e.target;
    if (e.metaKey || e.ctrlKey || e.altKey || e.shiftKey) return;
    if (/^(INPUT|TEXTAREA|SELECT)$/.test(t.tagName) || t.isContentEditable) return;
    if (t.closest && t.closest("pre, .md-typeset__scrollwrap, .highlight")) return;   // let wide code/tables scroll
    const dir = { ArrowLeft: "prev", ArrowRight: "next" }[e.key];
    const link = dir && document.querySelector(`.md-footer__link--${dir}`);
    if (link) link.click();
  });

  // Runs on first load and after every instant-navigation page swap.
  document$.subscribe(() => {
    const header = document.querySelector(".md-header");
    if (header && !header.querySelector(".reading-progress")) {
      bar = header.appendChild(document.createElement("div"));
      bar.className = "reading-progress";
    }
    progress();

    const read = load();
    const here = pathOf(location.href);
    const title = document.querySelector(".md-content h1")?.textContent.trim() || document.title;
    read[here] = { title, at: Date.now() };
    save(read);

    document.querySelectorAll(".md-sidebar--primary a.md-nav__link[href]").forEach((a) => {
      a.classList.toggle("is-read", pathOf(a.href) in read);
    });

    const box = document.getElementById("bge-continue");
    if (box) {
      const base = pathOf(".");                      // notes/before-gpu/
      const chapters = Object.entries(read).filter(([p]) => p.startsWith(base) && p !== base);
      if (chapters.length) {
        const [path, last] = chapters.sort((a, b) => b[1].at - a[1].at)[0];
        const total = document.querySelectorAll('.md-content a[href*="chapter-"]').length;
        box.replaceChildren();
        const p = box.appendChild(document.createElement("p"));
        p.append("↳ Continue reading: ");
        const a = p.appendChild(document.createElement("a"));
        a.href = path;
        a.textContent = last.title;
        p.append(` · ${chapters.length} / ${total} chapters opened`);
      }
    }
  });
})();
