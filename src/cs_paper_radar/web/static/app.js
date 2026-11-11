// Small progressive enhancements; every page works without JavaScript.
(function () {
  const root = document.documentElement;

  // Theme toggle (remembers the choice; default follows the OS).
  const toggle = document.getElementById("theme-toggle");
  if (toggle) {
    toggle.addEventListener("click", () => {
      const dark = root.dataset.theme
        ? root.dataset.theme === "dark"
        : window.matchMedia("(prefers-color-scheme: dark)").matches;
      root.dataset.theme = dark ? "light" : "dark";
      localStorage.setItem("theme", root.dataset.theme);
    });
  }

  // "/" focuses the search box.
  document.addEventListener("keydown", (e) => {
    if (e.key === "/" && !/^(INPUT|TEXTAREA|SELECT)$/.test(document.activeElement.tagName)) {
      e.preventDefault();
      const s = document.getElementById("global-search");
      if (s) { s.focus(); s.select(); }
    }
  });

  // Selects / date inputs that submit their form on change.
  document.querySelectorAll("[data-autosubmit]").forEach((el) =>
    el.addEventListener("change", () => el.form && el.form.submit())
  );

  // Expand / collapse every abstract in the page (or the section the button lives in).
  document.querySelectorAll("[data-toggle-abstracts]").forEach((btn) => {
    btn.addEventListener("click", () => {
      const scope = btn.closest("section") || document;
      const items = scope.querySelectorAll("details.abs");
      const open = btn.dataset.state !== "open";
      items.forEach((d) => (d.open = open));
      btn.dataset.state = open ? "open" : "closed";
      btn.textContent = open ? "Collapse abstracts" : "Expand abstracts";
    });
  });

  // Instant client-side filter for long lists.
  document.querySelectorAll("[data-filter]").forEach((input) => {
    const counter = document.querySelector(".filter-count");
    const items = Array.from(document.querySelectorAll(input.dataset.filter));
    const groups = Array.from(document.querySelectorAll("section.group"));
    let t;
    input.addEventListener("input", () => {
      clearTimeout(t);
      t = setTimeout(() => {
        const q = input.value.trim().toLowerCase();
        let shown = 0;
        items.forEach((el) => {
          const hit = !q || el.dataset.text.includes(q);
          el.hidden = !hit;
          if (hit) shown++;
        });
        groups.forEach((g) => (g.hidden = !g.querySelector(".paper:not([hidden])")));
        if (counter) counter.textContent = q ? `${shown} of ${items.length} papers` : "";
      }, 80);
    });
  });

  // Topic analysis can take a couple of seconds: show progress.
  document.querySelectorAll("form[data-loading]").forEach((form) =>
    form.addEventListener("submit", () => {
      const box = document.querySelector(".loading");
      if (box) box.hidden = false;
      const btn = form.querySelector("button");
      if (btn) { btn.disabled = true; btn.textContent = "Analysing…"; }
    })
  );
  document.querySelectorAll('a[href^="/topic?q="]').forEach((a) =>
    a.addEventListener("click", () => {
      const box = document.querySelector(".loading");
      if (box) box.hidden = false;
    })
  );

  // Copy-to-clipboard buttons.
  document.querySelectorAll("[data-copy]").forEach((btn) =>
    btn.addEventListener("click", async () => {
      try {
        await navigator.clipboard.writeText(btn.dataset.copy);
        const old = btn.textContent;
        btn.textContent = "Copied";
        setTimeout(() => (btn.textContent = old), 1200);
      } catch (_) { /* clipboard unavailable */ }
    })
  );
})();
