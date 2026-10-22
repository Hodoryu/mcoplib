(function () {
  function filterAll(query) {
    const q = (query || "").trim().toLowerCase();
    document.querySelectorAll("[data-operator-item]").forEach(function (el) {
      const hay = (el.getAttribute("data-search") || el.textContent || "").toLowerCase();
      el.style.display = (!q || hay.includes(q)) ? "" : "none";
    });
    document.querySelectorAll(".operator-card").forEach(function (el) {
      const hay = (el.getAttribute("data-search") || el.textContent || "").toLowerCase();
      el.style.display = (!q || hay.includes(q)) ? "" : "none";
    });
    const visibleCards = Array.from(document.querySelectorAll(".operator-card")).filter(function (el) {
      return el.style.display !== "none";
    }).length;
    const resultCount = document.getElementById("result-count");
    if (resultCount) resultCount.textContent = q ? ("匹配 " + visibleCards + " 个算子") : "197 个算子";
    const empty = document.getElementById("empty-state");
    if (empty) empty.style.display = (document.querySelectorAll(".operator-card").length && visibleCards === 0) ? "block" : "none";
  }
  document.querySelectorAll(".operator-search").forEach(function (input) {
    input.addEventListener("input", function () { filterAll(input.value); });
  });
  // Keep search boxes synchronized across desktop/mobile navigation.
  document.querySelectorAll(".operator-search").forEach(function (input) {
    input.addEventListener("input", function () {
      document.querySelectorAll(".operator-search").forEach(function (other) {
        if (other !== input) other.value = input.value;
      });
    });
  });
})();
