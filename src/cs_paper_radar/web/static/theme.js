// Applied before first paint (loaded synchronously in <head>) to avoid a theme flash.
(function () { var t = localStorage.getItem('theme'); if (t) document.documentElement.dataset.theme = t; })();
