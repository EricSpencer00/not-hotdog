// Main thread: input handling, the verdict, and the internals panel.
// All arithmetic lives in the worker; this file only moves pixels and paints.

(function () {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const view = $("view");
  const video = $("video");
  const preview = $("preview");
  const placeholder = $("placeholder");
  const verdict = $("verdict");
  const word = $("verdict-word");
  const sub = $("verdict-sub");
  const bar = $("bar");

  const worker = new Worker(
    URL.createObjectURL(new Blob([WORKER_SOURCE], { type: "text/javascript" }))
  );

  // Frames are downscaled to this before being handed over. The model only ever
  // sees 96x96, so sending a 4K frame would cost a large structured-clone copy
  // to throw almost all of it away.
  const GRAB = 256;
  const grab = document.createElement("canvas");
  grab.width = grab.height = GRAB;
  const gctx = grab.getContext("2d", { willReadFrequently: true });

  let ready = false;
  let camera = null;
  let showInternals = false;
  let seq = 0;
  let inFlight = false;
  let pending = null;

  // Verdict hysteresis. A borderline frame flipping the whole page green/red at
  // 30fps is unreadable, so a change has to survive three consecutive frames.
  const AGREE = 3;
  let streak = 0;
  let shown = null;

  // ── plumbing ─────────────────────────────────────────────────────────────

  function send(rgba, w, h) {
    if (!ready) return;
    if (inFlight) {
      pending = { rgba, w, h };
      return;
    }
    inFlight = true;
    worker.postMessage(
      { type: "infer", rgba: rgba.buffer, w, h, withTaps: showInternals, seq: ++seq },
      [rgba.buffer]
    );
  }

  function grabFrom(source, w, h) {
    const side = Math.min(w, h);
    gctx.drawImage(source, (w - side) / 2, (h - side) / 2, side, side, 0, 0, GRAB, GRAB);
    const d = gctx.getImageData(0, 0, GRAB, GRAB);
    send(new Uint8ClampedArray(d.data), GRAB, GRAB);
  }

  worker.onmessage = (e) => {
    const m = e.data;
    if (m.type === "ready") {
      ready = true;
      $("r-params").textContent = m.params.toLocaleString();
      $("r-size").textContent = (m.bytes / 1024).toFixed(0) + " KB int8";
      buildLayerSlots(m.layers);
      return;
    }
    if (m.type === "result") {
      inFlight = false;
      paint(m);
      if (pending) {
        const p = pending;
        pending = null;
        send(p.rgba, p.w, p.h);
      }
    }
  };

  // ── painting ─────────────────────────────────────────────────────────────

  function paint(m) {
    const conf = m.isHotdog ? m.prob : 1 - m.prob;
    $("r-conf").textContent = conf.toFixed(3);
    $("r-logit").textContent = m.logit.toLocaleString();
    $("r-ms").textContent = m.ms.toFixed(1) + " ms";
    bar.style.width = (conf * 100).toFixed(1) + "%";

    if (m.isHotdog === shown) {
      streak = 0;
    } else {
      streak++;
      if (streak >= (camera ? AGREE : 1)) {
        shown = m.isHotdog;
        streak = 0;
        setVerdict(m.isHotdog);
      }
    }
    if (m.tiles) drawTiles(m.tiles);
  }

  function setVerdict(isHotdog) {
    verdict.classList.remove("idle");
    const c = isHotdog ? "var(--yes)" : "var(--no)";
    document.documentElement.style.setProperty("--verdict", c);
    word.textContent = isHotdog ? "HOT DOG" : "NOT HOT DOG";
    sub.textContent = isHotdog ? "it is a hot dog" : "it is not a hot dog";
    verdict.classList.add("flip");
    setTimeout(() => verdict.classList.remove("flip"), 340);
  }

  // ── internals ────────────────────────────────────────────────────────────

  const slots = [];
  function buildLayerSlots(layers) {
    const host = $("layers");
    host.textContent = "";
    for (const l of layers) {
      const d = document.createElement("div");
      d.className = "layer";
      const c = document.createElement("canvas");
      const nm = document.createElement("div");
      nm.className = "nm";
      nm.textContent = l.name;
      const sh = document.createElement("div");
      sh.className = "sh";
      sh.textContent = l.shape;
      d.append(c, nm, sh);
      host.append(d);
      slots.push(c);
    }
  }

  function drawTiles(tiles) {
    for (let i = 0; i < tiles.length && i < slots.length; i++) {
      const t = tiles[i];
      const c = slots[i];
      if (c.width !== t.w || c.height !== t.h) {
        c.width = t.w;
        c.height = t.h;
      }
      const ctx = c.getContext("2d");
      const img = ctx.createImageData(t.w, t.h);
      const g = t.data;
      for (let p = 0, o = 0; p < g.length; p++, o += 4) {
        // Tint toward the site's ink rather than pure greyscale, so the panel
        // sits in the same palette as everything around it.
        const v = g[p];
        img.data[o] = v;
        img.data[o + 1] = v;
        img.data[o + 2] = (v * 0.94) | 0;
        img.data[o + 3] = 255;
      }
      ctx.putImageData(img, 0, 0);
    }
  }

  // ── image sources ────────────────────────────────────────────────────────

  function showStill(bitmapOrImg, w, h) {
    stopCamera();
    preview.width = 512;
    preview.height = 512;
    const side = Math.min(w, h);
    preview
      .getContext("2d")
      .drawImage(bitmapOrImg, (w - side) / 2, (h - side) / 2, side, side, 0, 0, 512, 512);
    preview.classList.remove("hide");
    video.classList.add("hide");
    placeholder.classList.add("hide");
    grabFrom(bitmapOrImg, w, h);
  }

  function loadBlob(blob) {
    const url = URL.createObjectURL(blob);
    const img = new Image();
    img.onload = () => {
      showStill(img, img.naturalWidth, img.naturalHeight);
      URL.revokeObjectURL(url);
    };
    img.onerror = () => {
      URL.revokeObjectURL(url);
      placeholder.textContent = "That file did not decode as an image.";
    };
    img.src = url;
  }

  // ── camera ───────────────────────────────────────────────────────────────

  function stopCamera() {
    if (camera) {
      camera.getTracks().forEach((t) => t.stop());
      camera = null;
      $("btn-cam").setAttribute("aria-pressed", "false");
      $("btn-cam").textContent = "Use camera";
    }
  }

  async function startCamera() {
    if (camera) {
      stopCamera();
      return;
    }
    try {
      camera = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: { ideal: "environment" }, width: { ideal: 640 } },
        audio: false,
      });
    } catch (err) {
      placeholder.classList.remove("hide");
      placeholder.textContent =
        err && err.name === "NotAllowedError"
          ? "Camera permission denied. Drop or paste an image instead."
          : "No camera available. Drop or paste an image instead.";
      return;
    }
    video.srcObject = camera;
    await video.play();
    video.classList.remove("hide");
    preview.classList.add("hide");
    placeholder.classList.add("hide");
    $("btn-cam").setAttribute("aria-pressed", "true");
    $("btn-cam").textContent = "Stop camera";
    loop();
  }

  function loop() {
    if (!camera) return;
    if (video.videoWidth) grabFrom(video, video.videoWidth, video.videoHeight);
    requestAnimationFrame(loop);
  }

  // ── samples ──────────────────────────────────────────────────────────────

  // Lazily fetched, and the only network request the page ever makes after
  // load. Nothing is fetched unless a sample is actually clicked.
  const SAMPLES = [
    ["hotdog", "hot dog"],
    ["corndog", "corn dog"],
    ["bratwurst", "brat"],
    ["sub", "sub"],
    ["burger", "burger"],
    ["dachshund", "dog"],
  ];

  function buildSamples() {
    const host = $("samples");
    for (const [file, label] of SAMPLES) {
      const b = document.createElement("button");
      b.type = "button";
      b.title = label;
      const img = document.createElement("img");
      img.loading = "lazy";
      img.alt = label;
      img.src = "./samples/" + file + ".webp";
      const l = document.createElement("span");
      l.className = "lbl mono";
      l.textContent = label;
      b.append(img, l);
      b.onclick = () => {
        if (img.complete && img.naturalWidth) {
          showStill(img, img.naturalWidth, img.naturalHeight);
        } else {
          img.onload = () => showStill(img, img.naturalWidth, img.naturalHeight);
        }
      };
      host.append(b);
    }
  }

  // ── wiring ───────────────────────────────────────────────────────────────

  $("btn-cam").onclick = startCamera;
  $("btn-file").onclick = () => $("file").click();
  $("file").onchange = (e) => e.target.files[0] && loadBlob(e.target.files[0]);

  $("btn-internals").onclick = (e) => {
    showInternals = !showInternals;
    e.target.setAttribute("aria-pressed", String(showInternals));
    e.target.textContent = showInternals ? "Hide internals" : "Show internals";
    $("internals").classList.toggle("hide", !showInternals);
  };

  ["dragenter", "dragover"].forEach((t) =>
    view.addEventListener(t, (e) => {
      e.preventDefault();
      view.classList.add("drag");
    })
  );
  ["dragleave", "drop"].forEach((t) =>
    view.addEventListener(t, (e) => {
      e.preventDefault();
      view.classList.remove("drag");
    })
  );
  document.addEventListener("drop", (e) => {
    e.preventDefault();
    const f = e.dataTransfer && e.dataTransfer.files[0];
    if (f) loadBlob(f);
  });
  document.addEventListener("dragover", (e) => e.preventDefault());

  document.addEventListener("paste", (e) => {
    for (const item of e.clipboardData.items) {
      if (item.type.startsWith("image/")) {
        loadBlob(item.getAsFile());
        return;
      }
    }
  });

  buildSamples();
  worker.postMessage({ type: "init" });
})();
