/**
 * Air-Pen Studio — Enhanced Web Client v2
 * Splash screen • Keyboard overlay • Lightbox gallery • Connection monitor • Live telemetry
 */

document.addEventListener("DOMContentLoaded", () => {

  // ────────────────────────────────────────────────────────
  //  SPLASH SCREEN
  // ────────────────────────────────────────────────────────
  const splashScreen  = document.getElementById("splashScreen");
  const splashBarFill = document.getElementById("splashBarFill");

  let splashProgress = 0;
  const splashInterval = setInterval(() => {
    splashProgress = Math.min(splashProgress + Math.random() * 18, 90);
    if (splashBarFill) splashBarFill.style.width = splashProgress + "%";
  }, 120);

  function hideSplash() {
    clearInterval(splashInterval);
    if (splashBarFill) splashBarFill.style.width = "100%";
    setTimeout(() => { if (splashScreen) splashScreen.classList.add("hidden"); }, 500);
  }

  let splashHidden = false;
  function maybeSplash() {
    if (!splashHidden) { splashHidden = true; hideSplash(); }
  }
  setTimeout(maybeSplash, 3000);

  // ────────────────────────────────────────────────────────
  //  ELEMENT REFS
  // ────────────────────────────────────────────────────────
  const gestureStatusPill = document.getElementById("gestureStatusPill");
  const gestureIcon       = document.getElementById("gestureIcon");
  const gestureName       = document.getElementById("gestureName");

  const telDist     = document.getElementById("telDist");
  const telReach    = document.getElementById("telReach");
  const statStrokes = document.getElementById("statStrokes");
  const statFps     = document.getElementById("statFps");
  const statPalm    = document.getElementById("statPalm");
  const canvasModeLabel = document.getElementById("canvasModeLabel");

  const btnUndo       = document.getElementById("btnUndo");
  const btnClear      = document.getElementById("btnClear");
  const btnMode       = document.getElementById("btnMode");
  const btnSave       = document.getElementById("btnSave");
  const btnGuidelines = document.getElementById("btnGuidelines");
  const btnDemo       = document.getElementById("btnDemo");

  const badgeActiveFinger = document.getElementById("badgeActiveFinger");
  const recCharVal        = document.getElementById("recCharVal");
  const recConfVal        = document.getElementById("recConfVal");
  const recFingerVal      = document.getElementById("recFingerVal");
  const btnClearWord      = document.getElementById("btnClearWord");
  const btnAddSpace       = document.getElementById("btnAddSpace");
  const btnBackspace      = document.getElementById("btnBackspace");
  const btnTypeWord       = document.getElementById("btnTypeWord");
  const recWordDisplay    = document.getElementById("recWordDisplay");

  const modeChips        = document.querySelectorAll(".mode-chip");
  const badgeWritingMode = document.getElementById("badgeWritingMode");
  const modeDescText     = document.getElementById("modeDescText");

  const brushSlider   = document.getElementById("brushSlider");
  const brushVal      = document.getElementById("brushVal");
  const reachSlider   = document.getElementById("reachSlider");
  const reachVal      = document.getElementById("reachVal");
  const swatchButtons = document.querySelectorAll(".swatch-btn");
  const optChips      = document.querySelectorAll(".opt-chip");
  const smoothChips   = document.querySelectorAll(".smooth-chip");

  const customColorPicker  = document.getElementById("customColorPicker");
  const gradEndColorPicker = document.getElementById("gradEndColorPicker");
  const gradChips          = document.querySelectorAll(".grad-chip");

  const galleryGrid       = document.getElementById("galleryGrid");
  const btnRefreshGallery = document.getElementById("btnRefreshGallery");
  const galleryCount      = document.getElementById("galleryCount");
  const toastBanner       = document.getElementById("toastBanner");
  const toastMsg          = document.getElementById("toastMsg");

  const connDot       = document.getElementById("connDot");
  const connLabel     = document.getElementById("connLabel");
  const offlineBanner = document.getElementById("offlineBanner");
  const cameraOffline = document.getElementById("cameraOffline");
  const liveFeed      = document.getElementById("liveFeed");

  const shortcutsOverlay  = document.getElementById("shortcutsOverlay");
  const closeShortcutsBtn = document.getElementById("closeShortcuts");
  const btnShortcuts      = document.getElementById("btnShortcuts");

  const lightboxOverlay = document.getElementById("lightboxOverlay");
  const lightboxClose   = document.getElementById("lightboxClose");
  const lightboxImg     = document.getElementById("lightboxImg");
  const lightboxPngDl   = document.getElementById("lightboxPngDl");
  const lightboxSvgDl   = document.getElementById("lightboxSvgDl");

  const gestureRows = {
    "Air-Draw":       document.getElementById("gRowDraw"),
    "Hover / Select": document.getElementById("gRowHover"),
    "Eraser":         document.getElementById("gRowErase"),
    "Brush Resize":   document.getElementById("gRowPinch"),
    "Idle / Pause":   document.getElementById("gRowIdle"),
  };

  const textWordInput = document.getElementById("textWordInput");

  // ────────────────────────────────────────────────────────
  //  TOAST
  // ────────────────────────────────────────────────────────
  let toastTimer = null;
  function showToast(msg) {
    if (!toastMsg || !toastBanner) return;
    toastMsg.textContent = msg;
    toastBanner.classList.add("show");
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => toastBanner.classList.remove("show"), 2400);
  }

  // ────────────────────────────────────────────────────────
  //  AUDIO CHIME
  // ────────────────────────────────────────────────────────
  let audioCtx = null;
  function playBeep(freq, type, duration) {
    freq = freq || 600; type = type || "sine"; duration = duration || 0.06;
    try {
      if (!audioCtx) audioCtx = new (window.AudioContext || window.webkitAudioContext)();
      const osc = audioCtx.createOscillator();
      const gain = audioCtx.createGain();
      osc.type = type;
      osc.frequency.value = freq;
      gain.gain.setValueAtTime(0.08, audioCtx.currentTime);
      gain.gain.exponentialRampToValueAtTime(0.001, audioCtx.currentTime + duration);
      osc.connect(gain); gain.connect(audioCtx.destination);
      osc.start(); osc.stop(audioCtx.currentTime + duration);
    } catch(e) {}
  }

  // ────────────────────────────────────────────────────────
  //  API HELPER
  // ────────────────────────────────────────────────────────
  async function apiAction(endpoint, params) {
    params = params || {};
    try {
      const url = new URL("/api/action/" + endpoint, window.location.origin);
      Object.keys(params).forEach(k => url.searchParams.append(k, params[k]));
      const res = await fetch(url);
      const data = await res.json();
      if (data.msg) showToast(data.msg);
      return data;
    } catch(err) { console.warn("Action failed:", err); }
  }

  // ────────────────────────────────────────────────────────
  //  CONNECTION MONITOR
  // ────────────────────────────────────────────────────────
  let failCount = 0;
  const MAX_FAILS = 5;

  function setOnline() {
    failCount = 0;
    if (connDot)   connDot.classList.remove("offline");
    if (connLabel) { connLabel.classList.remove("offline"); connLabel.textContent = "LIVE"; }
    if (offlineBanner) offlineBanner.classList.remove("show");
    if (cameraOffline) cameraOffline.classList.remove("show");
  }

  function setOffline() {
    failCount++;
    if (connDot)   connDot.classList.add("offline");
    if (connLabel) { connLabel.classList.add("offline"); connLabel.textContent = "OFFLINE"; }
    if (failCount >= MAX_FAILS) {
      if (offlineBanner) offlineBanner.classList.add("show");
      if (cameraOffline) cameraOffline.classList.add("show");
    }
  }

  // ────────────────────────────────────────────────────────
  //  SHORTCUT OVERLAY
  // ────────────────────────────────────────────────────────
  function openShortcuts() { if (shortcutsOverlay) shortcutsOverlay.classList.add("visible"); }
  function closeShortcutsPanel() { if (shortcutsOverlay) shortcutsOverlay.classList.remove("visible"); }

  if (btnShortcuts)      btnShortcuts.addEventListener("click", openShortcuts);
  if (closeShortcutsBtn) closeShortcutsBtn.addEventListener("click", closeShortcutsPanel);
  if (shortcutsOverlay)  shortcutsOverlay.addEventListener("click", (e) => { if (e.target === shortcutsOverlay) closeShortcutsPanel(); });

  // ────────────────────────────────────────────────────────
  //  LIGHTBOX
  // ────────────────────────────────────────────────────────
  function openLightbox(pngFile, svgFile) {
    if (!lightboxOverlay) return;
    if (lightboxImg)   lightboxImg.src = "/output/" + pngFile;
    if (lightboxPngDl) { lightboxPngDl.href = "/output/" + pngFile; lightboxPngDl.download = pngFile; }
    if (lightboxSvgDl) { lightboxSvgDl.href = "/output/" + svgFile; lightboxSvgDl.download = svgFile; }
    lightboxOverlay.classList.add("visible");
  }

  function closeLightbox() {
    if (!lightboxOverlay) return;
    lightboxOverlay.classList.remove("visible");
    setTimeout(() => { if (lightboxImg) lightboxImg.src = ""; }, 300);
  }

  if (lightboxClose)   lightboxClose.addEventListener("click", closeLightbox);
  if (lightboxOverlay) lightboxOverlay.addEventListener("click", (e) => { if (e.target === lightboxOverlay) closeLightbox(); });

  // ────────────────────────────────────────────────────────
  //  HEADER BUTTON ACTIONS
  // ────────────────────────────────────────────────────────
  if (btnDemo) {
    btnDemo.addEventListener("click", async () => {
      playBeep(880, "triangle", 0.15);
      showToast("Drawing AI Harmonic Spiral Demo...");
      await apiAction("demo");
      setTimeout(refreshGallery, 2500);
    });
  }

  if (btnUndo)  btnUndo.addEventListener("click",  () => { playBeep(450); apiAction("undo"); });
  if (btnClear) btnClear.addEventListener("click", () => { playBeep(350); apiAction("clear"); });

  if (btnMode) {
    btnMode.addEventListener("click", async () => {
      playBeep(520);
      const res = await apiAction("mode");
      if (res && res.mode && canvasModeLabel) canvasModeLabel.textContent = res.mode.toUpperCase();
    });
  }

  if (btnSave) {
    btnSave.addEventListener("click", async () => {
      playBeep(750, "sine", 0.2);
      await apiAction("save");
      refreshGallery();
    });
  }

  if (btnGuidelines) {
    btnGuidelines.addEventListener("click", async () => {
      playBeep(580);
      const res = await apiAction("guidelines");
      if (res && res.guidelines !== undefined) {
        if (res.guidelines) btnGuidelines.classList.add("active");
        else btnGuidelines.classList.remove("active");
      }
    });
  }

  if (btnClearWord) {
    btnClearWord.addEventListener("click", async () => {
      playBeep(320); await apiAction("clear_word");
      if (recWordDisplay) recWordDisplay.textContent = "--";
      if (recCharVal) recCharVal.textContent = "--";
      if (recConfVal) recConfVal.textContent = "Word Cleared";
    });
  }

  if (btnAddSpace) {
    btnAddSpace.addEventListener("click", async () => {
      playBeep(520, "sine", 0.05);
      const res = await apiAction("add_space");
      if (res && res.word !== undefined && recWordDisplay) recWordDisplay.textContent = res.word || "--";
    });
  }

  if (btnBackspace) {
    btnBackspace.addEventListener("click", async () => {
      playBeep(420, "triangle", 0.06);
      const res = await apiAction("backspace");
      if (res && res.word !== undefined && recWordDisplay) recWordDisplay.textContent = res.word || "--";
    });
  }

  if (btnTypeWord) {
    btnTypeWord.addEventListener("click", async () => {
      playBeep(880, "sine", 0.15);
      const res = await apiAction("type_word");
      if (res && res.ok && recWordDisplay) recWordDisplay.textContent = "--";
    });
  }

  const modeDescriptions = {
    pinch: "<strong>Pen-Grip:</strong> Pinch thumb + index to draw letters cleanly without unwanted cursive connections.",
    point: "<strong>Classic Point:</strong> Point with index finger to draw. Use peace sign (✌️) to hover pointer reticle.",
    depth: "<strong>Virtual Touch:</strong> Push finger forward in Z-depth towards screen to write. Pull back to hover."
  };

  let currentWritingMode = "pinch";
  function setWritingModeUI(mode) {
    currentWritingMode = mode;
    modeChips.forEach(c => {
      if (c.dataset.mode === mode) c.classList.add("active");
      else c.classList.remove("active");
    });
    if (badgeWritingMode) badgeWritingMode.textContent = mode.toUpperCase();
    if (modeDescText && modeDescriptions[mode]) modeDescText.innerHTML = modeDescriptions[mode];

    const rowDraw = gestureRows["Air-Draw"];
    const rowHover = gestureRows["Hover / Select"];
    if (mode === "pinch") {
      if (rowDraw) {
        const icon = rowDraw.querySelector(".g-icon"); if (icon) icon.textContent = "🤏";
        const strong = rowDraw.querySelector("strong"); if (strong) strong.textContent = "Pinch (Thumb+Index)";
        const span = rowDraw.querySelector("span"); if (span) span.textContent = "Air-Draw / Write (Pen Down)";
      }
      if (rowHover) {
        const icon = rowHover.querySelector(".g-icon"); if (icon) icon.textContent = "☝️";
        const strong = rowHover.querySelector("strong"); if (strong) strong.textContent = "Release Pinch / Point";
        const span = rowHover.querySelector("span"); if (span) span.textContent = "Hover Reticle (Pen Up)";
      }
    } else if (mode === "depth") {
      if (rowDraw) {
        const icon = rowDraw.querySelector(".g-icon"); if (icon) icon.textContent = "🎯";
        const strong = rowDraw.querySelector("strong"); if (strong) strong.textContent = "Push Forward (Z)";
        const span = rowDraw.querySelector("span"); if (span) span.textContent = "Virtual Touch (Pen Down)";
      }
      if (rowHover) {
        const icon = rowHover.querySelector(".g-icon"); if (icon) icon.textContent = "💨";
        const strong = rowHover.querySelector("strong"); if (strong) strong.textContent = "Pull Back from Screen";
        const span = rowHover.querySelector("span"); if (span) span.textContent = "Hover Reticle (Pen Up)";
      }
    } else {
      if (rowDraw) {
        const icon = rowDraw.querySelector(".g-icon"); if (icon) icon.textContent = "☝️";
        const strong = rowDraw.querySelector("strong"); if (strong) strong.textContent = "Any One Finger";
        const span = rowDraw.querySelector("span"); if (span) span.textContent = "Air-Draw / Write Words";
      }
      if (rowHover) {
        const icon = rowHover.querySelector(".g-icon"); if (icon) icon.textContent = "✌️";
        const strong = rowHover.querySelector("strong"); if (strong) strong.textContent = "Peace Sign";
        const span = rowHover.querySelector("span"); if (span) span.textContent = "Hover & Move Pointer";
      }
    }
  }

  modeChips.forEach(chip => {
    chip.addEventListener("click", () => {
      const mode = chip.dataset.mode;
      playBeep(620, "sine", 0.07);
      setWritingModeUI(mode);
      apiAction("pen_mode", { m: mode });
    });
  });

  // ────────────────────────────────────────────────────────
  //  MOUSE / TOUCH DRAWING ON VIDEO FEED
  // ────────────────────────────────────────────────────────
  let isMouseDown = false;

  function sendPoint(clientX, clientY, state) {
    if (!liveFeed) return;
    const rect = liveFeed.getBoundingClientRect();
    const x = Math.max(0, Math.min(1280, (clientX - rect.left) * (1280 / rect.width)));
    const y = Math.max(0, Math.min(720, (clientY - rect.top)  * (720  / rect.height)));
    apiAction("draw_point", { x: Math.round(x), y: Math.round(y), state: state });
  }

  if (liveFeed) {
    liveFeed.addEventListener("mousedown", (e) => { isMouseDown = true; sendPoint(e.clientX, e.clientY, "down"); });
    liveFeed.addEventListener("touchstart", (e) => { e.preventDefault(); sendPoint(e.touches[0].clientX, e.touches[0].clientY, "down"); }, { passive: false });
    liveFeed.addEventListener("touchmove",  (e) => { e.preventDefault(); sendPoint(e.touches[0].clientX, e.touches[0].clientY, "move"); }, { passive: false });
    liveFeed.addEventListener("touchend",   (e) => { e.preventDefault(); sendPoint(e.changedTouches[0].clientX, e.changedTouches[0].clientY, "up"); }, { passive: false });
  }

  window.addEventListener("mousemove", (e) => { if (isMouseDown) sendPoint(e.clientX, e.clientY, "move"); });
  window.addEventListener("mouseup",   (e) => { if (isMouseDown) { isMouseDown = false; sendPoint(e.clientX, e.clientY, "up"); } });

  // ────────────────────────────────────────────────────────
  //  COLOR CONTROLS
  // ────────────────────────────────────────────────────────
  swatchButtons.forEach(btn => {
    btn.addEventListener("click", () => {
      playBeep(650);
      swatchButtons.forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      apiAction("color", { c: btn.dataset.color });
    });
  });

  let customColorDebounce = null;
  function sendCustomColor(hex) {
    swatchButtons.forEach(b => b.classList.remove("active"));
    playBeep(640, "triangle", 0.07);
    clearTimeout(customColorDebounce);
    customColorDebounce = setTimeout(() => apiAction("custom_color", { hex: hex }), 120);
  }

  if (customColorPicker) {
    customColorPicker.addEventListener("input",  e => sendCustomColor(e.target.value));
    customColorPicker.addEventListener("change", e => sendCustomColor(e.target.value));
  }

  // ────────────────────────────────────────────────────────
  //  SLIDERS
  // ────────────────────────────────────────────────────────
  if (brushSlider) {
    brushSlider.addEventListener("input", (e) => {
      const val = e.target.value;
      if (brushVal) brushVal.textContent = val + "px";
      apiAction("brush", { r: val });
    });
  }

  if (reachSlider) {
    reachSlider.addEventListener("input", (e) => {
      const val = parseFloat(e.target.value);
      if (reachVal) reachVal.textContent = val.toFixed(1) + "x";
      apiAction("reach", { m: val });
    });
  }

  // ────────────────────────────────────────────────────────
  //  REACH & SMOOTHING CHIPS
  // ────────────────────────────────────────────────────────
  optChips.forEach(chip => {
    chip.addEventListener("click", () => {
      playBeep(600);
      const reach = parseFloat(chip.dataset.reach);
      optChips.forEach(c => c.classList.remove("active"));
      chip.classList.add("active");
      if (reachSlider) reachSlider.value = reach;
      if (reachVal) reachVal.textContent = reach.toFixed(1) + "x";
      apiAction("reach", { m: reach });
    });
  });

  smoothChips.forEach(chip => {
    chip.addEventListener("click", () => {
      playBeep(700);
      smoothChips.forEach(c => c.classList.remove("active"));
      chip.classList.add("active");
      apiAction("smooth", { p: chip.dataset.smooth });
    });
  });

  // ────────────────────────────────────────────────────────
  //  GRADIENT CHIPS
  // ────────────────────────────────────────────────────────
  function updateGradEndPickerVisibility(mode) {
    if (gradEndColorPicker) gradEndColorPicker.style.display = mode === "custom" ? "block" : "none";
  }
  updateGradEndPickerVisibility("none");

  gradChips.forEach(chip => {
    chip.addEventListener("click", () => {
      const mode = chip.dataset.grad;
      gradChips.forEach(c => c.classList.remove("active"));
      chip.classList.add("active");
      updateGradEndPickerVisibility(mode);
      const params = { mode: mode };
      if (mode === "custom" && gradEndColorPicker) params.end = gradEndColorPicker.value;
      playBeep(mode === "rainbow" ? 880 : 520, "sine", 0.08);
      apiAction("gradient", params);
    });
  });

  if (gradEndColorPicker) {
    let gradEndDebounce = null;
    gradEndColorPicker.addEventListener("input", e => {
      clearTimeout(gradEndDebounce);
      gradEndDebounce = setTimeout(() => {
        const activeGrad = document.querySelector(".grad-chip.active");
        if (activeGrad && activeGrad.dataset.grad === "custom") apiAction("gradient", { mode: "custom", end: e.target.value });
      }, 120);
    });
  }

  // ────────────────────────────────────────────────────────
  //  STATUS POLLING
  // ────────────────────────────────────────────────────────
  async function pollStatus() {
    try {
      const res = await fetch("/api/status");
      if (!res.ok) { setOffline(); return; }
      const st = await res.json();
      setOnline(); maybeSplash();

      if (telDist)  telDist.textContent  = st.distance_zone || "--";
      if (telReach && st.reach_gain) telReach.textContent = st.reach_gain;
      if (statStrokes) statStrokes.textContent = st.stroke_count || 0;
      if (statPalm) statPalm.textContent = st.palm_scale || "--";
      if (canvasModeLabel) canvasModeLabel.textContent = (st.canvas_mode || "BOARD").toUpperCase();

      if (statFps) {
        const fps = st.fps || 0;
        statFps.textContent = fps.toFixed(1);
        statFps.style.color = fps >= 50 ? "var(--accent-cyan)" : fps >= 30 ? "var(--accent-amber)" : "var(--accent-red)";
      }

      const g = st.gesture || "Idle / Pause";
      if (gestureName) gestureName.textContent = g;
      if (gestureStatusPill) {
        gestureStatusPill.className = "header-center-pill";
        const icons = { "Air-Draw": ["active-draw","☝️"], "Hover / Select": ["active-hover","✌️"], "Eraser": ["active-erase","👋"], "Brush Resize": ["active-draw","🤏"] };
        if (icons[g]) { gestureStatusPill.classList.add(icons[g][0]); if (gestureIcon) gestureIcon.textContent = icons[g][1]; }
        else if (gestureIcon) gestureIcon.textContent = "✊";
      }

      Object.keys(gestureRows).forEach(key => {
        const row = gestureRows[key];
        if (row) { if (key === g) row.classList.add("active"); else row.classList.remove("active"); }
      });

      swatchButtons.forEach(btn => {
        if (btn.dataset.color === st.color_key && st.color_key !== "custom") btn.classList.add("active");
        else btn.classList.remove("active");
      });
      if (st.color_key === "custom" && st.color_hex && customColorPicker) customColorPicker.value = st.color_hex;

      if (st.gradient_mode !== undefined) {
        gradChips.forEach(c => { if (c.dataset.grad === st.gradient_mode) c.classList.add("active"); else c.classList.remove("active"); });
        updateGradEndPickerVisibility(st.gradient_mode);
        if (st.gradient_mode === "custom" && st.gradient_end_hex && gradEndColorPicker) gradEndColorPicker.value = st.gradient_end_hex;
      }

      if (brushSlider && document.activeElement !== brushSlider && st.brush_radius) {
        brushSlider.value = st.brush_radius;
        if (brushVal) brushVal.textContent = st.brush_radius + "px";
      }

      if (reachSlider && document.activeElement !== reachSlider && st.reach_multiplier) {
        reachSlider.value = st.reach_multiplier;
        if (reachVal) reachVal.textContent = Number(st.reach_multiplier).toFixed(1) + "x";
      }

      if (st.smoothing_preset) {
        smoothChips.forEach(c => { if (c.dataset.smooth === st.smoothing_preset) c.classList.add("active"); else c.classList.remove("active"); });
      }

      if (recCharVal && st.recognized_char !== undefined) recCharVal.textContent = st.recognized_char || "--";
      if (recConfVal && st.recognized_conf !== undefined) recConfVal.textContent = st.recognized_conf > 0 ? Math.round(st.recognized_conf * 100) + "% Confidence" : "Air-writing active";
      if (recFingerVal && st.active_finger) recFingerVal.textContent = st.active_finger + " Finger";
      if (badgeActiveFinger && st.active_finger) badgeActiveFinger.textContent = "☝️ " + st.active_finger.toUpperCase();
      if (recWordDisplay && st.recognized_word !== undefined) recWordDisplay.textContent = st.recognized_word || "--";

      if (st.writing_mode && st.writing_mode !== currentWritingMode) {
        setWritingModeUI(st.writing_mode);
      }

      if (btnGuidelines && st.guidelines_enabled !== undefined) {
        if (st.guidelines_enabled) btnGuidelines.classList.add("active"); else btnGuidelines.classList.remove("active");
      }

    } catch(err) { setOffline(); }
  }

  // ────────────────────────────────────────────────────────
  //  GALLERY
  // ────────────────────────────────────────────────────────
  async function refreshGallery() {
    if (!galleryGrid) return;
    try {
      const res = await fetch("/api/gallery");
      const files = await res.json();

      if (!files || files.length === 0) {
        galleryGrid.innerHTML = `<div class="empty-state"><div class="empty-icon">🎨</div><span>No artworks yet</span><span class="empty-sub">Draw and hit <strong>Save</strong>!</span></div>`;
        if (galleryCount) galleryCount.textContent = "0";
        return;
      }

      if (galleryCount) galleryCount.textContent = files.length;
      galleryGrid.innerHTML = "";

      files.forEach(file => {
        const item = document.createElement("div");
        item.className = "gallery-item";
        const img = document.createElement("img");
        img.src = "/output/" + file.png; img.alt = file.png; img.loading = "lazy";
        const overlay = document.createElement("div");
        overlay.className = "gallery-overlay";
        const pngLink = document.createElement("a");
        pngLink.href = "/output/" + file.png; pngLink.target = "_blank"; pngLink.download = file.png; pngLink.textContent = "PNG";
        pngLink.addEventListener("click", e => e.stopPropagation());
        const svgLink = document.createElement("a");
        svgLink.href = "/output/" + file.svg; svgLink.target = "_blank"; svgLink.download = file.svg; svgLink.textContent = "SVG";
        svgLink.addEventListener("click", e => e.stopPropagation());
        overlay.appendChild(pngLink); overlay.appendChild(svgLink);
        item.appendChild(img); item.appendChild(overlay);
        item.addEventListener("click", () => openLightbox(file.png, file.svg));
        galleryGrid.appendChild(item);
      });
    } catch(err) { console.warn("Gallery load failed:", err); }
  }

  if (btnRefreshGallery) btnRefreshGallery.addEventListener("click", () => { playBeep(480); refreshGallery(); });

  // ────────────────────────────────────────────────────────
  //  TEXT WRITER
  // ────────────────────────────────────────────────────────
  const btnWriteText = document.getElementById("btnWriteText");
  const wordChips    = document.querySelectorAll(".word-chip");

  async function triggerWriteText(text) {
    if (!text || text.trim() === "") return;
    playBeep(780, "triangle", 0.12);
    showToast("Writing \"" + text.trim() + "\"...");
    await apiAction("write", { text: text.trim() });
  }

  if (btnWriteText) btnWriteText.addEventListener("click", () => triggerWriteText(textWordInput && textWordInput.value));
  if (textWordInput) textWordInput.addEventListener("keydown", (e) => { if (e.key === "Enter") triggerWriteText(textWordInput.value); });
  wordChips.forEach(chip => {
    chip.addEventListener("click", () => {
      const word = chip.dataset.word;
      if (textWordInput) textWordInput.value = word;
      triggerWriteText(word);
    });
  });

  // ────────────────────────────────────────────────────────
  //  KEYBOARD SHORTCUTS
  // ────────────────────────────────────────────────────────
  window.addEventListener("keydown", (e) => {
    if (textWordInput && document.activeElement === textWordInput) return;
    const k = e.key.toLowerCase();
    if (e.key === "Escape") { closeShortcutsPanel(); closeLightbox(); return; }
    if (e.key === "?")  { openShortcuts(); return; }
    if (k === "u") { playBeep(450); apiAction("undo"); }
    else if (k === "c") { playBeep(350); apiAction("clear"); }
    else if (k === "s") { playBeep(750); apiAction("save"); refreshGallery(); }
    else if (k === "m") { playBeep(520); apiAction("mode"); }
    else if (k === "g" && btnGuidelines) btnGuidelines.click();
    else if (k === "d" && btnDemo) btnDemo.click();
    else if (k === "1") { playBeep(600); setWritingModeUI("pinch"); apiAction("pen_mode", { m: "pinch" }); }
    else if (k === "2") { playBeep(640); setWritingModeUI("point"); apiAction("pen_mode", { m: "point" }); }
    else if (k === "3") { playBeep(680); setWritingModeUI("depth"); apiAction("pen_mode", { m: "depth" }); }
    else if (e.key === "Enter" && btnTypeWord) btnTypeWord.click();
    else if (e.key === " " && btnAddSpace) { e.preventDefault(); btnAddSpace.click(); }
    else if (e.key === "Backspace" && btnBackspace) { e.preventDefault(); btnBackspace.click(); }
  });

  // ────────────────────────────────────────────────────────
  //  START
  // ────────────────────────────────────────────────────────
  setInterval(pollStatus, 250);
  refreshGallery();
});
