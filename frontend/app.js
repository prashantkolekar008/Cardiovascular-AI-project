/**
 * CardioVision Pro - Clinical Enterprise 3D Anatomical Heart Engine
 * Loads Realistic Anatomical 3D Mesh (anatomical_heart.glb),
 * Overlays Continuous Multi-Vessel Coronary Arterial Networks (LAD, LCX, RCA),
 * Provides Real-Time Lead II ECG Waveform & Multi-Model Diagnostics.
 */

// --- Global Application State ---
let scene, camera, renderer, controls;
let heartGroup;
let realHeartMesh = null;
let vesselMeshes = { LAD: [], LCX: [], RCA: [] };
let lesionMarkers = { LAD: null, LCX: null, RCA: null };
let isPulsing = true;
let currentPatientData = null;
let currentPredictions = null;
let currentExplanations = null;
let raycaster = new THREE.Raycaster();
let mouse = new THREE.Vector2();
let hoveredVessel = null;
let targetCameraPos = null;

// ECG Monitor State
let ecgCanvas, ecgCtx;
let ecgX = 0;
let ecgStep = 0;
let ecgHeartRate = 75;
let ecgStElevation = false;

const API_BASE = "http://127.0.0.1:8000";

// --- 1. THREE.JS INITIALIZATION ---
function init3D() {
  const container = document.getElementById("canvas-container");
  const width = container.clientWidth;
  const height = container.clientHeight || 580;

  // Scene
  scene = new THREE.Scene();
  scene.background = new THREE.Color(0x06090f);

  // Camera
  camera = new THREE.PerspectiveCamera(40, width / height, 0.1, 100);
  camera.position.set(3.2, 1.4, 6.2);

  // Renderer
  renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false, powerPreference: "high-performance" });
  renderer.setSize(width, height);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.35;
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  container.appendChild(renderer.domElement);

  // Controls
  controls = new THREE.OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.06;
  controls.minDistance = 2.8;
  controls.maxDistance = 11;
  controls.target.set(0, 0, 0);

  // Studio Lighting
  setupMedicalLighting();

  // Root group for rotation and heartbeat pulsation
  heartGroup = new THREE.Group();
  scene.add(heartGroup);

  // Load High-Detail Anatomical 3D Heart Mesh
  loadAnatomicalHeartGLB();

  // Event Listeners
  window.addEventListener("resize", onWindowResize);
  renderer.domElement.addEventListener("mousemove", onMouseMove);
  renderer.domElement.addEventListener("click", onCanvasClick);

  // ECG
  initECGMonitor();

  // Animation Loop
  animate();
}

function setupMedicalLighting() {
  const ambientLight = new THREE.AmbientLight(0x1e293b, 1.4);
  scene.add(ambientLight);

  const keyLight = new THREE.DirectionalLight(0xffffff, 1.5);
  keyLight.position.set(5, 7, 6);
  scene.add(keyLight);

  const fillLight = new THREE.DirectionalLight(0x38bdf8, 0.85);
  fillLight.position.set(-6, 3, -4);
  scene.add(fillLight);

  const backLight = new THREE.DirectionalLight(0xf43f5e, 0.6);
  backLight.position.set(0, -5, -4);
  scene.add(backLight);
}

// --- 2. LOAD REAL ANATOMICAL GLB HEART MODEL & CONTINUOUS CORONARIES ---
function loadAnatomicalHeartGLB() {
  const loader = new THREE.GLTFLoader();

  loader.load(
    "anatomical_heart.glb",
    (gltf) => {
      realHeartMesh = gltf.scene;

      // Center and normalize scale
      const box = new THREE.Box3().setFromObject(realHeartMesh);
      const center = box.getCenter(new THREE.Vector3());
      const size = box.getSize(new THREE.Vector3());

      // Center at origin
      realHeartMesh.position.x -= center.x;
      realHeartMesh.position.y -= center.y;
      realHeartMesh.position.z -= center.z;

      // Normalize height to ~3.2 units
      const maxDim = Math.max(size.x, size.y, size.z);
      const scale = 3.2 / maxDim;
      realHeartMesh.scale.set(scale, scale, scale);

      // Enhance materials for rich medical depth
      realHeartMesh.traverse((child) => {
        if (child.isMesh) {
          child.castShadow = true;
          child.receiveShadow = true;
          if (child.material) {
            child.material.roughness = 0.42;
            child.material.metalness = 0.12;
            child.material.envMapIntensity = 1.0;
          }
        }
      });

      heartGroup.add(realHeartMesh);

      // Overlay continuous, anatomical coronary arterial trees
      buildContinuousCoronaryArteries();
    },
    (xhr) => {
      // Progress
      const pct = Math.round((xhr.loaded / (xhr.total || 7555476)) * 100);
      console.log(`Loading 3D Anatomy: ${pct}%`);
    },
    (error) => {
      console.warn("Could not load anatomical_heart.glb, building organic procedural fallback:", error);
      buildContinuousOrganicFallback();
      buildContinuousCoronaryArteries();
    }
  );
}

// Organic Continuous Single-Mesh Fallback (Seamless, no broken parts)
function buildContinuousOrganicFallback() {
  const geo = new THREE.SphereGeometry(1.6, 64, 64);
  const pos = geo.attributes.position;
  for (let i = 0; i < pos.count; i++) {
    let x = pos.getX(i);
    let y = pos.getY(i);
    let z = pos.getZ(i);

    if (y < 0) {
      const taper = 1.0 + (y / 1.6) * 0.55;
      x *= taper;
      z *= taper;
      x -= (-y * 0.25);
    } else {
      x *= 1.1;
      z *= 0.95;
    }
    pos.setXYZ(i, x, y, z);
  }
  geo.computeVertexNormals();

  const mat = new THREE.MeshStandardMaterial({
    color: 0x881337,
    roughness: 0.38,
    metalness: 0.12
  });
  realHeartMesh = new THREE.Mesh(geo, mat);
  heartGroup.add(realHeartMesh);
}

// Build Continuous, Anatomically-Grounded Coronary Arteries (No breaks, no gaps)
function buildContinuousCoronaryArteries() {
  vesselMeshes = { LAD: [], LCX: [], RCA: [] };

  function addArteryBranch(points, radius, vesselKey, name, territory) {
    const curve = new THREE.CatmullRomCurve3(points);
    const geo = new THREE.TubeGeometry(curve, 48, radius, 16, false);
    
    const mat = new THREE.MeshStandardMaterial({
      color: 0x10b981, // Default patent green
      roughness: 0.25,
      metalness: 0.2,
      emissive: 0x10b981,
      emissiveIntensity: 0.35,
    });

    const mesh = new THREE.Mesh(geo, mat);
    mesh.name = vesselKey;
    mesh.userData = { vessel: vesselKey, name: name, territory: territory };
    heartGroup.add(mesh);
    vesselMeshes[vesselKey].push(mesh);
    return mesh;
  }

  // ==========================================
  // 1. LAD (Left Anterior Descending)
  // Continuous arterial spine down anterior interventricular groove
  // ==========================================
  const ladSpine = [
    new THREE.Vector3(-0.15, 0.75, 0.82),   // Ostium at aortic root
    new THREE.Vector3(-0.32, 0.42, 1.02),   // Proximal LAD
    new THREE.Vector3(-0.48, 0.05, 1.15),   // Mid LAD
    new THREE.Vector3(-0.62, -0.45, 1.12),  // Distal LAD
    new THREE.Vector3(-0.76, -0.92, 0.94),  // Lower anterior wall
    new THREE.Vector3(-0.88, -1.35, 0.62),  // Reaches apex
    new THREE.Vector3(-0.92, -1.58, 0.35)   // Wraps cardiac apex
  ];
  addArteryBranch(ladSpine, 0.082, "LAD", "Left Anterior Descending (LAD)", "Anterior myocardial wall and interventricular septum");

  // Diagonal 1 (D1) branching seamlessly off Mid-LAD
  const d1Branch = [
    new THREE.Vector3(-0.48, 0.05, 1.15),
    new THREE.Vector3(-0.82, -0.12, 1.02),
    new THREE.Vector3(-1.12, -0.38, 0.78)
  ];
  addArteryBranch(d1Branch, 0.052, "LAD", "LAD - Diagonal 1 (D1)", "Anterolateral left ventricular wall");

  // Diagonal 2 (D2) branching seamlessly off Distal-LAD
  const d2Branch = [
    new THREE.Vector3(-0.62, -0.45, 1.12),
    new THREE.Vector3(-0.95, -0.68, 0.95),
    new THREE.Vector3(-1.18, -0.98, 0.68)
  ];
  addArteryBranch(d2Branch, 0.045, "LAD", "LAD - Diagonal 2 (D2)", "Apical lateral wall");

  // ==========================================
  // 2. LCX (Left Circumflex Artery)
  // Continuous curve traveling in the left atrioventricular groove
  // ==========================================
  const lcxSpine = [
    new THREE.Vector3(-0.15, 0.75, 0.82),   // Left Main bifurcation
    new THREE.Vector3(-0.58, 0.62, 0.62),   // Proximal LCX in AV sulcus
    new THREE.Vector3(-1.02, 0.38, 0.22),   // Mid LCX wrapping lateral wall
    new THREE.Vector3(-1.22, 0.02, -0.25),  // Lateral margin
    new THREE.Vector3(-1.12, -0.45, -0.68), // Posterior descending branch
    new THREE.Vector3(-0.78, -0.85, -0.92)  // Posterior left ventricle
  ];
  addArteryBranch(lcxSpine, 0.078, "LCX", "Left Circumflex (LCX)", "Lateral and posterior left ventricle");

  // Obtuse Marginal 1 (OM1) branching off LCX
  const om1Branch = [
    new THREE.Vector3(-1.02, 0.38, 0.22),
    new THREE.Vector3(-1.32, -0.08, 0.15),
    new THREE.Vector3(-1.42, -0.58, -0.05)
  ];
  addArteryBranch(om1Branch, 0.05, "LCX", "LCX - Obtuse Marginal (OM1)", "Obtuse margin of left ventricle");

  // ==========================================
  // 3. RCA (Right Coronary Artery)
  // Continuous curve running in right AV sulcus to the crux/inferior wall
  // ==========================================
  const rcaSpine = [
    new THREE.Vector3(0.25, 0.78, 0.78),    // Right aortic sinus ostium
    new THREE.Vector3(0.68, 0.52, 0.82),    // Proximal RCA in AV groove
    new THREE.Vector3(1.08, 0.12, 0.62),    // Mid RCA along acute margin
    new THREE.Vector3(1.18, -0.38, 0.32),   // Distal RCA
    new THREE.Vector3(1.02, -0.85, -0.15),  // Lower right AV groove
    new THREE.Vector3(0.65, -1.22, -0.52),  // Crux cordis
    new THREE.Vector3(0.22, -1.45, -0.68)   // Posterior Descending Artery (PDA)
  ];
  addArteryBranch(rcaSpine, 0.082, "RCA", "Right Coronary Artery (RCA)", "Right ventricle and inferior diaphragmatic wall");

  // Acute Marginal (AM) branching off RCA
  const amBranch = [
    new THREE.Vector3(1.08, 0.12, 0.62),
    new THREE.Vector3(1.28, -0.28, 0.75),
    new THREE.Vector3(0.98, -0.72, 0.88)
  ];
  addArteryBranch(amBranch, 0.048, "RCA", "RCA - Acute Marginal Branch", "Anterior right ventricular wall");

  // Build Stenosis Plaque Markers on vessels
  buildStenosisPlaqueMarkers();
}

function buildStenosisPlaqueMarkers() {
  const markerLocations = {
    LAD: new THREE.Vector3(-0.32, 0.42, 1.02),
    LCX: new THREE.Vector3(-1.02, 0.38, 0.22),
    RCA: new THREE.Vector3(1.08, 0.12, 0.62)
  };

  ["LAD", "LCX", "RCA"].forEach(vKey => {
    const loc = markerLocations[vKey];
    const collarGeo = new THREE.TorusGeometry(0.14, 0.038, 16, 32);
    const collarMat = new THREE.MeshBasicMaterial({
      color: 0xef4444,
      transparent: true,
      opacity: 0.0
    });
    const collarMesh = new THREE.Mesh(collarGeo, collarMat);
    collarMesh.position.copy(loc);
    collarMesh.lookAt(loc.clone().add(new THREE.Vector3(0, 1, 0)));
    heartGroup.add(collarMesh);
    lesionMarkers[vKey] = collarMesh;
  });
}

// --- 3. ANIMATION LOOP & PHYSIOLOGICAL PULSE ---
let clock = new THREE.Clock();

function animate() {
  requestAnimationFrame(animate);
  const elapsedTime = clock.getElapsedTime();

  // Natural Heartbeat Pulsation
  if (isPulsing && heartGroup) {
    const beatCycle = Math.pow(Math.sin(elapsedTime * (ecgHeartRate / 60) * Math.PI), 12);
    const scale = 1.0 + beatCycle * 0.032;
    heartGroup.scale.set(scale, scale, scale);

    // Lesion plaque pulsation
    ["LAD", "LCX", "RCA"].forEach(vKey => {
      const marker = lesionMarkers[vKey];
      if (marker && marker.material.opacity > 0) {
        const p = 1.0 + Math.sin(elapsedTime * 6.0) * 0.22;
        marker.scale.set(p, p, p);
      }
    });
  }

  // Smooth Camera Lerp
  if (targetCameraPos) {
    camera.position.lerp(targetCameraPos, 0.08);
    if (camera.position.distanceTo(targetCameraPos) < 0.04) {
      targetCameraPos = null;
    }
  }

  controls.update();
  renderer.render(scene, camera);
}

function onWindowResize() {
  const container = document.getElementById("canvas-container");
  const width = container.clientWidth;
  const height = container.clientHeight || 580;
  camera.aspect = width / height;
  camera.updateProjectionMatrix();
  renderer.setSize(width, height);
}

// --- 4. 3D RAYCASTING & INTERACTION ---
function onMouseMove(event) {
  const rect = renderer.domElement.getBoundingClientRect();
  mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
  mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

  raycaster.setFromCamera(mouse, camera);
  const allMeshes = [...vesselMeshes.LAD, ...vesselMeshes.LCX, ...vesselMeshes.RCA];
  const intersects = raycaster.intersectObjects(allMeshes, false);

  const tooltip = document.getElementById("vessel-tooltip");

  if (intersects.length > 0) {
    const hit = intersects[0].object;
    const vKey = hit.userData.vessel;

    if (vKey && currentPredictions && currentPredictions.vessels[vKey]) {
      hoveredVessel = vKey;
      renderer.domElement.style.cursor = "pointer";

      const vData = currentPredictions.vessels[vKey];
      document.getElementById("tooltip-title").textContent = vData.full_name;
      document.getElementById("tooltip-badge").textContent = `${vData.percentage}% Risk`;
      document.getElementById("tooltip-badge").className = `px-2 py-0.5 rounded text-[10px] font-bold uppercase font-mono-code ${
        vData.probability >= 0.65 ? 'bg-red-500/20 text-red-400 border border-red-500/30' : (vData.probability >= 0.35 ? 'bg-amber-500/20 text-amber-400 border border-amber-500/30' : 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30')
      }`;
      document.getElementById("tooltip-territory").textContent = vData.territory;
      document.getElementById("tooltip-status").textContent = vData.status;
      document.getElementById("tooltip-grade").textContent = vData.probability >= 0.65 ? "CAD-RADS 4A (Severe >70%)" : (vData.probability >= 0.35 ? "CAD-RADS 3 (Moderate 50-69%)" : "CAD-RADS 0 (Minimal/None)");

      tooltip.style.opacity = "1";
      return;
    }
  }

  renderer.domElement.style.cursor = "grab";
  tooltip.style.opacity = "0";
  hoveredVessel = null;
}

function onCanvasClick() {
  if (hoveredVessel) {
    selectVesselInDashboard(hoveredVessel);
  }
}

function selectVesselInDashboard(vKey) {
  document.querySelectorAll(".vessel-row").forEach(el => el.classList.remove("selected-vessel"));
  const card = document.getElementById(`card-vessel-${vKey.toLowerCase()}`);
  if (card) {
    card.classList.add("selected-vessel");
  }
}

// --- 5. CAMERA VIEWPORTS ---
function setupCameraControls() {
  document.getElementById("cam-anterior").addEventListener("click", () => {
    targetCameraPos = new THREE.Vector3(0.0, 0.1, 6.2); // Direct coronal/anterior LAD view
    selectVesselInDashboard("LAD");
  });
  document.getElementById("cam-lateral").addEventListener("click", () => {
    targetCameraPos = new THREE.Vector3(-5.8, 0.8, 2.8); // Lateral LCX view
    selectVesselInDashboard("LCX");
  });
  document.getElementById("cam-inferior").addEventListener("click", () => {
    targetCameraPos = new THREE.Vector3(5.2, -1.6, 3.2); // Inferior RCA view
    selectVesselInDashboard("RCA");
  });
  document.getElementById("cam-reset").addEventListener("click", () => {
    targetCameraPos = new THREE.Vector3(3.2, 1.4, 6.2); // Default 3/4 anatomical view
    document.querySelectorAll(".vessel-row").forEach(el => el.classList.remove("selected-vessel"));
  });
}

// --- 6. LEAD II ECG WAVEFORM RHYTHM STRIP ---
function initECGMonitor() {
  ecgCanvas = document.getElementById("ecg-canvas");
  if (!ecgCanvas) return;
  ecgCtx = ecgCanvas.getContext("2d");
  ecgCanvas.width = ecgCanvas.clientWidth * 2;
  ecgCanvas.height = ecgCanvas.clientHeight * 2;
  setInterval(drawECGTrace, 25);
}

function drawECGTrace() {
  if (!ecgCtx) return;
  const w = ecgCanvas.width;
  const h = ecgCanvas.height;
  const baseline = h / 2;

  ecgCtx.fillStyle = "#06090f";
  ecgCtx.fillRect(ecgX, 0, 10, h);

  const cycleLength = Math.max(25, Math.floor(1200 / ecgHeartRate));
  const phase = ecgStep % cycleLength;
  let voltage = 0;

  if (phase < 8) {
    voltage = Math.sin((phase / 8) * Math.PI) * 5;
  } else if (phase >= 12 && phase < 14) {
    voltage = -4;
  } else if (phase >= 14 && phase < 18) {
    voltage = 28;
  } else if (phase >= 18 && phase < 21) {
    voltage = -8;
  } else if (phase >= 21 && phase < 26) {
    voltage = ecgStElevation ? 9 : 0;
  } else if (phase >= 26 && phase < 38) {
    voltage = Math.sin(((phase - 26) / 12) * Math.PI) * (ecgStElevation ? 12 : 7);
  }

  const y = baseline - voltage;

  ecgCtx.strokeStyle = ecgStElevation ? "#f43f5e" : "#06b6d4";
  ecgCtx.lineWidth = 2.2;
  ecgCtx.lineCap = "round";

  ecgCtx.beginPath();
  ecgCtx.moveTo(ecgX - 3, baseline);
  ecgCtx.lineTo(ecgX, y);
  ecgCtx.stroke();

  ecgX += 3;
  if (ecgX >= w) ecgX = 0;
  ecgStep++;
}

// --- 7. API COMMUNICATION & DIAGNOSTIC REFRESH ---
async function fetchPredictions(patientPayload) {
  currentPatientData = patientPayload;
  try {
    const predRes = await fetch(`${API_BASE}/api/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(patientPayload)
    });
    currentPredictions = await predRes.json();

    const expRes = await fetch(`${API_BASE}/api/explain`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(patientPayload)
    });
    currentExplanations = await expRes.json();

    ecgHeartRate = patientPayload.PR || 75;
    ecgStElevation = (patientPayload["St Elevation"] === 1);
    const hrDisplay = document.getElementById("ecg-hr-display");
    if (hrDisplay) {
      hrDisplay.textContent = `${ecgHeartRate} BPM ${ecgStElevation ? '• ST-ELEVATION' : '• NORMAL'}`;
      hrDisplay.className = ecgStElevation ? "text-rose-400 font-bold" : "text-cyan-400 font-bold";
    }

    updateDashboardUI();
    update3DCoronaries();

  } catch (err) {
    console.error("Diagnostic API Error:", err);
  }
}

function update3DCoronaries() {
  if (!currentPredictions || !currentPredictions.vessels) return;

  ["LAD", "LCX", "RCA"].forEach(vKey => {
    const vData = currentPredictions.vessels[vKey];
    const meshes = vesselMeshes[vKey];
    const marker = lesionMarkers[vKey];

    if (meshes && vData) {
      const colorVal = new THREE.Color(vData.color);

      meshes.forEach(m => {
        m.material.color = colorVal;
        m.material.emissive = colorVal;
        m.material.emissiveIntensity = vData.probability >= 0.65 ? 0.65 : 0.25;
      });

      if (marker) {
        if (vData.probability >= 0.65) {
          marker.material.opacity = 0.85;
          marker.material.color = new THREE.Color(0xef4444);
        } else if (vData.probability >= 0.35) {
          marker.material.opacity = 0.45;
          marker.material.color = new THREE.Color(0xf59e0b);
        } else {
          marker.material.opacity = 0.0;
        }
      }
    }
  });
}

function updateDashboardUI() {
  if (!currentPredictions) return;

  const { overall_cad, vessels } = currentPredictions;

  // Gauge
  const arc = document.getElementById("gauge-arc");
  const circumference = 2 * Math.PI * 40;
  const offset = circumference - (overall_cad.probability * circumference);
  arc.style.strokeDashoffset = offset;
  arc.style.stroke = overall_cad.color;

  document.getElementById("cad-prob-text").textContent = `${overall_cad.percentage}%`;
  
  const classificationBadge = document.getElementById("cad-classification-badge");
  classificationBadge.textContent = overall_cad.prediction === "CAD" ? "CAD Positive (High Risk)" : "Low CAD Risk (Normal)";
  classificationBadge.className = `px-2.5 py-0.5 rounded text-xs font-bold font-mono-code ${
    overall_cad.prediction === "CAD" ? 'bg-rose-500/20 text-rose-400 border border-rose-500/30' : 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30'
  }`;

  const cadRads = document.getElementById("cad-rads-grade");
  if (overall_cad.probability >= 0.70) {
    cadRads.textContent = "4A: Severe Occlusion (70-99%)";
    cadRads.className = "text-rose-400 font-bold";
  } else if (overall_cad.probability >= 0.45) {
    cadRads.textContent = "3: Moderate Stenosis (50-69%)";
    cadRads.className = "text-amber-400 font-bold";
  } else {
    cadRads.textContent = "0: Minimal / No Plaque (<25%)";
    cadRads.className = "text-emerald-400 font-bold";
  }

  // Target Vessels
  ["LAD", "LCX", "RCA"].forEach(vKey => {
    const vData = vessels[vKey];
    const k = vKey.toLowerCase();

    const badge = document.getElementById(`badge-${k}`);
    const bar = document.getElementById(`bar-${k}`);
    const ind = document.getElementById(`indicator-${k}`);

    badge.textContent = `${vData.percentage}% Risk`;
    badge.className = `px-2 py-0.5 rounded text-[10px] font-bold font-mono-code ${
      vData.probability >= 0.65 ? 'bg-rose-500/20 text-rose-400' : (vData.probability >= 0.35 ? 'bg-amber-500/20 text-amber-400' : 'bg-emerald-500/20 text-emerald-400')
    }`;

    bar.style.width = `${vData.percentage}%`;
    bar.style.backgroundColor = vData.color;
    ind.style.backgroundColor = vData.color;
  });

  // Vitals
  if (currentPatientData) {
    document.getElementById("chip-age").textContent = `${currentPatientData.Age || '--'}y / ${(currentPatientData.Sex || 'M').charAt(0)}`;
    document.getElementById("chip-bp").textContent = `${currentPatientData.BP || '--'} / ${currentPatientData.PR || '--'}`;
    document.getElementById("chip-ef").textContent = `${currentPatientData["EF-TTE"] || '--'}%`;
    document.getElementById("chip-lipids").textContent = `${currentPatientData.LDL || '--'}`;
  }

  // Explainability
  const container = document.getElementById("explain-container");
  container.innerHTML = "";

  if (currentExplanations && currentExplanations.top_contributing_factors) {
    currentExplanations.top_contributing_factors.slice(0, 5).forEach(f => {
      const item = document.createElement("div");
      item.className = "bg-[#080d16] p-2.5 rounded-lg border border-[#162033] space-y-1";
      item.innerHTML = `
        <div class="flex items-center justify-between text-xs">
          <span class="font-semibold text-slate-200">${f.feature}</span>
          <span class="text-cyan-400 font-bold">${f.relative_importance_pct}% weight</span>
        </div>
        <div class="w-full bg-[#151f33] h-1.5 rounded-full overflow-hidden">
          <div class="bg-gradient-to-r from-blue-500 to-cyan-400 h-full rounded-full" style="width: ${Math.min(f.relative_importance_pct * 3.5, 100)}%"></div>
        </div>
        <div class="flex justify-between text-[10px] text-slate-400">
          <span>Recorded: <b class="text-slate-300">${f.patient_value}</b></span>
          <span class="${f.direction === 'Elevates Risk' ? 'text-rose-400' : 'text-slate-400'}">${f.direction}</span>
        </div>
      `;
      container.appendChild(item);
    });
  }
}

// --- 8. PRESET BUTTONS & SIMULATION MODAL ---
let presetsList = [];

async function setupPresets() {
  try {
    const res = await fetch(`${API_BASE}/api/presets`);
    presetsList = await res.json();

    document.getElementById("btn-preset-1").addEventListener("click", () => activatePreset(0, "btn-preset-1"));
    document.getElementById("btn-preset-2").addEventListener("click", () => activatePreset(1, "btn-preset-2"));
    document.getElementById("btn-preset-3").addEventListener("click", () => activatePreset(2, "btn-preset-3"));

    activatePreset(2, "btn-preset-3");
  } catch (err) {
    console.error("Presets loading error:", err);
  }
}

function activatePreset(index, btnId) {
  if (!presetsList[index]) return;
  const p = presetsList[index];

  document.querySelectorAll(".preset-btn").forEach(b => b.classList.remove("active-preset"));
  const btn = document.getElementById(btnId);
  if (btn) btn.classList.add("active-preset");

  document.getElementById("patient-name").textContent = p.name;
  document.getElementById("patient-desc").textContent = p.description;

  fetchPredictions(p.data);
}

function setupModal() {
  const modal = document.getElementById("custom-modal");
  const openBtn = document.getElementById("btn-open-modal");
  const closeBtn = document.getElementById("modal-close-btn");
  const cancelBtn = document.getElementById("modal-cancel-btn");
  const form = document.getElementById("vitals-form");

  openBtn.addEventListener("click", () => modal.classList.remove("hidden"));
  closeBtn.addEventListener("click", () => modal.classList.add("hidden"));
  cancelBtn.addEventListener("click", () => modal.classList.add("hidden"));

  form.addEventListener("submit", (e) => {
    e.preventDefault();
    modal.classList.add("hidden");

    const customData = {
      Age: parseFloat(document.getElementById("input-age").value),
      Sex: document.getElementById("input-sex").value,
      BP: parseFloat(document.getElementById("input-bp").value),
      PR: parseFloat(document.getElementById("input-pr").value),
      "EF-TTE": parseFloat(document.getElementById("input-ef").value),
      "Typical Chest Pain": parseInt(document.getElementById("input-tcp").value),
      "St Elevation": parseInt(document.getElementById("input-ste").value),
      "Region RWMA": parseInt(document.getElementById("input-rwma").value),
      LDL: parseFloat(document.getElementById("input-ldl").value),
      HDL: parseFloat(document.getElementById("input-hdl").value),
      FBS: parseFloat(document.getElementById("input-fbs").value),
      "Current Smoker": parseInt(document.getElementById("input-smoker").value),
    };

    document.querySelectorAll(".preset-btn").forEach(b => b.classList.remove("active-preset"));
    document.getElementById("patient-name").textContent = "Custom Simulation Cohort";
    document.getElementById("patient-desc").textContent = "Parameters custom-configured via diagnostic modal.";

    fetchPredictions(customData);
  });
}

// --- 9. BOOTSTRAP APPLICATION ---
window.addEventListener("DOMContentLoaded", () => {
  lucide.createIcons();
  init3D();
  setupCameraControls();
  setupPresets();
  setupModal();

  document.getElementById("card-vessel-lad").addEventListener("click", () => {
    document.getElementById("cam-anterior").click();
  });
  document.getElementById("card-vessel-lcx").addEventListener("click", () => {
    document.getElementById("cam-lateral").click();
  });
  document.getElementById("card-vessel-rca").addEventListener("click", () => {
    document.getElementById("cam-inferior").click();
  });
});
