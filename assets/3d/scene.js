// The Example Mine as a live web scene: the 007 video's diorama (mine.js), lit and graded
// the same way, but driven by requestAnimationFrame and sized to its container.
// mountMine(canvas, opts) is used twice on the landing: the hero and the Mine Map section,
// where setModule(id) highlights the part of the mine a Module works on.

import * as THREE from "three";
import { RoomEnvironment } from "three/addons/environments/RoomEnvironment.js";
import { EffectComposer } from "three/addons/postprocessing/EffectComposer.js";
import { RenderPass } from "three/addons/postprocessing/RenderPass.js";
import { UnrealBloomPass } from "three/addons/postprocessing/UnrealBloomPass.js";
import { OutputPass } from "three/addons/postprocessing/OutputPass.js";
import { makeMaterials, makeTerrain, makeHaulTruck, makeExcavator, makeCrusher, makeLightTower, makeSolarPanel, placeOnRoad } from "./mine.js";

const DEG = Math.PI / 180;
const TEAL = "#2fe8ce", AMBER = "#f2b234", HOT = "#ff6a3d";
const clamp01 = (x) => Math.min(1, Math.max(0, x));
const lerp = (a, b, u) => a + (b - a) * u;
const smooth = (u) => u * u * (3 - 2 * u);
const inOut = (u) => (u < 0.5 ? 4 * u * u * u : 1 - Math.pow(-2 * u + 2, 3) / 2);

// ---- graphics budget: the landing must stay cool on a fan-less laptop and usable on an
// office one. Quality tiers, cheapest first: a tier caps the pixel ratio and the canvas's
// pixel count and switches the costly passes. A scene starts at tier 2 (1 on phones and
// weak GPUs) and steps down while its frames run late; below tier 0 it freezes into a
// still frame that still turns on drag. Tier 3 is the original look, reached by ?gfx=3 only.
const TIERS = [
  { dpr: 1, px: 0.6e6, msaa: 0, bloom: false, shadow: 0 },
  { dpr: 1.25, px: 1.0e6, msaa: 0, bloom: true, shadow: 1024 },
  { dpr: 1.5, px: 1.6e6, msaa: 2, bloom: true, shadow: 1024 },
  { dpr: 2, px: 4.5e6, msaa: 4, bloom: true, shadow: 2048 },
];
const FPS = 30; // the scene moves slowly: 30 frames look the same as 120 at a quarter of the work
const WARMUP = 1000, WINDOW = 2000; // ms: ignored after a start or a step down (shader compiles), then measured
// ?gfx=0..3 fixes a tier (no adapting), ?gfx=still a still frame: for checking how weak devices see it
const GFX_PARAM = new URLSearchParams(location.search).get("gfx");
const FORCED = /^[0-3]$/.test(GFX_PARAM ?? "") ? +GFX_PARAM : GFX_PARAM === "still" ? 0 : undefined;
// shared by the hero and the Mine Map: what one learns about the device, the other starts from
const GFX = { tier: 2, frozen: GFX_PARAM === "still" };

/** The GPU's name: Chrome masks gl.RENDERER, Firefox deprecates the debug extension. */
function gpuName(gl) {
  try {
    const name = gl.getParameter(gl.RENDERER) || "";
    if (!/^webkit/i.test(name)) return name;
    const ext = gl.getExtension("WEBGL_debug_renderer_info");
    return (ext && gl.getParameter(ext.UNMASKED_RENDERER_WEBGL)) || name;
  } catch {
    return "";
  }
}

// Neutral tone mapping crushes very dark values (y = 6.25 x^2 below 0.08); feed the inverse
// so the backdrop lands exactly on the page's navy and the canvas edge disappears.
const invNeutral = (hex) => {
  const c = new THREE.Color(hex);
  const f = (v) => (v < 0.04 ? Math.sqrt(v / 6.25) : v + 0.04);
  return new THREE.Color(f(c.r), f(c.g), f(c.b));
};

/** Additive fresnel shell: the "selected in the twin" glow drawn over a mesh. */
function glowMaterial(color) {
  return new THREE.ShaderMaterial({
    transparent: true,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
    uniforms: { uColor: { value: new THREE.Color(color) }, uAlpha: { value: 0 } },
    vertexShader: `
      varying vec3 vN; varying vec3 vV;
      void main() {
        vec4 p = vec4(position, 1.0); vec3 n = normal;
        #ifdef USE_INSTANCING
          p = instanceMatrix * p; n = mat3(instanceMatrix) * n;
        #endif
        vec4 mv = modelViewMatrix * p;
        vN = normalize(normalMatrix * n); vV = normalize(-mv.xyz);
        gl_Position = projectionMatrix * mv;
      }`,
    fragmentShader: `
      uniform vec3 uColor; uniform float uAlpha; varying vec3 vN; varying vec3 vV;
      void main() {
        float f = pow(1.0 - abs(dot(normalize(vN), normalize(vV))), 2.2);
        gl_FragColor = vec4(uColor * (0.18 + 2.4 * f) * uAlpha, 1.0);
      }`,
  });
}

/** Give every mesh under obj a glow shell (a child, so it follows poses). Returns the material. */
function addGlow(obj, color) {
  const mat = glowMaterial(color);
  const meshes = [];
  obj.traverse((n) => n.isMesh && !n.userData.glow && meshes.push(n));
  for (const m of meshes) {
    const shell = m.isInstancedMesh ? new THREE.InstancedMesh(m.geometry, mat, m.count) : new THREE.Mesh(m.geometry, mat);
    if (m.isInstancedMesh) shell.instanceMatrix.copy(m.instanceMatrix);
    shell.userData.glow = true;
    shell.renderOrder = 20;
    shell.visible = false;
    m.add(shell);
    (mat.userData.shells ??= []).push(shell);
  }
  return mat;
}
function setGlow(mat, a) {
  mat.uniforms.uAlpha.value = a;
  for (const s of mat.userData.shells || []) s.visible = a > 0.005;
}

/**
 * Mount the mine on a canvas. opts:
 *   framing  "hero" | "map"   how much of the frame the tile fills
 *   labels   element          container for the pinned DOM labels (absolute, over the canvas)
 *   still    boolean          draw one frame and stop (reduced motion)
 * Returns { setModule(id|null), ready } or null when WebGL is unavailable.
 */
export function mountMine(canvas, opts = {}) {
  const framing = opts.framing ?? "hero";
  const small = matchMedia("(max-width: 860px), (pointer: coarse)").matches;

  let renderer;
  try {
    // without a real GPU (software rendering) the page keeps its poster instead
    renderer = new THREE.WebGLRenderer({ canvas, antialias: false, failIfMajorPerformanceCaveat: FORCED === undefined });
  } catch (err) {
    console.warn("[mine] WebGL unavailable:", err);
    return null;
  }
  const gpu = gpuName(renderer.getContext());
  if (FORCED === undefined && /swiftshader|llvmpipe|softpipe|software|basic render/i.test(gpu)) {
    renderer.dispose();
    return null;
  }
  const weak = small || /intel|mali|adreno|powervr|videocore/i.test(gpu) || (navigator.hardwareConcurrency || 8) <= 4 || (navigator.deviceMemory || 8) <= 4;
  let tier = FORCED ?? Math.min(GFX.tier, weak ? 1 : 2);
  let frozen = GFX.frozen;
  const isStill = () => opts.still || frozen;
  let dpr = 1;
  renderer.toneMapping = THREE.NeutralToneMapping;
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFShadowMap; // with shadow.radius it softens edges for less than PCFSoft

  const scene = new THREE.Scene();
  const BG = opts.background ?? "#0a1020";
  scene.background = invNeutral(BG);
  const camera = new THREE.PerspectiveCamera(21, 1, 1, 400);

  // ---- light: a studio environment (stands in for the video's HDRI), warm sun, blue rim
  const pmrem = new THREE.PMREMGenerator(renderer);
  scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
  scene.environmentIntensity = 0.5;
  scene.environmentRotation.y = 210 * DEG;
  pmrem.dispose();

  const EXT = 13;
  const sun = new THREE.DirectionalLight("#ffe6c4", 3.6);
  sun.position.set(-7, 6.5, 1.5).normalize().multiplyScalar(EXT * 2);
  Object.assign(sun.shadow.camera, { left: -EXT, right: EXT, top: EXT, bottom: -EXT, near: 0.1, far: EXT * 5 });
  sun.shadow.radius = 3;
  sun.shadow.bias = -0.0004;
  sun.shadow.normalBias = 0.02;
  scene.add(sun, sun.target);
  const rim = new THREE.DirectionalLight("#6aa8ff", 1.6);
  rim.position.set(-4, 5, -12);
  scene.add(rim);

  // ---- backdrop: a glow under the tile and a grid floor that fades out
  const floorMat = new THREE.ShaderMaterial({
    transparent: true,
    depthWrite: false,
    uniforms: { uGlow: { value: invNeutral("#16264a") }, uLine: { value: new THREE.Color("#3f8cf2") }, uLine2: { value: new THREE.Color(TEAL) } },
    vertexShader: `varying vec3 vW; void main(){ vec4 w = modelMatrix * vec4(position,1.0); vW = w.xyz; gl_Position = projectionMatrix * viewMatrix * w; }`,
    fragmentShader: `
      uniform vec3 uGlow, uLine, uLine2; varying vec3 vW;
      float gridLine(vec2 p, float s){ vec2 g = abs(fract(p/s - 0.5) - 0.5) / fwidth(p/s); return 1.0 - min(min(g.x,g.y),1.0); }
      void main(){
        float r = length(vW.xz);
        float glow = exp(-r*r/420.0);
        float l = gridLine(vW.xz, 2.0) * 0.55 + gridLine(vW.xz, 10.0) * 0.6;
        float fade = exp(-r*r/900.0) * smoothstep(12.0, 16.0, max(abs(vW.x), abs(vW.z)) + 3.0);
        vec3 lc = mix(uLine, uLine2, smoothstep(-20.0, 20.0, vW.x - vW.z));
        gl_FragColor = vec4(uGlow * glow + lc * l * fade * 0.16, max(glow * 0.9, l * fade * 0.5));
      }`,
  });
  const floor = new THREE.Mesh(new THREE.PlaneGeometry(160, 160), floorMat);
  floor.rotation.x = -Math.PI / 2;
  scene.add(floor);

  // ---- the mine, placed as in the video's style frame
  const mats = makeMaterials();
  const terrain = makeTerrain(mats, {});
  floor.position.y = terrain.base - 0.02;
  scene.add(terrain.group);
  const catcher = new THREE.Mesh(new THREE.PlaneGeometry(60, 60), new THREE.ShadowMaterial({ opacity: 0.5 }));
  catcher.rotation.x = -Math.PI / 2;
  catcher.position.y = terrain.base - 0.01;
  catcher.receiveShadow = true;
  catcher.renderOrder = 1;
  scene.add(catcher);
  const shadows = (o) => (o.traverse((n) => n.isMesh && ((n.castShadow = true), (n.receiveShadow = true))), o);
  terrain.group.traverse((n) => n.isMesh && (n.receiveShadow = true));

  const AZ = 42 * DEG;
  const screenRight = new THREE.Vector3(Math.cos(AZ), 0, -Math.sin(AZ));
  const truck1 = shadows(makeHaulTruck(mats, { loaded: false, seed: 4 }));
  truck1.position.set(terrain.center.x + screenRight.x * 0.95 - 0.3, -terrain.depth, terrain.center.y + screenRight.z * 0.95 - 0.3);
  truck1.rotation.set(0, AZ + 0.35, 0);
  truck1.updateMatrixWorld(true);
  scene.add(truck1);
  const exc = makeExcavator(mats);
  const bodyC = truck1.localToWorld(new THREE.Vector3(0, 0, -0.45));
  exc.group.position.copy(bodyC).addScaledVector(screenRight, -3.2);
  exc.group.position.y = -terrain.depth;
  exc.group.rotation.y = Math.atan2(bodyC.x - exc.group.position.x, bodyC.z - exc.group.position.z);
  scene.add(shadows(exc.group));
  const truck2 = shadows(makeHaulTruck(mats, { loaded: true, beacon: true, seed: 9 }));
  scene.add(truck2);

  const crusher = makeCrusher(mats, { side: 1 });
  const dumpEnd = terrain.haul.getPointAt(1), preDump = terrain.haul.getPointAt(0.97);
  crusher.position.set(dumpEnd.x, 0, dumpEnd.z);
  crusher.rotation.y = Math.atan2(preDump.x - dumpEnd.x, preDump.z - dumpEnd.z);
  crusher.position.addScaledVector(new THREE.Vector3(Math.sin(crusher.rotation.y), 0, Math.cos(crusher.rotation.y)), -1.9);
  scene.add(shadows(crusher));

  const towers = new THREE.Group();
  for (const [r, th] of [[1.12, 2.75], [1.12, 4.0], [1.16, 5.3]]) {
    const p = terrain.fromPolar(r, th);
    const lt = shadows(makeLightTower(mats));
    lt.position.set(p.x, terrain.heightAt(p.x, p.y), p.y);
    lt.rotation.y = Math.atan2(terrain.center.x - p.x, terrain.center.y - p.y);
    towers.add(lt);
  }
  scene.add(towers);

  // ---- data layer: the haul route with packets flowing to the crusher
  const ROUTE_SEG = 420;
  const flowMat = (color, dashes) => new THREE.ShaderMaterial({
    transparent: true,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
    uniforms: { uTime: { value: 0 }, uAlpha: { value: 1 }, uColor: { value: new THREE.Color(color) } },
    vertexShader: `varying vec2 vUv; void main(){ vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position,1.0); }`,
    fragmentShader: `
      uniform float uTime, uAlpha; uniform vec3 uColor; varying vec2 vUv;
      void main(){
        float dash = smoothstep(0.55, 1.0, fract(vUv.x * ${dashes.toFixed(1)} - uTime * 0.9));
        gl_FragColor = vec4(uColor * (0.5 + 2.6 * dash) * uAlpha, 1.0);
      }`,
  });
  const routeMat = flowMat(TEAL, 46);
  const routeGeo = new THREE.TubeGeometry(terrain.haul, ROUTE_SEG, 0.06, 8, false);
  routeGeo.translate(0, 0.07, 0);
  const route = new THREE.Mesh(routeGeo, routeMat);
  route.renderOrder = 5;
  scene.add(route);

  // tracking ring under truck-02 and telemetry packets rising from its beacon
  const ringMat = new THREE.MeshBasicMaterial({ color: new THREE.Color(TEAL).multiplyScalar(2.2), transparent: true, opacity: 0.9, side: THREE.DoubleSide, depthWrite: false });
  const ring = new THREE.Mesh(new THREE.RingGeometry(1.45, 1.53, 96), ringMat);
  ring.rotation.x = -Math.PI / 2;
  scene.add(ring);
  const packetMat = new THREE.MeshStandardMaterial({ color: "#000000", emissive: "#7ff3e4", emissiveIntensity: 6 });
  const packets = Array.from({ length: 3 }, () => {
    const m = new THREE.Mesh(new THREE.BoxGeometry(0.16, 0.16, 0.16), packetMat);
    scene.add(m);
    return m;
  });

  // ---- Module overlays (Mine Map section): each lights up the part of the mine it works on
  const glow = {
    transport: [addGlow(truck1, TEAL), addGlow(truck2, TEAL)],
    processing: [addGlow(crusher, TEAL)],
    maintenance: [addGlow(exc.group, HOT)],
    safety: [addGlow(towers, TEAL)],
    twin: [addGlow(terrain.group, TEAL)],
  };
  const solar = new THREE.Group();
  for (const [x, z] of [[5.5, -2.3], [4.3, -4.9]]) { // the free plateau behind the haul road
    const p = shadows(makeSolarPanel(mats));
    p.position.set(x, terrain.heightAt(x, z), z);
    p.rotation.y = 42 * DEG + 0.15;
    solar.add(p);
  }
  scene.add(solar);
  const crusherAt = crusher.getWorldPosition(new THREE.Vector3());
  const lineEnd = crusherAt.clone().lerp(new THREE.Vector3(5.5, 0, 0), 0.22);
  const lineCurve = new THREE.CatmullRomCurve3([[5.3, -1.4], [5.7, 0.6], [5.0, 2.4], [lineEnd.x, lineEnd.z]].map(([x, z]) => new THREE.Vector3(x, terrain.heightAt(x, z) + 0.08, z)));
  const powerMat = flowMat(AMBER, 26);
  const power = new THREE.Mesh(new THREE.TubeGeometry(lineCurve, 120, 0.05, 8, false), powerMat);
  power.renderOrder = 5;
  scene.add(power);
  glow.energy = [addGlow(solar, AMBER), addGlow(towers, AMBER)];

  // geology: the ore block model under the pit, drawn through the ground as mine software does
  const blockPts = [], blockGrade = [];
  {
    const rand = Math.sin;
    for (let x = -6.4; x <= 6.4; x += 0.42) for (let z = -6.4; z <= 6.4; z += 0.42) {
      const { r } = terrain.polar(x, z);
      if (r > 1.05) continue;
      const top = terrain.heightAt(x, z);
      for (let k = 1; k <= 3; k++) {
        const y = top - k * 0.42;
        if (y < terrain.base + 0.3) continue;
        const g = 0.5 + 0.5 * rand(x * 0.7 + 1.1) * rand(z * 0.6 - 0.4 + y * 0.8); // smooth fake grade
        blockPts.push(x, y, z);
        blockGrade.push(g * (1 - 0.5 * r));
      }
    }
  }
  const blockGeo = new THREE.BufferGeometry();
  blockGeo.setAttribute("position", new THREE.Float32BufferAttribute(blockPts, 3));
  blockGeo.setAttribute("grade", new THREE.Float32BufferAttribute(blockGrade, 1));
  const blockMat = new THREE.ShaderMaterial({
    transparent: true,
    depthWrite: false,
    depthTest: false,
    blending: THREE.AdditiveBlending,
    uniforms: { uAlpha: { value: 0 }, uLow: { value: new THREE.Color("#3f8cf2") }, uMid: { value: new THREE.Color(TEAL) }, uHigh: { value: new THREE.Color(AMBER) }, uPx: { value: 1 } },
    vertexShader: `
      attribute float grade; varying float vG; uniform float uPx;
      void main(){ vG = grade; vec4 mv = modelViewMatrix * vec4(position,1.0); gl_PointSize = uPx * (2.0 + 5.0 * grade); gl_Position = projectionMatrix * mv; }`,
    fragmentShader: `
      uniform float uAlpha; uniform vec3 uLow, uMid, uHigh; varying float vG;
      void main(){
        vec2 q = abs(gl_PointCoord - 0.5); if (max(q.x, q.y) > 0.42) discard;
        vec3 c = vG < 0.5 ? mix(uLow, uMid, vG * 2.0) : mix(uMid, uHigh, vG * 2.0 - 1.0);
        gl_FragColor = vec4(c * (0.4 + 1.6 * vG) * uAlpha, 1.0);
      }`,
  });
  const blocks = new THREE.Points(blockGeo, blockMat);
  blocks.visible = false;
  blocks.renderOrder = 30;
  scene.add(blocks);

  // drilling and blasting: a hole pattern on the far bench, fired row by row (delay timing)
  const holeMat = new THREE.MeshBasicMaterial({ transparent: true, depthWrite: false, blending: THREE.AdditiveBlending });
  const HOLES = [];
  for (let i = 0; i < 9; i++) for (let j = 0; j < 4; j++) HOLES.push({ th: 3.2 + i * 0.085 + (j % 2) * 0.042, r: 1.2 + j * 0.075, delay: i * 0.12 + j * 0.05 });
  const holes = new THREE.InstancedMesh(new THREE.CylinderGeometry(0.075, 0.075, 0.2, 12), holeMat, HOLES.length);
  {
    const m = new THREE.Matrix4();
    HOLES.forEach((h, k) => {
      const p = terrain.fromPolar(h.r, h.th);
      m.makeTranslation(p.x, terrain.heightAt(p.x, p.y) + 0.02, p.y);
      holes.setMatrixAt(k, m);
      holes.setColorAt(k, new THREE.Color(0));
    });
  }
  holes.visible = false;
  holes.renderOrder = 25;
  scene.add(holes);
  const holeColor = new THREE.Color(), amberHot = new THREE.Color(AMBER).multiplyScalar(2.6), hot = new THREE.Color(HOT).multiplyScalar(3.2);

  // safety: a keep-out circle around the working excavator
  const zoneMat = new THREE.MeshBasicMaterial({ color: new THREE.Color(HOT).multiplyScalar(1.6), transparent: true, opacity: 0, side: THREE.DoubleSide, depthWrite: false });
  zoneMat.depthTest = false;
  const zone = new THREE.Mesh(new THREE.RingGeometry(2.55, 2.68, 96), zoneMat);
  zone.renderOrder = 25;
  zone.rotation.x = -Math.PI / 2;
  zone.position.copy(exc.group.position).add(new THREE.Vector3(0, 0.05, 0));
  scene.add(zone);

  // your own Module: an empty slot waiting for something new
  const slotMat = new THREE.LineBasicMaterial({ color: new THREE.Color(TEAL).multiplyScalar(2.2), transparent: true, opacity: 0 });
  const slot = new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.BoxGeometry(1.6, 1.2, 1.6)), slotMat);
  slot.position.set(4.7, 0, -4.3);
  slot.position.y = terrain.heightAt(slot.position.x, slot.position.z) + 0.6;
  slot.visible = false;
  scene.add(slot);

  // Camera subject per Module: [world point, zoom (fov factor)]
  const pos = (o, y = 0.8) => o.getWorldPosition(new THREE.Vector3()).setY(o.position.y + y);
  const pitC = new THREE.Vector3(terrain.center.x, -terrain.depth * 0.6, terrain.center.y);
  const SUBJECT = {
    transport: () => [terrain.haul.getPointAt(0.55).setY(0), 0.86],
    geology: () => [pitC, 0.78],
    blasting: () => [new THREE.Vector3().setFromMatrixPosition(new THREE.Matrix4().fromArray(holes.instanceMatrix.array, 18 * 16)), 0.72],
    safety: () => [pos(exc.group, 0), 0.82],
    energy: () => [new THREE.Vector3(4.6, 0.6, -2.4), 0.8],
    processing: () => [pos(crusher, 1.0), 0.7],
    maintenance: () => [pos(exc.group, 1.0), 0.6],
    dashboards: () => [new THREE.Vector3(0.3, 0.6, 0), 0.95],
    visualization: () => [new THREE.Vector3(0.3, 0.6, 0), 0.95],
    own: () => [slot.position.clone(), 0.7],
  };

  // ---- DOM labels pinned to 3D points
  const labelsEl = opts.labels;
  const LABELS = [
    { id: "exc", text: "excavator-01", at: () => exc.group.localToWorld(new THREE.Vector3(0, 2.6, -0.4)) },
    { id: "truck", text: "truck-02", at: () => truck2.userData.beacon.getWorldPosition(new THREE.Vector3()).add(new THREE.Vector3(0, 0.25, 0)) },
    { id: "crusher", text: "crusher-01", at: () => crusher.localToWorld(new THREE.Vector3(0, 2.7, -1.75)) },
  ];
  if (labelsEl) {
    for (const l of LABELS) {
      const el = document.createElement("span");
      el.className = "mine-label";
      el.innerHTML = `<i></i>${l.text}`;
      labelsEl.append(el);
      l.el = el;
    }
  }

  // ---- post: MSAA render -> bloom -> tone map/sRGB (grain and vignette live in CSS)
  const rt = new THREE.WebGLRenderTarget(4, 4, { type: THREE.HalfFloatType, samples: TIERS[tier].msaa });
  const composer = new EffectComposer(renderer, rt);
  composer.addPass(new RenderPass(scene, camera));
  const bloom = new UnrealBloomPass(new THREE.Vector2(4, 4), 0.65, 0.55, 1.05);
  composer.addPass(bloom);
  composer.addPass(new OutputPass());

  // ---- framing: fit the tile into the canvas whatever its aspect
  const DIST = 99;
  const FIT = framing === "hero" ? (small ? 10.4 : 13.2) : small ? 9.2 : 11.2; // world half-size kept in view
  let W = 1, H = 1, drawn = false;
  function resize() {
    const w = canvas.clientWidth, h = canvas.clientHeight;
    if (!w || !h || (w === W && h === H)) return;
    W = w;
    H = h;
    const q = TIERS[tier];
    dpr = Math.min(devicePixelRatio || 1, q.dpr, Math.max(0.75, Math.sqrt(q.px / (W * H))));
    renderer.setPixelRatio(dpr);
    renderer.setSize(W, H, false);
    composer.setPixelRatio(dpr);
    composer.setSize(W, H);
    camera.aspect = W / H;
    // hero on a wide screen: push the mine toward the right edge, away from the headline
    // and up, so it sits level with the text rather than behind the facts strip
    // (the Mine Map stage only lifts it clear of its caption)
    const shift = framing === "hero" && !small ? Math.round(W * 0.07) : 0;
    const lift = Math.round(H * (framing === "hero" ? (small ? 0 : 0.07) : 0.05));
    if (shift || lift) camera.setViewOffset(W, H, -shift, lift, W, H);
    else camera.clearViewOffset();
    if (isStill() && drawn) draw(clock); // a resize clears the canvas and no loop redraws it
  }

  function applyTier(i) {
    tier = i;
    canvas.dataset.gfx = i; // the current tier, readable from devtools
    const q = TIERS[i];
    bloom.enabled = q.bloom;
    for (const t of [composer.renderTarget1, composer.renderTarget2]) {
      if (t.samples !== q.msaa) ((t.samples = q.msaa), t.dispose());
    }
    sun.castShadow = q.shadow > 0;
    if (q.shadow && sun.shadow.mapSize.x !== q.shadow) {
      sun.shadow.mapSize.setScalar(q.shadow);
      sun.shadow.map?.dispose();
      sun.shadow.map = null;
    }
    W = 0; // force resize() to apply the new pixel ratio
    resize();
  }
  applyTier(tier);
  if (frozen) canvas.dataset.gfx = "still";
  new ResizeObserver(resize).observe(canvas);

  // ---- interaction: drag rotates (horizontal only, so a phone still scrolls), then eases back
  let dragAz = 0, dragEl = 0, dragging = false, lastX = 0, lastY = 0, releasedAt = -1e9;
  canvas.addEventListener("pointerdown", (e) => {
    dragging = true;
    lastX = e.clientX;
    lastY = e.clientY;
    canvas.setPointerCapture(e.pointerId);
  });
  canvas.addEventListener("pointermove", (e) => {
    if (!dragging) return;
    dragAz -= (e.clientX - lastX) * 0.35;
    dragEl = Math.max(-14, Math.min(20, dragEl + (e.clientY - lastY) * 0.15));
    lastX = e.clientX;
    lastY = e.clientY;
    if (isStill()) draw(clock);
  });
  const release = () => ((dragging = false), (releasedAt = performance.now() / 1000));
  canvas.addEventListener("pointerup", release);
  canvas.addEventListener("pointercancel", release);

  // ---- module state (smoothed)
  let module = null;
  const level = {}; // id -> 0..1, eased toward the target each frame
  const camT = new THREE.Vector3(0.3, 0.6, 0), camGoal = new THREE.Vector3();
  let zoom = 1;

  // ---- per-frame state, all from t
  const haulLen = terrain.haul.getLength();
  const CYCLE = 24, GO = [0, 9], BACK = [12, 21]; // loaded run to the crusher, dump, empty run back
  const S0 = 0.3, S1 = 0.93;
  const v3 = new THREE.Vector3();
  let clock = 0, last = performance.now() / 1000;

  function draw(t) {
    const now = performance.now() / 1000;
    const dt = Math.min(0.1, now - last);
    last = now;

    // truck-02 shuttles between the pit and the crusher
    const c = t % CYCLE;
    // at each end it turns around in place while it dumps / gets loaded
    let s, turn = 0;
    if (c < GO[1]) s = lerp(S0, S1, inOut(c / GO[1]));
    else if (c < BACK[0]) ((s = S1), (turn = smooth((c - GO[1]) / (BACK[0] - GO[1]))));
    else if (c < BACK[1]) ((s = lerp(S1, S0, inOut((c - BACK[0]) / (BACK[1] - BACK[0])))), (turn = 1));
    else ((s = S0), (turn = 1 - smooth((c - BACK[1]) / (CYCLE - BACK[1]))));
    placeOnRoad(truck2, terrain, s);
    truck2.rotation.y += Math.PI * turn;
    if (turn > 0 && turn < 1) truck2.rotation.x = 0;
    else if (turn === 1) truck2.rotation.x *= -1;
    for (const w of truck2.userData.wheels) w.rotation.x = (s * haulLen) / 0.45 * (turn > 0.5 ? -1 : 1);

    const ph = (2 * Math.PI * t) / 4;
    exc.pose({ slew: 0.05 * Math.sin(ph), boom: 0.5 + 0.1 * Math.sin(ph), stick: 1.2 + 0.12 * Math.sin(ph + 1.0), bucket: 1.25 + 0.2 * Math.sin(ph + 2.0) });

    routeMat.uniforms.uTime.value = t;
    ring.position.copy(truck2.position).add(v3.set(0, 0.04, 0));
    ringMat.opacity = 0.55 + 0.35 * Math.sin(t * 2.4);
    const beacon = truck2.userData.beacon.getWorldPosition(new THREE.Vector3());
    packets.forEach((p, i) => {
      const u = (t * 0.45 + i / packets.length) % 1;
      p.position.copy(beacon).add(v3.set(0, 0.3 + u * 3.2, 0));
      p.scale.setScalar(Math.sin(Math.PI * u) * 0.9 + 0.05);
      p.rotation.set(t * 1.7 + i, t * 2.3 + i * 2, 0);
    });

    // module highlight levels ease toward the selection
    const k = 1 - Math.exp(-dt * 7);
    for (const id of Object.keys(SUBJECT)) level[id] = lerp(level[id] ?? 0, id === module ? 1 : 0, isStill() ? 1 : k);
    const pulse = 0.75 + 0.25 * Math.sin(t * 3.2);
    const L = level;
    for (const m of glow.transport) setGlow(m, L.transport * pulse);
    routeMat.uniforms.uAlpha.value = 1 + 1.4 * L.transport - 0.6 * L.visualization;
    for (const m of glow.processing) setGlow(m, L.processing * pulse);
    for (const m of glow.maintenance) setGlow(m, L.maintenance * (0.6 + 0.4 * Math.sin(t * 6)));
    setGlow(glow.safety[0], L.safety * pulse);
    setGlow(glow.energy[0], L.energy * pulse);
    setGlow(glow.energy[1], L.energy * 0.6 * pulse);
    setGlow(glow.twin[0], L.visualization * 0.55);
    for (const m of [...glow.transport, ...glow.processing]) if (L.visualization > 0.01) setGlow(m, Math.max(m.uniforms.uAlpha.value, L.visualization * 0.8));
    solar.visible = L.energy > 0.02;
    for (const p of solar.children) p.scale.setScalar(Math.max(0.001, smooth(clamp01(L.energy * 1.4))));
    power.visible = L.energy > 0.02;
    powerMat.uniforms.uTime.value = t;
    powerMat.uniforms.uAlpha.value = L.energy;
    blockMat.uniforms.uAlpha.value = L.geology;
    blockMat.uniforms.uPx.value = dpr * Math.min(W, H) / 640;
    blocks.visible = L.geology > 0.01;
    holes.visible = L.blasting > 0.01;
    if (holes.visible) {
      const cyc = (t % 3.2) - 0.6; // drilled holes glow amber, each fires in turn
      HOLES.forEach((h, k) => {
        const f = Math.exp(-Math.pow((cyc - h.delay) / 0.07, 2));
        holes.setColorAt(k, holeColor.copy(amberHot).lerp(hot, f).multiplyScalar(L.blasting * (0.55 + 1.6 * f)));
      });
      holes.instanceColor.needsUpdate = true;
    }

    // focus: when a Module is shown, the daylight dims so its glow reads (as in the twin views)
    const focus = Math.max(...Object.entries(L).map(([id, v]) => (id === "dashboards" ? 0.4 * v : v)), 0);
    sun.intensity = 3.6 * (1 - 0.55 * focus);
    rim.intensity = 1.6 * (1 - 0.3 * focus);
    scene.environmentIntensity = 0.5 * (1 - 0.5 * focus);
    zoneMat.opacity = L.safety * pulse;
    zone.visible = L.safety > 0.01;
    slotMat.opacity = L.own * (0.5 + 0.5 * Math.sin(t * 3));
    slot.visible = L.own > 0.01;
    slot.rotation.y = t * 0.4;

    // camera: gentle sway around the style-frame angle, drag offset easing back after release
    const idle = clamp01((now - releasedAt - 2.5) / 3);
    if (!dragging && idle > 0) {
      const back = 1 - Math.exp(-dt * 1.2 * idle);
      dragAz = lerp(dragAz, 0, back);
      dragEl = lerp(dragEl, 0, back);
    }
    const sway = opts.still ? 0 : 12 * Math.sin(t * 0.07);
    const az = (42 + sway + dragAz) * DEG, el = (36 + dragEl) * DEG;
    camGoal.set(0.3, 0.6, 0);
    let zGoal = 1;
    if (module && SUBJECT[module]) {
      const [p, z] = SUBJECT[module]();
      camGoal.lerp(p, 0.55);
      zGoal = z;
    }
    camT.lerp(camGoal, isStill() ? 1 : 1 - Math.exp(-dt * 2.2));
    zoom = lerp(zoom, zGoal, isStill() ? 1 : 1 - Math.exp(-dt * 2.2));
    const half = FIT * zoom;
    const vHalf = camera.aspect >= 1 ? half : half / camera.aspect;
    camera.fov = (2 * Math.atan(vHalf / DIST)) / DEG;
    camera.position.set(Math.cos(el) * Math.sin(az), Math.sin(el), Math.cos(el) * Math.cos(az)).multiplyScalar(DIST).add(camT);
    camera.lookAt(camT);
    camera.updateProjectionMatrix();
    camera.updateMatrixWorld();

    composer.render(); // OutputPass tone maps everything, the custom shaders included

    for (const l of LABELS) {
      if (!l.el) continue;
      const p = l.at().project(camera);
      l.el.style.transform = `translate3d(${(((p.x + 1) / 2) * W).toFixed(1)}px, ${(((1 - p.y) / 2) * H).toFixed(1)}px, 0)`;
      l.el.style.visibility = p.z < 1 ? "" : "hidden";
    }
  }

  // ---- loop: runs only while on screen and the tab is visible, at most FPS frames a second.
  // The time between drawn frames is the honest load signal: GPU work is asynchronous, so
  // timing draw() itself shows only the CPU side, while an overloaded GPU delays the frames.
  let onScreen = false, raf = 0, lastDraw = 0, winStart = 0, winDraws = 0;
  const restartWindow = () => ((winStart = performance.now() + WARMUP), (winDraws = 0));
  const t0 = performance.now() / 1000;
  function frame() {
    raf = 0;
    if (!onScreen || document.hidden || frozen) return;
    raf = requestAnimationFrame(frame);
    const a = performance.now();
    const gap = a - lastDraw;
    if (gap < 1000 / FPS - 4) return; // skip display frames beyond FPS (60/120 Hz screens)
    lastDraw = a;
    clock = a / 1000 - t0;
    draw(clock);
    if (FORCED !== undefined || a < winStart) return;
    if (gap > 1000) return restartWindow(); // a stall (page jank) says nothing about the GPU
    winDraws++;
    const span = a - winStart;
    if (span < WINDOW) return;
    const fps = (winDraws * 1000) / span;
    if (fps < 22) stepDown(fps < 12 ? 2 : 1);
    else ((winStart = a), (winDraws = 0));
  }
  function stepDown(steps) {
    restartWindow();
    if (tier > 0) applyTier(Math.max(0, tier - steps));
    else ((frozen = true), (canvas.dataset.gfx = "still"));
    GFX.tier = Math.min(GFX.tier, tier);
    GFX.frozen ||= frozen;
  }
  const kick = () => {
    if (GFX.tier < tier && FORCED === undefined) applyTier(GFX.tier); // the other scene already learned this device is slower
    if (GFX.frozen && !frozen) ((frozen = true), (canvas.dataset.gfx = "still"), draw(clock));
    if (opts.still || frozen || raf || !onScreen || document.hidden) return;
    restartWindow();
    last = performance.now() / 1000;
    raf = requestAnimationFrame(frame);
  };
  new IntersectionObserver(([e]) => ((onScreen = e.isIntersecting), kick()), { rootMargin: "100px" }).observe(canvas);
  document.addEventListener("visibilitychange", kick);

  renderer.compile(scene, camera);
  clock = 5.2;
  draw(clock);
  drawn = true;
  canvas.classList.add("is-live");

  return {
    setModule(id) {
      module = id;
      if (isStill()) draw(clock);
    },
  };
}
