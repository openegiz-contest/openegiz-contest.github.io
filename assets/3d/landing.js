// Landing page: the live mine in the hero and in the Mine Map section.
// Without WebGL the stages keep their poster image (see .mine-stage in style.css).

import { mountMine } from "./scene.js";

const still = matchMedia("(prefers-reduced-motion: reduce)").matches;

function mount(stage) {
  const canvas = stage.querySelector(".mine-canvas");
  const mine = mountMine(canvas, { framing: canvas.dataset.framing, labels: stage.querySelector(".mine-labels"), still });
  if (!mine) document.documentElement.classList.add("no-webgl");
  return mine;
}

const hero = document.querySelector(".hero .mine-stage");
if (hero) mount(hero);

// Mine Map: hovering (or focusing) a Module lights its part of the mine; when nobody
// points at the list, the Modules take turns so a phone sees them all too.
const mapStage = document.querySelector(".map-stage");
const links = [...document.querySelectorAll(".modules a[data-module]")];
if (mapStage && links.length) {
  const mine = mount(mapStage);
  const now = mapStage.querySelector(".map-now");
  let current = -1, held = false, timer = 0;

  const show = (i) => {
    current = i;
    const a = links[i];
    links.forEach((l) => l.classList.toggle("is-on", l === a));
    mapStage.dataset.module = a.dataset.module;
    now.querySelector("b").textContent = a.querySelector("b").textContent;
    now.querySelector("span").textContent = a.lastChild.textContent.trim();
    now.classList.remove("in");
    void now.offsetWidth; // restart the caption's entrance
    now.classList.add("in");
    mine?.setModule(a.dataset.module);
  };
  const cycle = () => {
    clearTimeout(timer);
    if (still) return;
    timer = setTimeout(() => (held || show((current + 1) % links.length), cycle()), 3600);
  };

  links.forEach((a, i) => {
    a.addEventListener("pointerenter", () => ((held = true), show(i)));
    a.addEventListener("focus", () => ((held = true), show(i)));
    a.addEventListener("pointerleave", () => ((held = false), cycle()));
    a.addEventListener("blur", () => ((held = false), cycle()));
  });
  show(0);
  cycle();
}
