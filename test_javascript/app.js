// import * as THREE from 'three';
import * as THREE from './resources/js/three.js'
import { OBJLoader } from './resources/js/OBJLoader.js';
// import { OBJLoader } from 'three/addons/loaders/OBJLoader.js';
// import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
// import { GUI } from 'dat.gui';
// import { LandscapeControls } from './resources/js/LandscapeControls'
// Create a scene
const scene = new THREE.Scene();

// Create a camera
const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);

// Create a renderer
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setClearColor(0x000000); // Background color
document.body.appendChild(renderer.domElement);

// Add OrbitControls for interaction
// Set initial camera position
camera.position.set(5, 5, 5);

// OrbitControls setup
const controls = new THREE.LandscapeControls(renderer.domElement, camera);

// Add lighting
const ambientLight = new THREE.AmbientLight(0x404040, 2); // Soft white light
scene.add(ambientLight);

const pointLight = new THREE.PointLight(0xffffff, 1.5); // White point light
pointLight.position.set(10, 10, 10);
scene.add(pointLight);


// Variables for blending
const blendParams = {
  blendFactor: 0 // Blend factor between 0 (lh_pial) and 1 (lh_inflated)
};

// Load both .obj files
const loader = new OBJLoader();
let lhPialMesh, lhInflatedMesh, blendedGeometry;

loader.load('lh_pial.obj', (object) => {
  lhPialMesh = object.children[0].clone(); // Assuming the loaded object has a single child mesh
  lhPialMesh.material = new THREE.MeshStandardMaterial({ color: 0xff0000, flatShading: true });
  scene.add(lhPialMesh);

  // Initialize blended geometry
  if (lhInflatedMesh) {
    createBlendedGeometry(); // Create the blended geometry if both meshes are loaded
  }
}, undefined, (error) => {
  console.error("Error loading lh_pial.obj:", error);
});

loader.load('lh_inflated.obj', (object) => {
  lhInflatedMesh = object.children[0].clone(); // Assuming the loaded object has a single child mesh
  lhInflatedMesh.material = new THREE.MeshStandardMaterial({ color: 0x00ff00, flatShading: true });
  lhInflatedMesh.visible = false;
  scene.add(lhInflatedMesh);

  // Initialize blended geometry
  if (lhPialMesh) {
    createBlendedGeometry(); // Create the blended geometry if both meshes are loaded
  }
}, undefined, (error) => {
  console.error("Error loading lh_inflated.obj:", error);
});

// Function to create blended geometry
function createBlendedGeometry() {
  // Ensure geometries have the same number of vertices
  const geometry1 = lhPialMesh.geometry;
  const geometry2 = lhInflatedMesh.geometry;

  if (geometry1.attributes.position.count !== geometry2.attributes.position.count) {
    console.warn("Geometries have different vertex counts and cannot be blended.");
    return;
  }

  // Create a new geometry to hold the blended mesh
  blendedGeometry = geometry1.clone(); // Start with lh_pial geometry
  blendMeshes(blendParams.blendFactor); // Apply initial blend factor
  const blendedMesh = new THREE.Mesh(blendedGeometry, lhPialMesh.material);
  scene.add(blendedMesh);
}

// Function to blend between lh_pial and lh_inflated
function blendMeshes(factor) {
  if (!blendedGeometry) return;

  const geometry1 = lhPialMesh.geometry;
  const geometry2 = lhInflatedMesh.geometry;

  const positions1 = geometry1.attributes.position.array;
  const positions2 = geometry2.attributes.position.array;
  const blendedPositions = blendedGeometry.attributes.position.array;

  // Interpolate between the two geometries
  for (let i = 0; i < positions1.length; i++) {
    blendedPositions[i] = positions1[i] * (1 - factor) + positions2[i] * factor;
  }

  blendedGeometry.attributes.position.needsUpdate = true;
}

// Set up dat.GUI
const gui = new GUI();
const blendFolder = gui.addFolder('Blend');
blendFolder.add(blendParams, 'blendFactor', 0, 1).name('Blend Factor').onChange((value) => {
  blendMeshes(value);
});
blendFolder.open(); // Open the folder by default


// Render loop
function animate() {
  requestAnimationFrame(animate);

  controls.update(); // Update controls to handle panning and zooming

  // updateSphericalFromCamera(); // Update spherical coordinates based on camera position

  renderer.render(scene, camera);
}

animate();
