import * as THREE from 'three';
import { OBJLoader } from 'three/addons/loaders/OBJLoader.js';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GUI } from 'dat.gui';

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
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.25;
controls.enableZoom = true;
controls.enablePan = true;
controls.enableRotate = true;

// Add lighting
const ambientLight = new THREE.AmbientLight(0x404040, 2); // Soft white light
scene.add(ambientLight);

const pointLight = new THREE.PointLight(0xffffff, 1.5); // White point light
pointLight.position.set(10, 10, 10);
scene.add(pointLight);

// Load the .obj file
const loader = new OBJLoader();
loader.load('lh_pial.obj', (object) => {
  // Center the object
  object.position.set(0, 0, 0);
  scene.add(object);
}, undefined, (error) => {
  console.error("Error loading OBJ file:", error);
});

// Variables for spherical coordinates
const sphericalParams = {
  azimuth: 0, // Horizontal angle in radians
  altitude: Math.PI / 4, // Vertical angle in radians
  radius: 10 // Distance from origin
};

// Function to update camera position based on spherical coordinates
function updateCameraPosition() {
  const azimuth = sphericalParams.azimuth;
  const altitude = sphericalParams.altitude;
  const radius = sphericalParams.radius;

  // Convert spherical to Cartesian coordinates
  const x = radius * Math.sin(altitude) * Math.cos(azimuth);
  const y = radius * Math.cos(altitude);
  const z = radius * Math.sin(altitude) * Math.sin(azimuth);

  // Update camera position
  camera.position.set(x, y, z);
  camera.lookAt(scene.position); // Keep the camera looking at the center of the scene

  // Update OrbitControls target
  controls.target.set(0, 0, 0);
}

// Set up dat.GUI
const gui = new GUI();
const cameraFolder = gui.addFolder('Camera');
cameraFolder.add(sphericalParams, 'azimuth', 0, Math.PI * 2).name('Azimuth').onChange(updateCameraPosition);
cameraFolder.add(sphericalParams, 'altitude', 0, Math.PI).name('Altitude').onChange(updateCameraPosition);
cameraFolder.add(sphericalParams, 'radius', 1, 100).name('Radius').onChange(updateCameraPosition);
cameraFolder.open(); // Open the folder by default

// Render loop
function animate() {
  requestAnimationFrame(animate);

  controls.update(); // Update controls to handle panning and zooming

  renderer.render(scene, camera);
}

animate();
