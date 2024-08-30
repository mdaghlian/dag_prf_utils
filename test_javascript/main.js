import * as THREE from 'three';
import { OBJLoader } from 'three/addons/loaders/OBJLoader.js';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js'; // Import OrbitControls

// Create a scene
const scene = new THREE.Scene();

// Create a camera
const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
camera.position.z = 5;

// Create a renderer
const renderer = new THREE.WebGLRenderer({ antialias: true }); // Enable antialiasing for smoother edges
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setClearColor(0x000000); // Background color
document.body.appendChild(renderer.domElement);

// Add OrbitControls for interaction
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true; // Optional: Add damping (inertia) for smoother interaction
controls.dampingFactor = 0.05; // Optional: Damping factor for smoothness
controls.enableZoom = true; // Enable zoom
controls.enablePan = true; // Enable panning
controls.enableRotate = true; // Enable rotation
controls.update(); // Update controls

// Add some lighting
const ambientLight = new THREE.AmbientLight(0x404040, 2); // Intensity increased
scene.add(ambientLight);

const pointLight = new THREE.PointLight(0xffffff, 1.5); // Intensity increased
pointLight.position.set(5, 5, 5);
scene.add(pointLight);

let loadedObject; // To hold the loaded object

// Load the .obj file
const loader = new OBJLoader();
loader.load('lh_pial.obj', (object) => {
  console.log("Object loaded successfully!", object); // Debugging
  object.position.set(0, 0, 0); // Adjust position as needed
  object.scale.set(1, 1, 1); // Default scale to 1

  object.traverse((child) => {
    if (child instanceof THREE.Mesh) {
      child.material = new THREE.MeshStandardMaterial({
        color: 0xff0000,
        flatShading: true,
      });
      child.geometry.computeBoundingBox(); // Compute bounding box for geometry scaling
    }
  });

  scene.add(object);
  loadedObject = object; // Store reference to the loaded object for later use
}, undefined, (error) => {
  console.error("Error loading OBJ file:", error);
});

// Function to update the magnitude of vertices
function updateVerticesMagnitude(scaleFactor) {
  if (!loadedObject) return; // Check if the object is loaded

  loadedObject.traverse((child) => {
    if (child instanceof THREE.Mesh) {
      const positions = child.geometry.attributes.position;
      const originalPositions = positions.array.slice(); // Make a copy of original positions

      // Update vertices
      for (let i = 0; i < positions.count; i++) {
        positions.setXYZ(
          i,
          originalPositions[i * 3] * scaleFactor,   // X coordinate
          originalPositions[i * 3 + 1] * scaleFactor, // Y coordinate
          originalPositions[i * 3 + 2] * scaleFactor  // Z coordinate
        );
      }
      positions.needsUpdate = true; // Inform Three.js to update the buffer attribute
    }
  });
}

// Add a slider to change the magnitude of vertices
const slider = document.createElement('input');
slider.type = 'range';
slider.min = '0.1';
slider.max = '10';
slider.step = '0.1';
slider.value = '1';
slider.id = 'magnitudeSlider';
slider.style.position = 'absolute';
slider.style.top = '10px';
slider.style.left = '10px';
slider.style.zIndex = '1';
document.body.appendChild(slider);

// Event listener for the slider
slider.addEventListener('input', (event) => {
  const scaleFactor = parseFloat(event.target.value);
  updateVerticesMagnitude(scaleFactor);
});

// Render loop
function animate() {
  requestAnimationFrame(animate);

  controls.update(); // Update OrbitControls

  renderer.render(scene, camera);
}
animate();
