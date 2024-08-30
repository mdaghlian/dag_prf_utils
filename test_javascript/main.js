import * as THREE from 'three';
import { OBJLoader } from 'three/addons/loaders/OBJLoader.js';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import GUI from 'dat.gui'; // Import dat.GUI (if using a module-based setup)

// Create a scene
const scene = new THREE.Scene();

// Create a camera
const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
camera.position.z = 5;

// Create a renderer
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setClearColor(0x000000); // Background color
document.body.appendChild(renderer.domElement);

// Add OrbitControls for interaction
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.05;
controls.enableZoom = true;
controls.enablePan = true;
controls.enableRotate = true;
controls.update();

// Add some lighting
const ambientLight = new THREE.AmbientLight(0x404040, 2);
scene.add(ambientLight);

const pointLight = new THREE.PointLight(0xffffff, 1.5);
pointLight.position.set(5, 5, 5);
scene.add(pointLight);

let loadedObject; // To hold the loaded object
let originalPositions; // To hold the original vertex positions for scaling

// Load the .obj file
const loader = new OBJLoader();
loader.load('lh_pial.obj', (object) => {
  object.position.set(0, 0, 0); // Adjust position as needed
  object.scale.set(1, 1, 1); // Default scale to 1

  object.traverse((child) => {
    if (child instanceof THREE.Mesh) {
      child.material = new THREE.MeshStandardMaterial({
        color: 0xff0000,
        flatShading: true,
      });

      // Store original positions to use for scaling
      const positions = child.geometry.attributes.position;
      originalPositions = positions.array.slice(); // Deep copy of original vertex positions
    }
  });

  scene.add(object);
  loadedObject = object; // Store reference to the loaded object for later use
}, undefined, (error) => {
  console.error("Error loading OBJ file:", error);
});

// Function to update the magnitude of vertices
function updateVerticesMagnitude(scaleFactor) {
  if (!loadedObject || !originalPositions) return; // Check if the object is loaded and original positions are available

  loadedObject.traverse((child) => {
    if (child instanceof THREE.Mesh) {
      const positions = child.geometry.attributes.position;

      // Update vertices
      for (let i = 0; i < positions.count; i++) {
        positions.setXYZ(
          i,
          originalPositions[i * 3] * scaleFactor,     // X coordinate
          originalPositions[i * 3 + 1] * scaleFactor, // Y coordinate
          originalPositions[i * 3 + 2] * scaleFactor  // Z coordinate
        );
      }
      positions.needsUpdate = true; // Inform Three.js to update the buffer attribute
      child.geometry.computeBoundingSphere(); // Recompute bounding sphere for proper rendering
    }
  });
}

// Set up dat.GUI
const gui = new GUI();
const cameraFolder = gui.addFolder('Camera');
cameraFolder.add(camera.position, 'x', -10, 10).name('Position X').listen();
cameraFolder.add(camera.position, 'y', -10, 10).name('Position Y').listen();
cameraFolder.add(camera.position, 'z', 0, 20).name('Position Z').listen();
cameraFolder.open(); // Open the folder by default

const objectFolder = gui.addFolder('Object Scaling');
const scalingController = objectFolder.add({ scale: 1 }, 'scale', 0.1, 10).name('Scale Magnitude');
scalingController.onChange(updateVerticesMagnitude);
objectFolder.open();

// Render loop
function animate() {
  requestAnimationFrame(animate);

  controls.update(); // Update OrbitControls

  renderer.render(scene, camera);
}
animate();
