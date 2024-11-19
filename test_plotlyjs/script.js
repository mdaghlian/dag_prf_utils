// To be filled  in
var meshx = swapx;
var meshy = swapy;
var meshz = swapz;
var meshi = swapi;
var meshj = swapj;
var meshk = swapk;


var colorSchemes = {
    // "Random Red-Blue": [
    //     'rgb(255, 0, 0)', 'rgb(0, 0, 255)', 'rgb(255, 0, 0)', 'rgb(0, 0, 255)',
    //     'rgb(255, 0, 0)', 'rgb(0, 0, 255)', 'rgb(255, 0, 0)', 'rgb(0, 0, 255)'
    // ],
    swapcol
};

var data = [{
    type: "mesh3d",
    x: meshx,
    y: meshy,
    z: meshz,
    i: meshi,
    j: meshj,
    k: meshk,
    // vertexcolor: colorSchemes["pol"],
    // intensity: [0, 0.33, 0.66, 1],
    // colorscale: [
    //   [0, 'rgb(255, 0, 0)'],
    //   [0.5, 'rgb(0, 255, 0)'],
    //   [1, 'rgb(0, 0, 255)']
    // ]
  }
];

// Function to generate update menus for each color scheme
function createColorDropdownOptions(colorSchemes) {
    return Object.keys(colorSchemes).map((schemeName, index) => ({
        label: schemeName,
        method: 'restyle',
        args: [{'vertexcolor': [colorSchemes[schemeName]]}]
    }));
}

// Layout with updatemenus for the dropdown
var layout = {
    title: "3D Mesh with Dynamic Color Dropdown",
    scene: { 
        xaxis: { title: 'X' },
        yaxis: { title: 'Y' },
        zaxis: { title: 'Z' }
    },
    updatemenus: [{
        buttons: createColorDropdownOptions(colorSchemes),
        direction: 'down',
        showactive: true,
        x: 0.1,
        xanchor: 'left',
        y: 1.1,
        yanchor: 'top'
    }]
};

Plotly.newPlot('myDiv', data, layout);

// inflate the mesh


