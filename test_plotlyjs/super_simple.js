

var meshx = *swapx*;
var meshy = *swapy*;
var meshz = *swapz*;
var meshi = *swapi*;
var meshj = *swapj*;
var meshk = *swapk*;


var data = [{
    type: "mesh3d",
    x: herex,
    y: herey,
    z: herez,
    i: herei,
    j: herej,
    k: herek,
    // intensity: [0, 0.33, 0.66, 1],
    // colorscale: [
    //   [0, 'rgb(255, 0, 0)'],
    //   [0.5, 'rgb(0, 255, 0)'],
    //   [1, 'rgb(0, 0, 255)']
    // ]
  }
];

  

Plotly.newPlot('myDiv', data, layout);

// inflate the mesh


