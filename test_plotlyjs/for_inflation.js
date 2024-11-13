

var meshx = *swapx*;
var meshy = *swapy*;
var meshz = *swapz*;
var meshi = *swapi*;
var meshj = *swapj*;
var meshk = *swapk*;

// Other coordinates
var mx0 = *swapx0*;
var my0 = *swapy0*;
var mz0 = *swapz0*;
var mx1 = *swapx1*;
var my1 = *swapy1*;
var mz1 = *swapz1*;
var mx2 = *swapx2*;
var my2 = *swapy2*;
var mz2 = *swapz2*;



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

// Layout with slider
var layout = {
    sliders: [{
      active: 5,
      currentvalue: {
        prefix: "Scale factor: ",
        font: { size: 20, color: '#666' }
      },
      pad: { t: 50 },
      steps: Array.from({ length: 20 }, (_, i) => {
        var scale = 0.5 + i * 0.1;
        return {
          label: scale.toFixed(1),
          method: 'restyle',
          args: [
            {
              x: [herex.map(v => v * scale)],
              y: [herey.map(v => v * scale)],
              z: [herez.map(v => v * scale)]
            }
          ]
        };
      })
    }]
  };
  

Plotly.newPlot('myDiv', data, layout);

// inflate the mesh


