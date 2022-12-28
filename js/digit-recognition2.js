let model;

var canvasWidth = 150;
var canvasHeight = 150;
var canvasStrokeStyle = "white";
var canvasLineJoin = "round";
var canvasLineWidth = 10;
var canvasBackgroundColor = "black";
var canvasId = "canvas";

document.getElementById("chart_box").innerHTML = "";
document.getElementById("chart_box").style.display = "none";

var canvasBox = document.getElementById("canvas_box");
var canvas = document.createElement("canvas");

canvas.setAttribute("width", canvasWidth);
canvas.setAttribute("height", canvasHeight);
canvas.setAttribute("id", canvasId);
canvas.style.backgroundColor = canvasBackgroundColor;
canvasBox.appendChild(canvas);
if (typeof G_vmlCanvasManager != "undefined") {
  canvas = G_vmlCanvasManager.initElement(canvas);
}

ctx = canvas.getContext("2d");

$("#canvas").mousedown(function (e) {
  var rect = canvas.getBoundingClientRect();
  var mouseX = e.clientX - rect.left;
  var mouseY = e.clientY - rect.top;
  drawing = true;
  addUserGesture(mouseX, mouseY);
  drawOnCanvas();
});

$("#canvas").mousemove(function (e) {
  if (drawing) {
    var rect = canvas.getBoundingClientRect();
    var mouseX = e.clientX - rect.left;
    var mouseY = e.clientY - rect.top;
    addUserGesture(mouseX, mouseY, true);
    drawOnCanvas();
  }
});

$("#canvas").mouseup(function (e) {
  drawing = false;
});

$("#canvas").mouseleave(function (e) {
  drawing = false;
});

function addUserGesture(x, y, dragging) {
  horizontal.push(x);
  vertical.push(y);
  drag_mouse.push(dragging);
}

function drawOnCanvas() {
  ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);

  ctx.strokeStyle = canvasStrokeStyle;
  ctx.lineJoin = canvasLineJoin;
  ctx.lineWidth = canvasLineWidth;

  for (var i = 0; i < horizontal.length; i++) {
    ctx.beginPath();
    if (drag_mouse[i] && i) {
      ctx.moveTo(horizontal[i - 1], vertical[i - 1]);
    } else {
      ctx.moveTo(horizontal[i] - 1, vertical[i]);
    }
    ctx.lineTo(horizontal[i], vertical[i]);
    ctx.closePath();
    ctx.stroke();
  }
}

$("#clear-button").click(async function () {
  ctx.clearRect(0, 0, canvasWidth, canvasHeight);
  horizontal = new Array();
  vertical = new Array();
  drag_mouse = new Array();
  $(".prediction-text").empty();
  $("#result_box").addClass("d-none");
});

async function loadModel() {
  console.log("model loading..");

  // clear the model variable
  model = undefined;

  // load the model using a HTTPS request (where you have stored your model files)
  model = await tf.loadLayersModel("models_tfjs/model.json");

  console.log("model loaded..");
}

loadModel();

function preprocessCanvas(image) {
  // resize the input image to target size of (1, 28, 28)
  let tensor = tf.browser
    .fromPixels(image)
    .resizeNearestNeighbor([28, 28])
    .mean(2)
    .expandDims(2)
    .expandDims()
    .toFloat();
  console.log(tensor.shape);
  return tensor.div(255.0);
}

$("#predict-button").click(async function () {
  // get image data from canvas
  var imageData = canvas.toDataURL();

  // preprocess canvas
  let tensor = preprocessCanvas(canvas);

  // make predictions on the preprocessed image tensor
  let predictions = await model.predict(tensor).data();

  // get the model's prediction results
  let results = Array.from(predictions);

  // display the predictions in chart
  $("#result_box").removeClass("d-none");
  displayChart(results);
  displayLabel(results);

  console.log(results);
});

var chart = "";
var firstTime = 0;
function loadChart(label, data, modelSelected) {
  var ctx = document.getElementById("chart_box").getContext("2d");
  chart = new Chart(ctx, {
    // The type of chart we want to create
    type: "bar",

    // The data for our dataset
    data: {
      labels: label,
      datasets: [
        {
          label: modelSelected + " prediction",
          backgroundColor: "#f50057",
          borderColor: "rgb(255, 99, 132)",
          data: data,
        },
      ],
    },

    // Configuration options go here
    options: {},
  });
}

function displayChart(data) {
  var select_model = document.getElementById("select_model");
  var select_option = "CNN";

  label = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"];
  if (firstTime == 0) {
    loadChart(label, data, select_option);
    firstTime = 1;
  } else {
    chart.destroy();
    loadChart(label, data, select_option);
  }
  document.getElementById("chart_box").style.display = "block";
}

function displayLabel(data) {
  var max = data[0];
  var maxIndex = 0;

  for (var i = 1; i < data.length; i++) {
    if (data[i] > max) {
      maxIndex = i;
      max = data[i];
    }
  }
  $(".prediction-text").html(
    "Predicting you draw <b>" +
      maxIndex +
      "</b> with <b>" +
      Math.trunc(max * 100) +
      "%</b> confidence"
  );
}
