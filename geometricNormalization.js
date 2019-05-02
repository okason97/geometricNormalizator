require('@tensorflow/tfjs-node')
var glob = require('glob')

// implements nodejs wrappers for HTMLCanvasElement, HTMLImageElement, ImageData
var canvas = require('canvas');
var faceapi = require('face-api.js');
var Batch = require('batch')
  , batch = new Batch;

  // patch nodejs environment, we need to provide an implementation of
// HTMLCanvasElement and HTMLImageElement, additionally an implementation
// of ImageData is required, in case you want to use the MTCNN
const { Canvas, Image, ImageData } = canvas
faceapi.env.monkeyPatch({ Canvas, Image, ImageData })

var config = require('./recognition')

var fs = require('fs')
var mathHelper = require('./mathHelper')

const desiredFaceWidth = 256
const desiredFaceHeight = desiredFaceWidth
const minFaceWidth = desiredFaceWidth*2
const minFaceHeight = minFaceWidth

async function geometricNormalization(image, name) {
    const options = new faceapi.TinyFaceDetectorOptions({
        inputSize: config.INPUT_SIZE,
        scoreThreshold: config.MIN_CONFIDENCE
    })

    console.log("procesando imagen "+name)

    var imageCanvas = canvas.createCanvas(minFaceWidth, minFaceHeight)
    var ctx = imageCanvas.getContext('2d')
    ctx.fillStyle = "black";
    ctx.fillRect(0, 0, canvas.width, canvas.height);    
    ctx.drawImage(image, (minFaceWidth-image.width)/2, (minFaceHeight-image.height)/2)

    console.log("detectando caras en "+name)

    var fullFaceDescriptions = await faceapi
        .detectAllFaces(imageCanvas, options)
        .withFaceLandmarks()
    console.log("deteccion completa en "+name)


    if (fullFaceDescriptions[0]){
        console.log("cara en "+name)
        var rightEyeCentroid = mathHelper.centroid(fullFaceDescriptions[0].landmarks.getRightEye())
        var leftEyeCentroid = mathHelper.centroid(fullFaceDescriptions[0].landmarks.getLeftEye())
        var faceAngle = Math.atan(mathHelper.slope(leftEyeCentroid,rightEyeCentroid))
        var canvasFace = extractFaces(image, imageCanvas, ctx, rightEyeCentroid, leftEyeCentroid, faceAngle, name)
    }else{
        console.log("no hay cara en "+name)
    }
}

function extractFaces(image, imageCanvas, ctx, rightEyeCentroid, leftEyeCentroid, faceAngle, name){
    var desiredLeftEye = {x: 0.35, y: 0.2}
    var desiredRightEyeX = 1.0 - desiredLeftEye.x
    // calculate scale to desired width
    var dist = mathHelper.distance(rightEyeCentroid, leftEyeCentroid)
    var desiredDist = (desiredRightEyeX - desiredLeftEye.x)
    desiredDist *= desiredFaceWidth
    var scale = desiredDist / dist

    // generate the box
    var eyesCentroid = mathHelper.centroid([leftEyeCentroid, rightEyeCentroid])
    const topLeftCorner = {
        x: eyesCentroid.x-desiredFaceWidth/2,
        y: eyesCentroid.y-desiredFaceHeight*desiredLeftEye.y
    }

    ctx.save();
    ctx.clearRect(0,0,imageCanvas.width, imageCanvas.height);
    ctx.translate(eyesCentroid.x, eyesCentroid.y)
    ctx.rotate(-faceAngle)
    ctx.scale(scale,scale)
    ctx.translate(-eyesCentroid.x, -eyesCentroid.y)
    ctx.rect(topLeftCorner.x,topLeftCorner.y,desiredFaceWidth, desiredFaceHeight)
    ctx.fill()
    ctx.drawImage(image, (minFaceWidth-image.width)/2, (minFaceHeight-image.height)/2)
    ctx.restore();

    // resize the new canvas to the size of the clipping area
    var canvasFace = canvas.createCanvas(desiredFaceWidth, desiredFaceHeight)
    var ctxFace = canvasFace.getContext('2d')
    ctxFace.clearRect(0, 0, canvasFace.width, canvasFace.height)

    // drawcd des    the clipped image from the main canvas to the new canvas
    ctxFace.drawImage(imageCanvas, topLeftCorner.x, topLeftCorner.y,
        desiredFaceWidth, desiredFaceHeight, 0, 0, desiredFaceWidth, desiredFaceHeight);
    saveCallback(canvasFace, name)
    return canvasFace
}

function saveCallback(canvas, name) {
    // Get the DataUrl from the Canvas
    const url = canvas.toDataURL('image/jpg', 1);

    // remove Base64 stuff from the Image
    const base64Data = url.replace(/^data:image\/png;base64,/, "");
    fs.writeFile('./processedImages/'+name, base64Data, 'base64', function (err) {
        console.log(err);
    });
}

async function runRecognition() {            
    await loadModels()
    glob('./data/lfw/*/*.jpg', async (er, files) => {
        // er is an error object or null.
        // files is an array of filenames.
        console.log(files)
        var i,j,filesBatch,chunk = 2;
        for (i=0,j=files.length; i<j; i+=chunk) {
            filesBatch = files.slice(i,i+chunk);
            filesBatch.forEach((file) => {
                batch.push(function(done){
                    console.log("procesando " + file)
                    canvas.loadImage(file).then(async (image) => {
                        await geometricNormalization(image, file.substr(file.lastIndexOf('/') + 1))
                        done()
                    })
                });
            })      
        }
        batch.end()
    })    
}

async function loadModels(){
    await faceapi.nets.tinyFaceDetector.loadFromDisk(config.MODEL_URL)
    await faceapi.nets.faceLandmark68Net.loadFromDisk(config.MODEL_URL)    
}

batch.concurrency(6);
runRecognition()