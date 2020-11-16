import {MyModel} from './model.js'

var input_video = document.getElementById('input_video')
var input_image = document.getElementById('input_image')
var output_canvas = document.getElementById('output_canvas')
const status = document.querySelector('p')
var frame_count = 0
// create input canvas to connect video & image
const input_canvas = document.createElement('canvas')
const ctx = input_canvas.getContext('2d')

let model = new MyModel()//init model class

main()

async function main() {
    // load model
    status.innerText = 'Model loading...'
    await model.load()
    status.innerText = 'Model is loaded!!!'
    //play video stream
    const stream = await navigator.mediaDevices.getUserMedia({video: true})
    input_video.srcObject = stream

    input_video.style.margin = 'auto'
    input_video.style.position='absolute'
    input_video.style.top='20%'
    input_video.style.left='20%'


    await input_video.play()
    input_canvas.width=360
    input_canvas.height=270

    output_canvas.style.position='absolute'
    output_canvas.style.top = input_video.style.top
    output_canvas.style.left = 0.2 * window.screen.width + 400
    status.innerText = 'Loading result...'
    refresh()
}

async function refresh() {
    if (frame_count == 1) {
        status.innerText = 'Result is loaded!!!'
    }
    frame_count++
    // //get input image from video stream
    ctx.drawImage(input_video, 0, 0, 360, 270)
    input_image.src = input_canvas.toDataURL('image/png')
    input_image.style.display = 'none'
    //get prediction & draw on canvas

    tf.tidy(() => {
        input_image.onload = () => {
            tf.engine().startScope()
            const fromPixels = 
                tf.browser.fromPixels(input_image).asType('float32')
            const prediction = model.predict(fromPixels)
            const output = model.blend(fromPixels, prediction)
            tf.browser.toPixels(output, output_canvas)
            tf.engine().endScope()
            //dispose tensor
            tf.dispose(fromPixels)
            tf.dispose(prediction)
            tf.dispose(output)
        }
    })
    setTimeout(refresh, 40)
}