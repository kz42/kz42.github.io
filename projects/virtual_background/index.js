import {MyModel} from './model.js'

var input_video = document.getElementById('input_video')
var output_canvas = document.getElementById('output_canvas')
const status = document.querySelector('p')
var frame_count = 0

let model = new MyModel()//init model class

main()

async function main() {
    // load model
    status.innerText = 'Model loading...'
    await model.load()
    status.innerText = 'Model is loaded!'
    //play video stream
    status.innerText = 'Loading result...'
    const stream = await navigator.mediaDevices.getUserMedia({video: true})
    const relativeLeft = ((window.screen.width - input_video.width)/2)/window.screen.width * 100

    input_video.srcObject = stream
    input_video.style.margin = 'auto'
    input_video.style.position ='absolute'
    input_video.style.top = '20%'
    input_video.style.left = relativeLeft.toString()+'%'
    await input_video.play()

    output_canvas.style.position='absolute'
    output_canvas.style.top = input_video.style.top
    output_canvas.style.left = input_video.style.left

    refresh()
}

async function refresh() {
    if (frame_count == 2) {
        status.innerText = 'Result is loaded!'
    }
    frame_count++
    tf.tidy(() => {
        tf.engine().startScope()
        const fromPixels = 
            tf.browser.fromPixels(input_video).resizeBilinear([256,256]).asType('float32')
        const prediction = model.predict(fromPixels)
        const output = model.blend(fromPixels, prediction)
        if (frame_count >= 2) {
            tf.browser.toPixels(output, output_canvas)
        }
        tf.engine().endScope()
        //dispose tensor
        tf.dispose(fromPixels)
        tf.dispose(prediction)
        tf.dispose(output)
    })
    setTimeout(refresh, 40)
}