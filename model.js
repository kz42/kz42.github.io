const MODEL_URL = 'model/my_webmodel/model.json';
const INPUT_NODE = 'image_holder:0';
const OUTPUT_NODE = 'output/softmax/Reshape_1:0';
var background_img = document.getElementById('background_image')
const BACKGROUND_URL = 'test_images/background_img.jpg'
var bkg_tensor;
export class MyModel {
    constructor() {}

    async load() {
        this.model = await tf.loadGraphModel(
            MODEL_URL);
        //get background tensor from img
        background_img.src = BACKGROUND_URL
        background_img.style.display = 'none'
        background_img.onload = () => {
            bkg_tensor = tf.browser.fromPixels(background_img).asType('float32');
        }
    }

    dispose() {
        if (this.model) {
            this.model.dispose();
        }
    }

    predict(input) {
        const reshapedInput = 
            input.reshape([1, ...input.shape]);
        const output = this.model.execute(
            {[INPUT_NODE]: reshapedInput}, OUTPUT_NODE);
        return output;
    }

    getForeGround(prediction) {
        const image_scalar = tf.scalar(256);
        const predictionList = tf.mul(prediction.slice([0,0,0,1], [-1,-1,-1,-1]).reshape([256,256]), image_scalar).asType('int32');
        return predictionList;
    }

    blend(fg, prediction) {
        //get fg & bg from softmax prediction
        const Mfg = prediction.slice([0,0,0,1], [-1,-1,-1,-1]).reshape([256,256,1])
        const Mbg = prediction.slice([0,0,0,0], [-1,-1,-1,1]).reshape([256,256,1])
        //
        const output = 
            tf.add(tf.mul(fg, Mfg), tf.mul(bkg_tensor, Mbg)).resizeBilinear([270,360]).asType('int32')
        return output;
    }
}

