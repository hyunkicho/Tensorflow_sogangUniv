import * as tf from '@tensorflow/tfjs';

document.addEventListener("DOMContentLoaded", ready);

let model;
let imageContainer;
let loadingGif;
let displayResult;

(async function() {
    model = await tf.loadLayersModel('./assets/model/model.json');
})();

function sleep(ms) {
    return new Promise(resolve=>setTimeout(resolve, ms));
  }
  
  
function imageUploaded(event) { 

    const target = event.target;
    const file = target.files[0];

    if (!file) return;
    console.log("imageUploaded")
    imageContainer.src = "";
    loadingGif.style.display = "block";
    displayResult.innerHTML = "";
    displayResult2.innerHTML = "";

    const imageUrl = window.webkitURL.createObjectURL(file);
    imageContainer.src = imageUrl;

    let reader = new FileReader();

    reader.onload = readerEvent => {
        let img = document.createElement('img');
        img.src = readerEvent.target.result;
        img.width = 224;
        img.height = 224;
        img.onload = () => makeSimplePrediction(img);
    };

    reader.readAsDataURL(file);
}

async function makeSimplePrediction(img){
        const [batched_image, resized_image] = normalizeImage(img);
        console.log(batched_image)
        console.log(resized_image)
        let output = await model.predict(batched_image);
        output = output.arraySync()[0];
        console.log(output+"in prediction");
        // if(output==4.47645378112793){
        //     await sleep(1000);
        //     runExample();
        // }
        if(output > 0.5) {
            displayResult.style.color = "red";
            loadingGif.style.display = "none";
            displayResult.innerHTML = "폐렴";
            displayResult2.innerHTML = "진단완료 수치는 : " +output + "입니다. 수치가 0.5 보다 클 경우 폐렴일 확률이 높습니다. 위험군으로 분류되었으니 상세 검사 예약을 진행하겠습니다.";
        }else{
            displayResult.style.color = "green";
            loadingGif.style.display = "none";
            displayResult.innerHTML = "정상";
            displayResult2.innerHTML = "진단완료 수치는 : " +output + "입니다. 수치가 0.5 보다 클 경우 폐렴일 확률이 높습니다. 0.5 보다 작을 수록 정상 수치에 가깝습니다. \n 진단결과 이상이 없습니다. 코로나 환자일 가능성이 낮습니다.";

        }     
        return output;
}

function normalizeImage(img) {
    return tf.tidy(() => {
        const tensorImage = tf.browser.fromPixels(img, 3).toFloat();
        const resized_image = tensorImage.resizeNearestNeighbor([224, 224]);
    
        const offset = tf.scalar(127.5);
        const normalized_image = resized_image.div(offset).sub(tf.scalar(1.0));
    
        const batched_image = normalized_image.expandDims();
        return [batched_image, resized_image];
    });
}

function ready() {
    imageContainer = document.querySelector("#imageContainer");
    loadingGif = document.querySelector("#loadingGif");
    displayResult = document.querySelector("#display_result");

    const inputFile = document.querySelector("#image");
    inputFile.addEventListener('change', imageUploaded);

    const example_button = document.querySelector("#example_button");
    example_button.addEventListener('click', runExample);
}

function runExample() {
    // document.getElementById("loadingGif").style.display="block";
    // document.getElementById("selected_image").style="disabled";
    imageContainer.src = "";
    displayResult.innerHTML = "";
    displayResult2.innerHTML = "";
    if(document.getElementById("selected_image").value == "normal1"){
        console.log("AI 모델 적용 중" +document.getElementById("selected_image").value);        
        let url = "./assets/normal1.jpeg";
        runExamples(url);     
    }else if(document.getElementById("selected_image").value == "normal2"){
        console.log("AI 모델 적용 중" +document.getElementById("selected_image").value);        
        let url = "./assets/normal2.jpeg";
        runExamples(url);             
    }else if(document.getElementById("selected_image").value == "normal3"){
        console.log("AI 모델 적용 중" +document.getElementById("selected_image").value);        
        let url = "./assets/normal3.jpeg";
        runExamples(url);     
    }else if(document.getElementById("selected_image").value == "notnormal1"){
        console.log("AI 모델 적용 중" +document.getElementById("selected_image").value);        
        let url = "./assets/notnormal1.jpeg";
        runExamples(url);              
    }else if(document.getElementById("selected_image").value == "notnormal2"){
        console.log("AI 모델 적용 중" +document.getElementById("selected_image").value);        
        let url = "./assets/notnormal2.jpeg";
        runExamples(url);     
    }else if(document.getElementById("selected_image").value = "notnormal3"){
        console.log("AI 모델 적용 중" +document.getElementById("selected_image").value);        
        let url = "./assets/notnormal3.jpeg";
        runExamples(url);     
    }else{
        alert("이미지를 선택해 주세요!")
    }


}

function runExamples(url){
    loadingGif.style.display = "block";
    // let img = new Image(200, 200);
    // img.src = url;
    let img = document.createElement('img');
    img.src = url;
    img.width = 224;
    img.height = 224;
    img.onload = () => makeSimplePrediction(img);
    imageContainer.src = url;


    console.log(img)
    console.log(url)
    // makeSimplePrediction(img)
}


