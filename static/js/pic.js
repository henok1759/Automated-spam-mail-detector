const input=document.getElementById("inputfile");
const pic=document.getElementById("pic");
const submit=document.getElementById("select");

input.onchange=function(){
    pic.src=URL.createObjectURL(input.files[0]);
}