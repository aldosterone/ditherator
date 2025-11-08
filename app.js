// app.js - WebGL Base Image Display (Fixed Upside Down Issue)

function main() {
    const canvas = document.getElementById('image-canvas');
    const gl = canvas.getContext('webgl'); 
    if (!gl) {
        alert('Your browser does not support WebGL!');
        return;
    }

    const imageUpload = document.getElementById('image-upload');
    let imageTexture = null;
    let imageWidth = 0;
    let imageHeight = 0;

    // ----------------------------------------------------
    // SHADER COMPILATION AND SETUP
    // ----------------------------------------------------

    function createShader(gl, type, source) {
        const shader = gl.createShader(type);
        gl.shaderSource(shader, source);
        gl.compileShader(shader);
        if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
            console.error('Shader compile error:', gl.getShaderInfoLog(shader));
            gl.deleteShader(shader);
            return null;
        }
        return shader;
    }

    function createProgram(gl, vertexSource, fragmentSource) {
        const vertexShader = createShader(gl, gl.VERTEX_SHADER, vertexSource);
        const fragmentShader = createShader(gl, gl.FRAGMENT_SHADER, fragmentSource);
        
        if (!vertexShader || !fragmentShader) {
            return null;
        }

        const program = gl.createProgram();
        gl.attachShader(program, vertexShader);
        gl.attachShader(program, fragmentShader);
        gl.linkProgram(program);
        if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
            console.error('Program link error:', gl.getProgramInfoLog(program));
            gl.deleteProgram(program);
            return null;
        }
        return program;
    }

    const vertexShaderSource = document.getElementById('vertex-shader').text;
    const fragmentShaderSource = document.getElementById('fragment-shader').text;
    const program = createProgram(gl, vertexShaderSource, fragmentShaderSource);
    
    if (!program) {
        return; 
    }
    
    gl.useProgram(program);

    const positionLocation = gl.getAttribLocation(program, 'a_position');
    const texCoordLocation = gl.getAttribLocation(program, 'a_texCoord');
    const imageLocation = gl.getUniformLocation(program, 'u_image');

    // --- Set up geometry (a full-screen quad) ---
    
    const positionBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([
        -1, -1, 1, -1, -1, 1, 
        -1, 1, 1, -1, 1, 1,
    ]), gl.STATIC_DRAW);

    const texCoordBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, texCoordBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([
        0, 0, 1, 0, 0, 1,
        0, 1, 1, 0, 1, 1,
    ]), gl.STATIC_DRAW);

    // ----------------------------------------------------
    // TEXTURE LOADING & RENDERING FUNCTIONS
    // ----------------------------------------------------

    function setupTextureAndImageLoad(image) {
        if (imageTexture) {
            gl.deleteTexture(imageTexture); 
        }
        imageTexture = gl.createTexture();
        
        gl.activeTexture(gl.TEXTURE0); // Use texture unit 0
        gl.bindTexture(gl.TEXTURE_2D, imageTexture);
        
        // 1. UPLOAD PLACEHOLDER 
        gl.texImage2D(
            gl.TEXTURE_2D, 0, gl.RGBA, 1, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE,
            new Uint8Array([0, 0, 0, 0]) 
        );

        // 2. Set texture parameters (Essential for NPOT images)
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);

        // 3. Define what happens when the image finishes loading
        image.onload = () => {
            gl.bindTexture(gl.TEXTURE_2D, imageTexture);
            
            // ⭐ VITAL FIX: Flip the Y-axis when uploading the image data
            gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true);
            
            // UPLOAD THE REAL IMAGE DATA
            gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, image);
            
            // Reset the flip flag after upload (good practice)
            gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);

            // Update canvas dimensions and viewport
            imageWidth = canvas.width = image.naturalWidth;
            imageHeight = canvas.height = image.naturalHeight;
            gl.viewport(0, 0, imageWidth, imageHeight);
            
            gl.uniform1i(imageLocation, 0); 
            
            drawSimpleImage(); 
        };
    }

    function drawSimpleImage() {
        if (!imageTexture) return;

        gl.clearColor(0, 0, 0, 0);
        gl.clear(gl.COLOR_BUFFER_BIT);

        // Bind Position Buffer and set pointer
        gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);
        gl.enableVertexAttribArray(positionLocation);
        gl.vertexAttribPointer(positionLocation, 2, gl.FLOAT, false, 0, 0);

        // Bind Texture Coordinate Buffer and set pointer
        gl.bindBuffer(gl.ARRAY_BUFFER, texCoordBuffer);
        gl.enableVertexAttribArray(texCoordLocation);
        gl.vertexAttribPointer(texCoordLocation, 2, gl.FLOAT, false, 0, 0);
        
        gl.activeTexture(gl.TEXTURE0); 
        gl.bindTexture(gl.TEXTURE_2D, imageTexture);
        
        gl.drawArrays(gl.TRIANGLES, 0, 6);
    }

    // ----------------------------------------------------
    // EVENT LISTENERS
    // ----------------------------------------------------

    imageUpload.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (!file) return;
        
        const reader = new FileReader();
        reader.onload = (event) => {
            const img = new Image();
            img.crossOrigin = ''; 

            setupTextureAndImageLoad(img);
            img.src = event.target.result;
        };
        reader.readAsDataURL(file);
    });

    drawSimpleImage();
}

window.onload = main;