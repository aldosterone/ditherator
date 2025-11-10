/**
 * @file viewer.js
 * @description Ditherator: WebGPU Dithering Viewer Core Logic. Handles image loading, effect dispatch (CPU/GPU),
 * and WebGPU pipeline setup.
 * @author Aldo Adriazola
 * @copyright (c) 2025 Aldo Adriazola. All rights reserved.
 * @license Licensed under the MIT License (or specify your chosen license).
 * @version 1.1.0 - Improved with error handling, save feature, and keyboard shortcuts
 */
"use strict";

const blueNoiseWidth = 64;
const blueNoiseHeight = 64;
const blueNoiseFileName = "blue_noise_64x64.png";

let currentImageBitmap = null; 

// --- GLOBAL UTILITY FUNCTIONS ---

function linearize(c) { return c * c; }
function unlinearize_approx(c) { return Math.sqrt(c); }
function toLinearGrayscale(r, g, b) {
    const linR = linearize(r);
    const linG = linearize(g);
    const linB = linearize(b);
    return 0.299 * linR + 0.587 * linG + 0.114 * linB;
}

/**
 * 🎨 Bayer Dithering (CPU Implementation)
 */
function bayerDither(inputData, width, height, threshold, isPerceptual) {
    const outputData = new Uint8Array(width * height * 4);
    
    const bayer = [
      [0, 32, 8, 40, 2, 34, 10, 42], [48, 16, 56, 24, 50, 18, 58, 26],
      [12, 44, 4, 36, 14, 46, 6, 38], [60, 28, 52, 20, 62, 30, 54, 22],
      [3, 35, 11, 43, 1, 33, 9, 41], [51, 19, 59, 27, 49, 17, 57, 25],
      [15, 47, 7, 39, 13, 45, 5, 37], [63, 31, 55, 23, 61, 29, 53, 21]
    ];
    const thresholdBias = threshold - 0.5;

    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            const i = (y * width + x) * 4;
            
            const R = inputData[i + 0] / 255.0;
            const G = inputData[i + 1] / 255.0;
            const B = inputData[i + 2] / 255.0;
            
            let oldGrayLinear = toLinearGrayscale(R, G, B);
            
            let oldGrayCompare = oldGrayLinear;
            if (isPerceptual) {
                oldGrayCompare = unlinearize_approx(oldGrayLinear); 
            }
            
            const bayerVal = (bayer[y % 8][x % 8] + 0.5) / 64.0;
            
            const quantizedValue = (oldGrayCompare > bayerVal + thresholdBias) ? 1.0 : 0.0;
            
            const outputVal = quantizedValue * 255;
            outputData[i + 0] = outputVal;
            outputData[i + 1] = outputVal;
            outputData[i + 2] = outputVal;
            outputData[i + 3] = 255;
        }
    }
    return outputData;
}

/**
 * 🎨 Floyd-Steinberg Dithering (CPU Implementation)
 */
function floydSteinbergDither(inputData, width, height, threshold, isPerceptual) {
    const outputData = new Uint8Array(width * height * 4);
    const pixelGrayscale = new Float32Array(width * height);
    const thresholdBias = threshold - 0.5;

    for (let i = 0; i < width * height; i++) {
        const i4 = i * 4;
        const r = inputData[i4 + 0] / 255.0;
        const g = inputData[i4 + 1] / 255.0;
        const b = inputData[i4 + 2] / 255.0;
        pixelGrayscale[i] = toLinearGrayscale(r, g, b);
    }
    
    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            const i = y * width + x;
            const i4 = i * 4;
            let oldGrayLinear = pixelGrayscale[i];
            
            let oldGrayCompare = oldGrayLinear;
            if (isPerceptual) {
                oldGrayCompare = unlinearize_approx(oldGrayLinear); 
            }

            const quantizedValue = (oldGrayCompare + thresholdBias > 0.5) ? 1.0 : 0.0;
            
            const error = oldGrayLinear - quantizedValue;

            if (x + 1 < width) {
                pixelGrayscale[i + 1] += error * (7 / 16);
            }
            if (y + 1 < height) {
                if (x > 0) {
                    pixelGrayscale[i + width - 1] += error * (3 / 16);
                }
                pixelGrayscale[i + width] += error * (5 / 16);
                
                if (x + 1 < width) {
                    pixelGrayscale[i + width + 1] += error * (1 / 16);
                }
            }

            const outputVal = quantizedValue * 255;
            outputData[i4 + 0] = outputVal;
            outputData[i4 + 1] = outputVal;
            outputData[i4 + 2] = outputVal;
            outputData[i4 + 3] = 255;
        }
    }
    return outputData;
}

/**
 * 🎨 Atkinson Dithering (CPU Implementation)
 */
function atkinsonDither(inputData, width, height, threshold, isPerceptual) {
    const outputData = new Uint8Array(width * height * 4);
    const pixelGrayscale = new Float32Array(width * height);
    const thresholdBias = threshold - 0.5;

    for (let i = 0; i < width * height; i++) {
        const i4 = i * 4;
        const r = inputData[i4 + 0] / 255.0;
        const g = inputData[i4 + 1] / 255.0;
        const b = inputData[i4 + 2] / 255.0;
        pixelGrayscale[i] = toLinearGrayscale(r, g, b);
    }
    
    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            const i = y * width + x;
            const i4 = i * 4;
            let oldGrayLinear = pixelGrayscale[i];
            
            let oldGrayCompare = oldGrayLinear;
            if (isPerceptual) {
                oldGrayCompare = unlinearize_approx(oldGrayLinear); 
            }

            const quantizedValue = (oldGrayCompare + thresholdBias > 0.5) ? 1.0 : 0.0;
            
            const error = oldGrayLinear - quantizedValue;
            const errorFraction = error * (1 / 8); 

            if (x + 1 < width) {
                pixelGrayscale[i + 1] += errorFraction;
            }
            if (x + 2 < width) {
                pixelGrayscale[i + 2] += errorFraction;
            }
            
            if (y + 1 < height) {
                if (x - 1 >= 0) {
                    pixelGrayscale[i + width - 1] += errorFraction;
                }
                pixelGrayscale[i + width] += errorFraction;
                
                if (x + 1 < width) {
                    pixelGrayscale[i + width + 1] += errorFraction;
                }
            }
            
            if (y + 2 < height) {
                pixelGrayscale[i + width * 2] += errorFraction;
            }

            const outputVal = quantizedValue * 255;
            outputData[i4 + 0] = outputVal;
            outputData[i4 + 1] = outputVal;
            outputData[i4 + 2] = outputVal;
            outputData[i4 + 3] = 255;
        }
    }
    return outputData;
}

/**
 * CPU Readback Function
 */
function readImageBitmapToUint8Array(imageBitmap) {
    if (!imageBitmap) return null;

    const width = imageBitmap.width;
    const height = imageBitmap.height;
    const canvas = document.createElement('canvas');
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext('2d');
    
    ctx.drawImage(imageBitmap, 0, 0);
    const imageData = ctx.getImageData(0, 0, width, height);
    
    return imageData.data; 
}

// --- ASYNC DATA FETCH UTILITIES ---
async function loadBlueNoiseTextureFromFile(device, url) {
    console.log(`Attempting to load blue noise texture from ${url}...`);
    try {
        const response = await fetch(url);
        if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
        const blob = await response.blob();
        const imageBitmap = await createImageBitmap(blob, { imageOrientation: 'none', premultiplyAlpha: 'none' });
        if (imageBitmap.width !== blueNoiseWidth || imageBitmap.height !== blueNoiseHeight)
            throw new Error(`Loaded noise image has incorrect dimensions! Expected ${blueNoiseWidth}x${blueNoiseHeight}, got ${imageBitmap.width}x${imageBitmap.height}`);
        const offscreenCanvas = new OffscreenCanvas(blueNoiseWidth, blueNoiseHeight);
        const ctx = offscreenCanvas.getContext("2d");
        ctx.drawImage(imageBitmap, 0, 0);
        
        const imageData = ctx.getImageData(0, 0, blueNoiseWidth, blueNoiseHeight); 
        
        const noiseData = new Uint8Array(blueNoiseWidth * blueNoiseHeight);
        for (let i = 0; i < noiseData.length; i++) noiseData[i] = imageData.data[i * 4];
        const noiseTexture = device.createTexture({
            label: "LoadedBlueNoiseTexture",
            size: [blueNoiseWidth, blueNoiseHeight],
            format: "r8unorm",
            usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
        });
        device.queue.writeTexture({ texture: noiseTexture }, noiseData, { bytesPerRow: blueNoiseWidth }, [blueNoiseWidth, blueNoiseHeight]);
        console.log("Blue noise texture created and uploaded successfully.");
        return noiseTexture;
    } catch (e) {
        console.error(`Failed to load or process noise texture from ${url}:`, e);
        return null;
    }
}

async function main() {
    console.log("--- Starting Improved Ditherator v1.1 ---");
    console.log("Browser:", navigator.userAgent);
    console.log("WebGPU available:", !!navigator.gpu);

    if (!navigator.gpu) {
        alert("WebGPU is not supported in your browser. Please use a compatible browser (Chrome, Edge, etc.)");
        throw new Error("WebGPU is not supported.");
    }
    
    console.log("Requesting WebGPU adapter...");
    
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) {
        alert("Failed to get WebGPU adapter. Your GPU may not support WebGPU.");
        throw new Error("No adapter.");
    }
    
    console.log("Got adapter, requesting device...");
    
    const device = await adapter.requestDevice();
    console.log("Got device successfully");
    
    let renderLoopId = null;
    
    device.lost.then(info => {
        console.error("WebGPU device was lost:", info.message);
        alert(`WebGPU device lost: ${info.message}. Please reload the page.`);
        if (renderLoopId) cancelAnimationFrame(renderLoopId);
    });

    const canvas = document.querySelector("canvas");
    const context = canvas.getContext("webgpu");
    const canvasFormat = navigator.gpu.getPreferredCanvasFormat();

    context.configure({
        device,
        format: canvasFormat,
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_DST,
        alphaMode: 'premultiplied',
    });
    console.log(`Canvas configured to fixed size ${canvas.width}x${canvas.height}`);

    const blueNoiseTexture = await loadBlueNoiseTextureFromFile(device, blueNoiseFileName);
    if (!blueNoiseTexture) {
        console.warn("Blue noise texture failed to load. Disabling blue noise option.");
        const blueNoiseOption = document.querySelector('#effect-selector option[value="blueNoise"]');
        if (blueNoiseOption) blueNoiseOption.disabled = true;
    }

    // --- SHADERS ---
    const renderShaderModule = device.createShaderModule({
        label: "RenderShader",
        code: `
            struct VSOut { @builtin(position) pos: vec4<f32>, @location(0) uv: vec2<f32> };
            @vertex fn vs(@builtin(vertex_index) i: u32) -> VSOut {
                let pos = array<vec2<f32>,6>(vec2(-1,-1),vec2(1,-1),vec2(1,1),vec2(-1,-1),vec2(1,1),vec2(-1,1));
                let uv  = array<vec2<f32>,6>(vec2(0,1),vec2(1,1),vec2(1,0),vec2(0,1),vec2(1,0),vec2(0,0));
                return VSOut(vec4(pos[i],0.0,1.0),uv[i]);
            }
            @group(0) @binding(0) var smplr: sampler;
            @group(0) @binding(1) var txtr: texture_2d<f32>;
            @fragment fn fs(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
                return textureSample(txtr, smplr, uv);
            }
        `,
    });

    const blueNoiseDitherShaderModule = device.createShaderModule({
        label: "BlueNoiseDitherComputeShader",
        code: `
            @group(0) @binding(0) var sourceTexture: texture_2d<f32>;
            @group(0) @binding(1) var outputTexture: texture_storage_2d<rgba8unorm, write>;
            @group(0) @binding(2) var blueNoiseTexture: texture_2d<f32>;
            @group(0) @binding(3) var<uniform> bias: f32;
            @group(0) @binding(4) var<uniform> isPerceptual: u32; 

            fn linearize(c: vec3<f32>) -> vec3<f32> { return c * c; }
            fn unlinearize_approx(c: f32) -> f32 { return sqrt(c); }

            @compute @workgroup_size(8,8)
            fn main(@builtin(global_invocation_id) id: vec3<u32>) {
                let dims = textureDimensions(sourceTexture);
                if (id.x >= dims.x || id.y >= dims.y) { return; }
                let src = textureLoad(sourceTexture, vec2<i32>(id.xy), 0);
                let gray_linear = dot(linearize(src.rgb), vec3(0.299, 0.587, 0.114));

                let g_compare = select(gray_linear, unlinearize_approx(gray_linear), isPerceptual > 0u);

                let noiseCoord = vec2<i32>(i32(id.x % 64u), i32(id.y % 64u));
                let threshold = textureLoad(blueNoiseTexture, noiseCoord, 0).r;
                
                let dither = select(0.0, 1.0, g_compare > threshold + bias);
                textureStore(outputTexture, vec2<i32>(id.xy), vec4(dither, dither, dither, 1.0));
            }
        `,
    });
    
    const grayscaleShaderModule = device.createShaderModule({
        label: "GrayscaleComputeShader",
        code: `
            @group(0) @binding(0) var sourceTexture: texture_2d<f32>;
            @group(0) @binding(1) var outputTexture: texture_storage_2d<rgba8unorm, write>;
            @group(0) @binding(2) var<uniform> isPerceptual: u32; 

            fn linearize(c: vec3<f32>) -> vec3<f32> { return c * c; }
            fn unlinearize_approx(c: f32) -> f32 { return sqrt(c); }

            @compute @workgroup_size(8,8)
            fn main(@builtin(global_invocation_id) id: vec3<u32>) {
                let dims = textureDimensions(sourceTexture);
                if (id.x >= dims.x || id.y >= dims.y) { return; }
                let src = textureLoad(sourceTexture, vec2<i32>(id.xy), 0);
                let grayLinear = dot(linearize(src.rgb), vec3(0.299, 0.587, 0.114));
                
                let final_gray = select(grayLinear, unlinearize_approx(grayLinear), isPerceptual > 0u);

                textureStore(outputTexture, vec2<i32>(id.xy), vec4(final_gray, final_gray, final_gray, 1.0));
            }
        `,
    });
    
    const thresholdShaderModule = device.createShaderModule({
        label: "ThresholdComputeShader",
        code: `
            @group(0) @binding(0) var sourceTexture: texture_2d<f32>;
            @group(0) @binding(1) var outputTexture: texture_storage_2d<rgba8unorm, write>;
            @group(0) @binding(2) var<uniform> threshold: f32;
            @group(0) @binding(3) var<uniform> isPerceptual: u32; 

            fn linearize(c: vec3<f32>) -> vec3<f32> { return c * c; }
            fn unlinearize_approx(c: f32) -> f32 { return sqrt(c); }
            
            @compute @workgroup_size(8,8)
            fn main(@builtin(global_invocation_id) id: vec3<u32>) {
                let dims = textureDimensions(sourceTexture);
                if (id.x >= dims.x || id.y >= dims.y) { return; }
                let src = textureLoad(sourceTexture, vec2<i32>(id.xy), 0);
                
                let grayLinear = dot(linearize(src.rgb), vec3(0.299, 0.587, 0.114));
                
                let g_compare = select(grayLinear, unlinearize_approx(grayLinear), isPerceptual > 0u);

                let quantized = select(0.0, 1.0, g_compare > threshold); 
                
                textureStore(outputTexture, vec2<i32>(id.xy), vec4(quantized, quantized, quantized, 1.0));
            }
        `,
    });

    // --- PIPELINES ---
    
    let grayscaleComputePipeline, thresholdComputePipeline;
    try {
        grayscaleComputePipeline = device.createComputePipeline({
            label: "GrayscaleComputePipeline",
            layout: "auto",
            compute: { module: grayscaleShaderModule, entryPoint: "main" },
        });
        thresholdComputePipeline = device.createComputePipeline({
            label: "ThresholdComputePipeline",
            layout: "auto",
            compute: { module: thresholdShaderModule, entryPoint: "main" },
        });
    } catch(e) { 
        console.error("Failed to create auto-layout pipelines:", e);
        alert("Failed to create compute pipelines. Your GPU may not fully support WebGPU.");
    }

    let blueNoiseComputePipeline;
    let blueNoiseComputeBindGroupLayout; 
    if (blueNoiseTexture) {
        try {
            blueNoiseComputeBindGroupLayout = device.createBindGroupLayout({
                label: "BlueNoiseComputeBindGroupLayout",
                entries: [
                    { binding: 0, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "unfilterable-float" } },
                    { binding: 1, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: "write-only", format: "rgba8unorm" } },
                    { binding: 2, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "unfilterable-float" } },
                    { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
                    { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } }
                ]
            });
            blueNoiseComputePipeline = device.createComputePipeline({
                label: "BlueNoiseComputePipeline",
                layout: device.createPipelineLayout({ bindGroupLayouts: [blueNoiseComputeBindGroupLayout] }), 
                compute: { module: blueNoiseDitherShaderModule, entryPoint: "main" },
            });
            console.log("Blue Noise compute pipeline created successfully.");
        } catch (e) { 
            console.error("Failed to create Blue Noise compute pipeline:", e); 
            const blueNoiseOption = document.querySelector('#effect-selector option[value="blueNoise"]');
            if (blueNoiseOption) blueNoiseOption.disabled = true;
        }
    }

    const renderPipeline = device.createRenderPipeline({
        label: "RenderPipeline",
        layout: "auto",
        vertex: { module: renderShaderModule, entryPoint: "vs" },
        fragment: { module: renderShaderModule, entryPoint: "fs", targets: [{ format: canvasFormat }] },
    });
    
    const sampler = device.createSampler({
        magFilter: "nearest",
        minFilter: "nearest"
    });

    // --- Uniform Buffers ---
    const biasValueArray = new Float32Array([0.30]);
    const biasUniformBuffer = device.createBuffer({
        label: "Bias Uniform Buffer",
        size: biasValueArray.byteLength,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(biasUniformBuffer, 0, biasValueArray);
    
    const thresholdValueArray = new Float32Array([0.5]);
    const thresholdUniformBuffer = device.createBuffer({
        label: "Threshold Uniform Buffer",
        size: thresholdValueArray.byteLength,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(thresholdUniformBuffer, 0, thresholdValueArray);
    
    const perceptualCheckbox = document.getElementById("perceptual-mode");
    const perceptualValueArray = new Uint32Array([perceptualCheckbox.checked ? 1 : 0]);
    const perceptualUniformBuffer = device.createBuffer({
        label: "Perceptual Uniform Buffer",
        size: perceptualValueArray.byteLength,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(perceptualUniformBuffer, 0, perceptualValueArray);

    // --- State ---
    let sourceTexture, ditheredTexture, currentRenderBindGroup;
    let needsRedraw = true;
    const runtimeDisplay = document.getElementById('runtime-display'); 

    // --- IMAGE LOADER WITH ERROR HANDLING ---
    document.getElementById("image-loader").addEventListener("change", async (event) => {
        const file = event.target.files[0];
        if (!file) return;
        
        try {
            // Validate file type
            if (!file.type.startsWith('image/')) {
                alert('Please select a valid image file');
                return;
            }
            
            // Validate file size (50MB limit)
            const maxSize = 50 * 1024 * 1024;
            if (file.size > maxSize) {
                alert('Image file is too large. Please select an image under 50MB.');
                return;
            }
            
            const imageBitmap = await createImageBitmap(file, { imageOrientation: 'none' });
            
            const originalWidth = imageBitmap.width;
            const originalHeight = imageBitmap.height;
            
            // Warn about very large images
            const maxDimension = 4096;
            if (originalWidth > maxDimension || originalHeight > maxDimension) {
                if (!confirm(`This image is very large (${originalWidth}x${originalHeight}). Processing may be slow. Continue?`)) {
                    return;
                }
            }

            currentImageBitmap = imageBitmap;
            
            // Set canvas internal resolution to match image
            canvas.width = originalWidth;
            canvas.height = originalHeight;
            
            // Calculate display size maintaining aspect ratio
            const maxDisplayWidth = window.innerWidth * 0.9;
            const maxDisplayHeight = window.innerHeight * 0.9;
            const aspectRatio = originalWidth / originalHeight;
            
            let displayWidth = originalWidth;
            let displayHeight = originalHeight;
            
            // Scale down if needed
            if (displayWidth > maxDisplayWidth) {
                displayWidth = maxDisplayWidth;
                displayHeight = displayWidth / aspectRatio;
            }
            if (displayHeight > maxDisplayHeight) {
                displayHeight = maxDisplayHeight;
                displayWidth = displayHeight * aspectRatio;
            }
            
            // Set CSS display size explicitly
            canvas.style.width = `${displayWidth}px`;
            canvas.style.height = `${displayHeight}px`;
            
            if (sourceTexture) sourceTexture.destroy();
            sourceTexture = device.createTexture({
                size: [originalWidth, originalHeight],
                format: "rgba8unorm", 
                usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT, 
            });
            
            device.queue.copyExternalImageToTexture(
                { source: currentImageBitmap }, 
                { texture: sourceTexture }, 
                [originalWidth, originalHeight]
            );

            console.log(`Image loaded at ${originalWidth}x${originalHeight}, displayed at ${displayWidth.toFixed(0)}x${displayHeight.toFixed(0)}.`);
            needsRedraw = true;
            
        } catch (error) {
            console.error('Error loading image:', error);
            alert('Failed to load image. Please try a different file.');
        }
    });

    // --- SAVE BUTTON (Safari Compatible) ---
    const saveButton = document.getElementById("save-button");
    
    if (saveButton) {
        saveButton.addEventListener("click", function(event) {
            console.log("Save button clicked"); // Debug log
            
            if (!sourceTexture) {
                alert('Please load an image first');
                return;
            }
            
            try {
                const effect = effectSelector.value;
                const timestamp = new Date().toISOString().slice(0, 19).replace(/:/g, '-');
                const filename = `dithered-${effect}-${timestamp}.png`;
                
                console.log("Attempting to save:", filename); // Debug log
                
                // Try toDataURL first (more reliable in Safari)
                try {
                    const dataUrl = canvas.toDataURL('image/png');
                    const link = document.createElement('a');
                    link.href = dataUrl;
                    link.download = filename;
                    link.style.display = 'none';
                    document.body.appendChild(link);
                    
                    // For Safari, we need to trigger with a slight delay
                    setTimeout(() => {
                        link.click();
                        setTimeout(() => {
                            document.body.removeChild(link);
                        }, 100);
                    }, 0);
                    
                    console.log(`Image saved as ${filename}`);
                    return;
                } catch (dataUrlError) {
                    console.warn('toDataURL failed, trying toBlob:', dataUrlError);
                }
                
                // Fallback to toBlob if toDataURL fails
                if (canvas.toBlob) {
                    canvas.toBlob((blob) => {
                        if (!blob) {
                            console.error('toBlob returned null');
                            alert('Could not save image. Try right-clicking the canvas and selecting "Save Image As..."');
                            return;
                        }
                        const url = URL.createObjectURL(blob);
                        const link = document.createElement('a');
                        link.href = url;
                        link.download = filename;
                        link.style.display = 'none';
                        document.body.appendChild(link);
                        link.click();
                        
                        setTimeout(() => {
                            document.body.removeChild(link);
                            URL.revokeObjectURL(url);
                        }, 100);
                        
                        console.log(`Image saved as ${filename} via toBlob`);
                    }, 'image/png');
                } else {
                    console.error('Neither toDataURL nor toBlob worked');
                    alert('Could not save image. Try right-clicking the canvas and selecting "Save Image As..."');
                }
                
            } catch (error) {
                console.error('Error saving image:', error);
                alert('Failed to save image: ' + error.message);
            }
        }, false); // Use capture phase = false for better Safari compatibility
    } else {
        console.error("Save button not found!");
    }

    // --- Event Listeners ---
    const effectSelector = document.getElementById("effect-selector");
    const sliderControls = document.getElementById("slider-controls");
    const biasSliderGroup = document.getElementById("bias-slider-group");
    const thresholdSliderGroup = document.getElementById("threshold-slider-group");
    const biasSlider = document.getElementById("bias-slider");
    const biasValue = document.getElementById("bias-value");
    const thresholdSlider = document.getElementById("threshold-slider");
    const thresholdValue = document.getElementById("threshold-value");

    effectSelector.addEventListener("change", () => {
        needsRedraw = true;
        const effect = effectSelector.value;
        if (effect === "blueNoise" || effect === "threshold" || effect === "bayer" || effect === "floydSteinberg" || effect === "atkinson") {
            sliderControls.style.display = "block";
        } else {
            sliderControls.style.display = "none";
        }

        if (effect === "blueNoise") {
            biasSliderGroup.style.display = "block";
            thresholdSliderGroup.style.display = "none";
        } else if (effect === "threshold" || effect === "bayer" || effect === "floydSteinberg" || effect === "atkinson") { 
            biasSliderGroup.style.display = "none";
            thresholdSliderGroup.style.display = "block";
        } else {
            biasSliderGroup.style.display = "none";
            thresholdSliderGroup.style.display = "none";
        }
    });

    biasSlider.addEventListener("input", (e) => {
        const newValue = e.target.valueAsNumber;
        biasValue.textContent = newValue.toFixed(2);
        biasValueArray[0] = newValue;
        device.queue.writeBuffer(biasUniformBuffer, 0, biasValueArray);
        needsRedraw = true;
    });
    
    thresholdSlider.addEventListener("input", (e) => {
        const newValue = e.target.valueAsNumber;
        thresholdValue.textContent = newValue.toFixed(2);
        thresholdValueArray[0] = newValue;
        device.queue.writeBuffer(thresholdUniformBuffer, 0, thresholdValueArray);
        needsRedraw = true;
    });
    
    perceptualCheckbox.addEventListener("change", () => {
        perceptualValueArray[0] = perceptualCheckbox.checked ? 1 : 0;
        device.queue.writeBuffer(perceptualUniformBuffer, 0, perceptualValueArray);
        needsRedraw = true;
    });

    // --- KEYBOARD SHORTCUTS (Safari Compatible) ---
    console.log("Setting up keyboard shortcuts..."); // Debug log
    
    document.addEventListener('keydown', function(e) {
        console.log("Key pressed:", e.key, "Target:", e.target.tagName); // Debug log
        
        // Ignore if user is typing in an input (but not file input or checkbox)
        if (e.target.tagName === 'INPUT') {
            if (e.target.type !== 'file' && e.target.type !== 'checkbox' && e.target.type !== 'range') {
                console.log("Ignoring - user is typing in input");
                return;
            }
        }
        
        // Ignore if user is in a textarea or select
        if (e.target.tagName === 'TEXTAREA' || e.target.tagName === 'SELECT') {
            console.log("Ignoring - user is in textarea/select");
            return;
        }
        
        // Get the key in a Safari-compatible way
        const key = e.key ? e.key.toLowerCase() : String.fromCharCode(e.keyCode || e.which).toLowerCase();
        console.log("Processed key:", key); // Debug log
        
        let handled = false;
        
        switch(key) {
            case '0':
                console.log("Switching to original");
                effectSelector.value = 'original';
                effectSelector.dispatchEvent(new Event('change', { bubbles: true }));
                handled = true;
                break;
            case '1':
                console.log("Switching to grayscale");
                effectSelector.value = 'grayscale';
                effectSelector.dispatchEvent(new Event('change', { bubbles: true }));
                handled = true;
                break;
            case '2':
                console.log("Switching to threshold");
                effectSelector.value = 'threshold';
                effectSelector.dispatchEvent(new Event('change', { bubbles: true }));
                handled = true;
                break;
            case '3':
                console.log("Switching to bayer");
                effectSelector.value = 'bayer';
                effectSelector.dispatchEvent(new Event('change', { bubbles: true }));
                handled = true;
                break;
            case '4':
                console.log("Switching to blue noise");
                const blueNoiseOption = document.querySelector('#effect-selector option[value="blueNoise"]');
                if (blueNoiseOption && !blueNoiseOption.disabled) {
                    effectSelector.value = 'blueNoise';
                    effectSelector.dispatchEvent(new Event('change', { bubbles: true }));
                }
                handled = true;
                break;
            case '5':
                console.log("Switching to Floyd-Steinberg");
                effectSelector.value = 'floydSteinberg';
                effectSelector.dispatchEvent(new Event('change', { bubbles: true }));
                handled = true;
                break;
            case '6':
                console.log("Switching to Atkinson");
                effectSelector.value = 'atkinson';
                effectSelector.dispatchEvent(new Event('change', { bubbles: true }));
                handled = true;
                break;
            case 'p':
                console.log("Toggling perceptual mode");
                perceptualCheckbox.checked = !perceptualCheckbox.checked;
                perceptualCheckbox.dispatchEvent(new Event('change', { bubbles: true }));
                handled = true;
                break;
            case 's':
                if (e.ctrlKey || e.metaKey || (!e.ctrlKey && !e.metaKey && !e.altKey)) {
                    console.log("Triggering save");
                    const saveBtn = document.getElementById('save-button');
                    if (saveBtn) {
                        saveBtn.click();
                    }
                    handled = true;
                }
                break;
        }
        
        if (handled) {
            e.preventDefault();
            e.stopPropagation();
        }
    }, false); // Use capture phase = false
    
    console.log("Keyboard shortcuts initialized");

    // --- Render Loop ---
    async function render() {
        if (!sourceTexture) { 
            if (runtimeDisplay) runtimeDisplay.textContent = 'Runtime: N/A';
            renderLoopId = requestAnimationFrame(render); 
            return; 
        }
        const effect = document.getElementById("effect-selector").value;
        let textureToDrawView;

        if (needsRedraw) {
            let startTime = performance.now(); 
            
            let computePipelineToUse = null;
            let computeBindGroupLayoutToUse = null; 
            let computeBindGroupEntries = [];
            let isCpuDither = false;
            let ditherFunction = null; 
            
            if (effect !== "original") {
                if (ditheredTexture) ditheredTexture.destroy();
                ditheredTexture = device.createTexture({
                    size: [sourceTexture.width, sourceTexture.height],
                    format: "rgba8unorm", 
                    usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
                });
            }

            if (effect === "floydSteinberg" || effect === "atkinson" || effect === "bayer") { 
                isCpuDither = true;
                if (effect === "bayer") ditherFunction = bayerDither;
                else if (effect === "floydSteinberg") ditherFunction = floydSteinbergDither;
                else ditherFunction = atkinsonDither;
            } else if (effect === "grayscale") {
                computePipelineToUse = grayscaleComputePipeline;
                computeBindGroupLayoutToUse = grayscaleComputePipeline.getBindGroupLayout(0); 
                computeBindGroupEntries = [
                    { binding: 0, resource: sourceTexture.createView() },
                    { binding: 1, resource: ditheredTexture.createView() }, 
                    { binding: 2, resource: { buffer: perceptualUniformBuffer } },
                ];
            } else if (effect === "threshold") {
                computePipelineToUse = thresholdComputePipeline;
                computeBindGroupLayoutToUse = thresholdComputePipeline.getBindGroupLayout(0); 
                computeBindGroupEntries = [
                    { binding: 0, resource: sourceTexture.createView() },
                    { binding: 1, resource: ditheredTexture.createView() }, 
                    { binding: 2, resource: { buffer: thresholdUniformBuffer } },
                    { binding: 3, resource: { buffer: perceptualUniformBuffer } },
                ];
            } else if (effect === "blueNoise" && blueNoiseComputePipeline) {
                computePipelineToUse = blueNoiseComputePipeline;
                computeBindGroupLayoutToUse = blueNoiseComputeBindGroupLayout; 
                computeBindGroupEntries = [
                    { binding: 0, resource: sourceTexture.createView() },
                    { binding: 1, resource: ditheredTexture.createView() }, 
                    { binding: 2, resource: blueNoiseTexture.createView() },
                    { binding: 3, resource: { buffer: biasUniformBuffer } },
                    { binding: 4, resource: { buffer: perceptualUniformBuffer } }, 
                ];
            }

            if (isCpuDither) {
                // --- CPU Execution Path ---
                console.log(`Starting ${effect} CPU dithering...`);
                
                if (runtimeDisplay) runtimeDisplay.textContent = 'Runtime: Processing...';
                
                const sourceData = readImageBitmapToUint8Array(currentImageBitmap); 
                
                if (!sourceData) {
                    console.error(`${effect} failed: Image data not available.`);
                    if (runtimeDisplay) runtimeDisplay.textContent = 'Runtime: Error';
                    needsRedraw = false;
                    renderLoopId = requestAnimationFrame(render);
                    return;
                }

                const ditheredCpuData = ditherFunction(
                    sourceData, 
                    currentImageBitmap.width, 
                    currentImageBitmap.height, 
                    thresholdValueArray[0], 
                    perceptualValueArray[0] === 1
                );
                
                device.queue.writeTexture(
                    { texture: ditheredTexture }, 
                    ditheredCpuData, 
                    { bytesPerRow: currentImageBitmap.width * 4 }, 
                    [currentImageBitmap.width, currentImageBitmap.height]
                );
                
                let endTime = performance.now(); 
                if (runtimeDisplay) runtimeDisplay.textContent = `Runtime: ${(endTime - startTime).toFixed(2)} ms (CPU)`;
                console.log(`${effect} CPU dithering complete in ${(endTime - startTime).toFixed(2)} ms.`);

            } else if (computePipelineToUse) {
                // --- GPU Compute Execution Path ---
                const encoder = device.createCommandEncoder();
                
                const computePass = encoder.beginComputePass();
                const computeBindGroup = device.createBindGroup({
                    layout: computeBindGroupLayoutToUse, 
                    entries: computeBindGroupEntries.sort((a,b) => a.binding - b.binding),
                });
                computePass.setPipeline(computePipelineToUse);
                computePass.setBindGroup(0, computeBindGroup);
                
                computePass.dispatchWorkgroups(Math.ceil(sourceTexture.width / 8), Math.ceil(sourceTexture.height / 8));
                computePass.end();

                device.queue.submit([encoder.finish()]);
                
                await device.queue.onSubmittedWorkDone(); 
                
                let endTime = performance.now(); 
                if (runtimeDisplay) runtimeDisplay.textContent = `Runtime: ${(endTime - startTime).toFixed(2)} ms (GPU)`;
            } else if (effect === "original") {
                 let endTime = performance.now(); 
                 if (runtimeDisplay) runtimeDisplay.textContent = `Runtime: ${(endTime - startTime).toFixed(2)} ms`;
            }

            // --- Final Render Bind Group Creation ---
            if (effect === "original") {
                textureToDrawView = sourceTexture.createView();
            } else {
                textureToDrawView = ditheredTexture.createView();
            }
            
            currentRenderBindGroup = device.createBindGroup({
                layout: renderPipeline.getBindGroupLayout(0),
                entries: [
                    { binding: 0, resource: sampler },
                    { binding: 1, resource: textureToDrawView },
                ],
            });

            needsRedraw = false;
        }

        // Final canvas drawing pass (always runs)
        const renderEncoder = device.createCommandEncoder();
        const view = context.getCurrentTexture().createView();
        const pass = renderEncoder.beginRenderPass({
            colorAttachments: [{
                view,
                loadOp: "clear",
                clearValue: [1, 1, 1, 1],
                storeOp: "store",
            }],
        });
        pass.setPipeline(renderPipeline);
        if (currentRenderBindGroup) {
            pass.setBindGroup(0, currentRenderBindGroup);
            pass.draw(6);
        }
        pass.end();
        device.queue.submit([renderEncoder.finish()]);
        renderLoopId = requestAnimationFrame(render);
    }

    render();
}

main().catch(error => {
    console.error("Fatal error in main():", error);
    alert(`Application error: ${error.message}. Please check the console for details.`);
});
