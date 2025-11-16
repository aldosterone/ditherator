/**
 * @file viewer.js
 * @description Ditherator: WebGPU Dithering Viewer Core Logic. Handles image loading, effect dispatch (CPU/GPU),
 * and WebGPU pipeline setup.
 * @author Aldo Adriazola
 * @copyright (c) 2025 Aldo Adriazola. All rights reserved.
 * @license Licensed under the MIT License (or specify your chosen license).
 * @version 1.2.1 - Fixed Bayer binding issue
 */
"use strict";

const blueNoiseWidth = 256;
const blueNoiseHeight = 256;
const blueNoiseFileName = "HDR_LA_0(256x256).png";

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
 * 🎨 Floyd-Steinberg Dithering (CPU Implementation)
 */
function floydSteinbergDither(inputData, width, height, brightness, isPerceptual) {
    const outputData = new Uint8Array(width * height * 4);
    const pixelGrayscale = new Float32Array(width * height);

    for (let i = 0; i < width * height; i++) {
        const i4 = i * 4;
        const r = inputData[i4 + 0] / 255.0;
        const g = inputData[i4 + 1] / 255.0;
        const b = inputData[i4 + 2] / 255.0;
        let gray = toLinearGrayscale(r, g, b);
        gray = Math.max(0.0, Math.min(1.0, gray + brightness));
        pixelGrayscale[i] = gray;
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

            const quantizedValue = (oldGrayCompare > 0.5) ? 1.0 : 0.0;
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
function atkinsonDither(inputData, width, height, brightness, isPerceptual) {
    const outputData = new Uint8Array(width * height * 4);
    const pixelGrayscale = new Float32Array(width * height);

    for (let i = 0; i < width * height; i++) {
        const i4 = i * 4;
        const r = inputData[i4 + 0] / 255.0;
        const g = inputData[i4 + 1] / 255.0;
        const b = inputData[i4 + 2] / 255.0;
        let gray = toLinearGrayscale(r, g, b);
        gray = Math.max(0.0, Math.min(1.0, gray + brightness));
        pixelGrayscale[i] = gray;
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

            const quantizedValue = (oldGrayCompare > 0.5) ? 1.0 : 0.0;
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
    console.log("--- Starting Improved Ditherator v1.2.1 ---");
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

    const canvas = document.querySelector("canvas#main-canvas");
    if (!canvas) {
        throw new Error("Main canvas not found");
    }
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

                let noiseDims = textureDimensions(blueNoiseTexture);
                let noiseCoord = vec2<i32>(i32(id.x % noiseDims.x), i32(id.y % noiseDims.y));
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
    
    const bayerDitherShaderModule = device.createShaderModule({
        label: "BayerDitherComputeShader",
        code: `
            @group(0) @binding(0) var sourceTexture: texture_2d<f32>;
            @group(0) @binding(1) var outputTexture: texture_storage_2d<rgba8unorm, write>;
            @group(0) @binding(2) var<uniform> isPerceptual: u32;
            @group(0) @binding(3) var<uniform> matrixSize: u32;

            fn linearize(c: vec3<f32>) -> vec3<f32> { return c * c; }
            fn unlinearize_approx(c: f32) -> f32 { return sqrt(c); }
            
            fn getBayer8(x: u32, y: u32) -> f32 {
                let bayer = array<array<f32, 8>, 8>(
                    array<f32, 8>(0.0, 32.0, 8.0, 40.0, 2.0, 34.0, 10.0, 42.0),
                    array<f32, 8>(48.0, 16.0, 56.0, 24.0, 50.0, 18.0, 58.0, 26.0),
                    array<f32, 8>(12.0, 44.0, 4.0, 36.0, 14.0, 46.0, 6.0, 38.0),
                    array<f32, 8>(60.0, 28.0, 52.0, 20.0, 62.0, 30.0, 54.0, 22.0),
                    array<f32, 8>(3.0, 35.0, 11.0, 43.0, 1.0, 33.0, 9.0, 41.0),
                    array<f32, 8>(51.0, 19.0, 59.0, 27.0, 49.0, 17.0, 57.0, 25.0),
                    array<f32, 8>(15.0, 47.0, 7.0, 39.0, 13.0, 45.0, 5.0, 37.0),
                    array<f32, 8>(63.0, 31.0, 55.0, 23.0, 61.0, 29.0, 53.0, 21.0)
                );
                return (bayer[y % 8u][x % 8u] + 0.5) / 64.0;
            }
            
            fn getBayer16(x: u32, y: u32) -> f32 {
                let bayer = array<array<f32, 16>, 16>(
                    array<f32, 16>(0.0, 128.0, 32.0, 160.0, 8.0, 136.0, 40.0, 168.0, 2.0, 130.0, 34.0, 162.0, 10.0, 138.0, 42.0, 170.0),
                    array<f32, 16>(192.0, 64.0, 224.0, 96.0, 200.0, 72.0, 232.0, 104.0, 194.0, 66.0, 226.0, 98.0, 202.0, 74.0, 234.0, 106.0),
                    array<f32, 16>(48.0, 176.0, 16.0, 144.0, 56.0, 184.0, 24.0, 152.0, 50.0, 178.0, 18.0, 146.0, 58.0, 186.0, 26.0, 154.0),
                    array<f32, 16>(240.0, 112.0, 208.0, 80.0, 248.0, 120.0, 216.0, 88.0, 242.0, 114.0, 210.0, 82.0, 250.0, 122.0, 218.0, 90.0),
                    array<f32, 16>(12.0, 140.0, 44.0, 172.0, 4.0, 132.0, 36.0, 164.0, 14.0, 142.0, 46.0, 174.0, 6.0, 134.0, 38.0, 166.0),
                    array<f32, 16>(204.0, 76.0, 236.0, 108.0, 196.0, 68.0, 228.0, 100.0, 206.0, 78.0, 238.0, 110.0, 198.0, 70.0, 230.0, 102.0),
                    array<f32, 16>(60.0, 188.0, 28.0, 156.0, 52.0, 180.0, 20.0, 148.0, 62.0, 190.0, 30.0, 158.0, 54.0, 182.0, 22.0, 150.0),
                    array<f32, 16>(252.0, 124.0, 220.0, 92.0, 244.0, 116.0, 212.0, 84.0, 254.0, 126.0, 222.0, 94.0, 246.0, 118.0, 214.0, 86.0),
                    array<f32, 16>(3.0, 131.0, 35.0, 163.0, 11.0, 139.0, 43.0, 171.0, 1.0, 129.0, 33.0, 161.0, 9.0, 137.0, 41.0, 169.0),
                    array<f32, 16>(195.0, 67.0, 227.0, 99.0, 203.0, 75.0, 235.0, 107.0, 193.0, 65.0, 225.0, 97.0, 201.0, 73.0, 233.0, 105.0),
                    array<f32, 16>(51.0, 179.0, 19.0, 147.0, 59.0, 187.0, 27.0, 155.0, 49.0, 177.0, 17.0, 145.0, 57.0, 185.0, 25.0, 153.0),
                    array<f32, 16>(243.0, 115.0, 211.0, 83.0, 251.0, 123.0, 219.0, 91.0, 241.0, 113.0, 209.0, 81.0, 249.0, 121.0, 217.0, 89.0),
                    array<f32, 16>(15.0, 143.0, 47.0, 175.0, 7.0, 135.0, 39.0, 167.0, 13.0, 141.0, 45.0, 173.0, 5.0, 133.0, 37.0, 165.0),
                    array<f32, 16>(207.0, 79.0, 239.0, 111.0, 199.0, 71.0, 231.0, 103.0, 205.0, 77.0, 237.0, 109.0, 197.0, 69.0, 229.0, 101.0),
                    array<f32, 16>(63.0, 191.0, 31.0, 159.0, 55.0, 183.0, 23.0, 151.0, 61.0, 189.0, 29.0, 157.0, 53.0, 181.0, 21.0, 149.0),
                    array<f32, 16>(255.0, 127.0, 223.0, 95.0, 247.0, 119.0, 215.0, 87.0, 253.0, 125.0, 221.0, 93.0, 245.0, 117.0, 213.0, 85.0)
                );
                return (bayer[y % 16u][x % 16u] + 0.5) / 256.0;
            }
            
            @compute @workgroup_size(8,8)
            fn main(@builtin(global_invocation_id) id: vec3<u32>) {
                let dims = textureDimensions(sourceTexture);
                if (id.x >= dims.x || id.y >= dims.y) { return; }
                
                let src = textureLoad(sourceTexture, vec2<i32>(id.xy), 0);
                let grayLinear = dot(linearize(src.rgb), vec3(0.299, 0.587, 0.114));
                
                let g_compare = select(grayLinear, unlinearize_approx(grayLinear), isPerceptual > 0u);
                
                let bayerValue = select(getBayer8(id.x, id.y), getBayer16(id.x, id.y), matrixSize == 16u);
                
                let quantized = select(0.0, 1.0, g_compare > bayerValue);
                
                textureStore(outputTexture, vec2<i32>(id.xy), vec4(quantized, quantized, quantized, 1.0));
            }
        `,
    });

    // --- PIPELINES ---
    
    let grayscaleComputePipeline, thresholdComputePipeline, bayerComputePipeline;
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
        bayerComputePipeline = device.createComputePipeline({
            label: "BayerComputePipeline",
            layout: "auto",
            compute: { module: bayerDitherShaderModule, entryPoint: "main" },
        });
        console.log("Compute pipelines created successfully (grayscale, threshold, bayer)");
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
    // Read initial values from HTML elements to keep them in sync
    const biasSlider = document.getElementById("bias-slider");
    const initialBiasValue = biasSlider ? parseFloat(biasSlider.value) : 0.15;
    const biasValueArray = new Float32Array([initialBiasValue]);
    const biasUniformBuffer = device.createBuffer({
        label: "Bias Uniform Buffer",
        size: biasValueArray.byteLength,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(biasUniformBuffer, 0, biasValueArray);
    
    const thresholdSlider = document.getElementById("threshold-slider");
    const initialThresholdValue = thresholdSlider ? parseFloat(thresholdSlider.value) : 0.5;
    const thresholdValueArray = new Float32Array([initialThresholdValue]);
    const thresholdUniformBuffer = device.createBuffer({
        label: "Threshold Uniform Buffer",
        size: thresholdValueArray.byteLength,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(thresholdUniformBuffer, 0, thresholdValueArray);
    
    const brightnessValueArray = new Float32Array([0.0]);
    
    const perceptualCheckbox = document.getElementById("perceptual-mode");
    const perceptualValueArray = new Uint32Array([perceptualCheckbox.checked ? 1 : 0]);
    const perceptualUniformBuffer = device.createBuffer({
        label: "Perceptual Uniform Buffer",
        size: perceptualValueArray.byteLength,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(perceptualUniformBuffer, 0, perceptualValueArray);
    
    // Read initial Bayer size from the active pattern swatch
    const activeBayerSwatch = document.querySelector('.pattern-swatch.active');
    const initialBayerSize = activeBayerSwatch ? parseInt(activeBayerSwatch.dataset.size) : 16;
    const bayerSizeValueArray = new Uint32Array([initialBayerSize]);
    const bayerSizeUniformBuffer = device.createBuffer({
        label: "Bayer Size Uniform Buffer",
        size: bayerSizeValueArray.byteLength,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(bayerSizeUniformBuffer, 0, bayerSizeValueArray);
    
    // --- BAYER PATTERN SWATCHES ---
    function renderBayerPatternSwatch(canvas, matrixSize) {
        if (!canvas) {
            console.error("Canvas element not found for pattern swatch");
            return;
        }
        
        const ctx = canvas.getContext('2d');
        if (!ctx) {
            console.error("Could not get 2d context for pattern swatch");
            return;
        }
        
        // Use the canvas width/height directly
        const width = canvas.width;
        const height = canvas.height;
        
        const bayer8 = [
            [0, 32, 8, 40, 2, 34, 10, 42],
            [48, 16, 56, 24, 50, 18, 58, 26],
            [12, 44, 4, 36, 14, 46, 6, 38],
            [60, 28, 52, 20, 62, 30, 54, 22],
            [3, 35, 11, 43, 1, 33, 9, 41],
            [51, 19, 59, 27, 49, 17, 57, 25],
            [15, 47, 7, 39, 13, 45, 5, 37],
            [63, 31, 55, 23, 61, 29, 53, 21]
        ];
        
        const bayer16 = [
            [0, 128, 32, 160, 8, 136, 40, 168, 2, 130, 34, 162, 10, 138, 42, 170],
            [192, 64, 224, 96, 200, 72, 232, 104, 194, 66, 226, 98, 202, 74, 234, 106],
            [48, 176, 16, 144, 56, 184, 24, 152, 50, 178, 18, 146, 58, 186, 26, 154],
            [240, 112, 208, 80, 248, 120, 216, 88, 242, 114, 210, 82, 250, 122, 218, 90],
            [12, 140, 44, 172, 4, 132, 36, 164, 14, 142, 46, 174, 6, 134, 38, 166],
            [204, 76, 236, 108, 196, 68, 228, 100, 206, 78, 238, 110, 198, 70, 230, 102],
            [60, 188, 28, 156, 52, 180, 20, 148, 62, 190, 30, 158, 54, 182, 22, 150],
            [252, 124, 220, 92, 244, 116, 212, 84, 254, 126, 222, 94, 246, 118, 214, 86],
            [3, 131, 35, 163, 11, 139, 43, 171, 1, 129, 33, 161, 9, 137, 41, 169],
            [195, 67, 227, 99, 203, 75, 235, 107, 193, 65, 225, 97, 201, 73, 233, 105],
            [51, 179, 19, 147, 59, 187, 27, 155, 49, 177, 17, 145, 57, 185, 25, 153],
            [243, 115, 211, 83, 251, 123, 219, 91, 241, 113, 209, 81, 249, 121, 217, 89],
            [15, 143, 47, 175, 7, 135, 39, 167, 13, 141, 45, 173, 5, 133, 37, 165],
            [207, 79, 239, 111, 199, 71, 231, 103, 205, 77, 237, 109, 197, 69, 229, 101],
            [63, 191, 31, 159, 55, 183, 23, 151, 61, 189, 29, 157, 53, 181, 21, 149],
            [255, 127, 223, 95, 247, 119, 215, 87, 253, 125, 221, 93, 245, 117, 213, 85]
        ];
        
        const matrix = matrixSize === 8 ? bayer8 : bayer16;
        const matrixMax = matrixSize === 8 ? 64 : 256;
        
        // Calculate pixel size to fill the entire canvas
        const pixelWidth = width / matrixSize;
        const pixelHeight = height / matrixSize;
        
        // Render each cell of the matrix
        for (let y = 0; y < matrixSize; y++) {
            for (let x = 0; x < matrixSize; x++) {
                const value = matrix[y][x];
                const gray = Math.floor((value / matrixMax) * 255);
                
                ctx.fillStyle = `rgb(${gray}, ${gray}, ${gray})`;
                ctx.fillRect(
                    x * pixelWidth, 
                    y * pixelHeight, 
                    pixelWidth, 
                    pixelHeight
                );
            }
        }
    }
    
    setTimeout(() => {
        const swatches = document.querySelectorAll('.pattern-swatch');
        console.log(`Found ${swatches.length} pattern swatches`);
        
        swatches.forEach(swatch => {
            const canvas = swatch.querySelector('canvas');
            const size = parseInt(swatch.dataset.size);
            
            console.log(`Rendering pattern swatch for ${size}x${size}, canvas:`, canvas);
            renderBayerPatternSwatch(canvas, size);
            
            swatch.addEventListener('click', () => {
                document.querySelectorAll('.pattern-swatch').forEach(s => s.classList.remove('active'));
                swatch.classList.add('active');
                
                bayerSizeValueArray[0] = size;
                device.queue.writeBuffer(bayerSizeUniformBuffer, 0, bayerSizeValueArray);
                console.log(`Bayer matrix size changed to ${size}x${size}`);
                needsRedraw = true;
            });
        });
    }, 100);

    // --- State ---
    let sourceTexture, ditheredTexture, currentRenderBindGroup;
    let needsRedraw = true;
    const runtimeDisplay = document.getElementById('runtime-display');
    let currentEffect = 'original';
    
    const effects = [
        { id: 'original', name: 'Original', needsSlider: false },
        { id: 'grayscale', name: 'Grayscale', needsSlider: false },
        { id: 'threshold', name: 'Threshold', needsSlider: 'threshold' },
        { id: 'floydSteinberg', name: 'Floyd-Steinberg', needsSlider: 'brightness' },
        { id: 'atkinson', name: 'Atkinson', needsSlider: 'brightness' },
        { id: 'bayer', name: 'Bayer', needsSlider: 'bayerSize' },
        { id: 'blueNoise', name: 'Blue Noise', needsSlider: 'bias' }
    ];

    // --- IMAGE LOADER ---
    document.getElementById("image-loader").addEventListener("change", async (event) => {
        const file = event.target.files[0];
        if (!file) return;
        
        try {
            if (!file.type.startsWith('image/')) {
                alert('Please select a valid image file');
                return;
            }
            
            const maxSize = 50 * 1024 * 1024;
            if (file.size > maxSize) {
                alert('Image file is too large. Please select an image under 50MB.');
                return;
            }
            
            const imageBitmap = await createImageBitmap(file, { imageOrientation: 'none' });
            
            const originalWidth = imageBitmap.width;
            const originalHeight = imageBitmap.height;
            
            const maxDimension = 4096;
            if (originalWidth > maxDimension || originalHeight > maxDimension) {
                if (!confirm(`This image is very large (${originalWidth}x${originalHeight}). Processing may be slow. Continue?`)) {
                    return;
                }
            }

            currentImageBitmap = imageBitmap;
            
            canvas.width = originalWidth;
            canvas.height = originalHeight;
            
            const maxDisplayWidth = window.innerWidth * 0.9;
            const maxDisplayHeight = window.innerHeight * 0.9;
            const aspectRatio = originalWidth / originalHeight;
            
            let displayWidth = originalWidth;
            let displayHeight = originalHeight;
            
            if (displayWidth > maxDisplayWidth) {
                displayWidth = maxDisplayWidth;
                displayHeight = displayWidth / aspectRatio;
            }
            if (displayHeight > maxDisplayHeight) {
                displayHeight = maxDisplayHeight;
                displayWidth = displayHeight * aspectRatio;
            }
            
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
            
            generateEffectThumbnails();
            
        } catch (error) {
            console.error('Error loading image:', error);
            alert('Failed to load image. Please try a different file.');
        }
    });

    console.log("Setting up save button...");
    const saveButton = document.getElementById("save-button");
    console.log("Save button element:", saveButton);
    
    if (saveButton) {
        saveButton.addEventListener("click", function(event) {
            console.log("=== SAVE BUTTON CLICKED ===");
            
            if (!sourceTexture) {
                alert('Please load an image first');
                return;
            }
            
            try {
                const effect = currentEffect;
                const timestamp = new Date().toISOString().slice(0, 19).replace(/:/g, '-');
                const filename = `dithered-${effect}-${timestamp}.png`;
                
                console.log("Attempting to save:", filename);
                
                try {
                    const dataUrl = canvas.toDataURL('image/png');
                    const link = document.createElement('a');
                    link.href = dataUrl;
                    link.download = filename;
                    link.style.display = 'none';
                    document.body.appendChild(link);
                    
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
        }, false);
        console.log("Save button listener attached successfully");
    } else {
        console.error("Save button not found!");
    }
    
    // --- THEME TOGGLE ---
    const themeToggle = document.getElementById('theme-toggle');
    const themeIcon = document.getElementById('theme-icon');
    const themeText = document.getElementById('theme-text');
    
    const savedTheme = localStorage.getItem('theme') || 'light';
    if (savedTheme === 'dark') {
        document.body.classList.add('dark-theme');
        themeIcon.textContent = '☀️';
        themeText.textContent = 'Light';
    }
    
    themeToggle.addEventListener('click', () => {
        document.body.classList.toggle('dark-theme');
        const isDark = document.body.classList.contains('dark-theme');
        
        if (isDark) {
            themeIcon.textContent = '☀️';
            themeText.textContent = 'Light';
            localStorage.setItem('theme', 'dark');
        } else {
            themeIcon.textContent = '🌙';
            themeText.textContent = 'Dark';
            localStorage.setItem('theme', 'light');
        }
    });

    // --- Event Listeners ---
    const sliderControls = document.getElementById("slider-controls");
    const biasSliderGroup = document.getElementById("bias-slider-group");
    const thresholdSliderGroup = document.getElementById("threshold-slider-group");
    const brightnessSliderGroup = document.getElementById("brightness-slider-group");
    const bayerSizeGroup = document.getElementById("bayer-size-group");
    // biasSlider and thresholdSlider already defined above when reading initial values
    const biasValue = document.getElementById("bias-value");
    const thresholdValue = document.getElementById("threshold-value");
    const brightnessSlider = document.getElementById("brightness-slider");
    const brightnessDisplayValue = document.getElementById("brightness-value");
    
    function setEffect(effectId) {
        currentEffect = effectId;
        needsRedraw = true;
        
        document.querySelectorAll('.effect-thumb').forEach(thumb => {
            thumb.classList.toggle('active', thumb.dataset.effect === effectId);
        });
        
        const effect = effects.find(e => e.id === effectId);
        if (!effect) return;
        
        if (effect.needsSlider && sliderControls) {
            sliderControls.style.display = "block";
            
            if (biasSliderGroup) biasSliderGroup.style.display = "none";
            if (thresholdSliderGroup) thresholdSliderGroup.style.display = "none";
            if (brightnessSliderGroup) brightnessSliderGroup.style.display = "none";
            if (bayerSizeGroup) bayerSizeGroup.style.display = "none";
            
            if (effect.needsSlider === 'bias' && biasSliderGroup) {
                biasSliderGroup.style.display = "block";
            } else if (effect.needsSlider === 'threshold' && thresholdSliderGroup) {
                thresholdSliderGroup.style.display = "block";
            } else if (effect.needsSlider === 'brightness' && brightnessSliderGroup) {
                brightnessSliderGroup.style.display = "block";
            } else if (effect.needsSlider === 'bayerSize' && bayerSizeGroup) {
                bayerSizeGroup.style.display = "block";
            }
        } else if (sliderControls) {
            sliderControls.style.display = "none";
        }
    }

    async function generateEffectThumbnails() {
        const container = document.getElementById('effect-thumbnails');
        container.innerHTML = '';
        
        if (!currentImageBitmap) return;
        
        const thumbSize = 150;
        const aspectRatio = currentImageBitmap.width / currentImageBitmap.height;
        let thumbWidth = thumbSize;
        let thumbHeight = thumbSize;
        
        if (aspectRatio > 1) {
            thumbHeight = thumbSize / aspectRatio;
        } else {
            thumbWidth = thumbSize * aspectRatio;
        }
        
        for (const effect of effects) {
            if (effect.id === 'blueNoise' && !blueNoiseTexture) continue;
            
            const thumbDiv = document.createElement('div');
            thumbDiv.className = 'effect-thumb';
            thumbDiv.dataset.effect = effect.id;
            if (effect.id === currentEffect) {
                thumbDiv.classList.add('active');
            }
            
            const thumbCanvas = document.createElement('canvas');
            thumbCanvas.width = thumbWidth;
            thumbCanvas.height = thumbHeight;
            
            const label = document.createElement('div');
            label.className = 'label';
            label.textContent = effect.name;
            
            thumbDiv.appendChild(thumbCanvas);
            thumbDiv.appendChild(label);
            container.appendChild(thumbDiv);
            
            generateThumbnailPreview(thumbCanvas, effect.id);
            
            thumbDiv.addEventListener('click', () => {
                setEffect(effect.id);
            });
        }
    }
    
    async function generateThumbnailPreview(thumbCanvas, effectId) {
        if (!currentImageBitmap) return;
        
        const ctx = thumbCanvas.getContext('2d');
        const width = thumbCanvas.width;
        const height = thumbCanvas.height;
        
        ctx.drawImage(currentImageBitmap, 0, 0, width, height);
        
        if (effectId === 'original') return;
        
        const imageData = ctx.getImageData(0, 0, width, height);
        const data = new Uint8Array(imageData.data);
        
        if (effectId === 'grayscale') {
            for (let i = 0; i < data.length; i += 4) {
                const gray = 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
                data[i] = data[i + 1] = data[i + 2] = gray;
            }
        } else if (effectId === 'threshold') {
            for (let i = 0; i < data.length; i += 4) {
                const gray = 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
                const val = gray > 127 ? 255 : 0;
                data[i] = data[i + 1] = data[i + 2] = val;
            }
        } else if (effectId === 'bayer') {
            const bayer = [
                [0, 32, 8, 40, 2, 34, 10, 42],
                [48, 16, 56, 24, 50, 18, 58, 26],
                [12, 44, 4, 36, 14, 46, 6, 38],
                [60, 28, 52, 20, 62, 30, 54, 22],
                [3, 35, 11, 43, 1, 33, 9, 41],
                [51, 19, 59, 27, 49, 17, 57, 25],
                [15, 47, 7, 39, 13, 45, 5, 37],
                [63, 31, 55, 23, 61, 29, 53, 21]
            ];
            for (let y = 0; y < height; y++) {
                for (let x = 0; x < width; x++) {
                    const i = (y * width + x) * 4;
                    const gray = 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
                    const threshold = (bayer[y % 8][x % 8] + 0.5) / 64.0 * 255;
                    const val = gray > threshold ? 255 : 0;
                    data[i] = data[i + 1] = data[i + 2] = val;
                }
            }
        } else if (effectId === 'blueNoise') {
            if (blueNoiseTexture) {
                for (let y = 0; y < height; y++) {
                    for (let x = 0; x < width; x++) {
                        const i = (y * width + x) * 4;
                        const gray = 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
                        
                        const noiseX = x % blueNoiseWidth;
                        const noiseY = y % blueNoiseHeight;
                        const noiseVal = ((noiseX * 73 + noiseY * 179) % 256) / 255.0 * 255;
                        
                        const val = gray > noiseVal ? 255 : 0;
                        data[i] = data[i + 1] = data[i + 2] = val;
                    }
                }
            } else {
                for (let i = 0; i < data.length; i += 4) {
                    const gray = 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
                    const val = gray > 127 ? 255 : 0;
                    data[i] = data[i + 1] = data[i + 2] = val;
                }
            }
        } else if (effectId === 'floydSteinberg') {
            const grayscale = new Float32Array(width * height);
            for (let i = 0; i < width * height; i++) {
                const i4 = i * 4;
                grayscale[i] = 0.299 * data[i4] + 0.587 * data[i4 + 1] + 0.114 * data[i4 + 2];
            }
            
            for (let y = 0; y < height; y++) {
                for (let x = 0; x < width; x++) {
                    const i = y * width + x;
                    const oldGray = grayscale[i];
                    const newVal = oldGray > 127 ? 255 : 0;
                    const error = oldGray - newVal;
                    
                    if (x + 1 < width) grayscale[i + 1] += error * 7/16;
                    if (y + 1 < height) {
                        if (x > 0) grayscale[i + width - 1] += error * 3/16;
                        grayscale[i + width] += error * 5/16;
                        if (x + 1 < width) grayscale[i + width + 1] += error * 1/16;
                    }
                    
                    const i4 = i * 4;
                    data[i4] = data[i4 + 1] = data[i4 + 2] = newVal;
                }
            }
        } else if (effectId === 'atkinson') {
            const grayscale = new Float32Array(width * height);
            for (let i = 0; i < width * height; i++) {
                const i4 = i * 4;
                grayscale[i] = 0.299 * data[i4] + 0.587 * data[i4 + 1] + 0.114 * data[i4 + 2];
            }
            
            for (let y = 0; y < height; y++) {
                for (let x = 0; x < width; x++) {
                    const i = y * width + x;
                    const oldGray = grayscale[i];
                    const newVal = oldGray > 127 ? 255 : 0;
                    const error = (oldGray - newVal) / 8;
                    
                    if (x + 1 < width) grayscale[i + 1] += error;
                    if (x + 2 < width) grayscale[i + 2] += error;
                    if (y + 1 < height) {
                        if (x > 0) grayscale[i + width - 1] += error;
                        grayscale[i + width] += error;
                        if (x + 1 < width) grayscale[i + width + 1] += error;
                    }
                    if (y + 2 < height) grayscale[i + width * 2] += error;
                    
                    const i4 = i * 4;
                    data[i4] = data[i4 + 1] = data[i4 + 2] = newVal;
                }
            }
        }
        
        const newImageData = new ImageData(new Uint8ClampedArray(data), width, height);
        ctx.putImageData(newImageData, 0, 0);
    }

    const legacySelector = document.getElementById("effect-selector");
    if (legacySelector) {
        legacySelector.addEventListener("change", () => {
            setEffect(legacySelector.value);
        });
    }

    if (biasSlider && biasValue) {
        biasSlider.addEventListener("input", (e) => {
            const newValue = e.target.valueAsNumber;
            biasValue.textContent = newValue.toFixed(2);
            biasValueArray[0] = newValue;
            device.queue.writeBuffer(biasUniformBuffer, 0, biasValueArray);
            needsRedraw = true;
        });
    }
    
    if (thresholdSlider && thresholdValue) {
        thresholdSlider.addEventListener("input", (e) => {
            const newValue = e.target.valueAsNumber;
            thresholdValue.textContent = newValue.toFixed(2);
            thresholdValueArray[0] = newValue;
            device.queue.writeBuffer(thresholdUniformBuffer, 0, thresholdValueArray);
            needsRedraw = true;
        });
    }
    
    if (brightnessSlider && brightnessDisplayValue) {
        brightnessSlider.addEventListener("input", (e) => {
            const newValue = e.target.valueAsNumber;
            brightnessDisplayValue.textContent = newValue.toFixed(2);
            brightnessValueArray[0] = newValue;
            needsRedraw = true;
        });
    }
    
    if (perceptualCheckbox) {
        perceptualCheckbox.addEventListener("change", () => {
            perceptualValueArray[0] = perceptualCheckbox.checked ? 1 : 0;
            device.queue.writeBuffer(perceptualUniformBuffer, 0, perceptualValueArray);
            needsRedraw = true;
        });
    }

    // --- KEYBOARD SHORTCUTS ---
    console.log("Setting up keyboard shortcuts...");
    console.log("Document ready state:", document.readyState);
    
    document.addEventListener('keydown', function(e) {
        console.log("=== KEY PRESSED ===", e.key, "keyCode:", e.keyCode, "Target:", e.target.tagName);
        
        if (e.target.tagName === 'INPUT') {
            if (e.target.type !== 'file' && e.target.type !== 'checkbox' && e.target.type !== 'range') {
                console.log("Ignoring - user is typing in input");
                return;
            }
        }
        
        if (e.target.tagName === 'TEXTAREA' || e.target.tagName === 'SELECT') {
            console.log("Ignoring - user is in textarea/select");
            return;
        }
        
        const key = e.key ? e.key.toLowerCase() : String.fromCharCode(e.keyCode || e.which).toLowerCase();
        console.log("Processed key:", key);
        
        let handled = false;
        
        switch(key) {
            case '0':
                console.log("Switching to original");
                setEffect('original');
                handled = true;
                break;
            case '1':
                console.log("Switching to grayscale");
                setEffect('grayscale');
                handled = true;
                break;
            case '2':
                console.log("Switching to threshold");
                setEffect('threshold');
                handled = true;
                break;
            case '3':
                console.log("Switching to bayer");
                setEffect('bayer');
                handled = true;
                break;
            case '4':
                console.log("Switching to blue noise");
                if (blueNoiseTexture) {
                    setEffect('blueNoise');
                }
                handled = true;
                break;
            case '5':
                console.log("Switching to Floyd-Steinberg");
                setEffect('floydSteinberg');
                handled = true;
                break;
            case '6':
                console.log("Switching to Atkinson");
                setEffect('atkinson');
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
    }, false);
    
    console.log("Keyboard shortcuts listener attached successfully");
    console.log("=== ALL EVENT LISTENERS REGISTERED ===");
    console.log("Ready for user interaction");
    
    setEffect('original');

    // --- Render Loop ---
    async function render() {
        if (!sourceTexture) { 
            if (runtimeDisplay) runtimeDisplay.textContent = 'Runtime: N/A';
            renderLoopId = requestAnimationFrame(render); 
            return; 
        }
        const effect = currentEffect;
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

            if (effect === "floydSteinberg" || effect === "atkinson") { 
                isCpuDither = true;
                if (effect === "floydSteinberg") ditherFunction = floydSteinbergDither;
                else ditherFunction = atkinsonDither;
            } else if (effect === "bayer") {
                computePipelineToUse = bayerComputePipeline;
                computeBindGroupLayoutToUse = bayerComputePipeline.getBindGroupLayout(0); 
                computeBindGroupEntries = [
                    { binding: 0, resource: sourceTexture.createView() },
                    { binding: 1, resource: ditheredTexture.createView() }, 
                    { binding: 2, resource: { buffer: perceptualUniformBuffer } },
                    { binding: 3, resource: { buffer: bayerSizeUniformBuffer } },
                ];
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
                    brightnessValueArray[0],
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
                console.log(`Starting ${effect} GPU compute...`);
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
    
    console.log("=== MAIN() COMPLETED SUCCESSFULLY ===");
}

main().catch(error => {
    console.error("=== FATAL ERROR IN MAIN() ===");
    console.error("Error name:", error.name);
    console.error("Error message:", error.message);
    console.error("Error stack:", error.stack);
    alert(`Application error: ${error.message}. Please check the console for details.`);
});
