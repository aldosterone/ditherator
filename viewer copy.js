// viewer.js - Attempt #58
"use strict";

// ARCHITECTURE: Targeted Fix for Bayer Compute Shader Color Space
// Based STRICTLY on Attempt #17 (working Original view).
// 1. Keeps `sourceTexture` format as `rgba8unorm`.
// 2. Keeps the simple passthrough Render Shader.
// 3. Modifies the Bayer Compute Shader ONLY to manually linearize the color
//    read from the `sourceTexture` before dithering.
// 4. Keeps the exact render loop logic from Attempt #17.

async function main() {
    console.log("--- Starting Attempt #58: Targeted Bayer Fix on #17 Base ---");

    if (!navigator.gpu) throw new Error("WebGPU is not supported.");
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) throw new Error("No adapter.");
    const device = await adapter.requestDevice();
     let renderLoopId = null; // Declare renderLoopId early for device.lost
     device.lost.then(info => {
        console.error("WebGPU device was lost:", info.message);
        alert(`WebGPU device lost: ${info.message}. Reload page.`);
        if (renderLoopId) cancelAnimationFrame(renderLoopId);
    });

    const canvas = document.querySelector("canvas");
    const context = canvas.getContext("webgpu");
    const canvasFormat = navigator.gpu.getPreferredCanvasFormat();

    // --- Configure the context ONCE (Identical to #17) ---
    context.configure({
        device,
        format: canvasFormat,
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_DST,
        alphaMode: 'premultiplied',
    });
    console.log(`Canvas configured ONCE to fixed size ${canvas.width}x${canvas.height}`);

    // --- RENDER SHADER (Identical to #17 - WORKING) ---
    const renderShaderModule = device.createShaderModule({
        label: "RenderShader",
        code: `
            struct VSOut { @builtin(position) pos: vec4<f32>, @location(0) uv: vec2<f32> };
            @vertex fn vs(@builtin(vertex_index) i: u32) -> VSOut {
                let pos = array<vec2<f32>, 6>(vec2(-1,-1), vec2(1,-1), vec2(1,1), vec2(-1,-1), vec2(1,1), vec2(-1,1));
                let uv  = array<vec2<f32>, 6>(vec2(0,1), vec2(1,1), vec2(1,0), vec2(0,1), vec2(1,0), vec2(0,0));
                return VSOut(vec4(pos[i], 0.0, 1.0), uv[i]);
            }
            @group(0) @binding(0) var smplr: sampler;
            @group(0) @binding(1) var txtr: texture_2d<f32>;
            @fragment fn fs(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
                return textureSample(txtr, smplr, uv);
            }
        `,
    });
    
    // --- COMPUTE SHADER (Modified *only* to add linearization) ---
    const ditherShaderModule = device.createShaderModule({
        label: "BayerDitherComputeShader_ManualLinear",
        code: `
            // Input texture (reads gamma-encoded data from rgba8unorm)
            @group(0) @binding(0) var sourceTexture: texture_2d<f32>; 
            // Output texture (writes linear 0.0 or 1.0)
            @group(0) @binding(1) var outputTexture: texture_storage_2d<rgba8unorm, write>;

            // *** NEW FUNCTION ***
            // Simple gamma correction (sRGB to linear approx)
            fn linearize(color: vec3<f32>) -> vec3<f32> {
                let safe_color = max(color, vec3(0.0));
                // Use the fast (gamma 2.0) approximation, as pow(2.2) may fail to compile
                return safe_color * safe_color;
            }

            const bayerMatrix: array<array<f32, 8>, 8> = array(
              array( 0.0, 32.0,  8.0, 40.0,  2.0, 34.0, 10.0, 42.0),
              array(48.0, 16.0, 56.0, 24.0, 50.0, 18.0, 58.0, 26.0),
              array(12.0, 44.0,  4.0, 36.0, 14.0, 46.0,  6.0, 38.0),
              array(60.0, 28.0, 52.0, 20.0, 62.0, 30.0, 54.0, 22.0),
              array( 3.0, 35.0, 11.0, 43.0,  1.0, 33.0,  9.0, 41.0),
              array(51.0, 19.0, 59.0, 27.0, 49.0, 17.0, 57.0, 25.0),
              array(15.0, 47.0,  7.0, 39.0, 13.0, 45.0,  5.0, 37.0),
              array(63.0, 31.0, 55.0, 23.0, 61.0, 29.0, 53.0, 21.0)
            );

            @compute @workgroup_size(8, 8)
            fn main(@builtin(global_invocation_id) id: vec3<u32>) {
                let dims = textureDimensions(sourceTexture);
                if (id.x >= dims.x || id.y >= dims.y) { return; }

                // *** MODIFIED LOGIC ***
                // 1. Read gamma color
                let sourceColorGamma = textureLoad(sourceTexture, vec2<i32>(id.xy), 0);
                // 2. Manually linearize
                let sourceColorLinear = linearize(sourceColorGamma.rgb);
                // 3. Calculate grayscale in linear space
                let grayLinear = dot(sourceColorLinear, vec3(0.299, 0.587, 0.114));
                
                // 4. Compare linear gray to linear threshold
                let threshold = bayerMatrix[id.y % 8u][id.x % 8u] / 64.0;
                let quantized = select(0.0, 1.0, grayLinear > threshold);
                
                // 5. Store the result
                textureStore(outputTexture, vec2<i32>(id.xy), vec4(quantized, quantized, quantized, 1.0));
            }
        `,
    });

    // --- Pipelines (Identical structure to #17) ---
    const renderPipeline = device.createRenderPipeline({
        label: "RenderPipeline",
        layout: "auto",
        vertex: { module: renderShaderModule, entryPoint: "vs" },
        fragment: { module: renderShaderModule, entryPoint: "fs", targets: [{ format: canvasFormat }] },
    });
    
    let computePipeline;
    try {
        computePipeline = device.createComputePipeline({
            label: "ComputePipeline",
            layout: "auto",
            compute: { module: ditherShaderModule, entryPoint: "main" }, // Use the modified shader
        });
        console.log("Compute pipeline created successfully.");
    } catch (e) {
        console.error("Failed to create compute pipeline:", e);
        const ditherOption = document.querySelector('#effect-selector option[value="dither"]');
        if (ditherOption) ditherOption.disabled = true;
    }

    const sampler = device.createSampler({ magFilter: "linear", minFilter: "linear" });

    // --- State Management (Identical to #17) ---
    let sourceTexture;
    let needsRedraw = true; 
    let ditheredTexture;
    let currentRenderBindGroup;

    // --- File Loader (Identical to #17) ---
    document.getElementById("image-loader").addEventListener("change", async (event) => {
        const file = event.target.files[0];
        if (!file) return;
        try {
            const imageBitmap = await createImageBitmap(file, { imageOrientation: 'none' });
            if (sourceTexture) sourceTexture.destroy();
            
            sourceTexture = device.createTexture({
                size: [imageBitmap.width, imageBitmap.height],
                format: 'rgba8unorm',
                usage: GPUTextureUsage.TEXTURE_BINDING | 
                       GPUTextureUsage.COPY_DST | 
                       GPUTextureUsage.RENDER_ATTACHMENT,
            });
            console.log("Source texture created with rgba8unorm format.");

            device.queue.copyExternalImageToTexture({ source: imageBitmap }, { texture: sourceTexture }, [imageBitmap.width, imageBitmap.height]);
            canvas.style.aspectRatio = imageBitmap.width / imageBitmap.height;
            needsRedraw = true;
        } catch (e) {
            console.error("Error processing image:", e);
        }
    });
    
    // --- Effect Selector (Identical to #17) ---
    document.getElementById("effect-selector").addEventListener("change", () => {
        needsRedraw = true; 
    });

    // --- Render Loop (Identical to #17) ---
    // renderLoopId is defined at top of main()
    function render() {
        if (!sourceTexture) {
            renderLoopId = requestAnimationFrame(render);
            return;
        }

        const effect = document.getElementById("effect-selector").value;
        let textureToDrawView; 

        if (needsRedraw) {
            const encoder = device.createCommandEncoder({ label: "EffectEncoder" });

            if (effect === "dither" && computePipeline) {
                console.log("Dispatching Bayer compute pass (with fast linearization)...");
                if (!ditheredTexture || ditheredTexture.width !== sourceTexture.width || ditheredTexture.height !== sourceTexture.height) {
                    if (ditheredTexture) ditheredTexture.destroy();
                    ditheredTexture = device.createTexture({
                        size: [sourceTexture.width, sourceTexture.height],
                        format: "rgba8unorm", 
                        usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
                    });
                }

                const computePass = encoder.beginComputePass();
                computePass.setPipeline(computePipeline);
                const computeBindGroup = device.createBindGroup({
                    layout: computePipeline.getBindGroupLayout(0),
                    entries: [
                        { binding: 0, resource: sourceTexture.createView() },
                        { binding: 1, resource: ditheredTexture.createView() },
                    ],
                });
                computePass.setBindGroup(0, computeBindGroup);
                computePass.dispatchWorkgroups(Math.ceil(sourceTexture.width / 8), Math.ceil(sourceTexture.height / 8));
                computePass.end();
                textureToDrawView = ditheredTexture.createView();
            } else { 
                textureToDrawView = sourceTexture.createView();
            }

            currentRenderBindGroup = device.createBindGroup({
                layout: renderPipeline.getBindGroupLayout(0),
                entries: [
                    { binding: 0, resource: sampler },
                    { binding: 1, resource: textureToDrawView },
                ],
            });
            
            device.queue.submit([encoder.finish()]);
            needsRedraw = false; 
        }

        const renderEncoder = device.createCommandEncoder({ label: "RenderEncoder" });
        let currentCanvasTextureView;
        try {
             currentCanvasTextureView = context.getCurrentTexture().createView();
        } catch(e) {
             console.warn("Could not get current texture. Skipping frame.", e);
             renderLoopId = requestAnimationFrame(render);
             return;
        }
        
        const renderPass = renderEncoder.beginRenderPass({
            colorAttachments: [{
                view: currentCanvasTextureView,
                loadOp: "clear",
                clearValue: [1, 1, 1, 1], // White
                storeOp: "store",
            }],
        });
        
        if (currentRenderBindGroup) {
            renderPass.setPipeline(renderPipeline);
            renderPass.setBindGroup(0, currentRenderBindGroup);
            renderPass.draw(6);
        }
        
        renderPass.end();
        device.queue.submit([renderEncoder.finish()]);
        
        renderLoopId = requestAnimationFrame(render);
    }
    
    render();
}
main().catch(console.error);