# Project Isekai
*Copyright (c) 2025 Martin Grimme, MIT License*

A playground project to build a voxel engine for rendering foreign worlds (jap. 異世界, isekai).
Because, as the saying goes, you should reinvent the wheel if you want to become an expert for wheels.

Current State:
* Runs in fairly modern web browsers with support for WebGL2.
* Voxels are rendered natively on the GPU without the use of polygons.
* Basic ray tracing with shadows, reflections, and light refraction.
* A virtual machine for generating textures programmatically with TASM machine language on the GPU.
* Virtually unlimited world sizes.
* Collision detection and gravity - you can explore the world by walking and jumping.
* Keyboard and Gamepad controls.

Right now, it uses:
* Shellfish for the UI
* WebGL2 for the rendering
* AssemblyScript for real-time terrain generation

## Voxels

Voxels (VOlume piXELs) are like pixels, but in 3D. Think of a scene being composed by millions of little
colored cubes instead of triangles, a bit like the Danish plastic toy bricks you might have had as a child.
Depending on the voxel resolution, scenes can be very detailed, just like pixel graphics can be in 2D.

In this engine, the surface material (color, texture, bumps, reflectivity) of each voxel is controllable by
code, and can be pretty detailed as well. The surface can even be animated.

## Ray Marching and Ray Tracing

Ray marching is a technique where for each pixel on the screen, you shoot a ray of vision from
the camera into the scene, in order to determine the color of that pixel. When the ray hits an
object, you take the color of that object (and maybe do some lighting computations on it as well) and
put it on the screen.

On reflective surfaces, that ray may bounce off and hit another object. Following the path
of the ray for several bounces is called ray tracing.

Modern GPUs are quite capable of handling ray marching and limited ray tracing in real-time.

DDA ray marching uses a digital differential analyzer (DDA) algorithm to efficiently march the grid of
voxels along the ray of vision from grid line intersection to grid line intersection until it hits a solid voxel.

One advantage of ray marching a scene of voxels is that the performance is bound by the screen's resolution.
You only have to process what's visible on screen. You may have billions of voxels in the scene, but since the
resolution of the screen is limited, the amount of voxels does not have an impact on the rendering performance
(provided that you get the vast amount of voxels to the GPU fast enough). This allows for very detailed graphics
without a loss of performance.

## TASM

TASM (Texture Assembly) is a custom machine language I have designed to be processed by a tiny virtual machine
in the fragment shader. It lets you write texture code to files, which are then loaded and processed
in the shader.

### How can TASM even be processed in the shader?

Short answer: it can't.

Shaders are pretty bad at working with branching code (which is needed for
interpreting different processor instructions in a virtual machine). Therefore, all the shader
does is implement a single complex instruction that is parameterized to emulate the various
TASM processor instructions. A firmware defines the parameters needed for each TASM instruction.
